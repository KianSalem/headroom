"""The bounded tool layer.

This is the only surface through which a model changes audio, and its shape is
a deliberate design position rather than a serialization detail.

**Every setter is absolute and idempotent.** ``set_gain(gain_db=-2.0)`` means
"the gain op is -2 dB", not "add -2 dB". The chain is a declarative document
re-applied to the source each render, so a relative tool surface would sit at
odds with the model underneath it and would make a repeated call ambiguous --
did the model intend to move again, or restate? Restating an absolute value is
detectable as a no-op, and is reported as one.

**Op identity is the tool layer's job, not the model's.** There are no op ids
in any schema. A gain op is *the* gain op; an EQ band is identified by which
analysis band it corrects. Exposing ids would invite a whole class of failure
-- hallucinated identifiers, edits to stale ops -- in exchange for expressing
configurations no scored dimension asks for.

**Bounds live in the schema, extracted from the same annotations the
validators use.** The model is told the legal range up front instead of
learning it from rejections, and the numbers cannot drift from the ones
actually enforced because there is only one copy.

**Refusal is structured and free.** An out-of-bounds request returns a
machine-readable violation naming the field, the value and the constraint. It
costs a tool round-trip, not a render, so recovering from a bad idea is cheap
in the currency the budget is denominated in.

**Ownership is enforced by omission.** A specialist is not asked politely to
leave another's ops alone; it is never handed the tools. A call to a tool
belonging to another role is rejected with the owner named, and the rejection
is counted -- the rate at which it happens is evidence about the prompt.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, cast

from headroom.analysis.spectral import BAND_EDGES, N_BANDS
from headroom.dsp.chain import Chain
from headroom.dsp.ops import (
    AttackMs,
    BandIndex,
    CeilingDbtp,
    CompRatio,
    CompressorOp,
    EqBand,
    EqFreqHz,
    EqGainDb,
    EqOp,
    EqQ,
    ExpanderOp,
    ExpRatio,
    GainDb,
    GainOp,
    LimiterOp,
    MakeupDb,
    OpKind,
    ReleaseMs,
    StereoWidthOp,
    ThresholdDb,
    Width,
    op_compressor,
    op_eq,
    op_eq_band,
    op_expander,
    op_gain,
    op_limiter,
    op_stereo_width,
)
from headroom.dsp.ops import (
    BoundViolationError as BoundViolation,
)

from .roles import OWNED_OPS, OWNER_OF_OP, Role

#: Below this a call changed nothing audible and is reported as ``no_change``
#: rather than accepted. Without it, restating a value would consume a render
#: and the loop would abort on ``proposal_empty`` with no explanation the
#: model could act on.
NO_CHANGE_EPS: Final[float] = 1e-6

#: Maximum bands in one EQ op, from :class:`~headroom.dsp.ops.EqOp`. There are
#: nine analysis bands and room for eight filters, so an EQ specialist that
#: wants to correct every band at once must choose. That pressure is real and
#: the error names the occupied slots so the choice can be informed.
MAX_EQ_BANDS: Final[int] = 8

FilterShape = Literal["peak", "low_shelf", "high_shelf"]


def band_center_hz(band: int) -> float:
    """Geometric centre of an analysis band. The same rule the heuristic uses,
    so a corrective filter placed by either system lands in the same place and
    the two are comparable."""
    return float((BAND_EDGES[band] * BAND_EDGES[band + 1]) ** 0.5)


def band_label(band: int) -> str:
    return f"{BAND_EDGES[band]:.0f}-{BAND_EDGES[band + 1]:.0f} Hz"


# --- schema generation --------------------------------------------------------


def _range_of(alias: object) -> tuple[float | None, float | None]:
    """Pull ``ge``/``le`` out of an annotated bound alias.

    Reading the constraint off the annotation, rather than restating it, is
    what guarantees the advertised range and the enforced range are the same
    range.
    """
    lo: float | None = None
    hi: float | None = None
    for meta in getattr(alias, "__metadata__", ()):
        for constraint in getattr(meta, "metadata", ()):
            got_lo = getattr(constraint, "ge", None)
            got_hi = getattr(constraint, "le", None)
            if got_lo is not None:
                lo = float(got_lo)
            if got_hi is not None:
                hi = float(got_hi)
    return lo, hi


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    description: str
    bound: object | None = None
    kind: Literal["number", "integer", "string"] = "number"
    enum: tuple[str, ...] = ()
    nullable: bool = False
    required: bool = True

    def json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"description": self.description}
        if self.enum:
            schema["type"] = "string"
            schema["enum"] = list(self.enum)
        elif self.nullable:
            schema["type"] = [self.kind, "null"]
        else:
            schema["type"] = self.kind
        if self.bound is not None:
            lo, hi = _range_of(self.bound)
            cast_to = int if self.kind == "integer" else float
            if lo is not None:
                schema["minimum"] = cast_to(lo)
            if hi is not None:
                schema["maximum"] = cast_to(hi)
        return schema


_REASON: Final[Param] = Param(
    name="reason",
    description=(
        "One short clause naming the measurement this call is meant to move and "
        "the direction, e.g. 'band 6 is 2.1 tol hot'."
    ),
    kind="string",
)


# --- outcomes -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """The result of one tool call.

    ``chain`` is always a valid chain: on failure it is the chain unchanged, so
    a caller can apply a whole batch of calls and keep whatever succeeded.
    """

    ok: bool
    chain: Chain
    #: Formatted "<parameter> <signed delta>", matching the heuristic's
    #: vocabulary so the critic's sign-flip detector reads agent traces with no
    #: special case and the two systems' traces are directly comparable.
    action: str
    payload: dict[str, Any]
    finished: bool = False

    @property
    def error(self) -> str:
        return str(self.payload.get("error", "")) if not self.ok else ""


def _fail(chain: Chain, payload: dict[str, Any]) -> ToolOutcome:
    return ToolOutcome(ok=False, chain=chain, action="", payload=payload)


def _ok(chain: Chain, action: str, note: str) -> ToolOutcome:
    return ToolOutcome(
        ok=True,
        chain=chain,
        action=action,
        payload={"applied": action, "note": note, "chain": chain.describe()},
    )


def _no_change(chain: Chain, what: str) -> ToolOutcome:
    return _fail(
        chain,
        {
            "error": "no_change",
            "detail": f"{what} already has that value; nothing was changed",
            "hint": "Choose a different value, or call finish if the chain is right.",
        },
    )


# --- argument coercion --------------------------------------------------------


class _BadArgsError(ValueError):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__(payload.get("detail", "bad arguments"))


def _number(args: Mapping[str, Any], name: str, default: float | None = None) -> float:
    if name not in args or args[name] is None:
        if default is not None:
            return default
        raise _BadArgsError(
            {"error": "missing_argument", "field": name, "detail": f"{name} is required"}
        )
    value = args[name]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _BadArgsError(
            {"error": "bad_argument", "field": name, "value": value, "detail": "expected a number"}
        )
    try:
        parsed = float(value)
    except ValueError as exc:
        raise _BadArgsError(
            {"error": "bad_argument", "field": name, "value": value, "detail": "expected a number"}
        ) from exc
    # NaN and infinity parse happily and then poison everything downstream:
    # a NaN threshold silently disables a compressor, and int(nan) raises.
    if not math.isfinite(parsed):
        raise _BadArgsError(
            {
                "error": "bad_argument",
                "field": name,
                "value": value,
                "detail": "expected a finite number",
            }
        )
    return parsed


def _band(args: Mapping[str, Any], name: str = "band_index") -> int:
    value = int(_number(args, name))
    if not 0 <= value < N_BANDS:
        raise _BadArgsError(
            {
                "error": "bad_argument",
                "field": name,
                "value": value,
                "detail": f"band index must be 0..{N_BANDS - 1}",
                "bands": {i: band_label(i) for i in range(N_BANDS)},
            }
        )
    return value


def _optional_band(args: Mapping[str, Any], name: str = "band_index") -> int | None:
    if name not in args or args[name] is None:
        return None
    return _band(args, name)


def _shape(args: Mapping[str, Any]) -> FilterShape:
    value = str(args.get("shape", "peak"))
    if value not in ("peak", "low_shelf", "high_shelf"):
        raise _BadArgsError(
            {
                "error": "bad_argument",
                "field": "shape",
                "value": value,
                "detail": "shape must be peak, low_shelf or high_shelf",
            }
        )
    return cast(FilterShape, value)


# --- op-level upserts ---------------------------------------------------------
# Each of these owns the identity rule for one op kind: what counts as "the
# same op" for the purpose of revising rather than stacking.


def _single(chain: Chain, kind: OpKind) -> Any | None:
    found = chain.of_kind(kind)
    return found[0] if found else None


def _set_gain(chain: Chain, args: Mapping[str, Any]) -> ToolOutcome:
    want = _number(args, "gain_db")
    existing = _single(chain, OpKind.GAIN)
    if existing is not None:
        current = cast(GainOp, existing).gain_db
        if abs(want - current) < NO_CHANGE_EPS:
            return _no_change(chain, "gain")
        return _ok(
            chain.edit(existing.id, gain_db=want),
            f"gain.gain_db {want - current:+.3f}",
            f"gain {current:+.2f} -> {want:+.2f} dB",
        )
    if abs(want) < NO_CHANGE_EPS:
        return _no_change(chain, "gain (absent, which is 0 dB)")
    return _ok(
        chain.add(op_gain(want)), f"gain.gain_db {want:+.3f}", f"gain op added at {want:+.2f} dB"
    )


def _set_limiter(chain: Chain, args: Mapping[str, Any]) -> ToolOutcome:
    ceiling = _number(args, "ceiling_dbtp")
    release = _number(args, "release_ms", 50.0)
    existing = _single(chain, OpKind.LIMITER)
    if existing is not None:
        op = cast(LimiterOp, existing)
        if (
            abs(ceiling - op.ceiling_dbtp) < NO_CHANGE_EPS
            and abs(release - op.release_ms) < NO_CHANGE_EPS
        ):
            return _no_change(chain, "limiter")
        return _ok(
            chain.edit(op.id, ceiling_dbtp=ceiling, release_ms=release),
            f"limiter.ceiling {ceiling - op.ceiling_dbtp:+.3f}",
            f"ceiling {op.ceiling_dbtp:+.2f} -> {ceiling:+.2f} dBTP",
        )
    return _ok(
        chain.add(op_limiter(ceiling_dbtp=ceiling, release_ms=release)),
        f"limiter.ceiling {ceiling:+.3f}",
        f"limiter added at {ceiling:+.2f} dBTP",
    )


def _remove(chain: Chain, kind: OpKind, action_key: str, magnitude: float) -> ToolOutcome:
    existing = _single(chain, kind)
    if existing is None:
        return _no_change(chain, f"{kind} (absent)")
    return _ok(
        chain.remove(existing.id),
        f"{action_key} {-magnitude:+.3f}",
        f"{kind} removed",
    )


def _remove_limiter(chain: Chain, args: Mapping[str, Any]) -> ToolOutcome:
    existing = _single(chain, OpKind.LIMITER)
    magnitude = abs(cast(LimiterOp, existing).ceiling_dbtp) if existing is not None else 0.0
    return _remove(chain, OpKind.LIMITER, "limiter.ceiling", magnitude)


def _eq_bands(chain: Chain) -> tuple[EqOp | None, list[EqBand]]:
    existing = _single(chain, OpKind.EQ)
    if existing is None:
        return None, []
    op = cast(EqOp, existing)
    return op, list(op.bands)


def _write_eq(
    chain: Chain, op: EqOp | None, bands: Sequence[EqBand], action: str, note: str
) -> ToolOutcome:
    payload = tuple(b.model_dump() for b in bands)
    if op is None:
        return _ok(chain.add(op_eq(bands)), action, note)
    if not bands:
        return _ok(chain.remove(op.id), action, note)
    return _ok(chain.edit(op.id, bands=payload), action, note)


def _upsert_eq_band(
    chain: Chain,
    *,
    freq_hz: float,
    gain_db: float,
    q: float,
    shape: FilterShape,
    action_key: str,
    label: str,
) -> ToolOutcome:
    op, bands = _eq_bands(chain)
    for i, band in enumerate(bands):
        if band.shape == shape and abs(band.freq_hz - freq_hz) < 1.0:
            if abs(gain_db - band.gain_db) < NO_CHANGE_EPS and abs(q - band.q) < NO_CHANGE_EPS:
                return _no_change(chain, label)
            bands[i] = op_eq_band(shape=shape, freq_hz=band.freq_hz, gain_db=gain_db, q=q)
            return _write_eq(
                chain,
                op,
                bands,
                f"{action_key} {gain_db - band.gain_db:+.3f}",
                f"{label} {band.gain_db:+.2f} -> {gain_db:+.2f} dB",
            )
    if abs(gain_db) < NO_CHANGE_EPS:
        return _no_change(chain, f"{label} (absent, which is 0 dB)")
    if len(bands) >= MAX_EQ_BANDS:
        return _fail(
            chain,
            {
                "error": "eq_full",
                "detail": f"the equalizer holds at most {MAX_EQ_BANDS} filters",
                "occupied": [f"{b.shape} {b.freq_hz:.0f}Hz {b.gain_db:+.1f}dB" for b in bands],
                "hint": "Remove the least useful filter first, or widen an existing one.",
            },
        )
    bands.append(op_eq_band(shape=shape, freq_hz=freq_hz, gain_db=gain_db, q=q))
    return _write_eq(
        chain, op, bands, f"{action_key} {gain_db:+.3f}", f"{label} added at {gain_db:+.2f} dB"
    )


def _set_eq_band(chain: Chain, args: Mapping[str, Any]) -> ToolOutcome:
    band = _band(args)
    return _upsert_eq_band(
        chain,
        freq_hz=band_center_hz(band),
        gain_db=_number(args, "gain_db"),
        q=_number(args, "q", 1.0),
        shape="peak",
        action_key=f"eq.band{band}",
        label=f"band {band} ({band_label(band)})",
    )


def _set_eq_filter(chain: Chain, args: Mapping[str, Any]) -> ToolOutcome:
    shape = _shape(args)
    freq = _number(args, "freq_hz")
    return _upsert_eq_band(
        chain,
        freq_hz=freq,
        gain_db=_number(args, "gain_db"),
        q=_number(args, "q", 0.707),
        shape=shape,
        action_key=f"eq.{shape}{freq:.0f}",
        label=f"{shape} at {freq:.0f} Hz",
    )


def _remove_eq_band(chain: Chain, args: Mapping[str, Any]) -> ToolOutcome:
    band = _band(args)
    freq = band_center_hz(band)
    op, bands = _eq_bands(chain)
    for i, existing in enumerate(bands):
        if existing.shape == "peak" and abs(existing.freq_hz - freq) < 1.0:
            removed = bands.pop(i)
            return _write_eq(
                chain,
                op,
                bands,
                f"eq.band{band} {-removed.gain_db:+.3f}",
                f"band {band} filter removed",
            )
    return _no_change(chain, f"band {band} (no filter there)")


def _set_compressor(chain: Chain, args: Mapping[str, Any]) -> ToolOutcome:
    threshold = _number(args, "threshold_db")
    ratio = _number(args, "ratio")
    attack = _number(args, "attack_ms", 10.0)
    release = _number(args, "release_ms", 120.0)
    makeup = _number(args, "makeup_db", 0.0)
    existing = _single(chain, OpKind.COMPRESSOR)
    if existing is not None:
        op = cast(CompressorOp, existing)
        same = (
            abs(threshold - op.threshold_db) < NO_CHANGE_EPS
            and abs(ratio - op.ratio) < NO_CHANGE_EPS
            and abs(attack - op.attack_ms) < NO_CHANGE_EPS
            and abs(release - op.release_ms) < NO_CHANGE_EPS
            and abs(makeup - op.makeup_db) < NO_CHANGE_EPS
        )
        if same:
            return _no_change(chain, "compressor")
        return _ok(
            chain.edit(
                op.id,
                threshold_db=threshold,
                ratio=ratio,
                attack_ms=attack,
                release_ms=release,
                makeup_db=makeup,
            ),
            f"comp.ratio {ratio - op.ratio:+.3f}",
            f"compressor {op.ratio:.2f}:1 @ {op.threshold_db:+.1f} -> "
            f"{ratio:.2f}:1 @ {threshold:+.1f} dB",
        )
    return _ok(
        chain.add(
            op_compressor(
                threshold_db=threshold,
                ratio=ratio,
                attack_ms=attack,
                release_ms=release,
                makeup_db=makeup,
            )
        ),
        f"comp.ratio {ratio - 1.0:+.3f}",
        f"compressor added, {ratio:.2f}:1 @ {threshold:+.1f} dB",
    )


def _set_expander(chain: Chain, args: Mapping[str, Any]) -> ToolOutcome:
    threshold = _number(args, "threshold_db")
    ratio = _number(args, "ratio")
    attack = _number(args, "attack_ms", 5.0)
    release = _number(args, "release_ms", 80.0)
    existing = _single(chain, OpKind.EXPANDER)
    if existing is not None:
        op = cast(ExpanderOp, existing)
        same = (
            abs(threshold - op.threshold_db) < NO_CHANGE_EPS
            and abs(ratio - op.ratio) < NO_CHANGE_EPS
            and abs(attack - op.attack_ms) < NO_CHANGE_EPS
            and abs(release - op.release_ms) < NO_CHANGE_EPS
        )
        if same:
            return _no_change(chain, "expander")
        return _ok(
            chain.edit(
                op.id,
                threshold_db=threshold,
                ratio=ratio,
                attack_ms=attack,
                release_ms=release,
            ),
            f"exp.ratio {ratio - op.ratio:+.3f}",
            f"expander {op.ratio:.2f}:1 -> {ratio:.2f}:1 @ {threshold:+.1f} dB",
        )
    return _ok(
        chain.add(
            op_expander(threshold_db=threshold, ratio=ratio, attack_ms=attack, release_ms=release)
        ),
        f"exp.ratio {ratio - 1.0:+.3f}",
        f"expander added, {ratio:.2f}:1 @ {threshold:+.1f} dB",
    )


def _remove_compressor(chain: Chain, args: Mapping[str, Any]) -> ToolOutcome:
    existing = _single(chain, OpKind.COMPRESSOR)
    magnitude = cast(CompressorOp, existing).ratio - 1.0 if existing is not None else 0.0
    return _remove(chain, OpKind.COMPRESSOR, "comp.ratio", magnitude)


def _remove_expander(chain: Chain, args: Mapping[str, Any]) -> ToolOutcome:
    existing = _single(chain, OpKind.EXPANDER)
    magnitude = cast(ExpanderOp, existing).ratio - 1.0 if existing is not None else 0.0
    return _remove(chain, OpKind.EXPANDER, "exp.ratio", magnitude)


def _set_stereo_width(chain: Chain, args: Mapping[str, Any]) -> ToolOutcome:
    band = _optional_band(args)
    want = _number(args, "width")
    label = "global" if band is None else f"band{band}"
    human = "the whole mix" if band is None else f"band {band} ({band_label(band)})"
    for op in chain.of_kind(OpKind.STEREO_WIDTH):
        width_op = cast(StereoWidthOp, op)
        if width_op.band == band:
            if abs(want - width_op.width) < NO_CHANGE_EPS:
                return _no_change(chain, f"width on {human}")
            return _ok(
                chain.edit(width_op.id, width=want),
                f"width.{label} {want - width_op.width:+.3f}",
                f"width on {human} {width_op.width:.3f} -> {want:.3f}",
            )
    if abs(want - 1.0) < NO_CHANGE_EPS:
        return _no_change(chain, f"width on {human} (absent, which is 1.0)")
    return _ok(
        chain.add(op_stereo_width(width=want, band=band)),
        f"width.{label} {want - 1.0:+.3f}",
        f"width on {human} set to {want:.3f}",
    )


def _remove_stereo_width(chain: Chain, args: Mapping[str, Any]) -> ToolOutcome:
    band = _optional_band(args)
    label = "global" if band is None else f"band{band}"
    for op in chain.of_kind(OpKind.STEREO_WIDTH):
        width_op = cast(StereoWidthOp, op)
        if width_op.band == band:
            return _ok(
                chain.remove(width_op.id),
                f"width.{label} {1.0 - width_op.width:+.3f}",
                f"width op on {label} removed",
            )
    return _no_change(chain, f"width on {label} (absent)")


def _finish(chain: Chain, args: Mapping[str, Any]) -> ToolOutcome:
    return ToolOutcome(
        ok=True,
        chain=chain,
        action="",
        payload={"finished": True, "reason": str(args.get("reason", ""))},
        finished=True,
    )


# --- the tool table -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    #: ``None`` means every role gets it.
    role: Role | None
    summary: str
    params: tuple[Param, ...]
    apply: Callable[[Chain, Mapping[str, Any]], ToolOutcome] = field(compare=False)

    def json_schema(self) -> dict[str, Any]:
        """Anthropic tool-definition shape."""
        return {
            "name": self.name,
            "description": self.summary,
            "input_schema": {
                "type": "object",
                "properties": {p.name: p.json_schema() for p in self.params},
                "required": [p.name for p in self.params if p.required],
            },
        }


def _spec(
    name: str,
    role: Role | None,
    summary: str,
    params: Sequence[Param],
    apply: Callable[[Chain, Mapping[str, Any]], ToolOutcome],
) -> ToolSpec:
    return ToolSpec(name=name, role=role, summary=summary, params=(*params, _REASON), apply=apply)


TOOLS: Final[dict[str, ToolSpec]] = {
    spec.name: spec
    for spec in (
        # --- loudness ---
        _spec(
            "set_gain",
            Role.LOUDNESS,
            "Set the output gain to an absolute value in dB. Moves integrated "
            "loudness roughly one-for-one and changes nothing else.",
            [Param("gain_db", "Absolute gain in dB.", GainDb)],
            _set_gain,
        ),
        _spec(
            "set_limiter",
            Role.LOUDNESS,
            "Set the true-peak limiter's ceiling. Use it to bring peaks under a "
            "delivery ceiling; it lowers crest factor as a side effect, so do not "
            "reach for it to fix level.",
            [
                Param("ceiling_dbtp", "True-peak ceiling in dBTP.", CeilingDbtp),
                Param("release_ms", "Release in ms.", ReleaseMs, required=False),
            ],
            _set_limiter,
        ),
        _spec("remove_limiter", Role.LOUDNESS, "Remove the limiter.", [], _remove_limiter),
        # --- eq ---
        _spec(
            "set_eq_band",
            Role.EQ,
            "Set a corrective peaking filter on one of the nine analysis bands, "
            "centred on that band. This is the direct lever on band_clr_<i>: a "
            "band reading too hot wants a negative gain here.",
            [
                Param(
                    "band_index",
                    "Analysis band, 0 (lowest) to 8 (highest): "
                    + ", ".join(f"{i}={band_label(i)}" for i in range(N_BANDS)),
                    BandIndex,
                    kind="integer",
                ),
                Param("gain_db", "Absolute filter gain in dB.", EqGainDb),
                Param(
                    "q",
                    "Filter Q. Around 1.0 moves a whole analysis band; higher "
                    "carves a notch inside it.",
                    EqQ,
                    required=False,
                ),
            ],
            _set_eq_band,
        ),
        _spec(
            "set_eq_filter",
            Role.EQ,
            "Set a free-form peaking or shelving filter at an arbitrary frequency. "
            "Use it for a narrow resonance that sits inside one analysis band, "
            "where a band-wide correction would dull everything around it.",
            [
                Param("shape", "Filter shape.", enum=("peak", "low_shelf", "high_shelf")),
                Param("freq_hz", "Centre or corner frequency in Hz.", EqFreqHz),
                Param("gain_db", "Absolute filter gain in dB.", EqGainDb),
                Param("q", "Filter Q.", EqQ, required=False),
            ],
            _set_eq_filter,
        ),
        _spec(
            "remove_eq_band",
            Role.EQ,
            "Remove the corrective filter on one analysis band.",
            [Param("band_index", "Analysis band, 0 to 8.", BandIndex, kind="integer")],
            _remove_eq_band,
        ),
        # --- dynamics ---
        _spec(
            "set_compressor",
            Role.DYNAMICS,
            "Set the compressor. Reduces crest factor and loudness range. The "
            "threshold must sit BELOW the programme level to do anything at all.",
            [
                Param(
                    "threshold_db",
                    "Threshold in dBFS. Must be below the programme level; a "
                    "threshold above it is a no-op.",
                    ThresholdDb,
                ),
                Param("ratio", "Compression ratio, 1.0 being no compression.", CompRatio),
                Param(
                    "attack_ms",
                    "Attack in ms. Short attacks blunt transients and lower the percussive ratio.",
                    AttackMs,
                    required=False,
                ),
                Param("release_ms", "Release in ms.", ReleaseMs, required=False),
                Param(
                    "makeup_db",
                    "Make-up gain in dB, applied after compression.",
                    MakeupDb,
                    required=False,
                ),
            ],
            _set_compressor,
        ),
        _spec(
            "set_expander",
            Role.DYNAMICS,
            "Set the downward expander. Restores crest factor on material that has "
            "been over-compressed. The threshold must sit ABOVE the programme "
            "level's quiet passages to reach them.",
            [
                Param("threshold_db", "Threshold in dBFS.", ThresholdDb),
                Param("ratio", "Expansion ratio, 1.0 being no expansion.", ExpRatio),
                Param("attack_ms", "Attack in ms.", AttackMs, required=False),
                Param("release_ms", "Release in ms.", ReleaseMs, required=False),
            ],
            _set_expander,
        ),
        _spec("remove_compressor", Role.DYNAMICS, "Remove the compressor.", [], _remove_compressor),
        _spec("remove_expander", Role.DYNAMICS, "Remove the expander.", [], _remove_expander),
        # --- stereo ---
        _spec(
            "set_stereo_width",
            Role.STEREO,
            "Set the mid/side width factor. 1.0 is unchanged, below 1.0 narrows "
            "toward mono, above 1.0 widens. Omit band_index to move the whole mix, "
            "or give one to move a single analysis band.",
            [
                Param("width", "Absolute width factor.", Width),
                Param(
                    "band_index",
                    "Analysis band 0 to 8, or null for the whole mix.",
                    BandIndex,
                    kind="integer",
                    nullable=True,
                    required=False,
                ),
            ],
            _set_stereo_width,
        ),
        _spec(
            "remove_stereo_width",
            Role.STEREO,
            "Remove a width op.",
            [
                Param(
                    "band_index",
                    "Analysis band, or null for the global op.",
                    BandIndex,
                    kind="integer",
                    nullable=True,
                    required=False,
                )
            ],
            _remove_stereo_width,
        ),
        # --- shared ---
        _spec(
            "finish",
            None,
            "Stop editing and hand the chain back to be rendered and measured. "
            "Call this once the edits you want for this round are in place.",
            [],
            _finish,
        ),
    )
}


def tools_for(role: Role) -> tuple[ToolSpec, ...]:
    return tuple(spec for spec in TOOLS.values() if spec.role in (None, role))


def schemas_for(role: Role) -> list[dict[str, Any]]:
    return [spec.json_schema() for spec in tools_for(role)]


def apply_call(chain: Chain, role: Role, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
    """Dispatch one call, converting every failure into a structured result.

    Nothing raises. A specialist that emits nonsense gets a payload it can read
    and retry from, and the retry costs a tool round-trip rather than a render.
    """
    spec = TOOLS.get(name)
    if spec is None:
        return _fail(
            chain,
            {
                "error": "unknown_tool",
                "tool": name,
                "available": [s.name for s in tools_for(role)],
            },
        )
    if spec.role is not None and spec.role is not role:
        return _fail(
            chain,
            {
                "error": "not_owned",
                "tool": name,
                "owner": str(spec.role),
                "your_role": str(role),
                "detail": (
                    f"{name} edits ops owned by the {spec.role} specialist. "
                    "Report the problem in your reason and finish; the supervisor "
                    "routes to that specialist."
                ),
                "available": [s.name for s in tools_for(role)],
            },
        )
    try:
        return spec.apply(chain, arguments)
    except _BadArgsError as exc:
        return _fail(chain, exc.payload)
    except BoundViolation as exc:
        payload = exc.as_tool_error()
        payload["hint"] = "Every bound is in the tool schema; choose a value inside it."
        return _fail(chain, payload)


def describe_owned(chain: Chain, role: Role) -> str:
    """The part of the chain this role controls, or a note that it is empty."""
    kinds = OWNED_OPS[role]
    mine = [op for op in chain.ops if OpKind(op.kind) in kinds]
    if not mine:
        return "(you have no ops in the chain yet)"
    return Chain(ops=tuple(mine)).describe().removeprefix("source -> ").removesuffix(" -> render")


def foreign_ops(chain: Chain, role: Role) -> str:
    """A one-line summary of what the other specialists have done.

    Enough for a specialist to know that a limiter is in circuit and is eating
    its transients; not enough to invite it to try to fix that itself.
    """
    kinds = OWNED_OPS[role]
    others = [op for op in chain.ops if OpKind(op.kind) not in kinds]
    if not others:
        return "(no other specialist has added anything)"
    return ", ".join(f"{op.kind} (owned by {OWNER_OF_OP[OpKind(op.kind)]})" for op in others)
