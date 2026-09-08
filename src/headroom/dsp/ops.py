"""Typed, bounded processing operations.

Two rules make this layer safe to hand to an agent:

1. **Bounds are enforced, never clamped.** A request for +40 dB of gain is
   rejected with a structured error naming the bound and the value. Silently
   clamping would tell the agent its move succeeded and then show it a
   measurement that does not match, which is the hardest kind of bug to
   attribute in a feedback loop.
2. **Ops are immutable values.** Editing an op returns a new op. The chain is
   a declarative description re-applied to the original source every render,
   so nothing accumulates.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from enum import IntEnum, StrEnum
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class OpKind(StrEnum):
    GAIN = "gain"
    EQ = "eq"
    COMPRESSOR = "compressor"
    EXPANDER = "expander"
    STEREO_WIDTH = "stereo_width"
    LIMITER = "limiter"


class ChainStage(IntEnum):
    """Canonical mastering order. The supervisor repositions out-of-order ops
    to these stages and logs the move: an agent that puts the limiter first
    should be corrected by the system rather than trusted."""

    GAIN_STAGING = 0
    EQ = 1
    DYNAMICS = 2
    STEREO = 3
    LIMITER = 4


STAGE_OF: Final[dict[OpKind, ChainStage]] = {
    OpKind.GAIN: ChainStage.GAIN_STAGING,
    OpKind.EQ: ChainStage.EQ,
    OpKind.COMPRESSOR: ChainStage.DYNAMICS,
    OpKind.EXPANDER: ChainStage.DYNAMICS,
    OpKind.STEREO_WIDTH: ChainStage.STEREO,
    OpKind.LIMITER: ChainStage.LIMITER,
}


class BoundViolationError(ValueError):
    """A parameter fell outside its hard bound.

    Carries the field, the offending value and the bound so the agent receives
    a machine-readable reason rather than prose it has to parse.
    """

    def __init__(self, violations: list[dict[str, object]]) -> None:
        self.violations = violations
        detail = "; ".join(
            f"{v['field']}={v['value']!r} violates {v['constraint']}" for v in violations
        )
        super().__init__(f"parameter out of bounds: {detail}")

    def as_tool_error(self) -> dict[str, object]:
        """Structured payload for a tool result."""
        return {"error": "bound_violation", "violations": self.violations}


def _rethrow(exc: ValidationError) -> BoundViolationError:
    violations: list[dict[str, object]] = []
    for err in exc.errors():
        violations.append(
            {
                "field": ".".join(str(p) for p in err["loc"]),
                "value": err.get("input"),
                "constraint": err["msg"],
                "type": err["type"],
            }
        )
    return BoundViolationError(violations)


# --- bound aliases, single source of truth for the whole system ---------------
GainDb = Annotated[float, Field(ge=-24.0, le=24.0)]
EqGainDb = Annotated[float, Field(ge=-18.0, le=18.0)]
EqFreqHz = Annotated[float, Field(ge=20.0, le=20000.0)]
EqQ = Annotated[float, Field(ge=0.1, le=10.0)]
ThresholdDb = Annotated[float, Field(ge=-60.0, le=0.0)]
CompRatio = Annotated[float, Field(ge=1.0, le=20.0)]
ExpRatio = Annotated[float, Field(ge=1.0, le=8.0)]
AttackMs = Annotated[float, Field(ge=0.1, le=500.0)]
ReleaseMs = Annotated[float, Field(ge=1.0, le=3000.0)]
MakeupDb = Annotated[float, Field(ge=-12.0, le=12.0)]
Width = Annotated[float, Field(ge=0.0, le=2.0)]
CeilingDbtp = Annotated[float, Field(ge=-3.0, le=-0.1)]
BandIndex = Annotated[int, Field(ge=0, le=8)]


class _Op(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def stage(self) -> ChainStage:
        return STAGE_OF[OpKind(self.kind)]  # type: ignore[attr-defined]


class EqBand(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shape: Literal["peak", "low_shelf", "high_shelf", "hpf", "lpf"]
    freq_hz: EqFreqHz
    gain_db: EqGainDb = 0.0
    q: EqQ = 0.707


class GainOp(_Op):
    kind: Literal[OpKind.GAIN] = OpKind.GAIN
    gain_db: GainDb


class EqOp(_Op):
    kind: Literal[OpKind.EQ] = OpKind.EQ
    bands: tuple[EqBand, ...] = Field(min_length=1, max_length=8)


class CompressorOp(_Op):
    """No knee parameter: ``pedalboard.Compressor`` does not expose one, and
    faking a soft knee in front of a hard-knee compressor would make the
    rendered result disagree with the declared parameters."""

    kind: Literal[OpKind.COMPRESSOR] = OpKind.COMPRESSOR
    threshold_db: ThresholdDb
    ratio: CompRatio
    attack_ms: AttackMs = 10.0
    release_ms: ReleaseMs = 100.0
    makeup_db: MakeupDb = 0.0


class ExpanderOp(_Op):
    """Downward expansion, to restore crest factor. There is no expander in
    ``pedalboard``, and without one the ``over_compress`` degradation is
    unrecoverable by construction -- a compressor cannot undo compression."""

    kind: Literal[OpKind.EXPANDER] = OpKind.EXPANDER
    threshold_db: ThresholdDb
    ratio: ExpRatio
    attack_ms: AttackMs = 5.0
    release_ms: ReleaseMs = 80.0


class StereoWidthOp(_Op):
    kind: Literal[OpKind.STEREO_WIDTH] = OpKind.STEREO_WIDTH
    width: Width
    #: ``None`` widens every band. An index restricts the move to one band,
    #: which is what makes "tighten the lows, widen the top" expressible.
    band: BandIndex | None = None


class LimiterOp(_Op):
    """``ceiling_dbtp`` is a true-peak ceiling. ``pedalboard.Limiter`` takes a
    sample-peak threshold with no lookahead or oversampling and cannot honour
    one, so the renderer uses an oversampled implementation instead."""

    kind: Literal[OpKind.LIMITER] = OpKind.LIMITER
    ceiling_dbtp: CeilingDbtp = -1.0
    release_ms: ReleaseMs = 50.0


Op = GainOp | EqOp | CompressorOp | ExpanderOp | StereoWidthOp | LimiterOp

OP_MODELS: Final[dict[OpKind, type[BaseModel]]] = {
    OpKind.GAIN: GainOp,
    OpKind.EQ: EqOp,
    OpKind.COMPRESSOR: CompressorOp,
    OpKind.EXPANDER: ExpanderOp,
    OpKind.STEREO_WIDTH: StereoWidthOp,
    OpKind.LIMITER: LimiterOp,
}


def make_op(kind: OpKind | str, **params: object) -> Op:
    """Construct an op, converting validation failures into :class:`BoundViolationError`."""
    model = OP_MODELS[OpKind(kind)]
    try:
        return model(**params)  # type: ignore[return-value]
    except ValidationError as exc:
        raise _rethrow(exc) from exc


def edit_op(op: Op, **params: object) -> Op:
    """Return a copy with ``params`` replaced. Revision, not correction-stacking:
    a 240 Hz cut can be *changed* from -3 dB to -1.5 dB instead of having a
    +1.5 dB boost stacked on top, which would be a different, worse filter."""
    merged = {**op.model_dump(), **params}
    try:
        # Revalidate rather than model_copy(update=...), which bypasses
        # validation and would let an out-of-bounds edit through silently.
        return type(op).model_validate(merged)
    except ValidationError as exc:
        raise _rethrow(exc) from exc


# --- typed constructors -------------------------------------------------------
# One per op kind, as SPEC 7.1 specifies. These exist so callers get a concrete
# type rather than the ``Op`` union: ``op_gain(-2.0).gain_db`` type-checks,
# whereas ``make_op(OpKind.GAIN, gain_db=-2.0).gain_db`` cannot, because the
# union has no such attribute. ``make_op`` remains for the tool layer, where the
# kind arrives as a string from the model and the concrete type is not known
# statically.


def op_gain(gain_db: float) -> GainOp:
    try:
        return GainOp(gain_db=gain_db)
    except ValidationError as exc:
        raise _rethrow(exc) from exc


def op_eq_band(
    shape: str = "peak", freq_hz: float = 1000.0, gain_db: float = 0.0, q: float = 0.707
) -> EqBand:
    """A single filter band. Exists so no caller has to construct :class:`EqBand`
    directly: the raw model raises pydantic's ``ValidationError``, which the tool
    layer does not recognise, so an out-of-range filter would escape as an
    exception instead of arriving as a structured refusal."""
    try:
        return EqBand(shape=shape, freq_hz=freq_hz, gain_db=gain_db, q=q)  # type: ignore[arg-type]
    except ValidationError as exc:
        raise _rethrow(exc) from exc


def op_eq(bands: Sequence[EqBand | Mapping[str, object]]) -> EqOp:
    try:
        return EqOp(
            bands=tuple(b if isinstance(b, EqBand) else EqBand.model_validate(b) for b in bands)
        )
    except ValidationError as exc:
        raise _rethrow(exc) from exc


def op_compressor(
    threshold_db: float,
    ratio: float,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    makeup_db: float = 0.0,
) -> CompressorOp:
    try:
        return CompressorOp(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
            makeup_db=makeup_db,
        )
    except ValidationError as exc:
        raise _rethrow(exc) from exc


def op_expander(
    threshold_db: float,
    ratio: float,
    attack_ms: float = 5.0,
    release_ms: float = 80.0,
) -> ExpanderOp:
    try:
        return ExpanderOp(
            threshold_db=threshold_db, ratio=ratio, attack_ms=attack_ms, release_ms=release_ms
        )
    except ValidationError as exc:
        raise _rethrow(exc) from exc


def op_stereo_width(width: float, band: int | None = None) -> StereoWidthOp:
    try:
        return StereoWidthOp(width=width, band=band)
    except ValidationError as exc:
        raise _rethrow(exc) from exc


def op_limiter(ceiling_dbtp: float = -1.0, release_ms: float = 50.0) -> LimiterOp:
    try:
        return LimiterOp(ceiling_dbtp=ceiling_dbtp, release_ms=release_ms)
    except ValidationError as exc:
        raise _rethrow(exc) from exc
