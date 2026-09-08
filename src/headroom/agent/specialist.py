"""The specialist interface and its two model-free implementations.

A specialist is anything that turns a :class:`~headroom.agent.briefing.Briefing`
and a chain into a new chain plus a record of how it got there. The interface
is deliberately that narrow: the supervisor, the tool layer, the memory, the
critic and the trace are all written against it, so an LLM-backed specialist
and a scripted one are interchangeable and every path through the architecture
can be exercised with no API key and no cost.

Two implementations live here.

:class:`ScriptedSpecialist` replays a fixed list of tool calls. It exists so
the failure modes that matter -- oscillation, bound saturation, a call to
another role's tool, a proposal that changes nothing -- are reproducible tests
rather than things hoped for in a live run.

:class:`ProportionalSpecialist` is a real policy: it reads the briefing and
maps each dimension it owns onto the tool that moves it, using **the same
correction constants as the heuristic baseline, imported rather than copied**.
That makes it a controlled ablation. It differs from the heuristic in one
designed respect -- it may make several coordinated edits per render, where
the heuristic makes one -- so the gap between them measures the architecture,
and the gap between it and the LLM-backed agent measures the model. Neither
number is interpretable without the other.

One incidental difference is recorded here rather than hidden: the two stereo
corrections that are not per-band (``correlation_z`` and ``mono_compat_db``)
share the heuristic's constants and clamps but are expressed as a width change
in dB here, where the heuristic applies them as a linear factor. The per-band
``width_i`` corrections are identical. The tools take absolute widths, so a dB
step is the natural unit on this side; the mapping was left as-is once results
were recorded rather than re-running every table to remove a difference that
only touches two of eleven stereo dimensions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, cast

from headroom.baselines.heuristic import BAND_K, CORRECTIVE_Q, DYNAMICS_K, GAIN_K, WIDTH_K
from headroom.dsp.chain import Chain
from headroom.dsp.ops import (
    CompressorOp,
    EqOp,
    ExpanderOp,
    GainOp,
    LimiterOp,
    OpKind,
    StereoWidthOp,
)

from . import tools
from .briefing import Briefing
from .roles import Role

#: Edits one specialist may make in a single round. Bounded so a runaway policy
#: cannot write an unbounded chain, and low enough that a round stays legible
#: in the trace.
MAX_EDITS_PER_TURN: Final[int] = 6


@dataclass(frozen=True, slots=True)
class ToolRecord:
    name: str
    arguments: dict[str, Any]
    ok: bool
    action: str
    payload: dict[str, Any]

    @property
    def error(self) -> str:
        return "" if self.ok else str(self.payload.get("error", "unknown"))


@dataclass(frozen=True, slots=True)
class SpecialistTurn:
    """One specialist's complete contribution to one round."""

    role: Role
    chain: Chain
    calls: tuple[ToolRecord, ...] = ()
    rationale: str = ""
    #: The specialist has nothing useful left to try. The supervisor routes
    #: elsewhere rather than treating this as a failure.
    stop: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    n_llm_calls: int = 0

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(c.action for c in self.calls if c.ok and c.action)

    @property
    def ownership_violations(self) -> int:
        return sum(1 for c in self.calls if c.error == "not_owned")

    @property
    def rejected(self) -> int:
        return sum(1 for c in self.calls if not c.ok and c.error != "no_change")

    @property
    def dominant_action(self) -> str:
        """The largest single move, used as the round's action string.

        The critic's oscillation detector reads one ``"<param> <signed delta>"``
        per step. A round with several edits is summarized by its biggest,
        which is the one a sign-flip would be about; the full list is in the
        note and the chain itself is in the trace.
        """
        scored: list[tuple[float, str]] = []
        for action in self.actions:
            _, _, tail = action.rpartition(" ")
            try:
                scored.append((abs(float(tail)), action))
            except ValueError:
                continue
        return max(scored)[1] if scored else ""


class Specialist(Protocol):
    """Structural interface. Implementations need a role and a call."""

    role: Role

    def __call__(self, briefing: Briefing, chain: Chain) -> SpecialistTurn: ...


def run_calls(
    role: Role, chain: Chain, calls: Sequence[tuple[str, Mapping[str, Any]]]
) -> tuple[Chain, tuple[ToolRecord, ...], bool]:
    """Apply a batch of calls in order, keeping whatever succeeds.

    Returns the resulting chain, one record per call, and whether ``finish``
    was reached. A failed call leaves the chain untouched and does not abort
    the batch: partial success is the honest outcome, and the records say
    exactly which parts landed.
    """
    records: list[ToolRecord] = []
    finished = False
    for name, arguments in calls:
        outcome = tools.apply_call(chain, role, name, arguments)
        chain = outcome.chain
        records.append(
            ToolRecord(
                name=name,
                arguments=dict(arguments),
                ok=outcome.ok,
                action=outcome.action,
                payload=outcome.payload,
            )
        )
        if outcome.finished:
            finished = True
            break
    return chain, tuple(records), finished


# --- scripted ----------------------------------------------------------------

Script = Sequence[Sequence[tuple[str, Mapping[str, Any]]]]


@dataclass
class ScriptedSpecialist:
    """Replays a fixed script, one entry per turn.

    Once the script runs out it stops, which is what a real specialist should
    also do when it has nothing left. Tests use it to drive the supervisor into
    a specific state -- an oscillation, a saturated bound, a rejected tool --
    without a network call.
    """

    role: Role
    script: Script
    rationale: str = "scripted"
    turn: int = 0

    def __call__(self, briefing: Briefing, chain: Chain) -> SpecialistTurn:
        if self.turn >= len(self.script):
            return SpecialistTurn(
                role=self.role, chain=chain, stop=True, rationale="script exhausted"
            )
        calls = self.script[self.turn]
        self.turn += 1
        new_chain, records, _ = run_calls(self.role, chain, calls)
        return SpecialistTurn(
            role=self.role,
            chain=new_chain,
            calls=records,
            rationale=self.rationale,
            stop=not any(r.ok and r.action for r in records),
        )


# --- proportional ------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _existing_band_gain(chain: Chain, freq_hz: float) -> float:
    for op in chain.of_kind(OpKind.EQ):
        for band in cast(EqOp, op).bands:
            if band.shape == "peak" and abs(band.freq_hz - freq_hz) < 1.0:
                return band.gain_db
    return 0.0


def _existing_width(chain: Chain, band: int | None) -> float:
    for op in chain.of_kind(OpKind.STEREO_WIDTH):
        width_op = cast(StereoWidthOp, op)
        if width_op.band == band:
            return width_op.width
    return 1.0


def _existing_gain(chain: Chain) -> float:
    ops = chain.of_kind(OpKind.GAIN)
    return cast(GainOp, ops[0]).gain_db if ops else 0.0


@dataclass(frozen=True, slots=True)
class PlannedEdit:
    """One requested move, expressed as a delta so conflicts can be resolved.

    ``slot`` is an identity key naming the control this edit writes. Several
    dimensions can name the same slot, and they must be *combined* rather than
    picked between -- found the hard way. ``correlation_z`` and
    ``mono_compat_db`` are two views of the same global width control and they
    routinely pull in opposite directions. Keeping whichever was larger this
    turn made the surviving dimension alternate between turns, so the width
    went up, down, up, down and the critic correctly killed the run for
    oscillation. The specialist was not confused; the collapse rule was wrong.

    Averaging the requested deltas is the fix, and it is right in both regimes.
    Two dimensions fighting produce their net request, which is small and
    stable. Two near-duplicate dimensions -- ``crest_factor_db`` and
    ``crest_short_p50`` measure almost the same thing -- produce roughly what
    either asked for alone, where summing would have doubled it and started
    the oscillation from the other end.
    """

    slot: str
    tool: str
    #: The tool argument carrying the absolute value, recomputed after merging.
    param: str
    current: float
    delta: float
    low: float
    high: float
    reason: str
    extra: dict[str, Any] = field(default_factory=dict)


Plan = list[PlannedEdit]


def _plan_eq(briefing: Briefing, chain: Chain) -> Plan:
    plan: Plan = []
    for d in briefing.mine:
        if not d.name.startswith("band_clr_"):
            continue  # flatness has no single filter that moves it predictably
        band = int(d.name.rsplit("_", 1)[1])
        freq = tools.band_center_hz(band)
        plan.append(
            PlannedEdit(
                slot=f"eq:{band}",
                tool="set_eq_band",
                param="gain_db",
                current=_existing_band_gain(chain, freq),
                delta=-d.delta * BAND_K * briefing.step_scale,
                low=-18.0,
                high=18.0,
                reason=f"{d.name} is {d.scaled:+.2f} tol",
                extra={"band_index": band, "q": CORRECTIVE_Q},
            )
        )
    return plan


def _plan_stereo(briefing: Briefing, chain: Chain) -> Plan:
    plan: Plan = []
    for d in briefing.mine:
        if d.name.startswith("width_"):
            band: int | None = int(d.name.rsplit("_", 1)[1])
            correction_db = -d.delta * WIDTH_K * briefing.step_scale
        elif d.name == "correlation_z":
            # More correlation means a narrower image, so too much correlation
            # wants more width.
            band = None
            correction_db = _clamp(d.delta * 0.25 * briefing.step_scale, -0.5, 0.5) * 10.0
        elif d.name == "mono_compat_db":
            band = None
            correction_db = -_clamp(abs(d.delta) * 0.1 * briefing.step_scale, 0.0, 0.5) * 10.0
        else:
            continue
        # Width is a ratio, so the correction is multiplicative: the delta is
        # expressed in width units against the current value.
        current = _existing_width(chain, band)
        want = current * float(10.0 ** (correction_db / 20.0))
        plan.append(
            PlannedEdit(
                slot=f"width:{band}",
                tool="set_stereo_width",
                param="width",
                current=current,
                delta=want - current,
                low=0.0,
                high=2.0,
                reason=f"{d.name} is {d.scaled:+.2f} tol",
                extra={"band_index": band},
            )
        )
    return plan


def _plan_dynamics(briefing: Briefing, chain: Chain) -> Plan:
    plan: Plan = []
    level = briefing.programme_level_db
    for d in briefing.mine:
        if d.name not in ("crest_factor_db", "crest_short_p50", "lra"):
            continue  # attack time and percussive ratio have no monotone lever
        if d.delta > 0.0:
            ops = chain.of_kind(OpKind.COMPRESSOR)
            current = cast(CompressorOp, ops[0]).ratio if ops else 1.0
            plan.append(
                PlannedEdit(
                    slot="dyn:comp",
                    tool="set_compressor",
                    param="ratio",
                    current=current,
                    delta=abs(d.delta) * DYNAMICS_K * briefing.step_scale,
                    low=1.0,
                    high=20.0,
                    reason=f"{d.name} is {d.scaled:+.2f} tol, too dynamic",
                    extra={
                        "threshold_db": round(_clamp(level - 6.0, -60.0, 0.0), 2),
                        "attack_ms": 10.0,
                        "release_ms": 120.0,
                    },
                )
            )
        else:
            ops = chain.of_kind(OpKind.EXPANDER)
            current = cast(ExpanderOp, ops[0]).ratio if ops else 1.0
            plan.append(
                PlannedEdit(
                    slot="dyn:exp",
                    tool="set_expander",
                    param="ratio",
                    current=current,
                    delta=abs(d.delta) * DYNAMICS_K * briefing.step_scale,
                    low=1.0,
                    high=8.0,
                    reason=f"{d.name} is {d.scaled:+.2f} tol, over-compressed",
                    extra={
                        "threshold_db": round(_clamp(level + 10.0, -60.0, 0.0), 2),
                        "attack_ms": 5.0,
                        "release_ms": 80.0,
                    },
                )
            )
    return plan


def _plan_loudness(briefing: Briefing, chain: Chain) -> Plan:
    plan: Plan = []
    for d in briefing.mine:
        if d.name == "lufs_integrated":
            plan.append(
                PlannedEdit(
                    slot="loud:gain",
                    tool="set_gain",
                    param="gain_db",
                    current=_existing_gain(chain),
                    delta=-d.delta * GAIN_K * briefing.step_scale,
                    low=-24.0,
                    high=24.0,
                    reason=f"level is {d.scaled:+.2f} tol off",
                )
            )
        elif d.name == "true_peak_dbtp":
            if d.delta > 0.0:
                ops = chain.of_kind(OpKind.LIMITER)
                current = cast(LimiterOp, ops[0]).ceiling_dbtp if ops else 0.0
                plan.append(
                    PlannedEdit(
                        slot="loud:limiter",
                        tool="set_limiter",
                        param="ceiling_dbtp",
                        current=current,
                        delta=_clamp(d.target, -3.0, -0.1) - current,
                        low=-3.0,
                        high=-0.1,
                        reason=f"true peak is {d.scaled:+.2f} tol over the ceiling",
                        extra={"release_ms": 50.0},
                    )
                )
            else:
                plan.append(
                    PlannedEdit(
                        slot="loud:gain",
                        tool="set_gain",
                        param="gain_db",
                        current=_existing_gain(chain),
                        delta=-d.delta * GAIN_K * briefing.step_scale,
                        low=-24.0,
                        high=24.0,
                        reason="peaks sit below the ceiling; level, not limiting",
                    )
                )
    return plan


def _direction_of(edit: PlannedEdit) -> str:
    """The edit's ``"<parameter> <sign>"`` identity, matching what the loop
    records when a move fails."""
    return f"{_ACTION_OF[edit.tool](edit)} {'+' if edit.delta > 0 else '-'}"


#: How each tool's edit maps onto the action parameter name the loop records.
#: Kept beside the planners because the two must agree: a mismatch would mean
#: the specialist silently ignores the failed-move set.
_ACTION_OF: Final[dict[str, Any]] = {
    "set_gain": lambda e: "gain.gain_db",
    "set_limiter": lambda e: "limiter.ceiling",
    "set_eq_band": lambda e: f"eq.band{e.extra['band_index']}",
    "set_compressor": lambda e: "comp.ratio",
    "set_expander": lambda e: "exp.ratio",
    "set_stereo_width": lambda e: (
        "width.global"
        if e.extra.get("band_index") is None
        else f"width.band{e.extra['band_index']}"
    ),
}


def merge_plan(plan: Plan, max_edits: int) -> tuple[list[tuple[str, Mapping[str, Any]]], list[str]]:
    """Combine edits that write the same control, then emit absolute calls.

    Deltas are averaged, not summed: see :class:`PlannedEdit` for why both
    halves of that choice matter. Order is preserved from the briefing, which
    is sorted by descending magnitude, so the largest problem is addressed
    first when the edit budget runs out.
    """
    grouped: dict[str, list[PlannedEdit]] = {}
    for edit in plan:
        grouped.setdefault(edit.slot, []).append(edit)

    calls: list[tuple[str, Mapping[str, Any]]] = []
    reasons: list[str] = []
    for edits in grouped.values():
        head = edits[0]
        net = sum(e.delta for e in edits) / len(edits)
        value = _clamp(head.current + net, head.low, head.high)
        if abs(value - head.current) < 1e-6:
            continue
        reason = "; ".join(dict.fromkeys(e.reason for e in edits))[:180]
        calls.append((head.tool, {**head.extra, head.param: round(value, 4), "reason": reason}))
        reasons.append(reason)
        if len(calls) >= max_edits:
            break
    return calls, reasons


_PLANNERS: Final[dict[Role, Any]] = {
    Role.EQ: _plan_eq,
    Role.DYNAMICS: _plan_dynamics,
    Role.STEREO: _plan_stereo,
    Role.LOUDNESS: _plan_loudness,
}


@dataclass
class ProportionalSpecialist:
    """Deterministic policy over the real tool layer.

    Used as the ``agent-scaffold`` system: the whole architecture, with the
    model replaced by arithmetic. Free to run, and the honest control for any
    claim that the model is doing the work.
    """

    role: Role
    max_edits: int = MAX_EDITS_PER_TURN
    #: Reasons collected per turn, for the trace.
    last_plan: list[str] = field(default_factory=list)

    def __call__(self, briefing: Briefing, chain: Chain) -> SpecialistTurn:
        plan = cast(Plan, _PLANNERS[self.role](briefing, chain))
        # The same rule the heuristic gets, so the ablation stays controlled: a
        # move whose direction already made the distance worse is not retried
        # in that direction.
        blocked = set(briefing.tried_and_failed)
        plan = [e for e in plan if _direction_of(e) not in blocked]
        calls, reasons = merge_plan(plan, self.max_edits)
        self.last_plan = reasons
        if not calls:
            return SpecialistTurn(
                role=self.role,
                chain=chain,
                stop=True,
                rationale="nothing I own maps onto an op that moves it",
            )
        new_chain, records, _ = run_calls(self.role, chain, calls)
        applied = [r for r in records if r.ok and r.action]
        return SpecialistTurn(
            role=self.role,
            chain=new_chain,
            calls=records,
            rationale="; ".join(reasons[: len(applied)]) or "no edit landed",
            stop=not applied,
        )
