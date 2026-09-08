"""The supervisor: deterministic routing over four specialists.

There is no model here, and that is the point. Routing is the one part of a
multi-agent system that is genuinely easy -- pick the specialist that owns the
largest weighted error -- and spending a model call to do arithmetic would add
cost, latency and a failure mode in exchange for nothing. What the supervisor
does own is the part that is not easy:

**Rerouting.** A specialist that has failed to improve twice running is routed
around, so one stuck role cannot spend the whole render budget. It is not
banned: if its dimensions are still the worst after another role has moved, it
gets another turn with fresh measurements.

**Handing the turn back.** A specialist may answer "nothing I own can move
this". That is a real answer, it costs no render, and the supervisor tries the
next role in the same round. Bounded, because for an LLM-backed specialist it
is not free in tokens.

**Refusing an empty proposal.** If a specialist's edits leave the chain
audibly identical, rendering it would burn a render to learn nothing. The
supervisor detects it from the fingerprint and reroutes instead.

**Correcting the order.** Ops are canonicalized to mastering order and every
move is recorded. A specialist that inserts a limiter before the EQ is
corrected by the system rather than trusted, and the correction rate is a
number in the trace instead of a thing nobody checked.

Everything above is a pure function of loop state, memory and the specialists'
returned turns, so the whole control surface is testable with scripted
specialists and no API key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from headroom.control.loop import Proposal
from headroom.control.state import LoopState
from headroom.dsp.chain import Chain

from . import briefing as briefing_mod
from .memory import WorkingMemory
from .roles import OWNED_FEATURES, Role
from .specialist import Specialist, SpecialistTurn

NAME: Final[str] = "agent"


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    #: Consecutive settled turns without improvement before a role is routed
    #: around. Two, because one bad move is information and two is a pattern.
    strike_limit: int = 2
    #: Roles consulted in one round before the supervisor gives up on it. Each
    #: consultation is free in renders but not in tokens, so this is the knob
    #: that bounds the cost of politeness.
    max_consults_per_round: int = 3
    #: How many times a role may hand the turn straight back before it is
    #: dropped for the rest of the run.
    stop_limit: int = 2


DEFAULT_CONFIG: Final[SupervisorConfig] = SupervisorConfig()


@dataclass(frozen=True, slots=True)
class RoleLoad:
    role: Role
    #: Summed weighted contribution of the out-of-tolerance dimensions it owns.
    contribution: float
    n_out: int
    worst: str


def role_loads(state: LoopState) -> list[RoleLoad]:
    """Rank roles by the weighted error they are responsible for.

    Weighted contribution rather than a count: eleven stereo dimensions each
    one tolerance out matter less in total than integrated loudness being six
    tolerances out, and the family weights already encode that judgement.
    """
    loads: list[RoleLoad] = []
    for role in Role:
        owned = set(OWNED_FEATURES[role])
        mine = [
            d for d in state.distance.breakdown.values() if d.name in owned and not d.in_tolerance
        ]
        if not mine:
            continue
        worst = max(mine, key=lambda d: abs(d.scaled))
        loads.append(
            RoleLoad(
                role=role,
                contribution=sum(d.contribution for d in mine),
                n_out=len(mine),
                worst=f"{worst.name} {worst.scaled:+.2f} tol",
            )
        )
    loads.sort(key=lambda load: -load.contribution)
    return loads


@dataclass
class Supervisor:
    """Routes rounds to specialists and returns one proposal per round."""

    specialists: Mapping[Role, Specialist]
    config: SupervisorConfig = DEFAULT_CONFIG
    memory: WorkingMemory = field(default_factory=WorkingMemory)
    #: Times each role handed the turn straight back.
    stops: dict[Role, int] = field(default_factory=dict)
    #: Ops repositioned into canonical order across the whole run.
    repositions: int = 0
    #: Rounds where a consulted specialist produced no audible change.
    empty_rounds: int = 0
    consults: int = 0

    def propose(self, state: LoopState) -> Proposal:
        if state.step_index > 0:
            self.memory.settle(state.distance.score)

        skipped: list[str] = []
        turns: list[SpecialistTurn] = []

        for load in role_loads(state):
            if len(turns) >= self.config.max_consults_per_round:
                break
            role = load.role
            if self.stops.get(role, 0) >= self.config.stop_limit:
                skipped.append(f"{role} dropped after {self.stops[role]} passes")
                continue
            if self.memory.strikes_for(role) >= self.config.strike_limit:
                skipped.append(f"{role} rerouted after {self.memory.strikes_for(role)} misses")
                continue

            brief = briefing_mod.build(state, self.memory, role)
            turn = self.specialists[role](brief, state.chain)
            turns.append(turn)
            self.consults += 1
            self.memory.ownership_violations += turn.ownership_violations
            self.memory.rejected_calls += turn.rejected

            if turn.stop or not turn.actions:
                self.stops[role] = self.stops.get(role, 0) + 1
                skipped.append(f"{role} passed: {turn.rationale or 'nothing to do'}")
                continue

            ordered, moves = turn.chain.canonical()
            self.repositions += len(moves)
            if ordered.fingerprint() == state.chain.fingerprint():
                self.empty_rounds += 1
                skipped.append(f"{role} produced no audible change")
                continue

            targeted = tuple(d.name for d in brief.mine)
            self.memory.open(
                step=state.step_index,
                role=role,
                targeted=targeted,
                actions=turn.actions,
                rationale=turn.rationale,
                score_before=state.distance.score,
            )
            return _proposal(
                chain=ordered,
                turn=turn,
                load=load,
                turns=turns,
                skipped=skipped,
                n_repositioned=len(moves),
            )

        return _give_up(state, turns, skipped)

    def as_proposer(self) -> Any:
        """Adapt to the shared loop's ``Proposer`` signature."""
        return self.propose

    def stats(self) -> dict[str, Any]:
        return {
            **self.memory.stats(),
            "consults": self.consults,
            "repositions": self.repositions,
            "empty_rounds": self.empty_rounds,
            "passes": {str(r): n for r, n in self.stops.items()},
        }


def _totals(turns: list[SpecialistTurn]) -> dict[str, float]:
    return {
        "in": float(sum(t.input_tokens for t in turns)),
        "out": float(sum(t.output_tokens for t in turns)),
        "cache": float(sum(t.cache_read_tokens for t in turns)),
        "cost": float(sum(t.cost_usd for t in turns)),
    }


def _proposal(
    *,
    chain: Chain,
    turn: SpecialistTurn,
    load: RoleLoad,
    turns: list[SpecialistTurn],
    skipped: list[str],
    n_repositioned: int,
) -> Proposal:
    totals = _totals(turns)
    note_parts = [
        f"role={turn.role}",
        f"targeting {load.worst}",
        f"{len(turn.actions)} edit(s): {'; '.join(turn.actions)}",
    ]
    if turn.rationale:
        note_parts.append(turn.rationale)
    if n_repositioned:
        note_parts.append(f"{n_repositioned} op(s) repositioned to canonical order")
    if turn.rejected:
        note_parts.append(f"{turn.rejected} call(s) rejected")
    if skipped:
        note_parts.append("skipped: " + "; ".join(skipped))
    return Proposal(
        chain=chain,
        action=turn.dominant_action,
        note=" | ".join(note_parts),
        input_tokens=int(totals["in"]),
        output_tokens=int(totals["out"]),
        cache_read_tokens=int(totals["cache"]),
        cost_usd=totals["cost"],
        role=str(turn.role),
        n_edits=len(turn.actions),
        n_rejected=sum(t.rejected for t in turns),
    )


def _give_up(state: LoopState, turns: list[SpecialistTurn], skipped: list[str]) -> Proposal:
    totals = _totals(turns)
    reason = "; ".join(skipped) if skipped else "every dimension is inside tolerance"
    return Proposal(
        chain=state.chain,
        give_up=True,
        note=f"no specialist could act: {reason}",
        input_tokens=int(totals["in"]),
        output_tokens=int(totals["out"]),
        cache_read_tokens=int(totals["cache"]),
        cost_usd=totals["cost"],
        n_rejected=sum(t.rejected for t in turns),
    )
