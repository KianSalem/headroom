"""What a specialist is shown, and nothing else.

Context isolation is the load-bearing claim of a multi-specialist design, so
it is implemented here as a hard filter rather than an instruction. A
specialist's briefing contains the dimensions it owns, the ops it owns, one
line about what the other roles have done, and its own recent attempts. The
other 18 to 26 scored dimensions are not in the prompt at all.

That is measurable rather than rhetorical: input tokens per call are recorded
in the trace, so the cost of isolation -- four calls instead of one -- can be
weighed against the cost of a single agent carrying all 28 dimensions plus a
growing transcript.

Everything here is deterministic. The briefing is a pure function of loop
state, memory and role, which is what lets the whole agent be tested against a
scripted specialist with no API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from headroom.control.state import LoopState
from headroom.target.distance import FeatureDelta

from . import tools
from .memory import WorkingMemory
from .roles import OWNED_FEATURES, ROLE_BRIEF, Role

#: Out-of-tolerance dimensions listed in full. Beyond this the tail is
#: summarized: a specialist that fixes its four worst dimensions has done its
#: job for the turn, and the rest will be re-measured anyway.
MAX_LISTED: Final[int] = 6

_RELATION: Final[dict[str, str]] = {
    "both": "match",
    "max": "stay at or below",
    "min": "stay at or above",
}


@dataclass(frozen=True, slots=True)
class Briefing:
    role: Role
    step: int
    renders_left: int
    render_budget: int
    score: float
    initial_score: float
    #: Damping from the critic. Below 1.0 means a previous move overshot or
    #: oscillated and this turn's move should be proportionally smaller.
    step_scale: float
    oscillating: bool
    mine: tuple[FeatureDelta, ...]
    n_mine_out: int
    programme_level_db: float
    own_chain: str
    other_chain: str
    own_history: str
    other_history: str
    frozen: tuple[str, ...]
    tried_and_failed: tuple[str, ...]

    def render(self) -> str:
        return "\n".join(self._blocks())

    def _blocks(self) -> list[str]:
        out = [
            f"## Round {self.step + 1}",
            f"Renders left in the whole run: {self.renders_left} of {self.render_budget}. "
            f"Your edits this round cost exactly one render, however many you make.",
            f"Total distance across all 28 dimensions: {self.score:.4f} "
            f"(started at {self.initial_score:.4f}; 0 is a perfect match).",
            f"Programme level: {self.programme_level_db:.1f} LUFS integrated. "
            f"Set any threshold relative to this, not to 0 dBFS.",
            "",
            "## Your dimensions",
            "delta is current minus target. scaled is delta in tolerance units, so "
            "+2.0 means two tolerances too high and anything within +/-1.0 is already "
            "acceptable. Fix the largest magnitudes first.",
            "",
            self._table(),
        ]
        if self.n_mine_out > len(self.mine):
            out.append(
                f"({self.n_mine_out - len(self.mine)} further dimensions of yours are out "
                f"of tolerance by smaller amounts.)"
            )
        out += [
            "",
            "## Your processing",
            self.own_chain,
            "",
            "## Elsewhere in the chain (not yours to edit)",
            self.other_chain,
            "",
            "## Your attempts on this track",
            self.own_history,
            "",
            "## What the other specialists last did",
            self.other_history,
        ]
        if self.step_scale < 1.0:
            out += [
                "",
                "## Step size",
                f"The critic has damped this round to {self.step_scale:.3f} of a full "
                f"correction, because earlier moves overshot or reversed themselves. "
                f"Make a move about {self.step_scale:.0%} of the size you otherwise would.",
            ]
        if self.oscillating:
            out += [
                "",
                "## Oscillation warning",
                "The distance has stopped falling. Repeating a move you have already "
                "tried, or reversing your last one, will end the run. Try a different "
                "dimension or a different op.",
            ]
        if self.frozen:
            out += [
                "",
                "## Frozen",
                "These dimensions reversed direction too often and are locked for the "
                "rest of the run; leave them alone: " + ", ".join(self.frozen),
            ]
        if self.tried_and_failed:
            out += [
                "",
                "## Already tried and made it worse",
                ", ".join(self.tried_and_failed),
            ]
        return out

    def _table(self) -> str:
        if not self.mine:
            return "Every dimension you own is inside tolerance. Call finish."
        header = (
            f"{'dimension':18s} {'current':>10s} {'':2s} {'target':>10s} "
            f"{'unit':>9s} {'delta':>8s} {'scaled':>8s}"
        )
        rows = [header, "-" * len(header)]
        for d in self.mine:
            rows.append(
                f"{d.name:18s} {d.current:+10.3f} {_RELATION[d.direction][:2]:>2s} "
                f"{d.target:+10.3f} {d.unit:>9s} {d.delta:+8.3f} {d.scaled:+8.2f}"
            )
        return "\n".join(rows)


def build(state: LoopState, memory: WorkingMemory, role: Role) -> Briefing:
    """Filter loop state down to one role's view."""
    owned = set(OWNED_FEATURES[role])
    mine = [
        d
        for d in state.distance.breakdown.values()
        if d.name in owned and not d.in_tolerance and d.name not in state.frozen_params
    ]
    mine.sort(key=lambda d: -abs(d.scaled))
    level = getattr(state.features, "lufs_integrated", -14.0)
    return Briefing(
        role=role,
        step=state.step_index,
        renders_left=state.renders_left,
        render_budget=state.render_budget,
        score=state.distance.score,
        initial_score=state.initial_distance,
        step_scale=state.step_scale,
        oscillating=state.oscillating,
        mine=tuple(mine[:MAX_LISTED]),
        n_mine_out=len(mine),
        programme_level_db=float(level),
        own_chain=tools.describe_owned(state.chain, role),
        other_chain=tools.foreign_ops(state.chain, role),
        own_history=memory.render_own(role),
        other_history=memory.render_others(role),
        frozen=tuple(sorted(state.frozen_params & owned)),
        tried_and_failed=tuple(sorted(a for a in state.tried_and_failed if _looks_like(a, role))),
    )


_ACTION_PREFIX: Final[dict[Role, tuple[str, ...]]] = {
    Role.EQ: ("eq.",),
    Role.DYNAMICS: ("comp.", "exp."),
    Role.STEREO: ("width.",),
    Role.LOUDNESS: ("gain.", "limiter."),
}


def _looks_like(action: str, role: Role) -> bool:
    return action.startswith(_ACTION_PREFIX[role])


def role_system_prompt(role: Role) -> str:
    """The static half of the prompt, cached across every call for this role.

    It contains no measurements and no chain state, so it is byte-identical for
    every call a role makes for the whole evaluation. That is what makes prompt
    caching worth having here: the tool schemas and the standing instructions
    are the large, unchanging part, and the per-turn briefing is the small one.
    """
    return "\n".join(
        [
            "You are one of four specialists in an automated audio mastering system.",
            ROLE_BRIEF[role],
            "",
            "How the system works:",
            "- Processing is a declarative chain re-applied to the original audio every "
            "time. You are editing a description, not damaging a file, so a value you "
            "set can be revised freely rather than corrected on top of.",
            "- Your setters are absolute. set_gain(gain_db=-2.0) means the gain op is "
            "-2 dB, not 2 dB quieter than now.",
            "- Make every edit you want this round, then call finish. All of them are "
            "rendered and measured together as one render, so a coordinated set of "
            "moves costs no more than a single timid one.",
            "- Every bound is in the tool schema. A value outside it is rejected with "
            "the bound named; that costs you nothing but the round-trip, so stay "
            "inside it and do not guess.",
            "- You will be told what you tried before and whether it helped. You get no "
            "conversation history beyond that.",
            "",
            "How to decide:",
            "- Work on the largest scaled magnitude first. A dimension inside +/-1.0 "
            "tolerance is done; do not polish it.",
            "- Prefer one confident correction of the right size over a sequence of "
            "cautious ones. The render budget is small.",
            "- If a previous move of yours made the distance worse, do not repeat it in "
            "the same direction; the amount was wrong or the op was.",
            "- If nothing you own is out of tolerance, or nothing you own can move what "
            "is, call finish and say so. Handing the turn back is a real answer and "
            "the supervisor will route elsewhere.",
        ]
    )
