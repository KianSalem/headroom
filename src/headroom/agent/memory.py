"""Working memory: tier 1 of SPEC 9, the only tier v1 builds.

The point of writing this down rather than relying on the conversation is that
a specialist is invoked **fresh each turn**, with no message history. That is a
deliberate cost decision -- a growing transcript would re-bill every earlier
measurement on every later call -- and it means the specialist's only knowledge
of what has already been tried is what this module chooses to show it.

Which turns the memory into a design surface rather than a log. Three rules:

**A specialist sees its own attempts in full and everyone else's as one line.**
It needs to know that its own +2 dB at band 6 made things worse. It does not
need eleven stereo dimensions to reason about a tonal problem, and showing them
would undo the context isolation that is the whole argument for splitting the
roles in the first place.

**Outcomes are attributed, not just recorded.** An entry is opened when a move
is proposed and settled on the following turn, once the render has been
measured, so what the specialist reads is "that move cost you 0.03" rather than
a list of moves and a list of scores it has to correlate itself.

**It is bounded.** The last few entries per role, not the whole run. An
unbounded scratchpad would grow the prompt without bound and re-introduce the
cost the fresh-call design exists to avoid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

from .roles import Role

#: Entries per role rendered into a prompt. Four is enough to see a
#: boost-cut-boost cycle, which is the pattern that matters most.
RECENT_PER_ROLE: Final[int] = 4


@dataclass
class MemoryEntry:
    step: int
    role: Role
    #: Scored feature names the supervisor routed on.
    targeted: tuple[str, ...]
    #: Action strings actually applied, in order.
    actions: tuple[str, ...]
    rationale: str
    score_before: float
    #: ``None`` until the render has been measured on the following turn.
    score_after: float | None = None
    #: Whether the move beat the best score of the run so far, as opposed to
    #: merely beating the step before it. The supervisor counts strikes on
    #: this; the specialist is shown the simpler comparison.
    improved_best: bool = False

    @property
    def settled(self) -> bool:
        return self.score_after is not None

    @property
    def delta(self) -> float | None:
        if self.score_after is None:
            return None
        return self.score_after - self.score_before

    @property
    def helped(self) -> bool | None:
        d = self.delta
        return None if d is None else d < 0.0

    def render(self) -> str:
        moves = "; ".join(self.actions) if self.actions else "(no edit)"
        if self.score_after is None:
            return f"step {self.step}: {moves} -> not yet measured"
        verdict = "BETTER" if self.helped else "WORSE"
        return (
            f"step {self.step}: {moves} -> distance "
            f"{self.score_before:.4f} to {self.score_after:.4f} ({verdict} "
            f"by {abs(self.delta or 0.0):.4f})"
        )

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["role"] = str(self.role)
        payload["delta"] = self.delta
        payload["helped"] = self.helped
        return payload


@dataclass
class WorkingMemory:
    entries: list[MemoryEntry] = field(default_factory=list)
    #: Consecutive settled turns that failed to beat the best score *ever*
    #: seen, per role. The supervisor reads this to reroute away from a
    #: specialist that is stuck rather than letting it spend the whole budget.
    #:
    #: "Beat the best" rather than "beat the previous step", because the
    #: previous-step rule was measurably evadable. A specialist overshooting
    #: and correcting produces a sawtooth -- worse, better, worse, better --
    #: and every recovery reset its strike count, so it held the route for
    #: seven straight renders while ending no better than it started. Scoring
    #: against the best-so-far is also what the critic already does, so the two
    #: now agree about what progress means.
    strikes: dict[Role, int] = field(default_factory=dict)
    #: Lowest distance seen this run. ``None`` until the first settle.
    best_score: float | None = None
    #: Attempted calls to tools the role does not own. Counted rather than
    #: merely rejected: the rate is a reported number, because a prompt that
    #: leaks the boundary is a prompt problem and this is how it shows up.
    ownership_violations: int = 0
    #: Tool calls rejected for bad arguments or out-of-bounds values.
    rejected_calls: int = 0

    def open(
        self,
        *,
        step: int,
        role: Role,
        targeted: tuple[str, ...],
        actions: tuple[str, ...],
        rationale: str,
        score_before: float,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            step=step,
            role=role,
            targeted=targeted,
            actions=actions,
            rationale=rationale,
            score_before=score_before,
        )
        self.entries.append(entry)
        return entry

    def settle(self, score_after: float) -> MemoryEntry | None:
        """Attribute the newly measured score to the most recent open entry.

        Two different notions of "it worked" are recorded, deliberately. What
        the specialist is *shown* is whether the move beat the step before it,
        because that is the feedback it can act on. What the *supervisor*
        counts is whether the move beat the best score of the whole run, which
        is what stops a sawtooth from holding the route forever.
        """
        for entry in reversed(self.entries):
            if entry.settled:
                continue
            entry.score_after = score_after
            baseline = min(entry.score_before, self.best_score or entry.score_before)
            entry.improved_best = score_after < baseline
            self.best_score = min(baseline, score_after)
            self.strikes[entry.role] = (
                0 if entry.improved_best else self.strikes.get(entry.role, 0) + 1
            )
            return entry
        return None

    def strikes_for(self, role: Role) -> int:
        return self.strikes.get(role, 0)

    def for_role(self, role: Role, limit: int = RECENT_PER_ROLE) -> list[MemoryEntry]:
        return [e for e in self.entries if e.role is role][-limit:]

    def render_own(self, role: Role, limit: int = RECENT_PER_ROLE) -> str:
        mine = self.for_role(role, limit)
        if not mine:
            return "(this is your first turn on this track)"
        return "\n".join(e.render() for e in mine)

    def render_others(self, role: Role, limit: int = RECENT_PER_ROLE) -> str:
        """One line per other role: what it last did and whether it worked.

        Deliberately coarse. A specialist that knows the loudness stage just
        added 3 dB understands why its own measurements moved; it is not
        thereby invited to reason about level itself.
        """
        lines: list[str] = []
        for other in Role:
            if other is role:
                continue
            theirs = [e for e in self.entries if e.role is other and e.settled]
            if not theirs:
                continue
            last = theirs[-1]
            outcome = "helped" if last.helped else "did not help"
            lines.append(f"{other}: {'; '.join(last.actions) or '(no edit)'} ({outcome})")
        return "\n".join(lines[-limit:]) if lines else "(no other specialist has run yet)"

    def stats(self) -> dict[str, Any]:
        settled = [e for e in self.entries if e.settled]
        helped = [e for e in settled if e.helped]
        return {
            "turns": len(self.entries),
            "settled": len(settled),
            "helped": len(helped),
            "hit_rate": round(len(helped) / len(settled), 4) if settled else 0.0,
            "by_role": {str(r): len(self.for_role(r, limit=10**6)) for r in Role},
            "ownership_violations": self.ownership_violations,
            "rejected_calls": self.rejected_calls,
        }
