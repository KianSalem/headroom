"""The critic: deterministic, no LLM.

Reads the distance history and decides whether to continue, stop, or give up.
It also owns oscillation handling, which is required rather than optional -- a
naive loop *will* oscillate: boost the highs, measure too bright, cut the
highs, measure dull, forever.

Detection follows SPEC 6.4: distance failing to decrease over a window, and a
sign flip on the same op parameter across consecutive edits. The response is
escalated in order rather than jumping straight to abort -- halve the step
size, then freeze the oscillating parameter and route elsewhere, then abort
with a named reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .state import AbortReason, LoopState, Verdict


@dataclass(frozen=True, slots=True)
class CriticConfig:
    #: Renders a system may spend. The budget currency is renders rather than
    #: LLM calls, so no system gets free exploration.
    render_budget: int = 14
    #: Window over which distance must show net progress.
    window: int = 4
    #: Steps allowed without beating the best score before giving up.
    no_improve_patience: int = 4
    #: Relative improvement that counts as progress at all.
    min_relative_improvement: float = 0.02
    #: Below this the damping has failed and oscillation is unresolved.
    min_step_scale: float = 0.125
    #: A run whose distance grows past this multiple of its start is diverging.
    divergence_multiple: float = 3.0


DEFAULT_CONFIG: Final[CriticConfig] = CriticConfig()


@dataclass(frozen=True, slots=True)
class CriticVerdict:
    verdict: Verdict
    step_scale: float
    oscillating: bool
    abort_reason: AbortReason | None
    note: str


def _sign_flipped(state: LoopState) -> bool:
    """True when the last two accepted actions pushed the same parameter in
    opposite directions -- the signature of a boost-cut-boost cycle."""
    actions = [s.action for s in state.history if s.action]
    if len(actions) < 3:
        return False
    recent = actions[-3:]
    # Actions are formatted "<param> <signed delta>", so the parameter name is
    # everything up to the last space.
    keyed: list[tuple[str, float]] = []
    for action in recent:
        head, _, tail = action.rpartition(" ")
        try:
            keyed.append((head, float(tail)))
        except ValueError:
            return False
    if len({k for k, _ in keyed}) != 1:
        return False
    signs = [1 if v > 0 else -1 for _, v in keyed if v != 0.0]
    return len(signs) == 3 and signs[0] != signs[1] and signs[1] != signs[2]


def _stalled(state: LoopState, config: CriticConfig) -> bool:
    scores = state.score_history()
    if len(scores) <= config.no_improve_patience:
        return False
    best_recent = min(scores[-config.no_improve_patience :])
    best_before = min(scores[: -config.no_improve_patience])
    if best_before <= 0.0:
        return True
    return (best_before - best_recent) / best_before < config.min_relative_improvement


def assess(state: LoopState, config: CriticConfig = DEFAULT_CONFIG) -> CriticVerdict:
    """Decide what the loop should do next."""
    scale = state.step_scale

    if state.distance.converged:
        return CriticVerdict(Verdict.CONVERGED, scale, False, None, "inside tolerance")

    if state.renders_left <= 0:
        return CriticVerdict(
            Verdict.ABORT, scale, state.oscillating, AbortReason.MAX_STEPS, "render budget spent"
        )

    if state.distance.score > state.initial_distance * config.divergence_multiple:
        return CriticVerdict(
            Verdict.ABORT,
            scale,
            state.oscillating,
            AbortReason.DIVERGED,
            f"distance grew past {config.divergence_multiple}x its start",
        )

    scores = state.score_history()
    window = scores[-config.window :]
    no_net_progress = len(window) >= config.window and min(window) >= window[0]
    oscillating = no_net_progress or _sign_flipped(state)

    if oscillating:
        # Escalate in order: damp, then freeze and reroute, then abort.
        damped = scale / 2.0
        if damped < config.min_step_scale:
            return CriticVerdict(
                Verdict.ABORT,
                scale,
                True,
                AbortReason.OSCILLATION_UNRESOLVED,
                f"damping bottomed out at {scale:.3f}",
            )
        return CriticVerdict(
            Verdict.CONTINUE, damped, True, None, f"oscillation: damping to {damped:.3f}"
        )

    if _stalled(state, config):
        return CriticVerdict(
            Verdict.ABORT,
            scale,
            oscillating,
            AbortReason.NO_IMPROVEMENT,
            f"no {config.min_relative_improvement:.0%} gain in {config.no_improve_patience} steps",
        )

    return CriticVerdict(Verdict.CONTINUE, scale, False, None, "")
