"""The shared control loop.

Every system is a ``Proposer``: given the measured state, return the next
chain. The loop owns rendering, measurement, the critic, budget accounting and
the trace, so all systems are held to identical rules.

Two rules worth stating because they decide what the numbers mean:

**The budget is denominated in renders.** Not LLM calls, not wall time. A
system that thinks longer is not thereby given more attempts on the audio.

**The reported result is the best chain found, not the last one.** Any real
tool keeps its best result, the rule applies identically to every system, and
without it a controller that explores then ends on a bad step would be scored
on the bad step. Every intermediate score is still in the trace.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Final

from headroom.analysis.features import analyze
from headroom.audio import AudioBuffer
from headroom.dsp.chain import Chain
from headroom.dsp.ops import BoundViolationError
from headroom.target.distance import DistanceResult, distance, recovery_ratio
from headroom.target.profile import TargetProfile

from .critic import DEFAULT_CONFIG, CriticConfig, assess
from .state import AbortReason, LoopState, RunTrace, StepRecord, Verdict

_TRACKED_PACKAGES: Final[tuple[str, ...]] = (
    "numpy",
    "scipy",
    "pedalboard",
    "librosa",
    "pyloudnorm",
    "soundfile",
)


@dataclass(frozen=True, slots=True)
class Proposal:
    """A system's next move."""

    chain: Chain
    #: Formatted "<parameter> <signed delta>" where possible, because the
    #: critic's sign-flip detector reads it. Free text otherwise.
    action: str = ""
    note: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    #: Set when a system has decided it has nothing useful left to try.
    give_up: bool = False
    #: Which specialist produced this move, for a multi-agent system. Empty
    #: for every non-agent system, which is why it lives here rather than in a
    #: subclass: one trace layout for all systems keeps the comparison honest.
    role: str = ""
    #: Edits bundled into this one render. The heuristic can only ever make
    #: one; a specialist may coordinate several, and the difference is the
    #: architecture's whole claim, so it is recorded per step.
    n_edits: int = 1
    #: Tool calls refused for bad arguments, bounds or ownership.
    n_rejected: int = 0


Proposer = Callable[[LoopState], Proposal]


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover
            out[name] = "unknown"
    return out


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):  # pragma: no cover
        return ""


def config_hash(config: CriticConfig, target: TargetProfile, norm: str) -> str:
    """Hash of everything that changes what a score means.

    Printed alongside results: two runs with different tolerances or weights
    are not comparable, and the hash makes that visible instead of leaving it
    to be discovered.
    """
    payload = {
        "critic": {
            "render_budget": config.render_budget,
            "window": config.window,
            "no_improve_patience": config.no_improve_patience,
            "min_relative_improvement": config.min_relative_improvement,
            "min_step_scale": config.min_step_scale,
            "divergence_multiple": config.divergence_multiple,
        },
        "norm": norm,
        "tolerance_overrides": target.tolerance_overrides,
        "weight_overrides": target.weight_overrides,
        "directions": target.directions,
        "n_constrained": len(target.targets),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()


def direction_key(action: str) -> str:
    """The identity of a *move*, for the "tried this and it was worse" set.

    Recording the whole action string was measurably useless. A proportional
    controller emits a different magnitude every step, so ``comp.ratio +3.063``
    and ``comp.ratio +4.249`` never matched and nothing was ever recognised as
    already-failed -- the heuristic baseline added compression seven times in a
    row while the distance climbed monotonically, then aborted.

    Recording the parameter alone is too strong: a move that overshot should be
    retried smaller, which the critic's damping already arranges. Parameter
    plus *sign* is the useful identity. "Pushing this parameter up made things
    worse" is real information; it does not forbid pulling the parameter back
    down, which is usually the right next move.
    """
    head, _, tail = action.rpartition(" ")
    if not head:
        return action
    try:
        value = float(tail)
    except ValueError:
        return action
    return f"{head} {'+' if value > 0 else '-'}"


def _worst_summary(result: DistanceResult, k: int = 5) -> list[dict[str, float | str]]:
    return [
        {
            "name": d.name,
            "family": d.family,
            "unit": d.unit,
            "delta": round(d.delta, 4),
            "scaled": round(d.scaled, 4),
            "excess": round(d.excess, 4),
        }
        for d in result.worst(k)
    ]


def run_loop(
    system: str,
    source: AudioBuffer,
    target: TargetProfile,
    propose: Proposer,
    *,
    track_id: str = "",
    degradation_kind: str = "none",
    degradation_seed: int = 0,
    degradation_params: dict[str, float | int | str] | None = None,
    degradation_lossy: bool = False,
    config: CriticConfig = DEFAULT_CONFIG,
    norm: str = "l2",
    model: str = "",
    effort: str = "",
) -> RunTrace:
    """Run one system against one target and return its trace.

    ``source`` is the audio to be repaired -- in evaluation that is the
    already-degraded render, not the pristine original.
    """
    started = time.perf_counter()
    renders = 0

    features = analyze(source)
    result = distance(features, target, norm=norm)
    initial = result.score

    state = LoopState(
        source=source,
        target=target,
        chain=Chain(),
        features=features,
        distance=result,
        initial_distance=initial,
        step_index=0,
        step_scale=1.0,
        renders_used=0,
        render_budget=config.render_budget,
    )

    best_chain, best_score = Chain(), initial
    abort_reason: AbortReason | None = None
    converged = False
    totals = {"in": 0, "out": 0, "cache": 0}
    total_cost = 0.0

    while True:
        verdict = assess(state, config)
        state.step_scale = verdict.step_scale
        state.oscillating = verdict.oscillating

        if verdict.verdict is not Verdict.CONTINUE:
            state.history.append(
                StepRecord(
                    index=state.step_index,
                    chain=state.chain,
                    chain_fingerprint=state.chain.fingerprint(),
                    distance_score=state.distance.score,
                    n_out_of_tolerance=state.distance.n_out_of_tolerance,
                    by_family={k: round(v, 6) for k, v in state.distance.by_family.items()},
                    worst=_worst_summary(state.distance),
                    action="",
                    step_scale=state.step_scale,
                    oscillating=state.oscillating,
                    verdict=verdict.verdict,
                    renders_used=renders,
                    elapsed_s=round(time.perf_counter() - started, 4),
                    note=verdict.note,
                )
            )
            converged = verdict.verdict is Verdict.CONVERGED
            abort_reason = verdict.abort_reason
            break

        try:
            proposal = propose(state)
        except BoundViolationError as exc:
            abort_reason = AbortReason.BOUND_SATURATION
            state.history.append(
                _terminal_step(state, renders, started, AbortReason.BOUND_SATURATION, str(exc))
            )
            break
        except Exception as exc:
            abort_reason = AbortReason.PROPOSAL_ERROR
            state.history.append(
                _terminal_step(state, renders, started, AbortReason.PROPOSAL_ERROR, repr(exc))
            )
            break

        totals["in"] += proposal.input_tokens
        totals["out"] += proposal.output_tokens
        totals["cache"] += proposal.cache_read_tokens
        total_cost += proposal.cost_usd

        if proposal.give_up:
            abort_reason = AbortReason.NO_IMPROVEMENT
            state.history.append(
                _terminal_step(
                    state, renders, started, AbortReason.NO_IMPROVEMENT, proposal.note or "gave up"
                )
            )
            break

        ordered, _ = proposal.chain.canonical()
        if ordered.fingerprint() == state.chain.fingerprint():
            abort_reason = AbortReason.PROPOSAL_EMPTY
            state.history.append(
                _terminal_step(
                    state, renders, started, AbortReason.PROPOSAL_EMPTY, "proposal changed nothing"
                )
            )
            break

        from headroom.dsp.backends.pedalboard import render_chain

        try:
            rendered = render_chain(source, ordered)
        except (FloatingPointError, ValueError) as exc:
            abort_reason = AbortReason.RENDER_FAILURE
            state.history.append(
                _terminal_step(state, renders, started, AbortReason.RENDER_FAILURE, repr(exc))
            )
            break

        renders += 1
        new_features = analyze(rendered)
        new_result = distance(new_features, target, norm=norm)

        if new_result.score >= state.distance.score and proposal.action:
            state.tried_and_failed.add(direction_key(proposal.action))

        state.history.append(
            StepRecord(
                index=state.step_index,
                chain=ordered,
                chain_fingerprint=ordered.fingerprint(),
                distance_score=new_result.score,
                n_out_of_tolerance=new_result.n_out_of_tolerance,
                by_family={k: round(v, 6) for k, v in new_result.by_family.items()},
                worst=_worst_summary(new_result),
                action=proposal.action,
                step_scale=state.step_scale,
                oscillating=state.oscillating,
                verdict=Verdict.CONTINUE,
                renders_used=renders,
                elapsed_s=round(time.perf_counter() - started, 4),
                input_tokens=proposal.input_tokens,
                output_tokens=proposal.output_tokens,
                cache_read_tokens=proposal.cache_read_tokens,
                cost_usd=proposal.cost_usd,
                role=proposal.role,
                n_edits=proposal.n_edits,
                n_rejected=proposal.n_rejected,
                note=proposal.note,
            )
        )

        if new_result.score < best_score:
            best_chain, best_score = ordered, new_result.score

        state.chain = ordered
        state.features = new_features
        state.distance = new_result
        state.renders_used = renders
        state.step_index += 1

    return RunTrace(
        system=system,
        track_id=track_id,
        degradation_kind=degradation_kind,
        degradation_seed=degradation_seed,
        degradation_params=degradation_params or {},
        degradation_lossy=degradation_lossy,
        source_hash=source.content_hash(),
        target_label=target.label,
        target_provenance=target.provenance,
        initial_distance=initial,
        final_distance=best_score,
        recovery_ratio=recovery_ratio(initial, best_score),
        converged=converged,
        abort_reason=abort_reason,
        final_chain=best_chain,
        steps=tuple(state.history),
        n_renders=renders,
        wall_time_s=round(time.perf_counter() - started, 4),
        total_cost_usd=total_cost,
        total_input_tokens=totals["in"],
        total_output_tokens=totals["out"],
        total_cache_read_tokens=totals["cache"],
        git_sha=git_sha(),
        config_hash=config_hash(config, target, norm),
        package_versions=package_versions(),
        model=model,
        effort=effort,
    )


def _terminal_step(
    state: LoopState, renders: int, started: float, reason: AbortReason, note: str
) -> StepRecord:
    return StepRecord(
        index=state.step_index,
        chain=state.chain,
        chain_fingerprint=state.chain.fingerprint(),
        distance_score=state.distance.score,
        n_out_of_tolerance=state.distance.n_out_of_tolerance,
        by_family={k: round(v, 6) for k, v in state.distance.by_family.items()},
        worst=_worst_summary(state.distance),
        action="",
        step_scale=state.step_scale,
        oscillating=state.oscillating,
        verdict=Verdict.ABORT,
        renders_used=renders,
        elapsed_s=round(time.perf_counter() - started, 4),
        note=f"{reason}: {note}",
    )
