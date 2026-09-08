from __future__ import annotations

import numpy as np
import pytest
from evals.degradations import make_degradation

from headroom.analysis.features import analyze
from headroom.audio import AudioBuffer
from headroom.baselines import heuristic, trivial
from headroom.control.critic import CriticConfig, assess
from headroom.control.loop import Proposal, config_hash, run_loop
from headroom.control.state import (
    TRACE_SCHEMA_VERSION,
    AbortReason,
    InfrastructureError,
    LoopState,
    RunTrace,
)
from headroom.dsp.backends.pedalboard import clear_cache, render_chain
from headroom.dsp.chain import Chain
from headroom.dsp.ops import op_gain
from headroom.target.distance import distance
from headroom.target.profile import TargetProfile

from .conftest import SR

BUDGET = CriticConfig(render_budget=14)


@pytest.fixture(scope="module")
def scene() -> tuple[AudioBuffer, TargetProfile]:
    rng = np.random.default_rng(3)
    t = np.arange(SR * 5) / SR
    bass = 0.18 * np.sin(2 * np.pi * 55 * t)
    mids = 0.10 * np.sin(2 * np.pi * 440 * t) + 0.08 * np.sin(2 * np.pi * 1320 * t)
    tops = 0.05 * rng.standard_normal(t.size)
    hits = np.zeros_like(t)
    for start in range(0, t.size - 3000, int(SR * 0.3)):
        hits[start : start + 3000] += (
            rng.standard_normal(3000) * 0.12 * np.exp(-np.arange(3000) / 500)
        )
    original = AudioBuffer(
        np.stack(
            [bass + mids + tops + hits, bass + mids + np.roll(tops, 71) + np.roll(hits, 37)],
            axis=1,
        ),
        SR,
    )
    return original, TargetProfile.from_features(analyze(original))


def _degraded(scene: tuple[AudioBuffer, TargetProfile], kind: str, seed: int = 0) -> AudioBuffer:
    original, _ = scene
    level = analyze(original).lufs_integrated
    degradation = make_degradation(kind, seed=seed, level_db=level)  # type: ignore[arg-type]
    return render_chain(original, degradation.chain)


def test_null_system_recovers_exactly_nothing(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """The sanity floor. If this is not ~0, the metric is wrong rather than the
    system impressive."""
    _, target = scene
    clear_cache()
    trace = run_loop(
        "null", _degraded(scene, "level_offset"), target, trivial.null_propose, config=BUDGET
    )
    assert trace.recovery_ratio == pytest.approx(0.0, abs=1e-12)
    assert trace.n_renders == 0
    assert not trace.converged


@pytest.mark.parametrize("kind", ["level_offset", "band_shift", "stereo_overwide"])
def test_heuristic_converges_on_single_feature_degradations(
    kind: str, scene: tuple[AudioBuffer, TargetProfile]
) -> None:
    """SPEC 9 acceptance: the heuristic must be genuinely good at single-feature
    numeric targets. A weak baseline would make the whole comparison
    worthless."""
    _, target = scene
    clear_cache()
    trace = run_loop(kind, _degraded(scene, kind), target, heuristic.propose, config=BUDGET)
    assert trace.converged, trace.summary()
    assert trace.recovery_ratio > 0.9


def test_heuristic_beats_random_decisively(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    _, target = scene
    source = _degraded(scene, "level_offset")
    clear_cache()
    heur = run_loop("heuristic", source, target, heuristic.propose, config=BUDGET)
    clear_cache()
    rand = run_loop("random", source, target, trivial.make_random_propose(0), config=BUDGET)
    assert heur.recovery_ratio > rand.recovery_ratio + 0.2


def test_heuristic_fixes_a_level_offset_in_one_render(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """Level maps onto gain one-for-one and is essentially decoupled, so a
    proportional controller should need a single move."""
    _, target = scene
    clear_cache()
    trace = run_loop(
        "heuristic", _degraded(scene, "level_offset"), target, heuristic.propose, config=BUDGET
    )
    assert trace.n_renders == 1
    assert trace.converged


def test_budget_is_denominated_in_renders(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """No system gets free attempts on the audio."""
    _, target = scene
    config = CriticConfig(render_budget=5)
    clear_cache()
    trace = run_loop(
        "random", _degraded(scene, "combo"), target, trivial.make_random_propose(1), config=config
    )
    assert trace.n_renders <= 5


def test_loop_reports_the_best_chain_not_the_last(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """Any real tool keeps its best result, and the rule applies identically to
    every system."""
    _, target = scene
    source = _degraded(scene, "level_offset")

    calls = {"n": 0}

    def good_then_bad(state: LoopState) -> Proposal:
        # Deliberately good-but-imperfect first, so the loop does not converge
        # and actually reaches the bad move whose result must be discarded.
        calls["n"] += 1
        if calls["n"] == 1:
            return Proposal(chain=Chain().add(op_gain(-7.0)), action="gain.gain_db -7.000")
        return Proposal(chain=Chain().add(op_gain(18.0)), action="gain.gain_db +18.000")

    clear_cache()
    trace = run_loop("probe", source, target, good_then_bad, config=CriticConfig(render_budget=6))
    scores = [s.distance_score for s in trace.steps if s.action]
    assert trace.final_distance == pytest.approx(min(scores))
    assert trace.final_distance < scores[-1]


def test_a_proposer_exception_ends_the_run_not_the_eval(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    _, target = scene

    def broken(state: LoopState) -> Proposal:
        raise RuntimeError("specialist blew up")

    clear_cache()
    trace = run_loop("broken", _degraded(scene, "combo"), target, broken, config=BUDGET)
    assert trace.abort_reason is AbortReason.PROPOSAL_ERROR
    assert "specialist blew up" in trace.steps[-1].note


def test_a_proposal_that_changes_nothing_is_named_as_such(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    _, target = scene

    def noop(state: LoopState) -> Proposal:
        return Proposal(chain=state.chain, action="noop 0.0")

    clear_cache()
    trace = run_loop("noop", _degraded(scene, "combo"), target, noop, config=BUDGET)
    assert trace.abort_reason is AbortReason.PROPOSAL_EMPTY


def test_divergence_is_named_for_what_was_observed(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """The target may well be reachable; the system ran away from it. Naming
    that "target unreachable" would misrepresent the abort distribution, which
    is itself a reported result."""
    _, target = scene

    def runaway(state: LoopState) -> Proposal:
        n = len(state.history)
        return Proposal(
            chain=Chain().add(op_gain(min(3.0 * (n + 1), 24.0))),
            action=f"gain.gain_db {3.0 * (n + 1):+.3f}",
        )

    clear_cache()
    trace = run_loop("runaway", _degraded(scene, "band_shift"), target, runaway, config=BUDGET)
    assert trace.abort_reason is AbortReason.DIVERGED


def test_every_abort_carries_a_named_reason(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """SPEC 6.4: every abort reason must be a named enum value, because the
    distribution across the evaluation is a headline result."""
    _, target = scene
    for name, proposer in (
        ("null", trivial.null_propose),
        ("random", trivial.make_random_propose(2)),
        ("hillclimb", trivial.make_hillclimb_propose(2)),
        ("heuristic", heuristic.propose),
    ):
        clear_cache()
        trace = run_loop(name, _degraded(scene, "combo", seed=1), target, proposer, config=BUDGET)
        assert trace.converged or isinstance(trace.abort_reason, AbortReason)


def test_critic_reports_convergence_at_zero_distance(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    original, target = scene
    features = analyze(original)
    result = distance(features, target)
    state = LoopState(
        source=original,
        target=target,
        chain=Chain(),
        features=features,
        distance=result,
        initial_distance=result.score,
        step_index=0,
        step_scale=1.0,
        renders_used=0,
        render_budget=14,
    )
    verdict = assess(state)
    assert verdict.verdict.value == "converged"


def test_trace_roundtrips_through_json(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """The report is generated from traces, never from live runs, so a trace
    has to survive serialisation exactly."""
    _, target = scene
    clear_cache()
    trace = run_loop(
        "heuristic", _degraded(scene, "band_shift"), target, heuristic.propose, config=BUDGET
    )
    restored = RunTrace.model_validate_json(trace.model_dump_json())
    assert restored == trace
    assert restored.trace_schema_version == TRACE_SCHEMA_VERSION


def test_trace_records_provenance(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """A number must always be traceable to the code and config that made it."""
    _, target = scene
    clear_cache()
    trace = run_loop(
        "heuristic", _degraded(scene, "band_shift"), target, heuristic.propose, config=BUDGET
    )
    assert trace.package_versions["pedalboard"]
    assert trace.package_versions["numpy"]
    assert trace.config_hash
    assert trace.source_hash


def test_config_hash_changes_when_the_metric_changes(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """Two runs with different tolerances are not comparable, and the hash
    makes that visible rather than leaving it to be discovered."""
    _, target = scene
    base = config_hash(BUDGET, target, "l2")
    assert config_hash(BUDGET, target, "l1") != base
    assert config_hash(CriticConfig(render_budget=99), target, "l2") != base
    loosened = target.model_copy(update={"tolerance_overrides": {"lufs_integrated": 5.0}})
    assert config_hash(BUDGET, loosened, "l2") != base


def test_config_hash_knows_which_features_are_constrained_not_just_how_many() -> None:
    """A spectral-only and a stereo-only target with the same feature count are
    different scores, and must not share a hash."""
    spotify = TargetProfile.from_preset("spotify")
    only_lufs = spotify.model_copy(update={"targets": {"lufs_integrated": -14.0}})
    only_peak = spotify.model_copy(update={"targets": {"true_peak_dbtp": -1.0}})
    assert len(only_lufs.targets) == len(only_peak.targets)
    assert config_hash(BUDGET, only_lufs, "l2") != config_hash(BUDGET, only_peak, "l2")


def test_an_infrastructure_failure_ends_the_evaluation_not_the_cell(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """A missing cassette or a dead API is not a fact about the proposer.
    Recording it as proposal_error would publish a +0.000 row that measures
    the harness, with a clean exit code."""
    _, target = scene

    class NoRecordingError(InfrastructureError):
        pass

    def unreplayable(state: LoopState) -> Proposal:
        raise NoRecordingError("no recording for this request")

    clear_cache()
    with pytest.raises(NoRecordingError):
        run_loop("agent", _degraded(scene, "combo"), target, unreplayable, config=BUDGET)
