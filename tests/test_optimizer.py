from __future__ import annotations

import numpy as np
import pytest
from evals.degradations import make_degradation

from headroom.analysis.features import analyze
from headroom.analysis.features import clear_cache as clear_analyze
from headroom.analysis.spectral import N_BANDS
from headroom.audio import AudioBuffer
from headroom.baselines import heuristic, optimizer
from headroom.control.critic import CriticConfig
from headroom.control.loop import run_loop
from headroom.dsp.backends.pedalboard import clear_cache, render_chain
from headroom.dsp.chain import Chain
from headroom.dsp.ops import op_gain
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
    original = AudioBuffer(
        np.stack([bass + mids + tops, bass + mids + np.roll(tops, 71)], axis=1), SR
    )
    return original, TargetProfile.from_features(analyze(original))


def test_identity_params_render_an_empty_chain(scene: tuple[AudioBuffer, TargetProfile]) -> None:
    """The optimizer must be able to express "change nothing"."""
    original, _ = scene
    chain = optimizer.build_chain(optimizer.identity_params(), -14.0)
    assert chain.ops == ()
    rendered = render_chain(original, chain, use_cache=False)
    np.testing.assert_array_equal(rendered.samples, original.samples)


def test_build_chain_omits_near_identity_stages() -> None:
    """A chain full of 0.01 dB filters and ratio-1.001 compressors would be
    unreadable, and the chain is meant to be the inspectable artifact."""
    params = optimizer.identity_params()
    params[0] = 0.001  # gain below the threshold
    params[2 + N_BANDS] = 1.005  # compressor effectively bypassed
    assert optimizer.build_chain(params, -14.0).ops == ()

    params[0] = 3.0
    assert len(optimizer.build_chain(params, -14.0).ops) == 1


def test_build_chain_respects_op_bounds() -> None:
    """Powell explores to the edge of its box, so the mapping has to clamp."""
    params = np.full(optimizer.N_PARAMS, 1e6, dtype=np.float64)
    chain = optimizer.build_chain(params, -14.0)
    assert Chain.model_validate_json(chain.model_dump_json()) == chain


def test_optimizer_bounds_every_seeded_system(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """The point of the whole baseline. Measured cold, blind Powell lost to the
    heuristic on coupled degradations, and a ceiling a competitor beats is not
    a ceiling."""
    original, target = scene
    level = analyze(original).lufs_integrated
    degradation = make_degradation("combo", seed=0, level_db=level)
    degraded = render_chain(original, degradation.chain)

    clear_cache()
    clear_analyze()
    heur = run_loop("heuristic", degraded, target, heuristic.propose, config=BUDGET)

    clear_cache()
    clear_analyze()
    bound = optimizer.run(
        degraded, target, render_budget=40, n_starts=1, seed_chains=[heur.final_chain]
    )
    assert bound.final_distance <= heur.final_distance + 1e-9
    assert bound.recovery_ratio >= heur.recovery_ratio - 1e-9


def test_seeding_with_a_perfect_chain_is_never_made_worse(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    original, target = scene
    level = analyze(original).lufs_integrated
    degradation = make_degradation("level_offset", seed=0, level_db=level)
    degraded = render_chain(original, degradation.chain)
    exact = Chain().add(op_gain(-float(degradation.params["gain_db"])))

    clear_cache()
    clear_analyze()
    bound = optimizer.run(degraded, target, render_budget=40, n_starts=1, seed_chains=[exact])
    assert bound.recovery_ratio > 0.99


def test_optimizer_stops_early_once_converged(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """Spending the rest of the budget after finding an exact answer would only
    inflate the reported render count, which is part of the comparison."""
    original, target = scene
    level = analyze(original).lufs_integrated
    degradation = make_degradation("level_offset", seed=0, level_db=level)
    degraded = render_chain(original, degradation.chain)
    exact = Chain().add(op_gain(-float(degradation.params["gain_db"])))

    clear_cache()
    clear_analyze()
    bound = optimizer.run(degraded, target, render_budget=250, n_starts=3, seed_chains=[exact])
    assert bound.converged
    assert bound.n_renders < 30, f"did not stop early: {bound.n_renders} renders"


def test_optimizer_respects_its_render_budget(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    original, target = scene
    level = analyze(original).lufs_integrated
    degraded = render_chain(
        original, make_degradation("spectral_tilt", seed=0, level_db=level).chain
    )
    clear_cache()
    clear_analyze()
    bound = optimizer.run(degraded, target, render_budget=25, n_starts=2)
    assert bound.n_renders <= 26  # +1 for the final confirmation render


def test_optimizer_never_makes_audio_worse_than_it_started(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """It keeps the best result found, and identity is always among the
    candidates, so recovery cannot be negative."""
    original, target = scene
    level = analyze(original).lufs_integrated
    degraded = render_chain(original, make_degradation("band_shift", seed=1, level_db=level).chain)
    clear_cache()
    clear_analyze()
    bound = optimizer.run(degraded, target, render_budget=30, n_starts=1)
    assert bound.recovery_ratio >= -1e-9
    assert bound.final_distance <= bound.initial_distance + 1e-9


def test_optimizer_trace_states_its_unequal_budget(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """The budget is unequal by design, so the trace has to say so -- a reader
    comparing it as a peer would be misled."""
    original, target = scene
    level = analyze(original).lufs_integrated
    degraded = render_chain(original, make_degradation("band_shift", seed=1, level_db=level).chain)
    clear_cache()
    clear_analyze()
    bound = optimizer.run(degraded, target, render_budget=20, n_starts=1)
    assert "Unequal budget by design" in bound.steps[0].note
    assert bound.system == "optimizer"
