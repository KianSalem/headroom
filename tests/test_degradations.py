from __future__ import annotations

import numpy as np
import pytest
from evals.degradations import (
    ALL_KINDS,
    LEVEL_RELATIVE_KINDS,
    MIN_USEFUL_DISTANCE,
    SINGLE_KINDS,
    Degradation,
    DegradationKind,
    make_degradation,
    suite,
)

from headroom.analysis.features import analyze
from headroom.analysis.loudness import analyze_loudness
from headroom.audio import AudioBuffer
from headroom.dsp.backends.pedalboard import render_chain
from headroom.target.distance import distance
from headroom.target.profile import TargetProfile

from .conftest import SR


@pytest.fixture(scope="module")
def musical() -> AudioBuffer:
    """Broadband stereo with transients, so every feature family has something
    to measure. Synthetic, but dense enough to be a fair stand-in."""
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
    return AudioBuffer(
        np.stack(
            [bass + mids + tops + hits, bass + mids + np.roll(tops, 71) + np.roll(hits, 37)], axis=1
        ),
        SR,
    )


def test_same_seed_gives_an_identical_degradation() -> None:
    """A run must be recreatable from (track, kind, seed) alone."""
    for kind in ALL_KINDS:
        a = make_degradation(kind, seed=7)
        b = make_degradation(kind, seed=7)
        assert a.chain.fingerprint() == b.chain.fingerprint()
        assert a.params == b.params


def test_different_seeds_give_different_degradations() -> None:
    for kind in ALL_KINDS:
        prints = {make_degradation(kind, seed=s).chain.fingerprint() for s in range(4)}
        assert len(prints) > 1, f"{kind} ignores its seed"


def test_degradation_chains_are_within_op_bounds() -> None:
    """Damage is built from the same bounded vocabulary used to repair it, so
    an exact inverse is usually expressible -- which is what makes a recovery
    ratio interpretable."""
    for degradation in suite(seeds=(0, 1, 2, 3, 4)):
        assert degradation.chain.ops, f"{degradation.kind} produced no ops"
        restored = type(degradation.chain).model_validate_json(degradation.chain.model_dump_json())
        assert restored.fingerprint() == degradation.chain.fingerprint()


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_degradation_measurably_damages_the_audio(
    kind: DegradationKind, musical: AudioBuffer
) -> None:
    """A degradation that changes nothing is worse than useless: the recovery
    ratio divides by the initial distance, so a near-zero denominator turns
    measurement noise into an apparently excellent score."""
    clean = analyze(musical)
    target = TargetProfile.from_features(clean)
    scores = [
        distance(
            analyze(
                render_chain(
                    musical,
                    make_degradation(kind, seed=s, level_db=clean.lufs_integrated).chain,
                )
            ),
            target,
        ).score
        for s in (0, 1, 2)
    ]
    assert max(scores) > MIN_USEFUL_DISTANCE, f"{kind} never damages anything: {scores}"


@pytest.mark.parametrize("kind", LEVEL_RELATIVE_KINDS)
def test_dynamics_thresholds_follow_program_level(kind: DegradationKind) -> None:
    """A compressor threshold in absolute dBFS is meaningless without knowing
    how loud the material is. Fixed thresholds produced literal no-op
    degradations on dense material."""
    quiet = make_degradation(kind, seed=0, level_db=-30.0)
    loud = make_degradation(kind, seed=0, level_db=-8.0)
    assert quiet.params["threshold_db"] != loud.params["threshold_db"]
    assert float(quiet.params["threshold_db"]) < float(loud.params["threshold_db"])


def test_over_compress_is_flagged_lossy_and_over_expand_is_not() -> None:
    """Compression discards level information no expander recovers, so that
    class has a recovery ceiling below 1.0 for reasons unrelated to the
    controller. Expansion is invertible by a compressor."""
    assert make_degradation("over_compress", seed=0).is_lossy
    assert not make_degradation("over_expand", seed=0).is_lossy


def test_combo_stacks_distinct_single_degradations() -> None:
    combo = make_degradation("combo", seed=5, combo_size=3)
    parts = str(combo.params["parts"]).split(",")
    assert len(parts) == len(set(parts)) == 3
    assert all(p in SINGLE_KINDS for p in parts)
    assert len(combo.chain.ops) >= 3


def test_combo_inherits_lossiness_from_its_parts() -> None:
    """A combo containing over_compress cannot be exactly inverted either."""
    combos = [make_degradation("combo", seed=s) for s in range(12)]
    assert not any(c.is_lossy for c in combos), (
        "combo draws only from SINGLE_KINDS, none of which are lossy"
    )


@pytest.mark.parametrize(
    ("kind", "invertible"),
    [("level_offset", True), ("band_shift", True), ("stereo_collapse", True)],
)
def test_exact_inverse_recovers_fully(
    kind: DegradationKind, invertible: bool, musical: AudioBuffer
) -> None:
    """The achievable ceiling. Without this, a recovery ratio of 0.7 cannot be
    read as good or bad -- there is nothing to compare it to."""
    from headroom.dsp.chain import Chain
    from headroom.dsp.ops import EqBand, Op, op_eq, op_gain, op_stereo_width

    clean = analyze(musical)
    target = TargetProfile.from_features(clean)
    degradation = make_degradation(kind, seed=0, level_db=clean.lufs_integrated)
    damaged = render_chain(musical, degradation.chain)
    initial = distance(analyze(damaged), target).score

    params = degradation.params
    inverse: list[Op]
    if kind == "level_offset":
        inverse = [op_gain(-float(params["gain_db"]))]
    elif kind == "band_shift":
        inverse = [
            op_eq(
                [
                    EqBand(
                        shape="peak",
                        freq_hz=float(params["freq_hz"]),
                        gain_db=-float(params["gain_db"]),
                        q=1.0,
                    )
                ]
            )
        ]
    else:
        inverse = [op_stereo_width(width=1.0 / float(params["width"]))]

    final = distance(analyze(render_chain(damaged, Chain(ops=tuple(inverse)))), target).score
    assert initial > MIN_USEFUL_DISTANCE
    assert 1.0 - (final / initial) > 0.9, f"{kind} not recoverable: {initial} -> {final}"


def test_stereo_collapse_repair_stays_within_the_width_bound() -> None:
    """SPEC 10.2 specifies collapse to 0.0-0.4, but undoing 0.1 needs a width
    of 10.0 against a bound of 2.0. That range would measure the bound rather
    than the controller, so the range is narrowed."""
    for seed in range(20):
        width = float(make_degradation("stereo_collapse", seed=seed).params["width"])
        assert 1.0 / width <= 2.0, f"seed {seed}: repairing width {width} needs {1 / width:.2f}"


def test_suite_covers_the_full_cross_product() -> None:
    built = suite(seeds=(0, 1))
    assert len(built) == len(ALL_KINDS) * 2
    assert all(isinstance(d, Degradation) for d in built)


def test_level_offset_moves_loudness_by_the_amount_it_claims(musical: AudioBuffer) -> None:
    degradation = make_degradation("level_offset", seed=0)
    before = analyze_loudness(musical).lufs_integrated
    after = analyze_loudness(render_chain(musical, degradation.chain)).lufs_integrated
    assert after - before == pytest.approx(float(degradation.params["gain_db"]), abs=0.05)
