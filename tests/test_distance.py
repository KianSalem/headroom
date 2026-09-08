from __future__ import annotations

import math

import numpy as np
import pytest

from headroom.analysis.features import REPORTED_NOT_SCORED, analyze
from headroom.analysis.spectral import N_BANDS
from headroom.audio import AudioBuffer
from headroom.dsp.backends.pedalboard import render_chain
from headroom.dsp.chain import Chain
from headroom.dsp.ops import op_gain
from headroom.target.distance import (
    FAMILY_WEIGHTS,
    FEATURE_WEIGHTS,
    SCORED,
    distance,
    recovery_ratio,
    to_scored,
)
from headroom.target.profile import PRESETS, TargetProfile

from .conftest import stereo, white


@pytest.fixture
def src() -> AudioBuffer:
    return stereo(white(seconds=6.0, seed=4))


def test_distance_to_self_is_zero_and_converged(src: AudioBuffer) -> None:
    """Phase 3 acceptance: distance(x, x) == 0."""
    fv = analyze(src)
    result = distance(fv, TargetProfile.from_features(fv))
    assert result.score == 0.0
    assert result.converged
    assert result.n_out_of_tolerance == 0


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_distance_is_monotone_under_growing_perturbation(src: AudioBuffer, norm: str) -> None:
    """Phase 3 acceptance: monotonic under increasing single-feature perturbation."""
    fv = analyze(src)
    target = TargetProfile.from_features(fv)
    scores = [
        distance(
            analyze(render_chain(src, Chain().add(op_gain(g)))),
            target,
            norm=norm,
        ).score
        for g in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    ]
    assert scores == sorted(scores)


def test_breakdown_sums_consistently_with_the_scalar_score(src: AudioBuffer) -> None:
    """Phase 3 acceptance: the breakdown sums consistently with the score."""
    fv = analyze(src)
    target = TargetProfile.from_features(fv)
    moved = analyze(render_chain(src, Chain().add(op_gain(3.0))))

    l2 = distance(moved, target, norm="l2")
    assert math.sqrt(sum(d.contribution for d in l2.breakdown.values())) == pytest.approx(
        l2.score, abs=1e-12
    )
    l1 = distance(moved, target, norm="l1")
    assert sum(d.contribution for d in l1.breakdown.values()) == pytest.approx(l1.score, abs=1e-12)


def test_weights_normalize_per_family_not_per_feature() -> None:
    """Ten of 28 scored dimensions are spectral and eleven are stereo. Uniform
    per-feature weights would give those two families 75% of the objective
    while lufs_integrated got 3.6% -- an implicit weighting choice pretending
    to be neutrality."""
    totals: dict[str, float] = {}
    for spec in SCORED:
        totals[spec.family] = totals.get(spec.family, 0.0) + FEATURE_WEIGHTS[spec.name]
    for family, want in FAMILY_WEIGHTS.items():
        assert totals[family] == pytest.approx(want, abs=1e-12)
    assert sum(totals.values()) == pytest.approx(1.0, abs=1e-12)


def test_collinear_features_are_excluded_from_the_score() -> None:
    """plr is exactly true_peak_dbtp - lufs_integrated; scoring all three would
    count loudness error three times."""
    scored_names = {s.name for s in SCORED}
    for excluded in REPORTED_NOT_SCORED:
        assert excluded not in scored_names
    assert "plr" in REPORTED_NOT_SCORED


def test_band_energy_enters_the_score_as_a_centered_log_ratio(src: AudioBuffer) -> None:
    """Band energy is compositional: normalized to sum to one, so its nine
    values carry eight degrees of freedom and its deltas sum to zero. The CLR
    transform is the correct treatment, and CLR values must sum to zero."""
    scored = to_scored(analyze(src))
    clr = [scored[f"band_clr_{i}"] for i in range(N_BANDS)]
    assert sum(clr) == pytest.approx(0.0, abs=1e-9)


def test_clr_is_invariant_to_level(src: AudioBuffer) -> None:
    quiet = to_scored(analyze(render_chain(src, Chain().add(op_gain(-12.0)))))
    loud = to_scored(analyze(render_chain(src, Chain().add(op_gain(6.0)))))
    for i in range(N_BANDS):
        assert quiet[f"band_clr_{i}"] == pytest.approx(loud[f"band_clr_{i}"], abs=0.05)


def test_bounded_features_are_transformed_not_used_raw(src: AudioBuffer) -> None:
    """correlation is bounded [-1, 1] and flatness [0, 1]; normalizing them raw
    is close to meaningless, so they enter as Fisher-z and logit."""
    fv = analyze(src)
    scored = to_scored(fv)
    assert scored["correlation_z"] == pytest.approx(
        math.atanh(min(fv.correlation, 0.999999)), abs=1e-6
    )
    assert scored["flatness_logit"] != fv.spectral_flatness


def test_inside_tolerance_contributes_exactly_zero(src: AudioBuffer) -> None:
    """This is what makes convergence meaningful and stops the agent chasing
    measurement noise."""
    fv = analyze(src)
    target = TargetProfile.from_features(fv)
    tiny = analyze(render_chain(src, Chain().add(op_gain(0.2))))
    result = distance(tiny, target)
    assert result.breakdown["lufs_integrated"].delta != 0.0
    assert result.breakdown["lufs_integrated"].excess == 0.0
    assert result.breakdown["lufs_integrated"].in_tolerance


def test_deltas_are_signed_so_the_agent_knows_which_way_to_move(src: AudioBuffer) -> None:
    fv = analyze(src)
    target = TargetProfile.from_features(fv)
    louder = analyze(render_chain(src, Chain().add(op_gain(6.0))))
    quieter = analyze(render_chain(src, Chain().add(op_gain(-6.0))))
    assert distance(louder, target).breakdown["lufs_integrated"].excess > 0
    assert distance(quieter, target).breakdown["lufs_integrated"].excess < 0


def test_partial_targets_renormalize_so_scores_stay_comparable(src: AudioBuffer) -> None:
    """One mechanism gives a full-vector target, a loudness preset, and a
    masked target. Without renormalization a two-feature preset would score
    tiny next to a full-vector one and the two could not be compared."""
    fv = analyze(src)
    for target in (
        TargetProfile.from_features(fv),
        TargetProfile.from_preset("spotify"),
        TargetProfile.masked(fv, {"loudness", "spectral"}),
    ):
        result = distance(fv, target)
        assert sum(d.weight for d in result.breakdown.values()) == pytest.approx(1.0, abs=1e-12)


def test_true_peak_in_a_preset_is_a_ceiling_not_a_setpoint(src: AudioBuffer) -> None:
    """Audio well under the ceiling is quiet, not wrong. Treating the ceiling
    as a setpoint would penalize a quiet master as harshly as a clipping one."""
    target = TargetProfile.from_preset("spotify")
    quiet = analyze(render_chain(src, Chain().add(op_gain(-12.0))))
    result = distance(quiet, target)
    assert result.breakdown["true_peak_dbtp"].direction == "max"
    assert result.breakdown["true_peak_dbtp"].delta < 0.0
    assert result.breakdown["true_peak_dbtp"].excess == 0.0


def test_exceeding_a_ceiling_is_penalized(src: AudioBuffer) -> None:
    target = TargetProfile.from_preset("spotify")
    hot = analyze(
        render_chain(
            src,
            Chain().add(op_gain(24.0)).add(op_gain(12.0)),
        )
    )
    assert distance(hot, target).breakdown["true_peak_dbtp"].excess > 0.0


def test_masked_target_leaves_other_families_unconstrained(src: AudioBuffer) -> None:
    fv = analyze(src)
    target = TargetProfile.masked(fv, {"loudness"})
    assert target.constrained_families() == ["loudness"]
    assert "band_clr_0" not in target.targets


def test_every_preset_is_reachable_and_well_formed() -> None:
    for name, preset in PRESETS.items():
        target = TargetProfile.from_preset(name)
        assert target.targets["lufs_integrated"] == preset.lufs_integrated
        assert target.directions["true_peak_dbtp"] == "max"
        assert -30.0 < preset.lufs_integrated < 0.0
        assert -3.0 <= preset.true_peak_dbtp <= -0.1


def test_unknown_preset_raises() -> None:
    with pytest.raises(KeyError):
        TargetProfile.from_preset("winamp")


def test_reference_plus_preset_keeps_tone_and_overrides_level(src: AudioBuffer) -> None:
    """ "Match that reference, but at Spotify level" is a real request."""
    fv = analyze(src)
    combined = TargetProfile.from_features(fv).with_loudness("spotify")
    assert combined.targets["lufs_integrated"] == -14.0
    assert combined.targets["band_clr_0"] == pytest.approx(to_scored(fv)["band_clr_0"])
    assert combined.directions["true_peak_dbtp"] == "max"


def test_target_survives_a_json_roundtrip(src: AudioBuffer) -> None:
    target = TargetProfile.from_features(analyze(src)).with_loudness("club")
    restored = TargetProfile.model_validate_json(target.model_dump_json())
    assert restored == target


def test_rejects_a_target_that_constrains_nothing(src: AudioBuffer) -> None:
    empty = TargetProfile(label="x", provenance="y", targets={})
    with pytest.raises(ValueError, match="constrains no scored features"):
        distance(analyze(src), empty)


def test_rejects_an_unknown_norm(src: AudioBuffer) -> None:
    fv = analyze(src)
    with pytest.raises(ValueError, match="norm must be"):
        distance(fv, TargetProfile.from_features(fv), norm="linf")


@pytest.mark.parametrize(
    ("initial", "final", "want"),
    [(1.0, 0.0, 1.0), (1.0, 0.5, 0.5), (1.0, 1.0, 0.0), (1.0, 3.0, -2.0), (0.0, 0.0, 0.0)],
)
def test_recovery_ratio(initial: float, final: float, want: float) -> None:
    assert recovery_ratio(initial, final) == pytest.approx(want)


def test_recovery_ratio_of_an_already_perfect_start_is_not_nan() -> None:
    assert np.isfinite(recovery_ratio(0.0, 0.0))
    assert recovery_ratio(0.0, 1.0) == float("-inf")
