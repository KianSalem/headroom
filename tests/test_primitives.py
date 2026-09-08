"""Tests for the DSP we wrote ourselves, because pedalboard has no equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from headroom.analysis.dynamics import crest_factor_db
from headroom.analysis.loudness import true_peak_dbtp
from headroom.analysis.spectral import BAND_EDGES
from headroom.audio import AudioBuffer
from headroom.dsp.primitives import expander, stereo_width, true_peak_limiter

from .conftest import SR, sine, stereo, white


@pytest.mark.parametrize("ceiling", [-0.1, -0.3, -1.0, -2.0, -3.0])
def test_limiter_guarantees_the_true_peak_ceiling(ceiling: float) -> None:
    """The ceiling is a guarantee, not an aspiration. If the limiter could miss
    it, the loudness specialist would be chasing a target its own tool cannot
    reach -- which appears in the loop as an oscillation that is really a
    tooling bug. 997 Hz near full scale is the worst case for inter-sample
    peaks.
    """
    hot = stereo(sine(997.0, -0.1)).samples
    out = true_peak_limiter(hot, SR, ceiling, 50.0)
    assert true_peak_dbtp(AudioBuffer(out, SR)) <= ceiling + 1e-6


def test_limiter_catches_inter_sample_peaks_a_sample_limiter_would_miss() -> None:
    """A sample-peak limiter leaves the ceiling exceeded in dBTP."""
    hot = stereo(sine(997.0, -0.5)).samples
    naive = np.clip(hot, -(10 ** (-1.0 / 20.0)), 10 ** (-1.0 / 20.0))
    assert true_peak_dbtp(AudioBuffer(naive, SR)) > -1.0
    ours = true_peak_limiter(hot, SR, -1.0, 50.0)
    assert true_peak_dbtp(AudioBuffer(ours, SR)) <= -1.0 + 1e-6


def test_limiter_leaves_quiet_audio_alone() -> None:
    quiet = stereo(sine(997.0, -20.0)).samples
    out = true_peak_limiter(quiet, SR, -1.0, 50.0)
    np.testing.assert_allclose(out, quiet, atol=1e-9)


def test_limiter_handles_silence() -> None:
    out = true_peak_limiter(np.zeros((SR, 2)), SR, -1.0, 50.0)
    assert np.all(out == 0.0)


def _squashed() -> np.ndarray:
    rng = np.random.default_rng(21)
    t = np.arange(SR * 6) / SR
    env = np.where((t % 1.0) < 0.12, 1.0, 0.25)
    return np.stack([env * rng.standard_normal(t.size) * 0.2] * 2, axis=1)


def test_expander_increases_crest_factor() -> None:
    """Without an expander the over_compress degradation is unrecoverable by
    construction: a compressor cannot undo compression."""
    sq = _squashed()
    before = crest_factor_db(sq)
    after = crest_factor_db(expander(sq, SR, -14.0, 3.0, 5.0, 80.0))
    assert after > before + 0.5


def test_expansion_is_monotone_in_ratio() -> None:
    sq = _squashed()
    crests = [crest_factor_db(expander(sq, SR, -14.0, r, 5.0, 80.0)) for r in (1.0, 1.5, 2.0, 3.0)]
    assert crests == sorted(crests)


def test_expander_ratio_one_is_exact_identity() -> None:
    sq = _squashed()
    assert np.array_equal(expander(sq, SR, -14.0, 1.0, 5.0, 80.0), sq)


def test_expander_below_the_signal_does_nothing() -> None:
    """A threshold under the whole signal must be a no-op, not a surprise."""
    sq = _squashed()
    out = expander(sq, SR, -60.0, 4.0, 5.0, 80.0)
    np.testing.assert_allclose(out, sq, atol=1e-9)


def test_global_width_zero_produces_mono() -> None:
    src = stereo(white(seed=1), white(seed=2)).samples
    out = stereo_width(src, SR, 0.0, None, BAND_EDGES)
    np.testing.assert_allclose(out[:, 0], out[:, 1], atol=1e-12)


def test_global_width_one_is_exact_identity() -> None:
    src = stereo(white(seed=1), white(seed=2)).samples
    np.testing.assert_allclose(stereo_width(src, SR, 1.0, None, BAND_EDGES), src, atol=1e-12)


def test_width_above_one_increases_side_energy() -> None:
    src = stereo(white(seed=1), white(seed=2)).samples

    def side_rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(((x[:, 0] - x[:, 1]) / 2.0) ** 2)))

    assert side_rms(stereo_width(src, SR, 1.5, None, BAND_EDGES)) > side_rms(src)


def test_band_limited_width_preserves_the_mid_signal() -> None:
    """Width only scales side energy, so the mono sum must be untouched."""
    src = stereo(white(seed=1), white(seed=2)).samples
    out = stereo_width(src, SR, 0.0, 4, BAND_EDGES)
    np.testing.assert_allclose(out.mean(axis=1), src.mean(axis=1), atol=1e-10)


@pytest.mark.parametrize("band", [0, 4, 8])
def test_band_limited_width_targets_its_band(band: int) -> None:
    from headroom.analysis.stereo import analyze_stereo

    src = stereo(white(seed=1), white(seed=2))
    before = analyze_stereo(src).width_per_band_db
    out = src.replace_samples(stereo_width(src.samples, SR, 0.0, band, BAND_EDGES))
    after = analyze_stereo(out).width_per_band_db
    deltas = [a - b for a, b in zip(after, before, strict=True)]
    assert deltas[band] < -8.0
    far = [abs(d) for i, d in enumerate(deltas) if abs(i - band) > 1]
    assert max(far) < 1.0
