from __future__ import annotations

import numpy as np
import pytest

from headroom.analysis.stereo import MONO_COMPAT_FLOOR_DB, analyze_stereo, mono_compat_db
from headroom.audio import AudioBuffer

from .conftest import sine, stereo, white


def test_identical_channels_correlate_at_one(white_noise: AudioBuffer) -> None:
    st = analyze_stereo(white_noise)
    assert st.correlation == pytest.approx(1.0, abs=1e-9)
    assert st.is_degenerate


def test_inverted_channels_correlate_at_minus_one() -> None:
    w = white()
    assert analyze_stereo(stereo(w, -w)).correlation == pytest.approx(-1.0, abs=1e-9)


def test_independent_channels_correlate_near_zero(uncorrelated: AudioBuffer) -> None:
    assert abs(analyze_stereo(uncorrelated).correlation) < 0.02


def test_mono_fold_of_uncorrelated_channels_loses_exactly_3db(
    uncorrelated: AudioBuffer,
) -> None:
    """Analytic: summing two uncorrelated equal-power signals as (L+R)/2 gives
    a 3.01 dB level loss."""
    assert mono_compat_db(uncorrelated) == pytest.approx(-3.01, abs=0.1)


def test_mono_fold_of_identical_channels_is_lossless(white_noise: AudioBuffer) -> None:
    assert mono_compat_db(white_noise) == pytest.approx(0.0, abs=0.01)


def test_out_of_phase_mono_fold_reports_the_floor() -> None:
    """The fold is digital silence. Reporting 0.0 dB would make the worst
    possible mono compatibility indistinguishable from the best."""
    w = white()
    assert mono_compat_db(stereo(w, -w)) == MONO_COMPAT_FLOOR_DB


def test_silence_mono_compat_is_zero(silence: AudioBuffer) -> None:
    assert mono_compat_db(silence) == 0.0


def test_width_per_band_is_reported_per_band(uncorrelated: AudioBuffer) -> None:
    st = analyze_stereo(uncorrelated)
    assert len(st.width_per_band_db) == 9
    assert all(np.isfinite(v) for v in st.width_per_band_db)


def test_more_side_energy_lowers_the_mid_side_ratio() -> None:
    """mid_side_ratio_db is mid-over-side, so adding side energy must lower it."""
    mid = sine(440.0, -20.0)
    side = white(seed=5)

    def ratio(side_gain: float) -> float:
        buf = stereo(mid + side_gain * side, mid - side_gain * side)
        return analyze_stereo(buf).mid_side_ratio_db

    readings = [ratio(g) for g in (0.02, 0.1, 0.5)]
    assert readings == sorted(readings, reverse=True)


def test_width_per_band_moves_only_the_band_it_targets() -> None:
    """A global width number cannot express "tighten the lows, widen the top",
    which is why width is measured per band."""
    from headroom.analysis.spectral import BAND_EDGES
    from headroom.dsp.primitives import stereo_width

    src = stereo(white(seed=1), white(seed=2))
    before = analyze_stereo(src).width_per_band_db
    target_band = 2
    narrowed = src.replace_samples(
        stereo_width(src.samples, src.sample_rate, 0.0, target_band, BAND_EDGES)
    )
    after = analyze_stereo(narrowed).width_per_band_db

    deltas = [a - b for a, b in zip(after, before, strict=True)]
    assert deltas[target_band] < -10.0
    others = [abs(d) for i, d in enumerate(deltas) if abs(i - target_band) > 1]
    assert max(others) < 1.0


def test_all_ratio_features_are_in_db_not_raw_ratios() -> None:
    """Raw side/mid ratios are heavy-right-tailed and bounded below by zero,
    so the metric normalizes dB values instead. A raw ratio would be negative
    -> impossible, whereas dB values legitimately are."""
    w = white()
    st = analyze_stereo(stereo(w, -w))
    assert st.mid_side_ratio_db < 0.0
