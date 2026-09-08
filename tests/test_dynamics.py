from __future__ import annotations

import pytest

from headroom.analysis.dynamics import analyze_dynamics, crest_factor_db
from headroom.analysis.loudness import analyze_loudness
from headroom.audio import AudioBuffer

from .conftest import sine, stereo


def test_sine_crest_factor_is_exactly_3_01_db() -> None:
    """Analytic: peak/rms of a sine is sqrt(2), so 20*log10(sqrt(2)) = 3.0103."""
    assert crest_factor_db(stereo(sine(1000.0, -20.0)).samples) == pytest.approx(3.0103, abs=1e-3)


def test_crest_factor_is_level_invariant() -> None:
    """It is a ratio, so scaling the signal must not change it."""
    a = crest_factor_db(stereo(sine(1000.0, -20.0)).samples)
    b = crest_factor_db(stereo(sine(1000.0, -6.0)).samples)
    assert a == pytest.approx(b, abs=1e-6)


def test_plr_is_true_peak_minus_loudness(white_noise: AudioBuffer) -> None:
    loud = analyze_loudness(white_noise)
    dyn = analyze_dynamics(white_noise, loud.true_peak_dbtp, loud.lufs_integrated)
    assert dyn.plr == pytest.approx(loud.true_peak_dbtp - loud.lufs_integrated, abs=1e-12)


def test_silence_gives_zero_crest_not_nan(silence: AudioBuffer) -> None:
    assert crest_factor_db(silence.samples) == 0.0
