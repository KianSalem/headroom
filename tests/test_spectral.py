from __future__ import annotations

import numpy as np
import pytest

from headroom.analysis.spectral import (
    BAND_EDGES,
    N_BANDS,
    analyze_spectral,
    band_edges_for,
    welch_power,
)
from headroom.audio import AudioBuffer

from .conftest import sine, stereo


def test_white_noise_flatness_approaches_one(white_noise: AudioBuffer) -> None:
    assert analyze_spectral(white_noise).spectral_flatness > 0.95


def test_pure_tone_flatness_approaches_zero() -> None:
    assert analyze_spectral(stereo(sine(1000.0, -20.0))).spectral_flatness < 0.01


def test_band_energy_sums_to_one(white_noise: AudioBuffer) -> None:
    """It measures balance, not level: without normalization it would
    correlate with loudness and the metric would count level twice."""
    be = analyze_spectral(white_noise).band_energy
    assert len(be) == N_BANDS
    assert sum(be) == pytest.approx(1.0, abs=1e-9)


def test_band_energy_is_level_invariant() -> None:
    """Scaling the signal must leave the balance untouched."""
    quiet = analyze_spectral(stereo(sine(1000.0, -40.0))).band_energy
    loud = analyze_spectral(stereo(sine(1000.0, -6.0))).band_energy
    np.testing.assert_allclose(quiet, loud, atol=1e-6)


def test_white_noise_band_energy_is_proportional_to_bandwidth(
    white_noise: AudioBuffer,
) -> None:
    """Analytic: white noise has constant power per Hz, so energy in each
    log-spaced band must be proportional to that band's width. The top band is
    ~300x wider than the bottom one."""
    be = analyze_spectral(white_noise).band_energy
    widths = np.diff(np.asarray(BAND_EDGES))
    expected = widths / widths.sum()
    np.testing.assert_allclose(np.asarray(be), expected, rtol=0.05)


def test_white_noise_tilt_is_flat(white_noise: AudioBuffer) -> None:
    assert abs(analyze_spectral(white_noise).spectral_tilt) < 0.2


def test_tone_centroid_lands_on_the_tone() -> None:
    assert analyze_spectral(stereo(sine(2000.0, -20.0))).spectral_centroid == pytest.approx(
        2000.0, rel=0.02
    )


def test_lowpassed_noise_has_lower_centroid_and_negative_tilt(
    white_noise: AudioBuffer,
) -> None:
    from scipy import signal as sig

    sos = sig.butter(6, 2000.0 / (white_noise.sample_rate / 2), btype="lowpass", output="sos")
    dark = white_noise.replace_samples(
        np.asarray(sig.sosfiltfilt(sos, white_noise.samples, axis=0), dtype=np.float64)
    )
    bright = analyze_spectral(white_noise)
    muffled = analyze_spectral(dark)
    assert muffled.spectral_centroid < bright.spectral_centroid
    assert muffled.spectral_tilt < bright.spectral_tilt
    assert muffled.spectral_rolloff_85 < bright.spectral_rolloff_85


def test_band_edges_clamp_to_nyquist() -> None:
    """At 44.1 kHz the 20 kHz edge fits; the clamp exists so a lower rate does
    not silently read zero energy in the top band."""
    assert band_edges_for(22050.0)[-1] == 20000.0
    assert band_edges_for(16000.0)[-1] == 16000.0


def test_welch_handles_a_signal_shorter_than_the_window() -> None:
    freqs, power = welch_power(np.zeros(512), 48000)
    assert freqs.size == power.size > 0
    assert np.all(np.isfinite(power))
