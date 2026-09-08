from __future__ import annotations

import pytest

from headroom.analysis.transient import analyze_transient
from headroom.audio import AudioBuffer

from .conftest import ramp_clicks, sine, stereo, white


@pytest.mark.parametrize(
    ("ramp_ms", "tolerance_ms"),
    [(1.0, 0.6), (5.0, 1.5), (20.0, 2.0), (50.0, 4.0)],
)
def test_attack_time_tracks_a_known_linear_ramp(ramp_ms: float, tolerance_ms: float) -> None:
    """Analytic: the 10-90% rise of a linear ramp is exactly 0.8x its length."""
    got = analyze_transient(stereo(ramp_clicks(ramp_ms / 1000.0))).attack_time_p50
    assert got == pytest.approx(0.8 * ramp_ms, abs=tolerance_ms)


def test_attack_time_is_monotone_in_ramp_length() -> None:
    times = [
        analyze_transient(stereo(ramp_clicks(ms / 1000.0))).attack_time_p50
        for ms in (1.0, 5.0, 20.0, 50.0)
    ]
    assert times == sorted(times)


def test_onset_rate_matches_a_known_tempo() -> None:
    """120 BPM is 2 hits per second."""
    got = analyze_transient(stereo(ramp_clicks(0.002, bpm=120.0))).onset_rate
    assert got == pytest.approx(2.0, abs=0.2)


def test_percussive_ratio_discriminates_material() -> None:
    tone = analyze_transient(stereo(sine(440.0, -20.0))).percussive_ratio
    drums = analyze_transient(stereo(ramp_clicks(0.001))).percussive_ratio
    noise = analyze_transient(stereo(white())).percussive_ratio
    assert tone < 0.05
    assert drums > 0.9
    assert 0.3 < noise < 0.7


def test_silence_is_flagged_degenerate(silence: AudioBuffer) -> None:
    t = analyze_transient(silence)
    assert t.attack_degenerate
    assert t.n_onsets == 0
    assert t.percussive_ratio == 0.0
