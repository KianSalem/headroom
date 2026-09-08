from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from headroom.analysis.loudness import analyze_loudness, sample_peak_dbfs, true_peak_dbtp
from headroom.audio import SILENCE_FLOOR_DB, AudioBuffer, save

from .conftest import sine, stereo


def test_1k_sine_reads_its_own_level() -> None:
    """Phase 1 acceptance: LUFS within 0.1 LU. K-weighting is ~unity at 1 kHz,
    so a -20 dBFS 1 kHz sine in both channels must read about -20 LUFS."""
    f = analyze_loudness(stereo(sine(1000.0, -20.0)))
    assert f.lufs_integrated == pytest.approx(-20.0, abs=0.1)


@pytest.mark.parametrize("dbfs", [-30.0, -20.0, -12.0, -6.0])
def test_loudness_tracks_level_exactly(dbfs: float) -> None:
    """A pure level change must move LUFS by the same amount."""
    ref = analyze_loudness(stereo(sine(1000.0, -20.0))).lufs_integrated
    got = analyze_loudness(stereo(sine(1000.0, dbfs))).lufs_integrated
    assert got - ref == pytest.approx(dbfs - (-20.0), abs=0.02)


@pytest.mark.parametrize("dbfs", [-1.0, -0.2, -6.0])
def test_true_peak_of_a_sine_equals_its_amplitude(dbfs: float) -> None:
    """Analytic ground truth: the continuous peak of a sine is its amplitude,
    independent of sampling. 997 Hz deliberately does not divide the sample
    rate, so the sample peak falls short and only a true-peak meter recovers it.
    """
    buf = stereo(sine(997.0, dbfs))
    assert true_peak_dbtp(buf) == pytest.approx(dbfs, abs=0.05)


def test_true_peak_exceeds_sample_peak_for_an_offgrid_sine() -> None:
    buf = stereo(sine(997.0, -1.0))
    assert true_peak_dbtp(buf) > sample_peak_dbfs(buf)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_true_peak_agrees_with_ffmpeg(tmp_path: Path) -> None:
    """Independent cross-check against ffmpeg's ebur128. ffmpeg reports true
    peak at 0.1 dB resolution, so the tolerance covers its rounding."""
    buf = stereo(sine(997.0, -1.0))
    path = tmp_path / "tp.wav"
    save(buf, path, subtype="FLOAT")
    out = subprocess.run(
        [
            "ffmpeg",
            "-nostats",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stderr
    tail = out.split("True peak:")[-1]
    ffmpeg_tp = float(
        next(t for t in tail.split() if t.replace("-", "").replace(".", "").isdigit())
    )
    assert true_peak_dbtp(buf) == pytest.approx(ffmpeg_tp, abs=0.06)


def test_silence_is_floored_not_infinite(silence: AudioBuffer) -> None:
    f = analyze_loudness(silence)
    assert f.lufs_integrated == SILENCE_FLOOR_DB
    assert f.true_peak_dbtp == SILENCE_FLOOR_DB
    assert np.isfinite(f.lufs_integrated)
    assert f.short_term_degenerate


def test_short_signal_flags_degenerate_short_term() -> None:
    """Under one 3 s window the percentiles are not measurable, and must be
    flagged rather than invented."""
    f = analyze_loudness(stereo(sine(1000.0, -20.0, seconds=1.0)))
    assert f.short_term_degenerate
    assert f.lufs_short_p50 == f.lufs_integrated


def test_short_term_percentiles_separate_quiet_from_loud() -> None:
    """The percentiles are ungated, so they must span the real 28 LU range."""
    quiet = sine(1000.0, -40.0, seconds=5.0)
    loud = sine(1000.0, -12.0, seconds=5.0)
    f = analyze_loudness(stereo(np.concatenate([quiet, loud])))
    assert f.lufs_short_p10 < f.lufs_short_p50 < f.lufs_short_p90
    assert f.lufs_short_p90 - f.lufs_short_p10 > 20.0


def test_lra_applies_the_r128_relative_gate() -> None:
    """LRA is not min-to-max. EBU R128 discards blocks more than 20 LU below
    the ungated loudness, so a -40 dB passage against a -12 dB one is gated
    out and LRA stays small even though the short-term spread is 28 LU. This
    is correct behaviour and worth pinning: a naive reading of LRA as "dynamic
    range" would make the loudness specialist chase a number that cannot move.
    """
    f = analyze_loudness(
        stereo(np.concatenate([sine(1000.0, -40.0, 5.0), sine(1000.0, -12.0, 5.0)]))
    )
    assert f.lufs_short_p90 - f.lufs_short_p10 > 25.0
    assert f.lra < 10.0


def test_lra_measures_range_inside_the_gate() -> None:
    """With both passages inside the 20 LU gate, LRA does track the range."""
    f = analyze_loudness(
        stereo(np.concatenate([sine(1000.0, -24.0, 5.0), sine(1000.0, -12.0, 5.0)]))
    )
    assert f.lra > 8.0


def test_stationary_signal_has_near_zero_lra(white_noise: AudioBuffer) -> None:
    assert analyze_loudness(white_noise).lra < 1.5
