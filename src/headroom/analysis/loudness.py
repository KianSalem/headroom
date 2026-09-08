"""Loudness and peak measurement.

BS.1770-4 loudness comes from ``pyloudnorm``; the K-weighting filter is not
hand-rolled. True peak is implemented here because ``pyloudnorm`` has no
true-peak meter -- it measures loudness only. The implementation is validated
against ``ffmpeg -af ebur128:peak=true`` in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pyloudnorm as pyln
from scipy import signal

from headroom.audio import SILENCE_FLOOR_DB, AudioBuffer, Samples, db

#: EBU R128 short-term window.
SHORT_TERM_S: Final[float] = 3.0

#: Hop for short-term percentiles. Shorter than the window, so a 30 s track
#: yields ~28 observations instead of 10 -- percentiles over 10 points are
#: mostly noise.
SHORT_TERM_HOP_S: Final[float] = 1.0

#: BS.1770-4 Annex 2 specifies at least 4x oversampling for true-peak.
TRUE_PEAK_OVERSAMPLE: Final[int] = 4


@dataclass(frozen=True, slots=True)
class LoudnessFeatures:
    lufs_integrated: float
    lufs_short_p10: float
    lufs_short_p50: float
    lufs_short_p90: float
    lra: float
    true_peak_dbtp: float
    sample_peak_dbfs: float
    #: True when the signal was too short for a 3 s window, so the short-term
    #: percentiles fall back to the integrated value rather than being invented.
    short_term_degenerate: bool


def sample_peak_dbfs(buf: AudioBuffer) -> float:
    return db(float(np.max(np.abs(buf.samples))) if buf.n_frames else 0.0)


def true_peak_dbtp(buf: AudioBuffer, oversample: int = TRUE_PEAK_OVERSAMPLE) -> float:
    """Inter-sample true peak, in dBTP.

    Polyphase-upsamples each channel and takes the maximum absolute value. The
    reconstructed waveform between samples can exceed the sample peak, which is
    why a limiter targeting sample peak still clips a downstream converter.
    """
    if buf.n_frames == 0:
        return SILENCE_FLOOR_DB
    up = signal.resample_poly(buf.samples, oversample, 1, axis=0)
    return db(float(np.max(np.abs(np.asarray(up, dtype=np.float64)))))


def _short_term_blocks(buf: AudioBuffer) -> Samples:
    """Per-window loudness over 3 s windows.

    ``pyloudnorm`` exposes block loudness as the ``blockwise_loudness``
    attribute, populated as a side effect of ``integrated_loudness``. Windows
    below the absolute gate come back non-finite and are dropped.
    """
    meter = pyln.Meter(
        buf.sample_rate,
        block_size=SHORT_TERM_S,
        overlap=1.0 - (SHORT_TERM_HOP_S / SHORT_TERM_S),
    )
    meter.integrated_loudness(buf.samples)
    blocks = np.asarray(meter.blockwise_loudness, dtype=np.float64).ravel()
    return blocks[np.isfinite(blocks)]


def analyze_loudness(buf: AudioBuffer) -> LoudnessFeatures:
    meter = pyln.Meter(buf.sample_rate)
    long_enough = buf.duration_s >= SHORT_TERM_S

    integrated = (
        float(meter.integrated_loudness(buf.samples))
        if buf.duration_s >= meter.block_size
        else SILENCE_FLOOR_DB
    )
    if not np.isfinite(integrated):
        integrated = SILENCE_FLOOR_DB

    lra = float(meter.loudness_range(buf.samples)) if long_enough else 0.0
    if not np.isfinite(lra):
        lra = 0.0

    blocks = _short_term_blocks(buf) if long_enough else np.array([], dtype=np.float64)
    if blocks.size:
        p10, p50, p90 = (float(v) for v in np.percentile(blocks, [10.0, 50.0, 90.0]))
        degenerate = False
    else:
        p10 = p50 = p90 = integrated
        degenerate = True

    return LoudnessFeatures(
        lufs_integrated=integrated,
        lufs_short_p10=p10,
        lufs_short_p50=p50,
        lufs_short_p90=p90,
        lra=lra,
        true_peak_dbtp=true_peak_dbtp(buf),
        sample_peak_dbfs=sample_peak_dbfs(buf),
        short_term_degenerate=degenerate,
    )
