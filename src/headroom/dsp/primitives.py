"""DSP primitives that ``pedalboard`` does not provide.

Three things are missing from the built-ins and all three are load-bearing:

*True-peak limiting.* ``pedalboard.Limiter`` takes a sample-peak threshold with
no lookahead and no oversampling, so it cannot honour a dBTP ceiling. Since
``true_peak_dbtp`` is a scored feature, a limiter that misses the ceiling would
have the loudness specialist chasing a target the tool cannot reach -- which
shows up in the loop as an oscillation that is really a tooling bug.

*Downward expansion.* Without an expander the ``over_compress`` degradation is
unrecoverable by construction: a compressor cannot undo compression.

*Band-limited stereo width.* A global width control cannot express "tighten the
lows, widen the top", which is the most common real stereo move.

Gain curves are computed at a control rate and linearly interpolated up to the
sample rate. That is both how hardware detectors behave and a ~100x speedup
over a per-sample Python loop; gain curves are slow-moving, so the
interpolation is inaudible and fully deterministic.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
from scipy import ndimage, signal

from headroom.audio import Samples

#: Control-rate hop in samples. 8 samples is 0.17 ms at 48 kHz -- finer than
#: any attack time the bounds allow.
CONTROL_HOP: Final[int] = 8

#: Limiter lookahead. The gain curve must start ducking before the peak
#: arrives, or the peak passes through un-attenuated.
LOOKAHEAD_MS: Final[float] = 1.5

_EPS: Final[float] = 1e-12


def _control_peak(x: Samples, hop: int = CONTROL_HOP) -> Samples:
    """Stereo-linked peak per control block. Linking the channels keeps the
    stereo image stable: independent per-channel gain would shift the image
    every time one side is louder."""
    n_blocks = max(x.shape[0] // hop, 1)
    trimmed = x[: n_blocks * hop]
    if trimmed.size == 0:
        return np.zeros(1, dtype=np.float64)
    return np.abs(trimmed).reshape(n_blocks, -1).max(axis=1).astype(np.float64)


def _coef(time_ms: float, control_rate: float) -> float:
    """One-pole coefficient for a time constant in ms."""
    tau = max(time_ms, 1e-3) / 1000.0
    return float(np.exp(-1.0 / max(tau * control_rate, 1e-9)))


def _follow(env: Samples, attack_coef: float, release_coef: float) -> Samples:
    """Asymmetric envelope follower: fast on the way up, slow on the way down."""
    out = np.empty_like(env)
    prev = float(env[0])
    for i in range(env.size):
        v = float(env[i])
        c = attack_coef if v > prev else release_coef
        prev = c * prev + (1.0 - c) * v
        out[i] = prev
    return out


def _release_only(gain: Samples, release_coef: float) -> Samples:
    """Gain may fall instantly but recovers on the release time constant."""
    out = np.empty_like(gain)
    prev = float(gain[0])
    for i in range(gain.size):
        v = float(gain[i])
        prev = v if v < prev else release_coef * prev + (1.0 - release_coef) * v
        out[i] = prev
    return out


def _to_sample_rate(control: Samples, n: int, hop: int = CONTROL_HOP) -> Samples:
    """Linearly interpolate a control-rate curve up to ``n`` samples."""
    if control.size == 1:
        return np.full(n, float(control[0]), dtype=np.float64)
    ctrl_idx = np.arange(control.size, dtype=np.float64) * hop
    return np.interp(np.arange(n, dtype=np.float64), ctrl_idx, control).astype(np.float64)


def expander(
    x: Samples,
    sample_rate: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
) -> Samples:
    """Downward expander. Below the threshold, level is pushed further down,
    which increases crest factor -- the inverse of over-compression.

    For an input level ``L`` below threshold ``T``, output level is
    ``T + (L - T) * ratio``, so the applied gain is ``(L - T) * (ratio - 1)``.
    """
    if x.shape[0] == 0 or ratio <= 1.0:
        return x.copy()
    control_rate = sample_rate / CONTROL_HOP
    env = _follow(_control_peak(x), _coef(attack_ms, control_rate), _coef(release_ms, control_rate))
    env_db = 20.0 * np.log10(np.maximum(env, _EPS))
    over = env_db - threshold_db
    gain_db = np.where(over < 0.0, over * (ratio - 1.0), 0.0)
    gain = np.power(10.0, gain_db / 20.0)
    return np.asarray(x * _to_sample_rate(gain, x.shape[0])[:, None], dtype=np.float64)


def true_peak_limiter(
    x: Samples,
    sample_rate: int,
    ceiling_dbtp: float,
    release_ms: float,
    oversample: int = 4,
) -> Samples:
    """Lookahead limiter with a guaranteed true-peak ceiling.

    The gain requirement is derived from the *oversampled* peak, so
    inter-sample peaks are attenuated rather than merely the sample peaks. A
    running minimum over the lookahead window guarantees the gain is already
    down when the peak arrives. A final static trim makes the ceiling a
    guarantee rather than an aspiration: without it the loudness specialist
    would be given a target its own tool cannot hit.
    """
    n = x.shape[0]
    if n == 0:
        return x.copy()
    ceiling = float(np.power(10.0, ceiling_dbtp / 20.0))

    up = np.asarray(signal.resample_poly(x, oversample, 1, axis=0), dtype=np.float64)
    peak = _control_peak(up, CONTROL_HOP * oversample)
    needed = np.minimum(1.0, ceiling / np.maximum(peak, _EPS))

    control_rate = sample_rate / CONTROL_HOP
    look = max(round(LOOKAHEAD_MS * 1e-3 * control_rate), 1)
    ducked = ndimage.minimum_filter1d(needed, size=2 * look + 1, mode="nearest")
    smoothed = _release_only(ducked, _coef(release_ms, control_rate))

    y = np.asarray(x * _to_sample_rate(smoothed, n)[:, None], dtype=np.float64)

    up_y = np.asarray(signal.resample_poly(y, oversample, 1, axis=0), dtype=np.float64)
    achieved = float(np.max(np.abs(up_y))) if up_y.size else 0.0
    if achieved > ceiling and achieved > 0.0:
        y *= ceiling / achieved
    return y


def _band_sos(band: int, sample_rate: int, edges: tuple[float, ...]) -> npt.NDArray[np.float64]:
    """Zero-phase filter for one band.

    The lowest band is a low-pass so nothing below 20 Hz is excluded. The
    highest band is a band-pass up to the top analysis edge (20 kHz): at the
    sample rates this project accepts that edge is always below Nyquist, and
    nothing above it is measured, so a high-pass would only add out-of-band
    energy the metric cannot see.
    """
    nyq = sample_rate / 2.0
    lo, hi = edges[band], min(edges[band + 1], nyq * 0.999)
    if band == 0:
        sos = signal.butter(4, hi / nyq, btype="lowpass", output="sos")
    else:
        sos = signal.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return np.asarray(sos, dtype=np.float64)


def stereo_width(
    x: Samples,
    sample_rate: int,
    width: float,
    band: int | None,
    edges: tuple[float, ...],
) -> Samples:
    """Scale side energy, optionally within one band only.

    Uses zero-phase filtering (``sosfiltfilt``) on the side signal. A
    minimum-phase filter would rotate phase inside the band and, on
    recombination with the unfiltered mid, smear the stereo image it is
    supposed to be adjusting.
    """
    if x.shape[0] == 0:
        return x.copy()
    mid = (x[:, 0] + x[:, 1]) / 2.0
    side = (x[:, 0] - x[:, 1]) / 2.0

    if band is None:
        side_out = side * width
    else:
        sos = _band_sos(band, sample_rate, edges)
        in_band = np.asarray(signal.sosfiltfilt(sos, side), dtype=np.float64)
        side_out = side - in_band + in_band * width

    return np.stack([mid + side_out, mid - side_out], axis=1).astype(np.float64)
