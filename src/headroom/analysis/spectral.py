"""Spectral measurement.

Every spectral feature is derived from a single object: the Welch-averaged
power spectrum of the whole signal, pooled across channels. Deriving them all
from one spectrum rather than from independent per-frame estimates keeps them
mutually consistent -- ``band_energy``, ``spectral_centroid`` and
``spectral_tilt`` are then guaranteed to describe the same measured spectrum.

Band energy is normalized to sum to one across the nine bands so it measures
*balance*, not level. Without that it would correlate with loudness and the
distance metric would count level error twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy import signal

from headroom.audio import AudioBuffer, Samples

#: Nine log-spaced bands. Edges in Hz; band i spans EDGES[i] to EDGES[i+1].
BAND_EDGES: Final[tuple[float, ...]] = (
    20.0,
    60.0,
    120.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
    20000.0,
)
N_BANDS: Final[int] = len(BAND_EDGES) - 1

#: 8192 at 48 kHz gives 5.9 Hz resolution, so the 20-60 Hz band gets ~7 bins
#: rather than the ~3 a 4096 window would give.
N_FFT: Final[int] = 8192
OVERLAP: Final[int] = 6144

#: Tilt is fitted over the middle of the spectrum. Including the extremes
#: would measure the anti-alias and high-pass rolloffs instead of the balance.
TILT_FIT_LO_HZ: Final[float] = 60.0
TILT_FIT_HI_HZ: Final[float] = 12000.0

#: Power floor, so log of an empty bin is finite.
_P_FLOOR: Final[float] = 1e-20


@dataclass(frozen=True, slots=True)
class SpectralFeatures:
    band_energy: tuple[float, ...]
    spectral_centroid: float
    spectral_flatness: float
    spectral_rolloff_85: float
    spectral_tilt: float


def welch_power(x: Samples, sample_rate: int) -> tuple[Samples, Samples]:
    """Averaged power spectrum. ``x`` may be 1-D or ``(frames, channels)``.

    Multi-channel input is pooled by averaging the per-channel spectra, which
    preserves out-of-phase energy that a mid-channel-only analysis would lose.
    """
    nperseg = min(N_FFT, x.shape[0]) if x.shape[0] else N_FFT
    noverlap = min(OVERLAP, max(nperseg - 1, 0))
    freqs, pxx = signal.welch(
        x, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, axis=0, detrend=False
    )
    p = np.asarray(pxx, dtype=np.float64)
    if p.ndim == 2:
        p = p.mean(axis=1)
    return np.asarray(freqs, dtype=np.float64), np.maximum(p, _P_FLOOR)


def band_edges_for(nyquist: float) -> tuple[float, ...]:
    """Band edges clamped to Nyquist, so a 44.1 kHz source does not ask for
    energy above 22.05 kHz and silently read zero."""
    return tuple(min(e, nyquist) for e in BAND_EDGES)


def band_energy(freqs: Samples, power: Samples, nyquist: float) -> npt.NDArray[np.float64]:
    """Normalized energy per band; sums to 1.0 across the nine bands."""
    edges = band_edges_for(nyquist)
    out = np.zeros(N_BANDS, dtype=np.float64)
    for i in range(N_BANDS):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        mask = (freqs >= lo) & (freqs < hi)
        out[i] = float(power[mask].sum()) if mask.any() else 0.0
    total = float(out.sum())
    if total <= 0.0:
        return np.full(N_BANDS, 1.0 / N_BANDS, dtype=np.float64)
    return out / total


def _analysis_mask(freqs: Samples, nyquist: float) -> npt.NDArray[np.bool_]:
    return (freqs >= BAND_EDGES[0]) & (freqs <= min(BAND_EDGES[-1], nyquist))


def spectral_centroid(freqs: Samples, power: Samples, nyquist: float) -> float:
    m = _analysis_mask(freqs, nyquist)
    p = power[m]
    total = float(p.sum())
    if total <= 0.0:
        return 0.0
    return float((freqs[m] * p).sum() / total)


def spectral_flatness(freqs: Samples, power: Samples, nyquist: float) -> float:
    """Geometric mean over arithmetic mean. 1.0 for white noise, ~0 for a tone."""
    p = power[_analysis_mask(freqs, nyquist)]
    if p.size == 0:
        return 0.0
    arithmetic = float(p.mean())
    if arithmetic <= 0.0:
        return 0.0
    geometric = float(np.exp(np.mean(np.log(p))))
    return float(np.clip(geometric / arithmetic, 0.0, 1.0))


def spectral_rolloff_85(freqs: Samples, power: Samples, nyquist: float) -> float:
    m = _analysis_mask(freqs, nyquist)
    p, f = power[m], freqs[m]
    total = float(p.sum())
    if total <= 0.0:
        return 0.0
    idx = int(np.searchsorted(np.cumsum(p), 0.85 * total))
    return float(f[min(idx, f.size - 1)])


def spectral_tilt(freqs: Samples, power: Samples, nyquist: float) -> float:
    """Slope of a linear fit of power in dB against log2(frequency), in dB/octave."""
    hi = min(TILT_FIT_HI_HZ, nyquist * 0.9)
    m = (freqs >= TILT_FIT_LO_HZ) & (freqs <= hi) & (power > _P_FLOOR)
    if int(m.sum()) < 8:
        return 0.0
    x = np.log2(freqs[m])
    y = 10.0 * np.log10(power[m])
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def analyze_spectral(buf: AudioBuffer) -> SpectralFeatures:
    freqs, power = welch_power(buf.samples, buf.sample_rate)
    nq = buf.nyquist
    return SpectralFeatures(
        band_energy=tuple(float(v) for v in band_energy(freqs, power, nq)),
        spectral_centroid=spectral_centroid(freqs, power, nq),
        spectral_flatness=spectral_flatness(freqs, power, nq),
        spectral_rolloff_85=spectral_rolloff_85(freqs, power, nq),
        spectral_tilt=spectral_tilt(freqs, power, nq),
    )
