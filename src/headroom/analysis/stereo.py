"""Stereo-image measurement.

Ratio features are stored in dB rather than as raw ratios. A raw side/mid
ratio is heavy-right-tailed and bounded below by zero, so z-scoring it -- which
the distance metric does -- is close to meaningless. In dB it is symmetric and
additive, which is also how an engineer thinks about it.

``width_per_band`` exists because a single global width number cannot express
the most common real stereo move: collapse the low end, widen the top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pyloudnorm as pyln

from headroom.audio import SILENCE_FLOOR_DB, AudioBuffer, db

from .spectral import N_BANDS, band_energy, welch_power

#: Clamp for width readings. Out-of-phase content can drive mid energy to
#: nearly zero, which would send the ratio to +inf.
WIDTH_CLAMP_DB: Final[float] = 40.0

#: Floor for mono-compatibility. A fully out-of-phase signal folds to digital
#: silence, so the true delta is -inf; reporting it as 0.0 would make the worst
#: possible mono compatibility indistinguishable from the best.
MONO_COMPAT_FLOOR_DB: Final[float] = -60.0


@dataclass(frozen=True, slots=True)
class StereoFeatures:
    mid_side_ratio_db: float
    correlation: float
    width_per_band_db: tuple[float, ...]
    mono_compat_db: float
    #: True when both channels are identical, so width and correlation are
    #: structurally fixed rather than measured properties of a stereo image.
    is_degenerate: bool


def correlation(buf: AudioBuffer) -> float:
    """Pearson correlation of L and R. 1.0 for mono, -1.0 fully out of phase."""
    left, right = buf.samples[:, 0], buf.samples[:, 1]
    if left.size < 2:
        return 1.0
    sl, sr_ = float(left.std()), float(right.std())
    if sl <= 0.0 or sr_ <= 0.0:
        return 1.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def mono_compat_db(buf: AudioBuffer) -> float:
    """Integrated-loudness change when summed to mono with ``(L + R) / 2``.

    Negative means the mono fold loses level to phase cancellation. The summing
    convention is fixed in ``AudioBuffer.mid``; a ``/ sqrt(2)`` sum would move
    every reading by 3 dB.
    """
    meter = pyln.Meter(buf.sample_rate)
    if buf.duration_s < meter.block_size:
        return 0.0
    stereo = float(meter.integrated_loudness(buf.samples))
    if not np.isfinite(stereo):
        # The source itself is silent. Folding says nothing about it.
        return 0.0
    mono = buf.mid()
    folded = float(meter.integrated_loudness(np.stack([mono, mono], axis=1)))
    if not np.isfinite(folded):
        # Total cancellation: the fold really is silence, not "no change".
        folded = SILENCE_FLOOR_DB
    return float(np.clip(folded - stereo, MONO_COMPAT_FLOOR_DB, 6.0))


def analyze_stereo(buf: AudioBuffer) -> StereoFeatures:
    mid, side = buf.mid(), buf.side()
    mid_rms, side_rms = float(np.sqrt(np.mean(mid**2))), float(np.sqrt(np.mean(side**2)))
    degenerate = side_rms <= 0.0

    ms_db = float(np.clip(db(mid_rms) - db(side_rms), -WIDTH_CLAMP_DB, WIDTH_CLAMP_DB))

    freqs, mid_p = welch_power(mid, buf.sample_rate)
    _, side_p = welch_power(side, buf.sample_rate)
    mid_bands = band_energy(freqs, mid_p, buf.nyquist) * float(mid_p.sum())
    side_bands = band_energy(freqs, side_p, buf.nyquist) * float(side_p.sum())

    width: list[float] = []
    for i in range(N_BANDS):
        ratio_db = 10.0 * np.log10(max(side_bands[i], 1e-20) / max(mid_bands[i], 1e-20))
        width.append(float(np.clip(ratio_db, -WIDTH_CLAMP_DB, WIDTH_CLAMP_DB)))

    return StereoFeatures(
        mid_side_ratio_db=ms_db,
        correlation=correlation(buf),
        width_per_band_db=tuple(width),
        mono_compat_db=mono_compat_db(buf),
        is_degenerate=degenerate,
    )
