"""Dynamic-range measurement.

``plr`` is retained because engineers read it directly, but note it is exactly
``true_peak_dbtp - lufs_integrated``. It is therefore excluded from the scored
feature set (see ``target.distance``) so loudness error is not counted twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from headroom.audio import AudioBuffer, db

CREST_WINDOW_S: Final[float] = 3.0
CREST_HOP_S: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class DynamicsFeatures:
    crest_factor_db: float
    plr: float
    crest_short_p50: float


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def crest_factor_db(samples: np.ndarray) -> float:
    """``20*log10(peak / rms)`` over the whole signal, both channels pooled."""
    if samples.size == 0:
        return 0.0
    rms = _rms(samples)
    if rms <= 0.0:
        return 0.0
    return db(float(np.max(np.abs(samples)))) - db(rms)


def analyze_dynamics(
    buf: AudioBuffer, true_peak_dbtp: float, lufs_integrated: float
) -> DynamicsFeatures:
    win = int(CREST_WINDOW_S * buf.sample_rate)
    hop = int(CREST_HOP_S * buf.sample_rate)

    per_window: list[float] = []
    if buf.n_frames >= win:
        for start in range(0, buf.n_frames - win + 1, hop):
            block = buf.samples[start : start + win]
            if _rms(block) > 0.0:
                per_window.append(crest_factor_db(block))

    return DynamicsFeatures(
        crest_factor_db=crest_factor_db(buf.samples),
        plr=true_peak_dbtp - lufs_integrated,
        crest_short_p50=(
            float(np.median(per_window)) if per_window else crest_factor_db(buf.samples)
        ),
    )
