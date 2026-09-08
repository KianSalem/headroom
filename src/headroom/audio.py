"""Audio I/O and the canonical in-memory buffer.

Convention: samples are ``float64`` with shape ``(frames, channels)``, channels
last. That matches ``soundfile`` and ``pyloudnorm``. ``librosa`` and
``pedalboard`` want ``(channels, frames)``, so every transpose is explicit and
local to its call site rather than implied by a global convention.

Analysis runs at the source sample rate. Nothing is resampled for measurement:
resampling perturbs true-peak and inserts a filter stage for no analytic gain.
Comparability across sample rates is instead guaranteed by specifying every
time constant in seconds and every frequency in Hz.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import soundfile as sf

Samples = npt.NDArray[np.float64]

#: Below this rate the 20 kHz top band edge exceeds Nyquist and the spectral
#: feature set stops being comparable. Sources below it are rejected at load.
MIN_SAMPLE_RATE: Final[int] = 44100

#: Loudness and peak readings for digital silence are ``-inf``. Clamp to this
#: floor so downstream arithmetic (z-scores, deltas) stays finite.
SILENCE_FLOOR_DB: Final[float] = -120.0


class AudioError(ValueError):
    """Raised when audio cannot be used for analysis."""


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    """Immutable stereo audio.

    Mono input is duplicated to two channels so every downstream feature has a
    defined value; ``was_mono`` records that so stereo features can be reported
    as degenerate rather than silently meaningful.
    """

    samples: Samples
    sample_rate: int
    was_mono: bool = False

    def __post_init__(self) -> None:
        if self.samples.ndim != 2 or self.samples.shape[1] != 2:
            raise AudioError(f"expected (frames, 2) stereo float64, got shape {self.samples.shape}")
        if self.samples.dtype != np.float64:
            raise AudioError(f"expected float64, got {self.samples.dtype}")
        if self.sample_rate < MIN_SAMPLE_RATE:
            raise AudioError(
                f"sample rate {self.sample_rate} below minimum {MIN_SAMPLE_RATE}; "
                "the 20 kHz band edge would exceed Nyquist"
            )

    @property
    def n_frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.sample_rate

    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2.0

    def mid(self) -> Samples:
        """Mono sum, ``(L + R) / 2``. The convention is fixed here because a
        ``/ sqrt(2)`` sum would shift every mono-compatibility reading by 3 dB."""
        return np.asarray(self.samples.mean(axis=1), dtype=np.float64)

    def side(self) -> Samples:
        return np.asarray((self.samples[:, 0] - self.samples[:, 1]) / 2.0, dtype=np.float64)

    def for_librosa(self) -> Samples:
        """``(channels, frames)`` view, as librosa expects."""
        return np.ascontiguousarray(self.samples.T)

    def content_hash(self) -> str:
        """Stable hash of the exact sample bytes plus rate. Used as a render
        cache key and recorded in traces so a result is tied to its input."""
        h = hashlib.blake2b(digest_size=16)
        h.update(np.ascontiguousarray(self.samples, dtype=np.float64).tobytes())
        h.update(str(self.sample_rate).encode())
        return h.hexdigest()

    def replace_samples(self, samples: Samples) -> AudioBuffer:
        return AudioBuffer(samples=samples, sample_rate=self.sample_rate, was_mono=self.was_mono)


def load(path: str | Path) -> AudioBuffer:
    """Read a file into an :class:`AudioBuffer`.

    Always float64 and always stereo. Channel counts above two are rejected
    rather than downmixed: silently folding a 5.1 file would produce readings
    that look valid and are not.
    """
    data, rate = sf.read(str(path), dtype="float64", always_2d=True)
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape[1] == 1:
        return AudioBuffer(np.repeat(arr, 2, axis=1), int(rate), was_mono=True)
    if arr.shape[1] != 2:
        raise AudioError(f"{path}: expected mono or stereo, got {arr.shape[1]} channels")
    return AudioBuffer(arr, int(rate))


def save(buf: AudioBuffer, path: str | Path, subtype: str = "PCM_24") -> None:
    """Write a buffer to disk. 24-bit PCM by default: enough headroom that the
    quantization floor sits below anything the feature vector measures."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), buf.samples, buf.sample_rate, subtype=subtype)


def db(x: float, floor: float = SILENCE_FLOOR_DB) -> float:
    """Amplitude ratio to dB, floored so silence does not produce ``-inf``."""
    if not np.isfinite(x) or x <= 0.0:
        return floor
    return max(float(20.0 * np.log10(x)), floor)
