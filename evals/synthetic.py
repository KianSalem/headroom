"""A synthetic corpus, so the pipeline runs with no downloads.

MUSDB18-HQ needs a Zenodo access request and cannot be redistributed, which
would otherwise mean a stranger cloning this repo can read the code but not run
it. These tracks are deliberately *not* presented as results material -- they
are stationary, synthetic and much easier to repair than real music, so numbers
from them describe the harness rather than the systems.

They exist for three jobs: proving the pipeline end to end, giving CI something
to exercise, and letting anyone reproduce the mechanics before deciding whether
to request the real corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf

SR: Final[int] = 48000


@dataclass(frozen=True, slots=True)
class Voicing:
    """One synthetic track's character, chosen to span the feature space so a
    degradation cannot be repaired the same way on every track."""

    seed: int
    bass: float
    bright: float
    transient: float
    width: float
    bpm: float


VOICINGS: Final[tuple[Voicing, ...]] = (
    Voicing(1, bass=0.18, bright=0.05, transient=0.12, width=0.6, bpm=120.0),
    Voicing(2, bass=0.09, bright=0.11, transient=0.20, width=0.9, bpm=140.0),
    Voicing(3, bass=0.26, bright=0.03, transient=0.06, width=0.3, bpm=90.0),
    Voicing(4, bass=0.14, bright=0.08, transient=0.16, width=0.7, bpm=128.0),
    Voicing(5, bass=0.20, bright=0.06, transient=0.10, width=0.5, bpm=100.0),
    Voicing(6, bass=0.07, bright=0.13, transient=0.24, width=1.0, bpm=150.0),
)


def render_voicing(voicing: Voicing, seconds: float = 24.0, sample_rate: int = SR) -> np.ndarray:
    rng = np.random.default_rng(voicing.seed)
    t = np.arange(int(sample_rate * seconds)) / sample_rate

    low = voicing.bass * np.sin(2 * np.pi * (55 + voicing.seed * 7) * t)
    low += 0.4 * voicing.bass * np.sin(2 * np.pi * (110 + voicing.seed * 9) * t)
    mid = 0.10 * np.sin(2 * np.pi * (330 + voicing.seed * 20) * t)
    mid += 0.07 * np.sin(2 * np.pi * (880 + voicing.seed * 31) * t)
    top = voicing.bright * rng.standard_normal(t.size)

    hits = np.zeros_like(t)
    period = int(sample_rate * 60.0 / voicing.bpm)
    envelope = int(sample_rate * 0.12)
    for start in range(0, t.size - envelope, period):
        hits[start : start + envelope] += (
            rng.standard_normal(envelope)
            * voicing.transient
            * np.exp(-np.arange(envelope) / (sample_rate * 0.02))
        )

    core = low + mid + hits
    side = voicing.width * (top - np.roll(top, 131))
    left, right = core + top + side, core + np.roll(top, 97) - side

    peak = max(float(np.abs(left).max()), float(np.abs(right).max()), 1e-12)
    return np.stack([left, right], axis=1) / peak * 0.7


def write_corpus(root: str | Path, seconds: float = 24.0, sample_rate: int = SR) -> list[Path]:
    """Write the synthetic corpus and return the files created."""
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for voicing in VOICINGS:
        path = destination / f"synth_{voicing.seed:02d}.wav"
        sf.write(path, render_voicing(voicing, seconds, sample_rate), sample_rate, subtype="PCM_24")
        written.append(path)
    return written
