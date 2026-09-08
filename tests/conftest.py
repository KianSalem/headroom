"""Synthetic signals with analytically known properties.

Phase 1 of the spec requires every feature to be validated against a signal
whose true value is known analytically rather than against another
implementation's output. These generators are those signals.
"""

from __future__ import annotations

import numpy as np
import pytest

from headroom.audio import AudioBuffer

SR = 48000


def stereo(left: np.ndarray, right: np.ndarray | None = None, sr: int = SR) -> AudioBuffer:
    r = left if right is None else right
    return AudioBuffer(np.stack([left, r], axis=1).astype(np.float64), sr)


def sine(freq: float, dbfs: float, seconds: float = 8.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return float(10.0 ** (dbfs / 20.0)) * np.sin(2.0 * np.pi * freq * t)


def white(seconds: float = 8.0, amp: float = 0.05, seed: int = 0, sr: int = SR) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(int(sr * seconds)) * amp


def ramp_clicks(
    attack_s: float, bpm: float = 120.0, seconds: float = 8.0, seed: int = 11, sr: int = SR
) -> np.ndarray:
    """Percussive hits with a linear attack of known length. The 10-90% rise
    time of a linear ramp is exactly 0.8x its length."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    y = np.zeros(n)
    env_len = int(sr * 0.15)
    at = max(int(sr * attack_s), 1)
    env = np.concatenate(
        [np.linspace(0.0, 1.0, at), np.exp(-np.arange(env_len - at) / (sr * 0.03))]
    )[:env_len]
    for start in range(0, n - env_len, int(sr * 60.0 / bpm)):
        y[start : start + env_len] += env * rng.standard_normal(env_len) * 0.3
    return y


@pytest.fixture
def sine_1k_minus20() -> AudioBuffer:
    return stereo(sine(1000.0, -20.0))


@pytest.fixture
def white_noise() -> AudioBuffer:
    return stereo(white())


@pytest.fixture
def uncorrelated() -> AudioBuffer:
    return stereo(white(seed=1), white(seed=2))


@pytest.fixture
def silence() -> AudioBuffer:
    return AudioBuffer(np.zeros((SR * 5, 2)), SR)
