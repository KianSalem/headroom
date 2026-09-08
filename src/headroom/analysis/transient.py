"""Transient measurement.

Attack time is the most implementation-sensitive feature in the set -- two
reasonable implementations can differ by 2x -- so every constant is named and
fixed here rather than left to a library default. The definition is: build an
RMS envelope, and for each detected onset measure the time from 10% to 90% of
the rise from the pre-onset baseline to the post-onset peak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import librosa
import numpy as np

from headroom.audio import AudioBuffer, Samples

#: The RMS window is the hard limit on measurable attack time: a 10 ms window
#: smooths a 1 ms attack into a 10 ms rise no matter how fine the hop. 2 ms
#: window / 0.25 ms hop resolves the 1-5 ms attacks that drum transients
#: actually occupy.
ENV_WIN_S: Final[float] = 0.002
ENV_HOP_S: Final[float] = 0.00025
ONSET_HOP_S: Final[float] = 0.010

#: Baseline is the envelope minimum in this window before the onset.
ATTACK_PRE_S: Final[float] = 0.010
#: Peak is the envelope maximum in this window after the onset.
ATTACK_POST_S: Final[float] = 0.060
#: Onsets closer together than this are dropped: the next transient would
#: contaminate the rise being measured.
MIN_ONSET_GAP_S: Final[float] = 0.060
#: A rise smaller than this is not a transient, it is envelope noise.
MIN_RISE_RATIO: Final[float] = 1.05

#: librosa HPSS defaults, pinned explicitly. Both materially change
#: percussive_ratio, so leaving them implicit would make the feature depend on
#: the installed librosa version.
HPSS_MARGIN: Final[float] = 1.0
HPSS_KERNEL_SIZE: Final[int] = 31


@dataclass(frozen=True, slots=True)
class TransientFeatures:
    onset_rate: float
    attack_time_p50: float
    percussive_ratio: float
    n_onsets: int
    #: True when no onset yielded a measurable rise, so attack_time_p50 is a
    #: fallback rather than a measurement.
    attack_degenerate: bool


def rms_envelope(x: Samples, sample_rate: int) -> tuple[Samples, Samples]:
    """RMS envelope and its frame times, both from window sizes in seconds."""
    win = max(round(ENV_WIN_S * sample_rate), 2)
    hop = max(round(ENV_HOP_S * sample_rate), 1)
    env = librosa.feature.rms(y=x, frame_length=win, hop_length=hop, center=True)[0]
    env = np.asarray(env, dtype=np.float64)
    times = np.arange(env.size, dtype=np.float64) * (hop / sample_rate)
    return env, times


def onset_times(x: Samples, sample_rate: int) -> Samples:
    hop = max(round(ONSET_HOP_S * sample_rate), 1)
    times = librosa.onset.onset_detect(
        y=x, sr=sample_rate, hop_length=hop, units="time", backtrack=False
    )
    return np.asarray(times, dtype=np.float64)


def _attack_ms(env: Samples, times: Samples, onset: float) -> float | None:
    """10-90% rise time in ms for one onset, or None if not measurable."""
    hop = float(times[1] - times[0]) if times.size > 1 else 0.0
    if hop <= 0.0:
        return None
    i_on = int(np.searchsorted(times, onset))
    i_pre = max(int(np.searchsorted(times, onset - ATTACK_PRE_S)), 0)
    i_post = min(int(np.searchsorted(times, onset + ATTACK_POST_S)), env.size - 1)
    if not (i_pre < i_on <= i_post):
        return None

    baseline = float(env[i_pre : i_on + 1].min())
    peak = float(env[i_on : i_post + 1].max())
    if peak <= baseline * MIN_RISE_RATIO or peak <= 0.0:
        return None

    span = peak - baseline
    thr10, thr90 = baseline + 0.1 * span, baseline + 0.9 * span
    seg = env[i_pre : i_post + 1]
    above10 = np.flatnonzero(seg >= thr10)
    if above10.size == 0:
        return None
    i10 = int(above10[0])
    above90 = np.flatnonzero(seg[i10:] >= thr90)
    if above90.size == 0:
        return None
    i90 = i10 + int(above90[0])
    return float((i90 - i10) * hop * 1000.0)


def analyze_transient(buf: AudioBuffer) -> TransientFeatures:
    mid = buf.mid()
    if buf.n_frames < int(0.1 * buf.sample_rate) or float(np.max(np.abs(mid))) <= 0.0:
        return TransientFeatures(0.0, 0.0, 0.0, 0, attack_degenerate=True)

    onsets = onset_times(mid, buf.sample_rate)
    env, times = rms_envelope(mid, buf.sample_rate)

    attacks: list[float] = []
    for i, onset in enumerate(onsets):
        nxt = onsets[i + 1] if i + 1 < onsets.size else np.inf
        if float(nxt - onset) < MIN_ONSET_GAP_S:
            continue
        a = _attack_ms(env, times, float(onset))
        if a is not None:
            attacks.append(a)

    harmonic, percussive = librosa.effects.hpss(
        mid, margin=HPSS_MARGIN, kernel_size=HPSS_KERNEL_SIZE
    )
    e_perc = float(np.sum(np.square(percussive)))
    e_total = e_perc + float(np.sum(np.square(harmonic)))
    perc_ratio = float(np.clip(e_perc / e_total, 0.0, 1.0)) if e_total > 0.0 else 0.0

    return TransientFeatures(
        onset_rate=float(onsets.size) / buf.duration_s,
        attack_time_p50=float(np.median(attacks)) if attacks else 0.0,
        percussive_ratio=perc_ratio,
        n_onsets=int(onsets.size),
        attack_degenerate=not attacks,
    )
