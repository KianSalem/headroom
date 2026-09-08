"""The feature vector: the objective function everything else is graded through.

This module holds the *raw, interpretable* measurement -- the numbers an
engineer would read off a meter. The separate, transformed representation used
by the distance metric lives in :mod:`headroom.target.distance`, because the
transforms it applies (log ratios, Fisher z, dropping collinear features) serve
the metric and would make these values harder for a human or an agent to read.

Some fields are deliberately reported but never scored:

``plr``
    Exactly ``true_peak_dbtp - lufs_integrated``. Scoring it would count
    loudness error three times.
``sample_peak_dbfs``
    Near-collinear with ``true_peak_dbtp``.
``lufs_short_p10/p50/p90``
    ``lufs_integrated`` plus ``lra`` already carry level and spread.
``spectral_centroid``, ``spectral_rolloff_85``, ``spectral_tilt``
    All are summaries of the same spectrum that ``band_energy`` describes in
    nine dimensions. A tilt or a brightness change shows up in the band
    energies; scoring the summaries too would double-count spectral shape.
``mid_side_ratio_db``
    A scalar summary of what ``width_per_band_db`` describes per band.
``onset_rate``
    Not a mastering target, and unreliable on sustained material -- librosa's
    onset detector fires on spectral-flux noise in a steady tone.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from headroom.audio import AudioBuffer, AudioError

from .dynamics import analyze_dynamics
from .loudness import analyze_loudness
from .spectral import N_BANDS, analyze_spectral
from .stereo import analyze_stereo
from .transient import analyze_transient


class FeatureVector(BaseModel):
    """Measured properties of one piece of audio. Frozen; construct via :func:`analyze`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- loudness (LUFS / LU / dBTP / dBFS) ---
    lufs_integrated: float
    lufs_short_p10: float
    lufs_short_p50: float
    lufs_short_p90: float
    lra: float
    true_peak_dbtp: float
    sample_peak_dbfs: float

    # --- dynamics (dB) ---
    crest_factor_db: float
    plr: float
    crest_short_p50: float

    # --- spectral ---
    band_energy: tuple[float, ...] = Field(description="9 log-spaced bands, sums to 1.0")
    spectral_centroid: float
    spectral_flatness: float
    spectral_rolloff_85: float
    spectral_tilt: float

    # --- stereo ---
    mid_side_ratio_db: float
    correlation: float
    width_per_band_db: tuple[float, ...] = Field(description="9 bands, side/mid in dB")
    mono_compat_db: float

    # --- transient ---
    onset_rate: float
    attack_time_p50: float
    percussive_ratio: float
    n_onsets: int

    # --- provenance ---
    sample_rate: int
    duration_s: float
    source_hash: str

    # --- degeneracy flags: a reading that is structural, not measured ---
    was_mono: bool = False
    short_term_degenerate: bool = False
    stereo_degenerate: bool = False
    attack_degenerate: bool = False

    @field_validator("band_energy", "width_per_band_db")
    @classmethod
    def _nine_bands(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if len(v) != N_BANDS:
            raise ValueError(f"expected {N_BANDS} bands, got {len(v)}")
        return v

    def summary(self) -> str:
        """Compact human-readable rendering, as an engineer would read a meter."""
        bands = " ".join(f"{v:.3f}" for v in self.band_energy)
        width = " ".join(f"{v:+.1f}" for v in self.width_per_band_db)
        return (
            f"LUFS {self.lufs_integrated:+.2f}  TP {self.true_peak_dbtp:+.2f} dBTP  "
            f"LRA {self.lra:.2f} LU  PLR {self.plr:.2f} dB\n"
            f"crest {self.crest_factor_db:.2f} dB (short p50 {self.crest_short_p50:.2f})\n"
            f"centroid {self.spectral_centroid:.0f} Hz  flatness {self.spectral_flatness:.3f}  "
            f"rolloff85 {self.spectral_rolloff_85:.0f} Hz  tilt {self.spectral_tilt:+.2f} dB/oct\n"
            f"bands   [{bands}]\n"
            f"width   [{width}] dB\n"
            f"corr {self.correlation:+.3f}  M/S {self.mid_side_ratio_db:+.1f} dB  "
            f"mono {self.mono_compat_db:+.2f} dB\n"
            f"onsets {self.onset_rate:.2f}/s  attack {self.attack_time_p50:.2f} ms  "
            f"percussive {self.percussive_ratio:.3f}"
        )


#: Fields measured but excluded from the distance metric, with the reason.
#: Kept as data so the report can render the choice rather than assert it.
REPORTED_NOT_SCORED: Final[dict[str, str]] = {
    "plr": "exactly true_peak_dbtp - lufs_integrated",
    "sample_peak_dbfs": "near-collinear with true_peak_dbtp",
    "lufs_short_p10": "level and spread already carried by lufs_integrated + lra",
    "lufs_short_p50": "level and spread already carried by lufs_integrated + lra",
    "lufs_short_p90": "level and spread already carried by lufs_integrated + lra",
    "spectral_centroid": "summary of the spectrum band_energy already describes",
    "spectral_rolloff_85": "summary of the spectrum band_energy already describes",
    "spectral_tilt": "summary of the spectrum band_energy already describes",
    "mid_side_ratio_db": "scalar summary of width_per_band_db",
    "onset_rate": "not a mastering target; unreliable on sustained material",
}


#: Measurement cache, keyed on exact sample content.
#:
#: HPSS inside the transient analyser is roughly 80% of the cost of a full
#: measurement, while hashing the samples is ~145x cheaper than measuring
#: them. The control loop re-measures repeated audio constantly -- a system
#: that reverts an edit, or a numerical optimizer sweeping a parameter back
#: over a value it already tried -- so the hit rate is high and the miss
#: overhead is under 2%.
#:
#: Bounded, because an evaluation run measures thousands of distinct renders
#: and an unbounded cache would hold every one of them in memory.
_CACHE: Final[OrderedDict[tuple[str, bool], FeatureVector]] = OrderedDict()
_CACHE_MAX: Final[int] = 256
_CACHE_HITS: Final[list[int]] = [0, 0]  # hits, misses


def cache_stats() -> dict[str, int]:
    return {"entries": len(_CACHE), "hits": _CACHE_HITS[0], "misses": _CACHE_HITS[1]}


def clear_cache() -> None:
    _CACHE.clear()
    _CACHE_HITS[0] = _CACHE_HITS[1] = 0


def analyze(buf: AudioBuffer, use_cache: bool = True) -> FeatureVector:
    """Compute the full feature vector. Deterministic; no LLM involved."""
    if buf.n_frames == 0:
        # An empty file loads without complaint and fails deep inside the
        # spectral path with an IndexError; say what is actually wrong.
        raise AudioError("cannot analyze an empty buffer (0 frames)")
    if use_cache:
        # was_mono is part of the key: two buffers can hold identical samples
        # while disagreeing about whether the stereo features are meaningful.
        key = (buf.content_hash(), buf.was_mono)
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE_HITS[0] += 1
            _CACHE.move_to_end(key)
            return cached
        _CACHE_HITS[1] += 1
        computed = _measure(buf)
        _CACHE[key] = computed
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
        return computed
    return _measure(buf)


def _measure(buf: AudioBuffer) -> FeatureVector:
    loud = analyze_loudness(buf)
    dyn = analyze_dynamics(buf, loud.true_peak_dbtp, loud.lufs_integrated)
    spec = analyze_spectral(buf)
    st = analyze_stereo(buf)
    tr = analyze_transient(buf)

    return FeatureVector(
        lufs_integrated=loud.lufs_integrated,
        lufs_short_p10=loud.lufs_short_p10,
        lufs_short_p50=loud.lufs_short_p50,
        lufs_short_p90=loud.lufs_short_p90,
        lra=loud.lra,
        true_peak_dbtp=loud.true_peak_dbtp,
        sample_peak_dbfs=loud.sample_peak_dbfs,
        crest_factor_db=dyn.crest_factor_db,
        plr=dyn.plr,
        crest_short_p50=dyn.crest_short_p50,
        band_energy=spec.band_energy,
        spectral_centroid=spec.spectral_centroid,
        spectral_flatness=spec.spectral_flatness,
        spectral_rolloff_85=spec.spectral_rolloff_85,
        spectral_tilt=spec.spectral_tilt,
        mid_side_ratio_db=st.mid_side_ratio_db,
        correlation=st.correlation,
        width_per_band_db=st.width_per_band_db,
        mono_compat_db=st.mono_compat_db,
        onset_rate=tr.onset_rate,
        attack_time_p50=tr.attack_time_p50,
        percussive_ratio=tr.percussive_ratio,
        n_onsets=tr.n_onsets,
        sample_rate=buf.sample_rate,
        duration_s=buf.duration_s,
        source_hash=buf.content_hash(),
        was_mono=buf.was_mono,
        short_term_degenerate=loud.short_term_degenerate,
        stereo_degenerate=st.is_degenerate,
        attack_degenerate=tr.attack_degenerate,
    )
