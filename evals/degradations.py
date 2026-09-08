"""The degradation suite.

Every degradation is a **seeded, parametric chain** built from the same op
vocabulary the system uses to repair it. That buys three things:

1. **Reproducibility.** ``(track_id, kind, seed)`` fully determines the
   degradation, so any run can be recreated exactly from three values in a
   trace.
2. **Inspectability.** A degradation is a readable chain, not an opaque
   transform, so the report can show what was done alongside what was undone.
3. **A knowable ceiling.** Because the damage is expressible in the repair
   vocabulary, an exact inverse usually exists, which is what makes a recovery
   ratio of 0.7 interpretable rather than just a number.

The exception is worth stating plainly: ``over_compress`` is **not** exactly
invertible. Compression discards level information that no expander recovers,
so that class has a recovery ceiling below 1.0 for reasons that have nothing
to do with the controller. The evaluation measures that ceiling per degradation
rather than pretending it is 1.0.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from headroom.analysis.spectral import BAND_EDGES, N_BANDS
from headroom.dsp.chain import Chain
from headroom.dsp.ops import (
    EqBand,
    Op,
    op_compressor,
    op_eq,
    op_expander,
    op_gain,
    op_stereo_width,
)

DegradationKind = Literal[
    "spectral_tilt",
    "band_shift",
    "resonant_peak",
    "over_compress",
    "over_expand",
    "stereo_collapse",
    "stereo_overwide",
    "level_offset",
    "combo",
]

ALL_KINDS: Final[tuple[DegradationKind, ...]] = (
    "spectral_tilt",
    "band_shift",
    "resonant_peak",
    "over_compress",
    "over_expand",
    "stereo_collapse",
    "stereo_overwide",
    "level_offset",
    "combo",
)

#: Single-feature degradations. The heuristic controller should win here.
SINGLE_KINDS: Final[tuple[DegradationKind, ...]] = (
    "spectral_tilt",
    "band_shift",
    "resonant_peak",
    "stereo_collapse",
    "stereo_overwide",
    "level_offset",
)

#: Degradations that cannot be undone by an inverse of the same op type.
#: Reported separately, because their recovery ceiling is a physical fact.
LOSSY_KINDS: Final[frozenset[DegradationKind]] = frozenset({"over_compress"})

#: Pivot for the tilt shelf pair.
TILT_PIVOT_HZ: Final[float] = 1000.0


class Degradation(BaseModel):
    """A reproducible insult to a track."""

    model_config = ConfigDict(frozen=True)

    kind: DegradationKind
    seed: int
    params: dict[str, float | int | str]
    chain: Chain

    def describe(self) -> str:
        args = " ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.kind}(seed={self.seed} {args})"

    @property
    def is_lossy(self) -> bool:
        """True when no exact inverse exists in the op vocabulary."""
        if self.kind in LOSSY_KINDS:
            return True
        if self.kind == "combo":
            parts = str(self.params.get("parts", ""))
            return any(k in parts for k in LOSSY_KINDS)
        return False


def _band_center(band: int) -> float:
    """Geometric centre of a band, which is the right centre on a log axis."""
    return float(np.sqrt(BAND_EDGES[band] * BAND_EDGES[band + 1]))


def _spectral_tilt(rng: np.random.Generator) -> tuple[dict[str, float | int | str], list[Op]]:
    """Rotate the spectrum about 1 kHz using a shelf pair.

    The nominal slope is recorded, but the *achieved* tilt is whatever the
    shelves produce -- the evaluation measures it rather than trusting it.
    """
    slope = float(rng.uniform(1.0, 6.0)) * float(rng.choice([-1.0, 1.0]))
    # Each shelf carries half the rotation, in opposite directions.
    half = float(np.clip(abs(slope) * 1.5, 0.5, 12.0)) / 2.0
    low = half if slope < 0 else -half
    return (
        {"slope_db_oct": round(slope, 3), "shelf_gain_db": round(half, 3)},
        [
            op_eq(
                [
                    EqBand(shape="low_shelf", freq_hz=TILT_PIVOT_HZ, gain_db=low, q=0.707),
                    EqBand(shape="high_shelf", freq_hz=TILT_PIVOT_HZ, gain_db=-low, q=0.707),
                ]
            )
        ],
    )


def _band_shift(rng: np.random.Generator) -> tuple[dict[str, float | int | str], list[Op]]:
    band = int(rng.integers(1, N_BANDS - 1))
    gain = float(rng.uniform(2.0, 9.0)) * float(rng.choice([-1.0, 1.0]))
    return (
        {"band": band, "gain_db": round(gain, 3), "freq_hz": round(_band_center(band), 1)},
        [op_eq([EqBand(shape="peak", freq_hz=_band_center(band), gain_db=gain, q=1.0)])],
    )


def _resonant_peak(rng: np.random.Generator) -> tuple[dict[str, float | int | str], list[Op]]:
    freq = float(np.exp(rng.uniform(np.log(120.0), np.log(6000.0))))
    gain = float(rng.uniform(4.0, 12.0))
    q = float(rng.uniform(2.0, 8.0))
    return (
        {"freq_hz": round(freq, 1), "gain_db": round(gain, 3), "q": round(q, 3)},
        [op_eq([EqBand(shape="peak", freq_hz=freq, gain_db=gain, q=q)])],
    )


def _over_compress(
    rng: np.random.Generator, level_db: float
) -> tuple[dict[str, float | int | str], list[Op]]:
    # Below program level, not above it: a compressor acts on what exceeds
    # its threshold, so heavy compression needs the threshold well under the
    # integrated loudness. A threshold above program level touched only the
    # very peaks and moved no scored feature outside tolerance.
    ratio = float(rng.uniform(4.0, 20.0))
    threshold = float(np.clip(level_db + rng.uniform(-20.0, -6.0), -60.0, 0.0))
    return (
        {"ratio": round(ratio, 3), "threshold_db": round(threshold, 3)},
        [op_compressor(threshold_db=threshold, ratio=ratio, attack_ms=5.0, release_ms=80.0)],
    )


def _over_expand(
    rng: np.random.Generator, level_db: float
) -> tuple[dict[str, float | int | str], list[Op]]:
    ratio = float(rng.uniform(1.5, 4.0))
    threshold = float(np.clip(level_db + rng.uniform(8.0, 18.0), -60.0, 0.0))
    return (
        {"ratio": round(ratio, 3), "threshold_db": round(threshold, 3)},
        [op_expander(threshold_db=threshold, ratio=ratio, attack_ms=5.0, release_ms=80.0)],
    )


def _stereo_collapse(rng: np.random.Generator) -> tuple[dict[str, float | int | str], list[Op]]:
    # SPEC 10.2 specifies 0.0-0.4. Narrowed, because undoing a collapse to
    # 0.1 needs a width of 10.0 and the op bound is 2.0 -- that range would
    # measure the bound rather than the controller. 0.5 is the tightest
    # collapse whose exact repair is still in bounds.
    width = float(rng.uniform(0.5, 0.95))
    return {"width": round(width, 3)}, [op_stereo_width(width=width)]


def _stereo_overwide(rng: np.random.Generator) -> tuple[dict[str, float | int | str], list[Op]]:
    width = float(rng.uniform(1.5, 2.0))
    return {"width": round(width, 3)}, [op_stereo_width(width=width)]


def _level_offset(rng: np.random.Generator) -> tuple[dict[str, float | int | str], list[Op]]:
    gain = float(rng.uniform(4.0, 12.0)) * float(rng.choice([-1.0, 1.0]))
    return {"gain_db": round(gain, 3)}, [op_gain(gain)]


#: Degradations whose threshold is only meaningful relative to program level.
#: A compressor threshold in absolute dBFS says nothing without knowing how
#: loud the material is: -20 dBFS is heavy compression on a quiet mix and
#: inaudible on a loud one. Fixed thresholds produced literal no-op
#: degradations, which would have inflated every system's recovery ratio by
#: dividing through a near-zero initial distance.
LEVEL_RELATIVE_KINDS: Final[frozenset[str]] = frozenset({"over_compress", "over_expand"})

_BUILDERS: Final[dict[str, object]] = {
    "spectral_tilt": _spectral_tilt,
    "band_shift": _band_shift,
    "resonant_peak": _resonant_peak,
    "over_compress": _over_compress,
    "over_expand": _over_expand,
    "stereo_collapse": _stereo_collapse,
    "stereo_overwide": _stereo_overwide,
    "level_offset": _level_offset,
}


def _build(
    kind: str, rng: np.random.Generator, level_db: float
) -> tuple[dict[str, float | int | str], list[Op]]:
    builder = _BUILDERS[kind]
    result = builder(rng, level_db) if kind in LEVEL_RELATIVE_KINDS else builder(rng)  # type: ignore[operator]
    params, ops = result
    return dict(params), list(ops)


def make_degradation(
    kind: DegradationKind,
    seed: int,
    level_db: float = -14.0,
    combo_size: int = 2,
) -> Degradation:
    """Build a reproducible degradation.

    ``level_db`` is the source's integrated loudness, used to place dynamics
    thresholds relative to program level (see ``LEVEL_RELATIVE_KINDS``). It is
    part of the reproducibility key rather than a weakening of it: the track
    determines the level, so the same track, kind and seed always produce the
    same degradation.

    ``combo`` stacks 2-3 distinct single-feature degradations. It is the case
    that matters: single degradations are where a proportional controller
    should win, and coupled ones are where coordinating moves across a system
    whose parts interact should start to earn its cost.
    """
    rng = np.random.default_rng(seed)

    if kind == "combo":
        n = int(np.clip(combo_size, 2, 3))
        picked = list(rng.choice(np.asarray(SINGLE_KINDS, dtype=object), size=n, replace=False))
        params: dict[str, float | int | str] = {"parts": ",".join(str(p) for p in picked)}
        ops: list[Op] = []
        for i, part in enumerate(picked):
            sub_params, sub_ops = _build(
                str(part), np.random.default_rng(seed + 1000 * (i + 1)), level_db
            )
            for key, value in sub_params.items():
                params[f"{part}.{key}"] = value
            ops.extend(sub_ops)
        return Degradation(kind=kind, seed=seed, params=params, chain=Chain(ops=tuple(ops)))

    params, ops = _build(kind, rng, level_db)
    return Degradation(kind=kind, seed=seed, params=params, chain=Chain(ops=tuple(ops)))


def suite(
    kinds: Sequence[DegradationKind] = ALL_KINDS,
    seeds: Sequence[int] = (0, 1, 2),
    level_db: float = -14.0,
) -> tuple[Degradation, ...]:
    """The full cross product of kinds and seeds."""
    return tuple(make_degradation(k, s, level_db=level_db) for k in kinds for s in seeds)


#: A degradation weaker than this leaves nothing to recover, and a recovery
#: ratio computed against it is dominated by measurement noise because the
#: denominator is near zero. The runner skips these cells and records that it
#: did, rather than letting them silently flatter every system.
MIN_USEFUL_DISTANCE: Final[float] = 0.25
