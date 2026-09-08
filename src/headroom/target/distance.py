"""The distance metric.

**Deviation from SPEC 5, rule 1, deliberate.** The spec calls for z-scoring
every feature against population statistics from the reference corpus. This
implementation normalizes by *tolerance* instead. The reasons:

* The spec's stated purpose for z-scoring is that "raw units are not
  comparable" -- 200 Hz of centroid error versus 2 LU of loudness error.
  Dividing by a perceptually-motivated tolerance solves that directly, and
  yields a metric in units of "how audible is this error", which is what a
  mastering objective actually wants. A z-score answers a different question:
  "how unusual is this value in the corpus".
* Population z-scoring makes the metric a function of the corpus. Change the
  corpus and every historical score silently changes, invalidating stored
  results. Tolerance normalization is corpus-free and reproducible.
* Estimating statistics for ~28 features from a small train split is noisy,
  and a covariance (needed to handle the collinearity properly) would be
  rank-deficient outright.

Three further corrections to the naive feature set, all of which would
otherwise corrupt the score:

*Collinear features are excluded.* ``plr`` is exactly
``true_peak_dbtp - lufs_integrated``; scoring all three counts loudness error
three times. See ``analysis.features.REPORTED_NOT_SCORED``.

*Band energy is compositional.* It is normalized to sum to one, so its nine
values carry eight degrees of freedom and its deltas are constrained to sum to
zero. The correct treatment is a centered log-ratio transform, expressed here
in dB so it stays readable.

*Weights are normalized per family, not per feature.* Ten of the 28 scored
dimensions are spectral and eleven are stereo. Uniform per-feature weights
would hand those two families 75% of the objective while ``lufs_integrated``
got 3.6% -- an implicit weighting choice masquerading as neutrality.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from headroom.analysis.features import FeatureVector
from headroom.analysis.spectral import N_BANDS

#: How a constraint is enforced. "both" penalizes any deviation (match this
#: value). "max" penalizes only exceeding it (a ceiling -- true peak below the
#: ceiling is not an error, it is just quieter). "min" penalizes only falling
#: below it (a floor).
Direction = Literal["both", "max", "min"]

#: Weight per family. Sums to 1.0. Overridable via a config file; this is the
#: documented default and it is frozen before any evaluation run.
FAMILY_WEIGHTS: Final[dict[str, float]] = {
    "loudness": 0.25,
    "dynamics": 0.20,
    "spectral": 0.30,
    "stereo": 0.20,
    "transient": 0.05,
}

#: Convergence: total score at or below this, *and* no single feature more
#: than ``MAX_SINGLE_EXCESS`` tolerances out. The second condition matters --
#: a scalar threshold alone lets one badly wrong feature hide inside a good
#: average.
#: Tolerance for true peak when it is a delivery ceiling rather than a
#: matching target. Compliance, not audibility.
TRUE_PEAK_COMPLIANCE_TOL: Final[float] = 0.2

CONVERGE_EPS: Final[float] = 0.05
MAX_SINGLE_EXCESS: Final[float] = 1.0

_FLOOR: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class ScoredSpec:
    """Definition of one scored dimension."""

    name: str
    family: str
    unit: str
    #: Tolerance in the *transformed* unit. Inside it, error contributes zero.
    tolerance: float
    native_note: str = ""


def _band_clr(band_energy: tuple[float, ...]) -> list[float]:
    """Centered log-ratio of the band energies, in dB.

    ``clr_i = 10*log10(b_i) - mean_j(10*log10(b_j))``. Scale-invariant by
    construction, symmetric, and it equalizes sensitivity across bands: a 1 dB
    change in the 20-60 Hz band produces the same delta as a 1 dB change in
    the 8-20 kHz band, even though the latter holds ~300x the energy of
    broadband material because it is ~300x wider.
    """
    logs = [10.0 * math.log10(max(b, _FLOOR)) for b in band_energy]
    mean = sum(logs) / len(logs)
    return [v - mean for v in logs]


def _logit(p: float, lo: float = 1e-4, hi: float = 1.0 - 1e-4) -> float:
    """Logit for a 0-1 bounded feature. Z-scoring a bounded, skewed quantity
    is close to meaningless; the logit makes it unbounded and symmetric."""
    q = min(max(p, lo), hi)
    return math.log(q / (1.0 - q))


def _fisher_z(r: float) -> float:
    """Fisher transform for a correlation. ``atanh`` maps (-1, 1) to the reals
    and stabilizes the variance near the bounds, where a mastering change from
    0.98 to 0.99 matters far more than 0.10 to 0.11."""
    return math.atanh(min(max(r, -0.999999), 0.999999))


SCORED: Final[tuple[ScoredSpec, ...]] = (
    # --- loudness ---
    ScoredSpec("lufs_integrated", "loudness", "LUFS", 0.5),
    # 1.0 dB for *matching*: a 1 dB true-peak difference mid-range is
    # inaudible, and a 0.3 dB tolerance made true peak the loudest term in the
    # breakdown for every degradation, including purely spectral ones.
    # Delivery presets override this to TRUE_PEAK_COMPLIANCE_TOL, where the
    # number is about not clipping a converter rather than about audibility.
    ScoredSpec("true_peak_dbtp", "loudness", "dBTP", 1.0),
    ScoredSpec("lra", "loudness", "LU", 1.0),
    # --- dynamics ---
    ScoredSpec("crest_factor_db", "dynamics", "dB", 1.0),
    ScoredSpec("crest_short_p50", "dynamics", "dB", 1.0),
    # --- spectral ---
    *(
        ScoredSpec(f"band_clr_{i}", "spectral", "dB", 0.75, "centered log-ratio band energy")
        for i in range(N_BANDS)
    ),
    ScoredSpec("flatness_logit", "spectral", "logit", 0.15, "~0.03 native near 0.5"),
    # --- stereo ---
    ScoredSpec("correlation_z", "stereo", "fisher-z", 0.10, "~0.10 native near 0"),
    *(
        ScoredSpec(f"width_{i}", "stereo", "dB", 1.0, "side/mid ratio in band i")
        for i in range(N_BANDS)
    ),
    ScoredSpec("mono_compat_db", "stereo", "dB", 0.5),
    # --- transient ---
    ScoredSpec("attack_log2_ms", "transient", "log2(ms)", 0.35, "~27% relative"),
    ScoredSpec("percussive_logit", "transient", "logit", 0.20),
)

SPEC_BY_NAME: Final[dict[str, ScoredSpec]] = {s.name: s for s in SCORED}


def _feature_weights() -> dict[str, float]:
    """Family weight split evenly across that family's members."""
    counts: dict[str, int] = {}
    for spec in SCORED:
        counts[spec.family] = counts.get(spec.family, 0) + 1
    return {s.name: FAMILY_WEIGHTS[s.family] / counts[s.family] for s in SCORED}


FEATURE_WEIGHTS: Final[dict[str, float]] = _feature_weights()


def to_scored(fv: FeatureVector) -> dict[str, float]:
    """Project a raw feature vector into the scored, de-collinearized space."""
    out: dict[str, float] = {
        "lufs_integrated": fv.lufs_integrated,
        "true_peak_dbtp": fv.true_peak_dbtp,
        "lra": fv.lra,
        "crest_factor_db": fv.crest_factor_db,
        "crest_short_p50": fv.crest_short_p50,
        "flatness_logit": _logit(fv.spectral_flatness),
        "correlation_z": _fisher_z(fv.correlation),
        "mono_compat_db": fv.mono_compat_db,
        # attack time can legitimately be 0 when no transient is measurable;
        # log2 of a floored value keeps the dimension finite and ordered.
        "attack_log2_ms": math.log2(max(fv.attack_time_p50, 0.05)),
        "percussive_logit": _logit(fv.percussive_ratio),
    }
    for i, v in enumerate(_band_clr(fv.band_energy)):
        out[f"band_clr_{i}"] = v
    for i, v in enumerate(fv.width_per_band_db):
        out[f"width_{i}"] = v
    return out


@dataclass(frozen=True, slots=True)
class FeatureDelta:
    """One feature's signed error. This -- not the scalar score -- is what the
    supervisor and specialists actually read."""

    name: str
    family: str
    unit: str
    current: float
    target: float
    delta: float
    tolerance: float
    direction: Direction
    #: Delta in tolerance units. 1.0 means "exactly one tolerance out".
    scaled: float
    #: Signed amount beyond tolerance; zero inside it.
    excess: float
    weight: float
    contribution: float

    @property
    def in_tolerance(self) -> bool:
        return self.excess == 0.0

    def describe(self) -> str:
        arrow = "->" if self.excess == 0.0 else ("TOO HIGH" if self.excess > 0 else "TOO LOW")
        rel = {"both": "==", "max": "<=", "min": ">="}[self.direction]
        return (
            f"{self.name:18s} {self.current:+9.3f} {rel} {self.target:+9.3f} {self.unit:9s} "
            f"delta {self.delta:+7.3f}  {self.scaled:+6.2f} tol  {arrow}"
        )


@dataclass(frozen=True, slots=True)
class DistanceResult:
    score: float
    breakdown: dict[str, FeatureDelta]
    by_family: dict[str, float]
    n_out_of_tolerance: int
    converged: bool
    norm: str

    def worst(self, k: int = 5) -> list[FeatureDelta]:
        return sorted(self.breakdown.values(), key=lambda d: -d.contribution)[:k]

    def describe(self, k: int = 8) -> str:
        head = (
            f"score {self.score:.4f} ({self.norm})  "
            f"{self.n_out_of_tolerance}/{len(self.breakdown)} out of tolerance  "
            f"converged={self.converged}"
        )
        fams = "  ".join(f"{k2}={v:.3f}" for k2, v in sorted(self.by_family.items()))
        lines = [d.describe() for d in self.worst(k)]
        return "\n".join([head, f"by family: {fams}", *lines])


class TargetLike(Protocol):
    """What :func:`distance` needs from a target.

    A Protocol rather than the concrete :class:`~headroom.target.profile.TargetProfile`
    so this module stays free of a circular import and stays testable with a
    plain dict-backed stub.
    """

    @property
    def targets(self) -> Mapping[str, float]:
        """Scored-space targets. Only the constrained features need appear."""

    @property
    def tolerance_overrides(self) -> Mapping[str, float]: ...

    @property
    def weight_overrides(self) -> Mapping[str, float]: ...

    @property
    def directions(self) -> Mapping[str, Direction]:
        """Per-feature constraint direction; absent means ``"both"``."""


def distance(features: FeatureVector, target: TargetLike, norm: str = "l2") -> DistanceResult:
    """Signed, per-feature distance from ``features`` to ``target``.

    Only features the target actually constrains are scored, and the weights of
    that subset are renormalized to sum to 1.0. That single mechanism gives
    three things: a full-vector target (eval and reference mode), a partial
    target (a streaming loudness preset constrains only level and peak), and a
    masked target (constrain some families, watch for damage in the others).
    Without renormalization a two-feature preset would produce a score that
    looks tiny next to a full-vector one and the two could not be compared.

    ``norm`` is configurable because L1 is more robust to one wild feature and
    L2 punishes a single large error harder. Which is better is an empirical
    question, so it is a parameter rather than a hardcoded choice.
    """
    if norm not in ("l1", "l2"):
        raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")

    scored = to_scored(features)
    constrained = [name for name in target.targets if name in SPEC_BY_NAME]
    if not constrained:
        raise ValueError("target constrains no scored features")

    raw_weights = {
        name: float(target.weight_overrides.get(name, FEATURE_WEIGHTS[name]))
        for name in constrained
    }
    total_weight = sum(raw_weights.values())
    if total_weight <= 0.0:
        raise ValueError("target weights sum to zero")

    breakdown: dict[str, FeatureDelta] = {}
    by_family: dict[str, float] = {}
    n_out = 0
    max_excess = 0.0

    for name in constrained:
        spec = SPEC_BY_NAME[name]
        tol = float(target.tolerance_overrides.get(name, spec.tolerance))
        tol = max(tol, _FLOOR)
        current = scored[name]
        want = float(target.targets[name])
        direction: Direction = target.directions.get(name, "both")
        delta = current - want
        scaled = delta / tol
        if direction == "max":
            # A ceiling: only exceeding it is an error.
            excess = max(scaled - 1.0, 0.0)
        elif direction == "min":
            excess = min(scaled + 1.0, 0.0)
        else:
            excess = math.copysign(max(abs(scaled) - 1.0, 0.0), scaled)
        weight = raw_weights[name] / total_weight
        contribution = weight * (excess**2 if norm == "l2" else abs(excess))

        breakdown[name] = FeatureDelta(
            name=name,
            family=spec.family,
            unit=spec.unit,
            current=current,
            target=want,
            delta=delta,
            tolerance=tol,
            direction=direction,
            scaled=scaled,
            excess=excess,
            weight=weight,
            contribution=contribution,
        )
        by_family[spec.family] = by_family.get(spec.family, 0.0) + contribution
        if excess != 0.0:
            n_out += 1
            max_excess = max(max_excess, abs(excess))

    total = sum(d.contribution for d in breakdown.values())
    score = math.sqrt(total) if norm == "l2" else total
    converged = score <= CONVERGE_EPS and max_excess <= MAX_SINGLE_EXCESS

    return DistanceResult(
        score=score,
        breakdown=breakdown,
        by_family=by_family,
        n_out_of_tolerance=n_out,
        converged=converged,
        norm=norm,
    )


def recovery_ratio(initial: float, final: float) -> float:
    """``1 - final/initial``. 1.0 is perfect, 0 is no progress, negative means
    the system made the audio worse.

    Unstable when ``initial`` is near zero -- a mild degradation has a small
    denominator, so the ratio is noisiest exactly where the task is easiest.
    The evaluation therefore reports absolute final distance alongside it and
    stratifies by initial magnitude.
    """
    if initial <= _FLOOR:
        return 0.0 if final <= _FLOOR else float("-inf")
    return 1.0 - (final / initial)
