"""Target profiles: what the audio is supposed to become.

Three ways to build one, all reduced to the same mechanism -- a dict of
constrained scored features:

*From a reference feature vector.* Every scored dimension is constrained. This
is both evaluation mode (target is the known original, so ground truth exists)
and reference mode (target derives from a reference track, which is how a
person would actually use the tool).

*From a loudness preset.* Only level and true peak are constrained. "Master
this for Spotify" is a real request and a partial target expresses it exactly.

*Masked.* Constrain some families and leave the rest free, so the evaluation
can ask whether hitting the specified targets damaged the unspecified ones.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from headroom.analysis.features import FeatureVector

from .distance import (
    FAMILY_WEIGHTS,
    SCORED,
    SPEC_BY_NAME,
    TRUE_PEAK_COMPLIANCE_TOL,
    Direction,
    to_scored,
)


class LoudnessPreset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    lufs_integrated: float
    true_peak_dbtp: float
    note: str = ""


#: Streaming and delivery targets. These are the platform-published
#: normalization levels; a real user's most common actual request.
PRESETS: Final[dict[str, LoudnessPreset]] = {
    "spotify": LoudnessPreset(
        name="spotify", lufs_integrated=-14.0, true_peak_dbtp=-1.0, note="Spotify normalization"
    ),
    "apple_music": LoudnessPreset(
        name="apple_music", lufs_integrated=-16.0, true_peak_dbtp=-1.0, note="Apple Sound Check"
    ),
    "youtube": LoudnessPreset(
        name="youtube", lufs_integrated=-14.0, true_peak_dbtp=-1.0, note="YouTube normalization"
    ),
    "broadcast_ebu": LoudnessPreset(
        name="broadcast_ebu", lufs_integrated=-23.0, true_peak_dbtp=-1.0, note="EBU R128"
    ),
    "club": LoudnessPreset(
        name="club", lufs_integrated=-9.0, true_peak_dbtp=-0.3, note="loud club/CD master"
    ),
}


class TargetProfile(BaseModel):
    """Satisfies :class:`~headroom.target.distance.TargetLike`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    provenance: str
    targets: dict[str, float]
    tolerance_overrides: dict[str, float] = Field(default_factory=dict)
    weight_overrides: dict[str, float] = Field(default_factory=dict)
    #: Absent means "both" -- match the value. A delivery preset sets
    #: ``true_peak_dbtp`` to "max" because it is a ceiling, not a setpoint.
    directions: dict[str, Direction] = Field(default_factory=dict)

    @classmethod
    def from_features(
        cls,
        fv: FeatureVector,
        label: str = "reference",
        provenance: str = "full feature vector",
        tolerance_overrides: Mapping[str, float] | None = None,
    ) -> TargetProfile:
        """Constrain every scored dimension to a reference vector's values."""
        return cls(
            label=label,
            provenance=provenance,
            targets=dict(to_scored(fv)),
            tolerance_overrides=dict(tolerance_overrides or {}),
        )

    @classmethod
    def from_preset(cls, name: str) -> TargetProfile:
        """Constrain level and true peak only, leaving tone and image free."""
        if name not in PRESETS:
            raise KeyError(f"unknown preset {name!r}; have {sorted(PRESETS)}")
        p = PRESETS[name]
        return cls(
            label=p.name,
            provenance=f"loudness preset: {p.note}",
            targets={
                "lufs_integrated": p.lufs_integrated,
                "true_peak_dbtp": p.true_peak_dbtp,
            },
            # Level is a setpoint you aim for; true peak is a ceiling you must
            # not exceed. Treating the ceiling as a setpoint would penalize a
            # quiet master as harshly as a clipping one.
            directions={"true_peak_dbtp": "max"},
            # A ceiling is about compliance, so it is held far tighter than
            # the matching tolerance used when copying a reference.
            tolerance_overrides={"true_peak_dbtp": TRUE_PEAK_COMPLIANCE_TOL},
        )

    @classmethod
    def masked(
        cls, fv: FeatureVector, families: Mapping[str, bool] | set[str], label: str = "masked"
    ) -> TargetProfile:
        """Constrain only the named families. The unconstrained ones are still
        measured, so the report can show whether they were damaged."""
        keep = (
            set(families)
            if not isinstance(families, Mapping)
            else {k for k, v in families.items() if v}
        )
        unknown = keep - set(FAMILY_WEIGHTS)
        if unknown:
            raise KeyError(f"unknown families {sorted(unknown)}; have {sorted(FAMILY_WEIGHTS)}")
        scored = to_scored(fv)
        return cls(
            label=label,
            provenance=f"families constrained: {sorted(keep)}",
            targets={s.name: scored[s.name] for s in SCORED if s.family in keep},
        )

    def with_loudness(self, preset: str) -> TargetProfile:
        """Override level and peak with a delivery preset, keeping tone and
        image from this profile. "Match that reference, but at Spotify level."
        """
        p = PRESETS[preset]
        merged = dict(self.targets)
        merged["lufs_integrated"] = p.lufs_integrated
        merged["true_peak_dbtp"] = p.true_peak_dbtp
        return self.model_copy(
            update={
                "targets": merged,
                "directions": {**self.directions, "true_peak_dbtp": "max"},
                "tolerance_overrides": {
                    **self.tolerance_overrides,
                    "true_peak_dbtp": TRUE_PEAK_COMPLIANCE_TOL,
                },
                "label": f"{self.label}+{p.name}",
                "provenance": f"{self.provenance}; level from preset {p.name}",
            }
        )

    def constrained_families(self) -> list[str]:
        fams = {SPEC_BY_NAME[n].family for n in self.targets if n in SPEC_BY_NAME}
        return sorted(fams)

    def describe(self) -> str:
        return (
            f"target '{self.label}' ({self.provenance}): "
            f"{len(self.targets)} features across {self.constrained_families()}"
        )
