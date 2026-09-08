"""Translating a natural-language brief into a measurable target.

This is the one place in the system where a model is doing something
arithmetic cannot do at all. Everywhere else the LLM competes with a
proportional controller and, on numeric targets, loses or ties. Here there is
no controller to compete with: *"more space, but keep the low end tight"* is
not a number, and no amount of signal processing turns it into one.

What makes it a measurable claim rather than a demo is that the model's output
is a **target**, not a decision. It emits signed offsets, in tolerance units,
against dimensions the deterministic metric already defines. The existing loop
then runs completely unchanged, and the existing metric grades the result. The
model never touches audio, never scores anything, and cannot widen its own
mandate: an offset naming a dimension that does not exist is rejected, and so
is one larger than the bound.

That also makes the translation itself gradable without an LLM judge. A brief
has a checkable signature: *brighter* means the top bands go up relative to
where they started, *tighter low end* means the bottom bands narrow. So the
evaluation asks three separate questions, all deterministic --

1. did the target move the dimensions the brief names, in the named direction;
2. did the render then move those dimensions toward that target;
3. did the families the brief said to hold stay inside tolerance

-- and the second and third are where a plausible-sounding translation that
cannot actually be executed gets caught.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from headroom.analysis.features import FeatureVector
from headroom.dsp.ops import BoundViolationError
from headroom.target.distance import FAMILY_WEIGHTS, SPEC_BY_NAME, to_scored
from headroom.target.profile import TargetProfile

from .briefing import DIMENSIONS
from .roles import Role
from .tools import band_label

#: The real band edges, spelled out in the prompt.
#:
#: This replaced a lows/mids/highs rule of thumb that was simply wrong: it said
#: mids were bands 3 to 5, so a request to take harshness out of "the upper
#: mids" was translated as bands 4 and 5 when 2-4 kHz is band 6. The model had
#: the frequency range right and the index wrong, because the prompt gave it
#: the wrong index. Naming the edges costs sixty tokens.
_BAND_LABELS: Final[tuple[str, ...]] = tuple(band_label(i) for i in range(9))

#: Largest offset a brief may request, in tolerance units. Eight tolerances is
#: already a dramatic change -- six dB of band energy, four dB of width -- and
#: an unbounded offset would let one adjective ask for something no chain can
#: reach, which shows up as an unreachable target rather than as a bad brief.
MAX_OFFSET_TOL: Final[float] = 8.0
#: Smallest offset worth asking for. Below one tolerance the request is inside
#: the metric's own indifference band, so it would be satisfied by doing
#: nothing at all.
MIN_OFFSET_TOL: Final[float] = 1.0

#: Adjustments one brief may carry. A brief that names more than this is
#: either very long or the model is spraying.
MAX_ADJUSTMENTS: Final[int] = 8


class Adjustment(BaseModel):
    """One dimension the brief asks to move, and by how much."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: str
    #: Signed, in tolerance units. Positive means "more of what this dimension
    #: measures", which is why the glossary matters: more ``correlation_z``
    #: means a *narrower* image, not a wider one.
    offset_tol: float = Field(ge=-MAX_OFFSET_TOL, le=MAX_OFFSET_TOL)
    reason: str = ""


class BriefTarget(BaseModel):
    """A brief, as a partial constraint over the scored space."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    brief: str
    adjustments: tuple[Adjustment, ...] = ()
    #: Names to pin at the source's current values -- either a family, or a
    #: single dimension. This is what "but keep the low end tight" is for: the
    #: request is as much about what must *not* change as about what must.
    #:
    #: Dimensions are accepted as well as families because briefs are regional
    #: and families are not. "More width, but keep the low end tight and mono"
    #: is a hold on the bottom width bands and on mono compatibility while the
    #: top bands move -- and every one of those lives in the stereo family, so
    #: a family-only hold could express it only by forbidding the request.
    #: Asked to translate exactly that brief, the model held the whole stereo
    #: family and was then left with nothing it was allowed to widen.
    hold: tuple[str, ...] = ()
    rationale: str = ""
    #: Dimensions the model named that do not exist. Kept rather than dropped:
    #: the rate is a reported number, and a translation that invents dimensions
    #: is a prompt problem worth seeing.
    rejected: tuple[str, ...] = ()
    #: Whether the adjustments arrived as a JSON string rather than an array
    #: and had to be decoded. Recorded so a repaired quirk stays countable.
    repaired_json: bool = False

    def named(self) -> dict[str, float]:
        return {a.dimension: a.offset_tol for a in self.adjustments}

    def apply_to(self, fv: FeatureVector, label: str = "brief") -> TargetProfile:
        """Turn the brief into a target relative to the audio's own measurements.

        Relative, not absolute: "brighter" means brighter *than this*, so the
        offsets are applied to the source's own scored vector. A dimension the
        brief does not name and no held family covers is left unconstrained, so
        the metric neither rewards nor penalizes what happened to it -- and the
        report still measures it, which is how collateral damage stays visible.
        """
        scored = to_scored(fv)
        targets: dict[str, float] = {}
        for name in self.hold:
            if name in SPEC_BY_NAME:
                targets[name] = scored[name]
                continue
            targets.update({n: v for n, v in scored.items() if SPEC_BY_NAME[n].family == name})
        for adjustment in self.adjustments:
            spec = SPEC_BY_NAME[adjustment.dimension]
            targets[adjustment.dimension] = (
                scored[adjustment.dimension] + adjustment.offset_tol * spec.tolerance
            )
        return TargetProfile(
            label=label,
            provenance=f"brief: {self.brief}",
            targets=targets,
        )

    def describe(self) -> str:
        parts = [f"brief: {self.brief!r}"]
        for a in self.adjustments:
            spec = SPEC_BY_NAME[a.dimension]
            parts.append(
                f"  {a.dimension:18s} {a.offset_tol:+5.1f} tol "
                f"({a.offset_tol * spec.tolerance:+6.2f} {spec.unit})"
                + (f"  -- {a.reason}" if a.reason else "")
            )
        if self.hold:
            parts.append(f"  hold: {', '.join(self.hold)}")
        if self.rejected:
            parts.append(f"  rejected (no such dimension): {', '.join(self.rejected)}")
        return "\n".join(parts)


# --- the tool the model is given ---------------------------------------------

TOOL_NAME: Final[str] = "set_target"


def tool_schema() -> dict[str, Any]:
    """One tool, one call. The bounds and the legal dimension names are in the
    schema, so an illegal request is a validation error rather than a
    surprise."""
    return {
        "name": TOOL_NAME,
        "description": (
            "Translate the brief into signed offsets against measured dimensions. "
            "Offsets are relative to the audio's current values and expressed in "
            "tolerance units, so +2 means 'two tolerances more of this than it has "
            "now'. Name only what the brief actually asks for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "adjustments": {
                    "type": "array",
                    "maxItems": MAX_ADJUSTMENTS,
                    "description": "The dimensions the brief asks to move.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dimension": {
                                "type": "string",
                                "enum": sorted(SPEC_BY_NAME),
                                "description": "A scored dimension name.",
                            },
                            "offset_tol": {
                                "type": "number",
                                "minimum": -MAX_OFFSET_TOL,
                                "maximum": MAX_OFFSET_TOL,
                                "description": (
                                    "Signed offset in tolerance units. Positive means "
                                    "more of what the dimension measures. Magnitude "
                                    f"below {MIN_OFFSET_TOL} is inside the metric's "
                                    "indifference band and will do nothing."
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": "The words in the brief this comes from.",
                            },
                        },
                        "required": ["dimension", "offset_tol", "reason"],
                    },
                },
                "hold": {
                    "type": "array",
                    "description": (
                        "Names to pin at their current values because the brief says "
                        "they must not change: either a whole family, or individual "
                        "dimensions. Use dimensions when the brief protects part of a "
                        "family while asking to move the rest -- 'more width but keep "
                        "the low end tight and mono' holds width_0, width_1 and "
                        "mono_compat_db while width_6 to width_8 go up. Never hold "
                        "something you are also adjusting."
                    ),
                    "items": {
                        "type": "string",
                        "enum": sorted(FAMILY_WEIGHTS) + sorted(SPEC_BY_NAME),
                    },
                },
                "rationale": {
                    "type": "string",
                    "description": "One or two sentences on how you read the brief.",
                },
            },
            "required": ["adjustments", "rationale"],
        },
    }


def system_prompt() -> str:
    """Static, so it is byte-identical on every translation call."""
    glossary = "\n".join(DIMENSIONS[role] for role in Role)
    return "\n".join(
        [
            "You translate a mastering brief into measurable targets. You do not "
            "process audio and you do not choose processing; a deterministic "
            "controller does that, and a deterministic metric grades the result.",
            "",
            "Your only output is a set of signed offsets against dimensions that "
            "are already measured, expressed in tolerance units relative to the "
            "audio's current values. One tolerance is roughly the smallest "
            "difference a listener would notice, so +2 is a clear change and +6 is "
            "dramatic. An offset below 1.0 does nothing at all.",
            "",
            "The dimensions and what they measure:",
            glossary,
            "",
            "How to translate:",
            "- Name only what the brief asks for. Every dimension you name becomes "
            "a constraint the controller must satisfy, and naming a dimension the "
            "brief is silent about spends the render budget on something nobody "
            "asked for.",
            "- Watch the sign. More correlation_z is a *narrower* image. More "
            "mono_compat_db is *better* mono compatibility. Higher attack_log2_ms "
            "is a *slower* transient.",
            "- A request for *more* of something is a positive offset on that "
            "region, and that stays true for the relative dimensions. band_clr is "
            "measured against the mean of all nine bands, but 'warmer' still means "
            "the low mids go up, not down: the relativity is in the unit, not in "
            "the direction of the request.",
            "- 'Keep X as it is' is a hold, not an offset. Holds are what stop the "
            "controller from trading away something the brief wanted preserved. "
            "Hold individual dimensions when the brief protects part of a family "
            "and moves the rest, which is the usual shape of a real request. "
            "Holding a whole family you were also asked to change leaves you "
            "nothing you are allowed to move.",
            "- When a brief names a frequency range, map it through the actual "
            "band edges rather than a rule of thumb:",
            *(f"    band {i}: {label}" for i, label in enumerate(_BAND_LABELS)),
            "  So 'upper mids' at 2-4 kHz is band 6, not band 5. 'Boxy' at "
            "300-600 Hz spans bands 3 and 4. 'Air' above 8 kHz is band 8.",
            "- 'Space' and 'width' are the width_* dimensions and correlation_z. "
            "'Punch' is crest factor and attack time. 'Loud' is lufs_integrated.",
            "- If the brief is genuinely about something not measured here, say so "
            "in the rationale and leave it out rather than approximating it with a "
            "dimension that means something else.",
        ]
    )


def parse(brief: str, arguments: dict[str, Any]) -> BriefTarget:
    """Validate a tool call into a :class:`BriefTarget`.

    Unknown dimension names are recorded and dropped rather than raising: one
    invented name should not throw away a translation that was otherwise
    correct, and the count is reported.
    """
    raw = arguments.get("adjustments")
    items, repaired = _as_list(raw)
    adjustments: list[Adjustment] = []
    rejected: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            rejected.append(repr(item)[:40])
            continue
        name = str(item.get("dimension", ""))
        if name not in SPEC_BY_NAME:
            rejected.append(name or "(unnamed)")
            continue
        try:
            adjustments.append(
                Adjustment(
                    dimension=name,
                    offset_tol=float(item.get("offset_tol", 0.0)),
                    reason=str(item.get("reason", "")),
                )
            )
        except (ValidationError, TypeError, ValueError):
            rejected.append(f"{name} (offset out of bounds)")

    raw_hold = arguments.get("hold")
    legal_holds = set(FAMILY_WEIGHTS) | set(SPEC_BY_NAME)
    adjusted = {a.dimension for a in adjustments}
    hold = tuple(
        h
        for h in (raw_hold if isinstance(raw_hold, list) else [])
        if h in legal_holds
        # Holding a dimension the same brief asks to move is contradictory, and
        # the adjustment is the more specific statement of intent, so it wins.
        # Keeping both would pin the value the request wanted changed, which
        # reads downstream as a target nothing can satisfy.
        and h not in adjusted
    )
    return BriefTarget(
        brief=brief,
        # An offset inside the indifference band is dropped: it would read as a
        # constraint the controller must satisfy while being satisfied already.
        adjustments=tuple(a for a in adjustments if abs(a.offset_tol) >= MIN_OFFSET_TOL),
        hold=hold,
        rationale=str(arguments.get("rationale", "")),
        rejected=tuple(rejected),
        repaired_json=repaired,
    )


#: A leading ``+`` on a number, which JSON does not allow.
#:
#: Both halves of this repair were found by reading a real transcript rather
#: than by anticipating them. A model asked for signed offsets will sometimes
#: serialize the array argument as a *string* of JSON, and will sometimes write
#: the positive ones as ``+2.0`` because that is how a signed quantity is
#: written -- which the standard rejects. Together they discarded a translation
#: that was, on inspection, exactly correct in every dimension and sign, and
#: scored it as a comprehension failure. The repair is narrow: a ``+`` is
#: removed only where it directly follows a key's colon and precedes a digit,
#: so a ``+`` inside a string value is untouched.
_LEADING_PLUS: Final[re.Pattern[str]] = re.compile(r"(:\s*)\+(\d|\.)")


def _as_list(raw: object) -> tuple[Sequence[Any], bool]:
    """Coerce the ``adjustments`` argument into a list, repairing what can be.

    The rate of repair is recorded rather than swallowed: a quirk that has been
    papered over silently is a quirk nobody knows about, and if it ever becomes
    common the number is the signal to change the prompt instead.
    """
    if isinstance(raw, list):
        return raw, False
    if not isinstance(raw, str):
        return [], False
    for candidate in (raw, _LEADING_PLUS.sub(r"\1\2", raw)):
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, list):
            return decoded, True
    return [], False


class EmptyBriefError(BoundViolationError):
    """The translation named nothing actionable."""

    def __init__(self, brief: str, rejected: Sequence[str]) -> None:
        super().__init__(
            [
                {
                    "field": "adjustments",
                    "value": list(rejected),
                    "constraint": (
                        f"a brief must name at least one scored dimension with an "
                        f"offset of at least {MIN_OFFSET_TOL} tolerances"
                    ),
                    "type": "empty_translation",
                }
            ]
        )
        self.brief = brief
