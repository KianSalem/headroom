"""Evaluating brief translation, without an LLM judge.

The obvious way to grade *"more space, but keep the low end tight"* is to ask a
model whether the output sounds like that, which measures the grader. This does
something else: a brief has a **checkable signature**. Whatever "brighter"
means in detail, it cannot mean less energy up top. Whatever "keep the low end
tight" means, it cannot mean a wider bottom. So each brief carries the regions
it must move and the direction, written down in advance, and the grading is
arithmetic.

Three separate questions get separate scores, because a system can fail any one
while passing the others and the failures mean different things:

**Translation.** Did the target name the right region with the right sign? A
failure here is a comprehension failure.

**Execution.** Did the render actually move it that way? This is where a
plausible-sounding translation that no chain can satisfy gets caught -- asking
for six tolerances of width the processor cannot produce reads as a perfect
translation and delivers a useless master.

**Collateral.** Did the families the brief said to leave alone stay inside
tolerance of where they started? "But keep the low end tight" is half the
brief, and nailing the first half by wrecking the second half is not the job.

## How the rubric is built, and how it was wrong first

An expectation is a **direction on a named region**, not on a specific
dimension. "Low mids" is a frequency range that spans two analysis bands, and a
brief that says "low mids" is not wrong to move either of them. The first
version of this file demanded specific band indices and scored a correct
translation as half wrong, which was a category error in the rubric rather than
a failure of the system. So an expectation is satisfied when *at least* ``need``
dimensions in the region moved the right way, and it is failed when **any**
dimension in the region moved the wrong way -- the second half is what keeps
the loosening from becoming a free pass.

``allow_hold`` exists for the same reason. "Keep the low end tight" is a
request to leave something as it is, and pinning a family at its current values
satisfies that exactly as well as pushing it down does. Refusing to count a
hold would mark the more correct answer wrong.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

from headroom.agent.brief import BriefTarget
from headroom.analysis.features import FeatureVector, analyze
from headroom.analysis.spectral import N_BANDS
from headroom.audio import AudioBuffer
from headroom.control.critic import CriticConfig
from headroom.control.loop import Proposer, run_loop
from headroom.control.state import RunTrace
from headroom.dsp.backends.pedalboard import render_chain
from headroom.target.distance import SPEC_BY_NAME, to_scored

#: A dimension must move by at least this many tolerances to count as having
#: moved. Below it the change is inside the metric's own indifference band, so
#: crediting it would be crediting noise.
MOVED_TOL: Final[float] = 0.5

#: How far a protected or held dimension may drift before it is damage. One
#: tolerance is, by construction, where the metric starts calling something an
#: error.
PROTECT_TOL: Final[float] = 1.0


def bands(sign: int, *indices: int) -> dict[str, int]:
    return {f"band_clr_{i}": sign for i in indices}


def widths(sign: int, *indices: int) -> dict[str, int]:
    return {f"width_{i}": sign for i in indices}


@dataclass(frozen=True, slots=True)
class Expectation:
    """A direction on a region of the measurement space.

    ``want`` carries a sign per dimension rather than one sign for the whole
    region, because some phrases are satisfiable through measurements that move
    opposite ways. "More punch" is more crest factor *or* a faster transient
    rise, and ``attack_log2_ms`` going down says the same thing as
    ``crest_factor_db`` going up. One sign across the region would have marked
    the second reading wrong.
    """

    label: str
    want: dict[str, int]
    #: Dimensions in the region that must move the right way. One by default:
    #: a brief naming a frequency range does not say how many of the bands
    #: covering it to touch.
    need: int = 1
    #: Whether pinning the region at its current values also satisfies this.
    #: True for "keep X as it is" phrasing, where a hold is the better answer.
    allow_hold: bool = False


@dataclass(frozen=True, slots=True)
class BriefCase:
    brief: str
    expect: tuple[Expectation, ...]
    #: Families that must survive the request intact.
    protect: tuple[str, ...] = ()
    note: str = ""

    @property
    def label(self) -> str:
        return self.brief if len(self.brief) <= 46 else self.brief[:43] + "..."

    def named_dims(self) -> set[str]:
        return {d for e in self.expect for d in e.want}


#: The signatures are physical. Phrases whose direction is a matter of taste
#: are not in this set.
CASES: Final[tuple[BriefCase, ...]] = (
    BriefCase(
        brief="Make it brighter and more open up top.",
        expect=(Expectation("top bands", bands(+1, 6, 7, 8)),),
        protect=("loudness",),
        note="cannot be satisfied by turning it up",
    ),
    BriefCase(
        brief="Warmer and rounder in the low mids, please.",
        expect=(Expectation("low mids 120-500 Hz", bands(+1, 2, 3)),),
        note="",
    ),
    BriefCase(
        brief="More space and width, but keep the low end tight and mono.",
        expect=(
            Expectation("upper-band width", widths(+1, 5, 6, 7, 8)),
            Expectation(
                "low end held tight",
                {**widths(-1, 0, 1), "mono_compat_db": +1},
                allow_hold=True,
            ),
        ),
        protect=("loudness",),
        note="the underspecified request: two regions, opposite directions",
    ),
    BriefCase(
        brief="Pull the stereo image in, it is too wide and diffuse.",
        expect=(
            Expectation("channel correlation", {"correlation_z": +1}),
            Expectation("band width", widths(-1, *range(N_BANDS)), need=2),
        ),
        note="more correlation is a narrower image",
    ),
    BriefCase(
        brief="Take the harshness out of the upper mids.",
        expect=(Expectation("upper mids 2-4 kHz", bands(-1, 5, 6)),),
        protect=("loudness",),
        note="",
    ),
    BriefCase(
        brief="Get it up to streaming level without squashing the dynamics.",
        expect=(Expectation("integrated loudness", {"lufs_integrated": +1}),),
        protect=("dynamics",),
        note="level up, crest factor preserved",
    ),
    BriefCase(
        brief="It needs more punch and impact.",
        expect=(
            Expectation(
                "transient impact",
                # More crest, or a faster transient rise. Both are punch.
                {"crest_factor_db": +1, "crest_short_p50": +1, "attack_log2_ms": -1},
            ),
        ),
        note="more peak relative to body",
    ),
    BriefCase(
        brief="Scoop the boxy mids out of it.",
        expect=(Expectation("boxy 250-1000 Hz", bands(-1, 3, 4)),),
        protect=("stereo",),
        note="without disturbing the image",
    ),
)

#: ``(brief, features) -> (target, usage)``. Narrow on purpose, so the eval
#: runs against a stub with no key. The usage half is deliberately untyped:
#: this module must not know what a token is.
Translator = Callable[[str, FeatureVector], tuple[BriefTarget, object]]


@dataclass
class BriefResult:
    case: BriefCase
    target: BriefTarget
    trace: RunTrace | None = None
    translation_score: float = 0.0
    execution_score: float = 0.0
    collateral_score: float = 1.0
    #: Dimensions the target named that no expectation covers. Reported rather
    #: than penalized: a brief can reasonably imply a move it does not say.
    extraneous: tuple[str, ...] = ()
    detail: dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.translation_score == 1.0
            and self.execution_score == 1.0
            and self.collateral_score == 1.0
        )


def _held(target: BriefTarget, dim: str) -> bool:
    return SPEC_BY_NAME[dim].family in target.hold


def score_translation(case: BriefCase, target: BriefTarget) -> tuple[float, tuple[str, ...]]:
    """Right region, right sign, and nothing in the region moved the wrong way."""
    named = target.named()
    hits = 0
    for expectation in case.expect:
        right = sum(
            1 for dim, sign in expectation.want.items() if dim in named and named[dim] * sign > 0
        )
        wrong = any(
            dim in named and named[dim] * sign < 0 for dim, sign in expectation.want.items()
        )
        if expectation.allow_hold and right == 0 and not wrong:
            right = sum(1 for dim in expectation.want if _held(target, dim))
        hits += right >= expectation.need and not wrong
    extraneous = tuple(sorted(set(named) - case.named_dims()))
    return hits / len(case.expect), extraneous


def score_execution(
    case: BriefCase, target: BriefTarget, before: dict[str, float], after: dict[str, float]
) -> tuple[float, dict[str, str]]:
    """Did the render move the region, audibly, and not the other way?"""
    hits = 0
    detail: dict[str, str] = {}
    for expectation in case.expect:
        moved = {
            dim: sign * (after[dim] - before[dim]) / SPEC_BY_NAME[dim].tolerance
            for dim, sign in expectation.want.items()
        }
        right = sum(1 for m in moved.values() if m >= MOVED_TOL)
        wrong = any(m <= -MOVED_TOL for m in moved.values())
        satisfied = right >= expectation.need and not wrong
        if expectation.allow_hold and not satisfied and not wrong:
            pinned = [dim for dim in expectation.want if _held(target, dim)]
            satisfied = bool(pinned) and all(abs(moved[dim]) <= PROTECT_TOL for dim in pinned)
        hits += satisfied
        detail[expectation.label] = (
            f"{right}/{len(expectation.want)} moved as asked"
            + (", some moved the wrong way" if wrong else "")
            + ("  ok" if satisfied else "  MISS")
        )
    return hits / len(case.expect), detail


def score_collateral(
    case: BriefCase, before: dict[str, float], after: dict[str, float]
) -> tuple[float, dict[str, str]]:
    """Did the protected families survive?

    A dimension the brief itself asked to move is exempt: "louder without
    squashing the dynamics" protects one family and asks for another, and a
    protection must not veto the request inside it.
    """
    asked = case.named_dims()
    protected = [
        spec.name
        for spec in SPEC_BY_NAME.values()
        if spec.family in case.protect and spec.name not in asked
    ]
    if not protected:
        return 1.0, {}
    detail: dict[str, str] = {}
    intact = 0
    for name in protected:
        drift = abs(after[name] - before[name]) / SPEC_BY_NAME[name].tolerance
        if drift <= PROTECT_TOL:
            intact += 1
        else:
            detail[name] = f"drifted {drift:.2f} tol"
    return intact / len(protected), detail


def run_case(
    case: BriefCase,
    source: AudioBuffer,
    translate: Translator,
    propose: Proposer,
    *,
    system: str = "agent",
    config: CriticConfig | None = None,
    track_id: str = "",
) -> BriefResult:
    """Translate, run the loop, and score all three questions."""
    features = analyze(source)
    try:
        target_spec, _ = translate(case.brief, features)
    except Exception as exc:  # a translation failure is a result, not a crash
        return BriefResult(
            case=case, target=BriefTarget(brief=case.brief), error=f"translation failed: {exc!r}"
        )

    translation_score, extraneous = score_translation(case, target_spec)
    if not target_spec.adjustments:
        return BriefResult(
            case=case,
            target=target_spec,
            translation_score=translation_score,
            extraneous=extraneous,
            error="translation named nothing actionable",
        )

    profile = target_spec.apply_to(features, label=f"brief:{case.label}")
    trace = run_loop(
        system,
        source,
        profile,
        propose,
        track_id=track_id,
        degradation_kind="brief",
        config=config or CriticConfig(),
    )
    rendered = render_chain(source, trace.final_chain)
    before, after = to_scored(features), to_scored(analyze(rendered))

    execution_score, execution_detail = score_execution(case, target_spec, before, after)
    collateral_score, collateral_detail = score_collateral(case, before, after)
    return BriefResult(
        case=case,
        target=target_spec,
        trace=trace,
        translation_score=translation_score,
        execution_score=execution_score,
        collateral_score=collateral_score,
        extraneous=extraneous,
        detail={**execution_detail, **collateral_detail},
    )


def summarize(results: Sequence[BriefResult]) -> str:
    """One line of aggregate scores. The cost named is the *controller's*: the
    translation is one call per brief and is billed separately, because the
    whole point of the comparison below is that the two are separable."""
    n = len(results) or 1
    controller = sum(r.trace.total_cost_usd for r in results if r.trace)
    return (
        f"{sum(r.passed for r in results)} of {len(results)} briefs satisfied all "
        f"three. Means: translation {sum(r.translation_score for r in results) / n:.0%}, "
        f"execution {sum(r.execution_score for r in results) / n:.0%}, "
        f"collateral {sum(r.collateral_score for r in results) / n:.0%}. "
        f"Controller cost ${controller:.4f}."
    )


def render_comparison(by_system: dict[str, Sequence[BriefResult]]) -> str:
    """Two controllers, the same translations, side by side.

    The translations are identical by construction -- replayed from the same
    cassette -- so the translation column has to match and any difference in
    the other two is the controller's alone. That is the whole experiment: the
    model reads the intent, and the question is who should close the loop.
    """
    if len(by_system) < 2:
        return ""
    lines = [
        "### Translation against controller",
        "",
        "The same recorded translations driving different controllers, so the "
        "`translation` column is identical by construction and any difference "
        "in the other two belongs to the controller alone.",
        "",
        "| controller | translation | execution | collateral | passed | controller cost |",
        "|---|---|---|---|---|---|",
    ]
    for system, results in by_system.items():
        n = len(results) or 1
        lines.append(
            f"| `{system}` | {sum(r.translation_score for r in results) / n:.0%} | "
            f"{sum(r.execution_score for r in results) / n:.0%} | "
            f"{sum(r.collateral_score for r in results) / n:.0%} | "
            f"{sum(r.passed for r in results)}/{len(results)} | "
            f"${sum(r.trace.total_cost_usd for r in results if r.trace):.4f} |"
        )
    return "\n".join(lines) + "\n"


def render_markdown(results: Sequence[BriefResult], heading: str = "### Briefs") -> str:
    """The brief table, as it appears in the report."""
    if not results:
        return ""
    lines = [
        *([heading, ""] if heading else []),
        "Nothing here is judged by a model. Each brief carries the regions it "
        "must move and the direction, written down in advance, and all three "
        "columns are arithmetic. `translation` is whether the target named the "
        "region correctly, `execution` whether the render actually moved it, "
        "and `collateral` whether the families the brief said to leave alone "
        "stayed inside tolerance. An expectation fails if anything in its "
        "region moved the wrong way.",
        "",
        "| brief | translation | execution | collateral | renders | cost |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        cost = r.trace.total_cost_usd if r.trace else 0.0
        renders = r.trace.n_renders if r.trace else 0
        lines.append(
            f"| {r.case.label} | {r.translation_score:.0%} | {r.execution_score:.0%} | "
            f"{r.collateral_score:.0%} | {renders} | ${cost:.4f} |"
        )
    lines.append("")
    lines.append(summarize(results))
    failures = [r for r in results if r.error]
    if failures:
        lines.append("")
        lines.extend(f"- `{r.case.label}`: {r.error}" for r in failures)
    return "\n".join(lines) + "\n"
