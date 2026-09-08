"""What a specialist is shown, and nothing else.

Context isolation is the load-bearing claim of a multi-specialist design, so
it is implemented here as a hard filter rather than an instruction. A
specialist's briefing contains the dimensions it owns, the ops it owns, one
line about what the other roles have done, and its own recent attempts. The
other 18 to 26 scored dimensions are not in the prompt at all.

That is measurable rather than rhetorical: input tokens per call are recorded
in the trace, so the cost of isolation -- four calls instead of one -- can be
weighed against the cost of a single agent carrying all 28 dimensions plus a
growing transcript.

Everything here is deterministic. The briefing is a pure function of loop
state, memory and role, which is what lets the whole agent be tested against a
scripted specialist with no API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from headroom.control.state import LoopState
from headroom.target.distance import FeatureDelta

from . import tools
from .memory import WorkingMemory
from .roles import OWNED_FEATURES, ROLE_BRIEF, Role

#: Out-of-tolerance dimensions listed in full. Beyond this the tail is
#: summarized: a specialist that fixes its four worst dimensions has done its
#: job for the turn, and the rest will be re-measured anyway.
MAX_LISTED: Final[int] = 6

_RELATION: Final[dict[str, str]] = {
    "both": "match",
    "max": "stay at or below",
    "min": "stay at or above",
}


@dataclass(frozen=True, slots=True)
class Briefing:
    role: Role
    step: int
    renders_left: int
    render_budget: int
    score: float
    initial_score: float
    #: Damping from the critic. Below 1.0 means a previous move overshot or
    #: oscillated and this turn's move should be proportionally smaller.
    step_scale: float
    oscillating: bool
    mine: tuple[FeatureDelta, ...]
    n_mine_out: int
    programme_level_db: float
    own_chain: str
    other_chain: str
    own_history: str
    other_history: str
    tried_and_failed: tuple[str, ...]

    def render(self) -> str:
        return "\n".join(self._blocks())

    def _blocks(self) -> list[str]:
        out = [
            f"## Round {self.step + 1}",
            f"Renders left in the whole run: {self.renders_left} of {self.render_budget}. "
            f"Your edits this round cost exactly one render, however many you make.",
            f"Total distance across all 28 dimensions: {self.score:.4f} "
            f"(started at {self.initial_score:.4f}; 0 is a perfect match).",
            f"Programme level: {self.programme_level_db:.1f} LUFS integrated. "
            f"Set any threshold relative to this, not to 0 dBFS.",
            "",
            "## Your dimensions",
            "delta is current minus target. scaled is delta in tolerance units, so "
            "+2.0 means two tolerances too high and anything within +/-1.0 is already "
            "acceptable. Fix the largest magnitudes first.",
            "",
            self._table(),
        ]
        if self.n_mine_out > len(self.mine):
            out.append(
                f"({self.n_mine_out - len(self.mine)} further dimensions of yours are out "
                f"of tolerance by smaller amounts.)"
            )
        out += [
            "",
            "## Your processing",
            self.own_chain,
            "",
            "## Elsewhere in the chain (not yours to edit)",
            self.other_chain,
            "",
            "## Your attempts on this track",
            self.own_history,
            "",
            "## What the other specialists last did",
            self.other_history,
        ]
        if self.step_scale < 1.0:
            out += [
                "",
                "## Step size",
                f"The critic has damped this round to {self.step_scale:.3f} of a full "
                f"correction, because earlier moves overshot or reversed themselves. "
                f"Make a move about {self.step_scale:.0%} of the size you otherwise would.",
            ]
        if self.oscillating:
            out += [
                "",
                "## Oscillation warning",
                "The distance has stopped falling. Repeating a move you have already "
                "tried, or reversing your last one, will end the run. Try a different "
                "dimension or a different op.",
            ]
        if self.tried_and_failed:
            out += [
                "",
                "## Already tried, in this direction, and it made things worse",
                "Do not push these the same way again. Pulling one back the other "
                "way is allowed and is often the right move: " + ", ".join(self.tried_and_failed),
            ]
        return out

    def _table(self) -> str:
        if not self.mine:
            return "Every dimension you own is inside tolerance. Call finish."
        header = (
            f"{'dimension':18s} {'current':>10s} {'':2s} {'target':>10s} "
            f"{'unit':>9s} {'delta':>8s} {'scaled':>8s}"
        )
        rows = [header, "-" * len(header)]
        for d in self.mine:
            rows.append(
                f"{d.name:18s} {d.current:+10.3f} {_RELATION[d.direction][:2]:>2s} "
                f"{d.target:+10.3f} {d.unit:>9s} {d.delta:+8.3f} {d.scaled:+8.2f}"
            )
        return "\n".join(rows)


def build(state: LoopState, memory: WorkingMemory, role: Role) -> Briefing:
    """Filter loop state down to one role's view."""
    owned = set(OWNED_FEATURES[role])
    mine = [d for d in state.distance.breakdown.values() if d.name in owned and not d.in_tolerance]
    mine.sort(key=lambda d: -abs(d.scaled))
    level = state.features.lufs_integrated
    return Briefing(
        role=role,
        step=state.step_index,
        renders_left=state.renders_left,
        render_budget=state.render_budget,
        score=state.distance.score,
        initial_score=state.initial_distance,
        step_scale=state.step_scale,
        oscillating=state.oscillating,
        mine=tuple(mine[:MAX_LISTED]),
        n_mine_out=len(mine),
        programme_level_db=float(level),
        own_chain=tools.describe_owned(state.chain, role),
        other_chain=tools.foreign_ops(state.chain, role),
        own_history=memory.render_own(role),
        other_history=memory.render_others(role),
        tried_and_failed=tuple(sorted(a for a in state.tried_and_failed if _looks_like(a, role))),
    )


_ACTION_PREFIX: Final[dict[Role, tuple[str, ...]]] = {
    Role.EQ: ("eq.",),
    Role.DYNAMICS: ("comp.", "exp."),
    Role.STEREO: ("width.",),
    Role.LOUDNESS: ("gain.", "limiter."),
}


def _looks_like(action: str, role: Role) -> bool:
    return action.startswith(_ACTION_PREFIX[role])


#: Per-role physics, in the system prompt because it never changes.
#:
#: This exists because of a measured failure. Without it the EQ specialist
#: answered a +2.3 dB band error with a +9.9 dB filter, overshot, read the
#: overshoot as a new error in the other direction and burned the render
#: budget. It had the delta in dB and the tolerance in dB; what it did not have
#: was the transfer function from the control it was given to the number it was
#: reading. Supplying that is the job -- it is domain knowledge a mastering
#: engineer has and a general-purpose model has no reason to.
#:
#: It also earns its place economically: the role prompt plus tool schemas is
#: the byte-identical prefix of every call a role makes, so anything invariant
#: belongs here rather than in the per-turn briefing.
CALIBRATION: Final[dict[Role, str]] = {
    Role.EQ: (
        "- A peaking filter at Q 1.0 centred on analysis band i moves "
        "band_clr_i by roughly 0.6 to 0.8 of its own gain, and pushes the two "
        "neighbouring bands the other way by about a fifth of it. So a delta "
        "of +2.0 dB on band 4 wants band 4 set near -2.5 dB. Not -9 dB.\n"
        "- Filters do not stack. The gain you set replaces whatever was there, "
        "so read your current value from 'Your processing' and set the value "
        "you want, not the change you want.\n"
        "- band_clr is a centered log-ratio: the nine values are relative to "
        "their own mean, so cutting one band raises all the others slightly. "
        "Correcting the largest two or three and re-measuring beats trying to "
        "solve all nine at once.\n"
        "- flatness_logit is how flat the spectrum is overall. No single filter "
        "moves it predictably; it follows from getting the bands right. Do not "
        "chase it directly."
    ),
    Role.DYNAMICS: (
        "- Compression ratio maps non-linearly onto crest factor and depends on "
        "the material. A ratio near 1 + 0.4 x |delta| is a sound first move.\n"
        "- A threshold above the programme level does nothing at all. Put it 4 "
        "to 8 dB below the programme level given above.\n"
        "- Expansion is what restores crest factor that compression removed; a "
        "compressor cannot undo compression. Its threshold has to sit above the "
        "quiet passages to reach them, so a few dB above programme level.\n"
        "- Attack time is what moves attack_log2_ms and percussive_logit: a "
        "short attack blunts transients, a long one lets them through. Ratio "
        "and threshold barely touch them."
    ),
    Role.STEREO: (
        "- Width is a ratio and not a dB value. 1.0 is unchanged, 0.0 is mono, "
        "2.0 is double the side signal.\n"
        "- width_i is a side/mid ratio in dB, so a delta of +3 dB on band i is "
        "corrected by multiplying that band's width by 10^(-3/20), about 0.71. "
        "A delta of -6 dB wants roughly double the width.\n"
        "- correlation_z and mono_compat_db both move with the *global* width "
        "control and they pull in opposite directions: widening lowers "
        "correlation and hurts mono compatibility. Move the global width in "
        "small steps and let the per-band controls do the shaping.\n"
        "- Width above about 1.6 on the low bands makes a mix fall apart in "
        "mono. Prefer widening the top."
    ),
    Role.LOUDNESS: (
        "- Output gain moves lufs_integrated one for one and changes nothing "
        "else. A delta of +3.0 LUFS is corrected by setting the gain 3.0 dB "
        "lower than its current value.\n"
        "- true_peak_dbtp in a delivery target is a ceiling, not a setpoint: "
        "sitting below it is not an error. The row's relation column tells you "
        "which it is.\n"
        "- The limiter brings peaks under a ceiling, and lowers crest factor as "
        "a side effect, which the dynamics specialist will then read as its own "
        "problem. Use it when the peak is over the ceiling, not to fix level.\n"
        "- If peaks sit below the ceiling and the level is low, that is a gain "
        "problem. Turn it up; do not reach for the limiter."
    ),
}


#: What each role's own dimensions actually measure.
#:
#: The highest-value thing that was missing. A model given a row reading
#: "mono_compat_db -8.4, delta -3.9" can tell it is out of tolerance and has no
#: idea what the number is, so it guesses at which control to reach for. Naming
#: the physical quantity is what turns the table from arithmetic into something
#: it can reason about.
DIMENSIONS: Final[dict[Role, str]] = {
    Role.EQ: (
        "- band_clr_0 through band_clr_8 are the energy in nine octave-ish "
        "bands, expressed as a centered log-ratio in dB: each band relative to "
        "the mean of all nine. Scale-invariant, so turning the whole mix up "
        "does not change them. Band 0 is 20-60 Hz, 4 is 500-1000 Hz, 8 is "
        "8-20 kHz.\n"
        "- flatness_logit is spectral flatness on a logit scale. High means "
        "noise-like and even, low means tonal and peaky."
    ),
    Role.DYNAMICS: (
        "- crest_factor_db is peak minus RMS over the whole file: how much "
        "dynamic range is left. Around 12 to 18 dB is a lively master, under "
        "8 dB is heavily limited.\n"
        "- crest_short_p50 is the median of the same measure over short "
        "windows, so it describes moment-to-moment punch rather than the "
        "whole-file figure one loud passage can dominate.\n"
        "- lra is EBU R128 loudness range: how much the loudness varies across "
        "the programme, after a relative gate drops the quiet parts.\n"
        "- attack_log2_ms is log2 of the median 10-90% transient rise time in "
        "milliseconds, so +1.0 is twice as slow.\n"
        "- percussive_logit is the percussive fraction of the signal after "
        "harmonic-percussive separation, on a logit scale."
    ),
    Role.STEREO: (
        "- width_0 through width_8 are the side/mid energy ratio in dB, per "
        "band. More negative is narrower. Real mixes are usually narrow in the "
        "low bands and wider up top.\n"
        "- correlation_z is the Fisher transform of the left/right correlation. "
        "High means the channels are nearly the same signal, which is a narrow "
        "image; near zero means uncorrelated and very wide.\n"
        "- mono_compat_db is how much level the mix loses when folded to mono. "
        "Around -3 dB is normal for uncorrelated material; much worse than "
        "that means side energy is cancelling and the mix will hollow out on a "
        "phone speaker."
    ),
    Role.LOUDNESS: (
        "- lufs_integrated is BS.1770-4 gated integrated loudness: perceived "
        "level over the whole programme, K-weighted so it tracks hearing rather "
        "than raw energy. Streaming platforms normalize to it, which is why "
        "hitting it matters more than peak level does.\n"
        "- true_peak_dbtp is the inter-sample peak, measured on 4x oversampled "
        "audio. It is higher than the sample peak, and it is the number a "
        "converter or a lossy encoder actually clips against, which is why "
        "delivery targets leave a dB of headroom below 0."
    ),
}


def role_system_prompt(role: Role) -> str:
    """The static half of the prompt, cached across every call for this role.

    It contains no measurements and no chain state, so it is byte-identical for
    every call a role makes for the whole evaluation. That is what makes prompt
    caching worth having here: the tool schemas and the standing instructions
    are the large, unchanging part, and the per-turn briefing is the small one.
    """
    return "\n".join(
        [
            "You are one of four specialists in an automated audio mastering system.",
            ROLE_BRIEF[role],
            "",
            "How the system works:",
            "- Processing is a declarative chain re-applied to the original audio every "
            "time. You are editing a description, not damaging a file, so a value you "
            "set can be revised freely rather than corrected on top of.",
            "- Your setters are absolute. set_gain(gain_db=-2.0) means the gain op is "
            "-2 dB, not 2 dB quieter than now.",
            "- Make every edit you want this round, then call finish. All of them are "
            "rendered and measured together as one render, so a coordinated set of "
            "moves costs no more than a single timid one.",
            "- Every bound is in the tool schema. A value outside it is rejected with "
            "the bound named; that costs you nothing but the round-trip, so stay "
            "inside it and do not guess.",
            "- You will be told what you tried before and whether it helped. You get no "
            "conversation history beyond that.",
            "",
            "How to decide:",
            "- Work on the largest scaled magnitude first. A dimension inside +/-1.0 "
            "tolerance is done; do not polish it.",
            "- The *size* of your correction matters as much as its direction. Every "
            "row gives you delta in the dimension's own physical unit. A move several "
            "times larger than the measured delta will overshoot, and the next round "
            "will read worse than this one.",
            "- Emit every edit you want, and finish, in a single response. Doing it in "
            "several rounds is billed for the whole context each time and buys nothing.",
            "- If a previous move of yours made the distance worse, do not repeat it in "
            "the same direction; the amount was wrong or the op was.",
            "- If nothing you own is out of tolerance, or nothing you own can move what "
            "is, call finish and say so. Handing the turn back is a real answer and "
            "the supervisor will route elsewhere.",
            "",
            "What your measurements mean:",
            DIMENSIONS[role],
            "",
            "How your controls map onto the measurements:",
            CALIBRATION[role],
        ]
    )
