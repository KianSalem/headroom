"""The heuristic controller: a proportional controller with damping, no LLM.

**This is the real competitor.** A weak baseline would make the whole
comparison worthless, so it is built to win where a proportional controller
should win: single-feature numeric targets, where it maps a measured error
straight onto the parameter that most directly causes it.

Design choices that make it strong rather than a strawman:

*It revises, it does not stack.* A 240 Hz cut is edited from -3 dB to -1.5 dB
rather than having a +1.5 dB boost added on top, which would be a different
and worse filter.

*Per-feature proportional constants.* Level maps onto gain one-for-one and is
essentially decoupled, so it is corrected fully. A band correction is damped
because a peaking filter bleeds into its neighbours. Dynamics corrections are
damped hardest because the mapping from ratio to crest factor is nonlinear and
material-dependent.

*One feature per step, largest contributor first.* This keeps credit
assignment clean -- each render attributes to exactly one decision -- and it
mirrors what a competent engineer does. It is also the honest weakness: with a
finite render budget, a coupled degradation with six bad features cannot be
fixed one feature at a time, which is precisely where coordinating moves
should start to earn its cost.

*It backs out of a move that did not work.* A correction whose direction made
the distance worse is not retried in that direction, and the controller routes
to the next largest contributor instead. Without this it added compression
seven renders in a row on ``over_compress`` while the distance climbed
monotonically -- a compressor with no make-up gain lowers loudness faster than
it lowers peaks, so compressing *raised* the crest factor it was trying to
lower. Leaving that in would have made the baseline a strawman, and the
architecture's headline result an artefact of it.
"""

from __future__ import annotations

from typing import Final

from headroom.analysis.spectral import BAND_EDGES, N_BANDS
from headroom.control.loop import Proposal, direction_key
from headroom.control.state import LoopState
from headroom.dsp.chain import Chain
from headroom.dsp.ops import (
    CompressorOp,
    EqBand,
    EqOp,
    ExpanderOp,
    GainOp,
    LimiterOp,
    OpKind,
    StereoWidthOp,
    op_compressor,
    op_eq,
    op_expander,
    op_gain,
    op_limiter,
    op_stereo_width,
)
from headroom.target.distance import FeatureDelta

NAME: Final[str] = "heuristic"

#: Fraction of a measured error to correct in one step, per feature family.
#: Level is decoupled and corrected fully; a peaking filter bleeds into its
#: neighbours; dynamics are nonlinear and material-dependent.
GAIN_K: Final[float] = 1.0
BAND_K: Final[float] = 0.7
WIDTH_K: Final[float] = 0.8
DYNAMICS_K: Final[float] = 0.4

#: Q for corrective peaking filters. Broad enough to move a whole band without
#: carving a notch inside it.
CORRECTIVE_Q: Final[float] = 1.0

_EPS: Final[float] = 1e-9


def _band_center(band: int) -> float:
    return float((BAND_EDGES[band] * BAND_EDGES[band + 1]) ** 0.5)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _upsert_gain(chain: Chain, delta_db: float) -> tuple[Chain, str]:
    existing = chain.of_kind(OpKind.GAIN)
    if existing:
        op = existing[0]
        assert isinstance(op, GainOp)
        want = _clamp(op.gain_db + delta_db, -24.0, 24.0)
        return chain.edit(op.id, gain_db=want), f"gain.gain_db {want - op.gain_db:+.3f}"
    want = _clamp(delta_db, -24.0, 24.0)
    return chain.add(op_gain(want)), f"gain.gain_db {want:+.3f}"


def _upsert_band(chain: Chain, band: int, delta_db: float) -> tuple[Chain, str]:
    """Revise the corrective filter for one band, or add one."""
    freq = _band_center(band)
    existing = chain.of_kind(OpKind.EQ)
    if existing:
        op = existing[0]
        assert isinstance(op, EqOp)
        bands = list(op.bands)
        for i, existing_band in enumerate(bands):
            if abs(existing_band.freq_hz - freq) < 1.0:
                want = _clamp(existing_band.gain_db + delta_db, -18.0, 18.0)
                bands[i] = existing_band.model_copy(update={"gain_db": want})
                moved = want - existing_band.gain_db
                return (
                    chain.edit(op.id, bands=tuple(b.model_dump() for b in bands)),
                    f"eq.band{band} {moved:+.3f}",
                )
        if len(bands) < 8:
            want = _clamp(delta_db, -18.0, 18.0)
            bands.append(EqBand(shape="peak", freq_hz=freq, gain_db=want, q=CORRECTIVE_Q))
            return (
                chain.edit(op.id, bands=tuple(b.model_dump() for b in bands)),
                f"eq.band{band} {want:+.3f}",
            )
        return chain, ""
    want = _clamp(delta_db, -18.0, 18.0)
    return (
        chain.add(op_eq([EqBand(shape="peak", freq_hz=freq, gain_db=want, q=CORRECTIVE_Q)])),
        f"eq.band{band} {want:+.3f}",
    )


def _upsert_width(chain: Chain, band: int | None, factor: float) -> tuple[Chain, str]:
    label = "global" if band is None else f"band{band}"
    for op in chain.of_kind(OpKind.STEREO_WIDTH):
        assert isinstance(op, StereoWidthOp)
        if op.band == band:
            want = _clamp(op.width * factor, 0.0, 2.0)
            return (
                chain.edit(op.id, width=want),
                f"width.{label} {want - op.width:+.3f}",
            )
    want = _clamp(factor, 0.0, 2.0)
    return chain.add(op_stereo_width(width=want, band=band)), f"width.{label} {want - 1.0:+.3f}"


def _upsert_dynamics(chain: Chain, level_db: float, crest_delta: float) -> tuple[Chain, str]:
    """Crest too high means compress; too low means expand.

    ``crest_delta`` is current minus target, so a positive value means the
    audio is more dynamic than the target and wants compression.
    """
    if crest_delta > 0.0:
        target_ratio = _clamp(1.0 + abs(crest_delta) * DYNAMICS_K, 1.0, 20.0)
        for op in chain.of_kind(OpKind.COMPRESSOR):
            assert isinstance(op, CompressorOp)
            want = _clamp(op.ratio + abs(crest_delta) * DYNAMICS_K, 1.0, 20.0)
            return chain.edit(op.id, ratio=want), f"comp.ratio {want - op.ratio:+.3f}"
        return (
            chain.add(
                op_compressor(
                    threshold_db=_clamp(level_db - 6.0, -60.0, 0.0),
                    ratio=target_ratio,
                    attack_ms=10.0,
                    release_ms=120.0,
                )
            ),
            f"comp.ratio {target_ratio - 1.0:+.3f}",
        )

    target_ratio = _clamp(1.0 + abs(crest_delta) * DYNAMICS_K, 1.0, 8.0)
    for op in chain.of_kind(OpKind.EXPANDER):
        assert isinstance(op, ExpanderOp)
        want = _clamp(op.ratio + abs(crest_delta) * DYNAMICS_K, 1.0, 8.0)
        return chain.edit(op.id, ratio=want), f"exp.ratio {want - op.ratio:+.3f}"
    return (
        chain.add(
            op_expander(
                threshold_db=_clamp(level_db + 10.0, -60.0, 0.0),
                ratio=target_ratio,
                attack_ms=5.0,
                release_ms=80.0,
            )
        ),
        f"exp.ratio {target_ratio - 1.0:+.3f}",
    )


def _upsert_limiter(chain: Chain, target_dbtp: float) -> tuple[Chain, str]:
    want = _clamp(target_dbtp, -3.0, -0.1)
    for op in chain.of_kind(OpKind.LIMITER):
        assert isinstance(op, LimiterOp)
        return (
            chain.edit(op.id, ceiling_dbtp=want),
            f"limiter.ceiling {want - op.ceiling_dbtp:+.3f}",
        )
    return chain.add(op_limiter(ceiling_dbtp=want)), f"limiter.ceiling {want:+.3f}"


def _correct(state: LoopState, worst: FeatureDelta) -> tuple[Chain, str]:
    """Map one feature error onto the parameter that most directly causes it."""
    chain = state.chain
    scale = state.step_scale
    name = worst.name
    # worst.delta is current minus target, so the correction is its negation.
    correction = -worst.delta * scale

    if name == "lufs_integrated":
        return _upsert_gain(chain, correction * GAIN_K)

    if name == "true_peak_dbtp":
        # Only a ceiling breach is actionable by the limiter. Too *little*
        # peak is a level problem, so it routes to gain instead.
        if worst.delta > 0.0:
            return _upsert_limiter(chain, worst.target)
        return _upsert_gain(chain, correction * GAIN_K)

    if name.startswith("band_clr_"):
        return _upsert_band(chain, int(name.rsplit("_", 1)[1]), correction * BAND_K)

    if name.startswith("width_"):
        band = int(name.rsplit("_", 1)[1])
        factor = float(10.0 ** (correction * WIDTH_K / 20.0))
        return _upsert_width(chain, band, factor)

    # Both global width corrections below are expressed in dB and converted,
    # matching the per-band branch just above and the scaffold's stereo
    # planner. Treating the clamped value as a linear factor instead -- which
    # is what this did in v1 -- shares the constants but not the units, and
    # silently gives the two controllers different maximum steps (1.5x here
    # against 1.78x there) on the one role where they were meant to be
    # identical.
    if name == "correlation_z":
        # More correlation means a narrower image, so a positive error wants
        # width increased.
        correction_db = _clamp(worst.delta * 0.25 * scale, -0.5, 0.5) * 10.0
        return _upsert_width(chain, None, float(10.0 ** (correction_db / 20.0)))

    if name == "mono_compat_db":
        # Poor mono compatibility is excess out-of-phase side energy.
        correction_db = -_clamp(abs(worst.delta) * 0.1 * scale, 0.0, 0.5) * 10.0
        return _upsert_width(chain, None, float(10.0 ** (correction_db / 20.0)))

    if name in ("crest_factor_db", "crest_short_p50"):
        return _upsert_dynamics(chain, state.features.lufs_integrated, worst.delta * scale)

    if name == "lra":
        return _upsert_dynamics(chain, state.features.lufs_integrated, worst.delta * scale)

    # flatness, attack time and percussive ratio have no single op that moves
    # them predictably. Rather than guess, fall through and let the next
    # largest contributor be addressed.
    return chain, ""


def propose(state: LoopState) -> Proposal:
    """Correct the largest remaining contributor."""
    ranked = sorted(state.distance.breakdown.values(), key=lambda d: -d.contribution)
    for candidate in ranked:
        if candidate.excess == 0.0:
            break
        chain, action = _correct(state, candidate)
        if not action or abs(float(action.rsplit(" ", 1)[1])) < _EPS:
            continue
        if direction_key(action) in state.tried_and_failed:
            continue
        return Proposal(chain=chain, action=action, note=f"targeting {candidate.name}")

    return Proposal(
        chain=state.chain,
        give_up=True,
        note="no actionable feature: remaining errors have no direct op mapping",
    )


def band_count() -> int:
    return N_BANDS
