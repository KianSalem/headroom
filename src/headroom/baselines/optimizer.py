"""The achievable ceiling: numerical optimization over the chain parameters.

**This is a reference, not a competitor.** It exists to answer two questions
that otherwise have no answer.

*What does a recovery ratio of 0.7 mean?* Nothing, on its own. It could be
near-perfect or half the available gain. Comparing against the best result any
method can reach with this op vocabulary turns it into a statement.

*Why is an LLM anywhere near a 13-dimensional continuous optimization with a
cheap deterministic objective?* A fair question, and the honest answer is a
measured one. This baseline will very likely produce the lowest distance of
any system here. What it cannot do is reach that answer in fourteen renders,
or act on "more space, but keep the low end tight" -- and those, not raw
distance, are what the rest of the project is about.

The budget is deliberately unequal and reported as such. Every other system
gets fourteen renders; this one gets hundreds, and its render count appears in
the results table beside its score so the trade is visible.

It is also **seeded with the other systems' final chains**, which is what makes
it a bound rather than just another controller. Measured cold, blind Powell
search lost to the heuristic on coupled degradations -- a 13-dimensional space
is not reliably searchable from a standing start -- and a "ceiling" that a
competitor beats is not a ceiling. A bound is allowed to use information no
controller has, including another system's answer; a peer is not, which is
exactly why this is not scored as one.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Final

import numpy as np
from scipy import optimize

from headroom.analysis.features import analyze
from headroom.analysis.spectral import BAND_EDGES, N_BANDS
from headroom.audio import AudioBuffer
from headroom.control.critic import DEFAULT_CONFIG, CriticConfig
from headroom.control.loop import config_hash, git_sha, package_versions
from headroom.control.state import AbortReason, RunTrace, StepRecord, Verdict
from headroom.dsp.backends.pedalboard import render_chain
from headroom.dsp.chain import Chain
from headroom.dsp.ops import EqBand, Op, op_compressor, op_eq, op_expander, op_gain, op_stereo_width
from headroom.target.distance import distance, recovery_ratio
from headroom.target.profile import TargetProfile

NAME: Final[str] = "optimizer"

#: Corrective filter Q, matching the heuristic so the two search the same space.
CORRECTIVE_Q: Final[float] = 1.0

#: (low, high) per parameter. Order: gain, nine band gains, global width,
#: compressor ratio, expander ratio.
BOUNDS: Final[tuple[tuple[float, float], ...]] = (
    (-24.0, 24.0),
    *((-18.0, 18.0) for _ in range(N_BANDS)),
    (0.0, 2.0),
    (1.0, 20.0),
    (1.0, 8.0),
)

N_PARAMS: Final[int] = len(BOUNDS)

#: Ratios below this are treated as bypass, so the optimizer can decline to
#: use a dynamics stage rather than being forced to include a near-identity one.
BYPASS_RATIO: Final[float] = 1.02
#: EQ and width moves smaller than these are omitted for the same reason: the
#: resulting chain should be readable, not littered with 0.01 dB filters.
MIN_EQ_DB: Final[float] = 0.05
MIN_WIDTH_DELTA: Final[float] = 0.01


def _band_center(band: int) -> float:
    return float((BAND_EDGES[band] * BAND_EDGES[band + 1]) ** 0.5)


def build_chain(params: np.ndarray, level_db: float) -> Chain:
    """Turn a parameter vector into a chain, omitting near-identity stages."""
    ops: list[Op] = []

    if abs(float(params[0])) >= MIN_EQ_DB:
        ops.append(op_gain(float(np.clip(params[0], -24.0, 24.0))))

    bands = [
        EqBand(
            shape="peak",
            freq_hz=_band_center(i),
            gain_db=float(np.clip(params[1 + i], -18.0, 18.0)),
            q=CORRECTIVE_Q,
        )
        for i in range(N_BANDS)
        if abs(float(params[1 + i])) >= MIN_EQ_DB
    ]
    if bands:
        ops.append(op_eq(bands[:8]))

    width = float(np.clip(params[1 + N_BANDS], 0.0, 2.0))
    if abs(width - 1.0) >= MIN_WIDTH_DELTA:
        ops.append(op_stereo_width(width=width))

    comp_ratio = float(np.clip(params[2 + N_BANDS], 1.0, 20.0))
    if comp_ratio > BYPASS_RATIO:
        ops.append(
            op_compressor(
                threshold_db=float(np.clip(level_db - 6.0, -60.0, 0.0)),
                ratio=comp_ratio,
                attack_ms=10.0,
                release_ms=120.0,
            )
        )

    exp_ratio = float(np.clip(params[3 + N_BANDS], 1.0, 8.0))
    if exp_ratio > BYPASS_RATIO:
        ops.append(
            op_expander(
                threshold_db=float(np.clip(level_db + 10.0, -60.0, 0.0)),
                ratio=exp_ratio,
            )
        )

    return Chain(ops=tuple(ops))


def identity_params() -> np.ndarray:
    """The parameter vector that renders an empty chain."""
    params = np.zeros(N_PARAMS, dtype=np.float64)
    params[1 + N_BANDS] = 1.0  # unity width
    params[2 + N_BANDS] = 1.0  # bypass compressor
    params[3 + N_BANDS] = 1.0  # bypass expander
    return params


def run(
    source: AudioBuffer,
    target: TargetProfile,
    *,
    track_id: str = "",
    degradation_kind: str = "none",
    degradation_seed: int = 0,
    degradation_params: dict[str, float | int | str] | None = None,
    degradation_lossy: bool = False,
    render_budget: int = 250,
    n_starts: int = 3,
    seed_chains: Sequence[Chain] = (),
    norm: str = "l2",
    seed: int = 0,
    config: CriticConfig = DEFAULT_CONFIG,
) -> RunTrace:
    """Multi-start Powell search over the chain parameters.

    Powell is derivative-free, which suits an objective that is only available
    by rendering audio. Multiple starts are used because a single local search
    from identity stalls on coupled degradations.

    ``seed_chains`` are evaluated directly before the search and the best is
    kept, so the reported result is never worse than the best system fed in.
    That is what makes this an upper bound instead of a competitor.
    """
    started = time.perf_counter()
    baseline_features = analyze(source)
    level_db = baseline_features.lufs_integrated
    initial = distance(baseline_features, target, norm=norm).score

    renders = 0
    best_params = identity_params()
    best_score = initial
    best_seed_chain: Chain | None = None
    trajectory: list[float] = []
    converged_early = False

    # Evaluate the seeded chains first. Cheap (one render each) and it is what
    # guarantees the bound holds.
    for candidate in seed_chains:
        try:
            rendered = render_chain(source, candidate)
        except (FloatingPointError, ValueError):
            continue
        renders += 1
        score = distance(analyze(rendered), target, norm=norm).score
        trajectory.append(score)
        if score < best_score:
            best_score, best_seed_chain = score, candidate

    def objective(params: np.ndarray) -> float:
        nonlocal renders, best_params, best_score, best_seed_chain, converged_early
        if renders >= render_budget or converged_early:
            # Powell has no hard evaluation cap, so the budget is enforced by
            # making further evaluation unattractive rather than by raising.
            return float(best_score + 1e3)
        chain = build_chain(params, level_db)
        try:
            rendered = render_chain(source, chain)
        except (FloatingPointError, ValueError):
            return float(initial * 10.0)
        renders += 1
        result = distance(analyze(rendered), target, norm=norm)
        score = result.score
        trajectory.append(score)
        if score < best_score:
            best_score = score
            best_params = np.array(params, dtype=np.float64)
            best_seed_chain = None
        if result.converged:
            # Nothing left to find. Spending the rest of the budget would only
            # inflate the reported render count.
            converged_early = True
        return score

    rng = np.random.default_rng(seed)
    starts = [identity_params()]
    for _ in range(max(n_starts - 1, 0)):
        starts.append(np.array([rng.uniform(low, high) for low, high in BOUNDS], dtype=np.float64))

    for start in starts:
        if renders >= render_budget or converged_early:
            break
        optimize.minimize(
            objective,
            start,
            method="Powell",
            bounds=BOUNDS,
            options={"maxfev": max(render_budget - renders, 1), "xtol": 1e-2, "ftol": 1e-3},
        )

    best_chain = (
        best_seed_chain if best_seed_chain is not None else build_chain(best_params, level_db)
    )
    final_result = distance(analyze(render_chain(source, best_chain)), target, norm=norm)

    steps = (
        StepRecord(
            index=0,
            chain=best_chain,
            chain_fingerprint=best_chain.fingerprint(),
            distance_score=final_result.score,
            n_out_of_tolerance=final_result.n_out_of_tolerance,
            by_family={k: round(v, 6) for k, v in final_result.by_family.items()},
            worst=[
                {"name": d.name, "delta": round(d.delta, 4), "scaled": round(d.scaled, 4)}
                for d in final_result.worst(5)
            ],
            action=f"powell.{n_starts}start {final_result.score - initial:+.4f}",
            step_scale=1.0,
            oscillating=False,
            verdict=Verdict.CONVERGED if final_result.converged else Verdict.ABORT,
            renders_used=renders,
            elapsed_s=round(time.perf_counter() - started, 4),
            note=(
                f"multi-start Powell, {renders} renders, "
                f"{len(seed_chains)} seeded chain(s)"
                + (", stopped early on convergence" if converged_early else "")
                + f". Unequal budget by design; every other system gets "
                f"{config.render_budget}."
            ),
        ),
    )

    return RunTrace(
        system=NAME,
        track_id=track_id,
        degradation_kind=degradation_kind,
        degradation_seed=degradation_seed,
        degradation_params=degradation_params or {},
        degradation_lossy=degradation_lossy,
        source_hash=source.content_hash(),
        target_label=target.label,
        target_provenance=target.provenance,
        initial_distance=initial,
        final_distance=final_result.score,
        recovery_ratio=recovery_ratio(initial, final_result.score),
        converged=final_result.converged,
        abort_reason=None if final_result.converged else AbortReason.MAX_STEPS,
        final_chain=best_chain,
        steps=steps,
        n_renders=renders,
        wall_time_s=round(time.perf_counter() - started, 4),
        git_sha=git_sha(),
        config_hash=config_hash(config, target, norm),
        package_versions=package_versions(),
    )
