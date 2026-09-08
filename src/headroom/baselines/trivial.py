"""Floor baselines: do-nothing, random search, and random hill-climbing.

``null`` is the sanity floor -- its recovery ratio must be ~0, and if it is
not, something is wrong with the metric rather than impressive about the
system.

``random`` is the floor SPEC 9 specifies. In a ~20-dimensional continuous
space it does essentially nothing, which makes it a weak floor: almost any
system clears it, so clearing it proves little.

``hillclimb`` is the floor that actually has teeth. It is a (1+1) evolution
strategy: perturb the best chain found so far, keep the perturbation only if
it helps. It costs nothing to build, uses the identical render budget, and is
often surprisingly competitive -- which is exactly why it belongs here. A
system that cannot beat random hill-climbing is not doing anything.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from headroom.analysis.spectral import BAND_EDGES, N_BANDS
from headroom.control.loop import Proposal, Proposer
from headroom.control.state import LoopState
from headroom.dsp.chain import Chain
from headroom.dsp.ops import (
    EqBand,
    Op,
    op_compressor,
    op_eq,
    op_expander,
    op_gain,
    op_limiter,
    op_stereo_width,
)

NULL_NAME: Final[str] = "null"
RANDOM_NAME: Final[str] = "random"
HILLCLIMB_NAME: Final[str] = "hillclimb"


def null_propose(state: LoopState) -> Proposal:
    """Do nothing, immediately."""
    return Proposal(chain=state.chain, give_up=True, note="null system takes no action")


def _random_op(rng: np.random.Generator) -> Op:
    """A uniformly random in-bounds op."""
    kind = int(rng.integers(0, 6))
    if kind == 0:
        return op_gain(float(rng.uniform(-12.0, 12.0)))
    if kind == 1:
        band = int(rng.integers(0, N_BANDS))
        freq = float((BAND_EDGES[band] * BAND_EDGES[band + 1]) ** 0.5)
        return op_eq(
            [
                EqBand(
                    shape="peak",
                    freq_hz=freq,
                    gain_db=float(rng.uniform(-9.0, 9.0)),
                    q=float(rng.uniform(0.5, 3.0)),
                )
            ]
        )
    if kind == 2:
        return op_compressor(
            threshold_db=float(rng.uniform(-40.0, -6.0)),
            ratio=float(rng.uniform(1.0, 8.0)),
            attack_ms=float(rng.uniform(1.0, 50.0)),
            release_ms=float(rng.uniform(20.0, 400.0)),
        )
    if kind == 3:
        return op_expander(
            threshold_db=float(rng.uniform(-30.0, -4.0)),
            ratio=float(rng.uniform(1.0, 4.0)),
        )
    if kind == 4:
        band_choice = int(rng.integers(-1, N_BANDS))
        return op_stereo_width(
            width=float(rng.uniform(0.2, 1.8)),
            band=None if band_choice < 0 else band_choice,
        )
    return op_limiter(ceiling_dbtp=float(rng.uniform(-3.0, -0.2)))


def make_random_propose(seed: int) -> Proposer:
    """Random search: keep stacking random in-bounds ops."""
    rng = np.random.default_rng(seed)

    def propose(state: LoopState) -> Proposal:
        op = _random_op(rng)
        return Proposal(
            chain=state.chain.add(op),
            action=f"random.{op.kind} {float(rng.uniform(-1.0, 1.0)):+.3f}",
            note="uniform random in-bounds op",
        )

    return propose


def make_hillclimb_propose(seed: int) -> Proposer:
    """(1+1) evolution strategy: perturb the best chain found so far.

    Reverting to the best-so-far before each perturbation is the whole
    difference from ``random``, and it is why this floor is a real one.
    """
    rng = np.random.default_rng(seed)

    def propose(state: LoopState) -> Proposal:
        best: Chain = state.chain
        best_score = state.distance.score
        for record in state.history:
            if record.distance_score < best_score:
                best, best_score = record.chain, record.distance_score

        op = _random_op(rng)
        # Replace an existing op of the same kind half the time, so the chain
        # does not just grow without bound over the budget.
        same_kind = best.of_kind(op.kind)
        if same_kind and rng.random() < 0.5:
            candidate = best.remove(same_kind[0].id).add(op)
        else:
            candidate = best.add(op)

        return Proposal(
            chain=candidate,
            action=f"hillclimb.{op.kind} {float(rng.uniform(-1.0, 1.0)):+.3f}",
            note=f"perturbing best-so-far (score {best_score:.4f})",
        )

    return propose
