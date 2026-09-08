"""Loop state, step records and the run trace.

Every system -- do-nothing, random, hillclimb, heuristic, numerical optimizer
and the agent -- runs inside the same loop and produces the same trace. That is
deliberate: if each system did its own bookkeeping, a comparison between them
would partly measure the bookkeeping. It also fixes the budget currency as
**renders**, not LLM calls or wall time, so no system gets free exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from headroom.analysis.features import FeatureVector
from headroom.dsp.chain import Chain
from headroom.target.distance import DistanceResult
from headroom.target.profile import TargetProfile

#: Bumped whenever the trace layout changes. Old traces stay readable, which
#: matters because the report is generated from traces, never from live runs.
#: Version 2 added the per-step agent fields (``role``, ``n_edits``,
#: ``n_rejected``); every one has a default, so version 1 traces still load.
TRACE_SCHEMA_VERSION: Final[str] = "2"


class AbortReason(StrEnum):
    """Why a run stopped without converging.

    Enumerated up front and never extended casually: the distribution of abort
    reasons across the evaluation is a headline result, and adding a value
    after the fact would mean re-running everything to keep the distribution
    comparable.
    """

    MAX_STEPS = "max_steps"
    NO_IMPROVEMENT = "no_improvement"
    OSCILLATION_UNRESOLVED = "oscillation_unresolved"
    BOUND_SATURATION = "bound_saturation"
    PROPOSAL_ERROR = "proposal_error"
    PROPOSAL_EMPTY = "proposal_empty"
    RENDER_FAILURE = "render_failure"
    TOKEN_BUDGET = "token_budget"
    COST_BUDGET = "cost_budget"
    #: The system made the audio dramatically worse than it started. Named
    #: for what was observed rather than for a cause: the target may well be
    #: reachable, the system just ran away from it.
    DIVERGED = "diverged"


class Verdict(StrEnum):
    CONTINUE = "continue"
    CONVERGED = "converged"
    ABORT = "abort"


class StepRecord(BaseModel):
    """One iteration, as written to the trace."""

    model_config = ConfigDict(frozen=True)

    index: int
    chain: Chain
    chain_fingerprint: str
    distance_score: float
    n_out_of_tolerance: int
    by_family: dict[str, float]
    #: Signed, tolerance-scaled deltas for the worst offenders. The full
    #: 28-dimension breakdown is recoverable from the chain and the source, so
    #: only the part that drove the decision is stored.
    worst: list[dict[str, float | str]]
    action: str
    step_scale: float
    oscillating: bool
    verdict: Verdict
    renders_used: int
    elapsed_s: float
    #: Populated only by LLM-backed systems; zero elsewhere.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    #: Populated by multi-agent systems: which specialist moved, how many
    #: edits it bundled into this render, and how many of its tool calls were
    #: refused. Zero or empty elsewhere.
    role: str = ""
    n_edits: int = 1
    n_rejected: int = 0
    note: str = ""


class RunTrace(BaseModel):
    """The complete record of one run. The report is generated from these."""

    model_config = ConfigDict(frozen=True)

    trace_schema_version: str = TRACE_SCHEMA_VERSION
    system: str
    track_id: str
    degradation_kind: str
    degradation_seed: int
    degradation_params: dict[str, float | int | str] = Field(default_factory=dict)
    degradation_lossy: bool = False

    source_hash: str
    target_label: str
    target_provenance: str

    initial_distance: float
    final_distance: float
    recovery_ratio: float
    converged: bool
    abort_reason: AbortReason | None = None

    final_chain: Chain
    steps: tuple[StepRecord, ...] = ()

    n_renders: int = 0
    wall_time_s: float = 0.0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0

    #: Provenance, so a number can always be traced to the code that made it.
    git_sha: str = ""
    config_hash: str = ""
    package_versions: dict[str, str] = Field(default_factory=dict)
    model: str = ""
    effort: str = ""
    replayed_from_cassette: bool = False
    #: Provenance a particular system wants published that the shared loop
    #: cannot know about -- the agent records its routing statistics here.
    #: Empty for every other system, so one trace layout still serves all of
    #: them and the comparison stays a comparison.
    system_stats: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        outcome = "CONVERGED" if self.converged else f"abort:{self.abort_reason}"
        return (
            f"{self.system:12s} {self.track_id[:24]:24s} "
            f"{self.degradation_kind}/{self.degradation_seed} "
            f"{self.initial_distance:7.3f} -> {self.final_distance:7.3f} "
            f"recovery {self.recovery_ratio:+.3f}  {len(self.steps):2d} steps  "
            f"{self.n_renders:3d} renders  ${self.total_cost_usd:.4f}  {outcome}"
        )


@dataclass
class LoopState:
    """What a proposer sees when deciding the next move.

    Mutable by the loop, read-only in spirit by proposers.
    """

    source: object  # AudioBuffer; typed loosely to keep this module import-light
    target: TargetProfile
    chain: Chain
    features: FeatureVector
    distance: DistanceResult
    initial_distance: float
    step_index: int
    step_scale: float
    renders_used: int
    render_budget: int
    history: list[StepRecord] = field(default_factory=list)
    oscillating: bool = False
    #: Op parameters whose last two edits flipped sign. Frozen to break a
    #: boost-cut-boost cycle.
    frozen_params: set[str] = field(default_factory=set)
    #: Actions that increased distance, so they are not retried.
    tried_and_failed: set[str] = field(default_factory=set)

    @property
    def renders_left(self) -> int:
        return max(self.render_budget - self.renders_used, 0)

    def score_history(self) -> list[float]:
        return [s.distance_score for s in self.history]
