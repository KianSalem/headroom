"""The evaluation runner.

Executes (track x degradation x seed x system) and writes one JSON trace per
run. Nothing is aggregated here beyond what a trace already holds: the report
is generated from traces on disk, never from a live run, so a published number
can always be recomputed from committed data.

Two guards exist because without them the headline metric quietly lies:

**Weak degradation cells are skipped.** ``recovery_ratio`` divides by the
initial distance, so a degradation that barely damaged anything turns
measurement noise into an apparently excellent score. Skipped cells are
recorded rather than silently dropped.

**Systems run against identical inputs.** Every system for a given cell gets
the same degraded render and the same target, so the comparison is paired and
a per-cell difference is meaningful.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

from headroom.analysis.features import analyze
from headroom.analysis.features import clear_cache as clear_analysis_cache
from headroom.audio import AudioBuffer
from headroom.baselines import heuristic, optimizer, trivial
from headroom.control.critic import CriticConfig
from headroom.control.loop import Proposer, run_loop
from headroom.control.state import RunTrace
from headroom.dsp.backends.pedalboard import clear_cache as clear_render_cache
from headroom.dsp.backends.pedalboard import render_chain
from headroom.target.profile import TargetProfile

from .corpus import CorpusManifest, TrackRecord
from .degradations import (
    ALL_KINDS,
    MIN_USEFUL_DISTANCE,
    DegradationKind,
    make_degradation,
)

#: Systems driven by the shared loop. The optimizer is handled separately
#: because its render budget is unequal by design.
LOOP_SYSTEMS: Final[tuple[str, ...]] = ("null", "random", "hillclimb", "heuristic")

#: Every system the runner knows how to execute.
ALL_SYSTEMS: Final[tuple[str, ...]] = (*LOOP_SYSTEMS, "optimizer")

#: How long a slice of each track to use. Whole tracks make the evaluation
#: dominated by measurement time; a fixed window keeps runs comparable across
#: tracks of different length.
CLIP_SECONDS: Final[float] = 20.0


def make_proposer(system: str, seed: int) -> Proposer:
    if system == "null":
        return trivial.null_propose
    if system == "random":
        return trivial.make_random_propose(seed)
    if system == "hillclimb":
        return trivial.make_hillclimb_propose(seed)
    if system == "heuristic":
        return heuristic.propose
    raise KeyError(f"{system!r} is not a loop system; have {LOOP_SYSTEMS}")


@dataclass
class SkippedCell:
    track_id: str
    degradation_kind: str
    degradation_seed: int
    initial_distance: float
    reason: str


@dataclass
class RunReport:
    traces: list[RunTrace] = field(default_factory=list)
    skipped: list[SkippedCell] = field(default_factory=list)

    def summary(self) -> str:
        return f"{len(self.traces)} traces, {len(self.skipped)} cells skipped"


def clip(buf: AudioBuffer, seconds: float = CLIP_SECONDS) -> AudioBuffer:
    """Take a centred window, so the slice is musical material rather than an
    intro or a fade."""
    want = int(seconds * buf.sample_rate)
    if buf.n_frames <= want:
        return buf
    start = (buf.n_frames - want) // 2
    return buf.replace_samples(buf.samples[start : start + want])


def run_cell(
    source: AudioBuffer,
    track_id: str,
    kind: DegradationKind,
    seed: int,
    systems: Sequence[str],
    *,
    config: CriticConfig,
    norm: str = "l2",
    optimizer_budget: int = 250,
    min_useful_distance: float = MIN_USEFUL_DISTANCE,
) -> tuple[list[RunTrace], SkippedCell | None]:
    """Run one (track, degradation, seed) cell across systems.

    The optimizer runs last and is seeded with every other system's final
    chain, which is what makes it an upper bound rather than a competitor.
    """
    clean = analyze(source)
    target = TargetProfile.from_features(clean, label=f"original:{track_id}")
    degradation = make_degradation(kind, seed=seed, level_db=clean.lufs_integrated)
    degraded = render_chain(source, degradation.chain)

    from headroom.target.distance import distance

    initial = distance(analyze(degraded), target, norm=norm).score
    if initial < min_useful_distance:
        return [], SkippedCell(
            track_id=track_id,
            degradation_kind=kind,
            degradation_seed=seed,
            initial_distance=initial,
            reason=(
                f"initial distance {initial:.3f} below {min_useful_distance}: "
                "recovery ratio would divide by near-zero"
            ),
        )

    shared = {
        "track_id": track_id,
        "degradation_kind": kind,
        "degradation_seed": seed,
        "degradation_params": degradation.params,
        "degradation_lossy": degradation.is_lossy,
    }

    traces: list[RunTrace] = []
    for system in systems:
        if system == "optimizer":
            continue
        traces.append(
            run_loop(
                system,
                degraded,
                target,
                make_proposer(system, seed),
                config=config,
                norm=norm,
                **shared,  # type: ignore[arg-type]
            )
        )

    if "optimizer" in systems:
        traces.append(
            optimizer.run(
                degraded,
                target,
                render_budget=optimizer_budget,
                seed_chains=[t.final_chain for t in traces if t.final_chain.ops],
                norm=norm,
                config=config,
                **shared,  # type: ignore[arg-type]
            )
        )

    return traces, None


def iter_cells(
    tracks: Sequence[TrackRecord],
    kinds: Sequence[DegradationKind],
    seeds: Sequence[int],
) -> Iterator[tuple[TrackRecord, DegradationKind, int]]:
    for track in tracks:
        for kind in kinds:
            for seed in seeds:
                yield track, kind, seed


def run_matrix(
    manifest: CorpusManifest,
    *,
    split: str = "test",
    kinds: Sequence[DegradationKind] = ALL_KINDS,
    seeds: Sequence[int] = (0, 1, 2),
    systems: Sequence[str] = ALL_SYSTEMS,
    out_dir: str | Path = "results/traces",
    config: CriticConfig | None = None,
    norm: str = "l2",
    optimizer_budget: int = 250,
    clip_seconds: float = CLIP_SECONDS,
    min_useful_distance: float = MIN_USEFUL_DISTANCE,
    verbose: bool = True,
) -> RunReport:
    """Run the full matrix and write one trace per run.

    ``split`` defaults to ``test`` because every headline number comes from the
    test split and nothing is ever tuned on it.
    """
    tracks = manifest.test() if split == "test" else manifest.train()
    critic = config or CriticConfig()
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    report = RunReport()
    for track, kind, seed in iter_cells(tracks, kinds, seeds):
        # The caches are per-cell: across cells they would only hold audio that
        # is never revisited, while inside a cell the hit rate is high.
        clear_render_cache()
        clear_analysis_cache()

        source = clip(track.load(), clip_seconds)
        traces, skipped = run_cell(
            source,
            track.track_id,
            kind,
            seed,
            systems,
            config=critic,
            norm=norm,
            optimizer_budget=optimizer_budget,
            min_useful_distance=min_useful_distance,
        )

        if skipped is not None:
            report.skipped.append(skipped)
            if verbose:
                print(f"skip {track.track_id} {kind}/{seed}: {skipped.reason}")
            continue

        for trace in traces:
            write_trace(trace, destination)
            report.traces.append(trace)
            if verbose:
                print(trace.summary())

    write_skipped(report.skipped, destination)
    return report


def trace_filename(trace: RunTrace) -> str:
    safe_track = trace.track_id.replace("/", "_") or "unknown"
    return f"{safe_track}__{trace.degradation_kind}_{trace.degradation_seed}__{trace.system}.json"


def write_trace(trace: RunTrace, out_dir: str | Path) -> Path:
    path = Path(out_dir) / trace_filename(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json.loads(trace.model_dump_json()), indent=2) + "\n")
    return path


def write_skipped(skipped: Sequence[SkippedCell], out_dir: str | Path) -> Path:
    """Skipped cells are recorded, not silently dropped: which degradations
    failed to damage which tracks is itself information."""
    path = Path(out_dir) / "_skipped.json"
    path.write_text(json.dumps([asdict(s) for s in skipped], indent=2, sort_keys=True) + "\n")
    return path


def load_traces(out_dir: str | Path) -> list[RunTrace]:
    """Read every trace in a directory. The report consumes only this."""
    traces: list[RunTrace] = []
    for path in sorted(Path(out_dir).glob("*.json")):
        if path.name.startswith("_"):
            continue
        traces.append(RunTrace.model_validate_json(path.read_text()))
    return traces
