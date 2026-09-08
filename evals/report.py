"""Aggregation and reporting.

Reads traces from disk and produces the results table. Nothing here touches
audio or the API: a published number must be recomputable from committed data.

Three choices about presentation, each because the obvious alternative
misleads:

**Median and IQR, not mean and standard deviation.** SPEC 10.3. One
catastrophic run should not be able to hide inside an average, and it should
not be able to dominate one either.

**Paired comparison.** Every system runs the identical (track, degradation,
seed) cell against the identical target, so the runs are paired. Comparing
marginal medians throws that structure away; a per-cell difference with a
signed-rank test uses it.

**Sliced by degradation type.** The interesting question is never "which system
is better" but "which cases does each win". A single overall number cannot
answer it, and SPEC 14 requires the report to be designed for that from the
start.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
from scipy import stats

from headroom.control.state import RunTrace

#: Minimum paired observations before a significance test says anything. Below
#: this the test has no power and quoting it would be theatre.
MIN_PAIRED_N: Final[int] = 6


@dataclass(frozen=True, slots=True)
class Cell:
    """Aggregated results for one system, optionally within one degradation."""

    system: str
    degradation_kind: str
    n: int
    recovery_median: float
    recovery_q1: float
    recovery_q3: float
    converged_rate: float
    oscillation_rate: float
    regression_rate: float
    renders_median: float
    wall_median_s: float
    cost_total_usd: float
    aborts: dict[str, int] = field(default_factory=dict)

    @property
    def recovery_iqr(self) -> float:
        return self.recovery_q3 - self.recovery_q1


def _aggregate(system: str, kind: str, traces: Sequence[RunTrace]) -> Cell:
    recoveries = np.array([t.recovery_ratio for t in traces], dtype=np.float64)
    finite = recoveries[np.isfinite(recoveries)]
    if finite.size == 0:
        finite = np.array([0.0])
    q1, median, q3 = (float(v) for v in np.percentile(finite, [25.0, 50.0, 75.0]))

    return Cell(
        system=system,
        degradation_kind=kind,
        n=len(traces),
        recovery_median=median,
        recovery_q1=q1,
        recovery_q3=q3,
        converged_rate=float(np.mean([t.converged for t in traces])),
        oscillation_rate=float(np.mean([any(s.oscillating for s in t.steps) for t in traces])),
        regression_rate=float(np.mean([t.recovery_ratio < 0.0 for t in traces])),
        renders_median=float(np.median([t.n_renders for t in traces])),
        wall_median_s=float(np.median([t.wall_time_s for t in traces])),
        cost_total_usd=float(sum(t.total_cost_usd for t in traces)),
        aborts=dict(Counter(str(t.abort_reason) for t in traces if t.abort_reason is not None)),
    )


@dataclass
class ResultsTable:
    overall: list[Cell] = field(default_factory=list)
    by_kind: list[Cell] = field(default_factory=list)
    n_traces: int = 0
    config_hashes: set[str] = field(default_factory=set)
    git_shas: set[str] = field(default_factory=set)

    @property
    def comparable(self) -> bool:
        """False when traces were produced under different metric configs.

        Two runs with different tolerances or weights are not comparable, and
        the report must say so rather than quietly averaging across them.
        """
        return len(self.config_hashes) <= 1


def aggregate(traces: Sequence[RunTrace]) -> ResultsTable:
    table = ResultsTable(
        n_traces=len(traces),
        config_hashes={t.config_hash for t in traces if t.config_hash},
        git_shas={t.git_sha for t in traces if t.git_sha},
    )
    systems = sorted({t.system for t in traces})
    kinds = sorted({t.degradation_kind for t in traces})

    for system in systems:
        rows = [t for t in traces if t.system == system]
        if rows:
            table.overall.append(_aggregate(system, "ALL", rows))
        for kind in kinds:
            subset = [t for t in rows if t.degradation_kind == kind]
            if subset:
                table.by_kind.append(_aggregate(system, kind, subset))
    return table


@dataclass(frozen=True, slots=True)
class PairedResult:
    system_a: str
    system_b: str
    n: int
    median_difference: float
    a_wins: int
    b_wins: int
    ties: int
    p_value: float | None
    note: str = ""

    def describe(self) -> str:
        significance = "n too small to test" if self.p_value is None else f"p={self.p_value:.4f}"
        return (
            f"{self.system_a} vs {self.system_b}: n={self.n}, "
            f"median difference {self.median_difference:+.3f} recovery, "
            f"{self.a_wins}W/{self.b_wins}L/{self.ties}T, {significance}"
        )


def paired(traces: Sequence[RunTrace], system_a: str, system_b: str) -> PairedResult:
    """Compare two systems on the cells both actually ran.

    A Wilcoxon signed-rank test is used rather than a t-test: recovery ratios
    are bounded above, unbounded below and not remotely normal, so a test that
    assumes normality would be measuring the wrong thing.
    """

    def key(trace: RunTrace) -> tuple[str, str, int]:
        return (trace.track_id, trace.degradation_kind, trace.degradation_seed)

    left = {key(t): t.recovery_ratio for t in traces if t.system == system_a}
    right = {key(t): t.recovery_ratio for t in traces if t.system == system_b}
    shared = sorted(set(left) & set(right))

    diffs = np.array(
        [left[k] - right[k] for k in shared if np.isfinite(left[k]) and np.isfinite(right[k])],
        dtype=np.float64,
    )
    if diffs.size == 0:
        return PairedResult(system_a, system_b, 0, 0.0, 0, 0, 0, None, "no shared cells")

    p_value: float | None = None
    note = ""
    if diffs.size >= MIN_PAIRED_N and float(np.abs(diffs).max()) > 0.0:
        try:
            p_value = float(stats.wilcoxon(diffs, zero_method="zsplit").pvalue)
        except ValueError as exc:  # pragma: no cover - degenerate input
            note = f"test skipped: {exc}"
    else:
        note = f"n<{MIN_PAIRED_N} or all differences zero"

    return PairedResult(
        system_a=system_a,
        system_b=system_b,
        n=int(diffs.size),
        median_difference=float(np.median(diffs)),
        a_wins=int(np.sum(diffs > 1e-9)),
        b_wins=int(np.sum(diffs < -1e-9)),
        ties=int(np.sum(np.abs(diffs) <= 1e-9)),
        p_value=p_value,
        note=note,
    )


def _fmt_cell(cell: Cell) -> str:
    return (
        f"| {cell.system} | {cell.n} | {cell.recovery_median:+.3f} | "
        f"{cell.recovery_q1:+.3f} to {cell.recovery_q3:+.3f} | "
        f"{cell.converged_rate:.0%} | {cell.oscillation_rate:.0%} | "
        f"{cell.regression_rate:.0%} | {cell.renders_median:.0f} | "
        f"{cell.wall_median_s:.1f} | ${cell.cost_total_usd:.4f} |"
    )


def render_markdown(table: ResultsTable) -> str:
    """The results table, as it appears in the README."""
    header = (
        "| system | n | recovery (median) | IQR | converged | oscillated | "
        "regressed | renders | wall s | cost |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    lines: list[str] = []

    if not table.comparable:
        lines.append(
            "> **Not comparable.** These traces were produced under "
            f"{len(table.config_hashes)} different metric configurations. "
            "Tolerances or weights changed between runs, so the rows below do "
            "not measure the same thing.\n"
        )

    lines.append("### Overall\n")
    lines.append(header)
    lines.extend(_fmt_cell(c) for c in sorted(table.overall, key=lambda c: -c.recovery_median))

    kinds = sorted({c.degradation_kind for c in table.by_kind})
    for kind in kinds:
        rows = [c for c in table.by_kind if c.degradation_kind == kind]
        lines.append(f"\n### {kind}\n")
        lines.append(header)
        lines.extend(_fmt_cell(c) for c in sorted(rows, key=lambda c: -c.recovery_median))

    aborts: Counter[str] = Counter()
    for cell in table.overall:
        aborts.update(cell.aborts)
    if aborts:
        lines.append("\n### Abort reasons\n")
        lines.append("| reason | count |\n|---|---|")
        lines.extend(f"| {reason} | {count} |" for reason, count in aborts.most_common())

    lines.append(
        f"\n_{table.n_traces} traces"
        + (f", git {sorted(table.git_shas)[0]}" if table.git_shas else "")
        + (f", config {sorted(table.config_hashes)[0]}" if table.config_hashes else "")
        + "_"
    )
    return "\n".join(lines) + "\n"


def render_comparisons(traces: Sequence[RunTrace], reference: str = "heuristic") -> str:
    """Paired comparisons against the system that actually has to be beaten."""
    systems = sorted({t.system for t in traces} - {reference})
    if not systems:
        return ""
    lines = [
        f"### Paired against `{reference}`\n",
        "Same track, degradation and seed for both systems, so differences are "
        "per-cell rather than marginal.\n",
        "| system | n | median difference | wins | losses | ties | signed-rank p |",
        "|---|---|---|---|---|---|---|",
    ]
    for system in systems:
        result = paired(traces, system, reference)
        p = "—" if result.p_value is None else f"{result.p_value:.4f}"
        lines.append(
            f"| {system} | {result.n} | {result.median_difference:+.3f} | "
            f"{result.a_wins} | {result.b_wins} | {result.ties} | {p} |"
        )
    return "\n".join(lines) + "\n"


def write_markdown(traces: Sequence[RunTrace], path: str | Path) -> Path:
    table = aggregate(traces)
    body = render_markdown(table) + "\n" + render_comparisons(traces)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    return out


def write_json(traces: Sequence[RunTrace], path: str | Path) -> Path:
    """Machine-readable aggregate, so the README table can be regenerated."""
    table = aggregate(traces)
    payload = {
        "n_traces": table.n_traces,
        "comparable": table.comparable,
        "config_hashes": sorted(table.config_hashes),
        "git_shas": sorted(table.git_shas),
        "overall": [asdict(c) for c in table.overall],
        "by_kind": [asdict(c) for c in table.by_kind],
        "paired_vs_heuristic": [
            asdict(paired(traces, s, "heuristic"))
            for s in sorted({t.system for t in traces} - {"heuristic"})
        ],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return out
