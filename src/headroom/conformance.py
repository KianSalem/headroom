"""Cross-implementation conformance for the scored vector.

The metric is the product, so any *second* implementation of it has to be held
against the first by measurement rather than by reading the diff. There are
three of those in this project's future and one in its past: a port of a
primitive to another language, a rewrite of an analyzer stage, a fix to a
measurement defect, and -- the reason this module exists -- the v1.1 pass that
corrects two of them. In every case the question is the same: which of the 28
dimensions moved, on which file, and by how much.

Agreement is reported in **tolerance units** rather than native ones. Every
scored dimension already carries a perceptually-motivated tolerance, so
``0.01 tol`` means the same thing for an integrated-loudness reading in LUFS
and a correlation in Fisher-z, and a single threshold can gate all 28 without
a per-feature table of what counts as close. A drift of 1.0 tol is, by
construction, exactly as audible as the smallest error the metric bothers to
score.

Each entry carries the audio's content hash, and a comparison refuses to
report drift for any file whose two hashes disagree. Without that, "the port
reads this track differently" and "the port decoded this track differently"
are the same number.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import median
from typing import Final

from pydantic import BaseModel, ConfigDict

from headroom import __version__
from headroom.analysis.features import analyze
from headroom.audio import load
from headroom.target.distance import SCORED, to_scored

#: Bumped when the meaning of a field changes, so an old dump cannot be
#: silently compared against a new one.
SCHEMA_VERSION: Final[int] = 1

AUDIO_SUFFIXES: Final[tuple[str, ...]] = (".wav", ".flac", ".aiff", ".aif")


class FileVector(BaseModel):
    """One file's scored vector, keyed to the bytes it was measured from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_hash: str
    scored: dict[str, float]


class VectorDump(BaseModel):
    """What an implementation says the corpus measures.

    ``implementation`` is free text naming what produced the dump, so a port
    can identify itself and a report can say which two things disagreed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    implementation: str
    files: dict[str, FileVector]


class FeatureDrift(BaseModel):
    """How far one dimension moved across the whole corpus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    family: str
    unit: str
    tolerance: float
    #: Worst absolute disagreement, in tolerance units.
    max_tol: float
    median_tol: float
    #: The same worst case in the dimension's own unit, for a reader who wants
    #: to know whether 3 tol is 0.3 dB or 30.
    max_native: float
    worst_file: str

    def describe(self) -> str:
        return (
            f"{self.name:<20s} {self.max_tol:8.4f} tol  "
            f"median {self.median_tol:8.4f}  "
            f"{self.max_native:+9.4f} {self.unit:<9s} {self.worst_file}"
        )


class DiffReport(BaseModel):
    """Per-dimension drift between two dumps, worst first."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    implementation_a: str
    implementation_b: str
    n_files: int
    drifts: list[FeatureDrift]
    #: Files present in both dumps whose audio hashes disagree. These are
    #: excluded from every number above: they are not a measurement
    #: disagreement, they are a different input.
    hash_mismatches: list[str]
    only_in_a: list[str]
    only_in_b: list[str]

    @property
    def max_tol(self) -> float:
        return max((d.max_tol for d in self.drifts), default=0.0)

    @property
    def comparable(self) -> bool:
        """True when the two dumps covered the same files, byte for byte."""
        return not (self.hash_mismatches or self.only_in_a or self.only_in_b)

    def moved(self, threshold: float) -> list[FeatureDrift]:
        return [d for d in self.drifts if d.max_tol > threshold]

    def describe(self, top: int = 12, threshold: float = 0.0) -> str:
        head = (
            f"{self.implementation_a}  ->  {self.implementation_b}\n"
            f"{self.n_files} files compared, {len(self.drifts)} dimensions, "
            f"{len(self.moved(threshold))} moved by more than {threshold:g} tol"
        )
        lines = [d.describe() for d in self.drifts[:top]]
        tail: list[str] = []
        if self.hash_mismatches:
            tail.append(
                f"EXCLUDED {len(self.hash_mismatches)} file(s) whose audio hash differs: "
                + ", ".join(self.hash_mismatches[:4])
            )
        if self.only_in_a:
            tail.append(f"only in A: {len(self.only_in_a)} file(s)")
        if self.only_in_b:
            tail.append(f"only in B: {len(self.only_in_b)} file(s)")
        return "\n".join([head, *lines, *tail])


def audio_files(paths: Iterable[str | Path]) -> list[Path]:
    """Every audio file under ``paths``, sorted, directories walked.

    Sorted so two dumps of the same corpus list their files in the same order
    whatever the filesystem feels like doing.
    """
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(
                q
                for q in sorted(p.rglob("*"))
                if q.is_file() and q.suffix.lower() in AUDIO_SUFFIXES
            )
        elif p.is_file():
            found.append(p)
    return sorted(set(found))


def key_for(path: Path) -> str:
    """The name a file goes under in a dump.

    Relative to the working directory and POSIX-separated, so a dump taken
    from the repository root on macOS and one taken on Linux -- or one taken
    by a port -- agree on what to call the same track.
    """
    return Path(os.path.relpath(path, Path.cwd())).as_posix()


def dump(paths: Sequence[str | Path], implementation: str | None = None) -> VectorDump:
    """Measure every file under ``paths`` and record the scored vector."""
    files: dict[str, FileVector] = {}
    for path in audio_files(paths):
        features = analyze(load(str(path)))
        files[key_for(path)] = FileVector(
            source_hash=features.source_hash,
            scored=to_scored(features),
        )
    return VectorDump(
        implementation=implementation or f"headroom {__version__}",
        files=files,
    )


def diff(a: VectorDump, b: VectorDump) -> DiffReport:
    """Per-dimension drift between two dumps of the same corpus."""
    if a.schema_version != b.schema_version:
        raise ValueError(
            f"schema versions differ ({a.schema_version} vs {b.schema_version}); "
            "the dumps do not mean the same thing"
        )

    shared = sorted(set(a.files) & set(b.files))
    mismatched = [k for k in shared if a.files[k].source_hash != b.files[k].source_hash]
    usable = [k for k in shared if k not in set(mismatched)]

    drifts: list[FeatureDrift] = []
    for spec in SCORED:
        per_file: list[tuple[float, float, str]] = []
        for k in usable:
            va, vb = a.files[k].scored.get(spec.name), b.files[k].scored.get(spec.name)
            if va is None or vb is None:
                continue
            native = vb - va
            per_file.append((abs(native) / spec.tolerance, native, k))
        if not per_file:
            continue
        worst = max(per_file, key=lambda t: t[0])
        drifts.append(
            FeatureDrift(
                name=spec.name,
                family=spec.family,
                unit=spec.unit,
                tolerance=spec.tolerance,
                max_tol=worst[0],
                median_tol=median(t[0] for t in per_file),
                max_native=worst[1],
                worst_file=worst[2],
            )
        )

    drifts.sort(key=lambda d: -d.max_tol)
    return DiffReport(
        implementation_a=a.implementation,
        implementation_b=b.implementation,
        n_files=len(usable),
        drifts=drifts,
        hash_mismatches=mismatched,
        only_in_a=sorted(set(a.files) - set(b.files)),
        only_in_b=sorted(set(b.files) - set(a.files)),
    )


__all__ = [
    "SCHEMA_VERSION",
    "DiffReport",
    "FeatureDrift",
    "FileVector",
    "VectorDump",
    "audio_files",
    "diff",
    "dump",
    "key_for",
]
