"""Corpus loading and the train/test split.

The evaluation paradigm is degrade-and-recover: take a well-produced file,
apply a seeded degradation, and measure how close the system gets back to the
original. The target is the original's own feature vector, so the source
material does not need to be a commercial master -- it needs to be
well-produced, legally shareable, and uncompressed enough that measurement is
not dominated by codec artifacts.

That is why this uses a public corpus rather than private masters: a stranger
can download the same files and reproduce every number in the results table.
See ``MUSDB18_HQ`` below for the licensing terms, which are not permissive
enough to redistribute -- so the corpus is downloaded, never vendored.

**The split is by hash of the track id, not by shuffle.** A seeded shuffle
reassigns every track when one is added, which would silently move tracks
across the train/test boundary and invalidate previously reported numbers.
Hashing is stable: adding a track never moves an existing one.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import soundfile as sf
from pydantic import BaseModel, ConfigDict

from headroom.audio import MIN_SAMPLE_RATE, AudioBuffer, load

Split = Literal["train", "test"]

#: Fraction of tracks assigned to test.
TEST_FRACTION: Final[float] = 0.35

AUDIO_SUFFIXES: Final[frozenset[str]] = frozenset({".wav", ".flac", ".aiff", ".aif"})


class CorpusSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    url: str
    license: str
    redistributable: bool
    note: str = ""


MUSDB18_HQ: Final[CorpusSource] = CorpusSource(
    name="MUSDB18-HQ",
    url="https://zenodo.org/record/3338373",
    license="mixed CC BY-NC-SA 4.0 / 3.0; academic use; per-track terms",
    redistributable=False,
    note=(
        "150 uncompressed stereo tracks with stems. Open access, but 22.66 GB. "
        "Not redistributed by this repository. For the same tracks at a size a "
        "laptop can hold, 'headroom fetch-corpus' pulls the 4.68 GB MUSDB18 or "
        "the 147 MB SiSEC18 excerpts instead; see evals/fetch.py."
    ),
)


class TrackRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    track_id: str
    path: str
    sample_rate: int
    duration_s: float
    channels: int
    split: Split
    license: str = ""
    source: str = ""

    def load(self) -> AudioBuffer:
        return load(self.path)


class CorpusManifest(BaseModel):
    """A frozen record of which tracks exist and which split each belongs to.

    Committed alongside results so a reader can verify that the reported test
    numbers came from tracks that were never used for tuning.
    """

    model_config = ConfigDict(frozen=True)

    corpus_name: str
    created_at: str
    test_fraction: float
    split_method: str = "blake2b(track_id) -- stable under insertion"
    #: Where the audio came from, and the digest that was actually verified on
    #: the way in. A results table is only as reproducible as its inputs are
    #: identifiable, and "MUSDB18" alone does not identify a distribution.
    source_url: str = ""
    source_md5: str = ""
    #: Whether audio derived from this corpus may be published -- embedded in
    #: the HTML report, committed under results/. Defaults to False because the
    #: real corpus is non-commercial with per-track terms, and the safe failure
    #: is a report with tables and no players, not a licence violation in git.
    redistributable: bool = False
    tracks: tuple[TrackRecord, ...] = ()

    def train(self) -> tuple[TrackRecord, ...]:
        return tuple(t for t in self.tracks if t.split == "train")

    def test(self) -> tuple[TrackRecord, ...]:
        """Every headline number comes from these, and nothing is ever tuned on
        them. Kept as a separate accessor so a tuning path cannot reach them by
        iterating ``tracks`` out of habit."""
        return tuple(t for t in self.tracks if t.split == "test")

    def summary(self) -> str:
        train, test = self.train(), self.test()
        total_s = sum(t.duration_s for t in self.tracks)
        return (
            f"{self.corpus_name}: {len(self.tracks)} tracks "
            f"({len(train)} train / {len(test)} test), "
            f"{total_s / 60.0:.1f} min total"
        )


def assign_split(track_id: str, test_fraction: float = TEST_FRACTION) -> Split:
    """Deterministic split from the track id alone.

    Stable under insertion: adding a track never reassigns an existing one, so
    a number reported last month still refers to the same test set.
    """
    digest = hashlib.blake2b(track_id.encode(), digest_size=8).digest()
    position = int.from_bytes(digest, "big") / float(1 << 64)
    return "test" if position < test_fraction else "train"


def iter_audio_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            yield path


#: Directory names that carry a corpus's own train/test split.
CANONICAL_SPLIT_DIRS: Final[frozenset[str]] = frozenset({"train", "test"})


def split_from_track_id(track_id: str) -> Split | None:
    """The split a corpus assigned itself, read off the path, or None.

    MUSDB18 ships its tracks under ``train/`` and ``test/``, and the source
    separation literature reports against that split. Adopting it costs
    nothing and makes these numbers comparable to that work -- whereas hashing
    would invent a third split nobody else uses.
    """
    head, _, rest = track_id.partition("/")
    if rest and head in CANONICAL_SPLIT_DIRS:
        return "train" if head == "train" else "test"
    return None


def scan_directory(
    root: str | Path,
    corpus_name: str = MUSDB18_HQ.name,
    test_fraction: float = TEST_FRACTION,
    license_note: str = MUSDB18_HQ.license,
    min_duration_s: float = 10.0,
    prefer_canonical_split: bool = True,
    source_url: str = "",
    source_md5: str = "",
    redistributable: bool = False,
) -> CorpusManifest:
    """Build a manifest by scanning a directory of audio files.

    Reads headers only -- no audio is decoded -- so scanning a large corpus is
    fast. Files below ``min_duration_s`` or under the minimum sample rate are
    skipped, because a two-second clip cannot support a 3 s short-term loudness
    window; so are files with more than two channels.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"corpus root not found: {root_path}")

    records: list[TrackRecord] = []
    canonical_seen = False
    for path in iter_audio_files(root_path):
        info = sf.info(str(path))
        if info.samplerate < MIN_SAMPLE_RATE or info.duration < min_duration_s:
            continue
        if info.channels > 2:
            continue
        track_id = path.relative_to(root_path).with_suffix("").as_posix()
        canonical = split_from_track_id(track_id) if prefer_canonical_split else None
        if canonical is not None:
            canonical_seen = True
        records.append(
            TrackRecord(
                track_id=track_id,
                path=str(path),
                sample_rate=int(info.samplerate),
                duration_s=float(info.duration),
                channels=int(info.channels),
                split=canonical or assign_split(track_id, test_fraction),
                license=license_note,
                source=corpus_name,
            )
        )

    return CorpusManifest(
        corpus_name=corpus_name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        test_fraction=test_fraction,
        split_method=(
            "corpus-canonical (train/ and test/ directories)"
            if canonical_seen
            else "blake2b(track_id) -- stable under insertion"
        ),
        source_url=source_url,
        source_md5=source_md5,
        redistributable=redistributable,
        tracks=tuple(records),
    )


def missing_audio(manifest: CorpusManifest) -> tuple[TrackRecord, ...]:
    """Tracks the manifest names that are not on disk.

    A committed manifest describes audio this repository deliberately does not
    contain: MUSDB18 is non-commercial with per-track terms, so the corpus is
    fetched, never vendored. That makes "manifest present, audio absent" the
    normal state of a fresh clone rather than an error, and it deserves a
    sentence telling the reader which command fixes it -- not a libsndfile
    stack trace from four frames deep.
    """
    return tuple(t for t in manifest.tracks if not Path(t.path).exists())


def save_manifest(manifest: CorpusManifest, path: str | Path) -> None:
    """Write the manifest with track paths stored *relative to it*.

    A manifest is committed alongside results so a reader can check that the
    reported test numbers came from tracks never used for tuning. Absolute
    paths would make that artefact carry one machine's directory layout and be
    useless anywhere else, so paths are relativized on the way out and resolved
    against the manifest's own location on the way back in. A corpus somewhere
    the relative path cannot reach -- a different drive -- stays absolute.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(manifest.model_dump_json())
    base = p.parent.resolve()
    for track in payload["tracks"]:
        # A corpus the relative path cannot reach -- a different Windows drive
        # -- keeps its absolute path rather than failing the save.
        with contextlib.suppress(ValueError):
            track["path"] = os.path.relpath(Path(track["path"]).resolve(), base)
    p.write_text(json.dumps(payload, indent=2) + "\n")


def load_manifest(path: str | Path) -> CorpusManifest:
    p = Path(path)
    payload = json.loads(p.read_text())
    base = p.parent.resolve()
    for track in payload["tracks"]:
        track["path"] = str(Path(base / track["path"]).resolve())
    return CorpusManifest.model_validate(payload)
