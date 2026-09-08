"""Fetching a real-music corpus without a 22 GB download.

The synthetic corpus (``headroom synth-corpus``) is stationary, and its
degradations are built from the same op vocabulary used to repair them. That
is the easiest possible case and it flatters every deterministic system, so
the honest headline number has to come from real music.

The obvious source is MUSDB18-HQ -- 150 uncompressed stereo mixes with stems,
the standard corpus for source separation. It is 22.66 GB, which is more disk
than a laptop reviewer is likely to hand over just to check somebody's
portfolio. So this module fetches one of two smaller distributions of the
*same* tracks:

``musdb18``
    4.68 GB. Full-length mixes, AAC-encoded stems in MP4. Clips of the same
    20 s length the synthetic run uses, so the two result tables are
    comparable.
``musdb18-7s``
    147 MB. The SiSEC18 7-second excerpts. Cheap enough to fetch on a whim,
    but a 6.8 s excerpt supports only about four 3 s short-term loudness
    windows, so ``lra`` -- and therefore the dynamics role -- is measured on
    thin evidence. Use it to prove the pipeline runs, not to quote numbers.

Three things this module is deliberately careful about:

**The archives are pinned by size and MD5**, taken from Zenodo's own API. A
corpus that silently changes upstream would silently change every number
reported against it, and the manifest records the digest that was actually
verified.

**Downloads never resume.** Zenodo ignores ``Range``, so a resumed transfer
appends a second full body to the partial one and produces an archive whose
central directory parses fine and whose members fail with "Bad magic number
for file header". That is a genuinely confusing failure -- it was hit while
writing this -- so a partial file is deleted and refetched rather than
resumed.

**Only the mixture is extracted.** A MUSDB18 stem file carries five audio
streams (mixture, drums, bass, other, vocals); mastering operates on the
mixture, which is stream 0. The stems carry no titles or handler names to
identify them by, so that ordering was checked rather than trusted: decoding
all five streams of ``test/Cristina Vane - So Easy`` and summing streams 1-4
reproduces stream 0 to within 0.018 peak absolute error, which is AAC coding
noise. Stream 0 is the linear mixture. The stems are decoded and thrown away,
which is why the download is several times larger than the audio this keeps.

The licence is non-commercial and per-track, so the audio is never committed,
never redistributed, and never embedded in the HTML report. What gets
committed is the manifest: track ids, the split each belongs to, and the
digest of the archive they came from.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

Split = Literal["train", "test"]

#: Stem index of the full mixture inside a MUSDB18 ``.stem.mp4``.
MIXTURE_STREAM: Final[int] = 0

#: How far into a track the extracted window starts, as a fraction of its
#: duration. The first 20 s of a song is frequently an intro -- sparse, quiet,
#: often mono-ish -- which is not what a mastering decision is made against.
#: A third of the way in usually lands in a full arrangement.
START_FRACTION: Final[float] = 1.0 / 3.0

#: Seconds of audio kept per track. Longer than the evaluation's own 20 s clip
#: so the runner has margin to clip from without hitting the end of the file.
KEEP_SECONDS: Final[float] = 30.0

MEMBER_SUFFIX: Final[str] = ".stem.mp4"

_UNSAFE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")


class RemoteArchive(BaseModel):
    """A pinned remote archive: URL plus the size and digest it must have."""

    model_config = ConfigDict(frozen=True)

    key: str
    name: str
    url: str
    record_url: str
    size_bytes: int
    md5: str
    license: str
    redistributable: bool = False
    note: str = ""

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1].split("?", 1)[0]


MUSDB18: Final[RemoteArchive] = RemoteArchive(
    key="musdb18",
    name="MUSDB18",
    url="https://zenodo.org/records/1117372/files/musdb18.zip?download=1",
    record_url="https://zenodo.org/record/1117372",
    size_bytes=4684228845,
    md5="af06762477334799bfc5abf237648207",
    license="non-commercial, per-track terms (Zenodo: other-nc)",
    note=(
        "150 full-length stereo mixes with stems, AAC in MP4. Open access -- no "
        "request needed, despite what the HQ record's reputation suggests."
    ),
)

MUSDB18_7S: Final[RemoteArchive] = RemoteArchive(
    key="musdb18-7s",
    name="MUSDB18 (SiSEC18 7s excerpts)",
    url="https://zenodo.org/records/3270814/files/MUSDB18-7-STEMS.zip?download=1",
    record_url="https://zenodo.org/record/3270814",
    size_bytes=147209385,
    md5="dc2bdfac5b46ce742bd60009866758a5",
    license="non-commercial, per-track terms (inherited from MUSDB18)",
    note=(
        "144 seven-second excerpts of the MUSDB18 tracks. Cheap, but too short "
        "to measure loudness range on; the record declares CC-BY-4.0 while the "
        "parent corpus is non-commercial, so the stricter terms are assumed."
    ),
)

ARCHIVES: Final[dict[str, RemoteArchive]] = {
    MUSDB18.key: MUSDB18,
    MUSDB18_7S.key: MUSDB18_7S,
}

DEFAULT_ARCHIVE: Final[str] = MUSDB18.key


class FetchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    archive: RemoteArchive
    root: str
    written: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    archive_kept: bool = False

    def summary(self) -> str:
        lines = [
            f"{self.archive.name}: wrote {len(self.written)} tracks to {self.root}",
            f"  verified md5 {self.archive.md5}",
        ]
        if self.skipped:
            lines.append(f"  skipped {len(self.skipped)} (already present)")
        if not self.archive_kept:
            lines.append("  archive deleted after extraction")
        return "\n".join(lines)


class FetchError(RuntimeError):
    """Raised for a missing tool, a failed digest, or a malformed archive."""


def slugify(name: str) -> str:
    """Filesystem- and shell-safe track name, still readable.

    MUSDB18 names tracks ``Artist - Title``, which is worth keeping in the
    results table, but a raw space in a track id reaches trace filenames and
    every shell command a reader might paste. Spaces become underscores and
    anything outside a conservative set is dropped.
    """
    return _UNSAFE.sub("_", name.replace(" ", "_")).strip("_") or "unknown"


def require_ffmpeg() -> str:
    """Locate ffmpeg, or explain how to get it.

    ffmpeg is an external binary rather than a Python dependency, so the clean
    clone that runs the test suite does not need it -- only fetching real music
    does.
    """
    found = shutil.which("ffmpeg")
    if found is None:
        raise FetchError(
            "ffmpeg not found on PATH, and MUSDB18 ships AAC streams in MP4 "
            "containers that need it. Install it (macOS: 'brew install ffmpeg', "
            "Debian/Ubuntu: 'apt install ffmpeg'), or run 'headroom synth-corpus' "
            "for the offline synthetic corpus instead."
        )
    return found


def md5_of(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.md5()  # Zenodo publishes md5, so md5 is what can be checked
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    archive: RemoteArchive,
    cache_dir: str | Path,
    *,
    log: Callable[[str], None] = lambda _: None,
) -> Path:
    """Fetch the archive into ``cache_dir``, verifying size and digest.

    A file already present and already matching is left alone, so re-running
    costs nothing. Anything else -- wrong size, wrong digest, half a transfer
    -- is deleted and fetched again from the start. See the module docstring
    for why resuming is not an option.
    """
    directory = Path(cache_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / archive.filename

    if target.exists():
        if target.stat().st_size == archive.size_bytes and md5_of(target) == archive.md5:
            log(f"{archive.filename}: already present and verified")
            return target
        log(f"{archive.filename}: present but does not match its pin -- refetching")
        target.unlink()

    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    expected_mb = archive.size_bytes / 1e6
    log(f"{archive.filename}: fetching {expected_mb:.0f} MB from {archive.record_url}")

    with urllib.request.urlopen(archive.url, timeout=120) as response, partial.open("wb") as out:
        seen = 0
        step = max(archive.size_bytes // 20, 1)
        next_mark = step
        for block in iter(lambda: response.read(1 << 20), b""):
            out.write(block)
            seen += len(block)
            if seen >= next_mark:
                log(f"  {seen / 1e6:.0f} / {expected_mb:.0f} MB")
                next_mark += step

    actual_size = partial.stat().st_size
    if actual_size != archive.size_bytes:
        partial.unlink(missing_ok=True)
        raise FetchError(
            f"{archive.filename}: expected {archive.size_bytes} bytes, got "
            f"{actual_size}. The transfer was truncated; nothing was kept."
        )
    actual_md5 = md5_of(partial)
    if actual_md5 != archive.md5:
        partial.unlink(missing_ok=True)
        raise FetchError(
            f"{archive.filename}: md5 {actual_md5} does not match the pinned "
            f"{archive.md5}. Either the transfer corrupted or the upstream "
            f"archive changed; refusing to build a corpus from it."
        )
    partial.replace(target)
    log(f"{archive.filename}: verified md5 {actual_md5}")
    return target


def plan_members(
    zf: zipfile.ZipFile,
    *,
    limit: dict[Split, int] | None = None,
) -> tuple[tuple[Split, str], ...]:
    """Which archive members to extract, in a deterministic order.

    The split comes from the archive's own ``train/`` and ``test/`` directories
    rather than from hashing the track id. MUSDB18 has a canonical split that
    the separation literature reports against, and adopting it costs nothing
    and makes these numbers comparable to that work.

    Members are sorted by name and taken from the front, so asking for six test
    tracks twice gives the same six tracks.
    """
    by_split: dict[Split, list[str]] = {"train": [], "test": []}
    for info in zf.infolist():
        if info.is_dir() or not info.filename.endswith(MEMBER_SUFFIX):
            continue
        head, _, _ = info.filename.partition("/")
        if head == "train":
            by_split["train"].append(info.filename)
        elif head == "test":
            by_split["test"].append(info.filename)

    splits: tuple[Split, ...] = ("train", "test")
    planned: list[tuple[Split, str]] = []
    for split in splits:
        names = sorted(by_split[split])
        cap = None if limit is None else limit.get(split)
        planned.extend((split, name) for name in names[:cap])
    return tuple(planned)


def _probe_duration(ffmpeg: str, path: Path) -> float:
    """Duration in seconds, via ffmpeg's own stderr.

    Uses ffmpeg rather than ffprobe so that a machine with only one of the two
    installed still works.
    """
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if match is None:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def extract_mixture(
    zf: zipfile.ZipFile,
    member: str,
    out_path: Path,
    *,
    ffmpeg: str,
    seconds: float = KEEP_SECONDS,
    start_fraction: float = START_FRACTION,
) -> None:
    """Decode one track's mixture stream into a wav clip.

    The member is written to a temporary file first because ffmpeg needs to
    seek within an MP4 container, which it cannot do on a pipe. The temporary
    file is the reason peak disk use is the archive plus one track, not the
    archive plus the whole corpus.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "track.stem.mp4"
        with zf.open(member) as src, staged.open("wb") as dst:
            shutil.copyfileobj(src, dst)

        duration = _probe_duration(ffmpeg, staged)
        start = max(0.0, (duration - seconds) * start_fraction) if duration else 0.0
        command = [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(staged),
            "-map",
            f"0:a:{MIXTURE_STREAM}",
            "-t",
            f"{seconds:.3f}",
            "-c:a",
            "pcm_s24le",
            "-f",
            "wav",
            str(part),
        ]
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not part.exists():
            # Nothing partial may survive: the next run treats any file at
            # out_path as a finished track, and the archive is gone by then.
            part.unlink(missing_ok=True)
            raise FetchError(f"ffmpeg failed on {member}: {proc.stderr.strip()[:400]}")
        part.replace(out_path)


def iter_planned(
    zf: zipfile.ZipFile, planned: Sequence[tuple[Split, str]]
) -> Iterator[tuple[Split, str, str]]:
    """Yield ``(split, member, track_id)`` for each planned member."""
    for split, member in planned:
        stem = member.rsplit("/", 1)[-1].removesuffix(MEMBER_SUFFIX)
        yield split, member, slugify(stem)


def fetch(
    archive_key: str = DEFAULT_ARCHIVE,
    out_dir: str | Path = "corpus/musdb18",
    *,
    cache_dir: str | Path = "~/.cache/headroom/archives",
    limit: dict[Split, int] | None = None,
    seconds: float = KEEP_SECONDS,
    start_fraction: float = START_FRACTION,
    keep_archive: bool = False,
    log: Callable[[str], None] = lambda _: None,
) -> FetchResult:
    """Download, verify, and decode a real-music corpus into ``out_dir``.

    ``keep_archive`` defaults to false: the archive is several times the size
    of the audio kept from it, and on the machine this was written for that was
    the difference between comfortable and full.
    """
    archive = ARCHIVES.get(archive_key)
    if archive is None:
        known = ", ".join(sorted(ARCHIVES))
        raise FetchError(f"unknown archive {archive_key!r}; known archives: {known}")

    ffmpeg = require_ffmpeg()
    zip_path = download(archive, cache_dir, log=log)
    root = Path(out_dir).expanduser()

    written: list[str] = []
    skipped: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        planned = plan_members(zf, limit=limit)
        if not planned:
            raise FetchError(
                f"{archive.filename} contained no '{MEMBER_SUFFIX}' members under "
                "train/ or test/; the archive layout is not what this expects."
            )
        log(f"extracting {len(planned)} mixtures (up to {seconds:.0f} s each)")
        for index, (split, member, track_id) in enumerate(iter_planned(zf, planned), 1):
            out_path = root / split / f"{track_id}.wav"
            if out_path.exists():
                skipped.append(track_id)
                continue
            extract_mixture(
                zf,
                member,
                out_path,
                ffmpeg=ffmpeg,
                seconds=seconds,
                start_fraction=start_fraction,
            )
            written.append(track_id)
            log(f"  [{index}/{len(planned)}] {split}/{track_id}")

    if not keep_archive:
        zip_path.unlink(missing_ok=True)

    return FetchResult(
        archive=archive,
        root=str(root),
        written=tuple(written),
        skipped=tuple(skipped),
        archive_kept=keep_archive,
    )
