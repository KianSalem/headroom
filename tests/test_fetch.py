"""Tests for the real-music corpus fetcher, none of which touch the network.

The fetcher's job is to turn a pinned remote archive into a train/test corpus
without ever letting unverified audio become a result. So what is tested here
is mostly refusal: a truncated transfer, a digest that does not match, an
archive whose layout is not what was expected, a missing ffmpeg. The one test
that decodes audio builds its own multi-stream file with ffmpeg and skips when
ffmpeg is absent, because a clean clone runs this suite without it.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from evals import fetch
from evals.corpus import scan_directory, split_from_track_id

from headroom.cli import build_parser

REPO_ROOT = Path(__file__).resolve().parents[1]


def _zip_with(names: list[str], payload: bytes = b"x") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name in names:
            zf.writestr(name, payload)
    return buffer.getvalue()


def _member_names() -> list[str]:
    return [
        "train/B Artist - Later.stem.mp4",
        "train/A Artist - Early.stem.mp4",
        "test/D Artist - Fourth.stem.mp4",
        "test/C Artist - Third.stem.mp4",
        "notes.txt",
        "extra/E Artist - Elsewhere.stem.mp4",
    ]


# --------------------------------------------------------------------------
# The pins themselves
# --------------------------------------------------------------------------


def test_every_archive_is_keyed_by_its_own_key() -> None:
    """A table where the key and the record disagree would fetch the wrong file."""
    for key, archive in fetch.ARCHIVES.items():
        assert archive.key == key


def test_every_archive_is_pinned_by_size_and_digest() -> None:
    """An unpinned corpus can change upstream and silently change every number."""
    for archive in fetch.ARCHIVES.values():
        assert archive.size_bytes > 0
        assert len(archive.md5) == 32
        assert int(archive.md5, 16) >= 0
        assert archive.url.startswith("https://")
        assert not archive.redistributable


def test_no_archive_claims_to_be_redistributable() -> None:
    """MUSDB18 is non-commercial with per-track terms.

    The flag is asserted rather than assumed because it is what justifies
    keeping the audio out of the repository and out of the HTML report.
    """
    assert all(not a.redistributable for a in fetch.ARCHIVES.values())
    assert all("non-commercial" in a.license for a in fetch.ARCHIVES.values())


def test_the_cli_offers_exactly_the_archives_the_fetcher_can_verify() -> None:
    """The advertised choices are read from the archive table, not restated."""
    parser = build_parser()
    args = parser.parse_args(["fetch-corpus", "--archive", "musdb18-7s"])
    assert args.archive in fetch.ARCHIVES
    with pytest.raises(SystemExit):
        parser.parse_args(["fetch-corpus", "--archive", "not-a-corpus"])


def test_the_default_destination_is_one_git_refuses_to_commit() -> None:
    """Corpus audio in a public repository is a licence violation, not a mess.

    Checked against .gitignore rather than trusting the default string, so
    changing the default to somewhere committable fails here.
    """
    args = build_parser().parse_args(["fetch-corpus"])
    top = Path(args.out).parts[0]
    ignored = {
        line.strip().rstrip("/*")
        for line in (REPO_ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith(("#", "!"))
    }
    assert top in ignored, f"{args.out} is not covered by .gitignore"


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Angels In Amplifiers - I'm Alright", "Angels_In_Amplifiers_-_I_m_Alright"),
        ("BKS - Bulldozer", "BKS_-_Bulldozer"),
        ("  ", "unknown"),
        ("!!!", "unknown"),
        ("a/b", "a_b"),
    ],
)
def test_slugify_keeps_names_readable_and_shell_safe(raw: str, expected: str) -> None:
    assert fetch.slugify(raw) == expected


def test_slugified_ids_survive_a_shell_and_a_filename() -> None:
    forbidden = set(r""" '"$`;|&<>()*?[]#~""")
    for name in ("Al James - Schoolboy Facination", "AM Contra - Heart Peripheral"):
        assert not (set(fetch.slugify(name)) & forbidden)


# --------------------------------------------------------------------------
# Planning which members to take
# --------------------------------------------------------------------------


def test_members_come_from_the_corpus_own_train_test_directories() -> None:
    with zipfile.ZipFile(io.BytesIO(_zip_with(_member_names()))) as zf:
        planned = fetch.plan_members(zf)
    splits = {split for split, _ in planned}
    assert splits == {"train", "test"}
    # notes.txt is not audio and extra/ is not a split, so neither is planned.
    assert all(name.endswith(fetch.MEMBER_SUFFIX) for _, name in planned)
    assert not any(name.startswith("extra/") for _, name in planned)


def test_planning_is_deterministic_and_takes_from_the_front() -> None:
    """Asking for two tracks twice has to give the same two tracks.

    Otherwise a re-run silently evaluates a different corpus than the one the
    committed manifest describes.
    """
    raw = _zip_with(_member_names())
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        first = fetch.plan_members(zf, limit={"train": 1, "test": 1})
        second = fetch.plan_members(zf, limit={"train": 1, "test": 1})
    assert first == second
    assert [name for _, name in first] == [
        "train/A Artist - Early.stem.mp4",
        "test/C Artist - Third.stem.mp4",
    ]


def test_a_limit_of_zero_takes_nothing_from_that_split() -> None:
    with zipfile.ZipFile(io.BytesIO(_zip_with(_member_names()))) as zf:
        planned = fetch.plan_members(zf, limit={"train": 0, "test": 2})
    assert {split for split, _ in planned} == {"test"}


def test_track_ids_are_slugged_without_their_split_prefix() -> None:
    with zipfile.ZipFile(io.BytesIO(_zip_with(_member_names()))) as zf:
        planned = fetch.plan_members(zf, limit={"train": 1, "test": 0})
        rows = list(fetch.iter_planned(zf, planned))
    assert rows == [("train", "train/A Artist - Early.stem.mp4", "A_Artist_-_Early")]


# --------------------------------------------------------------------------
# Downloading: everything about it is a refusal
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._stream = io.BytesIO(body)

    def read(self, size: int) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _pinned(body: bytes, **over: Any) -> fetch.RemoteArchive:
    fields: dict[str, Any] = {
        "key": "fake",
        "name": "Fake",
        "url": "https://example.invalid/fake.zip",
        "record_url": "https://example.invalid/record",
        "size_bytes": len(body),
        "md5": hashlib.md5(body).hexdigest(),
        "license": "non-commercial, per-track terms",
    }
    fields.update(over)
    return fetch.RemoteArchive(**fields)


def _serve(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda _url: _FakeResponse(body))


def test_a_verified_download_lands_and_reports_its_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"a zip would go here" * 100
    archive = _pinned(body)
    _serve(monkeypatch, body)
    path = fetch.download(archive, tmp_path)
    assert path.read_bytes() == body
    assert fetch.md5_of(path) == archive.md5


def test_a_truncated_transfer_is_deleted_rather_than_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short read must not leave something a later run would trust."""
    body = b"complete body"
    archive = _pinned(body, size_bytes=len(body) + 999)
    _serve(monkeypatch, body)
    with pytest.raises(fetch.FetchError, match="truncated"):
        fetch.download(archive, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_digest_mismatch_refuses_to_build_a_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"not what was pinned"
    archive = _pinned(body, md5="0" * 32)
    _serve(monkeypatch, body)
    with pytest.raises(fetch.FetchError, match="does not match the pinned"):
        fetch.download(archive, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_already_verified_archive_is_not_fetched_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"already here"
    archive = _pinned(body)
    (tmp_path / archive.filename).write_bytes(body)

    def _explode(_url: str) -> None:
        raise AssertionError("re-downloaded an archive that was already verified")

    monkeypatch.setattr(urllib.request, "urlopen", _explode)
    assert fetch.download(archive, tmp_path).read_bytes() == body


def test_a_stale_archive_is_replaced_rather_than_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file with the right name and the wrong contents is the dangerous case."""
    body = b"the real body"
    archive = _pinned(body)
    (tmp_path / archive.filename).write_bytes(b"stale contents of the wrong length")
    _serve(monkeypatch, body)
    assert fetch.download(archive, tmp_path).read_bytes() == body


def test_a_leftover_part_file_is_never_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zenodo ignores Range, so resuming appends a second body to the first.

    The result parses as a zip and fails on every member, which is a genuinely
    confusing way to lose an afternoon. A partial is discarded instead.
    """
    body = b"the real body"
    archive = _pinned(body)
    partial = tmp_path / (archive.filename + ".part")
    partial.write_bytes(b"half of a previous attempt")
    _serve(monkeypatch, body)
    assert fetch.download(archive, tmp_path).read_bytes() == body
    assert not partial.exists()


# --------------------------------------------------------------------------
# Fetch-level failures
# --------------------------------------------------------------------------


def test_an_unknown_archive_names_the_ones_that_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch, "require_ffmpeg", lambda: "/usr/bin/true")
    with pytest.raises(fetch.FetchError, match="known archives"):
        fetch.fetch("no-such-corpus")


def test_a_missing_ffmpeg_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ffmpeg is not a Python dependency, so a clean clone can lack it."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(fetch.FetchError) as caught:
        fetch.require_ffmpeg()
    message = str(caught.value)
    assert "brew install ffmpeg" in message
    assert "synth-corpus" in message


def test_an_unexpected_archive_layout_is_reported_not_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _zip_with(["something/else.txt"])
    archive = _pinned(body)
    monkeypatch.setattr(fetch, "ARCHIVES", {"fake": archive})
    monkeypatch.setattr(fetch, "require_ffmpeg", lambda: "/usr/bin/true")
    _serve(monkeypatch, body)
    with pytest.raises(fetch.FetchError, match="layout is not what this expects"):
        fetch.fetch("fake", tmp_path / "out", cache_dir=tmp_path / "cache")


# --------------------------------------------------------------------------
# The split, and what the manifest records about provenance
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("track_id", "expected"),
    [
        ("test/BKS_-_Bulldozer", "test"),
        ("train/ANiMAL_-_Clinic_A", "train"),
        ("synth_02", None),
        ("test", None),
        ("nested/test/thing", None),
    ],
)
def test_a_corpus_split_is_read_off_the_path_when_it_has_one(
    track_id: str, expected: str | None
) -> None:
    assert split_from_track_id(track_id) == expected


def test_a_canonical_layout_overrides_the_hash_split(tmp_path: Path) -> None:
    """MUSDB18's own split is what the separation literature reports against."""
    for split in ("train", "test"):
        directory = tmp_path / split
        directory.mkdir()
        for index in range(3):
            sf.write(
                directory / f"t{index}.wav",
                [[0.0, 0.0]] * 44100 * 11,
                44100,
                subtype="PCM_24",
            )
    manifest = scan_directory(tmp_path, corpus_name="fake", source_md5="abc")
    assert "canonical" in manifest.split_method
    assert manifest.source_md5 == "abc"
    assert {t.split for t in manifest.train()} == {"train"}
    assert len(manifest.train()) == 3
    assert len(manifest.test()) == 3


def test_a_flat_layout_still_uses_the_stable_hash_split(tmp_path: Path) -> None:
    for index in range(4):
        sf.write(
            tmp_path / f"t{index}.wav",
            [[0.0, 0.0]] * 44100 * 11,
            44100,
            subtype="PCM_24",
        )
    manifest = scan_directory(tmp_path, corpus_name="fake")
    assert "blake2b" in manifest.split_method


# --------------------------------------------------------------------------
# The one test that actually decodes audio
# --------------------------------------------------------------------------

_FFMPEG = shutil.which("ffmpeg")


@pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg is not installed")
def test_only_the_mixture_stream_is_extracted(tmp_path: Path) -> None:
    """A MUSDB18 stem file holds the mixture plus four stems, mixture first.

    Mastering operates on the mixture, so taking the wrong stream would
    silently evaluate the drums. The streams here are told apart by frequency
    rather than by level, because ffmpeg's own output level for a synthesised
    tone is not something this test should be asserting about.
    """
    assert _FFMPEG is not None
    source = tmp_path / "two_streams.mp4"
    subprocess.run(
        [
            _FFMPEG,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=200:duration=3:sample_rate=44100",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=5000:duration=3:sample_rate=44100",
            "-filter_complex",
            "[0:a]aformat=channel_layouts=stereo[a];[1:a]aformat=channel_layouts=stereo[b]",
            "-map",
            "[a]",
            "-map",
            "[b]",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    packed = tmp_path / "packed.zip"
    with zipfile.ZipFile(packed, "w") as zf:
        zf.write(source, "test/Some Artist - Some Title.stem.mp4")

    out = tmp_path / "out.wav"
    with zipfile.ZipFile(packed) as zf:
        fetch.extract_mixture(
            zf,
            "test/Some Artist - Some Title.stem.mp4",
            out,
            ffmpeg=_FFMPEG,
            seconds=2.0,
            start_fraction=0.0,
        )

    samples, rate = sf.read(out)
    assert rate == 44100
    assert samples.shape[1] == 2
    assert 1.5 < len(samples) / rate <= 2.05

    mono = samples.mean(axis=1)
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    dominant = float(np.fft.rfftfreq(len(mono), 1.0 / rate)[int(np.argmax(spectrum))])
    assert 150.0 < dominant < 260.0, (
        f"dominant tone {dominant:.0f} Hz -- that is the second stream, not the first"
    )


@pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg is not installed")
def test_a_broken_member_reports_ffmpeg_rather_than_writing_silence(
    tmp_path: Path,
) -> None:
    assert _FFMPEG is not None
    packed = tmp_path / "packed.zip"
    with zipfile.ZipFile(packed, "w") as zf:
        zf.writestr("test/Nope - Nope.stem.mp4", b"this is not an mp4")
    with zipfile.ZipFile(packed) as zf, pytest.raises(fetch.FetchError, match="ffmpeg"):
        fetch.extract_mixture(zf, "test/Nope - Nope.stem.mp4", tmp_path / "o.wav", ffmpeg=_FFMPEG)


# --------------------------------------------------------------------------
# Corpus character: the claim that synthetic material is easier, as a number
# --------------------------------------------------------------------------


def _write_tone(path: Path, seconds: float = 11.0, rate: int = 44100) -> None:
    t = np.arange(int(rate * seconds)) / rate
    tone = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    sf.write(path, np.column_stack([tone, tone]), rate, subtype="PCM_24")


def _stats_for(tmp_path: Path, name: str, blocks: list[float]) -> str:
    """Build a one-track corpus whose level follows ``blocks``, and measure it."""
    from evals.corpus import save_manifest

    from headroom.cli import main

    rate = 44100
    seconds_per_block = 4.0
    parts = []
    for gain in blocks:
        t = np.arange(int(rate * seconds_per_block)) / rate
        parts.append(gain * np.sin(2.0 * np.pi * 220.0 * t))
    tone = np.concatenate(parts)
    root = tmp_path / name
    root.mkdir()
    sf.write(root / "t.wav", np.column_stack([tone, tone]), rate, subtype="PCM_24")

    manifest = scan_directory(root, corpus_name=name, test_fraction=1.0)
    manifest_path = root / "manifest.json"
    save_manifest(manifest, manifest_path)
    assert main(["corpus-stats", "--manifest", str(manifest_path)]) == 0
    return name


def _reported_lra(out: str) -> float:
    line = next(line for line in out.splitlines() if line.strip().startswith("lra"))
    return float(line.split()[1])


def test_corpus_stats_separates_material_that_moves_from_material_that_does_not(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reason the real corpus exists is that synthetic audio does not move.

    Asserted as a comparison rather than against a threshold: a tone that
    starts abruptly gives loudness range about a unit to chew on from the
    K-weighting filter settling, so the absolute floor is not zero and pinning
    a number here would be pinning an artefact.
    """
    _stats_for(tmp_path, "held", [0.2, 0.2, 0.2])
    held = _reported_lra(capsys.readouterr().out)

    _stats_for(tmp_path, "moving", [0.02, 0.4, 0.02])
    moving = _reported_lra(capsys.readouterr().out)

    assert moving > held + 5.0, f"held {held}, moving {moving}"


def test_corpus_stats_refuses_a_split_it_has_no_tracks_for(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from evals.corpus import save_manifest

    from headroom.cli import main

    _write_tone(tmp_path / "tone.wav")
    manifest = scan_directory(tmp_path, corpus_name="held tones", test_fraction=1.0)
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)

    assert main(["corpus-stats", "--manifest", str(manifest_path), "--split", "train"]) == 2
    assert "no train tracks" in capsys.readouterr().err
