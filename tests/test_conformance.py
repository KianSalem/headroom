from __future__ import annotations

from pathlib import Path

import pytest

from headroom import conformance
from headroom.audio import save
from headroom.target.distance import SCORED, SPEC_BY_NAME

from .conftest import sine, stereo, white


def _corpus(root: Path) -> Path:
    """Two files, one of them in a subdirectory, so the walk is exercised."""
    save(stereo(sine(440.0, -20.0, seconds=3.0)), root / "tone.wav")
    save(stereo(white(seconds=3.0, seed=1), white(seconds=3.0, seed=2)), root / "sub" / "noise.wav")
    return root


def _dump(**scored: float) -> conformance.VectorDump:
    """A one-file dump whose vector is zero except where overridden."""
    values = dict.fromkeys((s.name for s in SCORED), 0.0)
    values.update(scored)
    return conformance.VectorDump(
        implementation="test",
        files={"a.wav": conformance.FileVector(source_hash="hash-a", scored=values)},
    )


def test_a_dump_measures_every_scored_dimension_for_every_file(tmp_path: Path) -> None:
    result = conformance.dump([_corpus(tmp_path)])

    assert len(result.files) == 2
    for entry in result.files.values():
        assert set(entry.scored) == {s.name for s in SCORED}
        assert entry.source_hash


def test_a_dump_round_trips_through_json(tmp_path: Path) -> None:
    original = conformance.dump([_corpus(tmp_path)])
    restored = conformance.VectorDump.model_validate_json(original.model_dump_json())
    assert restored == original


def test_an_implementation_compared_with_itself_shows_no_drift(tmp_path: Path) -> None:
    """The property a conformance run is asserting, in its degenerate case."""
    result = conformance.dump([_corpus(tmp_path)])
    report = conformance.diff(result, result)

    assert report.comparable
    assert report.max_tol == 0.0
    assert len(report.drifts) == len(SCORED)
    assert report.moved(0.0) == []


def test_drift_is_reported_in_tolerance_units_not_native_ones() -> None:
    """Half a tolerance of integrated loudness is 0.25 LUFS, and the point of
    the unit is that the same 0.5 means the same thing for every dimension."""
    tol = SPEC_BY_NAME["lufs_integrated"].tolerance
    report = conformance.diff(_dump(), _dump(lufs_integrated=0.5 * tol))

    worst = report.drifts[0]
    assert worst.name == "lufs_integrated"
    assert worst.max_tol == pytest.approx(0.5)
    assert worst.max_native == pytest.approx(0.5 * tol)
    assert worst.worst_file == "a.wav"
    assert [d.max_tol for d in report.drifts[1:]] == [0.0] * (len(SCORED) - 1)


def test_a_file_whose_audio_differs_is_excluded_rather_than_called_drift() -> None:
    """Otherwise "the port reads this track differently" and "the port was
    handed a different track" are the same number."""
    a = _dump()
    b = _dump(lufs_integrated=99.0)
    other = conformance.VectorDump(
        implementation=b.implementation,
        files={
            "a.wav": conformance.FileVector(source_hash="hash-b", scored=b.files["a.wav"].scored)
        },
    )

    report = conformance.diff(a, other)

    assert report.hash_mismatches == ["a.wav"]
    assert not report.comparable
    assert report.n_files == 0
    assert report.max_tol == 0.0


def test_files_present_on_only_one_side_are_named() -> None:
    a = _dump()
    b = conformance.VectorDump(implementation="test", files={})

    report = conformance.diff(a, b)

    assert report.only_in_a == ["a.wav"]
    assert report.only_in_b == []
    assert not report.comparable


def test_dumps_of_different_schema_versions_refuse_to_compare() -> None:
    a = _dump()
    b = a.model_copy(update={"schema_version": a.schema_version + 1})

    with pytest.raises(ValueError, match="schema versions differ"):
        conformance.diff(a, b)


def test_the_walk_finds_audio_anywhere_under_a_directory_and_sorts_it(tmp_path: Path) -> None:
    _corpus(tmp_path)
    (tmp_path / "notes.txt").write_text("not audio")

    found = conformance.audio_files([tmp_path])

    assert [p.name for p in found] == ["noise.wav", "tone.wav"]
