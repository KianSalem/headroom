"""Tests for the evaluation pipeline: corpus, runner, aggregation, report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from evals import html_report, report, runner
from evals.corpus import assign_split, load_manifest, save_manifest, scan_directory

from headroom.audio import AudioBuffer, load
from headroom.control.critic import CriticConfig
from headroom.control.state import RunTrace

from .conftest import SR

FAST = CriticConfig(render_budget=6)


def _write_corpus(root: Path, n: int = 4, seconds: float = 12.0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rng = np.random.default_rng(i + 1)
        t = np.arange(int(SR * seconds)) / SR
        low = 0.18 * np.sin(2 * np.pi * (55 + 7 * i) * t)
        mid = 0.10 * np.sin(2 * np.pi * (330 + 20 * i) * t)
        top = 0.05 * rng.standard_normal(t.size)
        left, right = low + mid + top, low + mid + np.roll(top, 97)
        peak = max(float(np.abs(left).max()), float(np.abs(right).max()))
        sf.write(
            root / f"synth_{i:02d}.wav",
            np.stack([left, right], axis=1) / peak * 0.7,
            SR,
            subtype="PCM_24",
        )


def test_split_is_stable_under_insertion() -> None:
    """A seeded shuffle reassigns every track when one is added, silently
    moving tracks across the train/test boundary and invalidating any number
    already reported. Hashing the id cannot do that."""
    before = {f"track_{i}": assign_split(f"track_{i}") for i in range(50)}
    after = {f"track_{i}": assign_split(f"track_{i}") for i in range(80)}
    for track_id, split in before.items():
        assert after[track_id] == split


def test_split_respects_the_requested_fraction() -> None:
    ids = [f"t{i}" for i in range(400)]
    test_share = sum(assign_split(i, 0.35) == "test" for i in ids) / len(ids)
    assert 0.28 < test_share < 0.42


def test_manifest_roundtrips_and_separates_splits(tmp_path: Path) -> None:
    _write_corpus(tmp_path / "audio")
    manifest = scan_directory(tmp_path / "audio", corpus_name="unit", test_fraction=0.5)
    save_manifest(manifest, tmp_path / "m.json")
    restored = load_manifest(tmp_path / "m.json")

    assert restored == manifest
    assert len(manifest.tracks) == 4
    assert set(manifest.train()) | set(manifest.test()) == set(manifest.tracks)
    assert not set(manifest.train()) & set(manifest.test())


def test_scan_skips_files_too_short_to_measure(tmp_path: Path) -> None:
    """A two-second clip cannot support a 3 s short-term loudness window."""
    root = tmp_path / "audio"
    _write_corpus(root, n=1, seconds=6.0)
    sf.write(root / "tiny.wav", np.zeros((SR * 2, 2)), SR, subtype="PCM_24")
    manifest = scan_directory(root, min_duration_s=5.0)
    assert [t.track_id for t in manifest.tracks] == ["synth_00"]


def test_clip_takes_a_centred_window() -> None:
    buf = AudioBuffer(np.arange(SR * 10 * 2, dtype=np.float64).reshape(-1, 2) / 1e7, SR)
    clipped = runner.clip(buf, 4.0)
    assert clipped.n_frames == SR * 4
    assert clipped.samples[0, 0] > buf.samples[0, 0]


def test_run_cell_gives_every_system_identical_input(tmp_path: Path) -> None:
    """The comparison is paired, so a per-cell difference is only meaningful if
    the inputs were the same."""
    _write_corpus(tmp_path / "audio", n=1)
    source = runner.clip(load(tmp_path / "audio" / "synth_00.wav"), 6.0)
    traces, skipped = runner.run_cell(
        source, "synth_00", "level_offset", 0, ("null", "heuristic"), config=FAST
    )
    assert skipped is None
    assert len({t.source_hash for t in traces}) == 1
    assert len({t.initial_distance for t in traces}) == 1
    assert len({t.target_label for t in traces}) == 1


def test_run_cell_skips_a_degradation_that_barely_damages(tmp_path: Path) -> None:
    """recovery_ratio divides by the initial distance, so a near-zero
    denominator turns measurement noise into an apparently excellent score."""
    _write_corpus(tmp_path / "audio", n=1)
    source = runner.clip(load(tmp_path / "audio" / "synth_00.wav"), 6.0)
    traces, skipped = runner.run_cell(
        source,
        "synth_00",
        "level_offset",
        0,
        ("heuristic",),
        config=FAST,
        min_useful_distance=1e6,
    )

    assert traces == []
    assert skipped is not None
    assert "divide by near-zero" in skipped.reason


def test_optimizer_is_seeded_by_the_other_systems(tmp_path: Path) -> None:
    """What makes it a bound instead of a competitor."""
    _write_corpus(tmp_path / "audio", n=1)
    source = runner.clip(load(tmp_path / "audio" / "synth_00.wav"), 6.0)
    traces, _ = runner.run_cell(
        source,
        "synth_00",
        "band_shift",
        1,
        ("heuristic", "optimizer"),
        config=FAST,
        optimizer_budget=15,
    )
    by_system = {t.system: t for t in traces}
    assert by_system["optimizer"].final_distance <= by_system["heuristic"].final_distance + 1e-9


def test_traces_written_and_reloaded_exactly(tmp_path: Path) -> None:
    """The report is generated from traces on disk, never a live run, so a
    published number can always be recomputed from committed data."""
    _write_corpus(tmp_path / "audio", n=1)
    source = runner.clip(load(tmp_path / "audio" / "synth_00.wav"), 6.0)
    traces, _ = runner.run_cell(
        source, "synth_00", "level_offset", 0, ("null", "heuristic"), config=FAST
    )
    out = tmp_path / "traces"
    for trace in traces:
        runner.write_trace(trace, out)
    reloaded = runner.load_traces(out)
    assert sorted(t.system for t in reloaded) == sorted(t.system for t in traces)
    assert reloaded[0] in traces


def test_skipped_cells_are_recorded_not_dropped(tmp_path: Path) -> None:
    """Which degradations failed to damage which tracks is itself
    information."""
    cell = runner.SkippedCell("t", "band_shift", 0, 0.01, "too weak")
    path = runner.write_skipped([cell], tmp_path)
    assert json.loads(path.read_text())[0]["track_id"] == "t"


def _fake_trace(system: str, key: tuple[str, str, int], recovery: float) -> RunTrace:
    from headroom.dsp.chain import Chain

    track, kind, seed = key
    return RunTrace(
        system=system,
        track_id=track,
        degradation_kind=kind,
        degradation_seed=seed,
        source_hash="h",
        target_label="t",
        target_provenance="p",
        initial_distance=1.0,
        final_distance=1.0 - recovery,
        recovery_ratio=recovery,
        converged=recovery > 0.95,
        final_chain=Chain(),
        config_hash="cfg",
    )


def test_aggregate_reports_median_and_iqr_not_mean() -> None:
    """One catastrophic run must not be able to hide inside an average, and it
    must not be able to dominate one either."""
    traces = [_fake_trace("s", ("t", "k", i), r) for i, r in enumerate([0.9, 0.9, 0.9, 0.9, -50.0])]
    cell = report.aggregate(traces).overall[0]
    assert cell.recovery_median == pytest.approx(0.9)
    assert cell.regression_rate == pytest.approx(0.2)
    assert float(np.mean([t.recovery_ratio for t in traces])) < 0.0


def test_paired_comparison_uses_only_shared_cells() -> None:
    traces = [
        *[_fake_trace("a", ("t", "k", i), 0.9) for i in range(8)],
        *[_fake_trace("b", ("t", "k", i), 0.2) for i in range(6)],
    ]
    result = report.paired(traces, "a", "b")
    assert result.n == 6
    assert result.a_wins == 6
    assert result.median_difference == pytest.approx(0.7)
    assert result.p_value is not None and result.p_value < 0.05


def test_paired_declines_to_test_when_n_is_too_small() -> None:
    """Quoting a p-value from three observations would be theatre."""
    traces = [
        *[_fake_trace("a", ("t", "k", i), 0.9) for i in range(3)],
        *[_fake_trace("b", ("t", "k", i), 0.2) for i in range(3)],
    ]
    result = report.paired(traces, "a", "b")
    assert result.p_value is None
    assert "n<" in result.note


def test_report_flags_traces_from_different_metric_configs() -> None:
    """Two runs with different tolerances are not comparable, and the report
    must say so rather than quietly averaging across them."""
    a = _fake_trace("s", ("t", "k", 0), 0.9)
    b = _fake_trace("s", ("t", "k", 1), 0.9).model_copy(update={"config_hash": "other"})
    table = report.aggregate([a, b])
    assert not table.comparable
    assert "Not comparable" in report.render_markdown(table)


def test_markdown_report_renders_every_slice() -> None:
    traces = [
        _fake_trace(s, ("t", k, 0), 0.5)
        for s in ("heuristic", "null")
        for k in ("level_offset", "combo")
    ]
    text = report.render_markdown(report.aggregate(traces))
    assert "### Overall" in text
    assert "### level_offset" in text and "### combo" in text
    assert "heuristic" in text and "null" in text


def test_html_report_is_self_contained_and_resolves_its_audio(tmp_path: Path) -> None:
    """The page is served from a static directory and has to work offline, so
    no external scripts, fonts or stylesheets."""
    _write_corpus(tmp_path / "audio", n=1)
    source = runner.clip(load(tmp_path / "audio" / "synth_00.wav"), 6.0)
    traces, _ = runner.run_cell(
        source, "synth_00", "level_offset", 0, ("null", "heuristic"), config=FAST
    )
    showcase = html_report.make_showcase("synth_00", "level_offset", 0, traces, source=source)
    out = html_report.build(traces, tmp_path / "report", showcases=[showcase])

    page = out.read_text()
    assert "<title>" in page and "<style>" in page
    assert "http://" not in page and "https://" not in page
    assert "cdn" not in page.lower()

    refs = [s.split('"')[0] for s in page.split('src="audio/')[1:]]
    assert refs
    for ref in refs:
        assert (tmp_path / "report" / "audio" / ref).is_file()


def test_html_report_states_that_measurements_are_on_lossless_audio(
    tmp_path: Path,
) -> None:
    """Players are transcoded for transport. Without saying so, a reader would
    reasonably wonder whether the numbers describe the files they just heard."""
    trace = _fake_trace("heuristic", ("t", "k", 0), 0.9)
    out = html_report.build([trace], tmp_path / "report")
    page = out.read_text()
    assert "lossless render" in page
    assert "Measured is not perceived" in page
    assert "upper bound rather than a competitor" in page


def test_a_showcase_refuses_audio_the_traces_were_not_measured_on(tmp_path: Path) -> None:
    """A source clipped to a different length than the run used would be
    re-degraded and labelled with numbers measured on different audio."""
    root = tmp_path / "corpus"
    _write_corpus(root, n=1, seconds=12.0)
    manifest = scan_directory(root, corpus_name="probe", test_fraction=1.0)
    track = manifest.tracks[0]
    full = track.load()
    traces, _ = runner.run_cell(
        full, track.track_id, "level_offset", 0, ["null"], config=CriticConfig(render_budget=2)
    )
    trace = traces[0]
    shorter = runner.clip(full, 6.0)
    assert shorter.content_hash() != full.content_hash()
    with pytest.raises(html_report.SourceMismatchError, match="clip-seconds"):
        html_report.make_showcase(track.track_id, "level_offset", 0, [trace], source=shorter)
    ok = html_report.make_showcase(track.track_id, "level_offset", 0, [trace], source=full)
    assert ok.degraded.content_hash() == trace.source_hash


def test_report_audio_names_keep_their_dots(tmp_path: Path) -> None:
    """MUSDB18 has "M.E.R.C. Music - Knockout". with_suffix() would have
    written every player for that track to one file named test_M.E.R.C.wav."""
    buf = AudioBuffer(np.zeros((4800, 2)), SR)
    written = html_report.write_audio(buf, tmp_path / "test_M.E.R.C._Music__heuristic")
    assert written.name.startswith("test_M.E.R.C._Music__heuristic.")
    assert written.suffix in {".wav", ".mp3"}


def test_check_against_turns_reproduction_into_an_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The published-row claim is 'run it again, land on the same recovery'.
    That is a command with an exit code, not a sentence."""
    from headroom.cli import main

    root = tmp_path / "corpus"
    _write_corpus(root, n=1, seconds=12.0)
    manifest = scan_directory(root, corpus_name="probe", test_fraction=1.0)
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)
    common = ["eval", "--manifest", str(manifest_path), "--systems", "null", "--seeds", "1"]
    first, second = tmp_path / "first", tmp_path / "second"
    assert main([*common, "--out", str(first)]) == 0
    assert main([*common, "--out", str(second), "--check-against", str(first)]) == 0
    out = capsys.readouterr().out
    assert "recovery mismatches against" in out and " 0 of " in out

    # Tamper with one published trace and the same command refuses.
    victim = sorted(p for p in first.glob("*.json") if not p.name.startswith("_"))[0]
    doctored = json.loads(victim.read_text())
    doctored["recovery_ratio"] = 0.5
    victim.write_text(json.dumps(doctored))
    assert main([*common, "--out", str(tmp_path / "third"), "--check-against", str(first)]) == 1
    assert victim.name in capsys.readouterr().out


def test_the_run_summary_states_replay_status_and_spend() -> None:
    from headroom.control.state import RunTrace

    base = RunTrace.model_validate_json(
        sorted(Path("results/traces-musdb18-20s").glob("*__agent.json"))[0].read_text()
    )
    paid = base.model_copy(update={"replayed_from_cassette": False, "total_cost_usd": 0.25})
    free = base.model_copy(update={"replayed_from_cassette": True, "total_cost_usd": 0.75})
    text = runner.RunReport(traces=[paid, free]).summary()
    assert "1/2 model-backed cells fully replayed" in text
    assert "actual spend $0.2500 (recorded cost $1.0000)" in text
    assert "model-backed" not in runner.RunReport(traces=[]).summary()
