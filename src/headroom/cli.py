"""Command line interface.

``master`` is the product surface: point it at a file and either a reference
track or a delivery preset, and it runs the closed loop and writes the result.
It defaults to a system that needs no credential, so the demo works on a fresh
clone; ``--system agent`` swaps in the model-backed specialists.

The rest is the measurement surface -- ``analyze``, ``compare``, ``render`` --
plus the evaluation commands, which build the results table from traces on
disk rather than from a live run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

import soundfile as sf
from evals.runner import AGENT_SYSTEMS, FREE_SYSTEMS

from headroom import __version__
from headroom.agent.client import DEFAULT_MODEL
from headroom.analysis.features import analyze
from headroom.audio import AudioBuffer, AudioError, load, save
from headroom.dsp.backends.pedalboard import render_chain
from headroom.dsp.chain import Chain
from headroom.target.distance import distance
from headroom.target.profile import PRESETS, TargetProfile

if TYPE_CHECKING:
    from evals.corpus import CorpusManifest


#: soundfile ships no type information, so its error class is named once here
#: with a type mypy can check an ``except`` clause against.
_UNREADABLE_AUDIO: type[Exception] = sf.LibsndfileError


class UsageError(ValueError):
    """A command line that cannot be acted on. Exits 2 with one sentence."""


class CorpusMissingError(RuntimeError):
    """A manifest names audio that is not on disk. Fetchable, not broken."""


def _cmd_analyze(args: argparse.Namespace) -> int:
    features = analyze(load(args.input))
    if args.json:
        sys.stdout.write(features.model_dump_json(indent=2) + "\n")
    else:
        sys.stdout.write(f"{Path(args.input).name}\n{features.summary()}\n")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    chain = Chain.model_validate_json(Path(args.chain).read_text())
    ordered, moves = chain.canonical()
    for move in moves:
        sys.stderr.write(
            f"repositioned {move.kind} {move.op_id}: {move.from_index} -> {move.to_index}\n"
        )
    sys.stdout.write(ordered.describe() + "\n")
    save(render_chain(load(args.input), ordered), args.output)
    sys.stdout.write(f"wrote {args.output}\n")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    current = analyze(load(args.input))
    if args.preset:
        target = TargetProfile.from_preset(args.preset)
    else:
        target = TargetProfile.from_features(analyze(load(args.reference)), label="reference")
    result = distance(current, target, norm=args.norm)
    sys.stdout.write(f"{target.describe()}\n{result.describe(args.top)}\n")
    return 0


def _cmd_presets(_: argparse.Namespace) -> int:
    for name, preset in PRESETS.items():
        sys.stdout.write(
            f"{name:14s} {preset.lufs_integrated:+6.1f} LUFS  "
            f"<= {preset.true_peak_dbtp:+.1f} dBTP   {preset.note}\n"
        )
    return 0


def _target_for(args: argparse.Namespace) -> TargetProfile:
    """Reference match, delivery preset, or a reference at a preset's level.

    The third is the request a person actually has: match that record, but hand
    me something Spotify will not turn down.
    """
    if args.reference:
        target = TargetProfile.from_features(
            analyze(load(args.reference)), label=Path(args.reference).stem
        )
        return target.with_loudness(args.target) if args.target else target
    return TargetProfile.from_preset(args.target)


def _brief_target(args: argparse.Namespace, source: AudioBuffer) -> TargetProfile:
    """Translate a natural-language brief into a target, then print it.

    Printed because this is the one step in the system a person cannot check by
    listening: if the translation read "brighter" as "louder", the master will
    be wrong in a way that sounds deliberate.
    """
    from headroom.agent.client import BriefTranslator, ModelClient, ModelConfig
    from headroom.agent.factory import cassette_for

    config = ModelConfig(model=args.model, effort=args.effort)
    translator = BriefTranslator(
        client=ModelClient(config=config, cassette=cassette_for(config.model))
    )
    spec, usage = translator(args.brief, analyze(source))
    sys.stdout.write(spec.describe() + "\n")
    if spec.rationale:
        sys.stdout.write(f"  reading: {spec.rationale}\n")
    sys.stdout.write(f"  translation cost {usage.input_tokens} in / {usage.output_tokens} out\n\n")
    if not spec.adjustments:
        raise SystemExit("the brief did not translate into anything measurable")
    profile = spec.apply_to(analyze(source), label=f"brief:{args.brief[:32]}")
    return profile.with_loudness(args.target) if args.target else profile


def _cmd_master(args: argparse.Namespace) -> int:
    from headroom.control.critic import CriticConfig
    from headroom.control.loop import run_loop

    if not (args.reference or args.target or args.brief):
        raise UsageError(
            "master needs something to aim at: --target <preset>, --reference <file>, "
            "or --brief '<what you want>'"
        )
    source = load(args.input)
    target = _brief_target(args, source) if args.brief else _target_for(args)
    config = CriticConfig(render_budget=args.budget)

    if args.system in AGENT_SYSTEMS:
        from headroom.agent.factory import build_system

        agent = build_system(args.system, model=args.model, effort=args.effort)
        trace = agent.annotate(
            run_loop(
                args.system,
                source,
                target,
                agent.propose,
                config=config,
                model=args.model if args.system == "agent" else "",
                effort=args.effort,
            )
        )
    else:
        from evals.runner import make_proposer

        trace = run_loop(args.system, source, target, make_proposer(args.system, 0), config=config)

    out = sys.stdout
    out.write(f"{target.describe()}\n")
    out.write(f"system {trace.system}" + (f" ({trace.model})" if trace.model else "") + "\n\n")
    for step in trace.steps:
        marker = "*" if step.action else " "
        out.write(
            f"{marker} {step.index:2d} {step.distance_score:7.4f} "
            f"{step.n_out_of_tolerance:2d} out  {step.action or step.verdict:26s} "
            f"{step.note[:80]}\n"
        )
    out.write(
        f"\ndistance {trace.initial_distance:.4f} -> {trace.final_distance:.4f}  "
        f"recovery {trace.recovery_ratio:+.3f}  "
        f"{'converged' if trace.converged else f'stopped: {trace.abort_reason}'}\n"
    )
    out.write(f"{trace.n_renders} renders, {trace.wall_time_s:.1f}s")
    if trace.total_cost_usd:
        out.write(
            f", ${trace.total_cost_usd:.4f} "
            f"({trace.total_input_tokens} in / {trace.total_output_tokens} out"
            f"{', replayed' if trace.replayed_from_cassette else ''})"
        )
    out.write("\n\n" + trace.final_chain.describe() + "\n")

    save(render_chain(source, trace.final_chain), args.output)
    out.write(f"wrote {args.output}\n")
    if args.chain:
        Path(args.chain).write_text(trace.final_chain.model_dump_json(indent=2) + "\n")
        out.write(f"wrote {args.chain}\n")
    if args.trace:
        Path(args.trace).write_text(trace.model_dump_json(indent=2) + "\n")
        out.write(f"wrote {args.trace}\n")
    return 0


def _require_corpus(manifest_path: str) -> CorpusManifest:
    """Load a manifest, or explain that its audio has to be fetched first."""
    from evals.corpus import load_manifest, missing_audio

    manifest = load_manifest(manifest_path)
    missing = missing_audio(manifest)
    if missing:
        names = ", ".join(t.track_id for t in missing[:3])
        more = f" (and {len(missing) - 3} more)" if len(missing) > 3 else ""
        raise CorpusMissingError(
            f"{manifest_path} names {len(missing)} of {len(manifest.tracks)} tracks "
            f"that are not on disk: {names}{more}.\n"
            f"This corpus is fetched rather than committed -- MUSDB18 is "
            f"non-commercial with per-track terms. Run:\n"
            f"  headroom fetch-corpus --archive musdb18\n"
            f"or 'headroom synth-corpus' for the offline synthetic corpus."
        )
    return manifest


def _cmd_eval(args: argparse.Namespace) -> int:
    from evals.runner import AgentOptions, run_matrix

    report = run_matrix(
        _require_corpus(args.manifest),
        split=args.split,
        seeds=tuple(range(args.seeds)),
        systems=tuple(args.systems),
        out_dir=args.out,
        clip_seconds=args.clip_seconds,
        agent_options=AgentOptions(
            model=args.model, effort=args.effort, cassette_mode=args.cassette_mode
        ),
    )
    sys.stdout.write(report.summary() + "\n")
    if args.check_against:
        # The reproduction claim as a command: the same cells, run again, land
        # on the same recovery as the traces that were published.
        from evals.runner import load_traces, trace_filename

        committed = {trace_filename(t): t for t in load_traces(args.check_against)}
        compared = [
            (t, committed[trace_filename(t)])
            for t in report.traces
            if trace_filename(t) in committed
        ]
        mismatched = [t for t, c in compared if abs(t.recovery_ratio - c.recovery_ratio) > 1e-9]
        sys.stdout.write(
            f"recovery mismatches against {args.check_against}: "
            f"{len(mismatched)} of {len(compared)} matched traces\n"
        )
        for t in mismatched:
            sys.stdout.write(f"  {trace_filename(t)}\n")
        return 1 if mismatched or not compared else 0
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from evals.report import write_json, write_markdown
    from evals.runner import load_traces

    traces = load_traces(args.traces)
    if not traces:
        sys.stderr.write(f"no traces in {args.traces}\n")
        return 1
    write_markdown(traces, args.markdown)
    write_json(traces, args.json_out)
    sys.stdout.write(f"{len(traces)} traces -> {args.markdown}, {args.json_out}\n")
    if args.html:
        from evals.corpus import load_manifest
        from evals.html_report import SourceMismatchError, make_showcase
        from evals.html_report import build as build_html
        from evals.runner import clip

        showcases = []
        manifest = (
            load_manifest(args.manifest) if args.manifest and Path(args.manifest).exists() else None
        )
        if manifest is not None and not manifest.redistributable:
            # results/ is committed. Writing a non-commercial corpus's audio
            # there is a licence violation one `git add` away, so the report
            # gets tables and no players unless the manifest says otherwise.
            sys.stderr.write(
                f"{manifest.corpus_name}: not marked redistributable; the report "
                "gets tables only, no listening examples\n"
            )
        elif manifest is not None:
            tracks = {t.track_id: t for t in manifest.tracks}
            # One showcase per degradation kind, so a listener hears each
            # failure mode once rather than the same track nine times.
            seen: set[str] = set()
            wanted = set(args.showcase) if args.showcase else None
            for trace in traces:
                if trace.degradation_kind in seen or trace.track_id not in tracks:
                    continue
                if wanted is not None and trace.degradation_kind not in wanted:
                    continue
                source = clip(tracks[trace.track_id].load(), args.clip_seconds)
                try:
                    showcase = make_showcase(
                        trace.track_id,
                        trace.degradation_kind,
                        trace.degradation_seed,
                        traces,
                        source=source,
                    )
                except SourceMismatchError as exc:
                    sys.stderr.write(f"skipping listening example: {exc}\n")
                    continue
                seen.add(trace.degradation_kind)
                showcases.append(showcase)
        build_html(traces, args.html, showcases)
        sys.stdout.write(f"wrote {args.html} with {len(showcases)} listening examples\n")
    return 0


def _cmd_brief_eval(args: argparse.Namespace) -> int:
    """Score brief translation on one track. Deterministic, no LLM judge."""
    from evals import briefs
    from evals.runner import clip

    from headroom.agent.client import BriefTranslator, ModelClient, ModelConfig
    from headroom.agent.factory import build_system, cassette_for
    from headroom.control.critic import CriticConfig
    from headroom.dsp.backends.pedalboard import clear_cache

    source = clip(load(args.input), args.clip_seconds)
    model_config = ModelConfig(model=args.model, effort=args.effort)
    translate = BriefTranslator(
        client=ModelClient(config=model_config, cassette=cassette_for(model_config.model))
    )
    critic = CriticConfig(render_budget=args.budget)
    by_system: dict[str, list[briefs.BriefResult]] = {}
    for system in args.system:
        sys.stdout.write(f"\ncontroller: {system}\n")
        results: list[briefs.BriefResult] = []
        for case in briefs.CASES:
            # Per-case caches: across cases they would only hold audio nothing
            # revisits, and a fresh supervisor keeps working memory per brief.
            clear_cache()
            agent = build_system(system, model=args.model, effort=args.effort)
            result = briefs.run_case(
                case,
                source,
                translate,
                agent.propose,
                system=system,
                config=critic,
                track_id=Path(args.input).stem,
            )
            results.append(result)
            sys.stdout.write(
                f"{'PASS' if result.passed else 'fail'} "
                f"T{result.translation_score:.0%} E{result.execution_score:.0%} "
                f"C{result.collateral_score:.0%}  {case.label}\n"
            )
            for name, detail in result.detail.items():
                sys.stdout.write(f"       {name}: {detail}\n")
            if result.error:
                sys.stdout.write(f"       {result.error}\n")
        by_system[system] = results

    repaired = sum(1 for rs in by_system.values() for r in rs if r.target.repaired_json) // len(
        by_system
    )
    sections = [
        f"## Briefs on `{Path(args.input).name}`",
        "",
        f"Translation model `{args.model}`. {len(briefs.CASES)} briefs, "
        f"{translate.client.n_calls // max(len(by_system), 1)} translation calls costing "
        f"${translate.client.cost / max(len(by_system), 1):.4f} to record and "
        "nothing to replay" + (f"; {repaired} needed a JSON repair." if repaired else "."),
        "",
    ]
    if len(by_system) > 1:
        sections += [briefs.render_comparison(dict(by_system)), ""]
    for system, results in by_system.items():
        sections += [
            f"### Controller `{system}`",
            "",
            briefs.render_markdown(results, heading=""),
            "",
        ]
    body = "\n".join(sections)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    sys.stdout.write("\n" + body + f"wrote {out}\n")
    return 0


def _cmd_corpus(args: argparse.Namespace) -> int:
    from evals.corpus import save_manifest, scan_directory

    manifest = scan_directory(
        args.root,
        corpus_name=args.name or Path(args.root).resolve().name,
        test_fraction=args.test_fraction,
        license_note=args.license,
        min_duration_s=args.min_duration,
        redistributable=args.redistributable,
    )
    save_manifest(manifest, args.out)
    sys.stdout.write(f"{manifest.summary()}\nwrote {args.out}\n")
    for split in ("train", "test"):
        if manifest.tracks and not any(t.split == split for t in manifest.tracks):
            # The hash split is stable under insertion, which on a handful of
            # tracks can mean every one lands on the same side. Say so here,
            # because `headroom eval` on an empty split runs zero cells and
            # reports that as a result rather than as a mistake.
            sys.stderr.write(
                f"warning: no tracks in the {split!r} split; raise --test-fraction "
                "or lay the corpus out as train/ and test/ directories\n"
            )
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    from evals.synthetic import write_corpus

    files = write_corpus(args.out, seconds=args.seconds)
    sys.stdout.write(f"wrote {len(files)} synthetic tracks to {args.out}\n")
    sys.stdout.write(
        "These are for exercising the pipeline, not for results: they are "
        "stationary and much easier to repair than real music.\n"
    )
    return 0


#: What each reported dimension is evidence *of*. The point of this command is
#: that "the synthetic corpus is easier than real music" should be a
#: measurement rather than a caveat in a readme, so each row names the property
#: it stands for.
_CHARACTER: Final[tuple[tuple[str, str, str], ...]] = (
    ("lra", "LU", "loudness moves over time -- 0 means stationary"),
    ("plr", "dB", "peak-to-loudness headroom"),
    ("spectral_flatness", "", "noise-like at 1, tonal near 0"),
    ("spectral_centroid", "Hz", "where the energy sits"),
    ("spectral_tilt", "dB/oct", "slope of the spectrum"),
    ("correlation", "", "stereo field: +1 mono, 0 wide, negative unnatural"),
    ("mono_compat_db", "dB", "what a mono fold-down costs"),
    ("attack_time_p50", "ms", "how fast transients rise"),
    ("percussive_ratio", "", "share of energy in transients"),
)


def _cmd_corpus_stats(args: argparse.Namespace) -> int:
    from statistics import median

    from evals.runner import clip

    manifest = _require_corpus(args.manifest)
    tracks = manifest.test() if args.split == "test" else manifest.train()
    if not tracks:
        sys.stderr.write(f"manifest has no {args.split} tracks\n")
        return 2

    measured = [analyze(clip(t.load(), args.clip_seconds)) for t in tracks]
    # The *measured* duration, not the requested clip: a track shorter than the
    # clip length is used whole, and a header that claimed otherwise would
    # misdescribe every number under it.
    seconds = sorted(float(f.duration_s) for f in measured)
    span = (
        f"{seconds[0]:.1f} s"
        if seconds[0] == seconds[-1]
        else f"{seconds[0]:.1f}-{seconds[-1]:.1f} s"
    )
    sys.stdout.write(
        f"{manifest.corpus_name} -- {len(tracks)} {args.split} tracks, {span} each\n\n"
    )
    width = max(len(name) for name, _, _ in _CHARACTER)
    for name, unit, why in _CHARACTER:
        values = [float(getattr(f, name)) for f in measured]
        mid, low, high = median(values), min(values), max(values)
        sys.stdout.write(
            f"  {name:<{width}}  {mid:+9.3f} {unit:<7} [{low:+.3f} .. {high:+.3f}]  {why}\n"
        )
    return 0


def fetch_destinations(archive_key: str, out: str | None, manifest: str | None) -> tuple[str, str]:
    """Where an archive's audio and manifest go unless the caller says.

    Keyed on the archive: the full-length and the 7 s distributions hold the
    same track names, so a shared default directory would make the second
    fetch skip every track as "already present" and then write a manifest
    claiming the wrong archive's digest for the first archive's audio.
    """
    out_dir = out or f"audio/{archive_key}"
    manifest_path = manifest or f"results/corpus_manifest_{archive_key.replace('-', '_')}.json"
    return out_dir, manifest_path


def _cmd_fetch_corpus(args: argparse.Namespace) -> int:
    from evals.corpus import save_manifest, scan_directory
    from evals.fetch import ARCHIVES, Split, fetch

    def log(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    archive = ARCHIVES[args.archive]
    out_dir, manifest_path = fetch_destinations(args.archive, args.out, args.manifest)
    limit: dict[Split, int] = {"train": args.train_tracks, "test": args.tracks}
    result = fetch(
        args.archive,
        out_dir,
        cache_dir=args.cache_dir,
        limit=limit,
        seconds=args.seconds,
        keep_archive=args.keep_archive,
        log=log,
    )
    sys.stdout.write(result.summary() + "\n")

    manifest = scan_directory(
        out_dir,
        corpus_name=archive.name,
        license_note=archive.license,
        min_duration_s=args.min_duration,
        source_url=archive.record_url,
        source_md5=archive.md5,
    )
    save_manifest(manifest, manifest_path)
    sys.stdout.write(f"{manifest.summary()}\nwrote {manifest_path}\n")
    sys.stdout.write(
        "The audio is non-commercial and per-track licensed: it is not "
        "committed, redistributed, or embedded in the report.\n"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Imported here so the archive choices offered on the command line cannot
    # drift from the archives the fetcher actually knows how to verify.
    from evals.fetch import ARCHIVES as _ARCHIVE_KEYS

    parser = argparse.ArgumentParser(prog="headroom", description=__doc__)
    parser.add_argument("--version", action="version", version=f"headroom {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="measure a file's feature vector")
    p_analyze.add_argument("input")
    p_analyze.add_argument("--json", action="store_true", help="emit the full vector as JSON")
    p_analyze.set_defaults(func=_cmd_analyze)

    p_render = sub.add_parser("render", help="apply a chain JSON file to audio")
    p_render.add_argument("input")
    p_render.add_argument("chain")
    p_render.add_argument("output")
    p_render.set_defaults(func=_cmd_render)

    p_compare = sub.add_parser("compare", help="measure distance to a reference or preset")
    p_compare.add_argument("input")
    group = p_compare.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference", help="match this file's sonic profile")
    group.add_argument("--preset", choices=sorted(PRESETS), help="delivery loudness target")
    p_compare.add_argument("--norm", choices=("l1", "l2"), default="l2")
    p_compare.add_argument("--top", type=int, default=8)
    p_compare.set_defaults(func=_cmd_compare)

    sub.add_parser("presets", help="list delivery targets").set_defaults(func=_cmd_presets)

    p_synth = sub.add_parser(
        "synth-corpus", help="generate a synthetic corpus so the pipeline runs offline"
    )
    p_synth.add_argument("--out", default="audio/synthetic")
    p_synth.add_argument("--seconds", type=float, default=24.0)
    p_synth.set_defaults(func=_cmd_synth)

    p_master = sub.add_parser(
        "master", help="run the closed loop against a reference or a delivery preset"
    )
    p_master.add_argument("input")
    p_master.add_argument("output")
    p_master.add_argument("--reference", help="match this file's sonic profile")
    p_master.add_argument(
        "--brief",
        help=(
            "a natural-language request, e.g. 'more space but keep the low end "
            "tight'. Translated into measurable offsets by a single bounded model "
            "call; the loop and the metric are unchanged."
        ),
    )
    p_master.add_argument(
        "--target",
        choices=sorted(PRESETS),
        help="delivery loudness target, alone or with --reference",
    )
    p_master.add_argument(
        "--system",
        default="agent-scaffold",
        choices=("heuristic", "agent-scaffold", "agent"),
        help="agent-scaffold is the full architecture with no model call, so it needs no key",
    )
    p_master.add_argument("--model", default=DEFAULT_MODEL)
    p_master.add_argument(
        "--effort", default="", help="output_config.effort, where the model takes it"
    )
    p_master.add_argument("--budget", type=int, default=14, help="render budget")
    p_master.add_argument("--chain", help="also write the chain as JSON")
    p_master.add_argument("--trace", help="also write the full run trace as JSON")
    p_master.set_defaults(func=_cmd_master)

    p_eval = sub.add_parser("eval", help="run the evaluation matrix and write traces")
    p_eval.add_argument("--manifest", default="results/corpus_manifest.json")
    p_eval.add_argument("--split", default="test", choices=("train", "test"))
    p_eval.add_argument("--seeds", type=int, default=3)
    p_eval.add_argument("--systems", nargs="+", default=list(FREE_SYSTEMS))
    p_eval.add_argument("--out", default="results/traces")
    p_eval.add_argument(
        "--check-against",
        metavar="TRACES_DIR",
        help="after the run, compare each cell's recovery with the trace of the same "
        "name in this directory and exit 1 on any mismatch; how a published row is "
        "shown to reproduce",
    )
    p_eval.add_argument("--clip-seconds", type=float, default=20.0)
    p_eval.add_argument("--model", default=DEFAULT_MODEL)
    p_eval.add_argument("--effort", default="")
    p_eval.add_argument(
        "--cassette-mode",
        default=None,
        choices=("auto", "replay", "record", "off"),
        help="replay costs nothing and fails on a prompt change",
    )
    p_eval.set_defaults(func=_cmd_eval)

    p_report = sub.add_parser("report", help="build the results table from traces on disk")
    p_report.add_argument("--traces", default="results/traces")
    p_report.add_argument("--markdown", default="results/RESULTS.md")
    p_report.add_argument("--json-out", default="results/results.json")
    p_report.add_argument("--html", default="", help="also build the listening report here")
    p_report.add_argument("--manifest", default="results/corpus_manifest.json")
    p_report.add_argument("--clip-seconds", type=float, default=20.0)
    p_report.add_argument(
        "--showcase",
        nargs="*",
        default=["spectral_tilt", "over_compress", "stereo_collapse"],
        help="degradations to render for listening; empty means every kind, which "
        "is a much heavier page",
    )
    p_report.set_defaults(func=_cmd_report)

    p_briefs = sub.add_parser(
        "brief-eval", help="score natural-language brief translation, no LLM judge"
    )
    p_briefs.add_argument("input", help="a track to apply every brief to")
    p_briefs.add_argument("--out", default="results/BRIEFS.md")
    p_briefs.add_argument(
        "--system",
        nargs="+",
        default=["agent-scaffold", "agent"],
        choices=("agent-scaffold", "agent"),
        help="controllers to run the same translations through; two makes it a "
        "controlled comparison of who should close the loop",
    )
    p_briefs.add_argument("--model", default=DEFAULT_MODEL)
    p_briefs.add_argument("--effort", default="")
    p_briefs.add_argument("--budget", type=int, default=8)
    p_briefs.add_argument("--clip-seconds", type=float, default=20.0)
    p_briefs.set_defaults(func=_cmd_brief_eval)

    p_stats = sub.add_parser(
        "corpus-stats",
        help="measure a corpus's character, so 'synthetic is easier' is a number",
    )
    p_stats.add_argument("--manifest", default="results/corpus_manifest.json")
    p_stats.add_argument("--split", default="test", choices=("train", "test"))
    p_stats.add_argument("--clip-seconds", type=float, default=20.0)
    p_stats.set_defaults(func=_cmd_corpus_stats)

    p_fetch = sub.add_parser(
        "fetch-corpus",
        help="download a real-music corpus, decode the mixtures, write a manifest",
    )
    p_fetch.add_argument(
        "--archive",
        choices=sorted(_ARCHIVE_KEYS),
        default="musdb18",
        help="musdb18 is 4.68 GB of full-length mixes; musdb18-7s is 147 MB of "
        "7 s excerpts, too short to measure loudness range on",
    )
    p_fetch.add_argument(
        "--out",
        default=None,
        help="where the decoded mixtures go (default: audio/<archive>, so the "
        "full-length and 7 s distributions never mix)",
    )
    p_fetch.add_argument(
        "--manifest",
        default=None,
        help="manifest to write (default: results/corpus_manifest_<archive>.json)",
    )
    p_fetch.add_argument(
        "--tracks",
        type=int,
        default=6,
        help="test tracks to fetch; every headline number comes from these",
    )
    p_fetch.add_argument("--train-tracks", type=int, default=2)
    p_fetch.add_argument("--seconds", type=float, default=30.0)
    p_fetch.add_argument("--min-duration", type=float, default=10.0)
    p_fetch.add_argument("--cache-dir", default="~/.cache/headroom/archives")
    p_fetch.add_argument(
        "--keep-archive",
        action="store_true",
        help="keep the downloaded archive; it is several times the size of the "
        "audio kept from it, so by default it is deleted after extraction",
    )
    p_fetch.set_defaults(func=_cmd_fetch_corpus)

    p_corpus = sub.add_parser("corpus", help="build a train/test manifest from a directory")
    p_corpus.add_argument("root")
    p_corpus.add_argument("--out", default="results/corpus_manifest.json")
    p_corpus.add_argument("--test-fraction", type=float, default=0.35)
    p_corpus.add_argument(
        "--name",
        default=None,
        help="corpus name recorded in the manifest (default: the directory's name); "
        "a committed manifest that names the wrong corpus is worse than no manifest",
    )
    p_corpus.add_argument(
        "--license",
        default="unspecified",
        help="licence note recorded in the manifest; fetch-corpus fills this in "
        "from the archive it verified, this command cannot guess it",
    )
    p_corpus.add_argument("--min-duration", type=float, default=10.0)
    p_corpus.add_argument(
        "--redistributable",
        action="store_true",
        help="the audio may be published: lets `report --html` embed it. Off by "
        "default; only say it for audio you generated or hold the rights to",
    )
    p_corpus.set_defaults(func=_cmd_corpus)

    return parser


def main(argv: list[str] | None = None) -> int:
    from evals.fetch import FetchError

    from headroom.agent.cassette import CassetteMissError
    from headroom.agent.client import ModelError

    args = build_parser().parse_args(argv)
    try:
        result: int = args.func(args)
    except CassetteMissError as exc:
        # Replay mode is the default in CI and the recommended way to reproduce
        # a published number, so a miss is a normal outcome of asking for
        # something nobody recorded -- not a bug worth a stack trace.
        sys.stderr.write(f"no recording for this request.\n{exc.args[0]}\n")
        return 2
    except ModelError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except CorpusMissingError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except FetchError as exc:
        # A missing ffmpeg or a digest mismatch is a fact about the machine or
        # the network, not a defect worth a traceback.
        sys.stderr.write(f"{exc}\n")
        return 2
    except UsageError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except (FileNotFoundError, AudioError, _UNREADABLE_AUDIO) as exc:
        # A path that does not exist or is not audio is the user's to fix.
        sys.stderr.write(f"{exc}\n")
        return 2
    return result


if __name__ == "__main__":
    raise SystemExit(main())
