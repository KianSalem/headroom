"""Command line interface.

Only the deterministic surface exists so far: measuring audio, rendering a
chain, and comparing a render to a target. The closed-loop commands land with
the controller and the agent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from headroom import __version__
from headroom.analysis.features import analyze
from headroom.audio import load, save
from headroom.dsp.backends.pedalboard import render_chain
from headroom.dsp.chain import Chain
from headroom.target.distance import distance
from headroom.target.profile import PRESETS, TargetProfile


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


def _cmd_corpus(args: argparse.Namespace) -> int:
    from evals.corpus import save_manifest, scan_directory

    manifest = scan_directory(args.root, test_fraction=args.test_fraction)
    save_manifest(manifest, args.out)
    sys.stdout.write(f"{manifest.summary()}\nwrote {args.out}\n")
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


def build_parser() -> argparse.ArgumentParser:
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

    p_corpus = sub.add_parser("corpus", help="build a train/test manifest from a directory")
    p_corpus.add_argument("root")
    p_corpus.add_argument("--out", default="results/corpus_manifest.json")
    p_corpus.add_argument("--test-fraction", type=float, default=0.35)
    p_corpus.set_defaults(func=_cmd_corpus)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
