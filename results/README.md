# What is in this directory

Generated output. Nothing here is written by hand, and everything here is
reproducible from the corpus manifests plus `headroom eval` and
`headroom report`. The argument these tables support is in
[docs/ANALYSIS.md](../docs/ANALYSIS.md); this file only says which file is
which, because the most guessable filename in the directory is not the
headline.

## Start here

| you want | read |
|---|---|
| **the headline** — real music, 20 s clips, the condition the README quotes | [`RESULTS-musdb18-20s.md`](RESULTS-musdb18-20s.md) |
| the same comparison at 6.8 s, which is what exposed the model's instability | [`RESULTS-musdb18-7s.md`](RESULTS-musdb18-7s.md) |
| the synthetic controls, both clip lengths | [`RESULTS-synth-control-20s.md`](RESULTS-synth-control-20s.md), [`RESULTS-synth-control.md`](RESULTS-synth-control.md) |
| natural-language brief translation, graded with no LLM judge | [`BRIEFS.md`](BRIEFS.md) |

`RESULTS.md` and `results.json` are **not** the headline. They are the default
output path of `headroom report`, so a fresh run lands there, and they
currently hold the superseded original synthetic run — the one whose `agent`
row does not replay from cassettes. Every other table is named for the
condition it reports.

## The four conditions

Two corpora by two clip lengths, 54 paired cells each, one seed, a 14-render
budget for every system. Running the comparison in all four is what tells you
whether the architecture's advantage is about the material or about the
sample size and the analysis window.

| file | corpus | clips | traces |
|---|---|---|---|
| `RESULTS-musdb18-20s.md` | six MUSDB18 test tracks | 20 s | [`traces-musdb18-20s/`](traces-musdb18-20s) |
| `RESULTS-musdb18-7s.md` | the same tracks, 7 s excerpts | 6.8 s | [`traces-musdb18-7s/`](traces-musdb18-7s) |
| `RESULTS-synth-control-20s.md` | generated corpus, no download | 20 s | [`traces-synth-control/`](traces-synth-control) |
| `RESULTS-synth-control.md` | generated corpus, no download | 6.8 s | [`traces-synth-control/`](traces-synth-control) |
| `RESULTS.md` | superseded original synthetic run | 6.8 s | [`traces/`](traces) |

Each `RESULTS-*.md` has a `results-*.json` beside it with the same numbers in
machine-readable form: per-system and per-degradation-kind aggregates, paired
tests against the heuristic, and the measured cost.

## Which numbers describe which metric

v1.1 corrected four defects in the metric and the controllers, quantified in
[SCOPE.md](../SCOPE.md#known-issues-in-v1-and-what-v11-did-about-them). Every
arithmetic system was re-run against the corrected metric. The model-backed
rows were not: the cassettes carry measured numbers inside the prompts, so a
metric change invalidates them by construction and re-recording costs money
rather than compute.

So where a table carries an `agent` row, or where `BRIEFS.md` reports
translation quality, those numbers describe the **v1** metric while the rows
around them describe **v1.1**. Each is marked in place.

## The rest

- `corpus_manifest*.json` — which tracks, which split, and the digests they
  are pinned to. The audio itself is never committed: MUSDB18 is
  non-commercial with per-track terms, so the manifest travels and the audio
  is fetched by `headroom fetch-corpus`.
- `report/` — the HTML listening page. Synthetic clips only;
  `headroom report --html` refuses to write audio for any corpus whose
  manifest is not marked redistributable.
- `traces*/` — one JSON per cell: the full step-by-step run, the chain at
  every step, the distance, and the named abort. These are what
  `headroom eval --check-against` compares a fresh run against, and what makes
  a published row falsifiable rather than quotable.
