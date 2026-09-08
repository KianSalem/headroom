# v1 scope

[SPEC.md](SPEC.md) is the full design and stays the north star. This file
records what v1 actually ships, what is deferred, and every place the
implementation deliberately departs from the spec.

The goal of v1 is a working system whose results you can **hear** and whose
numbers a stranger can **reproduce**, not a complete research programme.

## In scope

| Component | Status |
|---|---|
| Analyzer — every feature in SPEC §4 | **done**, validated against analytic signals |
| DSP chain — typed bounded ops, deterministic render | **done** |
| Distance metric — tolerance-scaled, per-feature breakdown | **done** |
| Corpus loader — MUSDB18-HQ, stable train/test split | **done** |
| CLI — `analyze`, `render`, `compare`, `presets`, `corpus` | **done** |
| Heuristic controller — proportional, damped, no LLM | not started |
| Optimizer ceiling (CMA-ES) + random-hillclimb floor | not started |
| Degradation suite + eval runner | not started |
| Supervisor + specialists + critic, bounded tool layer | not started |
| Working memory (tier 1), oscillation detection, named aborts | not started |
| Reference matching and delivery presets end to end | partly — targets exist, no controller |
| HTML report with audio players and the agent's chain | not started |

## Deferred, with the reason

- **Natural-language brief evaluation.** The most interesting test of the
  thesis, and the biggest build. v1 measures numeric recovery only.
- **Ablation suite and model sweep.** Seven ablations plus a model grid is
  months of runs. v1 reports one honest comparison table.
- **Blind ABX perceptual check.** Needs the system finished first.
- **Semantic and episodic memory (tiers 2 and 3).** Cross-run learning is
  valuable and none of it can be evaluated before tier 1 works.
- **Per-section optimization.** Requires time-varying processing.
- **Multiband compressor.** Large parameter space, no degradation needs it.
- **MCP server.** The tool layer is a plain typed module; an MCP adapter over
  it is a thin wrapper and a good demo, but it is not on the critical path.
- **LangGraph.** The supervisor at this scope is a small state machine. A
  hand-rolled loop is one less heavy dependency, avoids real friction under
  `mypy --strict`, and shows the control flow rather than importing it. The
  specialist interface stays narrow enough to swap.

## Deviations from SPEC

**Tolerance scaling replaces population z-scoring** (SPEC §5 rule 1). The
spec's justification for z-scoring is that raw units are not comparable.
Dividing by a perceptually-motivated tolerance solves that directly and yields
a metric in units of "how audible is this error", which is what a mastering
objective wants; a z-score answers "how unusual is this value in the corpus".
Z-scoring also makes the metric a function of the corpus — change the corpus
and every historical score silently changes — and estimating a covariance over
~28 features from a small train split would be rank-deficient outright.

**`op_saturation` cut.** SPEC §7.1 defines it but the §8.3 ownership table
assigns it to no specialist. It is also the only non-invertible op and no
degradation produces it.

**The limiter belongs to Loudness alone.** SPEC §8.3 gives `op_limiter` to
Dynamics and the limiter ceiling to Loudness, then states that two specialists
must never edit the same op type.

**`op_expander` added.** Not in the spec. Without it the `over_compress` and
`over_expand` degradations are unrecoverable by construction: a compressor
cannot undo compression.

**No `knee_db` on the compressor.** `pedalboard.Compressor` exposes no knee
control. Faking a soft knee ahead of a hard-knee compressor would make the
render disagree with the declared parameters.

**True-peak limiter, expander and band-limited width are hand-written.**
`pedalboard.Limiter` takes a sample-peak threshold with no lookahead or
oversampling and cannot honour a dBTP ceiling. Since `true_peak_dbtp` is a
scored feature, an unreachable ceiling would make the loudness specialist
oscillate against a tooling bug.

**Collinear and redundant features are reported but not scored.** `plr` is
exactly `true_peak_dbtp - lufs_integrated`; the spectral summaries duplicate
what the band energies already describe. See
`analysis.features.REPORTED_NOT_SCORED`.

**Band energy enters the score as a centered log-ratio.** It is normalized to
sum to one, so its nine values carry eight degrees of freedom and its deltas
are constrained to sum to zero.

**Weights normalize per family, not per feature.** Ten of 28 scored dimensions
are spectral and eleven are stereo; uniform weights would hand those two
families 75% of the objective while `lufs_integrated` got 3.6%.

**Constraints can be one-sided.** True peak in a delivery preset is a ceiling,
not a setpoint. Treating it as a setpoint penalized a quiet master as harshly
as a clipping one.

**Python 3.12+, not 3.11+.** numpy 2.5's type stubs use `type` statements that
require 3.12, so the spec's floor cannot be type-checked.

**Determinism is scoped to a platform.** SPEC §7.4 asks for bit-identical
renders. That holds for a given machine and pinned versions; across platforms
SIMD dispatch and denormal handling differ. CI asserts bit-identity within a
platform. Renders are float32 — pedalboard processes in float32 regardless of
input dtype — which puts the floor near −145 dB, far below anything measured.

## Corpus

**MUSDB18-HQ**, downloaded not vendored (mixed CC BY-NC-SA, academic use, needs
a one-time Zenodo access request). 150 uncompressed stereo tracks with stems
for a possible v2.

The degrade-and-recover paradigm targets the original's own feature vector, so
the source needs to be well-produced, legally shareable and uncompressed — not
a commercial master. A public corpus is therefore strictly better than private
material: anyone can download the same files and reproduce every number.

Demo audio for the report comes from separately-sourced CC-BY clips committed
under `audio/demo/`, so hosting them publicly is unambiguous.

## Cost

The agent evaluation is engineered to run for about ten dollars.

- **Record/replay cassettes.** Every API call is recorded on first run and
  replayed at zero cost afterwards. Cassettes are committed, so development
  iteration is free after the first pass, CI can test the agent layer with no
  API key (a public repo cannot expose secrets to fork pull requests), and a
  stranger can reproduce the whole results table for nothing.
- **Five of seven systems cost nothing** — null, random, hillclimb, heuristic
  and the optimizer ceiling are pure DSP.
- Agent runs use Sonnet 5 at low-to-medium effort with a 12-step cap, prompt
  caching, and render memoization. Development runs on Haiku 4.5.
- Measured spend is reported in the README rather than estimated.
