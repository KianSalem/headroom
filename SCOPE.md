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
| CLI — `master`, `eval`, `report` | **done** |
| Heuristic controller — proportional, damped, no LLM | **done** |
| Optimizer ceiling (multi-start Powell) + random-hillclimb floor | **done** |
| Degradation suite + eval runner | **done** |
| Supervisor + specialists + critic, bounded tool layer | **done** |
| Working memory (tier 1), oscillation detection, named aborts | **done** |
| Record/replay cassettes, measured token cost | **done** |
| Reference matching and delivery presets end to end | **done** — `headroom master` |
| HTML report with audio players and the agent's chain | **done** |

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
must never edit the same op type. Loudness wins the tie because
`true_peak_dbtp` is the feature the limiter exists to control.

**The tool surface is absolute and upsert-keyed, with no op ids.** SPEC §7.2
implies id-addressed edits. In practice `set_gain(gain_db=-2.0)` means "the
gain op is −2 dB": there is one gain op, one compressor, one EQ, and an EQ
band is identified by which analysis band it corrects. Exposing ids would buy
configurations no scored dimension asks for, in exchange for hallucinated
identifiers and edits to stale ops. Restating a value is detected and reported
as a no-op instead of spending a render to discover the chain did not change.

**Ownership is enforced by omission, and the violation rate is reported.** A
specialist is not asked to respect the boundary; it is never handed the tools
that cross it. A call to another role's tool returns a structured refusal
naming the owner, and the count appears in the trace — a prompt that leaks the
boundary is a prompt problem, and this is how it becomes visible.

**Routing is deterministic; there is no supervisor model call.** Picking the
specialist that owns the largest weighted error is arithmetic. Spending a model
call on it would add cost, latency and a failure mode for nothing. What the
supervisor does own is harder and still deterministic: rerouting around a stuck
role, accepting "nothing I own can move this" as an answer, refusing a render
whose chain is audibly unchanged, and canonicalizing op order.

**An `agent-scaffold` system was added as a controlled ablation.** Not in the
spec. It is the full architecture — supervisor, tool layer, working memory,
critic — with a proportional policy where the model goes, using the heuristic's
correction constants *imported rather than copied*. It differs from the
heuristic in exactly one respect: it may make several coordinated edits per
render. So the heuristic-to-scaffold gap measures the architecture and the
scaffold-to-agent gap measures the model. Neither number is interpretable
alone, and it costs nothing to run.

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

The agent evaluation is engineered to run for a few dollars.

- **Record/replay cassettes.** Every API call is recorded on first run and
  replayed at zero cost afterwards. Cassettes are committed, so development
  iteration is free after the first pass, CI can test the agent layer with no
  API key (a public repo cannot expose secrets to fork pull requests), and a
  stranger can reproduce the whole results table for nothing.
- **Six of seven systems cost nothing** — null, random, hillclimb, heuristic,
  the optimizer ceiling and the agent scaffold are pure arithmetic. `headroom
  eval` defaults to exactly those, so a fresh clone reproduces a full results
  table with no credential.
- Agent runs default to Haiku 4.5 with a 14-render cap, prompt caching on the
  role prompt and tool schemas, and render and measurement memoization.
- A missing price raises rather than defaulting to zero: a cost of $0.00 is the
  most misleading number this project could print.
