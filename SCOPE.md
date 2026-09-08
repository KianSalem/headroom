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
| Natural-language briefs, graded without an LLM judge | **done** |
| Record/replay reproduction of the agent row at zero cost | **done**, verified |

## Deferred, with the reason

- **Brief evaluation at scale.** Eight briefs on one track ship, graded
  deterministically. A larger brief set across the real corpus is the obvious
  next run and needs no new code.
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

**Two control bugs were found by running the system, not by reasoning about
it, and both are now regression tests.** Two dimensions driving the same
control alternated between turns and oscillated, because the collapse rule kept
whichever request was larger; averaging the requested deltas fixes that without
double-correcting near-duplicate dimensions. And strike counting against the
previous step let a specialist that overshoots and corrects hold the route
indefinitely -- a sawtooth resets the count on every recovery -- so strikes now
count against the best score of the run, which is what the critic already did.

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

Two corpora, and the difference between them is a result in its own right.

**`headroom synth-corpus`** — six stationary synthetic tracks, no download.
Runs the pipeline offline and in CI. Not a source of headline numbers.

**`headroom fetch-corpus`** — real music, via MUSDB18.

The degrade-and-recover paradigm targets the original's own feature vector, so
the source needs to be well-produced and legally shareable, not a commercial
master. A public corpus is therefore strictly better than private material:
anyone can download the same files and reproduce every number.

### Why not MUSDB18-HQ

SPEC named MUSDB18-HQ. It turned out to be the wrong choice for a portfolio
piece, for a reason that had nothing to do with audio quality: it is a **22.66
GB** single zip. A reviewer will not spend that to check somebody's numbers,
and the machine this was built on did not have the disk.

Two corrections to what this document previously claimed, both found by asking
Zenodo's API rather than repeating the folklore:

- MUSDB18-HQ does **not** need an access request. Its record is
  `access_right: open`. The 22 GB is the only real barrier.
- Its licence is **`other-nc`** — non-commercial, per-track — not
  CC BY-NC-SA. That matters, because non-commercial means the audio must stay
  out of a portfolio site's HTML report, not merely out of git.

`fetch-corpus` pulls the same tracks from a smaller distribution instead:
`musdb18` (4.68 GB, full-length mixes) or `musdb18-7s` (147 MB, the SiSEC18
7-second excerpts). Both are open access, both are pinned by the size and MD5
Zenodo publishes, and only the mixture stream is decoded out of each
five-stream stem file — so peak disk is the archive plus one track, and the
archive is deleted afterwards.

Three things that cost real time and are worth writing down:

- **Zenodo ignores `Range`.** A resumed transfer appends a second full body to
  the partial one. The result has a valid central directory and every member
  fails with "Bad magic number for file header". Partials are now discarded
  rather than resumed.
- **The stem streams carry no titles.** Nothing in the container says which of
  the five is the mixture. Rather than trust the convention, all five were
  decoded and summed: streams 1–4 reproduce stream 0 to within 0.018 peak
  absolute error, which is AAC coding noise. Stream 0 is the linear mixture.
- **The 7-second variant cannot support the dynamics measurement.** A 6.8 s
  excerpt gives about four 3 s short-term loudness windows, so `lra` rests on
  thin evidence — and `lra` is exactly where the architecture showed its
  largest win. It is kept for proving the pipeline runs, not for quoting.

### How the synthetic corpus actually differs, measured

`headroom corpus-stats` exists so that this is a number rather than a caveat.
**Six tracks each, both clipped to the same 6.8 s**, because the first version
of this table did not hold clip length constant and got a result badly wrong
— see below.

Six tracks each, reported at both clip lengths the evaluation uses, because
one row behaves completely differently at the two:

| | synth @20s | real @20s | synth @6.8s | real @6.8s |
|---|---|---|---|---|
| `spectral_flatness` | 0.274 | **0.004** | 0.273 | 0.004 |
| `spectral_centroid` | 3219 Hz | 709 Hz | 3213 Hz | 752 Hz |
| `spectral_tilt` | −0.43 dB/oct | **−5.39 dB/oct** | −0.43 | −5.34 |
| `percussive_ratio` | 0.025 | **0.287** | 0.025 | 0.255 |
| `attack_time_p50` | 8.3 ms | 18.5 ms | 9.5 | 20.2 |
| `plr` | 9.71 dB | 14.82 dB | 9.54 | 12.92 |
| `mono_compat_db` | −2.72 dB | −0.76 dB | −2.72 | −0.86 |
| `correlation` | +0.48 | +0.73 | +0.48 | +0.70 |
| `lra` | **0.04 LU** | **2.34 LU** | 1.74 | 1.97 |

Every row except the last says the same thing at either clip length: synthetic
material is noise-like where music is tonal, nearly flat where music falls 5 dB
per octave, and carries a tenth of the transient energy. One synthetic track
has *negative* stereo correlation, which no mix would. A proportional
controller inverting a known op on a signal like that is close to solving the
problem analytically, which is why every deterministic system does well on it.

`lra` is the exception, and it is the interesting one. At 20 s the synthetic
corpus is **63× more stationary** than real music — 0.04 LU against 2.34, with
one real track at 6.37. At 6.8 s that difference vanishes into the noise:
1.74 against 1.97. The audio did not change; the measurement stopped being
able to resolve it.

### A number this document got wrong, and why

An earlier version of the table above claimed the synthetic corpus was **65×
more stationary** than real music: `lra` 0.03 LU against 1.96 LU. That
comparison was invalid — not because the conclusion was wrong, but because
nothing in it was controlled. The synthetic figure was measured on 20 s clips
and the real one on 6.8 s clips, so it was as much a measurement of window
count as of material. Held at 6.8 s it is 1.74 against 1.97, no difference
worth reporting; held at 20 s it is 0.04 against 2.34, a factor of 63. The
right answer happened to be close to the number originally quoted, which is
the least reassuring way to be right.

The cause is worth knowing, because it is a property of the measurement rather
than of the audio. `lra` is built from 3 s short-term windows on a 1 s hop, and
K-weighting takes time to settle, so the first window of any clip is an
outlier. At 20 s there are about eighteen windows and one outlier barely moves
the 10th percentile; at 6.8 s there are about four and it dominates them. A
held sine tone measures `lra` 0.94 LU at 6.8 s purely from filter settling.

Two consequences, both acted on:

- The `musdb18-7s` variant cannot support `lra`, and therefore cannot support
  the dynamics role, which is why the 4.68 GB full-length archive is the
  default and the excerpts are labelled as pipeline-proving only.
- Any cross-corpus comparison has to hold clip length constant. The
  real-music table is accompanied by a synthetic run at the same clip length
  and the same cell count, so that "real music changed the result" is a claim
  about material rather than about window counts.

Demo audio for the report comes from separately-sourced CC-BY clips committed
under `audio/demo/`, so hosting them publicly is unambiguous. Real-music
results are reported as numbers only — no players — because the corpus licence
is non-commercial.

## What the results actually showed

Recorded here because the point of the evaluation was to be able to be wrong.

### On real music, 54 paired cells

**The architecture beats the heuristic, significantly.** `agent-scaffold`
against `heuristic`: 26 wins, 13 losses, 15 ties, signed-rank p=0.0280,
median recovery +0.996 against +0.962, a median 2 renders against 3.

**Four conditions, because one comparison proves nothing.** Two corpora × two
clip lengths, 54 cells each, same six-track count, same seed, same systems:

| corpus | clips | scaffold | heuristic | gap | W/L/T | p |
|---|---|---|---|---|---|---|
| synthetic | 6.8 s | +1.000 | +0.994 | 0.006 | 19/14/21 | 0.2302 |
| synthetic | 20 s | +0.999 | +0.991 | 0.008 | 21/9/24 | 0.0470 |
| real music | 6.8 s | +0.996 | +0.962 | 0.034 | 26/13/15 | 0.0280 |
| real music | 20 s | +0.991 | +0.959 | 0.032 | 27/15/12 | 0.0101 |

**A claim these controls cut down.** With only the 6.8 s rows it looked like
the advantage appeared *only* on real music, and this document said so.
Synthetic at 20 s is also significant, p=0.0470. The surviving claim is about
**effect size rather than significance**: the median gap is about five times
larger on real music, 0.033 against 0.007, at both clip lengths. Real music
does not create the advantage — it makes it big enough to matter.

**The original null was mostly a sample-size problem, not a material one.**
The first synthetic run reported p=0.120 at 18 cells and was read as "the
architecture is not significantly better". The identical condition at 54 cells
gives p=0.0470. An 0.008 gap was never going to be resolvable at n=18, and
that is a less flattering explanation than the material one, which is why it
is stated first.

**The synthetic corpus is a poor instrument either way**, and the ties say so:
21 and 24 of 54 cells are exact ties, because both systems finish at recovery
1.000 on material easy enough to saturate them. Real music leaves headroom —
heuristic +0.959, convergence 67% against 76% — and 12 ties instead of 24.

**A prediction this document made, and got half wrong.** It said real music
would *widen* the coupling the architecture exploits. The per-kind margins did
the opposite and collapsed: `spectral_tilt` +0.290 → +0.057, `over_compress`
+0.148 → +0.004. What grew was breadth, not depth — small wins on most kinds
instead of large wins on two. The conclusion held; the stated reason for it
did not.

**The model's problem is not its score. It is that it will not sit still.**
`agent` against `heuristic` is significantly worse at 6.8 s (18/26/10,
p=0.0319) and *not* significantly worse at 20 s (19/22/13, p=0.1676). So
"the model is significantly worse on numeric targets", which this document
asserted, is an overclaim and is retracted.

What replaced it is better. Change only the analysis window — same six tracks,
same degradations, same seed, 6.8 s against 20 s — and measure how far each
system's per-kind medians move:

| system | mean abs. change over nine kinds | worst kind |
|---|---|---|
| `agent-scaffold` | 0.012 | 0.053 (`over_expand`) |
| `heuristic` | 0.021 | 0.130 (`over_expand`) |
| `agent` | **0.219** | **0.753** (`combo`) |

Ten times the heuristic's movement and eighteen times the scaffold's.
Concretely, at 6.8 s the model collapsed on `combo` (+0.247) and
`stereo_collapse` (+0.334) and was the best of the three on `spectral_tilt`
(+0.958); at 20 s the collapses are gone (+1.000, +0.963) and `spectral_tilt`
is its worst kind (+0.774 against the scaffold's +0.920). An earlier version of
this section built a "wins where one coherent spectral judgement is enough,
collapses where features are coupled" story out of the 6.8 s column. The 20 s
column does not support it. The per-kind story was noise dressed as mechanism.

What does survive both conditions: 31% oscillation against 6–11%, convergence
54% against 67%, 226 specialist turns against 159 at hit rates of 52–96%
against 95–98%, the lowest median of the three, and $1.20 against $0. The
mechanism is in the traces — a deterministic controller has its step size
*imposed* by the critic, which multiplies every correction by the damping
factor, while a model is only *told* the factor, and only after oscillation is
already detected.

**The bound is not 1.000, and that changes who looks bad.** A multi-start
Powell optimizer at 250 renders a cell, seeded with every other system's final
chain, never loses a cell to the heuristic (31/0/23) and never oscillates. Its
per-kind ceiling shows two degradations that simply cannot be undone with the
ops available: `over_compress` caps at +0.890 and `over_expand` at +0.854.

Expressed as a fraction of that ceiling, averaged over the nine kinds:
scaffold 97.9%, heuristic 97.3%, agent 80.6%. And per kind the picture inverts
in places — on `over_compress` all three systems sit within 4% of everything
achievable, so the task is the limit rather than the controller. The one place
a deterministic controller genuinely leaves value behind is `spectral_tilt`,
where the heuristic reaches 87.8% of bound and the architecture 93.6%.

The number worth quoting is not the recovery: **the scaffold reaches 99.6% of
the ceiling in a median 2 renders against the optimizer's 250.** That ratio is
the argument for a routed, bounded, measured loop instead of brute search.

### On the synthetic corpus, 18 paired cells, 20 s clips

Kept because it is the run the real-music work was checked against, and
because two of its readings did not survive that check. Neither conclusion
below should be read on its own — see the four-condition table above.

**The architecture beats the heuristic where coupling exists, and the
difference is not significant overall.** `agent-scaffold` converges on 89% of
cells against 72% in a median 2 renders against 4, and the gap concentrates on
the coupled degradations — `spectral_tilt` +0.985 against +0.695,
`over_compress` +0.981 against +0.833, tying the optimizer bound on both.
Across all 18 cells the paired recovery difference is +0.000 at p=0.120.
*Superseded:* the same condition at 54 cells gives p=0.047. The null result was
a sample-size artefact.

**The model is significantly worse than the heuristic here** — 3 wins, 11
losses, p=0.025, 44% convergence, 28% oscillation, $0.386. *Superseded as a
general claim:* it does not replicate on real music at 20 s (p=0.168). What
holds across every condition is the step-size mechanism — a deterministic
controller has its step size *imposed* by the critic, which multiplies every
correction by the damping factor, while a model is only *told* the factor, and
only after oscillation is already detected.

**Sonnet 5 is worse and 3.2x the price** on the same cell: +0.764 against
+0.799, 2.6x the output tokens, one specialist held for seven renders.

**The model wins on briefs, which nothing else can do at all.** Translation
94%, and the same recorded translations executed by the deterministic
controller beat the model-backed one: execution 75% against 56%, collateral 88%
against 85%, 4 of 8 briefs fully satisfied against 2, at zero controller cost
against $0.112. So the design conclusion is a split — the model reads intent,
arithmetic closes the loop.

**Two of these findings only exist because a baseline bug was fixed.** Before
the heuristic could decline to retry a failed direction, the architecture won
at p=0.030; afterwards, p=0.120. The earlier number was an artefact of a
strawman.

### What none of it supports

The degradations are still built from the same op vocabulary used to repair
them, on both corpora. Real music fixes the material, not the task: a
controller still knows the damage was reachable by ops it owns. Damage from
outside the vocabulary — a bad room, a codec, a bass player having a bad day —
is the harder test and is not run.

Six tracks and one seed is 54 cells, so nine degradation kinds means six cells
per kind and every per-kind number is indicative rather than a result — the
model's clip-length instability is what six cells per kind looks like when the
system under test is not deterministic. The optimizer bound was run at 6.8 s
only, because at 250 renders a cell it is the slowest thing in the suite, so
the 20 s table is un-normalised.

The two agent rows were run at two clip lengths and disagree with each other
about the model. That is reported as the finding rather than resolved by
picking the more flattering one, but it does mean the model's *ranking* against
the heuristic on real music is genuinely unsettled at this sample size.

## Cost

The agent evaluation is engineered to run for a few dollars. Total spend to
produce everything in this repository: **about $2.13**, against a $4.95
allowance. The two agent evaluation rows are $0.386 (synthetic, 18 cells) and
$1.272 (real music, 54 cells); the rest is probes, the Sonnet comparison, the
brief translations, and the README demo. Every one of those calls is committed
as a cassette, so reproducing all of it costs nothing.

- **Record/replay cassettes.** Every API call is recorded on first run and
  replayed at zero cost afterwards. Cassettes are committed, so development
  iteration is free after the first pass, CI can test the agent layer with no
  API key (a public repo cannot expose secrets to fork pull requests), and a
  stranger can reproduce the whole results table for nothing.
- **Six of seven systems cost nothing** — null, random, hillclimb, heuristic,
  the optimizer ceiling and the agent scaffold are pure arithmetic. `headroom
  eval` defaults to exactly those, so a fresh clone reproduces a full results
  table with no credential.
- **The controls were free, which is why they got run — and why one of them
  cut a claim down.** The four-condition design (two corpora × two clip
  lengths, 54 cells each) is pure arithmetic and cost nothing, so the honest
  version of the experiment was never competing with the budget. That mattered:
  the fourth condition is the one that showed the advantage is not exclusive to
  real music, and a design that had to be paid for per cell would have been
  tempting to stop at three.
- Agent runs default to **Haiku 4.5**, and that is a measured choice rather
  than a budget concession. On the same cell Sonnet 5 cost 3.2x as much
  ($0.098 against $0.031) for slightly *worse* recovery (+0.764 against
  +0.799): it spent 2.6x the output tokens, stayed on one specialist for seven
  straight renders and overshot repeatedly. The task is arithmetic over a
  six-row table, and it does not reward a larger model.
- **Prompt caching is declared but does not engage on Haiku 4.5.** A cache
  breakpoint only takes effect once the prefix clears a per-model minimum, and
  the ~2.3k-token role prefix is under it: probing with the real prefix
  returned zero cache writes on every TTL, while quadrupling it wrote and then
  read an entry. It does engage on Sonnet 5, which has a lower threshold. The
  declaration is left in place because it costs nothing when inert and starts
  paying with no code change; the hit rate is reported either way. Padding the
  prompt to clear a cache threshold was considered and rejected.
- Cache writes are priced by TTL. A 1-hour entry costs 2x the input rate
  against 1.25x for a 5-minute one, and an evaluation reuses one prefix for as
  long as the matrix takes, so 1 hour is the right TTL and the right price.
- A missing price raises rather than defaulting to zero: a cost of $0.00 is the
  most misleading number this project could print.
