# Analysis

The full results story: what was run, what the numbers say, what was retracted
along the way, and what none of it supports. The [README](../README.md) carries
the headline; this file carries the argument. Every table here is regenerated
from committed traces by `headroom report`, and every agent row on real music
replays from committed cassettes with no key.

- [Design](#design)
- [Headline: real music, 20 s clips](#headline-real-music-20-s-clips)
- [Paired tests, both clip lengths](#paired-tests-both-clip-lengths)
- [The four-condition control, and three claims it cut down](#the-four-condition-control-and-three-claims-it-cut-down)
- [What the bound says](#what-the-bound-says)
- [Stability under a measurement change](#stability-under-a-measurement-change)
- [Where the model wins: briefs](#where-the-model-wins-briefs)
- [The original synthetic run](#the-original-synthetic-run)
- [Caveats that matter](#caveats-that-matter)
- [Honesty machinery](#honesty-machinery)
- [The measurement core](#the-measurement-core)

## Design

**Degrade and recover.** A clean track is damaged by a known, parameterised
degradation drawn from the same op vocabulary the systems repair with. Each
system then gets the same render budget to bring the damaged audio back toward
the clean track's measured profile. A *cell* is one track × one degradation kind
× one seed; every system runs every cell, so comparisons are paired.

**recovery** = `1 − final/initial` distance to target. 1.000 is a complete
repair, 0 is no progress, negative means the system made it worse. It is
self-contained per cell, so the table reads without a reference system; the
optimizer bound below says how much of that 1.000 was reachable at all.

Seven systems run inside the identical loop, against the identical degraded
input, with the identical render budget, scored by the identical feature vector:

| system | what it is | why it is in the table |
|---|---|---|
| `null` | does nothing | if recovery is not ~0 here, the metric is broken |
| `random` | random bounded ops | a floor that is not zero |
| `hillclimb` | random perturbation of the best chain so far | search without a model of the problem |
| `heuristic` | proportional controller, damped, one feature per render | **the real competitor**, built to win where a controller should |
| `agent-scaffold` | the full architecture with arithmetic where the model goes | isolates architecture from model |
| `agent` | the same architecture with LLM specialists | isolates the model's contribution |
| `optimizer` | multi-start Powell, 250 renders, seeded with every other system's answer | a **measured** achievable bound, not a competitor |

`agent-scaffold` uses the heuristic's correction constants *imported rather than
copied*, so it differs from the heuristic in one designed respect: it may make
several coordinated edits per render. (One incidental difference is recorded in
`src/headroom/agent/specialist.py`: two of the eleven stereo corrections are
expressed in dB on the scaffold side and as a linear factor on the heuristic
side, with the same constants and clamps.) The heuristic-to-scaffold gap
measures the architecture; the scaffold-to-agent gap measures the model.
Neither number means much alone, and the scaffold costs nothing to run.

## Headline: real music, 20 s clips

Six MUSDB18 test tracks, nine degradation kinds, one seed: 54 paired cells.
14-render budget for every system, `claude-haiku-4-5` for the agent. Clips are
20 s, the length at which loudness range is a real measurement rather than
filter settling. No cell was skipped for a weak degradation. Full tables in
[`results/RESULTS-musdb18-20s.md`](../results/RESULTS-musdb18-20s.md).

| system | recovery (median) | IQR | converged | oscillated | renders | cost |
|---|---|---|---|---|---|---|
| **`agent-scaffold`** | **+0.991** | +0.912 to +1.000 | **67%** | **6%** | **2** | **$0** |
| `heuristic` | +0.959 | +0.858 to +1.000 | 67% | 11% | 3 | $0 |
| `agent` *(Haiku 4.5)* | +0.934 | +0.821 to +1.000 | 54% | 31% | 3 | $1.20 |
| `hillclimb` | +0.001 | +0.000 to +0.061 | 0% | 76% | 6 | $0 |
| `random` | +0.000 | +0.000 to +0.001 | 0% | 76% | 6 | $0 |
| `null` | +0.000 | +0.000 to +0.000 | 0% | 0% | 0 | $0 |

The optimizer bound is not in this table because it was measured at 6.8 s, not
20 s; see [What the bound says](#what-the-bound-says).

## Paired tests, both clip lengths

Wilcoxon signed-rank against the heuristic on the same cells. Both clip lengths
are shown because they disagree about the agent, and that disagreement is the
interesting part:

| system | clips | wins | losses | ties | p |
|---|---|---|---|---|---|
| `agent-scaffold` | 20 s | 27 | 15 | 12 | **0.010** |
| `agent-scaffold` | 6.8 s | 26 | 13 | 15 | **0.028** |
| `agent` | 6.8 s | 18 | 26 | 10 | **0.032** (worse) |
| `agent` | 20 s | 19 | 22 | 13 | 0.168 |
| floors | either | 0 | 54 | 0 | ≤0.0001 |

## The four-condition control, and three claims it cut down

The architecture beating the heuristic is only worth reporting if it survives
holding everything else still, so the comparison was run in four conditions:
two corpora × two clip lengths, 54 cells each, same six-track count, same seed,
same systems. All of it is arithmetic and cost nothing.

| corpus | clips | `agent-scaffold` | `heuristic` | gap | wins | losses | ties | p |
|---|---|---|---|---|---|---|---|---|
| synthetic | 6.8 s | +1.000 | +0.994 | 0.006 | 19 | 14 | 21 | 0.230 |
| synthetic | 20 s | +0.999 | +0.991 | 0.008 | 21 | 9 | 24 | **0.047** |
| real music | 6.8 s | +0.996 | +0.962 | **0.034** | 26 | 13 | 15 | **0.028** |
| **real music** | **20 s** | **+0.991** | **+0.959** | **0.032** | 27 | 15 | 12 | **0.010** |

**Retracted: "the advantage only appears on real music."** It does not;
synthetic at 20 s is significant too (p=0.047). What survives is about **effect
size rather than significance**: the median gap is roughly five times larger on
real music, 0.033 against 0.007, consistently at both clip lengths. Real music
does not create the advantage; it makes it large enough to be worth caring
about.

**Retracted: "the model is significantly worse than the heuristic."** True at
6.8 s (p=0.032) and on the original synthetic run (p=0.025), *not* true at 20 s
on real music (p=0.168), which is the most informative condition of the three.
Stated as a general result it was an overclaim, and running the extra condition
is what caught it.

**Retracted: the original null result as a material effect.** The first
synthetic run reported p=0.120 and was read as "not significant". The identical
condition at 54 cells instead of 18 gives p=0.047. An 0.008 gap was never
resolvable at n=18, which is a less flattering explanation than the material
one, so it goes first.

Why the synthetic corpus is a weak instrument regardless: 21 and 24 of its 54
cells are exact ties, because both systems finish at recovery 1.000 on material
easy enough to saturate them. Real music leaves headroom and gives 12.
`headroom corpus-stats` says why it is harder:

| measure (20 s clips, 6 tracks each) | synthetic | real music |
|---|---|---|
| spectral flatness | 0.004 | 0.274 |
| spectral tilt | −5.39 dB/oct | −0.43 dB/oct |
| percussive ratio | 0.287 | 0.025 |
| loudness range | 0.04 LU | 2.34 LU |
| stereo correlation | +0.729 | +0.478 |

## What the bound says

A multi-start Powell optimizer, 250 renders a cell, seeded with every other
system's final chain, says what was reachable at all within the op vocabulary.
It never loses a cell to the heuristic (31 wins, 0 losses, 23 ties) and never
oscillates, which is what makes it usable as a ceiling rather than a competitor.
On the 6.8 s real-music condition:

| system | recovery (median) | renders | mean % of bound over the nine kinds |
|---|---|---|---|
| `optimizer` *(measured bound)* | +1.000 | 250 | 100% |
| **`agent-scaffold`** | **+0.996** | **2** | **97.9%** |
| `heuristic` | +0.962 | 3 | 97.3% |
| `agent` | +0.920 | 4 | 80.6% |

Two different statistics are in that table, so to be explicit: the scaffold's
median recovery of 0.996 against the optimizer's 1.000 is where "99.6% of the
ceiling in 2 renders against 250" comes from; 97.9% is the per-kind ratio
averaged over kinds, which weights the hard kinds equally with the easy ones.
Either way, that ratio, not the recovery number, is the argument for a routed,
bounded, measured loop over brute search.

The bound also reframes the dynamics results, which is exactly what it is for.
It is *not* 1.000 everywhere: `over_compress` caps at +0.890 and `over_expand`
at +0.854, so those degradations are not fully undoable with the ops available,
no matter how many renders are spent.

| kind | bound | `agent-scaffold` | `heuristic` | `agent` |
|---|---|---|---|---|
| `over_compress` | +0.890 | 96.5% | 96.1% | 96.6% |
| `over_expand` | +0.854 | 91.7% | 96.7% | **98.9%** |
| `spectral_tilt` | +0.969 | 93.6% | **87.8%** | **98.9%** |
| `combo` | +1.000 | **100.0%** | 96.1% | 24.7% |

So the systems looking weak on `over_compress` are all within 4% of everything
that was achievable: the task, not the controller, is the limit. Where a
controller genuinely leaves value behind is `spectral_tilt`, and that is the
one kind the architecture most improves on the heuristic (87.8% → 93.6%). It is
also where the model does best of the three at this clip length, at 98.9% of
bound, a result that does not hold at 20 s, per the instability below.

## Stability under a measurement change

**The architecture earns its keep, and it is the most stable thing in the
table.** The scaffold wins in three of four conditions, converges in a median 2
renders against 3, and halves the heuristic's oscillation rate (6% against
11%). Its coordinated-edit claim shows up in the traces as `edits/turn`, which
reaches 4.64 on the stereo role at 20 s (4.72 at 6.8 s).

**The model's problem is not that it scores worse. It is that it will not sit
still.** Change only the analysis window, same six tracks, same degradations,
same seed, 6.8 s to 20 s, and measure how far each system's per-kind results
move:

| system | mean \|Δ\| across the nine kinds | worst kind |
|---|---|---|
| `agent-scaffold` | **0.012** | 0.053 (`over_expand`) |
| `heuristic` | 0.021 | 0.130 (`over_expand`) |
| `agent` | **0.219** | **0.753** (`combo`) |

The model moves ten times as far as the heuristic and eighteen times as far as
the scaffold. Concretely: at 6.8 s it collapsed on `combo` (+0.247) and
`stereo_collapse` (+0.334) and was the best system on `spectral_tilt` (+0.958);
at 20 s the collapses are gone (+1.000, +0.963) and `spectral_tilt` is its
*worst* kind (+0.774, against the scaffold's +0.920). A per-kind story about the
model told at one clip length does not survive the other, and an earlier
version of this repository told one.

What does survive across both clip lengths: the model oscillates on 31% of
cells against 6–11%, converges on 54% against 67%, consults its specialists 226
turns against the scaffold's 159, and costs $1.20 against $0. On the three
roles that are routed to often (loudness, EQ, stereo) its specialist hit rates
are 63–94% at 20 s where the scaffold's are 95–98%; on dynamics both are poor
(18% and 11%), because the dynamics dimensions have no single op that moves
them cleanly. The mechanism is visible in the traces: a deterministic
controller has its step size **imposed** by the critic, which multiplies every
correction by the damping factor, while a model is only **told** the factor,
and only after oscillation has already been detected. Persuasion is a worse
actuator than multiplication.

**Sonnet 5 is not the answer either.** On the same cell it cost 3.2× as much as
Haiku for slightly worse recovery (+0.764 against +0.799), spending 2.6× the
output tokens and holding one specialist for seven straight renders. This task
is arithmetic over a six-row table; it does not reward a larger model.

## Where the model wins: briefs

Briefs are the one thing no controller can do. *"More space, but keep the low
end tight"* is not a number, and the model turns it into signed offsets against
dimensions the metric already defines, after which the loop and the metric run
unchanged.

Graded with no LLM judge. Each brief carries the regions it must move and the
direction, written down in advance; three separate scores are arithmetic. Same
recorded translations, two different controllers:

| controller | translation | execution | collateral | all three | controller cost |
|---|---|---|---|---|---|
| **`agent-scaffold`** | 94% | **75%** | **88%** | **4/8** | **$0.0000** |
| `agent` | 94% | 56% | 85% | 2/8 | $0.1117 |

The translation column is identical by construction, both rows replay the same
cassette, so the rest belongs to the controller alone. Eight translations cost
$0.039 to record and nothing to replay. Full detail in
[`results/BRIEFS.md`](../results/BRIEFS.md).

**The design conclusion is a split, and it is measured rather than asserted:
the model reads intent at 94%, and arithmetic closes the loop better than the
model does, for free.**

## The original synthetic run

The first run, 18 cells on two synthetic test tracks at 20 s with the optimizer
bound alongside, is in [`results/RESULTS.md`](../results/RESULTS.md). There the
scaffold reached +0.999 against the heuristic's +0.995 and the agent +0.913.
The real-music runs supersede it: three times the cells, harder material, and a
four-condition control design around it.

One provenance gap is recorded rather than hidden. That run's agent row was
recorded before cassettes were committed to the repository, to a scratch
directory that has since diverged from the current prompts, so it is the one
agent row here that **does not replay**. Its traces say so in their
`system_stats.client.cassette.path` field. Every agent row on real music, and
every brief translation, replays from committed cassettes with recovery
bit-identical to the committed traces.

## Caveats that matter

**The degradations come from the same op vocabulary used to repair them.** Real
music fixes the *material*, not the task: a controller still knows the damage
was reachable by ops it owns. A degradation drawn from outside the vocabulary,
a bad room, a codec, a bass player having a bad day, is the harder test and is
not run here.

**n=54 from six tracks is still small, and one seed.** Nine degradation kinds
mean six cells per kind, so every per-kind number is indicative and the overall
paired test is the one to read. The clip-length instability above is exactly
what six cells per kind looks like when a system is not deterministic. The
architecture result sits at p=0.010 and p=0.047 depending on corpus, which is
not a large margin either.

**A 6.8 s clip cannot measure loudness range, and one earlier conclusion rested
on that.** `lra` is built from 3 s windows on a 1 s hop, so 6.8 s gives about
four of them and K-weighting settling dominates the percentiles: a held sine
tone measures 0.94 LU this way. That weakens exactly one of 28 dimensions, and
it is the one the dynamics role turns on. The 20 s runs are the fix; the 6.8 s
tables are kept because comparing them is what exposed the model's instability.

**The optimizer bound is measured at 6.8 s, not 20 s.** At 250 renders a cell
the bound is the slowest thing here, so it was run on the shorter clips. The
percentages of bound above come from the 6.8 s condition, and the 20 s table is
un-normalised.

**Four defects found in review are fixed in v1.1, and every arithmetic row
here was re-run on the corrected metric.** Per-band stereo width and attack
time were both measured wrong; so were two of the heuristic's stereo
corrections, and the floors' action format, which could trip the oscillation
detector on a fabricated sign. What each fix moved is quantified in
[SCOPE.md](../SCOPE.md#known-issues-in-v1-and-what-v11-did-about-them), in
tolerance units over the whole committed corpus: attack time by a median of
4.74 tolerances, per-band width by 0.004 dB on real music, and the other
eighteen dimensions by nothing at all. The model-backed rows could *not* be
re-run. Their cassettes carry the measured numbers inside the prompts, so a
metric change invalidates them by construction, and re-recording needs a
credential. Where an `agent` number appears below it still describes the v1
metric, and says so.

**Real-music audio is not published.** MUSDB18's licence is non-commercial with
per-track terms, so results are numbers only, with no listening page. The
synthetic showcase in [`results/report`](../results/report) is what can be
hosted, and `headroom report --html` refuses to write audio for any corpus
whose manifest is not marked redistributable.

## Honesty machinery

Every one of these exists because its absence would let a flattering number
through.

- **An achievable bound is measured, not assumed.** A multi-start Powell
  optimizer, seeded with every other system's final chain, says what was
  reachable at all. Without it, "0.85 recovery" on `over_expand` reads as a
  controller failure when it is 99% of what any controller could do.
- **Weak cells are skipped, not silently averaged.** Recovery divides by the
  initial distance, so a degradation that barely damaged anything turns
  measurement noise into an excellent-looking score. Skipped cells are recorded
  with their reason.
- **Median and IQR, never mean and SD.** One catastrophic run should neither
  hide inside an average nor dominate one.
- **Wilcoxon signed-rank on paired cells.** Recovery ratios are bounded above,
  unbounded below and not remotely normal. Below six paired observations the
  test is not quoted at all.
- **A metric-config hash travels with every trace.** Two runs with different
  tolerances, weights, or constrained features are not comparable, and the
  report says so rather than quietly averaging them.
- **Best chain, not last chain**, applied identically to every system, and
  every intermediate score stays in the trace.
- **Every stop has an enumerated reason.** The distribution of abort reasons is
  a headline result, not a footnote. A failure of the harness (a missing
  cassette, a dead API) is not one of them: it ends the evaluation with a
  non-zero exit rather than becoming a +0.000 row.
- **Cost is read from API usage, and a model with no price entry raises**
  rather than defaulting to `$0.00`, which is the most misleading number this
  project could print. Cache writes are priced by TTL, since a 1-hour entry
  costs 2× the input rate against 1.25× for a 5-minute one.
- **The baseline gets fixed even when that costs the headline.** The heuristic
  could only ever *increase* a compressor ratio, so on `over_compress` it added
  compression seven renders running while the distance climbed. Letting it
  decline to retry a failed direction raised it from +0.643 to +0.833 and moved
  the architecture's advantage from p=0.030 to p=0.120. The earlier number was
  an artefact of a strawman.

## The measurement core

The objective function has to be correct before anything is graded through it,
so every feature is validated against a signal whose value is known
analytically rather than against another implementation:

- a sine's crest factor is exactly `20·log10(√2)` = 3.0103 dB
- a 1 kHz sine at −20 dBFS reads −20.03 LUFS (K-weighting is ~unity at 1 kHz)
- white noise band energy is proportional to band width: the top band is 300×
  wider than the bottom one and holds 301× the energy
- folding uncorrelated channels to mono loses exactly 3.01 dB
- the 10–90% rise of a linear ramp is 0.8× its length

Three DSP primitives are hand-written because `pedalboard` has no equivalent:
an oversampled **true-peak limiter** (its `Limiter` is sample-peak with no
lookahead and cannot honour a dBTP ceiling, and an unreachable ceiling would
make the loudness specialist oscillate against a tooling bug), a **downward
expander** (without one the `over_compress` degradation is unrecoverable by
construction, since a compressor cannot undo compression), and **band-limited
stereo width** (a global width control cannot express "tighten the lows, widen
the top"). The true-peak meter is also ours, since `pyloudnorm` measures
loudness only; it agrees with analytic ground truth to 0.006 dB and with
`ffmpeg`'s `ebur128` to that tool's display resolution.

Three corrections to the naive feature set, each of which would otherwise
corrupt the score: collinear features are **reported but not scored** (`plr` is
exactly `true_peak_dbtp - lufs_integrated`, so scoring all three counts
loudness error three times); band energy is **compositional**, normalized to
sum to one, so it enters as a centered log-ratio; and weights normalize **per
family, not per feature**, because ten of 28 scored dimensions are spectral and
eleven are stereo, and uniform weights would hand those two families 75% of the
objective while `lufs_integrated` got 3.6%.
