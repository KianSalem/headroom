# headroom

[![ci](https://github.com/KianSalem/headroom/actions/workflows/ci.yml/badge.svg)](https://github.com/KianSalem/headroom/actions/workflows/ci.yml)

A closed-loop agent system that performs mastering-engineer work on real audio and is graded
by measurement rather than by an LLM's opinion.

Give it audio and a target sonic profile. A deterministic supervisor reads the measured
distance to target, routes to the specialist that owns the largest error, applies its bounded
DSP edits, re-renders, re-measures, and iterates until the audio lands inside tolerance or a
critic stops it with a named reason.

**The AI never generates audio.** It makes engineering decisions about real audio, and every
decision is scored against physical measurement — integrated LUFS, true peak, spectral
balance, stereo correlation, crest factor. There is no LLM-as-judge in the primary metric,
and that is enforced by an import contract rather than by discipline.

```
headroom master mix.wav out.wav --reference ref.wav --target spotify
```

## What it does

Deliver to a platform:

```
$ headroom master mix.wav mastered.wav --target club
```

![headroom closing the loop against a delivery target](docs/media/demo-master.gif)

<details>
<summary>the same run as text</summary>

```
$ headroom master audio/synthetic/synth_01.wav mastered.wav --target club
target 'club' (loudness preset: loud club/CD master): 2 features across ['loudness']
system agent-scaffold

*  0  4.3904  1 out  gain.gain_db +4.234        role=loudness | targeting lufs_integrated -8.47 tol | 1 edit(s): gain.gain_db +4
*  1  0.0000  0 out  limiter.ceiling -0.300     role=loudness | targeting true_peak_dbtp +7.21 tol | 1 edit(s): limiter.ceiling 
   2  0.0000  0 out  converged                  inside tolerance

distance 5.2811 -> 0.0000  recovery +1.000  converged
2 renders, 5.6s

source -> gain[gain_db=+4.23] -> limiter[ceiling_dbtp=-0.30 release_ms=50.00] -> render
wrote mastered.wav
```

Every recording on this page is made by a committed
[tape script](docs/tapes), against the corpus `headroom synth-corpus` generates, with
`ANTHROPIC_API_KEY` empty. `vhs docs/tapes/master.tape` reproduces this file.

</details>

Or say what you want in words. The model's only job is to turn the sentence into
constraints; a deterministic controller and a deterministic metric do the rest:

![translating a natural-language brief into constraints](docs/media/demo-brief.gif)

<details>
<summary>the same run as text</summary>

```
$ headroom master audio/synthetic/synth_02.wav mastered.wav \
    --brief "More space and width, but keep the low end tight and mono."

brief: 'More space and width, but keep the low end tight and mono.'
  correlation_z       -2.0 tol ( -0.20 fisher-z)  -- more space
  width_5             +2.0 tol ( +2.00 dB)        -- more width
  width_6             +2.0 tol ( +2.00 dB)        -- more width
  width_7             +2.0 tol ( +2.00 dB)        -- more width
  width_8             +2.0 tol ( +2.00 dB)        -- more width
  hold at current value: width_0, width_1, width_2, width_3, width_4,
                         mono_compat_db, band_clr_0, band_clr_1, band_clr_2, band_clr_3
  translation cost 3326 in / 399 out

target 'brief:...': 15 features across ['spectral', 'stereo']

*  0  0.4724   3 out  width.band8 +0.202   role=stereo | 5 edits | targeting width_6 -2.00 tol
*  1  0.3114  10 out  width.global -0.166  role=stereo | targeting mono_compat_db -2.96 tol
   ...
distance 0.5330 -> 0.3114  recovery +0.416  stopped: no_improvement

source -> stereo_width[1.20 band=5] -> stereo_width[1.20 band=6] -> stereo_width[1.20 band=7]
       -> stereo_width[1.20 band=8] -> stereo_width[0.89] -> render
```

</details>

The half of that brief which says *don't* is the interesting half, and it becomes ten held
dimensions rather than a hope. Note also that it does not converge: holding mono
compatibility genuinely fights widening the top, the metric encodes that trade-off, and the
system reports a compromise at +0.416 instead of pretending. The translation is one API call
and is replayed from a committed cassette, so this exact run costs nothing to repeat.

Neither example needs an API key beyond that: `--system agent-scaffold` is the default and
makes no model calls at all. `--system agent` swaps the model-backed specialists in behind
the same interface.

## The architecture

Four specialists, a deterministic supervisor, and a bounded tool layer.

```
                    measurement (28 scored dimensions, no LLM)
                                     |
                          deterministic supervisor
                    routes to the role owning the largest
                      weighted error; reroutes when stuck
                                     |
        +----------------+-----------+-----------+----------------+
        | EQ             | Dynamics  | Stereo    | Loudness       |
        | 10 dimensions  | 5         | 11        | 2              |
        | owns: eq       | comp, exp | width     | gain, limiter   |
        +----------------+-----------+-----------+----------------+
                                     |
                      bounded tool layer (absolute setters,
                      structured refusals, ownership by omission)
                                     |
                    declarative chain -> render from source
```

Six decisions carry most of the weight, and each one exists because the obvious alternative
broke something measurable.

**Processing is a declarative chain re-applied to the original source every render.** Nothing
accumulates over forty iterations, a 240 Hz cut can be *revised* from −3 dB to −1.5 dB rather
than having a +1.5 dB boost stacked on top of it, and the chain — not the audio — is the
artifact you inspect, diff and export.

**Every scored dimension has exactly one owner, and so does every op kind.** Both are
asserted by tests. An unowned dimension is permanently unfixable; a twice-owned one is two
specialists correcting each other's drift across turns, which is oscillation nobody caused.

**A specialist sees only the dimensions it owns.** The other 18 to 26 are not in its prompt.
That is implemented as a filter over loop state, not as an instruction, and asserted against
the rendered prompt text.

**Ownership is enforced by omission.** A specialist is not asked to leave another's ops
alone; it is never handed the tools. A call to another role's tool returns a structured
refusal naming the owner, and the attempt is counted — a prompt that leaks the boundary is a
prompt problem, and this is how it becomes visible instead of becoming a bug.

**The budget is denominated in renders, not model calls.** A system that thinks longer is not
thereby given more attempts on the audio. Refusing a bad tool call therefore costs a
round-trip and not a render, which is what makes strict bounds cheap to enforce.

**A specialist may bundle several coordinated edits into one render.** The heuristic baseline
corrects one feature per render by construction. This is the architecture's entire numeric
claim, so `edits/turn` is a reported number rather than a design intention.

Routing itself is arithmetic and gets no model call. What the supervisor does own is harder
and still deterministic: rerouting around a role that has missed twice, accepting *"nothing I
own can move this"* as a real answer, refusing to spend a render on a chain that is audibly
unchanged, and correcting op order back to canonical mastering order — with the correction
rate recorded rather than assumed to be zero.

## The comparison is the deliverable

Seven systems run inside the identical loop, against the identical degraded input, with the
identical render budget, scored by the identical feature vector.

| system | what it is | why it is in the table |
|---|---|---|
| `null` | does nothing | if recovery is not ~0 here, the metric is broken |
| `random` | random bounded ops | a floor that is not zero |
| `hillclimb` | random perturbation of the best chain so far | search without a model of the problem |
| `heuristic` | proportional controller, damped, one feature per render | **the real competitor**, built to win where a controller should |
| `agent-scaffold` | the full architecture with arithmetic where the model goes | isolates architecture from model |
| `agent` | the same architecture with LLM specialists | isolates the model's contribution |
| `optimizer` | multi-start Powell, ~90–250 renders, seeded with every other system's answer | a **measured** achievable bound, not a competitor |

`agent-scaffold` uses the heuristic's correction constants *imported rather than copied*, so
it differs from the heuristic in exactly one respect: it may make several coordinated edits
per render. The heuristic-to-scaffold gap measures the architecture; the scaffold-to-agent
gap measures the model. Neither number means much alone, and the scaffold costs nothing to
run.

## Results

Real music — six MUSDB18 test tracks, 54 paired cells, one seed, 14-render budget for every
system, `claude-haiku-4-5` for the agent. Clips are 20 s, the length at which loudness range
is a real measurement rather than filter settling. No cell was skipped for a weak
degradation. Fetch the corpus with `headroom fetch-corpus`; full tables are in
[`results/RESULTS-musdb18-20s.md`](results/RESULTS-musdb18-20s.md).

| system | recovery (median) | IQR | converged | oscillated | renders | cost |
|---|---|---|---|---|---|---|
| **`agent-scaffold`** | **+0.991** | +0.912 to +1.000 | **67%** | **6%** | **2** | **$0** |
| `heuristic` | +0.959 | +0.858 to +1.000 | 67% | 11% | 3 | $0 |
| `agent` *(Haiku 4.5)* | +0.934 | +0.821 to +1.000 | 54% | 31% | 3 | $1.20 |
| `hillclimb` | +0.001 | +0.000 to +0.061 | 0% | 76% | 6 | $0 |
| `random` | +0.000 | +0.000 to +0.001 | 0% | 76% | 6 | $0 |
| `null` | +0.000 | +0.000 to +0.000 | 0% | 0% | 0 | $0 |

Paired against the heuristic, Wilcoxon signed-rank, same cells. Both clip lengths shown
because they disagree about the agent, and that disagreement is the interesting part:

| system | clips | median difference | wins | losses | ties | p |
|---|---|---|---|---|---|---|
| `agent-scaffold` | 20 s | +0.000 | 27 | 15 | 12 | **0.010** |
| `agent-scaffold` | 6.8 s | +0.000 | 26 | 13 | 15 | **0.028** |
| `agent` | 6.8 s | +0.000 | 18 | 26 | 10 | **0.032** |
| `agent` | 20 s | +0.000 | 19 | 22 | 13 | 0.168 |
| floors | either | −0.94 | 0 | 54 | 0 | ≤0.0001 |

### The controls, and two claims they cut down

The architecture beating the heuristic is only worth reporting if it survives holding
everything else still, so the comparison was run in four conditions — two corpora × two clip
lengths, 54 cells each, same six-track count, same seed, same systems. All of it is pure
arithmetic and cost nothing.

| corpus | clips | `agent-scaffold` | `heuristic` | gap | wins | losses | ties | p |
|---|---|---|---|---|---|---|---|---|
| synthetic | 6.8 s | +1.000 | +0.994 | 0.006 | 19 | 14 | 21 | 0.230 |
| synthetic | 20 s | +0.999 | +0.991 | 0.008 | 21 | 9 | 24 | **0.047** |
| real music | 6.8 s | +0.996 | +0.962 | **0.034** | 26 | 13 | 15 | **0.028** |
| **real music** | **20 s** | **+0.991** | **+0.959** | **0.032** | 27 | 15 | 12 | **0.010** |

**Retracted claim 1: "the advantage only appears on real music."** It does not — synthetic at
20 s is significant too (p=0.047). What survives is about **effect size rather than
significance**: the median gap is roughly five times larger on real music, 0.033 against
0.007, consistently at both clip lengths. Real music does not create the advantage; it makes
it large enough to be worth caring about.

**Retracted claim 2: "the model is significantly worse than the heuristic."** True at 6.8 s
(p=0.032) and on the original synthetic run (p=0.025), *not* true at 20 s on real music
(p=0.168) — which is the most informative condition of the three. Stated as a general result
it was an overclaim, and running the extra condition is what caught it.

**And the original null result was mostly a sample-size problem.** The first synthetic run
reported p=0.120 and was read as "not significant". The identical condition at 54 cells
instead of 18 gives p=0.047. An 0.008 gap was never resolvable at n=18, which is a less
flattering explanation than the material one, so it goes first.

Why the synthetic corpus is a weak instrument regardless: 21 and 24 of its 54 cells are exact
ties, because both systems finish at recovery 1.000 on material easy enough to saturate them.
Real music leaves headroom and gives 12. `headroom corpus-stats` says why it is harder —
spectral flatness 0.274 against 0.004, tilt −0.43 against −5.39 dB/oct, percussive ratio
0.025 against 0.287, loudness range 0.04 LU against 2.34.

### What the bound says, which the raw numbers hide

A multi-start Powell optimizer, 250 renders a cell, seeded with every other system's final
chain, says what was reachable at all within the op vocabulary. It never loses a cell to the
heuristic (31 wins, 0 losses, 23 ties) and never oscillates, which is what makes it usable as
a ceiling rather than a competitor. On the 6.8 s condition:

| system | recovery (median) | renders | % of bound, averaged over kinds |
|---|---|---|---|
| `optimizer` *(measured bound)* | +1.000 | 250 | 100% |
| **`agent-scaffold`** | **+0.996** | **2** | **97.9%** |
| `heuristic` | +0.962 | 3 | 97.3% |
| `agent` | +0.920 | 4 | 80.6% |

**The scaffold reaches 99.6% of the ceiling in 2 renders against the optimizer's 250.** That
ratio — not the recovery number — is the argument for a routed, bounded, measured loop over
brute search.

The bound also reframes the dynamics results, which is exactly what it is for. It is *not*
1.000 everywhere: `over_compress` caps at +0.890 and `over_expand` at +0.854, so those
degradations are not fully undoable with the ops available, no matter how many renders are
spent.

| kind | bound | `agent-scaffold` | `heuristic` | `agent` |
|---|---|---|---|---|
| `over_compress` | +0.890 | 96.5% | 96.1% | 96.6% |
| `over_expand` | +0.854 | 91.7% | 96.7% | **98.9%** |
| `spectral_tilt` | +0.969 | 93.6% | **87.8%** | **98.9%** |
| `combo` | +1.000 | **100.0%** | 96.1% | 24.7% |

So the systems looking weak on `over_compress` are all within 4% of everything that was
achievable — the task, not the controller, is the limit. Where a controller genuinely leaves
value behind is `spectral_tilt`, and that is the one kind the architecture most improves on
the heuristic (87.8% → 93.6%). It is also where the model does best of the three at this clip
length, at 98.9% of bound — a result that does not hold at 20 s, per the instability above.

### What that says

**The architecture earns its keep, and it is the most stable thing in the table.** The
scaffold is the same supervisor, tool layer, memory and critic as the agent, with the
heuristic's own correction constants imported rather than copied — so it differs from the
heuristic in exactly one respect, that it may make several coordinated edits per render.
That claim shows up in the traces as `edits/turn`, which reaches 4.72 on the stereo role. It
wins in three of four conditions, converges in a median 2 renders against 3, and halves the
heuristic's oscillation rate (6% against 11%).

**The model's problem is not that it scores worse. It is that it will not sit still.** Change
only the analysis window — same six tracks, same degradations, same seed, 6.8 s to 20 s — and
measure how far each system's per-kind results move:

| system | mean \|Δ\| across the nine kinds | worst kind |
|---|---|---|
| `agent-scaffold` | **0.012** | 0.053 (`over_expand`) |
| `heuristic` | 0.021 | 0.130 (`over_expand`) |
| `agent` | **0.219** | **0.753** (`combo`) |

The model moves ten times as far as the heuristic and eighteen times as far as the scaffold.
Concretely: at 6.8 s it collapsed on `combo` (+0.247) and `stereo_collapse` (+0.334) and was
the best system on `spectral_tilt` (+0.958); at 20 s the collapses are gone (+1.000, +0.963)
and `spectral_tilt` is its *worst* kind (+0.774, against the scaffold's +0.920). A per-kind
story about the model told at one clip length does not survive the other, and the earlier
version of this README told one.

What does survive across both: it oscillates on 31% of cells against 6–11%, converges on 54%
against 67%, consults its specialists 226 turns against the scaffold's 159 at hit rates of
52–96% where the scaffold's are 95–98%, and costs $1.20 against $0. The mechanism is visible
in the traces — a deterministic controller has its step size **imposed** by the critic, which
multiplies every correction by the damping factor, while a model is only **told** the factor,
and only after oscillation has already been detected. Persuasion is a worse actuator than
multiplication.

**Sonnet 5 is not the answer either.** On the same cell it cost 3.2× as much as Haiku for
slightly worse recovery (+0.764 against +0.799), spending 2.6× the output tokens and holding
one specialist for seven straight renders. This task is arithmetic over a six-row table; it
does not reward a larger model.

### The synthetic table, for comparison

The original run — 18 cells, 20 s clips, two test tracks, and a multi-start Powell optimizer
as a measured bound — is in [`results/RESULTS.md`](results/RESULTS.md). There the scaffold
reached +0.999 against the heuristic's +0.995 and the agent +0.913. The real-music runs
supersede it: three times the cells, harder material, and a four-condition control design
around it.

### Where the model does win

Briefs are the one thing no controller can do. *"More space, but keep the low end tight"* is
not a number, and the model turns it into signed offsets against dimensions the metric
already defines — after which the loop and the metric run unchanged.

Graded with no LLM judge. Each brief carries the regions it must move and the direction,
written down in advance; three separate scores are arithmetic. Same recorded translations,
two different controllers:

| controller | translation | execution | collateral | all three | controller cost |
|---|---|---|---|---|---|
| **`agent-scaffold`** | 94% | **75%** | **88%** | **4/8** | **$0.0000** |
| `agent` | 94% | 56% | 85% | 2/8 | $0.1117 |

The translation column is identical by construction — both rows replay the same cassette —
so the rest belongs to the controller alone. Eight translations cost $0.039 to record and
nothing to replay. Full detail in [`results/BRIEFS.md`](results/BRIEFS.md).

**So the design conclusion is a split, and it is measured rather than asserted: the model
reads intent at 94%, and arithmetic closes the loop better than the model does, for free.**

### Caveats that matter

**The degradations come from the same op vocabulary used to repair them.** Real music fixes
the *material*, not the task: a controller still knows the damage was reachable by ops it
owns. A degradation drawn from outside the vocabulary — a bad room, a codec, a bass player
having a bad day — is the harder test and is not run here.

**n=54 from six tracks is still small, and one seed.** Nine degradation kinds mean six cells
per kind, so every per-kind number is indicative and the overall paired test is the one to
read — the clip-length instability above is exactly what six cells per kind looks like when a
system is not deterministic. The architecture result sits at p=0.010 and p=0.047 depending on
corpus, which is not a large margin either.

**A 6.8 s clip cannot measure loudness range, and one earlier conclusion rested on that.**
`lra` is built from 3 s windows on a 1 s hop, so 6.8 s gives about four of them and
K-weighting settling dominates the percentiles — a held sine tone measures 0.94 LU this way.
That weakens exactly one of 28 dimensions, and it is the one the dynamics role turns on. The
20 s runs are the fix; the 6.8 s tables are kept because comparing them is what exposed the
model's instability.

**The optimizer bound is measured at 6.8 s, not 20 s.** At 250 renders a cell the bound is
the slowest thing here, so it was run on the shorter clips. The percentages of bound above
therefore come from the 6.8 s condition, and the 20 s table is un-normalised.

**Real-music audio is not published.** MUSDB18's licence is non-commercial with per-track
terms, so results are numbers only, with no listening page. The synthetic showcase in
[`results/report`](results/report) is what can be hosted.

## Honesty machinery

Every one of these exists because its absence would let a flattering number through.

- **`recovery_ratio` is measured against an achievable bound.** A multi-start Powell
  optimizer, seeded with every other system's final chain, says what was reachable at all.
  Without it, "0.85 recovery" is uninterpretable.
- **Weak cells are skipped, not silently averaged.** Recovery divides by the initial
  distance, so a degradation that barely damaged anything turns measurement noise into an
  excellent-looking score. Skipped cells are recorded with their reason.
- **Median and IQR, never mean and SD.** One catastrophic run should neither hide inside an
  average nor dominate one.
- **Wilcoxon signed-rank on paired cells.** Recovery ratios are bounded above, unbounded
  below and not remotely normal. Below six paired observations the test is not quoted at all.
- **A metric-config hash travels with every trace.** Two runs with different tolerances or
  weights are not comparable, and the report says so rather than quietly averaging them.
- **Best chain, not last chain** — applied identically to every system, and every
  intermediate score stays in the trace.
- **Every stop has an enumerated reason.** The distribution of abort reasons is a headline
  result, not a footnote.
- **Cost is read from API usage, and a model with no price entry raises** rather than
  defaulting to `$0.00`, which is the most misleading number this project could print. Cache
  writes are priced by TTL, since a 1-hour entry costs 2× the input rate against 1.25× for a
  5-minute one.
- **The baseline gets fixed even when that costs the headline.** The heuristic could only
  ever *increase* a compressor ratio, so on `over_compress` it added compression seven
  renders running while the distance climbed. Letting it decline to retry a failed direction
  raised it from +0.643 to +0.833 and moved the architecture's advantage from p=0.030 to
  p=0.120. The earlier number was an artefact of a strawman.

## Why the numbers are trustworthy

Every API call is recorded to a cassette committed in [`cassettes/`](cassettes/), so the
agent row reproduces at zero cost and with no key. That is verified rather than claimed:

```
$ headroom fetch-corpus --archive musdb18-7s     # the audio is fetched, never vendored
$ ANTHROPIC_API_KEY="" HEADROOM_CASSETTE_MODE=replay \
    headroom eval --manifest results/corpus_manifest_musdb18_7s.json --systems agent
54 traces, 0 cells skipped, 54/54 fully replayed
345 cassette hits, 0 misses, actual spend $0.0000 (recorded cost $1.2717)
recovery mismatches against the committed traces: 0 of 54 — bit-identical
```

The fetch is the one step that cannot be skipped: MUSDB18 is non-commercial with per-track
terms, so this repository ships the manifest and the digest, not the audio. Skipping it is a
common enough first move that it has its own error message naming the command, rather than a
libsndfile failure four frames down.

The cassettes are the API traffic verbatim, which makes the directory a readable record of
every prompt the system has ever sent — and means a test asserts across all 510 of them that
none carries a credential, that each one's key matches its own request, and that the recorded
tool calls still apply to a chain today. In CI, replay mode turns "the prompt changed and
nobody re-recorded" into a failing test rather than an unexpected bill.

Five of the six systems in the table are pure arithmetic and `headroom eval` defaults to
exactly those, so a fresh clone gets a full results table with no credential at all. The
corpus is public and fetched by one command that pins it to the digest Zenodo publishes;
`headroom synth-corpus` generates a deterministic synthetic corpus so the pipeline runs with
no download at all, and the manifest stores track paths relative to itself so a committed
manifest is portable rather than carrying one machine's directory layout.

## The measurement core

The objective function has to be correct before anything is graded through it, so every
feature is validated against a signal whose value is known analytically rather than against
another implementation:

- a sine's crest factor is exactly `20·log10(√2)` = 3.0103 dB
- a 1 kHz sine at −20 dBFS reads −20.03 LUFS (K-weighting is ~unity at 1 kHz)
- white noise band energy is proportional to band width — the top band is 300× wider than
  the bottom one and holds 301× the energy
- folding uncorrelated channels to mono loses exactly 3.01 dB
- the 10–90% rise of a linear ramp is 0.8× its length

Three DSP primitives are hand-written because `pedalboard` has no equivalent: an oversampled
**true-peak limiter** (its `Limiter` is sample-peak with no lookahead and cannot honour a
dBTP ceiling, and an unreachable ceiling would make the loudness specialist oscillate against
a tooling bug), a **downward expander** (without one the `over_compress` degradation is
unrecoverable by construction — a compressor cannot undo compression), and **band-limited
stereo width** (a global width control cannot express "tighten the lows, widen the top"). The
true-peak meter is also ours, since `pyloudnorm` measures loudness only; it agrees with
analytic ground truth to 0.006 dB and with `ffmpeg`'s `ebur128` to that tool's display
resolution.

Three corrections to the naive feature set, each of which would otherwise corrupt the score:
collinear features are **reported but not scored** (`plr` is exactly
`true_peak_dbtp - lufs_integrated`, so scoring all three counts loudness error three times);
band energy is **compositional**, normalized to sum to one, so it enters as a centered
log-ratio; and weights normalize **per family, not per feature**, because ten of 28 scored
dimensions are spectral and eleven are stereo, and uniform weights would hand those two
families 75% of the objective while `lufs_integrated` got 3.6%.

## Enforced invariants

```
$ lint-imports
Deterministic core imports nothing from anthropic    KEPT
Only agent.client may reach the model                KEPT
Core layers point one way                            KEPT
```

The second contract is the one that matters most. Exactly one module in the whole repository
can reach the API. Routing, the tool layer, the briefing filter and the working memory — the
parts the evaluation actually makes claims about — are held to the same rule as the
measurement core, which is why they are covered on every fork's pull request with the key
explicitly empty.

`mypy --strict` clean across 64 files, 288 tests, and `ruff` clean on the whole tree. Bounds
live in one set of annotations and are read from there both by the validators and by the JSON
schemas the model sees, so the advertised range and the enforced range cannot drift apart.

CI runs the whole suite with `ANTHROPIC_API_KEY` explicitly empty and the cassettes in replay
mode, on Linux and macOS, and then runs the demo above on a clean checkout — so "it works on
a fresh clone with no credential" is a job that fails rather than a claim in a readme.

## Commands

```
pip install -e .                                  # add [agent] for the model-backed system

headroom master mix.wav out.wav --reference ref.wav --target spotify
headroom master mix.wav out.wav --target club --system agent --model claude-haiku-4-5

headroom analyze mix.wav                          # the full feature vector
headroom compare mix.wav --preset spotify         # signed, per-feature distance to a target
headroom compare mix.wav --reference ref.wav      # ...or to a reference track's profile
headroom render mix.wav chain.json out.wav        # apply a declarative chain
headroom presets                                  # delivery targets

headroom synth-corpus --out audio/synthetic       # a corpus with no download
headroom fetch-corpus --archive musdb18 --tracks 6 # real music: 4.7 GB, pinned by md5
headroom corpus audio/synthetic                   # a stable train/test manifest
headroom corpus-stats                             # what the corpus is actually like
headroom eval                                     # the matrix; free systems by default
headroom report --html results/report             # tables plus a listening page
```

The recordings on this page are generated, not captured by hand: each one is a committed
[vhs](https://github.com/charmbracelet/vhs) tape in [`docs/tapes/`](docs/tapes) that runs
the real CLI against the synthetic corpus with `ANTHROPIC_API_KEY` empty, so `vhs
docs/tapes/brief.tape` reproduces the exact frames above. An `.mp4` sits beside each
`.gif` for use anywhere that is not a GitHub readme.

`fetch-corpus` exists because the honest headline needs real music and the standard corpus
for it, MUSDB18-HQ, is a 22.66 GB download. It pulls the same tracks from a smaller
distribution instead, verifies them against the MD5 Zenodo publishes, decodes only the
mixture stream out of each five-stream stem file, and deletes the archive afterwards —
peak disk is the archive plus one track, and what survives is about 50 MB of clips. The
audio is non-commercial and per-track licensed, so it is never committed, redistributed,
or embedded in the report; the manifest records the digest that was verified instead.

Distance is measured before anything is decided, and chain ordering is enforced by the
system rather than trusted from the input — hand it a chain with the limiter first and it
says what it moved:

![measuring distance to a target, then enforcing chain order](docs/media/demo-invariants.gif)

<details>
<summary>the same two runs as text</summary>

```
$ headroom compare audio/synthetic/synth_01.wav --preset club
target 'club' (loudness preset: loud club/CD master): 2 features across ['loudness']
score 5.2811 (l2)  1/2 out of tolerance  converged=False
by family: loudness=27.890
lufs_integrated      -13.234 ==    -9.000 LUFS      delta  -4.234   -8.47 tol  TOO LOW
true_peak_dbtp        -3.093 <=    -0.300 dBTP      delta  -2.793  -13.96 tol  ->

$ headroom render audio/synthetic/synth_02.wav docs/media/out-of-order-chain.json ordered.wav
repositioned gain 885431a3: 2 -> 0
repositioned stereo_width 7c0d51e2: 3 -> 2
repositioned limiter b975822f: 0 -> 3
source -> gain[gain_db=+2.00] -> eq[peak 240Hz -3.0dB Q1.40, high_shelf 8000Hz +2.5dB Q0.71] -> stereo_width[width=1.30 band=7] -> limiter[ceiling_dbtp=-1.00 release_ms=50.00] -> render
wrote ordered.wav
```

The chain fed to that second command is committed at
[`docs/media/out-of-order-chain.json`](docs/media/out-of-order-chain.json) with the limiter
at index 0 and the gain at index 2, so the repositioning is real rather than staged.

</details>

Distance is vector-valued and signed, because a scalar is useless to a controller. True peak
in a delivery preset is a **ceiling**, not a setpoint — 3 dB of headroom is not an error:

```
$ headroom compare mix.wav --preset spotify
score 0.7316 (l2)  1/2 out of tolerance  converged=False
lufs_integrated      -15.017 ==   -14.000 LUFS   delta  -1.017   -2.03 tol  TOO LOW
true_peak_dbtp        -4.014 <=    -1.000 dBTP   delta  -3.014  -10.05 tol  ->
```

## Reading the code

| path | what is in it |
|---|---|
| `src/headroom/analysis/` | the 28-dimension feature vector, validated analytically |
| `src/headroom/dsp/` | typed bounded ops, the declarative chain, deterministic render |
| `src/headroom/target/` | the tolerance-scaled distance metric and target profiles |
| `src/headroom/control/` | the shared loop and the deterministic critic |
| `src/headroom/baselines/` | heuristic, optimizer bound, random and hillclimb floors |
| `src/headroom/agent/` | roles, tools, briefing, memory, supervisor, model boundary |
| `evals/` | corpus fetch and split, degradations, runner, aggregation, HTML report |
| `docs/tapes/` | the tape scripts that record the demos on this page |

`src/headroom/agent/__init__.py` names the reading order for the agent package.

See [SPEC.md](SPEC.md) for the full design and [SCOPE.md](SCOPE.md) for what v1 ships, what
is deferred, and every deliberate deviation from the spec.
