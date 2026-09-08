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
$ headroom master mix.wav mastered.wav --target spotify
target 'spotify' (loudness preset: Spotify normalization): 2 features across ['loudness']
system agent-scaffold

*  0  0.0000  0 out  gain.gain_db -1.344   role=loudness | targeting lufs_integrated +2.69 tol
   1  0.0000  0 out  converged             inside tolerance

distance 1.1937 -> 0.0000  recovery +1.000  converged
1 renders, 2.2s

source -> gain[gain_db=-1.34] -> render
```

Or say what you want in words. The model's only job is to turn the sentence into
constraints; a deterministic controller and a deterministic metric do the rest:

```
$ headroom master mix.wav out.wav --brief "More space and width, but keep the low end tight and mono."

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

Test split, 18 paired cells, 14-render budget for every system, `claude-haiku-4-5` for the
agent. Full tables sliced by degradation type, the paired tests, the per-specialist
breakdown and the measured spend are in [`results/RESULTS.md`](results/RESULTS.md), and
every number is recomputable from the traces committed beside it.

| system | recovery (median) | IQR | converged | oscillated | renders | cost |
|---|---|---|---|---|---|---|
| `optimizer` *(measured bound)* | +1.000 | +0.990 to +1.000 | 94% | 0% | 90 | $0 |
| **`agent-scaffold`** | **+0.999** | +0.988 to +1.000 | **89%** | 6% | **2** | **$0** |
| `heuristic` | +0.995 | +0.951 to +1.000 | 72% | 6% | 4 | $0 |
| `agent` *(Haiku 4.5)* | +0.913 | +0.881 to +0.997 | 44% | 28% | 4 | $0.386 |
| `random` | +0.000 | +0.000 to +0.022 | 0% | 67% | 6 | $0 |
| `hillclimb` | +0.000 | +0.000 to +0.039 | 0% | 89% | 6 | $0 |
| `null` | +0.000 | +0.000 to +0.000 | 0% | 0% | 0 | $0 |

Paired against the heuristic, Wilcoxon signed-rank on the same 18 cells:

| system | median difference | wins | losses | ties | p |
|---|---|---|---|---|---|
| `agent-scaffold` | +0.000 | 8 | 3 | 7 | 0.120 |
| `agent` | −0.060 | 3 | 11 | 4 | **0.025** |
| `optimizer` | +0.000 | 8 | 0 | 10 | 0.010 |
| `random` / `hillclimb` / `null` | −0.95 | 0 | 18 | 0 | ≤0.0002 |

### What that says

**The architecture earns its keep, on the cases where it should.** The scaffold is the same
supervisor, tool layer, memory and critic as the agent, with the heuristic's own correction
constants imported rather than copied — so it differs from the heuristic in exactly one
respect, that it may make several coordinated edits per render. It converges on 89% of cells
against 72%, in a median 2 renders against 4, and the gap is concentrated exactly where
coupling exists: `spectral_tilt` +0.985 against +0.695, `over_compress` +0.981 against
+0.833. On both it ties the optimizer bound. Across all cells the recovery difference is not
significant at n=18 (p=0.120), and that is the honest reading.

**The model does not, on a numeric target.** The identical architecture with Haiku 4.5 in
place of the arithmetic is *significantly worse* than the heuristic: 3 wins against 11
losses, p=0.025, 44% convergence, 28% oscillation, and $0.386. Its specialists have hit
rates of 25–88% where the scaffold's are 78–100%. The mechanism is visible in the traces: a
deterministic controller has its step size **imposed** by the critic, which multiplies every
correction by the damping factor, while a model is only **told** the factor — and only after
oscillation has already been detected. Persuasion is a worse actuator than multiplication.

**Sonnet 5 is not the answer either.** On the same cell it cost 3.2× as much as Haiku for
slightly worse recovery (+0.764 against +0.799), spending 2.6× the output tokens and holding
one specialist for seven straight renders. This task is arithmetic over a six-row table; it
does not reward a larger model.

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

The corpus here is `headroom synth-corpus` — six stationary synthetic tracks — and the
degradations are built from the same op vocabulary used to repair them. That is the easiest
possible case, and it flatters every deterministic system in the table, because a
proportional controller inverting a known op is close to solving the problem analytically.
Real music should widen the coupling the architecture exploits and narrow the advantage
arithmetic has. n=18 is small; the significant results are the large ones.

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
$ ANTHROPIC_API_KEY="" HEADROOM_CASSETTE_MODE=replay headroom eval --systems agent
18 traces, 0 cells skipped, 18/18 fully replayed,
recorded cost $0.3857, actual spend $0.0000
recovery mismatches against the committed traces: none — identical
```

The cassettes are the API traffic verbatim, which makes the directory a readable record of
every prompt the system has ever sent — and means a test asserts across all 159 of them that
none carries a credential, that each one's key matches its own request, and that the recorded
tool calls still apply to a chain today. In CI, replay mode turns "the prompt changed and
nobody re-recorded" into a failing test rather than an unexpected bill.

Six of the seven systems are pure arithmetic and `headroom eval` defaults to exactly those,
so a fresh clone gets a full results table with no credential at all. The corpus is public
([MUSDB18-HQ](https://zenodo.org/record/3338373)); `headroom synth-corpus` generates a
deterministic synthetic corpus so the pipeline runs with no download, and the manifest stores
track paths relative to itself so a committed manifest is portable rather than carrying one
machine's directory layout.

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
headroom corpus audio/synthetic                   # a stable train/test manifest
headroom eval                                     # the matrix; free systems by default
headroom report --html results/report             # tables plus a listening page
```

Chain ordering is enforced by the system, not trusted from the input. Hand it a chain with
the limiter first and it says so:

```
$ headroom render mix.wav chain.json mastered.wav
repositioned gain 885431a3: 2 -> 0
repositioned limiter b975822f: 0 -> 3
source -> gain[gain_db=+2.00] -> eq[peak 240Hz -3.0dB Q1.40, high_shelf 8000Hz +2.5dB Q0.71]
       -> stereo_width[width=1.30 band=7] -> limiter[ceiling_dbtp=-1.00 release_ms=50.00] -> render
```

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
| `evals/` | corpus, degradations, runner, aggregation, HTML report |

`src/headroom/agent/__init__.py` names the reading order for the agent package.

See [SPEC.md](SPEC.md) for the full design and [SCOPE.md](SCOPE.md) for what v1 ships, what
is deferred, and every deliberate deviation from the spec.
