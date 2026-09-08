# headroom — build specification

This is the delivery document for the project. Read it fully before writing code.
It defines what to build, in what order, and what "correct" means at each stage.

---

## 1. What this is

**headroom** is a closed-loop agent system that performs mastering-engineer work on real
audio and is graded by measurement rather than by an LLM's opinion.

You give it an audio file and a target sonic profile. A supervisor agent reads the measured
distance between the current audio and the target, dispatches specialist agents that apply
bounded DSP operations through a tool layer, re-measures the render, and iterates until the
audio lands inside tolerance or the run is aborted.

The name is the audio term: headroom is the margin between the current level and clipping.

### The one-sentence thesis

A dumb controller can hit a LUFS target. It cannot handle *"more space, but keep the low end
tight."* The hypothesis this project tests is that an LLM's value in a control loop is
**translating underspecified perceptual goals into coordinated moves across a coupled
system** — and that on simple numeric targets, a plain heuristic controller wins.

We build both and find out. **The honest comparison is the deliverable**, not a demo.

### The rule that keeps this from being slop

**The AI never generates audio.** It makes engineering decisions about real audio, and every
decision is scored against physical measurement. There is no generative output anywhere in
the system, and no LLM-as-judge in the primary metric.

---

## 2. Non-goals

Explicitly out of scope. Do not build these, do not let them creep in.

- **No audio generation or synthesis.** No text-to-music, no stem generation.
- **No LLM-as-judge in the primary metric.** LLM judgment may appear in a clearly-labeled
  secondary experiment, never in the headline number.
- **No web UI in v1.** CLI plus generated HTML reports. A UI is a distraction until the
  numbers exist.
- **No real-time / low-latency processing.** Offline batch only.
- **No VST plugin hosting in v1.** `pedalboard` has built-in DSP; VST loading adds
  nondeterminism and licensing mess. Revisit only if built-ins prove insufficient.
- **No mixing from stems in v1.** v1 is stereo mastering, where ground truth is cleanest.
  Stems are v2 (§12).
- **No training or fine-tuning.** This is an orchestration project, not an ML project.

---

## 3. Core architecture

### 3.1 The chain model — read this before anything else

Processing is represented as an **ordered, declarative chain of typed operations**. The agent
never mutates audio directly. It edits a chain; the renderer applies the whole chain to the
*original* source every time.

```
source.wav ──▶ [gain: -2.1dB] ──▶ [eq: peak 240Hz -3.0dB Q1.4] ──▶ [comp: 2:1 -18dB] ──▶ render
                     ▲                        ▲
                     └── agent edits ops in place, adds, or removes ───┘
```

This is non-negotiable, and it buys four things:

1. **No cumulative degradation.** Re-rendering from source means 40 iterations don't stack 40
   generations of quantization and filter ringing.
2. **Revision instead of correction-stacking.** The agent can *change* the 240Hz cut from -3dB
   to -1.5dB. A destructive pipeline can only add a +1.5dB boost on top, which is a different
   and worse filter.
3. **The chain is the artifact.** It is human-readable, diffable, exportable, and it is what a
   real engineer would want to inspect.
4. **Determinism and reproducibility.** Same chain plus same source equals same bytes. This is
   testable, and §7 requires a test for it.

### 3.2 Component map

```
┌──────────────────────────────────────────────────────────────────┐
│  Analyzer (deterministic, no LLM)                                │
│  audio ──▶ FeatureVector                                         │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Distance (deterministic, no LLM)                                │
│  (FeatureVector, TargetProfile) ──▶ score + per-feature deltas   │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Critic (deterministic)                                          │
│  Reads distance history. Emits: CONTINUE | CONVERGED | ABORT,    │
│  plus an oscillation flag and a recommended step-size scalar.    │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Supervisor (LangGraph, LLM)                                     │
│  Reads the delta breakdown + memory. Chooses the next specialist │
│  and the constraint it operates under. Enforces chain ordering.  │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Specialists (LLM + tools): EQ · Dynamics · Stereo · Loudness    │
│  Each edits only the op types it owns, within hard bounds.       │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  DSP tool layer (MCP server) — bounded, validated, deterministic │
└──────────────────────────────────────────────────────────────────┘
```

Memory (§6) is read by the supervisor and specialists, and written by the loop after each
render.

---

## 4. The feature vector — the objective function

This is the most important deterministic component. Everything else is graded through it.
It must be correct, tested, and stable before any agent code exists.

All features are computed on a float64 stereo signal at the source sample rate. Every feature
is documented with its unit and expected range.

### 4.1 Loudness

| Feature | Definition | Unit |
|---|---|---|
| `lufs_integrated` | ITU-R BS.1770-4 gated integrated loudness | LUFS |
| `lufs_short_p10/p50/p90` | Percentiles of 3s short-term loudness | LUFS |
| `lra` | Loudness range (EBU R128) | LU |
| `true_peak_dbtp` | True peak, 4× oversampled | dBTP |
| `sample_peak_dbfs` | Raw sample peak | dBFS |

Use `pyloudnorm` for BS.1770. Do not hand-roll the K-weighting filter.

### 4.2 Dynamics

| Feature | Definition | Unit |
|---|---|---|
| `crest_factor_db` | 20·log10(peak / rms), whole signal | dB |
| `plr` | Peak-to-loudness ratio: `true_peak_dbtp - lufs_integrated` | dB |
| `crest_short_p50` | Median of per-3s-window crest factor | dB |

### 4.3 Spectral

| Feature | Definition | Unit |
|---|---|---|
| `band_energy[0..8]` | Energy in 9 log-spaced bands, normalized to sum to 1 | ratio |
| `spectral_centroid` | Power-weighted mean frequency | Hz |
| `spectral_flatness` | Geometric mean / arithmetic mean of spectrum | 0–1 |
| `spectral_rolloff_85` | Frequency below which 85% of energy sits | Hz |
| `spectral_tilt` | Slope of linear fit to log-power vs log-frequency | dB/oct |

Band edges (Hz): `[20, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 20000]`.

`band_energy` must be **normalized** so it measures balance, not level. Otherwise it
correlates with loudness and the distance metric double-counts.

### 4.4 Stereo

| Feature | Definition | Unit |
|---|---|---|
| `mid_side_ratio` | RMS(mid) / RMS(side) | ratio |
| `correlation` | Pearson correlation of L and R | -1 to 1 |
| `width_per_band[0..8]` | Side/mid energy ratio per band | ratio |
| `mono_compat_db` | Loudness delta when summed to mono | dB |

`width_per_band` matters: collapsing the low end while widening the top is a real mastering
move, and a single global width number cannot express it.

### 4.5 Transient

| Feature | Definition | Unit |
|---|---|---|
| `onset_rate` | Detected onsets per second | 1/s |
| `attack_time_p50` | Median 10–90% rise time across onsets | ms |
| `percussive_ratio` | Percussive energy / total, via HPSS | 0–1 |

### 4.6 Sections (non-stationarity)

Audio is not stationary. An intro, a drop, and a breakdown have genuinely different spectra,
and optimizing a global average can make every section individually wrong.

Detect sections with a self-similarity novelty curve (`librosa.segment`). Compute the full
feature vector **per section** in addition to globally.

- **v1:** optimize against the global vector; *report* per-section vectors and flag any
  section whose distance exceeds the global distance by more than 50%.
- **v2:** per-section targets and section-aware processing.

This ordering is deliberate — per-section optimization needs time-varying processing, which
is a much larger build.

---

## 5. Distance metric

```
distance(features, target) -> (score: float, breakdown: dict[str, FeatureDelta])
```

Rules:

1. **Z-score every feature** using population statistics computed across the reference
   corpus. Raw units are not comparable — 200 Hz of centroid error and 2 LU of loudness error
   must be put on the same scale before summing.
2. **Tolerance bands.** Each feature has a tolerance. Inside tolerance contributes exactly
   zero. This prevents the agent chasing noise and is what makes CONVERGED meaningful.
3. **Perceptual weights.** Features are not equally audible. Weights live in a config file,
   not in code. Ship a documented default; make it overridable.
4. **Return the breakdown, always.** The per-feature signed delta *is* the agent's input.
   A scalar score alone is useless to the supervisor.
5. **Vector-valued, not scalar.** `score` is for tracking; the breakdown drives decisions.

Default aggregation: weighted L2 over z-scored, tolerance-clipped deltas. Make the norm
configurable (L1/L2) — L1 is more robust to a single wild feature and worth an ablation.

### Two target modes

- **Eval mode** — target is the known original's feature vector. Ground truth exists.
  Used for every number in the report.
- **Reference mode** — target is derived from a reference track or the median of a reference
  set. This is the real-world use case, and it is how a human would actually use the tool.

---

## 6. Memory

Three tiers. This is the part most worth getting right; it is also the best interview
material in the project.

### 6.1 Working memory (per-run)

An append-only log of `(chain_snapshot, features, distance, breakdown, action_taken)`.

Derived on demand:
- **Local response curve** — for each op parameter touched, the observed Δfeature per Δparam
  *on this specific track*. This is the agent learning the plant it is controlling.
- **Tried-and-failed set** — actions that increased distance, so specialists stop retrying
  them.

### 6.2 Semantic memory (cross-run)

Durable records of `(starting_feature_delta, action, observed_response)`, aggregated into
expected response curves keyed by coarse context (genre tag, source loudness bucket, delta
sign). Answers: *"when the 250–500 Hz band is +0.8σ hot, what size cut typically fixes it
without dragging the centroid?"*

Storage: SQLite. Do not add a vector DB — the keys are numeric and low-dimensional, and
retrieval is a range query, not a similarity search.

### 6.3 Episodic memory (cross-run)

Full run traces with outcomes, retrievable by nearest-neighbour on the starting feature
vector. Used by the supervisor to prime a plan: *"the three most similar past runs converged
by leading with dynamics, not EQ."*

### 6.4 Contradiction and oscillation handling

Required, not optional. A naive loop **will** oscillate: boost highs, critic says too bright,
cut highs, critic says dull, forever.

Detect:
- Distance not monotonically decreasing over a window of N steps
- Sign flip on the same op parameter across consecutive edits
- Observed response contradicting semantic memory by more than a threshold

Respond, in order:
1. Halve the step-size scalar the critic passes to specialists.
2. Freeze the oscillating op and route to a different specialist.
3. Escalate to the supervisor for a full replan with the oscillation history in context.
4. Abort with a documented reason.

**Every abort reason must be a named enum value.** The distribution of abort reasons across
the eval is a headline result, not a footnote.

---

## 7. DSP tool layer

An MCP server exposing analysis and processing. Specialists reach DSP only through it.

### 7.1 Tools

```python
analyze(audio_ref: str) -> FeatureVector
render(source_ref: str, chain: Chain) -> audio_ref
distance_to_target(features: FeatureVector, target: TargetProfile) -> DistanceResult

# chain editing — return a new Chain, never mutate
chain_add_op(chain: Chain, op: Op, position: int | None) -> Chain
chain_edit_op(chain: Chain, op_id: str, params: dict) -> Chain
chain_remove_op(chain: Chain, op_id: str) -> Chain

# op constructors, each parameter-bounded
op_gain(gain_db: float)
op_eq(bands: list[EqBand])            # peak | low_shelf | high_shelf | hpf | lpf
op_compressor(threshold_db, ratio, attack_ms, release_ms, knee_db, makeup_db)
op_multiband_compressor(crossovers: list[float], bands: list[CompBand])
op_limiter(ceiling_dbtp, release_ms)
op_stereo_width(width: float, band: int | None)
op_saturation(drive: float, kind: Literal["tape","tube","soft"])
```

### 7.2 Bounds are enforced server-side

Every parameter has a hard min/max validated in the tool layer. A request for +40 dB is
rejected with a structured error naming the bound, not silently clamped — the agent must see
that it asked for something impossible.

Suggested starting bounds (tune with data, document changes):

| Parameter | Min | Max |
|---|---|---|
| `gain_db` | -24 | +24 |
| EQ `gain_db` | -18 | +18 |
| EQ `q` | 0.1 | 10 |
| Compressor `ratio` | 1.0 | 20 |
| Compressor `threshold_db` | -60 | 0 |
| `width` | 0.0 | 2.0 |
| Limiter `ceiling_dbtp` | -3.0 | -0.1 |

### 7.3 Signal chain ordering

The supervisor enforces canonical order: **gain staging → EQ → dynamics → stereo → saturation
→ limiter**. Ops inserted out of order are repositioned, and the repositioning is logged. An
agent that puts the limiter first should be corrected by the system, not trusted.

### 7.4 Determinism requirement

`render(source, chain)` must be bit-identical across runs for identical inputs. **Write this
test in Phase 2 and keep it green.** If `pedalboard` introduces nondeterminism (threading,
SIMD dispatch), pin the version and document it in the README.

---

## 8. Agent layer

### 8.1 SDK choice — read this before implementing

Use the **Anthropic Python SDK's beta tool runner** (`client.beta.messages.tool_runner`) for
specialists. The workload is custom DSP tools with no filesystem or bash requirement, which is
exactly what the tool runner is for, and it gives per-turn hooks for approval, logging, and
result modification.

Do **not** reach for the Claude Agent SDK (`claude-agent-sdk`) here. That is a different
package — the Claude Code harness with built-in Read/Write/Edit/Bash tools — and none of those
built-ins are useful for this loop. If you later want Agent SDK parity for its own sake, the
specialist interface (§8.3) is deliberately narrow enough to swap behind, but the tool runner
is the correct default and the tradeoff should be stated explicitly if it changes.

The **supervisor** is a LangGraph graph. LangGraph owns control flow and state; the LLM calls
inside nodes go through the Anthropic SDK.

### 8.2 Model configuration

```python
model = "claude-opus-5"              # default; never downgrade silently
thinking = {"type": "adaptive"}      # no budget_tokens — removed on Opus 5, returns 400
output_config = {"effort": "high"}   # low | medium | high | xhigh | max
```

Notes that will save you a debugging session:

- `budget_tokens` is **removed** on Opus 5. Sending it returns a 400. Use adaptive thinking.
- Assistant **prefill is removed** on Opus 5. Use structured outputs
  (`output_config={"format": {...}}`) or system-prompt instructions to shape responses.
- Use `strict: true` on tool definitions so tool arguments validate exactly. Schemas need
  `additionalProperties: false` and `required`.
- Always `json.loads()` tool inputs. Never string-match the serialized input.
- Model and effort must be **config-driven**, because the model sweep (§10) varies them across
  `claude-opus-5`, `claude-sonnet-5`, and `claude-haiku-4-5`.

### 8.3 Specialist contract

Every specialist takes the same input and returns the same output. Keep this interface
narrow — it is what makes the ablations cheap to run.

```python
@dataclass
class SpecialistRequest:
    current_features: FeatureVector
    target: TargetProfile
    breakdown: dict[str, FeatureDelta]   # signed, z-scored, tolerance-clipped
    chain: Chain
    directive: str                       # supervisor's constraint in natural language
    step_scale: float                    # critic's damping scalar, 0 < s <= 1
    memory: MemoryContext                # local response curve + relevant priors
    budget: StepBudget                   # max tool calls, max tokens

@dataclass
class SpecialistResult:
    chain: Chain                         # proposed new chain
    rationale: str                       # why — logged, never scored
    confidence: float
```

Specialists own disjoint op types:

| Specialist | Owns |
|---|---|
| EQ | `op_eq` |
| Dynamics | `op_compressor`, `op_multiband_compressor`, `op_limiter` |
| Stereo | `op_stereo_width` |
| Loudness | `op_gain`, limiter ceiling, final level |

Disjoint ownership is what makes credit assignment tractable. Do not let two specialists edit
the same op type.

### 8.4 The directive is the point

The supervisor's `directive` field is where the thesis lives. It should carry
underspecified perceptual intent — *"open up the top without making the snare splashy"* —
not a numeric setpoint. If directives degenerate into *"set band 6 to -2.3 dB"*, the LLM is
doing nothing a heuristic cannot, and the project has answered its own question in the
negative. **Log every directive; they are primary evidence in the writeup.**

---

## 9. Baselines — build these BEFORE the agent

This ordering is a hard rule. It keeps the project honest and guarantees a shippable result
even if the agent underperforms.

| Baseline | Description | Purpose |
|---|---|---|
| `null` | Do nothing | Sanity floor; recovery ratio must be ~0 |
| `random` | Random in-bounds ops, same step budget | Proves anything above it is learning |
| `heuristic` | Deterministic proportional controller with damping, no LLM | **The real competitor** |
| `single_agent` | One LLM with all tools, no supervisor, no memory | Isolates the value of orchestration |
| `full` | Supervisor + specialists + memory | The system |

### The heuristic controller

Build this properly, not as a strawman. A weak baseline makes the whole comparison worthless
and any reviewer will spot it immediately.

It should: map each feature delta to the op parameter that most directly affects it, apply a
proportional correction scaled by a damping factor, iterate, and stop on tolerance or
no-improvement. It will be genuinely good at single-feature numeric targets. That is the
point.

**If the heuristic wins overall, that is the result, and the writeup says so in the first
paragraph.** The interesting question then becomes *which subset of cases* the agent wins,
and the eval must be able to answer that — so slice every metric by degradation type.

---

## 10. Evaluation harness

### 10.1 Corpus

Source material: released Salemtech masters, unmastered full mixes, and stems (stems reserved
for v2).

**Split the corpus into train and test and never tune on test.** Every headline number comes
from the test split. This will be the first thing a serious reader checks.

### 10.2 Degradation suite

Parametric and seeded, so every run is reproducible from `(track_id, degradation, seed)`.

| Degradation | Parameters |
|---|---|
| `spectral_tilt` | slope ±1 to ±6 dB/oct |
| `band_shift` | one band, ±2 to ±9 dB |
| `resonant_peak` | freq, +4 to +12 dB, Q 2–8 |
| `over_compress` | ratio 4–20, threshold -30 to -12 dB |
| `over_expand` | inverse: increase crest factor |
| `stereo_collapse` | width 1.0 → 0.0–0.4 |
| `stereo_overwide` | width 1.0 → 1.5–2.0 |
| `level_offset` | ±12 dB |
| `combo` | 2–3 of the above stacked |

`combo` is the important one. Single degradations are where the heuristic should win; coupled
degradations are where orchestration should earn its cost. **Report them separately.**

### 10.3 Metrics

| Metric | Definition |
|---|---|
| **`recovery_ratio`** | `1 - (final_distance / initial_distance)`. **Primary.** 1.0 is perfect, 0 is no progress, negative means it made things worse. |
| `converged` | Reached tolerance within budget |
| `steps_to_converge` | Chain edits before convergence |
| `oscillation_rate` | Fraction of runs tripping the oscillation detector |
| `abort_reason` | Distribution over the named enum |
| `cost_usd` | Token cost per run |
| `wall_time_s` | Latency per run |
| `regression_rate` | Fraction of runs with `recovery_ratio < 0` |

Report **median and IQR**, not just the mean. One catastrophic run should not be able to hide
inside an average, and it should not be able to dominate one either.

### 10.4 Ablations

Each isolates one design decision. This section is what makes the project read as senior.

1. **No memory** — all three tiers off
2. **Working memory only** — no cross-run learning
3. **No supervisor** — specialists round-robin
4. **No damping** — critic always returns `step_scale = 1.0`
5. **No bounds** — parameters unclamped (expect instability; that is the finding)
6. **Numeric directives** — supervisor emits setpoints instead of perceptual language.
   *This is the direct test of the core thesis.*
7. **Model sweep** — `claude-opus-5` vs `claude-sonnet-5` vs `claude-haiku-4-5`, and effort
   `low` vs `high` vs `xhigh`. Produces a quality-per-dollar curve.

### 10.5 The perceptual check

Measured is not perceived. A system can hit every target and sound wrong.

Run a small blind ABX: for N test-set tracks, compare the system's output against the original
master and against the heuristic's output. Even n=1 (you) is worth doing if it is honestly
blind and honestly reported.

**Document where measurement and perception disagree.** That section will be the most-quoted
part of the writeup, and it is the part no one else bothers to write.

---

## 11. Repository layout

```
headroom/
├── README.md                    # what it is, results table, how to run
├── SPEC.md                      # this file
├── pyproject.toml
├── src/headroom/
│   ├── analysis/
│   │   ├── features.py          # FeatureVector assembly
│   │   ├── loudness.py          # BS.1770 via pyloudnorm
│   │   ├── spectral.py
│   │   ├── stereo.py
│   │   ├── transient.py
│   │   └── sections.py
│   ├── dsp/
│   │   ├── ops.py               # typed ops + bounds
│   │   ├── chain.py             # chain model + render
│   │   └── backends/pedalboard.py
│   ├── target/
│   │   ├── profile.py           # target vector, tolerances, weights
│   │   └── distance.py
│   ├── tools/server.py          # MCP server
│   ├── agents/
│   │   ├── graph.py             # LangGraph supervisor
│   │   ├── critic.py
│   │   ├── specialists/{eq,dynamics,stereo,loudness}.py
│   │   └── memory/{working,semantic,episodic}.py
│   ├── baselines/{heuristic,random_search,single_agent}.py
│   └── cli.py
├── evals/
│   ├── corpus.py                # train/test split, manifest
│   ├── degradations.py
│   ├── runner.py
│   └── report.py                # HTML + markdown results
├── tests/
└── .github/workflows/ci.yml
```

---

## 12. Build phases

Each phase has acceptance criteria. **Do not start a phase until the previous one passes.**

### Phase 0 — Scaffolding
Package layout, `pyproject.toml`, ruff + mypy + pytest, CI on push, audio I/O, corpus loader
with train/test manifest.
**Accept:** `pytest` green in CI; a WAV loads and round-trips bit-identically.

### Phase 1 — Analyzer
Every feature in §4. Unit tests with synthetic signals of known properties: a -20 LUFS sine,
white noise (flatness ≈ 1), a pure tone (flatness ≈ 0), a fully-correlated mono signal
(correlation = 1.0), an out-of-phase signal (correlation = -1.0).
**Accept:** every feature tested against a signal whose true value is known analytically;
LUFS within ±0.1 LU of a reference implementation.

### Phase 2 — DSP chain
Op types, bounds validation, chain model, `pedalboard` renderer.
**Accept:** determinism test passes (same chain + source → identical bytes); every bound
rejects out-of-range input with a structured error; a known chain produces the expected
measured change (e.g. -3 dB gain moves `lufs_integrated` by -3.0 ±0.05).

### Phase 3 — Target and distance
Target profiles, z-score population stats from the train split, tolerances, weights, distance
with breakdown.
**Accept:** `distance(x, x) == 0`; distance is monotonic under increasing single-feature
perturbation; the breakdown sums consistently with the scalar score.

### Phase 4 — Degradation suite + eval runner + trivial baselines
All degradations, seeded and reproducible. Runner executing `null` and `random`. Report
generation.
**Accept:** `null` recovery ratio ≈ 0; degradations are reproducible from a seed; the report
renders end to end.

### Phase 5 — Heuristic controller
The real baseline.
**Accept:** beats `random` decisively on every single-feature degradation; converges on
`level_offset` and `spectral_tilt` in a handful of steps.

### Phase 6 — MCP tool server
Expose analysis, render, and chain editing. Bounds enforced. Structured errors.
**Accept:** a manual MCP client can analyze, edit a chain, render, and re-analyze.

### Phase 7 — Single-agent baseline
One LLM, all tools, no supervisor, no memory.
**Accept:** completes runs without crashing; produces a recovery-ratio distribution; cost and
latency recorded per run.

### Phase 8 — Supervisor, specialists, critic
LangGraph graph, four specialists, oscillation detection, damping, named abort reasons.
**Accept:** oscillation rate materially below `single_agent`; every abort carries a named
reason; chain ordering is enforced and repositioning is logged.

### Phase 9 — Memory
All three tiers, SQLite-backed.
**Accept:** the no-memory ablation runs cleanly, and the memory-vs-no-memory delta is
measured and reported — **whatever its sign**.

### Phase 10 — Ablations, sweep, writeup
All ablations in §10.4, the model sweep, the perceptual check, and the README results table.
**Accept:** README leads with a results table including the heuristic baseline; failure
taxonomy documented; the model/effort cost curve is plotted.

---

## 13. Tech stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| DSP | `pedalboard` | Spotify's; fast, good built-ins |
| Loudness | `pyloudnorm` | BS.1770-4. Do not hand-roll |
| Analysis | `librosa`, `numpy`, `scipy` | |
| Audio I/O | `soundfile` | |
| LLM | `anthropic` | Beta tool runner for specialists |
| Orchestration | `langgraph` | Supervisor graph |
| Schemas | `pydantic` v2 | Ops, features, targets |
| Storage | `sqlite3` (stdlib) | Memory tiers. No vector DB |
| Quality | `pytest`, `ruff`, `mypy --strict` | |

Pin exact versions of `pedalboard`, `librosa`, and `numpy` — all three can shift numeric
output across minor releases, which would silently invalidate stored memory and cached
population statistics.

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| **The agent loses to the heuristic** | This is an acceptable, publishable outcome. Slice results by degradation type so the report can say precisely where each approach wins. Design the report for this from day one. |
| **Measurement ≠ perception** | §10.5 perceptual check, reported honestly. |
| **Oscillation eats the step budget** | §6.4 detection and damping, built in Phase 8, not bolted on later. |
| **Cost blowup during evals** | Per-run token budget; run sweeps on `claude-haiku-4-5`; cache the system prompt and tool definitions; keep tool definitions byte-stable so the cache actually hits. |
| **`pedalboard` nondeterminism** | Determinism test in Phase 2; pin the version. |
| **Scope creep into mixing/generation** | §2 non-goals. Stems are v2, and only after v1 ships with numbers. |
| **Corpus too small to be meaningful** | Degradation sampling multiplies the corpus: N tracks × M degradations × K seeds. Report N of each explicitly rather than just the product. |

---

## 15. Conventions

- **Type everything.** `mypy --strict` in CI. Ops and features are pydantic models.
- **No LLM in the deterministic core.** `analysis/`, `dsp/`, and `target/` must import nothing
  from `anthropic`. Enforce with an import-linter rule.
- **Every run writes a trace** — chain history, features, distances, directives, tool calls,
  tokens, timings — to a single JSON file. The eval report is generated from traces, never
  from live runs.
- **Seed everything.** Every stochastic component takes an explicit seed.
- **Commit messages state what changed and why.** No emoji.
- **README carries real numbers**, including the ones that are unflattering.

---

## 16. Definition of done for v1

1. The analyzer computes every §4 feature, tested against analytically-known signals.
2. `render` is deterministic and bit-reproducible.
3. All five systems in §9 run end to end over the degradation suite.
4. The test-split results table reports recovery ratio (median + IQR), convergence rate,
   oscillation rate, cost, and latency for all five, **sliced by degradation type**.
5. All seven ablations in §10.4 have been run and reported.
6. The perceptual check is done and documented, including disagreements with the metric.
7. The README opens with the results table and an honest one-paragraph statement of what the
   orchestration did and did not buy.
8. A stranger can clone the repo, run one command against a sample track, and get a mastered
   output plus a trace.

---

## 17. First message to Claude Code

Suggested opening prompt inside this repo:

> Read SPEC.md in full. Then implement Phase 0 and Phase 1 only. Do not start Phase 2.
> For Phase 1, write the tests before the implementation — every feature must be validated
> against a synthetic signal whose true value is known analytically. Report which features you
> could not test that way and why.
