# headroom

A closed-loop agent system that performs mastering-engineer work on real audio and is graded
by measurement rather than by an LLM's opinion.

Give it audio and a target sonic profile. A supervisor agent reads the measured distance to
target, dispatches specialist agents that apply bounded DSP operations, re-measures the
render, and iterates until the audio lands inside tolerance.

**The AI never generates audio.** It makes engineering decisions about real audio, and every
decision is scored against physical measurement — integrated LUFS, true peak, spectral
balance, stereo correlation, crest factor. There is no LLM-as-judge in the primary metric.

## The question this answers

A plain controller can hit a LUFS target. It cannot handle *"more space, but keep the low end
tight."* This project tests whether an LLM's value in a control loop is translating
underspecified perceptual goals into coordinated moves across a coupled system — where EQ
changes alter compression behavior, which alters loudness, which alters perceived brightness.

So it ships a heuristic controller alongside the agent and compares them honestly. **The
comparison is the deliverable, not a demo.**

## Status

**The deterministic core is built and tested.** The analyzer, the DSP chain and the
distance metric all work; the controller, the agent and the evaluation harness do not
exist yet.

That ordering is deliberate. The objective function has to be correct before anything is
graded through it, so every feature is validated against a signal whose value is known
analytically rather than against another implementation:

- a sine's crest factor is exactly `20·log10(√2)` = 3.0103 dB
- a 1 kHz sine at −20 dBFS reads −20.03 LUFS (K-weighting is ~unity at 1 kHz)
- white noise band energy is proportional to band width — the top band is 300× wider than
  the bottom one and holds 301× the energy
- folding uncorrelated channels to mono loses exactly 3.01 dB
- the 10–90% rise of a linear ramp is 0.8× its length

Three DSP primitives are hand-written because `pedalboard` has no equivalent: an
oversampled **true-peak limiter** (its `Limiter` is sample-peak with no lookahead and
cannot honour a dBTP ceiling), a **downward expander** (without one the `over_compress`
degradation is unrecoverable by construction — a compressor cannot undo compression), and
**band-limited stereo width** (a global width control cannot express "tighten the lows,
widen the top"). The true-peak meter is also ours, since `pyloudnorm` measures loudness
only; it agrees with analytic ground truth to 0.006 dB and with `ffmpeg`'s `ebur128` to
that tool's display resolution.

137 tests, `mypy --strict` clean, and an import-linter contract that makes the
"no LLM in the objective function" rule enforceable rather than aspirational —
`analysis/`, `dsp/` and `target/` cannot import `anthropic`.

See [SPEC.md](SPEC.md) for the full design and [SCOPE.md](SCOPE.md) for what v1 ships,
what is deferred, and every deliberate deviation from the spec.

## What works today

```
pip install -e .

headroom analyze mix.wav                      # the full feature vector
headroom compare mix.wav --preset spotify     # signed, per-feature distance to a target
headroom compare mix.wav --reference ref.wav  # ...or to a reference track's profile
headroom render mix.wav chain.json out.wav    # apply a declarative chain
headroom presets                              # delivery targets
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

Distance is vector-valued and signed, because a scalar is useless to a controller. Note
that true peak in a delivery preset is a **ceiling**, not a setpoint — 3 dB of headroom is
not an error:

```
$ headroom compare mix.wav --preset spotify
score 0.7316 (l2)  1/2 out of tolerance  converged=False
lufs_integrated      -15.017 ==   -14.000 LUFS   delta  -1.017   -2.03 tol  TOO LOW
true_peak_dbtp        -4.014 <=    -1.000 dBTP   delta  -3.014  -10.05 tol  ->
```

## Results

Not yet — there is no controller to report on. This section will carry the test-split
results table: recovery ratio (median and IQR), convergence rate, oscillation rate, cost
and latency across every system, sliced by degradation type, including the numbers that
don't flatter the agent.

Two things will make that table trustworthy. The corpus is public
([MUSDB18-HQ](https://zenodo.org/record/3338373)), so anyone can download the same files
and re-derive the numbers. And every API call is recorded to a committed cassette, so
reproducing the whole table costs nothing and needs no API key.
