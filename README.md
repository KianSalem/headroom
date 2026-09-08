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

Pre-implementation. See [SPEC.md](SPEC.md) for the full build specification: architecture,
feature vector, evaluation design, ablations, and phased acceptance criteria.

## Results

Not yet. This section will carry the test-split results table — recovery ratio, convergence
rate, oscillation rate, cost, and latency across all five systems, sliced by degradation type
— including the numbers that don't flatter the agent.
