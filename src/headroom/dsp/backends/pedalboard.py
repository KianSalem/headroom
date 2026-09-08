"""Rendering backend.

``pedalboard`` supplies gain, the EQ filter shapes and the compressor. The
true-peak limiter, the expander and band-limited stereo width are ours (see
:mod:`headroom.dsp.primitives`) because pedalboard has no equivalent.

Two determinism notes, both of which the test suite pins:

*Fresh plugin instances per render.* pedalboard plugins are stateful. A reused
instance can carry envelope state across renders, which would make the result
depend on call order rather than only on ``(source, chain)``.

*float32.* pedalboard processes and returns float32 whatever it is handed.
Rendered audio therefore carries float32 precision (a floor near -145 dB, far
below anything the feature vector measures) and is upcast to float64 for
analysis. Bit-identical reproduction is guaranteed for a given machine and
pinned dependency versions, not across platforms -- SIMD dispatch and denormal
handling differ. The suite asserts bit-identity within a platform and a tight
tolerance across them.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pedalboard as pb

from headroom.analysis.spectral import BAND_EDGES
from headroom.audio import AudioBuffer, Samples

from ..chain import Chain
from ..ops import (
    CompressorOp,
    EqOp,
    ExpanderOp,
    GainOp,
    LimiterOp,
    Op,
    OpKind,
    StereoWidthOp,
)
from ..primitives import expander, stereo_width, true_peak_limiter

#: Render cache. Keyed on (source content hash, chain fingerprint), so an agent
#: that reverts an edit gets the earlier render back for free. The loop
#: re-renders constantly, and renders dominate wall time for the optimizer
#: baseline, so this is a large and cheap win.
_CACHE: Final[dict[tuple[str, str], Samples]] = {}

_EQ_FILTERS: Final[dict[str, type]] = {
    "peak": pb.PeakFilter,
    "low_shelf": pb.LowShelfFilter,
    "high_shelf": pb.HighShelfFilter,
    "hpf": pb.HighpassFilter,
    "lpf": pb.LowpassFilter,
}


def cache_stats() -> dict[str, int]:
    return {"entries": len(_CACHE)}


def clear_cache() -> None:
    _CACHE.clear()


def _pedalboard_plugins(op: Op) -> list[pb.Plugin]:
    """Plugins for ops pedalboard can express. Fresh instances every call."""
    if isinstance(op, GainOp):
        return [pb.Gain(gain_db=op.gain_db)]
    if isinstance(op, EqOp):
        plugins: list[pb.Plugin] = []
        for band in op.bands:
            cls = _EQ_FILTERS[band.shape]
            if band.shape in ("hpf", "lpf"):
                plugins.append(cls(cutoff_frequency_hz=band.freq_hz))
            else:
                plugins.append(
                    cls(cutoff_frequency_hz=band.freq_hz, gain_db=band.gain_db, q=band.q)
                )
        return plugins
    if isinstance(op, CompressorOp):
        out: list[pb.Plugin] = [
            pb.Compressor(
                threshold_db=op.threshold_db,
                ratio=op.ratio,
                attack_ms=op.attack_ms,
                release_ms=op.release_ms,
            )
        ]
        if op.makeup_db != 0.0:
            # pedalboard.Compressor has no makeup control, so it is an explicit
            # trailing gain rather than a hidden one.
            out.append(pb.Gain(gain_db=op.makeup_db))
        return out
    return []


def _apply(samples: Samples, sample_rate: int, op: Op) -> Samples:
    if isinstance(op, ExpanderOp):
        return expander(
            samples, sample_rate, op.threshold_db, op.ratio, op.attack_ms, op.release_ms
        )
    if isinstance(op, StereoWidthOp):
        return stereo_width(samples, sample_rate, op.width, op.band, BAND_EDGES)
    if isinstance(op, LimiterOp):
        return true_peak_limiter(samples, sample_rate, op.ceiling_dbtp, op.release_ms)

    plugins = _pedalboard_plugins(op)
    if not plugins:
        raise NotImplementedError(f"no renderer for op kind {op.kind!r}")
    board = pb.Pedalboard(plugins)
    out = board(samples.astype(np.float32), sample_rate)
    return np.asarray(out, dtype=np.float64)


def render_chain(source: AudioBuffer, chain: Chain, use_cache: bool = True) -> AudioBuffer:
    """Apply ``chain`` to ``source`` in canonical order.

    Ordering is canonicalised here rather than trusted from the caller, so a
    chain that reached the renderer out of order still produces a correctly
    ordered result. The repositioning itself is reported by
    :meth:`Chain.canonical` for the caller to log.
    """
    ordered, _ = chain.canonical()
    key = (source.content_hash(), ordered.fingerprint())
    if use_cache and key in _CACHE:
        return source.replace_samples(_CACHE[key])

    samples = np.ascontiguousarray(source.samples, dtype=np.float64)
    for op in ordered.ops:
        samples = _apply(samples, source.sample_rate, op)
        if not np.all(np.isfinite(samples)):
            raise FloatingPointError(
                f"op {op.kind}:{op.id} produced non-finite samples; chain is unstable"
            )

    result = np.ascontiguousarray(samples, dtype=np.float64)
    if use_cache:
        _CACHE[key] = result
    return source.replace_samples(result)


def op_kinds_supported() -> tuple[OpKind, ...]:
    return tuple(OpKind)
