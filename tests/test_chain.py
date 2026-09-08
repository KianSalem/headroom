from __future__ import annotations

import numpy as np
import pytest

from headroom.analysis.loudness import analyze_loudness
from headroom.audio import AudioBuffer
from headroom.dsp.backends.pedalboard import cache_stats, clear_cache, render_chain
from headroom.dsp.chain import Chain, ChainError
from headroom.dsp.ops import (
    GainOp,
    OpKind,
    op_compressor,
    op_eq,
    op_gain,
    op_limiter,
    op_stereo_width,
)

from .conftest import stereo, white


@pytest.fixture
def src() -> AudioBuffer:
    return stereo(white(seconds=4.0, seed=9))


def test_empty_chain_renders_the_source_unchanged(src: AudioBuffer) -> None:
    out = render_chain(src, Chain(), use_cache=False)
    np.testing.assert_array_equal(out.samples, src.samples)


def test_render_is_bit_identical_across_runs(src: AudioBuffer) -> None:
    """Phase 2 acceptance: same chain + same source -> identical bytes.

    Guaranteed for a given machine and pinned dependency versions. Across
    platforms, SIMD dispatch and denormal handling differ, so CI asserts
    bit-identity within a platform and a tight tolerance across them.
    """
    chain = (
        Chain()
        .add(op_gain(-2.0))
        .add(op_compressor(threshold_db=-18.0, ratio=3.0, makeup_db=1.5))
        .add(op_limiter(ceiling_dbtp=-1.0))
    )
    clear_cache()
    a = render_chain(src, chain, use_cache=False)
    b = render_chain(src, chain, use_cache=False)
    assert np.array_equal(a.samples, b.samples)


def test_minus_3db_gain_moves_loudness_by_exactly_3db(src: AudioBuffer) -> None:
    """Phase 2 acceptance: a known chain produces the expected measured change."""
    before = analyze_loudness(src).lufs_integrated
    after = analyze_loudness(render_chain(src, Chain().add(op_gain(-3.0)))).lufs_integrated
    assert after - before == pytest.approx(-3.0, abs=0.05)


def test_canonical_order_is_enforced_and_repositioning_is_logged() -> None:
    """An agent that puts the limiter first should be corrected by the system,
    not trusted -- and the correction must be visible."""
    chain = (
        Chain()
        .add(op_limiter(ceiling_dbtp=-1.0))
        .add(op_gain(-2.0))
        .add(op_eq([{"shape": "peak", "freq_hz": 240, "gain_db": -3.0}]))
    )
    ordered, moves = chain.canonical()
    assert [op.kind for op in ordered.ops] == [OpKind.GAIN, OpKind.EQ, OpKind.LIMITER]
    assert len(moves) == 3
    assert {m.kind for m in moves} == {OpKind.GAIN, OpKind.EQ, OpKind.LIMITER}


def test_canonical_order_is_stable_within_a_stage() -> None:
    """Two EQs in a deliberate order must keep it."""
    a = op_eq([{"shape": "peak", "freq_hz": 100, "gain_db": 1.0}])
    b = op_eq([{"shape": "peak", "freq_hz": 8000, "gain_db": 1.0}])
    ordered, moves = Chain().add(a).add(b).canonical()
    assert [op.id for op in ordered.ops] == [a.id, b.id]
    assert moves == ()


def test_already_canonical_chain_reports_no_moves() -> None:
    chain = Chain().add(op_gain(-1.0)).add(op_limiter(ceiling_dbtp=-1.0))
    _, moves = chain.canonical()
    assert moves == ()


def test_fingerprint_ignores_op_ids_but_not_parameters() -> None:
    """Two chains with identical parameters render identical audio and must
    share a cache entry regardless of how their ops were named."""

    def build(gain: float) -> Chain:
        return Chain().add(op_gain(gain))

    assert build(-2.0).fingerprint() == build(-2.0).fingerprint()
    assert build(-2.0).fingerprint() != build(-2.5).fingerprint()


def test_fingerprint_is_order_independent_after_canonicalisation() -> None:
    chain = Chain().add(op_gain(-2.0)).add(op_limiter(ceiling_dbtp=-1.0))
    reversed_chain = Chain(ops=tuple(reversed(chain.ops)))
    assert chain.canonical()[0].fingerprint() == reversed_chain.canonical()[0].fingerprint()


def test_edit_and_remove_return_new_chains() -> None:
    op = op_gain(-2.0)
    chain = Chain().add(op)
    edited = chain.edit(op.id, gain_db=-4.0)
    removed = chain.remove(op.id)

    original, revised = chain.ops[0], edited.ops[0]
    assert isinstance(original, GainOp)
    assert isinstance(revised, GainOp)
    assert original.gain_db == -2.0
    assert revised.gain_db == -4.0
    assert removed.ops == ()


def test_unknown_op_id_raises() -> None:
    with pytest.raises(ChainError, match="no op with id"):
        Chain().edit("nope", gain_db=1.0)


def test_render_cache_serves_repeat_renders(src: AudioBuffer) -> None:
    clear_cache()
    chain = Chain().add(op_gain(-2.0))
    first = render_chain(src, chain)
    second = render_chain(src, chain)
    assert cache_stats()["entries"] == 1
    assert np.array_equal(first.samples, second.samples)


def test_describe_renders_readable_signal_flow() -> None:
    chain = (
        Chain()
        .add(op_gain(-2.1))
        .add(op_eq([{"shape": "peak", "freq_hz": 240, "gain_db": -3.0, "q": 1.4}]))
    )
    text = chain.describe()
    assert "source ->" in text and "-> render" in text
    assert "240Hz" in text and "-3.0dB" in text


def test_chain_survives_a_json_roundtrip() -> None:
    """The chain is the artifact: it has to be exportable and re-importable."""
    chain = (
        Chain()
        .add(op_gain(-2.0))
        .add(op_eq([{"shape": "high_shelf", "freq_hz": 8000, "gain_db": 1.5}]))
        .add(op_stereo_width(width=1.2, band=6))
    )
    restored = Chain.model_validate_json(chain.model_dump_json())
    assert restored.fingerprint() == chain.fingerprint()
    assert restored == chain


def test_unstable_chain_raises_rather_than_returning_nans(src: AudioBuffer) -> None:
    infected = src.replace_samples(np.full_like(src.samples, np.inf))
    with pytest.raises((FloatingPointError, ValueError)):
        render_chain(infected, Chain().add(op_limiter(ceiling_dbtp=-1.0)), use_cache=False)
