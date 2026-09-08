from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from headroom.audio import SILENCE_FLOOR_DB, AudioBuffer, AudioError, db, load, save

from .conftest import SR, sine, stereo


def test_rejects_wrong_dtype() -> None:
    with pytest.raises(AudioError, match="float64"):
        AudioBuffer(np.zeros((10, 2), dtype=np.float32), SR)


def test_rejects_low_sample_rate() -> None:
    """Below 44.1 kHz the 20 kHz band edge exceeds Nyquist."""
    with pytest.raises(AudioError, match="Nyquist"):
        AudioBuffer(np.zeros((10, 2)), 22050)


def test_rejects_non_stereo_shape() -> None:
    with pytest.raises(AudioError, match="stereo"):
        AudioBuffer(np.zeros((10, 3)), SR)


def test_wav_roundtrip_is_bit_identical(tmp_path: Path) -> None:
    """Phase 0 acceptance: a WAV loads and round-trips bit-identically."""
    buf = stereo(sine(1000.0, -20.0, seconds=1.0))
    path = tmp_path / "rt.wav"
    save(buf, path, subtype="FLOAT")
    back = load(path)
    assert back.sample_rate == buf.sample_rate
    assert np.array_equal(back.samples.astype(np.float32), buf.samples.astype(np.float32))


def test_mono_is_duplicated_and_flagged(tmp_path: Path) -> None:
    path = tmp_path / "mono.wav"
    import soundfile as sf

    sf.write(str(path), sine(1000.0, -20.0, seconds=1.0), SR, subtype="FLOAT")
    buf = load(path)
    assert buf.was_mono
    assert np.array_equal(buf.samples[:, 0], buf.samples[:, 1])


def test_content_hash_is_stable_and_discriminating() -> None:
    a = stereo(sine(1000.0, -20.0, seconds=1.0))
    b = stereo(sine(1000.0, -20.0, seconds=1.0))
    c = stereo(sine(1000.0, -21.0, seconds=1.0))
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()


def test_mid_side_reconstruct_exactly() -> None:
    buf = stereo(sine(440.0, -12.0, seconds=1.0), sine(660.0, -15.0, seconds=1.0))
    mid, side = buf.mid(), buf.side()
    np.testing.assert_allclose(mid + side, buf.samples[:, 0], atol=1e-15)
    np.testing.assert_allclose(mid - side, buf.samples[:, 1], atol=1e-15)


def test_db_floors_silence_instead_of_returning_inf() -> None:
    assert db(0.0) == SILENCE_FLOOR_DB
    assert db(1.0) == 0.0
    assert db(0.5) == pytest.approx(-6.0206, abs=1e-4)


def test_analyze_cache_returns_identical_results() -> None:
    """A memo that changes a result is far worse than a slow measurement.
    HPSS makes measurement ~145x more expensive than hashing the samples, so
    the loop caches -- but only if caching is provably transparent."""
    from headroom.analysis.features import analyze, cache_stats, clear_cache

    buf = stereo(sine(997.0, -12.0, seconds=4.0))
    clear_cache()
    uncached = analyze(buf, use_cache=False)
    first = analyze(buf)
    second = analyze(buf)

    assert first == uncached
    assert second == uncached
    stats = cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_analyze_cache_distinguishes_different_audio() -> None:
    from headroom.analysis.features import analyze, clear_cache

    clear_cache()
    a = analyze(stereo(sine(997.0, -12.0, seconds=4.0)))
    b = analyze(stereo(sine(997.0, -18.0, seconds=4.0)))
    assert a.lufs_integrated != b.lufs_integrated


def test_analyze_cache_distinguishes_mono_provenance() -> None:
    """Two buffers can hold identical samples while disagreeing about whether
    the stereo features mean anything, so was_mono is part of the key."""
    from headroom.analysis.features import analyze, clear_cache

    clear_cache()
    samples = np.stack([sine(997.0, -12.0, seconds=4.0)] * 2, axis=1)
    as_stereo = analyze(AudioBuffer(samples, SR, was_mono=False))
    as_mono = analyze(AudioBuffer(samples, SR, was_mono=True))
    assert as_stereo.was_mono is False
    assert as_mono.was_mono is True


def test_analyze_cache_is_bounded() -> None:
    """An evaluation measures thousands of distinct renders; an unbounded cache
    would hold every one of them in memory."""
    from headroom.analysis.features import _CACHE_MAX, analyze, cache_stats, clear_cache

    clear_cache()
    rng = np.random.default_rng(0)
    for _ in range(_CACHE_MAX + 12):
        analyze(AudioBuffer(rng.standard_normal((SR // 8, 2)) * 0.05, SR))
    assert cache_stats()["entries"] <= _CACHE_MAX


def test_an_empty_buffer_is_refused_with_a_reason() -> None:
    """An empty file loads without complaint; analysing it used to fail with an
    IndexError several frames inside the spectral code."""
    from headroom.analysis.features import analyze

    with pytest.raises(AudioError, match="0 frames"):
        analyze(AudioBuffer(np.zeros((0, 2)), SR))
