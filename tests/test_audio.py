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
