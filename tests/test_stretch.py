"""Does each stretch backend actually stretch TIME, and leave PITCH alone?

This is the test the module's own docstring asks for. `duration.wsola_stretch_cap`
used to fall back to `np.interp`, which is a pitch shifter wearing a time
stretcher's name, and nothing caught it because nothing ever measured the
output's f0. So: synthesise a 440 Hz tone, stretch it, and assert on both
numbers.

`linear` is included as the control. It is expected to FAIL the pitch
assertion -- a 1.10x linear resample lands at 484 Hz -- and the test asserts
that it does, because "the pitch-preserving backends preserve pitch" only
means something next to a backend that does not.

Availability is probed per filter, not per binary (only the imageio_ffmpeg
static build on this host carries `rubberband`; conda's ffmpeg 7.1 does not).
A genuinely absent backend is skipped, never failed.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from backend import stretch as S
from conftest import write_tone

FACTOR = 1.10          # speed up by 10%: the pipeline's actual working range
TONE_HZ = 440.0
TONE_S = 2.0
SR = 24000

PITCH_PRESERVING = ("rubberband", "ffmpeg_atempo", "librosa")


def dominant_hz(path) -> float:
    """Peak of the magnitude spectrum, parabolically interpolated.

    A 2 s window at 24 kHz has ~0.5 Hz bins, so the raw peak is already good
    to ~0.1%; the interpolation just removes the bin-quantisation floor so a
    2% tolerance is testing the backend and not the FFT.
    """
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float64)
    mag = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    k = int(np.argmax(mag))
    if 0 < k < len(mag) - 1:
        a, b, c = mag[k - 1], mag[k], mag[k + 1]
        denom = a - 2 * b + c
        k = k + (0.5 * (a - c) / denom if denom else 0.0)
    return float(k) * sr / len(audio)


@pytest.fixture(scope="module")
def source_tone(tmp_path_factory):
    return write_tone(tmp_path_factory.mktemp("stretch") / "tone.wav",
                      TONE_S, sample_rate=SR, freq=TONE_HZ)


def test_the_probe_itself_is_accurate(source_tone):
    """If `dominant_hz` is wrong, every other assertion here is meaningless."""
    assert dominant_hz(source_tone) == pytest.approx(TONE_HZ, rel=0.002)


def test_at_least_one_backend_exists():
    assert S.available_backends(), "no stretch backend at all on this host"
    assert "linear" in S.available_backends()


@pytest.mark.parametrize("backend", S.BACKENDS)
def test_duration_within_2pct(backend, source_tone, tmp_path):
    if backend not in S.available_backends():
        pytest.skip(f"{backend} is not available on this host "
                    f"(have: {', '.join(S.available_backends())})")
    out = tmp_path / f"{backend}.wav"
    res = S.time_stretch(source_tone, out, factor=FACTOR, backend=backend)

    assert res.backend == backend
    assert res.pitch_preserving == (backend != "linear")
    expected = TONE_S / FACTOR
    assert res.achieved_duration_s == pytest.approx(expected, rel=0.02), (
        f"{backend}: wanted {expected:.4f}s, got {res.achieved_duration_s:.4f}s")
    assert sf.info(str(out)).duration == pytest.approx(expected, rel=0.02)


@pytest.mark.parametrize("backend", PITCH_PRESERVING)
def test_pitch_preserved_within_2pct(backend, source_tone, tmp_path):
    if backend not in S.available_backends():
        pytest.skip(f"{backend} is not available on this host "
                    f"(have: {', '.join(S.available_backends())})")
    out = tmp_path / f"{backend}.wav"
    S.time_stretch(source_tone, out, factor=FACTOR, backend=backend)
    got = dominant_hz(out)
    assert got == pytest.approx(TONE_HZ, rel=0.02), (
        f"{backend} shifted the pitch: {got:.1f} Hz vs {TONE_HZ:.1f} Hz "
        f"(ratio {got / TONE_HZ:.4f}); a true time stretch must not")


def test_linear_shifts_pitch_by_the_factor(source_tone, tmp_path):
    """The control arm: `linear` is a resampler and must look like one."""
    out = tmp_path / "linear.wav"
    S.time_stretch(source_tone, out, factor=FACTOR, backend="linear")
    ratio = dominant_hz(out) / TONE_HZ
    assert ratio == pytest.approx(FACTOR, rel=0.02), (
        f"linear resampling should raise f0 by the stretch factor "
        f"({FACTOR}x); measured {ratio:.4f}x")


def test_named_backend_never_downgrades_silently(tmp_path, source_tone):
    """`resolve_backend` is the module's whole reason to exist."""
    absent = [b for b in S.BACKENDS if b not in S.available_backends()]
    if not absent:
        pytest.skip("every backend is available here; nothing to refuse")
    with pytest.raises(RuntimeError, match="not available"):
        S.time_stretch(source_tone, tmp_path / "x.wav",
                       factor=FACTOR, backend=absent[0])


def test_unknown_backend_raises(source_tone, tmp_path):
    with pytest.raises(ValueError, match="unknown stretch backend"):
        S.time_stretch(source_tone, tmp_path / "x.wav",
                       factor=FACTOR, backend="pyrubberband")


def test_stretch_to_duration_reports_the_cap(source_tone, tmp_path):
    """A capped segment still overruns; the result must say so, not hide it."""
    res = S.stretch_to_duration(source_tone, tmp_path / "capped.wav",
                                target_duration_s=TONE_S / 2.0,
                                max_stretch=0.10, backend="linear")
    assert res.capped is True
    assert res.requested_factor == pytest.approx(2.0, rel=1e-6)
    assert res.applied_factor == pytest.approx(1.10, rel=1e-6)
    assert res.achieved_duration_s == pytest.approx(TONE_S / 1.10, rel=0.02)
