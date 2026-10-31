"""Speech and internal-pause segmentation of a wav, clip-relative.

`backend.pause_absorb` calls `vad()` on every warp-plan run to find the gaps a
dub may absorb. Two backends, and which one answered is always in the result,
since they have different operating points.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

from backend.common.media_io import probe_duration
from backend.asr.transcripts import _require

log = logging.getLogger(__name__)

_PYANNOTE_BUNDLE = "pyannote/speaker-diarization-community-1"
_PYANNOTE_SEGMENTATION = "segmentation/pytorch_model.bin"

@dataclass
class VadResult:
    speech: List[Tuple[float, float]]
    pauses: List[Tuple[float, float]]
    min_pause_s: float
    # Load-bearing: the two backends have different operating points, so a
    # silent fallback would make this clip incomparable to the rest.
    backend: str = "unknown"
    min_speech_s: float = 0.0
    duration_s: float = 0.0
    error: Optional[str] = None


def _read_wav(wav: Path) -> Tuple["Any", int]:
    sf = _require("soundfile")
    np = _require("numpy")
    y, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    return np.ascontiguousarray(y), sr


@lru_cache(maxsize=2)
def _pyannote_vad_pipeline(min_speech_s: float, min_pause_s: float):
    """Build the pyannote VAD pipeline from the cached diarization bundle.

    Raises on failure; `vad()` turns that into the energy fallback and records
    why. Cached, since the checkpoint costs ~2 s to load.
    """
    from huggingface_hub import try_to_load_from_cache

    ckpt = try_to_load_from_cache(_PYANNOTE_BUNDLE, _PYANNOTE_SEGMENTATION)
    if not isinstance(ckpt, str):
        raise RuntimeError(
            f"{_PYANNOTE_BUNDLE}/{_PYANNOTE_SEGMENTATION} is not in the HF cache "
            "(offline, and we never fetch during a scoring run)"
        )
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    model = Model.from_pretrained(ckpt)
    pipe = VoiceActivityDetection(segmentation=model)
    # `min_duration_off` fills shorter gaps, `min_duration_on` drops shorter
    # speech.
    pipe.instantiate({"min_duration_on": min_speech_s, "min_duration_off": min_pause_s})
    try:
        import torch
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
    except Exception as e:                      # CPU is correct, just slower
        log.warning("vad: could not move pyannote to cuda (%s); staying on cpu", e)
    return pipe


def _vad_pyannote(wav: Path, *, min_pause_s: float, min_speech_s: float
                  ) -> List[Tuple[float, float]]:
    import torch
    y, sr = _read_wav(wav)
    pipe = _pyannote_vad_pipeline(min_speech_s, min_pause_s)
    ann = pipe({"waveform": torch.from_numpy(y).unsqueeze(0), "sample_rate": sr})
    return [(float(s.start), float(s.end)) for s in ann.get_timeline().support()]


def _vad_energy(wav: Path, *, min_pause_s: float, min_speech_s: float,
                thresh_db: float = -38.0) -> List[Tuple[float, float]]:
    """10 ms frame-energy VAD, at the same operating point
    is used downstream to pick clips."""
    np = _require("numpy")
    librosa = _require("librosa")
    hop = 160                                   # 10 ms at 16 kHz
    y, sr = _read_wav(wav)
    hop = max(1, int(round(sr * 0.010)))
    rms = librosa.feature.rms(y=y, frame_length=hop * 2 + 80, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    mask = db > thresh_db

    runs: List[List[float]] = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            runs.append([i * hop / sr, j * hop / sr])
            i = j
        else:
            i += 1
    merged: List[List[float]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_pause_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_speech_s]


def vad(wav: Path, *, min_pause_s: float = 0.25, min_speech_s: float = 0.20,
        backend: str = "auto") -> VadResult:
    """Speech and internal-pause segmentation of `wav`, clip-relative.

    `backend` is "auto" (pyannote, else energy), "pyannote" or "energy"; the
    one that actually answered is always in the result.
    """
    wav = Path(wav)
    try:
        duration = probe_duration(wav)
    except Exception as e:
        log.warning("vad: probe_duration(%s) failed (%s)", wav.name, e)
        duration = 0.0

    used, err = backend, None
    speech: List[Tuple[float, float]] = []
    if backend in ("auto", "pyannote"):
        try:
            speech = _vad_pyannote(wav, min_pause_s=min_pause_s, min_speech_s=min_speech_s)
            used = "pyannote"
        except Exception as e:
            if backend == "pyannote":
                return VadResult(speech=[], pauses=[], min_pause_s=min_pause_s,
                                 backend="pyannote", min_speech_s=min_speech_s,
                                 duration_s=duration, error=f"pyannote VAD failed: {e!r}")
            err = f"pyannote unavailable, fell back to energy VAD: {e!r}"
            log.warning("vad: %s", err)
            used = "energy"
    if used == "energy" or backend == "energy":
        try:
            speech = _vad_energy(wav, min_pause_s=min_pause_s, min_speech_s=min_speech_s)
            used = "energy"
        except Exception as e:
            return VadResult(speech=[], pauses=[], min_pause_s=min_pause_s,
                             backend="energy", min_speech_s=min_speech_s,
                             duration_s=duration, error=f"energy VAD failed: {e!r}")

    speech = sorted((float(a), float(b)) for a, b in speech)
    pauses = [(speech[i][1], speech[i + 1][0]) for i in range(len(speech) - 1)
              if speech[i + 1][0] - speech[i][1] >= min_pause_s]
    return VadResult(speech=speech, pauses=pauses, min_pause_s=min_pause_s,
                     backend=used, min_speech_s=min_speech_s, duration_s=duration,
                     error=err)


__all__ = ["VadResult", "vad"]
