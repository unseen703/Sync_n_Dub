"""Speech / internal-pause segmentation of a wav, clip-relative.

Lifted out of `backend/eval/reference.py` on 2026-10-28, when the eval
package was deleted with WhisperX. Nothing here was ever part of the
evaluation harness — `backend.pause_absorb` calls `vad()` on every warp-plan
run to find the gaps a dub is allowed to absorb — it just happened to live
next to the scoring code because the golden-reference builder wrote the same
segmentation to disk.

Two backends, and which one answered is always in the result. pyannote and
the energy VAD have different operating points, so a silent fallback would
make one clip incomparable to the rest.

pyannote ships no standalone VAD checkpoint: its segmentation model is a
file inside the cached diarization bundle, resolved through the HF cache and
never the network, so this works offline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .media_io import probe_duration
from .transcripts import _require

log = logging.getLogger(__name__)

_PYANNOTE_BUNDLE = "pyannote/speaker-diarization-community-1"
_PYANNOTE_SEGMENTATION = "segmentation/pytorch_model.bin"

SR = 16_000


@dataclass
class VadResult:
    speech: List[Tuple[float, float]]
    pauses: List[Tuple[float, float]]
    min_pause_s: float
    # `backend` is load-bearing: pyannote and the energy VAD have different
    # operating points, so a silent fallback makes that clip incomparable to
    # the rest of the sweep.
    backend: str = "unknown"
    min_speech_s: float = 0.0
    duration_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        total_speech = sum(e - s for s, e in self.speech)
        return {
            "schema": "golden.vad/1",
            "times": "clip-relative seconds; 0.0 == clip t0",
            "backend": self.backend,
            "min_pause_s": self.min_pause_s,
            "min_speech_s": self.min_speech_s,
            "duration_s": round(self.duration_s, 4),
            "n_speech": len(self.speech),
            "n_pauses": len(self.pauses),
            "speech_s": round(total_speech, 4),
            "speech_frac": round(total_speech / self.duration_s, 4) if self.duration_s else None,
            # Pauses are INTERNAL gaps only. Leading/trailing silence is
            # reported separately -- a clip boundary is not a pause the dub
            # has to reproduce.
            "lead_silence_s": round(self.speech[0][0], 4) if self.speech else None,
            "tail_silence_s": (round(self.duration_s - self.speech[-1][1], 4)
                               if self.speech and self.duration_s else None),
            "speech": [[round(s, 4), round(e, 4)] for s, e in self.speech],
            "pauses": [[round(s, 4), round(e, 4)] for s, e in self.pauses],
            "errors": [self.error] if self.error else [],
        }


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
    the reason. Cached -- the checkpoint costs ~2 s to load and a sweep calls
    this once per track per clip.
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
    # pyannote's knobs map 1:1: `min_duration_off` fills shorter gaps,
    # `min_duration_on` drops shorter speech.
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
    """10 ms frame-energy VAD.

    The same operating point `scripts/select_clips.py` used to pick the clips,
    so a fallback stays consistent with selection.
    """
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
    """Speech / internal-pause segmentation of `wav`, clip-relative.

    `backend`: "auto" (pyannote, else energy), "pyannote", or "energy".
    The backend that actually produced the answer is always in the result.
    """
    wav = Path(wav)
    try:
        duration = float(len(_read_wav(wav)[0])) / SR
    except Exception:
        duration = 0.0
    try:
        duration = probe_duration(wav)
    except Exception as e:
        log.warning("vad: probe_duration(%s) failed (%s); using sample count", wav.name, e)

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
