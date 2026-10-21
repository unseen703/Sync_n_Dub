"""Signal-level metrics over rendered dubbing artifacts.

These are the checks that catch damage the JSON manifests cannot see: a
stretch backend that transposed the voice, an `-shortest` mux that ate the
tail, a mix that clipped. Every function returns a dict and records what it
could not compute in `not_measured[]` with a reason.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:  # pragma: no cover - sibling module is owned by another agent
    from .reference import VadResult  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover
    VadResult = None  # type: ignore[assignment]

from .timing import as_spans

__all__ = [
    "clipping_stats",
    "truncation_check",
    "pitch_shift_cents",
    "roundtrip_wer",
    "vad_xcorr",
    "overlap_stats",
]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _require(pkg: str, pip_name: Optional[str] = None):
    """Import `pkg` or raise an ImportError that names the pip package.

    Same shim as `backend.transcripts._require`, duplicated rather than
    imported so these modules stay usable without the transcript stack.
    """
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError as e:
        raise ImportError(
            f"Audio evaluation needs `{pip_name or pkg}`. "
            f"Install it with `pip install {pip_name or pkg}`."
        ) from e


def _read_wav(src: Any) -> Tuple[np.ndarray, int]:
    """Load audio as float32 (n,) mono-max or (n, ch), plus sample rate.

    Accepts a path, an `(array, sr)` pair, or an already-loaded array with a
    separate sr supplied by the caller through the pair form.
    """
    if isinstance(src, tuple) and len(src) == 2 and not isinstance(src[0], (str, Path)):
        x = np.asarray(src[0], dtype=np.float64)
        return x, int(src[1])
    sf = _require("soundfile")
    x, sr = sf.read(str(src), dtype="float64", always_2d=False)
    return np.asarray(x, dtype=np.float64), int(sr)


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(axis=1)


def _ffprobe_bin() -> Optional[str]:
    return (os.environ.get("FFPROBE_BIN")
            or shutil.which("ffprobe")
            or shutil.which("ffprobe.exe"))


def _probe_audio_duration(path: Path) -> Tuple[Optional[float], Optional[str]]:
    """Duration of the FIRST AUDIO STREAM, not the container.

    The container duration of a muxed mp4 is the video length, which is
    exactly the number that hides an `-shortest` audio truncation.
    """
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        return None, "ffprobe not found on PATH (set FFPROBE_BIN)"
    cmd = [ffprobe, "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=duration,nb_samples,sample_rate",
           "-show_entries", "format=duration",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError) as e:
        return None, f"ffprobe failed on {path}: {e}"
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        return None, f"ffprobe emitted unparseable JSON for {path}: {e}"
    streams = data.get("streams") or []
    if streams:
        st = streams[0]
        d = st.get("duration")
        if d not in (None, "N/A"):
            return float(d), None
        # Some mp4 audio streams carry no duration tag; samples/rate is exact.
        ns, sr = st.get("nb_samples"), st.get("sample_rate")
        if ns not in (None, "N/A") and sr not in (None, "N/A") and float(sr) > 0:
            return float(ns) / float(sr), None
    fd = (data.get("format") or {}).get("duration")
    if fd not in (None, "N/A"):
        return float(fd), f"{path.name}: no audio-stream duration; used container duration"
    return None, f"{path} has no audio stream duration ffprobe could report"


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def clipping_stats(wav: Any) -> Dict[str, Any]:
    """Sample-peak and inter-sample (true) peak clipping of a rendered mix.

    True peak is measured on a 4x-oversampled copy: a signal can sit at
    0.99 on every sample and still reconstruct above 0 dBFS between them,
    which is what the listener's DAC actually produces. Reporting only the
    sample peak lets an audibly distorted mix pass.
    """
    x, sr = _read_wav(wav)
    m = _to_mono(x) if x.ndim > 1 else x
    # Clipping is per-sample across channels, so take the loudest channel.
    peak_src = np.max(np.abs(x), axis=1) if x.ndim > 1 else np.abs(x)

    n = int(peak_src.size)
    out: Dict[str, Any] = {
        "sample_rate": sr,
        "n_samples": n,
        "n_channels": int(x.shape[1]) if x.ndim > 1 else 1,
        "duration_s": (n / sr) if sr else None,
        "not_measured": [],
    }
    if n == 0:
        out.update({"max_abs": None, "n_clipped": 0, "pct_clipped": None,
                    "sample_peak_dbfs": None, "true_peak": None,
                    "true_peak_dbtp": None, "true_peak_over_0dbfs": None})
        out["not_measured"].append({
            "metric": "clipping_stats", "reason": "empty audio (0 samples)"})
        return out

    max_abs = float(peak_src.max())
    n_clip = int(np.count_nonzero(peak_src >= 0.999))
    out["max_abs"] = max_abs
    out["n_clipped"] = n_clip
    out["pct_clipped"] = 100.0 * n_clip / n
    out["sample_peak_dbfs"] = (20.0 * math.log10(max_abs)) if max_abs > 0 else None

    try:
        sps = _require("scipy.signal", "scipy")
        up = sps.resample_poly(m, 4, 1)
        tp = float(np.max(np.abs(up)))
        out["true_peak"] = tp
        out["true_peak_dbtp"] = (20.0 * math.log10(tp)) if tp > 0 else None
        out["true_peak_over_0dbfs"] = bool(tp > 1.0)
        out["oversample_factor"] = 4
    except Exception as e:  # noqa: BLE001
        out.update({"true_peak": None, "true_peak_dbtp": None,
                    "true_peak_over_0dbfs": None})
        out["not_measured"].append({
            "metric": "true_peak", "reason": f"4x oversampling failed: {e}"})
    return out


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def truncation_check(
    mix_wav: Any,
    mix_mp4: Any,
    *,
    tol_s: float = 0.05,
) -> Dict[str, Any]:
    """Did the video mux drop the tail of the dubbed audio?

    `delta_s = dur(wav) - dur(mp4 audio)`. Positive means the mp4 is shorter
    than the audio it was given — the classic `-shortest` symptom, where the
    dub overran the source video and the last line simply is not in the file.
    """
    wav_p = Path(mix_wav)
    mp4_p = Path(mix_mp4)
    out: Dict[str, Any] = {
        "wav": str(wav_p), "mp4": str(mp4_p),
        "tol_s": tol_s,
        "wav_duration_s": None, "mp4_audio_duration_s": None,
        "delta_s": None, "truncated": None,
        "not_measured": [],
    }

    if not wav_p.exists():
        out["not_measured"].append({
            "metric": "truncation_check", "reason": f"missing wav: {wav_p}"})
    else:
        d, err = _probe_audio_duration(wav_p)
        if d is None:
            # soundfile reads the wav header directly, no ffprobe needed.
            try:
                sf = _require("soundfile")
                info = sf.info(str(wav_p))
                d = float(info.frames) / float(info.samplerate)
                err = None
            except Exception as e:  # noqa: BLE001
                err = f"{err or ''}; soundfile fallback failed: {e}"
        out["wav_duration_s"] = d
        if d is None:
            out["not_measured"].append({
                "metric": "wav_duration_s", "reason": err or "unknown"})
        elif err:
            out["not_measured"].append({"metric": "wav_duration_s.note",
                                        "reason": err})

    if not mp4_p.exists():
        out["not_measured"].append({
            "metric": "truncation_check", "reason": f"missing mp4: {mp4_p}"})
    else:
        d, err = _probe_audio_duration(mp4_p)
        out["mp4_audio_duration_s"] = d
        if d is None:
            out["not_measured"].append({
                "metric": "mp4_audio_duration_s", "reason": err or "unknown"})
        elif err:
            out["not_measured"].append({"metric": "mp4_audio_duration_s.note",
                                        "reason": err})

    a, b = out["wav_duration_s"], out["mp4_audio_duration_s"]
    if a is not None and b is not None:
        out["delta_s"] = a - b
        out["truncated"] = bool((a - b) > tol_s)
        out["lost_tail_s"] = max(0.0, a - b)
    else:
        out["not_measured"].append({
            "metric": "delta_s",
            "reason": "need both durations; one or both could not be probed"})
    return out


# ---------------------------------------------------------------------------
# Pitch
# ---------------------------------------------------------------------------

def _voiced_f0(y: np.ndarray, sr: int, *, fmin: float, fmax: float,
               frame_length: int = 2048) -> Tuple[np.ndarray, int, int]:
    """librosa.yin f0 restricted to plausibly voiced frames.

    yin returns a pitch for every frame including silence, so the frames are
    gated two ways: RMS above a fraction of the frame-RMS median (drops
    silence) and f0 strictly inside (fmin, fmax) (drops octave/edge pinning).
    """
    librosa = _require("librosa")
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr,
                     frame_length=frame_length)
    rms = librosa.feature.rms(y=y, frame_length=frame_length,
                              hop_length=frame_length // 4)[0]
    n = min(f0.size, rms.size)
    f0, rms = f0[:n], rms[:n]
    if n == 0:
        return np.array([]), 0, 0
    thresh = 0.25 * float(np.median(rms[rms > 0])) if np.any(rms > 0) else 0.0
    keep = (rms > thresh) & (f0 > fmin * 1.02) & (f0 < fmax * 0.98)
    return f0[keep], int(np.count_nonzero(keep)), n


def pitch_shift_cents(
    raw_wav: Any,
    stretched_wav: Any,
    *,
    fmin: float = 65.0,
    fmax: float = 500.0,
) -> Dict[str, Any]:
    """Pitch transposition introduced by a time-stretch, in cents.

    This is the decisive stretch-backend discriminator. A true time-stretcher
    (rubberband / atempo / librosa phase-vocoder) leaves pitch alone: ~0 cents.
    Naive resampling (`linear`) changes the playback rate, so a 1.10x stretch
    drops the voice by ~165 cents — over a semitone and plainly audible.

    `cents = 1200 * log2(median f0 stretched / median f0 raw)`, medians taken
    over voiced frames only. 100 cents = 1 semitone.

    Also reports `flux_ratio` (mean spectral flux per frame, stretched/raw):
    a resampler packs the same spectral events into fewer frames and so pushes
    this above 1, while a phase vocoder keeps it near 1.
    """
    out: Dict[str, Any] = {
        "raw": str(raw_wav) if isinstance(raw_wav, (str, Path)) else "<array>",
        "stretched": (str(stretched_wav)
                      if isinstance(stretched_wav, (str, Path)) else "<array>"),
        "fmin": fmin, "fmax": fmax,
        "cents": None, "semitones": None,
        "f0_median_raw_hz": None, "f0_median_stretched_hz": None,
        "n_voiced_raw": 0, "n_voiced_stretched": 0,
        "duration_ratio": None, "flux_ratio": None,
        "centroid_shift_cents": None,
        "not_measured": [],
    }

    try:
        yr, sr_r = _read_wav(raw_wav)
        ys, sr_s = _read_wav(stretched_wav)
    except Exception as e:  # noqa: BLE001
        out["not_measured"].append({
            "metric": "pitch_shift_cents", "reason": f"could not read audio: {e}"})
        return out

    yr, ys = _to_mono(yr), _to_mono(ys)
    out["sample_rate_raw"] = sr_r
    out["sample_rate_stretched"] = sr_s
    out["duration_raw_s"] = yr.size / sr_r if sr_r else None
    out["duration_stretched_s"] = ys.size / sr_s if sr_s else None
    if out["duration_raw_s"]:
        out["duration_ratio"] = out["duration_stretched_s"] / out["duration_raw_s"]

    try:
        f0r, nvr, _ = _voiced_f0(yr, sr_r, fmin=fmin, fmax=fmax)
        f0s, nvs, _ = _voiced_f0(ys, sr_s, fmin=fmin, fmax=fmax)
    except Exception as e:  # noqa: BLE001
        out["not_measured"].append({
            "metric": "pitch_shift_cents", "reason": f"yin f0 failed: {e}"})
        return out

    out["n_voiced_raw"] = nvr
    out["n_voiced_stretched"] = nvs
    if nvr == 0 or nvs == 0:
        out["not_measured"].append({
            "metric": "pitch_shift_cents",
            "reason": f"no voiced frames (raw={nvr}, stretched={nvs}); "
                      f"f0 ratio is undefined"})
    else:
        mr, ms = float(np.median(f0r)), float(np.median(f0s))
        out["f0_median_raw_hz"] = mr
        out["f0_median_stretched_hz"] = ms
        out["cents"] = 1200.0 * math.log2(ms / mr)
        out["semitones"] = out["cents"] / 100.0

    try:
        librosa = _require("librosa")
        fr = librosa.onset.onset_strength(y=yr, sr=sr_r)
        fs = librosa.onset.onset_strength(y=ys, sr=sr_s)
        mr_f, ms_f = float(np.mean(fr)), float(np.mean(fs))
        out["flux_mean_raw"] = mr_f
        out["flux_mean_stretched"] = ms_f
        out["flux_ratio"] = (ms_f / mr_f) if mr_f > 0 else None
        cr = float(np.median(librosa.feature.spectral_centroid(y=yr, sr=sr_r)))
        cs = float(np.median(librosa.feature.spectral_centroid(y=ys, sr=sr_s)))
        out["centroid_raw_hz"] = cr
        out["centroid_stretched_hz"] = cs
        # Reported as context only, NOT as corroboration of the f0 reading.
        # Measured on the real edge-TTS artifacts, an upshift of +165 cents
        # moved the centroid *down* 34-84 cents: resampling pushes the top
        # of the band past Nyquist where it is discarded, and the centroid
        # is energy-weighted so losing that content drags it the wrong way.
        # `cents` is the verdict; this number must not be read as a check.
        out["centroid_shift_cents"] = (1200.0 * math.log2(cs / cr)
                                       if cr > 0 and cs > 0 else None)
    except Exception as e:  # noqa: BLE001
        out["not_measured"].append({
            "metric": "flux_ratio", "reason": f"spectral features failed: {e}"})
    return out


# ---------------------------------------------------------------------------
# Round-trip ASR
# ---------------------------------------------------------------------------

# Cached across calls: a sweep over 18 clips would otherwise pay the
# large-v3 load (tens of seconds, several GB) once per clip.
_WX_CACHE: Dict[Tuple[str, str, str], Any] = {}


def _whisperx_model(model: str, device: Optional[str], compute_type: Optional[str]):
    whisperx = _require("whisperx")
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    device = "cuda" if str(device).startswith("cuda") else "cpu"
    compute_type = compute_type or ("float16" if device == "cuda" else "float32")
    key = (model, device, compute_type)
    if key not in _WX_CACHE:
        log.info("roundtrip_wer: loading whisperx %s (device=%s, compute=%s)",
                 model, device, compute_type)
        _WX_CACHE[key] = whisperx.load_model(
            model, device=device, compute_type=compute_type)
    return _WX_CACHE[key], device, compute_type


def roundtrip_wer(
    wav: Any,
    expected_text: str,
    *,
    language: str = "hi",
    model: str = "large-v3",
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Re-transcribe a rendered dub and score it against the text that was sent to TTS.

    This answers "is the dub still intelligible after stretching/mixing?" —
    a stretch that saves the timing budget by squeezing 14 s of speech into
    12 s shows up here as a WER cliff even though every timing metric is green.

    RUN THIS ON THE DIALOGUE-ONLY RENDER, never on the M&E mix. Scoring the
    full mix measures how well HTDemucs separated the music bed, not how the
    timing method treated the voice, and the two move independently.
    """
    out: Dict[str, Any] = {
        "wav": str(wav), "language": language, "model": model,
        "wer": None, "cer": None, "hypothesis": None,
        "reference": expected_text,
        "warning": "must be run on the dialogue-only render, not the M&E mix",
        "not_measured": [],
    }
    p = Path(wav)
    if not p.exists():
        out["not_measured"].append({
            "metric": "roundtrip_wer", "reason": f"missing audio: {p}"})
        return out
    if not (expected_text or "").strip():
        out["not_measured"].append({
            "metric": "roundtrip_wer", "reason": "expected_text is empty"})
        return out

    try:
        whisperx = _require("whisperx")
        asr, dev, ct = _whisperx_model(model, device, compute_type)
        out["device"], out["compute_type"] = dev, ct
        audio = whisperx.load_audio(str(p))
        res = asr.transcribe(audio, batch_size=batch_size, language=language)
        hyp = " ".join(s.get("text", "").strip()
                       for s in res.get("segments", [])).strip()
    except Exception as e:  # noqa: BLE001
        out["not_measured"].append({
            "metric": "roundtrip_wer", "reason": f"whisperx transcription failed: {e}"})
        return out

    out["hypothesis"] = hyp
    out["detected_language"] = res.get("language")
    if not hyp:
        out["not_measured"].append({
            "metric": "roundtrip_wer",
            "reason": "whisperx returned no text for this audio"})
        return out

    try:
        jiwer = _require("jiwer")
        out["wer"] = float(jiwer.wer(expected_text, hyp))
        out["cer"] = float(jiwer.cer(expected_text, hyp))
    except Exception as e:  # noqa: BLE001
        out["not_measured"].append({
            "metric": "wer", "reason": f"jiwer scoring failed: {e}"})
    return out


# ---------------------------------------------------------------------------
# Cross-correlation lag
# ---------------------------------------------------------------------------

def vad_xcorr(
    hyp_mask,
    ref_mask,
    *,
    frame_s: float = 0.01,
    max_lag_s: float = 3.0,
) -> Dict[str, Any]:
    """Best global time offset between two speech-activity masks.

    Complements `offset_stats`: that one needs segments to match up, this one
    works on raw activity and finds a constant lag even when the segmentation
    disagrees. A large `peak_lag_ms` with a high `peak_correlation` means the
    dub is right but uniformly shifted — a fixable offset. A low peak
    correlation at any lag means no single shift saves it.

    Positive `peak_lag_ms` = the hypothesis is LATE relative to the reference.
    """
    a = np.asarray(hyp_mask).astype(np.float64).ravel()
    b = np.asarray(ref_mask).astype(np.float64).ravel()
    n = max(a.size, b.size)
    out: Dict[str, Any] = {
        "frame_s": frame_s, "max_lag_s": max_lag_s,
        "n_frames": int(n),
        "peak_lag_ms": None, "peak_correlation": None,
        "zero_lag_correlation": None,
        "sign_convention": "positive lag = hypothesis is late",
        "not_measured": [],
    }
    if n == 0:
        out["not_measured"].append({
            "metric": "vad_xcorr", "reason": "empty masks"})
        return out
    a = np.concatenate([a, np.zeros(n - a.size)])
    b = np.concatenate([b, np.zeros(n - b.size)])

    # Mean-removed and unit-normalised so the peak is a Pearson correlation
    # at the best lag, not an artefact of how much speech each side has.
    a = a - a.mean()
    b = b - b.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        out["not_measured"].append({
            "metric": "vad_xcorr",
            "reason": "a mask is constant (all speech or all silence); "
                      "correlation is undefined"})
        return out

    sps = _require("scipy.signal", "scipy")
    corr = sps.correlate(a, b, mode="full") / (na * nb)
    lags = np.arange(-(b.size - 1), a.size)
    lim = int(round(max_lag_s / frame_s))
    sel = np.abs(lags) <= lim
    if not np.any(sel):
        out["not_measured"].append({
            "metric": "vad_xcorr",
            "reason": f"max_lag_s={max_lag_s} is below one frame ({frame_s}s)"})
        return out
    c, l = corr[sel], lags[sel]
    k = int(np.argmax(c))
    out["peak_lag_ms"] = float(l[k] * frame_s * 1000.0)
    out["peak_correlation"] = float(c[k])
    zero = np.nonzero(l == 0)[0]
    out["zero_lag_correlation"] = float(c[zero[0]]) if zero.size else None
    out["correlation_gain_at_peak"] = (
        out["peak_correlation"] - out["zero_lag_correlation"]
        if out["zero_lag_correlation"] is not None else None)
    return out


# ---------------------------------------------------------------------------
# Placement overlap
# ---------------------------------------------------------------------------

_INTENDED_KEYS = ("intended_start", "ref_start", "orig_start",
                  "source_start", "start_s")


def overlap_stats(placements: Iterable[Any]) -> Dict[str, Any]:
    """How much dubbed audio plays on top of other dubbed audio, and how far clips slid.

    Accepts `mix.py`-style plan rows `(id, placed, wav, raw_dur, drift)`,
    dicts with start/end (or start/duration), or anything with .start/.end.

    `total_overlap_s` is the audible damage: two voices at once. `shift_*` is
    the cost of avoiding it — the shift-forward mixer never overlaps, it pays
    in displacement instead, and a report that shows only one of the two makes
    whichever strategy it favours look free.
    """
    rows = list(placements or [])
    spans = as_spans(rows)
    out: Dict[str, Any] = {
        "n_placements": len(rows),
        "n_timed": len(spans),
        "total_overlap_s": 0.0, "n_overlapping_pairs": 0,
        "n_segments_overlapped": 0,
        "max_overlap_s": None, "mean_overlap_s": None,
        "total_speech_s": float(sum(s.dur for s in spans)),
        "pct_speech_overlapped": None,
        "shift_median_s": None, "shift_p95_s": None, "shift_max_s": None,
        "not_measured": [],
    }
    if len(rows) != len(spans):
        out["not_measured"].append({
            "metric": "overlap_stats",
            "reason": f"{len(rows) - len(spans)} placements carried no usable "
                      f"start/end and were excluded"})
    if len(spans) < 2:
        out["not_measured"].append({
            "metric": "overlap_stats",
            "reason": f"need >= 2 timed placements, got {len(spans)}"})
        return out

    order = sorted(range(len(spans)), key=lambda i: (spans[i].start, spans[i].end))
    overlaps: List[float] = []
    touched: set = set()
    # Pairwise against every later-starting span, not just the neighbour: a
    # long clip can sit under several short ones.
    for ii, i in enumerate(order):
        for j in order[ii + 1:]:
            if spans[j].start >= spans[i].end:
                break
            ov = min(spans[i].end, spans[j].end) - max(spans[i].start, spans[j].start)
            if ov > 0:
                overlaps.append(ov)
                touched.add(i)
                touched.add(j)

    out["n_overlapping_pairs"] = len(overlaps)
    out["n_segments_overlapped"] = len(touched)
    out["total_overlap_s"] = float(sum(overlaps))
    out["max_overlap_s"] = float(max(overlaps)) if overlaps else 0.0
    out["mean_overlap_s"] = float(np.mean(overlaps)) if overlaps else 0.0
    if out["total_speech_s"] > 0:
        out["pct_speech_overlapped"] = 100.0 * out["total_overlap_s"] / out["total_speech_s"]

    shifts: List[float] = []
    for row in rows:
        if isinstance(row, dict):
            for k in ("drift", "drift_s", "shift", "shift_s"):
                if row.get(k) is not None:
                    shifts.append(float(row[k]))
                    break
            else:
                sp = _INTENDED_KEYS
                base = next((float(row[k]) for k in sp
                             if row.get(k) is not None), None)
                got = next((float(row[k]) for k in ("start", "placed")
                            if row.get(k) is not None), None)
                if base is not None and got is not None:
                    shifts.append(got - base)
        elif isinstance(row, (tuple, list)) and len(row) >= 5:
            # mix.py plan row: trailing element is `placed - intended_start`.
            try:
                shifts.append(float(row[4]))
            except (TypeError, ValueError):
                pass

    if shifts:
        a = np.abs(np.asarray(shifts, dtype=float))
        out["n_shifted"] = len(shifts)
        out["shift_median_s"] = float(np.median(a))
        out["shift_p95_s"] = float(np.percentile(a, 95))
        out["shift_max_s"] = float(a.max())
        out["shift_mean_signed_s"] = float(np.mean(shifts))
    else:
        out["not_measured"].append({
            "metric": "shift_stats",
            "reason": "placements carry no drift/intended-start field, so "
                      "displacement from the source timeline is unknown"})
    return out
