"""Shared MFCC + face-crop front end for the MFCC-based ASD adapters.

TalkNet and Light-ASD are different architectures taking the same two
inputs: 112x112 grayscale face crops at the track frame rate, and 13-d MFCC
over int16-valued PCM. Everything up to the first `forward_*` call is here.

Audio must stay on the int16 value scale. Both upstreams read their wav with
`scipy.io.wavfile.read` (int16 for 16-bit PCM) and pass it straight to
`python_speech_features.mfcc`. Read the same file as floats in [-1, 1] and
every score collapses toward zero without raising -- measured mean 0.044 vs
0.411 on the same clip. Use `load_audio_int16`; do not rescale in an
adapter. See docs/ROUGH_EDGES.md.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: `python_speech_features.mfcc` emits one row per 10 ms, whatever the fps.
MFCC_HZ = 100.0

#: int16 full scale, so the call site reads as a unit conversion.
INT16_SCALE = 32768.0


def load_audio_int16(audio_path: Path, *, model_name: str = "asd") -> np.ndarray:
    """Mono 16 kHz audio on the int16 value scale the MFCC front-ends expect.

    Returns float32: what has to match is the magnitude, not the storage
    type.
    """
    import soundfile as sf

    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(
            f"{model_name} expects 16 kHz audio; got {sr}. Resample upstream."
        )
    return audio * INT16_SCALE


def mfcc_features(audio: np.ndarray, *, fps: Optional[float] = None) -> np.ndarray:
    """13-d MFCC, optionally with the window scaled to a non-25 fps video.

    `fps=None` keeps `winlen=0.025 / winstep=0.010` fixed, which is what
    TalkNet's upstream does (it resamples the video to 25 fps first).
    Light-ASD instead scales both by `25 / fps`, so pass `fps` for that
    family.

    The asymmetry is deliberate -- each adapter matches the upstream it
    wraps, and at 25 fps the two are identical.
    """
    from python_speech_features import mfcc as mfcc_fn   # upstream dep

    winlen, winstep = 0.025, 0.010
    if fps:
        winlen = winlen * 25.0 / float(fps)
        winstep = winstep * 25.0 / float(fps)
    return mfcc_fn(audio, 16000, numcep=13, winlen=winlen, winstep=winstep)


def window_bounds(n_frames: int, window_frames: int,
                  *, min_tail: int = 8) -> List[Tuple[int, int]]:
    """Split a track into `[start, stop)` slices of at most `window_frames`.

    `window_frames <= 0` means one pass over the whole track. A tail shorter
    than `min_tail` is merged backwards.

    Windowing is a VRAM bound, not a tuning knob: the visual frontend holds
    ~0.77 MiB per frame, so a 10575-frame track asks for 7.91 GiB in one
    `batch_norm` and OOMs.
    """
    if window_frames <= 0 or n_frames <= window_frames:
        return [(0, n_frames)]
    bounds = [(i, min(i + window_frames, n_frames))
              for i in range(0, n_frames, window_frames)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < min_tail:
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    return bounds


def align_pair(audio_embed, video_embed, *, who: str = "asd"):
    """Trim two (B, T, ...) embeddings to their common T.

    The audio frontend downsamples 100 Hz MFCC by 4 and the visual one emits
    one embedding per crop, so the two disagree by up to a frame.

    Mandatory for Light-ASD, whose `forward_audio_visual_backend` is
    `x = x1 + x2`: a mismatch there raises rather than degrading.
    """
    t = min(audio_embed.shape[1], video_embed.shape[1])
    if audio_embed.shape[1] != video_embed.shape[1]:
        log.debug("%s: trimming a=%d v=%d -> %d", who,
                  audio_embed.shape[1], video_embed.shape[1], t)
    return audio_embed[:, :t], video_embed[:, :t]


def crop_all_tracks(
    tracks: Sequence,
    video_path: Path,
    size: int,
    cv2,
) -> Dict[str, np.ndarray]:
    """Sweep the video once and crop every frame each track needs.

    Returns {track_id: (N, size, size) uint8 grayscale}, in the same frame
    order as the track's `.frames`. A frame the decoder skipped is dropped;
    the caller trims scores to `min(len(crops), track.num_frames)`.

    One sweep for all tracks, not one per track -- the per-track shape is
    O(N * video_len) decode and ran for hours on 262-track footage.
    """
    buckets: Dict[int, list] = defaultdict(list)
    out: Dict[str, List[Optional[np.ndarray]]] = {}
    for tr in tracks:
        out[tr.track_id] = [None] * len(tr.frames)
        for pos, (f, bb) in enumerate(zip(tr.frames, tr.bboxes)):
            buckets[int(f)].append((tr.track_id, bb, pos))

    if not buckets:
        return {tid: np.zeros((0, size, size), dtype=np.uint8) for tid in out}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    max_needed = max(buckets)
    idx = 0
    try:
        while idx <= max_needed:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in buckets:
                for tid, bb, pos in buckets[idx]:
                    out[tid][pos] = crop_one(frame, bb, size, cv2)
            idx += 1
    finally:
        cap.release()

    packed: Dict[str, np.ndarray] = {}
    for tid, crops in out.items():
        valid = [c for c in crops if c is not None]
        packed[tid] = (np.stack(valid) if valid
                       else np.zeros((0, size, size), dtype=np.uint8))
    return packed


def crop_one(frame: np.ndarray, bb, size: int, cv2) -> np.ndarray:
    """One grayscale face patch, expanded 1.4x around the box.

    Both upstream encoders are trained on loose crops, so a tight box is a
    different input distribution.
    """
    x1, y1, x2, y2 = bb
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 1.4
    xa = int(max(0, cx - side / 2))
    ya = int(max(0, cy - side / 2))
    xb = int(min(frame.shape[1], cx + side / 2))
    yb = int(min(frame.shape[0], cy + side / 2))
    patch = frame[ya:yb, xa:xb]
    if patch.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)


__all__ = [
    "INT16_SCALE",
    "MFCC_HZ",
    "align_pair",
    "crop_all_tracks",
    "crop_one",
    "load_audio_int16",
    "mfcc_features",
    "window_bounds",
]
