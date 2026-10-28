"""Shared face-crop + VRAM-windowing front end for the ASD adapters.

An ASD adapter needs 112x112 grayscale face crops at the track frame rate,
decoded once for every track in the video, and a way to cut a long video
into windows that fit the card. Both live here so an adapter does not
re-derive them.

**The MFCC half was removed on 2026-10-28**, with the three MFCC-based
adapters (talknet, light_asd, lr_asd). LoCoNet takes VGGish audio, not MFCC,
so it shares only the crop half. If an MFCC adapter ever comes back, the
rule that made it work is worth repeating: the audio must stay on the
**int16 value scale**. Those upstreams read their wav with
`scipy.io.wavfile.read` (int16 for 16-bit PCM) and pass it straight to
`python_speech_features.mfcc`; read the same file as floats in [-1, 1] and
every score collapses toward zero without raising -- measured mean 0.044 vs
0.411 on the same clip. See docs/ROUGH_EDGES.md.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

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
    "crop_all_tracks",
    "crop_one",
    "window_bounds",
]
