"""Shared face-crop and VRAM-windowing front end for the ASD adapters.

Decodes 112x112 grayscale face crops once per video for every track, and cuts
a long track into windows that fit the card.

Any adapter added here must keep its audio on the INT16 value scale. Reading
the same wav as floats in [-1, 1] collapses every score toward zero without
raising — see docs/ROUGH_EDGES.md.
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

    `window_frames <= 0` means one pass over the whole track, prevents OOMs of VRAM.
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

    Returns `{track_id: (N, size, size) uint8 grayscale}` in the track's own
    frame order; a frame the decoder skipped is dropped, so the caller trims
    scores to `min(len(crops), track.num_frames)`. One sweep for all tracks —
    per-track decoding is O(N * video_len) and runs for hours on busy footage.
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

    The upstream encoders are trained on loose crops, so a tight box is a
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
