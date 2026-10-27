"""What every MFCC+crop ASD adapter here does identically.

TalkNet and Light-ASD are different architectures that take the SAME two
inputs: 112x112 grayscale face crops at the track's frame rate, and 13-d
MFCC over int16-valued PCM. Everything up to the first `forward_*` call is
shared, and this module is that part.

It exists because the alternative was tried. `ts_talknet_adapter.py` held
its own copy of this code and missed BOTH halves of the 2026-10-11 TalkNet
fix -- the `_crop_track` -> `_crop_all_tracks` split and the int16 rescale --
so it sat broken, and would have scored near-zero had it loaded at all. It
was deleted on 2026-10-23. A third hand-maintained copy is the same bet a
third time.

**The int16 rescale is the thing this module is really protecting.**
Both upstreams read their wav with `scipy.io.wavfile.read`, which yields
int16 for 16-bit PCM, and hand that straight to
`python_speech_features.mfcc`. Light-ASD is explicit about it --
`dataLoader.py` ends its audio path with `return audio.astype(numpy.int16)`.
Read the same file with `soundfile` and you get floats in [-1, 1], every
log-energy cepstral term shifts by ~log(32768), and every score collapses
toward zero WITHOUT RAISING. Measured on one 3 s single-speaker close-up:
mean 0.044 / max 0.129 / 0 of 75 frames above 0.5 as floats, against mean
0.411 / max 0.798 / 31 of 75 once rescaled. A stage that answers "nobody is
speaking" for every face, at exit code 0, is the most expensive failure in
this directory -- see docs/ROUGH_EDGES.md.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: `python_speech_features.mfcc` emits one row per 10 ms, whatever the video
#: fps. Both upstreams hard-code the 100-vs-25 ratio this represents.
MFCC_HZ = 100.0

#: int16 full scale. The rescale factor, named so the call site reads as a
#: unit conversion rather than a magic number.
INT16_SCALE = 32768.0


def load_audio_int16(audio_path: Path, *, model_name: str = "asd") -> np.ndarray:
    """Mono 16 kHz audio on the int16 VALUE SCALE the MFCC front-ends expect.

    Returns float32 (not int16 dtype) because `python_speech_features` does
    its own arithmetic in float anyway; what has to match is the magnitude,
    not the storage type. See this module's docstring for what happens when
    it does not.
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
    TalkNet's upstream does -- it resamples the video to 25 fps first and so
    never needs the correction. Light-ASD instead scales both by `25 / fps`
    (`dataLoader.py::load_audio`) so the MFCC grid lines up with the visual
    stream at whatever rate the source runs.

    The asymmetry is deliberate rather than an oversight to tidy up later:
    at 25 fps the two are identical, and away from 25 each adapter matches
    the upstream it wraps. Making TalkNet scale too would change the output
    of a working stage on every non-25 fps source.
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

    `window_frames <= 0` means one pass over the whole track.

    Windowing is a hard VRAM bound rather than a tuning knob: the visual
    frontend holds a (1, T, 64, 56, 56) float activation at ~0.77 MiB/frame
    and the crops are 112x112 whatever the source resolution, so T IS the
    memory story. A 7-minute single-face track (10575 frames) asked for
    7.91 GiB in one `batch_norm` and took job 20261024-155149 down at
    stage 6. Upstream never runs a whole track either -- TalkNet's
    `demoTalkNet.py`:237-245 cuts it into 1-6 s pieces.

    A tail shorter than `min_tail` is merged backwards: the 3D visual stem
    has a +/-2-frame kernel, so a stub window is scored almost entirely on
    padding and buys nothing.
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
    """Trim two (B, T, …) embeddings to their common T.

    The two frontends disagree by up to a frame on most clips: the audio one
    downsamples 100 Hz MFCC by 4, the visual one emits exactly one embedding
    per crop, and a window's frame span rarely divides into a whole number of
    MFCC windows. Upstream aligns earlier, on durations
    (`length = min((a.shape[0] - a.shape[0] % 4) / 100, v.shape[0] / 25)`);
    adapters that build their own tensors have to do it here instead.

    Trimming the tail is correct rather than merely safe -- the extra frame
    has no counterpart to attend to. It is also MANDATORY for Light-ASD,
    whose `forward_audio_visual_backend` is `x = x1 + x2`: a mismatch there
    is a broadcast error, not a graceful degradation.
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
    """Sweep the video once; dispatch each needed frame to every track that
    wants it.

    Returns {track_id: (N, size, size) uint8 grayscale array}, with crops in
    the same frame order as the track's `.frames` array. Any frame a track
    expected but the decoder skipped (e.g. truncated video) is dropped --
    the caller trims scores to `min(len(crops), track.num_frames)` so this
    is safe.

    One sweep, not one per track: the earlier shape reopened and resliced
    the video per track, i.e. O(N * video_len) decode, which ran for hours
    on busy footage (262 tracks).
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
    """One grayscale face patch, with the context the encoders were trained on.

    The 1.4x expansion is not cosmetic: both upstreams crop loosely and the
    visual encoders normalise against that framing, so a tight box is a
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
