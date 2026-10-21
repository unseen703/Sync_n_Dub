"""GPU discovery + ffmpeg media sharding helpers for multi-GPU stages.

Only the lip-sync stage currently uses this — the rest of the pipeline is
sequential and dependency-bound, so adding more GPUs would not help it.
Lip-sync is embarrassingly parallel in time: split the input video (and
driving audio) into N equal-duration chunks, run one adapter per GPU, then
concat the outputs. Each adapter does its own per-frame face detection, so
chunks are independent.

Env overrides:
  DUBBING_GPUS   comma-separated CUDA ids to pin to (e.g. "0,1"); when set,
                 overrides automatic enumeration.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from .media_io import _ffmpeg_bin

log = logging.getLogger(__name__)


_ORT_CUDA_READY: Optional[bool] = None


def ensure_onnxruntime_cuda() -> bool:
    """Make onnxruntime's CUDA EP actually loadable, and say whether it is.

    `onnxruntime-gpu` ships `libonnxruntime_providers_cuda.so`, which links
    against `libcublasLt.so.12`, `libcudnn.so.9` and friends. On this host
    those live inside torch's `nvidia-*` pip packages rather than anywhere on
    the default loader path, so the provider library fails to dlopen. The
    failure is quiet in the way that matters: `get_available_providers()` still
    lists `CUDAExecutionProvider`, and only `InferenceSession.get_providers()`
    afterwards reveals it silently fell back to CPU.

    `ort.preload_dlls()` dlopens those libraries from the nvidia wheels first,
    which is enough. Measured on `buffalo_l/det_10g.onnx` at 640x640:
    55.9 ms/infer on CPU versus 6.0 ms once the CUDA EP binds.

    Idempotent, and safe to call when onnxruntime is absent or too old to have
    `preload_dlls` -- it just returns False and the caller carries on.
    """
    global _ORT_CUDA_READY
    if _ORT_CUDA_READY is not None:
        return _ORT_CUDA_READY
    try:
        import onnxruntime as ort
    except ImportError:
        _ORT_CUDA_READY = False
        return False
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        try:
            preload()
        except Exception as e:                      # never fatal
            log.debug("onnxruntime preload_dlls failed: %s", e)
    ready = "CUDAExecutionProvider" in ort.get_available_providers()
    if not ready:
        log.warning(
            "onnxruntime has no CUDAExecutionProvider; stages using it will "
            "run on CPU. Check that onnxruntime-gpu (not onnxruntime) is the "
            "installed package."
        )
    _ORT_CUDA_READY = ready
    return ready


def visible_cuda_devices() -> List[int]:
    """Return CUDA device ids visible to this process.

    Honors `DUBBING_GPUS` (manual pin list) first, then
    `CUDA_VISIBLE_DEVICES` as remapped by the driver, then falls back to
    `torch.cuda.device_count()`. Returns [] when no GPUs are available.
    """
    env = os.getenv("DUBBING_GPUS", "").strip()
    if env:
        return [int(x) for x in env.split(",") if x.strip().isdigit()]

    cvd = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        # Already remapped: when CUDA_VISIBLE_DEVICES=2,5, torch sees them
        # as 0 and 1. We shard by position, not by physical id.
        return list(range(len([x for x in cvd.split(",") if x.strip()])))

    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass
    return []


def gpu_busy_pids() -> dict:
    """{device id -> [pids]} for compute processes that are NOT ours.

    "Free" means nobody else is on the card, not "has N GiB headroom". A
    fixed free-VRAM floor was tried and removed: our OWN process holds a
    CUDA context on whichever card it touched earlier in the pipeline
    (measured ~1.9-4.7 GiB for `ui.runner` after face detection), so a
    memory threshold reads the job's own context as a foreign tenant and
    refuses the card it is already using.

    Keyed in the same id space as `visible_cuda_devices()`; see
    `gpu_free_memory` for why the CUDA_VISIBLE_DEVICES remap matters.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        if proc.returncode != 0:
            return {}
        uuid_rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        phys = {}
        for line in (uuid_rows.stdout or "").strip().splitlines():
            idx, uu = [x.strip() for x in line.split(",", 1)]
            phys[uu] = int(idx)
        ours = _own_pids()
        busy = {}
        for line in (proc.stdout or "").strip().splitlines():
            uu, pid = [x.strip() for x in line.split(",")[:2]]
            if not pid.isdigit() or int(pid) in ours:
                continue
            g = phys.get(uu)
            if g is not None:
                busy.setdefault(g, []).append(int(pid))
    except Exception:                                    # noqa: BLE001
        return {}

    cvd = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and not os.getenv("DUBBING_GPUS", "").strip():
        ids = [int(x) for x in cvd.split(",") if x.strip().isdigit()]
        return {pos: busy[ph] for pos, ph in enumerate(ids) if ph in busy}
    return busy


def _own_pids() -> set:
    """This process and everything under it.

    Our own CUDA context must not make a card look occupied to us.
    """
    mine = {os.getpid(), os.getppid()}
    try:
        out = subprocess.run(["ps", "-eo", "pid,ppid", "--no-headers"],
                             capture_output=True, text=True, timeout=10)
        kids = {}
        for line in (out.stdout or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                kids.setdefault(int(parts[1]), []).append(int(parts[0]))
        stack = list(mine)
        while stack:
            for c in kids.get(stack.pop(), []):
                if c not in mine:
                    mine.add(c)
                    stack.append(c)
    except Exception:                                    # noqa: BLE001
        pass
    return mine


def free_devices() -> list:
    """Visible GPUs with no foreign compute process, in index order."""
    gpus = visible_cuda_devices()
    if not gpus:
        return []
    busy = gpu_busy_pids()
    return [g for g in gpus if not busy.get(g)]


def gpu_free_memory() -> dict:
    """{device id -> free MiB}, keyed in the SAME id space as
    `visible_cuda_devices()`.

    nvidia-smi reports PHYSICAL indices, but with `CUDA_VISIBLE_DEVICES=2,5`
    the process sees those as 0 and 1 -- so position i must be looked up
    under the physical id at position i, or the figures attach to the wrong
    cards. Kept for reporting and for the operator-facing message; it no
    longer gates selection.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        if proc.returncode != 0:
            return {}
        physical = {}
        for line in (proc.stdout or "").strip().splitlines():
            idx, free = [x.strip() for x in line.split(",")[:2]]
            physical[int(idx)] = int(free)
    except Exception:                                    # noqa: BLE001
        return {}

    cvd = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and not os.getenv("DUBBING_GPUS", "").strip():
        ids = [int(x) for x in cvd.split(",") if x.strip().isdigit()]
        return {pos: physical.get(phys, 0) for pos, phys in enumerate(ids)}
    return physical


def pick_devices(n: Optional[int] = None) -> List[int]:
    """Every free GPU by default; [] when none is free.

    Policy (set 2026-10-19): both cards free -> use both. One free -> use
    that one. None free -> return [] so the caller refuses the run and tells
    the operator, rather than queueing behind a stranger's job or OOMing
    beside it.

    `n` caps the count when a caller wants fewer. Returning [] is a REFUSAL,
    not "use the default device" -- an unpinned worker lands on whichever
    card the driver picks, which is exactly the collision this avoids.
    """
    gpus = visible_cuda_devices()
    if not gpus:
        return []
    free = free_devices()
    if not free:
        log.warning("gpu: every visible device has another process on it "
                    "(%s); refusing to start",
                    ", ".join(f"gpu{g}" for g in gpus))
        return []
    if len(free) < len(gpus):
        log.info("gpu: %d of %d device(s) free: %s",
                 len(free), len(gpus), ", ".join(f"gpu{g}" for g in free))
    want = len(free) if n is None else max(1, int(n))
    return free[:min(want, len(free))]


def probe_duration(media: Path) -> float:
    """Duration in seconds via ffprobe, 0.0 if unavailable."""
    # media_io._ffprobe_bin() rather than shutil.which: there is no
    # ffprobe on this host's PATH, and which() returning None here
    # silently degrades to "unknown" instead of failing (CLAUDE.md §1).
    from .media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        return 0.0
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(media)],
            capture_output=True, text=True, check=True,
        )
        return float(proc.stdout.strip() or 0.0)
    except Exception:
        return 0.0


def split_video_audio(
    video: Path,
    audio: Path,
    n: int,
    out_dir: Path,
    *,
    min_chunk_s: float = 15.0,
) -> List[Tuple[Path, Path]]:
    """Split (video, audio) into N frame-accurate chunks of equal duration.

    Returns a list of (video_chunk, audio_chunk) pairs. Video is re-encoded
    (libx264 ultrafast CRF 18) so each chunk starts on a keyframe — a
    stream-copy split would snap to the nearest prior keyframe and produce
    ragged boundaries that an audio-driven model would see as drift.

    Falls back to a single-chunk list when `n <= 1`, the clip is shorter
    than `n * min_chunk_s`, or ffprobe cannot read the duration.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dur = probe_duration(video)
    if n <= 1 or dur <= 0.0 or dur < n * min_chunk_s:
        return [(Path(video), Path(audio))]

    ff = _ffmpeg_bin()
    chunk = dur / n
    pairs: List[Tuple[Path, Path]] = []
    for i in range(n):
        start = i * chunk
        # Last chunk: no -t so we take whatever is left (avoids losing a
        # final frame to float rounding).
        length_args = [] if i == n - 1 else ["-t", f"{chunk:.6f}"]

        v_out = out_dir / f"part{i:02d}.mp4"
        a_out = out_dir / f"part{i:02d}.wav"

        v_cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{start:.6f}", "-i", str(video), *length_args,
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                 "-an", "-pix_fmt", "yuv420p", str(v_out)]
        a_cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{start:.6f}", "-i", str(audio), *length_args,
                 "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(a_out)]

        for cmd in (v_cmd, a_cmd):
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg split failed (rc={proc.returncode}): "
                    f"{proc.stderr.strip()[:400]}"
                )
        pairs.append((v_out, a_out))
    log.info("gpu.split: %s → %d chunks of ~%.2fs", video.name, n, chunk)
    return pairs


def concat_videos(parts: List[Path], out: Path) -> Path:
    """Concatenate chunk mp4s via ffmpeg's concat demuxer.

    All inputs must share codec parameters — guaranteed because
    `split_video_audio` re-encodes them with identical settings and each
    lip-sync adapter re-encodes with its own defaults (we re-encode once
    more on concat to normalize).
    """
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg_bin()
    listing = out.with_suffix(".concat.txt")
    listing.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in parts))
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", str(listing),
           "-c:v", "libx264", "-preset", "fast", "-crf", "18",
           "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
           str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:400]}"
        )
    try:
        listing.unlink()
    except OSError:
        pass
    return out
