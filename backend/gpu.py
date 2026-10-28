"""GPU discovery + ffmpeg media sharding helpers for multi-GPU stages.

Only the lip-sync stage uses this. It splits the input video and driving
audio into N equal-duration chunks, runs one adapter per GPU, then concats
the outputs; each adapter does its own per-frame face detection, so the
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
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .media_io import _ffmpeg_bin

log = logging.getLogger(__name__)


_ORT_CUDA_READY: Optional[bool] = None


def ensure_onnxruntime_cuda() -> bool:
    """Make onnxruntime's CUDA EP loadable, and say whether it is.

    `libonnxruntime_providers_cuda.so` links against `libcublasLt.so.12`,
    `libcudnn.so.9` and friends, which on this host live inside torch's
    `nvidia-*` pip packages rather than on the loader path, so the provider
    fails to dlopen. The failure is silent: `get_available_providers()` still
    lists `CUDAExecutionProvider` and only
    `InferenceSession.get_providers()` reveals the CPU fallback.
    `ort.preload_dlls()` fixes it (55.9 ms/infer on CPU vs 6.0 ms bound).

    Idempotent, and returns False rather than raising when onnxruntime is
    absent or too old to have `preload_dlls`.
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
    """CUDA device ids visible to this process, [] when there are none.

    `DUBBING_GPUS` (manual pin list) wins, then `CUDA_VISIBLE_DEVICES` as
    remapped by the driver, then `torch.cuda.device_count()`.
    """
    env = os.getenv("DUBBING_GPUS", "").strip()
    if env:
        return [int(x) for x in env.split(",") if x.strip().isdigit()]

    cvd = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        # Already remapped: with CUDA_VISIBLE_DEVICES=2,5 torch sees 0 and
        # 1. Shard by position, not by physical id.
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

    Ours are excluded because this process holds a CUDA context (~1.9-4.7
    GiB after face detection) on whichever card it touched earlier, and
    counting that would refuse the card it is already using.

    Keyed in the same id space as `visible_cuda_devices()`; see
    `gpu_free_memory` for the CUDA_VISIBLE_DEVICES remap.
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
    """This process and everything under it, so our own CUDA context does
    not make a card look occupied to us."""
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


#: Opt out of the one-tenant rule: every visible GPU counts as free and the
#: foreign tenants are only logged. Off by default.
#:
#: For the case the strict rule cannot see -- a neighbour that is resident
#: but idle. The risk is the operator's: the neighbour may grow and take one
#: of the two jobs with it.
GPU_ALLOW_SHARED_ENV = "DUBBING_GPU_ALLOW_SHARED"


def _allow_shared() -> bool:
    return (os.getenv(GPU_ALLOW_SHARED_ENV, "") or "").strip().lower() in (
        "1", "true", "yes", "on")


#: A card counts as free when BOTH numbers sit under these. 10% of a
#: 16311 MiB card is ~1631 MiB, so a resident-but-idle context (measured
#: 1117 MiB at 0% utilisation) passes while a working card does not.
#:
#: The memory figure is FOREIGN memory, not `memory.used`. This process
#: holds 1.9-4.7 GiB of its own CUDA context from face detection and ASD, so
#: a threshold over `memory.used` reads that as a stranger and refuses every
#: run that did any GPU work first.
GPU_BUSY_UTIL_PCT = 10.0        # override: DUBBING_GPU_BUSY_UTIL_PCT
GPU_BUSY_MEM_PCT = 10.0         # override: DUBBING_GPU_BUSY_MEM_PCT

#: `utilization.gpu` is an instantaneous sample, so a job between batches
#: reads 0%. Sampled `GPU_UTIL_SAMPLES` times and the MAX is taken -- a card
#: reading 0/0/90 is busy, and averaging would hide it.
GPU_UTIL_SAMPLES = 3
GPU_UTIL_SAMPLE_GAP_S = 0.2


def _pct_env(name: str, fallback: float) -> float:
    try:
        v = (os.getenv(name, "") or "").strip()
        return float(v) if v else fallback
    except ValueError:
        log.warning("gpu: %s is not a number; using %.1f", name, fallback)
        return fallback


def _remap_to_visible(physical: dict) -> dict:
    """Physical nvidia-smi ids -> the ids this process sees.

    With `CUDA_VISIBLE_DEVICES=2,5` the process sees those as 0 and 1, so
    position i is looked up under the physical id at position i. Getting it
    wrong attaches each card's figures to the other, and the gate then
    passes the busy card.
    """
    cvd = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and not os.getenv("DUBBING_GPUS", "").strip():
        ids = [int(x) for x in cvd.split(",") if x.strip().isdigit()]
        return {pos: physical[ph] for pos, ph in enumerate(ids)
                if ph in physical}
    return physical


def gpu_load() -> dict:
    """{visible device id -> (util_pct, foreign_mib, total_mib)}.

    A device missing from the result is UNKNOWN, and callers must treat
    unknown as busy -- failing open puts an unpinned worker on whichever
    card the driver picks.
    """
    try:
        util_max: dict = {}
        total: dict = {}
        for i in range(max(1, GPU_UTIL_SAMPLES)):
            if i:
                time.sleep(GPU_UTIL_SAMPLE_GAP_S)
            proc = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20)
            if proc.returncode != 0:
                return {}
            for line in (proc.stdout or "").strip().splitlines():
                idx, util, tot = [x.strip() for x in line.split(",")[:3]]
                g = int(idx)
                util_max[g] = max(util_max.get(g, 0.0), float(util))
                total[g] = int(tot)
        if not total:
            return {}

        # Foreign memory only -- see GPU_BUSY_MEM_PCT. Per-process rows are
        # keyed by uuid, so attributing them needs the same uuid->index map
        # `gpu_busy_pids` builds.
        uuid_rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        phys = {}
        for line in (uuid_rows.stdout or "").strip().splitlines():
            idx, uu = [x.strip() for x in line.split(",", 1)]
            phys[uu] = int(idx)
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        ours = _own_pids()
        foreign: dict = {g: 0 for g in total}
        for line in (apps.stdout or "").strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3 or not parts[1].isdigit():
                continue
            if int(parts[1]) in ours:
                continue
            g = phys.get(parts[0])
            if g is not None:
                try:
                    foreign[g] = foreign.get(g, 0) + int(parts[2])
                except ValueError:
                    continue
    except Exception:                                    # noqa: BLE001
        return {}

    return _remap_to_visible(
        {g: (util_max.get(g, 0.0), foreign.get(g, 0), total[g])
         for g in total})


def free_devices() -> list:
    """Visible GPUs that look idle, in index order.

    A load test, not an emptiness test: utilisation under
    `GPU_BUSY_UTIL_PCT` and foreign memory under `GPU_BUSY_MEM_PCT`. A
    neighbour that grows can still take one of the two jobs with it, and
    lip-sync is the likelier victim because MuseTalk answers an OOM by
    exiting 0 having written nothing.

    With `DUBBING_GPU_ALLOW_SHARED=1` every visible GPU counts as free and
    the foreign tenants are named in the log instead.
    """
    gpus = visible_cuda_devices()
    if not gpus:
        return []
    busy = gpu_busy_pids()
    if _allow_shared():
        shared = {g: busy[g] for g in gpus if busy.get(g)}
        if shared:
            log.warning(
                "gpu: %s set — using %s alongside %s. If that process grows, "
                "one of the two jobs dies, and MuseTalk answers an OOM by "
                "exiting 0 with no output.",
                GPU_ALLOW_SHARED_ENV,
                ", ".join(f"gpu{g}" for g in gpus),
                "; ".join(f"gpu{g}: pid {', '.join(map(str, p))}"
                          for g, p in shared.items()))
        return list(gpus)

    util_cap = _pct_env("DUBBING_GPU_BUSY_UTIL_PCT", GPU_BUSY_UTIL_PCT)
    mem_cap = _pct_env("DUBBING_GPU_BUSY_MEM_PCT", GPU_BUSY_MEM_PCT)
    load = gpu_load()
    free = []
    for g in gpus:
        if g not in load:
            # Unknown is busy, never free. See `gpu_load`.
            log.warning("gpu: gpu%d load unreadable; treating it as busy", g)
            continue
        util, foreign_mib, total_mib = load[g]
        mem_pct = (100.0 * foreign_mib / total_mib) if total_mib else 100.0
        if util < util_cap and mem_pct < mem_cap:
            if foreign_mib:
                log.info(
                    "gpu: gpu%d idle enough to use — %.0f%% util, %d MiB "
                    "(%.1f%%) held by %s",
                    g, util, foreign_mib, mem_pct,
                    ", ".join(f"pid {p}" for p in busy.get(g, [])) or "no pid")
            free.append(g)
        else:
            log.info("gpu: gpu%d busy — %.0f%% util (cap %.0f%%), %d MiB "
                     "foreign (%.1f%%, cap %.0f%%)",
                     g, util, util_cap, foreign_mib, mem_pct, mem_cap)
    return free


def gpu_free_memory() -> dict:
    """{device id -> free MiB}, in the same id space as
    `visible_cuda_devices()`.

    nvidia-smi reports PHYSICAL indices, so with `CUDA_VISIBLE_DEVICES=2,5`
    position i must be looked up under the physical id at position i, or the
    figures attach to the wrong cards.

    For reporting only; it does not gate selection.
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

    `n` caps the count when a caller wants fewer.

    [] is a REFUSAL, not "use the default device": an unpinned worker lands
    on whichever card the driver picks, which is the collision this avoids.
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


def _duration_or_zero(media: Path) -> float:
    """`media_io.probe_duration`, with a failure reported as 0.0.

    The 0.0 is for `split_video_audio`, which treats an unreadable duration
    as a reason to fall back to a single chunk; a raise there would kill the
    shard run.
    """
    from .media_io import probe_duration
    try:
        return float(probe_duration(media))
    except Exception:                               # noqa: BLE001
        log.warning("gpu: could not read the duration of %s; "
                    "treating it as unsplittable", media)
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
    (libx264 ultrafast CRF 18) so each chunk starts on a keyframe; a
    stream-copy split snaps to the nearest prior keyframe and gives ragged
    boundaries an audio-driven model reads as drift.

    Falls back to a single chunk when `n <= 1`, the clip is shorter than
    `n * min_chunk_s`, or ffprobe cannot read the duration.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dur = _duration_or_zero(video)
    if n <= 1 or dur <= 0.0 or dur < n * min_chunk_s:
        return [(Path(video), Path(audio))]

    ff = _ffmpeg_bin()
    chunk = dur / n
    pairs: List[Tuple[Path, Path]] = []
    for i in range(n):
        start = i * chunk
        # Last chunk: no -t, so float rounding cannot drop a final frame.
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

    All inputs must share codec parameters; this re-encodes once more on
    concat to normalise them.
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
