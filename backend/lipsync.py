"""Backend-facing lip-sync entry point.

Chains after the audio mix stage: takes the muxed MP4 (dialogue dubbed
over the M&E stem) and re-renders the on-screen faces so their mouth
movements match the dubbed audio. Writes the lip-synced video and
annotates the manifest JSON with `extra.lipsync.*`.

If the mix stage did not produce an MP4 (e.g. `--no-mux`), the stage
falls back to muxing the dubbed WAV onto the source video first.
"""
from __future__ import annotations

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models.lipsync import LipSyncRequest, available_models, get_lipsync

from .gpu import concat_videos, pick_devices, split_video_audio
from .media_io import _ffmpeg_bin, probe_duration

log = logging.getLogger(__name__)


@dataclass
class LipSyncOutput:
    model: str
    out_video: Path
    duration_s: float
    fps: float
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


def run_lipsync(
    merged_json: Path,
    *,
    model: str,
    source_video: Path,
    driving_audio: Path,
    out_dir: Path,
    pre_muxed_video: Optional[Path] = None,
    lipsync_kwargs: Optional[dict] = None,
    num_gpus: Optional[int] = None,
    restore: Optional[str] = None,
    restore_kwargs: Optional[dict] = None,
    window: Optional[tuple] = None,
) -> LipSyncOutput:
    """Lip-sync `source_video` (or `pre_muxed_video`) to `driving_audio`.

    The adapter sees a (video, audio) pair. We prefer the pre-muxed video
    (dubbed audio already baked in) so the output file has the correct
    audio track even if the adapter re-strips & re-attaches audio.

    `num_gpus` enables time-sharded multi-GPU inference:
      - None → auto: use every visible CUDA device.
      - 1    → single-GPU (legacy path, no splitting).
      - N>1  → split the input into N equal-duration chunks and run one
               adapter per GPU in parallel, each pinned via
               CUDA_VISIBLE_DEVICES. Short clips (< N * 15s) fall back to
               single-GPU to keep split/concat overhead from eating the win.

    `restore` names a face-restoration model from `models.restore` to run
    on the lip-synced render — the Wav2Lip-CodeFormer arrangement. The
    lip-sync models rebuild the mouth at a small working resolution
    (Wav2Lip paints a 96x96 crop), so the restored region comes back
    softer than the face around it; the restoration pass re-detects the
    face, rebuilds it at 512x512 and pastes it back. It runs after the
    sharded concat, so a multi-GPU render is restored as one video.

    `window=(start_s, end_s)` renders only that slice and splices it back
    into the full timeline, so the output is still a complete video. The
    parts outside the window are NOT passed through untouched at the byte
    level -- the concat re-encodes everything once -- but they are not sent
    to the lip-sync model, which is where the cost and the risk are. When
    no pre-muxed video was handed in, the head and tail are cut from a
    freshly muxed copy of the source carrying `driving_audio`; cutting them
    from the raw source would leave the untouched stretches speaking the
    ORIGINAL language.
    """
    if model not in available_models():
        raise ValueError(
            f"Unknown lip-sync model {model!r}. "
            f"Available: {', '.join(available_models())}"
        )
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    source_video = Path(source_video)
    driving_audio = Path(driving_audio)

    # Choose the video that goes into the lip-sync model as the face source.
    face_video = Path(pre_muxed_video) if pre_muxed_video and Path(pre_muxed_video).exists() \
                 else source_video
    out_video = out_dir / f"{source_video.stem}.{model}.lipsync.mp4"

    # Window: cut the face video and the driving audio down to the slice the
    # model will actually see, and remember what to splice it back into.
    splice_into: Optional[Path] = None
    w_start = w_end = 0.0
    if window is not None:
        w_start, w_end = sorted((float(window[0]), float(window[1])))
        full_dur = probe_duration(face_video)
        if full_dur > 0:
            w_end = min(w_end, full_dur)
        if w_end - w_start < 0.1:
            raise ValueError(
                f"lipsync window {w_start:.2f}-{w_end:.2f}s is empty "
                f"(video is {full_dur:.2f}s)")
        work = out_dir / "window"
        work.mkdir(parents=True, exist_ok=True)
        if face_video == source_video:
            # No pre-muxed video: build one, or the spliced-in head and tail
            # come back speaking the source language.
            dubbed_full = work / f"{source_video.stem}.dubbed.mp4"
            _remux_audio(source_video, driving_audio, dubbed_full)
            face_video = dubbed_full
        splice_into = face_video
        face_video = _cut_video(face_video, w_start, w_end,
                                work / "window.mp4")
        driving_audio = _cut_audio(driving_audio, w_start, w_end,
                                   work / "window.wav")
        log.info("lipsync: window %.2f-%.2fs of %.2fs -> rendering %.2fs",
                 w_start, w_end, full_dur, w_end - w_start)

    gpus = pick_devices(num_gpus)
    if len(gpus) > 1:
        result = _run_lipsync_sharded(
            model=model, face_video=face_video, driving_audio=driving_audio,
            out_video=out_video, out_dir=out_dir, gpus=gpus,
            lipsync_kwargs=lipsync_kwargs or {},
        )
    else:
        pinned_kwargs = dict(lipsync_kwargs or {})
        if gpus:
            pinned_kwargs.setdefault("cuda_device", gpus[0])
        lipsync = get_lipsync(model, **pinned_kwargs)
        log.info("lipsync[%s, gpu=%s]: %s + %s → %s",
                 model, pinned_kwargs.get("cuda_device"),
                 face_video, driving_audio, out_video)
        result = lipsync.sync(LipSyncRequest(
            video=face_video, audio=driving_audio, out_path=out_video,
        ))

    # Some adapters (notably Wav2Lip) output video-only or re-encode audio
    # losslessly; if the result has no audio track OR its audio doesn't
    # match the driving audio, re-mux the dubbed audio in.
    final_video = out_video
    if not result.error and _needs_remux(out_video):
        remuxed = out_dir / f"{source_video.stem}.{model}.lipsync.remux.mp4"
        _remux_audio(out_video, driving_audio, remuxed)
        final_video = remuxed

    # Optional face-restoration post-process. It re-encodes the video and
    # re-muxes the driving audio itself, so it supersedes the remux above.
    restore_info = None
    if restore and not result.error:
        restored_path = out_dir / f"{source_video.stem}.{model}.{restore}.mp4"
        restore_info = _run_restore(
            restore, video=final_video, audio=driving_audio,
            out_path=restored_path,
            kwargs=dict(restore_kwargs or {}), gpus=gpus,
        )
        if restore_info.get("error"):
            # Keep the un-restored render: it is a usable result, and
            # silently returning nothing would hide a working lip-sync.
            log.error("restore[%s] failed, keeping un-restored render: %s",
                      restore, restore_info["error"])
        else:
            final_video = restored_path

    # Splice the rendered window back into the full timeline.
    if splice_into is not None and not result.error:
        spliced = out_dir / f"{source_video.stem}.{model}.lipsync.windowed.mp4"
        try:
            final_video = _splice_window(
                splice_into, final_video, w_start, w_end, spliced,
                work_dir=out_dir / "window")
        except Exception as e:
            # The render is good; only the splice failed. Hand back the
            # window render and say so rather than losing the GPU work.
            log.error("lipsync: window splice failed (%s); returning the "
                      "window render only: %s", e, final_video)
            result.extra["window_splice_error"] = str(e)
        else:
            result.extra["window"] = [w_start, w_end]
            result.extra["window_spliced_into"] = str(splice_into)

    # Annotate the manifest if we have one.
    if merged_json and Path(merged_json).exists():
        data = json.loads(Path(merged_json).read_text())
        data.setdefault("extra", {})["lipsync"] = {
            "model": result.model_name,
            "out_video": str(final_video),
            "duration_s": result.duration_s,
            "fps": result.fps,
            "error": result.error,
            "driving_audio": str(driving_audio),
            "face_video": str(face_video),
            **result.extra,
        }
        if restore_info is not None:
            data["extra"]["lipsync"]["restore"] = restore_info
        Path(merged_json).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    log.info("lipsync[%s]: done → %s (err=%s)",
             model, final_video, result.error or "none")
    extra = dict(result.extra)
    if restore_info is not None:
        extra["restore"] = restore_info
    return LipSyncOutput(
        model=result.model_name, out_video=final_video,
        duration_s=result.duration_s, fps=result.fps,
        error=result.error, extra=extra,
    )


def _ff_run(cmd: list, what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg {what} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:400]}")


def _cut_video(src: Path, start: float, end: float, out: Path) -> Path:
    """Re-encoded cut, so the slice starts on a keyframe.

    A stream-copy cut snaps back to the previous keyframe, which would hand
    the model frames from before the window and shift every mouth it draws.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    _ff_run([_ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{start:.6f}", "-to", f"{end:.6f}", "-i", str(src),
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
             str(out)], f"window cut of {src.name}")
    return out


def _cut_audio(src: Path, start: float, end: float, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    _ff_run([_ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{start:.6f}", "-to", f"{end:.6f}", "-i", str(src),
             "-c:a", "pcm_s16le", str(out)], f"window cut of {src.name}")
    return out


def _splice_window(full: Path, rendered: Path, start: float, end: float,
                   out: Path, *, work_dir: Path) -> Path:
    """Put `rendered` back where [start, end] was in `full`.

    Measured on a 30 s 25 fps test clip: the spliced result comes out 0.040 s
    long -- one frame -- and it is the SAME one frame whether the splice has
    two parts or three, so it is a boundary rounding, not per-join drift that
    grows with the number of windows. One frame is well under the ~80 ms where
    lip-sync error becomes visible; several hundred joins would not be, so
    splice a window, not a hundred of them.
    """
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    dur = probe_duration(full)
    parts: list = []
    if start > 0.05:
        parts.append(_cut_video(full, 0.0, start, work_dir / "head.mp4"))
    parts.append(Path(rendered))
    if dur > 0 and end < dur - 0.05:
        parts.append(_cut_video(full, end, dur, work_dir / "tail.mp4"))
    if len(parts) == 1:
        return Path(rendered)
    concat_videos(parts, out)
    log.info("lipsync: spliced %d part(s) -> %s", len(parts), out)
    return out


def _run_restore(
    name: str,
    *,
    video: Path,
    audio: Path,
    out_path: Path,
    kwargs: dict,
    gpus: list,
) -> dict:
    """Run a `models.restore` adapter over a lip-synced render.

    Returns the result as a plain dict for the manifest. Failures come
    back as `{"error": ...}` rather than raising — a restoration fault
    should not throw away a lip-sync render that already succeeded.
    """
    from models.restore import RestoreRequest, available_models, get_restore

    if name not in available_models():
        return {"model_name": name,
                "error": f"Unknown restoration model {name!r}. "
                         f"Available: {', '.join(available_models())}"}
    if gpus:
        kwargs.setdefault("cuda_device", gpus[0])
    log.info("restore[%s, gpu=%s]: %s → %s",
             name, kwargs.get("cuda_device"), video, out_path)
    try:
        adapter = get_restore(name, **kwargs)
        result = adapter.restore(RestoreRequest(
            video=video, out_path=out_path, audio=audio,
        ))
    except Exception as e:                      # noqa: BLE001 - reported, not raised
        return {"model_name": name, "error": f"{type(e).__name__}: {e}"}

    info = result.to_jsonable()
    if not result.error and result.frames and not result.frames_with_faces:
        # Exit code 0 with no face in any frame means the stage did nothing.
        # Say so rather than letting an unchanged video read as a success.
        info["warning"] = (
            f"{name} detected no face in any of {result.frames} frames — "
            f"the output is byte-for-byte a re-encode of the input"
        )
        log.warning("restore[%s]: %s", name, info["warning"])
    return info


def _needs_remux(video: Path) -> bool:
    """True when the audio stream is missing or in a form players stumble on.

    Wav2Lip encodes AAC mono @16 kHz by default — technically valid, but
    many players (QuickTime, some web embeds, phone galleries) refuse it
    silently. Treat anything that isn't ≥44.1 kHz stereo as "needs remux".
    """
    # media_io._ffprobe_bin() rather than shutil.which: there is no ffprobe
    # on this host's PATH, so which() returns None and this function used to
    # answer "no remux needed" for every video -- which is exactly how
    # Wav2Lip's 16 kHz mono AAC shipped to a player that refused it.
    from .media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        return False      # best-effort; skip if ffprobe is unavailable
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error",
             "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, check=True,
        )
        lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
        if not lines:
            return True
        try:
            sr = int(lines[0]); ch = int(lines[1]) if len(lines) > 1 else 1
        except ValueError:
            return True
        return sr < 44100 or ch < 2
    except Exception:
        return False


def _run_lipsync_sharded(
    *,
    model: str,
    face_video: Path,
    driving_audio: Path,
    out_video: Path,
    out_dir: Path,
    gpus: list,
    lipsync_kwargs: dict,
):
    """Fan one job across `gpus`: split in time, run in parallel, concat.

    Returns a LipSyncResult-like object (model_name, out_video, duration_s,
    fps, error, extra) assembled from the final concat.
    """
    from models.lipsync import LipSyncResult
    n = len(gpus)
    shard_dir = out_dir / f"{out_video.stem}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    pairs = split_video_audio(face_video, driving_audio, n, shard_dir)
    if len(pairs) == 1:
        # Clip too short to shard profitably — fall back to the single GPU.
        pinned_kwargs = dict(lipsync_kwargs); pinned_kwargs["cuda_device"] = gpus[0]
        lipsync = get_lipsync(model, **pinned_kwargs)
        log.info("lipsync[%s, gpu=%s, sharded→fallback]: %s → %s",
                 model, gpus[0], face_video, out_video)
        return lipsync.sync(LipSyncRequest(
            video=face_video, audio=driving_audio, out_path=out_video,
        ))

    def _one(i_gpu_pair):
        i, gpu, (v, a) = i_gpu_pair
        pinned_kwargs = dict(lipsync_kwargs); pinned_kwargs["cuda_device"] = gpu
        lipsync = get_lipsync(model, **pinned_kwargs)
        part_out = shard_dir / f"part{i:02d}.lipsync.mp4"
        log.info("lipsync[%s, gpu=%d, shard=%d/%d]: %s → %s",
                 model, gpu, i + 1, len(pairs), v.name, part_out.name)
        return lipsync.sync(LipSyncRequest(video=v, audio=a, out_path=part_out))

    jobs = [(i, gpus[i], pair) for i, pair in enumerate(pairs)]
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(_one, jobs))

    errors = [r.error for r in results if r.error]
    if errors:
        return LipSyncResult(
            model_name=results[0].model_name, out_video=out_video,
            duration_s=0.0, fps=0.0,
            error=f"{len(errors)}/{len(results)} shard(s) failed: {errors[0]}",
        )

    concat_videos([r.out_video for r in results], out_video)
    # Keep the shard dir around for debugging; the concat.txt is already cleaned.

    fps, dur = _probe_video_fps_dur(out_video)
    extra = dict(results[0].extra); extra["sharded"] = {"n": n, "gpus": gpus}
    return LipSyncResult(
        model_name=results[0].model_name, out_video=out_video,
        duration_s=dur, fps=fps, extra=extra,
    )


def _probe_video_fps_dur(path: Path) -> tuple:
    """(fps, duration_s) via ffprobe; (0.0, 0.0) when unavailable."""
    # media_io._ffprobe_bin() rather than shutil.which: there is no
    # ffprobe on this host's PATH, and which() returning None here
    # silently degrades to "unknown" instead of failing (CLAUDE.md §1).
    from .media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe or not Path(path).exists():
        return (0.0, 0.0)
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
        fps = 0.0; dur = 0.0
        if lines:
            rate = lines[0]
            if "/" in rate:
                n, d = rate.split("/", 1)
                fps = float(n) / float(d) if float(d) else 0.0
            else:
                fps = float(rate)
        if len(lines) > 1:
            dur = float(lines[1])
        return (fps, dur)
    except Exception:
        return (0.0, 0.0)


def _remux_audio(video: Path, audio: Path, out: Path) -> None:
    """Mux `audio` onto `video` as 48 kHz stereo AAC — the broadest-compatibility
    audio profile. Wav2Lip's native 16 kHz mono passes ffprobe but is silently
    rejected by several common players."""
    ff = _ffmpeg_bin()
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
        "-shortest", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg remux failed (rc={proc.returncode}): {proc.stderr.strip()[:800]}"
        )
