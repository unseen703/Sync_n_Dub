"""Backend lip-sync stage."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.lipsync import LipSyncRequest, available_models, get_lipsync

from backend.common.gpu import (concat_videos, gpu_busy_pids, gpu_free_memory, gpu_load,
                  pick_devices, split_video_audio, visible_cuda_devices)
from backend.common.media_io import _ffmpeg_bin, probe_duration

# Patterns that identify lipsync render outputs (not intermediates like shards).
_RENDER_GLOBS = ("*.lipsync.mp4", "*.lipsync.remux.mp4",
                 "*.lipsync.windowed.mp4")

log = logging.getLogger(__name__)


def _archive_previous_renders(out_dir: Path, model: str) -> Optional[Path]:
    """Move this model's existing renders into ``prev/<timestamp>/``."""
    hits = []
    for pat in _RENDER_GLOBS:
        hits.extend(p for p in out_dir.glob(pat)
                    if model in p.name and p.is_file())
    if not hits:
        return None
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = out_dir / "prev" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for p in hits:
        target = dest / p.name
        shutil.move(str(p), str(target))
        log.info("lipsync: archived %s -> prev/%s/%s", p.name, stamp, p.name)
    return dest


# How far a target list may fall short of the clip it covers before the shortfall stops being rounding and becomes a real disagreement.
_COORD_PAD_FRAMES = 2


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
    window: Optional[tuple] = None,
    coord_list: Optional[list] = None,
    max_segment_s: Optional[float] = None,
    targets_npz: Optional[Path] = None,
) -> LipSyncOutput:
    """Lip-sync a video to dubbed audio, optionally rendering a window or splitting work across GPUs/segments. Supports per-frame face targets and records the result in the manifest."""
    if model not in available_models():
        raise ValueError(
            f"Unknown lip-sync model {model!r}. "
            f"Available: {', '.join(available_models())}"
        )
    if max_segment_s is None:
        from backend.media.scaling import lipsync_segment_s_default
        max_segment_s = lipsync_segment_s_default()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _archive_previous_renders(out_dir, model)
    source_video = Path(source_video)
    driving_audio = Path(driving_audio)

    # Choose the video that goes into the lip-sync model as the face source.
    face_video = Path(pre_muxed_video) if pre_muxed_video and Path(pre_muxed_video).exists() \
                 else source_video
    out_video = out_dir / f"{source_video.stem}.{model}.lipsync.mp4"

    # Window: cut the face video and driving audio down to the slice the model sees, and remember what to splice it back into.
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
            # No pre-muxed video: build one, or the spliced-in head and tail come back speaking the source language.
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
        if coord_list:
            # The cut clip renumbers frames from zero, so a list indexed against the whole video would animate the wrong frames -- the same hazard sharding and segmenting refuse below, except that this one cannot be answered by "render in one pass".
            fps_probe = _probe_fps(splice_into) or _probe_fps(face_video) or 0.0
            if fps_probe <= 0:
                log.warning("lipsync[%s]: cannot read the fps, so the %d "
                            "supplied target boxes cannot be aligned to the "
                            "window; dropping them rather than animating the "
                            "wrong frames.", model, len(coord_list))
                coord_list = None
            else:
                coord_list = _slice_coords_to_window(
                    coord_list, face_video, w_start, w_end, fps_probe,
                    model=model)
        if targets_npz:
            fps_probe = _probe_fps(splice_into) or _probe_fps(face_video) or 0.0
            if fps_probe <= 0:
                log.warning("lipsync[%s]: cannot read the fps, so the "
                            "supplied targets cannot be aligned to the "
                            "window; dropping them rather than animating the "
                            "wrong frames.", model)
                targets_npz = None
            else:
                targets_npz = _slice_targets_to_window(
                    targets_npz, face_video, w_start, w_end, fps_probe,
                    out_dir=work, model=model)

    # Cap the render input to 1080p.
    _MAX_RENDER_HEIGHT = 1080
    dims = _video_dims(face_video)
    if dims and dims[1] > _MAX_RENDER_HEIGHT:
        from backend.lipsync.wav2lip_adapter import cap_dims
        capped = cap_dims(dims, _MAX_RENDER_HEIGHT)
        cap_out = out_dir / f"{face_video.stem}.{capped[1]}p.mp4"
        if not cap_out.is_file():
            ff = _ffmpeg_bin() or "ffmpeg"
            subprocess.run(
                [ff, "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(face_video),
                 "-vf", f"scale={capped[0]}:{capped[1]}:flags=lanczos",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-ar", "48000", "-ac", "2",
                 "-b:a", "192k", str(cap_out)], check=True)
        log.info("lipsync: capped %dx%d -> %dx%d (%.1fx fewer pixels)",
                 dims[0], dims[1], capped[0], capped[1],
                 (dims[0] * dims[1]) / float(capped[0] * capped[1]))
        face_video = cap_out

    gpus = pick_devices(num_gpus)
    if not gpus and visible_cuda_devices():
        # Cards exist but none clears the free-VRAM floor.
        free = gpu_free_memory()
        busy = gpu_busy_pids()
        # Since 2026-10-22 the gate is a LOAD test, not "is anyone there", so the refusal has to quote the numbers it judged on -- a bare pid reads as the old one-tenant rule and sends the operator looking for a process to kill when the card is merely working.
        load = gpu_load()
        if load:
            who = ", ".join(
                f"gpu{g} at {u:.0f}% util, {m} MiB foreign"
                + (f" (pid {','.join(str(x) for x in busy[g])})" if busy.get(g) else "")
                for g, (u, m, _t) in sorted(load.items()))
        else:
            who = ", ".join(f"gpu{g} busy (pid {','.join(str(x) for x in p)})"
                            for g, p in sorted(busy.items())) or "all devices busy"
        return LipSyncOutput(
            model=model, out_video=out_video, duration_s=0.0, fps=0.0,
            error=(f"no GPU is free — {who}. Lip-sync needs a card to "
                   f"itself; waiting is safer than sharing one."),
            extra={"gpu_free_mib": free, "gpu_busy_pids": busy},
        )
    from backend.common.memguard import breach as _mem_breach, memory_snapshot
    _mem_why = _mem_breach()
    if _mem_why:
        return LipSyncOutput(
            model=model, out_video=out_video, duration_s=0.0, fps=0.0,
            error=(f"not starting lip-sync: {_mem_why}. Wait for memory to "
                   f"free up, or raise DUBBING_MEM_GUARD_PCT."),
            extra={"memory": memory_snapshot()},
        )

    seg_s = float(max_segment_s or 0.0)
    seg_dur = probe_duration(face_video) if seg_s > 0 else 0.0
    # The 1.05 slack stops a 2:03 clip becoming two parts, one of them three seconds long: the extra model load costs more than the memory a sliver saves.
    want_segments = seg_s > 0 and seg_dur > seg_s * 1.05
    if want_segments:
        result = _run_lipsync_segmented(
            model=model, face_video=face_video, driving_audio=driving_audio,
            out_video=out_video, out_dir=out_dir, gpus=gpus,
            lipsync_kwargs=lipsync_kwargs or {},
            seg_s=seg_s, duration_s=seg_dur, coord_list=coord_list,
            targets_npz=targets_npz,
        )
    else:
        result = _render_one(
            model=model, face_video=face_video, driving_audio=driving_audio,
            out_video=out_video, out_dir=out_dir, gpus=gpus,
            lipsync_kwargs=lipsync_kwargs or {}, coord_list=coord_list,
            targets_npz=targets_npz,
        )

    # Some adapters (notably Wav2Lip) output video-only or re-encode the audio, so re-mux the dubbed audio when the result has no audio track or its audio does not match the driving audio.
    final_video = out_video
    if not result.error and _needs_remux(out_video):
        remuxed = out_dir / f"{source_video.stem}.{model}.lipsync.remux.mp4"
        _remux_audio(out_video, driving_audio, remuxed)
        final_video = remuxed

    # Splice the rendered window back into the full timeline.
    if splice_into is not None and not result.error:
        spliced = out_dir / f"{source_video.stem}.{model}.lipsync.windowed.mp4"
        try:
            final_video = _splice_window(
                splice_into, final_video, w_start, w_end, spliced,
                work_dir=out_dir / "window")
        except Exception as e:
            # The render is good; only the splice failed.
            log.error("lipsync: window splice failed (%s); returning the "
                      "window render only: %s", e, final_video)
            result.extra["window_splice_error"] = str(e)
        else:
            result.extra["window"] = [w_start, w_end]
            result.extra["window_spliced_into"] = str(splice_into)

    # An adapter can exit 0 having written nothing: Wav2Lip's inference.py muxes with subprocess.call and discards the return code, and MuseTalk shelled into the wrong interpreter dies on import while the wrapper still returns.
    try:
        produced = final_video.is_file() and final_video.stat().st_size > 0
    except OSError:
        produced = False
    if not result.error and not produced:
        result.error = (f"{model} reported success but wrote no output to "
                        f"{final_video}")

    # FRAME PARITY.
    if produced and not result.error and splice_into is None and window is None:
        # A truncated video stream under a full-length audio stream plays as a freeze: the player holds the last decoded frame.
        try:
            _conform_to_source(final_video, source_video,
                               driving_audio, model=model)
        except Exception as e:                           # noqa: BLE001
            log.warning("lipsync[%s]: conform failed (%s); "
                        "the parity check below will report the shortfall",
                        model, e)

        # The reference for "was the picture truncated" is the SOURCE video, never the driving audio: a dub that overran its slot would otherwise be reported as a freeze that does not exist.
        v_dur, a_dur = _stream_durations(final_video)
        src_dur, _src_a = _stream_durations(source_video)
        if v_dur and src_dur and (src_dur - v_dur) > 0.5:
            result.error = (
                f"{model} video stream ends at {v_dur:.2f}s but the source "
                f"runs to {src_dur:.2f}s -- the picture is "
                f"{src_dur - v_dur:.1f}s short and will freeze")
            log.error("lipsync[%s]: TRUNCATED VIDEO STREAM — %s",
                      model, result.error)
        elif v_dur and a_dur and (a_dur - v_dur) > 0.5:
            # The render is the right length; the dub is too long for it.
            log.warning(
                "lipsync[%s]: dubbed audio runs %.1fs past the picture "
                "(%.2fs vs %.2fs) -- trimming to the picture; the timing "
                "plan did not fit the speech into its slot",
                model, a_dur - v_dur, a_dur, v_dur)
            try:
                trimmed = final_video.with_suffix(".trimmed.mp4")
                _remux_audio(final_video, driving_audio, trimmed,
                             shortest=True)
                shutil.move(str(trimmed), str(final_video))
            except Exception as e:                       # noqa: BLE001
                log.warning("lipsync[%s]: could not trim the overhang (%s); "
                            "the file ends with %.1fs of frozen picture",
                            model, e, a_dur - v_dur)

        src_n, out_n = _frame_count(source_video), _frame_count(final_video)
        if src_n and out_n and out_n > src_n:
            # A surplus is not a parity failure: the excess is the dub outlasting the picture (see _conform_to_source's overshoot branch).
            _conform_to_source(final_video, source_video, driving_audio,
                               model=model)
            out_n = _frame_count(final_video)
        if src_n and out_n and out_n > src_n:
            # Still over after the trim.
            log.info("lipsync[%s]: render is %d frame(s) longer than its "
                     "source (%d vs %d) -- the dub outlasts the picture; "
                     "nothing is missing", model, out_n - src_n, out_n, src_n)
        elif src_n and out_n and out_n < src_n:
            result.error = (
                f"{model} returned {out_n} frames for a {src_n}-frame source "
                f"({out_n - src_n:+d}); frames it could not place were "
                f"dropped rather than copied through, so the video freezes "
                f"and drifts")
            log.error("lipsync[%s]: FRAME PARITY FAILED — %s", model, result.error)
        elif src_n:
            log.info("lipsync[%s]: frame parity ok (%d frames)", model, src_n)

    # Annotate the manifest if we have one.
    if merged_json and Path(merged_json).exists():
        data = json.loads(Path(merged_json).read_text())
        data.setdefault("extra", {})["lipsync"] = {
            "model": result.model_name,
            "out_video": str(final_video),
            "duration_s": result.duration_s,
            "fps": result.fps,
            "error": result.error,
            "produced": produced,
            "driving_audio": str(driving_audio),
            "face_video": str(face_video),
            **result.extra,
        }
        Path(merged_json).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # The verdict leads and a failure is not INFO: logging "done" unconditionally with the error in a trailing parenthetical made a failed render read as a successful one in the UI's progress scrape.
    if result.error:
        log.error("lipsync[%s]: FAILED — %s", model, result.error)
        log.error("lipsync[%s]: no usable render; expected %s",
                  model, final_video)
    else:
        log.info("lipsync[%s]: done → %s", model, final_video)
    extra = dict(result.extra)
    extra["produced"] = produced
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


def _probe_fps(media: Optional[Path]) -> float:
    """Frames per second, or 0.0 when it cannot be determined."""
    if media is None:
        return 0.0
    try:
        import cv2

        cap = cv2.VideoCapture(str(media))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()
        return fps if fps > 0 else 0.0
    except Exception as e:                                # noqa: BLE001
        log.debug("could not probe fps for %s: %s", media, e)
        return 0.0


def _probe_frames(media: Path) -> int:
    """Frame count, or 0 when it cannot be determined."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(media))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        return max(0, n)
    except Exception as e:                                # noqa: BLE001
        log.debug("could not probe frame count for %s: %s", media, e)
        return 0


def _slice_targets_to_window(targets_npz: Path, window_video: Path,
                             w_start: float, w_end: float, fps: float,
                             *, out_dir: Path, model: str) -> Optional[Path]:
    """The slice of `targets_npz` covering the cut window clip."""
    from backend.asd.lipsync_targets import load_targets_payload, save_targets, slice_targets

    payload = load_targets_payload(Path(targets_npz))
    if payload is None:
        log.warning("lipsync[%s]: targets %s unreadable; the window renders "
                    "with stock detection", model, targets_npz)
        return None

    import numpy as np
    total = int(np.asarray(payload["frame_track"]).shape[0])
    f0 = int(round(w_start * fps))
    want = _probe_frames(window_video) or (int(round(w_end * fps)) - f0)
    stop = min(f0 + want, total)
    if f0 >= total:
        log.warning("lipsync[%s]: targets cover %d frame(s) but window starts "
                    "at frame %d; rendering with stock detection",
                    model, total, f0)
        return None
    try:
        piece = slice_targets(payload, f0, stop)
    except ValueError as e:
        log.warning("lipsync[%s]: targets window slice refused (%s); "
                    "rendering with stock detection", model, e)
        return None
    got = int(np.asarray(piece["frame_track"]).shape[0])
    short = want - got
    if short > _COORD_PAD_FRAMES:
        log.warning("lipsync[%s]: window needs %d frame(s) but targets slice "
                    "is only %d; rendering with stock detection",
                    model, want, got)
        return None
    dest = Path(out_dir) / "window.targets.npz"
    save_targets(piece, dest)
    log.info("lipsync[%s]: targets sliced to window (frames %d-%d of %d, "
             "%d kept, %d targeted)",
             model, f0, stop, total, got,
             int((piece["frame_track"] >= 0).sum()))
    return dest


def _slice_coords_to_window(coord_list: list, window_video: Path,
                            w_start: float, w_end: float, fps: float,
                            *, model: str) -> Optional[list]:
    """The slice of `coord_list` covering the cut window clip."""
    f0 = int(round(w_start * fps))
    want = _probe_frames(window_video) or (int(round(w_end * fps)) - f0)
    sliced = list(coord_list)[f0:f0 + want]
    short = want - len(sliced)
    if 0 < short <= _COORD_PAD_FRAMES and sliced:
        sliced.extend([sliced[-1]] * short)
        note = f", last box repeated x{short} to cover the clip"
    elif short > 0:
        note = (f", {short} SHORT of the clip's {want} frame(s) — too many "
                f"to be rounding, so any piece this is split into may "
                f"render ungated")
    else:
        note = ""
    log.info("lipsync[%s]: target boxes sliced to the window "
             "(frames %d-%d of %d, %d kept%s)",
             model, f0, f0 + want, len(coord_list), len(sliced), note)
    return sliced or None


def _split_coords(coord_list: Optional[list], pairs: list,
                  *, model: str, what: str) -> list:
    """One slice of `coord_list` per (video, audio) pair, in order."""
    if not coord_list:
        return [None] * len(pairs)
    out: list = []
    cursor = 0
    for i, (v, _a) in enumerate(pairs):
        n = _probe_frames(Path(v))
        if n <= 0:
            log.warning("lipsync[%s]: cannot read the frame count of %s "
                        "%d; it renders without ASD targeting",
                        model, what, i + 1)
            out.append(None)
            cursor = -1 if cursor < 0 else cursor
            continue
        short = (cursor + n) - len(coord_list)
        if cursor >= 0 and 0 < short <= _COORD_PAD_FRAMES \
                and cursor < len(coord_list):
            # A rounding-sized shortfall on the final piece.
            piece = list(coord_list[cursor:])
            piece.extend([piece[-1]] * short)
            log.info("lipsync[%s]: %s %d runs %d frame(s) past the target "
                     "list; repeating the last box rather than dropping the "
                     "targeting for all %d of its frames",
                     model, what, i + 1, short, n)
            out.append(piece)
            cursor = len(coord_list)
            continue
        if cursor < 0 or short > 0:
            log.warning("lipsync[%s]: target list is %d frame(s) but %s %d "
                        "needs frames %d-%d; it renders without ASD "
                        "targeting", model, len(coord_list), what, i + 1,
                        cursor, cursor + n)
            out.append(None)
            cursor = -1
            continue
        out.append(list(coord_list[cursor:cursor + n]))
        cursor += n
    return out


def _split_targets(targets_npz: Optional[Path], pairs: list, out_dir: Path,
                   *, model: str, what: str) -> list:
    """One targets .npz per (video, audio) pair, sliced and renumbered."""
    if not targets_npz:
        return [None] * len(pairs)

    from backend.asd.lipsync_targets import load_targets_payload, save_targets, slice_targets

    payload = load_targets_payload(Path(targets_npz))
    if payload is None:
        log.warning("lipsync[%s]: targets %s unreadable; every %s renders "
                    "with stock detection", model, targets_npz, what)
        return [None] * len(pairs)

    import numpy as np
    total = int(np.asarray(payload["frame_track"]).shape[0])
    out: list = []
    cursor = 0
    for i, (v, _a) in enumerate(pairs):
        n = _probe_frames(Path(v))
        if n <= 0:
            log.warning("lipsync[%s]: cannot read the frame count of %s %d; "
                        "it renders without targeting", model, what, i + 1)
            out.append(None)
            cursor = -1 if cursor < 0 else cursor
            continue
        short = (cursor + n) - total
        if cursor < 0 or short > _COORD_PAD_FRAMES:
            log.warning("lipsync[%s]: targets cover %d frame(s) but %s %d "
                        "needs frames %d-%d; it renders without targeting",
                        model, total, what, i + 1, cursor, cursor + n)
            out.append(None)
            cursor = -1
            continue
        stop = min(cursor + n, total)
        try:
            piece = slice_targets(payload, cursor, stop)
        except ValueError as e:                          # noqa: BLE001
            log.warning("lipsync[%s]: %s %d targets slice refused (%s); "
                        "it renders without targeting", model, what, i + 1, e)
            out.append(None)
            cursor += n
            continue
        got = int(np.asarray(piece["frame_track"]).shape[0])
        if got != n:
            # A rounding-sized shortfall on the last piece would be rejected by the consumer's own length check anyway.
            log.warning("lipsync[%s]: %s %d is %d frame(s) but its targets "
                        "slice is %d; it renders without targeting",
                        model, what, i + 1, n, got)
            out.append(None)
            cursor += n
            continue
        dest = Path(out_dir) / f"{Path(v).stem}.targets.npz"
        save_targets(piece, dest)
        log.info("lipsync[%s]: %s %d targets -> %s (%d frame(s), "
                 "%d targeted, %d run(s))", model, what, i + 1, dest.name,
                 got, int((piece["frame_track"] >= 0).sum()),
                 int(piece["run_start"].sum()))
        out.append(dest)
        cursor += n
    return out


def _cut_video(src: Path, start: float, end: float, out: Path) -> Path:
    """Re-encoded cut, so the slice starts on a keyframe."""
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
    """Put `rendered` back where [start, end] was in `full`."""
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


def _frame_count(video: Path) -> int:
    """Exact frame count, or 0 when ffprobe cannot answer."""
    # Imported here, as elsewhere: there is no ffprobe on this host's PATH and media_io resolves the one in the running env.
    from backend.common.media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        return 0
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
             str(video)], capture_output=True, text=True, timeout=300)
        # `-of csv=p=0` emits a trailing comma ("960,"), so int() on the raw line raises and the except answers 0, which reads as "cannot probe" and skips the check.
        first = (out.stdout or "").strip().splitlines()[0]
        return int(first.split(",")[0])
    except Exception:                                # noqa: BLE001
        return 0


def _conform_to_source(rendered: Path, source: Path, driving_audio: Path, *,
                       model: str) -> int:
    """Match the render's frame count to the source by trimming excess frames or appending the source tail."""
    src_n, out_n = _frame_count(source), _frame_count(rendered)
    if not src_n or not out_n or out_n == src_n:
        return 0

    # OVERSHOOT: the render can be LONGER than its source.
    if out_n > src_n:
        surplus = out_n - src_n
        work = rendered.parent / f"{rendered.stem}.conform"
        work.mkdir(parents=True, exist_ok=True)
        ff = _ffmpeg_bin()
        trimmed = work / "trimmed.mp4"
        # `-r` is NOT optional: without it ffmpeg re-times a `select` + `setpts` output at its default 25 fps, so a 23.976 fps shard comes back tagged 25/1 while its sibling keeps 24000/1001.
        rate = _frame_rate(rendered) or _frame_rate(source)
        _ff_run([ff, "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(rendered),
                 "-vf", f"select='lt(n\\,{src_n})',setpts=PTS-STARTPTS",
                 *(["-r", rate] if rate else []),
                 "-frames:v", str(src_n),
                 "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-pix_fmt", "yuv420p", str(trimmed)], "conform trim")
        got = _frame_count(trimmed)
        if got != src_n:
            log.warning("lipsync[%s]: trim produced %d frames, wanted %d; "
                        "leaving the render alone", model, got, src_n)
            return 0
        # `shortest=True` here, unlike the pad path below: the surplus IS the audio outlasting the picture.
        _remux_audio(trimmed, driving_audio, rendered, shortest=True)
        # ...except `-shortest` cuts to whichever stream is shorter and cannot tell which was meant.
        if _frame_count(rendered) < src_n:
            log.info("lipsync[%s]: the dub is shorter than its picture, so "
                     "-shortest cut %d frame(s) back off the conform; "
                     "re-muxing with the picture kept whole",
                     model, src_n - _frame_count(rendered))
            _remux_audio(trimmed, driving_audio, rendered, shortest=False)
        log.info("lipsync[%s]: trimmed %d frame(s) of audio-driven overshoot "
                 "(render ran %d past a %d-frame source)",
                 model, surplus, out_n, src_n)
        return -surplus

    missing = src_n - out_n
    work = rendered.parent / f"{rendered.stem}.conform"
    work.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg_bin()

    head = work / "head.mp4"          # the render, silent
    _ff_run([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(rendered),
             "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", str(head)], "conform head")

    # The tail must match the RENDER's dimensions, not the source's: Wav2Lip writes at `resize_factor` scale against a 1080p ceiling, and the concat demuxer silently drops a full-size tail.
    dims = _video_dims(rendered) or _video_dims(source)
    scale = f",scale={dims[0]}:{dims[1]}" if dims else ""
    rate = _frame_rate(rendered) or _frame_rate(source)

    tail = work / "tail.mp4"          # source frames out_n..src_n-1, silent
    _ff_run([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
             # `setpts=PTS-STARTPTS` is load-bearing: `select` keeps the SOURCE timestamps, so a tail taken from 14.9s carries that PTS span and the concat re-encode pads it out.
             "-vf", f"select='gte(n\\,{out_n})'{scale},setpts=PTS-STARTPTS",
             # No `-vsync 0` alongside `-r`: ffmpeg 7 rejects the pair.
             *(["-r", rate] if rate else ["-vsync", "0"]),
             "-frames:v", str(missing),
             "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", str(tail)], "conform tail")

    joined = work / "joined.mp4"
    # The concat FILTER, not `concat_videos`' demuxer: the demuxer needs byte-identical stream parameters and drops a mismatch without failing.
    _ff_run([ff, "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(head), "-i", str(tail),
             "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
             "-map", "[v]", "-c:v", "libx264", "-preset", "fast",
             "-crf", "18", "-pix_fmt", "yuv420p", str(joined)],
            "conform concat")
    # The demuxer drops a mismatched input without failing, so verify the frames were actually gained.
    gained = _frame_count(joined)
    if gained and gained != src_n:
        log.warning("lipsync[%s]: conform produced %d frames, wanted %d "
                    "(head %d + tail %d)", model, gained, src_n, out_n, missing)
    # NOT `-shortest`: the picture outlives the soundtrack here, and truncating to the audio would delete the frames just added.
    _remux_audio(joined, driving_audio, rendered, shortest=False)
    log.info("lipsync[%s]: conformed +%d frame(s) of untouched source "
             "(render covered %d of %d)", model, missing, out_n, src_n)
    return missing


def _concat_shards_verified(parts: list, out: Path, *, expected: int,
                            audio: Optional[Path] = None) -> Path:
    """Join rendered pieces and verify that no frames were lost."""
    parts = [Path(p) for p in parts]
    ff = _ffmpeg_bin()

    if audio is None:
        concat_videos([str(p) for p in parts], str(out))
        got = _frame_count(out)
        if not expected or not got or got >= expected:
            return out
        log.warning("lipsync: concat kept %d of %d frame(s) -- the demuxer "
                    "dropped a shard whose parameters did not match; rejoining "
                    "through the concat filter", got, expected)
        ins, labels = [], []
        for i, p in enumerate(parts):
            ins += ["-i", str(p)]
            labels.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}];")
        graph = ("".join(labels)
                 + "".join(f"[v{i}]" for i in range(len(parts)))
                 + f"concat=n={len(parts)}:v=1:a=0[v]")
        silent = out.with_suffix(".filterjoin.mp4")
        _ff_run([ff, "-y", "-hide_banner", "-loglevel", "error", *ins,
                 "-filter_complex", graph, "-map", "[v]", "-an",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-pix_fmt", "yuv420p", str(silent)], "shard filter-concat")
        # The demuxer join carried the audio; keep it rather than re-deriving.
        _remux_audio(silent, out, out, shortest=False)
        silent.unlink(missing_ok=True)
        log.info("lipsync: filter-concat recovered %d frame(s)",
                 _frame_count(out))
        return out

    # Video-only join, then remux the dub -- never decodes the pieces' audio.
    joined = out.with_suffix(".silent.mp4")
    listing = joined.with_suffix(".concat.txt")
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    _ff_run([ff, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", "-an", str(joined)], "shard concat")
    listing.unlink(missing_ok=True)

    got = _frame_count(joined)
    if expected and got and got < expected:
        log.warning("lipsync: concat kept %d of %d frame(s) -- the demuxer "
                    "dropped a piece whose parameters did not match; rejoining "
                    "through the concat filter", got, expected)
        ins, labels = [], []
        for i, p in enumerate(parts):
            ins += ["-i", str(p)]
            labels.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}];")
        graph = ("".join(labels)
                 + "".join(f"[v{i}]" for i in range(len(parts)))
                 + f"concat=n={len(parts)}:v=1:a=0[v]")
        fj = out.with_suffix(".filterjoin.mp4")
        _ff_run([ff, "-y", "-hide_banner", "-loglevel", "error", *ins,
                 "-filter_complex", graph, "-map", "[v]", "-an",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-pix_fmt", "yuv420p", str(fj)], "shard filter-concat")
        joined.unlink(missing_ok=True)
        joined = fj

    _remux_audio(joined, Path(audio), out, shortest=False)
    joined.unlink(missing_ok=True)
    log.info("lipsync: joined %d piece(s) video-only and remuxed the dub "
             "(%d frame(s))", len(parts), _frame_count(out))
    return out


def _frame_rate(video: Path):
    """`r_frame_rate` as ffmpeg writes it, e.g. "24000/1001", or None."""
    from backend.common.media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        return None
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
             str(video)], capture_output=True, text=True, timeout=60)
        val = (r.stdout or "").strip().splitlines()[0].split(",")[0]
        return val if "/" in val and not val.startswith("0") else None
    except Exception:                                    # noqa: BLE001
        return None


def _video_dims(video: Path):
    """(width, height) or None. Needed because a renderer may rescale."""
    from backend.common.media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        return None
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0",
             str(video)], capture_output=True, text=True, timeout=60)
        w, h = (r.stdout or "").strip().splitlines()[0].split(",")[:2]
        return int(w), int(h)
    except Exception:                                    # noqa: BLE001
        return None


def _stream_durations(video: Path) -> tuple:
    """(video_duration, audio_duration) in seconds; 0.0 where unknown."""
    from backend.common.media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        return 0.0, 0.0
    out = {}
    for kind in ("v", "a"):
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", f"{kind}:0",
                 "-show_entries", "stream=duration", "-of", "csv=p=0",
                 str(video)], capture_output=True, text=True, timeout=120)
            first = (r.stdout or "").strip().splitlines()[0]
            out[kind] = float(first.split(",")[0])
        except Exception:                               # noqa: BLE001
            out[kind] = 0.0
    return out.get("v", 0.0), out.get("a", 0.0)


def _needs_remux(video: Path) -> bool:
    """Return whether the audio track needs remuxing."""
    # media_io._ffprobe_bin() rather than shutil.which: there is no ffprobe on this host's PATH, so which() returns None and this answers "no remux needed" for every video.
    from backend.common.media_io import _ffprobe_bin
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


def _render_one(
    *,
    model: str,
    face_video: Path,
    driving_audio: Path,
    out_video: Path,
    out_dir: Path,
    gpus: list,
    lipsync_kwargs: dict,
    coord_list: Optional[list] = None,
    targets_npz: Optional[Path] = None,
):
    """One render of one (video, audio) pair: sharded if given >1 GPU."""
    if len(gpus) > 1:
        return _run_lipsync_sharded(
            model=model, face_video=face_video, driving_audio=driving_audio,
            out_video=out_video, out_dir=out_dir, gpus=gpus,
            lipsync_kwargs=lipsync_kwargs, coord_list=coord_list,
            targets_npz=targets_npz,
        )
    pinned_kwargs = dict(lipsync_kwargs)
    if gpus:
        pinned_kwargs.setdefault("cuda_device", gpus[0])
    lipsync = get_lipsync(model, **pinned_kwargs)
    log.info("lipsync[%s, gpu=%s]: %s + %s → %s",
             model, pinned_kwargs.get("cuda_device"),
             face_video, driving_audio, out_video)
    extra = {}
    if coord_list:
        extra["coord_list"] = coord_list
    if targets_npz:
        extra["targets_npz"] = str(targets_npz)
    return lipsync.sync(LipSyncRequest(
        video=face_video, audio=driving_audio, out_path=out_video,
        extra=extra,
    ))


def _run_lipsync_segmented(
    *,
    model: str,
    face_video: Path,
    driving_audio: Path,
    out_video: Path,
    out_dir: Path,
    gpus: list,
    lipsync_kwargs: dict,
    seg_s: float,
    duration_s: float,
    coord_list: Optional[list] = None,
    targets_npz: Optional[Path] = None,
):
    """Render a long clip in segments, reusing one model per GPU, then join the pieces."""
    import math
    from backend.lipsync import LipSyncRequest, LipSyncResult

    # Devices to spread pieces over.
    devices = list(gpus) if gpus else [None]
    g = max(1, len(devices))
    # Piece length so that `g` concurrent pieces sum to `seg_s` of frames.
    piece_s = seg_s / g
    n = max(1, math.ceil(duration_s / piece_s))
    seg_dir = out_dir / f"{out_video.stem}.segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # `min_chunk_s` guards against a tail sliver; `split_video_audio` returns a single pair when the clip is too short to divide.
    pairs = split_video_audio(face_video, driving_audio, n, seg_dir,
                              min_chunk_s=min(15.0, piece_s))
    if len(pairs) == 1:
        log.info("lipsync[%s]: %.0fs is one piece; rendering in one pass",
                 model, duration_s)
        return _render_one(
            model=model, face_video=face_video, driving_audio=driving_audio,
            out_video=out_video, out_dir=out_dir, gpus=gpus,
            lipsync_kwargs=lipsync_kwargs, coord_list=coord_list,
            targets_npz=targets_npz,
        )

    seg_coords = _split_coords(coord_list, pairs, model=model, what="piece")
    seg_targets = _split_targets(targets_npz, pairs, seg_dir,
                                 model=model, what="piece")

    # Round-robin pieces onto devices IN TIME ORDER; the whole set is reassembled by global index for the concat.
    groups: dict = {d_i: [] for d_i in range(len(devices))}
    for i in range(len(pairs)):
        groups[i % len(devices)].append(i)

    log.info("lipsync[%s]: %.0fs -> %d piece(s) of <=%.1fs across %d GPU(s), "
             "model loaded once per GPU (peak RAM tracks ~%.0fs of frames)",
             model, duration_s, len(pairs), piece_s, len(devices), seg_s)

    from backend.common.memguard import breach as _mem_breach

    # Refuse up front if the host is ALREADY out of memory; the per-batch watchdog inside each worker covers growth DURING the run.
    why = _mem_breach()
    if why:
        return LipSyncResult(
            model_name=model, out_video=out_video, duration_s=0.0, fps=0.0,
            error=(f"not starting lip-sync: {why}. Wait for memory to free "
                   f"up, or raise DUBBING_MEM_GUARD_PCT."))

    def _worker(d_i: int):
        """Render device `d_i`'s pieces in one warm `sync_many` call."""
        device = devices[d_i]
        idxs = groups[d_i]
        if not idxs:
            return d_i, []
        kwargs = dict(lipsync_kwargs)
        if device is not None:
            kwargs.setdefault("cuda_device", device)
        adapter = get_lipsync(model, **kwargs)
        reqs = []
        for i in idxs:
            v, a = pairs[i]
            extra: dict = {}
            if seg_coords[i]:
                extra["coord_list"] = seg_coords[i]
            if seg_targets[i]:
                extra["targets_npz"] = str(seg_targets[i])
            reqs.append(LipSyncRequest(
                video=v, audio=a,
                out_path=seg_dir / f"seg{i:02d}.lipsync.mp4", extra=extra))
        log.info("lipsync[%s, gpu=%s]: warm worker rendering %d piece(s): %s",
                 model, device, len(idxs),
                 ", ".join(f"#{i}" for i in idxs))
        return d_i, adapter.sync_many(reqs)

    # One warm worker per device.
    results_by_idx: dict = {}
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for d_i, res in pool.map(_worker, range(len(devices))):
            for i, r in zip(groups[d_i], res):
                results_by_idx[i] = r

    rendered: list = []
    for i, (v, a) in enumerate(pairs):
        r = results_by_idx.get(i)
        if r is None or r.error:
            # One lost piece means the join cannot be whole, so stop rather than ship a clip with a gap.
            err = r.error if r is not None else "no result returned"
            return LipSyncResult(
                model_name=getattr(r, "model_name", model),
                out_video=out_video, duration_s=0.0, fps=0.0,
                error=(f"piece {i + 1}/{len(pairs)} failed: {err}"))
        got = Path(r.out_video)
        # Conform each piece to ITS OWN source before joining: an audio-driven model can return a different frame count than it was given, and the join must not splice one piece's padding in front of the next.
        try:
            _conform_to_source(got, v, a, model=model)
        except Exception as e:                           # noqa: BLE001
            log.warning("lipsync[%s]: piece %d conform failed (%s)",
                        model, i + 1, e)
        rendered.append(got)

    short = []
    for i, ((v, _a), got) in enumerate(zip(pairs, rendered)):
        src_n, out_n = _frame_count(v), _frame_count(got)
        # Only a SHORTFALL is a problem; a longer render is the dub outlasting the picture, trimmed by the conform above.
        if src_n and out_n and out_n < src_n:
            short.append(f"piece {i + 1}/{len(pairs)}: {out_n} of {src_n}")
    if short:
        return LipSyncResult(
            model_name=model, out_video=out_video, duration_s=0.0, fps=0.0,
            error="frames missing after conform -- " + "; ".join(short),
        )

    _concat_shards_verified(
        rendered, out_video,
        expected=sum(_frame_count(v) for v, _a in pairs),
        audio=driving_audio)

    fps, dur = _probe_video_fps_dur(out_video)
    return LipSyncResult(
        model_name=model, out_video=out_video, duration_s=dur, fps=fps,
        extra={"segmented": {"n": len(pairs), "segment_s": seg_s,
                             "piece_s": piece_s, "warm": True,
                             "gpus": list(gpus)}},
    )


def _run_lipsync_sharded(
    *,
    model: str,
    face_video: Path,
    driving_audio: Path,
    out_video: Path,
    out_dir: Path,
    gpus: list,
    lipsync_kwargs: dict,
    coord_list: Optional[list] = None,
    targets_npz: Optional[Path] = None,
):
    """Split a clip across GPUs, render shards in parallel, then join them."""
    from backend.lipsync import LipSyncResult
    n = len(gpus)
    shard_dir = out_dir / f"{out_video.stem}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    pairs = split_video_audio(face_video, driving_audio, n, shard_dir)
    if len(pairs) == 1:
        # Clip too short to shard profitably -- fall back to one GPU.
        pinned_kwargs = dict(lipsync_kwargs); pinned_kwargs["cuda_device"] = gpus[0]
        lipsync = get_lipsync(model, **pinned_kwargs)
        log.info("lipsync[%s, gpu=%s, sharded→fallback]: %s → %s",
                 model, gpus[0], face_video, out_video)
        extra = {}
        if coord_list:
            extra["coord_list"] = coord_list
        if targets_npz:
            # Nothing was cut, so the whole-clip targets still line up.
            extra["targets_npz"] = str(targets_npz)
        return lipsync.sync(LipSyncRequest(
            video=face_video, audio=driving_audio, out_path=out_video,
            extra=extra,
        ))

    shard_coords = _split_coords(coord_list, pairs, model=model, what="shard")
    shard_targets = _split_targets(targets_npz, pairs, shard_dir,
                                   model=model, what="shard")

    def _one(i_gpu_pair):
        i, gpu, (v, a) = i_gpu_pair
        pinned_kwargs = dict(lipsync_kwargs); pinned_kwargs["cuda_device"] = gpu
        lipsync = get_lipsync(model, **pinned_kwargs)
        part_out = shard_dir / f"part{i:02d}.lipsync.mp4"
        coords = shard_coords[i]
        tgt = shard_targets[i]
        log.info("lipsync[%s, gpu=%d, shard=%d/%d]: %s → %s%s%s",
                 model, gpu, i + 1, len(pairs), v.name, part_out.name,
                 f" ({len(coords)} target boxes)" if coords else "",
                 f" (targets {tgt.name})" if tgt else "")
        extra = {}
        if coords:
            extra["coord_list"] = coords
        if tgt:
            extra["targets_npz"] = str(tgt)
        return lipsync.sync(LipSyncRequest(
            video=v, audio=a, out_path=part_out, extra=extra,
        ))

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

    # Conform EACH SHARD to its own source part, then check parity, then concat -- in that order.
    short = []
    for (i, _gpu, pair), r in zip(jobs, results):
        try:
            _conform_to_source(Path(r.out_video), pair[0], pair[1],
                               model=results[0].model_name)
        except Exception as e:                           # noqa: BLE001
            log.warning("lipsync: shard %d/%d conform failed (%s)", i + 1, n, e)
        src_n, out_n = _frame_count(pair[0]), _frame_count(r.out_video)
        # ONLY a shortfall is a problem.
        if src_n and out_n and out_n < src_n:
            short.append(f"shard {i + 1}/{n}: {out_n} of {src_n} frames "
                         f"({out_n - src_n:+d})")
    if short:
        return LipSyncResult(
            model_name=results[0].model_name, out_video=out_video,
            duration_s=0.0, fps=0.0,
            error=("frame count still wrong after conform -- "
                   + "; ".join(short)),
        )

    _concat_shards_verified([Path(r.out_video) for r in results], out_video,
                            expected=sum(_frame_count(pair[0])
                                         for _i, _g, pair in jobs),
                            audio=driving_audio)
    # Keep the shard dir around for debugging; the concat.txt is already cleaned.

    fps, dur = _probe_video_fps_dur(out_video)
    extra = dict(results[0].extra); extra["sharded"] = {"n": n, "gpus": gpus}
    return LipSyncResult(
        model_name=results[0].model_name, out_video=out_video,
        duration_s=dur, fps=fps, extra=extra,
    )


def _probe_video_fps_dur(path: Path) -> tuple:
    """(fps, duration_s) via ffprobe; (0.0, 0.0) when unavailable."""
    # media_io._ffprobe_bin() rather than shutil.which: there is no ffprobe on this host's PATH, and which() returning None degrades silently to "unknown".
    from backend.common.media_io import _ffprobe_bin
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


def _remux_audio(video: Path, audio: Path, out: Path, *,
                 shortest: bool = True) -> None:
    """Mux `audio` onto `video` as 48 kHz stereo AAC."""
    ff = _ffmpeg_bin()
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
    ]
    if shortest:
        cmd.append("-shortest")
    cmd.append(str(out))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg remux failed (rc={proc.returncode}): {proc.stderr.strip()[:800]}"
        )
