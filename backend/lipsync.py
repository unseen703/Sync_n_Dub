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
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models.lipsync import LipSyncRequest, available_models, get_lipsync

from .gpu import (concat_videos, gpu_busy_pids, gpu_free_memory, gpu_load,
                  pick_devices, split_video_audio, visible_cuda_devices)
from .media_io import _ffmpeg_bin, probe_duration

# Patterns that identify lipsync render outputs (not intermediates like shards).
_RENDER_GLOBS = ("*.lipsync.mp4", "*.lipsync.remux.mp4",
                 "*.lipsync.windowed.mp4")

log = logging.getLogger(__name__)


def _archive_previous_renders(out_dir: Path, model: str) -> Optional[Path]:
    """Move this model's existing renders into ``prev/<timestamp>/``.

    Called at the top of `run_lipsync` so a rerun never overwrites the
    previous render. On failure the old file is still reachable in prev/
    and `ui/media.py` falls back to it.
    """
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


# How far a target list may fall short of the clip it covers before the
# shortfall stops being rounding and starts being a real disagreement.
#
# Two frames, because that is all rounding can produce: a window slice
# rounds its start and end independently (±1 frame between them), and
# `_cut_video` re-encodes, so ffmpeg decides for itself whether the frame
# on the boundary is in or out (±1 more). Past that the list and the clip
# disagree about something — a different fps, a different source — and
# padding would be guessing at which frames the boxes belong to, which is
# the misalignment the whole module refuses elsewhere.
#
# A shortfall inside the tolerance is filled by REPEATING THE LAST BOX, not
# by the `(0,0,0,0)` untouched sentinel: a window that ends mid-sentence
# would otherwise freeze the mouth for its final frame, which is the
# stutter `lipsync_targets.MAX_GAP_FILL` exists to smooth out.
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
    max_segment_s: float = 60.0,
    targets_npz: Optional[Path] = None,
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

    There is no face-restoration pass any more. `models/restore/` and the
    `restore=` argument were removed on 2026-10-17 at the owner's request.
    The lip-sync models still rebuild the mouth at a small working
    resolution — Wav2Lip paints a **96x96** crop, MuseTalk 256x256 — so on
    a high-resolution source the animated region comes back softer than the
    face around it, and nothing in the pipeline now re-sharpens it. Pick
    MuseTalk over Wav2Lip where the face is large in frame; that is the
    only remaining lever.

    `coord_list` constrains WHICH face is animated. Nothing in the adapters
    does this on their own: Wav2Lip takes the first S3FD detection (`d =
    d[0]`) and MuseTalk vendors the same code plus a whole-frame DWPose
    estimate that is never matched to it, so on a multi-face shot both
    animate an arbitrary face and can flip between frames.
    `backend.lipsync_targets.build_coord_list` turns the tracker's face
    tracks plus TalkNet's per-frame scores into one box per frame --
    `(0,0,0,0)` where nobody on screen is speaking, which MuseTalk writes
    through unchanged. MuseTalk honours it today (via its own
    `--use_saved_coord` pickle); the other adapters ignore it rather than
    pretending, so a caller gets the old behaviour instead of a silent
    half-measure.

    It is NOT combinable with `num_gpus > 1` as written: the shards are cut
    by time and each adapter sees its own frame numbering, so a list indexed
    against the whole video would be applied to the wrong frames. Supplying
    one forces the single-GPU path, with a warning, rather than rendering
    mouths in the wrong places.

    `window=(start_s, end_s)` renders only that slice and splices it back
    into the full timeline, so the output is still a complete video. The
    parts outside the window are NOT passed through untouched at the byte
    level -- the concat re-encodes everything once -- but they are not sent
    to the lip-sync model, which is where the cost and the risk are. When
    no pre-muxed video was handed in, the head and tail are cut from a
    freshly muxed copy of the source carrying `driving_audio`; cutting them
    from the raw source would leave the untouched stretches speaking the
    ORIGINAL language.

    `max_segment_s` (default 60, 0 disables) caps how much video goes
    through the model at once. Anything longer is cut into <=1 min pieces,
    rendered ONE AFTER ANOTHER, and concatenated. This is a MEMORY bound,
    not a speed one -- it makes peak RAM a function of the segment length
    instead of the clip length.

    Lowered 120 -> 60 on 2026-10-23 by request. It buys RAM headroom and
    COSTS wall time, which is the trade to know before moving it: every
    segment pays a fresh model load per shard, and that fixed cost is the
    dominant term at this clip length. Measured on job 20261023-011027
    (150s, 1080p25, 2 GPUs, 348 of 3775 frames animated), a 37.48s shard
    took 4m55s-6m13s whether it animated 348 frames or none at all -- so
    the render is paying for process startup, decode and re-encode, not for
    the UNet. Halving the segment adds a third round of that.

    It exists because both vendored models load the whole clip into RAM:
    MuseTalk's `read_imgs()` and Wav2Lip's `full_frames.append(frame)` each
    retain one decoded uint8 frame per source frame. Measured on this host,
    MuseTalk is 5.87 GiB + 6.14 MiB/frame -- which matches a 1920x1080x3
    frame (5.93 MiB) to within 3%, i.e. it IS the pixels. A 7-minute 1080p
    clip therefore projects to ~70 GiB against 62 GiB of system RAM, and
    fails on host memory long before VRAM. At 2 minutes the same fit gives
    ~24 GiB, which fits with room for a neighbour.

    Sharding does not substitute for it: `num_gpus=2` halves the frames
    per PROCESS but the processes run at the same time, so the total
    resident set is unchanged. Segments run sequentially, so they bound it.
    The two compose -- each segment still uses whatever GPUs it was given.

    Like `coord_list` with sharding, a supplied `coord_list` disables
    segmentation: the boxes are indexed against the whole video and each
    segment renumbers frames from zero.

    `targets_npz` is LatentSync's answer to the same question `coord_list`
    answers for MuseTalk -- WHICH face to animate -- but it carries 106
    landmarks per frame rather than a box, because that is the only thing
    LatentSync's detector contributes. Unlike `coord_list` it does NOT
    disable sharding or segmenting: `_split_targets` cuts it the way the
    video is cut and writes one file per piece, so each adapter gets an
    array whose length matches the clip it was handed. Adapters that do not
    understand it ignore it.
    """
    if model not in available_models():
        raise ValueError(
            f"Unknown lip-sync model {model!r}. "
            f"Available: {', '.join(available_models())}"
        )
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _archive_previous_renders(out_dir, model)
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
        if coord_list:
            # The cut clip renumbers frames from zero, so a list indexed
            # against the whole video would animate the wrong frames here —
            # the same hazard sharding and segmenting refuse below, except
            # that this one cannot be answered by "render in one pass". It
            # CAN be answered exactly: we know the window and the fps, so
            # slice the list to match the clip the model will actually see.
            # The FULL video's fps, not the cut's: `f0` is an index into the
            # original timeline. They are the same today (`_cut_video`
            # re-encodes at the source rate), but if that ever stopped being
            # true the offset would be computed against the wrong clock.
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

    gpus = pick_devices(num_gpus)
    if not gpus and visible_cuda_devices():
        # Cards exist but none clears the free-VRAM floor. Starting anyway
        # means an unpinned worker on whichever card the driver picks, an
        # OOM part-way in, and -- because MuseTalk catches Exception and
        # exits 0 -- a run that reports "produced no mp4" after ASR, MT,
        # TTS and the mix have all been paid for. Refuse up front instead.
        free = gpu_free_memory()
        busy = gpu_busy_pids()
        # Since 2026-10-22 the gate is a LOAD test, not "is anyone there", so
        # the refusal has to quote the numbers it judged on -- a bare pid
        # reads as the old one-tenant rule and sends the operator looking for
        # a process to kill when the card is merely working.
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
    # Sharding and segmenting both used to be DISABLED whenever targets were
    # supplied, because each cuts the clip and renumbers frames from zero,
    # so a whole-video list landed on the wrong frames in every piece but
    # the first. That cost the second GPU and, worse, the memory bound —
    # `_run_lipsync_segmented` exists to keep peak RSS a function of the
    # segment rather than the clip, and turning it off put a long render
    # back on course to OOM. `_split_coords` cuts the list the same way the
    # video is cut, so neither has to be given up.
    # Refuse up front if the host is ALREADY out of memory. The per-render
    # watchdog protects against growth during a render; this one costs
    # nothing and avoids loading a model into a machine that cannot hold
    # it. Same shape as the GPU check above: refusing is cheaper than
    # being killed partway through.
    from .memguard import breach as _mem_breach, memory_snapshot
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
    # The 1.05 slack stops a 2:03 clip becoming two parts, one of them three
    # seconds long: the split/concat overhead and an extra model load cost
    # more than the memory a sliver saves.
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

    # Some adapters (notably Wav2Lip) output video-only or re-encode audio
    # losslessly; if the result has no audio track OR its audio doesn't
    # match the driving audio, re-mux the dubbed audio in.
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
            # The render is good; only the splice failed. Hand back the
            # window render and say so rather than losing the GPU work.
            log.error("lipsync: window splice failed (%s); returning the "
                      "window render only: %s", e, final_video)
            result.extra["window_splice_error"] = str(e)
        else:
            result.extra["window"] = [w_start, w_end]
            result.extra["window_spliced_into"] = str(splice_into)

    # An adapter exiting 0 having written nothing is a REAL failure mode here,
    # not a hypothetical: Wav2Lip's inference.py muxes with subprocess.call and
    # discards the return code, and MuseTalk shelled into the wrong interpreter
    # dies on import while the wrapper still returns. Both leave rc=0 and no
    # file. Trusting `result.error` alone therefore reports success for a run
    # that produced nothing, so the file itself is the arbiter.
    try:
        produced = final_video.is_file() and final_video.stat().st_size > 0
    except OSError:
        produced = False
    if not result.error and not produced:
        result.error = (f"{model} reported success but wrote no output to "
                        f"{final_video}")

    # FRAME PARITY. A lip-sync pass repaints mouths; it must not change how
    # many frames there are. MuseTalk DELETES any frame it cannot find a face
    # in -- `scripts/inference.py:167` skips `coord_placeholder` frames out of
    # the latent list, and the blend loop's bare `except: continue` (line 219)
    # never writes their PNG. The survivors are then reassembled at a fixed
    # fps, so the clip comes out SHORTER in frames and LONGER in seconds, and
    # the no-face stretch plays as a freeze.
    #
    # Measured 2026-10-18 on a 40 s Dark Knight clip: source 960 frames /
    # 40.040 s, output 950 frames / 40.499 s, with 12-20 s byte-identical
    # frame to frame while the source moves. Nothing errored and the file
    # played, so only counting frames catches it.
    if produced and not result.error and splice_into is None and window is None:
        # A truncated video stream under a full-length audio stream is what
        # the viewer actually experiences as a freeze: the player holds the
        # last decoded frame. Measured on shard 0 of the 2026-10-18 clip --
        # video 10.64 s, audio 20.02 s, i.e. 9.4 s of held picture. The
        # container duration reports the LONGER stream, so neither the file
        # duration nor a frame count alone reveals it.
        # Pad first, then judge: a render that is short only because the
        # audio ran out is recoverable, and the viewer must not lose picture.
        try:
            _conform_to_source(final_video, source_video,
                               driving_audio, model=model)
        except Exception as e:                           # noqa: BLE001
            log.warning("lipsync[%s]: conform failed (%s); "
                        "the parity check below will report the shortfall",
                        model, e)

        # The reference for "was the picture truncated" is the SOURCE video,
        # never the driving audio. Comparing against the audio was wrong and
        # failed a correct render on 2026-10-19: source 60.02 s / 1439
        # frames, render conformed to 1439, but the dubbed track was 62.22 s
        # because the TTS overran its slot -- so the check announced a 2.2 s
        # freeze that did not exist. The dub outrunning the source is a mix
        # concern, and `_remux_audio` already trims it with `-shortest`.
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
            # Nothing downstream trims this -- MuseTalk's own mux has no
            # `-shortest`, and `_needs_remux` is False when the audio track
            # already matches, so the delivered file ends with the picture
            # frozen for the overhang (measured 2026-10-19: video 60.02s,
            # audio 62.22s, 2.2s of held frame). Trim it here.
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
            # A surplus is recoverable and is NOT a parity failure: the
            # excess is the dub outlasting the picture (see the overshoot
            # branch in _conform_to_source). Trim it and carry on rather
            # than failing a render that is correct for its whole length.
            _conform_to_source(final_video, source_video, driving_audio,
                               model=model)
            out_n = _frame_count(final_video)
        if src_n and out_n and out_n > src_n:
            # Still over after the trim above. Not an error: the extra frames
            # are an audio tail with no picture behind them in the source.
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

    # "done" used to be logged unconditionally, with the error tucked into a
    # trailing parenthetical after a wall of upstream banner noise -- so a
    # failed render read as a successful one in the log and in the UI's own
    # progress scrape. The verdict leads now, and a failure is not INFO.
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
    """Frames per second, or 0.0 when it cannot be determined.

    cv2 rather than ffprobe, deliberately. `media_io._ffprobe_bin()` can
    answer `None`, and `None` is a SILENT answer in every caller — CLAUDE.md
    §1 records `fps=0.0` on a file that plays perfectly as exactly this bug.
    The one caller here treats 0.0 as "unknown" and drops the target boxes
    rather than aligning them against a guess, so a wrong number would be
    worse than no number.
    """
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
    """Frame count, or 0 when it cannot be determined.

    Used to split a `coord_list` across the pieces a sharded or segmented
    render is cut into. Counting ACTUAL frames beats recomputing time
    boundaries: `split_video_audio` divides by duration and gives the last
    chunk whatever is left over, so the arithmetic would have to reproduce
    both that asymmetry and ffmpeg's own rounding. A frame count cannot
    drift from the file it was read off.
    """
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


def _slice_coords_to_window(coord_list: list, window_video: Path,
                            w_start: float, w_end: float, fps: float,
                            *, model: str) -> Optional[list]:
    """The slice of `coord_list` covering the cut window clip.

    The OFFSET is arithmetic — `f0` indexes the original timeline, which
    only the window and the fps describe. The LENGTH is probed off the clip
    that was actually cut, with the arithmetic as a fallback when the probe
    cannot read it.

    The length used to be arithmetic too (`round(w_end*fps) - f0`), and the
    two rulers disagreed by a frame whenever the window's ends rounded in
    opposite directions. Measured 2026-10-23 on job 20261023-011027: a
    0.672-150.556s window at 25 fps sliced 3747 boxes for a clip ffmpeg had
    cut to 3748 frames. `_split_coords` measures with the probe, found
    itself one box short on the second of two segments, and answered by
    dropping the targeting for that whole segment — so 75 s of a 150 s dub
    rendered on MuseTalk's own `d = d[0]` first face, at exit code 0, with
    every artifact looking correct. Measuring both with the probe is what
    stops the shortfall arising in the first place; `_COORD_PAD_FRAMES`
    absorbs what ffmpeg's own rounding can still add.
    """
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
    """One slice of `coord_list` per (video, audio) pair, in order.

    The list is POSITIONAL and indexed against the clip it was built from:
    entry `i` is the box for frame `i`. Every path that cuts the clip
    renumbers frames from zero, so each piece needs its own slice or the
    boxes land on the wrong frames — silently, at exit code 0.

    Walks a cursor over the pieces' real frame counts. Returns a list the
    same length as `pairs`, holding None where the slice cannot be trusted
    (an unreadable frame count, or a list that ran out), because dropping
    the targeting for one piece is recoverable and misaligning it is not.
    """
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
            # A rounding-sized shortfall on the final piece. Dropping the
            # targeting for a whole shard over one frame is the expensive
            # answer to the cheap problem: it costs every frame in that
            # piece, not the frame that is missing.
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
    """One targets .npz per (video, audio) pair, sliced and renumbered.

    The targets twin of `_split_coords`, and the same contract: walk the
    pieces' MEASURED frame counts with a cursor, and return None for any
    piece whose slice cannot be trusted. A piece that renders without
    targeting falls back to LatentSync's own detector -- worse, but correct --
    whereas a misaligned slice animates the wrong face at exit code 0.

    Per piece rather than one file for all of them because the consumer
    checks length: `lipsync_pipeline` compares the targets against the clip
    it was handed and drops a mismatch. Handing every shard the whole-video
    array would therefore disable targeting on all of them.
    """
    if not targets_npz:
        return [None] * len(pairs)

    from .lipsync_targets import load_targets_payload, save_targets, slice_targets

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
            # A rounding-sized shortfall on the last piece is the clip and
            # the list disagreeing by a frame, which the consumer's own
            # length check would reject anyway. Say so here, where the piece
            # is named, rather than letting it fail silently downstream.
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


def _frame_count(video: Path) -> int:
    """Exact frame count, or 0 when ffprobe cannot answer.

    Counted rather than derived from duration x fps: the whole point is to
    catch an output whose duration and frame count disagree, so deriving one
    from the other would hide exactly the fault being looked for.
    """
    # Imported here, as elsewhere in this module: there is no ffprobe on this
    # host's PATH, and media_io resolves the one in the running env.
    from .media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        return 0
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
             str(video)], capture_output=True, text=True, timeout=300)
        # `-of csv=p=0` emits a trailing comma ("960,"), so int() on the raw
        # line raises and the except would quietly answer 0 -- which reads as
        # "cannot probe" and skips the check entirely. Take the first field.
        first = (out.stdout or "").strip().splitlines()[0]
        return int(first.split(",")[0])
    except Exception:                                # noqa: BLE001
        return 0


def _conform_to_source(rendered: Path, source: Path, driving_audio: Path, *,
                       model: str) -> int:
    """Pad a short render with the SOURCE tail so the picture never stops.

    MuseTalk renders `len(whisper_chunks)` frames -- a count derived from the
    AUDIO (`scripts/inference.py:211`), not from the video. So even with every
    frame written, the render can be shorter than its source: measured
    2026-10-18, a 480-frame shard with 20.020 s of audio produced 460.

    Once placeholder frames are written rather than skipped the PNG indices
    are contiguous from 0, so frame i of the render is frame i of the source
    and the whole shortfall sits at the END. The tail is therefore safe to
    append: the dub stays lip-synced for as long as there was audio to drive
    it, and the picture then continues exactly as the source plays.

    Frame-exact, not time-based. A re-encoded `-ss/-to` cut overshot by 22
    frames when first tried here (460 + 42 for a 20-frame request), because
    the rendered file's nominal fps and its real frame spacing disagree.
    `select='gte(n,M)'` with `-vsync 0` counts frames instead.

    Video only: the tail carries the source's ORIGINAL audio, which must not
    reach the dub, so both halves are muxed silent and the driving audio is
    laid back over the join. Returns frames appended (0 when already equal).
    """
    src_n, out_n = _frame_count(source), _frame_count(rendered)
    if not src_n or not out_n or out_n == src_n:
        return 0

    # OVERSHOOT. The render can be LONGER than its source, and the cause is
    # the mirror image of the shortfall below: Wav2Lip and MuseTalk both
    # render one frame per audio chunk, so when the driving audio outlasts
    # the picture they keep going past the last frame they have. The dub
    # routinely does outlast it -- a timing plan that cannot fit the speech
    # into its slot leaves the overhang on the end -- and time-sharding
    # hands that whole overhang to the LAST shard. Measured 2026-10-19:
    # shard 1 frame-exact at 480, shard 2 at 488 for the same 480-frame
    # source, the extra 0.334s matching the audio tail to the frame.
    #
    # Those frames have no picture behind them, so they are dropped rather
    # than kept. Verified before trusting it: on two 1500-frame shards the
    # first src_n frames aligned with the source at offset +0 (MAE 1.1 and
    # 2.7) while everything past src_n matched nothing nearby (MAE 8.4) --
    # the excess really is trailing, so a head trim is the right cut.
    #
    # This case was simply absent: the guard above read `out_n >= src_n`,
    # so an overshoot returned 0 having done nothing, and the caller's
    # `out_n != src_n` check then failed the run with "frames still missing
    # after conform" -- reporting a surplus as a shortage and discarding a
    # correct render. It cost three batches before it was fixed.
    if out_n > src_n:
        surplus = out_n - src_n
        work = rendered.parent / f"{rendered.stem}.conform"
        work.mkdir(parents=True, exist_ok=True)
        ff = _ffmpeg_bin()
        trimmed = work / "trimmed.mp4"
        # `-r` is NOT optional. Without it ffmpeg re-times a `select` +
        # `setpts` output at its own default of 25 fps: measured 2026-10-19,
        # a 23.976 fps shard came back tagged 25/1 with time_base 1/12800
        # while its sibling kept 24000/1001 and 1/24000. The shard then LOOKS
        # unprocessed, and `concat_videos`' demuxer -- which needs identical
        # stream parameters and drops a mismatch without failing -- silently
        # threw almost all of it away: 480 + 480 frames concatenated to 504.
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
        # `shortest=True` here, unlike the pad path below: the surplus IS
        # the audio outlasting the picture, so the soundtrack is the thing
        # that has to give.
        _remux_audio(trimmed, driving_audio, rendered, shortest=True)
        # ...except `-shortest` cuts to whichever stream is shorter, and it
        # cannot tell which one we meant. `trimmed` is already exactly src_n
        # frames, so when the shard's dub comes back a hair SHORTER than its
        # picture the mux silently takes the video back BELOW src_n and the
        # parity check fails a render that was correct. Measured 2026-10-24
        # on job 20261020-043027: shard 2 rendered 576, trimmed cleanly to
        # 575, then remuxed to 573 against a 575-frame source.
        #
        # Redo it keeping the picture whole. A soundtrack that ends early is
        # recoverable -- the caller trims an overhang, and silence at the tail
        # is survivable -- whereas frames deleted here are simply gone.
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

    # The tail must match the RENDER's dimensions, not the source's. Wav2Lip
    # writes at `resize_factor` scale -- which since 2026-10-22 is resolved
    # per source against a 1080p ceiling, so a 4K input renders 1920x1080
    # while a 1080p input renders at its own size -- and ffmpeg's concat
    # demuxer requires identical
    # parameters, so a full-size tail is silently dropped: measured
    # 2026-10-19, the conform logged "+3 frames" and the file gained none.
    dims = _video_dims(rendered) or _video_dims(source)
    scale = f",scale={dims[0]}:{dims[1]}" if dims else ""
    rate = _frame_rate(rendered) or _frame_rate(source)

    tail = work / "tail.mp4"          # source frames out_n..src_n-1, silent
    _ff_run([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
             # `setpts=PTS-STARTPTS` is load-bearing: `select` keeps the
             # SOURCE timestamps, so a 3-frame tail taken from 14.9s carries
             # that PTS span and the concat re-encode pads it out -- measured
             # 2026-10-19, a 3-frame request became 18 frames. Rebasing to
             # zero and pinning the rate makes the tail exactly `missing`
             # frames long.
             "-vf", f"select='gte(n\\,{out_n})'{scale},setpts=PTS-STARTPTS",
             # No `-vsync 0` alongside `-r`: ffmpeg 7 rejects the pair
             # ("One of -r/-fpsmax was specified together a non-CFR
             # -vsync/-fps_mode"). Constant rate plus a frame cap is what
             # we want anyway -- exactly `missing` frames at the head's rate.
             *(["-r", rate] if rate else ["-vsync", "0"]),
             "-frames:v", str(missing),
             "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", str(tail)], "conform tail")

    joined = work / "joined.mp4"
    # The concat FILTER, not `concat_videos`' demuxer. The demuxer requires
    # byte-identical stream parameters and drops a mismatch without failing:
    # measured 2026-10-19, head and tail were both 960x540 at 23.976 fps but
    # carried different timebases (1/24000 vs 1/11988) because one came from
    # wav2lip and one from an ffmpeg select filter, and the 3-frame tail was
    # silently discarded while the conform reported success. The filter
    # re-encodes through a graph and normalises both.
    _ff_run([ff, "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(head), "-i", str(tail),
             "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
             "-map", "[v]", "-c:v", "libx264", "-preset", "fast",
             "-crf", "18", "-pix_fmt", "yuv420p", str(joined)],
            "conform concat")
    # The concat is only believable if it actually gained the frames: the
    # demuxer drops a mismatched input without failing, so verify instead
    # of assuming.
    gained = _frame_count(joined)
    if gained and gained != src_n:
        log.warning("lipsync[%s]: conform produced %d frames, wanted %d "
                    "(head %d + tail %d)", model, gained, src_n, out_n, missing)
    # NOT `-shortest`: the whole point here is that the picture outlives the
    # soundtrack. Truncating to the audio would delete the frames just added.
    _remux_audio(joined, driving_audio, rendered, shortest=False)
    log.info("lipsync[%s]: conformed +%d frame(s) of untouched source "
             "(render covered %d of %d)", model, missing, out_n, src_n)
    return missing


def _concat_shards_verified(parts: list, out: Path, *, expected: int,
                            audio: Optional[Path] = None) -> Path:
    """Join shards/pieces, then CHECK the join actually kept every frame.

    `concat_videos` uses ffmpeg's concat demuxer, and its docstring assumes
    every input shares codec parameters because each adapter re-encodes with
    its own defaults. That assumption dies the moment `_conform_to_source`
    runs: it re-encodes ONLY the shard that needed padding or trimming and
    leaves the rest untouched, so one shard can carry a different frame rate
    or timebase than its siblings.

    The demuxer's response to a mismatch is the dangerous one -- it drops or
    mis-times the offending input, exits 0, and raises nothing. Measured
    2026-10-19: two 480-frame shards concatenated to 504 frames with rc=0,
    which then looked like a lip-sync fault rather than a join fault.

    So the fast path still runs, but its result is counted. A short join is
    redone through the concat FILTER, which decodes both inputs into one
    graph and normalises the timebases instead of demanding they match --
    the same reason `_conform_to_source` uses the filter internally.

    `audio` lays a known-good track over the join instead of carrying the
    pieces' own audio. A per-piece audio stream is not always re-decodable
    where it was NOT written: LatentSync muxes each piece with its own conda
    env's ffmpeg (`-c:a aac -q:a 0`, vendor .../lipsync_pipeline.py), and that
    native VBR AAC is rejected by the dubbing env's ffmpeg 7.1 decoder at
    concat time ("Reserved bit set / Prediction is not allowed in AAC-LC",
    rc=234) -- the failure surfaced once warm rendering left more pieces
    frame-exact, so `_conform_to_source` no-oped and stopped normalising the
    audio as a side effect. The per-piece audio is throwaway anyway (the dub
    covers the whole span), so when `audio` is supplied the pieces are joined
    VIDEO-ONLY -- no upstream audio stream is ever decoded -- and the dub is
    remuxed on. Without it the audio-carrying join is kept unchanged.
    """
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
    from .media_io import _ffprobe_bin
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
    from .media_io import _ffprobe_bin
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
    """(video_duration, audio_duration) in seconds; 0.0 where unknown.

    Per stream, not the container: the container reports the LONGEST stream,
    so a short picture under a full-length soundtrack looks correct there.
    """
    from .media_io import _ffprobe_bin
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
    """One render of one (video, audio) pair: sharded if given >1 GPU.

    Factored out so the segment loop and the un-segmented path cannot
    drift: both reach the adapter through here, so a change to pinning or
    to the sharded/single decision lands on both.
    """
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
    """Render a long clip as small pieces, warm model per GPU, then join.

    A memory bound, not a speed one. Both vendored models hold one decoded
    uint8 frame per source frame for the whole clip, so peak RSS is linear
    in LENGTH -- measured 5.87 GiB + 6.14 MiB/frame for MuseTalk at 1080p,
    which puts a 7-minute clip at ~70 GiB on a 62 GiB host. Cutting the clip
    into pieces makes peak RAM a function of the piece, not the clip.

    **Warm model.** The pieces for one GPU go to a single `sync_many` call,
    so the adapter loads the ~9 GiB of weights ONCE and cycles only the
    per-piece frame buffers in and out. The old path spawned a fresh process
    and reloaded the model for every piece, which the `run_lipsync` docstring
    measured as the dominant cost at this clip length (a 37 s shard cost
    ~5 min whether it animated 348 frames or none).

    **Peak RAM is held constant across GPU counts.** Each GPU's warm worker
    processes its pieces SEQUENTIALLY, so one GPU holds one piece's frames at
    a time. To keep the TOTAL resident set at `seg_s` worth of frames while
    `g` GPUs run concurrently, the piece is sized `seg_s / g`: `g` pieces in
    flight x `seg_s/g` each = `seg_s`. This is the same bound the old
    shard-within-segment path held (two half-segment shards resident), now
    reached without paying a model load per piece.
    """
    import math
    from models.lipsync import LipSyncRequest, LipSyncResult

    # Devices to spread pieces over. `[None]` means "inherit the parent env's
    # default GPU" -- the same convention `_render_one` uses when `gpus` is
    # empty -- so the single-worker path below still runs.
    devices = list(gpus) if gpus else [None]
    g = max(1, len(devices))
    # Piece length so that `g` concurrent pieces sum to `seg_s` of frames.
    piece_s = seg_s / g
    n = max(1, math.ceil(duration_s / piece_s))
    seg_dir = out_dir / f"{out_video.stem}.segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # `min_chunk_s` guards against a tail sliver; `split_video_audio`
    # returns a single pair when the clip is too short to divide, which the
    # branch below handles as the one-piece case.
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

    # Round-robin pieces onto devices IN TIME ORDER, so each device's warm
    # worker still gets a contiguous-enough spread and the whole set is
    # reassembled by global index for the concat.
    groups: dict = {d_i: [] for d_i in range(len(devices))}
    for i in range(len(pairs)):
        groups[i % len(devices)].append(i)

    log.info("lipsync[%s]: %.0fs -> %d piece(s) of <=%.1fs across %d GPU(s), "
             "model loaded once per GPU (peak RAM tracks ~%.0fs of frames)",
             model, duration_s, len(pairs), piece_s, len(devices), seg_s)

    from .memguard import breach as _mem_breach

    # Refuse up front if the host is ALREADY out of memory: loading a model
    # per GPU into a machine that cannot hold a piece's frames only buys a
    # render killed a minute later. The per-batch watchdog inside each worker
    # covers growth DURING the run.
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

    # One warm worker per device, in parallel. Sequential WITHIN a worker
    # (sync_many loops its pieces), concurrent ACROSS workers -- which is what
    # keeps peak RAM at `seg_s` while both cards stay busy.
    results_by_idx: dict = {}
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for d_i, res in pool.map(_worker, range(len(devices))):
            for i, r in zip(groups[d_i], res):
                results_by_idx[i] = r

    rendered: list = []
    for i, (v, a) in enumerate(pairs):
        r = results_by_idx.get(i)
        if r is None or r.error:
            # One lost piece means the join cannot be whole, so stop rather
            # than ship a clip with a gap. A shared cause (busy card, missing
            # checkpoint) would hit every piece anyway.
            err = r.error if r is not None else "no result returned"
            return LipSyncResult(
                model_name=getattr(r, "model_name", model),
                out_video=out_video, duration_s=0.0, fps=0.0,
                error=(f"piece {i + 1}/{len(pairs)} failed: {err}"))
        got = Path(r.out_video)
        # Conform each piece to ITS OWN source before joining: an audio-driven
        # model can return a different frame count than it was given, and the
        # join must not splice one piece's padding in front of the next.
        try:
            _conform_to_source(got, v, a, model=model)
        except Exception as e:                           # noqa: BLE001
            log.warning("lipsync[%s]: piece %d conform failed (%s)",
                        model, i + 1, e)
        rendered.append(got)

    short = []
    for i, ((v, _a), got) in enumerate(zip(pairs, rendered)):
        src_n, out_n = _frame_count(v), _frame_count(got)
        # Only a SHORTFALL is a problem; a longer render is the dub
        # outlasting the picture and is trimmed by the conform above.
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
    """Fan one job across `gpus`: split in time, run in parallel, concat.

    Returns a LipSyncResult-like object (model_name, out_video, duration_s,
    fps, error, extra) assembled from the final concat.

    A `coord_list` is split across the shards by frame count — see
    `_split_coords`. This used to cost the second GPU: `run_lipsync` forced
    a single device whenever targets were supplied, because a whole-video
    list applied to a shard that renumbers frames from zero animates the
    wrong frames.
    """
    from models.lipsync import LipSyncResult
    n = len(gpus)
    shard_dir = out_dir / f"{out_video.stem}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    pairs = split_video_audio(face_video, driving_audio, n, shard_dir)
    if len(pairs) == 1:
        # Clip too short to shard profitably — fall back to the single GPU.
        # The list still applies: nothing was cut, so it is still indexed
        # against the clip the model sees.
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

    # Conform EACH SHARD to its own source part, then check parity, then
    # concat -- in that order. A shard is routinely short: MuseTalk renders
    # `len(whisper_chunks)` frames, a count driven by the AUDIO, so measured
    # 2026-10-18 the two halves of one clip came back 425 of 444 and 436 of
    # 444. That is recoverable, not a failure, and the conform recovers it.
    #
    # It has to happen per shard rather than once on the concat: the join
    # would otherwise splice the source tail of shard 1 after shard 2, i.e.
    # out of order. It also has to happen BEFORE the parity check, or the
    # check rejects exactly the shortfall the conform was built to absorb --
    # which is what this code did when the check landed first, turning a
    # flawed render into a hard failure.
    short = []
    for (i, _gpu, pair), r in zip(jobs, results):
        try:
            _conform_to_source(Path(r.out_video), pair[0], pair[1],
                               model=results[0].model_name)
        except Exception as e:                           # noqa: BLE001
            log.warning("lipsync: shard %d/%d conform failed (%s)", i + 1, n, e)
        src_n, out_n = _frame_count(pair[0]), _frame_count(r.out_video)
        # ONLY a shortfall is a problem. A render longer than its source is
        # the dub outlasting the picture; the conform above trims it, and if
        # anything is still over it is surplus tail with no source frame
        # behind it -- nothing is lost by it being there. Failing on it is
        # what discarded three batches of correct renders.
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


def _remux_audio(video: Path, audio: Path, out: Path, *,
                 shortest: bool = True) -> None:
    """Mux `audio` onto `video` as 48 kHz stereo AAC — the broadest-compatibility
    audio profile. Wav2Lip's native 16 kHz mono passes ffprobe but is silently
    rejected by several common players.

    `shortest=False` keeps every video frame even when the audio runs out
    first. `-shortest` truncates the OUTPUT to the shorter input, so muxing a
    marginally shorter soundtrack onto a conformed video silently deletes the
    tail frames the conform had just restored: measured 2026-10-19, a shard
    conformed 690 -> 720 came back 717 because the audio was ~3 frames short
    of the picture. Callers restoring video continuity must pass False.
    """
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
