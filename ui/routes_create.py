"""Job creation: probe a URL, validate it, allocate a folder, launch.

Mounted by `ui/server.py` as `router`; this module owns no app and no
middleware.

Validation runs **before any work** — before yt-dlp is invoked, before a job
folder exists, before a GPU is touched — in this order:

 1. **URL scheme and host allowlist.** This is the SSRF answer. yt-dlp is a
    general-purpose fetcher: pointed at `http://169.254.169.254/` it will
    cheerfully try, and this server runs on a machine with credentials in its
    environment. The allowlist is the control; `DUBBING_UI_ALLOWED_HOSTS`
    widens it deliberately rather than by accident.
 2. **The post-probe `webpage_url` host, re-checked.** A shortener on an
    allowed host, or an extractor that follows a redirect, can land somewhere
    the first check would have refused. Checking only the submitted string
    makes the allowlist advisory.
 3. **Live and upcoming streams, refused.** They have no duration, so the
    window and the cap mean nothing.
 4. **Window sanity**, then
 5. **the lip-sync cap.** The cap is stated last in the plan, but it is
    computed *from* the window, so the window has to be sane first — a
    negative duration passes any cap.
 6. **The language**, against `ui.langs`, which builds its list from the
    adapters rather than from a copied table.

The cap is on the **lip-synced portion only**. A whole-video audio dub of a
two-hour lecture is fine; it is 20 minutes of lip-sync that turns into
between one and seven hours of GPU time on this host, and that is the reason
the refusal gives.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sys
import uuid as _uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import (APIRouter, File, HTTPException, Response,
                     UploadFile, status)
from fastapi.concurrency import run_in_threadpool

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:                           # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT))

from ui import jobqueue, langs, presets, ytmeta               # noqa: E402
from ui.detect_language import detect_from_probe               # noqa: E402
from ui.progress import plan_stages, read_progress, write_atomic  # noqa: E402
from ui.schemas import MIN_WINDOW_S, JobRequest, ProbeRequest    # noqa: E402
from ui._io import data_dir as _data_dir, env_int as _env_int   # noqa: E402

log = logging.getLogger("ui.routes_create")

DATA_DIR = _data_dir()
JOBS_DIR = DATA_DIR / "jobs"

#: Where yt-dlp may be pointed. Everything else is refused before the probe.
DEFAULT_ALLOWED_HOSTS = (
    "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
)

#: Lip-sync runs at 3-21x clip length here, so 20 minutes of video is between
#: one and seven hours of GPU. Longer is refused at submission with the
#: reason, rather than queued and discovered tomorrow.
MAX_LIPSYNC_S_DEFAULT = 20 * 60

router = APIRouter()


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def allowed_hosts() -> List[str]:
    raw = os.environ.get("DUBBING_UI_ALLOWED_HOSTS")
    if not raw:
        return list(DEFAULT_ALLOWED_HOSTS)
    hosts = [h.strip().lower() for h in raw.split(",") if h.strip()]
    return hosts or list(DEFAULT_ALLOWED_HOSTS)


def max_lipsync_s() -> int:
    return _env_int("DUBBING_UI_MAX_LIPSYNC_S", MAX_LIPSYNC_S_DEFAULT)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _bad(msg: str) -> HTTPException:
    return HTTPException(status.HTTP_400_BAD_REQUEST, msg)


def check_url(url: str, *, what: str = "That link") -> str:
    """Scheme + host allowlist. Returns the host; raises 400 otherwise.

    An IP literal is refused by the same rule that refuses any other host:
    `169.254.169.254` is not in the allowlist, so it never needs a special
    case — which is what makes the control hard to get wrong later.
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in ("http", "https"):
        raise _bad(f"{what} must be an http or https URL.")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise _bad(f"{what} has no host.")
    allowed = allowed_hosts()
    if host not in allowed:
        raise _bad(
            f"{what} points at {host}, which is not an allowed source. "
            f"This server only fetches from: {', '.join(allowed)}. "
            f"Set DUBBING_UI_ALLOWED_HOSTS to widen that deliberately."
        )
    return host


def check_window(window, duration_s: Optional[float]) -> Optional[Dict[str, float]]:
    """`None` for the whole video, else a sane `{start_s, end_s}`."""
    if window is None:
        return None
    start, end = float(window.start_s), float(window.end_s)
    if start < 0:
        raise _bad("The start of the window cannot be negative.")
    if end <= start:
        raise _bad("The window ends at or before it starts.")
    if end - start < MIN_WINDOW_S:
        raise _bad(f"That window is {end - start:.1f} s long. Use at least "
                   f"{MIN_WINDOW_S:g} s — below that there is nothing for "
                   f"diarization to segment.")
    if duration_s and end > float(duration_s) + 0.5:
        raise _bad(f"The window ends at {end:.0f} s but the video is only "
                   f"{float(duration_s):.0f} s long.")
    return {"start_s": start, "end_s": end}


def check_lipsync_cap(lipsynced_s: Optional[float], *, model: Optional[str]
                      ) -> None:
    """The compute cap, on the lip-synced portion only.

    **No longer reached from a submission.** Since 2026-10-20 nothing
    lip-syncs at submit time, so `model` is always None here and this
    returns immediately. The live cap moved to
    `ui/routes_edits._lipsync_blockers`, which is where the GPU hours are
    actually committed and where the model is known.

    Kept — rather than deleted — because it still correctly caps any caller
    that DOES name a model, and because deleting it would quietly remove the
    only enforcement on an upload path that might grow one later.
    """
    if not model or not lipsynced_s:
        return                      # audio-only dubs are uncapped
    cap = max_lipsync_s()
    if lipsynced_s <= cap:
        return
    est = presets.estimate_runtime(lipsynced_s, lipsync_model=model)
    raise _bad(
        f"That would lip-sync {lipsynced_s / 60:.0f} minutes of video, and "
        f"lip-sync is capped at {cap // 60} minutes here. This is a compute "
        f"limit, not a policy one: lip-sync runs at 3-21x the clip length on "
        f"this machine's two 16 GiB cards, so that render would take "
        f"{est['text']}. Either pick a shorter part, or choose the "
        f"audio-only preset — a whole-video dub of a longer source is fine, "
        f"the cap is only on the lip-synced portion."
    )


def check_tts(tts: Optional[str]) -> str:
    """Resolve and validate the engine, which the form no longer sends.

    An unknown id used to sail straight through: `languages_for` answers an
    unrecognised engine with the whole language table, so `check_language`
    passed, `build_kwargs` wrote it into `tts_model`, and the job was
    allocated, queued and launched before `synthesize_segments` raised
    `Unknown TTS model`. The operator saw a job card and then a dead job.
    Now it is a 400 before anything is written to disk.
    """
    from models.tts import available_models

    engine = (tts or presets.default_tts()).lower()
    if engine not in available_models():
        raise _bad(
            f"{engine!r} is not a TTS engine this install has. "
            f"Available: {', '.join(available_models())}."
            + (" Edge was removed on 2026-10-17."
               if engine == "edge" else "")
        )
    return engine


def check_style(style: Optional[str]) -> Optional[str]:
    """One of the four the form offers, `neutral`, or None for "work it out".

    Validated against `ui.speakers.SUBMIT_STYLES` rather than waved
    through: an unknown value is a **422 at the TTS engine**, mid-run,
    after ASR and MT have been paid for.

    The three non-values are not interchangeable, and this used to conflate
    two of them. An EMPTY style is the form's "work it out from the video",
    which leaves `presets.UI_STAGE_DEFAULTS`' `tts_style="auto"` in place and
    runs stage 9b1. `NEUTRAL_STYLE` is the opposite — it travels through as
    itself, 9b1 is not planned, and `models.tts.resolve_style` drops it at
    synthesis so the engine gets no style at all.
    """
    from models.tts import NEUTRAL_STYLE

    from .speakers import SUBMIT_STYLES

    v = (style or "").strip()
    if not v:
        return None
    if v == NEUTRAL_STYLE:
        return v
    if v not in SUBMIT_STYLES:
        raise _bad(f"{v!r} is not an offered delivery type. Choose one of: "
                   f"{NEUTRAL_STYLE}, {', '.join(SUBMIT_STYLES)}.")
    return v


def check_language(code: str, tts: Optional[str]) -> str:
    code = (code or "").strip().lower()
    available = langs.languages_for(tts)
    if code not in available:
        engine = tts or presets.default_tts()
        raise _bad(
            f"{langs.describe(code)} is not available with the {engine} voice "
            f"engine. It can speak: "
            f"{', '.join(langs.describe(c) for c in available[:12])}"
            + (" …" if len(available) > 12 else "")
        )
    return code


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

async def _probe_checked(url: str) -> Dict[str, Any]:
    """Allowlist, probe, re-check, refuse live. The whole front half."""
    check_url(url)
    try:
        meta = await run_in_threadpool(ytmeta.probe, url)
    except RuntimeError as e:
        raise _bad(str(e))
    # The extractor may have resolved somewhere else. Re-check, so the
    # allowlist is a control rather than a suggestion.
    check_url(str(meta.get("webpage_url") or url),
              what="That link resolves to a page that")
    if ytmeta.is_live(meta):
        raise _bad("That is a live or upcoming stream. There is nothing to "
                   "dub until it has finished and has a fixed duration.")
    return meta


@router.post("/api/youtube/probe")
async def youtube_probe(req: ProbeRequest) -> Dict[str, Any]:
    """What the form needs to render: title, duration, poster, estimates."""
    meta = await _probe_checked(req.url)
    dur = float(meta.get("duration_s") or 0.0)
    # The submission estimate is the DUB alone — no preset lip-syncs any
    # more, so there is no model to cost here and no "auto" to guess at.
    estimates = {}
    for pid, spec in presets.PRESETS.items():
        est = presets.estimate_runtime(dur, lipsync_model=spec["lipsync"])
        est.update(label=spec["label"], note=spec["note"])
        estimates[pid] = est
    meta["estimates"] = estimates

    # Lip-sync is quoted SEPARATELY and PER MODEL, because it is now a second
    # submission the operator makes after reviewing the dub. The models
    # differ by roughly 20x on this host (wav2lip 3-5x clip length,
    # musetalk 8-21x, latentsync ~83x) and one blended number would hide
    # exactly the difference the choice turns on.
    from models.lipsync import available_models
    meta["lipsync_estimates"] = {
        m: presets.estimate_runtime(dur, lipsync_model=m)
        for m in available_models()}
    meta["max_lipsync_s"] = max_lipsync_s()

    try:
        detected = await run_in_threadpool(detect_from_probe, meta)
        if detected:
            meta["detected_source_language"] = detected
    except Exception as e:
        log.warning("language detection failed during probe: %s", e)

    return meta


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

async def _resolve_source(req: JobRequest) -> Dict[str, Any]:
    """A URL or a staged upload, reduced to one metadata dict.

    Both branches answer with `duration_s`, `title` and `webpage_url`, so
    `create_job` below does not branch on where the video came from.
    """
    has_url = bool((req.url or "").strip())
    if has_url == bool(req.upload_id):
        raise _bad("Give either a link or an uploaded file — not both, and "
                   "not neither.")
    if has_url:
        return await _probe_checked(req.url)

    rec_path = DATA_DIR / "uploads" / f"{req.upload_id}.json"
    if not rec_path.is_file():
        raise _bad("That upload is no longer staged. Upload the file again.")
    rec = json.loads(rec_path.read_text())
    src = Path(rec["path"])
    if not src.is_file():
        raise _bad("The uploaded file is gone from the staging area. "
                   "Upload it again.")
    return {
        "duration_s": rec.get("duration_s"),
        "title": rec.get("title"),
        "webpage_url": None,
        "upload_id": req.upload_id,
        "source_path": str(src),
        "live_status": None,
    }


@router.post("/api/uploads", status_code=status.HTTP_201_CREATED)
async def upload_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accept a local video and answer with the same shape as the probe.

    The form treats an upload and a YouTube link identically from here on:
    both produce `{duration_s, title, estimates, max_lipsync_s}`, so the
    range picker and the quote need no second code path.

    `check_url` is NOT consulted. It is an SSRF control over hosts this
    server will *fetch from*, and an upload is not a fetch — widening it to
    admit a local path would turn an outbound guard into a local-file read.
    The upload's own constraints live in `ui/uploads.py`.

    The bytes land in a staging folder, not in a job folder: no job exists
    until `POST /api/jobs` validates language, window and cap. The returned
    `upload_id` is what ties the two calls together, and it is a uuid this
    server generated rather than anything the client sent.
    """
    from . import uploads

    try:
        ext = uploads.check_ext(file.filename or "")
    except ValueError as e:
        raise _bad(str(e))

    staging = DATA_DIR / "uploads"
    staging.mkdir(parents=True, exist_ok=True)
    upload_id = _uuid.uuid4().hex
    dest = staging / f"{upload_id}{ext}"

    cap = uploads.max_upload_mb() * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > cap:
                    # Stop at the cap rather than after it: a 4 GiB default
                    # is already generous and buffering past it is how a
                    # loopback UI fills a disk.
                    raise _TooBig()
                fh.write(chunk)
    except _TooBig:
        dest.unlink(missing_ok=True)
        raise _bad(f"That file is larger than the "
                   f"{uploads.max_upload_mb()} MB upload limit. Cut the "
                   f"section you want first, or raise "
                   f"DUBBING_UI_MAX_UPLOAD_MB.")
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if written == 0:
        dest.unlink(missing_ok=True)
        raise _bad("That upload was empty.")

    try:
        info = await run_in_threadpool(uploads.probe_file, dest)
    except Exception as e:
        # A file ffprobe cannot read is not a job worth allocating, and the
        # staged bytes are dropped rather than left to accumulate.
        dest.unlink(missing_ok=True)
        raise _bad(str(e))

    title = uploads.safe_stem(file.filename or "")
    dur = float(info["duration_s"])
    estimates = {}
    for pid, spec in presets.PRESETS.items():
        model = {"auto": "wav2lip"}.get(spec["lipsync"], spec["lipsync"])
        est = presets.estimate_runtime(dur, lipsync_model=model)
        est.update(label=spec["label"], note=spec["note"])
        estimates[pid] = est

    (staging / f"{upload_id}.json").write_text(json.dumps({
        "upload_id": upload_id, "path": str(dest),
        "original_filename": file.filename, "title": title,
        "created_at": _now(), **info,
    }, indent=2))
    log.info("upload %s: %s (%.1f MB, %.1fs)", upload_id, file.filename,
             written / 1e6, dur)

    result = {
        "upload_id": upload_id,
        "source": "upload",
        "title": title,
        "duration_s": dur,
        "width": info.get("width"), "height": info.get("height"),
        "has_audio": info.get("has_audio"),
        "size_bytes": info.get("size_bytes"),
        "estimates": estimates,
        "max_lipsync_s": max_lipsync_s(),
    }

    try:
        detected = await run_in_threadpool(
            detect_from_probe,
            {"title": title},
            video_path=dest,
        )
        if detected:
            result["detected_source_language"] = detected
    except Exception as e:
        log.warning("language detection failed during upload: %s", e)

    return result


class _TooBig(Exception):
    """Internal: the upload passed the size cap mid-stream."""


@router.post("/api/jobs", status_code=status.HTTP_201_CREATED)
async def create_job(req: JobRequest, response: Response) -> Dict[str, Any]:
    meta = await _resolve_source(req)

    duration_s = meta.get("duration_s")
    window = check_window(req.window, duration_s)
    clip_s = (window["end_s"] - window["start_s"]) if window else (duration_s or 0.0)

    preset = req.preset
    spec = presets.PRESETS[preset]
    # "off" is the form's way of saying "dub the audio, leave the picture
    # alone". It is NOT the same as omitting the field: omitting means "take
    # the preset's choice", which for `standard` IS lip-sync. The two were
    # conflated and the form's own "Off — dubbed audio only" option 400'd
    # with `Unknown lip-sync model 'off'`, because the server only ever knew
    # the four registry ids plus "auto". Normalise it to None here, where
    # `check_lipsync_cap` and `build_kwargs` both already read None as off.
    lipsync_off = (req.lipsync_model or "").lower() in ("off", "none")
    if lipsync_off:
        requested_ls = None
    else:
        requested_ls = (req.lipsync_model if req.lipsync_model is not None
                        else spec["lipsync"])
    if requested_ls and requested_ls not in ("auto",) + tuple(presets.LIPSYNC_FACTORS):
        raise _bad(f"Unknown lip-sync model {requested_ls!r}. Available: "
                   f"off, auto, {', '.join(sorted(presets.LIPSYNC_FACTORS))}.")

    # The cap bites on the model that will actually run. "auto" is costed at
    # musetalk, the expensive end, because that is what it may resolve to and
    # a cap that waves through the worst case is not a cap.
    cost_model = "musetalk" if requested_ls == "auto" else requested_ls
    check_lipsync_cap(clip_s, model=cost_model)

    tts = check_tts(req.tts)
    language = check_language(req.target_language, tts)
    style = check_style(req.style)

    # ---- everything validated; now allocate ----------------------------
    kwargs = presets.build_kwargs(
        preset, target_language=language, tts=tts,
        # "off" reaches `build_kwargs` as the literal, because None there
        # means "take the preset's choice" and `standard`'s choice is auto.
        lipsync_model=("off" if lipsync_off else requested_ls),
        num_speakers=req.num_speakers,
        skip_stems=req.skip_stems, style=style,
        source_language=(req.source_language or None),
    )
    cfg = presets.plan_config(kwargs)
    # The planned config must describe the run that will happen, including a
    # lip-sync stage whose model is not known until the face probe.
    if requested_ls:
        cfg["lipsync"] = cost_model

    # What the pipeline is pointed at: the page URL for a link, the staged
    # file's PATH for an upload. `backend.downloader.resolve_input` already
    # accepts either, so nothing below this line branches on the source.
    source_input = (meta.get("source_path")
                    or meta.get("webpage_url") or req.url)

    from backend.jobs import allocate, new_job_id
    job_id = new_job_id(source_input, cfg)
    job = await run_in_threadpool(
        allocate, DATA_DIR, source_input, cfg,
        job_id=job_id, argv=["ui", job_id],
    )
    job_root = Path(job.root)

    _write_meta(job_root, meta, window=window, preset=preset,
                lipsync_requested=requested_ls,
                is_upload=bool(req.upload_id))
    if not req.upload_id:
        await run_in_threadpool(ytmeta.fetch_thumbnail, meta, job_root)

    request_doc = {
        "version": 1,
        "job_id": job_id,
        "created_at": _now(),
        "url": source_input,
        "upload_id": req.upload_id,
        "target_language": language,
        "window": window,
        "preset": preset,
        "tts": tts,
        "lipsync_model": requested_ls,
        "num_speakers": req.num_speakers,
        "skip_stems": req.skip_stems,
        # Resolved here, not in the runner, so what was quoted to the
        # operator and what runs are the same dict.
        "pipeline": kwargs,
    }
    write_atomic(job_root / "request.json", request_doc)

    planned = plan_stages(cfg, pre=list(_pre_stages()))
    outcome = await run_in_threadpool(
        jobqueue.submit, job_id, job_root, JOBS_DIR, planned=planned,
    )

    body = {
        "job_id": job_id,
        "display_name": meta.get("title") or job_id,
        "state": outcome["state"],
        "window": window,
        "target_language": language,
        "preset": preset,
        "estimate": presets.estimate_runtime(
            clip_s, lipsync_model=cost_model),
    }
    if outcome["state"] == "queued":
        response.status_code = status.HTTP_202_ACCEPTED
        body["queued"] = True
        body["position"] = outcome["position"]
        body["note"] = (f"{jobqueue.occupancy(JOBS_DIR)} job(s) already using "
                        f"the GPUs; this one starts when they finish.")
    return body


def _pre_stages() -> List[str]:
    from ui.runner import PRE_STAGES
    return PRE_STAGES


def _write_meta(job_root: Path, meta: Dict[str, Any], *,
                window: Optional[Dict[str, float]], preset: str,
                lipsync_requested: Optional[str],
                is_upload: bool = False) -> None:
    """`<job>/meta.json` — written before the pipeline starts.

    The card needs a title while the download is still running, which is
    minutes. `ui/server.py::_job_meta` reads this file and falls back to the
    job-id slug when it is absent, so every field here is an improvement on a
    degraded display rather than a thing anything depends on.
    """
    doc = dict(meta)
    doc.pop("estimates", None)
    doc.update({
        # The name the card shows. Resolved here rather than in the UI so
        # that every reader — the job list, the spec panel, a later CLI
        # tool — agrees on what this job is called.
        # An upload has no page to name it, so its sanitised filename is
        # the title; `ytmeta.display_name` would fall back to a URL slug
        # that does not exist.
        "display_name": (meta.get("title") if is_upload else
                         ytmeta.display_name(
                             meta, url=str(meta.get("webpage_url") or ""))),
        "source": "upload" if is_upload else "youtube",
        "window": window,
        "preset": preset,
        "lipsync_requested": lipsync_requested,
        "created_at": _now(),
    })
    write_atomic(job_root / "meta.json", doc)


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------

@router.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    """Stop a running job or drop a queued one.

    The route is `{job_id}`, not `{job_id:path}`: the path converter is
    greedy and swallows `/cancel` into the id, so every sub-route under a job
    silently became the job route itself.
    """
    root = JOBS_DIR / job_id.strip("/")
    if not root.is_dir() or ".." in job_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job not found: {job_id}")
    result = await run_in_threadpool(jobqueue.cancel, root, JOBS_DIR)
    if result.get("state") in ("cancelled", "failed"):
        # The slot is free now; the finishing runner cannot say so because
        # there is no finishing runner.
        await run_in_threadpool(jobqueue.pump, JOBS_DIR)
    doc = read_progress(root) or {}
    return {"job_id": job_id, **result, "progress_state": doc.get("state")}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


__all__ = ["router", "check_url", "check_window", "check_lipsync_cap",
           "check_language", "allowed_hosts", "max_lipsync_s"]
