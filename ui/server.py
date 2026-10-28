"""Dubbing-pipeline inspector — FastAPI backend + static React UI.

Reads `data/jobs/<job_id>/` directly: each folder is one end-to-end run,
with `config.json`, `STATUS.json`, `VERIFY.json`, and stage subdirs
(videos/, audio/, faces/, asd/, lipsync/, mix/, …) all living inside.
Nothing is written outside a job folder — the UI reflects that.

Run:
    conda activate dubbing
    python ui/server.py                   # 127.0.0.1:7861
    python ui/server.py --host 0.0.0.0    # LAN
    python ui/server.py --port 7900

Open:  http://127.0.0.1:7861/
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import lipsync_models as _ls_models
from ._io import data_dir as _data_dir, env_int as _env_int

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# `DUBBING_UI_DATA_DIR` moves the whole UI. `_io.data_dir()` is the one place
# that resolves it, and the `DUBBING_DATA_DIR` alias, so every module agrees
# on where a job lives.
DATA_DIR = _data_dir()
JOBS_DIR = DATA_DIR / "jobs"

#: The built Vite app (`ui/web/` builds into it). The API still comes up if
#: the build is missing, so a long run stays reachable over /api.
WEB_DIST = Path(__file__).resolve().parent / "web-dist"
WEB_DIR  = WEB_DIST

LOG = logging.getLogger("ui")

#: Longest clip accepted for lip-sync. It runs at 3-21x clip length on this
#: host's 2x16 GiB cards, so 20 minutes is one to seven hours of render.
#: Anything longer is refused at submission rather than silently queued.
MAX_LIPSYNC_S = _env_int("DUBBING_UI_MAX_LIPSYNC_S", 20 * 60)

# ---------------------------------------------------------------------------
# Job folder shape. There is no single schema: clip-run jobs have
# STATUS.json + phases + stage subdirs, script outputs are flat with a few
# mp4s at the root. The folder is described empirically, not from a declared
# layout.
# ---------------------------------------------------------------------------

# `<timestamp>__<slug>__<config>` is the convention, parsed loosely so
# hand-named folders still work. Two stamp formats are live:
# `backend.jobs.new_job_id` writes "%Y%m%d-%H%M%S", the clip runner wrote
# "%Y%m%dT%H%M%S". Both must be handled or those jobs parse as nameless.
_JOB_ID_RE = re.compile(
    r"^(?P<ts>\d{8}[T-]\d{4,6}|PREFLIGHT)__(?P<slug>[^_]+(?:_[^_]+)*?)__(?P<cfg>.+)$"
)
_TS_FMT_RE = re.compile(
    r"^(\d{4})(\d{2})(\d{2})[T-](\d{2})(\d{2})(?:(\d{2}))?$")

# Classify any mp4 inside a job folder into a role, so the UI can line them
# up as original → lipsync → final without the user having to guess.
# Order matters — first match wins.
_VIDEO_ROLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("original",      re.compile(r"(^|/)videos/clip\.mp4$|\.video\.mp4$")),
    ("final",         re.compile(r"(^|/)final\..*\.mp4$|dubbed\..*\.mp4$|\.codeformer\.mp4$")),
    ("per_shot",      re.compile(r"(^|/)lipsync/shot\d+\.")),
    ("lipsync",       re.compile(r"\.lipsync\.mp4$|(^|/)lipsync/.*(musetalk|wav2lip|latentsync|spliced).*\.mp4$")),
    ("preview",       re.compile(r"(^|/)lipsync_preview/|(^|/)asd/.*\.mp4$|(^|/)faces/.*\.mp4$")),
)


def _role_for_video(rel_posix: str) -> str:
    for role, pat in _VIDEO_ROLE_PATTERNS:
        if pat.search(rel_posix):
            return role
    return "other"


def _parse_job_id(job_id: str) -> dict[str, Any]:
    m = _JOB_ID_RE.match(job_id)
    if not m:
        return {"job_id": job_id, "timestamp": None, "slug": job_id, "config_slug": None}
    ts_raw = m.group("ts")
    pretty_ts: str | None = None
    if ts_raw != "PREFLIGHT":
        tm = _TS_FMT_RE.match(ts_raw)
        if tm:
            yyyy, mo, dd, hh, mm, ss = tm.groups(default="00")
            pretty_ts = f"{yyyy}-{mo}-{dd} {hh}:{mm}:{ss}"
    return {
        "job_id":      job_id,
        "timestamp":   pretty_ts,
        "ts_raw":      ts_raw,
        "slug":        m.group("slug"),
        "config_slug": m.group("cfg"),
    }


from ._io import read_json as _read_json


# ---------------------------------------------------------------------------
# File-tree + categorization for one job
# ---------------------------------------------------------------------------

_MEDIA_EXTS = {
    ".mp4": "video", ".mkv": "video", ".webm": "video", ".mov": "video",
    ".wav": "audio", ".mp3": "audio", ".flac": "audio", ".m4a": "audio",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".webp": "image", ".svg": "image",
    ".json": "json", ".txt": "text", ".md": "text", ".vtt": "text",
    ".srt": "text", ".log": "text", ".rttm": "text",
    ".py":   "text",  # occasional preview helper scripts
    ".npz":  "binary",
}


def _data_url(rel_within_data: str) -> str:
    """Convert a path relative to `data/` into the browser-visible URL."""
    return f"/files/{Path(rel_within_data).as_posix()}"


def _walk_job(job_root: Path) -> list[dict[str, Any]]:
    """Flat list of every file inside a job folder, sorted by path."""
    out: list[dict[str, Any]] = []
    for p in sorted(job_root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(job_root)
        rel_posix = rel.as_posix()
        ext = p.suffix.lower()
        kind = _MEDIA_EXTS.get(ext, "other")
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        # URL under /files/ — the server serves anything under DATA_DIR.
        url = _data_url(f"jobs/{job_root.name}/{rel_posix}")
        entry = {
            "name":      p.name,
            "rel":       rel_posix,
            "url":       url,
            "size":      size,
            "ext":       ext,
            "kind":      kind,
            "stage":     rel.parts[0] if len(rel.parts) > 1 else "",
        }
        if kind == "video":
            entry["role"] = _role_for_video(rel_posix)
        out.append(entry)
    return out


def _dir_tree(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Nested {dirname -> {files:[], dirs:{...}}} tree, for the Files tab."""
    root: dict[str, Any] = {"files": [], "dirs": {}}
    for f in files:
        parts = Path(f["rel"]).parts
        node = root
        for part in parts[:-1]:
            node = node["dirs"].setdefault(part, {"files": [], "dirs": {}})
        node["files"].append(f)
    return root


def _describe_job(job_id: str) -> dict[str, Any]:
    job_root = JOBS_DIR / job_id
    if not job_root.is_dir():
        raise HTTPException(404, f"Job not found: {job_id}")

    files = _walk_job(job_root)

    # Pull out the structured status docs when they exist.
    status = _read_json(job_root / "STATUS.json")
    verify = _read_json(job_root / "VERIFY.json")
    cfg    = _read_json(job_root / "config.json")
    manifest = _read_json(job_root / "manifest.json")   # new v2 layout, rare

    # State: prefer STATUS.state, fall back to manifest.status, then infer.
    state = "unknown"
    if status and status.get("state"):
        state = status["state"]
    elif manifest and manifest.get("status"):
        state = manifest["status"]
    elif any(f["role"] == "final" for f in files if f["kind"] == "video"):
        state = "done"

    # Stage summary — top-level directory → file count + bytes.
    stages: dict[str, dict[str, int]] = {}
    for f in files:
        if not f["stage"]:
            continue
        s = stages.setdefault(f["stage"], {"count": 0, "bytes": 0})
        s["count"] += 1
        s["bytes"] += f["size"]
    stage_rows = [{"name": k, **v} for k, v in sorted(stages.items())]

    # Videos — ordered: original → preview → per_shot → lipsync → final → other.
    role_order = {"original": 0, "preview": 1, "per_shot": 2, "lipsync": 3,
                  "final": 4, "other": 5}
    videos = sorted(
        (f for f in files if f["kind"] == "video"),
        key=lambda f: (role_order.get(f["role"], 9), f["rel"]),
    )

    # MT JSON, if present — enables the segment-based Clips tab.
    mt_files = [f for f in files if f["name"].endswith(".mt.json")]
    segments = []
    tts_files_by_id: dict[str, str] = {}
    if mt_files:
        mt_payload = _read_json(job_root / mt_files[0]["rel"]) or {}
        # Index tts segment wavs by SEG_XXXX stem.
        for f in files:
            if f["ext"] == ".wav" and ("tts" in f["rel"] or "SEG_" in f["name"]):
                stem = Path(f["name"]).stem
                tts_files_by_id.setdefault(stem, f["url"])
        # Sort by start so the UI's `#N` matches `backend.edit_agent`'s cue
        # numbering, which also sorts by start.
        raw_segs = sorted(
            mt_payload.get("segments") or [],
            key=lambda s: float(s.get("start") or 0.0),
        )
        for idx, seg in enumerate(raw_segs, start=1):
            sid = seg.get("id") or f"SEG_{idx - 1:04d}"
            segments.append({
                "id":      sid,
                "cue_no":  idx,
                "speaker": seg.get("speaker"),
                "start":   float(seg.get("start") or 0.0),
                "end":     float(seg.get("end") or 0.0),
                "src_text":   seg.get("text") or "",
                "mt_text":    seg.get("mt_text") or "",
                "final_text": seg.get("final_text") or seg.get("mt_text") or "",
                "tts_url":    tts_files_by_id.get(sid),
            })

    parsed = _parse_job_id(job_id)

    return {
        **parsed,
        "root":        f"data/jobs/{job_id}",
        "state":       state,
        "total_bytes": sum(f["size"] for f in files),
        "n_files":     len(files),
        "config":      cfg,
        "status":      status,
        "verify":      verify,
        "manifest":    manifest,
        "stages":      stage_rows,
        "videos":      videos,
        "audios":      [f for f in files if f["kind"] == "audio"],
        "images":      [f for f in files if f["kind"] == "image"],
        "jsons":       [f for f in files if f["kind"] == "json"],
        "texts":       [f for f in files if f["kind"] == "text"],
        "tree":        _dir_tree(files),
        "files":       files,
        "segments":    segments,
        "has_mt":      bool(mt_files),
    }


def _epoch(value: Any) -> float | None:
    """Epoch seconds from an ISO-8601 string or a number; None if unreadable.

    A naive timestamp is read as UTC, not local: every writer here stamps
    with `datetime.now(timezone.utc)`, so assuming local would shift those
    rows by the machine's offset and interleave them wrongly.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _last_activity(job_dir: Path, parsed: dict[str, Any],
                   status: dict[str, Any] | None,
                   manifest: dict[str, Any] | None,
                   prog: dict[str, Any] | None) -> float:
    """When this job last RAN, in epoch seconds. Never None.

    `max()` over every stamp, not a first-match chain: a re-run (an edit
    replay, the lip-sync step) writes fresh progress into a folder whose name
    still carries the original allocation time.

    The folder mtime is a FALLBACK, never a candidate in the `max()` -- it is
    the most recent thing about any folder that has been written to at all,
    so including it turns the ordering into "whatever was touched last". It
    rescues only the job shapes with no recorded stamp.
    """
    recorded = [
        _epoch((prog or {}).get("updated")),
        _epoch((status or {}).get("updated")),
        _epoch((manifest or {}).get("finished_at")),
        _epoch((manifest or {}).get("started_at")),
        # `"2026-10-20 04:30:27"`, which `fromisoformat` accepts.
        _epoch(parsed.get("timestamp")),
    ]
    usable = [s for s in recorded if s]
    if usable:
        return max(usable)
    try:
        return job_dir.stat().st_mtime
    except OSError:                                      # pragma: no cover
        return 0.0


def _list_jobs() -> list[dict[str, Any]]:
    """Every job, most recently RUN first.

    Do not sort on the folder name: a hand-named job sorts by ASCII and
    pins itself above every timestamped one, and the name records when a job
    was ALLOCATED, which a re-run leaves alone. `JobRail.tsx` does not
    re-sort, so this is the only place the order can be set.
    """
    if not JOBS_DIR.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for d in JOBS_DIR.iterdir():
        if not d.is_dir():
            continue
        parsed = _parse_job_id(d.name)
        status = _read_json(d / "STATUS.json")
        cfg    = _read_json(d / "config.json")
        manifest = _read_json(d / "manifest.json")

        # Lightweight summary — avoid walking the full tree per row.
        files = list(d.iterdir())
        has_final_mp4 = any(f.name.startswith("final.") and f.suffix == ".mp4"
                            for f in files) \
                     or any(f.suffix == ".mp4" for f in files)

        # State comes from `read_progress`, the one normaliser over the four
        # job-folder shapes; nothing above it may branch on which runner
        # produced a job. In particular do not read `manifest.status`: it is
        # written at allocation and only updated on the pipeline's own error
        # path, so a job killed mid-stage reads "running" forever.
        prog = None
        try:
            from .progress import read_progress
            prog = read_progress(d)
        except Exception:                                # pragma: no cover
            prog = None
        state = ((prog or {}).get("state")
                 or (status or {}).get("state")
                 or (manifest or {}).get("status")
                 or ("done" if has_final_mp4 else "unknown"))

        meta = _read_json(d / "meta.json") or {}
        mcfg = (manifest or {}).get("config") or {}
        ran_at = _last_activity(d, parsed, status, manifest, prog)
        cutoff = datetime.now(timezone.utc).timestamp() - 7 * 86400
        if ran_at and ran_at < cutoff:
            continue
        rows.append({
            **parsed,
            # Epoch seconds, and the key this list is ordered by. Exposed so
            # a client can re-sort or render "3 minutes ago" without
            # re-deriving it from four possible stamps.
            "last_activity": ran_at,
            "last_activity_iso": (
                datetime.fromtimestamp(ran_at, timezone.utc).isoformat()
                if ran_at else None),
            # `display_name` is URL-derived at creation and upgraded once
            # the probe returns. `title` stays separate so a caller can tell
            # a real title from a placeholder.
            "title":   meta.get("title") or None,
            "display_name": (meta.get("display_name") or meta.get("title")
                             or _fallback_name(parsed, d)),
            "thumbnail_url": (f"/files/jobs/{d.name}/meta.thumb.jpg"
                              if (d / "meta.thumb.jpg").is_file() else None),
            "state":   state,
            "updated": ((status or {}).get("updated")
                        or (manifest or {}).get("finished_at")
                        or (manifest or {}).get("started_at")),
            "lipsync_model": ((cfg or {}).get("lipsync_model")
                              or mcfg.get("lipsync")),
            "language":      ((cfg or {}).get("language")
                              or mcfg.get("target_language")),
            "has_final":     has_final_mp4,
        })
    # Newest run first, job id as tie-break so the order is stable across
    # polls. Two jobs sharing a second would otherwise swap places between
    # refreshes and make the rail jump.
    rows.sort(key=lambda r: (r["last_activity"], r["job_id"]), reverse=True)
    return rows


#: Engine ids the registries know, plus retired ones. Used to tell an engine
#: name from a status sentence in the clip runner's `mt_tts` field, so it must
#: cover every value ever written to a job on disk -- `edge` (removed
#: 2026-10-17) and `indicf5` (removed 2026-10-28) stay for exactly that.
_KNOWN_TTS = ("edge", "bodhan", "indic-speak", "indicf5")


def _fallback_name(parsed: dict[str, Any], job_dir: Path) -> str:
    """A name for a job with no `meta.json` -- pre-UI jobs, and CLI ones.

    Never the raw job id: the slug is the video id or source filename, which
    at least identifies the material. `title` is None beside it, so a caller
    can tell this from a real title.
    """
    slug = parsed.get("slug") or ""
    if slug and slug != parsed.get("job_id"):
        return slug
    return job_dir.name


def _job_root_or_404(job_id: str) -> Path:
    root = JOBS_DIR / job_id.strip("/")
    if not root.is_dir():
        raise HTTPException(404, f"Job not found: {job_id}")
    return root


def _merged_json(job_root: Path) -> Path | None:
    """The merged ASR+MT JSON — the file stages 9b..9h read and write."""
    mt = job_root / "mt"
    cands = sorted(mt.glob("*.mt.json")) if mt.is_dir() else []
    if not cands:
        return None
    return max(cands, key=lambda q: q.stat().st_mtime)


def _job_config(job_root: Path) -> dict[str, Any]:
    """One config view over both runner shapes.

    The pipeline records its knobs in `manifest.json:config`, the clip runner
    in a flat `config.json` with different key names. Merged so every consumer
    above this line reads one dict.
    """
    manifest = _read_json(job_root / "manifest.json") or {}
    cfg = dict(manifest.get("config") or {})
    flat = _read_json(job_root / "config.json") or {}

    for src, dst in (("lipsync_model", "lipsync"), ("language", "target_language"),
                     ("asd_model", "asd")):
        if flat.get(src) is not None:
            cfg.setdefault(dst, flat[src])

    # `mt_tts` is free text in the clip runner, not an engine id -- on every
    # job on disk it reads "SKIPPED (...)". Take it only when it names a real
    # engine, or that sentence shows up in the UI as a voice name.
    mt_tts = str(flat.get("mt_tts") or "").strip().lower()
    if mt_tts in _KNOWN_TTS:
        cfg.setdefault("tts", mt_tts)

    # DUB_PROVENANCE.json exists to correct that field and is the authority
    # when present.
    prov = _read_json(job_root / "DUB_PROVENANCE.json") or {}
    if prov:
        chain = str(prov.get("pipeline") or "").lower()
        for engine in _KNOWN_TTS:
            if engine in chain:
                cfg["tts"] = engine
                break
        cfg.setdefault("dub_provenance", prov.get("pipeline"))

    for k, v in flat.items():
        cfg.setdefault(k, v)

    # `musetalk-v15` -> `musetalk`, so label lookups and the registry agree.
    # Through `lipsync_models.normalise`, which tests the LONGEST key first --
    # matching `wav2lip` before `wav2lip_hd` rewrites an HD run as plain
    # Wav2Lip for every reader downstream.
    ls = str(cfg.get("lipsync") or "").lower()
    known = _ls_models.normalise(ls)
    if known:
        cfg["lipsync"] = known
        cfg.setdefault("lipsync_variant", ls)
    return cfg


def _job_meta(job_root: Path) -> dict[str, Any]:
    meta = _read_json(job_root / "meta.json") or {}
    if not meta.get("title"):
        # No meta.json (pre-UI jobs). The job-id slug is the only name
        # available, and must read as the fallback it is.
        parsed = _parse_job_id(job_root.name)
        meta.setdefault("title", None)
        meta.setdefault("fallback_name", parsed.get("slug") or job_root.name)
    cfg = _job_config(job_root)
    win = cfg.get("window_abs_s")
    if win and not meta.get("window"):
        meta["window"] = win
    thumb = job_root / "meta.thumb.jpg"
    if thumb.is_file() and not meta.get("thumbnail_url"):
        meta["thumbnail_url"] = f"/files/jobs/{job_root.name}/meta.thumb.jpg"
    return meta


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Dubbing Studio")

# StaticFiles does not compress, so the first load would ship the whole JS
# bundle uncompressed. Registered before the mounts so it wraps them.
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    rows = _list_jobs()
    return {"jobs": rows, "count": len(rows)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    # NOT `{job_id:path}`: a path converter is greedy and swallows the
    # sub-routes like `/api/jobs/<id>/progress`. Job ids are directory names
    # and never contain a slash.
    return _describe_job(job_id.strip("/"))


def _safe_data_path(rel: str) -> Path:
    full = (DATA_DIR / rel).resolve()
    data_root = DATA_DIR.resolve()
    if not str(full).startswith(str(data_root)):
        raise HTTPException(400, "Path escapes data root")
    if not full.exists() or not full.is_file():
        raise HTTPException(404, f"Not found: {rel}")
    return full


@app.get("/files/{path:path}")
def serve_file(path: str, download: int = 0) -> Response:
    """Serve a job artifact. `?download=1` forces a save instead of inline.

    `ui/media.py` builds its `download_url` with that flag; without it the
    download button opens the file in a tab instead of saving it.
    """
    full = _safe_data_path(path)
    media = None
    ext = full.suffix.lower()
    if ext == ".vtt":
        media = "text/vtt"
    elif ext == ".wav":
        media = "audio/wav"
    elif ext == ".mp3":
        media = "audio/mpeg"
    elif ext == ".mp4":
        media = "video/mp4"
    elif ext == ".webm":
        media = "video/webm"
    elif ext == ".png":
        media = "image/png"
    elif ext in {".jpg", ".jpeg"}:
        media = "image/jpeg"
    elif ext == ".json":
        media = "application/json"
    elif ext in {".txt", ".md", ".log", ".rttm", ".py"}:
        media = "text/plain"
    headers = None
    if download:
        # The filename the user sees when saving; the UI may override it
        # through its own anchor.
        headers = {"Content-Disposition": f'attachment; filename="{full.name}"'}
    return FileResponse(full, media_type=media, headers=headers)


@app.get("/api/options")
def api_options(tts: str | None = None) -> dict[str, Any]:
    """Everything the new-job form needs to render its dropdowns.

    `tts_engines` has one row -- the form no longer offers a choice -- but
    stays a list, and `?tts=` is still honoured, because the language set
    differs per engine and an API caller overriding `tts` needs to ask which
    languages that buys. Omitting `?tts=` answers for `presets.default_tts()`
    rather than a hardcoded engine, so the form cannot disagree with what the
    server will run.
    """
    from . import presets
    from .langs import options as lang_options
    from . import presets
    from .speakers import submit_style_options
    engine = (tts or presets.default_tts()).lower()
    return {
        "languages": lang_options(engine),
        "tts_engines": [
            {"id": "bodhan", "label": "Bodhan Indic voices",
             "supports_tone": True,
             "note": "22 Indian languages · a distinct voice per speaker"},
        ],
        # Four delivery types, filtered from the engine's 14. Emotions are
        # excluded: stage 9b2 derives those per segment from the source
        # speaker, and picking one up front flattens the whole video.
        "styles": submit_style_options(),
        # Preselected source language. Served rather than baked into the
        # bundle so `DUBBING_UI_SOURCE_LANGUAGE` moves UI and pipeline
        # together.
        "default_source_language": presets.default_source_language(),
        # No `auto` row: nothing resolves it any more. The operator names
        # the model at `POST /api/jobs/{id}/lipsync`, against the face size
        # the run measured and shows.
        # Built from the REGISTRY, not hand-written, or a registered model
        # goes missing from the only place that enumerates them.
        "lipsync_models": _ls_models.options(),
        "limits": {
            "max_lipsync_s": MAX_LIPSYNC_S,
            "note": (f"Lip-sync is capped at {MAX_LIPSYNC_S // 60} minutes of "
                     f"video. It runs at 3–21× the clip length on this "
                     f"machine's GPUs, so a longer clip means hours. The cap "
                     f"applies when you START lip-sync, not when you submit "
                     f"the dub — a longer audio dub is fine."),
        },
    }


@app.get("/api/jobs/{job_id}/progress")
def api_progress(job_id: str) -> dict[str, Any]:
    from .progress import read_progress
    root = _job_root_or_404(job_id)
    doc = read_progress(root)
    if doc is None:
        raise HTTPException(404, f"No progress for {job_id}")
    return doc


@app.get("/api/jobs/{job_id}/media")
def api_media(job_id: str) -> dict[str, Any]:
    from .media import resolve
    root = _job_root_or_404(job_id)
    return resolve(root, data_dir=DATA_DIR, merged_json=_merged_json(root))


@app.get("/api/jobs/{job_id}/speakers")
def api_speakers(job_id: str) -> dict[str, Any]:
    from .speakers import build_table
    root = _job_root_or_404(job_id)
    table = build_table(root, _job_config(root), data_dir=DATA_DIR,
                        merged_json=_merged_json(root))
    table["job_id"] = root.name
    return table


@app.post("/api/jobs/{job_id}/speakers/{speaker_id}/label")
def api_set_speaker_label(job_id: str, speaker_id: str,
                          body: dict | None = Body(None)) -> dict[str, Any]:
    """Rename a speaker. Display only -- it dirties no pipeline stage.

    Kept in a job-local file rather than the merged JSON, so a rename never
    triggers a re-render.
    """
    from .speakers import write_label
    root = _job_root_or_404(job_id)
    labels = write_label(root, speaker_id, (body or {}).get("label"))
    return {"job_id": root.name, "labels": labels}


@app.get("/api/jobs/{job_id}/spec")
def api_spec(job_id: str) -> dict[str, Any]:
    from .progress import read_progress
    from .spec import build
    from .speakers import build_table
    root = _job_root_or_404(job_id)
    manifest = _read_json(root / "manifest.json") or {}
    manifest.setdefault("config", _job_config(root))
    n_spk = len(build_table(root, _job_config(root), data_dir=DATA_DIR,
                            merged_json=_merged_json(root))["speakers"])
    return build(root, manifest=manifest, meta=_job_meta(root),
                 progress=read_progress(root), n_speakers=n_spk)


@app.get("/api/jobs/{job_id}/log")
def api_log(job_id: str, tail: int = 200) -> dict[str, Any]:
    """Raw runner output.

    Lip-sync shells into other conda envs, so this is the only place the
    detail of a slow or wrong render exists.
    """
    root = _job_root_or_404(job_id)
    for name in ("run.log", "job.log"):
        p = root / name
        if p.is_file():
            lines = p.read_text(errors="replace").splitlines()
            return {"file": name, "total_lines": len(lines),
                    "lines": lines[-max(1, min(tail, 5000)):]}
    return {"file": None, "total_lines": 0, "lines": []}


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
# MUST be included ABOVE the static mount: Starlette matches in registration
# order, so a router added after `app.mount("/", ...)` is shadowed entirely --
# every GET 404s and every POST 405s from the static handler.
for _mod, _what in (("routes_edits", "voice/tone editing"),
                    ("routes_create", "job creation"),
                    ("routes_talkinghead", "talking head page"),
                    ("routes_enhance", "resolution enhancement")):
    try:
        _router = __import__(f"ui.{_mod}", fromlist=["router"]).router
    except ImportError as _e:
        # A missing router must not take the server down: the read-only
        # inspector is how a running GPU job gets checked.
        LOG.warning("ui.%s unavailable (%s); %s disabled", _mod, _e, _what)
    else:
        app.include_router(_router)
        LOG.info("mounted ui.%s (%s)", _mod, _what)


# Static UI — mounted last so /api/* and /files/* win first.
if (WEB_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")
else:
    LOG.warning("%s/index.html missing — API only. Build the UI: "
                "conda run -n dubui npm --prefix ui/web run build", WEB_DIR)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    # The adapters read credentials from the environment and nothing else
    # puts them there. Needed for in-process work (yt-dlp probes, edit
    # plans); the detached runner loads `.env` again on its own.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:                                  # pragma: no cover
        LOG.warning("python-dotenv is not installed; .env was not loaded")

    # `scripts/ui_run_job.sh` resolves its interpreter from `DUBBING_PY`,
    # falling back to a sibling conda env. Defaulting it to our own
    # interpreter runs a detached job under the env that launched this
    # server; `setdefault` keeps an operator value from `.env` winning.
    # `Popen` jobs inherit it via `os.environ`; the tmux path does not
    # inherit this process's environment and passes it explicitly.
    os.environ.setdefault("DUBBING_PY", sys.executable)

    LOG.info("Dubbing Studio listening on http://%s:%d  (ui=%s)",
             args.host, args.port, WEB_DIR.name)
    uvicorn.run("ui.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
