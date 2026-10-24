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

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import lipsync_models as _ls_models
from .auth import basic_auth

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# `DUBBING_UI_DATA_DIR` moves the whole UI, not part of it. `routes_create`
# and `routes_edits` already read it; this module did not, so with the var
# set the write path allocated jobs into one tree while the read path
# listed another — a job that creates successfully and never appears.
DATA_DIR = Path(os.environ.get("DUBBING_UI_DATA_DIR") or (REPO_ROOT / "data"))
JOBS_DIR = DATA_DIR / "jobs"

#: The built Vite app, and the hand-written inspector that predates it.
#: `WEB_DIR` prefers the build but falls back, so a broken or absent build
#: never takes the UI down — which matters on this host, where the UI is how
#: you check a four-hour GPU run.
WEB_DIST   = Path(__file__).resolve().parent / "web-dist"
WEB_LEGACY = Path(__file__).resolve().parent / "web"
WEB_DIR    = WEB_DIST if (WEB_DIST / "index.html").is_file() else WEB_LEGACY

LOG = logging.getLogger("ui")

#: Lip-sync runs at 3–21x clip length on this host's 2x16 GiB cards, so a
#: 20-minute render is between one and seven hours. Longer than that is
#: refused rather than silently queued — stated, with the reason, at the
#: point of submission.
MAX_LIPSYNC_S = int(os.environ.get("DUBBING_UI_MAX_LIPSYNC_S", 20 * 60))

# ---------------------------------------------------------------------------
# Job folder shape — the key realization is that there is no single schema.
# Some jobs come from the clip-run pipeline (STATUS.json + phases + stage
# subdirs); others are flat script outputs with a few mp4s at the root. We
# describe the folder empirically instead of trusting a declared layout.
# ---------------------------------------------------------------------------

# `<timestamp>__<slug>__<config>` is the convention; parse it loosely so we
# still cope with the hand-named `PREFLIGHT__…` folders.
# Two stamp formats are live: `backend.jobs.new_job_id` writes
# "%Y%m%d-%H%M%S" and the clip runner wrote "%Y%m%dT%H%M%S". Knowing only
# the second made every UI-created job fail to parse, so its card showed the
# raw id instead of a name and carried no timestamp at all.
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
        for seg in mt_payload.get("segments") or []:
            sid = seg.get("id") or f"SEG_{len(segments):04d}"
            segments.append({
                "id": sid,
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

    A NAIVE timestamp is read as UTC rather than local time. Every writer
    here stamps with `datetime.now(timezone.utc)`, so assuming local would
    shift those rows by the machine's offset and interleave them wrongly
    with the tz-aware ones — a five-and-a-half-hour reordering on this host.
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

    `max()` over every stamp rather than a first-match chain, because a job
    can be re-run: the edit agent's replay and the lip-sync step both write
    fresh progress into a folder whose NAME still carries the original
    allocation time. Taking the newest evidence is what makes "recent" mean
    recently executed rather than recently created.

    The folder's mtime is a FALLBACK, not a candidate. Putting it in the
    `max()` made it win every time — it is the most recent thing about any
    folder that has been written to at all — so the recorded stamps became
    decorative and the ordering silently became "whatever was touched last",
    which is not the same question. It now rescues only the job shapes with
    no recorded stamp whatsoever: the clip runner's and the bare
    `manifest.json` ones (CLAUDE.md §2 counts four shapes plus 16 folders
    with none).
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

    The order used to be `sorted(JOBS_DIR.iterdir(), reverse=True)` — the
    folder NAME, descending — which is right only for jobs whose name starts
    with an allocation timestamp. It was wrong twice over:

    1. A hand-named job sorts by ASCII, and lowercase letters sort ABOVE
       digits, so `w1l2_60to100_hi_musetalk_cli` and `bench_hi_120s` pinned
       themselves above every timestamped job on the machine no matter how
       old they were.
    2. The name records when a job was ALLOCATED. A re-run — an edit agent
       replay, or the lip-sync step — leaves the name alone, so a job that
       ran five minutes ago sorted as though it ran last week.

    `JobRail.tsx` does not re-sort; it relies on this order, so this is the
    only place it can be fixed.
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

        # State comes from `read_progress`, the ONE normaliser over the four
        # job-folder shapes (CLAUDE.md §2: nothing above it may branch on
        # which runner produced a job). This used to hand-roll the chain
        # `STATUS.json -> manifest.status -> "an mp4 exists"`, which never
        # looked at `progress.json` at all — the shape every UI-submitted
        # job uses. The card therefore showed whatever `manifest.status`
        # held, and that is written at allocation and only updated when the
        # pipeline exits through its own error path: a job killed mid-stage
        # (SIGTERM during TTS, say) leaves `"running"` there forever, so a
        # dead job advertised itself as still working, with a progress bar
        # frozen part-way. Measured on this host: 2 of 9 UI jobs.
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
        rows.append({
            **parsed,
            # Epoch seconds, and the key this list is ordered by. Exposed so
            # a client can re-sort or show "3 minutes ago" without having to
            # re-derive it from four possible stamps.
            "last_activity": ran_at,
            "last_activity_iso": (
                datetime.fromtimestamp(ran_at, timezone.utc).isoformat()
                if ran_at else None),
            # `display_name` is written at creation (URL-derived) and
            # upgraded to the real title once the probe returns. `title` is
            # kept separate so a caller can tell a real title from a
            # placeholder.
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
    # Newest run first. The job id is the tie-break so the order is STABLE
    # across polls — two jobs sharing a second (a batch submission, or two
    # folders whose only stamp is the same mtime) would otherwise swap
    # places between refreshes and make the rail jump under the cursor.
    rows.sort(key=lambda r: (r["last_activity"], r["job_id"]), reverse=True)
    return rows


#: Engine ids the registries know, plus the ones they used to. Used to tell
#: an engine name apart from a status sentence in the clip runner's `mt_tts`
#: field, so it has to cover every value ever written to a job on disk --
#: `edge` included, though nothing can be run with it any more.
_KNOWN_TTS = ("edge", "bodhan", "indic-speak", "indicf5")


def _fallback_name(parsed: dict[str, Any], job_dir: Path) -> str:
    """A name for a job that has no `meta.json` — every job made before the
    UI existed, and any made by the CLI.

    Still never the raw job id: the slug is the video id or the source
    filename, which at least identifies the material. The caller can tell
    this apart from a real title because `title` is None beside it.
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

    The pipeline records its knobs in `manifest.json:config`; the clip runner
    writes a flat `config.json` with different key names. Merge them so every
    consumer above this line reads one dict.
    """
    manifest = _read_json(job_root / "manifest.json") or {}
    cfg = dict(manifest.get("config") or {})
    flat = _read_json(job_root / "config.json") or {}

    for src, dst in (("lipsync_model", "lipsync"), ("language", "target_language"),
                     ("asd_model", "asd")):
        if flat.get(src) is not None:
            cfg.setdefault(dst, flat[src])

    # `mt_tts` is NOT an engine id in the clip runner — it is free text, and
    # on every job on disk it reads "SKIPPED (downloaded YouTube dub drives
    # audio)". Mapping it straight onto `tts` put that sentence in the UI as
    # the name of a voice engine. Take it only when it names a real engine.
    mt_tts = str(flat.get("mt_tts") or "").strip().lower()
    if mt_tts in _KNOWN_TTS:
        cfg.setdefault("tts", mt_tts)

    # DUB_PROVENANCE.json exists specifically to correct that field — its own
    # first key is `"corrects": "config.json field mt_tts, which is hardcoded
    # to 'SKIPPED'"`. It is the authority when present.
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
    #
    # Through `lipsync_models.normalise`, which tests the LONGEST key first.
    # The loop this replaces tried `wav2lip` before `wav2lip_hd`, so an HD run
    # had `cfg["lipsync"]` rewritten to `"wav2lip"` right here — and every
    # reader downstream (the job list, `spec.py`, `cost.py`) then described a
    # plain Wav2Lip run. The truth survived in `lipsync_variant`, which
    # nothing read.
    ls = str(cfg.get("lipsync") or "").lower()
    known = _ls_models.normalise(ls)
    if known:
        cfg["lipsync"] = known
        cfg.setdefault("lipsync_variant", ls)
    return cfg


def _job_meta(job_root: Path) -> dict[str, Any]:
    meta = _read_json(job_root / "meta.json") or {}
    if not meta.get("title"):
        # No meta.json (every job that predates the UI). The job-id slug is
        # the only name available, and it must look like the fallback it is.
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

# StaticFiles does not compress. Without this the first load ships the whole
# JS bundle uncompressed — the single cheapest win available on first render.
# Registered before the mounts so it wraps them.
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    rows = _list_jobs()
    return {"jobs": rows, "count": len(rows)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    # NOT `{job_id:path}`. A path converter is greedy, so it swallowed
    # `/api/jobs/<id>/progress` and answered "job not found: <id>/progress".
    # Job ids are directory names and never contain a slash, so the plain
    # converter is both correct and unambiguous against the sub-routes.
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

    `ui/media.py` hands the UI a `download_url` with that flag, and without
    it the "download the dubbed audio" button just plays the file in a new
    tab — which looks like the button is broken.
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
        # The filename the user sees when saving. `full.name` is a pipeline
        # stem like `clip.dubbed.hi.wav`, which is more useful than nothing
        # but worse than a title; the UI may override via its own anchor.
        headers = {"Content-Disposition": f'attachment; filename="{full.name}"'}
    return FileResponse(full, media_type=media, headers=headers)


@app.get("/api/options")
def api_options(tts: str | None = None) -> dict[str, Any]:
    """Everything the new-job form needs to render its dropdowns.

    `tts_engines` has exactly one row: the form no longer lets an operator
    pick an engine, so the list is what this install WILL use rather than a
    menu. It is still a list, and `?tts=` is still honoured, because the
    language set genuinely differs per engine and an API caller overriding
    `tts` needs to be able to ask which languages that buys them.

    Omitting `?tts=` answers for `presets.default_tts()` rather than for a
    hardcoded engine, so the form's language list cannot disagree with the
    engine the server will actually run.
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
        # The four delivery types the form offers, filtered from the
        # engine's own 14 — emotions are excluded because they are
        # per-segment moods stage 9b2 derives from the source speaker, and
        # picking one up front flattens the whole video into it.
        "styles": submit_style_options(),
        # What the form should PRESELECT as the spoken language. Served
        # rather than hardcoded in the bundle so `DUBBING_UI_SOURCE_LANGUAGE`
        # moves the UI and the pipeline together instead of only one.
        "default_source_language": presets.default_source_language(),
        # No `auto` row. It meant "picked from the face size in the video",
        # and that selection was removed on 2026-10-20: the operator names
        # the model at `POST /api/jobs/{id}/lipsync`, against the face size
        # the run measured and SHOWS. Leaving the row here would advertise a
        # value nothing resolves any more — the shape of the tone-picker bug
        # in CLAUDE.md §2, where the UI offered values the engine 422s on.
        # Built from the REGISTRY, not hand-written. This list carried three
        # rows while `models/lipsync/__init__.py` registered five, so
        # `wav2lip_hd` and `lipgan` were absent from the only place that
        # claims to enumerate the models — the same shape as the tone picker
        # in CLAUDE.md §2, inverted: a real option nothing advertised.
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


@app.post("/api/jobs/{job_id}/speakers/{speaker_id}/label",
          dependencies=[Depends(basic_auth)])
def api_set_speaker_label(job_id: str, speaker_id: str,
                          body: dict | None = Body(None)) -> dict[str, Any]:
    """Rename a speaker. Display only — it dirties no pipeline stage.

    `SPEAKER_01` is what diarization calls someone; "Prof. Mitra" is what
    the person watching calls them. Kept in a job-local file rather than the
    merged JSON precisely so that renaming never triggers a re-render.
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
    """Raw runner output. Lip-sync shells into other conda envs, so when a
    render is slow or wrong this is the only place the detail exists."""
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
# MUST be included ABOVE the static mount. Starlette matches routes in
# registration order, so a router added after `app.mount("/", ...)` is
# shadowed entirely: every GET answers 404 and every POST 405, from the
# static handler, which reads exactly like a routing typo.
for _mod, _what in (("routes_edits", "voice/tone editing"),
                    ("routes_create", "job creation"),
                    ("routes_talkinghead", "talking head page"),
                    ("routes_enhance", "resolution enhancement")):
    try:
        _router = __import__(f"ui.{_mod}", fromlist=["router"]).router
    except ImportError as _e:
        # A router that is not built yet must not take the whole server down
        # — the read-only inspector is how you check a running GPU job.
        LOG.warning("ui.%s unavailable (%s); %s disabled", _mod, _e, _what)
    else:
        app.include_router(_router)
        LOG.info("mounted ui.%s (%s)", _mod, _what)


# Static UI — mounted last so /api/* and /files/* win first.
# The old inspector stays reachable at /legacy during the transition; its
# Files/Images/Clips tabs are the fallback until they are ported.
if WEB_DIR is not WEB_LEGACY and WEB_LEGACY.is_dir():
    app.mount("/legacy", StaticFiles(directory=str(WEB_LEGACY), html=True),
              name="legacy")
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    # Load `.env` here for the same reason every other entry point does: the
    # adapters read their credentials from the environment and nothing else
    # puts them there. The server itself needs them for the in-process work
    # (yt-dlp probes, building an edit plan); the detached runner loads them
    # again on its own, because it is a separate process.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:                                  # pragma: no cover
        LOG.warning("python-dotenv is not installed; .env was not loaded")

    # Refuse a public bind without credentials, before uvicorn opens a socket.
    from .auth import require_auth_for_nonloopback
    require_auth_for_nonloopback(args.host)

    LOG.info("Dubbing Studio listening on http://%s:%d  (ui=%s)",
             args.host, args.port, WEB_DIR.name)
    uvicorn.run("ui.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
