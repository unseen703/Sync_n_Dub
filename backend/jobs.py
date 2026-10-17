"""Job-folder layout for the pipeline.

Every end-to-end run owns a self-contained tree under `data/jobs/<job_id>/`:

    data/jobs/<job_id>/
      manifest.json          <- status + config + stage inventory
      videos/ audio/ stems/ diarization/ faces/ asd/ asr/ align/ mt/
      eval/ voices/ tts/ vc/ mix/ lipsync/ exp/

Nothing a run produces escapes its job folder, so a job can be inspected,
moved, tarred or deleted on its own without disturbing any other run.

History, because it explains the version number: layout v1 was a manifest that
only *referenced* artifacts living in shared `data/<stage>/` folders. Those
folders were archived on 2026-10-11 and the v1 half of this module -- filename
parsing (`parse_mt_filename`), id synthesis (`job_id_for`), artifact collection
and `backfill` -- was removed on 2026-10-17, having had no caller outside its
own `main()` since the archive. `JOB_LAYOUT_VERSION` stays at 2 so a v1
manifest found in an old tarball is still identifiable.

Reading a run:

    $PY -m backend.jobs list              # every run, newest first
    $PY -m backend.jobs show <job_id>     # one run's manifest
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

JOB_DIR = "jobs"


def _fmt_bytes(n: Optional[int]) -> str:
    if not n:
        return "-"
    v = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if v < 1024 or unit == "T":
            return f"{v:.0f}{unit}" if unit == "B" else f"{v:.1f}{unit}"
        v /= 1024
    return f"{v:.1f}T"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Pipeline job folders")
    ap.add_argument("cmd", choices=("list", "show"),
                    help="list: every run; show: one run's manifest")
    ap.add_argument("job_id", nargs="?", help="required by `show`")
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--json", action="store_true", help="raw JSON output")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")

    if args.cmd == "list":
        rows = list_jobs(args.data_dir)
        if args.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
            return
        if not rows:
            print(f"no jobs under {args.data_dir / JOB_DIR}")
            return
        print(f"{'STATUS':<9} {'SIZE':>7}  JOB_ID")
        for r in rows:
            print(f"{r['status']:<9} {_fmt_bytes(r['total_bytes']):>7}  {r['job_id']}")
        print(f"\n{len(rows)} job(s) in {args.data_dir / JOB_DIR}")
        return

    if args.cmd == "show":
        if not args.job_id:
            raise SystemExit("`show` needs a job id (see `list`)")
        job = load(args.data_dir, args.job_id)
        m = job.read()
        if args.json:
            print(json.dumps(m, indent=2, ensure_ascii=False))
            return
        print(f"job      {m.get('job_id')}")
        print(f"status   {m.get('status')}")
        print(f"input    {m.get('input')}")
        print(f"started  {m.get('started_at')}")
        print(f"finished {m.get('finished_at')}")
        print(f"path     {job.root}")
        if m.get("error"):
            print(f"error    {m['error'].splitlines()[0]}")
        cfg = {k: v for k, v in (m.get("config") or {}).items() if v not in (None, False)}
        print("config   " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
        print("stages:")
        for name, st in (m.get("stages") or {}).items():
            if st.get("count"):
                print(f"  {name:<14} {st['count']:>4} file(s)  {_fmt_bytes(st['bytes'])}")
        print(f"total    {_fmt_bytes(m.get('total_bytes'))}")
        return

    raise SystemExit(f"unknown command {args.cmd!r}")


# ===========================================================================
# job folders
# ===========================================================================

JOB_LAYOUT_VERSION = 2

#: Per-stage subdirectories created inside every job folder. These mirror the
#: old `data/<stage>/` names so stage code needs no path rewriting — only the
#: root it is handed changes.
STAGE_DIRS: tuple[str, ...] = (
    "videos", "audio", "stems", "diarization", "faces", "asd",
    "asr", "align", "mt", "eval", "voices", "tts", "vc", "mix",
    "lipsync", "exp",
)

_SLUG_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_YT_ID = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/))"
    r"([A-Za-z0-9_-]{11})"
)


def slug_for_input(input_: str, *, max_len: int = 48) -> str:
    """A short, filesystem-safe token identifying the run's input.

    A YouTube URL yields its 11-character video id, any other URL a short
    hash of the URL, and a local path its filename stem. The result only
    labels the job folder — the authoritative input string is recorded in
    the manifest.
    """
    text = str(input_).strip()
    m = _YT_ID.search(text)
    if m:
        return m.group(1)
    if text.startswith(("http://", "https://")):
        tail = _SLUG_SAFE.sub("-", text.rstrip("/").rsplit("/", 1)[-1])[:24].strip("-")
        digest = hashlib.sha1(text.encode()).hexdigest()[:8]
        return f"{tail}-{digest}" if tail else f"url-{digest}"
    stem = Path(text).name
    for suffix in (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".wav", ".mp3"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    slug = _SLUG_SAFE.sub("-", stem).strip("-.")
    return (slug[:max_len] or "input")


def _tok(x: Optional[str]) -> str:
    """A stage choice as it appears in a job id; an unrun stage reads "none"."""
    return x if x else "none"


def config_slug(config: dict[str, Any]) -> str:
    """`<diar>.<asr>.<align>.<mt>.<lang>` — the stage choices in the job id."""
    return (
        f"{_tok(config.get('diarization'))}."
        f"{_tok(config.get('asr'))}."
        f"{_tok(config.get('align'))}."
        f"{_tok(config.get('mt'))}."
        f"{_tok(config.get('target_language'))}"
    )


def new_job_id(input_: str, config: dict[str, Any], *,
               now: Optional[_dt.datetime] = None) -> str:
    """`<timestamp>__<input-slug>__<config-slug>`.

    The timestamp leads so that a plain `ls` sorts jobs chronologically, and
    makes the id unique: re-running the same video with the same settings
    produces a new job rather than overwriting the previous one.
    """
    ts = (now or _dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{ts}__{slug_for_input(input_)}__{config_slug(config)}"


class Job:
    """One end-to-end run, rooted at `data/jobs/<job_id>/`.

    Created by `allocate()` before stage 1 runs, so the manifest exists (with
    `status: "running"`) even if the run later crashes. `finalize()` stamps
    the terminal status and the inventory of what landed on disk.
    """

    def __init__(self, root: Path, job_id: str, *, data_dir: Path):
        self.root = Path(root)
        self.job_id = job_id
        self.data_dir = Path(data_dir)

    # -- paths -------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def stage(self, name: str) -> Path:
        """Return (and create) the job-local directory for `name`."""
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def rel(self, p: Path | str | None) -> Optional[str]:
        """Path relative to the job root, for recording in the manifest."""
        if p is None:
            return None
        try:
            return str(Path(p).resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(p)

    # -- manifest ----------------------------------------------------------
    def read(self) -> dict[str, Any]:
        try:
            return json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, manifest: dict[str, Any]) -> Path:
        # Write-then-rename so a reader never sees a half-written manifest.
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        os.replace(tmp, self.manifest_path)
        return self.manifest_path

    def start(self, *, input_: str, config: dict[str, Any],
              argv: Optional[list[str]] = None) -> Path:
        return self._write({
            "version": JOB_LAYOUT_VERSION,
            "layout": "self-contained",
            "job_id": self.job_id,
            "status": "running",
            "input": str(input_),
            "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "finished_at": None,
            "argv": argv,
            "config": config,
            "stages": {},
        })

    def inventory(self) -> dict[str, dict[str, Any]]:
        """What actually landed in each stage folder.

        Built by walking the job tree rather than by guessing filenames —
        possible only because the run is self-contained, and it cannot go
        stale the way the v1 glob patterns did.
        """
        out: dict[str, dict[str, Any]] = {}
        for name in STAGE_DIRS:
            d = self.root / name
            if not d.exists():
                out[name] = {"status": "missing", "files": [], "count": 0, "bytes": 0}
                continue
            files = sorted(p for p in d.rglob("*") if p.is_file())
            out[name] = {
                "status": "ok" if files else "empty",
                "files": [self.rel(p) for p in files],
                "count": len(files),
                "bytes": sum(p.stat().st_size for p in files),
            }
        return out

    def finalize(self, result: Any = None, *, status: str = "ok",
                 error: Optional[str] = None) -> Path:
        manifest = self.read() or {
            "version": JOB_LAYOUT_VERSION,
            "layout": "self-contained",
            "job_id": self.job_id,
        }
        manifest["status"] = status
        manifest["finished_at"] = _dt.datetime.now(
            _dt.timezone.utc).isoformat(timespec="seconds")
        if error:
            manifest["error"] = error
        manifest["stages"] = self.inventory()
        manifest["total_bytes"] = sum(s["bytes"] for s in manifest["stages"].values())
        if result is not None:
            manifest["result"] = _result_summary(result, self)
        return self._write(manifest)


#: `PipelineResult` fields worth lifting into the manifest so that `jobs show`
#: answers "what did this run produce?" without loading the whole result.
_RESULT_FIELDS = (
    "num_turns", "num_speakers", "diarization_model", "stems_skipped",
    "num_face_tracks", "face_detector", "face_tracker", "asd_model",
    "num_on_screen", "asr_model", "asr_language", "align_model",
    "num_aligned_words", "mt_model", "mt_target_language", "mt_target_script",
    "translit_model", "codemix_retained_terms", "duration_model_language",
    "duration_model_voice", "duration_estimated_violations", "voice_engine",
    "tts_model", "tts_budget_violations", "tts_errors", "vc_model",
    "lipsync_model", "lipsync_error",
)

#: Result fields that are paths; recorded relative to the job root.
_RESULT_PATHS = (
    "video", "audio_16k", "dialogue_stem", "me_stem", "rttm", "manifest",
    "face_tracks_npz", "assignments_json", "asr_json", "align_json",
    "mt_json", "mt_txt", "voice_assignments_json", "speaker_references_dir",
    "tts_out_dir", "vc_out_dir", "mix_wav", "mix_mp4", "lipsync_mp4",
)


def _result_summary(result: Any, job: "Job") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in _RESULT_FIELDS:
        v = getattr(result, f, None)
        if v is not None:
            out[f] = v
    for f in _RESULT_PATHS:
        v = getattr(result, f, None)
        if v is not None:
            out[f] = job.rel(v)
    return out


#: The job allocated by the run in progress. The pipeline handles exactly one
#: input at a time (the CLI enforces it), so "the current job" is unambiguous
#: and lets the crash handler close out a manifest it never received a handle
#: to.
_CURRENT: Optional["Job"] = None


def current() -> Optional["Job"]:
    """The most recently allocated job, or None before any run started."""
    return _CURRENT


def allocate(data_dir: Path, input_: str, config: dict[str, Any], *,
             job_id: Optional[str] = None,
             argv: Optional[list[str]] = None) -> Job:
    """Create `data/jobs/<job_id>/` with its stage folders and manifest.

    Called once at the top of a run, before any stage executes.
    """
    data_dir = Path(data_dir)
    job_id = job_id or new_job_id(input_, config)
    root = data_dir / JOB_DIR / job_id
    root.mkdir(parents=True, exist_ok=True)
    job = Job(root, job_id, data_dir=data_dir)
    for name in STAGE_DIRS:
        job.stage(name)
    job.start(input_=input_, config=config, argv=argv)
    global _CURRENT
    _CURRENT = job
    log.info("job %s → %s", job_id, root)
    return job


def load(data_dir: Path, job_id: str) -> Job:
    """Reopen an existing job folder."""
    root = Path(data_dir) / JOB_DIR / job_id
    if not root.is_dir():
        raise FileNotFoundError(f"no such job: {root}")
    return Job(root, job_id, data_dir=Path(data_dir))


def list_jobs(data_dir: Path) -> list[dict[str, Any]]:
    """Every job folder, newest first, as manifest summaries."""
    root = Path(data_dir) / JOB_DIR
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        job = Job(d, d.name, data_dir=Path(data_dir))
        m = job.read()
        out.append({
            "job_id": d.name,
            "status": m.get("status", "unknown"),
            "input": m.get("input"),
            "started_at": m.get("started_at"),
            "config": m.get("config", {}),
            "total_bytes": m.get("total_bytes"),
            "path": str(d),
        })
    return out


if __name__ == "__main__":
    main()
