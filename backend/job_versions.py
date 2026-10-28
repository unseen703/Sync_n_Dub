"""One index that says which files belong to which version of a job.

`mt_versions` and `mix_versions` already keep every transcript and every mix
on disk, each numbered on its own. Neither says which edit produced a file,
and nothing ties a transcript to the mix rendered from it. This module is that
map, and only that: it LINKS to the snapshots those two write (relative path
plus sha256) and copies nothing.

    <job>/versions/index.json

    {"schema": 1, "current": 7, "versions": [
        {"v": 7, "parent": 6, "kind": "edit", "edit_id": "edit_003",
         "nlq": "...", "ops": [...], "dirty_stages": ["tts", "mix"],
         "status": "rendered",
         "artifacts": {"transcript": {"path": "mt/versions/...", "sha256": "..."},
                       "mix_wav": {...}, "mix_mp4": {...}, "lipsync_mp4": {...}},
         "created_at": "..."}]}

`current` is the only thing a reader may use for "latest" -- use `latest()`
instead of globbing a folder by mtime. Older versions stay on disk and in the
index; the live paths the UI reads are untouched.

Rules each of which is load-bearing:

1. **A version number is never reused.** `v` is one past the highest in the
   index, under a lock, so the UI server and the detached runner (two
   processes) cannot both take the same one.
2. **Revert is a new version**, not a deletion. `restore()` appends a
   `kind=revert` row and rewrites the live transcript; the version it undid
   stays.
3. **A revert leaves the mix alone.** The audio on disk still carries the
   undone edit until a re-render, so a revert row records only a transcript.
   `latest("mix_mp4")` therefore walks `parent` back to the version whose mix
   is actually on disk.
4. **Never a dependency.** Every writer here is called from a pipeline path,
   so each one is wrapped by its caller and logs instead of raising.

Per-cue TTS clips are not versioned: they are a hash cache that is rebuilt
from the transcript.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

log = logging.getLogger(__name__)

DIRNAME = "versions"
INDEX_NAME = "index.json"
LOCK_NAME = ".lock"
SCHEMA = 1

KINDS = ("original", "cast", "edit", "lipsync", "revert")
STATUSES = ("applied", "rendering", "rendered", "failed")


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def index_path(job_root: Path) -> Path:
    return Path(job_root) / DIRNAME / INDEX_NAME


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@contextlib.contextmanager
def _locked(job_root: Path) -> Iterator[None]:
    """Exclusive lock across processes. Read-modify-write happens inside it."""
    d = Path(job_root) / DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    with open(d / LOCK_NAME, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _load(job_root: Path) -> Dict[str, Any]:
    p = index_path(job_root)
    try:
        doc = json.loads(p.read_text())
        if isinstance(doc, dict) and isinstance(doc.get("versions"), list):
            return doc
    except (OSError, ValueError):
        pass
    return {"schema": SCHEMA, "current": None, "versions": []}


def _save(job_root: Path, doc: Dict[str, Any]) -> None:
    p = index_path(job_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    tmp.replace(p)


def read(job_root: Path) -> Dict[str, Any]:
    """The index, built from the files already on disk if there is none yet."""
    if not index_path(job_root).is_file():
        backfill(job_root)
    return _load(job_root)


def _row(doc: Dict[str, Any], v: int) -> Optional[Dict[str, Any]]:
    for r in doc["versions"]:
        if r.get("v") == v:
            return r
    return None


# ---------------------------------------------------------------------------
# linking existing files
# ---------------------------------------------------------------------------

def _artifact(job_root: Path, path: Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    try:
        rel = path.resolve().relative_to(Path(job_root).resolve()).as_posix()
    except ValueError:
        rel = str(path)
    return {"path": rel, "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def _index_rows(p: Path) -> List[Dict[str, Any]]:
    try:
        rows = json.loads(p.read_text())
        return rows if isinstance(rows, list) else []
    except (OSError, ValueError):
        return []


def link_transcript(job_root: Path, producer: str) -> Dict[str, Any]:
    """The newest `mt/versions/` snapshot written by `producer`."""
    vdir = Path(job_root) / "mt" / "versions"
    for row in reversed(_index_rows(vdir / INDEX_NAME)):
        if row.get("producer") == producer and row.get("file"):
            a = _artifact(job_root, vdir / row["file"])
            if a:
                return {"transcript": a}
    return {}


def link_mix(job_root: Path, tag: str) -> Dict[str, Any]:
    """The newest `mix/versions/` snapshot tagged `tag` (wav and mp4)."""
    vdir = Path(job_root) / "mix" / "versions"
    out: Dict[str, Any] = {}
    for row in reversed(_index_rows(vdir / INDEX_NAME)):
        if row.get("producer") != tag:
            continue
        for key, field in (("mix_wav", "wav"), ("mix_mp4", "mp4")):
            if row.get(field):
                a = _artifact(job_root, vdir / row[field])
                if a:
                    out[key] = a
        if out:
            return out
    return out


def _live_lipsync(job_root: Path) -> List[Path]:
    d = Path(job_root) / "lipsync"
    if not d.is_dir():
        return []
    return sorted((p for p in d.glob("*.mp4") if p.is_file()),
                  key=lambda p: p.stat().st_mtime)


def link_lipsync(job_root: Path) -> Dict[str, Any]:
    """The lip-sync render currently on disk (the newest in `lipsync/`)."""
    live = _live_lipsync(job_root)
    a = _artifact(job_root, live[-1]) if live else None
    return {"lipsync_mp4": a} if a else {}


def _live_transcript(job_root: Path) -> Optional[Path]:
    mt = Path(job_root) / "mt"
    cands = sorted(mt.glob("*.mt.json")) if mt.is_dir() else []
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


# ---------------------------------------------------------------------------
# writing versions
# ---------------------------------------------------------------------------

def append(job_root: Path, kind: str, *, parent: Optional[int] = None,
           edit_id: Optional[str] = None, nlq: str = "",
           ops: Sequence[Dict[str, Any]] = (),
           dirty_stages: Sequence[str] = (), status: str = "applied",
           artifacts: Optional[Dict[str, Any]] = None,
           extra: Optional[Dict[str, Any]] = None) -> int:
    """Add a version, make it current, return its number."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if index_path(job_root).is_file() is False:
        # An old job: record its history first, or v0 would claim the start.
        backfill(job_root)
    with _locked(job_root):
        doc = _load(job_root)
        v = 1 + max((r.get("v", -1) for r in doc["versions"]), default=-1)
        row: Dict[str, Any] = {
            "v": v,
            "parent": parent if parent is not None else doc.get("current"),
            "kind": kind, "edit_id": edit_id, "nlq": nlq,
            "ops": list(ops), "dirty_stages": list(dirty_stages),
            "status": status, "artifacts": dict(artifacts or {}),
            "created_at": _now(),
        }
        if extra:
            row.update(extra)
        doc["versions"].append(row)
        doc["current"] = v
        _save(job_root, doc)
    log.info("job version v%d (%s%s) recorded", v, kind,
             f" {edit_id}" if edit_id else "")
    return v


def update(job_root: Path, v: int, *, status: Optional[str] = None,
           artifacts: Optional[Dict[str, Any]] = None,
           error: Optional[str] = None) -> bool:
    if status is not None and status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    with _locked(job_root):
        doc = _load(job_root)
        row = _row(doc, v)
        if row is None:
            return False
        if status is not None:
            row["status"] = status
        if artifacts:
            row.setdefault("artifacts", {}).update(artifacts)
        if error is not None:
            row["error"] = error
        row["updated_at"] = _now()
        _save(job_root, doc)
    return True


def find_by_edit_id(job_root: Path, edit_id: str) -> Optional[int]:
    """Newest version recorded for `edit_NNN` / `cast`, or None."""
    for r in reversed(read(job_root)["versions"]):
        if r.get("edit_id") == edit_id:
            return r["v"]
    return None


def begin_render(job_root: Path, *, edit_id: str, kind: str = "edit") -> int:
    """Mark the version for `edit_id` as rendering, creating it if it has no row.

    A cast has no `commit()` behind it, so it is created here.
    """
    v = find_by_edit_id(job_root, edit_id)
    if v is None:
        return append(job_root, kind, edit_id=edit_id, status="rendering",
                      artifacts=_current_transcript(job_root))
    update(job_root, v, status="rendering")
    return v


def finish_render(job_root: Path, v: int, state: str,
                  error: Optional[str] = None) -> None:
    """Close a render: link what it produced, or record that it did not."""
    row = _row(_load(job_root), v) or {}
    if state != "done":
        update(job_root, v, status="failed", error=error or state)
        return
    tag = row.get("edit_id")
    arts: Dict[str, Any] = {}
    if tag:
        arts.update(link_mix(job_root, tag))
    arts.update(link_lipsync(job_root))
    update(job_root, v, status="rendered", artifacts=arts)


def _current_transcript(job_root: Path) -> Dict[str, Any]:
    live = _live_transcript(job_root)
    a = _artifact(job_root, live) if live else None
    return {"transcript_live": a} if a else {}


# ---------------------------------------------------------------------------
# lip-sync: the one artifact nothing snapshots
# ---------------------------------------------------------------------------

def stash_lipsync(job_root: Path) -> List[Dict[str, str]]:
    """Move the live render into `lipsync/versions/` before a new one writes.

    A rename, not a copy: the adapter overwrites `lipsync/*.mp4` in place, so
    without this the previous render is gone. No extra disk. The version that
    owned the file is re-pointed at its new path. Returns the moves, for
    `unstash_lipsync` if the render then fails.
    """
    moves: List[Dict[str, str]] = []
    live = _live_lipsync(job_root)
    if not live:
        return moves
    root = Path(job_root)
    vdir = root / "lipsync" / DIRNAME
    vdir.mkdir(parents=True, exist_ok=True)
    taken = [p.name for p in vdir.iterdir()]
    with _locked(root):
        doc = _load(root)
        for src in live:
            k = 0
            while any(n.startswith(f"{src.stem}.v{k:03d}.") for n in taken):
                k += 1
            dst = vdir / f"{src.stem}.v{k:03d}.mp4"
            src_rel = src.resolve().relative_to(root.resolve()).as_posix()
            os.replace(src, dst)
            taken.append(dst.name)
            dst_rel = dst.resolve().relative_to(root.resolve()).as_posix()
            for r in doc["versions"]:
                a = (r.get("artifacts") or {}).get("lipsync_mp4")
                if a and a.get("path") == src_rel:
                    a["path"] = dst_rel
            moves.append({"from": str(src), "to": str(dst),
                          "from_rel": src_rel, "to_rel": dst_rel})
        _save(root, doc)
    return moves


def unstash_lipsync(job_root: Path, moves: Sequence[Dict[str, str]]) -> None:
    """Undo `stash_lipsync` after a failed render, so the old one is live again."""
    root = Path(job_root)
    with _locked(root):
        doc = _load(root)
        for m in moves:
            dst = Path(m["to"])
            if dst.is_file() and not Path(m["from"]).exists():
                os.replace(dst, m["from"])
            for r in doc["versions"]:
                a = (r.get("artifacts") or {}).get("lipsync_mp4")
                if a and a.get("path") == m["to_rel"]:
                    a["path"] = m["from_rel"]
        _save(root, doc)


# ---------------------------------------------------------------------------
# reading "latest"
# ---------------------------------------------------------------------------

def latest(job_root: Path, key: str) -> Optional[Path]:
    """The file for `key` that belongs to the current version.

    Walks `parent` from `current` to the first version that has `key`, so a
    revert (which records only a transcript) still resolves the mix that is
    actually on disk. Keys: transcript, mix_wav, mix_mp4, lipsync_mp4.
    """
    doc = read(job_root)
    by_v = {r["v"]: r for r in doc["versions"]}
    v, seen = doc.get("current"), set()
    while v is not None and v not in seen:
        seen.add(v)
        row = by_v.get(v)
        if row is None:
            return None
        a = (row.get("artifacts") or {}).get(key)
        if a and (Path(job_root) / a["path"]).is_file():
            return Path(job_root) / a["path"]
        v = row.get("parent")
    return None


# ---------------------------------------------------------------------------
# revert
# ---------------------------------------------------------------------------

def restore(job_root: Path, v: int) -> int:
    """Make version `v`'s transcript the live one, as a NEW version.

    The undone versions stay in the index. The live file is replaced through a
    temp file that does not end in `.mt.json`, so it can never be globbed as
    the document (`edit_agent._find_merged_json`).
    """
    root = Path(job_root)
    doc = read(root)
    target = _row(doc, v)
    if target is None:
        raise ValueError(f"no version v{v} in {index_path(root)}")
    t = (target.get("artifacts") or {}).get("transcript")
    if not t or not (root / t["path"]).is_file():
        raise ValueError(f"v{v} has no transcript snapshot to restore")
    live = _live_transcript(root)
    if live is None:
        raise ValueError(f"{root.name} has no live merged JSON to replace")
    tmp = live.with_suffix(".json.tmp")
    shutil.copy2(root / t["path"], tmp)
    tmp.replace(live)

    # Snapshot the restored document so the revert row links an immutable file.
    arts: Dict[str, Any] = {}
    try:
        from .mt_versions import snapshot
        tag = f"revert_{v:03d}"
        if snapshot(live, tag):
            arts = link_transcript(root, tag)
    except Exception as e:                                # noqa: BLE001
        log.warning("job_versions: no snapshot for the revert (%s)", e)
    return append(root, "revert", artifacts=arts or {"transcript": t},
                  extra={"restores": v}, status="applied",
                  dirty_stages=["tts", "mix"])


def record_revert(job_root: Path, merged_json: Path,
                  segments: Optional[Sequence[str]] = None) -> Optional[int]:
    """Log a per-segment `edits.revert()` that already rewrote the live file.

    That revert restores `edit_backup` in place and records nothing. This
    snapshots the result and appends a `revert` row, so the undo is a version
    like any other. Returns the new `v`, or None if it could not be recorded.
    """
    try:
        read(job_root)
        from .mt_versions import snapshot
        arts: Dict[str, Any] = {}
        if snapshot(Path(merged_json), "revert"):
            arts = link_transcript(job_root, "revert")
        return append(job_root, "revert", artifacts=arts, status="applied",
                      dirty_stages=["tts", "mix"],
                      extra={"segments": list(segments) if segments else "all"})
    except Exception as e:                                # noqa: BLE001
        log.warning("job_versions: could not record the revert (%s)", e)
        return None


# ---------------------------------------------------------------------------
# old jobs
# ---------------------------------------------------------------------------

def backfill(job_root: Path) -> int:
    """Build an index from what an older job already left on disk.

    Order: original, cast, then every `edits/edit_NNN.json`. Read-only against
    the old files; the only thing it creates is `versions/index.json`.
    Returns how many versions it recorded. A job with no history gets an empty
    index so this is not repeated.
    """
    root = Path(job_root)
    with _locked(root):
        if index_path(root).is_file():
            return 0
        _save(root, {"schema": SCHEMA, "current": None, "versions": []})
    n = 0

    def add(kind, edit_id, arts, **kw):
        nonlocal n
        status = "rendered" if any(k.startswith("mix_") for k in arts) else "applied"
        append(root, kind, edit_id=edit_id, artifacts=arts,
               status=kw.pop("status", status), **kw)
        n += 1

    orig = {**link_transcript(root, "original"), **link_mix(root, "original")}
    if orig:
        add("original", "original", orig)
    cast = {**link_transcript(root, "tts"), **link_mix(root, "cast")}
    if link_mix(root, "cast"):
        add("cast", "cast", cast)
    edits = root / "edits"
    for rec in sorted(edits.glob("edit_*.json")) if edits.is_dir() else []:
        try:
            doc = json.loads(rec.read_text())
        except (OSError, ValueError):
            continue
        ops = doc.get("ops") or []
        kind = ("lipsync" if ops and all(o.get("op") == "lipsync_window"
                                         for o in ops) else "edit")
        res = doc.get("result") or {}
        add(kind, rec.stem,
            {**link_transcript(root, rec.stem), **link_mix(root, rec.stem)},
            nlq=doc.get("nlq") or "", ops=ops,
            dirty_stages=res.get("dirty_stages") or [])
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m backend.job_versions",
        description="List, inspect or restore a job's versions.")
    ap.add_argument("cmd", choices=("list", "show", "restore"))
    ap.add_argument("v", nargs="?", type=int)
    ap.add_argument("--job", "-j", required=True, help="job id under data/jobs/")
    ap.add_argument("--data-dir", default=None, type=Path)
    a = ap.parse_args(argv)

    from .edit_agent import DEFAULT_DATA_DIR
    root = Path(a.data_dir or DEFAULT_DATA_DIR) / "jobs" / a.job
    if not root.is_dir():
        print(f"no such job: {root}")
        return 2
    doc = read(root)
    if a.cmd == "list":
        print(f"{a.job}  current=v{doc['current']}")
        for r in doc["versions"]:
            mark = "*" if r["v"] == doc["current"] else " "
            print(f" {mark} v{r['v']:<3} {r['kind']:<8} {r['status']:<9} "
                  f"{r.get('edit_id') or '-':<10} {(r.get('nlq') or '')[:60]}")
        return 0
    if a.v is None:
        print("give a version number")
        return 2
    if a.cmd == "show":
        row = _row(doc, a.v)
        print(json.dumps(row, indent=2, ensure_ascii=False) if row
              else f"no v{a.v}")
        return 0 if row else 1
    try:
        new = restore(root, a.v)
    except ValueError as e:
        print(f"error: {e}")
        return 1
    print(f"restored v{a.v} as v{new}. Re-render to hear it: the audio on "
          f"disk still carries the later edits.")
    return 0


__all__ = ["append", "update", "read", "latest", "restore", "backfill",
           "record_revert",
           "link_transcript", "link_mix", "link_lipsync", "begin_render",
           "finish_render", "find_by_edit_id", "stash_lipsync",
           "unstash_lipsync", "index_path"]


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(_main())
