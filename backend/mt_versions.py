"""Snapshot every version of a job's merged ASR+MT JSON.

Stages 9b/9b2/9c/9c2/9e and `backend/edits.py` all rewrite the merged JSON in
place, so each writer calls `snapshot()` first to keep a copy.

Live file, the only thing stages read::

    <job>/mt/<stem>.<engine>.<lang>.mt.json

Versions, in their own subdirectory beside it::

    <job>/mt/versions/<stem>.<engine>.<lang>.v000.original.json
    <job>/mt/versions/<stem>.<engine>.<lang>.v001.codemix.json
    <job>/mt/versions/<stem>.<engine>.<lang>.v002.condense.json

Naming constraints, each load-bearing:

1. Versions go in a subdirectory. `edit_agent._find_merged_json` globs
   ``mt/*.mt.json`` and takes the newest by mtime; a version file in `mt/`
   would become "the merged JSON" for every later edit.
2. No version file ends in `.mt.json`, and `producer` may never be `mt`.
3. The index is zero-padded to three digits -- `v10` sorts before `v9`
   otherwise.

Every failure here is caught and logged: a job that cannot write a snapshot
still renders.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

VERSIONS_DIRNAME = "versions"
INDEX_NAME = "index.json"

#: `<anything>.v<NNN>.<producer>.json`
_VERSION_RE = re.compile(r"\.v(\d{3,})\.[^.]+\.json$")

#: Reserved: a version must never be mistakable for the live document.
_FORBIDDEN_PRODUCERS = {"mt"}

_PRODUCER_OK = re.compile(r"^[a-z0-9][a-z0-9_]{0,31}$")


def versions_dir(mt_json: Path) -> Path:
    return Path(mt_json).parent / VERSIONS_DIRNAME


def _base_name(mt_json: Path) -> str:
    """`a.bodhan.hi.mt.json` -> `a.bodhan.hi`.

    Drops `.mt` so a version name cannot end in `.mt.json`.
    """
    name = Path(mt_json).name
    for suffix in (".mt.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def next_index(vdir: Path) -> int:
    """One past the highest `v<NNN>` in `vdir`, or 0.

    Read from the directory, not a stored counter, so two processes
    snapshotting at once cannot reuse a number.
    """
    highest = -1
    if vdir.is_dir():
        for p in vdir.iterdir():
            m = _VERSION_RE.search(p.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return highest + 1


def snapshot(mt_json: Path, producer: str, *,
             data: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """Copy the merged JSON into `mt/versions/` under the next version number.

    Pass `data` when the caller already holds the document in memory; it is
    snapshotted instead of re-reading the file.

    Returns the path written, or None on any failure (a dub with no version
    history is still a dub).
    """
    try:
        mt_json = Path(mt_json)
        producer = str(producer).strip().lower()
        if not _PRODUCER_OK.match(producer):
            log.warning("mt_versions: refusing producer name %r "
                        "(need [a-z0-9_], 1-32 chars)", producer)
            return None
        if producer in _FORBIDDEN_PRODUCERS:
            log.warning("mt_versions: %r is reserved -- a version must not "
                        "be mistakable for the live document", producer)
            return None

        if data is None:
            if not mt_json.is_file():
                return None
            payload = mt_json.read_text()
        else:
            payload = json.dumps(data, indent=2, ensure_ascii=False)

        vdir = versions_dir(mt_json)
        vdir.mkdir(parents=True, exist_ok=True)
        base = _base_name(mt_json)

        # Never overwrite: if the name exists, take the next free number.
        for bump in range(100):
            idx = next_index(vdir) + bump
            dest = vdir / f"{base}.v{idx:03d}.{producer}.json"
            if not dest.exists():
                break
        else:                                             # pragma: no cover
            log.warning("mt_versions: could not find a free version number "
                        "in %s", vdir)
            return None

        dest.write_text(payload)
        _append_index(vdir, dest, producer, payload)
        log.info("mt version %s -> %s", producer, dest.name)
        return dest
    except Exception as e:                                # noqa: BLE001
        # A snapshot is a record, never a dependency.
        log.warning("mt_versions: could not snapshot %s (%s)", producer, e)
        return None


def _append_index(vdir: Path, dest: Path, producer: str, payload: str) -> None:
    """Append a row to `versions/index.json`.

    Rewritten whole rather than appended to: a half-written JSON array is
    unreadable.
    """
    import datetime as _dt

    index_path = vdir / INDEX_NAME
    rows: List[Dict[str, Any]] = []
    if index_path.is_file():
        try:
            loaded = json.loads(index_path.read_text())
            if isinstance(loaded, list):
                rows = loaded
        except Exception:                                 # noqa: BLE001
            # The version files are the record; the index is a convenience.
            log.warning("mt_versions: index at %s unreadable; starting a new "
                        "one (the version FILES are unaffected)", index_path)
    n_segments = None
    try:
        n_segments = len(json.loads(payload).get("segments") or [])
    except Exception:                                     # noqa: BLE001
        pass
    rows.append({
        "file": dest.name,
        "producer": producer,
        "bytes": len(payload.encode("utf-8")),
        "segments": n_segments,
        "written_at": _dt.datetime.now().isoformat(timespec="seconds"),
    })
    index_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def history(mt_json: Path) -> List[Path]:
    """Every version file for this job, oldest first."""
    vdir = versions_dir(mt_json)
    if not vdir.is_dir():
        return []
    found = [p for p in vdir.iterdir() if _VERSION_RE.search(p.name)]
    # Sort on the parsed number, not the name, so it survives a change in
    # zero-padding.
    return sorted(found, key=lambda p: int(_VERSION_RE.search(p.name).group(1)))


__all__ = ["snapshot", "history", "versions_dir", "next_index",
           "VERSIONS_DIRNAME", "INDEX_NAME"]
