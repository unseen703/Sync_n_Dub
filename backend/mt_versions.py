"""Keep every version of a job's merged ASR+MT JSON on disk.

The merged JSON is the document the whole post-MT half of the pipeline
rewrites IN PLACE: stage 9b code-mixes it, 9c adds durations, 9c2 shortens
over-long lines, 9b2 attaches styles, 9e writes TTS paths, and
`backend/edits.py` applies the operator's edits to it. Each of those is a
`write_text` over the same path, so the original translation stops existing
the moment the second stage runs.

Per-segment backups (`edit_backup`, `condense_backup`) survive, but they are
a diff, not a document: they cannot answer "what did the dub say before this
run of stages" without replaying them, and they are lost outright if a stage
rewrites a key nobody thought to back up.

So every writer snapshots, and the snapshots are the record.

Naming
------
Live file, unchanged and still the only thing stages read::

    <job>/mt/<stem>.<engine>.<lang>.mt.json

Versions, beside it in their own directory::

    <job>/mt/versions/<stem>.<engine>.<lang>.v000.original.json
    <job>/mt/versions/<stem>.<engine>.<lang>.v001.codemix.json
    <job>/mt/versions/<stem>.<engine>.<lang>.v002.condense.json
    <job>/mt/versions/<stem>.<engine>.<lang>.v003.edit_000.json

Four naming decisions, each of them defensive rather than cosmetic:

1. **A SUBDIRECTORY, not `mt/` itself.**
   `edit_agent._find_merged_json` globs ``mt/*.mt.json`` and, when it finds
   several, takes **the newest by mtime** -- with only a log line. A version
   file sitting in `mt/` would therefore become "the merged JSON" for every
   subsequent edit, silently and preferentially, because it is newer than
   the live one. `glob` is not recursive, so a subdirectory is invisible to
   it no matter what the files are called.

2. **No version file ends in `.mt.json`.** Belt and braces for (1): the
   suffix is `.<producer>.json`, so even if someone later globs the tree
   recursively, these do not answer to the pattern that means "the live
   document". `producer` is therefore never allowed to be `mt`.

3. **Zero-padded to three digits.** `v10` sorts BEFORE `v9` in every
   lexicographic listing -- `ls`, `sorted()`, a file browser -- which is how
   a version history silently starts reading in the wrong order once it
   passes ten entries. Three digits also leaves room for a job that is
   re-run and re-edited many times.

4. **The producer is in the name.** `v003.condense.json` says what made it
   without opening it or cross-referencing an index.

Nothing here may break a dub. Every failure is caught and logged: a job that
cannot write a snapshot still renders, it just has a thinner history.
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

    The `.mt` is dropped so the version cannot be reassembled into something
    ending `.mt.json` by a later suffix.
    """
    name = Path(mt_json).name
    for suffix in (".mt.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def next_index(vdir: Path) -> int:
    """One past the highest `v<NNN>` present, or 0.

    Derived from the DIRECTORY rather than from a counter held anywhere
    else, so a snapshot written by a different process -- the edit agent
    replaying while the UI holds the job open -- cannot reuse a number.
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
    """Copy the merged JSON to `mt/versions/` under the next version number.

    `data` is the in-memory document when the caller already has it, which
    saves a re-read and -- more importantly -- snapshots exactly what was
    written rather than whatever is on disk by the time this runs.

    Returns the path written, or None. **None is not an error worth raising
    on**: a dub with no version history is still a dub.
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

        # Never overwrite. If the computed name exists -- two processes
        # snapshotting at once -- take the next free number instead of
        # clobbering somebody else's version.
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
    """A readable manifest beside the versions.

    Rewritten whole rather than appended to, because a partially-written
    JSON array is unreadable and this file is small.
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
            # A corrupt index must not stop the next snapshot; the files
            # themselves are the record and the index is a convenience.
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
    """Every version for this job, oldest first."""
    vdir = versions_dir(mt_json)
    if not vdir.is_dir():
        return []
    found = [p for p in vdir.iterdir() if _VERSION_RE.search(p.name)]
    # By the parsed NUMBER, not by name: the zero-padding makes the two agree
    # today, and sorting on the number keeps them agreeing if it ever stops.
    return sorted(found, key=lambda p: int(_VERSION_RE.search(p.name).group(1)))


__all__ = ["snapshot", "history", "versions_dir", "next_index",
           "VERSIONS_DIRNAME", "INDEX_NAME"]
