"""Snapshot every version of a job's dubbed mix.

Mix rewrites `mix/<stem>.dubbed.wav` and `mix/<stem>.dubbed.mp4` in place on
every edit replay, so each mix calls `snapshot()` first. Same pattern as
`backend.mt_versions`.

Live files, the only ones the UI reads::

    <job>/mix/<stem>.dubbed.wav
    <job>/mix/<stem>.dubbed.mp4

Versions, in their own subdirectory beside them::

    <job>/mix/versions/<stem>.v000.original.dubbed.wav
    <job>/mix/versions/<stem>.v001.edit_000.dubbed.mp4

Naming constraints, each load-bearing:

1. Versions go in a subdirectory. `frontend/media.find_dubbed_video` falls back to
   "newest `.mp4` in `mix/`" when the manifest pointer is missing, so a
   version file in `mix/` would silently become the UI's dub.
2. The `.dubbed.<ext>` tail is kept so the files stay playable; the
   `.v<NNN>.<producer>` segment goes ahead of it.
3. The index is zero-padded to three digits -- `v10` sorts before `v9`
   otherwise.

Every failure here is caught and logged: a mix that cannot be snapshotted
still plays.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

VERSIONS_DIRNAME = "versions"
INDEX_NAME = "index.json"

#: `<stem>.v<NNN>.<producer>.dubbed.<ext>`
_VERSION_RE = re.compile(r"\.v(\d{3,})\.[^.]+\.dubbed\.(wav|mp4)$")

_PRODUCER_OK = re.compile(r"^[a-z0-9][a-z0-9_]{0,31}$")


def versions_dir(mix_dir: Path) -> Path:
    return Path(mix_dir) / VERSIONS_DIRNAME


def _stem(path: Path) -> str:
    """`a.dubbed.wav` -> `a`.

    Drops the `.dubbed.<ext>` tail so the version can grow a
    `.v<NNN>.<producer>` segment and keep the same ending.
    """
    name = Path(path).name
    for suffix in (".dubbed.wav", ".dubbed.mp4"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


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


def snapshot(wav: Optional[Path], mp4: Optional[Path], producer: str
             ) -> Tuple[Optional[Path], Optional[Path]]:
    """Copy the dubbed wav/mp4 into `mix/versions/` under the next number.

    Either file may be None; the one that exists is copied. Returns
    `(versioned_wav, versioned_mp4)`, each None when its input was missing
    or could not be copied. Failures are logged, not raised.
    """
    wav = Path(wav) if wav else None
    mp4 = Path(mp4) if mp4 else None
    try:
        producer = str(producer).strip().lower()
        if not _PRODUCER_OK.match(producer):
            log.warning("mix_versions: refusing producer name %r "
                        "(need [a-z0-9_], 1-32 chars)", producer)
            return (None, None)

        pick = wav if (wav and wav.is_file()) else (mp4 if (mp4 and mp4.is_file()) else None)
        if pick is None:
            log.warning("mix_versions: neither wav nor mp4 exists; nothing to "
                        "snapshot for producer %r", producer)
            return (None, None)
        mix_dir = pick.parent
        stem = _stem(pick)

        vdir = versions_dir(mix_dir)
        vdir.mkdir(parents=True, exist_ok=True)

        # Never overwrite: if the name exists, take the next free number.
        for bump in range(100):
            idx = next_index(vdir) + bump
            wav_dest = vdir / f"{stem}.v{idx:03d}.{producer}.dubbed.wav"
            mp4_dest = vdir / f"{stem}.v{idx:03d}.{producer}.dubbed.mp4"
            if not wav_dest.exists() and not mp4_dest.exists():
                break
        else:                                             # pragma: no cover
            log.warning("mix_versions: could not find a free version number "
                        "in %s", vdir)
            return (None, None)

        written: Dict[str, Optional[Path]] = {"wav": None, "mp4": None}
        for src, dst, key in ((wav, wav_dest, "wav"), (mp4, mp4_dest, "mp4")):
            if src is None or not src.is_file():
                continue
            try:
                shutil.copy2(src, dst)
                written[key] = dst
            except Exception as e:                         # noqa: BLE001
                log.warning("mix_versions: could not copy %s -> %s (%s)",
                            src.name, dst.name, e)

        if written["wav"] is None and written["mp4"] is None:
            return (None, None)

        _append_index(vdir, idx, producer, written["wav"], written["mp4"])
        log.info("mix version %s -> %s%s", producer,
                 written["wav"].name if written["wav"] else "",
                 (", " + written["mp4"].name) if written["mp4"] else "")
        return (written["wav"], written["mp4"])
    except Exception as e:                                # noqa: BLE001
        # A snapshot is a record, never a dependency.
        log.warning("mix_versions: could not snapshot %s (%s)", producer, e)
        return (None, None)


def _append_index(vdir: Path, idx: int, producer: str,
                  wav: Optional[Path], mp4: Optional[Path]) -> None:
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
            log.warning("mix_versions: index at %s unreadable; starting a "
                        "new one (the version FILES are unaffected)",
                        index_path)
    row: Dict[str, Any] = {
        "index": idx,
        "producer": producer,
        "written_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if wav is not None:
        try:
            row["wav"] = wav.name
            row["wav_bytes"] = wav.stat().st_size
        except Exception:                                 # noqa: BLE001
            pass
    if mp4 is not None:
        try:
            row["mp4"] = mp4.name
            row["mp4_bytes"] = mp4.stat().st_size
        except Exception:                                 # noqa: BLE001
            pass
    rows.append(row)
    index_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def history(mix_dir: Path) -> List[Path]:
    """Every version file for this job, oldest first.

    Returns the wav when both extensions are present (the mp4 is a mux of
    the same mix), otherwise whichever one is on disk.
    """
    vdir = versions_dir(mix_dir)
    if not vdir.is_dir():
        return []
    found = [p for p in vdir.iterdir() if _VERSION_RE.search(p.name)]
    found.sort(key=lambda p: (int(_VERSION_RE.search(p.name).group(1)),
                              _VERSION_RE.search(p.name).group(2)))
    # Dedup by (idx, producer), keeping wav over mp4 so a listing shows one
    # row per snapshot by default.
    seen: set = set()
    out: List[Path] = []
    for p in found:
        m = _VERSION_RE.search(p.name)
        key = (int(m.group(1)),)
        if key in seen:
            continue
        # Prefer wav for the canonical row.
        wav_sibling = p.with_name(p.name[:-4] + ".wav") if p.suffix == ".mp4" else p
        out.append(wav_sibling if wav_sibling.is_file() else p)
        seen.add(key)
    return out


__all__ = ["snapshot", "history", "versions_dir", "next_index",
           "VERSIONS_DIRNAME", "INDEX_NAME"]
