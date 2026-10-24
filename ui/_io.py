"""Small reads the UI does everywhere, written once.

A job folder is read, never locked, while a pipeline may be writing into it.
So every read here is best-effort by design: a file that is absent, truncated
mid-write, or not JSON at all is "nothing yet", not an exception that takes
down a status endpoint.

`read_json` had four copies — `ui.server`, `ui.speakers`, `ui.media`,
`ui.routes_edits` — and three of them agreed. `ui.media`'s returned `{}`
instead of `None`, which is the kind of divergence that only shows up at a
call site doing `.get()` on the result. The shared one returns **None**,
matching the majority and the way almost every call site already reads
(`_read_json(p) or {}`); `ui.media`'s two call sites gained that `or {}`.

`env_int` / `env_float` / `data_dir` are the same idea for a smaller, sharper
bug. `.env.example` ships several numeric vars blank on purpose (`VAR=`),
and `python-dotenv` loads that as the key PRESENT with an empty string —
`os.environ.get(name, default)` then returns `""`, not `default`, and
`int("")` raises. Three call sites (`ui.server`, `ui.ytmeta`, `ui.clip`) did
this as a bare module-level `int(...)`/`float(...)` with no guard, so
`cp .env.example .env` followed by `python -m ui.server` crashed at import,
before the port was bound. Four more call sites had already written the same
try/except by hand (`ui.jobqueue.max_concurrent`/`spawn_grace_s`,
`ui.uploads.max_upload_mb`, `ui.routes_create.max_lipsync_s`) — one
canonical version here instead of five.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

#: `ui/_io.py` -> repo root. Every `ui/*.py` module computes this the same
#: way from its own `__file__`; `data_dir()` needs it once, here.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> Optional[Any]:
    """Parsed JSON, or None if it is missing, unreadable or malformed.

    None rather than `{}` so a caller can tell "no file" from "a file holding
    an empty object" when it cares — and `or {}` is one character more for
    the callers that do not.
    """
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def env_int(name: str, default: int) -> int:
    """`int(os.environ[name])`, where missing OR BLANK both mean `default`.

    See the module docstring for why a bare `int(os.environ.get(name, d))`
    is not safe here. A value that is present but does not parse is treated
    the same as blank, with a warning, rather than taking the process down.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r is not an integer; using the default %s",
                    name, raw, default)
        return default


def env_float(name: str, default: float) -> float:
    """`float` counterpart of `env_int` — see its docstring."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using the default %s",
                    name, raw, default)
        return default


def data_dir() -> Path:
    """Where this UI's data lives — one answer, read by every module.

    `DUBBING_UI_DATA_DIR` is the documented name (`.env.example`).
    `DUBBING_DATA_DIR` is accepted as an older alias because
    `ui.routes_edits` and `ui.routes_enhance` shipped reading that name
    instead, which meant setting one var moved HALF the UI: jobs were
    created under one tree by `ui.server`/`ui.routes_create`/`ui.runner`
    while their edit, enhance and lip-sync routes looked in the other and
    404'd. Blank is treated as unset, same as `env_int`/`env_float` above.
    """
    for name in ("DUBBING_UI_DATA_DIR", "DUBBING_DATA_DIR"):
        raw = os.environ.get(name)
        if raw and raw.strip():
            return Path(raw.strip())
    return _REPO_ROOT / "data"
