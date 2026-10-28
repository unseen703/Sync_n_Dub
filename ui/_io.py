"""Small reads the UI does everywhere, written once.

A job folder is read, never locked, while a pipeline may be writing into it,
so every read here is best-effort: a file that is absent, truncated
mid-write or not JSON is "nothing yet", not an exception that takes down a
status endpoint.

`env_int` / `env_float` exist because `.env.example` ships several numeric
vars blank on purpose (`VAR=`), and python-dotenv loads that as the key
present with an empty string -- `os.environ.get(name, default)` returns
`""`, not `default`, and `int("")` raises at import time.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

#: `ui/_io.py` -> repo root. `data_dir()` needs it once, here.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> Optional[Any]:
    """Parsed JSON, or None if the file is missing, unreadable or malformed.

    None rather than `{}` so a caller can tell "no file" from "an empty
    object"; callers that do not care write `read_json(p) or {}`.
    """
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def env_int(name: str, default: int) -> int:
    """`int(os.environ[name])`, where missing or blank both mean `default`.

    A value that is present but does not parse is treated as blank, with a
    warning, rather than raising.
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
    """`float` counterpart of `env_int`."""
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
    """Where this UI's data lives -- one answer for every module.

    `DUBBING_UI_DATA_DIR` is the documented name (`.env.example`);
    `DUBBING_DATA_DIR` is accepted as an older alias. Both are needed:
    `routes_edits` and `routes_enhance` once read only the alias, so setting
    one var moved half the UI and the other half 404'd. Blank means unset.
    """
    for name in ("DUBBING_UI_DATA_DIR", "DUBBING_DATA_DIR"):
        raw = os.environ.get(name)
        if raw and raw.strip():
            return Path(raw.strip())
    return _REPO_ROOT / "data"
