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
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


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
