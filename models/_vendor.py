"""Where a vendored upstream lives, and how to find it either side of a move.

The lip-sync upstreams moved out of `third_party/` into
`models/lipsync/vendor/` on 2026-10-23, and TalkNet followed into
`models/asd/vendor/`, so that nothing under `models/` reaches outside it.

~21 GB has to move for that layout to be real, and the move is a shell
command an operator runs (`scripts/move_lipsync_vendor.sh`) rather than
something the code can do for itself. So the code resolves EITHER location,
by existence rather than preference, and is correct before the move, after
it, and on a checkout where it never happened.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

#: Where a lip-sync adapter's vendored upstream lives, relative to the repo
#: root. Moved out of `third_party/` on 2026-10-23 by request, so nothing
#: under `models/` reaches outside it.
VENDOR_ROOT = "models/lipsync/vendor"


def resolve_vendor_repo(name: str, env_var: str,
                        explicit: Optional[str] = None,
                        *, legacy: Optional[str] = None,
                        vendor_root: Optional[str] = None) -> Path:
    """Locate a vendored upstream: explicit, then env, then new, then legacy.

    The legacy `third_party/<Name>` fallback is the whole point of this
    helper existing rather than the paths simply being rewritten. ~21 GB has
    to move for the new layout to be real, and that move is a shell command
    the operator runs, not something the code can do for itself. Between the
    code change and the move -- and on any checkout where the move has not
    happened -- the old path is still the correct answer.

    It resolves by EXISTENCE, not by preference: whichever is actually on
    disk wins, so the same code is right before and after. When both exist
    (a half-finished move, or a copy rather than a move) the new location
    wins and the old one is named in a warning, because two copies of a
    vendored repo with local patches applied is the state where a fix gets
    made in the one nobody is running.
    """
    if explicit:
        return Path(explicit)

    # parents[1], not [2]. This file is `models/_vendor.py`; it was
    # `models/lipsync/base.py` until 2026-10-23 and the index came with it,
    # which resolved every vendor path one directory ABOVE the repo root --
    # silently, because the paths simply did not exist and the adapter then
    # reported "repo not found" at its new wrong location.
    root = Path(__file__).resolve().parents[1]
    new = root / VENDOR_ROOT / name
    if vendor_root:
        new = root / vendor_root / name
    old = Path(legacy) if legacy else None
    old_abs = (root / old) if old and not old.is_absolute() else old

    # An env var that points somewhere REAL is an instruction and wins.
    # One pointing at a path that no longer exists is a stale pin -- e.g.
    # `.env` still naming `third_party/TalkNet-ASD` after the move -- and
    # honouring it means failing with "repo not found" while the repo sits
    # at the new location. Saying where it went beats obeying a dead path.
    from_env = os.getenv(env_var)
    if from_env:
        p = Path(from_env)
        p_abs = (root / p) if not p.is_absolute() else p
        if p_abs.exists():
            return p_abs
        if new.exists():
            _log.warning(
                "%s=%s does not exist; using %s instead. Blank that variable "
                "in .env -- a pin that outlives its target is a failure "
                "waiting for the next checkout.", env_var, from_env, new)
            return new
        return p_abs

    if new.exists():
        if old_abs and old_abs.exists():
            _log.warning(
                "lipsync vendor %r exists in BOTH %s and %s; using the "
                "former. Two copies of a patched upstream is how a fix "
                "lands in the one nobody runs -- delete the stale one.",
                name, new, old_abs)
        return new
    if old_abs and old_abs.exists():
        return old_abs
    # Neither present: return the NEW path so the adapter's own "repo not
    # found" error points at where it should be, not where it used to be.
    return new
