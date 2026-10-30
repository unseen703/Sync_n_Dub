"""Locate a vendored upstream, either side of the `third_party/` move.

The vendored repos live under `models/<stage>/vendor/`, but ~21 GB has to be
moved there by hand. So this resolves either location by EXISTENCE, and is
correct before the move, after it, and on a checkout where it never
happened.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

#: Where a lip-sync adapter's vendored upstream lives, relative to the repo
#: root, so nothing under `models/` reaches outside it.
VENDOR_ROOT = "backend/lipsync/vendor"


def resolve_vendor_repo(name: str, env_var: str,
                        explicit: Optional[str] = None,
                        *, legacy: Optional[str] = None,
                        vendor_root: Optional[str] = None) -> Path:
    """Locate a vendored upstream: explicit, then env, then new, then legacy.

    Resolves by EXISTENCE, not preference, so the same code works before and
    after the move.
    """
    if explicit:
        return Path(explicit)

    # parents[1], not [2]: this file is `models/_vendor.py`. Getting the index
    # wrong resolves every vendor path above the repo root, and the only
    # symptom is "repo not found" at a path that was never right.
    root = Path(__file__).resolve().parents[1]
    new = root / VENDOR_ROOT / name
    if vendor_root:
        new = root / vendor_root / name
    old = Path(legacy) if legacy else None
    old_abs = (root / old) if old and not old.is_absolute() else old

    # An env var pointing somewhere REAL is an instruction and wins. One
    # pointing at a path that no longer exists is a stale pin, and obeying it
    # means failing while the repo sits at the resolved location.
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
    # Neither present: return the new path, so the adapter's "repo not found"
    # names where it should be rather than where it used to be.
    return new
