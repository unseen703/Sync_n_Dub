"""Face-restoration adapter registry.

  - gfpgan : TencentARC/GFPGAN v1.4 + Real-ESRGAN x4plus, run over a
             finished render by `backend/face_enhance.py`.

Restoration is orthogonal to lip-sync: video in, video out, with no idea which
model drew the mouths. It is therefore its own stage (`--lipsync X --restore
gfpgan`) rather than a variant of each lip-sync model, which keeps the
registry additive and lets a restoration tweak skip the re-render.
"""
from __future__ import annotations

import importlib
from typing import Optional, Type

from .base import BaseRestore, RestoreRequest, RestoreResult

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "gfpgan": (".gfpgan_adapter", "GFPGANRestore"),
}


def available_models() -> list[str]:
    return sorted(_REGISTRY_SPEC)


def get_restore(name: str, **kwargs) -> BaseRestore:
    key = (name or "").strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown restore model {name!r}. "
            f"Available: {', '.join(available_models())}")
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)(**kwargs)


__all__ = ["BaseRestore", "RestoreRequest", "RestoreResult",
           "available_models", "get_restore"]
