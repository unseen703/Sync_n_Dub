"""Face-restoration adapter registry.

  - gfpgan : TencentARC/GFPGAN v1.4 + Real-ESRGAN x4plus, run over a
             finished render by `scripts/face_enhance.py`.

**This is not a lip-sync model, and that distinction is the whole reason the
package exists.** It used to be one: `wav2lip_hd` was registered in
`models/lipsync/` as a fifth lip-sync adapter, when all it did was run
`wav2lip` and then this pass. That shape does not scale -- 5 lip-sync models
x {plain, restored} is 10 registry entries, each needing its own `FACTS` row,
measured rate band, caveat and op-catalogue line, and 15 once a second
restorer lands. It also forced a re-render for every restoration tweak,
which matters because the pass measured 1156 s against the MuseTalk render's
447 s: 2.6x the cost of the thing it cleans.

Restoration is orthogonal to lip-sync. It takes a video and gives back a
video, and has no idea which model drew the mouths -- so it is `--lipsync X
--restore gfpgan`, 5 + 2 rather than 5 x 2.
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
