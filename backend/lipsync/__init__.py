"""Lip-sync adapter registry.

  - wav2lip     : Rudrabha/Wav2Lip — ECCV 2020 baseline (non-commercial)
  - musetalk    : TMElyralab/MuseTalk 1.5 — latent inpainting, 256x256, fast
  - latentsync15: bytedance/LatentSync 1.5 — audio-conditioned latent
                  diffusion at 256x256, ~8 GB

Each adapter wraps a repo clone under `backend/lipsync/vendor/`; see
`backend/common/vendor.py`.

`server/service/lipsync_models.normalise()` matches on PREFIX, so a key that is a prefix
of another (`latentsync` of `latentsync15`) must never be added without its
own row there, or the longer id silently downgrades to it.
"""
from __future__ import annotations

from backend.common.registry import make_registry
from .base import BaseLipSync, LipSyncRequest, LipSyncResult

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "wav2lip":    (".wav2lip_adapter", "Wav2LipSync"),
    "musetalk":   (".musetalk_adapter", "MuseTalkSync"),
    "latentsync15": (".latentsync_adapter", "LatentSyncSync"),
}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="lip-sync model",
    base=BaseLipSync,
    import_noun="Lip-sync model",
)

available_models = _registry.available
get_lipsync = _registry.get

def __getattr__(name):
    if name == "run_lipsync":
        import importlib
        return getattr(importlib.import_module(".lipsync", __name__), "run_lipsync")
    raise AttributeError(name)

__all__ = [
    "BaseLipSync",
    "LipSyncRequest",
    "LipSyncResult",
    "available_models",
    "get_lipsync",
]
