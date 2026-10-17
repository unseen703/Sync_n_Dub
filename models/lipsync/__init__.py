"""Lip-sync adapter registry.

  - lipgan      : Rudrabha/LipGAN — ACM MM 2019 baseline (non-commercial)
  - wav2lip     : Rudrabha/Wav2Lip — ECCV 2020 baseline (non-commercial)
  - musetalk    : TMElyralab/MuseTalk 1.5 — latent inpainting, 256x256, fast
  - latentsync  : bytedance/LatentSync 1.6 — audio-conditioned latent
                  diffusion, 512x512, highest fidelity

Each adapter wraps an external repo clone under `third_party/`.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Optional, Type

from .base import BaseLipSync, LipSyncRequest, LipSyncResult

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "lipgan":     (".lipgan_adapter", "LipGANSync"),
    "wav2lip":    (".wav2lip_adapter", "Wav2LipSync"),
    "musetalk":   (".musetalk_adapter", "MuseTalkSync"),
    "latentsync": (".latentsync_adapter", "LatentSyncSync"),
}


def _load(key: str) -> Type[BaseLipSync]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_models() -> list[str]:
    return sorted(_REGISTRY_SPEC)


def _accepted_kwargs(cls: Type[BaseLipSync]) -> set[str]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set()
    return {
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def get_lipsync(name: str, *, device: Optional[str] = None, **kwargs) -> BaseLipSync:
    key = name.strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown lip-sync model {name!r}. "
            f"Available: {', '.join(available_models())}"
        )
    try:
        cls = _load(key)
    except ImportError as e:
        raise ImportError(
            f"Lip-sync model {key!r} is not installed. Original error: {e}"
        ) from e
    accepted = _accepted_kwargs(cls)
    if accepted:
        unknown = set(kwargs) - accepted
        if unknown:
            import warnings
            warnings.warn(
                f"{cls.__name__} does not accept kwargs {sorted(unknown)!r}; dropping."
            )
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(device=device, **kwargs)


__all__ = [
    "BaseLipSync",
    "LipSyncRequest",
    "LipSyncResult",
    "available_models",
    "get_lipsync",
]
