"""Voice-conversion adapter registry.

  - openvoice : MyShell OpenVoice v2 — timbre converter + tone-colour clone
  - vevo      : Open-MMLab Amphion Vevo — unified controllable VC/TTS

Both do zero-shot voice conversion: take a TTS-synthesised clip (source
timbre) and the diarized speaker's reference clip (target timbre) and
return a WAV that keeps the content and prosody but shifts to the target
voice. The dubbing pipeline chains VC after TTS when the stock-voice
assignment is "close but not the real person" (plan §8, E14).
"""
from __future__ import annotations

import importlib
import inspect
from typing import Optional, Type

from .base import BaseVC, VCRequest, VCResult, VCSegment

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "openvoice": (".openvoice_adapter", "OpenVoiceVC"),
    "vevo": (".vevo_adapter", "VevoVC"),
}


def _load(key: str) -> Type[BaseVC]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_models() -> list[str]:
    return sorted(_REGISTRY_SPEC)


def _accepted_kwargs(cls: Type[BaseVC]) -> set[str]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set()
    return {
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def get_vc(name: str, *, device: Optional[str] = None, **kwargs) -> BaseVC:
    key = name.strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown VC model {name!r}. Available: {', '.join(available_models())}"
        )
    try:
        cls = _load(key)
    except ImportError as e:
        raise ImportError(
            f"VC model {key!r} is not installed. Original error: {e}"
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
    "BaseVC",
    "VCRequest",
    "VCResult",
    "VCSegment",
    "available_models",
    "get_vc",
]
