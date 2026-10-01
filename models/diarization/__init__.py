"""Diarization model registry.

Add a new model by importing its class here and adding it to `_REGISTRY`.
The rest of the codebase should only ever call `get_diarizer(name, ...)` —
never import an adapter directly.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Optional, Type

from .base import BaseDiarizer, DiarizationResult, Turn

# Each adapter lives in its own venv (pyannote / nemo / diarizen) because
# their dependency chains conflict. Import lazily so a venv that only has
# one adapter installed can still `import models.diarization`.
_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "pyannote": (".pyannote_adapter", "PyannoteDiarizer"),
    "diarizen": (".diarizen_adapter", "DiariZenDiarizer"),
}


def _load(key: str) -> Type[BaseDiarizer]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_models() -> list[str]:
    """Names accepted by `get_diarizer` / `--model`."""
    return sorted(_REGISTRY_SPEC)


def _accepted_kwargs(cls: Type[BaseDiarizer]) -> set[str]:
    """Kwarg names the constructor of `cls` will accept."""
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set()   # sentinel — anything is accepted
    return {
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def get_diarizer(name: str, *, device: Optional[str] = None, **kwargs) -> BaseDiarizer:
    """Instantiate a diarizer by name. Weights load lazily on first `diarize()`.

    Unknown kwargs are dropped with a warning, NOT propagated. That way a
    single call site (CLI, pipeline) can pass `hf_token=..., batch_size=...`
    without knowing which adapter will consume which.
    """
    key = name.strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown diarization model {name!r}. "
            f"Available: {', '.join(available_models())}"
        )
    try:
        cls = _load(key)
    except ImportError as e:
        raise ImportError(
            f"Diarizer {key!r} is not installed in this environment. "
            f"Activate the matching venv (e.g. .venv-{key}) or install "
            f"requirements-{key}.txt. Original error: {e}"
        ) from e
    accepted = _accepted_kwargs(cls)
    if accepted:
        unknown = set(kwargs) - accepted
        if unknown:
            import warnings
            warnings.warn(
                f"{cls.__name__} does not accept kwargs {sorted(unknown)!r}; dropping. "
                f"Accepted: {sorted(accepted - {'device'})!r}."
            )
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(device=device, **kwargs)


__all__ = [
    "BaseDiarizer",
    "DiarizationResult",
    "Turn",
    "available_models",
    "get_diarizer",
]
