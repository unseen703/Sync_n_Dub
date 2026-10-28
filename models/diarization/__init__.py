"""Diarization model registry — pyannote community-1 only.

Add a new model by adding it to `_REGISTRY_SPEC`. The rest of the codebase
should only ever call `get_diarizer(name, ...)` — never import an adapter
directly.

`diarizen` (BUT-FIT DiariZen) and `nemotron` (NVIDIA NeMo) were removed on
2026-10-28. Both were alternatives nobody ran: the default has been
`pyannote` at every call site since the CLI was written, and each dragged in
a conflicting dependency chain that needed its own venv. In git at b823223
if they are wanted back.
"""
from __future__ import annotations

from .._registry import make_registry
from .base import BaseDiarizer, DiarizationResult, Turn

# Imported lazily so `import models.diarization` costs nothing until a
# diarizer is actually asked for.
_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "pyannote": (".pyannote_adapter", "PyannoteDiarizer"),
}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="diarization model",
    base=BaseDiarizer,
    import_noun="Diarizer",
    import_detail="is not installed in this environment. pyannote runs in "
                  "the main `dubbing` env; `pip install pyannote.audio` and "
                  "set HF_TOKEN.",
    verbose_warning=True,
    get_doc="Instantiate a diarizer by name. Weights load lazily on first "
            "`diarize()`.\n\n    Unknown kwargs are dropped with a warning, "
            "NOT propagated. That way a\n    single call site (CLI, pipeline) "
            "can pass `hf_token=..., batch_size=...`\n    without knowing "
            "which adapter will consume which.\n    ",
)

available_models = _registry.available
get_diarizer = _registry.get

__all__ = [
    "BaseDiarizer",
    "DiarizationResult",
    "Turn",
    "available_models",
    "get_diarizer",
]
