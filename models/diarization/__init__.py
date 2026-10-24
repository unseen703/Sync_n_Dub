"""Diarization model registry.

Add a new model by importing its class here and adding it to `_REGISTRY`.
The rest of the codebase should only ever call `get_diarizer(name, ...)` —
never import an adapter directly.
"""
from __future__ import annotations

from .._registry import make_registry
from .base import BaseDiarizer, DiarizationResult, Turn

# Each adapter lives in its own venv (pyannote / nemo / diarizen) because
# their dependency chains conflict. Import lazily so a venv that only has
# one adapter installed can still `import models.diarization`.
_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "pyannote": (".pyannote_adapter", "PyannoteDiarizer"),
    "diarizen": (".diarizen_adapter", "DiariZenDiarizer"),
    "nemotron": (".nemotron_adapter", "NemotronDiarizer"),
}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="diarization model",
    base=BaseDiarizer,
    import_noun="Diarizer",
    import_detail="is not installed in this environment. Activate the "
                  "matching venv (e.g. .venv-{key}) or install "
                  "requirements-{key}.txt.",
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
