"""Active-speaker-detection registry -- LoCoNet only.

  - loconet : LoCoNet (Wang, CVPR 2024) -- joint multi-speaker ASD

The upstream is not pip-installable; clone it under
`backend/asd/vendor/loconet` and leave `LOCONET_REPO` blank, since a pin that
outlives its target fails at stage 6.

    git clone https://github.com/SJTUwxz/LoCoNet_ASD.git \\
        backend/asd/vendor/loconet

LoCoNet does not solve crowds on its own: its AVA-only weights are biased
against dubbed footage, so `backend/lipsync_targets.py` gating stays.
"""
from __future__ import annotations

from backend.common.registry import make_registry
from .base import ASDResult, ASDScore, BaseASD

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    # Scores up to NUM_SPEAKERS faces jointly rather than each in isolation.
    # AVA-only weights; see the adapter docstring for the dubbed-footage bias
    # and the one-model-per-process caveat.
    "loconet": (".loconet_adapter", "LoCoNetASD"),
}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="ASD model",
    base=BaseASD,
    import_noun="ASD model",
    import_detail="is not installed/accessible. The upstream lives at "
                  "backend/asd/vendor/loconet; LOCONET_REPO overrides it.",
)

available_models = _registry.available
get_asd = _registry.get

_STAGE_EXPORTS = {
    "ASD_MODEL_PREFERENCE", "SCORES_SUFFIX", "SegmentAssignment",
    "assign_faces_to_turns", "assignments_to_jsonable",
    "find_scores_npz", "load_scores_npz", "save_scores_npz", "score_asd",
}


def __getattr__(name):
    if name in _STAGE_EXPORTS:
        import importlib
        return getattr(importlib.import_module(".asd", __name__), name)
    raise AttributeError(name)

__all__ = [
    "ASD_MODEL_PREFERENCE",
    "ASDResult",
    "ASDScore",
    "BaseASD",
    "SCORES_SUFFIX",
    "SegmentAssignment",
    "assign_faces_to_turns",
    "assignments_to_jsonable",
    "available_models",
    "find_scores_npz",
    "get_asd",
    "load_scores_npz",
    "save_scores_npz",
    "score_asd",
]
