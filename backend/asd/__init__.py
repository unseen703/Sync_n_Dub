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

__all__ = [
    "ASDResult",
    "ASDScore",
    "BaseASD",
    "available_models",
    "get_asd",
]
