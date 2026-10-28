"""Active-speaker-detection registry -- LoCoNet only.

  - loconet : LoCoNet (Wang, CVPR 2024) -- joint multi-speaker ASD

The upstream is not pip-installable. It is vendored under
`models/asd/vendor/loconet` so nothing in `models/` reaches outside it, and
resolves through `models/_vendor.py::resolve_vendor_repo` -- which finds
either the vendored or the legacy `third_party/` location by EXISTENCE, so a
checkout where the move has not been run still works.

    git clone https://github.com/SJTUwxz/LoCoNet_ASD.git \\
        models/asd/vendor/loconet

Leave `LOCONET_REPO` blank -- a pin that outlives its target fails at stage 6
while the repo sits at the resolved path.

**Why only this one.** TalkNet, Light-ASD and LR-ASD were removed on
2026-10-28. All three scored each candidate face in ISOLATION, which is the
per-face limitation `backend/lipsync_targets.py` exists to work around;
LoCoNet scores up to NUM_SPEAKERS faces at once and is the only model here
that can actually answer "which of these two is speaking". It has been the
default since it landed.

**It still does not solve crowds on its own.** `lipsync_targets.py` gating
stays: AVA-only weights carry a bias against dubbed footage, where the
mouth motion no longer matches the audio.
"""
from __future__ import annotations

from .._registry import make_registry
from .base import ASDResult, ASDScore, BaseASD

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    # LoCoNet (Wang et al., CVPR 2024), added 2026-10-26. The FIRST joint,
    # multi-speaker model here -- it scores up to NUM_SPEAKERS faces at once
    # instead of each in isolation, which is the one thing the Liao family
    # and TalkNet could not do. VGGish audio, not MFCC. AVA-only weights;
    # see the adapter docstring for the dubbed-footage bias and the
    # one-model-per-process caveat.
    #   git clone https://github.com/SJTUwxz/LoCoNet_ASD.git \
    #       models/asd/vendor/loconet
    "loconet": (".loconet_adapter", "LoCoNetASD"),
    # `ts_talknet` (Jiang-Yidi/TS-TalkNet) REMOVED 2026-10-23. Its upstream
    # was never cloned here -- `third_party/TS-TalkNet` does not exist and
    # never did -- so every `get_asd("ts_talknet")` raised on load.
    #
    # `talknet`, `light_asd` and `lr_asd` REMOVED 2026-10-28. All three
    # scored each candidate face in ISOLATION, which is the per-face
    # limitation `backend/lipsync_targets.py` exists to work around;
    # loconet has been the default since it landed and is the only one that
    # scores competing faces jointly. In git at b823223.
}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="ASD model",
    base=BaseASD,
    import_noun="ASD model",
    import_detail="is not installed/accessible. The upstream lives at "
                  "models/asd/vendor/loconet; LOCONET_REPO overrides it.",
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
