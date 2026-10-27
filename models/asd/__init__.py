"""Active-speaker-detection registry.

Three models, two implementations:
  - talknet   : TalkNet-ASD (Tao, ACM MM 2021) -- 15.7M params, 92.3 AVA mAP
  - light_asd : Light-ASD (Liao, CVPR 2023) -- 1.0M params, 94.06 AVA mAP
  - lr_asd    : LR-ASD (Liao, IJCV 2025) -- 0.84M params, 94.45 AVA mAP

`lr_asd` is the same authors' extended `light_asd` and shares its upstream
API exactly, so `LRASD` subclasses `LightASD` in one module rather than
duplicating a scoring path. **Prefer it**: 20% fewer parameters and FLOPs,
+0.39 AVA mAP, and +5.0 points of Columbia F1 on the AVA weights (86.1 vs
81.1) -- generalisation, which is the axis that matters on unseen footage.
`light_asd` stays registered so the two can be compared on real material.

No upstream here is pip-installable. All are vendored under
`models/asd/vendor/` so nothing in `models/` reaches outside it, and all
resolve their clone through `models/_vendor.py::resolve_vendor_repo` --
which finds either the vendored or the legacy `third_party/` location by
EXISTENCE, so a checkout where the move has not been run still works.

    git clone https://github.com/TaoRuijie/TalkNet-ASD.git \\
        models/asd/vendor/talknet
    git clone https://github.com/Junhua-Liao/Light-ASD.git \\
        models/asd/vendor/lightasd
    git clone https://github.com/Junhua-Liao/LR-ASD.git \\
        models/asd/vendor/lrasd

Leave `TALKNET_REPO` / `LIGHTASD_REPO` / `LRASD_REPO` blank -- a pin that
outlives its target fails at stage 6 while the repo sits at the resolved
path.

**None of them picks the speaking face out of several.** All three score
each candidate face in isolation, which is the per-face limitation the
whole `backend/lipsync_targets.py` gating exists to work around. The Liao
models are smaller, newer and better on AVA, but Light-ASD measures WORSE
than TalkNet when faces compete (16.73 vs 13.84 competing-speaker false
alarms on MSDWild many.val) -- so they are a cheaper second opinion, not a
fix for crowds.

**One Liao-family model per process.** Light-ASD and LR-ASD both ship a
top-level `ASD.py`, `loss.py` and `model/` package under identical names,
and Python caches imports by name -- so loading both in one process scores
the second with the first one's network, silently. `LightASD` refuses that
case rather than letting it happen; stage 6 runs one model per process, so
nothing hits it today.
"""
from __future__ import annotations

from .._registry import make_registry
from .base import ASDResult, ASDScore, BaseASD

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "talknet": (".talknet_adapter", "TalkNetASD"),
    "light_asd": (".lightasd_adapter", "LightASD"),
    "lr_asd": (".lightasd_adapter", "LRASD"),
    # LoCoNet (Wang et al., CVPR 2024), added 2026-10-26. The FIRST joint,
    # multi-speaker model here -- it scores up to NUM_SPEAKERS faces at once
    # instead of each in isolation, which is the one thing the Liao family
    # and TalkNet cannot do. VGGish audio, not MFCC, so it shares only the
    # crop half of `_mfcc_av`. AVA-only weights; see the adapter docstring
    # for the dubbed-footage bias and the one-model-per-process caveat.
    #   git clone https://github.com/SJTUwxz/LoCoNet_ASD.git \
    #       models/asd/vendor/loconet
    "loconet": (".loconet_adapter", "LoCoNetASD"),
    # `ts_talknet` (Jiang-Yidi/TS-TalkNet) REMOVED 2026-10-23. Its upstream
    # was never cloned here -- `third_party/TS-TalkNet` does not exist and
    # never did -- so every `get_asd("ts_talknet")` raised on load. It was a
    # reference to a repo nobody had, and the only remaining third_party
    # name outside langfuse once TalkNet moved. In git at 438c0c2.
}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="ASD model",
    base=BaseASD,
    import_noun="ASD model",
    import_detail="is not installed/accessible. The upstreams live under "
                  "models/asd/vendor/<name> ({key} -> talknet, lightasd, "
                  "lrasd, loconet); TALKNET_REPO / LIGHTASD_REPO / LRASD_REPO "
                  "/ LOCONET_REPO override it.",
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
