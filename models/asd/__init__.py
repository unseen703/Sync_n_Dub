"""Active-speaker-detection registry.

Two adapters:
  - talknet    : TalkNet-ASD (Tao Ruijie, Interspeech 2021)
  - ts_talknet : TS-TalkNet with target-speaker enrolment (Interspeech 2023)

Both wrap repos that are not pip-installable. The adapters expect the repo
checked out next to the project and expose a `TALKNET_REPO` env var to point
at the clone path:

    git clone https://github.com/TaoRuijie/TalkNet-ASD.git third_party/TalkNet
    git clone https://github.com/Jiang-Yidi/TS-TalkNet.git third_party/TS-TalkNet
    export TALKNET_REPO=$(pwd)/third_party/TalkNet
    export TS_TALKNET_REPO=$(pwd)/third_party/TS-TalkNet
"""
from __future__ import annotations

from .._registry import make_registry
from .base import ASDResult, ASDScore, BaseASD

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "talknet": (".talknet_adapter", "TalkNetASD"),
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
    import_detail="is not installed/accessible. The upstream lives under "
                  "models/asd/vendor/talknet (TALKNET_REPO overrides it).",
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
