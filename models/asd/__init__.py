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

import importlib
import inspect
from typing import Optional, Type

from .base import ASDResult, ASDScore, BaseASD

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "talknet": (".talknet_adapter", "TalkNetASD"),
    "ts_talknet": (".ts_talknet_adapter", "TSTalkNetASD"),
}


def _load(key: str) -> Type[BaseASD]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_models() -> list[str]:
    return sorted(_REGISTRY_SPEC)


def _accepted_kwargs(cls: Type[BaseASD]) -> set[str]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set()
    return {
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def get_asd(name: str, *, device: Optional[str] = None, **kwargs) -> BaseASD:
    key = name.strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown ASD model {name!r}. "
            f"Available: {', '.join(available_models())}"
        )
    try:
        cls = _load(key)
    except ImportError as e:
        raise ImportError(
            f"ASD model {key!r} is not installed/accessible. "
            f"Clone the upstream repo and set the matching env var "
            f"(TALKNET_REPO or TS_TALKNET_REPO). Original error: {e}"
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
    "ASDResult",
    "ASDScore",
    "BaseASD",
    "available_models",
    "get_asd",
]
