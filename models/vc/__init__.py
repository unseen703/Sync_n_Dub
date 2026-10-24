"""Voice-conversion adapter registry — EMPTY since 2026-10-23.

`openvoice` (MyShell OpenVoice v2) and `vevo` (Open-MMLab Amphion) were
removed by request, as part of leaving no reference to `third_party/`
outside langfuse. Neither upstream was ever installed on this host --
`third_party/OpenVoice` and `third_party/Amphion` do not exist and never
did -- so `get_vc(...)` raised on load and stage 9f could not run. The
adapters were a reference to a repo nobody had, not a capability.

The package stays so `--vc` keeps parsing (`backend/cli.py` builds its
`choices` from `available_models()`, which is now empty) and so a real VC
adapter can be added back without rebuilding the stage around it. Both
files are in git at 438c0c2 if they are wanted again.
"""
from __future__ import annotations

from .._registry import make_registry
from .base import BaseVC, VCRequest, VCResult, VCSegment

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="VC model",
    base=BaseVC,
    import_noun="VC model",
)

available_models = _registry.available
get_vc = _registry.get

__all__ = [
    "BaseVC",
    "VCRequest",
    "VCResult",
    "VCSegment",
    "available_models",
    "get_vc",
]
