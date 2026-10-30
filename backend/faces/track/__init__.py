"""Face-tracker registry — ByteTrack only."""
from __future__ import annotations

from backend.common.registry import make_registry
from ..base import BaseFaceTracker, FaceTracks

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bytetrack": (".bytetrack_adapter", "ByteTrackTracker"),
}

# `pass_device=False`: a tracker is pure geometry over boxes the detector
# already produced, so `get_tracker` takes no `device` at all.
_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="face tracker",
    base=BaseFaceTracker,
    import_noun="Face tracker",
    import_detail="is not installed. Install with `pip install supervision`.",
    pass_device=False,
)

available_trackers = _registry.available
get_tracker = _registry.get

__all__ = [
    "BaseFaceTracker",
    "FaceTracks",
    "available_trackers",
    "get_tracker",
]
