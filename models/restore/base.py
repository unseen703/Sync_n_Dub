"""Shared types for restoration adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class RestoreRequest:
    """One restoration pass over one rendered video.

    `source` is the video the lip-sync model was GIVEN, and it is optional
    only because not every caller has it. With it, the adapter can diff the
    two and restore just the frames that changed -- the difference between
    1156 s and ~130 s on a 1500-frame clip. Without it, every frame is
    restored, which is correct but pays for 1265 frames that are already the
    source pixels.

    `scale_to` is a separate axis from restoration on purpose. GFPGANer
    bundles the two (its `upscale` resizes the whole background and pastes
    the face onto that), and keeping them bundled is what made a same-size
    pass resample every frame to 4K and back for nothing.
    """
    video: Path
    out_path: Path
    source: Optional[Path] = None
    gated: bool = True
    strength: float = 1.0
    scale_to: Optional[Tuple[int, int]] = None
    only_center_face: bool = False
    bg_upsample: bool = False
    num_gpus: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RestoreResult:
    model_name: str
    out_video: Path
    frames: int = 0
    restored: int = 0
    passed_through: int = 0
    wall_s: float = 0.0
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def produced(self) -> bool:
        """The FILE is the arbiter, never the error field (CLAUDE.md S4)."""
        p = Path(self.out_video)
        return p.is_file() and p.stat().st_size > 0


class BaseRestore:
    name = "base"

    def restore(self, request: RestoreRequest) -> RestoreResult:  # pragma: no cover
        raise NotImplementedError
