"""Shared types for restoration adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class RestoreRequest:
    """One restoration pass over one rendered video.

    `source` is the video the lip-sync model was GIVEN. With it the adapter
    diffs the two and restores only the changed frames, roughly a 10x saving;
    without it every frame is restored, which is correct but wasteful.

    `scale_to` is a separate axis from restoration on purpose — GFPGANer
    bundles the two, which makes a same-size pass resample every frame and
    back for nothing.

    `workers_per_gpu` splits each card further. One worker leaves the GPU
    mostly idle (the per-frame path is CPU-bound: decode, paste_back, resize,
    x264), but peak VRAM rises linearly, so None keeps the script's default of
    1 rather than guessing.

    `seek_shards` lets each shard seek to its first frame instead of decoding
    everything before it. False by default and deliberately NOT exposed over
    HTTP: an off-by-one seek shifts every whole-video index without changing
    the frame count, so `--frame-mask` restores the wrong mouths and nothing
    downstream can tell. See `backend/face_enhance.py` for the guards.
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
    workers_per_gpu: Optional[int] = None
    seek_shards: bool = False
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
