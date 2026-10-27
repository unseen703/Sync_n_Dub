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

    `workers_per_gpu` is the second axis of the split `num_gpus` starts, and
    until now the only one with no way to reach it: `scripts/face_enhance.py`
    has had the flag since it learned to shard, and nothing in the repo ever
    set it. One worker measured 1691 MiB of 16311 and 33.9% mean GPU
    utilisation -- the per-frame path is CPU-bound (decode, paste_back,
    resize, x264), so a card given one worker sits idle two thirds of the
    time while its memory goes unused.

    None keeps the script's own default of 1, which is today's behaviour.
    That is deliberate: peak VRAM rises linearly per worker and the gain is
    bounded by CPU contention, so the useful value is a measurement on this
    host rather than a guess written into a default.

    `seek_shards` lets each shard seek to its first frame instead of decoding
    every frame before it -- four shards otherwise decode 2.5x the video
    between them. False by default and deliberately NOT exposed over HTTP:
    an off-by-one seek shifts every whole-video index in the shard without
    changing the frame count, so `--frame-mask` restores the wrong mouths and
    nothing downstream can tell. The guards and the check that proves it on a
    given source are in `scripts/face_enhance.py`'s module docstring.
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
