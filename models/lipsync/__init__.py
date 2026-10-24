"""Lip-sync adapter registry.

  - wav2lip     : Rudrabha/Wav2Lip — ECCV 2020 baseline (non-commercial)
  - musetalk    : TMElyralab/MuseTalk 1.5 — latent inpainting, 256x256, fast
  - latentsync  : bytedance/LatentSync 1.6 — audio-conditioned latent
                  diffusion, 512x512, highest fidelity, ~18 GB inference
  - latentsync15: the SAME code at 256x256 with the 1.5 checkpoint —
                  ~8 GB, so it fits these 16 GiB cards with room to spare.
                  1.6 exists because 1.5's teeth and lips are blurrier;
                  that is the trade.

Each adapter wraps an external repo clone under `models/lipsync/vendor/`
(moved out of `third_party/` on 2026-10-23; see `models/_vendor.py`).

REMOVED 2026-10-23 — `wav2lip_hd` and `lipgan`.

`wav2lip_hd` was never a model. Its `sync()` was `super().sync()` followed by
a Real-ESRGAN + GFPGAN pass, so stage 1 was byte-identical to `wav2lip`: the
same `wav2lip_gan.pth` at `img_size=96`, the same resize policy, the same
coord cache. Restoration is now stage 9i (`models/restore/`,
`--restore gfpgan`), which makes the pair exactly
`--lipsync wav2lip --restore gfpgan` and leaves nothing for an adapter to do.
Keeping it would also have meant 5 x 2 registry entries the moment a second
restorer landed, each needing its own measured rate band and caveat.

Both stay in `ui/lipsync_models.FACTS` as RETIRED, display-only rows. Three
job folders on disk name `wav2lip_hd`, and `normalise()` matches on prefix —
so deleting the row resolves them to plain `wav2lip`, which is the exact
silent-downgrade CLAUDE.md S4 records for this pair.
"""
from __future__ import annotations

from .._registry import make_registry
from .base import BaseLipSync, LipSyncRequest, LipSyncResult

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "wav2lip":    (".wav2lip_adapter", "Wav2LipSync"),
    "musetalk":   (".musetalk_adapter", "MuseTalkSync"),
    "latentsync": (".latentsync_adapter", "LatentSyncSync"),
    "latentsync15": (".latentsync_adapter", "LatentSync15Sync"),
}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="lip-sync model",
    base=BaseLipSync,
    import_noun="Lip-sync model",
)

available_models = _registry.available
get_lipsync = _registry.get

__all__ = [
    "BaseLipSync",
    "LipSyncRequest",
    "LipSyncResult",
    "available_models",
    "get_lipsync",
]
