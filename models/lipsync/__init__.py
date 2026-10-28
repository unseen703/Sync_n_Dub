"""Lip-sync adapter registry.

  - wav2lip     : Rudrabha/Wav2Lip — ECCV 2020 baseline (non-commercial)
  - musetalk    : TMElyralab/MuseTalk 1.5 — latent inpainting, 256x256, fast
  - latentsync15: bytedance/LatentSync 1.5 — audio-conditioned latent
                  diffusion at 256x256, ~8 GB, so it fits these 16 GiB
                  cards with room to spare

Each adapter wraps an external repo clone under `models/lipsync/vendor/`
(moved out of `third_party/` on 2026-10-23; see `models/_vendor.py`).

REMOVED 2026-10-23 — `wav2lip_hd` and `lipgan`.

`wav2lip_hd` was never a model. Its `sync()` was `super().sync()` followed by
a Real-ESRGAN + GFPGAN pass, so stage 1 was byte-identical to `wav2lip`: the
same `wav2lip_gan.pth` at `img_size=96`, the same resize policy, the same
coord cache. Restoration is now stage 9i (`models/restore/`,
`--restore gfpgan`), which makes the pair exactly
`--lipsync wav2lip --restore gfpgan`.

REMOVED 2026-10-28 — `latentsync` (1.6).

It defaulted to the 512x512 UNet, measured at ~18 GB against 16311 MiB
cards: the one registered model that could not run here, and an OOM
surfaced as "produced no mp4" rather than an error. 1.5 is the same code and
the same checkout at 256x256; only the checkpoint and config differ.

All three are gone from `ui/lipsync_models.FACTS` too, by request on
2026-10-28. Checked against disk first: `latentsync` and `lipgan` are named
by ZERO job folders, so dropping their rows changes nothing. `wav2lip_hd` is
named by three (`hd_bench1_40s`, `hd_w4l11_40s`, `hd_iitprof_2900to3000`),
and because `normalise()` matches on prefix those three now resolve to plain
`wav2lip` — they label as "Wav2Lip" and quote its 3-5x rate instead of
6-24x. Cosmetic for a finished render, and accepted deliberately.

That prefix rule is still live: a key that is a prefix of another (and
`latentsync` is a prefix of `latentsync15`) must never be re-added without a
row, or the longer id silently downgrades to it.
"""
from __future__ import annotations

from .._registry import make_registry
from .base import BaseLipSync, LipSyncRequest, LipSyncResult

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "wav2lip":    (".wav2lip_adapter", "Wav2LipSync"),
    "musetalk":   (".musetalk_adapter", "MuseTalkSync"),
    "latentsync15": (".latentsync_adapter", "LatentSyncSync"),
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
