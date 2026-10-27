"""Warm-model LatentSync driver: load once, render many pieces.

Run UNDER the `latentsync` conda env, with the vendored repo as the working
directory (relative paths like `configs`, `checkpoints/whisper/...` and the
HF id `stabilityai/sd-vae-ft-mse` resolve from there) -- exactly as the
`LatentSyncSync` adapter already shells out. The adapter invokes this instead
of the vendored `scripts/inference.py` when it has more than one piece for a
GPU: that script's `main()` loads the model and builds the pipeline for a
SINGLE (video, audio) call, so a segmented render used to pay a full diffusion-
stack load per piece.

This driver replicates that one-time setup (the vendored `scripts/inference.py`
`main()` body) once, then loops a JSON manifest of pieces through the same
`pipeline(...)` call, freeing the per-piece buffers in between. The pixels are
identical to the per-call path -- only when the weights load changes.

Nothing here edits the vendored clone; it imports the clone's internals, with
the repo put on `sys.path` from `--repo` so those imports resolve while the
driver file itself lives in this repo.

Manifest: a JSON list of objects
  {"video": ..., "audio": ..., "out": ..., "temp_dir": ...,
   "targets_npz": ... | null}
Per piece the driver prints one line the adapter parses from stdout:
  [worker] ok   <out>
  [worker] fail <out> :: <reason>
so one piece crashing is reported against that piece rather than aborting the
rest (the adapter also treats a missing output file as the failure arbiter).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback


def _mem_used_pct():
    """Share of system RAM in use, or None. Mirrors MuseTalk's self-contained
    guard (vendor/musetalk/scripts/inference.py) -- the `dubbing` env's
    `backend.memguard` is not importable here."""
    try:
        total = avail = None
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key == "MemTotal":
                    total = float(rest.split()[0])
                elif key == "MemAvailable":
                    avail = float(rest.split()[0])
                if total is not None and avail is not None:
                    break
        return 100.0 * (total - avail) / total if total else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="vendored LatentSync repo")
    ap.add_argument("--manifest", required=True, help="JSON list of pieces")
    ap.add_argument("--unet_config_path", required=True)
    ap.add_argument("--inference_ckpt_path", required=True)
    ap.add_argument("--inference_steps", type=int, default=20)
    ap.add_argument("--guidance_scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1247)
    ap.add_argument("--enable_deepcache", action="store_true")
    ap.add_argument("--memory_efficient", action="store_true")
    args = ap.parse_args()

    # The vendored package must win over anything shadowing it, so prepend.
    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import torch
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from accelerate.utils import set_seed
    from latentsync.whisper.audio2feature import Audio2Feature

    with open(args.manifest) as fh:
        pieces = json.load(fh)

    config = OmegaConf.load(args.unet_config_path)

    # ---- one-time model setup (vendor scripts/inference.py::main) ----------
    is_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16 else torch.float32

    scheduler = DDIMScheduler.from_pretrained("configs")
    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )
    unet = unet.to(dtype=dtype)
    pipeline = LipsyncPipeline(
        vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler,
    ).to("cuda")
    if args.enable_deepcache:
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()

    print(f"[worker] model loaded once for {len(pieces)} piece(s)", flush=True)

    # ---- loop the pieces, cycling only the per-piece buffers ---------------
    rc = 0
    for i, p in enumerate(pieces):
        out = p["out"]
        print(f"[worker] start {i + 1}/{len(pieces)} {out}", flush=True)
        pct = _mem_used_pct()
        limit = float(os.environ.get("DUBBING_MEM_GUARD_PCT", "95") or 95)
        if pct is not None and limit > 0 and pct >= limit:
            print(f"[worker] fail {out} :: system RAM {pct:.1f}%% >= {limit:.0f}%% "
                  f"before piece {i + 1}/{len(pieces)}", flush=True)
            rc = 1
            continue
        try:
            if args.seed != -1:
                set_seed(args.seed)
            else:
                torch.seed()
            pipeline(
                video_path=p["video"],
                audio_path=p["audio"],
                video_out_path=out,
                num_frames=config.data.num_frames,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                weight_dtype=dtype,
                width=config.data.resolution,
                height=config.data.resolution,
                mask_image_path=config.data.mask_image_path,
                temp_dir=p.get("temp_dir", "temp"),
                memory_efficient=args.memory_efficient,
                targets_npz=p.get("targets_npz"),
            )
            print(f"[worker] ok   {out}", flush=True)
        except Exception as e:                                # noqa: BLE001
            rc = 1
            print(f"[worker] fail {out} :: {e}", flush=True)
            traceback.print_exc()
        finally:
            # Release this piece's frames before the next decodes its own.
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
