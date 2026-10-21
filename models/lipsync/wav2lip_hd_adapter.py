"""saifhassan/Wav2Lip-HD — Wav2Lip followed by Real-ESRGAN/GFPGAN restoration.

**This is not a higher-resolution Wav2Lip.** Upstream's `run_final.sh` calls
`inference.py` with `--no_sr --no_segmentation`, so stage 1 is plain Wav2Lip
with `wav2lip_gan.pth` at `img_size = 96` — byte-for-byte the model
`wav2lip_adapter.py` already drives. The "HD" is stage 2: Real-ESRGAN
`x4plus` over the rendered frames with `--face_enhance` (GFPGAN).

Which makes it, architecturally, the restoration pass that was deleted from
this repo on 2026-10-17 with `models/restore/`. Recorded here because the
cost profile that motivated that removal applies again: CodeFormer measured
~13x clip length on this host, and a per-frame GAN is the same shape of
expense. It is a second pass, not a cheaper one.

**Stage 1 is the existing adapter, subclassed, not reimplemented.** That is
deliberate: `Wav2LipSync` carries the resize_factor=2 default (a 4K source
raises "Image too big to run face detection on GPU" at 1), the per-run
private scratch dir that stops sharded runs racing on `temp/result.avi`, the
coord cache, and the rc=0-with-no-output check. Reimplementing stage 1 would
fork all four. A `wav2lip_hd` run therefore uses whatever the `wav2lip`
defaults currently are, and tracks them when they change.

**Stage 2 runs in its own conda env.** `basicsr` imports
`torchvision.transforms.functional_tensor`, removed in torchvision 0.17,
while the `dubbing` env is on 0.23 — and it cannot simply be pinned down,
because these are Blackwell sm_120 cards that need torch >= 2.7 / cu128. So
the shim lives in a cloned `wav2liphd` env reached through
`WAV2LIPHD_PYTHON` or the sibling-env fallback, exactly as MuseTalk and
LatentSync are reached.

Stage 2 is skipped, with a warning rather than an error, when the env or the
weights are absent: the result is then an ordinary Wav2Lip render, which is
a usable dub. Failing the whole job after stage 1 has already paid for the
GPU time would be the worse answer.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from .base import LipSyncRequest, LipSyncResult, resolve_env_python
from .wav2lip_adapter import Wav2LipSync, _probe_video

log = logging.getLogger(__name__)

#: Where the four stage-2 checkpoints live. Fetched from the upstream GitHub
#: releases, not the repo's Google Drive links: `run_final.sh` names
#: `-n RealESRGAN_x4plus`, which is the standard published weight, and its
#: custom `esrgan_yunying.pth` is only read by the in-`inference.py` SR path
#: that the same script disables with `--no_sr`.
DEFAULT_WEIGHTS_DIR = "models/weights/wav2lip_hd"
DEFAULT_HD_REPO = "third_party/Wav2Lip-HD"

#: Stage 2 lives in `scripts/`, NOT inside the vendored clone, and that is
#: load-bearing rather than tidiness. Python puts a script's own directory on
#: sys.path, and `third_party/Wav2Lip-HD/basicsr/` is a partial vendored copy
#: of basicsr -- so a stage-2 script sitting in that directory shadows the
#: installed basicsr 1.4.2 and dies with
#: `cannot import name 'circular_lowpass_kernel'`. Keeping our file out of the
#: clone also leaves `third_party/` exactly as upstream shipped it.
ENHANCER = Path(__file__).resolve().parents[2] / "scripts/wav2liphd_enhance.py"


class Wav2LipHDSync(Wav2LipSync):
    name = "wav2lip_hd"
    licence = "non-commercial"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        hd_repo_path: Optional[str] = None,
        weights_dir: Optional[str] = None,
        hd_python: Optional[str] = None,
        face_enhance: bool = True,
        enhance_tile: int = 512,
        enhance: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(device=device, **kwargs)
        self.hd_repo_path = Path(
            hd_repo_path or os.getenv("WAV2LIPHD_REPO") or DEFAULT_HD_REPO)
        self.weights_dir = Path(
            weights_dir or os.getenv("WAV2LIPHD_WEIGHTS")
            or DEFAULT_WEIGHTS_DIR)
        self.hd_python = hd_python
        self.face_enhance = face_enhance
        self.enhance_tile = enhance_tile
        self.enhance = enhance

    # -- stage 2 ---------------------------------------------------------

    def _stage2_blockers(self) -> list[str]:
        """Everything missing that stage 2 needs, as readable sentences."""
        missing = []
        if not ENHANCER.is_file():
            missing.append(f"enhancer script missing at {ENHANCER}")
        for w in ("RealESRGAN_x4plus.pth", "GFPGANv1.4.pth"):
            if not (self.weights_dir / w).is_file():
                missing.append(f"weight missing at {self.weights_dir / w}")
        return missing

    def _enhance(self, rendered: Path, target: tuple[int, int]) -> Optional[str]:
        """Run stage 2 in place. Returns an error string, or None on success."""
        blockers = self._stage2_blockers()
        if blockers:
            return "; ".join(blockers)

        python_bin = resolve_env_python("wav2liphd", "wav2liphd", self.hd_python)
        enhanced = rendered.with_suffix(".hd.mp4")
        tw, th = target
        from backend.media_io import _ffmpeg_bin
        # The EXACT rate, as a rational, not cv2's float. `_frame_rate` reads
        # ffprobe's `r_frame_rate` ("24000/1001"); a decimal approximation of
        # it is what lets a re-encoded shard end up tagged differently from
        # its sibling, and a mismatched shard is dropped by the concat
        # demuxer without an error. Upstream Wav2Lip-HD truncates this to an
        # int and then recombines at a hardcoded 20 fps, which is exactly the
        # trap this avoids.
        from backend.lipsync import _frame_rate
        rate = _frame_rate(rendered)

        cmd = [
            python_bin, str(ENHANCER),
            "--input", str(rendered.resolve()),
            "--output", str(enhanced.resolve()),
            "--weights-dir", str(self.weights_dir.resolve()),
            "--target-width", str(tw), "--target-height", str(th),
            "--tile", str(self.enhance_tile),
            "--ffmpeg", _ffmpeg_bin() or "ffmpeg",
        ]
        if rate:
            cmd += ["--fps", rate]
        if self.face_enhance:
            cmd.append("--face-enhance")
        log.info("wav2lip_hd[gpu=%s]: stage 2 %s", self.cuda_device, " ".join(cmd))
        proc = subprocess.run(cmd, env=self._subprocess_env(),
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return (f"real-esrgan rc={proc.returncode}: "
                    f"{proc.stderr.strip()[-2000:]}")
        if not enhanced.is_file() or enhanced.stat().st_size == 0:
            return (f"real-esrgan exited 0 but wrote no {enhanced}. "
                    f"stderr tail: {proc.stderr.strip()[-1000:]}")

        # Carry stage 1's audio across: the enhancer writes video only.
        ff = _ffmpeg_bin() or "ffmpeg"
        muxed = rendered.with_suffix(".hd.mux.mp4")
        mux = subprocess.run(
            [ff, "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(enhanced), "-i", str(rendered),
             "-map", "0:v:0", "-map", "1:a:0?",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-shortest", str(muxed)],
            capture_output=True, text=True)
        final = muxed if (mux.returncode == 0 and muxed.is_file()) else enhanced
        if final is enhanced:
            log.warning("wav2lip_hd: audio remux failed (%s); keeping the "
                        "silent enhanced render",
                        mux.stderr.strip()[-300:])
        final.replace(rendered)
        for leftover in (enhanced, muxed):
            if leftover != final:
                leftover.unlink(missing_ok=True)
        return None

    # -- entry point -----------------------------------------------------

    def sync(self, request: LipSyncRequest) -> LipSyncResult:
        result = super().sync(request)
        result.model_name = self.name
        if result.error or not self.enhance:
            return result

        out = Path(result.out_video)
        if not out.is_file():
            return result

        # Stage 2 must return the SOURCE's dimensions, not stage 1's. Wav2Lip
        # renders at resize_factor=2, i.e. half the source, so this pass is
        # also what takes the frame back to full resolution -- which is the
        # one thing `resize_factor=1` buys, without its 3.7x VRAM cost.
        from backend.lipsync import _video_dims
        target = _video_dims(Path(request.video)) or _video_dims(out)
        if not target:
            result.extra = dict(result.extra or {})
            result.extra["hd_enhance"] = "skipped: could not probe dimensions"
            log.warning("wav2lip_hd: could not probe source dimensions; "
                        "returning the plain Wav2Lip render")
            return result

        err = self._enhance(out, target)
        result.extra = dict(result.extra or {})
        if err:
            # Deliberately NOT a failure: stage 1's render is on disk and is
            # a usable dub. Marking the run failed here would discard it.
            result.extra["hd_enhance"] = f"skipped: {err}"
            log.warning("wav2lip_hd: stage 2 skipped (%s); returning the "
                        "plain Wav2Lip render", err)
        else:
            result.extra["hd_enhance"] = {
                "model": "RealESRGAN_x4plus",
                "face_enhance": self.face_enhance,
                "gfpgan": "GFPGANv1.4" if self.face_enhance else None,
                "target": f"{target[0]}x{target[1]}",
            }
            fps, dur = _probe_video(out)
            result.fps, result.duration_s = fps, dur
        return result
