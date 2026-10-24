"""GFPGAN v1.4 (+ Real-ESRGAN) over a finished render.

Shells into `scripts/face_enhance.py`, which runs in the `wav2liphd` conda
env. That env exists because `basicsr` imports
`torchvision.transforms.functional_tensor`, removed in torchvision 0.17,
while `dubbing` is on 0.23 -- and it cannot simply be pinned back, because
these are Blackwell sm_120 cards needing torch >= 2.7 / cu128.

**The env keeps its name, the variable does not.** `RESTORE_PYTHON` is what
this reads; the sibling-env fallback still resolves `envs/wav2liphd/bin/python`
because that directory is a real thing on disk. Renaming the env would mean
rebuilding it for no gain, but a variable called `WAV2LIPHD_PYTHON` pointing
at the interpreter that restores a *MuseTalk* render is a lie the next
reader has to unpick.

**The weights directory keeps its historical name too** --
`models/weights/wav2lip_hd/` holds `GFPGANv1.4.pth`,
`RealESRGAN_x4plus.pth`, `detection_Resnet50_Final.pth` and
`parsing_parsenet.pth`, none of which are Wav2Lip's. Moving 600 MB to a
better-named directory is a separate change; `RESTORE_WEIGHTS` overrides it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from models.lipsync.base import resolve_env_python

from .base import BaseRestore, RestoreRequest, RestoreResult

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ENHANCER = ROOT / "scripts" / "face_enhance.py"
DIFF_MASK = ROOT / "scripts" / "lipsync_diff_mask.py"
DEFAULT_WEIGHTS_DIR = "models/weights/wav2lip_hd"
WEIGHT_FILES = ("GFPGANv1.4.pth", "RealESRGAN_x4plus.pth")

_SUMMARY_RE = re.compile(
    r"wrote (\d+) frame\(s\).*?restored (\d+), passed through (\d+)")


class GFPGANRestore(BaseRestore):
    name = "gfpgan"

    def __init__(self, *, weights_dir: Optional[str] = None,
                 python_bin: Optional[str] = None) -> None:
        self.weights_dir = Path(
            weights_dir or os.getenv("RESTORE_WEIGHTS")
            or DEFAULT_WEIGHTS_DIR)
        if not self.weights_dir.is_absolute():
            self.weights_dir = ROOT / self.weights_dir
        self.python_bin = python_bin

    # -- preflight -------------------------------------------------------

    def blockers(self) -> list[str]:
        """Everything missing, as sentences a person can act on.

        Returned rather than raised so a readiness endpoint can disable a
        button and say why, instead of the operator finding out after the
        GPU has been committed.
        """
        out = []
        if not ENHANCER.is_file():
            out.append(f"enhancer script missing at {ENHANCER}")
        for w in WEIGHT_FILES:
            if not (self.weights_dir / w).is_file():
                out.append(f"weight missing at {self.weights_dir / w}")
        try:
            py = resolve_env_python("restore", "wav2liphd", self.python_bin)
            if not Path(py).is_file():
                out.append(f"interpreter not found at {py}; set RESTORE_PYTHON")
        except Exception as e:                            # noqa: BLE001
            out.append(f"could not resolve an interpreter ({e})")
        return out

    # -- the mask --------------------------------------------------------

    def build_mask(self, *, before: Path, after: Path, out: Path,
                   ffprobe: Optional[str] = None) -> Optional[dict]:
        """Which frames the lip-sync model changed, by diffing.

        Model-agnostic on purpose. MuseTalk can report untouched frames via
        its `(0,0,0,0)` placeholders but only when ASD-gated, and nothing
        persists them; Wav2Lip has no concept of an untouched frame at all.
        A diff is the only answer that works for every adapter, and it
        cannot drift from what the model did because it IS what it did.

        Returns the mask document, or None when the diff fails -- the caller
        then restores every frame, which is slower and still correct.
        """
        from backend.media_io import _ffprobe_bin

        py = resolve_env_python("restore", "wav2liphd", self.python_bin)
        cmd = [py, str(DIFF_MASK), "--before", str(before),
               "--after", str(after), "--out", str(out)]
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if p.returncode != 0 or not out.is_file():
            log.warning("restore: mask build failed (rc=%s): %s; "
                        "restoring every frame instead",
                        p.returncode, (p.stderr or "").strip()[-500:])
            return None
        try:
            doc = json.loads(out.read_text())
        except Exception as e:                            # noqa: BLE001
            log.warning("restore: unreadable mask (%s)", e)
            return None
        log.info("restore: mask -> %s of %s frame(s) changed (%.1f%%)",
                 doc.get("changed_count"), doc.get("frames"),
                 doc.get("changed_fraction_pct") or 0.0)
        return doc

    # -- entry point -----------------------------------------------------

    def restore(self, request: RestoreRequest) -> RestoreResult:
        t0 = time.time()
        out = Path(request.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        blockers = self.blockers()
        if blockers:
            return RestoreResult(model_name=self.name, out_video=out,
                                 error="; ".join(blockers))

        from backend.media_io import _ffmpeg_bin, _ffprobe_bin
        ffmpeg = _ffmpeg_bin() or "ffmpeg"
        ffprobe = _ffprobe_bin() or "ffprobe"
        py = resolve_env_python("restore", "wav2liphd", self.python_bin)

        mask_doc = None
        mask_path = out.parent / f"{out.stem}.changed_frames.json"
        if request.gated and request.source and Path(request.source).is_file():
            mask_doc = self.build_mask(before=Path(request.source),
                                       after=Path(request.video),
                                       out=mask_path, ffprobe=ffprobe)

        cmd = [py, str(ENHANCER),
               "--input", str(Path(request.video).resolve()),
               "--output", str(out.resolve()),
               "--weights-dir", str(self.weights_dir.resolve()),
               "--face-enhance",
               "--strength", str(request.strength),
               "--ffmpeg", ffmpeg, "--ffprobe", ffprobe]
        if mask_doc is not None:
            cmd += ["--frame-mask", str(mask_path.resolve())]
        if request.scale_to:
            cmd += ["--target-width", str(request.scale_to[0]),
                    "--target-height", str(request.scale_to[1])]
        if request.only_center_face:
            cmd.append("--only-center-face")
        if request.bg_upsample:
            cmd.append("--bg-upsample")
        if request.num_gpus:
            cmd += ["--gpus", ",".join(str(i) for i in range(request.num_gpus))]

        log.info("restore[gfpgan]: %s", " ".join(cmd))
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        dt = time.time() - t0
        tail = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()

        res = RestoreResult(model_name=self.name, out_video=out, wall_s=dt,
                            extra={"mask": str(mask_path) if mask_doc else None,
                                   "changed_frames": (mask_doc or {}).get(
                                       "changed_count"),
                                   "strength": request.strength})
        m = _SUMMARY_RE.search(tail)
        if m:
            res.frames, res.restored, res.passed_through = (
                int(m.group(1)), int(m.group(2)), int(m.group(3)))

        # The FILE is the arbiter. rc=0 with no output is the shape CLAUDE.md
        # records for Wav2Lip, and a 32 MB file that decoded 370 of a claimed
        # 1500 frames is the shape this repo hit on 2026-10-22.
        if p.returncode != 0:
            res.error = f"face_enhance rc={p.returncode}: {tail[-2000:]}"
        elif not res.produced:
            res.error = (f"face_enhance exited 0 but wrote no {out}. "
                         f"tail: {tail[-1000:]}")
        if res.error:
            log.error("restore[gfpgan]: FAILED in %.0fs — %s", dt, res.error)
        else:
            log.info("restore[gfpgan]: done in %.0fs → %s "
                     "(%d frames, %d restored, %d passed through)",
                     dt, out, res.frames, res.restored, res.passed_through)
        return res
