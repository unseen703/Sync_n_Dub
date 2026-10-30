"""GFPGAN v1.4 (+ Real-ESRGAN) face restoration over a finished render.

Shells into `python -m backend.face_enhance` with the current interpreter.
basicsr / gfpgan / facexlib / realesrgan were folded into the `dubbing` env
on 2026-10-30, so no separate interpreter resolution is needed; the historical
`torchvision.transforms.functional_tensor` incompatibility is patched inline
by `install.sh` and the facexlib/gfpgan weights are placed by
`./fetch_assets.py` (feature `restore`). Weights live in
`models/weights/restore/`; `RESTORE_WEIGHTS` overrides the location.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import BaseRestore, RestoreRequest, RestoreResult

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = "models/weights/restore"
WEIGHT_FILES = ("GFPGANv1.4.pth", "RealESRGAN_x4plus.pth")

_SUMMARY_RE = re.compile(
    r"wrote (\d+) frame\(s\).*?restored (\d+), passed through (\d+)")


def _parent_json(tail: str) -> Optional[dict]:
    """The sharding parent's trailing JSON line, when the run sharded.

    `_SUMMARY_RE` matches a WORKER's summary, but a worker's stdout goes to its
    own part log, so on a sharded run (the multi-GPU default) nothing in the
    captured output matches and the counts read 0. The parent prints totals as
    JSON instead; the regex stays for the single-worker shape, which has no
    parent.
    """
    for line in reversed((tail or "").splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict) and "workers" in doc:
            return doc
    return None


def _open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"diff: cannot open {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return cap, w, h


def _build_change_mask(before: Path, after: Path, out: Path, *, log) -> Optional[dict]:
    b_cap, b_w, b_h = _open_video(str(before))
    f_cap, f_w, f_h = _open_video(str(after))

    threshold = 16
    min_pixels = max(16, round(f_w * f_h / 32000))
    pad = max(4, round(16 * f_h / 1080))
    log(f"diff: {f_w}x{f_h} -> threshold={threshold} "
        f"min_pixels={min_pixels} pad={pad}")
    if b_w != f_w or b_h != f_h:
        log(f"diff: before is {b_w}x{b_h}, after is {f_w}x{f_h} -- "
            f"resizing before to the render's size")

    changed: list[int] = []
    boxes: dict[str, list[int]] = {}
    counts: list[int] = []
    idx = 0
    while True:
        ok_b, fb = b_cap.read()
        ok_f, ff = f_cap.read()
        if not (ok_b and ok_f):
            break
        if (fb.shape[1], fb.shape[0]) != (ff.shape[1], ff.shape[0]):
            fb = cv2.resize(fb, (ff.shape[1], ff.shape[0]),
                            interpolation=cv2.INTER_AREA)
        d = cv2.absdiff(cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(ff, cv2.COLOR_BGR2GRAY))
        mask = (d >= threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n = int(mask.sum())
        counts.append(n)
        if n >= min_pixels:
            ys, xs = np.nonzero(mask)
            x1 = max(0, int(xs.min()) - pad)
            y1 = max(0, int(ys.min()) - pad)
            x2 = min(ff.shape[1], int(xs.max()) + pad)
            y2 = min(ff.shape[0], int(ys.max()) + pad)
            changed.append(idx)
            boxes[str(idx)] = [x1, y1, x2, y2]
        idx += 1
    b_cap.release()
    f_cap.release()

    arr = np.asarray(counts) if counts else np.zeros(1)
    frac = 100.0 * len(changed) / max(1, idx)
    doc = {
        "before": str(Path(before).resolve()),
        "after": str(Path(after).resolve()),
        "frames": idx,
        "threshold": threshold,
        "min_pixels": min_pixels,
        "pad": pad,
        "changed_count": len(changed),
        "changed_fraction_pct": round(frac, 2),
        "changed": changed,
        "boxes": boxes,
        "changed_pixels_p50": int(np.percentile(arr, 50)),
        "changed_pixels_p95": int(np.percentile(arr, 95)),
        "changed_pixels_max": int(arr.max()),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))
    return doc


class GFPGANRestore(BaseRestore):
    name = "gfpgan"

    def __init__(self, *, weights_dir: Optional[str] = None,
                 python_bin: Optional[str] = None) -> None:
        self.weights_dir = Path(
            weights_dir or os.getenv("RESTORE_WEIGHTS")
            or DEFAULT_WEIGHTS_DIR)
        if not self.weights_dir.is_absolute():
            self.weights_dir = ROOT / self.weights_dir
        # `python_bin` kept for backwards compatibility with old call sites /
        # request.json payloads; the restore env was merged into `dubbing` on
        # 2026-10-30, so sys.executable is always correct.
        self.python_bin = python_bin or sys.executable

    # -- preflight -------------------------------------------------------

    def blockers(self) -> list[str]:
        """Everything missing, as sentences a person can act on.

        Returned rather than raised, so a readiness endpoint can disable a
        button and say why before the GPU is committed.
        """
        out = []
        for w in WEIGHT_FILES:
            if not (self.weights_dir / w).is_file():
                out.append(f"weight missing at {self.weights_dir / w}")
        if not Path(self.python_bin).is_file():
            out.append(f"interpreter not found at {self.python_bin}")
        return out

    # -- the mask --------------------------------------------------------

    def build_mask(self, *, before: Path, after: Path, out: Path,
                   ffprobe: Optional[str] = None) -> Optional[dict]:
        """Which frames the lip-sync model changed, by diffing.

        Model-agnostic on purpose: MuseTalk only reports untouched frames when
        ASD-gated and nothing persists them, and Wav2Lip has no concept of one.
        Returns the mask document, or None when the diff fails — the caller
        then restores every frame, which is slower and still correct.

        Inlined from the former `scripts/lipsync_diff_mask.py` on 2026-10-30;
        the only caller is this adapter, and there is no interpreter boundary
        left to justify a subprocess.

        The encoder-noise trap: separate H.264 passes make UNTOUCHED frames
        differ by a thin, low-amplitude speckle; a repainted mouth differs by
        a tight, high-amplitude patch. So a pixel counts only above
        `threshold`, and a FRAME only when at least `min_pixels` of them
        cluster. Both thresholds scale with the render's area.
        """
        try:
            doc = _build_change_mask(before, after, out,
                                     log=lambda m: log.info("restore: %s", m))
        except Exception as e:                            # noqa: BLE001
            log.warning("restore: mask build failed (%s); "
                        "restoring every frame instead", e)
            return None
        if doc is None:
            log.warning("restore: mask build returned no document; "
                        "restoring every frame instead")
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
        py = self.python_bin

        mask_doc = None
        mask_path = out.parent / f"{out.stem}.changed_frames.json"
        if request.gated and request.source and Path(request.source).is_file():
            mask_doc = self.build_mask(before=Path(request.source),
                                       after=Path(request.video),
                                       out=mask_path, ffprobe=ffprobe)

        cmd = [py, "-m", "backend.face_enhance",
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
        if request.workers_per_gpu:
            # Omitted rather than defaulted: the script's own default is 1,
            # and passing it would make every run look like a choice.
            cmd += ["--workers-per-gpu", str(request.workers_per_gpu)]
        if request.seek_shards:
            cmd.append("--seek-shards")

        log.info("restore[gfpgan]: %s", " ".join(cmd))
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        dt = time.time() - t0
        tail = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()

        res = RestoreResult(model_name=self.name, out_video=out, wall_s=dt,
                            extra={"mask": str(mask_path) if mask_doc else None,
                                   "changed_frames": (mask_doc or {}).get(
                                       "changed_count"),
                                   "strength": request.strength})
        doc = _parent_json(tail)
        if doc is not None:
            res.frames = int(doc.get("frames") or 0)
            res.restored = int(doc.get("restored") or 0)
            res.passed_through = int(doc.get("passed_through") or 0)
            # The split that actually ran: the script collapses to fewer
            # workers when a shard would fall below MIN_FRAMES_PER_SHARD, so
            # the requested number is not evidence.
            res.extra["workers"] = doc.get("workers")
        else:
            m = _SUMMARY_RE.search(tail)
            if m:
                res.frames, res.restored, res.passed_through = (
                    int(m.group(1)), int(m.group(2)), int(m.group(3)))
                res.extra["workers"] = 1

        # The file is the arbiter: rc=0 with no output, or a file that decodes
        # far fewer frames than claimed, are both failures seen here.
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
