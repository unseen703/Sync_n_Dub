"""CodeFormer adapter (sczhou/CodeFormer, NeurIPS 2022).

Blind face restoration, used here as the post-process on a lip-synced
render — the arrangement of langzizhixin/Wav2Lip-CodeFormer, which runs
Wav2Lip to move the mouth and then CodeFormer to rebuild the face it
left blurry. CodeFormer is the better half of that pairing: it restores
through a learned discrete codebook, so it reconstructs a plausible
face rather than sharpening the one it was given.

We shell out to upstream's `inference_codeformer.py` rather than
importing it, for the same reason the lip-sync adapters do: the repo
vendors its own trimmed `basicsr`, and putting that on this process's
`sys.path` would shadow nothing useful and risk clashing with the rest
of the pipeline.

Two upstream behaviours are deliberately **not** used:

1. **Its video mode.** `--input_path foo.mp4` routes through
   `basicsr/utils/video_util.py`, which runs ffmpeg as the bare name
   `cmd='ffmpeg'` and probes with `ffmpeg.probe`. There is no ffmpeg on
   this host's PATH (CLAUDE.md §1), and `VideoReader.__init__` answers a
   `FileNotFoundError` with `sys.exit(0)` — a *zero* exit code and no
   output file. So we cut and reassemble frames ourselves with
   `backend.media_io._ffmpeg_bin()` and feed the folder instead.
2. **`--bg_upsampler realesrgan`.** Upstream's notebook turns it on, but
   it upsamples the whole frame, pulls in the Real-ESRGAN weights, and
   costs VRAM we do not have spare. Left off by default, so at
   `upscale=1` the only pixels that change are the detected faces and
   the output keeps the input's resolution. Pass `bg_upsampler="realesrgan"`
   to opt in.

`fidelity` (upstream `-w`) trades identity against quality: 0 is the
best-looking face, 1 is the most faithful to the input. Upstream
recommends 0.5-0.7 for real degraded faces; the Wav2Lip-CodeFormer
notebook uses 0.7, which is the default here.

Env:
  CODEFORMER_REPO    (default: third_party/CodeFormer)
  CODEFORMER_PYTHON  (default: current python; set if CodeFormer lives
                      in a separate venv)

Install:
  git clone https://github.com/sczhou/CodeFormer third_party/CodeFormer
  # weights/CodeFormer/codeformer.pth
  # weights/facelib/{detection_Resnet50_Final,parsing_parsenet}.pth
  # from https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .base import BaseRestore, RestoreRequest, RestoreResult

log = logging.getLogger(__name__)


DEFAULT_REPO = "third_party/CodeFormer"
# "	detect 3 faces" / "	detect 1 faces" — upstream prints this per frame.
_DETECT_RE = re.compile(r"detect\s+(\d+)\s+faces")


class CodeFormerRestore(BaseRestore):
    name = "codeformer"
    licence = "non-commercial"      # S-Lab 1.0: research / non-commercial only

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_path: Optional[str] = None,
        python_bin: Optional[str] = None,
        fidelity: float = 0.7,
        upscale: int = 1,
        detection_model: str = "retinaface_resnet50",
        only_center_face: bool = False,
        face_upsample: bool = False,
        bg_upsampler: str = "None",
        keep_frames: bool = False,
        cuda_device: Optional[int] = None,
    ):
        super().__init__(device=device, cuda_device=cuda_device)
        self.repo_path = Path(repo_path or os.getenv("CODEFORMER_REPO") or DEFAULT_REPO)
        self.python_bin = python_bin or os.getenv("CODEFORMER_PYTHON") or sys.executable
        self.fidelity = float(fidelity)
        self.upscale = int(upscale)
        self.detection_model = detection_model
        self.only_center_face = only_center_face
        self.face_upsample = face_upsample
        self.bg_upsampler = bg_upsampler
        self.keep_frames = keep_frames

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.repo_path.exists():
            raise RuntimeError(
                f"CodeFormer repo not found at {self.repo_path}. "
                f"Clone: git clone https://github.com/sczhou/CodeFormer "
                f"{self.repo_path}"
            )
        inference = self.repo_path / "inference_codeformer.py"
        if not inference.exists():
            raise RuntimeError(f"CodeFormer inference_codeformer.py missing at {inference}")
        # The vendored basicsr only imports once `version.py` exists; upstream
        # generates it in `basicsr/setup.py`, which we deliberately do not run
        # (it would install a second basicsr into the dubbing env).
        version = self.repo_path / "basicsr" / "version.py"
        if not version.exists():
            raise RuntimeError(
                f"{version} missing — the vendored basicsr cannot import. "
                f"Write it from {self.repo_path / 'basicsr' / 'VERSION'} "
                f"(see the file's header for the two lines it needs)."
            )
        for ckpt in (
            self.repo_path / "weights" / "CodeFormer" / "codeformer.pth",
            self.repo_path / "weights" / "facelib" / "detection_Resnet50_Final.pth",
            self.repo_path / "weights" / "facelib" / "parsing_parsenet.pth",
        ):
            if not ckpt.exists():
                raise RuntimeError(
                    f"CodeFormer checkpoint missing at {ckpt}. Download from "
                    f"https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0"
                )
        self._loaded = True

    def restore(self, request: RestoreRequest) -> RestoreResult:
        self._ensure_loaded()
        from backend.media_io import _ffmpeg_bin

        src = Path(request.video).resolve()
        out = Path(request.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Resolve before deriving anything from it. The subprocess below runs
        # with cwd=repo_path, so a relative --input_path/--output_path would
        # be interpreted against third_party/CodeFormer and silently find no
        # frames. Callers pass relative out_dirs routinely (`--out-dir
        # data/jobs/<id>`), so this is the normal case, not an edge one.
        out = out.resolve()
        ff = _ffmpeg_bin()
        fps, _dur = _probe_video(src)
        if not fps:
            return self._fail(out, f"could not probe fps of {src}")

        work = Path(tempfile.mkdtemp(prefix="codeformer_", dir=str(out.parent)))
        frames_dir = work / "frames"
        frames_dir.mkdir()
        try:
            # 1. decode to PNG. Lossless, so the only quality change in the
            #    whole stage is the one CodeFormer makes.
            rc = subprocess.run(
                [ff, "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(src), "-start_number", "0",
                 str(frames_dir / "%06d.png")],
                capture_output=True, text=True,
            )
            if rc.returncode != 0:
                return self._fail(out, f"frame extract rc={rc.returncode}: "
                                       f"{rc.stderr.strip()[-2000:]}")
            n_frames = len(list(frames_dir.glob("*.png")))
            if not n_frames:
                return self._fail(out, f"no frames decoded from {src}")

            # 2. restore. cwd must be the repo: upstream resolves `weights/`
            #    relative to the working directory, and PYTHONPATH must carry
            #    the repo root so the vendored basicsr is importable.
            cf_out = work / "restored"
            cmd = [
                self.python_bin, "inference_codeformer.py",
                "--input_path", str(frames_dir),
                "--output_path", str(cf_out),
                "--fidelity_weight", str(self.fidelity),
                "--upscale", str(self.upscale),
                "--detection_model", self.detection_model,
                "--bg_upsampler", self.bg_upsampler,
            ]
            if self.only_center_face:
                cmd.append("--only_center_face")
            if self.face_upsample:
                cmd.append("--face_upsample")

            env = self._subprocess_env()
            env["PYTHONPATH"] = (
                str(self.repo_path.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
            ).rstrip(os.pathsep)

            log.info("codeformer[gpu=%s, w=%.2f, s=%d]: %d frames from %s",
                     self.cuda_device, self.fidelity, self.upscale, n_frames, src.name)
            proc = subprocess.run(cmd, cwd=str(self.repo_path), env=env,
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                return self._fail(out, f"codeformer rc={proc.returncode}: "
                                       f"{proc.stderr.strip()[-3000:]}")

            # Upstream prints one "detect N faces" per frame it found a face
            # in, and prints nothing for the frames it did not. A run that
            # restores nothing still exits 0, so count rather than trust rc.
            hits = _DETECT_RE.findall(proc.stdout)
            frames_with_faces = len(hits)
            faces_total = sum(int(h) for h in hits)

            final = cf_out / "final_results"
            n_out = len(list(final.glob("*.png"))) if final.exists() else 0
            if n_out != n_frames:
                return self._fail(
                    out,
                    f"codeformer wrote {n_out} frames for {n_frames} input "
                    f"frames — refusing to assemble a video with holes in it",
                )

            # 3. reassemble, muxing the audio back. Written as 48 kHz stereo
            #    AAC for the same reason backend.lipsync._remux_audio does:
            #    Wav2Lip's native 16 kHz mono passes ffprobe and is then
            #    silently refused by several players.
            audio_src = Path(request.audio).resolve() if request.audio else src
            cmd = [
                ff, "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", f"{fps}", "-start_number", "0",
                "-i", str(final / "%06d.png"),
                "-i", str(audio_src),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
                "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
                "-shortest", str(out.resolve()),
            ]
            rc = subprocess.run(cmd, capture_output=True, text=True)
            if rc.returncode != 0:
                return self._fail(out, f"reassemble rc={rc.returncode}: "
                                       f"{rc.stderr.strip()[-2000:]}")

            out_fps, out_dur = _probe_video(out)
            return RestoreResult(
                model_name=self.name, out_video=out,
                duration_s=out_dur, fps=out_fps,
                frames=n_frames, frames_with_faces=frames_with_faces,
                extra={
                    "licence": self.licence,
                    "fidelity": self.fidelity,
                    "upscale": self.upscale,
                    "detection_model": self.detection_model,
                    "bg_upsampler": self.bg_upsampler,
                    "face_upsample": self.face_upsample,
                    "only_center_face": self.only_center_face,
                    "faces_detected_total": faces_total,
                    "audio_from": str(audio_src),
                    "work_dir": str(work) if self.keep_frames else None,
                },
            )
        finally:
            if not self.keep_frames:
                shutil.rmtree(work, ignore_errors=True)

    def _fail(self, out: Path, msg: str) -> RestoreResult:
        log.error("codeformer: %s", msg)
        return RestoreResult(model_name=self.name, out_video=out,
                             duration_s=0.0, fps=0.0, error=msg)


def _probe_video(path: Path) -> tuple[float, float]:
    """Return (fps, duration_s) using ffprobe when available, else zeros."""
    from backend.media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe or not Path(path).exists():
        return (0.0, 0.0)
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
        fps = 0.0
        dur = 0.0
        if lines:
            rate = lines[0]
            if "/" in rate:
                n, d = rate.split("/", 1)
                fps = float(n) / float(d) if float(d) else 0.0
            else:
                fps = float(rate)
        if len(lines) > 1:
            dur = float(lines[1])
        return (fps, dur)
    except Exception:
        return (0.0, 0.0)
