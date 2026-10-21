"""Open-MMLab Amphion Vevo adapter.

Vevo is the unified controllable VC/TTS stack from the Amphion project.
It exposes a `VevoInferencePipeline` that takes (`content_wav`, `style_wav`,
`timbre_wav`) — for simple VC we pass the TTS output as all of
content+style and the speaker reference as timbre.

The Amphion repo is expected at `third_party/Amphion` (plan §11), with
Vevo checkpoints under `third_party/Amphion/pretrained/Vevo/`.

Env:
  AMPHION_REPO  (default: third_party/Amphion)
  VEVO_CHECKPOINTS (default: third_party/Amphion/pretrained/Vevo)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from .base import BaseVC, VCRequest, VCResult, VCSegment

log = logging.getLogger(__name__)


DEFAULT_REPO = "third_party/Amphion"
DEFAULT_CKPTS = "third_party/Amphion/pretrained/Vevo"


class VevoVC(BaseVC):
    name = "vevo"
    licence = "CC-BY-NC-4.0"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        task: str = "vc",            # "vc" | "style" | "timbre"
    ):
        super().__init__(device=device)
        self.repo_path = Path(repo_path or os.getenv("AMPHION_REPO") or DEFAULT_REPO)
        self.ckpt_dir = Path(checkpoint_dir
                              or os.getenv("VEVO_CHECKPOINTS")
                              or DEFAULT_CKPTS)
        self.task = task
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.repo_path.exists():
            raise RuntimeError(
                f"Amphion repo not found at {self.repo_path}. "
                f"Clone: git clone https://github.com/open-mmlab/Amphion {self.repo_path}"
            )
        sys.path.insert(0, str(self.repo_path))
        try:
            # The exact import path has shifted across Amphion revisions —
            # try the two most common.
            try:
                from models.vc.vevo.vevo_utils import (  # type: ignore
                    VevoInferencePipeline,
                )
            except ImportError:
                from models.svc.vevo.vevo_utils import VevoInferencePipeline  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"Amphion Vevo modules not importable from {self.repo_path}: {e}. "
                f"Follow Amphion's install guide; Vevo lives under models/vc/vevo."
            ) from e
        self._pipeline = VevoInferencePipeline(
            content_tokenizer_ckpt_path=str(self.ckpt_dir / "contentvec_tokenizer"),
            content_style_tokenizer_ckpt_path=str(self.ckpt_dir / "style_tokenizer"),
            ar_cfg_path=str(self.ckpt_dir / "config_ar.json"),
            ar_ckpt_path=str(self.ckpt_dir / "ar.ckpt"),
            fmt_cfg_path=str(self.ckpt_dir / "config_fmt.json"),
            fmt_ckpt_path=str(self.ckpt_dir / "fmt.ckpt"),
            vocoder_cfg_path=str(self.ckpt_dir / "config_vocoder.json"),
            vocoder_ckpt_path=str(self.ckpt_dir / "vocoder.ckpt"),
            device=self.device,
        )
        self._loaded = True
        log.info("vevo: loaded pipeline from %s", self.ckpt_dir)

    # ------------------------------------------------------------------

    def convert(
        self,
        requests: List[VCRequest],
        *,
        out_dir: Path,
        language: str,
    ) -> VCResult:
        self._ensure_loaded()
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        segments: List[VCSegment] = []
        for req in requests:
            dest = out_dir / f"{req.id}.wav"
            if not Path(req.source_audio).exists():
                segments.append(VCSegment(
                    id=req.id, audio_path=dest, duration_s=0.0,
                    sample_rate=24000, speaker=req.speaker,
                    error="source audio missing",
                ))
                continue
            if not req.reference_audio or not Path(req.reference_audio).exists():
                segments.append(VCSegment(
                    id=req.id, audio_path=dest, duration_s=0.0,
                    sample_rate=24000, speaker=req.speaker,
                    error="reference audio missing",
                ))
                continue
            try:
                audio = self._pipeline.inference_fm(
                    src_wav_path=str(req.source_audio),
                    src_text=None,
                    style_ref_wav_path=str(req.source_audio),
                    timbre_ref_wav_path=str(req.reference_audio),
                    flow_matching_steps=32,
                )
                sr = getattr(self._pipeline, "sample_rate", 24000)
                sf.write(str(dest), audio, sr)
                info = sf.info(str(dest))
                segments.append(VCSegment(
                    id=req.id, audio_path=dest,
                    duration_s=float(info.duration),
                    sample_rate=int(info.samplerate),
                    speaker=req.speaker,
                ))
            except Exception as e:
                log.warning("vevo: convert %s failed: %s", req.id, e)
                segments.append(VCSegment(
                    id=req.id, audio_path=dest, duration_s=0.0,
                    sample_rate=24000, speaker=req.speaker, error=str(e),
                ))

        return VCResult(
            model_name=f"vevo/{self.task}",
            out_dir=out_dir, segments=segments,
            extra={"licence": self.licence},
        )
