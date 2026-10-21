"""MyShell OpenVoice v2 adapter.

OpenVoice v2 ships two pieces:
  1. A `ToneColorConverter` — ingests a source WAV + a target "tone color"
     and emits a converted WAV.
  2. A speaker-encoder (`se_extractor`) — extracts a tone-color embedding
     from a reference clip.

We expect the OpenVoice v2 clone to live at `third_party/OpenVoice` (same
pattern as TalkNet in the plan), with its model checkpoints under
`third_party/OpenVoice/checkpoints_v2/`.

Env:
  OPENVOICE_REPO        (default: third_party/OpenVoice)
  OPENVOICE_CHECKPOINTS (default: third_party/OpenVoice/checkpoints_v2)

Install path:
  git clone https://github.com/myshell-ai/OpenVoice third_party/OpenVoice
  cd third_party/OpenVoice && pip install -e .
  wget … checkpoints_v2.zip && unzip into checkpoints_v2/
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from .base import BaseVC, VCRequest, VCResult, VCSegment

log = logging.getLogger(__name__)


DEFAULT_REPO = "third_party/OpenVoice"
DEFAULT_CKPTS = "third_party/OpenVoice/checkpoints_v2"


class OpenVoiceVC(BaseVC):
    name = "openvoice"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        language: str = "EN_NEWEST",
    ):
        super().__init__(device=device)
        self.repo_path = Path(repo_path or os.getenv("OPENVOICE_REPO") or DEFAULT_REPO)
        self.ckpt_dir = Path(checkpoint_dir
                              or os.getenv("OPENVOICE_CHECKPOINTS")
                              or DEFAULT_CKPTS)
        self.language = language
        self._converter = None
        self._se_extractor = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.repo_path.exists():
            raise RuntimeError(
                f"OpenVoice repo not found at {self.repo_path}. "
                f"Clone: git clone https://github.com/myshell-ai/OpenVoice "
                f"{self.repo_path}"
            )
        sys.path.insert(0, str(self.repo_path))
        try:
            from openvoice.api import ToneColorConverter  # type: ignore
            from openvoice import se_extractor              # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"OpenVoice modules not importable from {self.repo_path}: {e}. "
                f"Did you `pip install -e {self.repo_path}`?"
            ) from e
        converter_dir = self.ckpt_dir / "converter"
        config = converter_dir / "config.json"
        ckpt = converter_dir / "checkpoint.pth"
        if not config.exists() or not ckpt.exists():
            raise RuntimeError(
                f"OpenVoice v2 converter files missing under {converter_dir}. "
                f"Expected config.json + checkpoint.pth."
            )
        self._converter = ToneColorConverter(str(config), device=self.device)
        self._converter.load_ckpt(str(ckpt))
        self._se_extractor = se_extractor
        self._loaded = True
        log.info("openvoice: loaded converter from %s", converter_dir)

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

        # Pre-extract target tone-color embeddings once per unique reference.
        import torch
        ref_cache: dict = {}
        for req in requests:
            ref = str(req.reference_audio) if req.reference_audio else None
            if ref and ref not in ref_cache and Path(ref).exists():
                try:
                    tgt_se, _ = self._se_extractor.get_se(
                        ref, self._converter,
                        vad=True, target_dir=str(out_dir / "_tmp_se"),
                    )
                    ref_cache[ref] = tgt_se
                except Exception as e:
                    log.warning("openvoice: tone-color for %s failed: %s", ref, e)
                    ref_cache[ref] = None

        # The "source" tone-color embedding is the one baked into the
        # TTS voice — OpenVoice ships pre-computed `base_speakers/<lang>.pth`
        # files for its own TTS, so for TTS output from other engines we
        # extract a per-source embedding on the fly.
        src_cache: dict = {}
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
            tgt_se = ref_cache.get(str(req.reference_audio))
            if tgt_se is None:
                segments.append(VCSegment(
                    id=req.id, audio_path=dest, duration_s=0.0,
                    sample_rate=24000, speaker=req.speaker,
                    error="reference tone-color extraction failed",
                ))
                continue
            src_key = str(req.source_audio)
            if src_key not in src_cache:
                try:
                    src_se, _ = self._se_extractor.get_se(
                        src_key, self._converter,
                        vad=False, target_dir=str(out_dir / "_tmp_se"),
                    )
                    src_cache[src_key] = src_se
                except Exception as e:
                    log.warning("openvoice: source tone-color for %s failed: %s",
                                src_key, e)
                    src_cache[src_key] = None
            src_se = src_cache[src_key]
            if src_se is None:
                segments.append(VCSegment(
                    id=req.id, audio_path=dest, duration_s=0.0,
                    sample_rate=24000, speaker=req.speaker,
                    error="source tone-color extraction failed",
                ))
                continue
            try:
                self._converter.convert(
                    audio_src_path=str(req.source_audio),
                    src_se=src_se, tgt_se=tgt_se,
                    output_path=str(dest), message="@dubbing",
                )
                import soundfile as sf
                info = sf.info(str(dest))
                segments.append(VCSegment(
                    id=req.id, audio_path=dest,
                    duration_s=float(info.duration),
                    sample_rate=int(info.samplerate),
                    speaker=req.speaker,
                ))
            except Exception as e:
                log.warning("openvoice: convert %s failed: %s", req.id, e)
                segments.append(VCSegment(
                    id=req.id, audio_path=dest, duration_s=0.0,
                    sample_rate=24000, speaker=req.speaker, error=str(e),
                ))

        return VCResult(
            model_name="openvoice-v2",
            out_dir=out_dir, segments=segments,
        )
