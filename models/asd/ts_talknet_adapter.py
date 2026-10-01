"""TS-TalkNet adapter (Jiang-Yidi/TS-TalkNet, Interspeech 2023).

Extends TalkNet with a target-speaker enrolment embedding so the model can
disambiguate which face is speaking in a mixed-audio multi-face scene. Needs
a reference waveform per speaker, which the dubbing pipeline already produces
from the diarized dialogue stem.

Falls back to the base TalkNet score when no reference audio is provided —
the architecture degenerates gracefully, same as the upstream repo.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..faces.base import FaceTrack, FaceTracks
from .base import ASDResult, ASDScore, BaseASD
from .talknet_adapter import _crop_track

log = logging.getLogger(__name__)


class TSTalkNetASD(BaseASD):
    name = "ts_talknet"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_path: Optional[Path] = None,
        checkpoint: Optional[Path] = None,
        speaker_encoder_checkpoint: Optional[Path] = None,
        crop_size: int = 112,
    ):
        super().__init__(device=device)
        env_repo = os.getenv("TS_TALKNET_REPO")
        self.repo_path = Path(repo_path or env_repo).expanduser() if (repo_path or env_repo) else None
        env_ckpt = os.getenv("TS_TALKNET_CHECKPOINT")
        self.checkpoint = Path(checkpoint or env_ckpt).expanduser() if (checkpoint or env_ckpt) else None
        env_spk = os.getenv("TS_TALKNET_SPEAKER_ENCODER")
        self.speaker_encoder_checkpoint = (
            Path(speaker_encoder_checkpoint or env_spk).expanduser()
            if (speaker_encoder_checkpoint or env_spk) else None
        )
        self.crop_size = crop_size
        self._model = None
        self._speaker_encoder = None

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.repo_path is None or not self.repo_path.exists():
            raise RuntimeError(
                "TS-TalkNet repo not found. Set TS_TALKNET_REPO to the clone "
                "of https://github.com/Jiang-Yidi/TS-TalkNet."
            )
        if self.checkpoint is None or not self.checkpoint.exists():
            raise RuntimeError(
                "TS-TalkNet checkpoint not found. Set TS_TALKNET_CHECKPOINT "
                "to the pretrained .model from the TS-TalkNet README."
            )
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        import importlib
        import torch
        mod = importlib.import_module(os.getenv("TS_TALKNET_MODEL_MODULE", "talkNet"))
        TalkNet = getattr(mod, os.getenv("TS_TALKNET_MODEL_CLASS", "talkNet"))
        model = TalkNet()
        state = torch.load(str(self.checkpoint), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        clean = {k.removeprefix("model."): v for k, v in state.items()}
        try:
            model.load_state_dict(clean, strict=True)
        except Exception:
            model.load_state_dict(state, strict=False)
        model.eval()
        if self.device.startswith("cuda"):
            model = model.cuda()
        self._model = model

        # ECAPA / ResNet speaker encoder — TS-TalkNet uses it to extract the
        # 192-d target speaker embedding fed into the cross-attention.
        if self.speaker_encoder_checkpoint and self.speaker_encoder_checkpoint.exists():
            try:
                from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
                self._speaker_encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cuda" if self.device.startswith("cuda") else "cpu"},
                )
            except Exception as e:  # noqa: BLE001
                log.warning("ts_talknet: speaker encoder unavailable (%s); "
                            "falling back to repo-internal encoder", e)
                self._speaker_encoder = None
        self._loaded = True

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def score(
        self,
        video_path: Path,
        audio_path: Path,
        tracks: FaceTracks,
        *,
        reference_audio: Optional[Dict[str, Path]] = None,
    ) -> ASDResult:
        self._ensure_loaded()

        import torch
        import cv2
        import soundfile as sf
        from python_speech_features import mfcc as mfcc_fn

        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"TS-TalkNet expects 16 kHz; got {sr}.")

        # Pre-compute target-speaker embeddings for every provided reference.
        ts_embeds: Dict[str, torch.Tensor] = {}
        if reference_audio:
            for spk_id, ref_path in reference_audio.items():
                emb = self._embed_reference(ref_path, torch, sf)
                if emb is not None:
                    ts_embeds[spk_id] = emb

        per_track: Dict[str, ASDScore] = {}
        for tr in tracks.tracks:
            per_track[tr.track_id] = self._score_track(
                tr, video_path, audio, tracks.fps,
                ts_embeds, mfcc_fn, cv2, torch,
            )

        return ASDResult(
            video_path=video_path,
            model_name=self.name,
            fps=tracks.fps,
            per_track=per_track,
            extra={"ts_embeds": sorted(ts_embeds)},
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _embed_reference(self, ref_path: Path, torch, sf):
        if self._speaker_encoder is None:
            return None
        wav, sr = sf.read(str(ref_path), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        emb = self._speaker_encoder.encode_batch(wav_t).squeeze(0).squeeze(0)
        if self.device.startswith("cuda"):
            emb = emb.cuda()
        return emb

    def _score_track(
        self,
        track: FaceTrack,
        video_path: Path,
        audio: np.ndarray,
        fps: float,
        ts_embeds,
        mfcc_fn,
        cv2,
        torch,
    ) -> ASDScore:
        if track.num_frames == 0:
            return ASDScore(track.track_id,
                            np.zeros((0,), dtype=np.int32),
                            np.zeros((0,), dtype=np.float32))

        crops = _crop_track(track, video_path, self.crop_size, cv2)
        n_frames = len(crops)
        samples_per_frame = int(round(16000 / fps))
        s0 = max(0, int(track.frames[0]) * samples_per_frame)
        s1 = min(len(audio), s0 + n_frames * samples_per_frame)
        clip = audio[s0:s1]
        if clip.size == 0:
            return ASDScore(track.track_id,
                            track.frames.copy(),
                            np.zeros((track.num_frames,), dtype=np.float32))

        feats = mfcc_fn(clip, 16000, numcep=13, winlen=0.025, winstep=0.010)
        audio_feat = torch.FloatTensor(np.asarray(feats))
        video_feat = torch.FloatTensor(np.asarray(crops))

        # Best-matching target-speaker embedding: TS-TalkNet is sensitive to
        # which reference is used; if we have N speakers, score the track
        # under each and keep the max per frame. For a single reference the
        # loop collapses to one pass.
        with torch.no_grad():
            if self.device.startswith("cuda"):
                audio_feat = audio_feat.cuda(); video_feat = video_feat.cuda()

            best_scores = None
            embed_list = list(ts_embeds.values()) or [None]
            for ts in embed_list:
                scores = self._forward_once(audio_feat, video_feat, ts, torch)
                if best_scores is None:
                    best_scores = scores
                else:
                    best_scores = np.maximum(best_scores, scores)

        n = min(len(best_scores), track.num_frames)
        return ASDScore(
            track_id=track.track_id,
            frames=track.frames[:n].astype(np.int32),
            scores=best_scores[:n].astype(np.float32),
        )

    def _forward_once(self, audio_feat, video_feat, ts_embed, torch):
        """Run one forward pass. The TS-TalkNet model wires in a 3rd arg to
        `forward_cross_attention` (`ts_embed`); the base TalkNet wrapper
        does not. We detect the signature at call time."""
        m = self._model.model
        audio_embed = m.forward_audio_frontend(audio_feat.unsqueeze(0))
        video_embed = m.forward_visual_frontend(video_feat.unsqueeze(0))
        import inspect
        sig = inspect.signature(m.forward_cross_attention)
        if ts_embed is not None and len(sig.parameters) >= 3:
            audio_embed, video_embed = m.forward_cross_attention(
                audio_embed, video_embed, ts_embed.unsqueeze(0),
            )
        else:
            audio_embed, video_embed = m.forward_cross_attention(audio_embed, video_embed)
        out = m.forward_audio_visual_backend(audio_embed, video_embed)
        logits = self._model.lossAV.FC(out)
        return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
