"""emotion2vec+ speech-emotion adapter, via FunASR.

Env:
  EMOTION2VEC_MODEL   model id (default: iic/emotion2vec_plus_large)
  EMOTION2VEC_HUB     "hf" (default) or "ms" for ModelScope

Needs `pip install funasr`; weights download on first use.

Two traps. The published label set is inconsistent (the HF card lists 5
classes, the README 9), so no vocabulary is hardcoded — labels are read at
runtime and reported in `EmotionBatch.labels`. And FunASR returns labels as
`<zh>/<en>`, so `_norm_label` keeps the English side.

The model is utterance-level: `backend.emotion` cuts per-segment clips first,
since a whole lecture returns one useless label.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from .base import BaseEmotion, EmotionBatch, EmotionResult

log = logging.getLogger(__name__)

DEFAULT_MODEL = "iic/emotion2vec_plus_large"


def _norm_label(raw: str) -> str:
    """`'angry'` -> `'angry'`; `'<unk>/unknown'` -> `'unknown'`."""
    s = (raw or "").strip()
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    return s.strip().lower().replace("<", "").replace(">", "")


class Emotion2VecEmotion(BaseEmotion):
    name = "emotion2vec"
    sample_rate = 16_000
    coverage = ("self-supervised, evaluated on 10 languages in the paper; "
                "no published Indic numbers. Read from SOURCE audio, so "
                "this only constrains Indic-source material.")

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        model: Optional[str] = None,
        hub: Optional[str] = None,
        disable_update: bool = True,
    ):
        super().__init__(device=device)
        self.model_id = model or os.getenv("EMOTION2VEC_MODEL") or DEFAULT_MODEL
        self.hub = (hub or os.getenv("EMOTION2VEC_HUB") or "hf").lower()
        self.disable_update = disable_update
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise ImportError(
                "emotion2vec needs FunASR. `pip install funasr` into the env"
            ) from e
        kwargs = {"model": self.model_id, "hub": self.hub}
        if self.device:
            kwargs["device"] = self.device
        if self.disable_update:
            # FunASR otherwise phones home on every construction and blocks
            # for a long time on a slow link.
            kwargs["disable_update"] = True
        log.info("emotion2vec: loading %s (hub=%s, device=%s)",
                 self.model_id, self.hub, self.device or "auto")
        self._model = AutoModel(**kwargs)

    def score(self, clips: Dict[str, Path]) -> EmotionBatch:
        self._ensure_loaded()
        results = []
        vocab: list[str] = []

        for clip_id, path in clips.items():
            path = Path(path)
            if not path.exists():
                results.append(EmotionResult(clip_id, "unknown", 0.0,
                                             error=f"missing {path}"))
                continue
            try:
                out = self._model.generate(
                    str(path),
                    granularity="utterance",
                    extract_embedding=False,
                )
                rec = out[0] if isinstance(out, list) and out else {}
                labels = [_norm_label(x) for x in (rec.get("labels") or [])]
                scores = [float(x) for x in (rec.get("scores") or [])]
                if not labels or not scores:
                    results.append(EmotionResult(clip_id, "unknown", 0.0,
                                                 error="no labels in response"))
                    continue
                if not vocab:
                    vocab = labels
                dist = dict(zip(labels, scores))
                best = max(dist, key=dist.get)
                results.append(EmotionResult(
                    clip_id, best, float(dist[best]), scores=dist,
                ))
            except Exception as e:                   # noqa: BLE001
                # One bad clip must not lose the other predictions.
                log.warning("emotion2vec: %s failed: %s", clip_id, e)
                results.append(EmotionResult(clip_id, "unknown", 0.0,
                                             error=str(e)))

        return EmotionBatch(
            model_name=f"{self.name}:{self.model_id}",
            results=results,
            labels=vocab,
            extra={"hub": self.hub, "coverage": self.coverage},
        )
