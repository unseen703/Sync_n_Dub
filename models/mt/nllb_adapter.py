"""NLLB-200 adapter (local, HuggingFace transformers).

Model: facebook/nllb-200-distilled-600M by default. Any NLLB checkpoint
exposing the same `AutoTokenizer` / `AutoModelForSeq2SeqLM` interface works
(e.g. `facebook/nllb-200-distilled-1.3B`, `facebook/nllb-200-3.3B`).

NLLB uses FLORES-200 language codes ("eng_Latn", "hin_Deva", …) rather
than ISO-639. We accept either and translate to FLORES internally based
on the requested `target_language` + `target_script` ("native" | "roman").
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

from backend.sanitize import sanitize_mt_output

from .base import (
    DEFAULT_DIVERSITY_PENALTY,
    DEFAULT_LENGTH_PENALTIES,
    BaseMT,
    MTRequest,
    MTResult,
    MTSegment,
    dedupe_candidates,
    group_beam_remote_kwargs,
    plain_beam_fallback,
    plan_beam_groups,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "facebook/nllb-200-distilled-600M"

# ISO-639 (+ optional script) → FLORES-200. "roman" is a convenience alias
# the pipeline passes through from `--mt-target-script`: for Indic scripts
# we route to the matching Latin variant NLLB ships, when one exists.
_ISO_TO_FLORES_NATIVE = {
    "en": "eng_Latn", "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu",
    "bn": "ben_Beng", "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda",
    "ml": "mal_Mlym", "pa": "pan_Guru", "ur": "urd_Arab", "or": "ory_Orya",
    "as": "asm_Beng", "ne": "npi_Deva", "si": "sin_Sinh",
    "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "it": "ita_Latn",
    "pt": "por_Latn", "ru": "rus_Cyrl", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "zh": "zho_Hans", "ar": "arb_Arab",
}
_ISO_TO_FLORES_ROMAN = {
    "hi": "hin_Latn", "ur": "urd_Latn",
}


def _to_flores(lang: str, script: Optional[str]) -> str:
    """Return a FLORES-200 code; accept FLORES codes verbatim."""
    if "_" in lang and len(lang) >= 7:
        return lang
    key = lang.lower()
    if script == "roman" and key in _ISO_TO_FLORES_ROMAN:
        return _ISO_TO_FLORES_ROMAN[key]
    if key in _ISO_TO_FLORES_NATIVE:
        return _ISO_TO_FLORES_NATIVE[key]
    raise ValueError(
        f"NLLB adapter: no FLORES mapping for lang={lang!r} script={script!r}. "
        f"Pass a FLORES code directly (e.g. 'hin_Deva')."
    )


class NLLBMT(BaseMT):
    name = "nllb"
    supports_nbest = True

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        model_id: Optional[str] = None,
        max_new_tokens: int = 512,
        num_beams: int = 4,
        batch_size: int = 8,
        dtype: Optional[str] = None,     # "float16" | "bfloat16" | "float32"
    ):
        super().__init__(device=device)
        self.model_id = model_id or os.getenv("NLLB_MODEL_ID") or DEFAULT_MODEL_ID
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)
        self.batch_size = int(batch_size)
        self.dtype = dtype or os.getenv("NLLB_DTYPE")
        self._tok = None
        self._model = None

    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "`pip install transformers sentencepiece` for the NLLB adapter."
            ) from e

        log.info("nllb: loading tokenizer+model %s on %s", self.model_id, self.device)
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        torch_dtype = None
        if self.dtype == "float16":
            torch_dtype = torch.float16
        elif self.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.dtype == "float32":
            torch_dtype = torch.float32
        elif self.device.startswith("cuda"):
            torch_dtype = torch.float16
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, torch_dtype=torch_dtype,
        ).to(self.device)
        self._model.eval()
        self._loaded = True

    # ------------------------------------------------------------------

    def translate(
        self,
        requests: List[MTRequest],
        *,
        target_language: str,
        target_script: Optional[str] = None,
        source_language: Optional[str] = None,
    ) -> MTResult:
        self._ensure_loaded()
        import torch

        tgt_flores = _to_flores(target_language, target_script)
        # NLLB needs the forced BOS to be the target lang token.
        tgt_id = self._tok.convert_tokens_to_ids(tgt_flores)

        segments: List[MTSegment] = [None] * len(requests)         # type: ignore[list-item]
        # Group empties out so we don't pay the model-call tax on them.
        work: List[tuple[int, MTRequest, str]] = []
        for i, req in enumerate(requests):
            text = (req.text or "").strip()
            if not text:
                segments[i] = MTSegment(
                    id=req.id, source_text=text, text="",
                    source_language=req.source_language or source_language,
                    target_language=target_language,
                    target_script=req.target_script or target_script,
                )
                continue
            src = req.source_language or source_language or "en"
            src_flores = _to_flores(src, None)
            work.append((i, req, src_flores))

        # NLLB's tokenizer expects `src_lang` to be set per encode call.
        for b_start in range(0, len(work), self.batch_size):
            batch = work[b_start:b_start + self.batch_size]
            # All items in a batch must share src_lang for a single tokenizer
            # call; split by src_lang if the caller mixed source languages.
            by_src: dict[str, list[tuple[int, MTRequest]]] = {}
            for i, req, src in batch:
                by_src.setdefault(src, []).append((i, req))
            for src, items in by_src.items():
                self._tok.src_lang = src
                texts = [r.text.strip() for _, r in items]
                enc = self._tok(
                    texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_new_tokens,
                ).to(self.device)
                with torch.no_grad():
                    out = self._model.generate(
                        **enc,
                        forced_bos_token_id=tgt_id,
                        max_new_tokens=self.max_new_tokens,
                        num_beams=self.num_beams,
                    )
                decoded = self._tok.batch_decode(out, skip_special_tokens=True)
                for (i, req), text in zip(items, decoded):
                    segments[i] = MTSegment(
                        id=req.id,
                        source_text=req.text.strip(),
                        text=sanitize_mt_output(text.strip(), seg_id=req.id),
                        source_language=req.source_language or source_language,
                        target_language=target_language,
                        target_script=req.target_script or target_script,
                    )

        return MTResult(
            model_name=self.model_id,
            source_language=source_language,
            target_language=target_language,
            target_script=target_script,
            segments=segments,
            extra={"flores_target": tgt_flores,
                   "num_beams": self.num_beams,
                   "batch_size": self.batch_size},
        )

    # ------------------------------------------------------------------
    # n-best
    # ------------------------------------------------------------------

    def translate_nbest(
        self,
        requests: List[MTRequest],
        *,
        target_language: str,
        target_script: Optional[str] = None,
        source_language: Optional[str] = None,
        n_best: int = 4,
        length_penalties: Optional[Sequence[float]] = None,
        diversity_penalty: Optional[float] = None,
    ) -> MTResult:
        """Diverse beam search + a length-penalty sweep, deduped per segment.

        Two knobs, because one is not enough. Diverse beam search (one beam per
        group, Hamming penalty across groups) buys *lexical* variety; the
        length-penalty sweep buys *length* variety, which is the axis
        `backend.duration.rank_candidates` actually scores on. Plain
        `num_return_sequences=K` buys neither -- the K hypotheses come off one
        beam tree and differ by a token.

        The top-1 is produced by a separate plain-beam pass with exactly the
        kwargs `translate()` uses, so `MTSegment.text` is unchanged whether or
        not n-best is on.
        """
        if int(n_best) <= 1:
            return super().translate_nbest(
                requests, target_language=target_language,
                target_script=target_script, source_language=source_language,
                n_best=1,
            )

        self._ensure_loaded()
        import torch

        lps = tuple(float(x) for x in (length_penalties or DEFAULT_LENGTH_PENALTIES))
        div = DEFAULT_DIVERSITY_PENALTY if diversity_penalty is None else float(diversity_penalty)
        beams, groups, n_ret = plan_beam_groups(int(n_best), self.num_beams)
        remote_group = group_beam_remote_kwargs()

        tgt_flores = _to_flores(target_language, target_script)
        tgt_id = self._tok.convert_tokens_to_ids(tgt_flores)

        segments: List[MTSegment] = [None] * len(requests)   # type: ignore[list-item]
        # index -> candidate strings in priority order (anchor first)
        pool: Dict[int, List[str]] = {}
        work: List[tuple[int, MTRequest, str]] = []
        for i, req in enumerate(requests):
            text = (req.text or "").strip()
            if not text:
                segments[i] = MTSegment(
                    id=req.id, source_text=text, text="",
                    source_language=req.source_language or source_language,
                    target_language=target_language,
                    target_script=req.target_script or target_script,
                    extra={"candidates": []},
                )
                continue
            src = req.source_language or source_language or "en"
            work.append((i, req, _to_flores(src, None)))
            pool[i] = []

        failures: List[str] = []
        degraded: List[str] = []

        for b_start in range(0, len(work), self.batch_size):
            batch = work[b_start:b_start + self.batch_size]
            by_src: Dict[str, List[tuple[int, MTRequest]]] = {}
            for i, req, src in batch:
                by_src.setdefault(src, []).append((i, req))
            for src, items in by_src.items():
                self._tok.src_lang = src
                texts = [r.text.strip() for _, r in items]
                enc = self._tok(
                    texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_new_tokens,
                ).to(self.device)

                # Pass 0: the anchor. Same kwargs as translate(), so the
                # resulting top-1 is identical to the single-hypothesis path.
                passes = [("anchor", dict(num_beams=self.num_beams))]
                for lp in lps:
                    passes.append((f"lp={lp:g}", dict(
                        num_beams=beams,
                        num_beam_groups=groups,
                        diversity_penalty=div,
                        length_penalty=lp,
                        num_return_sequences=n_ret,
                        do_sample=False,          # diverse beam forbids sampling
                        **remote_group,
                    )))

                for label, gen_kwargs in passes:
                    # transformers >= 4.57 refuses `num_beam_groups=` unless
                    # the caller opts into the hub `custom_generate` repo, so
                    # "skip the failed pass" would skip EVERY sweep pass and
                    # leave the anchor as the only candidate. Retry the same
                    # length_penalty as plain beam before giving up.
                    attempts = [(label, gen_kwargs)]
                    fallback = plain_beam_fallback(gen_kwargs)
                    if fallback != gen_kwargs:
                        attempts.append((f"{label}/plain-beam", fallback))
                    decoded = None
                    for attempt_label, kwargs in attempts:
                        try:
                            with torch.no_grad():
                                out = self._model.generate(
                                    **enc,
                                    forced_bos_token_id=tgt_id,
                                    max_new_tokens=self.max_new_tokens,
                                    **kwargs,
                                )
                            decoded = self._tok.batch_decode(out, skip_special_tokens=True)
                        except Exception as e:
                            # One bad decode pass must not cost us the whole
                            # batch; the anchor pass alone still yields a top-1.
                            log.warning("nllb n-best: pass %s failed (%s)",
                                        attempt_label, e)
                            failures.append(f"{attempt_label}: {e}")
                            continue
                        if attempt_label != label:
                            degraded.append(attempt_label)
                        break
                    if decoded is None:
                        continue
                    per = max(1, len(decoded) // max(1, len(items)))
                    for j, (i, _req) in enumerate(items):
                        pool[i].extend(
                            sanitize_mt_output(t.strip(), seg_id=_req.id)
                            for t in decoded[j * per:(j + 1) * per]
                        )

        for i, req, _src in work:
            cands = dedupe_candidates(pool.get(i, []), limit=int(n_best))
            segments[i] = MTSegment(
                id=req.id,
                source_text=req.text.strip(),
                text=cands[0] if cands else "",
                source_language=req.source_language or source_language,
                target_language=target_language,
                target_script=req.target_script or target_script,
                extra={"candidates": cands},
            )

        return MTResult(
            model_name=self.model_id,
            source_language=source_language,
            target_language=target_language,
            target_script=target_script,
            segments=segments,
            extra={"flores_target": tgt_flores,
                   "num_beams": beams,
                   "num_beam_groups": groups,
                   "diversity_penalty": div,
                   "length_penalties": list(lps),
                   "n_best_requested": int(n_best),
                   "n_best_supported": True,
                   "nbest_errors": failures,
                   "nbest_degraded_passes": degraded,
                   "diverse_beam_used": not degraded,
                   "batch_size": self.batch_size},
        )
