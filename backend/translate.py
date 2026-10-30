"""Backend-facing machine-translation entry point.

Takes an `ASRResult` and a target language and returns one translated segment
per input segment, written to a merged JSON beside the ASR output so later
stages do not re-translate.
"""
from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from models.asr import ASRResult
from models.mt import MTRequest, MTResult, available_models, get_mt

from .sanitize import sanitize_for_mt

log = logging.getLogger(__name__)


def _filter_to_signature(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the kwargs `fn` cannot accept, instead of letting it raise.

    The generation controls are translate-time and not every adapter takes
    them, so filtering here keeps an adapter that never heard of `n_best`
    working rather than dying on a TypeError.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):          # builtins / C callables
        return {}
    params = sig.parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return dict(kwargs)
    accepted = {
        p.name for p in params
        if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }
    dropped = sorted(set(kwargs) - accepted)
    if dropped:
        log.info("translate: %s does not accept %s; dropping",
                 getattr(fn, "__qualname__", fn), dropped)
    return {k: v for k, v in kwargs.items() if k in accepted}


def translate_asr(
    asr_result: ASRResult,
    *,
    target_language: str,
    model: str = "bodhan",
    target_script: Optional[str] = None,
    source_language: Optional[str] = None,
    device: Optional[str] = None,
    n_best: int = 1,
    length_penalties: Optional[Sequence[float]] = None,
    diversity_penalty: Optional[float] = None,
    **model_kwargs,
) -> MTResult:
    """Translate every segment of `asr_result` into `target_language`.

    `n_best > 1` routes through `BaseMT.translate_nbest` and records the
    alternatives in `MTSegment.extra["candidates"]`, which
    `backend.duration.plan_segment_durations` later ranks. The contract
    guarantees `candidates[0] == text`, so the top-1 is unchanged.

    `n_best <= 1` makes the plain `mt.translate(...)` call with no generation
    controls at all.
    """
    if model not in available_models():
        raise ValueError(
            f"Unknown MT model {model!r}. "
            f"Available: {', '.join(available_models())}"
        )

    n_best = max(1, int(n_best))

    requests: List[MTRequest] = [
        MTRequest(
            id=seg.id,
            text=sanitize_for_mt((seg.text or "").strip()),
            source_language=seg.language or source_language or asr_result.language,
            target_script=target_script,
            n_best=n_best,
        )
        for seg in asr_result.segments
    ]

    log.info("translate: %d segments → %s (model=%s, script=%s, n_best=%d)",
             len(requests), target_language, model, target_script or "default",
             n_best)
    mt = get_mt(model, device=device, **model_kwargs)

    # The generation controls are filtered against the chosen callable's
    # signature, so an adapter that does not take them is unaffected.
    call = mt.translate if n_best <= 1 else getattr(mt, "translate_nbest", mt.translate)
    gen_kwargs = _filter_to_signature(call, {
        "n_best": n_best,
        "length_penalties": length_penalties,
        "diversity_penalty": diversity_penalty,
    }) if n_best > 1 else {}

    result = call(
        requests,
        target_language=target_language,
        target_script=target_script,
        source_language=source_language or asr_result.language,
        **gen_kwargs,
    )
    non_empty = sum(1 for s in result.segments if s.text)
    log.info("translate: %d/%d segments produced non-empty translations",
             non_empty, len(result.segments))
    if n_best > 1:
        counts = [len((s.extra or {}).get("candidates") or []) for s in result.segments]
        multi = sum(1 for c in counts if c > 1)
        log.info("translate: n_best=%d → %d/%d segments have >1 candidate "
                 "(max %d)", n_best, multi, len(counts), max(counts or [0]))
    return result


def mt_result_to_jsonable_merged(asr_result: ASRResult,
                                  mt_result: MTResult) -> dict:
    """Return an ASR+MT merged JSON dict (what gets written to disk)."""
    by_id = {s.id: s for s in mt_result.segments}
    payload = asr_result.to_jsonable()
    for seg in payload["segments"]:
        m = by_id.get(seg["id"])
        if m is None:
            continue
        seg["mt_text"] = m.text
        seg["mt_target_language"] = m.target_language
        seg["mt_target_script"] = m.target_script
        # The key `plan_segment_durations` reads. Without it `pool` is always
        # `[base]` and ranking is silently a no-op.
        cands = (m.extra or {}).get("candidates") or []
        if len(cands) > 1:
            seg["candidates_text"] = list(cands)
    payload["extra"] = {
        **payload.get("extra", {}),
        "mt_model": mt_result.model_name,
        "mt_target_language": mt_result.target_language,
        "mt_target_script": mt_result.target_script,
    }
    return payload


def write_mt_outputs(out_json: Path, out_txt: Path,
                     asr_result: ASRResult, mt_result: MTResult) -> None:
    """Write the merged ASR+MT JSON and a plain-text translated transcript."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        mt_result_to_jsonable_merged(asr_result, mt_result), indent=2,
        ensure_ascii=False,
    ))
    out_txt.write_text(
        "\n".join(s.text.strip() for s in mt_result.segments if s.text) + "\n",
        encoding="utf-8",
    )
