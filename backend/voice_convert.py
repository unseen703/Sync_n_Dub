"""Backend-facing voice-conversion entry point.

Chains after TTS: reads the merged JSON, converts each
`segments[].tts_audio` toward the speaker's reference clip, and writes
the result into `segments[].vc_audio`. The mixer then prefers `vc_audio`
over `tts_audio` when it is present.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from models.vc import VCRequest, available_models, get_vc

log = logging.getLogger(__name__)


def convert_segments(
    merged_json: Path,
    *,
    vc_model: str,
    language: str,
    reference_audio: Dict[str, Path],   # speaker id → reference WAV
    out_dir: Path,
    vc_kwargs: Optional[dict] = None,
    out_json: Optional[Path] = None,
) -> dict:
    if vc_model not in available_models():
        raise ValueError(
            f"Unknown VC model {vc_model!r}. "
            f"Available: {', '.join(available_models())}"
        )
    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    requests: List[VCRequest] = []
    for seg in data.get("segments", []):
        src = seg.get("tts_audio")
        spk = seg.get("speaker")
        ref = reference_audio.get(spk) if spk else None
        if not src or not ref:
            continue
        requests.append(VCRequest(
            id=seg["id"],
            source_audio=Path(src),
            reference_audio=Path(ref),
            speaker=spk,
        ))

    if not requests:
        log.warning("vc: no segments eligible for conversion; skipping")
        return data

    vc = get_vc(vc_model, **(vc_kwargs or {}))
    log.info("vc: converting %d segments with %s", len(requests), vc_model)
    result = vc.convert(requests, out_dir=out_dir, language=language)

    by_id = {s.id: s for s in result.segments}
    errors = 0
    for seg in data.get("segments", []):
        rec = by_id.get(seg["id"])
        if rec is None:
            continue
        if rec.error:
            errors += 1
            seg.setdefault("qc", {})["vc_error"] = rec.error
            continue
        seg["vc_audio"] = str(rec.audio_path)
        seg["vc_duration_s"] = rec.duration_s

    data.setdefault("extra", {})["vc"] = {
        "model": result.model_name,
        "language": language,
        "segments": len(result.segments),
        "errors": errors,
        "out_dir": str(out_dir),
    }

    dest = Path(out_json) if out_json else merged_json
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info("vc: wrote %s (errors=%d)", dest, errors)
    return data
