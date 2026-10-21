#!/usr/bin/env python
"""Render one reference clip per catalog voice, so voice assignment can
actually compare timbre instead of only gender.

`backend.voices.assign_voices` already scores an ECAPA cosine between the
real speaker's reference clip and each candidate TTS voice:

    if voice_embs and spk["id"] in speaker_embs and v.voice in voice_embs:
        sim = _cosine(speaker_embs[spk["id"]], voice_embs[v.voice])
        why["embed_cos"] = sim
        score += 0.4 * sim

`voice_embs` is built from `Voice.reference_audio`, and the built-in catalog
ships names with no audio -- so that branch never fires and the similarity
term contributes nothing. Nothing errors; the assignment just silently
degrades to gender + native-language. This script renders the missing audio.

One TTS call per voice (45 for indic-speak), written to
`data/voice_refs/<engine>/<voice>.wav`, where `attach_voice_references()`
finds them by convention. Idempotent: existing non-empty clips are skipped,
so a re-run after a partial failure only fills the gaps.

The prompt sentence is deliberately fixed across voices -- the embedding
should describe the speaker, and holding the text constant keeps phonetic
content from leaking into the comparison.

These clips must be RENDERED BY THE ENGINE, never downloaded. The cosine
compares a real speaker against what the candidate voice actually sounds
like, so a clip of some other human filed under `Amit.wav` would score a
confident similarity against the wrong timbre -- worse than the term being
inert, because the log then looks like the feature is working. If a future
engine's "voices" are themselves reference clips (IndicF5 is zero-shot and
takes `ref_audio_path` + `ref_text`), that is a different catalog with a
different provenance story, not more files in this directory.

Usage:
    $PY scripts/build_voice_refs.py                    # all indic-speak voices
    $PY scripts/build_voice_refs.py --languages hi,ta  # just these
    $PY scripts/build_voice_refs.py --force            # re-render everything
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from backend.voices import VOICE_REF_DIR, load_voice_catalog

log = logging.getLogger("build_voice_refs")

#: Held constant across voices on purpose -- see module docstring. Neutral
#: register, no style, so the clip reflects the voice rather than a delivery.
PROMPT = {
    "hi": "नमस्ते, यह आवाज़ का एक नमूना है जिसे पहचान के लिए रिकॉर्ड किया गया है।",
    "en": "Hello, this is a short sample of my voice recorded for identification.",
}
#: Minimum plausible WAV size; anything smaller is a failed render, not a cache.
MIN_BYTES = 8_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="indic-speak")
    ap.add_argument("--tts", default="bodhan",
                    help="TTS adapter key (default: bodhan).")
    ap.add_argument("--languages", default=None,
                    help="Comma-separated language filter, e.g. hi,ta. "
                         "Default: every voice in the catalog.")
    ap.add_argument("--prompt-language", default="hi",
                    help="Which PROMPT sentence to speak (default: hi).")
    ap.add_argument("--out-dir", default=None,
                    help=f"Default: {VOICE_REF_DIR}")
    ap.add_argument("--catalog-json", default=None, type=Path)
    ap.add_argument("--force", action="store_true",
                    help="Re-render clips that already exist.")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Extra seconds between calls; the adapter already "
                         "paces itself, this is for a tighter server budget.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Bodhan's URL/KEY/MODEL are read by the adapter through `os.getenv`, and
    # nothing puts them there when this script is run directly -- `backend.cli`
    # and `backend.edit_agent` each call this for themselves. Without it every
    # render fails on auth, which reads like an outage rather than a missing
    # call. Values are never logged.
    load_dotenv()

    out_root = Path(args.out_dir) if args.out_dir else VOICE_REF_DIR
    out_dir = out_root / args.engine
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_voice_catalog(args.engine, catalog_json=args.catalog_json)
    if args.languages:
        keep = {s.strip().lower() for s in args.languages.split(",") if s.strip()}
        catalog = [v for v in catalog if (v.language or "").lower() in keep]
    if not catalog:
        log.error("no voices matched engine=%s languages=%s",
                  args.engine, args.languages)
        return 2

    text = PROMPT.get(args.prompt_language) or PROMPT["en"]
    log.info("%d voices -> %s", len(catalog), out_dir)

    from models.tts import get_tts
    from models.tts.base import TTSRequest

    tts = get_tts(args.tts)
    done = skipped = failed = 0

    for i, v in enumerate(catalog, 1):
        dest = out_dir / f"{v.voice}.wav"
        if dest.exists() and dest.stat().st_size >= MIN_BYTES and not args.force:
            skipped += 1
            continue

        # Synthesize into a scratch dir, then move the single clip into place:
        # adapters name their own output per segment id.
        work = out_dir / f".tmp_{v.voice}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            res = tts.synthesize(
                [TTSRequest(id=v.voice, text=text, voice=v.voice,
                            language=args.prompt_language)],
                out_dir=work,
                language=args.prompt_language,
            )
            seg = res.segments[0]
            if seg.error:
                raise RuntimeError(seg.error)
            src = Path(seg.audio_path)
            if not src.exists() or src.stat().st_size < MIN_BYTES:
                raise RuntimeError(
                    f"render too small ({src.stat().st_size if src.exists() else 0} B)"
                )
            src.replace(dest)
            done += 1
            log.info("[%d/%d] %-10s %-5s -> %s (%.1f s)",
                     i, len(catalog), v.voice, v.language or "?",
                     dest.name, seg.duration_s)
        except Exception as e:                      # noqa: BLE001 - report & continue
            failed += 1
            log.warning("[%d/%d] %-10s FAILED: %s", i, len(catalog), v.voice, e)
        finally:
            for leftover in work.glob("*"):
                leftover.unlink(missing_ok=True)
            work.rmdir()

        if args.sleep:
            time.sleep(args.sleep)

    log.info("rendered=%d skipped=%d failed=%d -> %s",
             done, skipped, failed, out_dir)
    if failed:
        log.warning("re-run to fill the %d gap(s); existing clips are skipped",
                    failed)
    # Confirm the catalog now actually picks them up.
    from backend.voices import attach_voice_references
    fresh = load_voice_catalog(args.engine, catalog_json=args.catalog_json)
    n = sum(1 for v in fresh if v.reference_audio is not None)
    log.info("catalog now resolves %d/%d reference clips", n, len(fresh))
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
