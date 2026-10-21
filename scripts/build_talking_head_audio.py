#!/usr/bin/env python
"""Render every Talking Head dialogue with Bodhan TTS, and wire the links.

The catalogue (`data/talking_head/dialogues.json`) ships every `audio_url`
null, so the page lists its lines marked "no audio yet". This fills them in:
one WAV per entry under `data/talking_head/audio/`, and the `audio_url`
written back, which is the only step that makes a line playable.

    $PY scripts/build_talking_head_audio.py                 # everything
    $PY scripts/build_talking_head_audio.py --language hi   # one language
    $PY scripts/build_talking_head_audio.py --force         # re-render

**Idempotent.** An entry whose WAV already exists is skipped, so a run
interrupted by rate limiting is resumed by running it again rather than
restarted. Expect 429s throughout a full pass -- `bodhan_adapter` backs off
on its own and this script does not add a second retry layer on top.

**`load_dotenv()` is called here, not inherited.** `backend/cli.py`,
`edit_agent.py` and `experiment.py` each do it for themselves; a new entry
point under `scripts/` needs the same, or it sees no `BODHAN_TTS_KEY` and
every render fails on auth -- which reads like an upstream outage rather
than a missing line (CLAUDE.md section 4, build_voice_refs.py).

**The voice is picked from the catalog, per language.** `assign_voices`'s
rule is a voice RECORDED in the target language; the same applies here, and
the two genders alternate so a page of lines is not one voice reading all of
them. English is the known gap: it is a valid engine `lang` with no voice
recorded in it, so it is rendered with a Hindi voice and REPORTED as
accented rather than passed off as native.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv                          # noqa: E402

load_dotenv(ROOT / ".env")

from backend.voices import load_voice_catalog           # noqa: E402
from models.tts import get_tts                          # noqa: E402
from models.tts.base import TTSRequest                  # noqa: E402

log = logging.getLogger("build_th_audio")

CATALOGUE = ROOT / "data" / "talking_head" / "dialogues.json"
AUDIO_DIR = ROOT / "data" / "talking_head" / "audio"
URL_PREFIX = "/files/talking_head/audio"

#: No voice is RECORDED in English. Rendering the English lines with a Hindi
#: voice is a deliberate, reported compromise, not a silent substitution --
#: the Edge adapter's habit of answering an unknown language with Hindi is
#: exactly the bug CLAUDE.md records for Punjabi.
ENGLISH_FALLBACK_LANG = "hi"

#: A line too short to be worth playing usually means the engine returned a
#: near-empty clip rather than speech.
MIN_BYTES = 4000


def voices_by_language():
    out = {}
    for v in load_voice_catalog("indic-speak"):
        out.setdefault((v.language or "").lower(), []).append(v)
    for k in out:
        # Deterministic, and alternating gender below depends on the order.
        out[k].sort(key=lambda v: (v.gender or "", v.voice))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--language", action="append", default=None,
                    help="Only this language (repeatable).")
    ap.add_argument("--force", action="store_true",
                    help="Re-render entries that already have a WAV.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    doc = json.loads(CATALOGUE.read_text())
    catalog = voices_by_language()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    tts = None if args.dry_run else get_tts("bodhan")

    wanted = set(args.language or doc["dialogues"])
    done = failed = skipped = 0
    notes: list[str] = []

    for lang, rows in doc["dialogues"].items():
        if lang not in wanted or not rows:
            continue
        speak_lang = ENGLISH_FALLBACK_LANG if lang == "en" else lang
        pool = catalog.get(speak_lang) or []
        if not pool:
            notes.append(f"{lang}: no catalog voice; skipped")
            continue
        if lang == "en":
            notes.append(f"en: rendered with {speak_lang} voices — Indic-accented "
                         f"English, since no voice is recorded in English")

        todo, reqs = [], []
        for i, row in enumerate(rows):
            dst = AUDIO_DIR / f"{row['id']}.wav"
            if dst.is_file() and dst.stat().st_size >= MIN_BYTES and not args.force:
                row["audio_url"] = f"{URL_PREFIX}/{dst.name}"
                skipped += 1
                continue
            voice = pool[i % len(pool)]          # alternate the genders
            todo.append((row, dst, voice))
            reqs.append(TTSRequest(id=row["id"], text=row["text"],
                                   voice=voice.voice, language=speak_lang))

        if not reqs:
            continue
        log.info("%s: %d line(s) -> %s", lang, len(reqs),
                 ", ".join(sorted({v.voice for _, _, v in todo})))
        if args.dry_run:
            continue

        res = tts.synthesize(reqs, out_dir=AUDIO_DIR, language=speak_lang,
                             sample_rate=24000)
        by_id = {s.id: s for s in res.segments}
        for row, dst, _v in todo:
            seg = by_id.get(row["id"])
            if seg is None or seg.error:
                failed += 1
                log.warning("  %s FAILED: %s", row["id"],
                            (seg.error if seg else "no result"))
                continue
            src = Path(seg.audio_path)
            if src.resolve() != dst.resolve():
                dst.write_bytes(src.read_bytes())
                src.unlink(missing_ok=True)
            if dst.stat().st_size < MIN_BYTES:
                failed += 1
                log.warning("  %s produced %d bytes; not wiring it up",
                            row["id"], dst.stat().st_size)
                dst.unlink(missing_ok=True)
                continue
            row["audio_url"] = f"{URL_PREFIX}/{dst.name}"
            done += 1

    if not args.dry_run:
        CATALOGUE.write_text(json.dumps(doc, ensure_ascii=False, indent=2))

    total = sum(len(v) for v in doc["dialogues"].values())
    wired = sum(1 for v in doc["dialogues"].values() for r in v if r["audio_url"])
    log.info("\nrendered %d, skipped %d, failed %d", done, skipped, failed)
    log.info("catalogue: %d of %d lines now playable", wired, total)
    for n in notes:
        log.info("note: %s", n)
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
