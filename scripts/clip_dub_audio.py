#!/usr/bin/env python
"""Translate one clip's cue-timed source text and synthesise a dub track for it.

Input is the cue JSON written by ``build_clip_cues.py``; output is a single wav
of exactly the window's duration, with each cue's speech placed at that cue's
own start time.

Placement is per-cue rather than one global tempo over the window. The consumer
is a lip-sync model, so what matters is that speech is present while the mouth
is moving; a single global ``atempo`` keeps the total duration right but lets
individual cues drift off the frames they belong to.

Per-cue stretching is capped (``--max-stretch``). A cue that still overruns its
slot after capping is allowed to overlap the next one rather than shifting
everything forward -- a shift accumulates into drift, which is the failure this
placement exists to avoid. Every cap and overlap is recorded in the JSON
sidecar instead of being silently absorbed.

Usage:
    python -I scripts/clip_dub_audio.py \
        --cues data/clips/NAG_240to270.cues.en.json --lang hi \
        --out-wav data/mix/NAG_240to270.hi.dubbed.wav --work-dir /tmp/x
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from backend.codemix import (
    RetainSpan, apply_substitutions, detect_retain_spans, extract_latin_tokens,
    load_termbase,
)
from backend.gender import predict_gender
from backend.media_io import _ffmpeg_bin
from backend.stretch import stretch_to_duration
from backend.voices import DEFAULT_INDIC_SPEAK
from models.mt import get_mt
from models.mt.base import MTRequest
from models.translit import TranslitRequest, get_transliterator
from models.tts import TTSRequest, get_tts

log = logging.getLogger("clip_dub")

# Below this a cue has less than one analysis window for the stretch backends,
# and synthesising it costs an API call to produce something unplaceable.
MIN_SLOT_S = 0.35


def default_voice(lang: str, gender: str) -> str | None:
    """Lead voice for `lang` matching a MEASURED gender (never a default)."""
    for v in DEFAULT_INDIC_SPEAK:
        if v.language == lang and v.gender == gender:
            return v.voice
    return None


def detect_speaker_gender(ref_audio: Path, cues: list[dict], *,
                          device: str, log_) -> dict:
    """Measure the speaker's gender from the window's own audio.

    Voted per cue rather than taken from one read of the whole window: these
    clips are cut on shot boundaries and a window can hold more than one
    person, so a single prediction over all 30 s can be decided by whoever
    talks longest rather than by whoever the lip-sync will animate.

    This is not optional and there is no manual override, because the failure
    it prevents is silent. An earlier sweep defaulted to a male voice and
    dubbed it over a female speaker in all 15 languages; nothing errored, and
    a human listening to the output is what caught it (CLAUDE.md §3).
    """
    ff = _ffmpeg_bin()
    votes: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        for i, c in enumerate(cues):
            if c["slot_s"] < 0.8:        # too short to read a pitch track from
                continue
            piece = Path(td) / f"g{i:03d}.wav"
            p = subprocess.run(
                [ff, "-y", "-v", "error", "-ss", str(c["start"]),
                 "-to", str(c["end"]), "-i", str(ref_audio),
                 "-ar", "16000", "-ac", "1", str(piece)],
                capture_output=True, text=True)
            if p.returncode != 0 or not piece.exists():
                continue
            try:
                votes.append({"cue": i, "start": c["start"], "end": c["end"],
                              "gender": predict_gender(piece, device=device)})
            except Exception as e:
                log_.warning("gender: cue %d failed: %s", i, e)

    if not votes:
        # Refuse to guess. A wrong voice is worse than a stopped run, and a
        # defaulted one is exactly what this function exists to remove.
        raise RuntimeError(
            f"could not measure speaker gender from {ref_audio} -- no cue "
            f"yielded a prediction. Pass --ref-audio with usable speech, or "
            f"--voice to name the voice outright.")

    n_f = sum(1 for v in votes if v["gender"] == "female")
    n_m = len(votes) - n_f
    gender = "female" if n_f > n_m else "male"
    conf = max(n_f, n_m) / len(votes)
    log_.info("gender: MEASURED %s from %d cue(s) (female=%d male=%d, "
              "confidence %.0f%%)", gender, len(votes), n_f, n_m, conf * 100)
    if conf < 0.7:
        log_.warning("gender: only %.0f%% agreement -- the window probably "
                     "holds more than one speaker; check the sidecar votes",
                     conf * 100)
    return {"gender": gender, "confidence": round(conf, 3),
            "votes_female": n_f, "votes_male": n_m, "n_votes": len(votes),
            "ref_audio": str(ref_audio), "per_cue": votes,
            "method": "backend.gender.predict_gender (ECAPA, F0 fallback)"}


def codemix_texts(src_texts: list[str], mt_texts: list[str], *,
                  target_language: str, target_script: str,
                  translit_model: str, termbase_path: Path | None,
                  log_) -> tuple[list[str], dict]:
    """Script-normalise MT output, per `backend.codemix` steps 1-4.

    Bodhan's MT leaves technical terms in Latin script *inconsistently* -- on
    one 30 s Hindi clip "WT" came back as `डब्ल्यूटी` in one cue, `WT` in
    another and `W` in a third. An Indic TTS voice fed a bare Latin token does
    not reliably say it, so this is a correctness problem for the dub, not a
    cosmetic one.

    Terms are kept (that is the code-mixing) but rendered in the target script
    so the voice can pronounce them, and the same term renders the same way
    everywhere in the clip.
    """
    termbase = load_termbase(termbase_path) if termbase_path else {}

    # 1. spans worth retaining, from the source, plus any Latin left in the MT
    spans_per: list[list[RetainSpan]] = []
    unique: dict[str, None] = {}
    for src, mt in zip(src_texts, mt_texts):
        spans = detect_retain_spans(src or "", termbase=termbase)
        for tok in extract_latin_tokens(mt or ""):
            if tok and not any(sp.text.lower() == tok.lower() for sp in spans):
                spans.append(RetainSpan(text=tok, start=-1, end=-1,
                                        type="latin_in_mt"))
        spans_per.append(spans)
        for sp in spans:
            if sp.text.lower() not in termbase:
                unique.setdefault(sp.text, None)

    # 2. one batched transliteration call for every unique term in the clip
    tmap: dict[str, str] = {}
    if unique:
        tr = get_transliterator(translit_model)
        reqs = [TranslitRequest(id=f"T{i}", text=t, target_script=target_script)
                for i, t in enumerate(unique)]
        res = tr.transliterate(reqs, target_language=target_language,
                               target_script=target_script)
        for req, seg in zip(reqs, res.segments):
            out = (getattr(seg, "text", "") or "").strip()
            if out:
                tmap[req.text] = out
        log_.info("codemix: %d unique Latin term(s) -> %s", len(unique), tmap)

    # 3. substitute back into each MT string
    finals = []
    for spans, mt in zip(spans_per, mt_texts):
        for sp in spans:
            sp.translit = tmap.get(sp.text, "")
        finals.append(apply_substitutions(mt or "", spans, termbase=termbase))

    leftover = sorted({t for f in finals for t in extract_latin_tokens(f)})
    return finals, {
        "translit_model": translit_model,
        "target_script": target_script,
        "unique_terms": sorted(unique),
        "translit_map": tmap,
        "termbase_entries": len(termbase),
        "latin_remaining_after_codemix": leftover,
    }


def assemble(pieces: list[tuple[Path, float]], out_wav: Path,
             duration: float, sample_rate: int) -> None:
    """Place each (wav, start_s) into one track of exactly `duration` seconds.

    `amix` runs with normalize=0: the default divides by the input count, which
    would make a track quieter purely because it has more cues.
    """
    ff = _ffmpeg_bin()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if not pieces:
        subprocess.run(
            [ff, "-y", "-f", "lavfi", "-i",
             f"anullsrc=r={sample_rate}:cl=stereo", "-t", str(duration),
             "-c:a", "pcm_s16le", str(out_wav)], check=True, capture_output=True)
        return

    cmd = [ff, "-y"]
    for wav, _ in pieces:
        cmd += ["-i", str(wav)]
    parts, labels = [], []
    for i, (_, at) in enumerate(pieces):
        ms = max(0, int(round(at * 1000)))
        parts.append(f"[{i}:a]aresample={sample_rate},"
                     f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                     f"adelay={ms}|{ms}[d{i}]")
        labels.append(f"[d{i}]")
    parts.append(f"{''.join(labels)}amix=inputs={len(pieces)}:normalize=0,"
                 f"apad,atrim=end={duration},asetpts=N/SR/TB[out]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[out]",
            "-ar", str(sample_rate), "-ac", "2", "-c:a", "pcm_s16le", str(out_wav)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg assemble failed:\n{p.stderr[-2000:]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cues", required=True, type=Path)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out-wav", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--mt", default="bodhan")
    ap.add_argument("--tts", default="bodhan")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--ref-audio", type=Path, default=None,
                    help="Source-language audio for this window. Defaults to "
                         "the cue JSON's ref_audio. Speaker gender is measured "
                         "from it; there is no manual gender switch.")
    ap.add_argument("--gender-device", default="cpu")
    ap.add_argument("--target-script", default="codemix",
                    choices=("native", "roman", "codemix"),
                    help="Bodhan's own script mode (DEFAULT codemix). codemix "
                         "keeps English technical terms in Latin inside the "
                         "target-script sentence, which is both how this "
                         "material is actually spoken and consistent across "
                         "cues -- `native` leaves Latin remnants at random.")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--max-stretch", type=float, default=0.30)
    ap.add_argument("--skip-mt", action="store_true",
                    help="cue text is already in the target language")
    ap.add_argument("--normalize-script", dest="codemix",
                    action="store_true", default=False,
                    help="Post-hoc pass that transliterates any remaining "
                         "Latin into the target script. OFF by default and "
                         "the OPPOSITE of --target-script codemix: only use "
                         "it with --target-script native, where Bodhan leaves "
                         "Latin remnants unintentionally.")
    ap.add_argument("--translit", default="bodhan")
    ap.add_argument("--termbase", default=None, type=Path)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    spec = json.loads(args.cues.read_text(encoding="utf-8"))
    cues = spec["cues"]
    duration = float(spec["duration_s"])
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)

    usable = [c for c in cues if c["slot_s"] >= MIN_SLOT_S]
    skipped = len(cues) - len(usable)

    # ---- speaker gender: always measured, never defaulted ----------------
    if args.voice:
        gender_info = {"gender": None, "method": "skipped (--voice given)"}
        voice = args.voice
    else:
        ref = args.ref_audio or (Path(spec["ref_audio"])
                                 if spec.get("ref_audio") else None)
        if not ref or not Path(ref).exists():
            log.error("no reference audio (cue JSON ref_audio=%r, --ref-audio=%r). "
                      "Speaker gender cannot be measured and this stage will not "
                      "guess it.", spec.get("ref_audio"), args.ref_audio)
            return 2
        gender_info = detect_speaker_gender(Path(ref), usable,
                                            device=args.gender_device, log_=log)
        voice = default_voice(args.lang, gender_info["gender"])

    log.info("lang=%s voice=%s gender=%s cues=%d (usable=%d) window=%s",
             args.lang, voice, gender_info.get("gender"), len(cues),
             len(usable), spec["window_abs_s"])

    # ---- translate -------------------------------------------------------
    if args.skip_mt:
        texts = [c["text"] for c in usable]
        mt_name = "none (cues already in target language)"
    else:
        mt = get_mt(args.mt)
        texts = []
        for i, c in enumerate(usable):
            try:
                r = mt.translate(
                    [MTRequest(id=f"c{i:03d}", text=c["text"],
                               source_language=spec.get("language", "en"),
                               target_script=args.target_script,
                               length_hint_s=c["slot_s"])],
                    target_language=args.lang)
                seg = r.segments[0]
                texts.append((seg.text or "").strip())
            except Exception as e:
                log.warning("MT failed on cue %d: %s", i, e)
                texts.append("")
        mt_name = args.mt
    n_mt_empty = sum(1 for t in texts if not t)
    if usable and n_mt_empty == len(usable):
        log.error("%s: every cue failed to translate -- aborting before TTS",
                  args.lang)
        return 2

    # ---- code-mix / script normalisation (default on) --------------------
    texts_mt = list(texts)
    codemix_info: dict = {"enabled": False}
    if args.codemix and not args.skip_mt:
        try:
            texts, codemix_info = codemix_texts(
                [c["text"] for c in usable], texts,
                target_language=args.lang, target_script=args.target_script,
                translit_model=args.translit, termbase_path=args.termbase,
                log_=log)
            codemix_info["enabled"] = True
        except Exception as e:
            # Falling through with Latin remnants is worse than loud failure:
            # the voice silently drops or mangles those tokens.
            log.error("codemix failed for %s: %s -- keeping raw MT text",
                      args.lang, e)
            texts = texts_mt
            codemix_info = {"enabled": False, "error": str(e)}

    # ---- synthesise ------------------------------------------------------
    tts = get_tts(args.tts)
    reqs = [TTSRequest(id=f"{args.lang}_c{i:03d}", text=t, voice=voice,
                       language=args.lang)
            for i, t in enumerate(texts) if t]
    idx_of = [i for i, t in enumerate(texts) if t]
    result = tts.synthesize(reqs, out_dir=work / "synth", language=args.lang,
                            sample_rate=24000) if reqs else None

    # ---- fit each cue to its slot and place ------------------------------
    fitted_dir = work / "fitted"; fitted_dir.mkdir(parents=True, exist_ok=True)
    pieces: list[tuple[Path, float]] = []
    report = []
    seg_by_idx = {idx_of[k]: s for k, s in enumerate(result.segments)} if result else {}

    for i, cue in enumerate(usable):
        rec = dict(cue)
        rec["text_src"] = cue["text"]
        rec["text_mt"] = texts_mt[i]
        rec["text_tgt"] = texts[i]
        rec["codemixed"] = bool(texts[i] != texts_mt[i])
        rec["voice"] = voice
        seg = seg_by_idx.get(i)
        if not texts[i]:
            rec["status"] = "mt_failed"; report.append(rec); continue
        if seg is None or seg.error or seg.duration_s <= 0:
            rec["status"] = f"tts_failed: {getattr(seg, 'error', 'no segment')}"
            report.append(rec); continue
        rec["tts_raw_s"] = round(seg.duration_s, 3)
        fitted = fitted_dir / f"c{i:03d}.wav"
        try:
            sr = stretch_to_duration(seg.audio_path, fitted,
                                     target_duration_s=cue["slot_s"],
                                     max_stretch=args.max_stretch)
        except Exception as e:
            rec["status"] = f"stretch_failed: {e}"; report.append(rec); continue
        rec.update(status="ok", fitted_s=round(sr.achieved_duration_s, 3),
                   wanted_factor=round(sr.requested_factor, 3),
                   applied_factor=round(sr.applied_factor, 3),
                   capped=bool(sr.capped),
                   overruns_slot=bool(sr.achieved_duration_s > cue["slot_s"] + 0.05),
                   backend=sr.backend)
        pieces.append((fitted, cue["start"]))
        report.append(rec)

    assemble(pieces, Path(args.out_wav), duration, args.sample_rate)

    ok = sum(1 for r in report if r.get("status") == "ok")
    capped = sum(1 for r in report if r.get("capped"))
    overrun = sum(1 for r in report if r.get("overruns_slot"))
    failed = sum(1 for r in report if "failed" in str(r.get("status")))

    sidecar = Path(args.out_wav).with_suffix(".cues.json")
    sidecar.write_text(json.dumps({
        "lang": args.lang, "voice": voice, "mt": mt_name, "tts": args.tts,
        "codemix": codemix_info,
        "speaker_gender": gender_info,
        "source_cues": str(args.cues),
        "source_kind": spec.get("source"), "source_file": spec.get("source_file"),
        "window_abs_s": spec["window_abs_s"], "duration_s": duration,
        "sample_rate": args.sample_rate, "max_stretch": args.max_stretch,
        "cues_total": len(cues), "cues_synthesised": ok,
        "cues_too_short_skipped": skipped, "cues_stretch_capped": capped,
        "cues_overrunning_slot": overrun, "cues_failed": failed,
        "out_wav": str(args.out_wav), "cues": report,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    log.info("%s: %d/%d cues ok (capped=%d overrun=%d failed=%d short=%d) -> %s",
             args.lang, ok, len(cues), capped, overrun, failed, skipped,
             args.out_wav)
    if ok == 0:
        log.error("%s: no cue produced audio -- the wav is silence", args.lang)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
