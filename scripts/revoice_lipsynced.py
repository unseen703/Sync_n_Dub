#!/usr/bin/env python
"""Re-synthesise a dub in the MEASURED speaker gender and swap it onto a render.

Audio-only repair for a finished lip-sync. Re-uses the translated, code-mixed
text already in the run's cue sidecar, so no MT call is repeated; only TTS and
the mux re-run. Cost is seconds per language instead of a re-render.

WHAT THIS DOES NOT FIX
----------------------
The mouth in the existing video was generated *conditioned on the old audio*.
Swapping the track keeps cue-level alignment -- every cue starts at the same
timestamp and is fitted to the same slot -- but inside a cue the new voice's
phonemes do not land where the old voice's did, so the lip-sync is correct at
phrase level and approximate at phoneme level. Re-running the lip-sync stage on
the new audio is the only way to make the mouth actually match it. This script
is the cheap repair, and it records that limitation in its sidecar rather than
letting a later reader assume the render was rebuilt.

Usage:
    python -I scripts/revoice_lipsynced.py \
        --job-glob 'data/jobs/2026*__musetalk_asdgated_*' \
        --video-pattern 'final.{lang}.musetalk.asdgated.mp4' \
        --sidecar-pattern 'data/mix/IIT_svwrTOm55y4_240to270_{lang}.dubbed.cues.json' \
        --ref-audio data/clips/IIT_svwrTOm55y4_240to270.en.16k.wav
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from backend.media_io import _ffmpeg_bin
from backend.stretch import stretch_to_duration
from models.tts import TTSRequest, get_tts

sys.path.insert(0, str(ROOT / "scripts"))
from clip_dub_audio import assemble, default_voice, detect_speaker_gender  # noqa: E402

log = logging.getLogger("revoice")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-glob", required=True)
    ap.add_argument("--video-pattern", required=True,
                    help="filename inside each job dir, with {lang}")
    ap.add_argument("--sidecar-pattern", required=True,
                    help="path to the existing cue sidecar, with {lang}")
    ap.add_argument("--ref-audio", required=True, type=Path)
    ap.add_argument("--out-suffix", default="revoiced")
    ap.add_argument("--tts", default="bodhan")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--max-stretch", type=float, default=0.30)
    ap.add_argument("--gender-device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    ff = _ffmpeg_bin()

    jobs = sorted(Path(p) for p in globmod.glob(args.job_glob) if Path(p).is_dir())
    if not jobs:
        log.error("no job dirs match %s", args.job_glob)
        return 1

    gender = None
    done, failed = [], []
    for job in jobs:
        lang = job.name.rsplit("_", 1)[-1]
        video = job / args.video_pattern.format(lang=lang)
        sidecar = Path(args.sidecar_pattern.format(lang=lang))
        if not video.exists() or not sidecar.exists():
            log.warning("%s: missing video or sidecar, skipping", lang)
            failed.append(lang)
            continue

        spec = json.loads(sidecar.read_text(encoding="utf-8"))
        cues = [c for c in spec["cues"] if c.get("status") == "ok"]
        duration = float(spec["duration_s"])

        # Measured once: every language dubs the same speaker in the same
        # window, so the per-cue vote does not change between them.
        if gender is None:
            ginfo = detect_speaker_gender(args.ref_audio, cues,
                                          device=args.gender_device, log_=log)
            gender = ginfo["gender"]
        voice = default_voice(lang, gender)
        old_voice = spec.get("voice")
        if voice == old_voice:
            log.info("%s: already voiced %s (%s) -- skipping", lang, gender, voice)
            continue
        log.info("%s: %s -> %s (%s)", lang, old_voice, voice, gender)
        if args.dry_run:
            continue

        work = ROOT / "data" / "mix" / "_revoice" / f"{job.name}"
        tts = get_tts(args.tts)
        reqs = [TTSRequest(id=f"{lang}_c{i:03d}", text=c["text_tgt"],
                           voice=voice, language=lang)
                for i, c in enumerate(cues)]
        res = tts.synthesize(reqs, out_dir=work / "synth", language=lang,
                             sample_rate=24000)

        fitted_dir = work / "fitted"; fitted_dir.mkdir(parents=True, exist_ok=True)
        pieces, ok = [], 0
        for i, (c, seg) in enumerate(zip(cues, res.segments)):
            if seg.error or seg.duration_s <= 0:
                log.warning("%s cue %d: tts failed (%s)", lang, i, seg.error)
                continue
            out = fitted_dir / f"c{i:03d}.wav"
            try:
                stretch_to_duration(seg.audio_path, out,
                                    target_duration_s=c["slot_s"],
                                    max_stretch=args.max_stretch)
            except Exception as e:
                log.warning("%s cue %d: stretch failed (%s)", lang, i, e)
                continue
            pieces.append((out, c["start"])); ok += 1

        if not ok:
            log.error("%s: no cue synthesised, leaving the render alone", lang)
            failed.append(lang); continue

        new_wav = job / "audio" / f"clip.dub.48k.stereo.{args.out_suffix}.wav"
        new_wav.parent.mkdir(parents=True, exist_ok=True)
        assemble(pieces, new_wav, duration, args.sample_rate)

        out_mp4 = video.with_name(video.stem + f".{args.out_suffix}.mp4")
        p = subprocess.run(
            [ff, "-y", "-v", "error", "-i", str(video), "-i", str(new_wav),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_mp4)],
            capture_output=True, text=True)
        if p.returncode != 0:
            log.error("%s: mux failed: %s", lang, p.stderr[-400:])
            failed.append(lang); continue

        (job / f"REVOICE.{args.out_suffix}.json").write_text(json.dumps({
            "what": "audio track replaced after render; video NOT re-rendered",
            "measured_gender": gender,
            "voice_old": old_voice, "voice_new": voice,
            "cues_resynthesised": ok, "mt_reused_from": str(sidecar),
            "new_audio": str(new_wav), "new_video": str(out_mp4),
            "original_video_kept": str(video),
            "limitation": ("the mouth was generated conditioned on the OLD "
                           "audio. Cue starts and slot lengths are identical, "
                           "so alignment holds at phrase level, but phonemes "
                           "inside each cue do not match the new voice. Re-run "
                           "the lip-sync stage to fix that."),
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        log.info("%s: %d cues -> %s", lang, ok, out_mp4.name)
        done.append(lang)

    log.info("revoiced=%d (%s) failed=%d (%s)",
             len(done), " ".join(done), len(failed), " ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
