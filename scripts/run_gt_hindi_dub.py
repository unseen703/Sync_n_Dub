#!/usr/bin/env python3
"""Generate Hindi-dubbed audio for all ground-truth source videos.

For each of the 4 source videos in the golden index, this script:
  1. Extracts two 30-second clips (from different regions of the video)
  2. Runs ASR  (indictranscribe_core — local canary model)
  3. Runs MT   (bodhan indic-translate — REST API)
  4. Runs code-mix / script normalisation (allow code-mixed output)
  5. Runs TTS  (bodhan indic-speak — REST API, needs BODHAN_TTS_KEY)

Usage:
    python -I scripts/run_gt_hindi_dub.py [--asr indictranscribe_core|indictranscribe_flex]
                                          [--tts bodhan|indicf5]
                                          [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from backend.media_io import extract_audio, probe_duration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_gt_hindi_dub")

DATA_DIR = REPO_ROOT / "data"
GOLDEN_INDEX = DATA_DIR / "golden" / "index.json"
OUTPUT_ROOT = DATA_DIR / "gt_hindi_dub"

CLIP_DURATION = 30  # seconds per clip
CLIPS_PER_VIDEO = 2


def _find_source_video(source_dir: str) -> Path:
    """Resolve the mp4 file inside a golden-index source_dir."""
    d = REPO_ROOT / source_dir
    if not d.exists():
        raise FileNotFoundError(f"Source dir not found: {d}")
    mp4s = sorted(d.glob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No .mp4 in {d}")
    return mp4s[0]


def _pick_clip_offsets(duration_s: float) -> list[float]:
    """Pick two non-overlapping 30s offsets spread across the video."""
    if duration_s < CLIP_DURATION * 2:
        return [0.0]
    margin = 10.0
    usable = duration_s - margin
    if usable < CLIP_DURATION * CLIPS_PER_VIDEO + margin:
        return [margin, margin + CLIP_DURATION + 5]
    third = usable / 3
    return [
        max(margin, third - CLIP_DURATION / 2),
        max(margin, 2 * third - CLIP_DURATION / 2),
    ]


def _cut_clip(source_mp4: Path, start_s: float, duration_s: float,
              out_mp4: Path) -> Path:
    """Extract a clip from source_mp4 using ffmpeg."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    from backend.media_io import _ffmpeg_bin
    ffmpeg = _ffmpeg_bin()
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start_s:.2f}",
        "-i", str(source_mp4),
        "-t", f"{duration_s:.2f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    log.info("cutting clip: %s @ %.1fs for %.1fs", source_mp4.name, start_s, duration_s)
    subprocess.run(cmd, check=True, capture_output=True)
    return out_mp4


def run_clip_pipeline(
    clip_mp4: Path,
    out_dir: Path,
    *,
    asr_model: str,
    tts_model: str,
    device: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Run ASR → MT → codemix → TTS for one 30s clip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_16k = out_dir / "audio_16k.wav"
    extract_audio(clip_mp4, audio_16k, sample_rate=16_000, mono=True)

    result: dict = {
        "clip": str(clip_mp4),
        "audio_16k": str(audio_16k),
    }

    if dry_run:
        log.info("[dry-run] would process %s", clip_mp4.name)
        return result

    # --- ASR ---
    log.info("=== ASR (%s) ===", asr_model)
    from backend.asr import transcribe as asr_transcribe
    asr_result = asr_transcribe(
        audio_16k, model=asr_model, device=device, language="en",
    )
    asr_json = out_dir / "asr.json"
    asr_json.write_text(json.dumps(asr_result.to_jsonable(), indent=2, ensure_ascii=False))
    asr_txt = out_dir / "asr.transcript.txt"
    asr_txt.write_text(
        "\n".join(s.text.strip() for s in asr_result.segments if s.text) + "\n",
        encoding="utf-8",
    )
    result["asr_json"] = str(asr_json)
    result["asr_text"] = asr_result.text
    log.info("ASR transcript: %s", asr_result.text[:120])

    # --- MT (bodhan indic-translate) ---
    log.info("=== MT (bodhan → hi) ===")
    from backend.translate import translate_asr, write_mt_outputs
    mt_result = translate_asr(
        asr_result,
        target_language="hi",
        model="bodhan",
        device=device,
    )
    mt_json = out_dir / "mt.json"
    mt_txt = out_dir / "mt.transcript.txt"
    write_mt_outputs(mt_json, mt_txt, asr_result, mt_result)
    result["mt_json"] = str(mt_json)
    mt_text = " ".join(s.text.strip() for s in mt_result.segments if s.text)
    result["mt_text"] = mt_text
    log.info("MT output: %s", mt_text[:120])

    # --- Code-mix / script normalisation (allow code-mixed output) ---
    log.info("=== Code-mix (bodhan translit) ===")
    try:
        from backend.codemix import codemix_segments
        cm = codemix_segments(
            mt_json,
            target_language="hi",
            target_script="native",
            translit_model="bodhan",
        )
        result["codemix"] = cm.get("extra", {}).get("codemix", {})
        log.info("Code-mix: %d retained terms",
                 result["codemix"].get("retained_terms", 0))
    except Exception as e:
        log.warning("Code-mix failed (non-fatal): %s", e)
        result["codemix_error"] = str(e)

    # --- TTS ---
    log.info("=== TTS (%s → hi) ===", tts_model)
    from backend.tts import synthesize_segments
    tts_out = out_dir / "tts"
    # No diarization → single default speaker. Edge TTS picks a Hindi voice.
    speaker_voice_map: dict = {
        None: {"engine": tts_model, "voice": None, "reference_audio": None},
    }
    # Check if merged JSON has speaker labels and build a map for each.
    merged_data = json.loads(mt_json.read_text())
    speakers = {seg.get("speaker") for seg in merged_data.get("segments", [])}
    speakers.discard(None)
    if speakers:
        # No diarization here and so no reference clip and no predicted
        # gender — `assign_voices` needs both to do better than this. Hand
        # out distinct Hindi-recorded catalog voices round-robin so two
        # speakers do not read in the same voice, and let the gender land
        # where it lands; a real run goes through stage 9d instead.
        from backend.voices import load_voice_catalog
        pool = [v.voice for v in load_voice_catalog("indic-speak", language="hi")
                if (v.language or "").lower() == "hi"]
        speaker_voice_map = {}
        for i, spk in enumerate(sorted(speakers)):
            speaker_voice_map[spk] = {
                "engine": tts_model,
                "voice": pool[i % len(pool)] if pool else None,
                "reference_audio": None,
            }
    synthesize_segments(
        mt_json,
        tts_model=tts_model,
        language="hi",
        speaker_voice_map=speaker_voice_map,
        out_dir=tts_out,
        sample_rate=24000,
        apply_stretch=True,
    )
    result["tts_dir"] = str(tts_out)

    # Collect TTS audio paths from the updated mt.json
    merged = json.loads(mt_json.read_text())
    tts_paths = []
    for seg in merged.get("segments", []):
        p = seg.get("tts_audio") or seg.get("tts_raw_audio")
        if p and Path(p).exists():
            tts_paths.append(p)
    result["tts_audio_files"] = tts_paths
    log.info("TTS: %d audio files generated", len(tts_paths))

    # Write summary
    summary = out_dir / "summary.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr", default="indictranscribe_core",
                    choices=["indictranscribe_core", "indictranscribe_flex"],
                    help="ASR model (default: indictranscribe_core)")
    ap.add_argument("--tts", default="bodhan",
                    choices=["bodhan", "indicf5"],
                    help="TTS model (default: bodhan)")
    ap.add_argument("--device", default=None,
                    help="Compute device (default: auto)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Cut clips and extract audio but skip model inference")
    args = ap.parse_args()

    if not GOLDEN_INDEX.exists():
        log.error("Golden index not found: %s", GOLDEN_INDEX)
        sys.exit(1)

    index = json.loads(GOLDEN_INDEX.read_text())

    # Deduplicate by video_id → source_dir
    videos: dict[str, str] = {}
    for entry in index:
        vid = entry["video_id"]
        if vid not in videos:
            videos[vid] = entry["source_dir"]

    log.info("Found %d unique source videos: %s", len(videos), list(videos.keys()))

    all_results: list[dict] = []
    for video_id, source_dir in videos.items():
        log.info("=" * 60)
        log.info("Processing video: %s", video_id)
        log.info("=" * 60)

        source_mp4 = _find_source_video(source_dir)
        duration = probe_duration(source_mp4)
        log.info("Source: %s (%.1fs)", source_mp4.name, duration)

        offsets = _pick_clip_offsets(duration)
        for clip_idx, offset in enumerate(offsets[:CLIPS_PER_VIDEO]):
            clip_tag = f"clip{clip_idx}_{int(offset)}to{int(offset + CLIP_DURATION)}"
            clip_dir = OUTPUT_ROOT / video_id / clip_tag
            clip_mp4 = clip_dir / f"{video_id}_{clip_tag}.mp4"

            _cut_clip(source_mp4, offset, CLIP_DURATION, clip_mp4)

            result = run_clip_pipeline(
                clip_mp4, clip_dir,
                asr_model=args.asr,
                tts_model=args.tts,
                device=args.device,
                dry_run=args.dry_run,
            )
            result["video_id"] = video_id
            result["clip_tag"] = clip_tag
            result["offset_s"] = offset
            all_results.append(result)

    # Write master manifest
    manifest = OUTPUT_ROOT / "manifest.json"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    log.info("=" * 60)
    log.info("Done. %d clips processed. Manifest: %s", len(all_results), manifest)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
