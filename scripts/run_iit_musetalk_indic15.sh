#!/bin/bash
# IIT profs panel, 4:00-4:30, 15 Indic languages, ASD-gated MuseTalk.
# Launch detached:
#   tmux new-window -t dubsync -n iit15 \
#     "bash /data1/dipak_data/dubbingstudio/scripts/run_iit_musetalk_indic15.sh"
set -u
cd /data1/dipak_data/dubbingstudio

PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
export PATH=/data1/dipak_data/miniconda3/envs/dubbing/bin:$PATH
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6

STEM_BASE=IIT_svwrTOm55y4_240to270
CLIP=data/clips/${STEM_BASE}.mp4
EN16K=data/clips/${STEM_BASE}.en.16k.wav
CUES=data/clips/${STEM_BASE}.cues.en.json
LANGS="${LANGS:-as bn gu hi kn ml mr ne or pa sa sd ta te ur}"

TS=$(date +%Y%m%dT%H%M%S)
LOG="logs/musetalk_iit_indic15_${TS}.log"
STATUS="logs/musetalk_iit_indic15_${TS}.status"
exec > "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] ================ CONFIG ================"
echo "  video        : '3 IIT Profs Explain Their Research' (svwrTOm55y4)"
echo "  window       : 240.0-270.0 s (4:00-4:30), 30 s, 25 fps"
echo "  clip         : $CLIP  (3840x2160 source pre-downscaled to 1920x1080)"
echo "                 NOTE: --start 0 --duration 30 because the clip is ALREADY"
echo "                 the window; the runner's phase_cut would otherwise seek"
echo "                 240 s into a 30 s file and get silence."
echo "  faces        : 4 shots -- ~5 s B-roll (0 faces), ~8 s 4-face wide shot,"
echo "                 ~17 s single-face close-ups"
echo "  asd gate     : min_speaking_frac=0.12 (default 0.25 rejected EVERY shot"
echo "                 in this window). Measured here: silent faces never cross"
echo "                 ASD 0.5 at all (frac 0.000), speaking faces sit at"
echo "                 0.158-0.264, so any threshold in 0.01-0.15 separates them."
echo "                 clip_asd_musetalk_run.py:695 gates on"
echo "                 len(present)==1 and len(speaking)==1, so this threshold"
echo "                 CANNOT enable the 4-face shot -- that is rejected by the"
echo "                 face-count term regardless."
echo "  face policy  : ASD-GATED. TalkNet scores each tracked face per shot;"
echo "                 only single-speaking-face shots are lip-synced, the rest"
echo "                 pass through untouched. So roughly 17 s of the 30 s is"
echo "                 animated -- deliberate, not a failure. Nothing in"
echo "                 backend/lipsync.py gates on who is speaking, so without"
echo "                 this the 4-face shot animates an arbitrary person."
echo "  text source  : WhisperX large-v3 ASR over the clip's English audio"
echo "                 (no en-orig caption on disk for this video, and the"
echo "                 YouTube timedtext endpoint is 429-throttled)"
echo "  translation  : Bodhan indic-translate, target_script=CODEMIX (default)"
echo "                 English technical terms stay in Latin inside the Indic"
echo "                 sentence; see README 'Translation script' for why."
echo "  tts          : Bodhan indic-speak, male lead voice per language"
echo "  audio fit    : per-cue, rubberband, stretch cap +/-30%"
echo "  lipsync      : musetalk v15, fp16, batch=8, bbox_shift=0"
echo "  device       : GPU 1 (GPU 0 is in use by another user, 10.8 GiB)"
echo "  languages    : $LANGS"
echo "  est. wall    : ~8 min/lang -> ~2 h for all 15"
echo "  CAVEAT       : clip_asd_musetalk_run.py hardcodes"
echo "                 mt_tts='SKIPPED (downloaded YouTube dub drives audio)'"
echo "                 in its config.json. That is WRONG for this run -- the"
echo "                 audio IS generated (ASR->MT->TTS). A DUB_PROVENANCE.json"
echo "                 recording the truth is written into each job folder."
echo "  log          : $LOG"
echo "  status       : $STATUS"
echo "[$(date +%H:%M:%S)] ========================================"

echo "state=starting langs_done=0/15 ts=$(date -Is)" > "$STATUS"
mkdir -p data/mix

i=0
for LG in $LANGS; do
  i=$((i+1))
  STEM="${STEM_BASE}_${LG}"
  JOB="${TS}__svwrTOm55y4_240to270__musetalk_asdgated_${LG}"
  echo "[$(date +%H:%M:%S)] ===== [$i/15] $LG : START (job=$JOB) ====="
  T0=$(date +%s)

  echo "[$(date +%H:%M:%S)] [$LG] P0 dub audio (MT + TTS + fit)"
  $PY -I scripts/clip_dub_audio.py \
      --cues "$CUES" --lang "$LG" \
      --out-wav "data/mix/${STEM}.dubbed.wav" \
      --work-dir "data/mix/_work/${STEM}" || { echo "[$LG] AUDIO FAILED, skipping"; continue; }

  echo "[$(date +%H:%M:%S)] [$LG] P1-P8 ASD-gated musetalk"
  $PY -I scripts/clip_asd_musetalk_run.py \
      --src "$CLIP" \
      --dub-audio "data/mix/${STEM}.dubbed.wav" \
      --en-audio "$EN16K" \
      --start 0 --duration 30 \
      --job-id "$JOB" --lang "$LG" \
      --width 1920 --height 1080 --fps 25 \
      --batch-size 8 --use-float16 --gpu 1 \
      --asd-min-speaking-frac 0.12
  RC=$?

  # The runner's own config.json claims MT/TTS were skipped. Record what the
  # audio actually is, next to it, so a later session reading the job folder
  # is not misled by that line.
  JD="data/jobs/$JOB"
  if [ -d "$JD" ]; then
    cat > "$JD/DUB_PROVENANCE.json" <<PROV
{
  "corrects": "config.json field mt_tts, which is hardcoded to 'SKIPPED'",
  "audio_path": "generated",
  "pipeline": "WhisperX large-v3 ASR (en) -> Bodhan indic-translate -> Bodhan indic-speak TTS -> per-cue rubberband fit",
  "not_used": "YouTube downloaded dub track",
  "language": "$LG",
  "dub_wav": "data/mix/${STEM}.dubbed.wav",
  "cue_sidecar": "data/mix/${STEM}.dubbed.cues.json",
  "source_cues": "$CUES"
}
PROV
  fi

  T1=$(date +%s)
  echo "[$(date +%H:%M:%S)] ===== [$i/15] $LG : DONE rc=$RC in $((T1-T0))s ====="
  echo "state=running langs_done=$i/15 last=$LG rc=$RC ts=$(date -Is)" > "$STATUS"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
done

echo "[$(date +%H:%M:%S)] ================ DONE ================"
echo "state=done langs_done=$i/15 ts=$(date -Is)" > "$STATUS"
ls -la data/jobs/${TS}__svwrTOm55y4_240to270__musetalk_asdgated_*/final.*.mp4 2>/dev/null
