#!/bin/bash
# IIT profs panel, 4:00-4:30, 15 Indic languages.
# FULL lip-sync re-run driven by the EXISTING female-voice TTS audio.
#
# Why: the 20261015T012750 renders had their mouths generated against a MALE
# voice (gender was defaulted, not measured). The revoice pass swapped the
# audio track afterwards, which fixed the voice but left the mouth animated
# for the wrong waveform -- right at phrase level, wrong at phoneme level.
# This re-runs the lip-sync itself so the mouth is conditioned on the audio
# that actually ships.
#
# MT and TTS are NOT re-run. Each language reuses
#   data/jobs/20261015T012750__..._<lang>/audio/clip.dub.48k.stereo.revoiced.wav
#
# Launch detached:
#   tmux new-window -t dubsync -n iitre \
#     "bash /data1/dipak_data/dubbingstudio/scripts/rerun_iit_musetalk_revoiced.sh"
set -u
cd /data1/dipak_data/dubbingstudio

PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
export PATH=/data1/dipak_data/miniconda3/envs/dubbing/bin:$PATH
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6

STEM_BASE=IIT_svwrTOm55y4_240to270
CLIP=data/clips/${STEM_BASE}.mp4
EN16K=data/clips/${STEM_BASE}.en.16k.wav
SRC_PREFIX=20261015T012750__svwrTOm55y4_240to270__musetalk_asdgated
LANGS="${LANGS:-as bn gu hi kn ml mr ne or pa sa sd ta te ur}"

# MuseTalk v15 fp16 at batch 8 peaked 8052 MiB on this host. Both cards are
# shared with other users, so wait for a card with real headroom instead of
# launching into an OOM that would look like a model fault.
NEED_MIB=${NEED_MIB:-9000}
WAIT_MAX_S=${WAIT_MAX_S:-86400}

TS=$(date +%Y%m%dT%H%M%S)
LOG="logs/musetalk_iit_revoiced_${TS}.log"
STATUS="logs/musetalk_iit_revoiced_${TS}.status"
exec > "$LOG" 2>&1

pick_gpu() {
  # Echo the index of a GPU with >= NEED_MIB free, else nothing.
  nvidia-smi --query-gpu=index,memory.total,memory.used \
             --format=csv,noheader,nounits |
  while IFS=, read -r idx total used; do
    free=$(( ${total// /} - ${used// /} ))
    if [ "$free" -ge "$NEED_MIB" ]; then echo "${idx// /}"; return; fi
  done
}

echo "[$(date +%H:%M:%S)] ================ CONFIG ================"
echo "  task         : FULL lip-sync re-run on the female-voice audio"
echo "  video        : '3 IIT Profs Explain Their Research' (svwrTOm55y4)"
echo "  window       : 240.0-270.0 s (4:00-4:30), 30 s, 25 fps, 1920x1080"
echo "  clip         : $CLIP   (--start 0 --duration 30; clip IS the window)"
echo "  audio        : REUSED from $SRC_PREFIX*_<lang>/audio/"
echo "                 clip.dub.48k.stereo.revoiced.wav  (female, 30.000 s)"
echo "  voice        : female lead per language -- MEASURED female at 100%"
echo "                 confidence over 10 cues (ECAPA), not defaulted"
echo "  MT / TTS     : NOT re-run. Text is the code-mixed translation already"
echo "                 in the 012750 sidecars; audio is the revoiced wav."
echo "  asd gate     : min_speaking_frac=0.12 (0.25 rejects every shot here)"
echo "  face policy  : ASD-gated; only single-speaking-face shots animated."
echo "                 Expect shots 3 (8.00-14.68s) and 5 (22.60-30.00s)."
echo "  lipsync      : musetalk v15, fp16, batch 8, bbox_shift 0"
echo "  gpu          : chosen at launch, needs >= ${NEED_MIB} MiB free"
echo "  KNOWN ISSUE  : the reused audio overruns its cue slots (Indic text is"
echo "                 longer than the English it replaces), so 16-47% of the"
echo "                 speech overlaps depending on language. That is NOT"
echo "                 fixed here -- fixing it needs --duration-adjust and a"
echo "                 re-translate. Re-running lip-sync against this audio"
echo "                 makes the mouth match what is actually heard, overlap"
echo "                 included."
echo "  languages    : $LANGS"
echo "  est. wall    : ~870 s/lang -> ~3.6 h for 15 (less if a card is idle)"
echo "  log          : $LOG"
echo "[$(date +%H:%M:%S)] ========================================"

echo "state=waiting_for_gpu langs_done=0/15 ts=$(date -Is)" > "$STATUS"
echo "[$(date +%H:%M:%S)] waiting for a GPU with >= ${NEED_MIB} MiB free..."
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
WAITED=0
GPU=""
while [ -z "$GPU" ]; do
  GPU=$(pick_gpu)
  [ -n "$GPU" ] && break
  if [ "$WAITED" -ge "$WAIT_MAX_S" ]; then
    echo "[$(date +%H:%M:%S)] gave up waiting after ${WAITED}s"
    echo "state=gave_up_no_gpu ts=$(date -Is)" > "$STATUS"; exit 1
  fi
  sleep 120; WAITED=$((WAITED+120))
  if [ $((WAITED % 1800)) -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] still waiting (${WAITED}s)"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  fi
done
echo "[$(date +%H:%M:%S)] GPU $GPU has headroom after ${WAITED}s; starting"
echo "state=running gpu=$GPU langs_done=0/15 ts=$(date -Is)" > "$STATUS"

i=0
for LG in $LANGS; do
  i=$((i+1))
  DUB="data/jobs/${SRC_PREFIX}_${LG}/audio/clip.dub.48k.stereo.revoiced.wav"
  JOB="${TS}__svwrTOm55y4_240to270__musetalk_revoiced_${LG}"
  echo "[$(date +%H:%M:%S)] ===== [$i/15] $LG : START (job=$JOB) ====="
  if [ ! -f "$DUB" ]; then
    echo "[$(date +%H:%M:%S)] [$LG] MISSING reused audio $DUB -- skipping"
    continue
  fi
  T0=$(date +%s)

  $PY -I scripts/clip_asd_musetalk_run.py \
      --src "$CLIP" \
      --dub-audio "$DUB" \
      --en-audio "$EN16K" \
      --start 0 --duration 30 \
      --job-id "$JOB" --lang "$LG" \
      --width 1920 --height 1080 --fps 25 \
      --batch-size 8 --use-float16 --gpu "$GPU" \
      --asd-min-speaking-frac 0.12
  RC=$?

  JD="data/jobs/$JOB"
  if [ -d "$JD" ]; then
    cat > "$JD/DUB_PROVENANCE.json" <<PROV
{
  "corrects": "config.json field mt_tts, hardcoded to 'SKIPPED' by the driver",
  "audio_path": "generated, REUSED from an earlier run (MT and TTS not re-run)",
  "reused_from": "$DUB",
  "pipeline": "WhisperX large-v3 ASR (en) -> Bodhan indic-translate (target_script=codemix) -> Bodhan indic-speak TTS (female lead voice) -> per-cue rubberband fit",
  "speaker_gender": "female, measured via backend.gender.predict_gender (ECAPA), 10/10 cue votes",
  "supersedes": "data/jobs/${SRC_PREFIX}_${LG} (mouth was animated against a MALE voice)",
  "language": "$LG",
  "known_issue": "the reused audio overruns its cue slots; 16-47% of speech overlaps depending on language. Needs --duration-adjust and a re-translate to fix; not addressed by this run."
}
PROV
  fi

  T1=$(date +%s)
  echo "[$(date +%H:%M:%S)] ===== [$i/15] $LG : DONE rc=$RC in $((T1-T0))s ====="
  echo "state=running gpu=$GPU langs_done=$i/15 last=$LG rc=$RC ts=$(date -Is)" > "$STATUS"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
done

echo "[$(date +%H:%M:%S)] ================ DONE ================"
echo "state=done langs_done=$i/15 ts=$(date -Is)" > "$STATUS"
ls -la data/jobs/${TS}__svwrTOm55y4_240to270__musetalk_revoiced_*/final.*.mp4 2>/dev/null
