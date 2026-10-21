#!/usr/bin/env bash
# Two 4-minute Hindi dubs with MuseTalk lip-sync, at 1080p, run SEQUENTIALLY.
#
# WHY SEQUENTIAL, not both at once:
#   Measured at 1920x1080 on this host (logs/scaling_bench.log): peak RSS is
#   7.6 GiB @30s, 12.0 @60s, 21.1 @120s -- ~0.148 GiB per second of clip,
#   because MuseTalk holds decoded frames and per-frame latents in process.
#   240s therefore needs ~39 GiB, and the host has 48 GiB available. Two at
#   once would need ~78 GiB and would be killed by the OOM reaper.
#
# WHY 2 GPUs PER VIDEO (sharded by time):
#   Same bench: VRAM 9799 MiB @30s -> 12235 @120s, ~0.027 GiB/s. A single-GPU
#   240s render projects to ~15.5 GiB of 16.3 -- about 800 MiB of headroom.
#   CLAUDE.md records that MuseTalk swallows an OOM and reports it only as
#   "produced no mp4", so that risk is paid AFTER an hour of GPU time.
#   Sharding halves each shard to 120s: 12.2 GiB VRAM and 21.1 GiB RSS each,
#   42 GiB combined, comfortably inside both ceilings -- and it halves wall
#   time as a side effect.
#
# WINDOWS (stated because only one was specified):
#   batman : 0-240s. The source is 278.6s, so this is nearly all of it.
#   iit    : 240-480s. NOT specified by the request; chosen because it is the
#            four-person panel, which has continuous speech and four faces on
#            screen -- the interesting case for diarization and ASD. Change
#            IIT_START to move it.
set -euo pipefail
cd /data1/dipak_data/dubbingstudio

PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
FF=/data1/dipak_data/miniconda3/envs/dubbing/bin/ffmpeg
# cv2's bundled libtbb needs the conda libstdc++; without this the faces/ASD
# stages fail by import ORDER, which looks like a package problem and is not.
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6

TS=$(date +%Y%m%dT%H%M%S)
LOG="logs/musetalk_4min_hi_${TS}.log"
mkdir -p logs data/clips
exec > "$LOG" 2>&1

IIT_SRC=data/videos/iit_profs_quantum_image_disaster_svwrTOm55y4/svwrTOm55y4.mp4
BAT_SRC=data/jobs/20261017-031438__L3oOldViIgY__pyannote.whisperx.mms_fa.bodhan.hi/source/L3oOldViIgY.mp4
IIT_START=240; BAT_START=0; DUR=240

say() { echo "[$(date +%H:%M:%S)] $*"; }

# Fail fast on a bad flag: `_parse_args` is the same parser the run uses, so
# this catches a typo before any clip is cut or any model is loaded. Two
# flags were wrong on the first launch (`--input` is not positional, and the
# diarization flag is `--model`), each costing a full launch to discover.
preflight() {
  "$PY" - <<'PRE' || { echo "PREFLIGHT FAILED — not launching"; exit 2; }
import sys; sys.path.insert(0, ".")
from backend.cli import _parse_args
_parse_args(["--input", "data/clips/batman_1080p_4min.mp4", "--job-id", "x",
             "--model", "pyannote", "--resegment-turns",
             "--refine-speaker-labels", "--face-detector", "retinaface",
             "--asd", "talknet", "--asr", "bodhan", "--align", "mms_fa",
             "--mt", "bodhan", "--mt-target-language", "hi",
             "--mt-target-script", "codemix", "--tts", "bodhan",
             "--voice-engine", "indic-speak", "--tts-style", "auto",
             "--mix", "--lipsync", "musetalk", "--lipsync-num-gpus", "2"])
print("preflight: argv parses")
PRE
}

say "=== CONFIG (the log has to explain the result, not just show progress) ==="
cat <<CFG
  model          musetalk (v15), --lipsync-num-gpus 2  (time-sharded)
  resolution     1920x1080  (both sources are 3840x2160 and are downscaled)
  duration       ${DUR}s each
  iit window     ${IIT_START}-$((IIT_START+DUR))s of $IIT_SRC
  batman window  ${BAT_START}-$((BAT_START+DUR))s of $BAT_SRC
  target lang    hi
  asr/mt/tts     bodhan / bodhan / bodhan  (+ --voice-engine indic-speak)
  order          SEQUENTIAL -- ~39 GiB RSS each, 48 GiB available
  projected      ~32 min lip-sync per video sharded (15.8x clip, halved)
  relaunch       bash scripts/run_4min_hi_musetalk.sh
CFG
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
free -g | head -2

cut_clip() {   # src start out
  local src="$1" start="$2" out="$3"
  if [ -s "$out" ]; then say "clip exists, reusing: $out"; return; fi
  say "cutting $out (${start}s +${DUR}s, scaling to 1080p)"
  "$FF" -y -v error -ss "$start" -i "$src" -t "$DUR" \
    -vf "scale=1920:1080:flags=lanczos" -r 25 \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
    -c:a aac -b:a 192k "$out"
  "$PY" -c "import sys;from backend.media_io import probe_duration;print('  ->',probe_duration('$out'),'s')"
}

run_one() {    # tag clip
  local tag="$1" clip="$2"
  say "=== PIPELINE START: $tag ==="
  local t0=$SECONDS
  # `--input`, not a positional path: the CLI takes exactly one input per
  # run and refuses extras outright (which is how this was caught in 2s
  # rather than after an hour of GPU time).
  "$PY" -m backend.cli --input "$clip" \
      --job-id "hi4min_${tag}_${TS}" \
      --model pyannote --resegment-turns --refine-speaker-labels \
      --face-detector retinaface --asd talknet \
      --asr bodhan --align mms_fa \
      --mt bodhan --mt-target-language hi --mt-target-script codemix \
      --tts bodhan --voice-engine indic-speak --tts-style auto \
      --mix --lipsync musetalk --lipsync-num-gpus 2 \
    && say "=== DONE: $tag in $(( (SECONDS-t0)/60 )) min ===" \
    || say "=== FAILED: $tag after $(( (SECONDS-t0)/60 )) min (rc=$?) ==="
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

preflight

cut_clip "$BAT_SRC" "$BAT_START" data/clips/batman_1080p_4min.mp4
cut_clip "$IIT_SRC" "$IIT_START" data/clips/iit_1080p_4min.mp4

run_one batman data/clips/batman_1080p_4min.mp4
run_one iit    data/clips/iit_1080p_4min.mp4

say "=== ALL DONE ==="
ls -la data/jobs/hi4min_*_"${TS}"/lipsync/ 2>/dev/null || true
