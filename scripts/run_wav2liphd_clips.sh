#!/bin/bash
# Three end-to-end Hindi dubs, each lip-synced with wav2lip_hd.
#
# End to end means the FULL pipeline per clip -- download/resolve, stems,
# diarization, 4c, faces, ASD, ASR, align, MT, register, voices, TTS, mix,
# lip-sync -- not a lip-sync pass over a pre-made dub. Stage 9h is
# wav2lip_hd, which is the existing wav2lip render (resize_factor=2, coord
# cache, pads 0,10,0,0 -- the current defaults, inherited by subclassing)
# followed by Real-ESRGAN x4plus + GFPGAN over every frame.
#
# Everything is written under data/: jobs to data/jobs/<id>/, logs beside
# the clips in data/gender_bench/cuts/hd_runs/logs/.
set -u

cd /data1/dipak_data/dubbingstudio
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python

C=data/gender_bench/cuts/hd_runs
L=$C/logs
mkdir -p "$L"

run_one () {
  local job="$1" clip="$2"
  local log="$L/${job}.log"
  echo "=== $(date -Is)  START $job  ($clip) ===" | tee -a "$L/summary.log"
  /usr/bin/time -v -o "$L/${job}.time" \
  $PY -m backend.cli \
    --input "$clip" \
    --job-id "$job" \
    -m pyannote --refine-speaker-labels \
    --face-detector retinaface --asd talknet \
    --asr bodhan --asr-language en \
    --align mms_fa \
    --mt bodhan --mt-target-language hi --mt-target-script codemix \
    --tts bodhan --tts-style auto \
    --mix \
    --lipsync wav2lip_hd \
    --lipsync-num-gpus 2 \
    > "$log" 2>&1
  local rc=$?
  echo "=== $(date -Is)  END   $job  rc=$rc ===" | tee -a "$L/summary.log"
  {
    echo "--- $job rc=$rc ---"
    grep -E "=== stage|register =|errors=|hd_enhance|FRAME PARITY|FAILED|ERROR" "$log" | tail -25
    echo "lipsync output:"
    ls -la "data/jobs/$job/lipsync/" 2>/dev/null | tail -5
  } >> "$L/summary.log"
  return $rc
}

echo "### wav2lip_hd batch starting $(date -Is)" > "$L/summary.log"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$L/summary.log"

run_one hd_bench1_40s            "$C/bench1_40s.mp4"
run_one hd_w4l11_40s             "$C/w4l11_40s.mp4"
run_one hd_iitprof_2900to3000    "$C/iitprof_2900to3000.mp4"

echo "### wav2lip_hd batch done $(date -Is)" >> "$L/summary.log"
