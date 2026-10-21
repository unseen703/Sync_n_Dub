#!/bin/sh
# Full clip run, backend.cli only. MUSETALK_PYTHON is exported because it is
# absent from .env (now documented in .env.example); load_dotenv uses
# override=False, so an exported value wins and .env needs no edit to test.
set -eu
cd /data1/dipak_data/dubbingstudio
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python

export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6
export CUDA_VISIBLE_DEVICES=0
export MUSETALK_PYTHON=/data1/dipak_data/miniconda3/envs/musetalk/bin/python

exec "$PY" -m backend.cli \
  --input data/videos/iitm_w1l2_genai_intro_problem_setting_HUunmwZfGzc/HUunmwZfGzc_60to100.mp4 \
  --job-id w1l2_60to100_hi_musetalk_cli \
  --asr bodhan --asr-language en \
  --align mms_fa \
  --mt bodhan --mt-target-language hi --mt-target-script codemix \
  --tts bodhan --tts-style auto \
  --refine-speaker-labels \
  --face-detector retinaface --asd talknet \
  --mix \
  --lipsync musetalk --lipsync-batch-size 10 --lipsync-num-gpus 2
