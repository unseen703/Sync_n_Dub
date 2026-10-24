#!/usr/bin/env bash
# Re-fit a finished dub with ENGINE-SIDE TTS speed control, and compare.
#
#   bash scripts/run_tts_speed_test.sh [job_id] [mode] [bounds]
#
# Defaults to the 2026-10-21 svwrTOm55y4 run, oneshot, 0.85-1.40.
#
# Bodhan's rate field is an INFERENCE (CLAUDE.md): their docs list only
# model/input/voice/instructions, so the adapter sends `speed`, borrowed
# from the OpenAI /v1/audio/speech shape this endpoint otherwise matches.
# If that name is wrong the API answers 422. This script tries the three
# candidate names in turn and stops at the first that gets through, so one
# invocation settles a question that has been open since 2026-10-17.
set -uo pipefail
cd /data1/dipak_data/dubbingstudio

JOB="${1:-20261021-114009__svwrTOm55y4__pyannote.bodhan.mms_fa.bodhan.hi}"
MODE="${2:-oneshot}"
BOUNDS="${3:-0.85,1.40}"
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python

[ -d "data/jobs/$JOB" ] || { echo "no such job: data/jobs/$JOB"; exit 2; }

for FIELD in speed rate instructions.speed; do
  OUT="data/jobs/${JOB}__spd${MODE}"
  # The crashed/previous attempt leaves mt/ populated, and the script
  # copies the merged JSON into that path.
  rm -rf "$OUT"

  echo
  echo "============================================================"
  echo " BODHAN_TTS_RATE_FIELD=$FIELD   mode=$MODE   bounds=$BOUNDS"
  echo "============================================================"
  BODHAN_TTS_RATE_FIELD="$FIELD" \
    "$PY" scripts/test_tts_speed_control.py -j "$JOB" \
          --mode "$MODE" --bounds "$BOUNDS"
  rc=$?

  if [ $rc -eq 0 ]; then
    echo
    echo "OK with BODHAN_TTS_RATE_FIELD=$FIELD"
    echo "Put it in .env to keep it:  BODHAN_TTS_RATE_FIELD=$FIELD"
    exit 0
  fi
  echo "-- rc=$rc with field '$FIELD'; trying the next name"
done

echo
echo "All three field names failed. That is a real answer: the endpoint"
echo "may take no speed parameter at all, in which case engine-side rate"
echo "is not available and the only levers left are a shorter translation"
echo "or a higher --speed-cap."
exit 1
