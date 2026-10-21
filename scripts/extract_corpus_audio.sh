#!/usr/bin/env bash
# 16 kHz mono WAV per language audio track, next to the source media.
# Needed by VAD, whisperx forced-align, and clip cutting; done once up front.
set -u
FF=/data1/dipak_data/miniconda3/envs/dubbing/bin/ffmpeg
cd /data1/dipak_data/dubbingstudio
for f in data/videos/*/*.webm data/videos/*/*.m4a; do
  [ -e "$f" ] || continue
  [ "$(stat -c%s "$f")" -gt 1048576 ] || { echo "skip (truncated): $f"; continue; }
  out="${f%.*}.16k.wav"
  if [ -s "$out" ]; then echo "= $(basename "$out")"; continue; fi
  echo "+ $(basename "$out")"
  "$FF" -v error -y -i "$f" -ac 1 -ar 16000 "$out" || echo "  !! failed: $f"
done
echo "done"
