#!/bin/bash
cd /data1/dipak_data/dubbingstudio
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
LOG="logs/captions_indic15_$(date +%Y%m%dT%H%M%S).log"
exec > "$LOG" 2>&1
LANGS="as,bn,gu,hi,kn,ml,mr,ne,or,pa,sa,sd,ta,te,ur"
echo "[$(date +%H:%M:%S)] ==== CONFIG ===="
echo "  task     : fetch YouTube auto-translated captions, 15 Indic languages"
echo "  videos   : svwrTOm55y4 (IIT profs), dIYDPtHWBNA (NAG deep learning)"
echo "  langs    : $LANGS"
echo "  window   : full-video VTT; 240-270s sliced later by the TTS driver"
echo "  strategy : one language per yt-dlp call, 8s gap, 30/60/90/120s backoff on 429"
echo "  log      : $LOG"
echo "[$(date +%H:%M:%S)] ==== START ===="
for ID in dIYDPtHWBNA svwrTOm55y4; do
  echo "[$(date +%H:%M:%S)] --- $ID ---"
  $PY -I scripts/fetch_indic_captions.py --video-id "$ID" --langs "$LANGS" \
      --out-dir "data/captions/$ID" --gap-s 8
  echo "[$(date +%H:%M:%S)] --- $ID rc=$? ---"
done
echo "[$(date +%H:%M:%S)] ==== DONE ===="
for ID in dIYDPtHWBNA svwrTOm55y4; do echo "$ID: $(ls data/captions/$ID/*.vtt 2>/dev/null | wc -l) vtt files"; done
