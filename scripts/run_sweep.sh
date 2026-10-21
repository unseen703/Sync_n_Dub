#!/usr/bin/env bash
# Run the eight-methods suite over the golden set, one experiment per clip.
#
# Why hard links (data/exp/clips/<clip_id>.mp4): every golden clip's video is
# literally named clip.mp4, so `--input data/golden/<id>/clip.mp4` derives the
# base name "clip" and all 18 clips would collide in exp/clip/. experiment.py
# guards against that and refuses. A SYMLINK does not fix it either -- the
# media stage resolves it and the base reverts to "clip" while later stages
# look for the link's name. A hard link has no link to resolve.
#
# Tiers (data/golden/index.json):
#   A -> ranks the methods          (6-NRiom8K9Y, ZPUtA3W-7_I)
#   B -> generalization, separate table, never pooled   (qjPH9njnaVU)
#   C -> no Hindi dub: source-side component tests only. NOT swept here.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
export PATH=/data1/dipak_data/miniconda3/envs/dubbing/bin:$PATH   # whisperx shells out to bare ffmpeg
set -a; . ./.env; set +a                                          # pyannote needs HF_TOKEN

SUITE=${SUITE:-experiments/suites/eight_methods.json}
DATA_DIR=${DATA_DIR:-data/exp/sweep1}
TIERS=${TIERS:-A,B}
LOGDIR=logs/sweep
mkdir -p "$LOGDIR" data/exp/clips

# refresh the hard links (cheap, idempotent)
for d in data/golden/*/; do
  c=$(basename "$d"); [ -f "$d/clip.mp4" ] || continue
  [ -e "data/exp/clips/$c.mp4" ] || ln -f "$d/clip.mp4" "data/exp/clips/$c.mp4" 2>/dev/null \
    || cp "$d/clip.mp4" "data/exp/clips/$c.mp4"
done

mapfile -t CLIPS < <("$PY" - "$TIERS" <<'PYEOF'
import json, sys, pathlib
want = set(sys.argv[1].split(","))
idx = json.loads(pathlib.Path("data/golden/index.json").read_text())
rows = idx if isinstance(idx, list) else idx.get("clips", [])
for e in rows:
    if e.get("tier") in want:
        print(e["clip_id"])
PYEOF
)

echo "sweep: ${#CLIPS[@]} clip(s), tiers=$TIERS, suite=$SUITE, data-dir=$DATA_DIR"
ok=0; fail=0; failed=()
for c in "${CLIPS[@]}"; do
  mp4="data/exp/clips/$c.mp4"
  if [ ! -f "$mp4" ]; then echo "SKIP $c (no $mp4)"; continue; fi
  echo "--- $c"
  if "$PY" -m backend.experiment run --suite "$SUITE" --input "$mp4" \
        --data-dir "$DATA_DIR" --log-level INFO > "$LOGDIR/$c.log" 2>&1; then
    n=$(grep -c "variant .* failed" "$LOGDIR/$c.log" || true)
    if [ "$n" -gt 0 ]; then echo "    $n arm(s) failed -- see $LOGDIR/$c.log"; fi
    ok=$((ok+1))
  else
    echo "    SWEEP FAILED -- see $LOGDIR/$c.log"; fail=$((fail+1)); failed+=("$c")
  fi
done
echo "sweep: $ok ok, $fail failed"
[ "$fail" -gt 0 ] && printf 'failed clips: %s\n' "${failed[*]}"
exit 0
