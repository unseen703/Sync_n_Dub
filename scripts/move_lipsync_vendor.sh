#!/usr/bin/env bash
# Move the lip-sync adapters' vendored upstreams out of third_party/ and
# under models/lipsync/vendor/, so nothing in models/ reaches outside it.
#
# Written 2026-10-23. ~21 GB moves: LatentSync 12G, MuseTalk 8.7G,
# Wav2Lip 517M. `mv` within one filesystem is a rename -- near-instant and
# it does not need 21 GB of free space. Across filesystems it is a copy
# plus a delete, which does. The preflight below checks for that case
# rather than discovering it half way through.
#
# The adapters already resolve EITHER location by existence
# (`models/lipsync/base.py::resolve_vendor_repo`), so the code is correct
# before and after this runs. That is deliberate: it means this script can
# be run, or not run, without a matching code change.
#
# Usage:
#   bash scripts/move_lipsync_vendor.sh --dry-run   # show what would move
#   bash scripts/move_lipsync_vendor.sh             # do it
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
DEST="$ROOT/models/lipsync/vendor"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# source-dir-under-third_party  ->  destination name
# src-under-third_party : dest-name : destination root
REPOS="LatentSync:latentsync:models/lipsync/vendor \
       MuseTalk:musetalk:models/lipsync/vendor \
       Wav2Lip:wav2lip:models/lipsync/vendor \
       TalkNet-ASD:talknet:models/asd/vendor"

echo "=== repo root : $ROOT"
echo "=== destination: $DEST"
echo

# --- preflight -------------------------------------------------------------
# A rename is free; a cross-device move is a 21 GB copy. Knowing which this
# is before starting is the difference between 2 seconds and 20 minutes on a
# disk that may not have the room.
src_dev=$(stat -c %d "$ROOT/third_party" 2>/dev/null || echo "?")
mkdir -p "$DEST"
dst_dev=$(stat -c %d "$DEST" 2>/dev/null || echo "?")
if [ "$src_dev" = "$dst_dev" ]; then
  echo "same filesystem -> each move is a rename (fast, no extra space)"
else
  echo "DIFFERENT filesystems -> each move COPIES then deletes."
  echo "Free space on the destination:"
  df -h "$DEST" | tail -1
  echo "You need ~21 GB free. Ctrl-C now if that is tight."
fi
echo

# --- the git hazard --------------------------------------------------------
# `third_party/` was ignored wholesale. `models/lipsync/vendor/` is ignored
# now too (.gitignore, 2026-10-23) and `*.pt` was added alongside it --
# LatentSync's UNet is a 4.7 GB `.pt` that only escaped git because of the
# third_party rule. Verify the ignore is live BEFORE moving anything, or a
# later `git add -A` writes a 4.7 GB blob into history that cannot be
# removed without rewriting it.
if ! git check-ignore -q "models/lipsync/vendor/x" 2>/dev/null \
   || ! git check-ignore -q "models/asd/vendor/x" 2>/dev/null; then
  echo "REFUSING: models/lipsync/vendor/ is not gitignored."
  echo "Add it to .gitignore first -- 21 GB is one 'git add -A' away from"
  echo "being committed, and that is not undoable without a history rewrite."
  exit 1
fi
echo "gitignore check: lipsync+asd vendor dirs ignored  OK"
echo

# --- move ------------------------------------------------------------------
for pair in $REPOS; do
  src_name="${pair%%:*}"
  rest="${pair#*:}"
  dst_name="${rest%%:*}"
  dst_root="${rest##*:}"
  src="$ROOT/third_party/$src_name"
  dst="$ROOT/$dst_root/$dst_name"
  mkdir -p "$ROOT/$dst_root"

  if [ ! -d "$src" ]; then
    echo "skip   $src_name  (not present)"
    continue
  fi
  if [ -e "$dst" ]; then
    echo "skip   $src_name  (destination $dst already exists)"
    continue
  fi
  size=$(du -sh "$src" 2>/dev/null | cut -f1)
  if [ "$DRY" = "1" ]; then
    echo "would  $src_name ($size)  ->  $dst_root/$dst_name"
  else
    echo "moving $src_name ($size)  ->  $dst_root/$dst_name"
    mv "$src" "$dst"
  fi
done

echo
if [ "$DRY" = "1" ]; then
  echo "dry run only; nothing moved."
  exit 0
fi

# --- verify ----------------------------------------------------------------
# The adapters are the thing that has to be right, so ask THEM rather than
# checking the paths by eye.
#
# ASD is checked too, since 2026-10-25. The list above moves TalkNet, so
# leaving the check at lip-sync meant the one ASD repo this script relocates
# was the one repo nothing verified afterwards. Light-ASD and LR-ASD are
# absent from REPOS on purpose -- both are cloned straight to
# models/asd/vendor/ and neither has ever been in third_party/ -- but they
# are still checked here, because "did you clone it" is the same question.
echo "=== what the adapters resolve now:"
PY="${PY:-/data1/dipak_data/miniconda3/envs/dubbing/bin/python}"
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, ".")
ok = True
for stage, lister, getter in (
    ("lipsync", "models.lipsync", "get_lipsync"),
    ("asd", "models.asd", "get_asd"),
):
    mod = __import__(lister, fromlist=["*"])
    get = getattr(mod, getter)
    for m in mod.available_models():
        try:
            a = get(m)
        except Exception as e:                       # noqa: BLE001
            ok = False
            print(f"  {stage:<8} {m:<14} ERROR   {type(e).__name__}: {e}")
            continue
        repo = getattr(a, "repo_path", None)
        exists = bool(repo and repo.exists())
        ok &= exists
        print(f"  {stage:<8} {m:<14} {'OK ' if exists else 'MISSING'} {repo}")
print("\nall adapters resolved" if ok else "\nSOME ADAPTERS CANNOT FIND THEIR REPO")
sys.exit(0 if ok else 1)
PYEOF
