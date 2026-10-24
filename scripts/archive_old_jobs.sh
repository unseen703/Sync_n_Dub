#!/usr/bin/env bash
# Zip job folders older than N days, verify the archive, then delete them.
#
# Dry run by default. Nothing is deleted without --apply, and nothing is
# deleted at all unless `unzip -t` passes on the archive first.
set -euo pipefail

ROOT="/data1/dipak_data/dubbingstudio"
JOBS="$ROOT/data/jobs"
ARCHIVE_DIR="$ROOT/data/_archive"
DAYS="${DAYS:-4}"
MINUTES=$(( DAYS * 24 * 60 ))
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

[ -d "$JOBS" ] || { echo "no $JOBS"; exit 1; }
mkdir -p "$ARCHIVE_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
ZIP="$ARCHIVE_DIR/jobs_older_than_${DAYS}d_${STAMP}.zip"

# Older than DAYS, by folder mtime. -mmin is exact; -mtime +4 would actually
# mean "5 days or more" because find truncates to whole days.
mapfile -t CAND < <(find "$JOBS" -mindepth 1 -maxdepth 1 -type d -mmin "+$MINUTES" | sort)

TAKE=(); SKIP=()
for d in ${CAND[@]+"${CAND[@]}"}; do
  if grep -qs '"state"[[:space:]]*:[[:space:]]*"running"'  "$d/progress.json" \
  || grep -qs '"status"[[:space:]]*:[[:space:]]*"running"' "$d/manifest.json"; then
    SKIP+=("$d")          # still running: never touch it
  else
    TAKE+=("$d")
  fi
done

echo "older than ${DAYS}d: ${#CAND[@]}   to archive: ${#TAKE[@]}   skipped (running): ${#SKIP[@]}"
for d in ${SKIP[@]+"${SKIP[@]}"}; do echo "  SKIP (running)  $(basename "$d")"; done
if [ "${#TAKE[@]}" -eq 0 ]; then echo "nothing to do."; exit 0; fi
for d in "${TAKE[@]}"; do printf "  %8s  %s\n" "$(du -sh "$d" | cut -f1)" "$(basename "$d")"; done
echo "total: $(du -sch "${TAKE[@]}" | tail -1 | cut -f1)"

if [ "$APPLY" -ne 1 ]; then
  echo; echo "DRY RUN. Re-run with --apply to archive and delete."; exit 0
fi

echo; echo "zipping -> $ZIP"
( cd "$JOBS" && printf '%s\n' "${TAKE[@]#"$JOBS"/}" | zip -q -r -9 -@ "$ZIP" )

echo "verifying..."
unzip -tqq "$ZIP" || { echo "ARCHIVE FAILED VERIFICATION — nothing deleted"; exit 1; }

# Every folder must still be listed inside the archive before it is removed.
for d in "${TAKE[@]}"; do
  name="${d#"$JOBS"/}"
  unzip -l "$ZIP" "$name/*" >/dev/null 2>&1 \
    || { echo "MISSING FROM ARCHIVE: $name — nothing deleted"; exit 1; }
done

echo "archive OK ($(du -sh "$ZIP" | cut -f1)). deleting originals..."
for d in "${TAKE[@]}"; do
  case "$d" in "$JOBS"/*) rm -rf -- "$d"; echo "  removed $(basename "$d")";;
               *) echo "  REFUSING (outside $JOBS): $d";; esac
done
echo "done -> $ZIP"
