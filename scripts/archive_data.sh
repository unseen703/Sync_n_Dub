#!/usr/bin/env bash
# Archive every data/ folder except data/jobs into compressed tarballs.
#
# The pipeline now writes each run self-contained under data/jobs/<job_id>/,
# so the old per-stage folders (videos/, audio/, stems/, mt/, ...) are dead
# weight. This packs each one into data/_archive/<name>.tar.zst, VERIFIES the
# tarball (integrity + non-directory entry count), and only then removes the
# original. Legacy manifest-only job folders are packed as jobs_legacy.tar.zst
# so data/jobs/ starts clean for the new layout.
#
# Safe to re-run: a folder whose tarball already exists is skipped.
set -euo pipefail

DATA_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}"
ARCHIVE_DIR="$DATA_DIR/_archive"
KEEP="jobs _archive"
ZSTD_LEVEL="${ZSTD_LEVEL:-6}"
THREADS="${THREADS:-$(nproc)}"
# Tarballs at or above this size are dropped rather than kept: they are bulk
# media (mp4/wav) that barely compresses, and keeping them defeats the point
# of archiving. 0 disables the rule.
MAX_ARCHIVE_BYTES="${MAX_ARCHIVE_BYTES:-$((500 * 1024 * 1024))}"

mkdir -p "$ARCHIVE_DIR"
log() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }

# Pack $1 (a directory) into $2 (a .tar.zst), verify, then delete $1.
# $3 is the -C base directory tar changes into, $4 the name it stores.
pack() {
    local src="$1" out="$2" base="$3" name="$4"
    local src_count tar_count

    if [[ -e "$out" ]]; then
        log "SKIP  $name (archive already exists)"
        return 0
    fi

    src_count=$(find "$src" ! -type d | wc -l)
    log "PACK  $name ($(du -sh "$src" | cut -f1), $src_count files)"

    # Write to .part so an interrupted run never leaves a tarball that the
    # "already exists" check would trust.
    tar --use-compress-program="zstd -$ZSTD_LEVEL -T$THREADS" \
        -cf "$out.part" -C "$base" "$name"

    # Verify 1: the compressed stream decodes end to end.
    zstd -q -t "$out.part"

    # Verify 2: every non-directory entry made it in.
    tar_count=$(tar --use-compress-program="zstd -d -T$THREADS" \
        -tf "$out.part" | grep -cv '/$' || true)
    if [[ "$tar_count" -ne "$src_count" ]]; then
        log "FAIL  $name: $src_count files on disk vs $tar_count in archive — keeping original"
        mv "$out.part" "$out.MISMATCH"
        return 1
    fi

    mv "$out.part" "$out"
    rm -rf "$src"

    # Oversized archives are bulk media; drop them instead of hoarding them.
    local out_bytes
    out_bytes=$(stat -c %s "$out")
    if (( MAX_ARCHIVE_BYTES > 0 && out_bytes >= MAX_ARCHIVE_BYTES )); then
        rm -f "$out"
        log "DROP  $name archive was $(numfmt --to=iec "$out_bytes") (>= $(numfmt --to=iec "$MAX_ARCHIVE_BYTES")) — deleted, not kept"
        return 0
    fi
    log "DONE  $name -> $(basename "$out") ($(du -sh "$out" | cut -f1))"
}

failed=()

# 1. Legacy job folders: pack the whole of data/jobs, then recreate it empty.
if [[ -d "$DATA_DIR/jobs" ]] && [[ -n "$(ls -A "$DATA_DIR/jobs" 2>/dev/null)" ]]; then
    staged="$DATA_DIR/jobs_legacy"
    rm -rf "$staged"
    mkdir -p "$staged"
    # Move the old per-run folders aside so data/jobs survives as an empty dir.
    find "$DATA_DIR/jobs" -mindepth 1 -maxdepth 1 -exec mv -t "$staged" {} +
    pack "$staged" "$ARCHIVE_DIR/jobs_legacy.tar.zst" "$DATA_DIR" "jobs_legacy" \
        || failed+=("jobs_legacy")
fi
mkdir -p "$DATA_DIR/jobs"

# 2. Every other top-level data folder.
for path in "$DATA_DIR"/*/; do
    name=$(basename "$path")
    case " $KEEP " in *" $name "*) log "KEEP  $name"; continue ;; esac
    pack "${path%/}" "$ARCHIVE_DIR/$name.tar.zst" "$DATA_DIR" "$name" \
        || failed+=("$name")
done

echo
log "archive dir: $ARCHIVE_DIR ($(du -sh "$ARCHIVE_DIR" | cut -f1))"
if (( ${#failed[@]} )); then
    log "FAILED (originals kept): ${failed[*]}"
    exit 1
fi
log "all folders archived and originals removed"
