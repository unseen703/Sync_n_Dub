#!/usr/bin/env python
"""Fetch YouTube auto-translated captions, one language at a time, patiently.

The batched `--sub-langs a,b,c,d` form in the youtube-media-fetch skill hits a
hard HTTP 429 on the *first* language of every batch once the timedtext budget
is spent, and yt-dlp then writes nothing at all for that batch. The failure is
silent in the way that matters: rc=0 on the process, "Writing video subtitles
to: ..." on stdout, and no file on disk.

So this driver goes one language per invocation, sleeps between them, and backs
off 30/60/90/120 s on a 429 before retrying. It is idempotent -- a language whose
.vtt already exists and parses is skipped -- so it can be re-run until the set
is complete.

Usage:
    python -I scripts/fetch_indic_captions.py --video-id dIYDPtHWBNA \
        --langs as,bn,gu,hi,kn,ml,mr,ne,or,pa,sa,sd,ta,te,ur \
        --out-dir data/captions/dIYDPtHWBNA
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

YTDLP = "/data1/dipak_data/miniconda3/envs/dubbing/bin/yt-dlp"
FFDIR = "/data1/dipak_data/miniconda3/envs/dubbing/bin"

BACKOFFS = [30, 60, 90, 120]


def _is_good(vtt: Path) -> bool:
    """A .vtt counts as fetched only if it parses into at least one cue.

    A truncated or empty file is a failure, not a cache -- re-running must
    replace it rather than skip it.
    """
    if not vtt.exists() or vtt.stat().st_size < 200:
        return False
    try:
        from backend.transcripts import parse_vtt
        return len(parse_vtt(vtt)) > 0
    except Exception:
        return False


def fetch_one(video_id: str, lang: str, out_dir: Path, *, gap_s: float) -> bool:
    dest = out_dir / f"{video_id}.{lang}.vtt"
    if _is_good(dest):
        print(f"  skip {lang:3s}  (already on disk, parses)", flush=True)
        return True

    url = f"https://www.youtube.com/watch?v={video_id}"
    for attempt in range(len(BACKOFFS) + 1):
        p = subprocess.run(
            [YTDLP, "--ffmpeg-location", FFDIR, "--skip-download",
             "--write-subs", "--write-auto-subs",
             "--sub-langs", lang, "--sub-format", "vtt",
             "--sleep-requests", "2",
             "-o", str(out_dir / "%(id)s.%(ext)s"), "--no-playlist", url],
            capture_output=True, text=True,
        )
        blob = (p.stdout or "") + (p.stderr or "")
        if _is_good(dest):
            n = len(__import__("backend.transcripts", fromlist=["parse_vtt"]).parse_vtt(dest))
            print(f"  ok   {lang:3s}  {dest.stat().st_size:7d} B, {n} cues", flush=True)
            time.sleep(gap_s)
            return True
        if "429" in blob:
            if attempt < len(BACKOFFS):
                wait = BACKOFFS[attempt]
                print(f"  429  {lang:3s}  backoff {wait}s "
                      f"(attempt {attempt + 1}/{len(BACKOFFS)})", flush=True)
                time.sleep(wait)
                continue
            print(f"  FAIL {lang:3s}  429 after {len(BACKOFFS)} backoffs", flush=True)
            return False
        # Anything that is not a 429 will not be fixed by waiting -- log the
        # last stderr line and move on so one dead language does not stall the
        # rest of the set.
        last = [l for l in blob.splitlines() if l.strip()]
        print(f"  FAIL {lang:3s}  {last[-1][:160] if last else 'no output'}", flush=True)
        return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--langs", required=True,
                    help="comma-separated ISO codes")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--gap-s", type=float, default=8.0,
                    help="sleep between successful fetches")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    print(f"[captions] {args.video_id}: {len(langs)} languages -> {out_dir}",
          flush=True)
    ok, bad = [], []
    for lang in langs:
        (ok if fetch_one(args.video_id, lang, out_dir, gap_s=args.gap_s)
         else bad).append(lang)

    print(f"\n[captions] {args.video_id}  OK({len(ok)})={ok}", flush=True)
    print(f"[captions] {args.video_id} BAD({len(bad)})={bad}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
