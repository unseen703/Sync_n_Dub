#!/usr/bin/env python3
"""Fetch the timing-experiment corpus: video, every language audio track, every subtitle.

Layout (project convention: per-asset folder named by the real video title):
    data/videos/<slug>_<video_id>/
        <id>.mp4                 video + English audio, <=720p
        <id>.<lang>.m4a          one per language-tagged audio track
        <id>.<lang>.vtt          manual subtitles
        <id>.<lang>-orig.vtt     auto captions (carry inline word timestamps)
        meta.json                provenance for every file
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.downloader import video_metadata   # noqa: E402
VIDEOS = ROOT / "data" / "videos"
YTDLP = "/data1/dipak_data/miniconda3/envs/dubbing/bin/yt-dlp"
# No system ffmpeg on this host; yt-dlp needs it pointed at explicitly for
# muxing and post-processing, or it silently leaves unmerged part files.
FFMPEG_DIR = "/data1/dipak_data/miniconda3/envs/dubbing/bin"
FF = ["--ffmpeg-location", FFMPEG_DIR]

CORPUS = {
    "6-NRiom8K9Y": "nptel_design_thinking_intro",
    "ZPUtA3W-7_I": "lex_narendra_modi",
    "qjPH9njnaVU": "lex_pavel_durov_telegram",
    "YrtANPtnhyg": "ted_sixthsense_pranav_mistry",
    "dIYDPtHWBNA": "nag_nesterov_accelerated_gradient_descent",
}

# Languages worth pulling audio for. Hindi is the experiment target; English is
# the source; the rest are free and enable extending past Hindi later.
AUDIO_LANGS = ["en", "en-US", "hi", "hi-Latn", "bn", "gu", "kn", "ml", "mr", "ta", "te"]
# Above this, fetch audio+subs only and cut video per clip window later.
VIDEO_FULL_MAX_S = 2400
# Anything smaller than this is a failed/truncated download, never a real track.
MIN_MEDIA_BYTES = 1 << 20


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("  $", " ".join(cmd[:6]), "...", flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(vid: str) -> dict:
    """Full metadata dump, so provenance comes from yt-dlp, not from guessing."""
    cp = run([YTDLP, "-J", "--skip-download", f"https://www.youtube.com/watch?v={vid}"])
    if cp.returncode != 0:
        raise RuntimeError(f"probe failed for {vid}: {cp.stderr[-2000:]}")
    return json.loads(cp.stdout)


def audio_tracks(info: dict) -> dict[str, dict]:
    """language -> {best audio-only format, dub provenance}.

    Provenance must be scanned across ALL formats for a language, not just the
    best one: YouTube only puts the "- dubbed" / "- original" marker in
    format_note on some variants (the m3u8 233-* set), so picking the
    highest-bitrate format alone silently reports every track as original.
    """
    best: dict[str, dict] = {}
    kinds: dict[str, str] = {}
    for f in info.get("formats", []):
        if f.get("vcodec") != "none" or f.get("acodec") == "none":
            continue
        lang = f.get("language")
        if not lang:
            continue
        note = f"{f.get('format_note') or ''} {f.get('format') or ''}"
        if "dubbed-auto" in note:
            kinds[lang] = "dubbed-auto"                       # YouTube machine dub
        elif "dubbed" in note and kinds.get(lang) != "dubbed-auto":
            kinds[lang] = "dubbed"                            # human / professional dub
        elif "original" in note and lang not in kinds:
            kinds[lang] = "original"
        abr = f.get("abr") or 0
        if lang not in best or abr > (best[lang].get("abr") or 0):
            best[lang] = f
    for lang, f in best.items():
        f["_dub_kind"] = kinds.get(lang, "unknown")
    return best


def fetch(vid: str, slug: str) -> dict:
    out = VIDEOS / f"{slug}_{vid}"
    out.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={vid}"
    print(f"\n{'='*70}\n{vid}  ->  {out.relative_to(ROOT)}\n{'='*70}", flush=True)

    info = probe(vid)
    tracks = audio_tracks(info)
    meta = {
        "video_id": vid,
        "title": info.get("title"),
        "channel": info.get("channel"),
        "duration_s": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "url": url,
        # Everything the probe knows that is not about the files themselves:
        # categories, description, tags, chapters, language. Shared with
        # `backend.downloader` so the two download paths cannot drift into
        # recording different things about the same video. The text fields
        # are uploader-authored -- see `video_metadata` before modelling on
        # them.
        "source_metadata": video_metadata(info),
        "audio_tracks": {},
        "subtitles": {},
        "files": {},
    }
    for lang, f in sorted(tracks.items()):
        meta["audio_tracks"][lang] = {
            "format_id": f.get("format_id"),
            "abr": f.get("abr"),
            "ext": f.get("ext"),
            # 'dubbed' = human/professional, 'dubbed-auto' = YouTube machine dub.
            "dub_kind": f["_dub_kind"],
        }
    meta["subtitles"] = {
        "manual": sorted(info.get("subtitles") or {}),
        "automatic": sorted(info.get("automatic_captions") or {}),
    }
    print(f"  title: {meta['title']}")
    print(f"  dur  : {meta['duration_s']}s   audio langs: {sorted(tracks)}")
    print(f"  cat  : {meta['source_metadata'].get('categories')}")

    # 1. canonical video: <=720p video + English audio, muxed to mp4.
    #    Skipped for long sources -- clip windows are cut later with
    #    --download-sections, which needs only the few minutes we use.
    mp4 = out / f"{vid}.mp4"
    full_video = (info.get("duration") or 0) <= VIDEO_FULL_MAX_S
    meta["video_strategy"] = "full" if full_video else "sections_on_demand"
    if mp4.exists() and mp4.stat().st_size <= MIN_MEDIA_BYTES:
        mp4.unlink()
    if not full_video:
        print(f"  ~ {meta['duration_s']}s > {VIDEO_FULL_MAX_S}s: deferring video to clip sections")
    elif not mp4.exists():
        cp = run([YTDLP, *FF, "-f",
                  "bv*[height<=720][ext=mp4]+ba[language^=en]/bv*[height<=720]+ba[language^=en]/b[height<=720]",
                  "--merge-output-format", "mp4", "-o", str(mp4), "--no-playlist", url])
        if cp.returncode != 0:
            print(f"  !! video failed: {cp.stderr[-800:]}", file=sys.stderr)
    else:
        print(f"  = {mp4.name} exists")

    # 2. each language audio track, separately
    for lang in AUDIO_LANGS:
        if lang not in tracks:
            continue
        existing = [p for p in out.glob(f"{vid}.{lang}.*")
                    if p.suffix in (".m4a", ".webm", ".opus")]
        # A 0-byte or truncated file is a failed download, not a cached one.
        good = [p for p in existing if p.stat().st_size > MIN_MEDIA_BYTES]
        for stale in set(existing) - set(good):
            print(f"  x {stale.name} is {stale.stat().st_size}B -- refetching")
            stale.unlink()
        if good:
            print(f"  = {vid}.{lang} audio exists")
            continue
        # native codec, no transcode: lossless and needs no post-processor
        cp = run([YTDLP, *FF, "-f", f"ba[language={lang}]",
                  "-o", str(out / f"{vid}.{lang}.%(ext)s"), "--no-playlist", url])
        if cp.returncode != 0:
            print(f"  !! audio {lang} failed: {cp.stderr[-400:]}", file=sys.stderr)

    # 3. Subtitles. Track keys must come from the probe, never guessed:
    #    this corpus has manual English under "en-LUU0EuDKgKo" (NPTEL Official)
    #    while a plain "en" request silently returns a Bengali translation.
    #    Manual tracks give trustworthy text; the per-audio-track auto captions
    #    carry the inline <00:00:00.378> word timestamps we need for timing GT.
    manual = list((info.get("subtitles") or {}).keys())
    auto = list((info.get("automatic_captions") or {}).keys())
    wanted = set(manual)
    for k in auto:
        # plain language codes only -- "hi-bn"/"en-gu" are machine translations
        # of another track and carry no independent timing.
        if k in ("en", "hi", "hi-Latn", "en-orig", "hi-orig", "en-US"):
            wanted.add(k)
    wanted = sorted(wanted)
    print(f"  subs: {len(manual)} manual + {len(wanted)-len(manual)} auto -> {wanted}")
    for i in range(0, len(wanted), 4):          # small batches; YouTube 429s on bursts
        batch = wanted[i:i + 4]
        for attempt in range(4):
            cp = run([YTDLP, *FF, "--skip-download", "--write-subs", "--write-auto-subs",
                      "--sub-langs", ",".join(batch), "--sub-format", "vtt",
                      "--sleep-requests", "1.5",
                      "-o", str(out / f"{vid}.%(ext)s"), "--no-playlist", url])
            if cp.returncode == 0:
                break
            if "429" in (cp.stderr or ""):
                wait = 30 * (attempt + 1)
                print(f"  .. 429 on {batch}; sleeping {wait}s")
                time.sleep(wait)
            else:
                print(f"  !! subs {batch}: {cp.stderr.strip().splitlines()[-1][:160]}", file=sys.stderr)
                break

    for p in sorted(out.iterdir()):
        if p.suffix == ".part" or re.search(r"\.f\d+(-\d+)?\.", p.name):
            p.unlink(); continue
        if p.is_file() and p.name != "meta.json":
            meta["files"][p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  -> {len(meta['files'])} files")
    return meta


if __name__ == "__main__":
    want = sys.argv[1:] or list(CORPUS)
    for vid in want:
        try:
            fetch(vid, CORPUS[vid])
        except Exception as e:
            print(f"!! {vid}: {e}", file=sys.stderr)
    print("\ndone")
