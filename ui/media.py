"""Resolve which files the player should offer for a job.

Finds the original video, the lip-synced render, and the original and dubbed
audio. The render has several possible names, so the manifest and merged JSON
are consulted before falling back to a ranked glob. Everything returned is a
`/files/...` URL.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("ui.media")

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")

#: Ranked best-first: a restored render beats a remuxed one, which beats a
#: bare one — the order `run_lipsync` produces them in.
_LIPSYNC_RANK: tuple = (
    # The clip runner's and flat scripts' declared output, so it outranks
    # everything; otherwise an intermediate in lipsync/ wins over the finished
    # file at the root.
    "final.*.mp4",
    "*.codeformer.mp4", "*.gfpgan.mp4", "*.restoreformer.mp4",
    "*.lipsync.windowed.mp4",
    "*.lipsync.remux.mp4",
    "*.lipsync.mp4",
    "*spliced*.mp4",
)

#: Never offer these as "the" render: per-shot parts and multi-GPU shards look
#: exactly like a render that only covered part of the video.
_LIPSYNC_EXCLUDE = ("shot", "part")

#: Substrings marking a file as an intermediate whatever else it matches.
#: Without these, `spliced.noaudio.mp4` matches `*spliced*.mp4` and the player
#: offers a silent video as the finished dub.
_INTERMEDIATE = ("noaudio", "no_audio", "temp", "tmp.", ".src.")


def _rel_url(job_root: Path, p: Path, data_dir: Path) -> str:
    try:
        rel = p.resolve().relative_to(data_dir.resolve())
    except ValueError:
        rel = Path("jobs") / job_root.name / p.name
    return "/files/" + rel.as_posix()


def _ref(job_root: Path, p: Optional[Path], data_dir: Path, *,
         label: str, **extra) -> Optional[Dict[str, Any]]:
    if p is None or not Path(p).is_file():
        return None
    p = Path(p)
    url = _rel_url(job_root, p, data_dir)
    out = {"url": url, "download_url": url + "?download=1", "label": label,
           "name": p.name, "bytes": p.stat().st_size}
    out.update(extra)
    return out


#: Answers None, not `{}`, so both call sites below add `or {}`.
from ._io import read_json as _read_json


def _excluded(p: Path) -> bool:
    n = p.name.lower()
    return (any(n.startswith(x) for x in _LIPSYNC_EXCLUDE)
            or any(x in n for x in _INTERMEDIATE)
            or ".shards" in p.parent.name.lower())


def _is_dub(p: Path) -> bool:
    """True for a file whose name marks it as dubbed output."""
    n = p.name.lower()
    return "dub" in n or "revoiced" in n or ".tts." in n


def _newest(paths: List[Path]) -> Optional[Path]:
    paths = [p for p in paths if p.is_file()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# per-artifact resolution
# ---------------------------------------------------------------------------

def find_original_video(job_root: Path, manifest: Dict[str, Any]) -> Optional[Path]:
    rec = (manifest.get("result") or {}).get("video")
    if rec:
        p = job_root / rec
        if p.is_file():
            return p
    vids = [p for p in (job_root / "videos").glob("*")
            if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    if vids:
        # Largest, not newest — a job folder can also hold a preview cut.
        # `edit_agent._find_video` uses the same rule.
        return max(vids, key=lambda p: p.stat().st_size)
    return None


def find_lipsync_video(job_root: Path, manifest: Dict[str, Any],
                       merged: Dict[str, Any]) -> Optional[Path]:
    # 1. The manifest names it, job-relative.
    rec = (manifest.get("result") or {}).get("lipsync_mp4")
    if rec:
        p = job_root / rec
        if p.is_file():
            return p

    # 2. The merged JSON names it absolutely. Re-root by filename so a moved
    #    or copied job folder still resolves.
    out = ((merged.get("extra") or {}).get("lipsync") or {}).get("out_video")
    if out:
        p = Path(out)
        if p.is_file():
            return p
        for cand in (job_root / "lipsync" / p.name, job_root / p.name):
            if cand.is_file():
                return cand

    # 3. No record — rank what is on disk.
    for pattern in _LIPSYNC_RANK:
        hits = [p for p in job_root.glob(pattern) if not _excluded(p)]
        hits += [p for p in (job_root / "lipsync").glob(pattern)
                 if not _excluded(p)]
        best = _newest(hits)
        if best is not None:
            return best

    # 4. Nothing current — check archived renders from previous runs.
    prev = job_root / "lipsync" / "prev"
    if prev.is_dir():
        # Newest archive subfolder first; timestamped names sort correctly.
        for sub in sorted(prev.iterdir(), reverse=True):
            if not sub.is_dir():
                continue
            for pattern in _LIPSYNC_RANK:
                hits = [p for p in sub.glob(pattern) if not _excluded(p)]
                best = _newest(hits)
                if best is not None:
                    return best
    return None


def find_enhanced_video(job_root: Path) -> tuple:
    """Stage 9i's output: `(path, warning)`, either may be None.

    `enhance/STATUS.json` is consulted FIRST, because it is what the rest of
    the enhance code already treats as the record — re-deriving the path by
    glob would be a second implementation that can disagree with the panel
    sitting next to the player.

    The warning covers the one outcome that would otherwise mislead. 9i
    writes video-only and remuxes the render's audio on afterwards; when
    that remux fails, `backend.restore.run_restore` deliberately keeps the
    SILENT intermediate rather than losing the pass, and returns it as the
    result. A silent file offered under a bare "Enhanced" label plays as a
    broken dub, so it is labelled as what it is.

    Shards are not a hazard here -- `_excluded` already drops anything whose
    parent directory ends `.shards` -- but the two finished names differ by
    one infix, so `*.enhanced.mp4` is matched before the silent one rather
    than both falling out of a single glob by mtime.
    """
    d = job_root / "enhance"
    if not d.is_dir():
        return None, None

    silent_note = ("This enhanced video has no audio — the pass finished but "
                   "the audio remux failed. The lip-synced render still has "
                   "its dub.")

    rec = (_read_json(d / "STATUS.json") or {})
    if rec.get("state") == "done":
        p = Path(rec.get("video") or "")
        if p.is_file() and p.stat().st_size:
            return p, (silent_note if ".silent." in p.name else None)

    best = _newest([p for p in d.glob("*.enhanced.mp4")
                    if not _excluded(p)])
    if best is not None:
        return best, None
    best = _newest([p for p in d.glob("*.enhanced.silent.mp4")
                    if not _excluded(p)])
    if best is not None:
        return best, silent_note
    return None, None


def find_dubbed_video(job_root: Path, manifest: Dict[str, Any]) -> Optional[Path]:
    """The dub muxed onto the ORIGINAL video — no lip-sync."""
    rec = (manifest.get("result") or {}).get("mix_mp4")
    if rec and (job_root / rec).is_file():
        return job_root / rec
    return _newest(list((job_root / "mix").glob("*.mp4")))


def find_dubbed_audio(job_root: Path, manifest: Dict[str, Any]) -> Optional[Path]:
    rec = (manifest.get("result") or {}).get("mix_wav")
    if rec and (job_root / rec).is_file():
        return job_root / rec
    for pat, sub in (("*.dubbed.wav", "mix"), ("*.wav", "mix"),
                     ("*dub*.wav", "audio")):
        best = _newest(list((job_root / sub).glob(pat)))
        if best is not None:
            return best
    return None


def find_original_audio(job_root: Path, manifest: Dict[str, Any]
                        ) -> tuple:
    """Returns `(path, full_quality, note)`.

    With stems skipped there is no stereo extraction and the only original
    audio on disk is the 16 kHz mono file made for diarization. Fine for an
    A/B listen, but not the source audio — hence the flag and note.
    """
    audio = job_root / "audio"
    # Every candidate is filtered for `dub`: the clip runner writes the DUBBED
    # track as `clip.dub.48k.stereo.wav`, which a naive `*.48k.stereo.wav` glob
    # would offer as the original.
    stereo = _newest([p for p in (list(audio.glob("*.44k.stereo.wav"))
                                  + list(audio.glob("*.48k.stereo.wav")))
                      if not _is_dub(p)])
    if stereo is not None:
        return stereo, True, None

    rec = (manifest.get("result") or {}).get("audio_16k")
    mono = (job_root / rec) if rec and (job_root / rec).is_file() else None
    if mono is None:
        mono = _newest([p for p in (list(audio.glob("*.16k.mono.wav"))
                                    + list(audio.glob("*en*.wav")))
                        if not _is_dub(p)])
    if mono is not None:
        return mono, False, ("16 kHz mono — extracted for speech processing. "
                             "The full-quality original is inside the video.")
    return None, False, None


# ---------------------------------------------------------------------------
# the endpoint payload
# ---------------------------------------------------------------------------

def resolve(job_root: Path, *, data_dir: Path,
            merged_json: Optional[Path] = None) -> Dict[str, Any]:
    job_root = Path(job_root)
    manifest = _read_json(job_root / "manifest.json") or {}
    merged = ((_read_json(merged_json) or {})
              if merged_json and Path(merged_json).is_file() else {})

    orig_v = find_original_video(job_root, manifest)
    lip_v = find_lipsync_video(job_root, manifest, merged)
    enh_v, enh_warning = find_enhanced_video(job_root)
    dub_v = find_dubbed_video(job_root, manifest)
    dub_a = find_dubbed_audio(job_root, manifest)
    orig_a, full_q, note = find_original_audio(job_root, manifest)

    lip_info = ((merged.get("extra") or {}).get("lipsync") or {})
    poster = job_root / "meta.thumb.jpg"

    # Detect a previous render standing in for the current one.
    lip_warning: Optional[str] = None
    if lip_v is not None:
        try:
            prev_dir = job_root / "lipsync" / "prev"
            if lip_v.resolve().is_relative_to(prev_dir.resolve()):
                lip_warning = ("The latest lip-sync run failed. "
                               "Showing the previous render.")
        except (OSError, ValueError):
            pass

    return {
        "poster_url": (_rel_url(job_root, poster, data_dir)
                       if poster.is_file() else None),
        "video": {
            "original": _ref(job_root, orig_v, data_dir, label="Original"),
            "lipsync": _ref(job_root, lip_v, data_dir, label="Lip-synced",
                            model=lip_info.get("model"),
                            window=lip_info.get("window")),
            # Stage 9i. Never part of a submission -- it cannot run until a
            # render exists -- so its presence is the whole readiness
            # signal. 9i writes no `progress.json`, which means there is no
            # milestone to gate this on and the FILE is the only arbiter.
            "enhanced": _ref(job_root, enh_v, data_dir, label="Enhanced"),
            "dubbed_no_lipsync": _ref(job_root, dub_v, data_dir,
                                      label="Dubbed (no lip-sync)"),
            "lipsync_warning": lip_warning,
            "enhanced_warning": enh_warning,
        },
        "audio": {
            "original": _ref(job_root, orig_a, data_dir, label="Original audio",
                             full_quality=full_q, note=note),
            "dubbed": _ref(job_root, dub_a, data_dir, label="Dubbed audio",
                           full_quality=True, note=None),
        },
    }
