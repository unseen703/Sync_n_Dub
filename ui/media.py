"""One answer to "what do I play?".

The player needs four things — original video, lip-synced video, original
audio, dubbed audio — and each lives in a different place depending on which
runner made the job and how far it got. The lip-synced render alone has five
possible filenames, because `run_lipsync` chains remux → restore → splice and
each step writes a new file:

    <stem>.<model>.lipsync.mp4            the bare render
    <stem>.<model>.lipsync.remux.mp4      when _needs_remux fired
    <stem>.<model>.<restore>.mp4          after face restoration
    <stem>.<model>.lipsync.windowed.mp4   after a window splice
    final.<lang>.<model>.mp4              the clip runner / flat scripts

Guessing wrong means showing the user a soft, un-restored render and calling
it the result. So the order here is: ask the record first (the manifest, then
the merged JSON, both of which name the authoritative output), and only fall
back to a ranked glob when there is no record.

Everything returned is a `/files/...` URL, which the existing route already
serves with HTTP range support.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("ui.media")

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")

#: Ranked best-first. A restored render supersedes a remuxed one, which
#: supersedes a bare one — the same order `run_lipsync` produces them in.
_LIPSYNC_RANK: tuple = (
    # `final.*.mp4` is the clip runner's and the flat scripts' DECLARED
    # output — it outranks everything, because the alternative is picking an
    # intermediate out of lipsync/ while the finished file sits at the root.
    "final.*.mp4",
    "*.codeformer.mp4", "*.gfpgan.mp4", "*.restoreformer.mp4",
    "*.lipsync.windowed.mp4",
    "*.lipsync.remux.mp4",
    "*.lipsync.mp4",
    "*spliced*.mp4",
)

#: Never offer these as "the" render: per-shot parts and multi-GPU shards are
#: intermediates, and showing one as the result is indistinguishable from a
#: render that only covered part of the video.
_LIPSYNC_EXCLUDE = ("shot", "part")

#: Substrings that mark a file as an intermediate whatever else it matches.
#: `spliced.noaudio.mp4` is the one that caught this out: it matched
#: `*spliced*.mp4`, so the player offered a SILENT video as the finished dub.
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


#: The shared helper answers None where this module's own copy answered `{}`;
#: both call sites below add `or {}` so `.get()` on the result still works.
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
        # Largest, not newest: `edit_agent._find_video` uses the same rule,
        # and a job folder can also hold a small preview cut.
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

    # 2. The merged JSON names it absolutely. Re-root by filename so a job
    #    folder that was moved or copied still resolves.
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
    return None


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
    """Returns (path, full_quality, note).

    When stems were skipped there is no 44.1 kHz stereo extraction and the
    only original audio on disk is the 16 kHz mono file made for diarization.
    That is perfectly serviceable for an A/B listen, but it is NOT the source
    audio and must not be presented as if it were — hence the flag and note.
    """
    audio = job_root / "audio"
    # Every candidate must be filtered for `dub`: the clip runner writes the
    # DUBBED track as `clip.dub.48k.stereo.wav`, which matches a naive
    # `*.48k.stereo.wav` glob. Offering that as "original audio" would play
    # the dub under the wrong label — the one failure here that actively
    # misleads rather than merely disappoints.
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
    dub_v = find_dubbed_video(job_root, manifest)
    dub_a = find_dubbed_audio(job_root, manifest)
    orig_a, full_q, note = find_original_audio(job_root, manifest)

    lip_info = ((merged.get("extra") or {}).get("lipsync") or {})
    poster = job_root / "meta.thumb.jpg"

    return {
        "poster_url": (_rel_url(job_root, poster, data_dir)
                       if poster.is_file() else None),
        "video": {
            "original": _ref(job_root, orig_v, data_dir, label="Original"),
            "lipsync": _ref(job_root, lip_v, data_dir, label="Lip-synced",
                            model=lip_info.get("model"),
                            window=lip_info.get("window")),
            "dubbed_no_lipsync": _ref(job_root, dub_v, data_dir,
                                      label="Dubbed (no lip-sync)"),
        },
        "audio": {
            "original": _ref(job_root, orig_a, data_dir, label="Original audio",
                             full_quality=full_q, note=note),
            "dubbed": _ref(job_root, dub_a, data_dir, label="Dubbed audio",
                           full_quality=True, note=None),
        },
    }
