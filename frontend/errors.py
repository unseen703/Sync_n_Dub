"""Turn a runner traceback into something a person can act on.

Matches the failure against the ones this pipeline produces and returns what
happened, whose problem it is and what to do next. Unrecognised failures fall
back to the exception's own last line. The raw log stays at
`GET /api/jobs/{id}/log`.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

#: (pattern, title, explanation, what to do). First match wins, so order is
#: most-specific first. Every entry is a failure actually seen on this host.
_RULES: List[Tuple[str, str, str, Optional[str]]] = [
    (r"requires HF_TOKEN",
     "Missing Hugging Face credentials",
     "Finding who speaks when uses a model that needs a Hugging Face token, "
     "and none was available to the job.",
     "Add HF_TOKEN to .env and accept the model's licence at "
     "hf.co/pyannote/speaker-diarization-community-1."),

    (r"no GPU is free|another process is using gpu",
     "The GPUs are busy right now",
     "Another job is using every graphics card, and lip-sync needs one to "
     "itself.",
     "Wait for the running job to finish and start this one again. Nothing "
     "was lost — the check runs before any work begins."),

    (r"CUDA out of memory|CUDA error: out of memory|OutOfMemoryError",
     "Ran out of GPU memory",
     "The lip-sync render needed more video memory than this machine's "
     "cards have free.",
     "Use a shorter section, pick the faster lip-sync model, or wait for "
     "the other job on the GPU to finish."),

    (r"\b429\b|rate.?limit|Too Many Requests",
     "The translation service is rate-limiting us",
     "Too many requests were sent to the hosted translation or speech API "
     "in a short window.",
     "Wait a few minutes and run it again. Nothing was lost — the job can "
     "be re-run from where it stopped."),

    (r"internal_error|50[0234] (Server|Bad|Service)",
     "The speech service is having an outage",
     "The hosted Bodhan endpoint returned a server error. This is an "
     "outage on their side, not a problem with the video or the settings.",
     "Try again later, or switch the voice engine to Edge, which needs no "
     "account."),

    (r"is not a valid|Sign in to confirm|Private video|Video unavailable",
     "That video could not be downloaded",
     "YouTube refused the download — the video may be private, age-gated, "
     "region-locked or removed.",
     "Check the link opens in a browser while signed out."),

    # 401 really is "not accessible"; 403 on a media URL usually is not — it
    # is YouTube rate-limiting the download host, and the same link fetches
    # fine minutes later. The downloader retries with backoff, so reaching
    # this card means the retries were exhausted too.
    (r"HTTP Error 403|Forbidden",
     "The video host refused the download",
     "YouTube returned 403 after the retries ran out. This is usually "
     "rate limiting rather than a private video — the metadata probe that "
     "found the title had already succeeded.",
     "Wait a few minutes and start it again. If it keeps failing, check "
     "the link opens in a browser while signed out."),

    (r"HTTP Error 401",
     "That video is not accessible",
     "The video host refused the request outright.",
     "Check the link is public."),

    (r"No space left on device|Disk quota",
     "The disk is full",
     "There was no room to write the job's audio or video.",
     "Free some space under data/jobs and run it again."),

    (r"ffmpeg|ffprobe",
     "A media conversion step failed",
     "One of the audio or video conversion steps did not complete.",
     "This usually means the source file is unusual. Try a different "
     "section of the video."),

    # `Stopped on signal N` is the string `frontend/runner.py`'s handler actually
    # raises; it does not write "SIGTERM". Without that alternative every Stop
    # click lands in the unrecognised fallback and shows as "The job failed".
    (r"KeyboardInterrupt|SIGTERM|SIGINT|cancelled|Stopped on signal \d+",
     "The job was stopped",
     "Someone cancelled this run, or the machine shut it down.",
     None),

    # Not what the adapter's own message says. insightface IS installed; the
    # cause is the import-order trap in CLAUDE.md §1, where cv2's bundled
    # libtbb.so.12 needs CXXABI_1.3.15 that the system libstdc++ lacks, so
    # whichever loads first decides whether the import works.
    # `./ui_run_job.sh` exports the preload, so a UI job hitting this
    # was started some other way.
    (r"RetinaFace needs either|No module named 'retinaface'|"
     r"CXXABI_1\.3\.15|libtbb\.so",
     "Face detection could not start",
     "The face detector could not load its library. insightface is "
     "installed, so this is almost always the C++ runtime load order, not "
     "a missing package.",
     "Start the job through ./ui_run_job.sh, which exports "
     "LD_PRELOAD=<env>/lib/libstdc++.so.6. Never launch frontend.runner with a "
     "bare Popen."),

    (r"stopped without reporting a result",
     "The job stopped unexpectedly",
     "The process that was running this job is gone and did not report why. "
     "That usually means the machine ran out of memory or was restarted.",
     "Running it again is safe."),
]


def explain(raw: Optional[str], *, state: Optional[str] = None
            ) -> Optional[Dict[str, Any]]:
    """A user-facing reading of a failure, or None when nothing is wrong.

    Returns `{title, detail, action, recognised, raw}`. `raw` is one line, not
    the whole log, so a "technical details" disclosure can show the original
    without fetching the file.
    """
    # A cancelled job is never a failure, whatever the error text matched:
    # the state is the operator's recorded intent and outranks a string match
    # on the exception that carried it out.
    if (state or "").lower() == "cancelled":
        return {"title": "The job was stopped",
                "detail": "This run was cancelled before it finished.",
                "action": "Starting it again runs from the beginning.",
                "recognised": True,
                "raw": _first_line(str(raw)) if raw else None}

    if not raw or not str(raw).strip():
        if state == "failed":
            return {"title": "The job failed",
                    "detail": "No reason was recorded.",
                    "action": "The log may say more.",
                    "recognised": False, "raw": None}
        return None

    text = str(raw).strip()
    for pattern, title, detail, action in _RULES:
        if re.search(pattern, text, re.I):
            return {"title": title, "detail": detail, "action": action,
                    "recognised": True, "raw": _first_line(text)}

    return {"title": "The job failed",
            "detail": _first_line(text),
            "action": None,
            "recognised": False,
            "raw": _first_line(text)}


def _first_line(text: str, limit: int = 300) -> str:
    """The most informative single line of a traceback.

    Python puts the exception last, so the last non-empty line is the one
    worth showing.
    """
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return ""
    for ln in reversed(lines):
        if not ln.startswith(("File \"", "Traceback", "  ")):
            return ln[:limit]
    return lines[-1][:limit]
