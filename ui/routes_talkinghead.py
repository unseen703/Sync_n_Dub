"""Routes for the Talking Head page.

Mounted the same way as the other routers — ABOVE the StaticFiles mount in
`ui/server.py`, or Starlette's registration-order matching shadows every one
of these behind the static handler (404 on GET, 405 on POST) with nothing in
`app.routes` to show for it.

Three calls, in the order the page uses them:

  POST /api/talking-head/photo      a picture -> is there a face?
  GET  /api/talking-head/options    languages + categories for the dropdowns
  GET  /api/talking-head/dialogues  the lines for one language
  POST /api/talking-head/render     animate it (refuses until a clip exists)

The render call is deliberately a REFUSAL today. Every catalogue entry ships
`audio_url: null`, and a talking head with no audio is not a smaller version
of the feature -- it is a still frame. Saying so with a 409 and the reason
beats queueing a job that cannot produce anything.
"""
from __future__ import annotations

import logging
import sys
import uuid as _uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import (APIRouter, Depends, File, HTTPException, UploadFile,
                     status)
from fastapi.concurrency import run_in_threadpool

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:                      # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT))

from .auth import basic_auth                            # noqa: E402
from . import talking_head as TH                        # noqa: E402

log = logging.getLogger("ui.talkinghead")
router = APIRouter()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

#: Pictures only. The dubbing uploader's extension list is for video, and
#: reusing it here would accept an .mp4 for a page that cannot use one.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

#: A portrait is not a feature film. 40 MB is generous for a photo and keeps
#: a loopback UI from filling the disk on a mis-drag.
MAX_IMAGE_MB = 40


def _bad(msg: str) -> HTTPException:
    return HTTPException(status.HTTP_400_BAD_REQUEST, msg)


@router.get("/api/talking-head/options")
def options() -> Dict[str, Any]:
    """Dropdown contents.

    Languages come from `ui.langs`, which derives them from the voice
    catalog — not from a list written here. A hand-kept copy is how the tone
    picker once shipped values the engine 422s on.
    """
    from .langs import LANGUAGES, options as lang_options

    have = set(TH.languages_with_dialogues())
    langs = []
    seen = set()
    for row in lang_options("bodhan"):
        seen.add(row["code"])
        langs.append({**row, "n_dialogues": len(TH.dialogues_for(row["code"])),
                      "has_dialogues": row["code"] in have})
    # A language the CATALOGUE has lines for but the TTS catalog does not.
    # This page plays a supplied clip; it never synthesises, so it is not
    # bound to the voice table the dub form is bound to. English is exactly
    # this case: a valid engine `lang` with no voice RECORDED in it, so
    # `languages_for("bodhan")` leaves it out — and leaving it out of here
    # too would hide lines that are sitting in the file, ready to play.
    for code in sorted(have - seen):
        meta = LANGUAGES.get(code) or {}
        langs.append({"code": code,
                      "english": meta.get("english") or code.upper(),
                      "native": meta.get("native") or "",
                      "n_dialogues": len(TH.dialogues_for(code)),
                      "has_dialogues": True})
    return {
        "languages": langs,
        "categories": TH.categories(),
        "note": TH.ANIMATION_NOTE,
        "max_image_mb": MAX_IMAGE_MB,
    }


@router.get("/api/talking-head/dialogues")
def dialogues(language: str) -> Dict[str, Any]:
    rows = TH.dialogues_for(language)
    return {
        "language": language,
        "categories": TH.categories(),
        "dialogues": rows,
        "n_playable": sum(1 for r in rows if r["playable"]),
        # Said plainly rather than leaving an empty list to be interpreted.
        "message": (None if rows else
                    f"No lines for {language!r} yet. Add them to "
                    f"data/talking_head/dialogues.json — no code change "
                    f"is needed."),
    }


@router.post("/api/talking-head/photo",
             status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(basic_auth)])
async def upload_photo(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Stage a photo and report whether it holds a usable face.

    The face check is the page's gate, so it happens HERE rather than at
    render time: finding out that a picture has no face after a GPU job has
    been queued is the expensive way to learn it.
    """
    name = (file.filename or "").lower()
    ext = next((e for e in IMAGE_EXTS if name.endswith(e)), None)
    if ext is None:
        raise _bad(f"That is not an image. Accepted: "
                   f"{', '.join(IMAGE_EXTS)}")

    staging = DATA_DIR / "uploads" / "talking_head"
    staging.mkdir(parents=True, exist_ok=True)
    photo_id = _uuid.uuid4().hex
    dest = staging / f"{photo_id}{ext}"

    cap = MAX_IMAGE_MB * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > cap:
                    raise _Big()
                fh.write(chunk)
    except _Big:
        dest.unlink(missing_ok=True)
        raise _bad(f"That image is larger than the {MAX_IMAGE_MB} MB limit.")
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if written == 0:
        dest.unlink(missing_ok=True)
        raise _bad("That upload was empty.")

    probe = await run_in_threadpool(TH.probe_image, dest)
    if not probe.get("ok"):
        # Keep the file: the message explains what was wrong and the user
        # may want to see the picture they sent. Staging is swept, not
        # permanent.
        log.info("talking head: %s rejected (%s)", dest.name,
                 probe.get("reason"))
    try:
        rel = dest.resolve().relative_to(DATA_DIR.resolve())
        url = "/files/" + rel.as_posix()
    except ValueError:                                  # pragma: no cover
        url = None
    return {"photo_id": photo_id, "url": url, "bytes": written, **probe}


@router.post("/api/talking-head/render", dependencies=[Depends(basic_auth)])
def render(body: Dict[str, Any]) -> Dict[str, Any]:
    """Animate the photo against a chosen line.

    Refuses while the catalogue has no audio. The renderer needs a waveform:
    MuseTalk's whisper front-end drives the mouth from it, so without a clip
    there is nothing to animate TO and the output would be the input frame
    repeated. A 409 naming the entry is more useful than a job that runs and
    returns a still.
    """
    photo_id = str((body or {}).get("photo_id") or "").strip()
    dialogue_id = str((body or {}).get("dialogue_id") or "").strip()
    language = str((body or {}).get("language") or "").strip()
    if not photo_id or not dialogue_id:
        raise _bad("photo_id and dialogue_id are both required.")

    rows = {r["id"]: r for r in TH.dialogues_for(language)}
    row = rows.get(dialogue_id)
    if row is None:
        raise HTTPException(404, f"No dialogue {dialogue_id!r} for "
                                 f"language {language!r}.")
    if not row.get("playable"):
        raise HTTPException(
            409,
            f"{dialogue_id!r} has no audio yet. The line is in the "
            f"catalogue but `audio_url` is still null, and MuseTalk drives "
            f"the mouth from a waveform — without one the render would be "
            f"the photo, unchanged. Add the link in "
            f"data/talking_head/dialogues.json.")

    # Reached only once a clip exists. Left unimplemented on purpose rather
    # than half-built against a shape the real clips have not fixed yet.
    raise HTTPException(
        501, "Audio is present but the render path is not wired up yet.")


class _Big(Exception):
    pass
