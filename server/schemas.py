"""Request bodies for the job-creation routes.

Deliberately thin: pydantic checks SHAPE only. Every check that needs to know
something about this install — supported languages, allowed hosts, whether the
window fits the video — lives in `server/routes/routes_create.py`, where it can produce a
sentence the operator can act on instead of a validation blob.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .presets import PRESET_IDS, resolve_preset

#: The shortest window worth a run. Below this diarization has nothing to
#: segment and the GPU stages spend longer loading models than rendering.
MIN_WINDOW_S = 2.0


class Window(BaseModel):
    """A half-open time range in the source video, in seconds."""
    model_config = ConfigDict(extra="forbid")

    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)

    @property
    def duration_s(self) -> float:
        return float(self.end_s) - float(self.start_s)


class ProbeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2048)


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Exactly one of `url` / `upload_id`. A URL goes through the host
    #: allowlist and yt-dlp; an upload was already staged and probed. The
    #: "exactly one" rule is enforced in `routes_create.create_job`, where the
    #: refusal can be a readable sentence.
    url: Optional[str] = Field(default=None, min_length=1, max_length=2048)
    upload_id: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    target_language: str = Field(min_length=2, max_length=16)
    #: What is SPOKEN in the source. None lets the ASR detect it, which is
    #: unreliable on short turns even though the adapter pins one answer for
    #: the file — a stated language beats a guessed one.
    source_language: Optional[str] = Field(default=None, min_length=2,
                                           max_length=16)
    #: Whole-dub delivery register (`tts_style`). One of
    #: `server.service.speakers.SUBMIT_STYLES`, or None for the engine's neutral read.
    style: Optional[str] = Field(default=None, max_length=64)
    window: Optional[Window] = None
    #: One flow (`presets.PRESET_ID`). Older ids are still accepted and map
    #: onto it, since a browser holding a stale bundle still posts them.
    preset: str = Field(default="dub")
    #: None means "the install default" (`presets.default_tts()`), not "no
    #: TTS" — a dub with no TTS is not a thing this form can ask for.
    tts: Optional[str] = None
    #: "auto" defers to the face probe; an explicit model overrides it; None
    #: takes the preset's choice.
    lipsync_model: Optional[str] = None
    num_speakers: Optional[int] = Field(default=None, ge=1, le=32)
    skip_stems: Optional[bool] = None
    #: UI "Code-mix" checkbox. True keeps English technical terms in Latin
    #: script (Bodhan `target_script=codemix`); False omits the argument so
    #: the server picks the script. None means "no choice sent" and the UI
    #: default (True) applies in `routes_create.create_job`.
    code_mix: Optional[bool] = None

    @field_validator("preset")
    @classmethod
    def _known_preset(cls, v: str) -> str:
        # Anything unrecognised is REJECTED, not coerced: a typo that quietly
        # produced a different run would only surface in the GPU bill.
        try:
            return resolve_preset(v)
        except ValueError as e:
            raise ValueError(str(e)) from None

    @field_validator("target_language", "tts", "lipsync_model")
    @classmethod
    def _tidy(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().lower() if isinstance(v, str) else v
