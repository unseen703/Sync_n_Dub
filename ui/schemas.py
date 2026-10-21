"""Request bodies for the job-creation routes.

Deliberately thin. Pydantic checks *shape* here — a number is a number, a
preset is one of three strings — and nothing else. Every check that needs to
know something about this install (which languages a TTS engine can speak,
which hosts yt-dlp may be pointed at, whether the window fits inside the
video, whether the lip-synced portion is under the compute cap) lives in
`ui/routes_create.py`, where it can produce a sentence the operator can act
on instead of a validation-error blob.
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
    #: allowlist and yt-dlp; an upload was already staged and probed by
    #: `POST /api/uploads` and carries neither. `routes_create.create_job`
    #: enforces the "exactly one" rule, because the message it produces
    #: ("give a link or upload a file, not both") is worth more than a
    #: pydantic validation blob.
    url: Optional[str] = Field(default=None, min_length=1, max_length=2048)
    upload_id: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    target_language: str = Field(min_length=2, max_length=16)
    #: What is SPOKEN in the source. None means "let the ASR detect it",
    #: which on short diarization turns is unreliable: an English lecture
    #: came back on 2026-10-17 transcribed into Telugu, Gujarati, Malayalam
    #: and English across nine turns. The adapter now detects once and pins
    #: it, but a stated language beats a guessed one.
    source_language: Optional[str] = Field(default=None, min_length=2,
                                           max_length=16)
    #: Whole-dub delivery register (`tts_style`). One of
    #: `ui.speakers.SUBMIT_STYLES`, or None for the engine's neutral read.
    style: Optional[str] = Field(default=None, max_length=64)
    window: Optional[Window] = None
    #: One flow since 2026-10-20 (`presets.PRESET_ID`). The retired ids
    #: `draft` / `standard` / `best` are still accepted and map onto it,
    #: because a browser holding the older bundle still posts `standard`.
    preset: str = Field(default="dub")
    #: None means "the install default" (`presets.default_tts()`), not "no
    #: TTS" — a dub with no TTS is not a thing this form can ask for.
    tts: Optional[str] = None
    #: "auto" defers to the face probe; an explicit model overrides it; None
    #: takes the preset's choice.
    lipsync_model: Optional[str] = None
    num_speakers: Optional[int] = Field(default=None, ge=1, le=32)
    skip_stems: Optional[bool] = None

    @field_validator("preset")
    @classmethod
    def _known_preset(cls, v: str) -> str:
        # A retired id maps onto the single flow; anything else is REJECTED
        # rather than silently coerced, because a typo that quietly produced
        # a different run would be discovered only by the GPU bill.
        try:
            return resolve_preset(v)
        except ValueError as e:
            raise ValueError(str(e)) from None

    @field_validator("target_language", "tts", "lipsync_model")
    @classmethod
    def _tidy(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().lower() if isinstance(v, str) else v
