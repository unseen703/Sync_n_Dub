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

from .presets import PRESET_IDS

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

    url: str = Field(min_length=1, max_length=2048)
    target_language: str = Field(min_length=2, max_length=16)
    window: Optional[Window] = None
    preset: str = Field(default="standard")
    #: None means "the install default" (`presets.default_tts()`), not "no
    #: TTS" — a dub with no TTS is not a thing this form can ask for.
    tts: Optional[str] = None
    #: "auto" defers to the face probe; an explicit model overrides it; None
    #: takes the preset's choice.
    lipsync_model: Optional[str] = None
    num_speakers: Optional[int] = Field(default=None, ge=1, le=32)
    skip_stems: Optional[bool] = None
    #: Face restoration after lip-sync. OPT-IN: None means off, whatever the
    #: preset once said. It roughly doubles GPU time (~10-16x clip length on
    #: top of lip-sync's own 3-21x), rebuilds every face it finds rather than
    #: only the one that was animated, and cannot be measured by diffing the
    #: two rendered videos. Ask for it deliberately: `"restore_model":
    #: "codeformer"`.
    restore_model: Optional[str] = None

    @field_validator("preset")
    @classmethod
    def _known_preset(cls, v: str) -> str:
        # Rejected rather than silently coerced to "standard": a typo that
        # quietly produced a different, slower run would be discovered only
        # by the GPU bill.
        p = (v or "").strip().lower()
        if p not in PRESET_IDS:
            raise ValueError(f"unknown preset {v!r}; expected one of "
                             f"{', '.join(PRESET_IDS)}")
        return p

    @field_validator("target_language", "tts", "lipsync_model")
    @classmethod
    def _tidy(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().lower() if isinstance(v, str) else v
