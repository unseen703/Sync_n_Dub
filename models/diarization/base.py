"""Shared types and abstract base for the diarization adapters.

Every adapter takes a mono 16 kHz WAV and returns a `DiarizationResult`, so
downstream stages consume one shape whichever model produced it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Turn:
    """A single speaker turn."""
    start: float          # seconds
    end: float            # seconds
    speaker: str          # canonical label, e.g. "SPK_00"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class DiarizationResult:
    """Uniform output of every diarization adapter."""
    turns: List[Turn]
    audio_path: Path
    model_name: str
    extra: dict = field(default_factory=dict)   # adapter-specific extras

    @property
    def num_speakers(self) -> int:
        return len({t.speaker for t in self.turns})

    def to_rttm(self, uri: Optional[str] = None) -> str:
        """Render as an RTTM string, NIST RTTM v2 columns.

        Never empty, even with no turns detected: a 0-byte artefact is
        indistinguishable from a writer crash, and the file's presence is what
        proves the stage ran.
        """
        uri = uri or self.audio_path.stem
        if not self.turns:
            return (
                f";;;; No speaker turns detected for {uri} "
                f"(model={self.model_name}).\n"
            )
        lines = [
            f"SPEAKER {uri} 1 {t.start:.3f} {t.duration:.3f} "
            f"<NA> <NA> {t.speaker} <NA> <NA>"
            for t in self.turns
        ]
        return "\n".join(lines) + "\n"

    def write_rttm(self, out_path: Path, uri: Optional[str] = None) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.to_rttm(uri=uri))
        return out_path


class BaseDiarizer(ABC):
    """Every diarization adapter subclasses this.

    Construction must not need weights — load them in `_ensure_loaded()` — so
    the CLI and UI can enumerate models without paying the download cost.
    """

    name: str = "base"

    def __init__(self, device: Optional[str] = None, **kwargs):
        self.device = device or _resolve_device()
        self._loaded = False

    # --- lifecycle ---------------------------------------------------------

    @abstractmethod
    def _ensure_loaded(self) -> None:
        """Load model weights on demand. Called from `diarize()`."""

    # --- inference ---------------------------------------------------------

    @abstractmethod
    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """Run diarization on a mono 16 kHz WAV file."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_device() -> str:
    """Pick cuda if torch reports it, else cpu. `DUBBING_DEVICE` overrides."""
    import os
    override = os.getenv("DUBBING_DEVICE", "").strip().lower()
    if override in {"cpu", "cuda", "mps"}:
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def canonicalize_speaker_labels(turns: List[Turn]) -> List[Turn]:
    """Rewrite an adapter's speaker labels to `SPK_00`, `SPK_01`, …

    Ordered by first appearance, so the mapping is deterministic per audio.
    """
    mapping: dict[str, str] = {}
    out: List[Turn] = []
    for t in turns:
        if t.speaker not in mapping:
            mapping[t.speaker] = f"SPK_{len(mapping):02d}"
        out.append(Turn(start=t.start, end=t.end, speaker=mapping[t.speaker]))
    return out
