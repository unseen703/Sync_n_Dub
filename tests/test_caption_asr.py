"""Captions-instead-of-ASR: the decision, and the shape it must produce.

The feature only pays off if a caption-sourced transcript is INDISTINGUISHABLE
downstream from a turn-sliced ASR one. Everything after stage 7 keys off
`seg["speaker"]` and `SEG_NNNN` ids, so a transcript that loses either would
collapse the cast onto one voice while every artifact still looked valid.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend.caption_asr import (          # noqa: E402
    MIN_TURN_COVERAGE, captions_to_asr_result, caption_quality_problem,
    pick_caption_track, transcript_from_captions,
)


@dataclass
class _Turn:
    start: float
    end: float
    speaker: str


def _vtt(tmp_path: Path, cues, name="t.en.vtt") -> Path:
    def ts(t):
        h, rem = divmod(float(t), 3600.0)
        m, s = divmod(rem, 60.0)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    body = "WEBVTT\n\n" + "\n\n".join(
        f"{ts(a)} --> {ts(b)}\n{txt}" for a, b, txt in cues)
    p = tmp_path / name
    p.write_text(body + "\n", encoding="utf-8")
    return p


# --- which track gets picked ---------------------------------------------

def test_manual_is_preferred_and_auto_is_refused_by_default():
    """An automatic track is YouTube's own ASR, not a transcript.

    Accepting it would swap the engine this repo chose for an unconfigurable
    one, on exactly the Indic-accented English it mostly processes. That is
    not a saving, so it needs an explicit opt-in.
    """
    subs = {"en": {"path": "/x/a.vtt", "source": "auto"}}
    assert pick_caption_track(subs, language="en") is None
    got = pick_caption_track(subs, language="en", allow_auto=True)
    assert got and got["source"] == "auto"

    subs["en-GB"] = {"path": "/x/m.vtt", "source": "manual"}
    got = pick_caption_track(subs, language="en")
    assert got and got["source"] == "manual" and got["path"] == "/x/m.vtt"


def test_language_is_matched_by_base_code():
    """YouTube publishes manual English under suffixed keys.

    In this corpus the manual English track lives under `en-LUU0EuDKgKo` --
    the channel's own id -- and asking for plain `en` there returns a
    machine translation of another language instead. An exact-key match
    would miss the real transcript and fall back to ASR for no reason.
    """
    subs = {"en-LUU0EuDKgKo": {"path": "/x/real.vtt", "source": "manual"}}
    got = pick_caption_track(subs, language="en")
    assert got and got["path"] == "/x/real.vtt"


def test_a_different_language_is_not_substituted():
    subs = {"hi": {"path": "/x/hi.vtt", "source": "manual"}}
    assert pick_caption_track(subs, language="en") is None


# --- the shape produced ---------------------------------------------------

def test_cues_are_reshaped_onto_turns_and_keep_the_speaker():
    """One segment per TURN, carrying that turn's speaker.

    A caption cue breaks on line length, not on who is talking. Using cues
    as segments would hand stage 9d a transcript with no speaker labels, and
    `backend/tts.py` resolves every segment through `seg["speaker"]` -- so
    the whole dub would be read in one voice with nothing looking wrong.
    """
    tmp = Path(__file__).parent
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        vtt = _vtt(Path(d), [(0.0, 2.0, "hello there"),
                             (2.5, 4.0, "general kenobi")])
        turns = [_Turn(0.0, 2.2, "SPK_00"), _Turn(2.4, 4.1, "SPK_01")]
        r = captions_to_asr_result(vtt, audio_path=Path(d) / "a.wav",
                                   turns=turns, language="en")
    assert [s.id for s in r.segments] == ["SEG_0000", "SEG_0001"]
    assert [s.speaker for s in r.segments] == ["SPK_00", "SPK_01"]
    assert r.segments[0].text == "hello there"
    assert r.segments[1].text == "general kenobi"


def test_a_cue_lands_in_exactly_one_turn():
    """A cue spanning two turns goes to the one it overlaps MOST.

    Adding it to every turn it touches would repeat the sentence in the dub,
    once per speaker -- worse than losing it.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        vtt = _vtt(Path(d), [(0.0, 10.0, "one long sentence")])
        turns = [_Turn(0.0, 2.0, "SPK_00"), _Turn(2.0, 10.0, "SPK_01")]
        r = captions_to_asr_result(vtt, audio_path=Path(d) / "a.wav",
                                   turns=turns, language="en")
    texts = [s.text for s in r.segments]
    assert texts.count("one long sentence") == 1
    assert texts[1] == "one long sentence", "should land in the longer overlap"


# --- every refusal falls back to ASR rather than failing -------------------

def test_thin_coverage_falls_back_to_asr():
    """Forced captions on a few signs are not a transcript.

    Dubbing a fraction of the video is worse than paying for ASR.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        vtt = _vtt(Path(d), [(0.0, 1.0, "[MUSIC]")])
        turns = [_Turn(float(i), float(i) + 1, f"SPK_{i%2:02d}")
                 for i in range(20)]
        subs = {"en": {"path": str(vtt), "source": "manual"}}
        got = transcript_from_captions(subs, audio_path=Path(d) / "a.wav",
                                       turns=turns, language="en")
    assert got is None, f"coverage floor is {MIN_TURN_COVERAGE}"


def test_an_unreadable_track_returns_none_rather_than_raising(tmp_path):
    """The caller's alternative is running ASR. A caption problem must cost
    the saving, never the run."""
    bad = tmp_path / "b.en.vtt"
    bad.write_bytes(b"\x00\x01 not a caption file")
    assert caption_quality_problem(bad) is not None
    subs = {"en": {"path": str(bad), "source": "manual"}}
    assert transcript_from_captions(subs, audio_path=tmp_path / "a.wav",
                                    turns=[_Turn(0, 1, "SPK_00")],
                                    language="en") is None


def test_missing_file_is_not_an_exception(tmp_path):
    subs = {"en": {"path": str(tmp_path / "nope.vtt"), "source": "manual"}}
    assert transcript_from_captions(subs, audio_path=tmp_path / "a.wav",
                                    language="en") is None


# --- the wiring -----------------------------------------------------------

def test_pipeline_defaults_to_captions_but_not_to_auto_ones():
    src = (REPO / "backend" / "pipeline.py").read_text()
    assert "captions_as_transcript: bool = True" in src
    assert "captions_allow_auto: bool = False" in src


def test_both_stage7_paths_write_the_same_artifacts():
    """`asr.json`, transcript.txt and `result.asr_*` cannot depend on which
    path produced the text -- the manifest is the durable record of a run."""
    src = (REPO / "backend" / "pipeline.py").read_text()
    shared = src[src.index("if asr_model and asr_result is not None:"):]
    for token in ("asr_json.write_text", "result.asr_model",
                  "result.asr_json", "asr_txt.write_text"):
        assert token in shared[:2000], f"{token} is not on the shared path"
