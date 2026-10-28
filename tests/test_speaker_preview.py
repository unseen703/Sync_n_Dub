"""Pre-cast speaker previews: hear a speaker before the voices stage runs.

The voices stage writes `voices/<stem>.refs/*.wav` only after the cast is
chosen, so a job waiting on the cast has no reference clips. The preview is
cut from the source audio instead, and must keep clear of the `*.refs` glob
that `ui.speakers._reference_clips` treats as the pipeline's own output.
"""
from __future__ import annotations

from pathlib import Path

from ui import speaker_media as M
from ui import speakers as S


def _rttm(root: Path, rows) -> None:
    d = root / "diarization"
    d.mkdir(parents=True)
    (d / "c.pyannote.rttm").write_text("".join(
        f"SPEAKER c 1 {s} {dur} <NA> <NA> {spk} <NA> <NA>\n"
        for s, dur, spk in rows))


def test_solo_spans_cut_out_overlap():
    turns = [(0.0, 10.0, "A"), (4.0, 6.0, "B")]
    spans = M._solo_spans(turns)
    assert sorted(spans["A"]) == [(0.0, 4.0), (6.0, 10.0)]
    assert spans["B"] == []


def test_preview_absent_without_source_audio(tmp_path):
    _rttm(tmp_path, [(0.0, 5.0, "SPK_00")])
    assert M.speaker_preview_paths(tmp_path) == {}


def test_existing_previews_are_reused(tmp_path):
    p = tmp_path / M.PREVIEW_SUBDIR
    p.mkdir(parents=True)
    (p / "SPK_00.wav").write_bytes(b"\0")
    assert M.speaker_preview_paths(tmp_path) == {"SPK_00": p / "SPK_00.wav"}


def test_preview_dir_is_not_seen_as_reference_clips(tmp_path):
    p = tmp_path / M.PREVIEW_SUBDIR
    p.mkdir(parents=True)
    (p / "SPK_00.wav").write_bytes(b"\0")
    assert S._reference_clips(tmp_path) == {}
