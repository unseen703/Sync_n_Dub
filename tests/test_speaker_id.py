"""Stage 4c: one label must hold one person, and one person one label.

The pipeline already guarantees one VOICE per label (`speaker_voice_map` is
keyed by speaker id and `backend.tts` resolves every segment through
`seg["speaker"]`). What it cannot guarantee is that a label holds one person
-- pyannote under-clusters noisy material -- and since the voice is per label,
an impure label reads several characters in one voice with nothing erroring.

No audio and no speechbrain here: `_embed_turns` is replaced by synthetic
vectors, so these pin the clustering decisions rather than the encoder.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend import speaker_id
from backend.speaker_id import label_purity, refine_speakers
from models.diarization.base import DiarizationResult, Turn


def _result(turns, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"\0")          # existence is all refine_speakers checks
    return DiarizationResult(turns=turns, audio_path=wav, model_name="stub")


#: Real ECAPA width. At dim=32 two random directions sit at cosine ~0.31 --
#: nothing like two real speakers -- and the fixture quietly stops testing
#: what it claims to.
DIM = 192

#: Jitter that reproduces the MEASURED single-speaker spread. On real audio,
#: turns of one speaker scored mean 0.45-0.55 at this window length and turns
#: of different speakers 0.007-0.141 between centroids; 1.2 lands a synthetic
#: speaker at ~0.50. Thresholds in `speaker_id` are calibrated against the real
#: numbers, so the fixture is matched to them and never the other way round.
SAME_SPEAKER_JITTER = 1.2


def _voice(seed: int, dim: int = DIM) -> np.ndarray:
    """A stable per-person direction."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


def _other_voice(a: np.ndarray, seed: int) -> np.ndarray:
    """A second person, orthogonal to `a` -- what two real speakers look like."""
    b = _voice(seed)
    b = b - (b @ a) * a
    return b / np.linalg.norm(b)


def _jitter(base, seed, scale=SAME_SPEAKER_JITTER):
    """Another turn by the same person."""
    rng = np.random.default_rng(seed)
    v = base + rng.normal(scale=scale / np.sqrt(len(base)), size=base.shape)
    return v / np.linalg.norm(v)


def _patch_embeddings(monkeypatch, vecs):
    monkeypatch.setattr(speaker_id, "_embed_turns",
                        lambda *a, **k: dict(enumerate(vecs)))


def test_pure_label_is_left_alone(tmp_path, monkeypatch):
    """One person, many turns -> untouched. The common case must not churn."""
    alice = _voice(1)
    vecs = [_jitter(alice, i) for i in range(8)]
    _patch_embeddings(monkeypatch, vecs)
    turns = [Turn(i * 3.0, i * 3.0 + 2.5, "SPK_00") for i in range(8)]

    out = refine_speakers(_result(turns, tmp_path))

    assert out.num_speakers == 1
    assert out.extra["speaker_id"]["splits"] == {}
    assert {t.speaker for t in out.turns} == {"SPK_00"}


def test_label_holding_two_people_is_split(tmp_path, monkeypatch):
    """The defect this stage exists for."""
    alice = _voice(1); bob = _other_voice(alice, 2)
    vecs = ([_jitter(alice, i) for i in range(6)] +
            [_jitter(bob, 100 + i) for i in range(6)])
    _patch_embeddings(monkeypatch, vecs)
    turns = [Turn(i * 5.0, i * 5.0 + 4.0, "SPK_00") for i in range(12)]

    out = refine_speakers(_result(turns, tmp_path))

    assert out.num_speakers == 2, "two people in one label must separate"
    first = {t.speaker for t in out.turns[:6]}
    second = {t.speaker for t in out.turns[6:]}
    assert len(first) == 1 and len(second) == 1
    assert first != second
    assert out.extra["speaker_id"]["splits"] == {"SPK_00": 2}


def test_one_person_under_two_labels_is_merged(tmp_path, monkeypatch):
    """The opposite failure: pyannote over-split one speaker."""
    alice = _voice(1)
    vecs = [_jitter(alice, i, scale=0.3) for i in range(8)]
    _patch_embeddings(monkeypatch, vecs)
    turns = ([Turn(i * 3.0, i * 3.0 + 2.5, "SPK_00") for i in range(4)] +
             [Turn(20 + i * 3.0, 22.5 + i * 3.0, "SPK_01") for i in range(4)])

    out = refine_speakers(_result(turns, tmp_path), merge_ceiling=0.55)

    assert out.num_speakers == 1
    assert out.extra["speaker_id"]["merges"]


def test_fragment_too_small_to_cast_is_folded_into_who_it_resembles(tmp_path, monkeypatch):
    """A 2 s sliver should not claim its own catalog voice when it belongs
    to someone already present."""
    alice = _voice(1)
    vecs = [_jitter(alice, i, scale=0.3) for i in range(7)]
    _patch_embeddings(monkeypatch, vecs)
    turns = [Turn(i * 10.0, i * 10.0 + 9.0, "SPK_00") for i in range(6)]
    turns.append(Turn(100.0, 101.2, "SPK_00"))          # 1.2 s sliver

    out = refine_speakers(_result(turns, tmp_path), min_speaker_s=4.0)

    assert out.num_speakers == 1
    assert all(t.speaker == "SPK_00" for t in out.turns)


def test_unlike_fragment_is_kept_standalone_not_absorbed(tmp_path, monkeypatch):
    """Absorbing by "nearest sibling" always finds someone.

    On the real clip an ungated absorb put SPK_00 back to 0.331 and SPK_02 to
    0.274 -- below the floor the split had just cleaned them past. A bit-part
    with one line is a real speaker with thin evidence.
    """
    alice = _voice(1); stranger = _other_voice(alice, 9)
    vecs = ([_jitter(alice, i) for i in range(6)] +
            [_jitter(stranger, 200)])
    _patch_embeddings(monkeypatch, vecs)
    turns = [Turn(i * 10.0, i * 10.0 + 8.0, "SPK_00") for i in range(6)]
    turns.append(Turn(100.0, 102.0, "SPK_00"))          # 2 s, nobody's voice

    out = refine_speakers(_result(turns, tmp_path),
                          min_speaker_s=4.0, absorb_min_cosine=0.35)

    sid = out.extra["speaker_id"]
    assert out.num_speakers == 2, "an unlike bit-part keeps its own label"
    assert sid["stranded"], "and is recorded as stranded, not absorbed"


def test_labels_are_canonical_and_ordered_by_speech_time(tmp_path, monkeypatch):
    """SPK_00 is the lead, matching how `assign_voices` already ranks."""
    alice = _voice(1); bob = _other_voice(alice, 2)
    vecs = ([_jitter(bob, 100 + i) for i in range(4)] +     # short, goes first
            [_jitter(alice, i) for i in range(8)])          # long
    _patch_embeddings(monkeypatch, vecs)
    turns = ([Turn(i * 2.0, i * 2.0 + 1.5, "SPK_00") for i in range(4)] +
             [Turn(20 + i * 12.0, 30 + i * 12.0, "SPK_00") for i in range(8)])

    out = refine_speakers(_result(turns, tmp_path))

    by = {}
    for t in out.turns:
        by[t.speaker] = by.get(t.speaker, 0.0) + t.duration
    assert sorted(by) == [f"SPK_{i:02d}" for i in range(len(by))], "gap-free"
    assert max(by, key=by.get) == "SPK_00", "most speech gets SPK_00"


def test_missing_speechbrain_leaves_labels_untouched(tmp_path, monkeypatch):
    """Optional dependency: refuse to guess rather than mangle the labels."""
    monkeypatch.setattr(speaker_id, "_embed_turns", lambda *a, **k: None)
    turns = [Turn(i * 3.0, i * 3.0 + 2.0, f"SPK_{i % 2:02d}") for i in range(6)]

    out = refine_speakers(_result(turns, tmp_path))

    assert [t.speaker for t in out.turns] == [t.speaker for t in turns]
    assert out.extra["speaker_id"]["ran"] is False


def test_input_result_is_not_mutated(tmp_path, monkeypatch):
    alice = _voice(1); bob = _other_voice(alice, 2)
    vecs = [_jitter(alice, i) for i in range(6)] + [_jitter(bob, 100 + i) for i in range(6)]
    _patch_embeddings(monkeypatch, vecs)
    turns = [Turn(i * 5.0, i * 5.0 + 4.0, "SPK_00") for i in range(12)]
    src = _result(turns, tmp_path)

    refine_speakers(src)

    assert {t.speaker for t in src.turns} == {"SPK_00"}


def test_purity_of_a_mixed_label_is_lower_than_a_clean_one():
    """The statistic the whole stage turns on."""
    alice = _voice(1); bob = _other_voice(alice, 2)
    clean = [_jitter(alice, i) for i in range(6)]
    mixed = [_jitter(alice, i) for i in range(3)] + [_jitter(bob, 100 + i) for i in range(3)]
    assert label_purity(clean) > label_purity(mixed)
    assert label_purity([clean[0]]) is None


# --- anti-drift: CLAUDE.md's rule about new stage markers ------------------

def test_stage_4c_marker_is_grouped_and_labelled():
    """A marker missing from PHASE_GROUPS shows as a nameless 8th segment."""
    from ui.progress import PHASE_GROUPS, STAGE_LABELS, STAGE_MARKER_RE
    line = "=== stage 4c: speaker label refinement ==="
    m = STAGE_MARKER_RE.search(line)
    assert m and m.group(1) == "4c", "the scraper must match the real marker"
    assert "4c" in STAGE_LABELS
    assert any("4c" in keys for _, _, keys in PHASE_GROUPS)


def test_pipeline_still_emits_the_4c_marker():
    """Renaming the marker silently freezes the progress bar."""
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "backend" / "pipeline.py"
    assert '=== stage 4c: speaker label refinement ===' in src.read_text()
