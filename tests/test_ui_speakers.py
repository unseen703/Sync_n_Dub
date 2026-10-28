"""Does the speaker table know who got which voice?

The hard case is the legacy one. Stage 9d writes
`voices/<stem>.<engine>.voices.json` only inside `if voice_engine:`; a job
with no voice catalog takes the `else` branch and persists nothing at all.
That was the Edge and IndicF5 path, and both are gone — but the jobs they
wrote are still on disk, and there are zero `*.voices.json` files under
`data/` right now, so a table that only read that file would be empty on
every job that exists.

`_replay_else_branch` reproduces that branch exactly rather than guessing.
The pipeline does:

    for spk in sorted(speaker_refs):
        speaker_voice_map[spk] = {..., "voice": None, ...}

and `speaker_refs` keys are the stems of `voices/<stem>.refs/*.wav`. The
tests below pin that correspondence, because if the pipeline's enumeration
ever changes, this reconstruction becomes confidently wrong — which is worse
than being empty.

Also pinned: the tone column is hidden on exactly the engines that ignore a
style hint. That predicate has to match `backend.edits._h_set_tone`, or the
UI offers a control that silently does nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ui import speakers as S


def _job(tmp_path: Path, speakers=("SPEAKER_00", "SPEAKER_01", "SPEAKER_02"),
         rttm=True) -> Path:
    root = tmp_path / "jobs" / "J"
    refs = root / "voices" / "clip.refs"
    refs.mkdir(parents=True)
    for s in speakers:
        (refs / f"{s}.wav").write_bytes(b"\0" * 64)
    if rttm:
        d = root / "diarization"
        d.mkdir()
        d.joinpath("clip.pyannote.rttm").write_text("".join(
            f"SPEAKER clip 1 {i * 10}.00 {10 - i}.00 <NA> <NA> {s} <NA> <NA>\n"
            for i, s in enumerate(speakers)))
    return root


def _table(root, cfg, merged=None):
    mj = None
    if merged is not None:
        mj = root / "mt" / "clip.bodhan.hi.mt.json"
        mj.parent.mkdir(parents=True, exist_ok=True)
        mj.write_text(json.dumps(merged))
    return S.build_table(root, cfg, data_dir=root.parent.parent, merged_json=mj)


#: A config whose voices come from 9d's `else` branch — the replay path.
#: Since Edge was removed that branch is the CLONING path, so the engine
#: here is IndicF5 and the expected voice is None rather than a name.
CLONE_HI = {"tts": "indicf5", "target_language": "hi"}
#: The catalog path, for the tests that need a real voice name.
BODHAN_HI = {"tts": "bodhan", "target_language": "hi"}


# ---------------------------------------------------------------------------
# the reconstruction
# ---------------------------------------------------------------------------

def test_the_replay_branch_reports_no_voice(tmp_path):
    """9d's `else` branch assigns none, so the replay must not invent one.

    It used to deal out `edge_pool[i % len(pool)]` over
    `sorted(speaker_refs)`. With Edge gone the branch is reached only by the
    cloning engines, where the speaker's own reference clip IS the voice and
    there is no name to show. A plausible-looking name here would be fiction
    presented as a record.
    """
    root = _job(tmp_path)
    got = {s["id"]: s["voice"] for s in _table(root, CLONE_HI)["speakers"]}
    assert got == {"SPEAKER_00": None, "SPEAKER_01": None, "SPEAKER_02": None}


def test_reconstruction_is_marked_as_derived_not_assigned(tmp_path):
    """Provenance matters: "derived" is a faithful replay, not a record."""
    root = _job(tmp_path)
    assert {s["voice_source"] for s in _table(root, CLONE_HI)["speakers"]} == {"derived"}


def test_voices_json_wins_over_the_reconstruction(tmp_path):
    root = _job(tmp_path)
    (root / "voices" / "clip.indic-speak.voices.json").write_text(json.dumps([
        {"speaker": "SPEAKER_00", "engine": "indic-speak", "voice": "Kavya",
         "score": 1.3, "rationale": {"gender": 1.0}},
    ]))
    rows = {s["id"]: s for s in _table(root, CLONE_HI)["speakers"]}
    assert rows["SPEAKER_00"]["voice"] == "Kavya"
    assert rows["SPEAKER_00"]["voice_source"] == "assigned"
    assert rows["SPEAKER_00"]["voice_score"] == 1.3
    # The others still fall through to the replay.
    assert rows["SPEAKER_01"]["voice_source"] == "derived"


def test_speaker_voice_map_is_read_when_present(tmp_path):
    root = _job(tmp_path)
    (root / "voices" / "clip.speaker_voice_map.json").write_text(json.dumps(
        {"SPEAKER_01": {"engine": "indic-speak", "voice": "Amit"}}))
    rows = {s["id"]: s for s in _table(root, BODHAN_HI)["speakers"]}
    assert rows["SPEAKER_01"]["voice_source"] == "assigned"


def test_a_per_segment_edit_beats_the_reconstruction(tmp_path):
    """Once `backend.edits` has run, `tts_voice` is the truth."""
    root = _job(tmp_path)
    merged = {"segments": [
        {"id": "s0", "speaker": "SPEAKER_00", "start": 0, "end": 2,
         "tts_voice": "Amit"},
        {"id": "s1", "speaker": "SPEAKER_00", "start": 2, "end": 4,
         "tts_voice": "Amit"},
    ]}
    rows = {s["id"]: s for s in _table(root, BODHAN_HI, merged)["speakers"]}
    assert rows["SPEAKER_00"]["voice"] == "Amit"
    assert rows["SPEAKER_00"]["voice_source"] == "edit"


# ---------------------------------------------------------------------------
# before 9d has run
# ---------------------------------------------------------------------------

def test_table_appears_from_the_rttm_alone(tmp_path):
    """Requirement: the cast shows up as soon as diarization finishes, long
    before any voice is assigned."""
    root = tmp_path / "jobs" / "J"
    d = root / "diarization"
    d.mkdir(parents=True)
    d.joinpath("x.pyannote.rttm").write_text(
        "SPEAKER x 1 0.00 12.50 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "SPEAKER x 1 12.50 4.00 <NA> <NA> SPEAKER_01 <NA> <NA>\n")
    rows = S.build_table(root, CLONE_HI, data_dir=root.parent.parent)["speakers"]
    assert [r["id"] for r in rows] == ["SPEAKER_00", "SPEAKER_01"]
    assert rows[0]["speech_s"] == 12.5
    assert rows[0]["voice"] is None
    assert rows[0]["reference_url"] is None


def test_speakers_are_ordered_by_speech_time(tmp_path):
    """Protagonist first — the same ranking `assign_voices` uses."""
    root = _job(tmp_path)
    rows = _table(root, CLONE_HI)["speakers"]
    times = [r["speech_s"] for r in rows]
    assert times == sorted(times, reverse=True)


def test_each_speaker_gets_a_stable_colour(tmp_path):
    root = _job(tmp_path)
    rows = _table(root, CLONE_HI)["speakers"]
    colours = [r["color"] for r in rows]
    assert len(set(colours)) == len(colours)
    assert all(c.startswith("#") for c in colours)


def test_reference_clip_url_is_offered_when_it_exists(tmp_path):
    root = _job(tmp_path)
    rows = _table(root, CLONE_HI)["speakers"]
    assert rows[0]["reference_url"].endswith("/voices/clip.refs/SPEAKER_00.wav")


# ---------------------------------------------------------------------------
# tone visibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine,expected", [
    ("bodhan", True), ("indic-speak", True),
    ("indicf5", False), ("", False),
    # `edge` is a removed engine, but jobs recorded before 2026-10-17 still
    # carry it and must not be offered a tone control they never had.
    ("edge", False),
])
def test_tone_is_offered_only_where_the_engine_forwards_it(engine, expected):
    assert S.supports_tone({"tts": engine}) is expected


def test_tone_predicate_matches_the_edits_module():
    """If these two disagree, the UI offers a control the pipeline ignores —
    or hides one that works."""
    import inspect

    from backend import edits
    src = inspect.getsource(edits._h_set_tone)
    # edits warns for any engine NOT in this pair; speakers.py must hide the
    # column for exactly the same set.
    assert '("bodhan", "indic-speak")' in src
    assert S.TONE_ENGINES == ("bodhan", "indic-speak")


def test_tone_options_are_empty_when_unsupported(tmp_path):
    root = _job(tmp_path)
    assert _table(root, CLONE_HI)["tone_options"] == []
    assert _table(root, {"tts": "bodhan", "target_language": "hi"})["tone_options"]


def test_current_tone_is_read_back_from_the_segments(tmp_path):
    root = _job(tmp_path)
    merged = {"segments": [
        {"id": "s0", "speaker": "SPEAKER_00", "start": 0, "end": 1,
         "tts_style": "formal"}]}
    cfg = {"tts": "bodhan", "target_language": "hi"}
    rows = {s["id"]: s for s in _table(root, cfg, merged)["speakers"]}
    assert rows["SPEAKER_00"]["tone"] == "formal"


# ---------------------------------------------------------------------------
# the voice picker's options
# ---------------------------------------------------------------------------

def test_the_picker_offers_both_genders_in_the_target_language(tmp_path):
    """Hindi has Kavya and Suhani (female) and Amit (male) recorded in it,
    and the picker must open on those rather than on whatever the catalog
    lists first."""
    opts = S.voice_options(BODHAN_HI)
    native = [o for o in opts if o["language"] == "hi"]
    assert {o["voice"] for o in native} == {"Kavya", "Suhani", "Amit"}
    assert {o["gender"] for o in native} == {"female", "male"}


def test_bodhan_options_come_from_the_catalog(tmp_path):
    opts = S.voice_options({"tts": "bodhan", "target_language": "ta"})
    assert len(opts) > 20
    # Native-language voices first, the ordering load_voice_catalog applies.
    assert opts[0]["language"] == "ta"


def test_unknown_engine_does_not_explode(tmp_path):
    assert isinstance(S.voice_options({"tts": "nope", "target_language": "hi"}), list)


# ---------------------------------------------------------------------------
# the one backend change this UI needed
# ---------------------------------------------------------------------------

def test_pipeline_persists_the_speaker_voice_map():
    """Stage 9d used to leave no record on the default Edge path.

    `voices.json` is written only inside `if voice_engine:`. Everything else
    built the map in memory and dropped it, so nothing downstream could say
    which voice spoke for which speaker. `ui/speakers.py` can reconstruct it
    (see `_replay_else_branch`), but a reconstruction is a replay of an
    algorithm, not a record of a decision — and it silently becomes wrong the
    day that algorithm changes. This asserts the record is now written.
    """
    import inspect

    from backend import pipeline
    src = inspect.getsource(pipeline.run_post_mt_stages)
    assert "speaker_voice_map.json" in src, (
        "stage 9d no longer persists speaker_voice_map; ui/speakers.py falls "
        "back to replaying 9d's else branch, which is exact only while that "
        "branch is unchanged.")
    assert '"speaker_voice_map"] = speaker_voice_map' in src


def test_resolver_reads_the_persisted_map(tmp_path):
    """The file the pipeline now writes must be the one the UI reads."""
    root = _job(tmp_path)
    (root / "voices" / "clip.speaker_voice_map.json").write_text(json.dumps({
        "SPEAKER_00": {"engine": "indic-speak", "voice": "Kavya",
                       "reference_audio": None},
        "SPEAKER_01": {"engine": "indic-speak", "voice": "Amit",
                       "reference_audio": None},
    }))
    got = S.resolve_speaker_voices(root, BODHAN_HI, {})
    assert got["SPEAKER_00"]["source"] == "assigned"
    assert got["SPEAKER_01"]["voice"] == "Amit"


# ---------------------------------------------------------------------------
# tones must be the engine's own, not a plausible-looking invention
# ---------------------------------------------------------------------------

def test_tone_values_are_the_adapter_s_verbatim():
    """This module first shipped an invented list — "formal", "casual",
    "polite". None of those are Bodhan styles.

    `INDIC_SPEAK_STYLES` is quoted from the hosted API docs, case and spacing
    are significant, and an unknown value is a **422, not a silent
    fallback**. So an invented vocabulary does not degrade — every tone
    change fails at the API with nothing in the UI to explain it.

    `NEUTRAL_STYLE` leads the list and is the ONE exception, added
    2026-10-21. It is not an invented engine style: it never reaches the
    engine at all, because `models.tts.resolve_style` maps it to None and a
    None style is simply omitted from the payload. It is how a speaker is
    taken OFF a register, which the dropdown previously could not express —
    every option in it was another register.
    """
    from models.tts import NEUTRAL_STYLE
    from models.tts.bodhan_adapter import INDIC_SPEAK_STYLES
    values = [t["value"] for t in S.tone_vocabulary()]
    assert values == [NEUTRAL_STYLE] + list(INDIC_SPEAK_STYLES)


def test_tone_labels_never_replace_the_wire_value():
    """The dropdown may prettify "AIR style news"; the POST may not."""
    from models.tts import NEUTRAL_STYLE
    from models.tts.bodhan_adapter import INDIC_SPEAK_STYLES
    for t in S.tone_vocabulary():
        assert t["value"] in INDIC_SPEAK_STYLES or t["value"] == NEUTRAL_STYLE
        assert t["label"]


def test_neutral_is_offered_on_both_dropdowns():
    """The per-speaker Delivery picker AND the new-job Type dropdown.

    Neither could ask for a plain reading before. The form's other
    non-engine value, "auto", means *classify a register from the video's
    metadata* — the opposite of asking for none — so a dub could be given a
    register by three different routes and taken off one by none.
    """
    from models.tts import NEUTRAL_STYLE
    assert S.tone_vocabulary()[0]["value"] == NEUTRAL_STYLE
    assert S.submit_style_options()[0]["value"] == NEUTRAL_STYLE


def test_the_submit_form_still_offers_the_four_real_formats():
    """Neutral is an addition, not a replacement."""
    from models.tts.bodhan_adapter import INDIC_SPEAK_STYLES
    values = [o["value"] for o in S.submit_style_options()]
    for style in S.SUBMIT_STYLES:
        assert style in values
        assert style in INDIC_SPEAK_STYLES


def test_emotions_are_grouped_apart_from_delivery_styles():
    """14 flat items is a scroll; two groups is a choice. The six Ekman
    emotions come from AI4Bharat's Rasa corpus and are a different kind of
    thing from "TV style news"."""
    groups = {t["label"]: t["group"] for t in S.tone_vocabulary()}
    assert groups["Sad"] == "Emotion"
    assert groups["TV-style news"] == "Delivery"


def test_tone_options_absent_when_the_engine_ignores_style(tmp_path):
    root = _job(tmp_path)
    assert _table(root, CLONE_HI)["tone_options"] == []


# ---------------------------------------------------------------------------
# gender, and the voices that follow from it
# ---------------------------------------------------------------------------

def _with_voice_map(tmp_path, genders):
    root = _job(tmp_path, speakers=tuple(genders))
    (root / "voices" / "clip.speaker_voice_map.json").write_text(json.dumps({
        sid: {"engine": "bodhan", "voice": "X", "gender": g}
        for sid, g in genders.items()}))
    return root


def test_gender_is_read_from_the_persisted_map(tmp_path):
    """Stage 9d computes gender with ECAPA and now writes it down.
    Re-deriving it here would mean loading a speaker-embedding model inside
    a GET, which is not a thing a web request should do."""
    root = _with_voice_map(tmp_path, {"SPEAKER_00": "female", "SPEAKER_01": "male"})
    rows = {r["id"]: r for r in _table(root, {"tts": "bodhan",
                                              "target_language": "hi"})["speakers"]}
    assert rows["SPEAKER_00"]["gender"] == "female"
    assert rows["SPEAKER_01"]["gender"] == "male"


def test_an_explicit_gender_edit_beats_the_prediction(tmp_path):
    root = _with_voice_map(tmp_path, {"SPEAKER_00": "female"})
    merged = {"segments": [{"id": "s0", "speaker": "SPEAKER_00", "start": 0,
                            "end": 1, "edit_gender": "male"}]}
    rows = {r["id"]: r for r in
            _table(root, {"tts": "bodhan", "target_language": "hi"}, merged)["speakers"]}
    assert rows["SPEAKER_00"]["gender"] == "male"


def test_voices_are_recommended_by_matching_gender(tmp_path):
    root = _with_voice_map(tmp_path, {"SPEAKER_00": "female", "SPEAKER_01": "male"})
    t = _table(root, {"tts": "bodhan", "target_language": "hi"})
    for o in t["voice_options"]:
        if "SPEAKER_00" in o["recommended_for"]:
            assert o["gender"] == "female"
        if "SPEAKER_01" in o["recommended_for"]:
            assert o["gender"] == "male"


def test_every_voice_is_still_offered(tmp_path):
    """Recommendation, not filtering. ECAPA's gender call is a prediction,
    and a user who disagrees must still be able to pick the other voice."""
    root = _with_voice_map(tmp_path, {"SPEAKER_00": "female"})
    t = _table(root, {"tts": "bodhan", "target_language": "hi"})
    assert any(o["gender"] == "male" for o in t["voice_options"])


# ---------------------------------------------------------------------------
# renaming a speaker
# ---------------------------------------------------------------------------

def test_a_speaker_can_be_renamed_and_read_back(tmp_path):
    root = _job(tmp_path)
    S.write_label(root, "SPEAKER_00", "Prof. Mitra")
    rows = {r["id"]: r for r in _table(root, CLONE_HI)["speakers"]}
    assert rows["SPEAKER_00"]["label"] == "Prof. Mitra"
    assert rows["SPEAKER_01"]["label"] is None


def test_an_empty_name_clears_it(tmp_path):
    root = _job(tmp_path)
    S.write_label(root, "SPEAKER_00", "Temp")
    assert S.write_label(root, "SPEAKER_00", "") == {}


def test_renaming_does_not_touch_the_merged_json(tmp_path):
    """A rename is a display preference. If it landed in the merged JSON it
    would dirty a stage and cost a re-render for changing a caption."""
    root = _job(tmp_path)
    merged = {"segments": [{"id": "s0", "speaker": "SPEAKER_00",
                            "start": 0, "end": 1}]}
    mj = root / "mt" / "clip.bodhan.hi.mt.json"
    mj.parent.mkdir(parents=True, exist_ok=True)
    mj.write_text(json.dumps(merged))
    before = mj.read_text()
    S.write_label(root, "SPEAKER_00", "Prof. Mitra")
    assert mj.read_text() == before
