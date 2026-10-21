"""The staged human-in-the-loop flow: review the cast, review the cues, then
lip-sync.

Three new surfaces, each with one way to fail silently:

  * `set_speaker` moves a cue onto another speaker. It is the only op that
    touches `seg["speaker"]`, which is what `backend/tts.py` resolves the
    voice through -- so a mis-attributed line is read in the wrong person's
    voice while every artifact still looks right.
  * the transcript endpoint is a READ. If it ever re-derived text, a cue
    could disagree with the dub it describes.
  * `POST .../lipsync` finishes a `draft` run. That job never configured the
    stage, so the replay set has to gain it -- otherwise the render is a
    no-op reported as success.
"""
from __future__ import annotations

import json

import pytest

from backend import edits as E
from backend.edit_agent import _rule_parse


def _merged(tmp_path, segs):
    p = tmp_path / "merged.json"
    p.write_text(json.dumps({"segments": segs, "extra": {}}))
    return p


def _seg(i, spk, start, text="hello", **kw):
    d = {"id": f"SEG_{i:04d}", "start": float(start), "end": float(start + 2),
         "speaker": spk, "text": text, "final_text": f"{text}-hi"}
    d.update(kw)
    return d


class _Ctx:
    tts_engine = "bodhan"; voice_engine = "indic-speak"
    target_language = "hi"; voice_catalog_json = None
    lipsync_model = None; translit_model = None


# --- set_speaker ----------------------------------------------------------

def test_the_op_exists_and_is_wired():
    """A spec without a handler (or vice versa) is caught upstream by an
    assert; this pins that `set_speaker` is actually one of the nine."""
    assert "set_speaker" in E.OPS
    assert E.OPS["set_speaker"].dirty_from == "voices"


def test_moving_a_cue_changes_whose_voice_reads_it(tmp_path):
    mj = _merged(tmp_path, [_seg(1, "SPK_00", 0), _seg(2, "SPK_00", 10),
                            _seg(3, "SPK_01", 20)])
    ops = [E.EditOp(op="set_speaker", scope=E.Scope(segments=["SEG_0002"]),
                    params={"speaker": "SPK_01"})]
    E.apply_edits(mj, ops, ctx=_Ctx(), dry_run=False)

    segs = json.loads(mj.read_text())["segments"]
    assert segs[1]["speaker"] == "SPK_01", "the cue must move"
    assert segs[0]["speaker"] == "SPK_00", "its neighbours must not"
    assert segs[1]["edit_backup"]["speaker"] == "SPK_00", "revertible"


def test_a_stale_per_segment_voice_is_cleared(tmp_path):
    """Otherwise the cue moves to the new speaker and is still read in the
    old one's voice: `tts.py` reads `tts_voice` AHEAD of the speaker map."""
    mj = _merged(tmp_path, [_seg(1, "SPK_00", 0, tts_voice="Amit"),
                            _seg(2, "SPK_01", 10)])
    ops = [E.EditOp(op="set_speaker", scope=E.Scope(segments=["SEG_0001"]),
                    params={"speaker": "SPK_01"})]
    E.apply_edits(mj, ops, ctx=_Ctx(), dry_run=False)

    seg = json.loads(mj.read_text())["segments"][0]
    assert seg["speaker"] == "SPK_01"
    assert not seg["tts_voice"], "the previous speaker's voice must not stick"
    assert seg["edit_backup"]["tts_voice"] == "Amit"


def test_an_invented_speaker_is_refused(tmp_path):
    """A new id has no reference clip, no gender vote and no catalog voice,
    so TTS would fall back to the engine default and read it in a voice
    nobody chose."""
    mj = _merged(tmp_path, [_seg(1, "SPK_00", 0)])
    ops = [E.EditOp(op="set_speaker", scope=E.Scope(all=True),
                    params={"speaker": "SPK_99"})]
    with pytest.raises(E.EditError, match="unknown speaker"):
        E.apply_edits(mj, ops, ctx=_Ctx(), dry_run=True)


def test_emptying_a_speaker_is_announced(tmp_path):
    """9d re-derives the cast from what is there, so the operator's speaker
    list is about to change -- silence here would be a surprise."""
    mj = _merged(tmp_path, [_seg(1, "SPK_00", 0), _seg(2, "SPK_01", 10)])
    ops = [E.EditOp(op="set_speaker", scope=E.Scope(segments=["SEG_0001"]),
                    params={"speaker": "SPK_01"})]
    res = E.apply_edits(mj, ops, ctx=_Ctx(), dry_run=False)
    warned = " ".join(w for a in res.applied for w in a.warnings)
    assert "SPK_00" in warned and "no cues" in warned


def test_it_dirties_from_voices_not_tts(tmp_path):
    """The speaker SET can change, and 9d is what re-derives the map."""
    mj = _merged(tmp_path, [_seg(1, "SPK_00", 0), _seg(2, "SPK_01", 10)])
    ops = [E.EditOp(op="set_speaker", scope=E.Scope(segments=["SEG_0001"]),
                    params={"speaker": "SPK_01"})]
    res = E.apply_edits(mj, ops, ctx=_Ctx(), dry_run=True)
    assert res.applied[0].dirty_from == "voices"


# --- the fallback parser, which is all there is without an LLM key --------

@pytest.mark.parametrize("q,spk", [
    ("the line at 1:20 is SPK_02, not SPK_00", "SPK_02"),
    ("from 0:30 to 0:45 that is SPEAKER_01", "SPK_01"),
])
def test_reattribution_parses_without_an_llm(q, spk):
    ops = _rule_parse(q)
    assert [o.op for o in ops] == ["set_speaker"]
    assert ops[0].params["speaker"] == spk


def test_a_bare_catalog_voice_now_parses():
    """The regex matched only `xx-XX-NameNeural`, an Edge-era shape removed
    from the registry on 2026-10-17. Bodhan's voices are bare names, so
    "use Amit" silently did nothing."""
    ops = _rule_parse("use Amit for SPEAKER_00")
    assert [o.op for o in ops] == ["set_voice"]
    assert ops[0].params["voice"] == "Amit"


@pytest.mark.parametrize("q,expect", [
    ("use faster speech", "set_speed"),
    ("use a female voice", "set_gender"),
])
def test_the_widened_voice_regex_does_not_swallow_other_intents(q, expect):
    assert expect in [o.op for o in _rule_parse(q)]


# --- the transcript read --------------------------------------------------

def test_cues_carry_what_a_review_needs(tmp_path):
    from ui import transcript as T
    mj = _merged(tmp_path, [
        _seg(1, "SPK_00", 0, tts_audio=str(tmp_path / "a.wav")),
        _seg(2, "SPK_01", 10, qc={"tts_error": "502"}),
    ])
    out = T.cues(mj, data_dir=tmp_path, labels={"SPK_00": "Narrator"})

    assert out["total"] == 2 and len(out["cues"]) == 2
    a, b = out["cues"]
    assert a["speaker"] == "SPK_00" and a["speaker_label"] == "Narrator"
    assert a["source_text"] and a["dub_text"], "both sides, or it cannot be judged"
    assert b["tts_error"] == "502", "a silent-failure line must be visible"
    assert out["n_failed"] == 1
    assert out["speakers"] == ["SPK_00", "SPK_01"]


def test_the_transcript_is_a_read_not_a_second_source_of_truth(tmp_path):
    from ui import transcript as T
    segs = [_seg(1, "SPK_00", 0)]
    mj = _merged(tmp_path, segs)
    before = mj.read_text()
    T.cues(mj, data_dir=tmp_path)
    assert mj.read_text() == before, "reading the transcript must not write"


def test_paging(tmp_path):
    from ui import transcript as T
    mj = _merged(tmp_path, [_seg(i, "SPK_00", i * 3) for i in range(40)])
    out = T.cues(mj, data_dir=tmp_path, offset=10, limit=5)
    assert out["total"] == 40 and len(out["cues"]) == 5
    assert out["cues"][0]["id"] == "SEG_0010"


# --- the finish-the-job route --------------------------------------------

def test_lipsync_route_is_registered_and_guarded():
    from ui import routes_edits as R
    paths = {getattr(r, "path", "") for r in R.router.routes}
    assert "/api/jobs/{job_id}/lipsync" in paths
    assert "/api/jobs/{job_id}/transcript" in paths


def test_a_whole_video_window_re_enables_the_stage(tmp_path):
    """The mechanism the finish route depends on: an op that INTRODUCES
    lip-sync adds it back for a job that never configured it."""
    mj = _merged(tmp_path, [_seg(1, "SPK_00", 0), _seg(2, "SPK_00", 30)])
    ops = [E.EditOp(op="lipsync_window", scope=E.Scope(all=True),
                    params={"model": "musetalk"})]
    res = E.apply_edits(mj, ops, ctx=_Ctx(), dry_run=True)
    assert res.lipsync_window is not None, "a window must be derived"
    assert res.lipsync_model == "musetalk", "and an explicit model kept"


# --- speaker card media ---------------------------------------------------

def test_the_dub_sample_is_the_longest_clip_not_the_first(tmp_path):
    """An opening cue is often a one-word answer; the sample exists so a
    human can judge a VOICE."""
    from ui.speaker_media import dub_sample_paths
    merged = {"segments": [
        _seg(1, "SPK_00", 0, tts_audio="/a/short.wav", tts_duration_s=1.0),
        _seg(2, "SPK_00", 10, tts_audio="/a/long.wav", tts_duration_s=4.0),
        _seg(3, "SPK_01", 20, tts_audio="/a/other.wav", tts_duration_s=2.0),
    ]}
    got = dub_sample_paths(merged)
    assert got["SPK_00"] == "/a/long.wav"
    assert got["SPK_01"] == "/a/other.wav"


def test_a_failed_line_is_never_offered_as_a_sample(tmp_path):
    """A Bodhan outage sets `tts_error` and leaves no audio; offering it
    would be a dead play button on the review card."""
    from ui.speaker_media import dub_sample_paths
    merged = {"segments": [
        _seg(1, "SPK_00", 0, tts_audio="/a/x.wav", tts_duration_s=5.0,
             qc={"tts_error": "502"}),
    ]}
    assert dub_sample_paths(merged) == {}


def test_a_breath_is_too_short_to_judge_a_voice_by():
    from ui.speaker_media import dub_sample_paths, MIN_SAMPLE_S
    merged = {"segments": [_seg(1, "SPK_00", 0, tts_audio="/a/x.wav",
                                tts_duration_s=MIN_SAMPLE_S / 2)]}
    assert dub_sample_paths(merged) == {}


def test_no_face_evidence_means_no_crop_not_a_broken_link(tmp_path):
    """Stage 6 deliberately omits a speaker seen too briefly to trust;
    inventing a crop would assert evidence the pipeline refused to."""
    from ui.speaker_media import face_crop_paths
    (tmp_path / "videos").mkdir()
    assert face_crop_paths(tmp_path) == {}


# --- the bbox review render ----------------------------------------------

def test_status_trusts_the_file_over_the_status_field(tmp_path):
    """A killed render leaves a stale "running"/"done" behind forever.
    CLAUDE.md records what trusting a status field over the artifact costs."""
    from ui import bbox_video as B
    B._write_status(tmp_path, state="done")
    st = B.status(tmp_path)
    assert st["ready"] is False
    assert st["state"] == "failed", "claimed done, wrote no video"
    assert "error" in st


def test_absent_means_absent(tmp_path):
    from ui import bbox_video as B
    assert B.status(tmp_path)["state"] == "absent"


def test_render_without_face_tracks_fails_loudly(tmp_path):
    from ui import bbox_video as B
    st = B.render(tmp_path)
    assert st["state"] == "failed" and "stage 5" in st["error"]


def test_it_reads_assignments_not_a_scores_npz():
    """The first version looked for `asd/*.scores.npz`, which stage 6 never
    writes, so every box rendered grey: the render ran, looked plausible,
    and conveyed nothing."""
    from pathlib import Path
    body = (Path(__file__).resolve().parent.parent / "ui" / "bbox_video.py").read_text()
    assert 'glob("asd/*.assignments.json")' in body, "must read what 9f writes"
    # The substring survives in a comment explaining WHY it is not used, so
    # assert on the code shape rather than the file text.
    assert 'glob("asd/*.scores.npz")' not in body
    assert "np.load" not in body
