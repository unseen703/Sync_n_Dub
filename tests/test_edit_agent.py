"""Does an NLQ edit reach the stage that has to act on it?

Every assertion here traces to a way this feature can look like it works and
not work:

  * a time window that selects nothing, or selects the wrong cue because it
    demanded containment where it should have demanded overlap;
  * a speed edit written as a shorter slot and then silently clamped back to
    +-10% by the stretch cap, so "30% faster" arrives as 10% faster with a
    budget_violation nobody reads;
  * a per-segment voice override written under a key `backend.tts` does not
    read -- the same shape as the gender-matching dead code that shipped a
    female voice over an all-male lecture;
  * a rerun plan that promises stages the job cannot replay.

No network, no models, no GPU: the LLM is never called (ops are built
directly or through the regex fallback), and nothing here synthesises audio.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend import edits as E
from backend import edit_agent as A


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _merged(tmp_path: Path) -> Path:
    """Five cues, two speakers, with a real gap between 46 s and 75 s."""
    segs = [
        ("seg000", 0.0, 5.0, "SPEAKER_00", "The gradient is here.", "ग्रेडिएंट यहाँ है।"),
        ("seg001", 5.0, 12.0, "SPEAKER_00", "We apply back propagation.", "बैक प्रोपेगेशन लागू करते हैं।"),
        ("seg002", 30.5, 38.0, "SPEAKER_01", "The second speaker.", "दूसरा वक्ता।"),
        ("seg003", 38.0, 46.0, "SPEAKER_01", "Loss must be differentiable.", "लॉस अवकलनीय होना चाहिए।"),
        ("seg004", 75.0, 84.0, "SPEAKER_00", "We evaluate on the test set.", "टेस्ट सेट पर मूल्यांकन।"),
    ]
    data = {
        "audio_path": "x.wav", "model_name": "bodhan", "language": "en",
        "segments": [
            {"id": i, "start": a, "end": b, "speaker": s, "text": en,
             "mt_text": hi, "final_text": hi}
            for i, a, b, s, en, hi in segs
        ],
        "extra": {"mt_model": "bodhan", "mt_target_language": "hi"},
    }
    p = tmp_path / "demo.bodhan.hi.mt.json"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return p


@pytest.fixture
def merged(tmp_path: Path) -> Path:
    return _merged(tmp_path)


def _ctx(**kw) -> E.EditContext:
    base = dict(target_language="hi", tts_engine="bodhan")
    base.update(kw)
    return E.EditContext(**base)


def _apply(merged: Path, op: E.EditOp, **ctx_kw) -> E.EditResult:
    return E.apply_edits(merged, [op], ctx=_ctx(**ctx_kw), dry_run=False)


# ---------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,secs", [
    (42, 42.0), ("42", 42.0), ("1:23", 83.0), ("1:23.5", 83.5),
    ("00:01:23", 83.0), ("2:00:00", 7200.0),
])
def test_timestamps_parse(raw, secs):
    assert E._as_time(raw) == pytest.approx(secs)


def test_bad_timestamp_refuses():
    with pytest.raises(E.EditError):
        E._as_time("half past two")


def test_window_selects_overlapping_cues(merged):
    """0:30-0:45 must catch the cue that STRADDLES 0:45.

    Containment would drop seg003 (38-46) and the edit would stop audibly
    mid-sentence, which is the failure this rule exists to prevent.
    """
    data = json.loads(merged.read_text())
    assert E.Scope(start_s=30, end_s=45).resolve(data) == ["seg002", "seg003"]


def test_point_inside_a_cue(merged):
    data = json.loads(merged.read_text())
    assert E.Scope(start_s=80).resolve(data) == ["seg004"]


@pytest.mark.parametrize("t,expected", [
    (70.0, "seg004"),   # 5 s before seg004 starts, 24 s after seg003 ended
    (50.0, "seg003"),   # 4 s after seg003 ended, 25 s before seg004 starts
])
def test_point_in_a_gap_takes_the_nearest_cue(merged, t, expected):
    """46-75 s is silence; a point in it resolves to whichever cue is closer.

    Both directions are asserted, because "nearest" implemented as "the next
    one" would still pass a one-sided test and would quietly move every edit
    in a gap forwards.
    """
    data = json.loads(merged.read_text())
    assert E.Scope(start_s=t).resolve(data) == [expected]


def test_speaker_scope(merged):
    data = json.loads(merged.read_text())
    assert E.Scope(speaker="SPEAKER_01").resolve(data) == ["seg002", "seg003"]


def test_unknown_speaker_names_the_known_ones(merged):
    data = json.loads(merged.read_text())
    with pytest.raises(E.EditError, match="SPEAKER_00"):
        E.Scope(speaker="SPEAKER_99").resolve(data)


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------

def test_set_text_writes_both_text_fields(merged):
    """`synthesize_segments` prefers final_text and falls back to mt_text.

    Writing only one of them is how an edit silently does nothing on a run
    that did (or did not) code-mix.
    """
    _apply(merged, E.EditOp(op="set_text", scope=E.Scope(segments=["seg000"]),
                            params={"text": "नया वाक्य।"}))
    seg = json.loads(merged.read_text())["segments"][0]
    assert seg["mt_text"] == "नया वाक्य।"
    assert seg["final_text"] == "नया वाक्य।"


def test_set_speed_shortens_the_slot_and_lifts_the_cap(merged):
    """A 1.3x request needs its own stretch cap.

    `stretch_to_duration` clamps to +-max_stretch (0.10 by default), so
    without a per-segment cap the slot says 1.3x and the audio comes out at
    1.1x with a budget_violation flag -- green run, wrong output.
    """
    _apply(merged, E.EditOp(op="set_speed", scope=E.Scope(start_s=30, end_s=45),
                            params={"factor": 1.3}))
    segs = {s["id"]: s for s in json.loads(merged.read_text())["segments"]}
    assert segs["seg002"]["target_duration_s"] == pytest.approx(7.5 / 1.3, abs=1e-3)
    assert segs["seg002"]["tts_max_stretch"] >= 0.30
    assert segs["seg002"]["tts_rate_override"] == pytest.approx(1.3)
    assert "tts_max_stretch" not in segs["seg000"], "untouched cue kept its budget"


@pytest.mark.parametrize("factor", [0.0, -1.0, 3.0, 0.2])
def test_set_speed_refuses_nonsense(merged, factor):
    with pytest.raises(E.EditError):
        _apply(merged, E.EditOp(op="set_speed", scope=E.Scope(all=True),
                                params={"factor": factor}))


def test_set_gender_picks_a_matching_voice(merged):
    res = _apply(merged, E.EditOp(op="set_gender", scope=E.Scope(start_s=80),
                                  params={"gender": "male"}))
    seg = {s["id"]: s for s in json.loads(merged.read_text())["segments"]}["seg004"]
    assert seg["tts_voice"] == "Amit"      # the lead male voice recorded in hi
    assert seg["edit_gender"] == "male"
    assert res.dirty_stages[0] == "tts"


def test_speaker_wide_gender_goes_through_the_voice_stage(merged):
    res = _apply(merged, E.EditOp(op="set_gender",
                                  scope=E.Scope(speaker="SPEAKER_01"),
                                  params={"gender": "female"}))
    assert res.gender_overrides == {"SPEAKER_01": "female"}
    # Kavya, not Suhani: both are female and Hindi, and `tier == "lead"`
    # breaks the tie.
    assert res.speaker_voice_overrides["SPEAKER_01"]["voice"] == "Kavya"
    assert res.dirty_stages[0] == "voices"


def test_gender_with_no_voice_in_the_catalog_refuses(merged, tmp_path):
    """Better a refusal than the opposite gender.

    Assigning the nearest available voice regardless of gender is exactly
    what put a female voice on an all-male lecture once already.

    An unknown target LANGUAGE no longer triggers this: indic-speak voices
    read any language, so `load_voice_catalog` orders by language and never
    filters by it, and a female voice is still found. The refusal is about
    the GENDER having no voice, which is what this forces with a male-only
    catalog.
    """
    cat = tmp_path / "male_only.json"
    cat.write_text(json.dumps([
        {"engine": "indic-speak", "voice": "Solus", "gender": "male",
         "language": "hi", "tier": "lead"},
    ]))
    with pytest.raises(E.EditError):
        _apply(merged, E.EditOp(op="set_gender", scope=E.Scope(all=True),
                                params={"gender": "female"}),
               tts_engine="bodhan", voice_catalog_json=cat)


def test_punjabi_gender_change_picks_the_punjabi_voice_of_that_gender():
    """The case that used to be unservable, now that Edge is gone.

    Edge had no Punjabi voice at all -- its table claimed
    `pa-IN-OjasNeural` for both genders and the service has no such voice --
    so a gender change on a Punjabi dub was reported as impossible. The
    indic-speak catalog records one of each: Kaur and Manpreet.
    """
    assert E.voice_for_gender("female", engine="bodhan", language="pa")[0] == "Kaur"
    assert E.voice_for_gender("male", engine="bodhan", language="pa")[0] == "Manpreet"


def test_lipsync_window_from_a_time_range(merged):
    res = _apply(merged, E.EditOp(op="lipsync_window",
                                  scope=E.Scope(start_s=60, end_s=90),
                                  params={"model": "musetalk"}))
    assert res.lipsync_window == (60.0, 90.0)
    assert res.lipsync_model == "musetalk"
    assert res.dirty_stages == ["lipsync"]


def test_lipsync_window_too_short_refuses(merged):
    with pytest.raises(E.EditError):
        _apply(merged, E.EditOp(op="lipsync_window",
                                scope=E.Scope(start_s=10, end_s=10.2)))


def test_llm_only_ops_refuse_without_a_rewriter(merged):
    with pytest.raises(E.EditError, match="OPENROUTER_API_KEY"):
        _apply(merged, E.EditOp(op="retranslate", scope=E.Scope(all=True),
                                params={"instruction": "simpler"}))


def test_retranslate_uses_the_injected_rewriter(merged):
    calls = []

    def fake(system, user):
        calls.append(user)
        return "पुनः अनुवादित वाक्य।"

    _apply(merged, E.EditOp(op="retranslate", scope=E.Scope(segments=["seg001"]),
                            params={"instruction": "simpler"}),
           rewriter=fake)
    seg = {s["id"]: s for s in json.loads(merged.read_text())["segments"]}["seg001"]
    assert seg["final_text"] == "पुनः अनुवादित वाक्य।"
    assert "We apply back propagation." in calls[0], "source line reaches the model"


def test_code_mix_writes_a_termbase(merged, tmp_path):
    res = _apply(merged, E.EditOp(op="code_mix", scope=E.Scope(all=True),
                                  params={"terms": ["gradient", "weight"]}),
                 translit_model="bodhan", work_dir=tmp_path / "edits")
    assert res.translit_model == "bodhan"
    rows = res.termbase_path.read_text().splitlines()
    # `backend.codemix.load_termbase` reads `source<TAB>replacement`; a term
    # mapped to itself is what keeps it in English.
    assert "gradient\tgradient" in rows


def test_dry_run_writes_nothing(merged, tmp_path):
    before = merged.read_text()
    res = E.apply_edits(
        merged, [E.EditOp(op="code_mix", scope=E.Scope(all=True),
                          params={"terms": ["gradient"]})],
        ctx=_ctx(translit_model="bodhan", work_dir=tmp_path / "edits"),
        dry_run=True)
    assert merged.read_text() == before
    assert not res.termbase_path.exists()


# ---------------------------------------------------------------------------
# revert
# ---------------------------------------------------------------------------

def test_revert_restores_every_edited_field(merged):
    before = json.loads(merged.read_text())
    _apply(merged, E.EditOp(op="set_speed", scope=E.Scope(all=True),
                            params={"factor": 1.2}))
    _apply(merged, E.EditOp(op="set_text", scope=E.Scope(segments=["seg000"]),
                            params={"text": "बदला हुआ।"}))
    assert E.revert(merged)
    after = json.loads(merged.read_text())
    assert after["segments"] == before["segments"]


def test_revert_is_idempotent(merged):
    _apply(merged, E.EditOp(op="set_text", scope=E.Scope(segments=["seg000"]),
                            params={"text": "x"}))
    assert E.revert(merged) == ["seg000"]
    assert E.revert(merged) == []


def test_two_edits_to_one_field_keep_the_original_backup(merged):
    for text in ("first", "second"):
        _apply(merged, E.EditOp(op="set_text", scope=E.Scope(segments=["seg000"]),
                                params={"text": text}))
    seg = json.loads(merged.read_text())["segments"][0]
    assert seg["edit_backup"]["mt_text"] == "ग्रेडिएंट यहाँ है।"


# ---------------------------------------------------------------------------
# stage invalidation
# ---------------------------------------------------------------------------

def test_rerun_runs_from_the_earliest_dirty_stage():
    assert E.rerun_stages(["mix", "tts"])[0] == "tts"
    assert E.rerun_stages(["lipsync"]) == ["lipsync"]


def test_rerun_skips_stages_the_job_never_had():
    """A job with no VC and no timing plan must not be told to replay them."""
    got = E.rerun_stages(["tts"], available=["voices", "tts", "mix", "lipsync"])
    assert got == ["tts", "mix", "lipsync"]


def test_stage_order_matches_the_pipeline():
    """Guards against STAGE_ORDER drifting away from run_post_mt_stages."""
    import inspect
    from backend import pipeline
    src = inspect.getsource(pipeline.run_post_mt_stages)
    markers = ["9b.", "9c.", "9d.", "9e.", "9f.", "9f2.", "9g.", "9h."]
    assert all(m in src for m in markers)
    assert E.STAGE_ORDER == ("codemix", "duration", "voices", "tts", "vc",
                             "timing", "mix", "lipsync")


# ---------------------------------------------------------------------------
# the fields the TTS stage has to read back
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [
    "tts_voice", "tts_style", "tts_rate_override", "tts_max_stretch",
    "tts_reference_audio",
])
def test_tts_stage_reads_every_override_the_editor_writes(key):
    """The dead-code guard.

    `backend.edits` writing a key that `backend.tts` never reads is a silent
    no-op: the run is green, the manifest looks edited, and the audio is
    unchanged. That is precisely how the gender-matching bug survived several
    dubs. Cheap source check, aimed at the exact failure.
    """
    src = Path(__file__).resolve().parent.parent / "backend" / "tts.py"
    assert f'"{key}"' in src.read_text(), f"backend/tts.py never reads {key}"


def test_pipeline_accepts_the_overrides_the_editor_produces():
    import inspect
    from backend import pipeline
    params = inspect.signature(pipeline.run_post_mt_stages).parameters
    for name in ("gender_overrides", "speaker_voice_overrides",
                 "lipsync_window"):
        assert name in params


def test_lipsync_accepts_a_window():
    import inspect
    from backend import lipsync
    assert "window" in inspect.signature(lipsync.run_lipsync).parameters


# ---------------------------------------------------------------------------
# the regex fallback (what runs with no API key)
# ---------------------------------------------------------------------------

def test_rule_parse_speed_window():
    ops = A._rule_parse("speed up the audio between 0:30 and 0:45 by 1.2x")
    assert [o.op for o in ops] == ["set_speed"]
    assert ops[0].params["factor"] == pytest.approx(1.2)
    assert (ops[0].scope.start_s, ops[0].scope.end_s) == (30.0, 45.0)


def test_rule_parse_slower_inverts_the_factor():
    ops = A._rule_parse("slow down from 10 to 20 by 1.25x")
    assert ops[0].params["factor"] == pytest.approx(0.8)


def test_rule_parse_percent():
    ops = A._rule_parse("speed up at 1:00 by 20%")
    assert ops[0].params["factor"] == pytest.approx(1.2)


def test_rule_parse_lipsync_window_with_model():
    ops = A._rule_parse("lip-sync only from 1:00 to 1:30 with musetalk")
    assert ops[0].op == "lipsync_window"
    assert ops[0].params["model"] == "musetalk"
    assert (ops[0].scope.start_s, ops[0].scope.end_s) == (60.0, 90.0)


def test_rule_parse_gender_point():
    ops = A._rule_parse("the speaker at 1:20 should sound female")
    assert ops[0].op == "set_gender"
    assert ops[0].params["gender"] == "female"
    assert ops[0].scope.start_s == pytest.approx(80.0)


def test_rule_parse_codemix_terms():
    ops = A._rule_parse("code-mix the translation, keep gradient and weight in english")
    assert ops[0].op == "code_mix"
    assert set(ops[0].params["terms"]) == {"gradient", "weight"}


def test_rule_parse_two_ops_in_one_sentence():
    ops = A._rule_parse(
        "code-mix the translation and speed up between 0:30 and 0:45 by 1.1x")
    assert {o.op for o in ops} == {"code_mix", "set_speed"}


def test_rule_parse_gives_up_rather_than_guessing():
    with pytest.raises(E.EditError):
        A.parse_request("please make it nicer somehow",
                        data={"segments": []}, client=None)


def test_unknown_op_is_refused():
    with pytest.raises(E.EditError, match="unknown op"):
        E.EditOp.from_dict({"op": "delete_everything"})


def test_flat_op_shape_is_tolerated():
    """Small models emit params flat as often as nested."""
    op = E.EditOp.from_dict({"op": "set_speed", "factor": 1.5,
                             "scope": {"start": "0:10", "end": "0:20"}})
    assert op.params["factor"] == 1.5
    assert op.scope.start_s == 10.0


# ---------------------------------------------------------------------------
# JSON extraction from a chatty model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reply", [
    '{"ops": []}',
    'Sure!\n```json\n{"ops": []}\n```\nHope that helps.',
    'Here you go: {"ops": []} — let me know.',
    '```\n{"ops": []}\n```',
])
def test_extract_json_survives_prose_and_fences(reply):
    from backend.llm import extract_json
    assert extract_json(reply) == {"ops": []}


def test_extract_json_respects_braces_inside_strings():
    from backend.llm import extract_json
    got = extract_json('{"text": "a } b", "n": 1}')
    assert got == {"text": "a } b", "n": 1}


def test_extract_json_refuses_garbage():
    from backend.llm import extract_json
    with pytest.raises(ValueError):
        extract_json("no json at all")


def test_client_without_a_key_is_unavailable(monkeypatch):
    from backend.llm import OpenRouterClient, LLMUnavailable
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    c = OpenRouterClient()
    assert not c.available()
    assert "NO KEY" in c.describe()
    with pytest.raises(LLMUnavailable):
        c.chat([{"role": "user", "content": "hi"}])


def test_describe_never_prints_the_key(monkeypatch):
    from backend.llm import OpenRouterClient
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-SECRETVALUE")
    assert "SECRET" not in OpenRouterClient().describe()


# ---------------------------------------------------------------------------
# re-run plumbing
# ---------------------------------------------------------------------------

def _job(tmp_path: Path, **cfg_over) -> Path:
    """A minimal job folder: the dirs the replay bootstrap looks for, the
    merged JSON, a manifest, and a stand-in video file."""
    root = tmp_path / "jobs" / "20261016T1200__demo__hi"
    for d in ("mt", "videos", "audio", "mix", "lipsync", "voices", "tts"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "videos" / "demo.mp4").write_bytes(b"\0" * 64)
    merged = _merged(root / "mt")
    cfg = {"diarization": "pyannote", "asr": "bodhan", "mt": "bodhan",
           "target_language": "hi", "target_script": "codemix",
           "asr_language": "en", "tts": "bodhan", "vc": None,
           "voice_engine": None, "lipsync": "musetalk", "restore": None,
           "stem_model": "htdemucs", "skip_stems": False,
           "timing_producer": None, "variant_tag": None}
    cfg.update(cfg_over)
    (root / "manifest.json").write_text(json.dumps({
        "version": 2, "job_id": root.name, "status": "ok",
        "input": str(root / "videos" / "demo.mp4"),
        "config": cfg, "argv": None}, indent=2))
    return root


def test_rerun_drives_only_the_dirty_stages(tmp_path, monkeypatch):
    """A text edit must re-run TTS onward — and nothing above it.

    The guard is on the kwargs: `translit_model=None` and the absence of any
    upstream call are what prove stages 1..9a were skipped rather than
    quietly repeated.
    """
    root = _job(tmp_path)
    view = A.open_job(root.name, data_dir=tmp_path)
    plan = A.make_plan('change the line at 1:20 to "नया।"',
                       view=view, client=None)
    A.commit(plan)
    assert plan.stages == ["tts", "mix", "lipsync"]

    seen = {}

    class FakePost:
        mix_wav = tmp_path / "m.wav"; mix_mp4 = None
        lipsync_mp4 = tmp_path / "l.mp4"; lipsync_error = None

    from backend import pipeline
    monkeypatch.setattr(pipeline, "run_post_mt_stages",
                        lambda mj, **kw: (seen.update(kw), FakePost())[1])
    out = A.rerun(plan)

    assert out["status"] == "ok"
    assert seen["tts_model"] == "bodhan"        # dirty
    assert seen["mix"] is True                  # dirty
    assert seen["lipsync_model"] == "musetalk"  # dirty
    assert seen["translit_model"] is None       # NOT dirty: no re-code-mix
    assert seen["duration_adjust"] is False
    assert seen["vc_model"] is None
    assert seen["timing_producer"] is None
    assert seen["out_root"] == root


def test_rerun_passes_speaker_overrides_through(tmp_path, monkeypatch):
    root = _job(tmp_path)
    view = A.open_job(root.name, data_dir=tmp_path)
    plan = A.make_plan("SPEAKER_01 should sound female", view=view, client=None)
    A.commit(plan)
    seen = {}

    class FakePost:
        mix_wav = mix_mp4 = lipsync_mp4 = None; lipsync_error = None

    from backend import pipeline
    monkeypatch.setattr(pipeline, "run_post_mt_stages",
                        lambda mj, **kw: (seen.update(kw), FakePost())[1])
    A.rerun(plan)
    assert seen["gender_overrides"] == {"SPEAKER_01": "female"}
    assert seen["speaker_voice_overrides"]["SPEAKER_01"]["voice"]


def test_lipsync_only_edit_skips_the_post_mt_chain(tmp_path, monkeypatch):
    """A window edit must not re-synthesise anything.

    `run_post_mt_stages` would refuse it anyway — its 9g/9h guards both need a
    TTS pass in the same call — so routing it there would silently do nothing.
    """
    root = _job(tmp_path)
    (root / "mix" / "demo.dub.wav").write_bytes(b"\0" * 64)
    view = A.open_job(root.name, data_dir=tmp_path)
    plan = A.make_plan("lip-sync only from 1:00 to 1:20", view=view, client=None)
    A.commit(plan)
    assert plan.stages == ["lipsync"]

    seen = {}

    class FakeOut:
        out_video = tmp_path / "out.mp4"; error = None

    from backend import lipsync as LS
    monkeypatch.setattr(LS, "run_lipsync",
                        lambda mj, **kw: (seen.update(kw), FakeOut())[1])
    from backend import pipeline
    monkeypatch.setattr(pipeline, "run_post_mt_stages",
                        lambda *a, **k: pytest.fail("post-MT chain must not run"))
    out = A.rerun(plan)
    assert out["stages"] == ["lipsync"]
    assert seen["window"] == (60.0, 80.0)
    assert seen["model"] == "musetalk"


def test_lipsync_only_refuses_without_a_mix(tmp_path):
    root = _job(tmp_path)          # mix/ exists but is empty
    view = A.open_job(root.name, data_dir=tmp_path)
    plan = A.make_plan("lip-sync only from 1:00 to 1:20", view=view, client=None)
    with pytest.raises(E.EditError, match="mix"):
        A.rerun(plan)


def test_job_without_mt_says_why(tmp_path):
    root = tmp_path / "jobs" / "bare"
    (root / "mt").mkdir(parents=True)
    with pytest.raises(E.EditError, match="MT stage"):
        A.open_job("bare", data_dir=tmp_path)


def test_whole_dub_gender_change_gives_each_speaker_its_own_voice(merged):
    """Two speakers must not end up reading in the same voice."""
    res = _apply(merged, E.EditOp(op="set_gender", scope=E.Scope(all=True),
                                  params={"gender": "female"}),
                 tts_engine="bodhan", voice_engine="indic-speak")
    voices = [v["voice"] for v in res.speaker_voice_overrides.values()]
    assert len(voices) == 2 and len(set(voices)) == 2


def test_exhausted_gender_pool_reports_the_reuse(merged, tmp_path):
    """Two speakers, one voice of that gender: they must share it AND the
    plan must say so rather than pretending they differ.

    indic-speak has 23 female voices, so the exhaustion is forced with a
    one-voice catalog rather than by finding a corner of the real one.
    """
    cat = tmp_path / "tiny.json"
    cat.write_text(json.dumps([
        {"engine": "indic-speak", "voice": "Solo", "gender": "female",
         "language": "hi", "tier": "lead"},
        {"engine": "indic-speak", "voice": "Solus", "gender": "male",
         "language": "hi", "tier": "lead"},
    ]))
    res = _apply(merged, E.EditOp(op="set_gender", scope=E.Scope(all=True),
                                  params={"gender": "female"}),
                 tts_engine="bodhan", voice_catalog_json=cat)
    voices = [v["voice"] for v in res.speaker_voice_overrides.values()]
    assert voices == ["Solo", "Solo"]
    assert any("no further female voice free" in w for w in res.warnings)


def test_rerun_replays_the_runs_speed_setup(tmp_path, monkeypatch):
    """An edit to a `--speed-mode segment` run must re-render in segment mode.

    `--speed-mode` and `--timing-method` override `mix_strategy`,
    `mix_clip_source` and `tts_apply_stretch` AFTER argparse, so a re-run
    built from the raw namespace comes back as a global-tempo run that also
    stretches twice — visibly different audio from the same job.
    """
    root = _job(tmp_path, timing_producer="naive_slot")
    manifest = json.loads((root / "manifest.json").read_text())
    video = root / "videos" / "demo.mp4"
    manifest["argv"] = ["-i", str(video), "--asr", "bodhan", "--mt", "bodhan",
                        "--mt-target-language", "hi", "--tts", "bodhan",
                        "--mix", "--speed-mode", "segment", "--speed-cap", "1.4"]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    view = A.open_job(root.name, data_dir=tmp_path)
    ns = A.recover_args(view)
    assert ns is not None and ns.speed_mode == "segment"

    plan = A.make_plan('change the line at 1:20 to "नया।"', view=view,
                       client=None, ns=ns)
    A.commit(plan)
    assert "timing" in plan.stages

    seen = {}

    class FakePost:
        mix_wav = mix_mp4 = lipsync_mp4 = None; lipsync_error = None

    from backend import pipeline
    monkeypatch.setattr(pipeline, "run_post_mt_stages",
                        lambda mj, **kw: (seen.update(kw), FakePost())[1])
    A.rerun(plan, data_dir=tmp_path)

    assert seen["mix_strategy"] == "timing_plan"
    assert seen["mix_clip_source"] == "raw"
    assert seen["tts_apply_stretch"] is False
    assert seen["timing_producer"] == "naive_slot"
    assert seen["timing_params"]["max_tempo"] == pytest.approx(1.4)
    assert seen["timing_params"]["safety_margin_s"] == 0.0
    assert seen["timing_params"]["use_target_duration"] is True
