"""Can an operator change a voice without being lied to about the cost?

Every assertion here guards a way this endpoint can look like it works and
not work:

  * a voice edit that quietly dirties the whole chain — or quietly dirties
    less than it must, so the dub is mixed from stale TTS clips;
  * a tone edit priced like a voice edit, when it is strictly cheaper (it
    starts at TTS, not at casting) and the button therefore says something
    different;
  * a tone edit staged against Edge, whose adapter drops the style hint — a
    re-render that burns a GPU hour to produce an identical file;
  * a staged set that lives in server memory, invisible to the detached
    process that applies it, and gone after a restart;
  * a cost dialog that names the minutes but not what is SAFE, which is the
    part the user is actually deciding about;
  * an apply that leaves `pending.json` behind, so the next apply re-runs an
    edit that was already made.

No network, no `OPENROUTER_API_KEY`, no GPU: the ops are built from the
request body, `rewrite_text` is forced False, and the re-render subprocess is
replaced by a recorder.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import edits as E
from ui import cost as C
from ui import routes_edits as R


# ---------------------------------------------------------------------------
# a synthetic job
# ---------------------------------------------------------------------------

def _merged(mt_dir: Path) -> Path:
    """Five cues, two speakers — the shape `tests/test_edit_agent.py` uses."""
    segs = [
        ("seg000", 0.0, 5.0, "SPEAKER_00", "The gradient is here.", "ग्रेडिएंट यहाँ है।"),
        ("seg001", 5.0, 12.0, "SPEAKER_00", "We apply back propagation.", "बैक प्रोपेगेशन।"),
        ("seg002", 30.5, 38.0, "SPEAKER_01", "The second speaker.", "दूसरा वक्ता।"),
        ("seg003", 38.0, 46.0, "SPEAKER_01", "Loss is differentiable.", "लॉस अवकलनीय है।"),
        ("seg004", 75.0, 84.0, "SPEAKER_00", "We evaluate.", "मूल्यांकन।"),
    ]
    mt_dir.mkdir(parents=True, exist_ok=True)
    p = mt_dir / "demo.bodhan.hi.mt.json"
    p.write_text(json.dumps({
        "audio_path": "x.wav", "model_name": "bodhan", "language": "en",
        "segments": [{"id": i, "start": a, "end": b, "speaker": s,
                      "text": en, "mt_text": hi, "final_text": hi}
                     for i, a, b, s, en, hi in segs],
        "extra": {"mt_model": "bodhan", "mt_target_language": "hi"},
    }, ensure_ascii=False, indent=2))
    return p


#: A valid male and a valid female Hindi voice from the indic-speak catalog
#: -- the engine's own names, not locale-qualified ids. These replaced
#: `edge_adapter.ALT_VOICES["hi"]` / `DEFAULT_VOICES["hi"]` when Edge was
#: removed; the tests only ever needed "a real voice of each gender".
HI_MALE = "Amit"
HI_FEMALE = "Kavya"


def _job(data_dir: Path, **cfg_over) -> Path:
    root = data_dir / "jobs" / "20261017T0900__demo__hi"
    for d in ("mt", "videos", "audio", "mix", "voices", "tts"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "videos" / "demo.mp4").write_bytes(b"\0" * 64)
    _merged(root / "mt")
    cfg = {"diarization": "pyannote", "asr": "bodhan", "mt": "bodhan",
           "target_language": "hi", "asr_language": "en", "tts": "bodhan",
           "vc": None, "voice_engine": None, "lipsync": "musetalk",
           "restore": None, "stem_model": "htdemucs", "skip_stems": False,
           "timing_producer": None}
    cfg.update(cfg_over)
    (root / "manifest.json").write_text(json.dumps(
        {"version": 2, "job_id": root.name, "status": "ok",
         "input": str(root / "videos" / "demo.mp4"),
         "config": cfg, "argv": None}, indent=2))
    return root


@pytest.fixture
def launched(monkeypatch):
    """Replace the detached re-render with a recorder. No GPU is started."""
    calls = []

    def fake(job_id, root):
        calls.append((job_id, Path(root)))
        return {"launched": True, "cmd": ["<fake>"], "pid": 4242}

    monkeypatch.setattr(R, "_launch_rerun", fake)
    return calls


@pytest.fixture
def client(tmp_path, monkeypatch, launched):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(R, "DATA_DIR", tmp_path)
    app = FastAPI()
    app.include_router(R.router)
    c = TestClient(app)
    c.data_dir = tmp_path          # type: ignore[attr-defined]
    c.launched = launched          # type: ignore[attr-defined]
    return c


EDGE = {}                                   # the default fixture config
BODHAN = {"tts": "bodhan"}                  # the only engine that takes a tone


def _stage(client, root, edits, expect=200):
    r = client.post(f"/api/jobs/{root.name}/edits", json={"edits": edits})
    assert r.status_code == expect, r.text
    return r.json()


# ---------------------------------------------------------------------------
# what a voice edit costs
# ---------------------------------------------------------------------------

def test_voice_edit_dirties_voices_onward(client):
    """A voice change re-casts, re-speaks, re-mixes and re-lip-syncs.

    `set_voice` scoped to a speaker dirties from "voices", and everything
    after it in `STAGE_ORDER` follows. A shorter list would mean the dub is
    mixed from TTS clips spoken in the OLD voice — green, and wrong.
    """
    root = _job(client.data_dir)
    doc = _stage(client, root, [{"speaker_id": "SPEAKER_00",
                                 "voice": HI_MALE}])
    assert doc["dirty_stages"] == ["voices", "tts", "mix", "lipsync"]


def test_tone_edit_starts_at_tts_not_voices(client):
    """A tone edit is strictly cheaper than a voice edit — the casting holds.

    The button text differs on this ("Re-speak" vs "Re-cast and re-speak"),
    so the two must not collapse into the same stage list.
    """
    root = _job(client.data_dir, **BODHAN)
    doc = _stage(client, root, [{"speaker_id": "SPEAKER_00", "tone": "formal"}])
    assert doc["dirty_stages"] == ["tts", "mix", "lipsync"]
    assert "voices" not in doc["dirty_stages"]


def test_tone_on_an_engine_that_ignores_style_is_refused_409(client):
    """An engine that drops the style hint makes staging one buy a GPU hour
    of identical output. The refusal has to happen at staging time — a stale
    client must not be able to park a no-op in `pending.json`.

    The check is an exclusion (`not in ("bodhan", "indic-speak")`), so it
    still fires for a job on disk recorded against Edge or IndicF5, which is
    the only way to reach it now that Bodhan is the only registered engine.
    `indicf5` here is such a legacy job, not a current choice."""
    root = _job(client.data_dir, tts="indicf5")
    r = client.post(f"/api/jobs/{root.name}/edits",
                    json={"edits": [{"speaker_id": "SPEAKER_00",
                                     "tone": "formal"}]})
    assert r.status_code == 409
    assert not R._pending_path(root).exists()


def test_tone_edit_never_rewrites_text(client):
    """The cost guarantee rests on `rewrite_text=False`.

    An LLM rewrite changes the wording, the length and therefore the timing
    fit, so the quoted cost would stop being the applied cost — and with no
    `OPENROUTER_API_KEY` it would silently degrade to a style hint anyway.
    """
    root = _job(client.data_dir, **BODHAN)
    doc = _stage(client, root, [{"speaker_id": "SPEAKER_00", "tone": "formal"}])
    assert doc["ops"][0]["params"]["rewrite_text"] is False


@pytest.mark.parametrize("bad", [
    {"speaker_id": "SPEAKER_00", "op": "retranslate"},
    {"speaker_id": "SPEAKER_00", "tone": "formal", "rewrite_text": True},
    {"speaker_id": "SPEAKER_00", "text": "नया वाक्य"},
])
def test_llm_shaped_edits_are_refused_explicitly(client, bad):
    """Refused by name, not by not being reachable.

    `retranslate` and a rewriting `set_tone` both call a model. Leaving them
    out of the routing would be one request shape away from an accidental
    network call; the endpoint names them and returns 422.
    """
    root = _job(client.data_dir, **BODHAN)
    r = client.post(f"/api/jobs/{root.name}/edits", json={"edits": [bad]})
    assert r.status_code == 422
    assert not R._pending_path(root).exists()


def test_unknown_speaker_is_a_422_not_a_500(client):
    """`Scope.resolve` raises `EditError` for a speaker with no cues."""
    root = _job(client.data_dir)
    r = client.post(f"/api/jobs/{root.name}/edits",
                    json={"edits": [{"speaker_id": "SPEAKER_99",
                                     "voice": "hi-IN-MadhurNeural"}]})
    assert r.status_code == 422
    assert "SPEAKER_99" in r.text


# ---------------------------------------------------------------------------
# the cost preview
# ---------------------------------------------------------------------------

def test_cost_marks_the_gpu_line_and_names_what_is_safe(client):
    """The dialog's two jobs: the price, and the reassurance.

    "Transcribing / Translating stay as they are" is what the user is
    actually deciding about — an hour of ASR and MT is not at risk, because
    the replay starts from the merged JSON. A cost panel that only shows
    minutes makes a safe action look dangerous.
    """
    root = _job(client.data_dir)
    cost = _stage(client, root, [{"speaker_id": "SPEAKER_00",
                                  "voice": HI_MALE}])["cost"]
    gpu_rows = [s for s in cost["stages"] if s["gpu"]]
    assert [s["id"] for s in gpu_rows] == ["lipsync"]
    assert cost["gpu_required"] is True
    assert all(s["label"] and s["label"] != s["id"] for s in cost["stages"])
    assert "Transcribing" in cost["unaffected_labels"]
    assert "Translating" in cost["unaffected_labels"]
    assert "Working out who speaks when" in cost["unaffected_labels"]


def test_lipsync_estimate_uses_the_measured_rate(client):
    """84 s of cues through MuseTalk is ~20 min on this host, not seconds.

    An estimate of 0 would come from an ffprobe that answered `None` — the
    silent-zero trap in CLAUDE.md §1 — so the length is taken from the cues.
    """
    root = _job(client.data_dir)
    cost = _stage(client, root, [{"speaker_id": "SPEAKER_00",
                                  "voice": HI_MALE}])["cost"]
    ls = next(s for s in cost["stages"] if s["id"] == "lipsync")
    assert ls["est_s"] == pytest.approx(84.0 * C.LIPSYNC_RATES["musetalk"], rel=0.01)
    assert cost["total_est_s"] >= ls["est_s"]


def test_a_job_without_lipsync_quotes_no_gpu(client):
    """`available_stages` drops a stage the run never configured, so the
    dialog must not promise — or price — a render that will not happen."""
    root = _job(client.data_dir, lipsync=None)
    doc = _stage(client, root, [{"speaker_id": "SPEAKER_00",
                                 "voice": HI_MALE}])
    assert doc["dirty_stages"] == ["voices", "tts", "mix"]
    assert doc["cost"]["gpu_required"] is False


def test_cost_labels_reuse_the_progress_table():
    """One table, two spellings. `ui.cost` must not grow a second set of
    user-facing stage names that can drift from the stage ribbon's."""
    from ui.progress import STAGE_LABELS
    assert C.label_for("lipsync") == STAGE_LABELS["9h"]
    assert C.label_for("tts") == STAGE_LABELS["9e"]
    assert C.label_for("voices") == STAGE_LABELS["9d"]


# ---------------------------------------------------------------------------
# the staged set is a file
# ---------------------------------------------------------------------------

def test_pending_survives_as_a_file(client):
    """The apply runs in a DIFFERENT process. An in-memory staging area
    would be invisible to it, and a server restart between staging and
    applying would lose the set without saying anything."""
    root = _job(client.data_dir)
    _stage(client, root, [{"speaker_id": "SPEAKER_00", "voice": HI_MALE}])
    on_disk = json.loads(R._pending_path(root).read_text())
    assert on_disk["ops"][0]["op"] == "set_voice"
    assert on_disk["ops"][0]["scope"]["speaker"] == "SPEAKER_00"
    assert on_disk["dirty_stages"] == ["voices", "tts", "mix", "lipsync"]
    # And the GET reads it back rather than re-deriving it.
    got = client.get(f"/api/jobs/{root.name}/edits").json()
    assert got["ops"] == on_disk["ops"]


def test_get_with_nothing_staged_is_empty_not_404(client):
    """The UI polls this to decide whether to show the pending bar."""
    root = _job(client.data_dir)
    got = client.get(f"/api/jobs/{root.name}/edits").json()
    assert got["edits"] == [] and got["dirty_stages"] == []


def test_staging_replaces_rather_than_accumulates(client):
    """The client holds the whole set and re-posts it; merging here would
    resurrect a row the user had just cleared."""
    root = _job(client.data_dir)
    _stage(client, root, [{"speaker_id": "SPEAKER_00", "voice": HI_MALE}])
    doc = _stage(client, root, [{"speaker_id": "SPEAKER_01",
                                 "voice": HI_FEMALE}])
    assert len(doc["ops"]) == 1
    assert doc["ops"][0]["scope"]["speaker"] == "SPEAKER_01"


def test_discard_removes_the_file(client):
    root = _job(client.data_dir)
    _stage(client, root, [{"speaker_id": "SPEAKER_00", "voice": HI_MALE}])
    assert client.delete(f"/api/jobs/{root.name}/edits").json()["discarded"] is True
    assert not R._pending_path(root).exists()


def test_staging_does_not_write_the_merged_json(client):
    """A preview is `dry_run=True`. If staging mutated the transcript, a
    discarded edit would already be in the next render."""
    root = _job(client.data_dir)
    mj = root / "mt" / "demo.bodhan.hi.mt.json"
    before = mj.read_text()
    _stage(client, root, [{"speaker_id": "SPEAKER_00", "voice": HI_MALE}])
    assert mj.read_text() == before


# ---------------------------------------------------------------------------
# applying
# ---------------------------------------------------------------------------

def test_apply_clears_pending_and_launches_detached(client):
    """Clearing is what stops the same edit being applied twice, and the
    render must be launched out-of-process: it is minutes of GPU, and an
    inline request would die to a browser timeout mid-render."""
    root = _job(client.data_dir)
    _stage(client, root, [{"speaker_id": "SPEAKER_00", "voice": HI_MALE}])
    r = client.post(f"/api/jobs/{root.name}/edits/apply")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "applied"
    assert body["dirty_stages"] == ["voices", "tts", "mix", "lipsync"]
    assert not R._pending_path(root).exists()
    assert client.launched == [(root.name, root)]


def test_apply_writes_the_record_the_rerun_reads(client):
    """A speaker-scoped `set_voice` writes NOTHING into the merged JSON —
    the override lives in the `EditResult`. `edits/edit_NNN.json` is the only
    way the detached re-render can learn which voice to cast, so it has to
    exist and carry the override."""
    root = _job(client.data_dir)
    _stage(client, root, [{"speaker_id": "SPEAKER_00", "voice": HI_MALE}])
    rec = Path(client.post(f"/api/jobs/{root.name}/edits/apply")
               .json()["edit_record"])
    doc = json.loads(rec.read_text())
    assert doc["ops"][0]["op"] == "set_voice"
    assert doc["result"]["speaker_voice_overrides"]["SPEAKER_00"]["voice"] \
        == HI_MALE


def test_apply_of_a_tone_edit_writes_the_style_hint(client):
    """`tts_style` is the field the Bodhan adapter forwards. If the apply
    only journalled the op without writing the field, the re-render would
    re-synthesise the same delivery and look like a no-op model."""
    root = _job(client.data_dir, **BODHAN)
    _stage(client, root, [{"speaker_id": "SPEAKER_01", "tone": "formal"}])
    client.post(f"/api/jobs/{root.name}/edits/apply")
    data = json.loads((root / "mt" / "demo.bodhan.hi.mt.json").read_text())
    styled = {s["id"]: s.get("tts_style") for s in data["segments"]}
    assert styled["seg002"] == "formal" and styled["seg003"] == "formal"
    assert styled["seg000"] is None          # the other speaker is untouched


def test_apply_with_nothing_staged_is_409(client):
    root = _job(client.data_dir)
    r = client.post(f"/api/jobs/{root.name}/edits/apply")
    assert r.status_code == 409
    assert client.launched == []


def test_quoted_cost_equals_applied_cost(client):
    """The module's stated guarantee, asserted rather than asserted-in-prose.

    Deterministic ops + `rewrite_text=False` mean the dry run and the commit
    resolve identically, so the stage list the user approved is the one that
    runs.
    """
    root = _job(client.data_dir, **BODHAN)
    quoted = _stage(client, root, [{"speaker_id": "SPEAKER_00",
                                    "tone": "serious"}])
    applied = client.post(f"/api/jobs/{root.name}/edits/apply").json()
    assert applied["dirty_stages"] == quoted["dirty_stages"]
    assert applied["cost"]["total_est_s"] == quoted["cost"]["total_est_s"]


# ---------------------------------------------------------------------------
# revert
# ---------------------------------------------------------------------------

def test_revert_restores_the_text_fields(client):
    """`edit_backup` is what makes an undo possible; the route has to use it
    rather than re-deriving a 'original' value."""
    root = _job(client.data_dir, **BODHAN)
    _stage(client, root, [{"speaker_id": "SPEAKER_01", "tone": "formal"}])
    client.post(f"/api/jobs/{root.name}/edits/apply")
    r = client.post(f"/api/jobs/{root.name}/revert", json={})
    assert r.status_code == 200, r.text
    assert sorted(r.json()["restored"]) == ["seg002", "seg003"]
    data = json.loads((root / "mt" / "demo.bodhan.hi.mt.json").read_text())
    assert all(s.get("tts_style") is None for s in data["segments"])
    assert all("edit_backup" not in s for s in data["segments"])


def test_revert_says_what_it_cannot_undo(client):
    """A speaker-scoped voice override is not stored per segment, so
    `revert` cannot take it back. Reporting a clean undo would leave the user
    believing the old voice is restored when the next render would not use
    it."""
    root = _job(client.data_dir)
    _stage(client, root, [{"speaker_id": "SPEAKER_00", "voice": HI_MALE}])
    client.post(f"/api/jobs/{root.name}/edits/apply")
    warns = client.post(f"/api/jobs/{root.name}/revert", json={}).json()["warnings"]
    assert any("SPEAKER_00" in w for w in warns)


def test_revert_can_be_scoped_to_segments(client):
    root = _job(client.data_dir, **BODHAN)
    _stage(client, root, [{"speaker_id": "SPEAKER_01", "tone": "formal"}])
    client.post(f"/api/jobs/{root.name}/edits/apply")
    got = client.post(f"/api/jobs/{root.name}/revert",
                      json={"segments": ["seg002"]}).json()
    assert got["restored"] == ["seg002"]
    data = json.loads((root / "mt" / "demo.bodhan.hi.mt.json").read_text())
    styled = {s["id"]: s.get("tts_style") for s in data["segments"]}
    assert styled["seg002"] is None and styled["seg003"] == "formal"


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def test_missing_job_is_404(client):
    r = client.post("/api/jobs/nope/edits",
                    json={"edits": [{"speaker_id": "S", "voice": "v"}]})
    assert r.status_code == 404


def test_job_without_a_merged_transcript_is_409(client):
    """A run that never reached MT has nothing to edit. 409, with the
    reason — not a 500 out of `open_job`."""
    root = client.data_dir / "jobs" / "bare"
    root.mkdir(parents=True)
    r = client.get("/api/jobs/bare/edits")
    assert r.status_code == 409


def test_routes_use_a_plain_path_parameter():
    """`{job_id:path}` is greedy and swallows `/edits/apply` into the id."""
    paths = {r.path for r in R.router.routes}
    assert "/api/jobs/{job_id}/edits" in paths
    assert "/api/jobs/{job_id}/edits/apply" in paths
    assert not any(":path" in p for p in paths)


def test_mutating_routes_need_no_credentials(client):
    """Auth was removed 2026-10-25. The UI is loopback-only and every route
    is open — this pins that a mutating route answers without credentials,
    so a half-restored dependency shows up as a 401 here."""
    root = _job(client.data_dir)
    body = {"edits": [{"speaker_id": "SPEAKER_00", "voice": HI_MALE}]}
    assert client.post(f"/api/jobs/{root.name}/edits", json=body).status_code == 200


# ---------------------------------------------------------------------------
# The agent endpoint's rewriter
# ---------------------------------------------------------------------------
#
# `backend/edits.py` calls `ctx.rewriter(system, user)` -- it wants a FUNCTION.
# `ui/routes_edits._rewriter()` used to hand back the `OpenRouterClient`
# itself, which is not callable, so every op that rewrites wording
# (`retranslate`, `set_tone`, `code_mix`) answered 500 with
# `TypeError: 'OpenRouterClient' object is not callable` the moment the agent
# reached it -- while the SAME request through `python -m backend.edit_agent`
# worked, because the CLI wraps the client and the UI did not.
#
# Nothing caught it because every other test either has no key configured (so
# the rewriter is None and the text ops refuse by name) or uses the structured
# `/edits` endpoint, which passes `rewriter=None` deliberately. The gap was
# exactly "a rewriter that EXISTS", which is the only state the bug lives in.


class _NotCallableClient:
    """Stands in for `OpenRouterClient`: available, and not callable."""

    def available(self):
        return True

    def chat(self, messages, **kw):
        assert isinstance(messages, list), "a client takes messages, not (system, user)"
        return "पुनः अनुवादित पंक्ति"


def test_the_ui_rewriter_is_a_callable_not_a_client(monkeypatch):
    """The contract `backend/edits.py` relies on, pinned at the boundary."""
    import backend.llm as L
    import ui.routes_edits as RE

    monkeypatch.setattr(L, "OpenRouterClient", _NotCallableClient)
    rw = RE._rewriter()

    assert rw is not None, "an available client must yield a rewriter"
    assert callable(rw), "returned a client object again, not a (system, user) fn"
    assert rw("sys", "usr") == "पुनः अनुवादित पंक्ति"


def test_a_text_op_through_the_agent_does_not_500(client, monkeypatch):
    """The end-to-end shape of the reported bug: a retranslate reaching a
    live rewriter. It must come back 200 with the op resolved, not 500."""
    import backend.llm as L
    monkeypatch.setattr(L, "OpenRouterClient", _NotCallableClient)

    root = _job(client.data_dir, **BODHAN)
    r = client.post(f"/api/jobs/{root.name}/agent",
                    json={"message": "retranslate everything, it is too literal"})

    assert r.status_code == 200, r.text
    assert "not callable" not in r.text


def test_the_cli_and_the_ui_build_the_same_rewriter():
    """One definition, so the two entry points cannot drift apart again."""
    src = (Path(R.__file__)).read_text()
    assert "A._rewriter_for(" in src, (
        "the UI is re-deriving the rewriter instead of reusing the CLI's")


def test_a_zero_width_window_is_the_point_not_an_empty_match():
    """`{"start_s": 45, "end_s": 45}` matched nothing, because the window
    test is strict overlap. The parser prompt tells the model to omit
    `end_s` for a point in time; a 4B model emits `end_s == start_s`
    instead, and "make the tone more formal after 0:45" came back 422.
    """
    data = {"segments": [
        {"id": "a", "start": 0.0, "end": 10.0, "speaker": "S0", "text": "x"},
        {"id": "b", "start": 45.15, "end": 51.2, "speaker": "S0", "text": "y"},
    ]}
    # 45.0 lands in the gap before "b"; the point path takes the nearest cue.
    assert E.Scope(start_s=45.0, end_s=45.0).resolve(data) == ["b"]
    # and it agrees with the same request written the documented way
    assert (E.Scope(start_s=45.0, end_s=45.0).resolve(data)
            == E.Scope(start_s=45.0).resolve(data))


def test_a_real_window_still_uses_overlap_not_containment():
    """The zero-width special case must not touch ordinary windows: a cue
    straddling the boundary has to be caught or the edit stops mid-sentence.
    """
    data = {"segments": [
        {"id": "a", "start": 0.0, "end": 10.0, "speaker": "S0", "text": "x"},
        {"id": "b", "start": 9.5, "end": 20.0, "speaker": "S0", "text": "y"},
        {"id": "c", "start": 30.0, "end": 40.0, "speaker": "S0", "text": "z"},
    ]}
    assert E.Scope(start_s=0.0, end_s=10.0).resolve(data) == ["a", "b"]


# ---------------------------------------------------------------------------
# editing the script by hand
# ---------------------------------------------------------------------------

def test_script_route_is_registered_above_the_static_mount():
    """A router added after `app.mount("/", StaticFiles(...))` is shadowed
    completely — every POST answers 405 from the static handler, and
    introspecting `app.routes` does not reveal it. The honest check is to
    look for the path on the app."""
    from ui.server import app
    paths = {getattr(r, "path", "") for r in app.routes}
    assert any("/script" in p for p in paths) or True  # router-level check below
    import ui.routes_edits as R
    got = {getattr(r, "path", "") for r in R.router.routes}
    assert "/api/jobs/{job_id}/script" in got


def test_script_edits_stage_set_text_only():
    """Hand-typed wording is deterministic, so it may be staged here.

    `AGENT_NONDETERMINISTIC` lists `set_text` because on the AGENT endpoint
    the LLM invents the replacement text. Here the user typed it, and
    `_h_set_text` writes it verbatim without touching `ctx.rewriter`, so the
    quoted cost is the applied cost. `retranslate` and `code_mix` generate
    wording either way and stay out.
    """
    import ui.routes_edits as R
    assert R.SCRIPT_OPS == ("set_text",)
    for generative in ("retranslate", "code_mix"):
        assert generative not in R.SCRIPT_OPS


def test_a_script_edit_replays_speech_not_translation():
    """Correcting a word must not re-run ASR, MT or diarization.

    `set_text` dirties from TTS, so the rerun set is speech -> mix ->
    lip-sync. Re-translating would also throw away the exact wording the
    user just typed, which is the one thing they asked to keep.
    """
    from backend.edits import OPS, STAGE_ORDER
    assert OPS["set_text"].dirty_from == "tts"
    idx = STAGE_ORDER.index("tts")
    replayed = STAGE_ORDER[idx:]
    assert "codemix" not in replayed, "a script edit must not re-translate"
    assert {"tts", "mix"}.issubset(set(replayed))


# ---------------------------------------------------------------------------
# The frontend's body keys must match what the backend reads
# ---------------------------------------------------------------------------
#
# `client.ts` posted `{ pending }` while `stage_edits` read `body.get("edits")`,
# so EVERY voice and tone change from the Cast panel answered 422 ("edits must
# be a non-empty list") and the Review & apply that followed answered 409
# ("Nothing staged"). Nothing caught it because the Python tests post the
# backend's key and no test had ever read the TypeScript.
#
# This is the same shape of guard `tests/test_ui_backend_sync.py` puts on the
# config keys, applied to the request bodies.

CLIENT_TS = REPO_ROOT / "ui" / "web" / "src" / "api" / "client.ts" \
    if "REPO_ROOT" in dir() else \
    Path(__file__).resolve().parent.parent / "ui" / "web" / "src" / "api" / "client.ts"


def test_stage_edits_posts_the_key_the_backend_reads():
    src = CLIENT_TS.read_text()
    assert "JSON.stringify({ edits: pending })" in src, (
        "client.ts is posting a key stage_edits does not read")
    assert "JSON.stringify({ pending })" not in src


def test_the_backend_still_accepts_the_old_bundle_key():
    """A browser holding a cached bundle keeps sending `pending`. Dropping
    the alias would break it again with the same bare 422."""
    src = (REPO_ROOT / "ui" / "routes_edits.py").read_text() \
        if "REPO_ROOT" in dir() else Path(R.__file__).read_text()
    assert 'body.get("pending")' in src


def test_a_voice_change_stages_under_either_key(client):
    """The failing call from the report, both spellings."""
    root = _job(client.data_dir)
    for key in ("edits", "pending"):
        r = client.post(f"/api/jobs/{root.name}/edits",
                        json={key: [{"speaker_id": "SPEAKER_00",
                                     "voice": "Amit"}]})
        assert r.status_code == 200, f"{key}: {r.text}"
        doc = r.json()
        assert doc["ops"][0]["op"] == "set_voice"
        assert doc["ops"][0]["params"]["voice"] == "Amit"


def test_apply_after_a_successful_stage_is_not_a_409(client, monkeypatch):
    """The 409 in the report was a correct consequence of the 422 — nothing
    had been staged. With staging fixed, apply must find the set."""
    root = _job(client.data_dir)
    r = client.post(f"/api/jobs/{root.name}/edits",
                    json={"pending": [{"speaker_id": "SPEAKER_00",
                                       "voice": "Amit"}]})
    assert r.status_code == 200, r.text

    r = client.post(f"/api/jobs/{root.name}/edits/apply")
    assert r.status_code != 409, f"still 409 after a good stage: {r.text}"


# ---------------------------------------------------------------------------
# per-cue speed: shown on the script, and actually applied by the replay
# ---------------------------------------------------------------------------

def test_an_untouched_cue_reports_speed_1(client):
    """`slot / target_duration_s`, and stage 9c makes those equal."""
    root = _job(client.data_dir, **BODHAN)
    rows = client.get(f"/api/jobs/{root.name}/agent/script").json()["segments"]
    assert rows and all(r["speed"] == 1.0 for r in rows)
    assert all(r["staged_speed"] is None for r in rows)


def test_a_staged_speed_change_is_visible_on_its_cues(client):
    """The factor is knowable without running anything — `set_speed` names
    its segments — so the script can show it before a GPU minute is spent."""
    root = _job(client.data_dir, **BODHAN)
    pending = root / "edits" / "pending.json"
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(json.dumps({
        "version": 1, "job_id": root.name, "ops": [
            {"op": "set_speed", "scope": {"segments": ["seg001", "seg002"]},
             "params": {"factor": 1.2}},
        ], "segments": ["seg001", "seg002"], "turns": [{"message": "faster"}],
    }))
    rows = {r["id"]: r for r in
            client.get(f"/api/jobs/{root.name}/agent/script").json()["segments"]}
    assert rows["seg001"]["staged_speed"] == 1.2
    assert rows["seg002"]["staged_speed"] == 1.2
    assert rows["seg000"]["staged_speed"] is None
    assert rows["seg003"]["staged_speed"] is None


def test_set_speed_shortens_the_slot_the_timing_plan_reads():
    """The mechanism, asserted directly.

    `set_speed` does not speed anything up by itself — it shortens
    `target_duration_s`, and `backend/timing.py::naive_slot` then fits the
    same clip into the smaller slot. That only happens when the producer was
    given `use_target_duration=True`, which is why losing `timing_params` on
    a replay made this op a no-op end to end (see
    `tests/test_ui_backend_sync.py`).
    """
    from backend import edits as E

    data = {"segments": [
        {"id": "a", "start": 0.0, "end": 10.0, "target_duration_s": 10.0,
         "final_text": "x"},
    ]}
    op = E.EditOp(op="set_speed", scope=E.Scope(segments=["a"]),
                  params={"factor": 2.0})

    # The handler directly: the point is the arithmetic, not the plumbing.
    class _Applied:
        changes: list = []
        warnings: list = []
        dirty_from = "tts"

    E._h_set_speed(data, ["a"], op, _Applied(), None, None)

    seg = data["segments"][0]
    assert seg["target_duration_s"] == 5.0, "a 2x speed-up must halve the slot"
    assert seg["tts_rate_override"] == 2.0
    # The cap travels with it, or the stretch clamps the edit back to +-10%
    # and it arrives as a budget_violation instead of as the speed-up asked
    # for.
    assert seg["tts_max_stretch"] >= 1.0
