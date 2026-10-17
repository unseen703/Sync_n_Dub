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
    """IndicF5 drops the style hint, so staging one would buy a GPU hour of
    identical output. The refusal has to happen at staging time — a stale
    client must not be able to park a no-op in `pending.json`.

    This used to be the Edge case; with Edge gone, IndicF5 is the engine
    that reads neither `style` nor `rate`."""
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


def test_mutating_routes_carry_auth():
    """A guard that refuses to bind but protects nothing once bound is worse
    than none — every mutating route takes the dependency."""
    from ui.auth import basic_auth
    for route in R.router.routes:
        if set(route.methods) & {"POST", "DELETE", "PUT", "PATCH"}:
            deps = [d.call for d in route.dependant.dependencies]
            assert basic_auth in deps, route.path


def test_auth_is_enforced_not_merely_declared(client, monkeypatch):
    """Gradio enforced `auth=` itself; FastAPI does not. Porting only the
    refuse-to-bind half would leave a guard that protects nothing once bound,
    so the dependency is checked against a live request, not just listed."""
    root = _job(client.data_dir)
    monkeypatch.setenv("DUBBING_UI_USER", "op")
    monkeypatch.setenv("DUBBING_UI_PASS", "hunter2")
    body = {"edits": [{"speaker_id": "SPEAKER_00", "voice": HI_MALE}]}
    assert client.post(f"/api/jobs/{root.name}/edits", json=body).status_code == 401
    assert client.post(f"/api/jobs/{root.name}/edits", json=body,
                       auth=("op", "hunter2")).status_code == 200
