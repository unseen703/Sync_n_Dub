"""The Talking Head page: photo -> face check -> language -> line.

Each test pins a decision that would otherwise regress quietly:

  * the face check is the page's GATE, so a face-free photo must be rejected
    at upload rather than at render time;
  * a 4K still is accepted but downscaled, because MuseTalk's own S3FD finds
    no face above ~1280px and the run then dies on a divide-by-zero;
  * a line with no audio is OFFERED and marked, not hidden — the catalogue
    ships every `audio_url` null on purpose;
  * the render call refuses rather than queueing a job that cannot produce
    anything.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ui import talking_head as TH

REPO = Path(__file__).resolve().parent.parent


# --- the catalogue --------------------------------------------------------

def test_catalogue_is_valid_json_with_the_five_categories():
    doc = json.loads(TH.CATALOGUE.read_text())
    assert {c["id"] for c in doc["categories"]} == {
        "slogan", "inspiring", "funny", "catchy", "famous"}


def test_every_entry_ships_without_audio_by_design():
    """`audio_url` is null everywhere until the clips are supplied. If one
    ever arrives, this test is the reminder that `playable` starts working
    and the render path must be able to cope."""
    doc = json.loads(TH.CATALOGUE.read_text())
    for lang, rows in doc["dialogues"].items():
        for r in rows:
            assert "audio_url" in r, f"{lang}:{r.get('id')} missing the key"


def test_a_line_without_audio_is_offered_not_hidden(monkeypatch):
    """An empty page would suggest the feature is missing, when the only
    missing piece is a link.

    Driven by SYNTHETIC rows on purpose. The first version asserted that
    every Hindi line was unplayable, which was true only until the clips
    were rendered — it tested the catalogue's contents, not the rule.
    """
    monkeypatch.setattr(TH, "_load", lambda: {
        "categories": [{"id": "famous", "label": "Famous"}],
        "dialogues": {"xx": [
            {"id": "a", "category": "famous", "text": "with", "audio_url": "/f/a.wav"},
            {"id": "b", "category": "famous", "text": "without", "audio_url": None},
        ]}})
    rows = TH.dialogues_for("xx")
    assert len(rows) == 2, "the one with no audio must still be listed"
    assert [r["playable"] for r in rows] == [True, False]
    assert all(r["text"] for r in rows), "a line with no text is not a line"


def test_the_live_catalogue_has_no_entry_without_text():
    """Whatever the audio state, a blank line is never a line."""
    doc = json.loads(TH.CATALOGUE.read_text())
    for lang, rows in doc["dialogues"].items():
        for r in rows:
            assert r.get("text"), f"{lang}:{r.get('id')} has no text"


def test_every_entry_declares_a_known_category():
    doc = json.loads(TH.CATALOGUE.read_text())
    known = {c["id"] for c in doc["categories"]}
    for lang, rows in doc["dialogues"].items():
        for r in rows:
            assert r["category"] in known, f"{lang}:{r['id']} -> {r['category']}"


def test_ids_are_unique_across_the_catalogue():
    """The render route looks a line up by id; a duplicate would make which
    line you get depend on dict ordering."""
    doc = json.loads(TH.CATALOGUE.read_text())
    seen = [r["id"] for rows in doc["dialogues"].values() for r in rows]
    assert len(seen) == len(set(seen))


def test_an_unknown_language_is_empty_not_an_error():
    assert TH.dialogues_for("zz") == []
    assert TH.dialogues_for("") == []


# --- the face gate --------------------------------------------------------

def _img(tmp_path, w, h, name="x.png"):
    import cv2
    import numpy as np
    p = tmp_path / name
    cv2.imwrite(str(p), np.full((h, w, 3), 60, np.uint8))
    return p


def test_a_picture_with_no_face_is_refused(tmp_path):
    out = TH.probe_image(_img(tmp_path, 640, 480))
    assert out["ok"] is False and out["reason"] == "no_face"
    assert "portrait" in out["message"]


def test_an_unreadable_file_says_so_rather_than_raising(tmp_path):
    p = tmp_path / "not.png"
    p.write_bytes(b"nonsense")
    out = TH.probe_image(p)
    assert out["ok"] is False and out["reason"] == "unreadable"


def test_oversize_is_downscaled_to_what_the_renderer_can_handle(tmp_path):
    """Measured 2026-10-19: a 3840x2160 still made MuseTalk's S3FD return
    [None], and with no detections a print statement divides by zero. The
    render dies AFTER the model loads."""
    src = _img(tmp_path, 3840, 2160, "big.png")
    dst = TH.prepare_image(src, tmp_path / "small.png")
    import cv2
    h, w = cv2.imread(str(dst)).shape[:2]
    assert max(w, h) == TH.MAX_EDGE
    assert abs((w / h) - (3840 / 2160)) < 0.01, "aspect must be preserved"


def test_a_small_image_is_not_upscaled(tmp_path):
    src = _img(tmp_path, 400, 300, "small.png")
    import cv2
    h, w = cv2.imread(str(TH.prepare_image(src, tmp_path / "o.png"))).shape[:2]
    assert (w, h) == (400, 300)


# --- the routes -----------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from ui.server import app
    return TestClient(app)


def test_routes_are_mounted_above_the_static_handler(client):
    """Starlette matches in registration order; a router added after
    `app.mount("/", StaticFiles(...))` answers 404/405 from the static
    handler, and `app.routes` does not reveal it. Calling them is the only
    honest check."""
    assert client.get("/api/talking-head/options").status_code == 200
    assert client.get("/api/talking-head/dialogues?language=hi").status_code == 200


def test_languages_are_derived_never_hand_written(client):
    """A hand-kept copy is how the tone picker once shipped values the
    engine 422s on. Two derived sources, and nothing outside them: the TTS
    voice catalog, plus any language the DIALOGUE catalogue has lines for.

    The second source is not slack — this page plays a supplied clip and
    never synthesises, so English (a valid engine `lang` with no voice
    recorded in it) belongs here and is absent from `languages_for`.
    """
    from ui.langs import languages_for
    from ui import talking_head as T
    got = {l["code"] for l in client.get("/api/talking-head/options").json()["languages"]}
    engine = set(languages_for("bodhan"))
    catalogue = set(T.languages_with_dialogues())
    assert engine <= got, "every engine language must still be offered"
    assert got <= engine | catalogue, "no code may be invented here"
    assert got - engine <= catalogue


def test_a_language_with_no_lines_says_where_to_add_them(client):
    body = client.get("/api/talking-head/dialogues?language=ks").json()
    assert body["dialogues"] == []
    assert "dialogues.json" in body["message"]


def test_a_non_image_upload_is_refused(client):
    r = client.post("/api/talking-head/photo",
                    files={"file": ("clip.mp4", io.BytesIO(b"x"), "video/mp4")})
    assert r.status_code == 400 and "not an image" in r.json()["detail"]


def test_render_refuses_a_line_whose_audio_is_missing(client, monkeypatch):
    """409 naming the entry, rather than queueing a job that cannot produce
    anything: MuseTalk drives the mouth from a waveform, so with no clip the
    output would be the photo unchanged.

    Synthetic again — the real catalogue is fully rendered now, so pinning
    this to a live id would only hold until someone wired that id up.
    """
    from ui import routes_talkinghead as R
    monkeypatch.setattr(R.TH, "dialogues_for", lambda lang: [
        {"id": "silent", "category": "famous", "text": "x",
         "audio_url": None, "playable": False}])
    r = client.post("/api/talking-head/render",
                    json={"photo_id": "abc", "dialogue_id": "silent",
                          "language": "hi"})
    assert r.status_code == 409
    assert "no audio yet" in r.json()["detail"]


def test_a_rendered_line_gets_past_the_audio_gate(client):
    """The catalogue is rendered, so a real id must reach the render path.

    It then stops on the PHOTO, which is the next gate and the right one:
    `abc` was never uploaded. Before 2026-10-21 this asserted 501 — the
    render path did not exist — and the whole point of the change is that it
    does now, so the assertion moves to the next refusal rather than away.
    """
    r = client.post("/api/talking-head/render",
                    json={"photo_id": "abc", "dialogue_id": "hi-famous-1",
                          "language": "hi"})
    assert r.status_code == 404
    assert "no longer staged" in r.json()["detail"]


def test_render_refuses_a_link_whose_clip_is_not_on_disk(client, monkeypatch):
    """`playable` only means the URL is non-null.

    A catalogue entry pointing at a clip nobody copied across looks exactly
    like a working one, right up to the moment the GPU job fails. Checking
    the disk is the only way to tell them apart, and it costs a `stat`.
    """
    from ui import routes_talkinghead as R
    monkeypatch.setattr(R.TH, "dialogues_for", lambda lang: [
        {"id": "ghost", "category": "famous", "text": "x",
         "audio_url": "/files/talking_head/audio/not-there.wav",
         "playable": True}])
    r = client.post("/api/talking-head/render",
                    json={"photo_id": "abc", "dialogue_id": "ghost",
                          "language": "hi"})
    assert r.status_code == 409
    assert "not a file under data/" in r.json()["detail"]


@pytest.mark.parametrize("bad", [
    "../../etc", "a/b", "", ".", "..", "/abs", "x" * 200,
])
def test_a_render_id_cannot_escape_the_renders_directory(bad):
    """`render_id` arrives from the client on the poll, and `Path / ".."` is
    not a string operation.

    Rejected rather than sanitised: a cleaned-up id would read a DIFFERENT
    render's status and report it as this one's, which is worse than an
    error — the page would show someone else's video as the answer.
    """
    with pytest.raises(ValueError):
        TH.render_paths(bad)


def test_a_bad_render_id_polls_as_absent_rather_than_raising(client):
    """The route must answer, not 500: the page polls this on a timer."""
    r = client.get("/api/talking-head/render/..%2F..%2Fetc")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert r.json()["ready"] is False


def test_an_audio_url_cannot_escape_the_data_directory():
    """Catalogue rows are DATA. A `../` in one must not read an arbitrary
    file off this host just because the file happens to exist."""
    assert TH.audio_path_for("/files/../../etc/passwd") is None
    assert TH.audio_path_for("/etc/passwd") is None
    assert TH.audio_path_for("") is None


def test_a_real_catalogue_clip_resolves_to_a_file():
    """The converse — the 50 supplied clips must actually resolve, or every
    render would refuse at the gate above."""
    rows = [r for r in TH.dialogues_for("hi") if r["playable"]]
    if not rows:
        pytest.skip("no playable Hindi lines in the catalogue")
    p = TH.audio_path_for(rows[0]["audio_url"])
    assert p is not None and p.is_file()


def test_a_render_that_wrote_no_video_is_not_reported_done(tmp_path, monkeypatch):
    """The FILE is the arbiter, not the status field.

    CLAUDE.md §4 records what trusting the field costs: `run_lipsync` logged
    "done -> <path>" and recorded the path it had ASKED for, so a 40 s run
    exited 0 and left a manifest naming an mp4 that was never created.
    """
    monkeypatch.setattr(TH, "DATA_DIR", tmp_path)
    TH._write_render_status("r1", state="done", model="musetalk")
    st = TH.render_status("r1")
    assert st["state"] == "failed"
    assert st["ready"] is False
    assert "wrote no video" in st["error"]


def test_a_render_with_a_real_file_is_ready_and_has_a_url(tmp_path, monkeypatch):
    monkeypatch.setattr(TH, "DATA_DIR", tmp_path)
    TH._write_render_status("r2", state="done", model="musetalk")
    TH.render_paths("r2")["video"].write_bytes(b"not really an mp4")
    st = TH.render_status("r2")
    assert st["state"] == "done" and st["ready"] is True
    assert st["url"].startswith("/files/talking_head/renders/r2/")


def test_render_404s_on_a_line_that_does_not_exist(client):
    r = client.post("/api/talking-head/render",
                    json={"photo_id": "abc", "dialogue_id": "nope",
                          "language": "hi"})
    assert r.status_code == 404


def test_the_page_states_that_only_the_mouth_moves(client):
    """Measured 0.87% of the frame changing between output frames. A user
    expecting head motion would read a correct render as a broken one."""
    note = client.get("/api/talking-head/options").json()["note"]
    assert "mouth" in note.lower()


# --- the two bugs a real upload found -------------------------------------

@pytest.mark.parametrize("w,h", [(454, 675), (455, 674), (455, 675), (1001, 999)])
def test_prepared_images_always_have_even_sides(tmp_path, w, h):
    """libx264 with `-vf format=yuv420p` subsamples chroma by two and REFUSES
    an odd dimension:

        Could not open encoder before EOF
        Nothing was written into output file

    MuseTalk then trips over `save_dir_full` (assigned only on the VIDEO
    input path) while cleaning up, and its broad `except Exception` reports
    THAT, so the real cause never reaches the log.

    Found on a real 454x675 upload. The 1280x720 test image was even on both
    sides and hid this completely.
    """
    import cv2
    src = _img(tmp_path, w, h, f"{w}x{h}.png")
    out = cv2.imread(str(TH.prepare_image(src, tmp_path / "out.png")))
    oh, ow = out.shape[:2]
    assert ow % 2 == 0 and oh % 2 == 0, f"{ow}x{oh} would not encode"
    assert ow <= w and oh <= h, "crop down, never pad — a pad is a black edge"


def test_the_detector_import_happens_after_torch():
    """`import insightface` alone raises

        libstdc++.so.6: version `CXXABI_1.3.15' not found

    unless the conda libstdc++ is already loaded. The LD_PRELOAD that covers
    this is exported by `scripts/ui_run_job.sh`, which launches pipeline
    JOBS — the UI SERVER is started by hand and has none, so a lazy import
    inside a request turned every upload into a bare 500.
    """
    body = (Path(__file__).resolve().parent.parent / "ui" / "talking_head.py").read_text()
    torch_at = body.index("import torch")
    insight_at = body.index("from insightface.app import FaceAnalysis")
    assert torch_at < insight_at, "torch must load the conda libstdc++ first"


def test_a_dead_detector_is_reported_not_a_500(tmp_path, monkeypatch):
    """An environment fault and a bad photo need different things from
    whoever reads the message; a 500 put the reason in the server's stderr
    where the person uploading never sees it."""
    monkeypatch.setattr(TH, "_detector",
                        lambda device="cpu": (_ for _ in ()).throw(RuntimeError("boom")))
    out = TH.probe_image(_img(tmp_path, 640, 480))
    assert out["ok"] is False
    assert out["reason"] == "detector_unavailable"
    assert "not the problem" in out["message"]


def test_a_missing_insightface_names_the_interpreter(tmp_path, monkeypatch):
    """"insightface not found" is not a detector fault, it is a WRONG ENV.

    Seen live on 2026-10-21: `python -m ui.server` had been started from
    `.venv-diarization` (a uv venv carrying torch and cv2 but not
    insightface) instead of from `envs/dubbing`, and every photo upload came
    back saying the detector could not start. The message named the exception
    class, which is the one fact that does not help — `ModuleNotFoundError`
    is true of a typo, a broken install and a wrong interpreter alike.

    So this case is separated out and the message carries `sys.executable`:
    the only thing that identifies which python is actually running.
    """
    import sys

    monkeypatch.setattr(
        TH, "_detector",
        lambda device="cpu": (_ for _ in ()).throw(
            ImportError(f"insightface is not installed in the interpreter "
                        f"running this server ({sys.executable}).")))
    out = TH.probe_image(_img(tmp_path, 640, 480))
    assert out["ok"] is False
    assert out["reason"] == "detector_not_installed"
    assert sys.executable in out["message"]


def test_the_real_import_error_says_which_python_and_which_to_use(monkeypatch):
    """The sentence `_detector` builds, not the one the test above supplies."""
    import builtins
    import sys

    real = builtins.__import__

    def _no_insightface(name, *a, **k):
        if name.startswith("insightface"):
            raise ImportError("No module named 'insightface'")
        return real(name, *a, **k)

    monkeypatch.setattr(TH, "_APP", None)
    monkeypatch.setattr(builtins, "__import__", _no_insightface)
    with pytest.raises(ImportError) as e:
        TH._detector()
    msg = str(e.value)
    assert sys.executable in msg, "does not say which python is running"
    assert "dubbing" in msg, "does not say which python to use instead"


def test_english_is_offered_even_though_no_voice_is_recorded_in_it(client):
    """This page plays a SUPPLIED clip and never synthesises, so it is not
    bound to the TTS voice table the dub form is bound to. English is a
    valid engine `lang` with no voice recorded in it, so
    `languages_for("bodhan")` leaves it out — and so would this page have."""
    langs = {l["code"]: l for l in client.get("/api/talking-head/options").json()["languages"]}
    assert "en" in langs and langs["en"]["has_dialogues"]
    assert langs["en"]["n_dialogues"] > 0


def test_a_wired_audio_link_turns_the_line_playable(client):
    """The claim the page makes: add a link, no code change."""
    rows = client.get("/api/talking-head/dialogues?language=en").json()
    wired = [r for r in rows["dialogues"] if r["audio_url"]]
    assert wired, "en-famous-1 was given a real clip"
    assert all(r["playable"] for r in wired)
