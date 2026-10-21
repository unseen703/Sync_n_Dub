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
    """The catalogue is rendered, so a real id must now reach the render
    path (501, unimplemented) instead of being turned back at the gate."""
    r = client.post("/api/talking-head/render",
                    json={"photo_id": "abc", "dialogue_id": "hi-famous-1",
                          "language": "hi"})
    assert r.status_code == 501


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
                        lambda device="cpu": (_ for _ in ()).throw(ImportError("boom")))
    out = TH.probe_image(_img(tmp_path, 640, 480))
    assert out["ok"] is False
    assert out["reason"] == "detector_unavailable"
    assert "not the problem" in out["message"]


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
