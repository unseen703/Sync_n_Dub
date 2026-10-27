"""The HTTP surface: routing, file serving, and the bind guard.

Three of these pin bugs that were found by running the server rather than by
reading it:

  * `/api/jobs/{job_id:path}` is GREEDY. It matched `/api/jobs/<id>/progress`
    with `job_id="<id>/progress"` and answered 404 "job not found", so every
    sub-route was dead while looking like a missing job.
  * `?download=1` was emitted by `ui/media.py` and ignored by the route, so
    the download buttons played the file in a new tab instead of saving it.
  * `StaticFiles` does not compress, so without `GZipMiddleware` the first
    load ships the whole bundle uncompressed.

The rest cover the config normalisation, where the clip runner's free-text
`mt_tts` field ("SKIPPED (downloaded YouTube dub drives audio)") was being
rendered in the UI as the name of a voice engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ui import server as S

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def client():
    return TestClient(S.app)


@pytest.fixture
def a_job():
    jobs = S.JOBS_DIR
    if not jobs.is_dir():
        pytest.skip("no data/jobs on this host")
    dirs = sorted((d for d in jobs.iterdir() if d.is_dir()),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    if not dirs:
        pytest.skip("data/jobs is empty")
    return dirs[0].name


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", ["progress", "media", "speakers", "spec", "log"])
def test_sub_routes_are_not_swallowed_by_the_job_route(client, a_job, suffix):
    """The greedy `:path` converter made every one of these 404."""
    r = client.get(f"/api/jobs/{a_job}/{suffix}")
    assert r.status_code == 200, r.text[:200]
    assert isinstance(r.json(), dict)


def test_job_detail_still_works(client, a_job):
    r = client.get(f"/api/jobs/{a_job}")
    assert r.status_code == 200
    assert r.json()["job_id"] == a_job


def test_unknown_job_is_404_not_500(client):
    for path in ("/api/jobs/no-such-job",
                 "/api/jobs/no-such-job/progress",
                 "/api/jobs/no-such-job/media"):
        assert client.get(path).status_code == 404


def test_job_list_still_works(client):
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert "jobs" in r.json() and "count" in r.json()


def test_job_list_state_comes_from_the_progress_normaliser(tmp_path, monkeypatch):
    """A dead job must not advertise itself as still running.

    `_list_jobs` used to resolve state as
    `STATUS.json -> manifest.status -> "an mp4 exists"` and never looked at
    `progress.json`, which is the shape every UI-submitted job uses. The
    manifest's `status` is written at allocation and only corrected when the
    pipeline exits through its own error path, so a job killed mid-stage
    kept `"running"` there forever — the card showed Running with a frozen
    progress bar. CLAUDE.md §2 is explicit that `read_progress` is the one
    normaliser over the four folder shapes and nothing above it may branch
    on which runner produced a job.
    """
    jobs = tmp_path / "jobs"
    root = jobs / "20261017-010101__vid__pyannote.bodhan.bodhan.hi"
    root.mkdir(parents=True)
    # The exact disagreement seen on this host: the manifest is stale.
    (root / "manifest.json").write_text(json.dumps(
        {"version": 2, "job_id": root.name, "status": "running",
         "config": {"target_language": "hi", "lipsync": "wav2lip"}}))
    (root / "progress.json").write_text(json.dumps(
        {"version": 1, "job_id": root.name, "kind": "run", "state": "failed",
         "stages": [], "error": "boom", "current": None}))

    monkeypatch.setattr(S, "JOBS_DIR", jobs)
    row = next(r for r in S._list_jobs() if r["job_id"] == root.name)
    assert row["state"] == "failed", "the stale manifest won again"


def test_options_builds_from_the_adapters(client):
    """The language list must come from the voice catalog, not a hand-copied
    list that drifts the first time an adapter gains a voice."""
    from backend.voices import load_voice_catalog

    got = client.get("/api/options").json()
    codes = {r["code"] for r in got["languages"]}
    assert codes == {(v.language or "").lower()
                     for v in load_voice_catalog("indic-speak") if v.language}
    assert len(codes) == 22          # the scheduled languages
    assert got["limits"]["max_lipsync_s"] == S.MAX_LIPSYNC_S


def test_options_offers_exactly_one_engine(client):
    """The form states the engine rather than offering a choice of engine,
    so a second row here would put a dead picker back in the UI."""
    engines = client.get("/api/options").json()["tts_engines"]
    assert [e["id"] for e in engines] == ["bodhan"]
    assert engines[0]["supports_tone"] is True


def test_options_defaults_to_the_engine_the_server_will_run(client):
    """`GET /api/options` with no `?tts=` must answer for `default_tts()`.

    The form sends no engine any more, so if this answered for a hardcoded
    one the language dropdown could offer a language the run cannot speak.
    """
    from ui import presets
    assert client.get("/api/options").json()["languages"] == \
           client.get(f"/api/options?tts={presets.default_tts()}").json()["languages"]


def test_options_states_the_lipsync_limit_with_its_reason(client):
    note = client.get("/api/options").json()["limits"]["note"]
    assert "minute" in note and ("GPU" in note or "gpu" in note)


# ---------------------------------------------------------------------------
# file serving
# ---------------------------------------------------------------------------

def test_download_flag_forces_an_attachment(client, a_job):
    media = client.get(f"/api/jobs/{a_job}/media").json()
    ref = media["audio"]["dubbed"] or media["audio"]["original"] \
        or media["video"]["original"]
    if ref is None:
        pytest.skip("this job has no playable artifact")
    assert client.get(ref["url"]).headers.get("content-disposition") is None
    disp = client.get(ref["download_url"]).headers.get("content-disposition")
    assert disp and disp.startswith("attachment;")


def test_range_requests_are_supported(client, a_job):
    """Without 206 the video scrubber cannot seek."""
    media = client.get(f"/api/jobs/{a_job}/media").json()
    ref = media["video"]["original"] or media["video"]["lipsync"]
    if ref is None:
        pytest.skip("this job has no video")
    r = client.get(ref["url"], headers={"Range": "bytes=0-1023"})
    assert r.status_code == 206
    assert r.headers.get("content-range", "").startswith("bytes 0-1023/")


def test_path_traversal_is_refused(client):
    for bad in ("../CLAUDE.md", "../../etc/passwd"):
        assert client.get(f"/files/{bad}").status_code in (400, 404)


def test_gzip_middleware_is_installed():
    """StaticFiles does not compress; without this the first load ships the
    whole bundle uncompressed."""
    assert "GZipMiddleware" in [m.cls.__name__ for m in S.app.user_middleware]


def test_static_mount_is_last():
    """`/api/*` and `/files/*` must win over the catch-all static mount."""
    paths = [getattr(r, "path", "") for r in S.app.routes]
    assert paths[-1] in ("", "/"), paths[-3:]


# ---------------------------------------------------------------------------
# config normalisation across the two runner shapes
# ---------------------------------------------------------------------------

def test_free_text_mt_tts_is_not_read_as_an_engine(tmp_path):
    """The clip runner writes `mt_tts: "SKIPPED (downloaded YouTube dub
    drives audio)"`. Mapping that onto `tts` put the sentence on screen as
    the name of a voice engine."""
    root = tmp_path / "J"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({
        "mt_tts": "SKIPPED (downloaded YouTube dub drives audio)",
        "lipsync_model": "musetalk-v15", "language": "hi"}))
    cfg = S._job_config(root)
    assert cfg.get("tts") != "SKIPPED (downloaded YouTube dub drives audio)"
    assert cfg.get("tts") is None


def test_dub_provenance_corrects_the_hardcoded_field(tmp_path):
    """DUB_PROVENANCE.json exists precisely to correct `mt_tts` — its own
    first key says so — so it is the authority when present."""
    root = tmp_path / "J"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"mt_tts": "SKIPPED"}))
    (root / "DUB_PROVENANCE.json").write_text(json.dumps({
        "corrects": "config.json field mt_tts, which is hardcoded to 'SKIPPED'",
        "pipeline": "WhisperX large-v3 ASR (en) -> Bodhan indic-translate -> "
                    "Bodhan indic-speak TTS -> per-cue rubberband fit"}))
    assert S._job_config(root)["tts"] == "bodhan"


def test_lipsync_variant_is_normalised(tmp_path):
    """`musetalk-v15` must resolve to `musetalk` or every label lookup and
    registry call misses."""
    root = tmp_path / "J"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"lipsync_model": "musetalk-v15"}))
    cfg = S._job_config(root)
    assert cfg["lipsync"] == "musetalk"
    assert cfg["lipsync_variant"] == "musetalk-v15"


def test_manifest_config_wins_over_the_flat_one(tmp_path):
    root = tmp_path / "J"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(
        {"config": {"target_language": "ta"}}))
    (root / "config.json").write_text(json.dumps({"language": "hi"}))
    assert S._job_config(root)["target_language"] == "ta"


# ---------------------------------------------------------------------------
# router mounting — order is load-bearing
# ---------------------------------------------------------------------------

def test_routers_are_not_shadowed_by_the_static_mount(client):
    """`include_router` must be called ABOVE `app.mount("/", StaticFiles(...))`.

    Starlette matches in registration order, so a router registered after the
    catch-all static mount is shadowed completely: every GET answers 404 and
    every POST 405, served by the static handler. That reads exactly like a
    routing typo and is the kind of thing nobody finds until a button does
    nothing.

    Introspecting `app.routes` does NOT catch this — this FastAPI version
    wraps included routers in an opaque `_IncludedRouter` entry, so the
    routes are invisible there whether or not they work. The only honest
    check is to call them.
    """
    probes = [
        ("GET", "/api/jobs/does-not-exist/edits"),
        ("POST", "/api/jobs/does-not-exist/edits"),
        ("POST", "/api/youtube/probe"),
        ("POST", "/api/jobs"),
        ("POST", "/api/jobs/does-not-exist/cancel"),
    ]
    shadowed = []
    for method, path in probes:
        r = client.request(method, path, json={})
        # A shadowed route is answered by StaticFiles: a bare 404 "Not Found"
        # for a GET, or a 405 for a POST it has no handler for. A LIVE route
        # answers with a domain error — 400/404-with-a-reason/409/422.
        body = r.text.lower()
        if r.status_code == 405 or (r.status_code == 404 and "not found" == body.strip('"{} ').replace('"detail":', '').strip()):
            shadowed.append((method, path, r.status_code))
    assert not shadowed, (
        f"these routes are shadowed by the static mount — move the "
        f"include_router calls above it: {shadowed}")


def test_a_missing_router_does_not_take_the_server_down(monkeypatch):
    """The read-only inspector is how you check a running GPU job. A router
    that fails to import must log and be skipped, not raise at startup."""
    import inspect

    from ui import server
    src = inspect.getsource(server)
    assert "except ImportError" in src
    assert "include_router" in src


def test_static_ui_still_serves_after_the_routers(client):
    assert client.get("/").status_code == 200


def test_rerun_launches_through_the_preloading_script():
    """A voice change dirties through to lip-sync, so an apply runs stage 9h.

    Stage 9h's adapters touch cv2/insightface, and whichever libstdc++ loads
    first decides whether `import insightface` works at all (CLAUDE.md §1).
    `scripts/ui_run_job.sh` exports the conda one; a bare
    `Popen([sys.executable, "-m", "ui.runner", ...])` does not, and the
    resulting failure looks intermittent and package-related rather than like
    a missing environment variable.
    """
    import inspect

    from ui import routes_edits
    src = inspect.getsource(routes_edits._launch_rerun)
    assert "ui_run_job.sh" in src
    assert "LD_PRELOAD" in src, "the reason must stay next to the code"

    launcher = REPO / "scripts" / "ui_run_job.sh"
    assert launcher.is_file()
    text = launcher.read_text()
    assert "LD_PRELOAD" in text
    # It must forward the flag, or --rerun silently becomes a full re-run.
    assert '"$@"' in text


# ---------------------------------------------------------------------------
# .env loading — every entry point must, or gated models refuse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module,func", [
    ("ui.server", "main"),
    ("ui.runner", "main"),
])
def test_entry_points_load_dotenv(module, func):
    """An entry point that skips `load_dotenv()` fails on the first gated model.

    The credentials the adapters need (HF_TOKEN, the Bodhan keys) live in
    `.env` and nothing puts them in the environment on its own. Every other
    entry point in the repo loads it — `backend/cli.py`,
    `backend/experiment.py`, `backend/edit_agent.py` and five scripts. The UI
    runner originally did not, and the symptom was:

        RuntimeError: pyannote community-1 requires HF_TOKEN in the environment

    which reads like a missing token rather than an unloaded file, and only
    appears once a job has already downloaded its video.
    """
    import importlib
    import inspect

    mod = importlib.import_module(module)
    src = inspect.getsource(getattr(mod, func))
    assert "load_dotenv" in src, (
        f"{module}.{func}() does not load .env — the first gated model "
        f"(pyannote, Bodhan) will refuse with what looks like a missing key")


def test_dotenv_is_a_declared_dependency():
    """`load_dotenv` is wrapped in try/ImportError, so a missing dependency
    degrades to a warning rather than a crash. It still has to be declared,
    or every fresh clone hits that degraded path."""
    req = (REPO / "requirements.txt").read_text()
    assert "python-dotenv" in req


# ---------------------------------------------------------------------------
# job cards — what the list actually shows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("job_id,ts,slug", [
    # backend.jobs.new_job_id writes "%Y%m%d-%H%M%S" ...
    ("20261017-032044__L3oOldViIgY__pyannote.whisperx.mms_fa.bodhan.hi",
     "2026-10-17 03:20:44", "L3oOldViIgY"),
    # ... the clip runner wrote "%Y%m%dT%H%M%S".
    ("20261015T012750__svwrTOm55y4_240to270__musetalk_asdgated_hi",
     "2026-10-15 01:27:50", "svwrTOm55y4_240to270"),
])
def test_both_job_id_stamp_formats_parse(job_id, ts, slug):
    """Knowing only one format made every UI-created job fail to parse, so
    its card showed the whole raw id and carried no timestamp — which is
    exactly what the job list looked like."""
    got = S._parse_job_id(job_id)
    assert got["timestamp"] == ts
    assert got["slug"] == slug


def test_the_job_list_shows_the_video_title(tmp_path, monkeypatch):
    """`meta.json` already held the title; the list endpoint never read it,
    so every card fell back to its job id."""
    jobs = tmp_path / "jobs"
    d = jobs / "20261017-032044__abc__pyannote.whisperx.none.bodhan.hi"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"title": "Kill the Batman"}))
    (d / "manifest.json").write_text(json.dumps(
        {"status": "ok", "config": {"target_language": "hi",
                                    "lipsync": "wav2lip"}}))
    monkeypatch.setattr(S, "JOBS_DIR", jobs)
    row = S._list_jobs()[0]
    assert row["title"] == "Kill the Batman"
    # And the two fields that read "not recorded" on screen.
    assert row["language"] == "hi"
    assert row["lipsync_model"] == "wav2lip"


def test_language_falls_back_to_the_clip_runner_shape(tmp_path, monkeypatch):
    """Pipeline jobs keep it in manifest config; clip jobs in config.json."""
    jobs = tmp_path / "jobs"
    d = jobs / "20261015T012750__abc__musetalk_hi"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(
        {"language": "ta", "lipsync_model": "musetalk-v15"}))
    monkeypatch.setattr(S, "JOBS_DIR", jobs)
    row = S._list_jobs()[0]
    assert row["language"] == "ta"
    assert row["lipsync_model"] == "musetalk-v15"


# ---------------------------------------------------------------------------
# a job always has a name, and it improves rather than changing identity
# ---------------------------------------------------------------------------

def test_a_name_exists_before_the_title_is_fetched():
    """The probe takes seconds and can fail. A job created in that window
    still has to be called something, and the job id is not a name."""
    from ui.ytmeta import name_from_url
    assert name_from_url("https://www.youtube.com/watch?v=L3oOldViIgY") == \
        "YouTube L3oOldViIgY"
    assert name_from_url("https://youtu.be/dQw4w9WgXcQ") == "YouTube dQw4w9WgXcQ"
    assert name_from_url("") == "Untitled video"


def test_the_fetched_title_replaces_the_placeholder():
    from ui.ytmeta import display_name
    url = "https://www.youtube.com/watch?v=L3oOldViIgY"
    assert display_name({}, url=url) == "YouTube L3oOldViIgY"
    assert display_name({"title": "Kill the Batman"}, url=url) == "Kill the Batman"


def test_every_job_gets_a_display_name_even_without_meta(tmp_path, monkeypatch):
    """Jobs made before the UI existed, and any made by the CLI, have no
    `meta.json`. They must still be called something other than their id."""
    jobs = tmp_path / "jobs"
    d = jobs / "20261015T012750__svwrTOm55y4_240to270__musetalk_hi"
    d.mkdir(parents=True)
    monkeypatch.setattr(S, "JOBS_DIR", jobs)
    row = S._list_jobs()[0]
    assert row["display_name"] == "svwrTOm55y4_240to270"
    # ...and `title` stays None, so a caller can tell a real title from a
    # stand-in and dim it accordingly.
    assert row["title"] is None


def test_a_real_title_outranks_the_placeholder(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    d = jobs / "20261017-032044__abc__pyannote.none.none.bodhan.hi"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "display_name": "YouTube abc", "title": "Kill the Batman"}))
    monkeypatch.setattr(S, "JOBS_DIR", jobs)
    row = S._list_jobs()[0]
    assert row["title"] == "Kill the Batman"


def test_cards_carry_a_timestamp_to_tell_reruns_apart(tmp_path, monkeypatch):
    """Six runs of one video is the normal way of working here, so the name
    alone cannot identify a card."""
    jobs = tmp_path / "jobs"
    for stamp in ("20261017-032044", "20261017-031438"):
        (jobs / f"{stamp}__abc__pyannote.none.none.bodhan.hi").mkdir(parents=True)
    monkeypatch.setattr(S, "JOBS_DIR", jobs)
    stamps = [r["timestamp"] for r in S._list_jobs()]
    assert all(stamps) and len(set(stamps)) == 2
