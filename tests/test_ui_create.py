"""Can the UI start a job without starting something it shouldn't?

No network, no GPU, no download. `ui.ytmeta.probe` and `ui.jobqueue.spawn`
are monkeypatched, so what is exercised here is the decision-making: what is
refused, what is allowed, and what lands on disk.

Each test says which failure it guards against, because most of them guard
against something that would be invisible in a passing run:

* **The host allowlist.** yt-dlp is a general-purpose fetcher. Without the
  allowlist, `http://169.254.169.254/` is a valid thing to paste into the
  form, and this server runs on a machine with credentials in its
  environment. An IP literal is not a special case — it simply is not in the
  list — and the test pins that it stays refused.
* **The 20-minute cap is on the lip-synced portion only.** The easy mistake
  is to cap submission duration outright, which would refuse a whole-video
  audio dub of a two-hour lecture that costs minutes. Both halves are pinned.
* **The face-size picker.** Three measured clips, three recorded answers. If
  the threshold or the comparison flips, every auto-picked job quietly gets
  the wrong model and nothing errors — the only symptom is the GPU bill, or a
  blurry mouth.
* **Queue occupancy comes from disk.** A `kill -9`'d runner leaves
  `state: "running"` behind forever. If occupancy trusted that, the queue
  would wedge after the first crash and every later job would sit as
  "queued" with nothing running.
* **argv carries only the job id.** The URL is operator-supplied and must not
  reach a shell, `ps`, or a tmux window title.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ui import facepick, jobqueue, presets
from ui import routes_create as RC

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

FAKE_META = {
    "video_id": "svwrTOm55y4",
    "title": "Thermal Physics L4",
    "uploader": "NPTEL",
    "duration_s": 725.0,
    "thumbnail_url": None,
    "webpage_url": "https://www.youtube.com/watch?v=svwrTOm55y4",
    "live_status": "not_live",
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A bare app carrying only this router, rooted at a tmp data dir."""
    data = tmp_path / "data"
    (data / "jobs").mkdir(parents=True)
    monkeypatch.setattr(RC, "DATA_DIR", data)
    monkeypatch.setattr(RC, "JOBS_DIR", data / "jobs")
    monkeypatch.delenv("DUBBING_UI_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("DUBBING_UI_MAX_LIPSYNC_S", raising=False)
    monkeypatch.delenv("DUBBING_UI_MAX_CONCURRENT", raising=False)

    launched = []

    def _fake_spawn(job_id, job_root, *, rerun=False):
        launched.append({"job_id": job_id, "job_root": Path(job_root),
                         "rerun": rerun})
        return {"how": "test"}

    monkeypatch.setattr(jobqueue, "spawn", _fake_spawn)

    app = FastAPI()
    app.include_router(RC.router)
    c = TestClient(app)
    c.jobs_dir = data / "jobs"          # type: ignore[attr-defined]
    c.launched = launched               # type: ignore[attr-defined]
    return c


@pytest.fixture
def probe_ok(monkeypatch):
    """`ytmeta.probe` answering with a fixed 12-minute video."""
    from ui import ytmeta
    calls = []

    def _probe(url, _meta=FAKE_META):
        calls.append(url)
        return dict(_meta)

    monkeypatch.setattr(ytmeta, "probe", _probe)
    return calls


def _body(**over):
    body = {"url": "https://www.youtube.com/watch?v=svwrTOm55y4",
            "target_language": "hi", "preset": "draft", "tts": "bodhan"}
    body.update(over)
    return body


def test_an_engine_this_install_does_not_have_is_refused(client):
    """400 before anything is allocated, not a dead job later.

    An unrecognised engine used to pass every check: `langs.languages_for`
    answers one it does not know with the WHOLE language table, so the
    language check passed, `build_kwargs` wrote the id into `tts_model`, and
    the job was allocated, queued and launched before `synthesize_segments`
    raised `Unknown TTS model`. `edge` is the live case — jobs on disk still
    name it and a stale client could still send it.
    """
    r = client.post("/api/jobs", json=_body(tts="edge"))
    assert r.status_code == 400
    assert "removed" in r.json()["detail"].lower()
    # Refused before `allocate` ran, so no job folder was left behind.
    assert list(RC.JOBS_DIR.iterdir()) == []


# ---------------------------------------------------------------------------
# the host allowlist — the SSRF control
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://evil.example.com/watch?v=x",
    "http://169.254.169.254/latest/meta-data/",
    "https://127.0.0.1:8080/x",
    "http://[::1]/x",
    "file:///etc/passwd",
    "https://youtube.com.evil.example/watch?v=x",
])
def test_host_allowlist_refuses_everything_but_youtube(url):
    """Guards against yt-dlp being pointed at the cloud metadata endpoint, at
    loopback services, or at a lookalike domain. Each would be a server-side
    fetch of an attacker-chosen URL on a host that holds credentials."""
    with pytest.raises(Exception) as ei:
        RC.check_url(url)
    assert getattr(ei.value, "status_code", None) == 400


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=svwrTOm55y4",
    "https://youtu.be/svwrTOm55y4",
    "https://m.youtube.com/watch?v=svwrTOm55y4",
])
def test_host_allowlist_admits_youtube(url):
    assert "youtu" in RC.check_url(url)


def test_allowlist_is_overridable_but_not_by_accident(monkeypatch):
    """Widening the allowlist has to be a deliberate act, and an empty or
    whitespace value must not be read as 'allow everything'."""
    monkeypatch.setenv("DUBBING_UI_ALLOWED_HOSTS", "vimeo.com, youtu.be")
    assert RC.check_url("https://vimeo.com/1") == "vimeo.com"
    monkeypatch.setenv("DUBBING_UI_ALLOWED_HOSTS", "   ")
    assert RC.allowed_hosts() == list(RC.DEFAULT_ALLOWED_HOSTS)


def test_post_probe_redirect_is_rechecked(client, monkeypatch):
    """Guards against a shortener on an allowed host bouncing past the check:
    the submitted URL is fine, and `webpage_url` is somewhere else."""
    from ui import ytmeta
    monkeypatch.setattr(ytmeta, "probe", lambda url: dict(
        FAKE_META, webpage_url="http://169.254.169.254/latest/"))
    r = client.post("/api/jobs", json=_body())
    assert r.status_code == 400
    assert "169.254.169.254" in r.json()["detail"]


def test_live_stream_is_refused(client, monkeypatch):
    """A live stream has no duration, so the window and the cap mean nothing
    and the pipeline would run against a growing file."""
    from ui import ytmeta
    monkeypatch.setattr(ytmeta, "probe",
                        lambda url: dict(FAKE_META, live_status="is_live"))
    r = client.post("/api/jobs", json=_body())
    assert r.status_code == 400
    assert "live" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# the compute cap — lip-synced portion only
# ---------------------------------------------------------------------------

def test_cap_refuses_a_long_lipsync(client, monkeypatch):
    """90 minutes of lip-sync is 4.5 to 31 hours of GPU here. It is refused at
    submission, with the compute reason, rather than queued and discovered
    tomorrow."""
    from ui import ytmeta
    monkeypatch.setattr(ytmeta, "probe",
                        lambda url: dict(FAKE_META, duration_s=90 * 60))
    r = client.post("/api/jobs", json=_body(preset="standard"))
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "20 minutes" in detail
    # The refusal has to say *why*, or it reads as an arbitrary policy.
    assert "compute" in detail.lower() and "3-21x" in detail


def test_cap_allows_a_long_audio_only_dub(client, probe_ok, monkeypatch):
    """The cap is on the lip-synced portion, not on the submission. A
    whole-video audio dub of a two-hour lecture costs minutes and must go
    through — capping it outright is the easy, wrong implementation."""
    from ui import ytmeta
    monkeypatch.setattr(ytmeta, "probe",
                        lambda url: dict(FAKE_META, duration_s=2 * 60 * 60))
    r = client.post("/api/jobs", json=_body(preset="draft"))
    assert r.status_code == 201, r.text
    assert r.json()["state"] in ("starting", "queued")


def test_cap_costs_auto_at_the_expensive_end():
    """`auto` may resolve to MuseTalk, so the cap must cost it as MuseTalk. A
    cap that waves through its own worst case is not a cap."""
    RC.check_lipsync_cap(10 * 60, model=None)            # audio-only: no cap
    with pytest.raises(Exception):
        RC.check_lipsync_cap(30 * 60, model="musetalk")


# ---------------------------------------------------------------------------
# window validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window,needle", [
    ({"start_s": 100.0, "end_s": 50.0},  "ends at or before"),
    ({"start_s": 10.0,  "end_s": 11.0},  "at least"),
    ({"start_s": 700.0, "end_s": 900.0}, "only"),
])
def test_window_sanity(client, probe_ok, window, needle):
    """Guards against a window that cannot produce a clip: reversed, shorter
    than diarization can segment, or running off the end of the video. Each
    would otherwise surface as an ffmpeg error minutes into a detached run."""
    r = client.post("/api/jobs", json=_body(window=window))
    assert r.status_code == 400
    assert needle in r.json()["detail"]


def test_negative_start_is_refused(client, probe_ok):
    r = client.post("/api/jobs",
                    json=_body(window={"start_s": -5.0, "end_s": 30.0}))
    assert r.status_code == 422        # pydantic: ge=0 on the field itself


def test_a_good_window_is_kept_verbatim(client, probe_ok):
    r = client.post("/api/jobs",
                    json=_body(window={"start_s": 222.0, "end_s": 264.0}))
    assert r.status_code == 201, r.text
    assert r.json()["window"] == {"start_s": 222.0, "end_s": 264.0}


# ---------------------------------------------------------------------------
# languages
# ---------------------------------------------------------------------------

def test_language_is_checked_against_the_engine(client, probe_ok):
    """Guards against a job that runs for minutes and then cannot synthesise
    a word. `ui.langs` builds its list from the adapters, so this stays true
    when a voice table gains a language."""
    r = client.post("/api/jobs", json=_body(target_language="zz"))
    assert r.status_code == 400
    assert "not available" in r.json()["detail"]


# ---------------------------------------------------------------------------
# the face-size picker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("clip,median,expected",
                         [(k, v[0], v[1]) for k, v in facepick.CALIBRATION.items()])
def test_picker_reproduces_the_measured_clips(clip, median, expected):
    """The three clips the threshold was drawn from. Nothing errors when this
    flips — the only symptom is MuseTalk's 8-21x spent on a wide shot, or
    Wav2Lip's 96x96 crop on a close-up."""
    model, reason = facepick.pick_lipsync_model(median)
    assert model == expected, f"{clip}: median {median} -> {model}"
    assert f"{100 * median:.0f}%" in reason


def test_picker_states_its_basis():
    """The caveat is load-bearing: three clips from one video is a default,
    not a validated classifier, and an operator who is not told that cannot
    judge whether to override it."""
    _, reason = facepick.pick_lipsync_model(0.42)
    assert "three clips from one video" in reason


def test_picker_handles_no_faces():
    """A clip with no detected face must not be read as 'tiny faces' — and
    must not crash the run either."""
    model, reason = facepick.pick_lipsync_model(0.0)
    assert model == "wav2lip" and "No faces" in reason


def test_threshold_is_overridable(monkeypatch):
    monkeypatch.setenv("DUBBING_UI_FACE_THRESHOLD", "0.10")
    assert facepick.pick_lipsync_model(0.105)[0] == "musetalk"
    monkeypatch.setenv("DUBBING_UI_FACE_THRESHOLD", "nonsense")
    assert facepick.configured_threshold() == facepick.DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# the queue
# ---------------------------------------------------------------------------

def _progress(jobs_dir: Path, job_id: str, **fields):
    root = jobs_dir / job_id
    root.mkdir(parents=True, exist_ok=True)
    doc = {"version": 1, "job_id": job_id, "state": "running", "pid": None,
           "stages": []}
    doc.update(fields)
    (root / "progress.json").write_text(json.dumps(doc))
    (root / "request.json").write_text("{}")
    return root


def test_occupancy_ignores_a_dead_pid(tmp_path):
    """Guards against the queue wedging after the first `kill -9`. A killed
    runner leaves `state: "running"` on disk forever; if that counted, every
    later job would queue behind a process that does not exist."""
    jobs = tmp_path / "jobs"
    _progress(jobs, "dead", state="running", pid=_unused_pid())
    _progress(jobs, "alive", state="running", pid=os.getpid())
    running = {d["job_id"] for d in jobqueue.running(jobs)}
    assert running == {"alive"}
    assert jobqueue.occupancy(jobs) == 1


def test_occupancy_is_read_from_disk_not_memory(tmp_path):
    """The runner is detached on purpose, so a server restart must not forget
    a tmux job that outlived it. Nothing was registered with this process and
    the scan still finds it."""
    jobs = tmp_path / "jobs"
    _progress(jobs, "outlived-the-server", state="running", pid=os.getpid())
    assert jobqueue.occupancy(jobs) == 1


def test_queue_position_is_stable(tmp_path):
    jobs = tmp_path / "jobs"
    _progress(jobs, "a", state="queued", queued_at="2026-10-17T00:00:00+00:00")
    _progress(jobs, "b", state="queued", queued_at="2026-10-17T00:01:00+00:00")
    assert jobqueue.position(jobs, "a") == 1
    assert jobqueue.position(jobs, "b") == 2
    assert jobqueue.position(jobs, "missing") == 0


def test_second_job_queues_behind_a_running_one(client, probe_ok, monkeypatch):
    """`DUBBING_UI_MAX_CONCURRENT` defaults to 1 because MuseTalk alone peaks
    at 15.3 GiB of a 16 GiB card. The second submission must come back 202
    with a position, not 201."""
    _progress(client.jobs_dir, "already-running",
              state="running", pid=os.getpid())
    r = client.post("/api/jobs", json=_body())
    assert r.status_code == 202, r.text
    assert r.json()["queued"] is True and r.json()["position"] == 1
    assert client.launched == []       # nothing was started


def _unused_pid() -> int:
    """A pid that is not in use. 2^22 is above every Linux pid_max default."""
    pid = 4 * 1024 * 1024
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return pid
        pid += 1                                       # pragma: no cover


# ---------------------------------------------------------------------------
# what lands on disk
# ---------------------------------------------------------------------------

def test_request_json_is_written_and_argv_carries_only_the_job_id(
        client, probe_ok):
    """Guards against the operator's URL reaching a shell, `ps`, or a tmux
    window title. The runner reads everything but the job id from
    request.json, so the launcher needs no other argument."""
    url = "https://www.youtube.com/watch?v=svwrTOm55y4"
    r = client.post("/api/jobs", json=_body(url=url, preset="standard",
                                            window={"start_s": 10.0,
                                                    "end_s": 70.0}))
    assert r.status_code == 201, r.text
    job_id = r.json()["job_id"]
    root = client.jobs_dir / job_id

    req = json.loads((root / "request.json").read_text())
    assert req["url"] == url
    assert req["window"] == {"start_s": 10.0, "end_s": 70.0}
    assert req["target_language"] == "hi"
    assert req["pipeline"]["mt_target_language"] == "hi"
    # "auto" survives into the request; the runner resolves it after the
    # face probe, because nothing can resolve it before the clip exists.
    assert req["lipsync_model"] == "auto"

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["argv"] == ["ui", job_id]
    assert url not in json.dumps(manifest["argv"])

    assert client.launched and client.launched[0]["job_id"] == job_id


def test_meta_json_carries_the_title(client, probe_ok):
    """The title exists nowhere else: `download_youtube` discards yt-dlp's
    info, and a pre-cut job never calls it. Without this file the card falls
    back to the job-id slug forever."""
    r = client.post("/api/jobs", json=_body())
    root = client.jobs_dir / r.json()["job_id"]
    meta = json.loads((root / "meta.json").read_text())
    assert meta["title"] == "Thermal Physics L4"
    assert meta["video_id"] == "svwrTOm55y4"
    assert r.json()["display_name"] == "Thermal Physics L4"


def test_draft_preset_asks_for_no_lipsync(client, probe_ok):
    """`draft` exists to answer 'is the translation any good' in minutes. If
    a lip-sync model leaked into its kwargs it would take hours instead."""
    r = client.post("/api/jobs", json=_body(preset="draft"))
    root = client.jobs_dir / r.json()["job_id"]
    kw = json.loads((root / "request.json").read_text())["pipeline"]
    assert "lipsync_model" not in kw and "restore_model" not in kw
    assert kw["mix"] is True


def test_standard_preset_defers_the_lipsync_model(client, probe_ok):
    """`standard` leaves the lip-sync model to the face probe.

    `lipsync_model` must be ABSENT rather than None — a None reads as
    "lip-sync is off" to `plan_stages`, which is how the Lip sync node went
    missing from the progress bar for every default-preset job.
    """
    r = client.post("/api/jobs", json=_body(preset="standard"))
    kw = json.loads((client.jobs_dir / r.json()["job_id"]
                     / "request.json").read_text())["pipeline"]
    assert kw["face_detector"] == "retinaface"
    assert "lipsync_model" not in kw


def test_off_turns_lipsync_off_even_on_a_preset_that_has_it(client, probe_ok):
    """The form's "Off — dubbed audio only" option must actually work.

    It 400'd with `Unknown lip-sync model 'off'`: the dropdown adds that
    row client-side and the server only ever knew the registry ids plus
    "auto". Normalising it to None was not enough either — None means "take
    the preset's choice", and `standard`'s choice is `auto`, so the fix had
    to make "off" a value `build_kwargs` understands rather than an absence.
    """
    r = client.post("/api/jobs", json=_body(preset="standard",
                                            lipsync_model="off"))
    assert r.status_code == 201, r.text
    kw = json.loads((client.jobs_dir / r.json()["job_id"]
                     / "request.json").read_text())["pipeline"]
    assert "lipsync_model" not in kw
    assert "face_detector" not in kw      # no face probe for an audio-only dub
    assert kw["mix"] is True


def test_off_is_uncapped_because_nothing_is_lipsynced(client, probe_ok,
                                                      monkeypatch):
    """The lip-sync cap is on the lip-synced portion, so "off" is exempt.

    The form already knows this (`tooLong = lipsync !== 'off' && …`); the
    server has to agree or a long audio-only dub is refused for compute it
    will never spend.
    """
    monkeypatch.setenv("DUBBING_UI_MAX_LIPSYNC_S", "10")
    r = client.post("/api/jobs", json=_body(
        preset="standard", lipsync_model="off",
        window={"start_s": 0, "end_s": 600}))
    assert r.status_code == 201, r.text


def test_restoration_is_opt_in_not_a_preset_default(client, probe_ok):
    """No preset turns CodeFormer on, and that is deliberate.

    It WAS a preset default until 2026-10-17, and that was a promise the
    pipeline did not keep: the UI wrote `restore: codeformer` into every
    standard/best manifest while `_run_pipeline` never forwarded
    `restore_model` to the stage that runs it, so no UI job was ever
    restored. The forwarding bug is fixed; turning the default back on would
    have converted a silent no-op into a silent doubling of GPU time
    (CodeFormer measures ~13x clip length), so it now has to be asked for by
    name. This test is the record of that choice — if a preset starts
    shipping `restore_model` again, it should be because someone decided to,
    not because a default drifted back.
    """
    for preset in ("draft", "standard", "best"):
        r = client.post("/api/jobs", json=_body(
            preset=preset, window={"start_s": 0.0, "end_s": 60.0}))
        kw = json.loads((client.jobs_dir / r.json()["job_id"]
                         / "request.json").read_text())["pipeline"]
        assert kw.get("restore_model") is None, preset


def test_best_preset_pins_musetalk(client, probe_ok):
    r = client.post("/api/jobs",
                    json=_body(preset="best",
                               window={"start_s": 0.0, "end_s": 60.0}))
    root = client.jobs_dir / r.json()["job_id"]
    doc = json.loads((root / "request.json").read_text())
    assert doc["lipsync_model"] == "musetalk"
    assert doc["pipeline"]["lipsync_model"] == "musetalk"


def test_unknown_preset_is_refused_not_coerced(client, probe_ok):
    """A typo that quietly ran a different, slower preset would be found only
    on the GPU bill."""
    r = client.post("/api/jobs", json=_body(preset="turbo"))
    assert r.status_code == 422


def test_pipeline_kwargs_are_only_what_we_mean(client, probe_ok):
    """`_run_pipeline` takes ~60 keyword arguments. Passing one we did not
    intend is how a UI silently changes the pipeline's own defaults.

    A named parameter is not the only legitimate key: `_run_pipeline` also
    takes `**adapter_kwargs`, a flat bag it hands to every stage, each of
    which `_pick`s the names it recognises. `render_mode` lives there. So
    the check is "named parameter OR a name some stage actually picks" —
    which still catches a typo, because a misspelling is in neither set and
    would otherwise be swallowed by the bag in silence.
    """
    import inspect
    import re

    from backend import pipeline as P
    r = client.post("/api/jobs", json=_body(preset="standard"))
    root = client.jobs_dir / r.json()["job_id"]
    kw = json.loads((root / "request.json").read_text())["pipeline"]

    named = set(inspect.signature(P._run_pipeline).parameters)
    # Every name any `_pick(adapter_kwargs, (...))` in the pipeline forwards.
    src = inspect.getsource(P)
    picked = set(re.findall(r'"([a-z_]+)"',
                            " ".join(re.findall(r"_pick\(adapter_kwargs,\s*\(([^)]*)\)",
                                                src, re.S))))
    stray = set(kw) - named - picked
    assert not stray, (
        f"{sorted(stray)} is neither a _run_pipeline parameter nor a name any "
        f"stage picks out of adapter_kwargs — it would be accepted and then "
        f"silently ignored")


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------

def test_cancel_drops_a_queued_job(client, probe_ok):
    """A queued job has no process to signal, so cancel has to seal the doc
    itself — otherwise it sits in the queue forever."""
    _progress(client.jobs_dir, "busy", state="running", pid=os.getpid())
    r = client.post("/api/jobs", json=_body())
    job_id = r.json()["job_id"]
    c = client.post(f"/api/jobs/{job_id}/cancel")
    assert c.status_code == 200
    assert c.json()["state"] == "cancelled"
    doc = json.loads((client.jobs_dir / job_id / "progress.json").read_text())
    assert doc["state"] == "cancelled"


def test_cancel_seals_a_job_whose_process_is_gone(client, tmp_path):
    """Stored state says running, the pid is gone. Left alone it occupies a
    queue slot that does not exist."""
    _progress(client.jobs_dir, "zombie", state="running", pid=_unused_pid())
    r = client.post("/api/jobs/zombie/cancel")
    assert r.status_code == 200 and r.json()["state"] == "failed"


def test_cancel_404s_on_an_unknown_job(client):
    assert client.post("/api/jobs/nope/cancel").status_code == 404


def test_cancel_route_is_not_swallowed_by_a_path_converter():
    """`{job_id:path}` is greedy: it matches `abc/cancel` and the sub-route
    never fires. This pins the plain converter."""
    paths = {r.path for r in RC.router.routes}
    assert "/api/jobs/{job_id}/cancel" in paths
    assert not any(":path" in p for p in paths)


# ---------------------------------------------------------------------------
# the runner, with the download and the pipeline stubbed out
# ---------------------------------------------------------------------------

def test_runner_opens_the_three_unlogged_phases(tmp_path, monkeypatch):
    """`ui/progress.py`'s writer half had no caller before this. The pipeline
    logs 24 stage markers but logs nothing for download, cut or face probe —
    minutes of wall time each — so a job would show an unmoving bar while
    yt-dlp worked. This pins that the runner opens them itself, and that the
    markers the pipeline DOES log still land through the root logger."""
    import logging as _logging

    from ui import runner as R

    jobs = tmp_path / "jobs"
    job_id = "j1"
    root = jobs / job_id
    root.mkdir(parents=True)
    (root / "request.json").write_text(json.dumps({
        "url": "https://www.youtube.com/watch?v=x",
        "window": {"start_s": 1.0, "end_s": 3.0},
        "lipsync_model": None,
        "pipeline": {"asr_model": "whisperx", "mt_model": "bodhan",
                     "mt_target_language": "hi", "tts_model": "bodhan",
                     "mix": True, "skip_stems": True},
    }))

    src = root / "source" / "v.mp4"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    monkeypatch.setattr(R, "_fetch", lambda url, job_root: src)
    monkeypatch.setattr(R, "_cut", lambda s, j, w: j / "videos" / "clip.mp4")
    monkeypatch.setattr(R, "DATA_DIR", tmp_path)

    def _fake_pipeline(video, **kw):
        # The pipeline's own markers reach the writer through the ROOT
        # logger, which is why the handler is installed there.
        _logging.getLogger("backend.pipeline").info("=== stage 2: extract audio ===")
        _logging.getLogger("backend.pipeline").info("=== stage 9e: TTS (bodhan) ===")
        return None

    import backend.pipeline as BP
    monkeypatch.setattr(BP, "run_pipeline", _fake_pipeline)

    assert R.run(job_id, jobs_dir=jobs) == 0
    doc = json.loads((root / "progress.json").read_text())
    states = {s["key"]: s["state"] for s in doc["stages"]}
    assert doc["state"] == "done"
    for phase in R.PRE_STAGES:
        assert states[phase] in ("done", "skipped"), f"{phase}: {states[phase]}"
    assert states["2"] == "done" and states["9e"] == "done"


def test_runner_seals_the_doc_when_the_pipeline_raises(tmp_path, monkeypatch):
    """A crashed runner that left `state: "running"` behind would show as a
    job forever mid-render AND hold a queue slot nothing is using."""
    from ui import runner as R

    jobs = tmp_path / "jobs"
    root = jobs / "j2"
    root.mkdir(parents=True)
    (root / "request.json").write_text(json.dumps(
        {"url": "https://www.youtube.com/watch?v=x", "window": None,
         "lipsync_model": None, "pipeline": {}}))
    monkeypatch.setattr(R, "_fetch",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("404")))
    monkeypatch.setattr(R, "DATA_DIR", tmp_path)

    assert R.run("j2", jobs_dir=jobs) == 1
    doc = json.loads((root / "progress.json").read_text())
    assert doc["state"] == "failed" and "404" in doc["error"]


def test_runner_refuses_argv_as_a_source_of_truth(tmp_path):
    """Everything but the job id comes from request.json. Without it there is
    nothing to run, and guessing would mean re-deriving a URL from a folder
    name."""
    from ui import runner as R
    jobs = tmp_path / "jobs"
    (jobs / "j3").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        R.run("j3", jobs_dir=jobs)


# ---------------------------------------------------------------------------
# the re-render handoff
# ---------------------------------------------------------------------------

def test_rerun_reads_the_committed_record_not_pending_json(tmp_path):
    """`edits/pending.json` is DELETED on commit, and a speaker-scoped
    `set_voice` writes its `speaker_voice_overrides` only into the committed
    record — nothing goes into the merged JSON. Reading the wrong file would
    re-synthesise with the old voices, exit 0, and look to the operator
    exactly like an edit that did nothing."""
    from ui import runner
    d = tmp_path / "edits"
    d.mkdir()
    (d / "pending.json").write_text(json.dumps({"ops": [{"op": "WRONG"}]}))
    (d / "edit_000.json").write_text(json.dumps({"ops": [{"op": "old"}]}))
    (d / "edit_001.json").write_text(json.dumps({"ops": [{"op": "newest"}]}))
    rec = runner.latest_edit_record(tmp_path)
    assert rec["ops"] == [{"op": "newest"}]


def test_rerun_says_so_when_there_is_nothing_committed(tmp_path):
    """A bare pending.json is not a committed edit. Failing loudly beats
    replaying an empty op list and reporting success."""
    from ui import runner
    (tmp_path / "edits").mkdir()
    (tmp_path / "edits" / "pending.json").write_text("{}")
    with pytest.raises(FileNotFoundError) as ei:
        runner.latest_edit_record(tmp_path)
    assert "pending.json is not it" in str(ei.value)


def test_every_edit_stage_maps_to_a_pipeline_marker():
    """The ProgressWriter matches `=== stage 9e ===`, not the edit chain's
    own names. An unmapped stage shows a re-render with an empty stage list
    and a progress bar that never moves."""
    from backend.edits import STAGE_ORDER
    from ui.progress import STAGE_LABELS
    from ui.runner import EDIT_STAGE_KEYS
    assert set(STAGE_ORDER) == set(EDIT_STAGE_KEYS), (
        f"unmapped: {set(STAGE_ORDER) ^ set(EDIT_STAGE_KEYS)}")
    for name, key in EDIT_STAGE_KEYS.items():
        assert key in STAGE_LABELS, f"{name} -> stage {key} has no label"


# ---------------------------------------------------------------------------
# estimates
# ---------------------------------------------------------------------------

def test_estimates_use_the_measured_factors():
    """The form shows a number for every video. It must come from the
    measured table, not from a guess — and it must be marked as a range."""
    fast = presets.estimate_runtime(60, lipsync_model="wav2lip")
    slow = presets.estimate_runtime(60, lipsync_model="musetalk")
    slower = presets.estimate_runtime(60, lipsync_model="musetalk",
                                      restore_model="codeformer")
    assert fast["high_s"] < slow["high_s"] < slower["high_s"]
    assert fast["exact"] is False
    assert "3-5x" in fast["basis"]


# ---------------------------------------------------------------------------
# which hosted services a default run depends on
# ---------------------------------------------------------------------------

def test_asr_defaults_to_bodhan():
    """Changed from whisperx on 2026-10-17, at the user's request.

    Pinned because it moves the one stage that used to run locally onto the
    network: a Bodhan outage now takes ASR, MT and TTS together, where it
    used to leave transcription standing.
    """
    from ui import presets
    assert presets.default_asr() == "bodhan"
    assert presets.build_kwargs("standard", target_language="hi")["asr_model"] \
        == "bodhan"


def test_asr_is_still_overridable_to_a_local_model(monkeypatch):
    """The manual fallback for an outage. If this stops working there is no
    way back to a local transcriber without editing code."""
    from ui import presets
    monkeypatch.setenv("DUBBING_UI_ASR_MODEL", "whisperx")
    assert presets.build_kwargs("standard", target_language="hi")["asr_model"] \
        == "whisperx"


def test_code_mixing_is_the_mt_api_s_job_not_a_second_pass():
    """Stage 9b is deliberately not planned.

    The UI used to ask Bodhan for `native` script and then run the
    transliterator to put the English terms back — an extra call per
    retained span, editing a translation it had just asked to be made
    differently. `codemix` is `bodhan_adapter`'s own default, chosen there
    on measured grounds, so the UI now simply asks for it.

    `translit_model` must be ABSENT rather than None: `plan_stages` keys off
    presence, and a None would still read as a configured stage.
    """
    from ui import presets
    kw = presets.build_kwargs("standard", target_language="hi")
    assert kw["mt_target_script"] == "codemix"
    assert "translit_model" not in kw


def test_the_progress_bar_drops_the_codemix_stage():
    """The bar must reflect what will run, not what used to."""
    from ui import presets
    from ui.progress import plan_stages
    cfg = presets.plan_config(presets.build_kwargs("standard",
                                                   target_language="hi"))
    assert "9b" not in [s["key"] for s in plan_stages(cfg)]


def test_codemix_matches_the_adapter_s_own_default():
    """If the adapter's default ever changes, this says so rather than the
    UI silently diverging from the engine it is driving."""
    from models.mt.bodhan_adapter import DEFAULT_TARGET_SCRIPT

    from ui import presets
    assert presets.build_kwargs("standard", target_language="hi")[
        "mt_target_script"] == DEFAULT_TARGET_SCRIPT
