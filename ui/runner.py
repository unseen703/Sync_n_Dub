"""The detached process that actually runs a job.

    python -m ui.runner <job_id> [--rerun]

**The argv is only the job id.** Everything else — the URL, the language, the
window, the resolved pipeline kwargs — is read from `<job>/request.json`.
That is not tidiness: a URL passed on a command line reaches a shell, reaches
`ps`, and reaches the tmux window title, and this URL comes from whoever can
reach the form. Keeping it in a file also makes the job restartable, because
everything needed to re-run it is already on disk.

Sequence, which is also the sequence the progress bar shows:

    begin("fetch")  download the source            (not logged by the pipeline)
    begin("cut")    cut the window, or hardlink    (not logged by the pipeline)
    begin("probe")  measure face size, pick model  (not logged by the pipeline)
                    attach ProgressWriter to the ROOT logger
                    run_pipeline(...)              (logs its own 24 markers)
    finish()

The three leading phases exist in `progress.json` because they are minutes of
wall time that the pipeline never logs — a job that shows nothing for four
minutes while yt-dlp works looks hung.

`--rerun` is a different job entirely: it replays only the stages a committed
edit invalidated, reconstructing the plan from `<job>/edits/edit_NNN.json`
rather than from `request.json`. See `_run_rerun`.

SIGTERM is turned into an exception rather than letting the default handler
kill the process outright. A killed runner leaves `state: "running"` and a
dead pid behind; the queue then has to infer the slot is free, and the
operator sees a job that is forever "Lip-syncing". Raising means the `finally`
seals the doc as `cancelled` on the way down.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui import facepick, presets                             # noqa: E402
from ui.clip import adopt_whole, cut_window                  # noqa: E402
from ui.progress import ProgressWriter, plan_stages          # noqa: E402
from ui._io import data_dir as _data_dir                     # noqa: E402

log = logging.getLogger("ui.runner")

DATA_DIR = _data_dir()
JOBS_DIR = DATA_DIR / "jobs"

REQUEST_FILE = "request.json"

#: Phases the pipeline does not log, in the order they happen.
PRE_STAGES = ["fetch", "cut", "probe"]

#: `backend.edits.STAGE_ORDER` names -> the `=== stage N ===` markers the
#: pipeline logs for them. A re-render replays a subset of the post-MT chain,
#: and the ProgressWriter matches on the pipeline's keys, not on these names,
#: so without this map a re-render shows an empty stage list.
EDIT_STAGE_KEYS = {
    "codemix": "9b", "duration": "9c", "voices": "9d", "tts": "9e",
    "vc": "9f", "timing": "9f2", "mix": "9g", "lipsync": "9h",
}


class Cancelled(BaseException):
    """Raised from the SIGTERM handler so the `finally` clause runs.

    Derives from `BaseException`, NOT `Exception`, and that is load-bearing.
    `backend/` and `models/` are full of `except Exception` blocks that exist
    to survive a flaky upstream -- `models/tts/bodhan_adapter.py` catches one
    per segment and moves on to the next. An `Exception`-derived Cancelled is
    swallowed by every one of them, so the signal kills a segment instead of
    the run: on 2026-10-17 a Stop click during Bodhan TTS was logged 26 times
    as "synth failed for SEG_xxxx: Stopped on signal 15" while the pipeline
    carried on to mix and lip-sync a dub with 26 silent gaps in it.
    `KeyboardInterrupt` is a `BaseException` for exactly this reason.

    The two `except Cancelled` clauses in this module are each followed by
    `except BaseException`, so both still seal the progress doc on the way
    down. Keep that ordering if either handler is touched.
    """


def _terminate_children(sig: int = signal.SIGTERM) -> int:
    """Signal everything this runner shelled out to. Returns how many.

    Raising `Cancelled` in the parent does nothing to its subprocesses, and
    lip-sync is not a single child: `backend/lipsync.py` fans one job across
    every visible GPU, so a cancelled run leaves one MuseTalk/Wav2Lip process
    per card still holding ~15 GiB each. They are not in a killable process
    group of their own -- the runner is the group leader, so signalling the
    group would kill this process before it can seal the progress doc.
    """
    try:
        import psutil
    except ImportError:                                      # pragma: no cover
        return 0
    try:
        kids = psutil.Process().children(recursive=True)
    except Exception:                                        # pragma: no cover
        return 0
    for p in kids:
        try:
            p.send_signal(sig)
        except Exception:                                    # already gone
            pass
    if kids:
        psutil.wait_procs(kids, timeout=5)
        for p in kids:
            try:
                if p.is_running():
                    p.kill()
            except Exception:
                pass
    return len(kids)


def _preflight_gpu() -> None:
    """Refuse the job unless some GPU clears the free-VRAM floor.

    `backend.gpu.pick_devices` applies the same rule when the render
    actually starts; this only moves the verdict to the front so the
    operator is told immediately instead of 20 minutes in. Raises
    RuntimeError, which `ui/errors.py` maps to "The GPUs are busy right
    now" with a wait-and-retry action.

    Free means no OTHER process on the card. Our own CUDA context does not
    count against us -- it is on whichever device this pipeline already
    touched.

    No GPUs at all is NOT a refusal: a CPU-only host is a deployment
    choice, and blocking it here would be a new restriction rather than a
    guard against a measured failure.
    """
    from backend.gpu import (free_devices, gpu_busy_pids,
                             visible_cuda_devices)
    visible = visible_cuda_devices()
    if not visible:
        return
    free = free_devices()
    if free:
        log.info("gpu pre-flight: %d of %d free (%s)",
                 len(free), len(visible), ", ".join(f"gpu{g}" for g in free))
        return
    busy = gpu_busy_pids()
    who = ", ".join(f"gpu{g} (pid {','.join(str(x) for x in p)})"
                    for g, p in sorted(busy.items())) or "all devices"
    raise RuntimeError(
        f"no GPU is free — another process is using {who}. "
        f"Lip-sync needs a card to itself; the run cannot proceed.")


def _install_sigterm() -> None:
    def _handler(signum, _frame):
        # Kill the children FIRST. If anything downstream still swallows the
        # exception, the GPUs are freed either way.
        n = _terminate_children()
        if n:
            log.warning("stopped %d subprocess(es) on signal %d", n, signum)
        raise Cancelled(f"Stopped on signal {signum}.")
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):                        # pragma: no cover
            pass


def read_request(job_root: Path) -> Dict[str, Any]:
    p = Path(job_root) / REQUEST_FILE
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} is missing — the runner takes everything but the job id "
            f"from this file, so there is nothing to run.")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# the phases
# ---------------------------------------------------------------------------

def _fetch(url: str, job_root: Path) -> Path:
    """Download the source into `<job>/source/`, not `<job>/videos/`.

    `videos/` is where the *clip* goes, because `_adopt_into_job` leaves a
    file already inside it alone. Putting the full download there too would
    leave two playable mp4s in the folder that every media-resolution glob
    then has to tell apart.
    """
    from backend.downloader import resolve_input

    dest = Path(job_root) / "source"
    dest.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in dest.glob("*") if p.is_file()
                      and p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov"))
    if existing:
        log.info("source already present: %s", existing[0].name)
        return existing[0]
    return Path(resolve_input(url, dest))


def _cut(src: Path, job_root: Path, window: Optional[Dict[str, float]]) -> Path:
    videos = Path(job_root) / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    if window:
        start = float(window["start_s"])
        end = float(window["end_s"])
        out = videos / f"{src.stem}_{int(start)}to{int(end)}.mp4"
        if out.is_file() and out.stat().st_size:
            # Resuming a job whose download succeeded and whose pipeline did
            # not. Re-encoding the same window again would cost minutes and
            # produce the same bytes.
            log.info("clip already cut: %s", out.name)
            return out
        return cut_window(src, out, start, end)
    return adopt_whole(src, videos / f"{src.stem}{src.suffix or '.mp4'}")


def _probe_face(clip: Path) -> Dict[str, Any]:
    """MEASURE the face size. Does not choose anything.

    This used to be `_probe_and_pick`, and resolving `"auto"` here meant the
    most expensive stage in the pipeline was selected by a number the
    operator never saw — and silently fell back to wav2lip whenever the
    probe failed. Both halves of that are gone: the measurement is recorded
    and shown on the review screen, and the model is named by the operator at
    `POST /api/jobs/{id}/lipsync`.

    A failed probe is no longer load-bearing, so it is just a missing number.
    """
    try:
        stats = facepick.face_size_stats(clip)
    except Exception as e:                                    # noqa: BLE001
        log.warning("face probe failed (%s); the review screen will say so", e)
        return {"lipsync_model": None, "face": None,
                "reason": f"The face-size probe failed ({e})."}
    median = stats.get("median")
    hint = facepick.pick_lipsync_model(median)[1] if median else None
    log.info("face size: median=%s p90=%s", median, stats.get("p90"))
    return {"lipsync_model": None, "face": stats,
            "reason": hint or "No face was found in the sampled frames."}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def latest_edit_record(job_root: Path) -> Dict[str, Any]:
    """The newest `<job>/edits/edit_NNN.json`.

    **Not `edits/pending.json`** — that file is deleted on commit. The
    committed record is the only place a speaker-scoped `set_voice` survives:
    it writes `speaker_voice_overrides` into the record and nothing into the
    merged JSON, so a re-render that read only the merged JSON would
    re-synthesise with the OLD voices, succeed, and look to the operator
    exactly like an edit that did nothing.
    """
    d = Path(job_root) / "edits"
    records = sorted(d.glob("edit_*.json")) if d.is_dir() else []
    if not records:
        raise FileNotFoundError(
            f"{d}/edit_NNN.json is missing — there is no committed edit to "
            f"re-render. (pending.json is not it: it is deleted on commit.)")
    newest = max(records, key=lambda p: (p.name, p.stat().st_mtime))
    log.info("re-rendering from %s", newest.name)
    return json.loads(newest.read_text())


def _run_rerun(job_id: str, job_root: Path, jobs_dir: Path) -> int:
    """Replay only the stages the committed edit invalidated.

    The plan is **reconstructed**, not re-derived from the request: the
    operator approved a cost computed from these exact ops against this
    exact job, and a second derivation from different inputs could promise a
    different set of stages than the one that runs.
    """
    from backend import edit_agent, edits as _edits

    record = latest_edit_record(job_root)

    view = edit_agent.open_job(job_id, data_dir=DATA_DIR)
    ns = edit_agent.recover_args(view)
    # rewriter=None: no LLM and no network on this path. `rerun` replays the
    # run's own pacing through `cli.resolve_speed_setup` by itself.
    ctx = edit_agent.build_context(view, rewriter=None, ns=ns)
    ops = [_edits.EditOp.from_dict(o) for o in record["ops"]]

    res = _edits.apply_edits(view.merged_json, ops, ctx=ctx, nlq="(ui)",
                             dry_run=True)
    res.dirty_stages = _edits.rerun_stages(
        [a.dirty_from for a in res.applied],
        available=edit_agent.available_stages(view, res, ns=ns))

    planned = [{"key": EDIT_STAGE_KEYS.get(s, s), "title": None,
                "label": _label(EDIT_STAGE_KEYS.get(s, s)),
                "state": "pending", "started_at": None, "wall_s": None}
               for s in res.dirty_stages]
    log.info("re-render replays: %s", ", ".join(res.dirty_stages) or "(nothing)")

    # HOW the original submission was paced, straight from the request that
    # launched it.
    #
    # `edit_agent.rerun` normally recovers this by re-parsing the run's argv,
    # and a UI job has none: `ui.runner` is launched with the job id alone,
    # so the pipeline parser exits on it, `ns` is None, and the replay fell
    # through to argparse DEFAULTS. Those describe a run nobody launched —
    # `timing_params=None`, `mix_strategy=global_tempo`, stretch ON — where
    # this submission ran `naive_slot` with `use_target_duration=True`,
    # `timing_plan` and stretch OFF.
    #
    # The visible cost was a speed edit that did nothing: `set_speed` works
    # entirely by writing `target_duration_s`, and `use_target_duration`
    # defaults to False, so the replay re-fitted every clip to its untouched
    # `end - start` slot and the dub came back at the original speed. Job
    # 20261022-044302 has both plans on disk, one per pacing, as evidence.
    #
    # Read from `request.json` rather than the manifest because it is already
    # there for every job ever submitted from this UI — including the ones
    # that predate the manifest recording any of this.
    setup = None
    try:
        pipeline_kwargs = (read_request(job_root) or {}).get("pipeline") or {}
        setup = edit_agent.speed_setup_from_config(pipeline_kwargs)
    except Exception as e:                           # noqa: BLE001
        log.warning("could not read this job's pacing from request.json "
                    "(%s); the replay will fall back to the manifest", e)

    writer = ProgressWriter(job_root, job_id=job_id, planned=planned,
                            kind="rerun")
    _attach(writer)
    _install_sigterm()
    state, error = "done", None
    try:
        edit_agent.rerun(
            edit_agent.EditPlan(view=view, nlq="(ui)", ops=ops, source="ui",
                                result=res, ctx=ctx, ns=ns),
            speed_setup=setup,
            data_dir=DATA_DIR,
        )
    except Cancelled as e:
        state, error = "cancelled", str(e)
        log.warning("re-render %s cancelled: %s", job_id, e)
    except BaseException as e:                   # noqa: BLE001 - sealed below
        state, error = "failed", f"{type(e).__name__}: {e}"
        log.exception("re-render %s failed", job_id)
    finally:
        logging.getLogger().removeHandler(writer)
        writer.finish(state, error)
        writer.close()
        _pump(jobs_dir)
    return 0 if state == "done" else 1


def _label(key: str) -> str:
    from ui.progress import label_for
    return label_for(key)


def _attach(writer: ProgressWriter) -> None:
    """Install the writer on the ROOT logger, and make sure INFO reaches it.

    The handler is set to INFO, but a handler never sees a record its
    *logger* filtered first. The root logger defaults to WARNING, so without
    this line every `=== stage N ===` marker is dropped before the handler is
    consulted and `progress.json` stays on whatever `begin()` last wrote —
    a progress bar frozen at "Looking at the faces" for the whole run, with
    nothing in the log to say why. `main()` happens to call `basicConfig`
    first, which is exactly what makes this easy to not notice.
    """
    root = logging.getLogger()
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    root.addHandler(writer)


def _pump(jobs_dir: Path) -> None:
    """This process knows the slot just freed; the server may have restarted."""
    try:
        from ui.jobqueue import pump
        started = pump(jobs_dir)
        if started:
            log.info("started queued job(s): %s", ", ".join(started))
    except Exception as e:                               # pragma: no cover
        log.warning("could not start the next queued job: %s", e)


def run(job_id: str, *, rerun: bool = False, jobs_dir: Optional[Path] = None
        ) -> int:
    jobs_dir = Path(jobs_dir or JOBS_DIR)
    job_root = jobs_dir / job_id
    if not job_root.is_dir():
        raise FileNotFoundError(f"no such job folder: {job_root}")

    if rerun:
        return _run_rerun(job_id, job_root, jobs_dir)

    req = read_request(job_root)
    kwargs: Dict[str, Any] = dict(req.get("pipeline") or {})
    window = req.get("window")
    # `req["lipsync_model"]` is deliberately NOT read. An older bundle still
    # posts one and `request.json` still records what was asked for, but a
    # submission no longer lip-syncs, so acting on it here would reinstate
    # exactly the implicit render this change removed.

    plan_cfg = presets.plan_config(kwargs)
    # No lip-sync node is planned any more, and that is now honest rather
    # than a gap: phase one ends at the mix. The bar used to be padded with a
    # seventh node here because `"auto"` meant a stage that WOULD run without
    # being in `kwargs` yet. Lip-sync is a separate submission now, with its
    # own progress.
    planned = plan_stages(plan_cfg, pre=list(PRE_STAGES))
    writer = ProgressWriter(job_root, job_id=job_id, planned=planned)
    _attach(writer)

    _install_sigterm()
    state, error = "done", None
    try:
        # GPU pre-flight, BEFORE the download. A job that cannot lip-sync
        # should cost nothing: refusing here is seconds, whereas discovering
        # it at stage 9h is after the download, stems, diarization, faces,
        # ASD, and the billable Bodhan ASR/MT/TTS calls have all been paid
        # for -- and MuseTalk answers an OOM by exiting 0, so it surfaces
        # only as "produced no mp4". Audio-only dubs are not blocked.
        # Phase one never lip-syncs, so there is no GPU render to clear the
        # way for. The check moved to `POST /api/jobs/{id}/lipsync`, which is
        # where the GPU is actually about to be spent.

        writer.begin("fetch", "download")
        src = _fetch(str(req["url"]), job_root)

        writer.begin("cut", "trim to the chosen part")
        clip = _cut(src, job_root, window)

        writer.begin("probe", "face size")
        # Measured, recorded, shown — never acted on. `kwargs` is NOT given a
        # lip-sync model here: phase one always stops at the mix.
        _record_pick(job_root, _probe_face(clip))

        from backend.pipeline import run_pipeline
        result = run_pipeline(str(clip), job_id=job_id, data_dir=DATA_DIR,
                              **kwargs)

        # A lip-sync failure does NOT raise. `_run_pipeline` catches it into
        # `result.lipsync_error` and returns normally, on the reasoning that
        # the audio dub is still a usable artifact — which it is. But the
        # runner then wrote `done`, and the job card said done at 100% for a
        # job with no lip-synced video in it. Two real jobs on 2026-10-17
        # (CFymHrr5iQw, X2FV2E34Bwo) read done/100% having lost every shard
        # to the shared-temp race in `wav2lip_adapter`.
        #
        # The run asked for a lip-synced dub and did not produce one, so the
        # job failed. The message says what DID survive, because the mixed
        # audio is worth collecting.
        if getattr(result, "lipsync_error", None):
            state = "failed"
            error = (f"Lip-sync failed: {result.lipsync_error} "
                     f"The dubbed audio and its muxed video were still "
                     f"written — only the lip-synced render is missing.")
            log.error("job %s: %s", job_id, error)
    except Cancelled as e:
        state, error = "cancelled", str(e)
        log.warning("job %s cancelled: %s", job_id, e)
    except BaseException as e:                       # noqa: BLE001 - sealed below
        state = "failed"
        error = f"{type(e).__name__}: {e}"
        log.exception("job %s failed", job_id)
    finally:
        logging.getLogger().removeHandler(writer)
        writer.finish(state, error)
        writer.close()
        _pump(jobs_dir)
    return 0 if state == "done" else 1


def _record_pick(job_root: Path, picked: Dict[str, Any]) -> None:
    """Persist why this model was chosen, for the spec panel.

    Merged into `meta.json` rather than written beside it: the UI already
    reads one metadata file per job, and a second one would be a second thing
    to keep in step.
    """
    p = Path(job_root) / "meta.json"
    try:
        meta = json.loads(p.read_text()) if p.is_file() else {}
    except (OSError, ValueError):
        meta = {}
    meta["lipsync_pick"] = {
        "model": picked.get("lipsync_model"),
        "reason": picked.get("reason"),
        "face": picked.get("face"),
    }
    from ui.progress import write_atomic
    write_atomic(p, meta)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run one UI-submitted dubbing job. Reads everything but "
                    "the job id from <job>/request.json.")
    ap.add_argument("job_id")
    ap.add_argument("--rerun", action="store_true",
                    help="re-run a job folder that already has artifacts")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Every other entry point in this repo does this — `backend/cli.py`,
    # `backend/experiment.py`, `backend/edit_agent.py` and five scripts —
    # because the credentials the adapters need live in `.env` and nothing
    # puts them in the environment on their own. This runner is launched
    # detached from a web request, so it inherits even less than a shell
    # would. Without it the first gated model refuses:
    #
    #   RuntimeError: pyannote community-1 requires HF_TOKEN in the environment
    #
    # which reads like a missing token rather than an unloaded file.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:                                  # pragma: no cover
        log.warning("python-dotenv is not installed; .env was not loaded, so "
                    "HF_TOKEN and the Bodhan keys will be missing")

    return run(args.job_id, rerun=args.rerun)


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
