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

log = logging.getLogger("ui.runner")

DATA_DIR = Path(os.environ.get("DUBBING_UI_DATA_DIR") or (REPO_ROOT / "data"))
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


def _probe_and_pick(clip: Path, requested: Optional[str]
                    ) -> Dict[str, Any]:
    """Resolve `"auto"` into a real model. Anything else passes through."""
    if requested is None:
        return {"lipsync_model": None, "reason": "Lip-sync was not requested.",
                "face": None}
    if requested != "auto":
        return {"lipsync_model": requested,
                "reason": f"{requested} was chosen explicitly.", "face": None}
    try:
        stats = facepick.face_size_stats(clip)
        median = stats.get("median")
    except Exception as e:
        # A failed probe must not lose the run. Wav2Lip is the cheap choice,
        # so defaulting to it when the measurement is missing costs GPU hours
        # only in the case where MuseTalk would have been better.
        log.warning("face probe failed (%s); defaulting to wav2lip", e)
        return {"lipsync_model": "wav2lip", "face": None,
                "reason": (f"The face-size probe failed ({e}), so the cheaper "
                           f"model was used. Re-run with an explicit model if "
                           f"the speaker fills the frame.")}
    model, reason = facepick.pick_lipsync_model(median)
    log.info("lip-sync model: %s — %s", model, reason)
    return {"lipsync_model": model, "reason": reason, "face": stats}


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

    writer = ProgressWriter(job_root, job_id=job_id, planned=planned,
                            kind="rerun")
    _attach(writer)
    _install_sigterm()
    state, error = "done", None
    try:
        edit_agent.rerun(
            edit_agent.EditPlan(view=view, nlq="(ui)", ops=ops, source="ui",
                                result=res, ctx=ctx, ns=ns),
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
    requested_ls = req.get("lipsync_model")

    plan_cfg = presets.plan_config(kwargs)
    # `lipsync_model: "auto"` means the model is picked by the face probe a
    # few seconds from now, so `kwargs["lipsync_model"]` is still None here.
    # The STAGE runs either way, and the bar has to show it from the first
    # second — otherwise every default-preset job renders a six-node bar and
    # grows a seventh when lip-sync starts.
    if not plan_cfg.get("lipsync") and requested_ls:
        plan_cfg["lipsync"] = requested_ls
    planned = plan_stages(plan_cfg, pre=list(PRE_STAGES))
    writer = ProgressWriter(job_root, job_id=job_id, planned=planned)
    _attach(writer)

    _install_sigterm()
    state, error = "done", None
    try:
        writer.begin("fetch", "download")
        src = _fetch(str(req["url"]), job_root)

        writer.begin("cut", "trim to the chosen part")
        clip = _cut(src, job_root, window)

        writer.begin("probe", "face size")
        picked = _probe_and_pick(clip, requested_ls)
        if picked["lipsync_model"]:
            kwargs["lipsync_model"] = picked["lipsync_model"]
            kwargs.setdefault("face_detector", "retinaface")
        _record_pick(job_root, picked)

        from backend.pipeline import run_pipeline
        run_pipeline(str(clip), job_id=job_id, data_dir=DATA_DIR, **kwargs)
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
