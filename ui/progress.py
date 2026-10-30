"""Pipeline progress: one writer, one reader, three on-disk shapes.

Why a logging handler
---------------------
`backend.jobs.Job` writes `manifest.json` exactly twice — at `allocate()` and
at `finalize()`. Between stage 1 and stage 10b nothing on disk changes, so a
UI polling the manifest sees `status: "running"` and nothing else for the
entire run.

The information *does* exist: `backend/pipeline.py` logs 24 stage markers of
the form `=== stage 9e: TTS synthesis (bodhan) ===`, each carrying the stage key
and a human title. Capturing them with a `logging.Handler` costs **no edit to
`backend/`** — which matters, because `pipeline.py` is the file the CLI, the
sweep, `backend.experiment` and `edit_agent.rerun` all depend on, and 24 edits
to it for a UI feature is 24 chances to break shared code.

The handler is installed by `ui/runner.py` (the process tmux launches), never
by the pipeline itself. `backend/` does not know this module exists.

The one fragility is that a renamed marker silently stops being matched.

Reading
-------
`read_progress(job_root)` is the only function `ui/server.py` calls. It
normalises **all three shapes** found under `data/jobs/`, so nothing above it
ever branches on which runner produced a job:

  1. `progress.json`  — written by `ProgressWriter` (new jobs)
  2. `STATUS.json`    — legacy clip-runner shape (phases[], no plan)
  3. `manifest.json`  — `backend.jobs.Job` alone (start/finish only)
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("ui.progress")

PROGRESS_VERSION = 1
PROGRESS_FILE = "progress.json"

#: Matches both marker shapes in `backend/pipeline.py`:
#:   `=== stage 9f1b: viseme extraction (mms_char) ===`   (colon)
#:   `=== stages 1-9a SKIPPED (post-MT replay) ===`       (no colon)
STAGE_MARKER_RE = re.compile(
    r"^===\s+stages?\s+"
    r"(?P<key>[0-9][0-9a-z]*(?:-[0-9a-z]+)?)"
    r"[:\s]\s*(?P<title>.*?)\s*===$"
)

#: Stage key -> what a person should read. The pipeline's own titles are
#: runner vocabulary ("code-mix / script-norm (bodhan)"); these are not.
#: Shared with the spec panel so the two never disagree.
STAGE_LABELS: Dict[str, str] = {
    # `meta` and `fetch` are the runner's own steps, before the pipeline
    # starts. `1` is the pipeline resolving what the runner already put on
    # disk -- it used to share "Fetching the video" with `fetch`, so the
    # Prepare drawer listed the same sentence twice for two different things.
    "meta":   "Reading the video's details",
    "fetch":  "Downloading the video",
    "cut":    "Trimming to the chosen part",
    "probe":  "Looking at the faces",
    "1":      "Finding the downloaded video",
    "2":      "Extracting the audio",
    "3":      "Separating background noise from audio",
    "4":      "Diarization",
    "4b":     "Tidying up the segments",
    "4c":     "Telling the speakers apart",
    "5":      "Finding faces",
    "6":      "Active speaker detection",
    "6b":     "Gender detection",
    "7":      "Transcribing",
    "8":      "Timing the words",
    "9":      "Translating",
    "9b":     "Keeping English terms in English",
    "9b1":    "Working out the tone",
    "9b2":    "Emotion prediction",
    "9c":     "Fitting the text to the timing",
    "9c2":    "Shortening lines that run long",
    "9d":     "Voice assignment",
    "9e":     "Speech generation",
    "9f1":    "Timing the new speech",
    "9f1b":   "Reading mouth shapes",
    "9f2":    "Planning the timing of new speech",
    "9g":     "Mixing the new soundtrack",
    "9h":     "Lip-syncing",
    "9i":     "Enhancing resolution",
    "10a":    "Checking the transcript",
    "10b":    "Checking the translation",
    "1-9a":   "Reusing the earlier run",
}

#: The pipeline as a non-technical user describes it: four outcome-level
#: stages plus a synthetic Finished pill. The 24 internal markers fold into
#: these; the substages still travel in each phase for the expandable detail
#: row, but the ribbon itself shows only this many nodes.
#: Grouped by what an operator expects to find under each heading, not by
#: where the stage sits in the pipeline source. Diarization and ASR are
#: filed under `Translate` because "who said what, and what did they say"
#: is the work that produces the script; ASD under `Lip-sync` because
#: picking the speaking face is what the render needs it for.
#:
#: 9b1 is under `Translate` and 6b under `Prepare` for one shared reason:
#: both run BEFORE the MT gate. Filed under `Dub audio` either one would
#: make a job stopped at the gate draw a Dub pill with a stage done in it.
PHASE_GROUPS: List[tuple] = [
    ("prepare",   "Prepare",    ("meta", "fetch", "cut", "probe", "1", "2",
                                 "3", "1-9a", "5", "6b")),
    ("translate", "Translate",  ("4", "4b", "4c", "7", "8",
                                 "9", "9b", "9b1", "9c", "9c2")),
    ("dub",       "Dub audio",  ("9b2", "9d", "9e",
                                 "9f1", "9f1b", "9f2", "9g")),
    ("lipsync",   "Lip-sync",   ("6", "9h", "9i")),
]

#: Stages 10a/10b score the ASR and MT against ground-truth subtitles. They
#: are an optional QA pass that most runs skip, and they produce no part of
#: the dub — so they are kept out of the bar rather than appearing as an
#: eighth stage nobody asked about. They still show in the log.
#:
#: The second group is housekeeping the operator did not ask for and cannot
#: act on (2026-10-28, by request): trimming, the face-size probe, resolving
#: the file the runner just downloaded, the two optional diarization
#: clean-ups, face detection, and word-level alignment. Each still runs,
#: still logs, and still sits in `doc["stages"]` — they are just not steps
#: anyone reads the bar to find out about.
#:
#: 9i (enhancement) is NOT hidden: it belongs under Lip-sync and shows there
#: whenever it runs. What it must not be is PLANNED on a run that is not
#: doing it -- see `plan_stages`.
HIDDEN_KEYS: tuple = ("10a", "10b",
                      "cut", "probe", "1", "4b", "4c", "5", "8",
                      "1-9a", "9b", "9b1", "9c", "9f1", "9f1b")

_GROUP_OF: Dict[str, str] = {
    key: gid for gid, _label, keys in PHASE_GROUPS for key in keys
}

#: The clip runner's phases map onto the same ribbon, so a clip job and a
#: pipeline job read the same way on screen. Without this they all land in a
#: single meaningless "Other" pill.
_GROUP_OF.update({
    # P3_ASD follows stage 6 into `lipsync`: same work, same reason.
    "P1_CUT": "prepare", "P2_FACES": "prepare", "P3_ASD": "lipsync",
    "P4_MIX": "dub", "P5_LIPSYNC_BBOX_PREVIEW": "lipsync",
    "P6_LIPSYNC": "lipsync", "P7_SPLICE": "lipsync", "P8_VERIFY": "lipsync",
    "P5_SEGMENT": "prepare",
})

#: Phase names used by the legacy clip-runner's STATUS.json.
CLIP_PHASE_LABELS: Dict[str, str] = {
    "P1_CUT":                    "Trimming to the chosen part",
    "P2_FACES":                  "Finding faces",
    "P3_ASD":                    "Working out who is on screen",
    "P4_MIX":                    "Mixing the new soundtrack",
    "P5_LIPSYNC_BBOX_PREVIEW":   "Previewing the face crop",
    "P6_LIPSYNC":                "Lip-syncing",
    "P7_SPLICE":                 "Reassembling the video",
    "P8_VERIFY":                 "Checking the result",
}

TERMINAL = ("done", "failed", "cancelled")


def label_for(key: str, fallback: str = "") -> str:
    return STAGE_LABELS.get(key) or fallback or f"Stage {key}"


# ---------------------------------------------------------------------------
# planning the stage list up front
# ---------------------------------------------------------------------------

def plan_stages(config: Dict[str, Any], *, pre: Optional[List[str]] = None,
                stop_after_mt: bool = False) -> List[Dict[str, Any]]:
    """The stages this config will run, so the UI can say "5 of 18".

    Mirrors the conditionals in `_run_pipeline`, so it can drift. It is
    reconciled as markers arrive: an unplanned key is appended, a planned key
    that is skipped over is marked `skipped`. A drift degrades the up-front
    count, never the live state.

    `stop_after_mt` is what a UI submission plans with: the run ends at the
    gate. The boundary is `run_post_mt_stages`, not the Translate pill -- 9b
    and 9c live in that pill but run after the gate, and 9b1 before it.

    Post-gate keys are still planned, marked `deferred`. Dropping them left
    the `dub` and `lipsync` groups with no substages, and `_group_phases`
    reads an empty group as `skipped`, which the ribbon paints done -- so the
    bar claimed a lip-sync render that had never happened.
    """
    keys: List[str] = list(pre or ["fetch", "cut"])
    keys += ["1", "2"]
    if not config.get("skip_stems"):
        keys.append("3")
    keys.append("4")
    if config.get("resegment_turns"):
        keys.append("4b")
    if config.get("refine_speakers"):
        keys.append("4c")
    if config.get("face_detector") or config.get("lipsync"):
        keys.append("5")
    # 6 (ASD) is NOT planned here any more. Its marker is emitted at the
    # lip-sync call site, because the pre-gate ASD pass belongs to 6b's
    # gender vote -- so the row is planned down beside 9h, where it fires.
    # Planned up here it never opened, got swept to `skipped` on a clean
    # finish, and drew a done-looking Lip-sync pill: the original bug.
    if stop_after_mt:
        # 6b: every speaker's gender, predicted before the gate so the cast
        # table has one to show when it asks for a voice. Phase one only --
        # a CLI run gets gender inside 9d.
        keys.append("6b")
    if config.get("asr"):
        keys.append("7")
    if config.get("align"):
        keys.append("8")
    if config.get("mt"):
        keys.append("9")

    # 9b1 is the last marker phase one emits, so under `stop_after_mt` it
    # belongs on this side of the gate even though every other 9b* key does
    # not.
    gated_9b1 = stop_after_mt and config.get("tts_style") == "auto"
    if gated_9b1:
        keys.append("9b1")

    # Everything appended below this line runs inside `run_post_mt_stages`.
    # Under `stop_after_mt` the run stops before all of it, so every key from
    # here on is marked deferred at the bottom of this function.
    gate = len(keys)
    if config.get("translit"):
        keys.append("9b")
    if config.get("tts_style") == "auto" and not gated_9b1:
        keys.append("9b1")
    if config.get("emotion"):
        keys.append("9b2")
    if config.get("duration_adjust"):
        keys.append("9c")
    # 9c2 only runs inside the `duration_adjust` branch -- it acts on the
    # per-segment estimates 9c writes -- so it cannot be planned without it.
    if config.get("duration_adjust") and config.get("condense_overlong_text"):
        keys.append("9c2")
    if config.get("voice_engine") or config.get("tts"):
        keys.append("9d")
    if config.get("tts"):
        keys.append("9e")
    if config.get("tts_force_align"):
        keys.append("9f1")
    if config.get("visemes"):
        keys.append("9f1b")
    if config.get("timing_producer"):
        keys.append("9f2")
    if config.get("mix") or config.get("lipsync"):
        keys.append("9g")

    # 9h is planned even when this run will not reach it -- the one place
    # this function deliberately does not mirror `_run_pipeline`.
    #
    # A draft submission sets `lipsync: None` and stops at the mix; lip-sync
    # is then a separate act (`POST .../lipsync`). Keyed off the config, such
    # a job drew a bar reading "Finished - all 6 stages" over a dub still
    # waiting for its most expensive stage. The node is drawn and left
    # pending instead.
    #
    # Only when the run produces something to lip-sync TO: a config with no
    # TTS and no mix was never going near 9h.
    #
    # `deferred` carries this through `finish()` and `_from_manifest`, which
    # otherwise resolve a leftover `pending` to `skipped`. Skipped is a
    # decision; deferred is a stage that has not had its turn.
    deferred: set = set()
    _lipsync_keys = (["6"] if config.get("asd") else []) + ["9h"]
    if config.get("lipsync"):
        keys += _lipsync_keys
    elif config.get("mix") or config.get("tts"):
        keys += _lipsync_keys
        deferred = set(_lipsync_keys)

    # 9i is planned ONLY when this run will actually restore. It used to be
    # planted deferred-and-pending beside every 9h, and because enhancement
    # never writes `progress.json` -- `POST .../enhance` runs in its own
    # thread with its own polling -- that row could never resolve: once a
    # real 9h render finished the group rolled up ["done", "pending"] ->
    # "running" and held `Finished` dark forever.
    #
    # Enhancement still appears under Lip-sync when it happens: the pipeline
    # logs `=== stage 9i ===` and `_open` appends an unplanned row for it.
    # Reality beats the forecast, which is the rule the whole file follows.
    if config.get("restore"):
        keys.append("9i")

    # Under `stop_after_mt` the whole post-gate half is deferred: planned, so
    # the pills stay drawn, but never counted as done or skipped.
    if stop_after_mt:
        deferred = deferred | set(keys[gate:])
        # Every gated job is awaiting cast by construction -- `ui/runner.py`
        # forces `stop_after_mt=True` on every UI submission, so there is no
        # such thing as a gated job that will never be cast. If a falsy or
        # missing `tts`/`mix`/`lipsync` left 9d/9e/9g/9h out of `keys`, the
        # Dub-audio and Lip-sync groups come out with no substages at all --
        # `_group_phases` reads an empty group as `skipped`, and
        # `_finished_phase` counts `skipped` as complete, so the bar reads
        # Finished over a dub that has not been cast and a render that does
        # not exist. ("6" is deliberately excluded: see the comment above on
        # why ASD is never planned here.)
        for extra in ("9d", "9e", "9g", "9h"):
            if extra not in keys:
                keys.append(extra)
                deferred.add(extra)
    return _plan_rows(keys, deferred)


def _plan_rows(keys: List[str], deferred: Optional[set] = None
               ) -> List[Dict[str, Any]]:
    deferred = deferred or set()
    return [{"key": k, "title": None, "label": label_for(k),
             "state": "pending", "started_at": None, "wall_s": None,
             **({"deferred": True} if k in deferred else {})}
            for k in keys]


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

class ProgressWriter(logging.Handler):
    """Turns `=== stage N: … ===` log lines into `<job>/progress.json`.

    Install on the ROOT logger, because the markers come from
    `backend.pipeline` and the adapters log under their own names.
    """

    def __init__(self, job_root: Path, *, job_id: str,
                 planned: Optional[List[Dict[str, Any]]] = None,
                 kind: str = "run", heartbeat_s: float = 10.0):
        super().__init__(level=logging.INFO)
        self.job_root = Path(job_root)
        self.path = self.job_root / PROGRESS_FILE
        self._lock = threading.RLock()
        #: The keys THIS run owns. Everything else in the file belongs to an
        #: earlier run and is read-only here. The sweeps in `_open` and
        #: `finish` are scoped to it, or a cast run opening 9d would mark
        #: every inherited phase-one row `skipped`.
        self._owned = {s["key"] for s in (planned or [])}
        stages, started_at = self._merge(planned, job_id)
        self._doc: Dict[str, Any] = {
            "version": PROGRESS_VERSION,
            "job_id": job_id,
            "kind": kind,
            "pid": os.getpid(),
            "state": "running",
            # The JOB's start, carried across runs -- the job list sorts on
            # it. This run's own start is `run_started_at`.
            "started_at": started_at or _now(),
            "run_started_at": _now(),
            "updated_at": _now(),
            "heartbeat": _now(),
            "current": None,
            "log": "run.log",
            "error": None,
            "stages": stages,
        }
        self._stop = threading.Event()
        self._beat = threading.Thread(target=self._heartbeat, args=(heartbeat_s,),
                                      daemon=True, name="progress-heartbeat")
        self._flush()
        self._beat.start()

    def _merge(self, planned: Optional[List[Dict[str, Any]]],
               job_id: str) -> tuple:
        """Fold `planned` into the progress doc already on disk.

        A job folder is written by several processes over its life: phase
        one, the cast run, a re-render, the lip-sync job. Rows this run owns
        are reset (it is about to run them); rows it does not own are kept
        exactly as the last run left them.

        Replacing the file instead erased everything before it, leaving a
        single `9h` row -- so the ribbon drew four empty groups, called them
        `skipped`, and painted them done.

        Any trouble reading the old file -- missing, corrupt, a different
        schema version or a different job -- falls back to replacing it. A
        progress file must never be able to fail a run.
        """
        rows = [dict(r) for r in (planned or [])]
        try:
            doc = json.loads(self.path.read_text())
            prior = doc["stages"]
            if (doc.get("version") != PROGRESS_VERSION
                    or doc.get("job_id") != job_id):
                return rows, None
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return rows, None

        by_key = {r["key"]: r for r in rows if isinstance(r, dict) and "key" in r}
        out: List[Dict[str, Any]] = []
        for old in prior:
            if not isinstance(old, dict) or "key" not in old:
                continue
            fresh = by_key.pop(old["key"], None)
            if fresh is None:
                out.append(old)
                continue
            # Ours, and it has run before. Reset it: a stale `done` with the
            # previous attempt's wall time is wrong the moment we start.
            # `deferred` goes too -- we own it now.
            fresh.pop("deferred", None)
            fresh.update(state="pending", started_at=None, wall_s=None)
            out.append(fresh)
        for key, fresh in by_key.items():
            out.insert(_insert_at(out, key), fresh)
        return out, doc.get("started_at")

    # -- public ----------------------------------------------------------

    def begin(self, key: str, title: str = "") -> None:
        """Open a stage the pipeline does not log (fetch, cut, probe)."""
        with self._lock:
            self._open(key, title)
            self._flush()

    def finish(self, state: str = "done", error: Optional[str] = None) -> None:
        """Close the run out.

        On success, any stage still `pending` was conditionally skipped — the
        run got past it without logging it. Leaving those pending means a
        finished job reads "14 of 18" forever. On failure they stay pending,
        because they genuinely never got their turn and saying "skipped"
        would imply a decision nobody made.

        A `deferred` stage is the second case even on success: `plan_stages`
        marks 9h deferred on a draft run, where lip-sync is a separate act the
        operator has not performed yet. Calling that "skipped" would read as
        "this dub will never be lip-synced" — and `StageRibbon` counts skipped
        as done, so the bar would claim the whole pipeline had finished.
        """
        with self._lock:
            self._close_current("done" if state == "done" else state)
            if state == "done":
                for st in self._doc["stages"]:
                    # `self._owned`: a row an earlier run left pending is not
                    # this run's to judge. The lip-sync replay plans one key;
                    # without this it marked every inherited phase-one row
                    # `skipped`, which the ribbon drew as done.
                    if (st["state"] == "pending"
                            and st["key"] in self._owned
                            and not st.get("deferred")):
                        st["state"] = "skipped"
            self._doc["state"] = state
            self._doc["error"] = error
            self._doc["current"] = None
            self._flush()
        self._stop.set()

    def close(self) -> None:          # logging.Handler
        self._stop.set()
        super().close()

    # -- handler ---------------------------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:                               # pragma: no cover
            return
        if "===" not in msg:
            return                                      # cheap reject
        m = STAGE_MARKER_RE.match(msg.strip())
        if not m:
            return
        key, title = m.group("key"), m.group("title")
        with self._lock:
            # `=== stage 3: SKIPPED ===` opens and immediately closes.
            skipped = title.upper().startswith("SKIPPED")
            self._open(key, title)
            if skipped:
                self._close_current("skipped")
            self._flush()

    # -- internals -------------------------------------------------------

    def _open(self, key: str, title: str) -> None:
        self._close_current("done")
        stages = self._doc["stages"]
        idx = next((i for i, s in enumerate(stages)
                    if s["key"] == key and s["state"] == "pending"), None)
        if idx is not None:
            # Anything still pending before this one was conditionally
            # skipped -- but only among the keys THIS run owns, and never a
            # deferred row. An inherited row belongs to a closed run, a
            # deferred one to a submission that has not happened yet.
            for s in stages[:idx]:
                if (s["state"] == "pending"
                        and s["key"] in self._owned
                        and not s.get("deferred")):
                    s["state"] = "skipped"
        else:
            # No PENDING entry for this key. Two different situations, and
            # they used to be collapsed into "append":
            #
            #  * the key is already in the list in a terminal state — this is
            #    a RE-RUN re-entering a stage the previous attempt had
            #    already passed, skipped or failed. `--rerun` and the edit
            #    agent's replay both land here, as does any second attempt
            #    on the same job folder. Appending produced a SECOND row with
            #    the same key, and every reader takes the first match, so the
            #    UI showed the stale verdict: stage 9b2 read "skipped" on a
            #    run whose log says it executed for 11.9 s.
            #  * the key is absent entirely — the plan drifted from the
            #    pipeline. Appending is right there: reality beats forecast.
            idx = next((i for i, s in enumerate(stages)
                        if s["key"] == key), None)
            if idx is None:
                stages.append({"key": key, "title": title,
                               "label": label_for(key, title),
                               "state": "pending", "started_at": None,
                               "wall_s": None})
                idx = len(stages) - 1
            else:
                # Re-entering: clear the previous attempt's verdict so the
                # row reports THIS attempt rather than the last one.
                stages[idx]["wall_s"] = None
        st = stages[idx]
        st.update(state="running", title=title or st.get("title"),
                  label=label_for(key, title), started_at=_now(),
                  _t0=time.monotonic())
        self._doc["current"] = key

    def _close_current(self, state: str) -> None:
        for s in self._doc["stages"]:
            if s["state"] == "running":
                t0 = s.pop("_t0", None)
                if t0 is not None:
                    s["wall_s"] = round(time.monotonic() - t0, 2)
                s["state"] = state

    def _heartbeat(self, every: float) -> None:
        while not self._stop.wait(every):
            with self._lock:
                self._doc["heartbeat"] = _now()
                self._flush()

    def _flush(self) -> None:
        doc = dict(self._doc)
        doc["updated_at"] = _now()
        doc["stages"] = [{k: v for k, v in s.items() if not k.startswith("_")}
                         for s in doc["stages"]]
        done = sum(1 for s in doc["stages"] if s["state"] in ("done", "skipped"))
        doc["n_done"], doc["n_total"] = done, len(doc["stages"])
        write_atomic(self.path, doc)


def write_atomic(path: Path, doc: Dict[str, Any]) -> None:
    """tmp + os.replace, so a reader never sees a half-written file.

    Same guarantee `backend.jobs.Job._write` gives, for the same reason: the
    UI polls this file every 1.5 s while it is being rewritten.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# reading — the normaliser
# ---------------------------------------------------------------------------

def read_progress(job_root: Path) -> Optional[Dict[str, Any]]:
    """Normalised progress for a job, whichever runner produced it.

    Returns None only when the folder holds none of the three shapes.
    """
    job_root = Path(job_root)
    doc = (_from_progress_json(job_root)
           or _from_status_json(job_root)
           or _from_manifest(job_root)
           or _from_files(job_root))
    if doc is None:
        return None
    doc["milestones"] = milestones(job_root, doc)
    # A user-facing reading of the failure, so the UI never has to show a
    # traceback. The raw log stays at GET /api/jobs/{id}/log for debugging.
    try:
        from .errors import explain
        doc["failure"] = explain(doc.get("error"), state=doc.get("state"))
    except Exception:                                    # pragma: no cover
        doc["failure"] = None
    doc["phases"] = _group_phases(doc["stages"])
    doc["pct"] = _pct(doc)
    return doc


def _blank(job_root: Path) -> Dict[str, Any]:
    return {"version": PROGRESS_VERSION, "job_id": job_root.name,
            "kind": "run", "pid": None, "state": "unknown",
            "started_at": None, "updated_at": None, "heartbeat": None,
            "current": None, "error": None, "log": None,
            "stages": [], "n_done": 0, "n_total": 0, "source": None}


def _from_progress_json(job_root: Path) -> Optional[Dict[str, Any]]:
    p = job_root / PROGRESS_FILE
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        log.warning("progress.json unreadable in %s: %s", job_root.name, e)
        return None
    out = _blank(job_root)
    out.update(doc)
    out["source"] = "progress.json"
    out["stages"] = _backfill_erased(job_root, doc)
    out["n_done"] = sum(1 for st in out["stages"]
                        if st.get("state") in ("done", "skipped"))
    out["n_total"] = len(out["stages"])

    # A process that died without finalising still reads "running". Only the
    # pid can tell us otherwise, and only on this machine.
    if out.get("state") == "running" and not _pid_alive(out.get("pid")):
        out["state"] = "failed"
        out["error"] = out.get("error") or (
            "The run stopped without reporting a result — the process is gone. "
            "See run.log for what it was doing.")
    return out


def _backfill_erased(job_root: Path, doc: Dict[str, Any]
                     ) -> List[Dict[str, Any]]:
    """Repair a doc a pre-2026-10-28 rerun flattened.

    Until `ProgressWriter._merge` existed, a cast or re-render replaced this
    file with its own two or three keys, so the job's whole history went. The
    signature is a `rerun`/`cast` doc with not one Prepare-group key in it --
    no download, no diarization, nothing that must have happened for a
    re-render to be possible at all.

    Rather than rewrite those files, re-plan the run from the manifest and
    put the recovered rows in front of what survived. New jobs never match,
    so this costs one set-membership test per read.
    """
    stages = [st for st in (doc.get("stages") or [])
              if isinstance(st, dict) and "key" in st]
    if doc.get("kind") not in ("rerun", "cast"):
        return stages
    if any(_GROUP_OF.get(st["key"]) == "prepare" for st in stages):
        return stages
    try:
        recovered = _from_manifest(job_root)
    except Exception:                                     # noqa: BLE001
        return stages
    if not recovered:
        return stages
    seen = {st["key"] for st in stages}
    return [st for st in recovered["stages"]
            if st.get("key") not in seen] + stages


def _from_status_json(job_root: Path) -> Optional[Dict[str, Any]]:
    """Legacy clip-runner's STATUS.json shape: phases[], no plan."""
    p = job_root / "STATUS.json"
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    raw_state = str(doc.get("state") or "")
    stages = []
    for ph in doc.get("phases") or []:
        name = str(ph.get("name", "?"))
        stages.append({
            "key": name,
            "title": name,
            "label": CLIP_PHASE_LABELS.get(name, name.replace("_", " ").title()),
            "state": "done" if ph.get("done") else "running",
            "started_at": None,
            "wall_s": ph.get("wall_s"),
        })
    out = _blank(job_root)
    out.update({
        "source": "STATUS.json",
        "state": _clip_state(raw_state),
        "updated_at": doc.get("updated"),
        "stages": stages,
        # No planned list exists for this runner, so the denominator is what
        # has been seen so far. "4 of 5 so far" is truthful; a made-up total
        # would not be.
        "n_done": sum(1 for s in stages if s["state"] == "done"),
        "n_total": len(stages),
        "n_total_is_partial": True,
        "current": next((s["key"] for s in stages if s["state"] == "running"), None),
    })
    if out["state"] == "failed":
        out["error"] = raw_state[len("failed:"):].strip() or raw_state
    return out


def _clip_state(raw: str) -> str:
    low = raw.lower()
    if low.startswith("failed"):
        return "failed"
    if low in ("done", "ok", "complete", "completed"):
        return "done"
    if low.startswith("running") or low:
        return "running"
    return "unknown"


def _from_manifest(job_root: Path) -> Optional[Dict[str, Any]]:
    """Last resort: the two writes `backend.jobs.Job` makes."""
    p = job_root / "manifest.json"
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    status = str(doc.get("status") or "unknown")
    state = {"ok": "done", "running": "running",
             "failed": "failed"}.get(status, status)
    cfg = doc.get("config") or {}
    # A job sitting at the gate stamped `awaiting_cast` here, and the cast run
    # has not written its own progress doc yet. Plan what phase one ran, or
    # this fallback reports a finished job as stalled mid-dub. A CLI job never
    # carries the flag and still plans end to end.
    stages = plan_stages(cfg, stop_after_mt=bool(doc.get("awaiting_cast")))
    if state == "done":
        for s in stages:
            # `deferred` stays pending — see `ProgressWriter.finish`. The
            # manifest says the RUN finished, which on a draft job is true
            # and says nothing about lip-sync having happened.
            if not s.get("deferred"):
                s["state"] = "done"
    out = _blank(job_root)
    out.update({
        "source": "manifest.json",
        "state": state,
        "started_at": doc.get("started_at"),
        "updated_at": doc.get("finished_at") or doc.get("started_at"),
        "error": doc.get("error"),
        "stages": stages,
        "n_done": sum(1 for s in stages if s["state"] == "done"),
        "n_total": len(stages),
        # The manifest cannot say which stage is running — it is written twice.
        "coarse": True,
    })
    return out


def _from_files(job_root: Path) -> Optional[Dict[str, Any]]:
    """No status file of any kind — infer from what landed on disk.

    16 of the 51 folders here are flat script outputs: a `config.json` and a
    `final.<lang>.<model>.mp4` at the root, nothing else. They are real
    results and must appear in the UI; they just cannot report progress.
    """
    finals = sorted(job_root.glob("final.*.mp4")) + sorted(job_root.glob("*.lipsync.mp4"))
    if not finals and not (job_root / "config.json").is_file():
        return None

    # Build the stage list from EVIDENCE, not from assumption. A finished
    # render proves lip-sync ran; it says nothing about whether this job
    # transcribed anything, and several of these folders were driven by a
    # pre-made dub with no ASR at all. So each stage appears only when
    # something on disk vouches for it, and the bar is correspondingly
    # short rather than seven nodes of invented history.
    def has(sub: str, pattern: str) -> bool:
        d = job_root / sub
        return d.is_dir() and any(d.glob(pattern))

    evidence = [
        ("1",   bool(finals) or has("videos", "*.mp4")),
        ("4",   has("diarization", "*.rttm")),
        ("5",   has("faces", "*")),
        ("7",   has("asr", "*.json")),
        ("9",   has("mt", "*.json")),
        ("9e",  has("tts", "*")),
        ("9g",  has("mix", "*.wav") or has("audio", "*dub*.wav")),
        ("9h",  bool(finals)),
    ]
    stages = [{"key": k, "title": None, "label": label_for(k), "state": "done",
               "started_at": None, "wall_s": None}
              for k, proven in evidence if proven]

    out = _blank(job_root)
    out.update({
        "source": "files",
        "state": "done" if finals else "unknown",
        "updated_at": _mtime_iso(finals[0] if finals else job_root),
        "coarse": True,
        "stages": stages,
        "n_done": len(stages),
        "n_total": len(stages),
        "n_total_is_partial": True,
    })
    return out


def _mtime_iso(p: Path) -> Optional[str]:
    try:
        return _dt.datetime.fromtimestamp(
            p.stat().st_mtime, _dt.timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return None


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _pct(doc: Dict[str, Any]) -> int:
    if doc.get("state") == "done":
        return 100
    total = doc.get("n_total") or 0
    if not total:
        return 0
    return max(0, min(99, int(round(100 * (doc.get("n_done") or 0) / total))))


#: Where a key belongs in the stage list, so a row introduced by a later run
#: (the cast step's 9d into a doc that stops at 9) lands in pipeline order
#: rather than at the end.
_CANON_ORDER: Dict[str, int] = {
    key: i
    for i, key in enumerate(
        k for _gid, _label, keys in PHASE_GROUPS for k in keys)
}


def _insert_at(rows: List[Dict[str, Any]], key: str) -> int:
    """The index `key` should take in `rows`. Unknown keys go last."""
    rank = _CANON_ORDER.get(key)
    if rank is None:
        return len(rows)
    for i, r in enumerate(rows):
        other = _CANON_ORDER.get(r.get("key"))
        if other is not None and other > rank:
            return i
    return len(rows)


#: The last pill, and the only synthetic one. It is NOT in `PHASE_GROUPS`
#: because that table maps real `=== stage N ===` markers only.
FINISHED_ID = "finished"
FINISHED_LABEL = "Finished"


def _group_phases(stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold the stage list into the ribbon's pills.

    **Always the same pills, in the same order, on every run** — the seven
    groups plus `Finished`. Changed 2026-10-22 by request.

    It used to `continue` past a group with no planned substages, and
    `plan_stages` is config-driven, so a run that enabled few optional
    stages rendered a bar with one or two pills on it. The operator could
    not tell "this pipeline has no transcribe step" from "the bar is
    broken", and the bar's length changed between two runs of the same
    video — which made it useless as a thing to glance at.

    A group with nothing planned is now `skipped` rather than absent: the
    pipeline HAS that step, this run is not doing it, and those are
    different claims from "it does not exist".
    """
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for s in stages:
        if s["key"] in HIDDEN_KEYS:
            continue
        by_group.setdefault(_GROUP_OF.get(s["key"], "other"), []).append(s)

    # Every pill opens, including one this run planned nothing in -- so
    # `contains` carries the group's FULL membership, and the ribbon lists
    # the steps a run did not take as "not in this run" rather than showing
    # an empty drawer or refusing to open.
    #
    # Suppressed for the clip runner and the evidence-based fallback, whose
    # rows (`P1_CUT`, ...) are not pipeline keys: listing the pipeline's own
    # stages against those would describe a pipeline that is not running.
    canonical_doc = any(s["key"] in _CANON_ORDER for s in stages)

    out: List[Dict[str, Any]] = []
    for gid, label, _keys in PHASE_GROUPS:
        subs = by_group.pop(gid, [])
        planned = {s["key"] for s in subs}
        rest = ([{"key": k, "label": label_for(k)} for k in _keys
                 if k not in planned and k not in HIDDEN_KEYS]
                if canonical_doc or not stages else [])
        out.append({"id": gid, "label": label,
                    "contains": rest,
                    # No substages means this run never configured the step.
                    # `skipped`, not `pending`: pending reads as "still to
                    # come" and would leave a finished run looking unfinished.
                    "state": _rollup(subs) if subs else "skipped",
                    # True when the group exists only as work a later
                    # submission does. The ribbon draws a skipped group muted
                    # and a deferred one as a waiting node. Never both -- an
                    # empty group is skipped, not deferred.
                    "deferred": bool(subs) and not _live(subs),
                    "wall_s": _sum_opt(s.get("wall_s") for s in subs),
                    "substages": subs})
    for gid, subs in by_group.items():            # unknown keys, kept visible
        out.append({"id": gid, "label": gid.replace("_", " ").title(),
                    "contains": [],
                    "state": _rollup(subs),
                    "deferred": bool(subs) and not _live(subs),
                    "wall_s": _sum_opt(s.get("wall_s") for s in subs),
                    "substages": subs})

    out.append(_finished_phase(out))
    return out


def _finished_phase(phases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The `Finished` pill: done only when every other pill is.

    `skipped` counts as complete — a step this run was never going to take
    is not work outstanding. A `deferred` substage does NOT, and that is the
    load-bearing case: a draft run leaves 9h pending-and-deferred because
    lip-sync is a separate, expensive, explicit act. Treating it as complete
    would light `Finished` over a dub nobody has lip-synced, which is the
    screen asserting something false.
    """
    states = [p["state"] for p in phases]
    if "failed" in states:
        return {"id": FINISHED_ID, "label": FINISHED_LABEL,
                "state": "failed", "deferred": False,
                "contains": [], "wall_s": None, "substages": []}
    deferred = any(s.get("deferred") and s.get("state") == "pending"
                   for p in phases for s in p["substages"])
    complete = all(s in ("done", "skipped") for s in states) and not deferred
    return {"id": FINISHED_ID, "label": FINISHED_LABEL,
            "state": "done" if complete else "pending",
            # Always present so the TS type can treat them as non-optional.
            "deferred": False,
            # A moment, not a step: it contains nothing, so its pill is the
            # one that does not open.
            "contains": [],
            # None, not the sum of the others: the ribbon adds `wall_s`
            # across phases, so a total here would double the runtime. This
            # pill marks a moment, not work.
            "wall_s": None,
            "substages": []}


def _live(subs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The substages this run is actually going to touch.

    A deferred-and-pending row is work a later submission does (the cast
    step, the lip-sync job), so it must not drag the group into `running`
    nor be mistaken for a stage that was decided against.
    """
    return [s for s in subs
            if not (s.get("deferred") and s.get("state") == "pending")]


def _rollup(subs: List[Dict[str, Any]]) -> str:
    """Fold a group's substages into one pill state.

    Takes the rows, not bare states, because `deferred` lives on the row and
    changes the answer: a group of nothing but deferred rows reads `pending`
    ("not run yet"), never `skipped` -- which the ribbon paints as done.
    """
    live = _live(subs)
    if not live:
        return "pending"
    states = [s["state"] for s in live]
    if "running" in states:
        return "running"
    if "failed" in states:
        return "failed"
    if all(s in ("done", "skipped") for s in states):
        return "done" if any(s == "done" for s in states) else "skipped"
    if any(s in ("done", "skipped") for s in states):
        return "running"
    return "pending"


def _sum_opt(vals) -> Optional[float]:
    nums = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(nums), 2) if nums else None


# ---------------------------------------------------------------------------
# milestones — what the UI is allowed to show yet
# ---------------------------------------------------------------------------

def milestones(job_root: Path, doc: Optional[Dict[str, Any]] = None
               ) -> Dict[str, bool]:
    """Capability flags, computed from stage states OR'd with what is on disk.

    The disk half is not redundant. A stage is only *marked* done when the
    NEXT marker fires, so for a config where diarization is the last stage
    "done" would land at process exit. Checking for the RTTM closes that.
    """
    job_root = Path(job_root)
    states = {s["key"]: s["state"] for s in ((doc or {}).get("stages") or [])}

    def done(*keys: str) -> bool:
        return any(states.get(k) in ("done", "skipped") for k in keys)

    def has(sub: str, pattern: str) -> bool:
        d = job_root / sub
        return d.is_dir() and any(d.glob(pattern))

    # The finished render lands in three different places depending on which
    # runner made the job: `lipsync/*.lipsync*.mp4` (pipeline),
    # `final.*.mp4` at the job root (clip runner and the flat scripts), or
    # nothing at all. Check all of them rather than trusting a stage key that
    # only one of the three ever writes.
    # Root-level renders come in three namings across the runners:
    # `final.<lang>.<model>.mp4`, `<stem>.<model>.lipsync.mp4`, and
    # `<stem>.<model>.codeformer.mp4`. Only the first was checked at first,
    # which left a finished wav2lip job reading "not lip-synced yet".
    final_at_root = any(
        any(job_root.glob(pat))
        for pat in ("final.*.mp4", "*.lipsync.mp4", "*.codeformer.mp4")
    )
    mix_done = (done("9g", "P4_MIX") or has("mix", "*.wav")
                or has("audio", "*dub*.wav"))

    # Phase one stopped after MT for the operator to set gender + voice. The
    # runner stamps `awaiting_cast` on the manifest; gate it on the mix not
    # existing yet so a stale flag clears itself once the cast run completes
    # (or a crashed cast leaves the gate up rather than a half-done job).
    awaiting_cast = False
    try:
        import json as _json
        _mani = job_root / "manifest.json"
        if _mani.is_file():
            _mani_doc = _json.loads(_mani.read_text())
            awaiting_cast = bool(
                _mani_doc.get("awaiting_cast")
                or _mani_doc.get("result", {}).get("awaiting_cast"))
    except (OSError, ValueError):
        awaiting_cast = False

    mix_on_disk = has("mix", "*.wav") or has("audio", "*dub*.wav")

    return {
        "video_ready":      done("1", "cut", "P1_CUT") or has("videos", "*.mp4"),
        "diarization_done": done("4") or has("diarization", "*.rttm"),
        "awaiting_cast":    awaiting_cast and not mix_on_disk,
        "speakers_ready":   done("9d") or has("voices", "*"),
        "mix_done":         mix_done,
        "lipsync_ready":    (done("9h", "P6_LIPSYNC", "P7_SPLICE") or final_at_root)
                            and (final_at_root or has("lipsync", "*.lipsync*.mp4")),
    }
