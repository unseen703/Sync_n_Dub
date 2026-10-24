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
`tests/test_ui_progress.py` greps `pipeline.py` for every marker literal and
asserts this module still matches all of them, so that failure is loud.

Reading
-------
`read_progress(job_root)` is the only function `ui/server.py` calls. It
normalises **all three shapes** found under `data/jobs/`, so nothing above it
ever branches on which runner produced a job:

  1. `progress.json`  — written by `ProgressWriter` (new jobs)
  2. `STATUS.json`    — `scripts/clip_asd_musetalk_run.py` (34 of 51 folders)
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
    "fetch":  "Fetching the video",
    "cut":    "Trimming to the chosen part",
    "probe":  "Looking at the faces",
    "1":      "Fetching the video",
    "2":      "Extracting the audio",
    "3":      "Separating music from speech",
    "4":      "Working out who speaks when",
    "4b":     "Tidying up the segments",
    "4c":     "Telling the speakers apart",
    "5":      "Finding faces",
    "6":      "Working out who is on screen",
    "7":      "Transcribing",
    "8":      "Timing the words",
    "9":      "Translating",
    "9b":     "Keeping English terms in English",
    "9b1":    "Working out the tone",
    "9b2":    "Listening for emotion",
    "9c":     "Fitting the text to the timing",
    "9c2":    "Shortening lines that run long",
    "9d":     "Casting voices",
    "9e":     "Speaking the translation",
    "9f":     "Converting the voice",
    "9f1":    "Timing the new speech",
    "9f1b":   "Reading mouth shapes",
    "9f2":    "Planning the timing",
    "9g":     "Mixing the new soundtrack",
    "9h":     "Lip-syncing",
    "9i":     "Enhancing resolution",
    "10a":    "Checking the transcript",
    "10b":    "Checking the translation",
    "1-9a":   "Reusing the earlier run",
}

#: The pipeline as a person describes it: seven stages, in order, each one a
#: thing you can picture. The 24 internal markers fold into these.
#:
#: This is deliberately NOT a tree. The sub-stages exist (`9f1b: viseme
#: extraction`) and are still carried per stage for the log, but the UI shows
#: one bar across these seven and nothing finer — a reader wants to know how
#: far along the dub is, not which forced aligner is running.
PHASE_GROUPS: List[tuple] = [
    ("download",   "Download video",     ("fetch", "cut", "probe", "1", "2",
                                          "3", "1-9a")),
    ("speakers",   "Speaker annotation", ("4", "4b", "4c", "5", "6")),
    ("transcribe", "Transcribe",         ("7", "8")),
    ("translate",  "Translate",          ("9", "9b", "9c", "9c2")),
    # 9b1 (content register) and 9b2 (source emotion) both decide HOW the
    # line will be delivered, so they sit with speech generation rather than
    # with translation, even though their markers are numbered next to 9b.
    ("speech",     "Speech generation",  ("9b1", "9b2", "9d", "9e", "9f",
                                          "9f1", "9f1b", "9f2")),
    ("mix",        "Mix",                ("9g",)),
    ("lipsync",    "Lip sync",           ("9h",)),
    # Face restoration over the finished render. Its own pill rather than a
    # substage of Lip sync, because it is not part of lip-syncing: it runs on
    # any render from any model, costs 2.6x the MuseTalk pass it cleans
    # (1156 s vs 447 s, measured 2026-10-22), and is always a separate act.
    # Folding it into Lip sync would make a lit "Lip sync" mean two different
    # amounts of work.
    ("enhance",    "Enhance resolution", ("9i",)),
]

#: Stages 10a/10b score the ASR and MT against ground-truth subtitles. They
#: are an optional QA pass that most runs skip, and they produce no part of
#: the dub — so they are kept out of the bar rather than appearing as an
#: eighth stage nobody asked about. They still show in the log.
HIDDEN_KEYS: tuple = ("10a", "10b")

_GROUP_OF: Dict[str, str] = {
    key: gid for gid, _label, keys in PHASE_GROUPS for key in keys
}

#: The clip runner's phases map onto the same ribbon, so a clip job and a
#: pipeline job read the same way on screen. Without this they all land in a
#: single meaningless "Other" pill.
_GROUP_OF.update({
    "P1_CUT": "download", "P2_FACES": "speakers", "P3_ASD": "speakers",
    "P4_MIX": "mix", "P5_LIPSYNC_BBOX_PREVIEW": "lipsync",
    "P6_LIPSYNC": "lipsync", "P7_SPLICE": "lipsync", "P8_VERIFY": "lipsync",
    # Shot segmentation — the clip runner cuts on shot boundaries so it can
    # lip-sync only single-speaker shots. It is part of working out who is
    # on screen, not a stage of its own. Present in 11 jobs on disk, and
    # without this line it rendered as an eighth node labelled "Other".
    "P5_SEGMENT": "speakers",
})

#: Phase names used by `scripts/clip_asd_musetalk_run.py`'s STATUS.json.
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

def plan_stages(config: Dict[str, Any], *, pre: Optional[List[str]] = None
                ) -> List[Dict[str, Any]]:
    """The stages this config will run, so the UI can say "5 of 18".

    Mirrors the conditionals in `_run_pipeline`, which means it *can* drift.
    It is reconciled against reality as markers arrive: an unplanned key is
    appended, and a planned key that is skipped over is marked `skipped`. So
    a drift degrades the up-front count, never the live state.
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
    if config.get("asd"):
        keys.append("6")
    if config.get("asr"):
        keys.append("7")
    if config.get("align"):
        keys.append("8")
    if config.get("mt"):
        keys.append("9")
    if config.get("translit"):
        keys.append("9b")
    if config.get("tts_style") == "auto":
        keys.append("9b1")
    if config.get("emotion"):
        keys.append("9b2")
    if config.get("duration_adjust"):
        keys.append("9c")
    # 9c2 only runs inside the `duration_adjust` branch -- it acts on the
    # per-segment estimates 9c writes -- so it cannot be planned without it.
    if config.get("duration_adjust") and config.get("condense_overlong_text"):
        keys.append("9c2")
    if config.get("voice_engine") or config.get("tts") or config.get("vc"):
        keys.append("9d")
    if config.get("tts"):
        keys.append("9e")
    if config.get("vc"):
        keys.append("9f")
    if config.get("tts_force_align"):
        keys.append("9f1")
    if config.get("visemes"):
        keys.append("9f1b")
    if config.get("timing_producer"):
        keys.append("9f2")
    if config.get("mix") or config.get("lipsync"):
        keys.append("9g")
    # 9h is planned even when this run will not reach it, which is the one
    # place this function deliberately does not mirror `_run_pipeline`.
    #
    # A `draft` submission sets `lipsync: None` and the dub stops at the mix;
    # lip-sync is then a separate, explicit act (`POST .../lipsync`). Keying
    # the node off the config meant such a job drew a SIX-node bar reading
    # "Finished · all 6 stages" — a dub that is waiting for its most expensive
    # stage, rendered identically to one that will never have it. The node is
    # drawn and left pending instead, so the bar says what is still to come.
    #
    # **Only when the run actually produces something to lip-sync TO.** A
    # config with no TTS and no mix is not a dub awaiting its last stage; it
    # is a run that was never going anywhere near 9h, and showing the node as
    # "still to come" there would promise a step nobody is waiting for. That
    # case keeps the old behaviour — no node, and `_group_phases` reads the
    # group as `skipped`.
    #
    # `deferred` is what carries the first case through `finish()` and
    # `_from_manifest`, both of which otherwise resolve a leftover `pending`
    # to `skipped` on a successful run. Skipped is a decision; deferred is a
    # stage that has not had its turn, and the ribbon counts the two
    # differently.
    deferred: set = set()
    if config.get("lipsync"):
        keys.append("9h")
    elif config.get("mix") or config.get("tts"):
        keys.append("9h")
        deferred = {"9h"}
    # 9i is 9h's case one step further out. Enhancement is NEVER part of a
    # submission -- there is nothing to restore until a render exists, so it
    # is always a separate act on a finished job (`POST .../enhance`), the
    # same shape as `POST .../lipsync` on a draft. It is therefore planned
    # whenever there is a 9h node to enhance, and deferred unless this run
    # was explicitly configured to do it.
    #
    # Deferred and not `skipped`: `StageRibbon.isDone` counts skipped AS
    # DONE, so marking it skipped would light the bar over a render nobody
    # has enhanced — exactly the bug `deferred` was added to 9h to fix.
    if "9h" in keys:
        keys.append("9i")
        if not config.get("restore"):
            deferred = deferred | {"9i"}
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
        self._doc: Dict[str, Any] = {
            "version": PROGRESS_VERSION,
            "job_id": job_id,
            "kind": kind,
            "pid": os.getpid(),
            "state": "running",
            "started_at": _now(),
            "updated_at": _now(),
            "heartbeat": _now(),
            "current": None,
            "log": "run.log",
            "error": None,
            "stages": list(planned or []),
        }
        self._stop = threading.Event()
        self._beat = threading.Thread(target=self._heartbeat, args=(heartbeat_s,),
                                      daemon=True, name="progress-heartbeat")
        self._flush()
        self._beat.start()

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
                    if st["state"] == "pending" and not st.get("deferred"):
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
            # Anything still pending before this one was conditionally skipped.
            for s in stages[:idx]:
                if s["state"] == "pending":
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
    # A process that died without finalising still reads "running". Only the
    # pid can tell us otherwise, and only on this machine.
    if out.get("state") == "running" and not _pid_alive(out.get("pid")):
        out["state"] = "failed"
        out["error"] = out.get("error") or (
            "The run stopped without reporting a result — the process is gone. "
            "See run.log for what it was doing.")
    return out


def _from_status_json(job_root: Path) -> Optional[Dict[str, Any]]:
    """`scripts/clip_asd_musetalk_run.py`'s shape: phases[], no plan."""
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
    stages = plan_stages(cfg)
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


#: The last pill, and the only synthetic one. It is NOT in `PHASE_GROUPS`
#: because that table maps real `=== stage N ===` markers and
#: `tests/test_ui_progress.py` greps `pipeline.py` for every key in it; a
#: group with no marker would fail that grep for doing its job.
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

    out = []
    for gid, label, _keys in PHASE_GROUPS:
        subs = by_group.pop(gid, [])
        out.append({"id": gid, "label": label,
                    # No substages means this run never configured the step.
                    # `skipped`, not `pending`: pending reads as "still to
                    # come" and would leave a finished run looking unfinished.
                    "state": _rollup([s["state"] for s in subs]) if subs
                             else "skipped",
                    "wall_s": _sum_opt(s.get("wall_s") for s in subs),
                    "substages": subs})
    for gid, subs in by_group.items():            # unknown keys, kept visible
        out.append({"id": gid, "label": gid.replace("_", " ").title(),
                    "state": _rollup([s["state"] for s in subs]),
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
                "state": "failed", "wall_s": None, "substages": []}
    deferred = any(s.get("deferred") and s.get("state") == "pending"
                   for p in phases for s in p["substages"])
    complete = all(s in ("done", "skipped") for s in states) and not deferred
    return {"id": FINISHED_ID, "label": FINISHED_LABEL,
            "state": "done" if complete else "pending",
            # None, NOT the sum of the others. The ribbon adds `wall_s`
            # across phases to show elapsed time, so a total here would be
            # counted twice and report double the real runtime. This pill
            # marks a moment, not work.
            "wall_s": None,
            "substages": []}


def _rollup(states: List[str]) -> str:
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
    return {
        "video_ready":      done("1", "cut", "P1_CUT") or has("videos", "*.mp4"),
        "diarization_done": done("4") or has("diarization", "*.rttm"),
        "speakers_ready":   done("9d") or has("voices", "*"),
        "mix_done":         done("9g", "P4_MIX") or has("mix", "*.wav")
                            or has("audio", "*dub*.wav"),
        "lipsync_ready":    (done("9h", "P6_LIPSYNC", "P7_SPLICE") or final_at_root)
                            and (final_at_root or has("lipsync", "*.lipsync*.mp4")),
    }
