"""Voice and tone edits over a finished dub — stage, cost, apply, revert.

Mount with `app.include_router(ui.routes_edits.router)` — and mount it
**above** `app.mount("/", StaticFiles(...))` in `ui/server.py`, not after the
import. Starlette matches routes in registration order and that mount matches
every path, so a router included after it is shadowed completely: every route
here answers 404 (GET) or 405 (POST) from the static handler, which looks
exactly like a routing typo. Verified both ways on the real app.

What this endpoint is
---------------------
The operator picks a different voice for a speaker, or a different register,
and is told what re-rendering that costs *before* anything happens. Nothing
here parses a sentence, calls a model, or reaches the network:
`backend.edits` already does the whole job deterministically, and this module
is the HTTP skin over it.

    POST   /api/jobs/{id}/edits        stage a set, return the cost preview
    GET    /api/jobs/{id}/edits        the current pending set
    DELETE /api/jobs/{id}/edits        discard it
    POST   /api/jobs/{id}/edits/apply  commit + launch the re-render detached
    POST   /api/jobs/{id}/revert       undo recorded per-segment edits

**The guarantee: the quoted cost is the applied cost.** `set_voice` is pure
table lookup, and `set_tone` is forced to `rewrite_text=False`, so it writes
only the engine style hint. Both are deterministic, so the `dry_run=True`
pass that produced the preview and the `commit()` that follows resolve to the
same segments, the same fields and the same dirty stages. That holds *only*
while this endpoint refuses the two ops that would break it — `retranslate`,
and a `set_tone` that rewrites wording — because both call an LLM, and an LLM
rewrite can change text length, which changes the timing fit, which changes
what the re-render actually does. They are therefore refused **explicitly**
(422, naming the op) rather than by not being reachable: refusal by omission
is one request-shape away from being a silent LLM call.

Why the staged set is a file
----------------------------
`<job>/edits/pending.json`. The apply runs in a different process — the
re-render is minutes of GPU and is launched detached — so an in-memory
staging area would be invisible to it, and a server restart between staging
and applying would lose the set without saying so.

Handing off to the re-render
----------------------------
`commit()` writes `<job>/edits/edit_NNN.json`, carrying both the ops and the
`speaker_voice_overrides` the replay needs (a speaker-scoped `set_voice`
writes **no** per-segment field — the override lives only in the
`EditResult`). That file, not `pending.json`, is the contract with
`ui/runner.py`: the runner reconstructs the plan from the newest one with

    view = open_job(job_id, data_dir=DATA_DIR)
    ns   = recover_args(view)
    ctx  = build_context(view, rewriter=None, ns=ns)
    ops  = [EditOp.from_dict(o) for o in record["ops"]]
    res  = apply_edits(view.merged_json, ops, ctx=ctx, dry_run=True)
    res.dirty_stages = rerun_stages([a.dirty_from for a in res.applied],
                                    available=available_stages(view, res, ns=ns))
    rerun(EditPlan(view, "", ops, "ui", res, ctx, ns), data_dir=DATA_DIR)

and `pending.json` is cleared the moment the commit succeeds, so a staged set
is never applied twice.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:                      # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT))

from backend import tracing as _tracing
from backend import edits as E                          # noqa: E402
from backend import edit_agent as A                     # noqa: E402

from .auth import basic_auth                            # noqa: E402
from .cost import estimate as cost_estimate             # noqa: E402
from .speakers import TONE_ENGINES, supports_tone        # noqa: E402
from . import transcript as _transcript                  # noqa: E402
from . import bbox_video as _bbox                        # noqa: E402
from . import presets                                    # noqa: E402

log = logging.getLogger("ui.edits")

#: Overridable so a test (and a relocated data folder) does not need the repo
#: checkout's own `data/`. `ui.server` keeps its own copy of this constant;
#: both resolve to the same place.
DATA_DIR = Path(os.environ.get("DUBBING_DATA_DIR") or (REPO_ROOT / "data"))

PENDING_VERSION = 1

#: The only two ops this endpoint will build. Everything else in `OPS` either
#: needs an LLM or changes text, and both break the cost guarantee above.
#: What the STRUCTURED endpoint (`POST .../edits`) will stage. Both are
#: deterministic, which is what lets it promise that the quoted cost is the
#: applied cost.
ALLOWED_OPS = ("set_voice", "set_tone")

#: What the AGENT endpoint (`POST .../agent`) will stage. The whole op
#: catalogue, because the point of a sentence is that you do not have to
#: know which control it maps to.
#:
#: The cost guarantee is weaker here and the response says so per op rather
#: than in a docstring nobody reads. Three ops rewrite wording through an
#: LLM, and an LLM is not idempotent: the text resolved for the preview is
#: not guaranteed to be the text resolved at apply, and different wording is
#: a different length, which is a different timing fit. `AGENT_NONDETERMINISTIC`
#: is that set, and every staged op drawn from it carries a warning.
AGENT_OPS = tuple(sorted(E.OPS))
AGENT_NONDETERMINISTIC = ("retranslate", "set_text", "code_mix")

router = APIRouter()


# ---------------------------------------------------------------------------
# job plumbing
# ---------------------------------------------------------------------------

from ._io import read_json as _read_json


def _open(job_id: str) -> A.JobView:
    """`open_job`, with its errors turned into HTTP ones."""
    root = DATA_DIR / "jobs" / job_id.strip("/")
    if not root.is_dir():
        raise HTTPException(404, f"Job not found: {job_id}")
    try:
        return A.open_job(job_id.strip("/"), data_dir=DATA_DIR)
    except E.EditError as e:
        # The common case is a run that never reached MT: there is no merged
        # transcript, so there is nothing to edit. Say which, not "500".
        raise HTTPException(409, str(e)) from None


def _config(view: A.JobView) -> Dict[str, Any]:
    """The job's knobs, over both folder shapes.

    `manifest.json:config` is the pipeline's record; the clip runner writes a
    flat `config.json` with different key names. Merge them so `supports_tone`
    and the cost estimate see one dict either way.
    """
    cfg = dict(view.config or {})
    flat = _read_json(view.root / "config.json") or {}
    if isinstance(flat, dict):
        for src, dst in (("tts", "tts"), ("mt_tts", "tts"),
                         ("language", "target_language"),
                         ("target_language", "target_language"),
                         ("lipsync_model", "lipsync"), ("lipsync", "lipsync"),
                         ("restore", "restore")):
            if not cfg.get(dst) and flat.get(src):
                cfg[dst] = flat[src]
    return cfg


def _clip_seconds(merged: Dict[str, Any]) -> Optional[float]:
    """Media length as the edit sees it: the end of the last cue.

    Deliberately not an ffprobe call — `_ffprobe_bin()` answering `None` is a
    silent `0.0` in every caller (CLAUDE.md §1), and a cost dialog quoting
    `0 s` for a lip-sync render is the exact failure that note exists for.
    """
    ends = [float(s.get("end") or 0.0) for s in (merged.get("segments") or [])]
    return round(max(ends), 2) if ends else None


def _pending_path(root: Path) -> Path:
    return root / "edits" / "pending.json"


def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# request -> ops
# ---------------------------------------------------------------------------

def _build_ops(raw_edits: List[Dict[str, Any]], *, config: Dict[str, Any],
               ) -> List[E.EditOp]:
    """Validate the request body and turn it into typed ops.

    Order matters for the refusals: a stale client sending a tone edit to an
    engine that ignores the style hint gets a 409 and stages nothing, rather
    than staging an op that would re-render the dub into an identical result.
    """
    if not isinstance(raw_edits, list) or not raw_edits:
        raise HTTPException(
            422, "edits must be a non-empty list (send it under `edits`, or `pending` for the older UI bundle)")

    tone_ok = supports_tone(config)
    engine = (config.get("tts") or config.get("voice_engine") or "") or None
    ops: List[E.EditOp] = []

    for i, item in enumerate(raw_edits):
        if not isinstance(item, dict):
            raise HTTPException(422, f"edits[{i}] is not an object")
        where = f"edits[{i}]"

        op_name = str(item.get("op") or "").strip().lower()
        if op_name and op_name not in ALLOWED_OPS:
            raise HTTPException(
                422, f"{where}: this endpoint only stages "
                     f"{' and '.join(ALLOWED_OPS)}; {op_name!r} is refused "
                     f"because it needs an LLM, which would make the quoted "
                     f"cost different from the applied cost")
        for forbidden in ("text", "instruction", "target_language"):
            if item.get(forbidden) is not None:
                raise HTTPException(
                    422, f"{where}: {forbidden!r} belongs to a text edit "
                         f"(retranslate / set_text), which this endpoint "
                         f"refuses — it needs an LLM")
        if item.get("rewrite_text"):
            raise HTTPException(
                422, f"{where}: rewrite_text must stay false. A tone edit "
                     f"here sets the engine style hint only; rewriting the "
                     f"wording needs an LLM and breaks the cost guarantee")

        speaker = str(item.get("speaker_id") or item.get("speaker") or "").strip()
        if not speaker:
            raise HTTPException(422, f"{where}: speaker_id is required")

        voice = item.get("voice")
        tone = item.get("tone")
        if voice in ("", None) and tone in ("", None):
            raise HTTPException(
                422, f"{where}: nothing to change — give a voice, a tone, "
                     f"or both")

        if voice not in ("", None):
            ops.append(E.EditOp(
                op="set_voice", scope=E.Scope(speaker=speaker),
                params={"voice": str(voice), "engine": engine},
                note=f"UI: voice for {speaker}"))
        if tone not in ("", None):
            if not tone_ok:
                raise HTTPException(
                    409, f"{where}: the {engine or 'configured'} TTS engine "
                         f"ignores a style hint, so a tone edit would "
                         f"re-render the dub into an identical result. "
                         f"Only {' and '.join(TONE_ENGINES)} forward it.")
            ops.append(E.EditOp(
                op="set_tone", scope=E.Scope(speaker=speaker),
                # rewrite_text=False is the whole reason this needs no LLM.
                params={"tone": str(tone), "rewrite_text": False},
                note=f"UI: tone for {speaker}"))

    return ops


def _preview(view: A.JobView, ops: List[E.EditOp]) -> Dict[str, Any]:
    """Resolve the ops without writing, and price the result."""
    ns = A.recover_args(view)
    ctx = A.build_context(view, rewriter=None, ns=ns)
    try:
        res = E.apply_edits(view.merged_json, ops, nlq="", ctx=ctx,
                            dry_run=True)
    except E.EditError as e:
        raise HTTPException(422, str(e)) from None
    available = A.available_stages(view, res, ns=ns)
    res.dirty_stages = E.rerun_stages(
        [a.dirty_from for a in res.applied], available=available)

    merged = _read_json(view.merged_json) or {}
    cfg = _config(view)
    cost = cost_estimate(
        res.dirty_stages, config=cfg,
        clip_s=_clip_seconds(merged),
        n_segments=len(merged.get("segments") or []),
        available=available, warnings=res.warnings)
    return {"result": res, "ns": ns, "ctx": ctx, "cost": cost,
            "available": available, "config": cfg}


def _payload(job_id: str, raw_edits: List[Dict[str, Any]],
             ops: List[E.EditOp], prev: Dict[str, Any]) -> Dict[str, Any]:
    res: E.EditResult = prev["result"]
    return {
        "version": PENDING_VERSION,
        "job_id": job_id,
        "updated": _now(),
        "edits": raw_edits,
        "ops": [o.to_dict() for o in ops],
        "dirty_stages": list(res.dirty_stages),
        "speaker_voice_overrides": res.speaker_voice_overrides,
        "segments": res.touched,
        "warnings": list(res.warnings),
        "cost": prev["cost"],
    }


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@router.post("/api/jobs/{job_id}/edits", dependencies=[Depends(basic_auth)])
def stage_edits(job_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Stage a voice/tone set and return what re-rendering it would cost.

    Replaces whatever was pending: the UI holds the whole set and re-posts it,
    so a merge here would resurrect a row the user had just cleared.
    """
    view = _open(job_id)
    cfg = _config(view)
    # `pending` is the key the shipped Vite bundle sends; `edits` is what
    # this endpoint has always documented and what the tests post. They
    # drifted apart unnoticed because the tests exercise the backend
    # contract and nothing checked the frontend agreed — the symptom was a
    # bare 422 ("edits must be a non-empty list") on every voice change,
    # followed by a 409 on apply because nothing had been staged.
    # Both are accepted so an un-rebuilt bundle keeps working.
    rows = body.get("edits")
    if rows is None:
        rows = body.get("pending")
    ops = _build_ops(rows, config=cfg)
    prev = _preview(view, ops)
    payload = _payload(view.job_id, rows, ops, prev)
    _write_atomic(_pending_path(view.root), payload)
    # The server is long-lived, so nothing would otherwise push this batch
    # until the SDK's own interval — and `atexit` never fires on a process
    # that keeps running. Without it a chat message appears in Langfuse
    # late, or not at all if the server is restarted first.
    _tracing.flush()
    log.info("edits: staged %d op(s) for %s -> %s", len(ops), view.job_id,
             payload["dirty_stages"])
    return payload


@router.get("/api/jobs/{job_id}/review-video")
def get_review_video(job_id: str) -> Dict[str, Any]:
    """State of the ASD review render, plus its URL once there is one."""
    view = _open(job_id)
    st = _bbox.status(view.root)
    if st.get("ready"):
        vid = _bbox.paths(view.root)["video"]
        try:
            rel = vid.resolve().relative_to(DATA_DIR.resolve())
            st["url"] = "/files/" + rel.as_posix()
        except ValueError:
            st["url"] = None
    return {"job_id": view.job_id, **st}


@router.post("/api/jobs/{job_id}/review-video",
             dependencies=[Depends(basic_auth)])
def make_review_video(job_id: str) -> Dict[str, Any]:
    """Render the boxes, in a thread.

    A full decode + composite over a 1080p clip is minutes, so this returns
    immediately and the UI polls the GET above — the same shape the stage
    progress already uses. Rendering inline would die to a browser timeout
    while the work carried on.
    """
    view = _open(job_id)
    if not any(view.root.glob("faces/*.npz")):
        raise HTTPException(
            409, "This job has no face tracks, so there are no boxes to "
                 "draw. It ran without a face detector.")
    try:
        from .speakers import read_labels
        labels = read_labels(view.root)
    except Exception:                                   # noqa: BLE001
        labels = {}
    return {"job_id": view.job_id, **_bbox.start(view.root, labels=labels)}


@router.get("/api/jobs/{job_id}/transcript")
def get_transcript(job_id: str, offset: int = 0,
                   limit: int = _transcript.DEFAULT_LIMIT) -> Dict[str, Any]:
    """The cue list the operator reviews the dub against.

    Paged, because a 40-minute lecture is a few thousand cues and shipping
    them all makes this the slowest page in the UI for no benefit.
    """
    view = _open(job_id)
    try:
        from .speakers import read_labels
        labels = read_labels(view.root)
    except Exception:                                   # noqa: BLE001
        labels = {}
    return _transcript.cues(view.merged_json, data_dir=DATA_DIR,
                            offset=max(0, offset),
                            limit=max(1, min(limit, 2000)),
                            labels=labels)


@router.get("/api/jobs/{job_id}/edits")
def get_edits(job_id: str) -> Dict[str, Any]:
    """The pending set, or an empty one. Never 404s on "nothing staged" —
    the UI polls this to decide whether to show the pending bar."""
    view = _open(job_id)
    doc = _read_json(_pending_path(view.root))
    if not isinstance(doc, dict):
        return {"version": PENDING_VERSION, "job_id": view.job_id,
                "edits": [], "ops": [], "dirty_stages": [], "cost": None,
                "warnings": [], "updated": None}
    doc.setdefault("job_id", view.job_id)
    return doc


@router.delete("/api/jobs/{job_id}/edits", dependencies=[Depends(basic_auth)])
def discard_edits(job_id: str) -> Dict[str, Any]:
    view = _open(job_id)
    p = _pending_path(view.root)
    existed = p.is_file()
    if existed:
        p.unlink()
    return {"job_id": view.job_id, "discarded": existed}


# TWO different things are needed from the LLM here, and one helper used to
# serve both, which is the whole bug:
#
#   * `A.parse_request(client=...)` wants the CLIENT — it calls
#     `client.available()` and `client.chat_json(messages)`.
#   * `ctx.rewriter` wants a `(system, user) -> str` CALLABLE —
#     `backend/edits.py` invokes it as `ctx.rewriter(system, user)`.
#
# Passing the client as the rewriter gave every op that rewrites wording
# (`retranslate`, `set_tone`, `code_mix`) a 500 —
# `TypeError: 'OpenRouterClient' object is not callable` — the moment the
# agent reached one, while the SAME request through
# `python -m backend.edit_agent` worked, because `plan()` builds both.
# Passing the callable as the client fails the other way, at
# `client.available()`. They are not interchangeable, so they are two
# functions now, and `_rewriter_for` is the CLI's own so the wrapper has
# exactly one definition.


def _llm_client():
    """The raw LLM client for `A.parse_request`, or None.

    Never raises: `OPENROUTER_API_KEY` being empty is a supported state
    (CLAUDE.md §1), and the agent still parses with its regex fallback and
    still stages every op that does not rewrite wording. Only the three
    text ops refuse, and they refuse by name.
    """
    try:
        from backend.llm import OpenRouterClient
        c = OpenRouterClient()
        return c if c.available() else None
    except Exception as e:                       # pragma: no cover
        log.warning("agent: no LLM client (%s)", e)
        return None


def _rewriter(client=None):
    """A `(system, user) -> str` callable for the text ops, or None."""
    return A._rewriter_for(client if client is not None else _llm_client())


def _agent_preview(view: A.JobView, ops: List[E.EditOp], *, rewriter
                   ) -> Dict[str, Any]:
    """`_preview`, but with a rewriter so the text ops actually resolve.

    `_preview` passes `rewriter=None` on purpose — the structured endpoint
    refuses every op that would need one. The agent does not, so it has to
    build the context itself rather than quietly previewing a text edit
    that cannot run.
    """
    ns = A.recover_args(view)
    ctx = A.build_context(view, rewriter=rewriter, ns=ns)
    try:
        res = E.apply_edits(view.merged_json, ops, nlq="", ctx=ctx,
                            dry_run=True)
    except E.EditError as e:
        raise HTTPException(422, str(e)) from None
    available = A.available_stages(view, res, ns=ns)
    res.dirty_stages = E.rerun_stages(
        [a.dirty_from for a in res.applied], available=available)
    merged = _read_json(view.merged_json) or {}
    cfg = _config(view)
    cost = cost_estimate(
        res.dirty_stages, config=cfg, clip_s=_clip_seconds(merged),
        n_segments=len(merged.get("segments") or []),
        available=available, warnings=res.warnings)
    return {"result": res, "ns": ns, "ctx": ctx, "cost": cost,
            "available": available, "config": cfg}


@router.get("/api/jobs/{job_id}/agent/params")
def agent_params(job_id: str) -> Dict[str, Any]:
    """Everything the agent can change, with its CURRENT and STAGED value.

    One endpoint rather than the panel assembling this from `/speakers`,
    `/media` and the manifest, because the point of the panel is to show
    the effect of a sentence — and that only reads as an effect if the
    before and after come from the same snapshot. Three round trips would
    let them disagree by a poll.

    `staged` is None where the pending set does not touch that parameter.
    A field whose `staged` differs from `current` is what the UI highlights.
    """
    view = _open(job_id)
    cfg = _config(view)
    merged = _read_json(view.merged_json) or {}
    segs = merged.get("segments") or []
    pending = _read_json(_pending_path(view.root))
    pending = pending if isinstance(pending, dict) else {}
    ops = []
    try:
        ops = [E.EditOp.from_dict(o) for o in (pending.get("ops") or [])]
    except E.EditError:
        ops = []

    # Staged values, read off the ops rather than re-resolved: this endpoint
    # must not call an LLM, because the panel polls it.
    staged_voice: Dict[str, Any] = dict(
        pending.get("speaker_voice_overrides") or {})
    staged_tone: Dict[str, Any] = {}
    staged_gender: Dict[str, Any] = {}
    staged_window = None
    staged_lipsync = None
    for o in ops:
        spk = o.scope.speaker
        if o.op == "set_tone" and spk:
            staged_tone[spk] = o.params.get("tone")
        elif o.op == "set_gender" and spk:
            # Read off the op, like the tone above, rather than resolved.
            # `_h_set_gender` also picks a VOICE for the new gender, and that
            # half already arrives through `speaker_voice_overrides` — so the
            # cast row can show both halves of "make SPEAKER_00 female"
            # (gender AND the voice it implies) without this endpoint, which
            # the panel polls, loading a catalog or a model.
            staged_gender[spk] = o.params.get("gender")
        elif o.op == "lipsync_window":
            staged_window = {"start_s": o.scope.start_s, "end_s": o.scope.end_s}
            staged_lipsync = o.params.get("model") or staged_lipsync

    from .speakers import build_table, tone_vocabulary
    table = build_table(view.root, cfg, data_dir=DATA_DIR,
                        merged_json=view.merged_json)
    # Gender is `{current, staged}` like the other two, not the bare string it
    # used to be. The cast table reads this endpoint now, and a gender edit
    # typed into the chat box was the one staged change with nowhere on screen
    # to appear: "make SPEAKER_00 female" looked like it had done nothing
    # until it was applied and the dub came back in a different voice.
    speakers = [{
        "id": r["id"],
        "label": r.get("label"),
        "gender": {"current": r.get("gender"),
                   "staged": staged_gender.get(r["id"])},
        "speech_s": r.get("speech_s"),
        "voice": {"current": r.get("voice"),
                  "staged": (staged_voice.get(r["id"]) or {}).get("voice")},
        "tone": {"current": r.get("tone"), "staged": staged_tone.get(r["id"])},
    } for r in table.get("speakers") or []]

    clip_s = _clip_seconds(merged)
    return {
        "job_id": view.job_id,
        # Clip size: the edit's own view of the media, not ffprobe's —
        # see `_clip_seconds` for why that distinction is load-bearing.
        "clip": {"current": {"start_s": 0.0, "end_s": clip_s,
                             "duration_s": clip_s,
                             "n_segments": len(segs)},
                 "staged": None},
        "language": {"current": cfg.get("target_language"), "staged": None,
                     "editable": False,
                     "note": ("The target language is fixed once a dub has "
                              "been translated — changing it is a new run, "
                              "not an edit.")},
        "lipsync": {"current": cfg.get("lipsync"), "staged": staged_lipsync,
                    "window": {"current": None, "staged": staged_window}},
        "tts_engine": cfg.get("tts") or cfg.get("voice_engine"),
        "speakers": speakers,
        "voice_options": table.get("voice_options") or [],
        "tone_options": table.get("tone_options") or [],
        "tone_vocabulary": tone_vocabulary(),
        "dirty_stages": list(pending.get("dirty_stages") or []),
        "cost": pending.get("cost"),
        "turns": pending.get("turns") or [],
        "warnings": list(pending.get("warnings") or []),
        # What the agent can do, in the user's words. Served rather than
        # hard-coded in the frontend for the reason `tone_vocabulary()`
        # exists: `ui/speakers.py` first shipped an INVENTED tone list, and
        # every tone change then failed at the API with nothing on screen
        # to explain it. A list the UI authors itself is a list that can
        # describe an op the backend does not have.
        "actions": E.action_catalogue(),
        "prompts": [{"text": t, "does": d} for t, d in E.sample_prompts()],
    }


@router.get("/api/jobs/{job_id}/agent/script")
def agent_script(job_id: str, limit: int = 2000) -> Dict[str, Any]:
    """The segmented script, with any staged text change beside each cue.

    Separate from `/agent/params` because it is the long payload and the
    panel above it is the one that has to stay cheap to poll.
    """
    view = _open(job_id)
    merged = _read_json(view.merged_json) or {}
    pending = _read_json(_pending_path(view.root))
    touched = set((pending or {}).get("segments") or [])

    # The staged WORDING, for cues whose pending change is a plain
    # `set_text`. The panel used to show only a "staged" badge beside the
    # ORIGINAL line, on the reasoning that resolving a staged change needs
    # the LLM — true for `retranslate` or `set_tone`, and not true for text
    # the user typed themselves. The effect was that editing a line, then
    # editing it again, looked like it did nothing: the row redrew with the
    # original text and the same badge both times.
    #
    # Last op wins, matching how they apply.
    staged_text: Dict[str, str] = {}
    # The staged SPEED, resolved the same way and for the same reason. A
    # `set_speed` op names its segments outright, so the factor is knowable
    # here without running anything.
    staged_speed: Dict[str, float] = {}
    for raw in ((pending or {}).get("ops") or []):
        if not isinstance(raw, dict):
            continue
        scope = raw.get("scope") or {}
        params = raw.get("params") or {}
        if raw.get("op") == "set_text":
            segs = scope.get("segments") or []
            text = params.get("text")
            if len(segs) == 1 and isinstance(text, str):
                staged_text[str(segs[0])] = text
        elif raw.get("op") == "set_speed":
            try:
                factor = float(params.get("factor") or params.get("speed") or 0)
            except (TypeError, ValueError):
                continue
            if factor > 0:
                for s in (scope.get("segments") or []):
                    staged_speed[str(s)] = factor

    rows = []
    for seg in (merged.get("segments") or [])[:max(1, min(limit, 5000))]:
        sid = str(seg.get("id") or "")
        rows.append({
            "id": sid,
            "speaker": seg.get("speaker"),
            "start": float(seg.get("start") or 0.0),
            "end": float(seg.get("end") or 0.0),
            "src_text": seg.get("text") or "",
            "text": seg.get("final_text") or seg.get("mt_text") or "",
            "tts_voice": seg.get("tts_voice"),
            "tts_style": seg.get("tts_style"),
            # Whether the pending set would rewrite this cue.
            "staged": sid in touched,
            # The staged wording when it is known without running anything
            # — a `set_text` the user typed. None for a change whose result
            # only exists after the LLM runs (`retranslate`, `set_tone`), so
            # the panel can tell "this will change, text unknown" from
            # "this will read exactly thus".
            "staged_text": staged_text.get(sid),
            # How fast this cue is spoken, as a factor of its own slot.
            #
            # Derived, not stored: `set_speed` SHORTENS `target_duration_s`
            # and the timing plan then fits the clip into the smaller slot,
            # so the factor is `slot / target`. Reading `tts_rate_override`
            # instead would show only the speed the ENGINE was asked for,
            # which on a UI run is nothing at all — `supports_rate` gates it
            # and the default `--tts-speed-control off` never sends it. The
            # ratio is what the listener actually hears either way.
            "speed": _cue_speed(seg),
            "staged_speed": staged_speed.get(sid),
        })
    return {"job_id": view.job_id, "segments": rows,
            "total": len(merged.get("segments") or [])}


def _cue_speed(seg: Dict[str, Any]) -> float:
    """`slot / target_duration_s`, or 1.0 when the cue is untouched.

    1.0 for an ordinary cue: stage 9c writes `target_duration_s` equal to the
    slot, so the ratio is already 1 and only an edit moves it.
    """
    try:
        slot = float(seg.get("end") or 0.0) - float(seg.get("start") or 0.0)
        target = seg.get("target_duration_s")
        if target is None or slot <= 0:
            return 1.0
        target = float(target)
        if target <= 0:
            return 1.0
        return round(slot / target, 3)
    except (TypeError, ValueError):
        return 1.0


@router.post("/api/jobs/{job_id}/agent", dependencies=[Depends(basic_auth)])
def agent_message(job_id: str, body: Dict[str, Any] = Body(...)
                  ) -> Dict[str, Any]:
    """Turn one sentence into staged edits. Stages; does NOT re-render.

    The agent itself is `backend.edit_agent`, which has existed and been
    reachable from the CLI all along. This is the HTTP skin, and the one
    design decision in it is that a message ACCUMULATES onto the pending set
    rather than replacing it. A chat is a conversation: "make him female"
    followed by "and speed up the intro" is two edits, not a correction of
    the first. `replace: true` starts over, which is what a Clear button
    sends.

    Nothing here re-renders. The staged set and its cost come back exactly
    as the structured endpoint returns them, and `POST .../edits/apply` is
    still the only thing that spends a GPU minute.
    """
    message = str(body.get("message") or "").strip()
    if not message:
        raise HTTPException(422, "message is required")
    if len(message) > 2000:
        raise HTTPException(422, "message is too long (max 2000 characters)")

    view = _open(job_id)
    data = _read_json(view.merged_json) or {}
    if not data.get("segments"):
        raise HTTPException(
            409, "This job has no transcript yet, so there is nothing to "
                 "edit. The agent works on a finished dub.")

    client = _llm_client()
    rewriter = _rewriter(client)
    # The CLI gets its trace from `A.make_plan`, which this endpoint does
    # NOT call — it assembles the plan itself so it can merge the turn into
    # `pending.json`. So the grouping span has to be opened here too, or a
    # chat request shows up in Langfuse as loose generations with nothing
    # tying the rewrites to the message that caused them.
    with _tracing.trace("ui.agent_message",
                        input=message,
                        metadata={"job_id": job_id,
                                  "segments": len(data.get("segments") or [])}):
        return _agent_message_inner(job_id, body, view, data, message,
                                    client, rewriter)


def _agent_message_inner(job_id, body, view, data, message, client, rewriter):
    """The body of `agent_message`, split out so the tracing span above is a
    flat `with` rather than a re-indent of the whole handler."""
    try:
        new_ops, source = A.parse_request(message, data=data, client=client)
    except E.EditError as e:
        # A sentence the parser could not place is a 422 carrying the
        # agent's own guidance, which names the supported phrasings. It is
        # the most useful thing we can put in a chat bubble.
        raise HTTPException(422, str(e)) from None

    for o in new_ops:
        if o.op not in AGENT_OPS:
            raise HTTPException(422, f"unknown op {o.op!r}")

    turns: List[Dict[str, Any]] = []
    ops: List[E.EditOp] = []
    if not body.get("replace"):
        prior = _read_json(_pending_path(view.root))
        if isinstance(prior, dict):
            turns = [t for t in (prior.get("turns") or [])
                     if isinstance(t, dict)]
            try:
                ops = [E.EditOp.from_dict(o) for o in (prior.get("ops") or [])]
            except E.EditError:
                ops = []            # a stale set is dropped, not fatal
    ops = ops + new_ops

    prev = _agent_preview(view, ops, rewriter=rewriter)
    payload = _payload(view.job_id, [], ops, prev)

    nd = sorted({o.op for o in ops if o.op in AGENT_NONDETERMINISTIC})
    if nd:
        payload["warnings"] = list(payload["warnings"]) + [
            f"{', '.join(nd)} rewrites wording through an LLM, which is not "
            f"idempotent: applying may produce slightly different text than "
            f"this preview, and different wording is a different length, so "
            f"the timing fit can differ too."
        ]
    if nd and rewriter is None:
        payload["warnings"] = list(payload["warnings"]) + [
            "No OPENROUTER_API_KEY is configured, so the text ops above will "
            "refuse at apply time. Everything else will still run."
        ]

    turns = turns + [{"at": _now(), "message": message, "source": source,
                      "ops": [o.to_dict() for o in new_ops]}]
    payload["turns"] = turns
    payload["nlq"] = message
    payload["source"] = source
    _write_atomic(_pending_path(view.root), payload)
    log.info("agent: %s staged %d op(s) from %r (source=%s) -> %s",
             view.job_id, len(new_ops), message[:80], source,
             payload["dirty_stages"])
    return payload


@router.post("/api/jobs/{job_id}/edits/apply",
             dependencies=[Depends(basic_auth)])
def apply_pending(job_id: str) -> Dict[str, Any]:
    """Write the staged edits and launch the re-render, detached.

    Never runs the render inline: it is minutes of GPU, and a request that
    long dies to a browser timeout while the work continues, which is the
    worst of both.
    """
    view = _open(job_id)
    pending = _read_json(_pending_path(view.root))
    if not isinstance(pending, dict) or not pending.get("ops"):
        raise HTTPException(409, "Nothing staged for this job")

    try:
        ops = [E.EditOp.from_dict(o) for o in pending["ops"]]
    except E.EditError as e:
        raise HTTPException(422, f"stale pending set: {e}") from None
    # Which rules apply depends on WHICH endpoint staged this set, and the
    # `turns` key is the honest marker: only the agent writes it.
    #
    # A structured set keeps its guarantee intact — two deterministic ops,
    # `rewrite_text` re-forced here rather than trusted from a file that
    # could have been hand-edited between staging and applying.
    #
    # An agent set was never under that guarantee (the stage response said
    # so, per op), so narrowing it to `ALLOWED_OPS` here would mean the UI
    # could stage a speed change it could never apply.
    from_agent = bool(pending.get("turns"))
    allowed = AGENT_OPS if from_agent else ALLOWED_OPS
    for o in ops:
        if o.op not in allowed:
            raise HTTPException(422, f"refusing to apply {o.op!r}")
        if o.op == "set_tone" and not from_agent:
            o.params["rewrite_text"] = False

    prev = (_agent_preview(view, ops, rewriter=_rewriter()) if from_agent
            else _preview(view, ops))
    plan = A.EditPlan(view=view, nlq=pending.get("nlq") or "",
                      ops=ops, source=("agent" if from_agent else "ui"),
                      result=prev["result"],
                      ctx=prev["ctx"], ns=prev["ns"])
    try:
        res = A.commit(plan)
    except E.EditError as e:
        raise HTTPException(422, str(e)) from None

    # Only after a successful commit, so a failed apply leaves the set intact
    # and retryable.
    p = _pending_path(view.root)
    if p.is_file():
        p.unlink()

    records = sorted((view.root / "edits").glob("edit_*.json"))
    launch = _launch_rerun(view.job_id, view.root)
    return {
        "job_id": view.job_id,
        "status": "applied",
        "dirty_stages": list(res.dirty_stages),
        "cost": prev["cost"],
        "warnings": list(res.warnings),
        "edit_record": (str(records[-1]) if records else None),
        "rerun": launch,
    }


def _lipsync_blockers(view, cfg: Dict[str, Any], *,
                      model: Optional[str] = None) -> List[str]:
    """Why a lip-sync render would be refused, or [] if it would not.

    ONE definition, read by both `GET .../lipsync/readiness` (which drives a
    disabled button and its tooltip) and `POST .../lipsync` (which enforces
    it). Writing the same conditions twice is how a UI comes to offer an
    action the server then refuses.

    `model` is known only at the POST; the GET passes None and the compute
    cap then states the limit without quoting a runtime for a model nobody
    has chosen yet.
    """
    out: List[str] = []
    pending = _read_json(_pending_path(view.root))
    n = len(pending.get("ops") or []) if isinstance(pending, dict) else 0
    if n:
        out.append(f"{n} edit(s) are staged. Apply or discard them first, or "
                   f"the render would use the audio from before those edits.")
    if not cfg.get("mix") and not cfg.get("tts"):
        out.append("This job has no dubbed audio yet, so there is nothing to "
                   "lip-sync to.")

    # The compute cap, moved here from submit time on 2026-10-20. It used to
    # live in `routes_create.check_lipsync_cap`, which took the model the
    # submission asked for — and submissions no longer ask for one, so the
    # guard went quiet exactly when lip-sync became a separate, explicit act.
    # This is where the GPU hours are actually committed.
    from .routes_create import max_lipsync_s

    clip_s = _clip_seconds(_read_json(view.merged_json) or {})
    cap = max_lipsync_s()
    if clip_s and clip_s > cap:
        tail = ""
        if model:
            est = presets.estimate_runtime(clip_s, lipsync_model=model)
            tail = f" With {model} that render would take {est['text']}."
        out.append(
            f"This job is {clip_s / 60:.0f} minutes long and lip-sync is "
            f"capped at {cap // 60} minutes here. This is a compute limit, "
            f"not a policy one: lip-sync runs at 3-21x the clip length on "
            f"this machine's two 16 GiB cards.{tail} The audio dub is "
            f"unaffected — the cap is only on the lip-synced portion.")
    return out


def _asd_face_size(view) -> Dict[str, Any]:
    """Average size of the face ASD attributed to each speaker.

    NOT "the biggest face in the frame", which is what the submit-time probe
    measured and what `pick_lipsync_model` was fed. On a multi-face shot the
    biggest face is frequently not the one talking, and lip-sync quality
    depends on the size of the mouth actually being repainted.

    Reads stage 6's `assignments.json` through the same reader the gender
    vote uses, so there is one definition of "this speaker's face" in the
    repo rather than two. Returns `available: False` with a reason instead of
    raising — a job that ran before stage 6 was switched on simply has no
    number, and the screen says so.
    """
    import numpy as np

    vids = sorted((view.root / "videos").glob("*.mp4"))
    if not vids:
        return {"available": False, "reason": "no video in this job folder"}
    try:
        from backend.pipeline import _speaker_face_tracks
        tracks = _speaker_face_tracks(view.root, vids[0])
    except Exception as e:                                    # noqa: BLE001
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}
    if not tracks:
        return {"available": False,
                "reason": "no face track was attributed to a speaker "
                          "(stage 5 + 6 may not have run on this job)"}

    import cv2
    cap = cv2.VideoCapture(str(vids[0]))
    frame_h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frame_h <= 0:
        return {"available": False, "reason": "could not read the frame height"}

    per_speaker, all_fracs = [], []
    for spk, (frames, bboxes) in sorted(tracks.items()):
        bb = np.asarray(bboxes, dtype=float)
        if bb.size == 0:
            continue
        # Height, not width: a lip-sync crop is driven by the face box's
        # vertical extent and width varies with head turn.
        fracs = (bb[:, 3] - bb[:, 1]) / frame_h
        per_speaker.append({
            "speaker": spk,
            "frames": int(len(fracs)),
            "avg_frac": round(float(fracs.mean()), 4),
            "avg_px": int(round(float(fracs.mean()) * frame_h)),
        })
        all_fracs.extend(fracs.tolist())

    if not all_fracs:
        return {"available": False, "reason": "the attributed tracks are empty"}

    avg = float(sum(all_fracs) / len(all_fracs))
    from ui import facepick
    _suggested, why = facepick.pick_lipsync_model(avg)
    return {
        "available": True,
        "source": "asd",
        "frame_height": int(frame_h),
        "avg_frac": round(avg, 4),
        "avg_px": int(round(avg * frame_h)),
        "per_speaker": per_speaker,
        # A SUGGESTION, shown next to the number. Nothing acts on it.
        "suggests": _suggested,
        "why": why,
    }


@router.get("/api/jobs/{job_id}/lipsync/readiness")
def lipsync_readiness(job_id: str) -> Dict[str, Any]:
    """Everything the "ready to lip-sync?" screen needs, in one read.

    The review flow is: check the cast (`GET .../speakers`), the cues
    (`GET .../transcript`), and the measured face size (here), correct
    anything in plain language (`POST .../agent`), apply (`POST
    .../edits/apply`), then render (`POST .../lipsync`).

    `can_start` is the single answer to "will the render be accepted", and
    `blockers` says why not — so the button's disabled state and its tooltip
    come from the server rather than being re-derived in the browser and
    drifting from what the POST actually enforces.
    """
    view = _open(job_id)
    cfg = _config(view)
    from models.lipsync import available_models

    pending = _read_json(_pending_path(view.root))
    n_pending = len(pending.get("ops") or []) if isinstance(pending, dict) else 0
    blockers = _lipsync_blockers(view, cfg)

    already = bool(list((view.root / "lipsync").glob("*.mp4")))

    # One estimate PER MODEL, because the operator is now the one choosing
    # and the models differ by roughly 20x on this host (wav2lip 3-5x clip
    # length, wav2lip_hd 6-24x, musetalk 8-21x, latentsync ~83x). A model
    # list without the hours next to it moves the decision without informing
    # it.
    from .routes_create import max_lipsync_s
    from . import lipsync_models as ls_models
    clip_s = _clip_seconds(_read_json(view.merged_json) or {})
    estimates = {}
    if clip_s:
        for m in available_models():
            try:
                estimates[m] = presets.estimate_runtime(
                    clip_s, lipsync_model=m)["text"]
            except Exception:                                 # noqa: BLE001
                estimates[m] = None

    return {
        "job_id": view.job_id,
        "can_start": not blockers,
        "blockers": blockers,
        "pending_edits": n_pending,
        "already_rendered": already,
        # No default and no pre-selection: the operator chooses, and
        # `POST .../lipsync` 422s without one.
        "model_required": True,
        # `models` stays a list of ids — it is what the POST takes, and an
        # older bundle still reads it. `model_rows` carries the same ids with
        # the label, the speed band and the caveat that belongs in front of
        # the click: that wav2lip_hd is a second per-frame pass, and that
        # LatentSync's default 512 UNet does not fit these cards. The panel
        # used to render the raw registry slug with nothing beside it.
        "models": list(available_models()),
        "model_rows": ls_models.options(),
        "clip_s": clip_s,
        "max_lipsync_s": max_lipsync_s(),
        "estimates": estimates,
        "face": _asd_face_size(view),
    }


@router.delete("/api/jobs/{job_id}/agent/turns/{index}",
               dependencies=[Depends(basic_auth)])
def drop_staged_turn(job_id: str, index: int) -> Dict[str, Any]:
    """Remove ONE staged change, keeping the rest.

    Before this the only retraction was `DELETE .../edits`, which throws the
    whole set away — so one misparsed sentence among five cost the other
    four, and the natural move was to apply the bad one and fix it after.

    No LLM runs. Every turn carries the ops it produced, so the remaining
    set is rebuilt from what is already on disk: the voice/tone `edits`
    (deterministic, rebuilt through `_build_ops`) plus the ops of the turns
    that survive. The plan is then re-previewed, because dropping an op
    changes which stages replay and what the render costs — returning the
    old cost next to the new set is how a UI comes to quote a price for work
    it is not going to do.
    """
    view = _open(job_id)
    pending = _read_json(_pending_path(view.root))
    turns = [t for t in ((pending or {}).get("turns") or []) if isinstance(t, dict)]
    if not turns:
        raise HTTPException(409, "Nothing staged for this job")
    if not 0 <= index < len(turns):
        raise HTTPException(
            422, f"No staged change at position {index}; there "
                 f"{'is' if len(turns) == 1 else 'are'} {len(turns)}.")

    kept = turns[:index] + turns[index + 1:]
    raw_edits = [e for e in ((pending or {}).get("edits") or [])
                 if isinstance(e, dict)]

    # Nothing left at all: clear the file rather than leave an empty set
    # behind, so `can_start` on the lip-sync step stops reporting a blocker.
    if not kept and not raw_edits:
        _pending_path(view.root).unlink(missing_ok=True)
        return {"job_id": view.job_id, "removed": index, "turns": [],
                "ops": [], "dirty_stages": [], "edits": [], "cleared": True}

    ops: List[E.EditOp] = []
    if raw_edits:
        ops.extend(_build_ops(raw_edits, config=_config(view)))
    for t in kept:
        for o in (t.get("ops") or []):
            try:
                ops.append(E.EditOp.from_dict(o))
            except E.EditError:
                continue            # a stale op is dropped, not fatal

    if not ops:
        _pending_path(view.root).unlink(missing_ok=True)
        return {"job_id": view.job_id, "removed": index, "turns": [],
                "ops": [], "dirty_stages": [], "edits": [], "cleared": True}

    try:
        prev = _agent_preview(view, ops, rewriter=_rewriter())
    except HTTPException:
        raise
    payload = _payload(view.job_id, raw_edits, ops, prev)
    payload["turns"] = kept
    _write_atomic(_pending_path(view.root), payload)
    log.info("agent: dropped staged change %d for %s; %d left",
             index, view.job_id, len(kept))
    return payload


@router.post("/api/jobs/{job_id}/lipsync", dependencies=[Depends(basic_auth)])
def run_lipsync_now(job_id: str,
                    body: Optional[Dict[str, Any]] = Body(None)) -> Dict[str, Any]:
    """Finish a job that was dubbed without lip-sync — the staged flow's
    last step, after the cast and the transcript have been reviewed.

    A `draft` run sets `lipsync: None`, so the stage was never configured and
    `available_stages` would rightly refuse to replay it. The route back in
    is the same one the edit agent uses: an op that INTRODUCES the stage adds
    it back (`edit_agent.available_stages`). So this stages a whole-video
    `lipsync_window` with an explicit model rather than inventing a second
    launch path — one replay mechanism, one set of guarantees.

    The model must be explicit. `_h_lipsync_window` falls back to
    `ctx.lipsync_model`, which on a draft job is None, and a None model
    reaches the render as "no lip-sync" — the stage would appear to run and
    produce nothing.
    """
    view = _open(job_id)
    cfg = _config(view)
    from models.lipsync import available_models

    # The operator names the model. REQUIRED, not defaulted.
    #
    # This used to fall back to musetalk, and `"auto"` used to resolve from a
    # face-size probe at submit time — so the most expensive stage in the
    # pipeline could be chosen by a number nobody saw, on a job where the
    # operator had expressed no opinion. musetalk is 8-21x clip length here
    # against wav2lip's 3-5x, so guessing it costs real hours.
    #
    # `GET /api/jobs/{id}/lipsync/readiness` is where the face measurement
    # and the models live, so the refusal points there rather than just
    # listing names.
    model = str((body or {}).get("model") or "").strip()
    if model.lower() in ("auto", "none", "null"):
        model = ""
    if not model:
        raise HTTPException(
            422, "A lip-sync model is required to proceed. Choose one of "
                 f"{', '.join(available_models())} — "
                 f"GET /api/jobs/{job_id}/lipsync/readiness reports the "
                 f"measured face size and what it suggests.")
    if model not in available_models():
        raise HTTPException(
            422, f"Unknown lip-sync model {model!r}. "
                 f"Available: {', '.join(available_models())}")

    # Checked AFTER the model is known, and from the same function the
    # readiness endpoint reports — so the button's disabled state cannot
    # drift from what this enforces, and the compute cap can quote the
    # runtime for the model actually chosen rather than a generic one.
    blockers = _lipsync_blockers(view, cfg, model=model)
    if blockers:
        raise HTTPException(409, " ".join(blockers))

    # Moved here from the submit path, which no longer renders anything on a
    # GPU: this is the moment the VRAM is about to be spent.
    try:
        from ui.runner import _preflight_gpu
        _preflight_gpu()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from None
    except ImportError:                                       # pragma: no cover
        pass

    ops = [E.EditOp(op="lipsync_window", scope=E.Scope(all=True),
                    params={"model": model},
                    note=f"lip-sync the whole dub with {model}")]
    try:
        prev = _preview(view, ops)
    except E.EditError as e:
        raise HTTPException(422, str(e)) from None
    if "lipsync" not in (prev["result"].dirty_stages or []):
        # Defensive: if the stage still is not in the replay set, the render
        # would be a no-op reported as success.
        raise HTTPException(
            409, "lip-sync could not be added to this job's replay set; "
                 "the run may predate the stage being configurable.")

    plan = A.EditPlan(view=view, nlq=f"lip-sync the whole dub with {model}",
                      ops=ops, source="ui", result=prev["result"],
                      ctx=prev["ctx"], ns=prev["ns"])
    try:
        res = A.commit(plan)
    except E.EditError as e:
        raise HTTPException(422, str(e)) from None

    launch = _launch_rerun(view.job_id, view.root)
    log.info("lipsync: %s queued a whole-video render with %s",
             view.job_id, model)
    return {"job_id": view.job_id, "status": "queued", "model": model,
            "dirty_stages": list(res.dirty_stages), "cost": prev["cost"],
            "warnings": list(res.warnings), "rerun": launch}


@router.post("/api/jobs/{job_id}/revert", dependencies=[Depends(basic_auth)])
def revert_edits(job_id: str,
                 body: Optional[Dict[str, Any]] = Body(None)) -> Dict[str, Any]:
    """Undo recorded per-segment edits (all of them, or only `segments`).

    Honest about its reach: `backend.edits.revert` restores from each
    segment's `edit_backup`, so it undoes a tone hint and a per-segment voice
    override. A **speaker-scoped** `set_voice` writes nothing into the merged
    JSON — the override lives in the `EditResult` and reaches the replay
    through `edits/edit_NNN.json` — so reverting cannot take it back. The
    response says so rather than reporting a clean undo that did not happen.
    """
    view = _open(job_id)
    segments = (body or {}).get("segments") or None
    if segments is not None and not isinstance(segments, list):
        raise HTTPException(422, "segments must be a list of segment ids")
    restored = E.revert(view.merged_json, segments=segments)

    warnings: List[str] = []
    records = sorted((view.root / "edits").glob("edit_*.json"))
    spk_overrides = {}
    for rec in records:
        doc = _read_json(rec) or {}
        spk_overrides.update(((doc.get("result") or {})
                              .get("speaker_voice_overrides") or {}))
    if spk_overrides:
        warnings.append(
            "speaker-level voice assignments ("
            + ", ".join(sorted(spk_overrides))
            + ") are not stored per segment and were not reverted; re-cast "
              "them with a new voice edit to change them back")
    return {"job_id": view.job_id, "restored": restored,
            "n_restored": len(restored), "warnings": warnings}


# ---------------------------------------------------------------------------
# launching the re-render
# ---------------------------------------------------------------------------

def _launch_rerun(job_id: str, root: Path) -> Dict[str, Any]:
    """`scripts/ui_run_job.sh <job_id> --rerun`, detached.

    Goes through the shell launcher rather than calling `python -m ui.runner`
    directly, for one reason: **LD_PRELOAD**. A voice change dirties the
    chain through to `lipsync`, so a re-render runs stage 9h, and the
    adapters there touch cv2/insightface — whose import order decides which
    `libstdc++` the process gets (CLAUDE.md section 1). The launcher exports
    the conda one; a bare `Popen` of the module does not, and the resulting
    failure looks intermittent and package-related rather than like a missing
    environment variable.

    `start_new_session=True` puts it in its own process group, so it survives
    the server being restarted — the same property the repo's tmux convention
    buys for a CLI run. The launcher appends to the job folder's own
    `run.log`, which is already servable.

    A failure to launch is reported here rather than raised, because the
    edits are already committed by this point and failing the request would
    imply they were not.
    """
    logf = root / "run.log"
    launcher = REPO_ROOT / "scripts" / "ui_run_job.sh"
    if launcher.is_file():
        cmd = [str(launcher), job_id, "--rerun"]
    else:
        # Fall back rather than refuse, but say so: without the launcher the
        # preload is missing and a lip-sync re-render may fail oddly.
        log.warning("%s is missing; launching ui.runner directly, WITHOUT "
                    "LD_PRELOAD — a lip-sync re-render may fail on the "
                    "cv2/insightface import-order trap", launcher)
        cmd = [sys.executable, "-m", "ui.runner", job_id, "--rerun"]
    try:
        logf.parent.mkdir(parents=True, exist_ok=True)
        fh = open(logf, "a", buffering=1)
    except OSError as e:                                  # pragma: no cover
        return {"launched": False, "cmd": cmd, "error": str(e)}
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError as e:
        fh.close()
        log.warning("could not launch the re-render: %s", e)
        return {"launched": False, "cmd": cmd, "error": str(e)}
    finally:
        try:
            fh.close()
        except Exception:                                 # pragma: no cover
            pass
    return {"launched": True, "cmd": cmd, "pid": proc.pid,
            "log": str(logf)}


# ---------------------------------------------------------------------------
# editing the script directly
# ---------------------------------------------------------------------------

#: What `POST .../script` stages. `set_text` only, and that is not a
#: loosening of the rule above -- it is the other side of it.
#:
#: `AGENT_NONDETERMINISTIC` lists `set_text` because on the AGENT endpoint
#: the replacement wording is INVENTED BY THE LLM from a sentence, so the
#: text resolved at preview need not be the text resolved at apply. Here the
#: user typed the wording. `backend.edits._h_set_text` writes
#: `params["text"]` verbatim and never touches `ctx.rewriter`, so the quoted
#: cost IS the applied cost -- the same guarantee `/edits` protects.
#:
#: `retranslate` and `code_mix` stay refused here whatever the body says:
#: those generate wording even when reached from this route.
SCRIPT_OPS = ("set_text",)


@router.post("/api/jobs/{job_id}/script", dependencies=[Depends(basic_auth)])
def stage_script(job_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Stage hand-edited lines and price the re-render.

    Body: `{"edits": [{"segment_id": "SEG_0003", "text": "..."}, ...]}`.

    `set_text` dirties from TTS, so a script edit replays speech, mix and
    lip-sync and leaves download, diarization, ASR and translation exactly
    as they are. That is the point: correcting a word should not re-run an
    hour of the pipeline, and it must not re-translate the line either --
    the wording the user typed is the wording that gets spoken.
    """
    view = _open(job_id)
    raw = body.get("edits")
    if not isinstance(raw, list) or not raw:
        raise HTTPException(422, "edits must be a non-empty list")

    merged = _read_json(view.merged_json) or {}
    known = {str(s.get("id")): s for s in (merged.get("segments") or [])}
    if not known:
        raise HTTPException(409, "this job has no segments to edit yet")

    ops: List[E.EditOp] = []
    cleaned: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        where = f"edits[{i}]"
        if not isinstance(item, dict):
            raise HTTPException(422, f"{where} is not an object")
        sid = str(item.get("segment_id") or item.get("id") or "").strip()
        if not sid:
            raise HTTPException(422, f"{where}: segment_id is required")
        if sid not in known:
            # Inventing a segment would silently drop the user's edit while
            # the response still said "staged".
            raise HTTPException(
                422, f"{where}: this job has no segment {sid!r}")
        text = item.get("text")
        if text is None or not str(text).strip():
            raise HTTPException(
                422, f"{where}: text is required and cannot be blank. To "
                     f"drop a line, clear it in the dub rather than here — "
                     f"an empty cue still occupies its slot in the timing "
                     f"plan.")
        text = str(text).strip()
        before = (known[sid].get("mt_text") or known[sid].get("text") or "")
        if text == str(before).strip():
            continue                       # unchanged; nothing to stage
        ops.append(E.EditOp(
            op="set_text", scope=E.Scope(segments=[sid]),
            params={"text": text},
            note=f"UI: script edit on {sid}"))
        cleaned.append({"segment_id": sid, "text": text, "was": before})

    if not ops:
        raise HTTPException(
            422, "no changes — every line matches what is already there")

    # MERGE with what is already staged, and record this as a turn.
    #
    # This used to build `ops` from the script edits alone and overwrite
    # `pending.json` outright, which did two wrong things at once: it
    # DESTROYED every op the agent had staged, and it wrote no turn — so the
    # edit the user had just typed did not appear in the staged list either.
    # From the screen it read as "typing a correction deletes my staged
    # changes and does nothing".
    #
    # Same shape as `agent_message`: prior turns and prior ops are carried,
    # and the new work is appended as one more turn.
    prior = _read_json(_pending_path(view.root))
    turns: List[Dict[str, Any]] = []
    prior_ops: List[E.EditOp] = []
    prior_edits: List[Dict[str, Any]] = []
    if isinstance(prior, dict):
        turns = [t for t in (prior.get("turns") or []) if isinstance(t, dict)]
        prior_edits = [e for e in (prior.get("edits") or []) if isinstance(e, dict)]
        try:
            prior_ops = [E.EditOp.from_dict(o) for o in (prior.get("ops") or [])]
        except E.EditError:
            prior_ops = []          # a stale set is dropped, not fatal

    # A second edit of the same cue REPLACES the first rather than stacking
    # another `set_text` on it. Two ops on one segment are not wrong — the
    # later wins when applied — but they make the op count, the cost line
    # and the staged list all describe work that will not happen, and they
    # grow without bound as someone tunes a line.
    rewritten = {c["segment_id"] for c in cleaned}

    def _superseded(raw: Any) -> bool:
        """A plain `set_text` on a cue this request rewrites."""
        if not isinstance(raw, dict) or raw.get("op") != "set_text":
            return False
        segs = ((raw.get("scope") or {}).get("segments") or [])
        return len(segs) == 1 and str(segs[0]) in rewritten

    all_ops = [o for o in prior_ops if not _superseded(o.to_dict())] + ops

    # Keep the turns consistent with the ops. A turn keeps only its ops that
    # survived, and a turn left with none is dropped — otherwise the staged
    # list advertises an edit whose op is no longer in the plan, which is
    # the same "says one thing, does another" this whole screen is for.
    pruned: List[Dict[str, Any]] = []
    for t in turns:
        kept_ops = [o for o in (t.get("ops") or []) if not _superseded(o)]
        if kept_ops:
            pruned.append({**t, "ops": kept_ops})
    turns = pruned

    prev = _preview(view, all_ops)
    payload = _payload(view.job_id, prior_edits, all_ops, prev)
    n = len(ops)
    payload["turns"] = turns + [{
        "at": _now(),
        "message": (f"Rewrote {n} line{'' if n == 1 else 's'}: "
                    + ", ".join(c["segment_id"] for c in cleaned)),
        "source": "script",
        "ops": [o.to_dict() for o in ops],
    }]
    _write_atomic(_pending_path(view.root), payload)
    _tracing.flush()
    log.info("script: staged %d line edit(s) for %s -> %s",
             len(ops), view.job_id, payload["dirty_stages"])
    return payload
