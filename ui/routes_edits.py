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

from backend import edits as E                          # noqa: E402
from backend import edit_agent as A                     # noqa: E402

from .auth import basic_auth                            # noqa: E402
from .cost import estimate as cost_estimate             # noqa: E402
from .speakers import TONE_ENGINES, supports_tone        # noqa: E402

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

def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


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
        raise HTTPException(422, "edits must be a non-empty list")

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
    ops = _build_ops(body.get("edits"), config=cfg)
    prev = _preview(view, ops)
    payload = _payload(view.job_id, body.get("edits"), ops, prev)
    _write_atomic(_pending_path(view.root), payload)
    log.info("edits: staged %d op(s) for %s -> %s", len(ops), view.job_id,
             payload["dirty_stages"])
    return payload


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


def _rewriter():
    """An LLM client for the text ops, or None when no key is configured.

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
    staged_window = None
    staged_lipsync = None
    for o in ops:
        spk = o.scope.speaker
        if o.op == "set_tone" and spk:
            staged_tone[spk] = o.params.get("tone")
        elif o.op == "lipsync_window":
            staged_window = {"start_s": o.scope.start_s, "end_s": o.scope.end_s}
            staged_lipsync = o.params.get("model") or staged_lipsync

    from .speakers import build_table, tone_vocabulary
    table = build_table(view.root, cfg, data_dir=DATA_DIR,
                        merged_json=view.merged_json)
    speakers = [{
        "id": r["id"],
        "label": r.get("label"),
        "gender": r.get("gender"),
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
            # Whether the pending set would rewrite this cue. The panel
            # marks these; it does not show the new text, because resolving
            # it needs the LLM and this endpoint is polled.
            "staged": sid in touched,
        })
    return {"job_id": view.job_id, "segments": rows,
            "total": len(merged.get("segments") or [])}


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

    rewriter = _rewriter()
    try:
        new_ops, source = A.parse_request(message, data=data, client=rewriter)
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
