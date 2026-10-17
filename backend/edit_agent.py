"""NLQ edit agent — "say what to change", applied to a finished dub.

    $ PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
    $ $PY -m backend.edit_agent --job 20261016T1012__nag__hi \\
          "the speaker at 1:20 should sound female, and code-mix the
           translation between 0:30 and 1:00"

What it does, in order:

  1. opens the job folder and its merged ASR+MT JSON (the file stages 9b..9h
     read and write in place);
  2. turns the sentence into typed `backend.edits.EditOp`s — with the
     OpenRouter LLM when a key is configured, with regexes when it is not;
  3. resolves every scope against real segment ids and PRINTS THE PLAN,
     including which stages the edit invalidates and roughly what that costs;
  4. only with `--apply` does it write the JSON; only with `--run` does it
     re-render.

Dry-run is the default on purpose. An edit to a translated line is cheap; the
TTS + mix + lip-sync chain it invalidates is not, and a lip-sync pass over a
video that did not need one is the most expensive way to find out the sentence
was misread.

Re-rendering replays stages 9b..9h only. Download, diarization, face
detection, ASD, ASR, alignment and translation are not re-run: they already
produced the merged JSON, and `backend.pipeline._bootstrap_post_mt_only`
points the replay at the artifacts they left in the job folder.

The LLM
-------
OpenRouter, default `google/gemma-3-4b-it` (see `backend.llm` for why that
slug). It is used for two things: reading the request, and rewriting text for
`retranslate` / `set_tone` / `code_mix(mode=llm)`. Everything else —
resolving a timestamp to a segment, picking a voice of a given gender,
computing a stretch factor, deciding which stages are dirty — is ordinary
code, because those are the parts that must not be improvised.

Without `OPENROUTER_API_KEY` the agent still runs: `_rule_parse` handles the
common shapes ("speed up X to Y by 1.2x", "lip-sync 1:00 to 1:30",
"code-mix", "make the speaker at 1:20 female"), and any op that genuinely
needs a rewriter refuses with a message saying so.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import jobs as _jobs
from .edits import (STAGE_ORDER, EditContext, EditError, EditOp, EditResult,
                    OP_SPECS, OPS, Scope, apply_edits, rerun_stages, revert,
                    _hhmmss, _seg_end, _seg_start, _seg_text)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

#: How many cues go into the prompt. A 4B model's attention over a 300-cue
#: table is worse than over a focused 60-cue one, and the window the request
#: names is what matters anyway.
_MAX_PROMPT_SEGMENTS = 60


# ---------------------------------------------------------------------------
# locating the job
# ---------------------------------------------------------------------------

@dataclass
class JobView:
    """A job folder plus everything an edit needs to know about it."""
    job_id: str
    root: Path
    manifest: Dict[str, Any]
    merged_json: Path
    config: Dict[str, Any] = field(default_factory=dict)
    argv: Optional[List[str]] = None
    video: Optional[Path] = None

    @property
    def input(self) -> str:
        return str(self.manifest.get("input") or (self.video or ""))


def open_job(job_id: Optional[str], *, data_dir: Path,
             merged_json: Optional[Path] = None) -> JobView:
    """Open a job by id, or infer one from an explicit merged JSON path."""
    if merged_json is not None:
        merged_json = Path(merged_json).resolve()
        if not merged_json.exists():
            raise EditError(f"no such merged JSON: {merged_json}")
        root = _job_root_of(merged_json) or merged_json.parent
        job_id = job_id or root.name
    else:
        if not job_id:
            raise EditError("pass --job <job_id> or --merged <path>")
        root = Path(data_dir) / "jobs" / job_id
        if not root.is_dir():
            raise EditError(
                f"no such job: {root}\n"
                f"  known: {', '.join(_recent_job_ids(data_dir)) or '(none)'}")
        merged_json = _find_merged_json(root)

    manifest = {}
    mpath = root / "manifest.json"
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text())
        except ValueError as e:
            log.warning("manifest unreadable (%s); falling back to defaults", e)
    return JobView(
        job_id=job_id, root=root, manifest=manifest, merged_json=merged_json,
        config=dict(manifest.get("config") or {}),
        argv=manifest.get("argv"),
        video=_find_video(root),
    )


def _job_root_of(merged_json: Path) -> Optional[Path]:
    for anc in list(merged_json.resolve().parents)[:6]:
        if (anc / "audio").is_dir() and (anc / "videos").is_dir():
            return anc
    return None


def _recent_job_ids(data_dir: Path, limit: int = 8) -> List[str]:
    root = Path(data_dir) / "jobs"
    if not root.is_dir():
        return []
    return [d.name for d in sorted(root.iterdir(), reverse=True)
            if d.is_dir()][:limit]


def _find_merged_json(root: Path) -> Path:
    """The merged ASR+MT JSON this job's post-MT stages read and write."""
    mt_dir = root / "mt"
    cands = sorted(mt_dir.glob("*.mt.json")) if mt_dir.is_dir() else []
    if not cands:
        raise EditError(
            f"{root.name} has no merged ASR+MT JSON under {mt_dir}. "
            f"An edit works on the translated transcript, so the run has to "
            f"have reached the MT stage (--mt …).")
    if len(cands) > 1:
        newest = max(cands, key=lambda p: p.stat().st_mtime)
        log.info("job has %d merged JSONs; using the newest (%s)",
                 len(cands), newest.name)
        return newest
    return cands[0]


_VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".avi")


def _find_video(root: Path) -> Optional[Path]:
    d = root / "videos"
    if not d.is_dir():
        return None
    vids = [p for p in sorted(d.iterdir())
            if p.suffix.lower() in _VIDEO_SUFFIXES and p.is_file()]
    if not vids:
        return None
    return max(vids, key=lambda p: p.stat().st_size)


def recover_args(view: JobView) -> Optional[argparse.Namespace]:
    """Re-parse the run's own argv with the pipeline's own parser.

    The manifest's `config` records a dozen knobs; the original command line
    records all of them (termbase, duration-adjust, mix gains, restore
    fidelity…). Re-parsing it is the difference between replaying the run the
    user actually launched and replaying a defaults-shaped approximation.
    """
    if not view.argv:
        return None
    try:
        from .cli import _parse_args
        return _parse_args(list(view.argv))
    except BaseException as e:            # SystemExit included
        log.info("could not re-parse the run's argv (%s); using manifest "
                 "config + defaults", e)
        return None


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

def build_context(view: JobView, *, rewriter=None,
                  ns: Optional[argparse.Namespace] = None) -> EditContext:
    cfg = view.config
    data = json.loads(view.merged_json.read_text())
    extra = data.get("extra", {}) or {}
    return EditContext(
        target_language=(cfg.get("target_language")
                         or extra.get("mt_target_language")),
        source_language=cfg.get("asr_language") or data.get("language"),
        tts_engine=cfg.get("tts") or (extra.get("tts") or {}).get("model"),
        voice_engine=cfg.get("voice_engine"),
        voice_catalog_json=(Path(ns.voice_catalog)
                            if ns is not None and getattr(ns, "voice_catalog", None)
                            else None),
        translit_model=(getattr(ns, "translit", None) if ns is not None else None)
                       or (extra.get("codemix") or {}).get("model"),
        lipsync_model=cfg.get("lipsync"),
        termbase_path=(Path(ns.termbase)
                       if ns is not None and getattr(ns, "termbase", None)
                       else None),
        work_dir=view.root / "edits",
        rewriter=rewriter,
    )


# ---------------------------------------------------------------------------
# NLQ -> ops
# ---------------------------------------------------------------------------

def op_catalogue() -> str:
    """The op list, rendered for the prompt straight from `OP_SPECS`."""
    lines = []
    for spec in OP_SPECS:
        params = "; ".join(f"{k}: {v}" for k, v in spec.params.items())
        lines.append(f'- "{spec.name}" — {spec.summary}\n'
                     f'    params: {params or "(none)"}')
        for ex in spec.examples:
            lines.append(f'    e.g. "{ex}"')
    return "\n".join(lines)


def segment_digest(data: Dict[str, Any], *, limit: int = _MAX_PROMPT_SEGMENTS,
                   around: Optional[List[float]] = None) -> str:
    """A compact cue table for the prompt, centred on any times mentioned."""
    segs = sorted(data.get("segments") or [], key=_seg_start)
    if not segs:
        return "(no segments)"
    chosen = segs
    if len(segs) > limit:
        if around:
            keep = set()
            for t in around:
                i = min(range(len(segs)),
                        key=lambda j: abs(_seg_start(segs[j]) - t))
                keep.update(range(max(0, i - limit // (2 * len(around)) - 1),
                                  min(len(segs), i + limit // (2 * len(around)) + 2)))
            chosen = [segs[i] for i in sorted(keep)][:limit]
        else:
            step = max(1, len(segs) // limit)
            chosen = segs[::step][:limit]
    rows = []
    for s in chosen:
        text = _seg_text(s).replace("\n", " ")
        if len(text) > 90:
            text = text[:87] + "…"
        rows.append(f'{s.get("id")}\t{_hhmmss(_seg_start(s))}-'
                    f'{_hhmmss(_seg_end(s))}\t{s.get("speaker") or "?"}\t{text}')
    head = (f"{len(segs)} cue(s) total"
            + (f", showing {len(chosen)}" if len(chosen) < len(segs) else ""))
    return head + "\nid\tstart-end\tspeaker\ttext\n" + "\n".join(rows)


_SYSTEM = """You convert a dubbing-studio edit request into JSON operations.

Reply with ONE JSON object and nothing else:
  {"ops": [{"op": "...", "scope": {...}, "params": {...}, "note": "..."}]}

scope keys (use the fewest that express the request):
  "start_s"  seconds, or "M:SS" / "H:MM:SS"
  "end_s"    seconds — omit for a single point in time
  "speaker"  a speaker id exactly as it appears in the cue table
  "segments" a list of cue ids from the table
  "all"      true for the whole dub

Rules:
- Use ONLY the ops listed. Never invent an op, a param or a scope key.
- One request can need several ops; emit one object per distinct change.
- Copy timestamps from the request. Do not round them to whole minutes.
- If the request names no time, speaker or cue, set "all": true.
- "note" is one short sentence saying what you understood. It is shown to
  the user, so write it for them, not for yourself."""


def parse_request(nlq: str, *, data: Dict[str, Any],
                  client=None) -> Tuple[List[EditOp], str]:
    """Return `(ops, source)` where source is "llm" or "rules"."""
    if client is not None and client.available():
        try:
            ops = _llm_parse(nlq, data=data, client=client)
            if ops:
                return ops, "llm"
            log.warning("llm returned no ops; falling back to the rule parser")
        except Exception as e:
            log.warning("llm parse failed (%s); falling back to the rule "
                        "parser", e)
    ops = _rule_parse(nlq)
    if not ops:
        raise EditError(
            "could not turn that request into an edit.\n"
            f"  Supported: {', '.join(sorted(OPS))}\n"
            "  With OPENROUTER_API_KEY set, free-form phrasing is understood; "
            "without it, phrase it like: "
            '\"speed up 0:30 to 0:45 by 1.2x\", "lip-sync 1:00 to 1:30", '
            '"code-mix the translation", "make the speaker at 1:20 female".')
    return ops, "rules"


def _llm_parse(nlq: str, *, data: Dict[str, Any], client) -> List[EditOp]:
    times = [t for t in _find_times(nlq)]
    speakers = sorted({str(s.get("speaker")) for s in data.get("segments", [])
                       if s.get("speaker")})
    user = (
        f"Available ops:\n{op_catalogue()}\n\n"
        f"Speakers: {', '.join(speakers) or '(none labelled)'}\n"
        f"Cues:\n{segment_digest(data, around=times or None)}\n\n"
        f"Request: {nlq}\n\n"
        f"JSON:")
    payload = client.chat_json(
        [{"role": "system", "content": _SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=1200,
    )
    raw = payload.get("ops") if isinstance(payload, dict) else payload
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise EditError(f"model returned {type(raw).__name__}, expected a list")
    return [EditOp.from_dict(o) for o in raw if isinstance(o, dict)]


# ---------------------------------------------------------------------------
# rule-based fallback
# ---------------------------------------------------------------------------

_TS = r"(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:seconds|second|secs|sec|s)\b)"
_RANGE_RE = re.compile(
    rf"(?:between|from)\s+{_TS}\s*(?:and|to|-|–|—|until)\s*{_TS}"
    rf"|{_TS}\s*(?:-|–|—|to)\s*{_TS}", re.I)
_POINT_RE = re.compile(rf"\bat\s+{_TS}", re.I)
_FACTOR_RE = re.compile(r"(?:by\s*)?(\d+(?:\.\d+)?)\s*(?:x|times)\b", re.I)
_PERCENT_RE = re.compile(r"(?:by\s*)?(\d+(?:\.\d+)?)\s*(?:%|percent)", re.I)
_QUOTED_RE = re.compile(r'["“”\'](.{2,400}?)["“”\']', re.S)


def _clean_ts(s: str) -> str:
    return re.sub(r"\s*(?:seconds|second|secs|sec|s)$", "", s.strip(), flags=re.I)


def _find_times(text: str) -> List[float]:
    from .edits import _as_time
    out = []
    for m in re.finditer(_TS, text, re.I):
        try:
            out.append(_as_time(_clean_ts(m.group(0))))
        except EditError:
            continue
    return [t for t in out if t is not None]


def _scope_from(text: str) -> Scope:
    """Best-effort scope: a range beats a point beats a speaker beats all."""
    from .edits import _as_time
    m = _RANGE_RE.search(text)
    if m:
        groups = [g for g in m.groups() if g]
        if len(groups) >= 2:
            return Scope(start_s=_as_time(_clean_ts(groups[0])),
                         end_s=_as_time(_clean_ts(groups[1])))
    m = _POINT_RE.search(text)
    if m:
        return Scope(start_s=_as_time(_clean_ts(m.group(1))))
    m = re.search(r"\b(SPEAKER[_\s-]?\d+|spk[_\s-]?\d+)\b", text, re.I)
    if m:
        return Scope(speaker=re.sub(r"[\s-]", "_", m.group(1)).upper())
    return Scope(all=True)


def _rule_parse(nlq: str) -> List[EditOp]:
    """Regex parser. Deliberately conservative: it emits only what it is sure
    of, so a half-understood sentence becomes "I did not understand" rather
    than a confident wrong edit."""
    t = nlq.strip()
    low = t.lower()
    ops: List[EditOp] = []
    scope = _scope_from(t)

    # lip-sync window
    if re.search(r"lip[\s-]?sync|lipsync", low):
        model = None
        for name in ("wav2lip", "musetalk", "latentsync", "lipgan"):
            if name in low:
                model = name
                break
        sc = scope
        if sc.end_s is None and sc.start_s is None and not sc.speaker:
            sc = Scope(all=True)
        ops.append(EditOp(op="lipsync_window", scope=sc,
                          params={k: v for k, v in (("model", model),) if v},
                          note="lip-sync the named window only"))

    # speed
    if re.search(r"\bspeed\s*(?:up|down)?\b|\bfaster\b|\bslower\b|\bslow\s*down\b",
                 low):
        factor = None
        m = _FACTOR_RE.search(t)
        if m:
            factor = float(m.group(1))
        else:
            m = _PERCENT_RE.search(t)
            if m:
                pct = float(m.group(1)) / 100.0
                factor = 1.0 + pct
        slower = bool(re.search(r"\bslower\b|\bslow\s*down\b|\bspeed\s*down\b", low))
        if factor is None:
            factor = 0.85 if slower else 1.2
        elif slower and factor > 1.0:
            factor = round(1.0 / factor, 4)
        ops.append(EditOp(op="set_speed", scope=scope, params={"factor": factor},
                          note=f"{'slow down' if factor < 1 else 'speed up'} "
                               f"by {factor:.2f}x"))

    # code-mix
    if re.search(r"code[\s-]?mix", low):
        terms = re.search(r"keep\s+(.+?)\s+in\s+english", low)
        params: Dict[str, Any] = {}
        if terms:
            params["terms"] = [w.strip() for w in
                               re.split(r",|\band\b", terms.group(1))
                               if w.strip()]
        ops.append(EditOp(op="code_mix", scope=scope, params=params,
                          note="code-mix the translation"))

    # gender
    m = re.search(r"\b(male|female|man|woman)\b", low)
    if m and re.search(r"gender|voice|sound|gender|speak", low):
        gender = "male" if m.group(1) in ("male", "man") else "female"
        ops.append(EditOp(op="set_gender", scope=scope,
                          params={"gender": gender},
                          note=f"use a {gender} voice"))

    # explicit voice id
    m = re.search(r"\b(?:use|switch to|change .*? voice to)\s+"
                  r"([A-Za-z]{2}-[A-Z]{2}-[A-Za-z]+Neural)\b", t)
    if m:
        ops.append(EditOp(op="set_voice", scope=scope,
                          params={"voice": m.group(1)},
                          note=f"pin voice {m.group(1)}"))

    # literal replacement text
    if re.search(r"\b(change|replace|set)\b.*\bto\b", low):
        q = _QUOTED_RE.search(t)
        if q:
            ops.append(EditOp(op="set_text", scope=scope,
                              params={"text": q.group(1).strip()},
                              note="replace the line verbatim"))

    # tone
    m = re.search(r"\b(formal|informal|casual|friendly|serious|energetic|"
                  r"calm|polite|neutral)\b", low)
    if m and re.search(r"tone|register|style|sound more|make it more", low):
        ops.append(EditOp(op="set_tone", scope=scope,
                          params={"tone": m.group(1)},
                          note=f"shift the register to {m.group(1)}"))

    # retranslate
    if re.search(r"re[\s-]?translate|better translation|wrong translation|"
                 r"fix the translation|translation is", low) and not any(
                     o.op in ("set_text",) for o in ops):
        ops.append(EditOp(op="retranslate", scope=scope,
                          params={"instruction": t},
                          note="re-translate the scoped cues"))
    return ops


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

@dataclass
class EditPlan:
    view: JobView
    nlq: str
    ops: List[EditOp]
    source: str
    result: EditResult
    ctx: EditContext
    ns: Optional[argparse.Namespace] = None

    @property
    def stages(self) -> List[str]:
        return self.result.dirty_stages

    def render(self) -> str:
        r = self.result
        L: List[str] = []
        L.append(f"job        {self.view.job_id}")
        L.append(f"merged     {self.view.merged_json}")
        L.append(f"request    {self.nlq}")
        L.append(f"parsed by  {self.source}"
                 + ("  (regex fallback — OPENROUTER_API_KEY unset)"
                    if self.source == "rules" else ""))
        L.append("")
        L.append(f"{len(r.applied)} operation(s):")
        for a, op in zip(r.applied, self.ops):
            L.append(f"  • {op.op} on {a.scope}")
            if op.note:
                L.append(f"      understood as: {op.note}")
            L.append(f"      {len(a.segments)} cue(s): "
                     f"{', '.join(a.segments[:8])}"
                     f"{' …' if len(a.segments) > 8 else ''}")
            for c in a.changes[:8]:
                L.append(f"      - {c}")
            if len(a.changes) > 8:
                L.append(f"      - … {len(a.changes) - 8} more")
            for w in a.warnings:
                L.append(f"      ! {w}")
        L.append("")
        if r.lipsync_window:
            s, e = r.lipsync_window
            L.append(f"lip-sync window  {_hhmmss(s)}–{_hhmmss(e)} "
                     f"({e - s:.1f}s)")
        L.append(f"stages to re-run  {' → '.join(r.dirty_stages) or '(none)'}")
        L.append(f"cost              {_cost_note(r.dirty_stages, r)}")
        L.append("stages NOT re-run  download, diarization, faces, ASD, ASR, "
                 "align, MT (the merged JSON already holds their output)")
        return "\n".join(L)


def _cost_note(stages: List[str], r: EditResult) -> str:
    bits = []
    if "codemix" in stages:
        bits.append("one transliteration call per cue")
    if "tts" in stages:
        bits.append("a full TTS pass (every cue is re-synthesised)")
    if "mix" in stages:
        bits.append("an ffmpeg mix + mux")
    if "lipsync" in stages:
        if r.lipsync_window:
            s, e = r.lipsync_window
            bits.append(f"a GPU lip-sync render of {e - s:.0f}s")
        else:
            bits.append("a GPU lip-sync render of the WHOLE video")
    return "; ".join(bits) or "nothing to re-render"


def make_plan(nlq: str, *, view: JobView, client=None,
              ns: Optional[argparse.Namespace] = None) -> EditPlan:
    data = json.loads(view.merged_json.read_text())
    ops, source = parse_request(nlq, data=data, client=client)
    rewriter = _rewriter_for(client)
    ctx = build_context(view, rewriter=rewriter, ns=ns)
    # Resolve and validate without writing: `render()` has to be able to show
    # the real cue ids and the real replacement text before anything is saved.
    result = apply_edits(view.merged_json, ops, ctx=ctx, nlq=nlq, dry_run=True)
    result.dirty_stages = rerun_stages(
        [a.dirty_from for a in result.applied],
        available=available_stages(view, result, ns=ns))
    return EditPlan(view=view, nlq=nlq, ops=ops, source=source,
                    result=result, ctx=ctx, ns=ns)


def available_stages(view: JobView, res: EditResult,
                     ns: Optional[argparse.Namespace] = None) -> List[str]:
    """Stages this job can actually replay.

    A run with no voice conversion and no timing plan must not be told to
    re-run them: the plan would promise work that never happens, and the cost
    estimate the user approves would be for a different run than the one that
    executes. A stage the EDIT introduces (code-mix on a job that never
    code-mixed, lip-sync on a job that never lip-synced) is added back.
    """
    cfg = view.config

    def nsget(name, default=None):
        return getattr(ns, name, default) if ns is not None else default

    out = ["voices"]                      # 9d always runs when TTS does
    if cfg.get("tts"):
        out.append("tts")
    if cfg.get("vc"):
        out.append("vc")
    if cfg.get("timing_producer"):
        out.append("timing")
    if cfg.get("tts"):
        out.append("mix")
    if cfg.get("lipsync"):
        out.append("lipsync")
    if nsget("duration_adjust"):
        out.append("duration")
    # Added by the edit itself.
    if res.translit_model:
        out.append("codemix")
    if res.lipsync_model or res.lipsync_window:
        out.append("lipsync")
    return sorted(set(out), key=lambda s: STAGE_ORDER.index(s))


def _rewriter_for(client):
    if client is None or not client.available():
        return None

    def rewrite(system: str, user: str) -> str:
        return client.chat([{"role": "system", "content": system},
                            {"role": "user", "content": user}],
                           max_tokens=512)
    return rewrite


def commit(plan: EditPlan) -> EditResult:
    """Write the edits. Re-runs the handlers so LLM rewrites are not reused
    from the dry run — a rewrite is non-deterministic, and showing one string
    while saving another is the kind of mismatch nobody catches later."""
    res = apply_edits(plan.view.merged_json, plan.ops, ctx=plan.ctx,
                      nlq=plan.nlq, dry_run=False)
    res.dirty_stages = rerun_stages(
        [a.dirty_from for a in res.applied],
        available=available_stages(plan.view, res, ns=plan.ns))
    out_dir = plan.view.root / "edits"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = len(list(out_dir.glob("edit_*.json")))
    (out_dir / f"edit_{stamp:03d}.json").write_text(
        json.dumps({"nlq": plan.nlq, "parsed_by": plan.source,
                    "ops": [o.to_dict() for o in plan.ops],
                    "result": res.to_jsonable()}, indent=2, ensure_ascii=False))
    plan.result = res
    return res


# ---------------------------------------------------------------------------
# re-running
# ---------------------------------------------------------------------------

def rerun(plan: EditPlan, *, device: Optional[str] = None,
          num_gpus: Optional[int] = None,
          data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Replay the stages the edit invalidated, and nothing else.

    `data_dir` is the caller's, not the original run's: a job folder that has
    been moved or copied still replays against where it is NOW. It only
    matters as `_bootstrap_post_mt_only`'s fallback, since that function
    finds the upstream artifacts by walking up from the merged JSON.
    """
    from .pipeline import (_bootstrap_post_mt_only, run_post_mt_stages)

    view, res, ns = plan.view, plan.result, plan.ns
    cfg = view.config
    stages = set(res.dirty_stages)
    if not stages:
        return {"status": "nothing to re-run"}

    # Lip-sync-only: the mix is still valid, so drive the stage directly
    # rather than through the post-MT chain, whose 9g/9h guards both require
    # a TTS pass in the same call.
    if stages <= {"lipsync"}:
        return _rerun_lipsync_only(plan, num_gpus=num_gpus)

    input_ = str(view.video) if view.video and view.video.exists() else view.input
    boot = _bootstrap_post_mt_only(
        input_, view.merged_json,
        data_dir=Path(data_dir or getattr(ns, "data_dir", None)
                      or DEFAULT_DATA_DIR),
        diarization_model=cfg.get("diarization") or "pyannote",
        stem_model=cfg.get("stem_model") or "htdemucs",
        skip_stems=bool(cfg.get("skip_stems")),
        asr_language=cfg.get("asr_language"),
        mt_model=cfg.get("mt"),
        mt_target_language=cfg.get("target_language"),
        mt_target_script=cfg.get("target_script"),
        videos_dir=view.root / "videos",
    )
    _result, video, dia_input, me_stem, turns = boot

    def nsget(name, default=None):
        return getattr(ns, name, default) if ns is not None else default

    # Replay the run's OWN speed setup. Reading `--mix-strategy` and
    # `--no-tts-stretch` off the namespace is not enough: `--speed-mode
    # segment` and `--timing-method` override both AFTER argparse, so a
    # segment-fit run re-rendered from the raw namespace would come back as a
    # global-tempo run that also stretches twice.
    setup = None
    if ns is not None:
        try:
            from .cli import resolve_speed_setup
            setup = resolve_speed_setup(ns)
        except BaseException as e:                 # SystemExit included
            log.info("could not replay the run's speed setup (%s); using the "
                     "manifest's producer with its defaults", e)
    if setup is not None:
        log.info("edit rerun: speed setup producer=%s strategy=%s clip=%s "
                 "tts_stretch=%s", setup.timing_producer, setup.mix_strategy,
                 setup.mix_clip_source, setup.tts_apply_stretch)

    kwargs: Dict[str, Any] = dict(
        source_video=video,
        me_stem=me_stem,
        reference_audio_src=dia_input,
        out_root=view.root,
        target_language=cfg.get("target_language"),
        target_script=cfg.get("target_script"),
        source_language=cfg.get("asr_language"),
        turns=turns,
        device=device or nsget("device"),
        # 9b
        translit_model=(res.translit_model if "codemix" in stages else None),
        termbase_path=res.termbase_path or nsget("termbase"),
        # 9c
        duration_adjust=bool(nsget("duration_adjust")) if "duration" in stages else False,
        duration_voice=nsget("duration_voice", "default"),
        # 9d
        voice_engine=(cfg.get("voice_engine") if "voices" in stages else None),
        voice_catalog_json=nsget("voice_catalog"),
        allow_voice_reuse=bool(nsget("allow_voice_reuse")),
        gender_overrides=res.gender_overrides or None,
        speaker_voice_overrides=res.speaker_voice_overrides or None,
        # 9e
        tts_model=(cfg.get("tts") if "tts" in stages else None),
        tts_sample_rate=nsget("tts_sample_rate", 24000),
        tts_max_stretch=nsget("tts_max_stretch", 0.10),
        stretch_backend=nsget("stretch_backend", "auto"),
        tts_apply_stretch=(setup.tts_apply_stretch if setup is not None
                           else not bool(nsget("no_tts_stretch"))),
        # 9f
        vc_model=(cfg.get("vc") if "vc" in stages else None),
        # 9f2
        timing_producer=((setup.timing_producer if setup is not None
                          else cfg.get("timing_producer"))
                         if "timing" in stages else None),
        timing_params=(setup.timing_params if setup is not None else None),
        # 9g
        mix=("mix" in stages),
        mix_sample_rate=nsget("mix_sample_rate", 48000),
        mix_tts_gain_db=nsget("mix_tts_gain_db", 3.0),
        mix_me_gain_db=nsget("mix_me_gain_db", -3.0),
        mix_strategy=(setup.mix_strategy if setup is not None
                      else nsget("mix_strategy", "global_tempo")),
        mix_tempo_cap=(setup.mix_tempo_cap if setup is not None
                       else nsget("mix_tempo_cap", 1.5)),
        mix_clip_source=(setup.mix_clip_source if setup is not None
                         else nsget("mix_clip_source", "auto")),
        mux_video=not bool(nsget("no_mux")),
        # 9h
        lipsync_model=((res.lipsync_model or cfg.get("lipsync"))
                       if "lipsync" in stages else None),
        lipsync_num_gpus=num_gpus if num_gpus is not None else nsget("lipsync_num_gpus"),
        lipsync_window=res.lipsync_window,
        restore_model=(cfg.get("restore") if "lipsync" in stages else None),
    )
    log.info("edit rerun: stages %s", " → ".join(res.dirty_stages))
    post = run_post_mt_stages(view.merged_json, **kwargs)
    return {
        "status": "ok",
        "stages": res.dirty_stages,
        "mix_wav": str(post.mix_wav) if post.mix_wav else None,
        "mix_mp4": str(post.mix_mp4) if post.mix_mp4 else None,
        "lipsync_mp4": str(post.lipsync_mp4) if post.lipsync_mp4 else None,
        "lipsync_error": post.lipsync_error,
    }


def _rerun_lipsync_only(plan: EditPlan, *, num_gpus: Optional[int]) -> Dict[str, Any]:
    from .lipsync import run_lipsync

    view, res = plan.view, plan.result
    cfg = view.config
    model = res.lipsync_model or cfg.get("lipsync")
    if not model:
        raise EditError(
            "no lip-sync model: the job did not run one and the request did "
            "not name one. Add e.g. 'with musetalk' to the request.")
    mix_dir = view.root / "mix"
    wavs = sorted(mix_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    mp4s = sorted(mix_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not wavs:
        raise EditError(
            f"no dubbed audio in {mix_dir}; a lip-sync edit needs the mix "
            f"stage to have run. Re-run the dub first.")
    video = view.video
    if video is None:
        raise EditError(f"no source video in {view.root / 'videos'}")
    log.info("edit rerun: lip-sync only (%s)%s", model,
             f", window {res.lipsync_window}" if res.lipsync_window else "")
    out = run_lipsync(
        view.merged_json, model=model, source_video=video,
        driving_audio=wavs[-1], out_dir=view.root / "lipsync",
        pre_muxed_video=(mp4s[-1] if mp4s else None),
        num_gpus=num_gpus, window=res.lipsync_window,
        restore=cfg.get("restore"),
    )
    return {"status": "ok", "stages": ["lipsync"],
            "lipsync_mp4": str(out.out_video), "lipsync_error": out.error}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m backend.edit_agent",
        description="Edit a finished dub by describing the change in words.",
        epilog="Dry-run by default: nothing is written without --apply and "
               "nothing is re-rendered without --run.",
    )
    ap.add_argument("request", nargs="*",
                    help="what to change, in plain language")
    ap.add_argument("--job", "-j", default=None, help="job id under data/jobs/")
    ap.add_argument("--merged", default=None, type=Path,
                    help="merged ASR+MT JSON, instead of --job")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), type=Path)
    ap.add_argument("--apply", action="store_true",
                    help="write the edits into the merged JSON")
    ap.add_argument("--run", action="store_true",
                    help="re-run the invalidated stages (implies --apply)")
    ap.add_argument("--revert", action="store_true",
                    help="undo every recorded edit on this job and exit")
    ap.add_argument("--list-ops", action="store_true",
                    help="print the supported operations and exit")
    ap.add_argument("--check-llm", action="store_true",
                    help="make one tiny OpenRouter round trip and report "
                         "whether the model and key work; prints no secrets")
    ap.add_argument("--llm-model", default=None,
                    help="OpenRouter model slug (default: $OPENROUTER_MODEL "
                         "or google/gemma-3-4b-it)")
    ap.add_argument("--no-llm", action="store_true",
                    help="never call the LLM; use the regex parser only")
    ap.add_argument("--device", default=None, help="cuda | cpu")
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="GPUs for a lip-sync re-render")
    ap.add_argument("--json", action="store_true",
                    help="print the plan as JSON instead of text")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-7s %(message)s")

    if args.list_ops:
        print(op_catalogue())
        return 0

    if args.check_llm:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        from .llm import LLMUnavailable, OpenRouterClient, extract_json
        client = OpenRouterClient(model=args.llm_model)
        print(f"endpoint  {client.describe()}")
        if not client.available():
            print("result    NOT CONFIGURED — set OPENROUTER_API_KEY in .env")
            return 1
        try:
            got = client.chat_json(
                [{"role": "user",
                  "content": 'Reply with exactly {"ok": true} and nothing else.'}],
                max_tokens=32)
        except LLMUnavailable as e:
            print(f"result    FAILED — {e}")
            return 1
        print(f"reply     {got}")
        print("result    OK — the key, the slug and JSON mode all work"
              if isinstance(got, dict) and got.get("ok") is True else
              "result    REACHABLE, but the model did not follow the JSON "
              "instruction. It will still work: the parser tolerates prose, "
              "and the regex fallback catches the rest.")
        return 0

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        view = open_job(args.job, data_dir=args.data_dir,
                        merged_json=args.merged)
    except EditError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.revert:
        ids = revert(view.merged_json)
        print(f"reverted {len(ids)} segment(s) in {view.merged_json}"
              if ids else "nothing to revert")
        if ids:
            print("re-run the dub to hear it: the audio on disk still carries "
                  "the edits.")
        return 0

    nlq = " ".join(args.request).strip()
    if not nlq:
        print("error: say what to change, e.g. "
              '"speed up 0:30 to 0:45 by 1.2x"', file=sys.stderr)
        return 2

    client = None
    if not args.no_llm:
        from .llm import OpenRouterClient
        client = OpenRouterClient(model=args.llm_model)
        if not client.available():
            log.warning("OPENROUTER_API_KEY unset — using the regex parser. "
                        "Ops that rewrite text (retranslate, set_tone, "
                        "code_mix mode=llm) will refuse.")

    ns = recover_args(view)
    try:
        plan = make_plan(nlq, view=view, client=client, ns=ns)
    except EditError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"nlq": nlq, "parsed_by": plan.source,
                          "ops": [o.to_dict() for o in plan.ops],
                          "result": plan.result.to_jsonable()},
                         indent=2, ensure_ascii=False))
    else:
        print(plan.render())

    if not (args.apply or args.run):
        print("\n(dry run — nothing written. Add --apply to save, "
              "--run to save and re-render.)")
        return 0

    commit(plan)
    print(f"\nwritten → {view.merged_json}")
    if not args.run:
        print("(not re-rendered — add --run, or re-run the stages yourself.)")
        return 0

    out = rerun(plan, device=args.device, num_gpus=args.num_gpus,
                data_dir=args.data_dir)
    print("\nre-ran: " + " → ".join(out.get("stages") or []))
    for k in ("mix_wav", "mix_mp4", "lipsync_mp4"):
        if out.get(k):
            print(f"  {k:12s} {out[k]}")
    if out.get("lipsync_error"):
        print(f"  lipsync_error {out['lipsync_error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
