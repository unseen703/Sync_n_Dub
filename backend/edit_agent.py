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
OpenRouter, default `google/gemma-3-12b-it` (see `backend.llm` for why that
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
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import jobs as _jobs
from . import tracing as _tracing
from .edits import (STAGE_ORDER, sample_prompts_text, EditContext, EditError, EditOp, EditResult,
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
    records all of them (termbase, duration-adjust, mix gains, timing
    method…). Re-parsing it is the difference between replaying the run the
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


#: The operation catalogue lives HERE, in the system prompt, not in the user
#: message, because it is fixed text that never varies with the request.
#:
#: IT DID NOT SAVE TOKENS, and an earlier version of this comment claimed it
#: did. Measured on Langfuse either side of the change: 1653 input tokens per
#: parse before, 2156 after — UP 30%. The user turn did shrink (~3400 -> ~1100
#: chars), but this prompt is 1772 tokens against the old `op_catalogue()`'s
#: 635, so the system half grew by more than the user half fell. The win is
#: correctness, not cost: 7 of 8 representative requests resolve correctly
#: where the old prompt mangled several, and the fixed prefix is at least the
#: shape a prompt cache can reuse.
#:
#: The trade is that this text is now hand-maintained while `OPS` is
#: generated, so the two can drift. `tests/test_edit_agent.py` greps every
#: registered op name out of `OPS` and fails if one is missing from here,
#: which is the same guard `tests/test_ui_backend_sync.py` puts on the other
#: pair that must agree.
#:
#: It already caught one: `set_speaker` is registered in `backend/edits.py`
#: but `op_catalogue()` never advertised it, so the model could not emit an
#: op the pipeline has supported all along. This prompt lists all nine.
_SYSTEM = """You are an edit-planning agent for a dubbing and lip-syncing pipeline.

Your task is to convert a user's natural-language editing request into a minimal list of valid JSON operations. You do not execute edits. You only specify the edits the pipeline should perform.

## OUTPUT CONTRACT

Return exactly one valid JSON object with this structure:

{
  "ops": [
    {
      "op": "operation_name",
      "scope": {},
      "params": {},
      "note": "Short description of the requested change."
    }
  ]
}

Return JSON only. Do not include Markdown, code fences, explanations, or additional keys.

Every operation must contain exactly these keys:
- op
- scope
- params
- note

Use double-quoted JSON strings. Escape special characters correctly. Do not emit comments, trailing commas, or invalid JSON.

## AVAILABLE OPERATIONS

Use only the following operation names and parameters.

1. set_text
   Replace the translated text verbatim.
   Required params: text

2. retranslate
   Re-translate from the source transcript according to an instruction.
   Required params: instruction
   Optional params: target_language (only when the requested target language changes)

3. set_tone
   Change the delivery tone.
   Required params: tone
   Optional params: rewrite_text (boolean; defaults to true)

4. code_mix
   Preserve specified English terms in Latin script within the target-language translation.
   Optional params: terms (list of strings), mode ("stage" or "llm"), termbase (path string)
   Include only parameters supported by the request and available configuration.

5. set_gender
   Change the speaker's apparent gender by selecting an appropriate TTS voice.
   Required params: gender ("male" or "female")

6. set_speaker
   Reattribute dialogue to another existing speaker.
   Required params: speaker (valid speaker ID)

7. set_voice
   Assign a specific TTS voice.
   Required params: voice
   Optional params: engine (only if changing or explicitly specifying the engine)

8. set_speed
   Change the playback speed of the scoped audio.
   Required params: factor (number; greater than zero)

9. lipsync_window
   Run lip-sync only on the specified time window.
   Optional params: model ("wav2lip", "musetalk", or "latentsync"; include only when requested or needed to change the configured model)

10. replace_term
   Replace one word or phrase inside the line, leaving every other character identical.
   Required params: find (the word or phrase exactly as it appears now)
   Optional params: replace (the exact replacement; omit it to have `find` translated into the target language)

Do not invent operations, parameters, speaker IDs, cue IDs, or configuration values.

## SCOPE RULES

Scope may contain only these keys:
- start_s
- end_s
- speaker
- segments
- all

Use the smallest scope that accurately represents the request.

1. Explicit cue or segment IDs:
   Use "segments" with the exact cue IDs provided.

2. Explicit speaker:
   Use "speaker" with the exact speaker ID.
   Do not add timestamps unless the user also specifies a time restriction.

3. Explicit time or time range:
   Use "start_s" and, when applicable, "end_s".
   Copy timestamps from the request exactly, preserving seconds and fractional seconds.
   Accept formats such as "M:SS", "M:SS.ss", and "H:MM:SS".

4. Explicit speaker and time range:
   Include both the speaker and the requested time bounds.

5. Whole-dub request:
   Use {"all": true} only when the request applies to the entire dub and specifies no narrower scope.

6. Unspecified scope:
   If the request identifies no speaker, cue, or time restriction, use {"all": true}.

7. Never infer a time range from the beginning or end of the cue table.
   Never extend an explicitly scoped edit beyond its requested boundaries.

8. If the request refers to a speaker or cue that does not exist, do not silently substitute another speaker or cue. Return an empty ops array and explain the ambiguity briefly in the note of a single clarification operation is not possible. Do not invent a clarification operation; if no valid operation can express the issue, return {"ops": []}.

## OPERATION SELECTION

Choose the operation that most directly matches the requested change.

- Exact replacement wording: set_text. Preserve the user's replacement text exactly.
- Change the translation while retaining the source meaning: retranslate.
- Change ONE word or phrase and leave the rest of the line exactly as it is: replace_term. Prefer it over retranslate whenever the request names a specific word, including "translate the word X", "replace X with Y" or "keep X but change it to Y". retranslate regenerates the whole line and will alter wording the request never mentioned.
- Change delivery style, such as formal, casual, serious, or energetic: set_tone.
- Keep specified English technical terms in the translated sentence: code_mix.
- Change the perceived voice gender: set_gender.
- Correct who is speaking: set_speaker.
- Assign a particular TTS voice: set_voice.
- Change speech speed: set_speed.
- Restrict lip-sync processing to a time window or request a specific lip-sync model: lipsync_window.

Do not substitute one operation for another merely because they can produce similar results.

For requests involving distinct changes, create one operation per change. If one change fully satisfies the request, emit exactly one operation.

For example, "make SPK_01 more formal" requires set_tone with scope {"speaker":"SPK_01"}. Do not add a time range.

"Make the translation more formal and preserve the words gradient and weight in English" requires set_tone and code_mix, each with the same appropriate scope.

## PARAMETER RULES

- Include all required parameters and only relevant optional parameters.
- Do not include default values unless they are needed to make the requested behavior explicit.
- For set_tone, use rewrite_text=true unless the user explicitly asks to change vocal delivery without rewriting wording.
- For code_mix, include terms only when the user identifies terms to preserve. Do not invent a termbase path.
- For set_speed, use the numeric factor specified or unambiguously implied by the user. Do not invent a factor for vague requests such as "a little faster"; if the operation requires a factor and no configured default exists, return {"ops": []}.
- For lipsync_window, use the requested time window. Do not change the lip-sync model unless requested or necessary to specify a model.
- Do not add target_language unless the user explicitly requests a language change.
- Do not modify text unless the selected operation requires it.

## AMBIGUITY AND CONFLICTS

Resolve straightforward references using the supplied speaker list and cue table.

If a request is ambiguous in a way that affects which content is edited, which speaker is targeted, or what value must be supplied, do not guess. If no valid operation can represent the request safely, return {"ops": []}.

If two instructions conflict, prioritize the user's latest explicit instruction while preserving all non-conflicting requirements.

Never invent timestamps, replacement text, voice IDs, model IDs, or parameter values.

## FINAL VALIDATION

Before responding, silently verify:

1. The response is valid JSON.
2. Every operation is in the allowed operation list.
3. Every scope key is allowed and correctly reflects the user's request.
4. Speaker IDs and cue IDs exist when referenced.
5. Time bounds are copied accurately and are not inferred from unrelated cues.
6. Required parameters are present and correctly typed.
7. No unnecessary operations or parameters are included.
8. The note briefly describes the actual requested edit.

Return the JSON object only."""


def parse_request(nlq: str, *, data: Dict[str, Any],
                  client=None) -> Tuple[List[EditOp], str]:
    """Return `(ops, source)`; source is "llm", "rules" or "rules-unverified".

    **An empty ops array from the LLM is a verdict, not a miss.** Rule 8 of
    `_SYSTEM` tells the model to return `{"ops": []}` rather than guess when
    a request is ambiguous or names a speaker or cue the job does not have.
    This used to log that at WARNING and run the regex parser anyway -- so
    the one component that had actually read the job's speakers and cues
    said "I cannot tell what this means", and a pattern match silently
    overrode it. A regex cannot detect ambiguity: it either matches or it
    does not, and matching is not the same as being right.

    The verdict is carried instead. If the model abstained and the regexes
    still produce something, it comes back as `"rules-unverified"` with a
    loud warning so the caller can show the reading before anything runs.
    If neither produces ops, the error offers phrasings that work rather
    than the internal op names, which are the agent's vocabulary and not
    anything the user typed.
    """
    llm_abstained = False
    if client is not None and client.available():
        try:
            ops = _llm_parse(nlq, data=data, client=client)
            if ops:
                return ops, "llm"
            llm_abstained = True
            log.warning(
                "the LLM could not interpret %r -- it returned no operations, "
                "which is what it is instructed to do when a request is "
                "ambiguous or names a speaker or cue this job does not have",
                nlq)
        except Exception as e:
            log.warning("llm parse failed (%s); falling back to the rule "
                        "parser", e)
    ops = _rule_parse(nlq)
    if not ops:
        raise EditError(
            f"could not interpret: {nlq!r}\n\n"
            + ("  The LLM read this job's speakers and cues and could not "
               "match your request to any of them.\n\n"
               if llm_abstained else
               "  No OPENROUTER_API_KEY is set, so only fixed phrasings are "
               "understood.\n\n")
            + "  Try one of these shapes:\n\n"
            + sample_prompts_text(per_op=2, indent="    "))
    if llm_abstained:
        log.warning(
            "UNVERIFIED READING: the LLM declined to interpret this request, "
            "but the pattern matcher produced %d operation(s) from it. A "
            "regex matches text; it cannot tell whether the match means what "
            "you meant. Check the plan below before applying it.", len(ops))
        return ops, "rules-unverified"
    return ops, "rules"

def _llm_parse(nlq: str, *, data: Dict[str, Any], client) -> List[EditOp]:
    times = [t for t in _find_times(nlq)]
    speakers = sorted({str(s.get("speaker")) for s in data.get("segments", [])
                       if s.get("speaker")})
    # The op catalogue is NOT repeated here — it is fixed text and lives in
    # `_SYSTEM`. What belongs in the user turn is only what varies with the
    # request: this job's speakers, the cues (windowed around any timestamps
    # the request mentions), and the request itself.
    user = (
        f"Speakers: {', '.join(speakers) or '(none labelled)'}\n"
        f"Cues:\n{segment_digest(data, around=times or None)}\n\n"
        f"Request: {nlq}\n\n"
        f"JSON:")
    payload = client.chat_json(
        [{"role": "system", "content": _SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=1200,
        trace_name="edit_agent.parse",
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


#: Words that follow "use"/"switch to" in a sentence that is NOT naming a
#: voice. Without these, "use faster speech" pins a voice called "Faster".
_VOICE_STOPWORDS = {
    "the", "this", "that", "a", "an", "male", "female", "man", "woman",
    "hindi", "tamil", "telugu", "bengali", "marathi", "english",
    "musetalk", "wav2lip", "latentsync", "lipgan", "bodhan", "indicf",
    "faster", "slower", "slow", "fast", "same", "another", "other",
    "default", "neutral", "formal", "casual",
}


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

    # reattribution — "that line is SPK_02, not SPK_00"
    #
    # Ordered BEFORE gender/voice because "the speaker at 1:20 is SPK_02"
    # trips the gender regex on the bare word "speaker" otherwise, and a
    # misread reattribution is a wrong voice on a real line.
    m = re.search(r"\b(?:is|are|should\s+be|belongs?\s+to|=)\s*"
                  r"(SPK_\d+|SPEAKER[_ ]?\d+)\b", t, re.I)
    if m and not re.search(r"\bvoice\b", low):
        spk = m.group(1).upper().replace("SPEAKER ", "SPK_").replace(
            "SPEAKER_", "SPK_")
        ops.append(EditOp(op="set_speaker", scope=scope,
                          params={"speaker": spk},
                          note=f"reattribute the scoped cues to {spk}"))

    # gender
    m = re.search(r"\b(male|female|man|woman)\b", low)
    if m and re.search(r"gender|voice|sound|gender|speak", low):
        gender = "male" if m.group(1) in ("male", "man") else "female"
        ops.append(EditOp(op="set_gender", scope=scope,
                          params={"gender": gender},
                          note=f"use a {gender} voice"))

    # explicit voice id
    #
    # Two shapes. The `xx-XX-NameNeural` one is Azure/Edge, which was removed
    # from the registry on 2026-10-17 -- kept because jobs recorded before
    # that name it on disk. The bare capitalised name is `indic-speak`, which
    # is the ONLY engine now: `Amit`, `Kavya`, `Dhara`. Matching only the
    # first shape meant the fallback parser could not pin any voice that
    # currently exists, so "use Amit for SPEAKER_00" silently did nothing.
    #
    # Deliberately permissive: `_h_set_voice` checks the catalog and warns on
    # a miss, so a wrong guess here surfaces as a warning rather than a
    # silent no-op. `_VOICE_STOPWORDS` keeps the obvious false positives out.
    m = re.search(r"\b(?:use|switch to|change .*? voice to)\s+"
                  r"([A-Za-z]{2}-[A-Z]{2}-[A-Za-z]+Neural)\b", t)
    if not m:
        cand = re.search(r"\b(?:use|switch to|change .*? voice to)\s+"
                         r"([A-Z][a-z]{2,15})\b", t)
        if cand and cand.group(1).lower() not in _VOICE_STOPWORDS:
            m = cand
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
    # One request is one parse call plus up to N rewrite calls. Read as N+1
    # unrelated traces that tells you nothing about which rewrite served
    # which request, so the whole thing is one trace and the generations
    # nest inside it by OTel context.
    with _tracing.trace("edit_agent.request",
                        input=nlq,
                        metadata={"job_id": view.root.name,
                                  "segments": len(data.get("segments") or [])}):
        ops, source = parse_request(nlq, data=data, client=client)
        rewriter = _rewriter_for(client)
        ctx = build_context(view, rewriter=rewriter, ns=ns)
        # Resolve and validate without writing: `render()` has to be able to
        # show the real cue ids and the real replacement text before
        # anything is saved.
        result = apply_edits(view.merged_json, ops, ctx=ctx, nlq=nlq,
                             dry_run=True)
        result.dirty_stages = rerun_stages(
            [a.dirty_from for a in result.applied],
            available=available_stages(view, result, ns=ns))
    # Flush here rather than only at exit: the UI server is long-lived, so
    # without it a staged edit would not appear in Langfuse until the next
    # batch tick, and `--apply` from the CLI could race its own atexit.
    _tracing.flush()
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
                           max_tokens=512, trace_name="edit_agent.rewrite")
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
    # Rebuild the render settings from THIS job's manifest config. This
    # path is how the UI finishes a draft (`POST .../lipsync`), and it used
    # to pass no kwargs at all -- so the adapter fell back to its own
    # defaults and the `lipsync_batch_size` the submission recorded was
    # written to the manifest and then ignored. That was invisible while
    # the two numbers happened to agree; it stops being invisible the
    # moment either side moves. `ui/presets.py` says this config exists
    # precisely "because the replay that renders it rebuilds its kwargs
    # from this job's manifest config" -- it now actually does.
    ls_kwargs: Dict[str, Any] = {}
    if cfg.get("lipsync_batch_size"):
        ls_kwargs["batch_size"] = cfg["lipsync_batch_size"]
    log.info("edit rerun: lip-sync kwargs from manifest: %s",
             ls_kwargs or "(none recorded; adapter defaults apply)")
    out = run_lipsync(
        view.merged_json, model=model, source_video=video,
        driving_audio=wavs[-1], out_dir=view.root / "lipsync",
        pre_muxed_video=(mp4s[-1] if mp4s else None),
        num_gpus=num_gpus, window=res.lipsync_window,
        lipsync_kwargs=ls_kwargs,
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
    ap.add_argument("--list-speakers", action="store_true",
                    help="print this job's speaker ids, how long each one "
                         "talks and a sample line, then exit. `set_speaker` "
                         "refuses an id the job does not have rather than "
                         "substituting one, so this is how you find the id "
                         "to name.")
    ap.add_argument("--examples", action="store_true",
                    help="print sample requests you can copy and adapt, then "
                         "exit. These are phrasings, not op names -- "
                         "--list-ops shows the machine vocabulary, this "
                         "shows what to type.")
    ap.add_argument("--check-llm", action="store_true",
                    help="make one tiny OpenRouter round trip and report "
                         "whether the model and key work; prints no secrets")
    ap.add_argument("--check-langfuse", action="store_true",
                    help="verify the Langfuse key PAIR and host with one "
                         "round trip; prints no secrets")
    ap.add_argument("--llm-model", default=None,
                    help="OpenRouter model slug (default: $OPENROUTER_MODEL "
                         "or google/gemma-3-12b-it)")
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

    if args.examples:
        from .edits import sample_prompts_text
        print("Sample requests — copy one and change the times, speaker or "
              "wording.\n")
        print(sample_prompts_text(per_op=2, indent="  "))
        print("\n  Times accept 1:20, 0:30-0:45, '1:00 to 1:30' or seconds.")
        print("  Speaker ids are this job's own — run with --list-speakers "
              "or read them off the UI; a speaker the job does not have is "
              "refused rather than substituted.")
        return 0

    if args.list_speakers:
        view = open_job(args.job, data_dir=Path(args.data_dir),
                        merged_json=Path(args.merged) if args.merged else None)
        data = json.loads(view.merged_json.read_text())
        segs = data.get("segments") or []
        by: Dict[str, Dict[str, Any]] = {}
        for sg in segs:
            spk = str(sg.get("speaker") or "(none)")
            e = by.setdefault(spk, {"n": 0, "secs": 0.0, "sample": ""})
            e["n"] += 1
            e["secs"] += max(0.0, float(sg.get("end", 0)) - float(sg.get("start", 0)))
            if not e["sample"]:
                e["sample"] = (sg.get("mt_text") or sg.get("text") or "").strip()
        if not by:
            print("this job has no segments with speakers")
            return 1
        print(f"{view.job_id}: {len(by)} speaker(s)\n")
        for spk, e in sorted(by.items(), key=lambda kv: -kv[1]["secs"]):
            mins, secs = divmod(int(e["secs"]), 60)
            print(f"  {spk:12} {mins:d}:{secs:02d} over {e['n']:3d} line(s)")
            if e["sample"]:
                print(f"               e.g. {e['sample'][:70]}")
        print('\n  Use one in a request, e.g. '
              '"from 0:30 to 0:45 that is SPK_01".')
        return 0

    if args.check_langfuse:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        host = os.environ.get(_tracing.HOST_ENV) or _tracing.DEFAULT_HOST
        have_pub = bool((os.environ.get(_tracing.PUBLIC_KEY_ENV) or "").strip())
        have_sec = bool((os.environ.get(_tracing.SECRET_KEY_ENV) or "").strip())
        print(f"host      {host}")
        print(f"keys      public={'set' if have_pub else 'MISSING'}  "
              f"secret={'set' if have_sec else 'MISSING'}")
        if have_sec and not have_pub:
            # The reported case: only the private key in hand. Say where the
            # other one is rather than just failing, because it is NOT a
            # secret and is sitting in the UI.
            print("hint      Langfuse signs in with HTTP Basic as "
                  "public_key:secret_key and needs BOTH. The public key is "
                  "not a secret — copy the pk-lf-… from Langfuse → Project "
                  "Settings → API Keys (only the sk-lf-… is shown once).")
        ok, msg = _tracing.check()
        print(f"content   {'prompts+completions' if _tracing.capture_content() else 'metadata only'}")
        print(f"result    {msg}")
        return 0 if ok else 1

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
