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
                    _lipsync_model_ids, _tone_style_ids,
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

    A UI job has no pipeline command line — `ui/runner.py` is launched with
    the job id alone — so this returns None for one, and the caller recovers
    the pacing from the recorded config instead (`speed_setup_from_config`).

    **That case is detected, not discovered by failing.** Handing the bare
    job id to the pipeline parser made argparse print its full 40-line usage
    block and `server.py: error: the following arguments are required:
    --input/-i` to stderr before raising, on EVERY staged edit. The
    `SystemExit` was always caught and nothing was wrong, but the operator
    saw a wall of text with the word "error" in it and reasonably read their
    voice change as having failed. `--input` is required by the parser, so
    its absence is the cheap, exact test for "this is not a pipeline argv".
    """
    argv = list(view.argv or ())
    if not argv:
        return None
    if not any(a == "--input" or a == "-i" or a.startswith("--input=")
               for a in argv):
        log.debug("no pipeline argv for this job (%r); the replay will take "
                  "its pacing from the recorded config", argv[:1])
        return None
    try:
        from .cli import _parse_args
        # stderr is redirected because argparse writes usage there itself,
        # before raising, and there is no parser hook to stop it.
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            return _parse_args(argv)
    except BaseException as e:            # SystemExit included
        log.info("could not re-parse the run's argv (%s); the replay will "
                 "take its pacing from the recorded config", e)
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


def _cue_number_map(data: Dict[str, Any]) -> Dict[int, str]:
    """`{1: "SEG_0000", 2: "SEG_0001", ...}` — N is 1-based in start order.

    The number is **what the UI shows** on each flashcard, so a user who says
    "cue 5" to the edit agent means the 5th row in the Clips panel. Having
    one function own that mapping is why the UI and the agent stay in sync.
    """
    segs = sorted(data.get("segments") or [], key=_seg_start)
    return {i + 1: s.get("id") for i, s in enumerate(segs) if s.get("id")}


# `cue 5`, `cues 1 and 3`, `cue #4`, `cues 1 to 3`, `cue 2, 4, 6`
# The leading boundary tolerates sentence-start, whitespace, or punctuation.
_CUE_PHRASE_RE = re.compile(
    r"(?P<lead>\bcues?)\s+#?\s*(?P<nums>\d+(?:\s*(?:[-–—]|,|\band\b|\bto\b)\s*\d+)*)",
    re.I,
)

# Bare `#5` outside a `cue …` phrase. Word boundary before, digit after.
# Only matches when preceded by start-of-string or whitespace/punctuation so
# a stray `#` inside a quoted replacement reads as literal text.
_HASH_CUE_RE = re.compile(r"(?:^|(?<=[\s(.,;:!?]))#\s*(?P<num>\d+)\b")

# `segment 4`, `segment 0004`, `seg 12`, `segments 2 and 5`.
#
# DELIBERATELY NOT THE SAME MAPPING AS `cue N`, and the difference is the
# whole reason this is a second pattern rather than another alternative in
# `_CUE_PHRASE_RE`. "cue N" is the 1-based row number on the UI flashcard,
# so cue 5 is `SEG_0004`. "segment N" is the JSON's own word, and the number
# a person reads there is the one inside the id, which is 0-based: segment 4
# is `SEG_0004`. Folding the two together would silently shift every
# `segment N` request by one cue.
#
# The underscore in `SEG_0004` is not whitespace, so an id that is already
# canonical does not match this and is left alone.
_SEGMENT_REF_RE = re.compile(
    r"\b(?:segments?|segs?)\s+#?\s*(?P<nums>\d+(?:\s*(?:[-–—]|,|\band\b|\bto\b)\s*\d+)*)",
    re.I,
)


def _expand_number_list(raw: str) -> List[int]:
    """`"1-3"` → `[1,2,3]`,  `"1 and 4"` → `[1,4]`,  `"2, 3 to 5"` → `[2,3,4,5]`."""
    tokens = re.split(r"\s*(?:,|\band\b)\s*", raw, flags=re.I)
    out: List[int] = []
    for tok in tokens:
        m = re.match(r"\s*(\d+)\s*(?:[-–—]|to)\s*(\d+)\s*$", tok, re.I)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            out.extend(range(lo, hi + 1))
            continue
        m = re.match(r"\s*(\d+)\s*$", tok)
        if m:
            out.append(int(m.group(1)))
    # Preserve order but drop dups so "cues 1, 2, 1" stays "SEG_0000, SEG_0001".
    seen = set()
    return [n for n in out if not (n in seen or seen.add(n))]


def resolve_cue_aliases(nlq: str, data: Dict[str, Any]) -> str:
    """Rewrite `cue N` and `#N` inside an edit request into their SEG ids.

    Downstream parsing (LLM and rule) already understands SEG ids, so the
    simplest way to add short-number support is to translate before parsing.
    An out-of-range number is left as literal text so the LLM can see it and
    refuse rather than silently redirecting to a different cue.
    """
    nums = _cue_number_map(data)
    if not nums:
        return nlq

    def _replace_phrase(m: re.Match) -> str:
        wanted = _expand_number_list(m.group("nums"))
        resolved = [nums[n] for n in wanted if n in nums]
        if not resolved or len(resolved) != len(wanted):
            # At least one out-of-range number — leave the whole phrase alone.
            return m.group(0)
        if len(resolved) == 1:
            return resolved[0]
        return ", ".join(resolved)

    def _replace_hash(m: re.Match) -> str:
        n = int(m.group("num"))
        return nums[n] if n in nums else m.group(0)

    # `segment N` carries its own id number, so it is checked against the
    # ids that exist rather than against the 1-based cue map above.
    known_ids = {str(s.get("id")) for s in data.get("segments") or []}

    def _replace_segment(m: re.Match) -> str:
        wanted = _expand_number_list(m.group("nums"))
        resolved = [f"SEG_{n:04d}" for n in wanted]
        if not resolved or any(sid not in known_ids for sid in resolved):
            # An id this job does not have is left as literal text, so the
            # model sees it and refuses rather than being handed a
            # plausible neighbour.
            return m.group(0)
        return ", ".join(resolved)

    out = _CUE_PHRASE_RE.sub(_replace_phrase, nlq)
    out = _SEGMENT_REF_RE.sub(_replace_segment, out)
    out = _HASH_CUE_RE.sub(_replace_hash, out)
    return out


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
    # Row number is global (over `segs`), not local to `chosen` — a windowed
    # view still shows the same `#N` the UI shows, so "cue 42" lines up even
    # when 42 isn't in the window.
    idx_of = {id(s): i + 1 for i, s in enumerate(segs)}
    rows = []
    for s in chosen:
        text = _seg_text(s).replace("\n", " ")
        if len(text) > 90:
            text = text[:87] + "…"
        rows.append(f'#{idx_of[id(s)]}\t{s.get("id")}\t'
                    f'{_hhmmss(_seg_start(s))}-{_hhmmss(_seg_end(s))}\t'
                    f'{s.get("speaker") or "?"}\t{text}')
    head = (f"{len(segs)} cue(s) total"
            + (f", showing {len(chosen)}" if len(chosen) < len(segs) else ""))
    return head + "\n#\tid\tstart-end\tspeaker\ttext\n" + "\n".join(rows)


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
_SYSTEM_TEMPLATE = """You are an edit-planning agent for a dubbing and lip-syncing pipeline.

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
   Required params: tone — one of exactly these engine styles: {{tone_styles}}
     Do not invent a register. "formal", "casual", "warm" and "urgent" are NOT engine styles and are rejected by the pipeline. If the user asks for a register that is not in the list and none of the listed styles clearly means the same thing, return {"ops": []}.
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
   Optional params: model ({{lipsync_models}}; include only when requested or needed to change the configured model)

10. replace_term
   Replace one word or phrase inside the line, leaving every other character identical.
   Required params: find
     Copy the phrase CHARACTER FOR CHARACTER as it appears in the cue text, including any punctuation that is part of it. If the user quotes a phrase that ends in "?", "!", "।" or "." and that mark is present in the cue, it belongs in `find`. Leaving it out strands the mark in the line after the phrase around it is gone.
   Optional params: replace
     - A string: the exact replacement.
     - "" (an empty string): DELETE `find` from the line. This is how "remove", "drop", "take out", "get rid of" and "delete this phrase" are expressed. Spacing around the cut is repaired for you.
     - Omitted entirely: translate `find` into the target language. Use this ONLY when the user asks for a translation. Never omit `replace` for a removal request -- omitting it substitutes a translation where a deletion was asked for.
   To remove a phrase and leave punctuation standing in its place, pass that punctuation as `replace`. That is one operation, not two.

11. mute_cue
   Remove the scoped cue(s) from the dub: TTS skips them, the music-and-effects bed plays through, and their time is released as a pause the neighbouring lines may borrow. This is the operation for "mute", "silence", "delete", "remove" and "cut" when the user means a WHOLE cue. There is no separate delete operation.
   Required params: none

12. set_target_language
   Re-translate the scoped cues (or the whole dub) into a different Indic language via Bodhan MT, then re-cast voices and re-speak. Use this when the user asks to change the dub's language ("translate to Marathi", "dub it in Tamil", "change the target language to Bengali"). This is a distinct operation from retranslate: retranslate rewrites the SAME language with a hint, while set_target_language switches the output language entirely.
   Required params: target_language (ISO code of a supported Indic language)
   Optional params: target_script (script override, only when the request names one)
   Scope rules:
     - Whole dub: use {"all": true}. This is the default when no narrower scope is explicit.
     - Cue list or time window: honour it, so a mixed-language result is produced.
     - Speaker-only scope is REFUSED by the pipeline. If the user names a speaker alone, pair it with a time range the user supplied; otherwise return {"ops": []}.
   Do not invent an ISO code the user did not name. If the language is not one of the supported Indic languages, return {"ops": []}.

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
   Use "segments" with the exact cue IDs provided. The user may refer to a cue by its short number ("cue 5", "#5", "cues 1 and 3", "cues 2-4"); these are rewritten into their SEG ids before you see this prompt, so you always receive canonical ids. If a number falls outside the table, it is left in the request literally — treat it as an unknown cue reference.

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
- Change delivery style, such as a news read, a lecture, or an emotion: set_tone. The style must be one of the engine styles listed under operation 3; there is no "formal" and no "casual".
- Keep specified English technical terms in the translated sentence: code_mix.
- Change the perceived voice gender: set_gender.
- Correct who is speaking: set_speaker.
- Assign a particular TTS voice: set_voice.
- Change speech speed: set_speed.
- Restrict lip-sync processing to a time window or request a specific lip-sync model: lipsync_window.
- Remove a word or phrase from inside a line, leaving the rest of the line standing: replace_term with replace set to "". Do not use mute_cue for this -- it discards the whole cue, not the phrase.
- Silence, delete, remove or cut a WHOLE cue: mute_cue. The distinction that matters is phrase versus cue, not "mute" versus "delete": a request naming text inside the line is replace_term, a request naming the cue itself is mute_cue.
- Change the dub's output language ("translate the dub to Marathi", "dub it in Tamil", "change the target language"): set_target_language. Never use retranslate for a language switch — retranslate rewrites the SAME target language.

Do not substitute one operation for another merely because they can produce similar results.

For requests involving distinct changes, create one operation per change. If one change fully satisfies the request, emit exactly one operation.

For example, "read SPK_01's lines as a news anchor" requires set_tone with tone "news" and scope {"speaker":"SPK_01"}. Do not add a time range.

"Make it sound sad and preserve the words gradient and weight in English" requires set_tone (tone "sad") and code_mix, each with the same appropriate scope.

## PARAMETER RULES

- Include all required parameters and only relevant optional parameters.
- Do not include default values unless they are needed to make the requested behavior explicit.
- For set_tone, use rewrite_text=true unless the user explicitly asks to change vocal delivery without rewriting wording. Never pass a tone that is not in the engine list; an unknown style is refused by the pipeline, not quietly ignored.
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

#: The prompt the model actually sees. The lip-sync model list is filled in
#: from the REGISTRY rather than typed into the template, for the reason
#: `edits._lipsync_model_ids` exists at all: the hand-written list read
#: "wav2lip, musetalk, latentsync" while five adapters were registered, so
#: two of them were unreachable through the agent. Substituting at import
#: keeps `_SYSTEM` a plain string, which is what the drift guards in
#: `tests/test_edit_agent.py` grep.
_SYSTEM = (_SYSTEM_TEMPLATE
           .replace("{{lipsync_models}}", _lipsync_model_ids())
           .replace("{{tone_styles}}", _tone_style_ids()))


def parse_request(nlq: str, *, data: Dict[str, Any],
                  client=None) -> Tuple[List[EditOp], str]:
    """Return `(ops, source)`; source is "rules" or "llm".

    **The rule parser runs FIRST.** Most edit requests are a fixed shape —
    "speed this up by 1.2x", "mute cue 5", "use Amit for SPEAKER_00" — and
    the regexes read every one of them exactly, locally, in microseconds and
    for nothing. Sending those to a 12B model over the network bought a
    round trip per message and a non-deterministic reading of a sentence
    that had only one reading. The LLM is now what it is good at: the
    sentences the patterns genuinely cannot place.

    This reverses a deliberate earlier decision, so the reasoning it
    replaces is worth keeping. The LLM used to go first because it reads the
    job's actual speakers and cues and can therefore ABSTAIN — return
    `{"ops": []}` when a request is ambiguous or names a cue that does not
    exist — and a regex cannot: it either matches or it does not, and
    matching is not the same as being right. Running the rules first gives
    that veto up for every sentence they match.

    What makes the trade acceptable is `_rule_parse`'s own conservatism: it
    emits only shapes it is sure of, each branch guards the ones that could
    swallow it, and a match now carries a real scope (`_scope_from` learned
    cue ids at the same time as this change — before that, a matched rule
    could be right about the op and silently wrong about half the dub).
    What remains true is that a regex cannot detect ambiguity. If a wrong
    reading ever reaches a render, the fix is a narrower branch here, not a
    second opinion after the fact.

    If neither parser produces ops, the error offers phrasings that work
    rather than the internal op names, which are the agent's vocabulary and
    not anything the user typed.
    """
    # `cue 5` / `#5` → `SEG_0004` before either parser runs, so both see the
    # same canonical cue ids the rest of the agent and `edits.py` understand.
    nlq = resolve_cue_aliases(nlq, data)

    ops = _rule_parse(nlq)
    if ops:
        return ops, "rules"

    if client is not None and client.available():
        try:
            ops = _llm_parse(nlq, data=data, client=client)
            if ops:
                return ops, "llm"
            log.warning(
                "the LLM could not interpret %r -- it returned no operations, "
                "which is what it is instructed to do when a request is "
                "ambiguous or names a speaker or cue this job does not have",
                nlq)
            raise EditError(
                f"could not interpret: {nlq!r}\n\n"
                "  The patterns did not match, and the LLM read this job's "
                "speakers and cues and could not match your request to any "
                "of them.\n\n"
                "  Try one of these shapes:\n\n"
                + sample_prompts_text(per_op=2, indent="    "))
        except EditError:
            raise
        except Exception as e:
            log.warning("llm parse failed (%s); the rule parser had already "
                        "declined this request", e)

    raise EditError(
        f"could not interpret: {nlq!r}\n\n"
        "  No OPENROUTER_API_KEY is set, so only fixed phrasings are "
        "understood.\n\n"
        "  Try one of these shapes:\n\n"
        + sample_prompts_text(per_op=2, indent="    "))

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
#: Canonical cue ids. `resolve_cue_aliases` has already turned `cue 5`, `#5`
#: and `segment 5` into these before either parser runs, so this is what a
#: cue-scoped request looks like by the time `_scope_from` sees it.
_SEG_ID_RE = re.compile(r"\bSEG_\d{4}\b", re.I)


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
    """Best-effort scope: explicit cue ids beat a range beats a point beats a
    speaker beats all.

    The cue-id branch is FIRST and it is the one that was missing. By the
    time this runs `resolve_cue_aliases` has rewritten `cue 5` into
    `SEG_0004`, and nothing here matched that shape -- so every cue-scoped
    request the rules parsed fell through to `Scope(all=True)`. "cue 5:
    speed this up by 1.2x" sped up the whole dub, silently, and the UI's own
    selection prefix was the most common way to produce that sentence.
    """
    from .edits import _as_time
    segs = _SEG_ID_RE.findall(text)
    if segs:
        # First-seen order, de-duplicated: "SEG_0004 and SEG_0004" is one cue.
        return Scope(segments=list(dict.fromkeys(x.upper() for x in segs)))
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
    "musetalk", "wav2lip", "latentsync", "bodhan", "indicf",
    "faster", "slower", "slow", "fast", "same", "another", "other",
    "default", "neutral", "formal", "casual",
}


def _rule_parse(nlq: str) -> List[EditOp]:
    """Regex parser. Deliberately conservative: it emits only what it is sure
    of, so a half-understood sentence becomes "I did not understand" rather
    than a confident wrong edit.

    This is now the FIRST parser, not the fallback — see `parse_request`. The
    branches are ordered so that the narrowest reading wins: a request that
    quotes a phrase is a phrase edit, not a line edit and not a language
    switch, and the guards below say so explicitly rather than relying on
    which regex happens to be tried first.
    """
    t = nlq.strip()
    low = t.lower()
    ops: List[EditOp] = []
    scope = _scope_from(t)
    quotes = [q for q in _QUOTED_RE.findall(t) if q.strip()]

    def _has(*names: str) -> bool:
        return any(o.op in names for o in ops)

    # mute / delete a WHOLE cue
    #
    # First, and with one guard: a quoted phrase means the request names text
    # INSIDE the line, which is `replace_term` below. "There is no separate
    # delete operation" (`_SYSTEM`, op 11) — mute IS the delete here.
    #
    # Until this branch existed `mute_cue` was reachable only through the LLM
    # and through the mute button, so "mute the line at 0:15" — its own
    # advertised example — did not parse at all without an API key.
    if not quotes and (
            re.search(r"\b(mute|mutes|muted|silence|silenced)\b", low)
            or (re.search(r"\b(delete|remove|drop|cut|skip|kill)\b", low)
                and re.search(r"\b(cue|cues|line|lines|segment|segments"
                              r"|seg_\d{4})\b", low))):
        ops.append(EditOp(op="mute_cue", scope=scope, params={},
                          note="silence the scoped cues"))

    # one word or phrase inside the line
    #
    # Ordered BEFORE the language switch, because `translate the word
    # "conditional" into Hindi` — `replace_term`'s own advertised example —
    # otherwise matched `into Hindi` further down and came back as a
    # whole-dub language swap: a one-word request re-translating the entire
    # video and re-casting every voice.
    #
    # The quotes are the signal. `find` has to be character-for-character
    # what is in the cue, so a request that does not quote the phrase is left
    # to the LLM rather than guessed at from loose words.
    if quotes and re.search(r"\b(replace|replaces|swap|substitute|change"
                            r"|remove|delete|drop|get rid of|take out"
                            r"|translate)\b", low):
        params: Optional[Dict[str, Any]] = {"find": quotes[0].strip()}
        if len(quotes) > 1:
            params["replace"] = quotes[1]
        elif re.search(r"\b(remove|delete|drop|get rid of|take out|cut)\b",
                       low):
            # "" DELETES `find`; omitting it would translate it instead,
            # which is the one substitution this op must never make by
            # accident. Spacing and stranded punctuation are repaired by
            # `_h_replace_term`.
            params["replace"] = ""
        elif re.search(r"\btranslate\b", low):
            pass                    # omitted -> translate `find` in place
        else:
            # A single quoted string with a verb like "change" is the whole
            # new line, not a term: leave it to `set_text` below.
            params = None
        if params is not None:
            ops.append(EditOp(op="replace_term", scope=scope, params=params,
                              note=f"replace {params['find']!r} in place"))

    # lip-sync window
    #
    # The model is resolved first, from the REGISTRY, and a named model is
    # itself a trigger. The old condition needed the literal word "lipsync",
    # so "run musetalk on the last 20 seconds" — an advertised example —
    # matched nothing. LONGEST first, so "wav2lip_hd" is not read as the
    # plain model: that silent downgrade is what this loop exists to stop.
    from models.lipsync import available_models
    probe = low.replace("-", "_").replace(" ", "_")
    ls_model = None
    for name in sorted(available_models(), key=len, reverse=True):
        if name in probe:
            ls_model = name
            break
    if re.search(r"lip[\s-]?sync|lipsync", low) or (
            ls_model and re.search(r"\b(run|use|redo|rerun|re-?run|with|on)\b",
                                   low)):
        sc = scope
        if sc.is_empty():
            sc = Scope(all=True)
        ops.append(EditOp(op="lipsync_window", scope=sc,
                          params={k: v for k, v in (("model", ls_model),) if v},
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
    #
    # "keep X in English" is the phrasing the op catalogue advertises and the
    # one people actually type; the literal words "code-mix" are the jargon.
    # Matching only the jargon left the advertised example unparseable.
    if re.search(r"code[\s-]?mix", low) or re.search(
            r"\b(keep|keeps|leave|retain|preserve)\b[^.]*\bin\s+english\b", low):
        terms = re.search(r"\b(?:keep|leave|retain|preserve)\s+(.+?)\s+in\s+english",
                          low)
        params = {}
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
    #
    # `actually`/`really` are allowed between the verb and the id because the
    # workspace's own fill phrasing puts one there.
    m = re.search(r"\b(?:is|are|should\s+be|belongs?\s+to|=)\s+"
                  r"(?:actually\s+|really\s+)?"
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
    #
    # Skipped when `replace_term` already claimed the quoted phrase: two ops
    # on one sentence would replace the whole line AND a word inside it.
    if re.search(r"\b(change|replace|set)\b.*\bto\b", low) and not _has(
            "replace_term"):
        q = _QUOTED_RE.search(t)
        if q:
            ops.append(EditOp(op="set_text", scope=scope,
                              params={"text": q.group(1).strip()},
                              note="replace the line verbatim"))

    # tone
    #
    # The vocabulary is the ENGINE's, read through `edits.tone_styles()`.
    # This used to match "formal | casual | friendly | serious | energetic",
    # and Bodhan accepts none of those five — so every tone the rules parsed
    # was written to `tts_style` and then ignored at synthesis.
    # Longest-first so "news" inside "AIR style news" does not win.
    from .edits import tone_styles
    tone = None
    for style in sorted(tone_styles(), key=len, reverse=True):
        if re.search(rf"(?<![\w]){re.escape(style.lower())}(?![\w])", low):
            tone = style
            break
    if tone and re.search(r"tone|register|style|deliver|delivery|sounds?\b"
                          r"|read it|say it|make it|more\b", low):
        ops.append(EditOp(op="set_tone", scope=scope, params={"tone": tone},
                          note=f"shift the register to {tone}"))

    # target-language switch -- `translate ... to <lang>`, `change (the)
    # (target) language to <lang>`, `dub (it) in <lang>`, `retranslate ...
    # into <lang>`. Ordered AFTER replace_term so that `translate the word
    # "X" into Hindi` is a term translation, not a whole-dub swap. We
    # additionally require a known Indic language on the right -- `translate
    # to English` is latn and comes back as nothing rather than a wrong op.
    iso = _resolve_target_language(t) if not _has("replace_term") else None
    if iso:
        # No scope hint -> whole dub. A scope term the user included is
        # honoured verbatim (segments / time window). Speaker-only is left
        # for the handler to refuse.
        lang_scope = scope if not scope.is_empty() else Scope(all=True)
        ops.append(EditOp(op="set_target_language",
                          scope=lang_scope,
                          params={"target_language": iso},
                          note=f"switch target language to {iso}"))

    # retranslate
    #
    # Widened to the phrasings the catalogue advertises ("make the
    # translation simpler", "translate this again"), and guarded against
    # every op that also rewrites wording — `code_mix` in particular, whose
    # own example "code-mix the translation" contains the trigger words.
    if re.search(r"re[\s-]?translate|translate\b[^.]*\bagain\b"
                 r"|better translation|wrong translation|fix the translation"
                 r"|translation is|the translation\b", low) and not _has(
                     "set_text", "set_target_language", "replace_term",
                     "code_mix"):
        ops.append(EditOp(op="retranslate", scope=scope,
                          params={"instruction": t},
                          note="re-translate the scoped cues"))
    return ops


def _resolve_target_language(text: str) -> Optional[str]:
    """Pull an Indic ISO code out of a free-text request, or None.

    Only fires when the request names a language-switch verb -- `translate`,
    `retranslate`, `dub`, or `(change the) target language` -- followed by
    `to`/`in`/`into` and a token the shared `ui.langs.LANGUAGES` table
    recognises (ISO code, English name, or native-script name).
    """
    if not re.search(r"\b(?:translate|retranslate|dub|language)\b",
                     text, re.I):
        return None
    try:
        from ui.langs import LANGUAGES
    except Exception:
        return None
    name_to_iso: Dict[str, str] = {}
    for iso, meta in LANGUAGES.items():
        if meta.get("script") == "latn":
            continue
        name_to_iso[iso.lower()] = iso
        name_to_iso[(meta.get("english") or "").strip().lower()] = iso
        native = (meta.get("native") or "").strip()
        if native:
            name_to_iso[native] = iso
    # `to Marathi`, `into Tamil`, `in मराठी`. English/native tokens may
    # include spaces (none do in the current table but keep the group
    # permissive).
    m = re.search(r"\b(?:to|into|in)\s+([A-Za-zऀ-᱿]+)\b", text)
    if not m:
        return None
    key = m.group(1).strip().lower()
    iso = name_to_iso.get(key) or name_to_iso.get(m.group(1).strip())
    return iso


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
    if cfg.get("timing_producer"):
        out.append("timing")
    if cfg.get("tts"):
        out.append("mix")
    if cfg.get("lipsync"):
        out.append("lipsync")
    # `cfg` first: a UI job has no argv, so reading this off the namespace
    # alone dropped stage 9c from the replayable set on every one of them.
    # It is recorded in the manifest (`ui.presets.plan_config`), which is the
    # durable record either way.
    if cfg.get("duration_adjust") or nsget("duration_adjust"):
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
    # Before anything is written, so an older job's history is back-filled
    # from the files as they were and does not pick up this edit twice.
    _jv_ready = _job_versions_ready(plan.view.root)
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
    # The EDITED document, not just the request that produced it.
    # `edit_NNN.json` records what was asked and what the handlers reported;
    # it is not the transcript, so "what did the dub actually say after edit
    # 3" could only be answered by replaying. The producer carries the same
    # NNN as the journal entry, so the two line up by name.
    from .mt_versions import snapshot as _mt_snapshot
    _mt_snapshot(plan.view.merged_json, f"edit_{stamp:03d}")
    # Map this edit to the files it produced. A record, never a dependency.
    if _jv_ready:
        try:
            from . import job_versions as _jv
            ops_d = [o.to_dict() for o in plan.ops]
            _jv.append(
                plan.view.root,
                ("lipsync" if ops_d and all(o["op"] == "lipsync_window"
                                            for o in ops_d) else "edit"),
                edit_id=f"edit_{stamp:03d}", nlq=plan.nlq, ops=ops_d,
                dirty_stages=res.dirty_stages,
                artifacts=_jv.link_transcript(plan.view.root,
                                              f"edit_{stamp:03d}"))
        except Exception as e:                           # noqa: BLE001
            log.warning("job_versions: could not record edit_%03d (%s)",
                        stamp, e)
    plan.result = res
    return res


def _job_versions_ready(root: Path) -> bool:
    """Create (or back-fill) the job's version index; False if that failed."""
    try:
        from . import job_versions as _jv
        _jv.read(root)
        return True
    except Exception as e:                               # noqa: BLE001
        log.warning("job_versions: index unavailable for %s (%s)",
                    Path(root).name, e)
        return False


# ---------------------------------------------------------------------------
# re-running
# ---------------------------------------------------------------------------

def speed_setup_from_config(cfg: Dict[str, Any]):
    """The run's pacing, read off a recorded config when there is no argv.

    Takes any dict using the pipeline's own kwarg names, which is both the
    manifest's `config` (via `ui.presets.plan_config`) and the `pipeline`
    block of a UI job's `request.json`. The second matters because it is
    already on disk for every job ever submitted from the UI, so a job that
    predates this fix still replays correctly — `ui/runner.py` passes it.

    `recover_args` re-parses the command line because it carries knobs the
    manifest never did. A UI job has no command line: `ui/runner.py` is
    launched with the job id alone, so the pipeline parser exits on it and
    `ns` comes back None. The replay then fell through to argparse DEFAULTS
    — which describe a run nobody launched.

    The damage was specific and silent. A UI submission runs
    `timing_producer=naive_slot` with `use_target_duration=True`,
    `tts_apply_stretch=False`, `mix_strategy=timing_plan`,
    `mix_clip_source=raw` (`ui/presets.UI_TIMING_DEFAULTS`). The defaults are
    `timing_params=None`, stretch ON, `global_tempo`, `auto`. So every edit
    replayed on a UI job came back paced differently from the dub it was
    editing — and `set_speed` came back at the ORIGINAL SPEED, because its
    entire mechanism is writing `target_duration_s` and
    `use_target_duration` defaults to False, which makes `naive_slot` ignore
    that field and re-fit every clip to its untouched `end - start` slot.
    `backend/timing.py` predicts exactly this in its own docstring.

    Returns None when the manifest records no pacing at all — a job written
    before 2026-10-22, or by a runner that does not use `presets.plan_config`
    — so the caller keeps its previous behaviour rather than inventing a
    setup from nothing.
    """
    from .cli import SpeedSetup

    keys = ("timing_params", "tts_apply_stretch", "mix_strategy",
            "mix_clip_source", "mix_tempo_cap")
    if not any(cfg.get(k) is not None for k in keys):
        log.info("edit rerun: the manifest records no pacing; the replay "
                 "will use defaults, which may not match the original run")
        return None

    params = cfg.get("timing_params")
    setup = SpeedSetup(
        timing_producer=cfg.get("timing_producer"),
        timing_params=dict(params) if isinstance(params, dict) else None,
        tts_apply_stretch=bool(cfg.get("tts_apply_stretch")),
        tts_force_align=bool(cfg.get("tts_force_align")),
        mix_strategy=cfg.get("mix_strategy") or "global_tempo",
        mix_tempo_cap=cfg.get("mix_tempo_cap") or 1.5,
        mix_clip_source=cfg.get("mix_clip_source") or "auto",
    )
    log.info("edit rerun: pacing recovered from the manifest (no argv)")
    return setup


def rerun(plan: EditPlan, *, device: Optional[str] = None,
          num_gpus: Optional[int] = None,
          speed_setup=None,
          data_dir: Optional[Path] = None,
          mix_producer_tag: Optional[str] = None) -> Dict[str, Any]:
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
        mt_target_language=(res.target_language_override
                            or cfg.get("target_language")),
        mt_target_script=(res.target_script_override
                          or cfg.get("target_script")),
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
    # Three sources, most authoritative first. The caller's beats the argv
    # because only a caller that HAS the submission can supply it —
    # `ui/runner.py` reads `request.json`, which records the resolved kwargs
    # verbatim — and the manifest is the fallback for everyone else.
    setup = speed_setup
    if setup is None and ns is not None:
        try:
            from .cli import resolve_speed_setup
            setup = resolve_speed_setup(ns)
        except BaseException as e:                 # SystemExit included
            log.info("could not replay the run's speed setup (%s); using the "
                     "manifest's producer with its defaults", e)
    if setup is None:
        # No argv to re-parse — which is EVERY UI job, not an edge case. The
        # UI calls `run_pipeline()` directly and records `['<job_id>']` as
        # argv, so `recover_args` hands the pipeline parser a bare job id, it
        # exits, and `ns` is None. See `speed_setup_from_config`.
        setup = speed_setup_from_config(cfg)
    if setup is not None:
        log.info("edit rerun: speed setup producer=%s strategy=%s clip=%s "
                 "tts_stretch=%s", setup.timing_producer, setup.mix_strategy,
                 setup.mix_clip_source, setup.tts_apply_stretch)

    kwargs: Dict[str, Any] = dict(
        source_video=video,
        me_stem=me_stem,
        reference_audio_src=dia_input,
        out_root=view.root,
        # `set_target_language` writes the new language onto `res` so the
        # replay uses it even though `cfg` (read off the manifest) still
        # names the old one. The handler also mutates `data["target_language"]`
        # in the merged JSON for the per-cue consumers.
        target_language=(res.target_language_override
                         or cfg.get("target_language")),
        target_script=(res.target_script_override
                       or cfg.get("target_script")),
        source_language=cfg.get("asr_language"),
        turns=turns,
        device=device or nsget("device"),
        # 6 (relocated). ASD scores faces at the gate (for gender) and at the
        # lip-sync replay (for face selection). Pass the configured model when
        # either stage is replayed; `_run_asd_stage` is idempotent and reuses
        # scores already on disk, so this never double-scores.
        asd_model=("loconet" if ("lipsync" in stages or "voices" in stages) else None),
        asd_on_screen_threshold=cfg.get("asd_on_screen_threshold", 0.3),
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
        tts_max_stretch=nsget("tts_max_stretch",
                              cfg.get("tts_max_stretch") or 0.10),
        stretch_backend=nsget("stretch_backend", "auto"),
        tts_apply_stretch=(setup.tts_apply_stretch if setup is not None
                           else not bool(nsget("no_tts_stretch"))),
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
        # Caller-supplied tag -- `ui/runner.py` derives it from the edit
        # record filename, so `mix/versions/` grows one dubbed.wav/mp4 pair
        # per edit replay. None skips the snapshot.
        mix_producer_tag=mix_producer_tag,
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


def _latentsync_targets_npz(root: Path, video: Path, *, device: str = "cuda"):
    """Build the per-frame face targets LatentSync reads, from the ASD scores.

    LatentSync ignores `coord_list` (it needs 106 landmarks per frame), so the
    speaking-face choice reaches it only through this npz. Built from whatever
    `asd/*.scores.npz` is on disk (loconet, after `_run_asd_stage`). Best
    effort: on any failure the adapter falls back to its own largest-face
    choice, which is the pre-ASD behaviour rather than a hard error.
    """
    try:
        from .asd import find_scores_npz, load_scores_npz
        from .lipsync_targets import build_targets, save_targets
        from models.faces.base import FaceTracks

        npz = find_scores_npz(root / "asd", video.stem)
        faces = sorted((root / "faces").glob("*.npz"))
        if not npz or not faces:
            log.warning("latentsync targets: no ASD scores/faces under %s; the "
                        "adapter will pick a face itself", root)
            return None
        tracks = FaceTracks.load_npz(faces[0])
        asd = load_scores_npz(npz)
        payload, _ = build_targets(video, tracks, asd, device=device)
        out = root / "targets" / f"{video.stem}.targets.npz"
        save_targets(payload, out)
        log.info("latentsync targets -> %s", out)
        return out
    except Exception as e:                               # noqa: BLE001
        log.warning("latentsync targets build failed (%s); the adapter will "
                    "pick a face itself", e)
        return None


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
    # LatentSync 1.5 UI defaults for the render button. `memory_efficient`
    # composites straight into the encoder so only one copy of the clip is
    # resident instead of three -- measured ~12 GiB -> ~4.7 GiB per shard,
    # and the pixels are identical. `inference_steps=10` + `enable_deepcache`
    # roughly halve the UNet cost; this is the fast combo, and it DOES soften
    # the mouth (DeepCache leaves ~4 full passes at 10 steps -- see
    # `latentsync_adapter.py`), which is the trade being taken on purpose.
    #
    # latentsync15 ONLY: the registry drops kwargs an adapter does not accept,
    # but gating keeps the other models on their own defaults and makes the
    # log honest about what was set. `cfg` still overrides inference_steps and
    # deepcache -- but NOT memory_efficient (see below).
    if str(model).strip() == "latentsync15":
        # memory_efficient is FORCED on for every UI latentsync1.5 render, not
        # merely defaulted. It only changes WHEN frames are freed -- one copy of
        # the clip resident instead of three -- so the pixels are identical, and
        # it roughly thirds peak host RAM: a 90s-segment run measured ~60 GB with
        # it off. There is no quality reason to turn it off, so cfg cannot.
        ls_kwargs["memory_efficient"] = True
        ls_kwargs.setdefault(
            "inference_steps", int(cfg.get("lipsync_inference_steps", 10)))
        ls_kwargs.setdefault(
            "enable_deepcache", bool(cfg.get("lipsync_deepcache", True)))
    log.info("edit rerun: lip-sync kwargs from manifest: %s",
             ls_kwargs or "(none recorded; adapter defaults apply)")
    # WHICH face to animate, same as stage 9h in the full chain.
    #
    # This path bypasses `run_post_mt_stages` entirely — the mix is still
    # valid, so it drives the stage directly — which means it needs its own
    # call here or it keeps MuseTalk's first-detection face choice. And this
    # is the path that matters most: it is how the UI finishes a draft, so
    # leaving it out would have fixed the pipeline while the button everyone
    # actually presses carried on animating whoever was found first.
    from .pipeline import _asd_coord_list, _run_asd_stage, _turns_from_merged

    # Run ASD (loconet) just before lip-sync if it has not been scored yet.
    # This path bypasses `run_post_mt_stages`, so without this it only READ
    # existing scores -- and since ASD was relocated out of the main pipeline,
    # a fresh job reaches here with none and lip-syncs UNGATED. `_run_asd_stage`
    # is idempotent: it scores loconet the first time and reuses after (and
    # reuses the cast gate's scores when present).
    asd_model = "loconet"
    try:
        _turns = _turns_from_merged(
            json.loads(Path(view.merged_json).read_text()))
    except Exception:                                    # noqa: BLE001
        _turns = []
    _audio_dir = view.root / "audio"
    _ref_audio = _audio_dir / f"{video.stem}.dialogue.16k.mono.wav"
    if not _ref_audio.exists():
        _ref_audio = _audio_dir / f"{video.stem}.16k.mono.wav"
    _run_asd_stage(
        view.root, video, asd_model=asd_model, turns=_turns,
        reference_audio=_ref_audio,
        asd_on_screen_threshold=float(cfg.get("asd_on_screen_threshold", 0.3)))

    # MuseTalk reads `coord_list`; LatentSync ignores it and reads a targets
    # npz. Build whichever the chosen model consumes -- both off the loconet
    # scores just written -- so the speaking face reaches either adapter.
    _targets = _asd_coord_list(
        view.root, video, model,
        bbox_shift=int(cfg.get("bbox_shift") or 0))
    _targets_npz = (_latentsync_targets_npz(view.root, video)
                    if str(model).strip().startswith("latentsync") else None)
    out = run_lipsync(
        view.merged_json, model=model, source_video=video,
        driving_audio=wavs[-1], out_dir=view.root / "lipsync",
        pre_muxed_video=(mp4s[-1] if mp4s else None),
        num_gpus=num_gpus, window=res.lipsync_window,
        lipsync_kwargs=ls_kwargs,
        coord_list=(_targets[0] if _targets else None),
        targets_npz=_targets_npz,
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
        _job_versions_ready(view.root)    # back-fill before the file changes
        ids = revert(view.merged_json)
        print(f"reverted {len(ids)} segment(s) in {view.merged_json}"
              if ids else "nothing to revert")
        if ids:
            from . import job_versions as _jv
            _jv.record_revert(view.root, view.merged_json)
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
