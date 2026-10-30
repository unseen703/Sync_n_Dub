"""Shorten a translation that will not fit the time it has.

Stage 9c2, between duration planning (9c) and TTS (9e). When a line needs
more speed-up than the mixer can apply, nothing downstream can fix it --
`atempo` is already clamped and Bodhan TTS has no speed parameter -- so the
only remaining lever is to make the sentence shorter.

How it works:

* **Gate.** A segment is rewritten only when `estimated time(est) > target * speed_cap`
  (1.5 by default). A line needing 1.2x is left alone because `atempo` fits
  it, and rewriting text that already renders correctly is a translation
  risk for nothing.
* **It rewrites `mt_text`, not the ASR ouput.** `original_language_asr` is passed as
  context so the model can verify meaning and spot ASR errors, but
  retranslating from the source throws away a translation that is usually
  correct and merely long.
* **The prompt is `translate_agetn.prompt.txt` verbatim**, with
  `{{target_language}}` and `{{source_language}}` filled in. It carries its
  own syllable arithmetic,
  a cheapest-first cutting strategy, the preservation rules (negation,
  modality, numbers, protected terms) and a QC schema.
* **The model's metrics are recorded, not trusted.** `est_duration_s` as
  returned goes in the log; the real decision is `backend/duration.py`
  re-scoring the returned text. A rewrite that is not actually shorter is
  discarded.
* **`infeasible` ends the rounds for that segment.** A second ask cannot
  make an incompressible line compressible.
* **Segments go up in batches of `batch_size` (default 2).** Mapping is by
  the echoed `segment_id`; a reply that cannot be mapped is re-asked one
  segment at a time rather than matched by guesswork. Batching can cost an
  extra call; it cannot cost a segment.
* **Every failure keeps the original text.** No key, a dead model slug, bad
  JSON or a refusal all fall back to what was there. A shortening pass must
  never be the reason a dub fails.

The original wording is kept in `seg["condense_backup"]`, mirroring
`backend/edits.py`.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

#: Model for the rewrite. A small fast one suits the task -- "say this
#: shorter in the same language", not reasoning.
#:
#: This slug has NOT been confirmed against OpenRouter's live catalogue from
#: this host. A wrong slug degrades to "no shortening": the call fails, the
#: original text is kept, and the dub still renders, so it can be wrong for
#: a long time without anyone noticing. It shows in `extra.condense.error`
#: and the Langfuse trace. Confirm with:
#:   curl -s https://openrouter.ai/api/v1/models -H "Authorization: Bearer $OPENROUTER_API_KEY" \
#:     | python -c "import json,sys;[print(m['id']) for m in json.load(sys.stdin)['data'] if 'gemini' in m['id']]"
#: and set CONDENSE_MODEL if it prints something different.
DEFAULT_MODEL = os.getenv("CONDENSE_MODEL") or "google/gemini-3.1-flash-lite"

#: The speed-up the mix can actually apply. A segment is left alone while
#: `atempo` can absorb it, and rewritten only once it needs more than this.
DEFAULT_SPEED_CAP = 1.5

#: Slack above the cap before a rewrite is worth a call, and the epsilon on
#: "does it fit now". 0.0 means the cap is the cap.
DEFAULT_TOLERANCE = 0.0

#: Give up on a segment after this many rewrites. Each round costs a call
#: and the returns fall off fast.
DEFAULT_MAX_ROUNDS = 2

#: The chunk is `min(batch_size, len(todo))`, so a single overlong segment
#: is still a single-segment call.
DEFAULT_BATCH_SIZE = 2


def _env_int(name: str, default: int) -> int:
    """Read an int env var, tolerating the blank a copied template leaves.

    `int(os.getenv("X", 2))` raises on `X=`, which turns an empty line in
    .env into a crash at import.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        log.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default

#: The reference prompt, verbatim from `translate_agetn.prompt.txt` with
#: `{{target_language}}` substituted at call time. Kept as one string so it
#: can be diffed against that file.
_SYSTEM_TEMPLATE = (Path(__file__).resolve().parent.parent
                    / "translate_agetn.prompt.txt")

#: Fallback if the file is missing, so a deployment that lost it still
#: dubs. Deliberately terse -- the real prompt is the file.
_FALLBACK_SYSTEM = """You are a bilingual dubbing QC reviewer on a dubbing pipeline that goes {{source_language}} -> {{target_language}}. Shorten machine-translated lines so each one fits its own audio slot, preserving meaning, tone and protected terminology.

You receive a `segments` array of one or more segment objects, each with `segment_id`, `original_language_asr`, `mt_text`, `target_duration_s`, `est_duration_s`, `protected_terms` and `stretch_cap_pct`.

Each segment is an independent rewrite. Use only that segment's own fields. Never move, borrow or blend wording between segments, and never use one segment as context for another. Echo `segment_id` exactly. Return exactly one output object per input segment, in input order, with the same count -- never merged, split, omitted or added.

Rewrite `mt_text`, not `original_language_asr`. Use `original_language_asr` only to verify meaning. Preserve negation, modality, conditions, named entities, numbers, units, dates and every protected term exactly. Never add facts. Never pad to fill the slot. Keep natural spoken {{target_language}}, including code-mixing.

Return exactly one JSON object: {"segments": [{"segment_id": "<echo>", "text": "<rewritten line>", "fits": false, "infeasible": false, "meaning_loss": "none", "adequacy": "ok", "adequacy_issues": [], "preserved_terms": [], "asr_suspect": false, "note": null}]}"""


def _language_name(code: Optional[str], default: str) -> str:
    """ISO code -> English language name, or the code itself if unknown."""
    try:
        from frontend.langs import english_name
    except Exception:                                     # noqa: BLE001
        return code or default
    return english_name(code, default)


def _system_prompt(language: str, source_language: str) -> str:
    """The reference prompt with both languages filled in."""
    try:
        raw = _SYSTEM_TEMPLATE.read_text()
    except OSError as e:                                  # noqa: BLE001
        log.warning("condense: %s unreadable (%s); using the short fallback "
                    "prompt", _SYSTEM_TEMPLATE.name, e)
        raw = _FALLBACK_SYSTEM
    return (raw.replace("{{target_language}}", language)
               .replace("{{source_language}}", source_language))


def _segment_payload(seg: Dict[str, Any], *, text: str, est_s: float,
                     target_s: float, speed_cap: float) -> Dict[str, Any]:
    """One segment as the prompt's own input contract describes it.

    JSON rather than prose because the prompt names its fields
    (`original_language_asr`, `mt_text`, `target_duration_s`, ...) and reads them back
    in its arithmetic. `original_language_asr` is ASR output of the speaker -- the one
    field here that did not originate inside this pipeline, so it is passed
    as data.

    `stretch_cap_pct` is derived from the speed cap, not left at the
    prompt's default of 10: the prompt computes `fits` as
    `est <= target * (1 + stretch_cap_pct/100)`, so 10 would report
    `fits: false` for lines that render perfectly well and cut them far
    harder than necessary.
    """
    protected = [sp["text"] for sp in (seg.get("retain_spans") or [])
                 if isinstance(sp, dict) and sp.get("text")]
    return {
        "segment_id": str(seg.get("id", "")),
        "original_language_asr": str(seg.get("text") or ""),
        "mt_text": text,
        "target_duration_s": round(float(target_s), 2),
        "est_duration_s": round(float(est_s), 2),
        "protected_terms": protected,
        "stretch_cap_pct": round((speed_cap - 1.0) * 100.0),
    }


def _user_payload(rows: List[Dict[str, Any]]) -> str:
    """The chunk, in the `{"segments": [...]}` shape the prompt declares.

    Always an array, even for one segment: a request whose shape changes
    with the batch size is two contracts.
    """
    return json.dumps({"segments": rows}, ensure_ascii=False, indent=2)


def _rewrite_prompt(rows: List[Dict[str, Any]], *,
                    language: str,
                    source_language: str) -> List[Dict[str, str]]:
    return [
        {"role": "system",
         "content": _system_prompt(language, source_language)},
        {"role": "user", "content": _user_payload(rows)},
    ]


def _max_tokens(n: int) -> int:
    """Output budget for a chunk of `n` segments.
    """
    return min(8192, max(2048, 900 * max(1, n)))


def _chunks(items: List[Any], size: int) -> List[List[Any]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


class _BatchMismatch(ValueError):
    """The reply cannot be mapped onto the segments that were asked about.
    """


def _parse_batch(reply: Any, expected: List[str]) -> Dict[str, Dict[str, Any]]:
    """Map a batch reply to `{segment_id: row}`.

    Accepts `{"segments": [...]}` (what the prompt asks for), a bare list,
    and -- only for a single-segment ask -- one flat object.

    Matching is by the echoed `segment_id`. Positional matching is allowed
    only when no row carries an id and the count matches exactly; anything
    else raises, because a half-labelled reply is where a wrong guess is
    both easy and invisible.
    """
    rows: Any = reply
    if isinstance(reply, dict):
        if isinstance(reply.get("segments"), list):
            rows = reply["segments"]
        elif len(expected) == 1:
            rows = [reply]
        else:
            raise _BatchMismatch("no `segments` array in the reply")
    if not isinstance(rows, list):
        raise _BatchMismatch(f"expected a list of segments, got {type(rows).__name__}")
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        raise _BatchMismatch("no segment objects in the reply")

    ids = [str(r.get("segment_id") or "") for r in rows]
    if any(ids):
        if not all(ids):
            raise _BatchMismatch("some rows carry no segment_id")
        if len(set(ids)) != len(ids):
            raise _BatchMismatch("duplicate segment_id in the reply")
        unknown = [i for i in ids if i not in set(expected)]
        if unknown:
            raise _BatchMismatch(f"reply names segments that were not asked about: {unknown}")
        return dict(zip(ids, rows))

    # No ids anywhere. Position is the only signal left, and it is only
    # trustworthy when the count is exactly right.
    if len(rows) != len(expected):
        raise _BatchMismatch(
            f"{len(rows)} unlabelled row(s) for {len(expected)} segment(s)")
    return dict(zip(expected, rows))


#: Fields the model reports about its own rewrite. Recorded for QC only --
#: `duration.predict` decides whether the rewrite is kept.
_QC_FIELDS = ("fits", "infeasible", "reduction_pct", "syllables",
              "mt_syllables", "techniques", "meaning_loss", "adequacy",
              "adequacy_issues", "preserved_terms", "asr_suspect", "note")


def _qc_of(reply: Dict[str, Any]) -> Dict[str, Any]:
    return {k: reply[k] for k in _QC_FIELDS if k in reply}


def _seg_text(seg: Dict[str, Any]) -> str:
    for k in ("final_text", "mt_text", "text"):
        v = seg.get(k)
        if v and str(v).strip():
            return str(v)
    return ""


def _set_seg_text(seg: Dict[str, Any], text: str) -> None:
    # Write every key the segment uses for its dub text. `tts.py` reads
    # `final_text` when present and falls back to `mt_text`; updating only
    # one leaves the other stale.
    for k in ("final_text", "mt_text"):
        if k in seg:
            seg[k] = text
    seg.setdefault("final_text", text)


class _State:
    """One overlong segment, carried across rounds.

    A round asks about every segment still over budget, so each segment's
    best-so-far cannot live in a local variable.
    """

    __slots__ = ("seg", "sid", "original", "target_s", "est0",
                 "best_text", "best_est", "qc", "infeasible", "done")

    def __init__(self, seg: Dict[str, Any]):
        self.seg = seg
        # The same expression `_segment_payload` uses, so this id is the one
        # the model is asked to echo back.
        self.sid = str(seg.get("id", ""))
        self.original = _seg_text(seg)
        self.target_s = float(seg["target_duration_s"])
        self.est0 = float(seg["est_duration_s"])
        self.best_text = self.original
        self.best_est = self.est0
        self.qc: Dict[str, Any] = {}
        self.infeasible = False
        self.done = False


def condense_overlong(
    mt_json: Path,
    *,
    language: str,
    source_language: Optional[str] = None,
    voice: str = "default",
    cache_dir: Optional[Path] = None,
    client: Any = None,
    model: Optional[str] = None,
    speed_cap: float = DEFAULT_SPEED_CAP,
    tolerance: float = DEFAULT_TOLERANCE,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Rewrite segments the tempo stretch cannot fit. Returns a summary.

    A segment is left alone while `atempo` can absorb it, and rewritten only
    once it needs more than `speed_cap` -- the point at which the stretch
    clamps and the dub runs past the picture.

    `batch_size` segments go in one request (default 2, or
    `CONDENSE_BATCH_SIZE`); the chunk is `min(batch_size, overlong)`. A
    batch that fails or comes back unmappable is re-asked one segment at a
    time, so batching can cost an extra call but never a segment.

    `source_language` names what is SPOKEN in the source, for the prompt's
    own wording and for the `original_language_asr` field it reads. Left
    unset it comes from the merged JSON's own `language`, which is what ASR
    detected or was pinned to -- so an auto-detected non-English source is
    described to the model correctly without the caller doing anything.

    Requires `plan_segment_durations` (stage 9c) to have run: it reads the
    `target_duration_s` and `est_duration_s` that stage writes.
    """
    mt_json = Path(mt_json)
    batch = max(1, int(batch_size or _env_int("CONDENSE_BATCH_SIZE",
                                              DEFAULT_BATCH_SIZE)))
    data = json.loads(mt_json.read_text())
    src_name = _language_name(source_language or data.get("language"),
                              "the source language")
    segs = data.get("segments") or []
    summary: Dict[str, Any] = {
        "model": None, "considered": 0, "rewritten": 0, "rejected": 0,
        "failed": 0, "saved_s": 0.0, "segments": [],
        "speed_cap": speed_cap, "infeasible": 0,
        # `calls` is the checkable form of "did batching happen": with a
        # batch of 2 it should be about half `considered`.
        "batch_size": batch, "calls": 0,
    }
    if not segs:
        return summary

    # One formula, used both to decide what to rewrite and to decide when a
    # rewrite is done, so the two cannot disagree about what "fits" means.
    def limit_for(tgt: float) -> float:
        return tgt * speed_cap * (1.0 + tolerance)

    # Which ones overrun? Decided BEFORE any call, so the cost is known.
    todo = []
    for seg in segs:
        tgt = seg.get("target_duration_s")
        est = seg.get("est_duration_s")
        if not tgt or not est or tgt <= 0:
            continue
        if est > limit_for(float(tgt)):
            todo.append(seg)
    summary["considered"] = len(todo)
    if not todo:
        log.info("condense: every segment fits within %.2fx speed-up; "
                 "nothing to rewrite", speed_cap)
        return summary

    if client is None:
        try:
            from backend.common.llm import OpenRouterClient
            client = OpenRouterClient(model=model or DEFAULT_MODEL)
        except Exception as e:                           # noqa: BLE001
            log.warning("condense: no LLM client (%s); leaving the text as is", e)
            summary["error"] = str(e)
            return summary
    if not client.available():
        log.warning("condense: OPENROUTER_API_KEY unset; %d overlong "
                    "segment(s) left as they are", len(todo))
        summary["error"] = "no api key"
        return summary
    summary["model"] = getattr(client, "model", model or DEFAULT_MODEL)

    from .duration import load_or_default_model
    dmodel = load_or_default_model(
        Path(cache_dir) if cache_dir else Path("."),
        language=language, voice=voice)

    log.info("condense: %d of %d segment(s) need more than %.2fx speed-up; "
             "model=%s batch=%d", len(todo), len(segs), speed_cap,
             summary["model"], batch)

    states = [_State(seg) for seg in todo]

    def _accept(st: _State, row: Optional[Dict[str, Any]], round_: int) -> None:
        """Score one returned row against the segment it belongs to."""
        row = row if isinstance(row, dict) else {}
        cand = re.sub(r"\s+", " ", str(row.get("text") or "")).strip()
        if not cand:
            log.warning("condense[%s]: round %d returned no text",
                        st.sid or "?", round_ + 1)
            st.done = True
            return

        st.qc = _qc_of(row)
        # The model's own verdict that the required cut exceeds 40%.
        # Asking again cannot make an incompressible line compressible. Its
        # text is still taken if it measures shorter.
        st.infeasible = bool(row.get("infeasible"))

        cand_est = dmodel.predict(cand)
        # Verify: a model asked for brevity will happily return something
        # longer, and only the measurement knows.
        if cand_est >= st.best_est:
            log.info("condense[%s]: round %d was not shorter "
                     "(%.2fs vs %.2fs); keeping the previous text",
                     st.sid or "?", round_ + 1, cand_est, st.best_est)
            summary["rejected"] += 1
            st.done = True
            return
        st.best_text, st.best_est = cand, cand_est
        if st.best_est <= limit_for(st.target_s) or st.infeasible:
            st.done = True                     # the stretch can fit it now

    def _ask(chunk: List[_State], round_: int) -> List[_State]:
        """One request for one chunk. Returns the states it did not resolve.

        A chunk of one always resolves -- it is the fallback, so handing
        back an unresolved single would loop.
        """
        single = len(chunk) == 1
        rows = [_segment_payload(st.seg, text=st.best_text, est_s=st.best_est,
                                 target_s=st.target_s, speed_cap=speed_cap)
                for st in chunk]
        summary["calls"] += 1
        try:
            reply = client.chat_json(
                _rewrite_prompt(rows, language=language,
                                source_language=src_name),
                temperature=0.2,
                # Scaled: the client's 2048 default truncates a
                # multi-segment reply into unparseable JSON.
                max_tokens=_max_tokens(len(chunk)),
                trace_name="condense.rewrite",
            )
        except Exception as e:                           # noqa: BLE001
            if not single:
                log.warning("condense: a batch of %d failed on round %d (%s); "
                            "re-asking one at a time", len(chunk), round_ + 1, e)
                return list(chunk)
            log.warning("condense[%s]: round %d failed (%s)",
                        chunk[0].sid or "?", round_ + 1, e)
            summary["failed"] += 1
            chunk[0].done = True
            return []

        try:
            by_id = _parse_batch(reply, [st.sid for st in chunk])
        except _BatchMismatch as e:
            # Never guessed at: a rewrite pasted onto the wrong segment is
            # short, fluent and wrong, and the duration check cannot see it.
            if not single:
                log.warning("condense: unusable batch reply on round %d (%s); "
                            "re-asking %d segment(s) one at a time",
                            round_ + 1, e, len(chunk))
                return list(chunk)
            log.warning("condense[%s]: unusable reply (%s)",
                        chunk[0].sid or "?", e)
            by_id = {}

        missing = [st for st in chunk if st.sid not in by_id]
        if missing and not single:
            log.warning("condense: the reply omitted %d of %d segment(s); "
                        "re-asking them one at a time", len(missing), len(chunk))
        for st in chunk:
            row = by_id.get(st.sid)
            if row is None and not single:
                continue                       # handed back as unresolved
            _accept(st, row, round_)
        return [] if single else missing

    pending = list(states)
    for round_ in range(max_rounds):
        if not pending:
            break
        for chunk in _chunks(pending, batch):
            unresolved = _ask(chunk, round_)
            for st in unresolved:
                _ask([st], round_)
        pending = [st for st in pending if not st.done]

    for st in states:
        sid, qc = st.sid or "?", st.qc
        tgt, est = st.target_s, st.est0
        # Reported whatever happened to the text: a `major_issue` on a line
        # that was not rewritten still means the translation has a problem.
        if qc.get("adequacy") == "major_issue" or qc.get("asr_suspect"):
            log.warning("condense[%s]: adequacy=%s asr_suspect=%s %s",
                        sid, qc.get("adequacy"), qc.get("asr_suspect"),
                        qc.get("adequacy_issues") or "")
            summary.setdefault("flagged", []).append(
                {"id": sid, **{k: qc[k] for k in
                               ("adequacy", "adequacy_issues", "asr_suspect",
                                "note") if k in qc}})

        if st.best_text != st.original:
            seg = st.seg
            seg.setdefault("condense_backup", {})["text"] = st.original
            seg["condense_backup"]["est_duration_s"] = est
            _set_seg_text(seg, st.best_text)
            seg["est_duration_s"] = st.best_est
            seg.setdefault("qc", {})["condensed"] = True
            summary["rewritten"] += 1
            if st.infeasible:
                summary["infeasible"] += 1
            summary["saved_s"] += est - st.best_est
            summary["segments"].append({
                "id": sid, "target_s": round(tgt, 2), "qc": qc,
                "before_s": round(est, 2), "after_s": round(st.best_est, 2),
                # Against the speed cap, not the raw slot: a line the
                # stretch can reach is a line that fits.
                "rate_before": round(est / tgt, 2),
                "rate_after": round(st.best_est / tgt, 2),
                "fits": bool(st.best_est <= limit_for(tgt)),
                "before_chars": len(st.original),
                "after_chars": len(st.best_text),
            })
            log.info("condense[%s]: %.2fs -> %.2fs (slot %.2fs, "
                     "%.2fx -> %.2fx)%s%s",
                     sid, est, st.best_est, tgt, est / tgt, st.best_est / tgt,
                     "" if st.best_est <= limit_for(tgt) else "  still over cap",
                     "  [model: infeasible]" if st.infeasible else "")

    summary["saved_s"] = round(summary["saved_s"], 2)
    data.setdefault("extra", {})["condense"] = summary
    mt_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from backend.pipeline.mt_versions import snapshot as _mt_snapshot
    _mt_snapshot(mt_json, "condense", data=data)
    log.info("condense: rewrote %d/%d segment(s), saved %.1fs of speech "
             "in %d call(s)", summary["rewritten"], summary["considered"],
             summary["saved_s"], summary["calls"])
    return summary
