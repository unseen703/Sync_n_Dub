"""Shorten a translation that will not fit the time it has.

Stage 9c2, between duration planning (9c) and TTS (9e).

The problem it solves, measured on the 2026-10-21 svwrTOm55y4 run: 50.9 s
of Hindi had to fit 30.6 s of slots -- 1.66x -- against a 1.5x tempo cap.
Three of five segments were clamped and the dub ran ~8 s long. Nothing
downstream can fix that. `atempo` was already at its limit, and Bodhan's
TTS has no speed parameter at all (checked against their API docs on
2026-10-22: the endpoint takes `model`, `input`, `voice` and
`instructions` and nothing else). The only remaining lever is to make the
sentence shorter, which is what this does.

**It edits only what the tempo stretch cannot reach.** Changed 2026-10-22
by request: the gate is now `est > target * speed_cap` (1.5 by default),
not a flat 10% overrun. A line needing 1.2x is left alone because `atempo`
fits it perfectly well; only a line past the cap, where the stretch clamps
and the dub runs long, is worth a call. The old gate rewrote 4 of the 5
segments on the svwrTOm55y4 run where only 3 ever clamped, and rewriting
text that already renders correctly is a translation risk for nothing.

**It rewrites `mt_text`, NOT the English, and the English is context.**
Corrected 2026-10-22 against the author's reference prompt
(`translate_agetn.prompt.txt`), which states it outright: "Rewrite
`mt_text`, not `english_asr`, unless the MT is unusable. Use `english_asr`
to verify the intended meaning, resolve mistranslations, and identify
possible ASR errors." An earlier revision of this module retranslated from
the source as its primary path, which throws away a translation that is
usually correct and merely long, and re-opens every decision the MT already
made about register and code-mixing.

**The prompt is the author's, verbatim**, with `{{target_language}}` filled
in — see `_SYSTEM_TEMPLATE`. It carries its own syllable arithmetic, a
cutting strategy ordered cheapest-first, the non-negotiable preservation
rules (negation, modality, numbers, protected terms) and a QC schema.

**Its metrics are reported, not trusted.** The prompt says so itself --
"The duration estimate is a proxy, not a guarantee" -- so `est_duration_s`
as returned by the model is recorded for the log and the real decision is
still made by `backend/duration.py` re-scoring the returned text. That is
the one property this stage cannot give up.

`infeasible` (the model's own "required cut exceeds 40%") ENDS the rounds
for that segment. A second ask cannot make a line compressible that the
first correctly judged incompressible, and the prompt is explicit that the
remainder is the pipeline's problem, not the rewriter's.

**It verifies the result rather than trusting it.** The same duration
model that flagged the segment re-scores the rewrite, and a rewrite that
is not actually shorter is DISCARDED. An LLM asked for brevity will
cheerfully return something longer, and the only way to know is to measure.

**Segments go up in batches, and the batch is deliberately small.** Added
2026-10-23. The system prompt is the author's 10.7 KB reference file --
~2.9k tokens -- and it was re-sent once per segment, so N overlong lines
paid for it N times. `batch_size` (default **2**, `CONDENSE_BATCH_SIZE`,
`--condense-batch-size`) sends several segments in one request; the chunk
is `min(batch_size, overlong)`, so the ordinary one-segment case is
unchanged.

**2 rather than 5 because the dangerous failure is invisible.** A dropped
or duplicated segment is caught by `_parse_batch`, and a rewrite that is
too long is caught by the duration re-score. What neither can see is a
line that borrowed its neighbour's wording: it is short, fluent, and in
the wrong place. The prompt now forbids it explicitly ("Segment isolation
and identity"), but a model attends less to each line the more of them it
holds, so the batch stays small until a run has been read against its
source. `batch_size=1` is exactly the pre-batch behaviour.

**Mapping is by the echoed `segment_id`, and a reply that cannot be mapped
is re-asked one segment at a time** rather than matched by guesswork --
positional matching is allowed only when NO row carries an id and the
count is exactly right. So batching can cost an extra call; it cannot
cost a segment.

**Every failure keeps the original text.** No key, a dead model slug, bad
JSON, a refusal -- each falls back to what was already there. A shortening
pass is an optimisation; it must never be the reason a dub fails.

The original wording is kept in `seg["condense_backup"]`, mirroring how
`backend/edits.py` makes its changes revertible.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

#: Model for the rewrite. A small fast one is right for this: the task is
#: "say this shorter in the same language", not reasoning.
#:
#: Set to `google/gemini-3.1-flash-lite` on 2026-10-22, by the author, who
#: named the slug. It replaces `google/gemini-2.5-flash-lite`.
#:
#: **Still unconfirmed against the live catalogue from here.** Two attempts
#: to read OpenRouter's model list on 2026-10-22 returned a malformed id
#: (`anthropic/google/gemini-3.8-flash[1m]`) and then a page with no Gemini
#: rows at all, so no slug in this family has been checked from this host.
#: CLAUDE.md records what a wrong one costs: `gemma-3b` was invented and no
#: such model exists. Confirm with
#:   curl -s https://openrouter.ai/api/v1/models -H "Authorization: Bearer $OPENROUTER_API_KEY" \
#:     | python -c "import json,sys;[print(m['id']) for m in json.load(sys.stdin)['data'] if 'gemini' in m['id']]"
#: and set CONDENSE_MODEL if it prints something different.
#:
#: A wrong slug degrades to "no shortening" -- the call fails, the original
#: text is kept, and the dub still renders. It does not break a run, which
#: is also why it can be wrong for a long time without anyone noticing:
#: `extra.condense.error` and the Langfuse trace are where it shows.
DEFAULT_MODEL = os.getenv("CONDENSE_MODEL") or "google/gemini-3.1-flash-lite"

#: The speed-up the mix can actually apply. A segment is left ALONE while
#: `atempo` can still absorb it, and is only rewritten once it needs more
#: than this — the point at which the stretch clamps and the dub runs long.
#:
#: Changed 2026-10-22 by request, from a flat 0.10 overrun. The old gate
#: fired at 1.10x, which spent an LLM call rewriting lines the tempo stretch
#: would have fitted perfectly well: on the svwrTOm55y4 run that is 4 of 5
#: segments rewritten where only 3 ever clamped. Rewriting text that already
#: renders correctly is a translation risk taken for nothing.
#:
#: **1.5, not the 1.8 `--speed-cap` now defaults to.** They are different
#: questions and were asked separately: 1.8 is how hard the mixer is ALLOWED
#: to push, 1.5 is the point past which the author would rather re-word the
#: line than listen to it. Hindi TTS already runs 1.21-1.65x longer than the
#: English it covers, so a gate at 1.8 would almost never fire. Raise
#: `--condense-speed-cap` to 1.8 to make the two agree.
DEFAULT_SPEED_CAP = 1.5

#: Slack ABOVE the cap before a rewrite is worth a call, and the epsilon on
#: "does it fit now". 0.0 means the cap is the cap.
DEFAULT_TOLERANCE = 0.0

#: Give up on a segment after this many rewrites. Each round costs a call
#: and the returns fall off fast -- a model that cannot shorten a sentence
#: in two tries is not going to manage it in five.
DEFAULT_MAX_ROUNDS = 2

#: Segments per request. The system prompt is 10.7 KB -- ~2.9k tokens --
#: and it used to be re-sent once per segment, so N overlong lines paid for
#: it N times. Batching sends it once per chunk instead.
#:
#: **2, deliberately low.** A model handling several lines at once attends
#: less to each, and the failure that matters is not a dropped segment (the
#: parser catches that) but a plausible rewrite that borrowed wording from
#: its neighbour -- which `duration.predict` cannot see, because the wrong
#: line can be perfectly short. 2 halves the prompt cost and keeps the
#: model's attention almost undivided; raise it once a run has been read
#: side by side with its source. `1` is exactly the old behaviour.
#:
#: The chunk is `min(batch_size, len(todo))`, so a single overlong segment
#: is still a single-segment call and the common case is unchanged.
DEFAULT_BATCH_SIZE = 2


def _env_int(name: str, default: int) -> int:
    """Read an int env var, tolerating the blank a copied template leaves.

    Same shape as `llm._env_num`: `int(os.getenv("X", 2))` raises on `X=`,
    which turns a harmlessly empty line in .env into a crash at import.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        log.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default

#: The author's reference prompt, verbatim from `translate_agetn.prompt.txt`
#: with `{{target_language}}` substituted at call time. It is kept as one
#: string rather than assembled from parts so it can be diffed against that
#: file; `tests/test_condense.py` asserts the two have not drifted.
_SYSTEM_TEMPLATE = (Path(__file__).resolve().parent.parent
                    / "translate_agetn.prompt.txt")

#: Fallback if the file is missing — a deployment that lost it should still
#: dub. Deliberately terse: the real prompt is the file, and a silently
#: substituted "good enough" version is how two prompts end up in a repo.
_FALLBACK_SYSTEM = """You are a bilingual dubbing QC reviewer for an English -> {{target_language}} dubbing pipeline. Shorten machine-translated lines so each one fits its own audio slot, preserving meaning, tone and protected terminology.

You receive a `segments` array of one or more segment objects, each with `segment_id`, `english_asr`, `mt_text`, `target_duration_s`, `est_duration_s`, `protected_terms` and `stretch_cap_pct`.

Each segment is an independent rewrite. Use only that segment's own fields. Never move, borrow or blend wording between segments, and never use one segment as context for another. Echo `segment_id` exactly. Return exactly one output object per input segment, in input order, with the same count -- never merged, split, omitted or added.

Rewrite `mt_text`, not `english_asr`. Use `english_asr` only to verify meaning. Preserve negation, modality, conditions, named entities, numbers, units, dates and every protected term exactly. Never add facts. Never pad to fill the slot. Keep natural spoken {{target_language}}, including code-mixing.

Return exactly one JSON object: {"segments": [{"segment_id": "<echo>", "text": "<rewritten line>", "fits": false, "infeasible": false, "meaning_loss": "none", "adequacy": "ok", "adequacy_issues": [], "preserved_terms": [], "asr_suspect": false, "note": null}]}"""


def _system_prompt(language: str) -> str:
    """The reference prompt with the target language filled in."""
    try:
        raw = _SYSTEM_TEMPLATE.read_text()
    except OSError as e:                                  # noqa: BLE001
        log.warning("condense: %s unreadable (%s); using the short fallback "
                    "prompt", _SYSTEM_TEMPLATE.name, e)
        raw = _FALLBACK_SYSTEM
    return raw.replace("{{target_language}}", language)


def _segment_payload(seg: Dict[str, Any], *, text: str, est_s: float,
                     target_s: float, speed_cap: float) -> Dict[str, Any]:
    """One segment as the prompt's own input contract describes it.

    JSON rather than prose because the prompt names its fields
    (`english_asr`, `mt_text`, `target_duration_s`, ...) and reads them back
    in its arithmetic. It is also DATA by construction: `english_asr` is ASR
    output of the speaker, the one field here that did not originate inside
    this pipeline.

    `stretch_cap_pct` is derived from the SPEED CAP, not left at the
    prompt's default of 10. The prompt computes `fits` as
    `est <= target * (1 + stretch_cap_pct/100)`, and on this pipeline a line
    fits when the mix can stretch it -- 1.5x, i.e. 50%. Leaving it at 10
    would have the model report `fits: false` for lines that render
    perfectly well, and then cut them far harder than necessary.
    """
    protected = [sp["text"] for sp in (seg.get("retain_spans") or [])
                 if isinstance(sp, dict) and sp.get("text")]
    return {
        "segment_id": str(seg.get("id", "")),
        "english_asr": str(seg.get("text") or ""),
        "mt_text": text,
        "target_duration_s": round(float(target_s), 2),
        "est_duration_s": round(float(est_s), 2),
        "protected_terms": protected,
        "stretch_cap_pct": round((speed_cap - 1.0) * 100.0),
    }


def _user_payload(rows: List[Dict[str, Any]]) -> str:
    """The chunk, in the `{"segments": [...]}` shape the prompt declares.

    Always an ARRAY, even for one segment. A request whose shape changes
    with the batch size is two contracts, and the one that runs rarely is
    the one that breaks unnoticed.
    """
    return json.dumps({"segments": rows}, ensure_ascii=False, indent=2)


def _rewrite_prompt(rows: List[Dict[str, Any]], *,
                    language: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt(language)},
        {"role": "user", "content": _user_payload(rows)},
    ]


def _max_tokens(n: int) -> int:
    """Output budget for a chunk of `n` segments.

    The client's default is 2048, which is enough for one Hindi line plus
    the 16-key QC schema and not enough for several: Devanagari costs 2-3
    tokens per akshara, so a truncated reply is unparseable JSON and the
    whole chunk is lost. The floor keeps a small batch at today's budget --
    a bare `per_segment * n` would have LOWERED the cap at n=2.
    """
    return min(8192, max(2048, 900 * max(1, n)))


def _chunks(items: List[Any], size: int) -> List[List[Any]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


class _BatchMismatch(ValueError):
    """The reply cannot be mapped onto the segments that were asked about.

    Raised rather than guessed at. Pasting segment A's rewrite onto segment
    B produces a short, fluent, plausible line in the wrong place -- and
    `duration.predict` cannot catch it, because the wrong line measures
    perfectly well. The caller re-asks the chunk one segment at a time.
    """


def _parse_batch(reply: Any, expected: List[str]) -> Dict[str, Dict[str, Any]]:
    """Map a batch reply to `{segment_id: row}`.

    Accepts `{"segments": [...]}` (what the prompt asks for), a bare list,
    and -- only for a single-segment ask -- one flat object, which is what
    the pre-batch prompt returned and what a model drops back to.

    Matching is by the ECHOED `segment_id`. Positional matching is allowed
    only when NO row carries an id and the count matches exactly; anything
    else raises, because a half-labelled reply is the case where a wrong
    guess is both easy and invisible.
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


#: Fields the model reports about its own rewrite. Recorded for QC; none of
#: them decides whether the rewrite is kept -- `duration.predict` does that.
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
    # Write every key the segment already uses for its dub text. `tts.py`
    # reads `final_text` when present and falls back to `mt_text`; updating
    # only one leaves the other stale and whichever is read first wins.
    for k in ("final_text", "mt_text"):
        if k in seg:
            seg[k] = text
    seg.setdefault("final_text", text)


class _State:
    """One overlong segment, carried across rounds.

    Rounds used to be the inner loop of a per-segment outer loop. With a
    batch they invert — a round asks about every segment still over its
    budget — so each segment's best-so-far has to live somewhere other than
    a local variable.
    """

    __slots__ = ("seg", "sid", "original", "target_s", "est0",
                 "best_text", "best_est", "qc", "infeasible", "done")

    def __init__(self, seg: Dict[str, Any]):
        self.seg = seg
        # The same expression `_segment_payload` uses, so the id this is
        # keyed by is the id the model is asked to echo back.
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

    A segment is left alone while `atempo` can still absorb it. It is only
    rewritten once it needs more than `speed_cap` — the point at which the
    stretch clamps and the dub runs past the picture.

    `batch_size` segments go in one request (default 2, or
    `CONDENSE_BATCH_SIZE`); the chunk is `min(batch_size, overlong)`, so one
    overlong line is still one call. A batch that fails or comes back
    unmappable is re-asked one segment at a time, so batching can cost an
    extra call but never a segment.

    Requires `plan_segment_durations` to have run: it reads the
    `target_duration_s` and `est_duration_s` that stage 9c writes.
    """
    mt_json = Path(mt_json)
    batch = max(1, int(batch_size or _env_int("CONDENSE_BATCH_SIZE",
                                              DEFAULT_BATCH_SIZE)))
    data = json.loads(mt_json.read_text())
    segs = data.get("segments") or []
    summary: Dict[str, Any] = {
        "model": None, "considered": 0, "rewritten": 0, "rejected": 0,
        "failed": 0, "saved_s": 0.0, "segments": [],
        "speed_cap": speed_cap, "infeasible": 0,
        # `calls` is the checkable form of "did batching happen": with a
        # batch of 2 it should be about half `considered`, and equal to it
        # when every batch had to be re-asked singly.
        "batch_size": batch, "calls": 0,
    }
    if not segs:
        return summary

    # ONE formula, used to decide what to rewrite and to decide when a
    # rewrite is done. The speed-up the mixer can apply is what both mean by
    # "fits", so they cannot drift into disagreeing about it.
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
            from .llm import OpenRouterClient
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
        # The model's own verdict that the required cut exceeds 40% and
        # cannot be made without breaking a preservation rule. Asking again
        # cannot make an incompressible line compressible, and the prompt is
        # explicit that the remainder is the pipeline's problem. Its text is
        # still taken if it measures shorter.
        st.infeasible = bool(row.get("infeasible"))

        cand_est = dmodel.predict(cand)
        # VERIFY. A model asked for brevity will happily return something
        # longer, and only the measurement knows. The prompt says the same
        # of its own number: "a proxy, not a guarantee".
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

        A chunk of one always resolves — it is the fallback, and handing
        back an unresolved single would loop. Anything the caller gets back
        is therefore a batch that has to be re-asked segment by segment.
        """
        single = len(chunk) == 1
        rows = [_segment_payload(st.seg, text=st.best_text, est_s=st.best_est,
                                 target_s=st.target_s, speed_cap=speed_cap)
                for st in chunk]
        summary["calls"] += 1
        try:
            reply = client.chat_json(
                _rewrite_prompt(rows, language=language),
                temperature=0.2,
                # Scaled: the client's 2048 default truncates a multi-segment
                # reply into unparseable JSON, which loses the whole chunk.
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
            # Never guessed at. A rewrite pasted onto the wrong segment is
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
        # Reported whatever happened to the text, because a `major_issue` on
        # a line that was NOT rewritten is still the interesting signal --
        # it means the translation has a problem the length pass noticed and
        # could not fix.
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
                # Against the SPEED CAP, not against the raw slot: a line the
                # stretch can reach is a line that fits, and reporting it as
                # "still over" would send someone chasing a solved problem.
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
    from .mt_versions import snapshot as _mt_snapshot
    _mt_snapshot(mt_json, "condense", data=data)
    log.info("condense: rewrote %d/%d segment(s), saved %.1fs of speech "
             "in %d call(s)", summary["rewritten"], summary["considered"],
             summary["saved_s"], summary["calls"])
    return summary
