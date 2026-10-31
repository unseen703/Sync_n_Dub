"""Typed edits over a finished dub, and the stages each one invalidates.

This is the deterministic half of the NLQ edit agent: no LLM, no network, no
guessing. `backend.edit_agent` turns a sentence into `EditOp`s; everything
here takes those ops, resolves them against a run's merged ASR+MT JSON,
mutates the JSON, and reports which pipeline stages have to be re-run.

Why the merged JSON is the edit surface
---------------------------------------
Stages 9b..9h (`backend.pipeline.run_post_mt_stages`) all read and write one
merged JSON in place, and `run_pipeline(post_mt_only_from=…)` replays that
chain without touching the hours of upstream work. So an edit is: change
fields in that file, then replay from the earliest stage the change
invalidated. Nothing re-downloads, re-diarizes, re-transcribes or re-detects
faces.

Fields written, and who reads them back
---------------------------------------
  mt_text, final_text      `backend.tts.synthesize_segments` (final_text wins)
  tts_voice                per-segment voice override      (added for this)
  tts_style                per-segment engine style hint   (added for this)
  tts_rate_override        per-segment synthesis rate      (added for this)
  tts_max_stretch          per-segment stretch cap         (added for this)
  target_duration_s        the slot the clip is stretched into
  edit_backup              pre-edit value of every key this module overwrote

`edit_backup` is what makes `revert()` possible, and it is written once per
key — a second edit to the same field does not overwrite the original.

Stage invalidation
------------------
`STAGE_ORDER` is the post-MT chain in execution order. Every op declares the
earliest stage it dirties; the rerun set is that stage and everything after
it, intersected with the stages the job was actually configured with. A text
edit therefore starts at TTS, not at code-mix, so a literal replacement the
user typed is never re-written by the transliterator behind their back.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

EDITS_VERSION = 1


def _lipsync_model_ids() -> str:
    """The registered lip-sync model names, for the op catalogue.

    Read from the registry rather than typed out. The hand-written list said
    "wav2lip | musetalk | latentsync" while five adapters were registered, so
    the LLM was never told `wav2lip_hd` or `lipgan` could be asked for — and
    an op catalogue that under-reports is indistinguishable, from the model's
    side, from the feature not existing.

    Falls back to the literal set if the registry will not import: this is a
    prompt string, and a failed import here must not take down an edit that
    has nothing to do with lip-sync.
    """
    try:
        from backend.lipsync.lipsync import available_models
        return " | ".join(available_models())
    except Exception:                               # noqa: BLE001
        return "wav2lip | musetalk | latentsync15"


def tone_styles() -> Tuple[str, ...]:
    """The delivery styles the TTS engine actually accepts, plus `neutral`.

    Read from `models.tts.bodhan_adapter.INDIC_SPEAK_STYLES`, for exactly the
    reason `_lipsync_model_ids` is read from the lip-sync registry. The op
    catalogue, the system prompt and the regex parser all used to carry a
    hand-written list -- "formal | casual | friendly | serious | energetic" --
    and NONE of those five is an engine value. A tone edit through the agent
    therefore wrote `tts_style="formal"`, which Bodhan does not know, so the
    one op whose whole purpose is to change delivery changed nothing. The
    speaker dropdown never had the bug because `server.service.speakers.tone_vocabulary`
    already read the engine; this is the same read, placed where `backend/`
    can use it without importing `server/`.

    `NEUTRAL_STYLE` leads the list: it is not an engine value but a sentinel
    `models.tts.resolve_style` turns back into "send no style".
    """
    try:
        from backend.tts import NEUTRAL_STYLE
        from backend.tts.bodhan_adapter import INDIC_SPEAK_STYLES
        return (NEUTRAL_STYLE,) + tuple(INDIC_SPEAK_STYLES)
    except Exception:                               # noqa: BLE001
        # Same policy as above: a prompt string must not take down an edit.
        return ("neutral", "happy", "sad", "news")


def _tone_style_ids() -> str:
    return " | ".join(tone_styles())


def _voice_ids() -> str:
    """Flat list of every voice id in the indic-speak catalog.

    Pulled from `backend.tts.voices.DEFAULT_INDIC_SPEAK` so the prompt and the
    catalog cannot drift. Returned as a comma-separated string; the planner
    prompt substitutes it into `{{voice_ids}}`.
    """
    from backend.tts.voices import DEFAULT_INDIC_SPEAK
    return ", ".join(v.voice for v in DEFAULT_INDIC_SPEAK)


#: The delivery styles the EDIT WORKSPACE offers, narrowed on purpose.
#:
#: `tone_styles()` is the full set the engine accepts — fifteen, most of them
#: emotions (anger, fear, disgust) that belong to a performance, not to a
#: lecture recording being dubbed. Offering all fifteen in a chip list made
#: the common choice hard to find and invited a register the material does
#: not support. These two are the formats this pipeline's material actually
#: comes in.
#:
#: It narrows what is OFFERED, never what is ACCEPTED: typing "make it sad"
#: still parses and still applies, because `_h_set_tone` validates against
#: `tone_styles()` and the regex matches every engine style. If the offered
#: pair ever needs a third, add it here — one list, served to the frontend.
TONE_CHOICES: Tuple[str, ...] = ("TV style news", "educational lecture")

#: Post-MT chain, in execution order. See `run_post_mt_stages`.
STAGE_ORDER: Tuple[str, ...] = (
    "codemix", "duration", "voices", "tts", "timing", "mix", "lipsync",
)

#: Default stretch cap in `backend.tts`. A speed edit past this gets its own
#: per-segment cap, or the stretch is silently clamped and the "+30% faster"
#: the user asked for arrives as +10% with a budget_violation flag.
_DEFAULT_MAX_STRETCH = 0.10


class EditError(ValueError):
    """An op that cannot be applied as written."""


# ---------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------

@dataclass
class Scope:
    """Which segments an op touches.

    Exactly one of these is normally set, and they AND together when several
    are: `{speaker: SPEAKER_01, start_s: 60, end_s: 90}` is "that speaker,
    within that window".

    A window selects every segment that OVERLAPS it, not only those contained
    in it — "speed up between 0:30 and 0:45" must catch a cue that straddles
    0:44, or the edit audibly stops mid-sentence.

    `start_s` with no `end_s` is a POINT: the segment containing it, or, when
    it falls in a gap, the nearest one. That is what "change the gender at
    1:20" means.
    """
    segments: List[str] = field(default_factory=list)
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    speaker: Optional[str] = None
    all: bool = False

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Scope":
        d = dict(d or {})
        segs = d.get("segments") or d.get("segment_ids") or []
        if isinstance(segs, str):
            segs = [segs]
        return cls(
            segments=[str(s) for s in segs],
            start_s=_as_time(d.get("start_s", d.get("start"))),
            end_s=_as_time(d.get("end_s", d.get("end"))),
            speaker=(str(d["speaker"]) if d.get("speaker") else None),
            all=bool(d.get("all", False)),
        )

    def is_empty(self) -> bool:
        return not (self.segments or self.speaker or self.all
                    or self.start_s is not None or self.end_s is not None)

    def describe(self) -> str:
        bits = []
        if self.all:
            bits.append("whole dub")
        if self.segments:
            bits.append(f"segments {', '.join(self.segments)}")
        if self.speaker:
            bits.append(f"speaker {self.speaker}")
        if self.start_s is not None and self.end_s is not None:
            bits.append(f"{_hhmmss(self.start_s)}–{_hhmmss(self.end_s)}")
        elif self.start_s is not None:
            bits.append(f"at {_hhmmss(self.start_s)}")
        elif self.end_s is not None:
            bits.append(f"up to {_hhmmss(self.end_s)}")
        return ", ".join(bits) or "unscoped"

    def resolve(self, data: Dict[str, Any]) -> List[str]:
        """Return the ids of the segments this scope selects, in time order."""
        segs = data.get("segments") or []
        if not segs:
            return []
        by_id = {str(s.get("id")): s for s in segs}

        picked = list(segs)
        if self.segments:
            missing = [s for s in self.segments if s not in by_id]
            if missing:
                raise EditError(f"no such segment id(s): {', '.join(missing)}")
            picked = [by_id[s] for s in self.segments]
        if self.speaker:
            picked = [s for s in picked if s.get("speaker") == self.speaker]
            if not picked:
                known = sorted({str(s.get("speaker")) for s in segs
                                if s.get("speaker")})
                raise EditError(
                    f"no segments for speaker {self.speaker!r} "
                    f"(known: {', '.join(known) or 'none'})")

        # A ZERO-WIDTH window is the point path wearing the wrong shape, and
        # it matches almost nothing: the overlap test is strict
        # (`end > lo and start < hi`), so `45–45` catches only a cue that
        # straddles 45.0 exactly and answers "matched no segments" otherwise.
        # The parser prompt says to OMIT `end_s` for a single point in time
        # and a 4B model routinely emits `end_s == start_s` instead -- which
        # is how "make the tone more formal after 0:45" became a 422. Treat
        # it as the point it plainly means; it has no other sensible reading.
        zero_width = (self.start_s is not None and self.end_s is not None
                      and float(self.start_s) == float(self.end_s))

        if self.start_s is not None and self.end_s is not None and not zero_width:
            lo, hi = sorted((float(self.start_s), float(self.end_s)))
            picked = [s for s in picked
                      if _seg_end(s) > lo and _seg_start(s) < hi]
        elif self.start_s is not None or self.end_s is not None:
            t = float(self.start_s if self.start_s is not None else self.end_s)
            hit = [s for s in picked
                   if _seg_start(s) <= t <= _seg_end(s)]
            if not hit and picked:
                hit = [min(picked, key=lambda s: min(abs(_seg_start(s) - t),
                                                     abs(_seg_end(s) - t)))]
                log.info("scope: t=%.2fs falls in a gap; nearest cue is %s "
                         "(%.2f–%.2f)", t, hit[0].get("id"),
                         _seg_start(hit[0]), _seg_end(hit[0]))
            picked = hit

        return [str(s.get("id")) for s in
                sorted(picked, key=_seg_start)]


def _as_time(v: Any) -> Optional[float]:
    """Accept 83, 83.5, "83", "1:23", "00:01:23.5" — reject the rest."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    if ":" in s:
        parts = s.split(":")
        if len(parts) > 3:
            raise EditError(f"unparseable timestamp {v!r}")
        try:
            total = 0.0
            for p in parts:
                total = total * 60.0 + float(p or 0)
        except ValueError as e:
            raise EditError(f"unparseable timestamp {v!r}") from e
        return total
    try:
        return float(s)
    except ValueError as e:
        raise EditError(f"unparseable timestamp {v!r}") from e


def _hhmmss(t: float) -> str:
    t = max(0.0, float(t))
    m, s = divmod(t, 60.0)
    h, m = divmod(int(m), 60)
    return (f"{h:d}:{m:02d}:{s:05.2f}" if h else f"{m:d}:{s:05.2f}")


def _seg_start(seg: Dict[str, Any]) -> float:
    return float(seg.get("start", seg.get("start_s", 0.0)) or 0.0)


def _seg_end(seg: Dict[str, Any]) -> float:
    return float(seg.get("end", seg.get("end_s", 0.0)) or 0.0)


def _seg_text(seg: Dict[str, Any]) -> str:
    return (seg.get("final_text") or seg.get("mt_text")
            or seg.get("text") or "")


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------

@dataclass
class EditOp:
    op: str
    scope: Scope = field(default_factory=Scope)
    params: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EditOp":
        name = str(d.get("op") or d.get("type") or "").strip().lower()
        if name not in OPS:
            raise EditError(
                f"unknown op {name!r}. Known: {', '.join(sorted(OPS))}")
        params = dict(d.get("params") or {})
        # Tolerate a flat shape — small models emit {"op":…, "factor":1.3}
        # as often as they nest it under "params".
        for k, v in d.items():
            if k in ("op", "type", "scope", "params", "note"):
                continue
            params.setdefault(k, v)
        return cls(op=name, scope=Scope.from_dict(d.get("scope")),
                   params=params, note=str(d.get("note") or ""))

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "scope": asdict(self.scope),
                "params": self.params, "note": self.note}

    def describe(self) -> str:
        return f"{self.op}({_fmt_params(self.params)}) on {self.scope.describe()}"


def classify_op(op: "EditOp", source_language: Optional[str] = None) -> str:
    """Return "rule", "script" or "llm" for a parsed op.

    Three buckets, decided by what the op's handler and its post-MT replay
    actually touch:

    - **rule**   — the handler mutates text or metadata deterministically,
                   with no outside call. The replay still runs the dirty
                   stages, but nothing in the op itself needs the LLM or
                   another hosted model to speak for it.
    - **script** — the handler stays local, but the replay re-runs a
                   pipeline stage via an external MODEL that is not an LLM
                   (today: the Bodhan transliterator for `code_mix mode=stage`,
                   Bodhan MT for `set_target_language`, and `retranslate` on
                   the fast path where target language equals the dub's
                   source language — a verbatim copy from the ASR cue).
    - **llm**    — the handler itself reaches `ctx.rewriter` to rewrite text
                   at apply time (retranslate across languages, set_tone with
                   rewrite_text, code_mix mode=llm, replace_term with no
                   replacement given).

    The kind is a property of the PARSED op, not of its `OpSpec`, because
    `code_mix`, `set_tone`, `replace_term` and `retranslate` flip bucket on
    their params and — for retranslate — on the dub's own source language.

    `source_language` is `data["language"]` for the plan (also carried on
    `EditContext.source_language`). Pass it so `retranslate` with
    `target_language == source_language` is classified as `script` at plan
    time; without it the function returns the conservative default `"llm"`
    for retranslate so the preview never understates the cost. The
    comparison is `.strip().lower()` on both sides — no ISO alias table.
    """
    name = op.op
    if name == "retranslate":
        tgt = str(op.params.get("target_language") or "").strip().lower()
        src = str(source_language or "").strip().lower()
        if tgt and src and tgt == src:
            return "script"
        return "llm"
    if name == "set_tone":
        return "llm" if bool(op.params.get("rewrite_text", True)) else "rule"
    if name == "code_mix":
        mode = str(op.params.get("mode") or "stage").strip().lower()
        return "llm" if mode == "llm" else "script"
    if name == "replace_term":
        return "llm" if op.params.get("replace") is None else "rule"
    if name == "set_target_language":
        return "script"
    if name == "merge_cues":
        return "llm"
    return "rule"


#: Sentence-final punctuation in the languages this pipeline dubs into.
#: `।`/`॥` are the Devanagari danda and double danda; the ASCII set covers a
#: code-mixed line that ends on a Latin clause.
_SENT_END = "।॥.?!"

#: Stands in for an ellipsis while the run-collapse below is applied, so
#: `...` is not mistaken for three sentence-enders and flattened to one.
_ELLIPSIS_SENTINEL = "\x00ELL\x00"


def _tidy_text(s: str) -> str:
    """Repair the seams a substitution leaves behind.

    `_h_replace_term` edits a line with `re.sub`, which is exact and knows
    nothing about the text either side of the cut. Three artefacts come out
    of that, all of them invisible in a diff and audible in the dub:

      * a DOUBLE SPACE where a deleted phrase used to be;
      * a space STRANDED before the punctuation that replaced it, so the
        danda floats off its word (`है ।` rather than `है।`);
      * a RUN of sentence-enders, when the replacement lands next to one
        that was already in the line.

    The third is the one that sent us here. `replace "ठीक है" with "।"` over
    `... है ठीक है? और ...` leaves `... है ।? और ...`: the request named the
    phrase without its question mark, so the mark outlived the phrase it
    belonged to. Bodhan reads what it is handed, so this is repaired at the
    edit rather than left for the engine to interpret.

    A run keeps its FIRST member: the replacement is what was asked for, and
    the survivor is the mark it displaced.
    """
    s = s.replace("…", "...").replace("...", _ELLIPSIS_SENTINEL)
    s = re.sub(rf"([{_SENT_END}])(?:[ \t]*[{_SENT_END}])+", r"\1", s)
    s = re.sub(rf"[ \t]+([{_SENT_END},;:])", r"\1", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.replace(_ELLIPSIS_SENTINEL, "...").strip()


def _fmt_params(p: Dict[str, Any]) -> str:
    out = []
    for k, v in p.items():
        s = str(v)
        if len(s) > 48:
            s = s[:45] + "…"
        out.append(f"{k}={s}")
    return ", ".join(out)


@dataclass
class OpSpec:
    """One supported edit, described once for the LLM, the CLI and the code."""
    name: str
    dirty_from: str                  # earliest stage in STAGE_ORDER it breaks
    params: Dict[str, str]           # param → human description
    summary: str
    examples: List[str] = field(default_factory=list)
    needs_llm: bool = False


OP_SPECS: List[OpSpec] = [
    OpSpec(
        name="set_text", dirty_from="tts",
        params={"text": "the exact replacement text, in the target language"},
        summary="Replace the translated text of the scoped segment(s) verbatim.",
        examples=['change the line at 1:20 to "यह गलत अनुवाद था"'],
    ),
    OpSpec(
        name="replace_term", dirty_from="tts",
        params={"find": "the word or phrase exactly as it appears now, "
                        "including any punctuation that belongs to it",
                "replace": 'the exact replacement; "" deletes `find`; omit '
                           "it to have `find` translated into the target "
                           "language"},
        summary="Replace one word or phrase inside the line and leave every "
                "other character identical.",
        examples=['translate the word "conditional" into Hindi',
                  'replace "cloud" with "क्लाउड" everywhere'],
    ),
    OpSpec(
        name="retranslate", dirty_from="tts",
        params={"instruction": "how the new translation should differ",
                "target_language": "ISO code, only if it changes"},
        summary="Re-translate the scoped segment(s) from the source transcript, "
                "following a free-text instruction.",
        examples=[],
        needs_llm=True,
    ),
    OpSpec(
        # The style list is READ FROM THE ENGINE, never typed here. It used
        # to read "formal | casual | friendly | serious | energetic", none of
        # which Bodhan accepts -- see `tone_styles`.
        name="set_tone", dirty_from="tts",
        params={"tone": _tone_style_ids(),
                "rewrite_text": "true (default) to also rewrite the wording"},
        summary="Change the delivery register: an engine style hint, and (by "
                "default) an LLM rewrite of the wording to match.",
        examples=["deliver the lines after 1:00 as educational lecture",
                  "read from 2:00 to 2:30 as TV style news"],
        needs_llm=True,
    ),
    OpSpec(
        name="code_mix", dirty_from="codemix",
        params={"terms": "source-language terms to keep unchanged",
                "mode": "stage (re-run the transliterator) | llm (rewrite)",
                "termbase": "path to a termbase TSV"},
        summary="Code-mix the translation — keep technical terms in the "
                "source language inside the target-language sentence.",
        examples=["code-mix the translation",
                  "keep gradient and weight in English throughout",
                  "keep the technical terms in the original language"],
    ),
    OpSpec(
        name="set_gender", dirty_from="voices",
        params={"gender": "male | female"},
        summary="Change the speaker's apparent gender by reassigning the TTS "
                "voice from the matching-gender pool.",
        examples=["the speaker at 1:20 should sound female",
                  "SPEAKER_01 is male, not female"],
    ),
    OpSpec(
        name="create_speaker", dirty_from="",
        # The one op that MINTS a speaker id. Every other op refuses an
        # unknown speaker; this one carves out a hole for the case diarization
        # failed on -- two people merged into one SPK, or a wholly missed
        # voice -- and the user corrects it by naming the voice directly.
        #
        # Voice is required (not optional, not a warning): a brand-new speaker
        # has no reference clip, no ASD vote and no catalog pick, so without a
        # voice TTS would fall back to the engine default -- the silent-
        # wrong-output trap `_h_set_speaker` already cites. Gender follows the
        # voice from the catalog; tone defaults to `neutral` and is read from
        # the request if named.
        params={"voice": "engine voice id for the new speaker, "
                         "e.g. Deepika, Chinmay, Adarsh",
                "gender": "male | female (optional; follows the voice "
                          "when omitted)",
                "tone": f"optional delivery style ({_tone_style_ids()}); "
                         "defaults to neutral",
                "engine": "tts engine, only if it changes"},
        summary="Mint a new speaker id for the scoped cues and assign it the "
                "named voice. Use when diarization merged two people onto one "
                "SPK, or when a stretch has no speaker who fits.",
        examples=[
            "create a new speaker for the voice from 0:20 to 0:25 and "
            "assign the Deepika voice",
            "the speaker from 0:00 to 0:09 is wrong -- create a new speaker "
            "with the Chinmay voice",
            "create a new speaker and assign the Adarsh voice to cue 4 "
            "and cue 5",
        ],
    ),
    OpSpec(
        name="set_speaker", dirty_from="voices",
        params={"speaker": "the speaker id that really talks here, "
                           "e.g. SPK_02"},
        summary="Reattribute a stretch of dialogue to a different speaker, "
                "so it is spoken in THAT speaker's voice.",
        # Both examples name the speaker EXPLICITLY. "that is the second
        # speaker" was advertised for months and parses to nothing: a rule
        # parser cannot turn an ordinal into a diariser label without
        # guessing whether the count is 0- or 1-based, and a wrong guess is
        # a real line read in the wrong voice.
        examples=["the line at 1:20 is SPK_02, not SPK_00",
                  "from 0:30 to 0:45 that is actually SPK_02"],
    ),
    OpSpec(
        name="set_voice", dirty_from="voices",
        # `indic-speak` voices are BARE NAMES (Amit, Kavya, Suhani). This
        # read "hi-IN-MadhurNeural", the Azure/Edge shape, for as long as
        # Edge was registered — and kept reading it for six days after Edge
        # was deleted on 2026-10-17, so both the LLM and the regex fallback
        # were being shown a pattern no voice in the catalogue matches.
        #
        # This is also how a gender change is expressed through NLQ: pick a
        # voice of the desired gender from the catalog. The handler reads the
        # catalog entry's gender and writes it to `gender_overrides` so the
        # speaker's apparent gender follows the voice. A bare "make this
        # female" with no voice named is deliberately refused by the LLM
        # prompt and the regex, because set_gender's pool-pick path is
        # reserved for the cast-table dropdown — see `server/routes/routes_edits.py`.
        params={"voice": "engine voice id, e.g. Amit, Deepika, Chinmay",
                "engine": "tts engine, only if it changes"},
        summary="Pin a specific TTS voice for the named speaker. Requires "
                "the speaker to be named; a per-cue voice edit is a "
                "create_speaker, not a set_voice. The voice's gender becomes "
                "the speaker's apparent gender.",
        examples=["use the Deepika voice for SPEAKER_00",
                  "give SPK_01 the Chinmay voice",
                  "SPK_01 should sound female, assign the Deepika voice"],
    ),
    OpSpec(
        name="set_speed", dirty_from="tts",
        params={"factor": "1.1 = 10% faster, 0.9 = 10% slower"},
        summary="Speed the dubbed speech up or down over a window, by "
                "re-fitting each clip into a shorter/longer slot.",
        examples=["speed up the audio between 0:30 and 0:45 by 1.1x",
                  "slow down the intro a bit"],
    ),
    OpSpec(
        name="lipsync_window", dirty_from="lipsync",
        params={"model": f"{_lipsync_model_ids()}, only if it changes"},
        summary="Lip-sync only the scoped time window instead of the whole "
                "video, splicing the render back into the full timeline.",
        examples=[],
    ),
    OpSpec(
        # `timing`, not `tts`, because `_h_mute_cue` sets exactly that on
        # the applied record: TTS still synthesises muted cues so a revert
        # has the clip, and the freed slot has to be re-planned before mix
        # drops it. The spec said `tts` while the handler said `timing`, so
        # the UI quoted a re-speak that never happened.
        name="mute_cue", dirty_from="timing",
        params={},
        summary="Silence the scoped segment(s) — TTS skips them, the mix "
                "leaves the M&E bed playing through, and their time is "
                "released as a pause the neighbouring cues may borrow.",
        examples=["mute the line at 0:15",
                  "delete the garbage cue SEG_0009"],
    ),
    OpSpec(
        # Re-translates the scoped cues through Bodhan MT -- NOT the LLM.
        # Scope policy is enforced in the handler: `all` sweeps the whole
        # dub (dirty_from=codemix, voices re-cast from the new pool); a
        # cue list or time window re-translates only those cues and pins
        # a per-cue voice from the new language's catalog (dirty_from=tts).
        # Speaker-only scope is refused -- a bare speaker is not a time
        # bound and the mixed-language result would be unpredictable.
        name="set_target_language", dirty_from="codemix",
        params={"target_language": "ISO code of the new target language, "
                                   "e.g. hi, mr, ta — one of the 22 Indic "
                                   "languages the home-page picker lists",
                "target_script": "script override (deva | taml | …); "
                                 "omit to use Bodhan's default"},
        summary="Re-translate the scoped cues (or the whole dub) into a "
                "different Indic language via Bodhan MT. Scope=all swaps "
                "the dub globally and re-casts every voice from that "
                "language's pool; a cue or time scope re-translates only "
                "those cues and pins a voice per cue, leaving the rest of "
                "the dub in its original language.",
        examples=["translate the whole dub to Marathi",
                  "retranslate 0:30 to 0:45 into Tamil",
                  "change the target language to Bengali"],
    ),
    OpSpec(
        # Merges exactly TWO consecutive cues of the same speaker into one.
        # The LLM is given both source lines AND both current translations
        # and asked for a single merged target-language line, so the join
        # reads naturally rather than being a string-concat of two cues.
        # Refusals (apply time): ids missing, not index-adjacent, speakers
        # differ, or the gap between cue1.end and cue2.start exceeds 2.0s.
        name="merge_cues", dirty_from="codemix",
        params={},
        summary="Merge two consecutive cues from the same speaker into one. "
                "The first cue absorbs the second: timing extends to the "
                "second's end, originals are joined, and the translation is "
                "rewritten by the LLM to read as a single utterance.",
        examples=["merge cue 4 and cue 5",
                  "combine SEG_0004 with SEG_0005"],
        needs_llm=True,
    ),
    OpSpec(
        # Split a cue in two at a user-given timestamp and (optionally)
        # assign the right half to a different speaker, or mint a new one.
        # Deterministic: no LLM is required. The two halves inherit the
        # parent's text split on the nearest word boundary that fits inside
        # each half; translation text is kept on the LEFT half and the right
        # half is left for the operator to retranslate (or hand off to the
        # agent).
        name="split_cue", dirty_from="codemix",
        params={"at_s": "timestamp inside the cue to split at (seconds)",
                "speaker": "optional existing speaker id for the right half",
                "voice": "optional voice name — mints a new speaker for "
                         "the right half (gender follows the voice)"},
        summary="Split one cue into two at a given timestamp. The right "
                "half becomes a new cue; its speaker is kept, reassigned, "
                "or minted from the voice you name.",
        examples=["split the cue at 0:12.5",
                  "split SEG_0007 at 7.8 and assign the right half to SPK_02",
                  "split at 1:03 and create a new speaker with "
                  "the Deepika voice"],
    ),
]

OPS: Dict[str, OpSpec] = {s.name: s for s in OP_SPECS}


#: What re-running from a stage actually means, in a sentence. Keyed by the
#: `dirty_from` values above, so the explanation a user reads is derived from
#: the rerun logic rather than written alongside it and left to rot.
STAGE_PLAIN: Dict[str, str] = {
    "codemix":  "re-translates the affected lines, then re-speaks, remixes "
                "and re-syncs them",
    "duration": "re-fits the timing, then re-speaks, remixes and re-syncs",
    "voices":   "re-casts the voice, then re-speaks those lines, remixes "
                "the audio and re-syncs the lips",
    # "re-fits", not "re-speaks": `backend/tts.py` is incremental. It hashes
    # a RAW signature (text, voice, style, reference, language, model) and a
    # STRETCH signature (target duration, cap) per segment, and only the
    # segments whose raw signature moved are sent to the engine. A pure speed
    # edit moves only the stretch signature, so the cached clip is re-fitted
    # with no engine call at all -- which is why `set_speed` dirties from
    # `tts` and still costs nothing at Bodhan. Saying "re-speaks" here quoted
    # a bill the stage does not run up.
    "tts":      "re-fits the affected lines — re-speaking only those whose "
                "wording, voice or style changed — then remixes the audio "
                "and re-syncs the lips",
    "timing":   "re-fits the timing, remixes and re-syncs",
    "mix":      "remixes the audio and re-syncs the lips",
    "lipsync":  "re-syncs the lips only",
}

#: op name -> how it reads to a person. `title` is the action, `takes` is
#: what the user supplies, `changes` is what it does to the DUB.
#:
#: The replay cost is deliberately NOT written here -- it is derived from
#: the op's own `dirty_from` through `STAGE_PLAIN`. Authoring both halves by
#: hand is how a description comes to promise a cheaper re-render than the
#: op really causes. Splitting them also keeps `changes` honest in the other
#: direction: `retranslate` replays from TTS, but what it CHANGES is the
#: wording, and a reader told only "re-speaks the lines" would not know the
#: translation itself moves.
#:
#: Every op must have an entry here, else it surfaces in the UI as a bare
#: identifier. Four extra keys drive the UI "edit workspace":
#:
#:   category  the intent group it belongs under — one of six, in the order
#:             `OP_SPECS` already lists them, so the group order is derived
#:             from the registry rather than written down a second time.
#:   scopes    which kinds of on-screen selection this op can be aimed at:
#:             "segment" (a cue/time), "speaker", or "all" (the whole dub).
#:             The UI reads it to decide which actions a clicked cue or
#:             speaker makes relevant, and whether a selection can be folded
#:             into the phrasing automatically.
#:   quick     one of the handful shown before "See all actions".
#:   fill      a scope-AGNOSTIC phrasing. The UI prefixes it with whatever is
#:             selected ("cue 5: …", "SPEAKER_00: …") so the user never types
#:             a timestamp or a speaker id. Deliberately not the same string
#:             as `examples`: those carry their own baked-in scope and are
#:             used as-is when nothing is selected.
#:   needs     the ONE value the user still has to supply, named by the
#:             `<angle>` placeholder inside `fill`, or None when the fill is
#:             complete as written. Four of these fills used to carry `…` or
#:             `<voice>` and were sent verbatim: the ellipsis became the
#:             line's new text, and "translate the whole dub to this
#:             language" was a 422 every single time, because the prompt
#:             forbids inventing an ISO code and no regex matches "this".
#:             The UI reads `needs` to open the right picker (voice, tone,
#:             language) or to select the placeholder, and refuses to send
#:             while an unfilled `<…>` is still in the box. A placeholder
#:             cannot be made parseable by prefilling junk; it can only be
#:             made impossible to send.
USER_ACTIONS: Dict[str, Dict[str, Any]] = {
    "set_text": {
        "title": "Rewrite a line",
        "takes": "a moment in the video, and the exact wording you want",
        "changes": "that line is spoken with your wording",
        "category": "Translation", "scopes": ["segment"], "quick": True,
        "fill": 'change this line to "<new wording>"',
        "needs": "text",
    },
    "replace_term": {
        "title": "Change one word everywhere",
        "takes": "the word as it appears now, and either the replacement or "
                 "nothing (it will be translated for you)",
        "changes": "only that word changes; the rest of the line stays "
                   "exactly as it was",
        "category": "Translation", "scopes": ["segment", "all"], "quick": False,
        "fill": 'replace "<word>" with "<replacement>"',
        "needs": "text",
    },
    "retranslate": {
        "title": "Translate a passage again",
        "takes": "a time range, and how you would like the translation revised",
        "changes": "the translation of those lines is rewritten, then spoken",
        "category": "Translation", "scopes": ["segment"], "quick": True,
        "fill": "please rewrite this passage in a more formal tone",
    },
    "set_tone": {
        "title": "Change how a line is delivered",
        "takes": "a time range or a speaker, and one of two delivery "
                 "styles: TV style news or educational lecture",
        "changes": "the same words are spoken in a different manner",
        "category": "Voice", "scopes": ["segment", "speaker"], "quick": False,
        "fill": "deliver this in an educational lecture style",
        "needs": "tone",
        "options": list(TONE_CHOICES),
    },
    "code_mix": {
        "title": "Code-mix the translation",
        "takes": "optionally the terms to protect, e.g. gradient, weight",
        "changes": "technical terms stay in the source language inside the "
                   "translated sentence instead of being translated",
        "category": "Translation", "scopes": ["segment", "all"], "quick": False,
        "fill": "keep technical terms such as gradient and weight in English",
    },
    "create_speaker": {
        "title": "Create a new speaker",
        "takes": "a time range or cue(s), and the voice to assign",
        "changes": "a new speaker id is minted, those cues move onto it, "
                   "and it is read in the voice you named; the speaker's "
                   "gender follows the voice",
        "category": "Speakers", "scopes": ["segment"], "quick": True,
        "fill": "create a new speaker for these cues and "
                "assign the <voice> voice",
        "needs": "voice",
    },
    "set_speaker": {
        "title": "Reassign who is talking",
        "takes": "a time range, and which speaker it really is",
        "changes": "those lines move to that speaker and are read in THEIR "
                   "voice; who speaks when is not recomputed",
        "category": "Speakers", "scopes": ["segment"], "quick": False,
        "fill": "assign these lines to <speaker>",
        "needs": "speaker",
    },
    "set_voice": {
        "title": "Set the voice / gender of a speaker",
        "takes": "a speaker, and a voice from the catalogue "
                 "(gender follows the voice)",
        "changes": "that speaker is read by the voice you named, and their "
                   "apparent gender follows the voice's catalog entry",
        # Speaker-only: a cue-scoped voice edit is almost always what
        # create_speaker was added for. The prompt and the handler both say
        # so; the UI chip hides the segment scope to match.
        "category": "Voice", "scopes": ["speaker"], "quick": True,
        "fill": "use the <voice> voice for this speaker",
        "needs": "voice",
    },
    "set_speed": {
        "title": "Speed up or slow down",
        "takes": "a time range, and how much faster or slower",
        "changes": "the dubbed speech is re-fitted into a shorter or "
                   "longer slot; the picture is untouched",
        "category": "Timing & audio", "scopes": ["segment"], "quick": True,
        "fill": "make this line 1.1 times faster",
    },
    "mute_cue": {
        "title": "Mute a cue",
        "takes": "a time range or segment id",
        "changes": "silences those lines — TTS skips them and the "
                   "music-and-effects bed plays through; their time becomes "
                   "a pause the lines either side may borrow instead of "
                   "speeding up",
        "category": "Cleanup", "scopes": ["segment"], "quick": True,
        "fill": "mute this line but keep the background audio",
    },
    "set_target_language": {
        "title": "Change the target language",
        "takes": "an Indic language, and optionally a cue or time range "
                 "(defaults to the whole dub)",
        "changes": "the scoped lines are re-translated into the new "
                   "language via Bodhan MT; a whole-dub change re-casts "
                   "every voice from that language's pool, a scoped change "
                   "pins a native voice per cue and leaves the rest alone",
        "category": "Translation", "scopes": ["all", "segment"], "quick": True,
        "fill": "change the target language to <language>",
        "needs": "language",
    },
    "split_cue": {
        "title": "Split one cue in two",
        "takes": "a timestamp inside the cue, and optionally a speaker "
                 "to assign the right half to (or a voice to mint a new "
                 "speaker for it)",
        "changes": "the cue is cut at that moment; the right half becomes "
                   "a new cue, which keeps the original speaker unless you "
                   "reassign it or mint a new one",
        "category": "Cleanup", "scopes": ["segment"], "quick": False,
        "fill": "split this cue at <time>",
        "needs": "time",
    },
    "merge_cues": {
        "title": "Merge two consecutive cues",
        "takes": "two adjacent cue ids from the same speaker",
        "changes": "the two cues become one — timing extends to the second "
                   "cue's end, the originals are joined, and the translation "
                   "is rewritten so it reads as a single utterance",
        # Typed-command only: not surfaced in the quick-action strip, and no
        # `fill` so the workspace does not open a picker. The LLM parser
        # still accepts it; the user names both ids in free text.
        "category": "Cleanup", "scopes": ["segment"], "quick": False,
    },
}


def action_catalogue() -> List[Dict[str, Any]]:
    """Every op as plain language: what it is, takes, changes, and the cost.

    Served to the UI so the list cannot drift from the op registry. The repo
    has paid for that before: `server/service/speakers.py` first shipped an invented
    tone vocabulary, and every tone change then failed at the API with
    nothing on screen to explain it.
    """
    out: List[Dict[str, Any]] = []
    for spec in OP_SPECS:
        meta = USER_ACTIONS.get(spec.name)
        if not meta:
            continue
        out.append({
            "op": spec.name,
            "title": meta["title"],
            "takes": meta["takes"],
            "changes": meta["changes"],
            # Derived, never authored -- see the note above.
            "replays": STAGE_PLAIN.get(spec.dirty_from, spec.dirty_from),
            "examples": list(spec.examples or [])[:2],
            # The workspace metadata. See the USER_ACTIONS note for each.
            "category": meta["category"],
            "scopes": list(meta.get("scopes", [])),
            "quick": bool(meta.get("quick", False)),
            "fill": meta.get("fill", ""),
            "needs": meta.get("needs"),
            # A narrowed pick list, when the op offers fewer values than the
            # catalogue holds. None means "use the full list for `needs`".
            "options": list(meta["options"]) if meta.get("options") else None,
        })
    return out


def sample_prompts(per_op: int = 2) -> List[Tuple[str, str]]:
    """`(phrasing, what it does)` pairs, in user language, for every op.

    The op names (`set_speaker`, `mute_cue`) are the agent's
    vocabulary, not a person's. Listing them when a request fails to parse
    tells the user what the CODE is called, which is not what they asked.
    Each OpSpec already carries worked `examples` written as someone would
    actually type them; this surfaces those instead.
    """
    out: List[Tuple[str, str]] = []
    for spec in OP_SPECS:
        for ex in (spec.examples or [])[:per_op]:
            out.append((ex, spec.summary))
    return out


def sample_prompts_text(per_op: int = 2, indent: str = "  ") -> str:
    """`sample_prompts` rendered for a terminal."""
    lines = []
    for spec in OP_SPECS:
        ex = (spec.examples or [])[:per_op]
        if not ex:
            continue
        lines.append(f"{indent}{spec.summary}")
        for e in ex:
            lines.append(f'{indent}  "{e}"')
        lines.append("")
    return "\n".join(lines).rstrip()



# ---------------------------------------------------------------------------
# voice resolution
# ---------------------------------------------------------------------------

def voice_for_gender(
    gender: str, *,
    engine: Optional[str],
    language: Optional[str],
    catalog_json: Optional[Path] = None,
    exclude: Sequence[str] = (),
) -> Tuple[Optional[str], str]:
    """Pick a voice of `gender` for `engine`/`language`.

    Returns `(voice_id, why)`. `voice_id` is None when the engine's catalog
    has nothing of that gender for that language — reported, never silently
    substituted with the opposite gender, because that is exactly the failure
    the voice-assignment stage already shipped once.
    """
    gender = (gender or "").strip().lower()
    if gender not in ("male", "female"):
        raise EditError(f"gender must be 'male' or 'female', got {gender!r}")
    eng = (engine or "").strip().lower()
    prim = (language or "hi").split("-")[0].lower()

    from backend.tts.voices import load_voice_catalog
    cat_engine = "indic-speak" if eng in ("", "bodhan", "indic-speak") else eng
    catalog = load_voice_catalog(cat_engine, language=prim,
                                 catalog_json=catalog_json)
    pool = [v for v in catalog
            if (v.gender or "").lower() == gender and v.voice not in exclude]
    if not pool:
        return None, (f"{cat_engine} catalog has no {gender} voice "
                      f"for {prim!r}")
    native = [v for v in pool if (v.language or "").lower() == prim]
    lead = [v for v in (native or pool) if v.tier == "lead"]
    pick = (lead or native or pool)[0]
    return pick.voice, (
        f"{cat_engine} {gender} voice for {prim!r}"
        + ("" if native else " (cross-language — no native one in the catalog)")
    )


# ---------------------------------------------------------------------------
# applying
# ---------------------------------------------------------------------------

@dataclass
class AppliedOp:
    op: str
    scope: str
    params: Dict[str, Any]
    segments: List[str]
    changes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dirty_from: str = ""


@dataclass
class EditResult:
    merged_json: Path
    applied: List[AppliedOp] = field(default_factory=list)
    dirty_stages: List[str] = field(default_factory=list)
    speaker_voice_overrides: Dict[str, dict] = field(default_factory=dict)
    gender_overrides: Dict[str, str] = field(default_factory=dict)
    lipsync_window: Optional[Tuple[float, float]] = None
    lipsync_model: Optional[str] = None
    translit_model: Optional[str] = None
    asd_model: Optional[str] = None
    termbase_path: Optional[Path] = None
    #: Set by `_h_set_target_language` on a whole-dub change. `rerun()`
    #: prefers it over `cfg.get("target_language")` so the post-MT chain
    #: replays against the new language, and `commit()` persists it.
    target_language_override: Optional[str] = None
    target_script_override: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def touched(self) -> List[str]:
        seen: List[str] = []
        for a in self.applied:
            for s in a.segments:
                if s not in seen:
                    seen.append(s)
        return seen

    def to_jsonable(self) -> dict:
        return {
            "merged_json": str(self.merged_json),
            "applied": [asdict(a) for a in self.applied],
            "dirty_stages": list(self.dirty_stages),
            "speaker_voice_overrides": self.speaker_voice_overrides,
            "gender_overrides": self.gender_overrides,
            "lipsync_window": list(self.lipsync_window) if self.lipsync_window else None,
            "lipsync_model": self.lipsync_model,
            "translit_model": self.translit_model,
            "termbase_path": str(self.termbase_path) if self.termbase_path else None,
            "target_language_override": self.target_language_override,
            "target_script_override": self.target_script_override,
            "warnings": list(self.warnings),
        }


@dataclass
class EditContext:
    """Everything an op needs that is not in the merged JSON itself."""
    target_language: Optional[str] = None
    source_language: Optional[str] = None
    tts_engine: Optional[str] = None
    voice_engine: Optional[str] = None
    voice_catalog_json: Optional[Path] = None
    translit_model: Optional[str] = None
    lipsync_model: Optional[str] = None
    termbase_path: Optional[Path] = None
    #: Where an op may write a file it generates (the job folder's `edits/`).
    work_dir: Optional[Path] = None
    #: Set by `apply_edits`. A dry run must not leave files behind either —
    #: the plan still names the termbase it would write, it just does not
    #: write it.
    dry_run: bool = False
    #: `(system, user) -> str`. Injected so this module stays import-clean of
    #: the network; `backend.edit_agent` passes the OpenRouter client in.
    rewriter: Optional[Callable[[str, str], str]] = None


def apply_edits(
    merged_json: Path,
    ops: Sequence[EditOp],
    *,
    ctx: Optional[EditContext] = None,
    nlq: str = "",
    dry_run: bool = False,
) -> EditResult:
    """Apply `ops` to `merged_json` and report what has to be re-run.

    With `dry_run=True` everything is resolved, checked and reported, and the
    file is not written — the intended default for a first look at an edit.
    """
    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    ctx = ctx or EditContext()
    ctx.dry_run = bool(dry_run)
    res = EditResult(merged_json=merged_json)

    if not ops:
        raise EditError("no edit operations to apply")

    for op in ops:
        spec = OPS.get(op.op)
        if spec is None:
            raise EditError(f"unknown op {op.op!r}")
        if op.scope.is_empty():
            # An unscoped edit means the whole dub. Say so rather than
            # guessing a window; a silent whole-file rewrite is the expensive
            # kind of surprise.
            op.scope.all = True

        # merge_cues is DESTRUCTIVE -- the handler removes the second cue from
        # the segments list. The commit path then persists that to the merged
        # JSON, and `server/runner.py::_run_rerun` re-applies the stored ops
        # against the ALREADY-MUTATED document to compute `dirty_stages`.
        # Scope.resolve would then raise "no such segment id(s): SEG_XXXX" on
        # the vanished cue and the replay dies before TTS runs. Detect that
        # state here and record the op as already-applied -- the mutation is
        # baked into the file, so there is nothing to re-mutate, but the
        # `dirty_from` still has to propagate so the post-MT stages replay.
        if op.op == "merge_cues" and op.scope.segments:
            present = {str(s.get("id")) for s in data.get("segments") or []}
            wanted = list(op.scope.segments)
            missing = [s for s in wanted if s not in present]
            kept = [s for s in wanted if s in present]
            if missing and kept:
                applied = AppliedOp(
                    op=op.op, scope=op.scope.describe(),
                    params=dict(op.params), segments=wanted,
                    dirty_from=spec.dirty_from)
                applied.warnings.append(
                    f"merge_cues: {', '.join(missing)} not in the cue list "
                    f"(already merged into {', '.join(kept)}); re-apply "
                    f"skipped, downstream replay unchanged")
                res.applied.append(applied)
                res.warnings.extend(applied.warnings)
                continue

        seg_ids = op.scope.resolve(data)
        if not seg_ids and op.op != "lipsync_window":
            raise EditError(
                f"{op.op}: scope '{op.scope.describe()}' matched no segments")
        handler = _HANDLERS[op.op]
        applied = AppliedOp(op=op.op, scope=op.scope.describe(),
                            params=dict(op.params), segments=seg_ids,
                            dirty_from=spec.dirty_from)
        handler(data, seg_ids, op, applied, ctx, res)
        res.applied.append(applied)
        res.warnings.extend(applied.warnings)

    res.dirty_stages = rerun_stages([a.dirty_from for a in res.applied])

    journal = data.setdefault("extra", {}).setdefault("edits", {
        "version": EDITS_VERSION, "history": [],
    })
    journal.setdefault("version", EDITS_VERSION)
    journal.setdefault("history", []).append({
        "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "nlq": nlq,
        "ops": [op.to_dict() for op in ops],
        "applied": [asdict(a) for a in res.applied],
        "dirty_stages": res.dirty_stages,
    })

    if not dry_run:
        tmp = merged_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.replace(merged_json)
        log.info("edits: wrote %d op(s) into %s", len(res.applied), merged_json)
    return res


def rerun_stages(dirty_from: Sequence[str],
                 *, available: Optional[Sequence[str]] = None) -> List[str]:
    """Stages to re-run: the earliest dirty one and everything after it.

    `available` narrows the set to the stages the job was configured with, so
    a run that never did voice conversion is not suddenly asked to.
    """
    if not dirty_from:
        return []
    known = [STAGE_ORDER.index(s) for s in dirty_from if s in STAGE_ORDER]
    if not known:
        return []
    idx = min(known)
    out = list(STAGE_ORDER[idx:])
    if available is not None:
        keep = set(available)
        out = [s for s in out if s in keep]
    return out


def _backup(seg: Dict[str, Any], key: str) -> None:
    """Record the pre-edit value of `key`, once, so `revert` can restore it."""
    bk = seg.setdefault("edit_backup", {})
    if key not in bk:
        bk[key] = seg.get(key)


def revert(merged_json: Path, *, segments: Optional[Sequence[str]] = None,
           dry_run: bool = False) -> List[str]:
    """Undo every recorded edit (or only those on `segments`).

    Returns the ids restored. Clears `edit_backup` as it goes, so a revert is
    idempotent rather than re-restoring stale values on a second call.
    """
    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    want = set(segments) if segments else None
    restored: List[str] = []
    for seg in data.get("segments", []):
        sid = str(seg.get("id"))
        if want is not None and sid not in want:
            continue
        bk = seg.pop("edit_backup", None)
        if not bk:
            continue
        for k, v in bk.items():
            if v is None:
                seg.pop(k, None)
            else:
                seg[k] = v
        restored.append(sid)
    if restored and not dry_run:
        tmp = merged_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.replace(merged_json)
    return restored


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------

def _set_segment_text(seg: Dict[str, Any], text: str) -> None:
    """Write a new translation into both fields TTS might read.

    `synthesize_segments` prefers `final_text`, falls back to `mt_text`. Only
    writing one of them is how an edit gets silently ignored on a run that
    did (or did not) code-mix.
    """
    _backup(seg, "mt_text")
    _backup(seg, "final_text")
    seg["mt_text"] = text
    seg["final_text"] = text


def _h_set_text(data, seg_ids, op, applied, ctx, res):
    text = op.params.get("text")
    if not text or not str(text).strip():
        raise EditError("set_text needs a non-empty `text`")
    text = str(text).strip()
    if len(seg_ids) > 1:
        applied.warnings.append(
            f"set_text applied the same text to {len(seg_ids)} segments — "
            f"scope a narrower window if that was not intended")
    by_id = {str(s.get("id")): s for s in data.get("segments", [])}
    for sid in seg_ids:
        seg = by_id[sid]
        before = _seg_text(seg)
        _set_segment_text(seg, text)
        applied.changes.append(f"{sid}: {before!r} → {text!r}")


#: Leading junk a small model prefixes to a "reply with the line only"
#: answer. Measured on gemma-3-12b: a retranslate came back as
#: `': cloud और इसी तरह …'` — the whole correct line behind a stray colon,
#: which would then be SPOKEN by the TTS engine. Cheaper to strip here than
#: to keep tightening the prompt against it.
_REWRITE_JUNK = re.compile(r'^[\s:：\-–—•*>]+')


def _clean_rewrite(text: str) -> str:
    """Trim a rewritten line of the wrapper a model puts around it.

    Quotes and code fences too: the prompts all say "reply with the line
    ONLY", and the failure when that is not obeyed is silent — the stray
    character is just read aloud.
    """
    out = (text or "").strip()
    if out.startswith("```"):
        out = out.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    out = _REWRITE_JUNK.sub("", out).strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'\u201c\u201d":
        out = out[1:-1].strip()
    return out


def _h_replace_term(data, seg_ids, op, applied, ctx, res):
    """Swap one term inside the line; everything else is byte-identical.

    `retranslate` could not do this. It hands the whole line to an LLM and
    takes a whole line back, so even told to change one word the model
    rewrites around it — measured on one cue, "translate the word
    'conditional'" also turned `text generators` into `पाठ जनरेटर` in one
    segment and `टेक्स्ट जेनरेटर` in the next. Two different renderings of a
    phrase nobody asked about, in adjacent lines.

    Here the substitution is `str`-level and the model, when it is used at
    all, is asked for ONE WORD rather than a sentence. The rest of the cue
    cannot drift because it is never regenerated.

    `replace` omitted means "translate `find` into the target language" — one
    LLM call per distinct term for the whole op, not one per segment, because
    the same word in two cues must come out the same way. That consistency is
    the other half of what `retranslate` got wrong.
    """
    find = str(op.params.get("find") or op.params.get("term")
               or op.params.get("word") or "").strip()
    if not find:
        raise EditError("replace_term needs a `find` word or phrase")

    repl = op.params.get("replace", op.params.get("replacement",
                                                  op.params.get("to")))
    repl = None if repl is None else str(repl)

    if repl is None:
        if ctx.rewriter is None:
            raise EditError(
                f"replace_term can translate {find!r} only with the LLM "
                f"(OPENROUTER_API_KEY unset). Pass the replacement outright: "
                f'replace "{find}" with "<your text>".')
        lang = ctx.target_language or "the target language"
        from backend.common.prompts import PROMPTS as _PROMPTS
        system = _PROMPTS["term_translation"]
        user = (f"Target language: {lang}\n"
                f"Translate this term: {find}")
        repl = _clean_rewrite(ctx.rewriter(system, user))
        if not repl:
            raise EditError(f"the model returned no translation for {find!r}")
        applied.changes.append(f"term: {find!r} → {repl!r} (translated)")

    # Case-insensitive, because the cue may carry `Conditional` mid-sentence
    # while the request typed it lower-case. `re.escape` keeps a term with
    # punctuation in it literal rather than letting it act as a pattern.
    pattern = re.compile(re.escape(find), re.IGNORECASE)
    by_id = {str(s.get("id")): s for s in data.get("segments", [])}
    hits = 0
    for sid in seg_ids:
        seg = by_id[sid]
        cur = _seg_text(seg)
        if not pattern.search(cur):
            # Named rather than silently skipped: a scope that caught cues
            # without the term is usually an off-by-one in the request, and
            # the old behaviour rewrote those cues anyway.
            applied.warnings.append(f"{sid}: {find!r} not in this line; left alone")
            continue
        new_text = _tidy_text(pattern.sub(repl.replace("\\", "\\\\"), cur))
        if new_text == cur:
            continue
        _set_segment_text(seg, new_text)
        applied.changes.append(f"{sid}: {cur!r} → {new_text!r}")
        hits += 1

    if not hits:
        raise EditError(
            f"{find!r} does not appear in any of the scoped cue(s). Check the "
            f"spelling, or widen the scope.")


def _h_retranslate(data, seg_ids, op, applied, ctx, res):
    """Re-translate scoped cues, with two fast paths that skip the LLM.

    Two cue-level shortcuts come first:

    1. **Source cue is empty.** The ASR captured nothing to translate from —
       no model can invent one. The cue is left as it stands, and the
       history records `"SEG_XXXX: source cue is empty, no change applied"`.
    2. **Target language equals the dub's source language** (case-agnostic,
       strip+lower, compared against `data["language"]`). "keep this cue in
       english" on an English-source dub is a verbatim copy from the ASR
       transcript into `mt_text`. No LLM call, deterministic. This path is
       what `classify_op` tags `"script"` at plan time.

    Everything else goes through the LLM path below, which is `"llm"`-kind
    classification and matches the historical retranslate behaviour.
    """
    instruction = str(op.params.get("instruction")
                      or op.params.get("how") or "").strip()
    tgt_raw = op.params.get("target_language") or ctx.target_language or ""
    tgt_norm = str(tgt_raw).strip().lower()
    src_norm = str(data.get("language") or "").strip().lower()
    lang = tgt_raw or "the target language"
    by_id = {str(s.get("id")): s for s in data.get("segments", [])}

    same_language = bool(tgt_norm and src_norm and tgt_norm == src_norm)

    for sid in seg_ids:
        seg = by_id[sid]
        src = seg.get("text") or ""
        cur = _seg_text(seg)

        # 1. Empty source — nothing to translate from. Warn and move on, so
        #    the edit journal names the cue as deliberately skipped rather
        #    than silently dropped.
        if not src.strip():
            applied.warnings.append(
                f"{sid}: source cue is empty, no change applied")
            continue

        # 2. Target language is the source language — copy ASR verbatim.
        #    Deterministic, no LLM, classifies as script.
        if same_language:
            if cur == src:
                applied.warnings.append(
                    f"{sid}: already matches the source line, no change")
                continue
            _set_segment_text(seg, src)
            applied.changes.append(
                f"{sid}: {cur!r} → {src!r} (source verbatim, same language)")
            continue

        # 3. Different language — go through the LLM.
        if ctx.rewriter is None:
            raise EditError(
                "retranslate needs the LLM (OPENROUTER_API_KEY unset). Use "
                "set_text with the exact wording instead.")
        # The instruction describes a CHANGE to a line that is already
        # translated, and saying so matters. Told only "reply with the
        # replacement line", gemma-3-12b answered a word-level instruction
        # ("translate the word 'conditional' into Hindi") with just that
        # word — `'.शर्तिया टेक्स्ट जेनरेटर'` in place of the whole
        # sentence, which is a destroyed cue rather than a failed one.
        from backend.common.prompts import PROMPTS as _PROMPTS
        system = _PROMPTS["retranslate"]
        user = (
            f"Source line (original language): {src}\n"
            f"Current translation ({lang}): {cur}\n"
            f"Instruction: {instruction or 'improve the translation'}\n"
            f"Write the corrected {lang} line.")
        new = _clean_rewrite(ctx.rewriter(system, user))
        if not new or new == cur:
            applied.warnings.append(f"{sid}: model returned no change")
            continue
        _set_segment_text(seg, new)
        applied.changes.append(f"{sid}: {cur!r} → {new!r}")


def _h_set_tone(data, seg_ids, op, applied, ctx, res):
    tone = str(op.params.get("tone") or op.params.get("style") or "").strip()
    if not tone:
        raise EditError("set_tone needs a `tone`")

    # Pin the value to something the engine knows, and say so loudly when it
    # is not. Until this check existed a tone edit from the agent wrote
    # whatever adjective the sentence contained -- "formal", "warm",
    # "urgent" -- none of which Bodhan accepts, so the op reported success
    # and changed nothing audible. Matching is case-insensitive because the
    # engine's own spellings are mixed ("AIR style news", "Customer Care")
    # and nobody types them that way; the stored value is the engine's.
    _known = {t.lower(): t for t in tone_styles()}
    if tone.lower() not in _known:
        raise EditError(
            f"{tone!r} is not a delivery style this engine accepts. "
            f"Pick one of: {', '.join(tone_styles())}")
    tone = _known[tone.lower()]
    rewrite = op.params.get("rewrite_text")
    rewrite = True if rewrite is None else bool(rewrite)
    by_id = {str(s.get("id")): s for s in data.get("segments", [])}

    engine = (ctx.tts_engine or "").lower()
    if engine and engine not in ("bodhan", "indic-speak"):
        applied.warnings.append(
            f"tts engine {engine!r} ignores the style hint "
            f"(only the Bodhan adapter forwards it); the tone change will "
            f"come from the rewritten wording alone")

    for sid in seg_ids:
        seg = by_id[sid]
        _backup(seg, "tts_style")
        seg["tts_style"] = tone
        applied.changes.append(f"{sid}: style={tone}")

    # "Neutral" is the absence of a register, not a register to write toward.
    # `models.tts.resolve_style` turns the sentinel into "send no style", and
    # there is nothing for an LLM to rewrite the wording INTO -- asking it to
    # would get a paraphrase at best and, because a paraphrase is a different
    # length, a different timing fit for no stated reason.
    from backend.tts import NEUTRAL_STYLE
    if rewrite and tone.strip().lower() == NEUTRAL_STYLE:
        applied.warnings.append(
            "neutral removes the delivery style rather than setting one, so "
            "the wording was left exactly as it is")
        rewrite = False

    if rewrite:
        if ctx.rewriter is None:
            applied.warnings.append(
                "tone rewrite skipped — OPENROUTER_API_KEY unset; only the "
                "engine style hint was set")
            return
        lang = ctx.target_language or "the target language"
        for sid in seg_ids:
            seg = by_id[sid]
            cur = _seg_text(seg)
            if not cur.strip():
                continue
            from backend.common.prompts import PROMPTS as _PROMPTS
            system = _PROMPTS["tone_rewrite"]
            user = (f"Language: {lang}\nRegister: {tone}\nLine: {cur}")
            new = _clean_rewrite(ctx.rewriter(system, user))
            if not new or new == cur:
                continue
            _set_segment_text(seg, new)
            applied.changes.append(f"{sid}: {cur!r} → {new!r}")


def _language_name(code, default="the source language"):
    """ISO code -> English language name, or the code itself if unknown."""
    try:
        from server.utils.langs import english_name
    except Exception:                                   # noqa: BLE001
        return code or default
    return english_name(code, default)


def _h_code_mix(data, seg_ids, op, applied, ctx, res):
    src_name = _language_name(ctx.source_language)
    terms = op.params.get("terms") or []
    if isinstance(terms, str):
        terms = [t.strip() for t in terms.replace(",", " ").split() if t.strip()]
    mode = str(op.params.get("mode") or "").strip().lower()
    if mode not in ("stage", "llm"):
        # Stage 9b is the designed path for this edit — it has the termbase,
        # the retain-span detector and the script normaliser. The LLM rewrite
        # is the fallback for when no transliterator can be resolved.
        mode = "stage"

    termbase = op.params.get("termbase") or ctx.termbase_path
    if termbase:
        res.termbase_path = Path(termbase)

    if terms and mode != "llm":
        # The transliterator's termbase is the documented way to force a term
        # to survive translation, so write one rather than inventing a second
        # mechanism: `backend.codemix.load_termbase` reads `source<TAB>repl`,
        # and a row mapping a term to itself is what keeps it untranslated.
        res.termbase_path = _write_termbase(
            terms, base=res.termbase_path, work_dir=ctx.work_dir,
            dry_run=ctx.dry_run)
        applied.changes.append(
            f"terms kept in {src_name}: {', '.join(terms)} "
            f"(termbase {res.termbase_path})")
    elif terms:
        applied.changes.append(f"terms kept in {src_name}: {', '.join(terms)}")

    if mode == "stage":
        model = ctx.translit_model
        if not model:
            # The job never code-mixed, so there is no model recorded. Take
            # the registry's first choice rather than refusing: stage 9b is
            # the designed path for this edit, and an LLM rewrite of every
            # line is a bigger change than the user asked for.
            try:
                from backend.translate.translit import available_models
                model = (available_models() or [None])[0]
            except Exception as e:                      # registry unimportable
                log.info("translit registry unavailable: %s", e)
                model = None
            if model:
                applied.warnings.append(
                    f"this job never code-mixed; using translit={model}. "
                    f"It needs its credentials in .env — pass "
                    f"mode=llm to rewrite with the LLM instead.")
        if model:
            res.translit_model = model
            applied.changes.append(
                f"stage 9b will re-run with translit={model}")
            applied.dirty_from = "codemix"
        else:
            applied.warnings.append(
                "no transliteration model available; falling back to the "
                "LLM rewrite")
            mode = "llm"

    if mode == "llm":
        if ctx.rewriter is None:
            raise EditError(
                "code_mix(mode=llm) needs the LLM (OPENROUTER_API_KEY unset), "
                "and no transliteration model is configured for this job.")
        applied.dirty_from = "tts"      # text is rewritten here and now
        by_id = {str(s.get("id")): s for s in data.get("segments", [])}
        lang = ctx.target_language or "the target language"
        for sid in seg_ids:
            seg = by_id[sid]
            cur = _seg_text(seg)
            src = seg.get("text") or ""
            if not cur.strip():
                continue
            from backend.common.prompts import PROMPTS as _PROMPTS
            system = (_PROMPTS["code_mix"]
                      .replace("{{source_language}}", src_name))
            user = (f"Target language: {lang}\n"
                    f"Source line ({src_name}): {src}\n"
                    f"Current translation: {cur}\n"
                    + (f"Always keep these in {src_name}: "
                       f"{', '.join(terms)}\n" if terms else ""))
            new = _clean_rewrite(ctx.rewriter(system, user))
            if not new or new == cur:
                continue
            _set_segment_text(seg, new)
            applied.changes.append(f"{sid}: {cur!r} → {new!r}")


def _write_termbase(terms: Sequence[str], *, base: Optional[Path],
                    work_dir: Optional[Path], dry_run: bool = False) -> Path:
    """Write `terms` as a termbase TSV, carrying over an existing one.

    A row is `term<TAB>term`: `backend.codemix` treats the second column as
    the exact string to put back, so mapping a term to itself is how it
    survives translation in Latin script.
    """
    rows: Dict[str, str] = {}
    if base and Path(base).exists():
        for ln in Path(base).read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                rows[parts[0].strip()] = parts[1].strip()
    for t in terms:
        rows.setdefault(str(t).strip(), str(t).strip())
    out_dir = Path(work_dir) if work_dir else Path.cwd()
    out = out_dir / "edit_termbase.tsv"
    if dry_run:
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# written by backend.edits — one `source<TAB>replacement` per line\n"
        + "".join(f"{k}\t{v}\n" for k, v in sorted(rows.items())),
        encoding="utf-8")
    return out


def _h_set_gender(data, seg_ids, op, applied, ctx, res):
    gender = str(op.params.get("gender") or "").strip().lower()
    engine = ctx.tts_engine or ctx.voice_engine
    lang = ctx.target_language
    by_id = {str(s.get("id")): s for s in data.get("segments", [])}

    # Speaker-wide when the scope named a speaker or selected every one of
    # that speaker's cues; otherwise the override is per segment.
    speakers = {by_id[s].get("speaker") for s in seg_ids if by_id.get(s)}
    speakers.discard(None)
    whole_speaker = bool(op.scope.speaker) or op.scope.all

    voice, why = voice_for_gender(
        gender, engine=engine, language=lang,
        catalog_json=ctx.voice_catalog_json)
    if voice is None:
        raise EditError(f"cannot honour gender={gender!r}: {why}")

    if whole_speaker:
        # One voice per speaker, not one voice for all of them: handing two
        # speakers the same voice turns a dialogue into one person reading
        # both parts, which is the thing `allow_voice_reuse=False` exists to
        # stop upstream. When the pool runs out the reuse is reported.
        taken: List[str] = []
        for spk in sorted(str(x) for x in speakers):
            v, w = voice_for_gender(
                gender, engine=engine, language=lang,
                catalog_json=ctx.voice_catalog_json, exclude=taken)
            if v is None:
                v, w = voice, f"{why} (reused — no further {gender} voice free)"
                applied.warnings.append(f"{spk}: {w}")
            taken.append(v)
            res.gender_overrides[spk] = gender
            res.speaker_voice_overrides[spk] = {
                "engine": engine, "voice": v, "source": "edit:set_gender",
            }
            applied.changes.append(f"{spk}: gender={gender}, voice={v}")
        applied.dirty_from = "voices"
    else:
        for sid in seg_ids:
            seg = by_id[sid]
            _backup(seg, "tts_voice")
            _backup(seg, "edit_gender")
            seg["tts_voice"] = voice
            seg["edit_gender"] = gender
            applied.changes.append(f"{sid}: gender={gender}, voice={voice}")
        applied.dirty_from = "tts"
    applied.changes.append(why)


def _h_set_speaker(data, seg_ids, op, applied, ctx, res):
    """Move scoped cues onto another speaker.

    This is the one op that edits `seg["speaker"]`, which is upstream of
    everything voice-related: `backend/tts.py` resolves each segment through
    `speaker_voice_map[seg["speaker"]]`, so moving a cue moves which voice
    reads it. That is the whole point -- diarization attributing a line to
    the wrong person is the failure this corrects, and no other op can.

    `dirty_from="voices"` rather than `"tts"` because the speaker set itself
    can change: moving the last cue off a speaker leaves it with no segments,
    and stage 9d re-derives the map from what is actually there.

    The target must already exist in the job. Inventing a speaker id would
    give it no reference clip, no gender vote and no catalog voice, so TTS
    would fall back to the engine default and read it in a voice nobody
    chose -- the silent-wrong-output shape CLAUDE.md records for
    `--voice-engine`.
    """
    target = str(op.params.get("speaker") or "").strip()
    if not target:
        raise EditError("set_speaker needs a speaker id, e.g. SPK_02")

    by_id = {str(s.get("id")): s for s in data.get("segments", [])}
    known = {str(s.get("speaker")) for s in data.get("segments", [])
             if s.get("speaker")}
    if target not in known:
        raise EditError(
            f"unknown speaker {target!r}; this job has "
            f"{', '.join(sorted(known)) or 'no labelled speakers'}. "
            f"A new id would have no reference clip and no assigned voice.")

    moved = 0
    for sid in seg_ids:
        seg = by_id.get(sid)
        if seg is None:
            continue
        before = seg.get("speaker")
        if str(before) == target:
            continue
        _backup(seg, "speaker")
        # The per-segment voice override is whatever the PREVIOUS speaker was
        # pinned to. Leaving it would silently defeat the reattribution: the
        # cue would move to the new speaker and still be read in the old
        # one's voice, because `tts.py` reads `tts_voice` ahead of the map.
        if seg.get("tts_voice"):
            _backup(seg, "tts_voice")
            seg["tts_voice"] = None
        seg["speaker"] = target
        moved += 1
        applied.changes.append(f"{sid}: speaker {before} -> {target}")

    if not moved:
        applied.warnings.append(
            f"every scoped cue was already {target}; nothing to move")
    else:
        # Say when a speaker has been emptied: 9d will drop it, the voice it
        # held is freed, and the operator's speaker list is about to change.
        left = {str(s.get("speaker")) for s in data.get("segments", [])
                if s.get("speaker")}
        for gone in sorted(known - left):
            applied.warnings.append(
                f"{gone} now has no cues and will disappear from the cast")
    applied.dirty_from = "voices"


def _h_set_voice(data, seg_ids, op, applied, ctx, res):
    voice = str(op.params.get("voice") or "").strip()
    if not voice:
        raise EditError("set_voice needs a `voice`")
    # set_voice is now a speaker-scoped operation: the LLM prompt says
    # "Requires Speaker id explicitly" and the handler enforces the same.
    # A per-cue voice swap that does not change the speaker is almost
    # always what create_speaker was added for -- the cue stays on its
    # existing speaker and that speaker gets re-voiced, which is a
    # cast-table change dressed as a cue edit. The explicit refusal here
    # makes the misuse visible rather than silently wholesale-re-voicing
    # that speaker from one cue's intent.
    if op.scope.all:
        raise EditError(
            "set_voice needs a speaker scope. A whole-dub re-voice would "
            "pin the same voice onto every speaker. Name the speaker, or "
            "use create_speaker for a stretch of cues.")
    if not op.scope.speaker:
        raise EditError(
            "set_voice needs a speaker scope, not a cue or time window. "
            "If you want a cue stretch to be read in a different voice, "
            "use create_speaker instead.")
    engine = op.params.get("engine") or ctx.tts_engine or ctx.voice_engine
    by_id = {str(s.get("id")): s for s in data.get("segments", [])}
    canonical, cat_gender, why = _voice_catalog_lookup(
        voice, engine=engine, language=ctx.target_language,
        catalog_json=ctx.voice_catalog_json)
    if canonical is None:
        applied.warnings.append(
            f"{voice!r} is not in the {engine!r} catalog ({why}); passing it "
            f"through anyway — the engine will reject it if it is wrong")
    else:
        voice = canonical

    # The voice carries the gender: writing `gender_overrides` here is how
    # the merged NLQ flow collapses what used to be a separate `set_gender`.
    speakers = {by_id[s].get("speaker") for s in seg_ids if by_id.get(s)}
    speakers.discard(None)
    for spk in speakers:
        res.speaker_voice_overrides[str(spk)] = {
            "engine": engine, "voice": voice, "source": "edit:set_voice"}
        if cat_gender:
            res.gender_overrides[str(spk)] = cat_gender
            applied.changes.append(
                f"{spk}: voice={voice}, gender={cat_gender}")
        else:
            applied.changes.append(f"{spk}: voice={voice}")
    applied.dirty_from = "voices"


def _voice_catalog_lookup(
    voice: str, *, engine: Optional[str],
    language: Optional[str], catalog_json: Optional[Path],
) -> Tuple[Optional[str], Optional[str], str]:
    """Case-insensitive catalog lookup that also returns the voice's gender.

    `_known_voice` returns only the canonical name; `_h_set_voice` and
    `_h_create_speaker` both need the gender too, so the lookup is factored
    out rather than done twice. Returns `(canonical, gender, why)` with
    `canonical` None when the voice is not in the catalog.
    """
    eng = (engine or "").lower()
    try:
        from backend.tts.voices import load_voice_catalog
        cat = load_voice_catalog(
            "indic-speak" if eng in ("", "bodhan", "indic-speak") else eng,
            language=language, catalog_json=catalog_json)
        match = next((v for v in cat if v.voice.lower() == voice.lower()),
                     None)
        if match is None:
            return None, None, f"{len(cat)} voices known"
        return (match.voice, (match.gender or None),
                f"{len(cat)} voices known")
    except Exception as e:                       # catalog is advisory only
        return voice, None, f"catalog unreadable: {e}"


def _h_create_speaker(data, seg_ids, op, applied, ctx, res):
    """Mint a new speaker id, move the scoped cues onto it, assign the voice.

    Required: `voice` and a scope that resolves to at least one cue, that is
    NOT the whole dub and NOT a bare speaker. Optional: `gender` (follows the
    voice when omitted) and `tone` (defaults to neutral).

    `dirty_from="voices"`: 9d reads `speaker_voice_overrides` and the newly
    minted id surfaces in the cast. No reference clip is written -- the voice
    override is read ahead of the reference-based pick, so TTS does not need
    one. Any source speaker left with no cues is reported, same as
    `_h_set_speaker`, because the operator's cast list is about to change.
    """
    voice = str(op.params.get("voice") or "").strip()
    if not voice:
        raise EditError(
            "create_speaker needs a `voice` -- a new speaker with no voice "
            "has no way to be read, and the pipeline would fall back to the "
            "engine default silently. Name the voice: "
            "Deepika, Chinmay, Adarsh, etc.")

    # Scope: refuse `all` and a bare speaker. Both would mean "mint a speaker
    # for every cue there is" or "rename an existing speaker", neither of
    # which is what this op is for. A time window or cue list is the only
    # valid shape -- the LLM prompt says the same, this is the defense.
    if op.scope.all:
        raise EditError(
            "create_speaker needs a time range or cue list; `all` would "
            "mint a speaker that owns the whole dub")
    if op.scope.speaker and not (op.scope.start_s is not None
                                  or op.scope.end_s is not None
                                  or op.scope.segments):
        raise EditError(
            "create_speaker needs a time range or cue list; a bare speaker "
            "scope is a rename, not a creation -- use set_speaker instead")
    if not seg_ids:
        raise EditError(
            "create_speaker: the scope resolved to no cues; name a time "
            "range or cue id the new speaker will own")

    engine = op.params.get("engine") or ctx.tts_engine or ctx.voice_engine
    canonical, cat_gender, why = _voice_catalog_lookup(
        voice, engine=engine, language=ctx.target_language,
        catalog_json=ctx.voice_catalog_json)
    if canonical is None:
        # Hard refuse rather than warn: the whole point of the op is to pin
        # a known voice to a new speaker. A pass-through here is the silent-
        # wrong-output trap this op was added to AVOID.
        raise EditError(
            f"create_speaker: {voice!r} is not in the {engine!r} catalog "
            f"({why})")
    voice = canonical

    gender_param = str(op.params.get("gender") or "").strip().lower() or None
    if gender_param and gender_param not in ("male", "female"):
        raise EditError(
            f"create_speaker: gender must be 'male' or 'female', "
            f"got {gender_param!r}")
    gender = gender_param or cat_gender

    tone = str(op.params.get("tone") or "").strip() or None
    if tone:
        known_tones = {t.lower(): t for t in tone_styles()}
        canon = known_tones.get(tone.lower())
        if canon is None:
            raise EditError(
                f"create_speaker: tone {tone!r} is not an engine style. "
                f"Pick one of: {', '.join(tone_styles())}")
        tone = canon

    # Mint the new id. Scan the DIARIZATION label space (SPK_NN / SPEAKER_NN)
    # for the biggest number seen, pick next, two-digit zero-pad. Width is
    # fixed at SPK_0N so the editor's own ids stay distinguishable from the
    # upstream labels when both widths are read later.
    known = {str(s.get("speaker")) for s in data.get("segments", [])
             if s.get("speaker")}
    used_nums: List[int] = []
    for name in known:
        m = re.match(r"(?:SPK|SPEAKER)[_ ]?(\d+)$", name, re.I)
        if m:
            used_nums.append(int(m.group(1)))
    next_num = (max(used_nums) + 1) if used_nums else 0
    new_id = f"SPK_{next_num:02d}"

    by_id = {str(s.get("id")): s for s in data.get("segments", [])}
    moved = 0
    for sid in seg_ids:
        seg = by_id.get(sid)
        if seg is None:
            continue
        _backup(seg, "speaker")
        # The previous speaker's per-cue voice override would silently defeat
        # the new assignment -- same reasoning as _h_set_speaker.
        if seg.get("tts_voice"):
            _backup(seg, "tts_voice")
            seg["tts_voice"] = None
        seg["speaker"] = new_id
        if tone and tone.lower() != "neutral":
            _backup(seg, "tts_style")
            seg["tts_style"] = tone
        moved += 1

    res.speaker_voice_overrides[new_id] = {
        "engine": engine, "voice": voice, "source": "edit:create_speaker"}
    if gender:
        res.gender_overrides[new_id] = gender

    # Edit-only op: 9d is NOT replaying, so the voices/*.voices.json file the
    # cast panel reads will not grow a row for `new_id` on its own. Write the
    # assignment where `_from_speaker_voice_map` already looks for it -- the
    # merged JSON's `extra.speaker_voice_map` -- and patch every on-disk
    # `voices/*.gender.json` with the new speaker's gender, so the cast table
    # shows the voice and gender immediately rather than "needs a voice".
    svm = data.setdefault("extra", {}).setdefault("speaker_voice_map", {})
    svm[new_id] = {"engine": engine, "voice": voice,
                   "gender": gender, "source": "edit:create_speaker"}

    if not ctx.dry_run:
        voices_dir = res.merged_json.parent.parent / "voices"
        if voices_dir.is_dir() and gender:
            for gpath in sorted(voices_dir.glob("*.gender.json")):
                try:
                    doc = json.loads(gpath.read_text())
                    if not isinstance(doc, dict):
                        continue
                    doc[new_id] = {
                        "gender": gender,
                        # `pipeline.py` builds a vote entry for every speaker
                        # from this field and then reads `v["gender"]`. Without
                        # it the dict defaults to `{}` and run_post_mt_stages
                        # crashes with KeyError('gender') the next time
                        # TTS/mix replays.
                        "gender_vote": {
                            "gender": gender,
                            "decided_by": "override",
                            "votes": {},
                            "tally": {"male": 0, "female": 0},
                        },
                        "gender_basis": "edit",
                        "source": "edit:create_speaker",
                    }
                    tmp = gpath.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(doc, indent=2,
                                              ensure_ascii=False))
                    tmp.replace(gpath)
                except (OSError, ValueError) as e:
                    log.warning("create_speaker: could not patch %s: %s",
                                gpath, e)

    applied.changes.append(
        f"{new_id}: minted, voice={voice}"
        + (f", gender={gender}" if gender else "")
        + (f", tone={tone}" if tone and tone.lower() != "neutral" else "")
        + f", moved {moved} cue(s)")
    if not gender:
        applied.warnings.append(
            f"{new_id}: no gender override written -- catalog has no "
            f"gender for {voice!r} and the request named none")

    left = {str(s.get("speaker")) for s in data.get("segments", [])
            if s.get("speaker")}
    for gone in sorted(known - left):
        applied.warnings.append(
            f"{gone} now has no cues and will disappear from the cast")

    # Edit-only op: the speaker map is updated, nothing else runs. The user
    # triggers TTS + mix separately via the "Start TTS & mix" button.
    applied.dirty_from = ""
    applied.warnings.append(
        "speaker map updated; TTS and mix were NOT re-rendered. "
        "Run them from the cast panel when ready.")


def _known_voice(voice: str, *, engine: Optional[str],
                 language: Optional[str],
                 catalog_json: Optional[Path]) -> Tuple[Optional[str], str]:
    """Case-insensitive catalog lookup.  Returns ``(canonical_name, why)``
    where *canonical_name* is the catalog spelling when found, else ``None``.
    """
    eng = (engine or "").lower()
    try:
        from backend.tts.voices import load_voice_catalog
        cat = load_voice_catalog(
            "indic-speak" if eng in ("", "bodhan", "indic-speak") else eng,
            language=language, catalog_json=catalog_json)
        match = next((v.voice for v in cat
                      if v.voice.lower() == voice.lower()), None)
        return match, f"{len(cat)} voices known"
    except Exception as e:                       # catalog is advisory only
        return voice, f"catalog unreadable: {e}"


def _h_set_speed(data, seg_ids, op, applied, ctx, res):
    try:
        factor = float(op.params.get("factor")
                       or op.params.get("speed") or 0.0)
    except (TypeError, ValueError) as e:
        raise EditError(f"set_speed: bad factor {op.params.get('factor')!r}") from e
    if factor <= 0:
        raise EditError("set_speed needs a positive `factor` (1.2 = 20% faster)")
    if not 0.5 <= factor <= 2.0:
        raise EditError(
            f"set_speed factor {factor} is outside 0.5–2.0; beyond that the "
            f"stretch stops being intelligible")

    by_id = {str(s.get("id")): s for s in data.get("segments", [])}
    # The stretch cap is what silently clamps a big speed change, so raise it
    # for exactly these segments and leave every other segment's budget alone.
    needed = abs(factor - 1.0)
    cap = max(_DEFAULT_MAX_STRETCH, needed + 0.05)
    for sid in seg_ids:
        seg = by_id[sid]
        slot = seg.get("target_duration_s")
        if slot is None:
            slot = _seg_end(seg) - _seg_start(seg)
        slot = float(slot or 0.0)
        if slot <= 0:
            applied.warnings.append(f"{sid}: no usable slot length; skipped")
            continue
        _backup(seg, "target_duration_s")
        _backup(seg, "tts_max_stretch")
        _backup(seg, "tts_rate_override")
        seg["target_duration_s"] = round(slot / factor, 4)
        seg["tts_max_stretch"] = round(cap, 4)
        seg["tts_rate_override"] = round(factor, 4)
        applied.changes.append(
            f"{sid}: slot {slot:.2f}s → {seg['target_duration_s']:.2f}s "
            f"({factor:.2f}x)")
    applied.changes.append(
        "speech is re-fitted inside its original slot, so the timeline and "
        "the video stay aligned; a faster clip ends early and leaves silence")


def _h_lipsync_window(data, seg_ids, op, applied, ctx, res):
    start = op.scope.start_s
    end = op.scope.end_s
    if start is None or end is None:
        if seg_ids:
            by_id = {str(s.get("id")): s for s in data.get("segments", [])}
            picked = [by_id[s] for s in seg_ids]
            start = min(_seg_start(s) for s in picked)
            end = max(_seg_end(s) for s in picked)
        else:
            raise EditError(
                "lipsync_window needs a time range (start and end)")
    start, end = sorted((float(start), float(end)))
    if end - start < 0.5:
        raise EditError(
            f"lipsync window {end - start:.2f}s is too short to render")
    res.lipsync_window = (start, end)
    model = op.params.get("model") or ctx.lipsync_model
    if model:
        from backend.lipsync.lipsync import available_models
        known = available_models()
        if str(model) not in known:
            raise EditError(
                f"unknown lip-sync model {model!r}. "
                f"Available: {', '.join(known)}")
        res.lipsync_model = str(model)
    applied.segments = seg_ids
    applied.changes.append(
        f"lip-sync {_hhmmss(start)}–{_hhmmss(end)}"
        + (f" with {model}" if model else "")
        + "; the rest of the video is passed through untouched")


def _h_mute_cue(data, seg_ids, op, applied, ctx, res):
    """Silence cues and hand their time back to the timeline.

    Muting is the only way to drop a cue, and it does both halves of that.
    Mix skips the segment, so nothing is spoken; and `timing` takes the
    slot out of `_next_slot_start`'s lookahead, so the freed seconds merge
    into the gap after the preceding cue instead of sitting behind a
    boundary nobody may cross.

    That second half is what makes the silence USABLE rather than merely
    silent. Whether it is actually spent is the timing layer's decision and
    stays there: `--speed-borrow-gap` is 0.0 by default, so by default a
    muted cue reads as clean M&E bed exactly as it always did. Turn borrow
    on and the same seconds become budget for a neighbouring line that
    would otherwise have been sped up. Nothing here forces that trade.

    `dirty_from` is `timing`, not `tts`: TTS still synthesises muted cues,
    so the clip is there for a revert and a mute costs no Bodhan calls.
    `timing` rather than `mix` because the freed slot has to be re-planned
    before mix drops the clip.

    The text is left in the document rather than blanked: `revert` needs
    it, and a cue the operator wants back is a `--revert`, not a
    re-translation.
    """
    by_id = {str(s.get("id")): s for s in data.get("segments", [])}
    for sid in seg_ids:
        seg = by_id.get(sid)
        if seg is None:
            continue
        _backup(seg, "muted")
        seg["muted"] = True
        freed = max(0.0, _seg_end(seg) - _seg_start(seg))
        applied.changes.append(
            f"{sid}: muted, {freed:.2f}s released as pause")
    applied.dirty_from = "timing"


def _h_set_target_language(data, seg_ids, op, applied, ctx, res):
    """Swap the target language on scoped cues via Bodhan MT.

    Two paths, chosen by scope:

    - **Whole dub** (`scope.all`): re-translate every cue, swap
      `data["target_language"]`, clear every voice pin, and let the post-MT
      replay re-cast voices and re-speak from scratch. `dirty_from="codemix"`
      — codemix → duration → voices → tts → timing → mix → lipsync.
    - **Cue list or time window**: re-translate only those cues, pin a
      per-cue voice from the new language's catalog, and leave the global
      target language and the other cues untouched. `dirty_from="tts"` — the
      voices stage is skipped because the voice is already chosen.

    **Speaker-only scope is refused.** A bare speaker has no time bound; a
    mixed-language result keyed by diarizer labels is unpredictable, and the
    operator usually means "the speaker, in this window" or "the whole dub".
    Say so rather than guess.

    MT itself is `models.mt.get_mt("bodhan")`, called once with the full
    request list — the Bodhan adapter paces its own loop. `BODHAN_MT_KEY` is
    required; without it the adapter raises on the first call and we surface
    that as a clean `EditError` rather than a backtrace. The LLM is not
    touched: this op reaches the pipeline's native translator, not a rewrite.
    """
    iso = str(op.params.get("target_language") or "").strip().lower()
    if not iso:
        raise EditError(
            "set_target_language needs a 'target_language' ISO code, e.g. "
            "hi, mr, ta")

    # One source of truth for the catalogue: the same table the home-page
    # dropdown reads from. Avoids a second list drifting the first time an
    # adapter gains or loses support for a language.
    from server.utils.langs import LANGUAGES
    meta = LANGUAGES.get(iso)
    indic = sorted(k for k, v in LANGUAGES.items() if v["script"] != "latn")
    if meta is None or meta["script"] == "latn":
        raise EditError(
            f"unknown or non-Indic target language {iso!r}. "
            f"Supported: {', '.join(indic)}")

    # Scope guard: a bare speaker is refused. `scope.all` is the whole-dub
    # path; segments / time windows / speaker+time all have cue bounds and
    # are honoured as the per-cue path.
    sc = op.scope
    speaker_only = (sc.speaker and not sc.all
                    and not sc.segments
                    and sc.start_s is None and sc.end_s is None)
    if speaker_only:
        raise EditError(
            "set_target_language cannot scope to a speaker alone — add a "
            "time range (e.g. 'SPK_01 from 0:30 to 0:45'), or use the whole "
            "dub.")

    script = op.params.get("target_script") or None

    try:
        from backend.translate import get_mt
        from backend.translate.base import MTRequest
        bodhan = get_mt("bodhan")
    except Exception as e:
        raise EditError(f"Bodhan MT adapter is not available: {e}") from e

    by_id = {str(s.get("id")): s for s in data.get("segments", [])}
    requests: List = []
    for sid in seg_ids:
        seg = by_id.get(sid)
        if seg is None:
            continue
        src = (seg.get("text") or "").strip()
        if not src:
            applied.warnings.append(f"{sid}: no source text, left alone")
            continue
        requests.append(MTRequest(
            id=sid, text=src,
            source_language=ctx.source_language,
            target_script=script,
        ))

    if not requests:
        raise EditError(
            "set_target_language: none of the scoped cues carry source text "
            "to translate")

    try:
        result = bodhan.translate(
            requests,
            target_language=iso,
            target_script=script,
            source_language=ctx.source_language,
        )
    except Exception as e:
        raise EditError(f"Bodhan MT refused the translation: {e}") from e

    for mt_seg in result.segments:
        seg = by_id.get(mt_seg.id)
        if seg is None:
            continue
        _set_segment_text(seg, mt_seg.text or "")
        _backup(seg, "edit_target_language")
        seg["edit_target_language"] = iso

    whole = bool(sc.all)
    if whole:
        data["target_language"] = iso
        if script:
            data["target_script"] = script
        for seg in data.get("segments") or []:
            if seg.get("tts_voice"):
                _backup(seg, "tts_voice")
                seg["tts_voice"] = None
        res.target_language_override = iso
        if script:
            res.target_script_override = script
        applied.dirty_from = "codemix"
        applied.changes.append(
            f"target language -> {iso} ({meta['english']} / {meta['native']}); "
            f"{len(requests)} cue(s) re-translated, voices cleared for re-cast")
    else:
        engine = ctx.tts_engine or ctx.voice_engine
        pinned = 0
        for mt_seg in result.segments:
            seg = by_id.get(mt_seg.id)
            if seg is None:
                continue
            raw_gender = (seg.get("gender") or seg.get("edit_gender")
                          or "female")
            gender = str(raw_gender).lower()
            if gender not in ("male", "female"):
                gender = "female"
            try:
                voice, why = voice_for_gender(
                    gender, engine=engine, language=iso,
                    catalog_json=ctx.voice_catalog_json)
            except EditError as e:
                applied.warnings.append(f"{mt_seg.id}: {e}")
                continue
            if voice is None:
                applied.warnings.append(
                    f"{mt_seg.id}: {why}; left on the original voice")
                continue
            _backup(seg, "tts_voice")
            seg["tts_voice"] = voice
            pinned += 1
            applied.changes.append(
                f"{mt_seg.id}: -> {iso}, voice={voice}")
        applied.dirty_from = "tts"
        applied.changes.insert(
            0,
            f"{len(requests)} cue(s) re-translated to {iso} "
            f"({meta['english']} / {meta['native']}); "
            f"pinned {pinned} native voice(s)")


#: Maximum silence between two cues that still counts as "consecutive" for a
#: merge. Beyond this the user asked to splice across a real pause -- a merge
#: here would widen one cue over a stretch where the picture shows nobody
#: speaking, so `_h_merge_cues` refuses.
MERGE_MAX_GAP_S = 2.0


def _h_merge_cues(data, seg_ids, op, applied, ctx, res):
    """Merge two consecutive cues of the same speaker into one.

    Refuses (as EditError) unless all of the following hold:

      * exactly two cue ids are in scope;
      * both ids exist in the cue list;
      * they are INDEX-adjacent in `data["segments"]` (not merely time-close);
      * `speaker` matches on both;
      * `cue2.start - cue1.end <= MERGE_MAX_GAP_S` (2.0s).

    The first cue absorbs the second: originals are space-joined, timing
    extends to `cue2.end`, and the translation is rewritten by the LLM from
    both originals plus both translations so it reads as a single utterance
    in the target language. The first cue keeps its id, speaker, register
    and tone. Cue2 is removed from the live segments list.

    Dirties from `codemix`, so the replay re-runs 9c/9c2, TTS, mix and
    lip-sync for the merged window — same scope as `retranslate`.
    """
    if len(seg_ids) != 2:
        raise EditError(
            "merge_cues needs exactly two cue ids in its scope")
    segs = data.get("segments", [])
    idx_by_id = {str(s.get("id")): i for i, s in enumerate(segs)}
    missing = [s for s in seg_ids if s not in idx_by_id]
    if missing:
        raise EditError(
            f"merge_cues: unknown cue(s): {', '.join(missing)}")

    sid_a, sid_b = seg_ids[0], seg_ids[1]
    i_a, i_b = idx_by_id[sid_a], idx_by_id[sid_b]
    # Accept the pair in either order from the scope; normalise to
    # earlier-then-later so the merge direction is deterministic.
    if i_b < i_a:
        i_a, i_b = i_b, i_a
        sid_a, sid_b = sid_b, sid_a
    if i_b != i_a + 1:
        raise EditError(
            f"merge_cues: {sid_a} and {sid_b} are not consecutive")

    seg_a, seg_b = segs[i_a], segs[i_b]
    spk_a = str(seg_a.get("speaker") or "")
    spk_b = str(seg_b.get("speaker") or "")
    if spk_a != spk_b:
        raise EditError(
            f"merge_cues: speakers differ "
            f"({spk_a or '?'} vs {spk_b or '?'})")

    gap = _seg_start(seg_b) - _seg_end(seg_a)
    if gap > MERGE_MAX_GAP_S:
        raise EditError(
            f"merge_cues: {gap:.2f}s gap between {sid_a} and {sid_b} "
            f"(max {MERGE_MAX_GAP_S:.1f}s)")

    if ctx.rewriter is None:
        raise EditError(
            "merge_cues needs the LLM (OPENROUTER_API_KEY unset)")

    src_a = (seg_a.get("text") or "").strip()
    src_b = (seg_b.get("text") or "").strip()
    cur_a = _seg_text(seg_a).strip()
    cur_b = _seg_text(seg_b).strip()
    lang = (ctx.target_language
            or data.get("target_language")
            or "the target language")

    from backend.common.prompts import PROMPTS as _PROMPTS
    system = _PROMPTS["merge_cues"]
    user = (
        f"Target language: {lang}\n"
        f"Speaker: {spk_a or 'unknown'}\n"
        f"Original 1: {src_a}\n"
        f"Original 2: {src_b}\n"
        f"Current translation 1: {cur_a}\n"
        f"Current translation 2: {cur_b}\n"
        f"Write the merged {lang} line.")
    merged = _clean_rewrite(ctx.rewriter(system, user))
    if not merged:
        raise EditError("merge_cues: model returned no text")

    # Backup the fields a revert needs. Cue2 is backed up wholesale so the
    # versioned snapshot in `mt/versions/` is the only recovery path for the
    # deletion; the live revert() can at least restore cue1's shape.
    for key in ("text", "end", "end_s"):
        _backup(seg_a, key)

    # Join originals and extend timing to cue2's end. Both end keys are kept
    # in sync when either side carried them, so downstream readers (speed
    # fit, TTS, mix) see a coherent cue regardless of which key they prefer.
    seg_a["text"] = (src_a + " " + src_b).strip()
    _set_segment_text(seg_a, merged)
    end_b = _seg_end(seg_b)
    if "end" in seg_a or "end" in seg_b:
        seg_a["end"] = end_b
    if "end_s" in seg_a or "end_s" in seg_b:
        seg_a["end_s"] = end_b

    # Remove cue2 from the live list. The op's `applied.segments` records
    # both ids so the journal shows what disappeared.
    segs.pop(i_b)

    applied.changes.append(
        f"{sid_a} + {sid_b} -> {sid_a}: "
        f"{cur_a!r} + {cur_b!r} -> {merged!r} "
        f"(end={end_b:.2f}s)")
    applied.warnings.append(
        f"{sid_b} was removed from the cue list; use mt/versions/ to revert")


def _mint_speaker_id(data: Dict[str, Any]) -> str:
    """Pick the next free SPK_NN id, matching `_h_create_speaker`'s scheme."""
    known = {str(s.get("speaker")) for s in data.get("segments", [])
             if s.get("speaker")}
    used: List[int] = []
    for name in known:
        m = re.match(r"(?:SPK|SPEAKER)[_ ]?(\d+)$", name, re.I)
        if m:
            used.append(int(m.group(1)))
    nxt = (max(used) + 1) if used else 0
    return f"SPK_{nxt:02d}"


def _split_text_proportional(text: str, left_fraction: float) -> Tuple[str, str]:
    """Cut `text` at the whitespace boundary closest to `left_fraction`.

    A deterministic fallback for when word-level timings are not available
    on a cue. Picks the whitespace split whose offset ratio is nearest the
    target, so the two halves are at least shaped like the time split.
    """
    text = text or ""
    stripped = text.strip()
    if not stripped:
        return "", ""
    frac = max(0.05, min(0.95, float(left_fraction)))
    # candidate split points are the whitespace runs
    boundaries = [i for i, ch in enumerate(text) if ch.isspace()]
    if not boundaries:
        # one unbroken token — give it to the left, nothing to the right.
        return text, ""
    target = int(round(len(text) * frac))
    cut = min(boundaries, key=lambda i: abs(i - target))
    return text[:cut].rstrip(), text[cut:].lstrip()


def _h_split_cue(data, seg_ids, op, applied, ctx, res):
    """Split one cue at a user-given timestamp into two cues.

    Scope resolves to exactly one cue (either an explicit SEG id, or the
    cue that contains `at_s`). The timestamp must fall strictly inside the
    cue — a split at the exact boundary is a no-op and refused rather than
    producing an empty cue.

    The left half keeps the original id, speaker, voice pins and
    translation; its end moves to `at_s`. The right half gets a fresh
    SEG id, start=`at_s`, the original end, and — unless `speaker` or
    `voice` is given — the same speaker as the left half.

    Word-level timings on the parent are partitioned by `end <= at_s` →
    left, else → right, and each half's `text` is rebuilt from its words.
    Without words we fall back to a proportional whitespace split.

    `mt_text` / `final_text` are split proportionally between the halves
    by the same time fraction as the cut, so BOTH halves have readable
    text immediately. The post-MT replay chain does not re-call Bodhan MT
    (only an explicit `retranslate` op does), and leaving the halves
    empty would send TTS into silence -- the "SEG shows blank" trap. A
    warning flags that the word boundary may not match the sentence
    boundary; `retranslate` on either half cleans the grammar up.

    `dirty_from="codemix"`: both halves need re-translation, re-fit,
    re-TTS, remix and re-sync. The right half is a brand-new cue so TTS
    has nothing cached for it; the left half's wording changed so its
    cached clip is invalid too.
    """
    if len(seg_ids) != 1:
        raise EditError(
            "split_cue needs exactly one cue in its scope "
            f"(got {len(seg_ids)}). Name the cue by id or give a "
            "timestamp inside it.")

    at_s = _as_time(op.params.get("at_s"))
    if at_s is None:
        # Fall back to the scope's point timestamp, if any.
        if op.scope.start_s is not None and op.scope.end_s is None:
            at_s = float(op.scope.start_s)
        elif (op.scope.start_s is not None and op.scope.end_s is not None
              and float(op.scope.start_s) == float(op.scope.end_s)):
            at_s = float(op.scope.start_s)
    if at_s is None:
        raise EditError(
            "split_cue needs `at_s` (seconds) — the moment to cut at")

    segs = data.get("segments") or []
    idx_by_id = {str(s.get("id")): i for i, s in enumerate(segs)}
    sid = seg_ids[0]
    if sid not in idx_by_id:
        raise EditError(f"split_cue: unknown cue {sid!r}")
    i_a = idx_by_id[sid]
    seg_a = segs[i_a]
    s0, s1 = _seg_start(seg_a), _seg_end(seg_a)
    MIN_HALF = 0.15
    if not (s0 + MIN_HALF <= at_s <= s1 - MIN_HALF):
        raise EditError(
            f"split_cue: at_s={at_s:.3f}s must sit at least "
            f"{MIN_HALF:.2f}s inside {sid} ({s0:.2f}–{s1:.2f}s)")

    # Resolve the right-half speaker policy.
    want_speaker = str(op.params.get("speaker") or "").strip() or None
    want_voice = str(op.params.get("voice") or "").strip() or None
    known_speakers = {str(s.get("speaker")) for s in segs if s.get("speaker")}
    orig_speaker = str(seg_a.get("speaker") or "") or None

    new_speaker_id: Optional[str] = None
    voice_assignment: Optional[Tuple[str, str, Optional[str]]] = None  # (engine, voice, gender)
    if want_speaker and want_voice:
        raise EditError(
            "split_cue: pass `speaker` OR `voice`, not both — pick an "
            "existing speaker, or mint a new one with a voice")
    if want_speaker:
        if want_speaker not in known_speakers:
            raise EditError(
                f"split_cue: unknown speaker {want_speaker!r} "
                f"(known: {', '.join(sorted(known_speakers)) or 'none'})")
        right_speaker = want_speaker
    elif want_voice:
        engine = op.params.get("engine") or ctx.tts_engine or ctx.voice_engine
        canonical, cat_gender, why = _voice_catalog_lookup(
            want_voice, engine=engine, language=ctx.target_language,
            catalog_json=ctx.voice_catalog_json)
        if canonical is None:
            raise EditError(
                f"split_cue: {want_voice!r} is not in the {engine!r} "
                f"catalog ({why})")
        new_speaker_id = _mint_speaker_id(data)
        voice_assignment = (engine, canonical, cat_gender)
        right_speaker = new_speaker_id
    else:
        right_speaker = orig_speaker or ""

    # Partition words by their END time: a word ending at or before `at_s`
    # belongs to the left half; the rest, including a word straddling the
    # cut, go right. Keeps the picture-locked boundary at a word boundary
    # on the LEFT at the cost of the right half starting slightly early;
    # the alternative (straddling word → left) would smear mouth timing on
    # the first word of the right half.
    words = list(seg_a.get("words") or [])
    if words:
        left_words = [w for w in words if _as_time(w.get("end")) is not None
                      and float(w["end"]) <= at_s + 1e-6]
        right_words = words[len(left_words):]
        left_text = " ".join((w.get("text") or "").strip()
                             for w in left_words).strip()
        right_text = " ".join((w.get("text") or "").strip()
                              for w in right_words).strip()
        if not left_text or not right_text:
            # Degenerate word timing: fall back to proportional split.
            left_words, right_words = [], []
            left_text, right_text = _split_text_proportional(
                str(seg_a.get("text") or ""),
                (at_s - s0) / max(1e-6, (s1 - s0)))
    else:
        left_text, right_text = _split_text_proportional(
            str(seg_a.get("text") or ""),
            (at_s - s0) / max(1e-6, (s1 - s0)))
        left_words, right_words = [], []

    # Mint the new id. Scan ALL existing SEG ids for the biggest number
    # seen and pick next (zero-padded to 4 digits — the width the pipeline
    # already writes). This keeps new ids out of collision with both
    # upstream SEG_NNNN and any earlier splits.
    seg_nums: List[int] = []
    for s in segs:
        m = re.match(r"SEG[_ ]?(\d+)$", str(s.get("id") or ""), re.I)
        if m:
            seg_nums.append(int(m.group(1)))
    new_seg_id = f"SEG_{(max(seg_nums) + 1) if seg_nums else 0:04d}"

    # Back up every field the left half overwrites, so revert() restores it.
    for key in ("end", "end_s", "text", "words", "mt_text", "final_text",
                "target_duration_s", "est_duration_s", "candidates", "qc"):
        if key in seg_a:
            _backup(seg_a, key)

    # Build the right half FIRST (from the parent's pre-mutation state) so
    # we can carry forward the fields that still make sense (speaker,
    # tts_style, language, mt_target_* ) without inheriting left-only
    # artefacts like `candidates` or `qc`.
    new_seg: Dict[str, Any] = {
        "id": new_seg_id,
        "start": at_s,
        "end": s1,
        "text": right_text,
        "speaker": right_speaker,
        "language": seg_a.get("language"),
    }
    if "end_s" in seg_a:
        new_seg["end_s"] = s1
    if "start_s" in seg_a:
        new_seg["start_s"] = at_s
    if right_words:
        new_seg["words"] = right_words
    for carry in ("mt_target_language", "mt_target_script",
                  "tts_style", "tts_engine"):
        if carry in seg_a:
            new_seg[carry] = seg_a[carry]
    # Translation split. The post-MT replay chain does NOT re-call Bodhan
    # MT -- only an explicit `retranslate` op does. If we cleared both
    # halves here, TTS would walk into empty `final_text` and the cues
    # would synthesise as silence, which is the "SEG shows blank in the
    # UI" bug this comment exists to prevent. So split the parent's
    # translation proportionally by the same time fraction as the split
    # itself, warn that the grammar is approximate, and let the operator
    # `retranslate` either half if the word boundary matters.
    parent_mt = str(seg_a.get("mt_text") or "")
    parent_final = str(seg_a.get("final_text") or parent_mt or "")
    left_frac = max(0.05, min(0.95, (at_s - s0) / max(1e-6, (s1 - s0))))
    left_mt, right_mt = _split_text_proportional(parent_mt, left_frac)
    left_final, right_final = _split_text_proportional(parent_final, left_frac)
    # Fallback: if the parent had no translation at all, there is nothing
    # to carry forward; leave both halves empty with a stronger warning.
    new_seg["mt_text"] = right_mt
    new_seg["final_text"] = right_final

    # Mutate the left half in place.
    seg_a["end"] = at_s
    if "end_s" in seg_a:
        seg_a["end_s"] = at_s
    if left_text:
        _set_segment_text(seg_a, left_text)
        seg_a["text"] = left_text
    if left_words:
        seg_a["words"] = left_words
    # The left half's wording changed; its downstream pacing fields are
    # now stale. Drop them so the replay re-derives from the shorter slot.
    for key in ("target_duration_s", "est_duration_s", "candidates", "qc"):
        seg_a.pop(key, None)
    seg_a["mt_text"] = left_mt
    seg_a["final_text"] = left_final

    # Insert the new cue right after the parent.
    segs.insert(i_a + 1, new_seg)

    # Speaker bookkeeping if we minted one.
    if new_speaker_id and voice_assignment:
        engine, voice, gender = voice_assignment
        res.speaker_voice_overrides[new_speaker_id] = {
            "engine": engine, "voice": voice, "source": "edit:split_cue"}
        if gender:
            res.gender_overrides[new_speaker_id] = gender
        svm = data.setdefault("extra", {}).setdefault("speaker_voice_map", {})
        svm[new_speaker_id] = {"engine": engine, "voice": voice,
                               "gender": gender, "source": "edit:split_cue"}

    applied.changes.append(
        f"{sid}: split at {at_s:.3f}s -> {sid} ({s0:.2f}–{at_s:.2f}s) + "
        f"{new_seg_id} ({at_s:.2f}–{s1:.2f}s, speaker={right_speaker or '?'})")
    if parent_mt or parent_final:
        applied.warnings.append(
            f"{sid} / {new_seg_id}: translation split proportionally; "
            f"the word boundary may not match the sentence boundary -- "
            f"`retranslate` either half for cleaner grammar")
    else:
        applied.warnings.append(
            f"{sid} / {new_seg_id}: parent had no translation; both halves "
            f"are empty -- `retranslate` to populate")
    applied.dirty_from = "codemix"


_HANDLERS: Dict[str, Callable] = {
    "set_text": _h_set_text,
    "retranslate": _h_retranslate,
    "set_tone": _h_set_tone,
    "code_mix": _h_code_mix,
    "set_gender": _h_set_gender,
    "create_speaker": _h_create_speaker,
    "set_speaker": _h_set_speaker,
    "set_voice": _h_set_voice,
    "replace_term": _h_replace_term,
    "set_speed": _h_set_speed,
    "lipsync_window": _h_lipsync_window,
    "mute_cue": _h_mute_cue,
    "set_target_language": _h_set_target_language,
    "merge_cues": _h_merge_cues,
    "split_cue": _h_split_cue,
}

assert set(_HANDLERS) == set(OPS), "handler/spec drift"
