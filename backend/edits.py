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

#: Post-MT chain, in execution order. See `run_post_mt_stages`.
STAGE_ORDER: Tuple[str, ...] = (
    "codemix", "duration", "voices", "tts", "vc", "timing", "mix", "lipsync",
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
        params={"find": "the word or phrase exactly as it appears now",
                "replace": "the exact replacement; omit it to have `find` "
                           "translated into the target language"},
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
        examples=["retranslate 0:30–0:45, it is too literal",
                  "make the translation simpler between 2:00 and 2:30"],
        needs_llm=True,
    ),
    OpSpec(
        name="set_tone", dirty_from="tts",
        params={"tone": "formal | casual | friendly | serious | energetic | …",
                "rewrite_text": "true (default) to also rewrite the wording"},
        summary="Change the delivery register: an engine style hint, and (by "
                "default) an LLM rewrite of the wording to match.",
        examples=["make the tone more formal after 1:00"],
        needs_llm=True,
    ),
    OpSpec(
        name="code_mix", dirty_from="codemix",
        params={"terms": "English terms to keep in Latin script",
                "mode": "stage (re-run the transliterator) | llm (rewrite)",
                "termbase": "path to a termbase TSV"},
        summary="Code-mix the translation — keep English technical terms in "
                "English inside the target-language sentence.",
        examples=["code-mix the translation",
                  "keep gradient and weight in English throughout"],
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
        name="set_speaker", dirty_from="voices",
        params={"speaker": "the speaker id that really talks here, "
                           "e.g. SPK_02"},
        summary="Reattribute a stretch of dialogue to a different speaker, "
                "so it is spoken in THAT speaker's voice.",
        examples=["the line at 1:20 is SPK_02, not SPK_00",
                  "from 0:30 to 0:45 that is the second speaker"],
    ),
    OpSpec(
        name="set_voice", dirty_from="voices",
        params={"voice": "engine voice id, e.g. hi-IN-MadhurNeural or Aditi",
                "engine": "tts engine, only if it changes"},
        summary="Pin a specific TTS voice for the scoped segments or speaker.",
        examples=["use hi-IN-MadhurNeural for SPEAKER_00"],
    ),
    OpSpec(
        name="set_speed", dirty_from="tts",
        params={"factor": "1.25 = 25% faster, 0.9 = 10% slower"},
        summary="Speed the dubbed speech up or down over a window, by "
                "re-fitting each clip into a shorter/longer slot.",
        examples=["speed up the audio between 0:30 and 0:45 by 1.2x",
                  "slow down the intro a bit"],
    ),
    OpSpec(
        name="lipsync_window", dirty_from="lipsync",
        params={"model": "wav2lip | musetalk | latentsync, only if it changes"},
        summary="Lip-sync only the scoped time window instead of the whole "
                "video, splicing the render back into the full timeline.",
        examples=["lip-sync only from 1:00 to 1:30",
                  "run musetalk on the last 20 seconds"],
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
    "tts":      "re-speaks the affected lines, remixes the audio and "
                "re-syncs the lips",
    "vc":       "re-converts the voice, remixes and re-syncs",
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
#: `tests/test_edit_agent.py` fails if an op has no entry, which is what
#: stops a newly added op from surfacing in the UI as a bare identifier.
USER_ACTIONS: Dict[str, Dict[str, str]] = {
    "set_text": {
        "title": "Rewrite a line",
        "takes": "a moment in the video, and the exact wording you want",
        "changes": "that line is spoken with your wording, word for word",
    },
    "replace_term": {
        "title": "Change one word everywhere",
        "takes": "the word as it appears now, and either the replacement or "
                 "nothing (it will be translated for you)",
        "changes": "only that word changes; the rest of the line stays "
                   "exactly as it was",
    },
    "retranslate": {
        "title": "Translate a passage again",
        "takes": "a time range, and what to change about it "
                 "(\"too literal\", \"simpler\")",
        "changes": "the translation of those lines is rewritten, then spoken",
    },
    "set_tone": {
        "title": "Change how a line is delivered",
        "takes": "a time range or a speaker, and a tone such as formal, "
                 "warm or urgent",
        "changes": "the same words are spoken in a different manner",
    },
    "code_mix": {
        "title": "Keep English words in English",
        "takes": "optionally the terms to protect, e.g. gradient, weight",
        "changes": "technical terms stay English inside the translated "
                   "sentence instead of being translated",
    },
    "set_gender": {
        "title": "Change a voice's gender",
        "takes": "a speaker or a moment, and male or female",
        "changes": "that speaker is re-cast from the other gender's voices",
    },
    "set_speaker": {
        "title": "Reassign who is talking",
        "takes": "a time range, and which speaker it really is",
        "changes": "those lines move to that speaker and are read in THEIR "
                   "voice; who speaks when is not recomputed",
    },
    "set_voice": {
        "title": "Pick a specific voice",
        "takes": "a speaker, and a voice from the catalogue",
        "changes": "that speaker is read by the voice you named",
    },
    "set_speed": {
        "title": "Speed up or slow down",
        "takes": "a time range, and how much faster or slower",
        "changes": "the dubbed speech is re-fitted into a shorter or "
                   "longer slot; the picture is untouched",
    },
    "lipsync_window": {
        "title": "Lip-sync only part of the video",
        "takes": "a time range, and optionally which lip-sync model",
        "changes": "only that stretch has its mouth re-animated and is "
                   "spliced back into the full timeline",
    },
}


def action_catalogue() -> List[Dict[str, Any]]:
    """Every op as plain language: what it is, takes, changes, and the cost.

    Served to the UI so the list cannot drift from the op registry. The repo
    has paid for that before: `ui/speakers.py` first shipped an invented
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
        })
    return out


def sample_prompts(per_op: int = 2) -> List[Tuple[str, str]]:
    """`(phrasing, what it does)` pairs, in user language, for every op.

    The op names (`lipsync_window`, `set_speaker`) are the agent's
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

    from .voices import load_voice_catalog
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
    termbase_path: Optional[Path] = None
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
    idx = min(STAGE_ORDER.index(s) for s in dirty_from if s in STAGE_ORDER)
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
        system = (
            "You translate ONE word or short phrase for a subtitle. Reply "
            "with the translation ONLY — no quotes, no commentary, no "
            "alternatives, no explanation, no transliteration in brackets.")
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
        new_text = pattern.sub(repl.replace("\\", "\\\\"), cur)
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
    if ctx.rewriter is None:
        raise EditError(
            "retranslate needs the LLM (OPENROUTER_API_KEY unset). Use "
            "set_text with the exact wording instead.")
    instruction = str(op.params.get("instruction")
                      or op.params.get("how") or "").strip()
    lang = op.params.get("target_language") or ctx.target_language or "the target language"
    by_id = {str(s.get("id")): s for s in data.get("segments", [])}
    for sid in seg_ids:
        seg = by_id[sid]
        src = seg.get("text") or ""
        cur = _seg_text(seg)
        # The instruction describes a CHANGE to a line that is already
        # translated, and saying so matters. Told only "reply with the
        # replacement line", gemma-3-12b answered a word-level instruction
        # ("translate the word 'conditional' into Hindi") with just that
        # word — `'.शर्तिया टेक्स्ट जेनरेटर'` in place of the whole
        # sentence, which is a destroyed cue rather than a failed one.
        system = (
            "You are a subtitle translator for a dubbing pipeline. You are "
            "given a line that is ALREADY translated, plus an instruction "
            "describing a change to it.\n\n"
            "Apply the instruction to the current translation and reply with "
            "the COMPLETE corrected line. Preserve every part of the line the "
            "instruction does not mention — wording, word order and "
            "punctuation. If the instruction concerns one word or phrase, "
            "change only that and leave the rest identical.\n\n"
            "Reply with the line ONLY: no quotes, no commentary, no "
            "romanisation, no alternatives, and never a fragment.")
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
            system = (
                "You rewrite dubbing lines to a requested register. Keep the "
                "meaning and the approximate length — the line has to fit the "
                "same time slot. Reply with the rewritten line ONLY.")
            user = (f"Language: {lang}\nRegister: {tone}\nLine: {cur}")
            new = _clean_rewrite(ctx.rewriter(system, user))
            if not new or new == cur:
                continue
            _set_segment_text(seg, new)
            applied.changes.append(f"{sid}: {cur!r} → {new!r}")


def _h_code_mix(data, seg_ids, op, applied, ctx, res):
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
        # and a row mapping a term to itself is what keeps it in English.
        res.termbase_path = _write_termbase(
            terms, base=res.termbase_path, work_dir=ctx.work_dir,
            dry_run=ctx.dry_run)
        applied.changes.append(
            f"terms kept in English: {', '.join(terms)} "
            f"(termbase {res.termbase_path})")
    elif terms:
        applied.changes.append(f"terms kept in English: {', '.join(terms)}")

    if mode == "stage":
        model = ctx.translit_model
        if not model:
            # The job never code-mixed, so there is no model recorded. Take
            # the registry's first choice rather than refusing: stage 9b is
            # the designed path for this edit, and an LLM rewrite of every
            # line is a bigger change than the user asked for.
            try:
                from models.translit import available_models
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
            system = (
                "You code-mix dubbing lines the way Indian lecturers actually "
                "speak: the sentence stays in the target language, but "
                "technical terms, proper nouns, acronyms and units stay in "
                "English, in Latin script. Keep the meaning and the length. "
                "Reply with the rewritten line ONLY.")
            user = (f"Target language: {lang}\n"
                    f"English source: {src}\n"
                    f"Current translation: {cur}\n"
                    + (f"Always keep these in English: {', '.join(terms)}\n"
                       if terms else ""))
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
    engine = op.params.get("engine") or ctx.tts_engine or ctx.voice_engine
    by_id = {str(s.get("id")): s for s in data.get("segments", [])}
    known, why = _known_voice(voice, engine=engine, language=ctx.target_language,
                              catalog_json=ctx.voice_catalog_json)
    if not known:
        applied.warnings.append(
            f"{voice!r} is not in the {engine!r} catalog ({why}); passing it "
            f"through anyway — the engine will reject it if it is wrong")

    if op.scope.speaker or op.scope.all:
        speakers = {by_id[s].get("speaker") for s in seg_ids if by_id.get(s)}
        speakers.discard(None)
        for spk in speakers:
            res.speaker_voice_overrides[str(spk)] = {
                "engine": engine, "voice": voice, "source": "edit:set_voice"}
            applied.changes.append(f"{spk}: voice={voice}")
        applied.dirty_from = "voices"
    else:
        for sid in seg_ids:
            seg = by_id[sid]
            _backup(seg, "tts_voice")
            seg["tts_voice"] = voice
            applied.changes.append(f"{sid}: voice={voice}")
        applied.dirty_from = "tts"


def _known_voice(voice: str, *, engine: Optional[str],
                 language: Optional[str],
                 catalog_json: Optional[Path]) -> Tuple[bool, str]:
    eng = (engine or "").lower()
    try:
        from .voices import load_voice_catalog
        cat = load_voice_catalog(
            "indic-speak" if eng in ("", "bodhan", "indic-speak") else eng,
            language=language, catalog_json=catalog_json)
        return any(v.voice == voice for v in cat), f"{len(cat)} voices known"
    except Exception as e:                       # catalog is advisory only
        return True, f"catalog unreadable: {e}"


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
        res.lipsync_model = str(model)
    applied.segments = seg_ids
    applied.changes.append(
        f"lip-sync {_hhmmss(start)}–{_hhmmss(end)}"
        + (f" with {model}" if model else "")
        + "; the rest of the video is passed through untouched")


_HANDLERS: Dict[str, Callable] = {
    "set_text": _h_set_text,
    "retranslate": _h_retranslate,
    "set_tone": _h_set_tone,
    "code_mix": _h_code_mix,
    "set_gender": _h_set_gender,
    "set_speaker": _h_set_speaker,
    "set_voice": _h_set_voice,
    "replace_term": _h_replace_term,
    "set_speed": _h_set_speed,
    "lipsync_window": _h_lipsync_window,
}

assert set(_HANDLERS) == set(OPS), "handler/spec drift"
