"""Timing plans: where each piece of dubbed speech lands on the output timeline.

A plan is not a duration per segment. It is a list of `Piece`s -- each a
contiguous chunk of a raw TTS clip with its own placement on the output
timeline and its own local tempo. Pauses are the gaps between consecutive
`dst_start`s, not objects, so a pause-aware plan costs nothing extra to render.

Every producer emits the same `TimingPlan`, and `backend.mix` has one strategy
that renders any of them:

    warped         pieces addressing a pre-stretched clip; see `backend.warp_plan`

Time units:
  * `slot`, `src_speech`, `src_pauses`, `dst_start` are ABSOLUTE seconds on the
    source/output timeline.
  * `Piece.src_span` is RELATIVE to the start of the clip the mix resolves for
    that segment -- `tts_raw_audio` for every producer except `warped`, whose
    pieces address `tts_warped_audio`.
  * `segments[].words[]` are absolute; `segments[].tts_words[]` are
    clip-relative. They are not interchangeable.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

PRODUCERS = ("warped",)

SCHEMA_VERSION = 1

# A pause shorter than this is coarticulation, not structure worth preserving.
DEFAULT_PAUSE_MIN_S = 0.25


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

@dataclass
class Span:
    start: float
    end: float

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

    def to_jsonable(self) -> dict:
        return {"start": round(self.start, 4), "end": round(self.end, 4)}

    @classmethod
    def from_jsonable(cls, d) -> "Span":
        if isinstance(d, (list, tuple)):
            return cls(float(d[0]), float(d[1]))
        return cls(float(d["start"]), float(d["end"]))


@dataclass
class Piece:
    """One contiguous chunk of a raw TTS clip, placed on the output timeline."""

    src_span: Span            # seconds INTO the raw clip (pre-stretch)
    dst_start: float          # absolute seconds on the output timeline
    local_tempo: float = 1.0  # >1 speeds up; dst_dur = src_span.dur / local_tempo
    anchor: str = "whole"     # "whole" | "clause" | "phrase" | "word"
    text: str = ""            # debug only; never read by the renderer
    src_word_range: Optional[Tuple[int, int]] = None   # index into tts_words[]

    @property
    def dst_dur(self) -> float:
        return self.src_span.dur / max(1e-6, self.local_tempo)

    @property
    def dst_end(self) -> float:
        return self.dst_start + self.dst_dur

    def to_jsonable(self) -> dict:
        return {
            "src_span": self.src_span.to_jsonable(),
            "dst_start": round(self.dst_start, 4),
            "local_tempo": round(self.local_tempo, 6),
            "anchor": self.anchor,
            "text": self.text,
            "src_word_range": list(self.src_word_range) if self.src_word_range else None,
        }

    @classmethod
    def from_jsonable(cls, d: dict) -> "Piece":
        wr = d.get("src_word_range")
        return cls(
            src_span=Span.from_jsonable(d["src_span"]),
            dst_start=float(d["dst_start"]),
            local_tempo=float(d.get("local_tempo", 1.0)),
            anchor=d.get("anchor", "whole"),
            text=d.get("text", ""),
            src_word_range=tuple(wr) if wr else None,
        )


@dataclass
class SegmentPlan:
    id: str
    speaker: Optional[str] = None

    # --- what the SOURCE did; this is the thing worth preserving ---
    slot: Span = field(default_factory=lambda: Span(0.0, 0.0))
    src_speech: List[Span] = field(default_factory=list)
    src_pauses: List[Span] = field(default_factory=list)
    src_lead_in_s: float = 0.0      # slot.start -> first word onset
    src_tail_out_s: float = 0.0     # last word offset -> slot.end

    # --- what the TARGET will do ---
    pieces: List[Piece] = field(default_factory=list)
    inserted_pauses: List[Tuple[int, float]] = field(default_factory=list)

    cost: dict = field(default_factory=dict)
    qc: dict = field(default_factory=dict)

    @property
    def placement_s(self) -> float:
        return self.pieces[0].dst_start if self.pieces else self.slot.start

    @property
    def total_duration_s(self) -> float:
        if not self.pieces:
            return 0.0
        return max(p.dst_end for p in self.pieces) - self.pieces[0].dst_start

    @property
    def dst_end(self) -> float:
        return max((p.dst_end for p in self.pieces), default=self.slot.start)

    def to_jsonable(self) -> dict:
        return {
            "id": self.id,
            "speaker": self.speaker,
            "slot": self.slot.to_jsonable(),
            "src_speech": [s.to_jsonable() for s in self.src_speech],
            "src_pauses": [s.to_jsonable() for s in self.src_pauses],
            "src_lead_in_s": round(self.src_lead_in_s, 4),
            "src_tail_out_s": round(self.src_tail_out_s, 4),
            "pieces": [p.to_jsonable() for p in self.pieces],
            "inserted_pauses": [[int(i), round(float(s), 4)]
                                for i, s in self.inserted_pauses],
            "placement_s": round(self.placement_s, 4),
            "total_duration_s": round(self.total_duration_s, 4),
            "cost": self.cost,
            "qc": self.qc,
        }

    @classmethod
    def from_jsonable(cls, d: dict) -> "SegmentPlan":
        return cls(
            id=d["id"],
            speaker=d.get("speaker"),
            slot=Span.from_jsonable(d.get("slot", {"start": 0.0, "end": 0.0})),
            src_speech=[Span.from_jsonable(x) for x in d.get("src_speech", [])],
            src_pauses=[Span.from_jsonable(x) for x in d.get("src_pauses", [])],
            src_lead_in_s=float(d.get("src_lead_in_s", 0.0)),
            src_tail_out_s=float(d.get("src_tail_out_s", 0.0)),
            pieces=[Piece.from_jsonable(x) for x in d.get("pieces", [])],
            inserted_pauses=[(int(a), float(b))
                             for a, b in d.get("inserted_pauses", [])],
            cost=d.get("cost", {}),
            qc=d.get("qc", {}),
        )


@dataclass
class TimingPlan:
    producer: str = ""
    version: int = SCHEMA_VERSION
    params: dict = field(default_factory=dict)
    segments: List[SegmentPlan] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        return {
            "version": self.version,
            "producer": self.producer,
            "params": self.params,
            "segments": [s.to_jsonable() for s in self.segments],
            "summary": self.summary,
        }

    @classmethod
    def from_jsonable(cls, d: dict) -> "TimingPlan":
        return cls(
            version=int(d.get("version", SCHEMA_VERSION)),
            producer=d.get("producer", ""),
            params=d.get("params", {}),
            segments=[SegmentPlan.from_jsonable(x) for x in d.get("segments", [])],
            summary=d.get("summary", {}),
        )

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_jsonable(), indent=2, ensure_ascii=False))
        return path

    @classmethod
    def read(cls, path: Path) -> "TimingPlan":
        return cls.from_jsonable(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# shared helpers for every producer
# ---------------------------------------------------------------------------

def raw_clip_duration(seg: dict) -> Optional[float]:
    """Length of the segment's UNSTRETCHED TTS clip, or None.

    Prefers `tts_raw_audio`; `tts_audio` may already have been stretched by the
    TTS stage, and planning against that compresses twice.
    """
    import soundfile as sf

    for key in ("tts_raw_audio", "tts_audio"):
        p = seg.get(key)
        if not p:
            continue
        try:
            return float(sf.info(str(p)).duration)
        except Exception:
            continue
    return None


# How far outside its slot a word may sit before the stream is judged
# off-timeline. Diarization boundaries and forced alignment routinely disagree
# by a few tens of ms.
WORD_SLOT_TOLERANCE_S = 0.50
# Below this fraction outside, treat it as boundary noise and keep the stream.
WORD_SLOT_MAX_OUTSIDE_FRAC = 0.25


def source_speech_and_pauses(
    seg: dict, *, pause_min_s: float = DEFAULT_PAUSE_MIN_S,
) -> Tuple[List[Span], List[Span]]:
    """Split a segment's source words into speech runs and the gaps between them.

    The word timings come from the forced-alignment stage and are the basis for
    pause-aware placement. Falls back to one speech run spanning the slot when
    words are missing, so a segment without alignment degrades to naive
    placement instead of failing.
    """
    words = [w for w in (seg.get("words") or [])
             if w.get("start") is not None and w.get("end") is not None]
    start = float(seg.get("start", seg.get("start_s", 0.0)))
    end = float(seg.get("end", seg.get("end_s", 0.0)))
    if not words:
        return [Span(start, end)], []

    # `words[]` must be absolute and inside this segment's own slot. A segment
    # can carry its predecessor's word stream -- monotonic and plausible, but
    # seconds out of place -- and merged JSONs on disk still carry that damage.
    # Refuse such a stream and degrade to the slot rather than anchor to it.
    off = [w for w in words
           if not (start - WORD_SLOT_TOLERANCE_S <= float(w["start"])
                   <= end + WORD_SLOT_TOLERANCE_S)]
    if len(off) > len(words) * WORD_SLOT_MAX_OUTSIDE_FRAC:
        log.warning(
            "timing: segment %s has %d/%d words outside its slot [%.2f, %.2f] "
            "(word range [%.2f, %.2f]) -- word stream is on a different "
            "timeline; falling back to slot-level placement. Re-run alignment.",
            seg.get("id", "?"), len(off), len(words), start, end,
            min(float(w["start"]) for w in words),
            max(float(w["end"]) for w in words))
        seg.setdefault("qc", {})["words_off_timeline"] = {
            "n_outside": len(off), "n_words": len(words),
            "slot": [start, end],
            "word_range": [min(float(w["start"]) for w in words),
                           max(float(w["end"]) for w in words)],
        }
        return [Span(start, end)], []

    words.sort(key=lambda w: float(w["start"]))
    speech: List[Span] = []
    pauses: List[Span] = []
    cur = Span(float(words[0]["start"]), float(words[0]["end"]))
    for w in words[1:]:
        ws, we = float(w["start"]), float(w["end"])
        gap = ws - cur.end
        if gap >= pause_min_s:
            speech.append(cur)
            pauses.append(Span(cur.end, ws))
            cur = Span(ws, we)
        else:
            cur.end = max(cur.end, we)
    speech.append(cur)
    return speech, pauses


def _base_segment_plan(seg: dict, *, pause_min_s: float) -> SegmentPlan:
    start = float(seg.get("start", seg.get("start_s", 0.0)))
    end = float(seg.get("end", seg.get("end_s", 0.0)))
    speech, pauses = source_speech_and_pauses(seg, pause_min_s=pause_min_s)
    sp = SegmentPlan(
        id=seg.get("id", "SEG_????"),
        speaker=seg.get("speaker"),
        slot=Span(start, end),
        src_speech=speech,
        src_pauses=pauses,
        src_lead_in_s=max(0.0, speech[0].start - start) if speech else 0.0,
        src_tail_out_s=max(0.0, end - speech[-1].end) if speech else 0.0,
    )
    # Carried on the plan because `_next_slot_start` sees plans, not segments,
    # and a muted cue must drop out of that lookahead for its seconds to read
    # as gap. See `edits._h_mute_cue`.
    if seg.get("muted"):
        sp.qc["muted"] = True
    return sp


def _clamp_tempo(t: float, lo: float, hi: float) -> Tuple[float, bool]:
    c = max(lo, min(hi, t))
    return c, (abs(c - t) > 1e-9)


def summarize(plan: TimingPlan, *, source_duration_s: Optional[float] = None) -> dict:
    """Plan-level aggregates.

    Overlap is computed here, not per segment: it is a property of the
    sequence, and it says whether a producer is stacking speech on itself.
    """
    segs = plan.segments
    if not segs:
        return {"segments": 0}
    ordered = sorted(segs, key=lambda s: s.placement_s)
    overlap_s = 0.0
    overlap_n = 0
    for a, b in zip(ordered, ordered[1:]):
        ov = a.dst_end - b.placement_s
        if ov > 1e-3:
            overlap_s += ov
            overlap_n += 1
    tempos = [p.local_tempo for s in segs for p in s.pieces]
    # Speech-only tempo -- the rate a listener hears. `tempo_max` below is a
    # PIECE number, which under the blob path covers speech and silence
    # together and overstates the rate. A producer that does not record
    # `qc["speech_tempo"]` is absent here rather than guessed at.
    sp_tempos = [float(s.qc["speech_tempo"]) for s in segs
                 if s.qc.get("speech_tempo") is not None]
    fit_end = max((s.dst_end for s in segs), default=0.0)
    out = {
        "segments": len(segs),
        "pieces": sum(len(s.pieces) for s in segs),
        "inserted_pauses": sum(len(s.inserted_pauses) for s in segs),
        "overlap_segments": overlap_n,
        "overlap_s": round(overlap_s, 3),
        "tempo_min": round(min(tempos), 4) if tempos else None,
        "tempo_max": round(max(tempos), 4) if tempos else None,
        "tempo_mean": round(sum(tempos) / len(tempos), 4) if tempos else None,
        "tempo_cap_hits": sum(1 for s in segs if s.qc.get("tempo_cap_hit")),
        "speech_tempo_max": round(max(sp_tempos), 4) if sp_tempos else None,
        "speech_tempo_mean": (round(sum(sp_tempos) / len(sp_tempos), 4)
                              if sp_tempos else None),
        # How many cues fitted their slot without the speech being touched.
        "speech_untouched_segments": sum(
            1 for s in segs if s.qc.get("speech_untouched")),
        "pause_absorbed_s": round(
            sum(float(s.qc.get("pause_absorbed_s") or 0.0) for s in segs), 3),
        "pause_added_s": round(
            sum(float(s.qc.get("pause_added_s") or 0.0) for s in segs), 3),
        # Separate from `overrun_into_gap_s`: the same seconds of gap, but a
        # borrow is a decision and an overrun is what the cap could not absorb.
        "borrowed_s": round(
            sum(float(s.qc.get("borrowed_s") or 0.0) for s in segs), 3),
        "borrowed_segments": sum(1 for s in segs if s.qc.get("borrowed_s")),
        "overrun_into_gap_s": round(
            sum(float(s.qc.get("overrun_into_gap_s") or 0.0) for s in segs), 3),
        # Slot time a producer could not fill because its slowdown floor bound
        # first -- audible silence left inside a segment's own slot. 0.0 means
        # every short clip was stretched to cover its slot; a producer that
        # never tries leaves this absent rather than zero.
        "unfilled_slot_s": round(
            sum(float(s.qc.get("unfilled_s") or 0.0) for s in segs), 3),
        "unfilled_slot_segments": sum(1 for s in segs if s.qc.get("unfilled_s")),
        "fit_duration_s": round(fit_end, 3),
    }
    if source_duration_s:
        out["source_duration_s"] = round(source_duration_s, 3)
        out["drift_s"] = round(fit_end - source_duration_s, 3)
        out["drift_pct"] = round((fit_end - source_duration_s) / source_duration_s * 100, 3)
    return out




# ---------------------------------------------------------------------------
# merged-JSON integration
# ---------------------------------------------------------------------------

def plan_warped(merged_json: Path, **params) -> TimingPlan:
    """The `warped` producer. See `backend.warp_plan` for what it does.

    A thin wrapper so `plan()`, which looks for `plan_<name>` on this module,
    keeps working. The import is lazy: `backend.warp_plan` pulls in scipy,
    soundfile and a VAD.

    This is the one producer that RENDERS AUDIO -- it writes a warped clip per
    cue and `tts_warped_audio` onto each segment, because a `Piece` cannot
    express a time-varying rate. The plan it returns is only meaningful
    against those files. Mix it with `clip_source="warped"`.
    """
    from .warp_plan import plan_warped as _impl

    return _impl(merged_json, **params)


def apply_plan_to_merged(
    plan: TimingPlan,
    merged_json: Path,
    *,
    out_json: Optional[Path] = None,
) -> dict:
    """Write the plan into `extra["timing_plan"]` plus per-segment qc flags.

    Stored in the merged JSON, the pipeline's data bus: mix and lipsync
    already read `extra.{duration,tts,mix}` and need no new wiring.
    """
    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    by_id = {s.id: s for s in plan.segments}
    for seg in data.get("segments", []):
        sp = by_id.get(seg.get("id"))
        if sp is None:
            continue
        if sp.qc:
            seg.setdefault("qc", {}).update(
                {f"timing_{k}": v for k, v in sp.qc.items()})
        seg["timing_placement_s"] = round(sp.placement_s, 4)
        seg["timing_pieces"] = len(sp.pieces)
    data.setdefault("extra", {})["timing_plan"] = plan.to_jsonable()
    dest = Path(out_json) if out_json else merged_json
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def load_plan_from_merged(merged_json: Path) -> Optional[TimingPlan]:
    data = json.loads(Path(merged_json).read_text())
    d = (data.get("extra") or {}).get("timing_plan")
    return TimingPlan.from_jsonable(d) if d else None


def plan(method: str, merged_json: Path, **params) -> TimingPlan:
    """Dispatch by producer name."""
    if method not in PRODUCERS:
        raise ValueError(f"unknown timing producer {method!r}; "
                         f"choose one of {PRODUCERS}")
    from . import timing as _self
    fn = getattr(_self, f"plan_{method}", None)
    if fn is None:
        raise NotImplementedError(
            f"producer {method!r} is declared in PRODUCERS but not implemented")
    return fn(merged_json, **params)


