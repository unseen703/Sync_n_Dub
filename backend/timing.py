"""Timing plans: where each piece of dubbed speech lands on the output timeline.

The problem this solves. Indic translations run 1.15-1.42x longer than their
English source (`comparison-results.md` §3), and the pipeline's only answer has
been a +/-10% per-segment stretch followed by one global `atempo`. Both are
uniform: they compress everything equally, which moves every speech onset. A
5 s clip placed in a 5 s slot is still badly synced if one of them is silent for
its first second, and lip-sync responds to where speech events actually fall,
not to matching totals.

So a plan is not "a duration per segment". It is a list of `Piece`s, each a
contiguous chunk of a raw TTS clip, with its own placement on the output
timeline and its own local tempo. Pauses are not objects that need rendering --
they are the gaps between consecutive `dst_start`s, which is why a pause-aware
plan costs nothing extra to render.

Every producer emits the same `TimingPlan`, and `backend.mix` grows exactly one
strategy that renders any of them:

    naive_slot     one Piece per segment, target = slot - margin  (the control arm)
    pause_aware    N Pieces anchored to the SOURCE speech onsets, pauses reinserted
    word_anchored  N Pieces cut at clause boundaries from TTS word timings
    dp             the same object, chosen by dynamic programming over a cost

Keeping one output type is what makes the hardest producer (dp) cheap to
integrate: it has no new rendering surface, only a different search.

Time conventions, which are the likeliest source of an off-by-a-segment bug:
  * `slot`, `src_speech`, `src_pauses`, `dst_start` are ABSOLUTE seconds on the
    source/output timeline.
  * `Piece.src_span` is RELATIVE to the start of that segment's raw TTS clip.
  * `segments[].words[]` in the merged JSON are absolute; `segments[].tts_words[]`
    are clip-relative. They are not interchangeable.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

PRODUCERS = ("naive_slot", "pause_aware", "word_anchored", "dp")

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

    Always prefers `tts_raw_audio`: `tts_audio` may already have been stretched
    by the TTS stage, and planning against an already-stretched clip is how the
    pipeline came to compress twice.
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


# A word may sit this far outside its segment's slot before the stream is
# judged off-timeline: diarization boundaries and forced alignment disagree by
# a few tens of ms routinely, and that is not corruption.
WORD_SLOT_TOLERANCE_S = 0.50
# Below this fraction outside, treat it as boundary noise and keep the stream.
WORD_SLOT_MAX_OUTSIDE_FRAC = 0.25


def source_speech_and_pauses(
    seg: dict, *, pause_min_s: float = DEFAULT_PAUSE_MIN_S,
) -> Tuple[List[Span], List[Span]]:
    """Split a segment's source words into speech runs and the gaps between them.

    These word timings are produced by the forced-alignment stage and, before
    this module existed, were read by nothing at all (`comparison-results.md` §7
    measured that aligner choice changed zero of 44 outputs). They are the whole
    basis for pause-aware placement.

    Falls back to one speech run spanning the slot when words are missing, so a
    segment without alignment degrades to the naive behaviour instead of failing.
    """
    words = [w for w in (seg.get("words") or [])
             if w.get("start") is not None and w.get("end") is not None]
    start = float(seg.get("start", seg.get("start_s", 0.0)))
    end = float(seg.get("end", seg.get("end_s", 0.0)))
    if not words:
        return [Span(start, end)], []

    # `words[]` must be ABSOLUTE and inside this segment's own slot. A
    # whisperx bug (fixed in models/align/whisperx_adapter.py) shipped 17 of
    # the 142 merged JSONs on disk with each segment carrying its PREDECESSOR's
    # word stream -- still monotonic, still plausible, and silently wrong.
    # Anchoring to those times puts speech seconds away from where it belongs,
    # so refuse the stream and degrade to the slot rather than trust it.
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
    return SegmentPlan(
        id=seg.get("id", "SEG_????"),
        speaker=seg.get("speaker"),
        slot=Span(start, end),
        src_speech=speech,
        src_pauses=pauses,
        src_lead_in_s=max(0.0, speech[0].start - start) if speech else 0.0,
        src_tail_out_s=max(0.0, end - speech[-1].end) if speech else 0.0,
    )


def _clamp_tempo(t: float, lo: float, hi: float) -> Tuple[float, bool]:
    c = max(lo, min(hi, t))
    return c, (abs(c - t) > 1e-9)


def summarize(plan: TimingPlan, *, source_duration_s: Optional[float] = None) -> dict:
    """Plan-level aggregates. Overlap is computed here rather than per segment
    because it is a property of the sequence, and it is the number that tells
    you whether a producer has started stacking speech on top of itself."""
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
        # Slot time a producer could not fill because its slowdown floor
        # bound first -- i.e. audible silence left inside a segment's own
        # slot. 0.0 means every short clip was stretched out to cover its
        # slot; a producer that never tries leaves this absent, not zero.
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
# producer: naive_slot -- the control arm
# ---------------------------------------------------------------------------

#: What to do with a cue whose slot starts inside the previous cue's slot.
#: "serialize" is the historical behaviour -- the shift-forward cursor pushes
#: it past the cue it overlaps. "keep" places it where the source put it.
OVERLAP_POLICIES = ("serialize", "keep")


def _next_slot_start(plans: List["SegmentPlan"], i: int) -> Optional[float]:
    """Where the next NON-overlapping slot begins, or None at the end.

    Nested cues are skipped on purpose. The caller uses this to measure the
    source gap after slot `i`, and a cue that starts *inside* slot `i` does
    not end it -- taking the next slot in list order would measure a negative
    gap for exactly the segments that are followed by music.
    """
    end = plans[i].slot.end
    for j in range(i + 1, len(plans)):
        if plans[j].slot.start >= end - 1e-3:
            return plans[j].slot.start
    return None


def plan_naive_slot(
    merged_json: Path,
    *,
    safety_margin_s: float = 0.10,
    max_tempo: float = 1.10,
    min_tempo: float = 0.90,
    shift_forward: bool = True,
    use_target_duration: bool = False,
    overlap_policy: str = "serialize",
    gap_barrier_s: float = 0.0,
    out_json: Optional[Path] = None,
) -> TimingPlan:
    """Reproduce today's behaviour AS A TIMING PLAN.

    This exists so the control arm and the treatment arms go through the
    identical renderer. Without it, any measured difference between "current
    pipeline" and a new producer would be confounded by the rendering path --
    different ffmpeg graphs, different resampling, different cap semantics --
    and the comparison would not be about timing at all.

    One Piece per segment, target = slot - margin, and the same shift-forward
    cursor `backend.mix` uses today, so overruns cascade rather than overlap.

    The tempo is `raw / target` in BOTH directions: a clip longer than its
    slot is compressed, and a clip SHORTER than its slot is expanded to fill
    it, so a short dub line does not leave a hole in the middle of its own
    slot. `min_tempo` bounds the expansion -- past roughly 1.4x it stops
    sounding like speech -- and whatever the floor could not absorb is
    recorded per segment as `qc["unfilled_s"]` and summed into the plan
    summary as `unfilled_slot_s`. That number is the honest answer to "is
    there still a pause?", and it is why the floor is reported rather than
    just applied.

    `use_target_duration` reads `segments[].target_duration_s` as the slot
    length when it is present, instead of `end - start`. That is the field
    stage 9c (duration planning) and `backend.edits.set_speed` both write, so
    without it a per-segment speed edit is silently ignored by this producer.
    It defaults to False so the m2 control arm keeps the exact semantics it
    was measured with.

    `overlap_policy="keep"` stops the cursor serialising SIMULTANEOUS SPEECH.
    A cue whose slot starts inside the previous cue's slot is a second voice
    talking over the first, not a cue that follows it, and the monotone cursor
    cannot express that: it pushes the nested cue past the one it overlaps,
    which on job 20261022-044302 moved a 0.35 s backchannel (26.22-26.58,
    inside a 23.93-30.73 turn) out to ~32.2 s -- into the middle of a 7.2 s
    stretch of music. `keep` places such a cue at its own slot start and does
    NOT advance the cursor, so the cue it overlaps still owns the timeline.
    The mix renders this already: `backend.mix._ffmpeg_mix_plan` is one
    `adelay` per clip into `amix` and has never required non-overlap.
    `summarize` then reports the overlap honestly instead of reporting zero
    because the overlap was displaced rather than absent.

    `gap_barrier_s` (0.0 = off) stops the cascade walking into that music in
    the first place. A source gap at least this long after a cue is a wall:
    the cue's target is shortened to whatever is left before its slot ends,
    so it is sped up (within `max_tempo`) rather than allowed to run on over
    non-speech audio. Whatever the cap still cannot absorb is recorded as
    `qc["overrun_into_gap_s"]` -- the clip is never truncated, because losing
    a line is worse than a short bleed. Clamping the cue also resets the
    cursor for free: it now ends before the gap, so the next cue is placed at
    its own slot start instead of inheriting the drift.
    """
    if overlap_policy not in OVERLAP_POLICIES:
        raise ValueError(f"unknown overlap_policy {overlap_policy!r}; "
                         f"choose one of {OVERLAP_POLICIES}")
    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    plan = TimingPlan(producer="naive_slot", params={
        "safety_margin_s": safety_margin_s,
        "max_tempo": max_tempo, "min_tempo": min_tempo,
        "shift_forward": shift_forward,
        "use_target_duration": use_target_duration,
        "overlap_policy": overlap_policy,
        "gap_barrier_s": gap_barrier_s,
    })

    # Built up front because the gap barrier needs to look AHEAD at the next
    # slot, which the old one-pass loop could not do.
    segs = list(data.get("segments", []))
    plans = [_base_segment_plan(s, pause_min_s=DEFAULT_PAUSE_MIN_S) for s in segs]

    cursor = 0.0
    # End of the last slot that advanced the cursor. A nested cue must not
    # become the thing the NEXT cue is tested against, or one overlap would
    # cascade into a chain of them.
    prev_slot_end: Optional[float] = None
    for i, (seg, sp) in enumerate(zip(segs, plans)):
        raw = raw_clip_duration(seg)
        if raw is None or raw <= 0:
            sp.qc["no_audio"] = True
            plan.segments.append(sp)
            continue

        # 1. where it goes
        nested = (overlap_policy == "keep" and prev_slot_end is not None
                  and sp.slot.start < prev_slot_end - 1e-3)
        if nested:
            placed = sp.slot.start
        else:
            placed = max(sp.slot.start, cursor) if shift_forward else sp.slot.start

        # 2. how long it is allowed to be
        slot_dur = sp.slot.dur
        if use_target_duration and seg.get("target_duration_s") is not None:
            try:
                slot_dur = max(0.0, float(seg["target_duration_s"]))
            except (TypeError, ValueError):
                pass
        target = max(0.01, slot_dur - safety_margin_s)

        barrier = False
        if gap_barrier_s > 0 and not nested:
            nxt = _next_slot_start(plans, i)
            if nxt is not None and nxt - sp.slot.end >= gap_barrier_s:
                barrier = True
                sp.qc["gap_after_s"] = round(nxt - sp.slot.end, 3)
                # Run to the music and not one frame past it. `placed` is
                # already the drifted position, so this is what is actually
                # left, not the slot's nominal length.
                target = max(0.01, min(target, sp.slot.end - placed))

        # 3. how fast
        tempo, capped = _clamp_tempo(raw / target, min_tempo, max_tempo)
        sp.pieces = [Piece(src_span=Span(0.0, raw), dst_start=placed,
                           local_tempo=tempo, anchor="whole",
                           text=seg.get("final_text") or seg.get("mt_text") or "")]
        if capped:
            sp.qc["tempo_cap_hit"] = True
            # Only an expansion that hit the FLOOR leaves silence. A
            # compression pinned at the ceiling overruns instead, which the
            # shift-forward cursor already accounts for.
            unfilled = target - raw / tempo
            if unfilled > 1e-3:
                sp.qc["unfilled_s"] = round(unfilled, 4)
        if placed > sp.slot.start + 1e-3:
            sp.qc["shifted_s"] = round(placed - sp.slot.start, 4)
        if barrier:
            over = (placed + sp.pieces[0].dst_dur) - sp.slot.end
            if over > 1e-3:
                sp.qc["overrun_into_gap_s"] = round(over, 4)
        if nested:
            # Deliberately does NOT touch the cursor: the cue this one talks
            # over still owns the timeline.
            sp.qc["overlap_kept"] = True
        else:
            cursor = placed + sp.pieces[0].dst_dur
            prev_slot_end = (sp.slot.end if prev_slot_end is None
                             else max(prev_slot_end, sp.slot.end))
        plan.segments.append(sp)

    plan.summary = summarize(plan)
    if out_json:
        plan.write(Path(out_json))
    log.info("timing[naive_slot]: %d segments, %d pieces, drift_end=%.2fs",
             plan.summary.get("segments", 0), plan.summary.get("pieces", 0),
             plan.summary.get("fit_duration_s", 0.0))
    return plan


# ---------------------------------------------------------------------------
# merged-JSON integration
# ---------------------------------------------------------------------------

def apply_plan_to_merged(
    plan: TimingPlan,
    merged_json: Path,
    *,
    out_json: Optional[Path] = None,
) -> dict:
    """Write the plan into `extra["timing_plan"]` plus per-segment qc flags.

    Stored inside the merged JSON because that is the pipeline's existing data
    bus -- mix and lipsync already read `extra.{duration,tts,mix}` and need no
    new wiring to see this.
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
    """Dispatch by producer name. The experiment runner's single entry point."""
    if method not in PRODUCERS:
        raise ValueError(f"unknown timing producer {method!r}; "
                         f"choose one of {PRODUCERS}")
    from . import timing as _self
    fn = getattr(_self, f"plan_{method}", None)
    if fn is None:
        # The heavier producers (word_anchored, dp) live in `timing_advanced`
        # to keep this module importable without their deps. Import lazily so
        # a plain `plan("pause_aware", ...)` never pays for them.
        try:
            from . import timing_advanced as _adv
        except Exception as e:                       # pragma: no cover
            raise NotImplementedError(
                f"producer {method!r} needs backend.timing_advanced, which "
                f"failed to import: {type(e).__name__}: {e}") from e
        fn = getattr(_adv, f"plan_{method}", None)
    if fn is None:
        raise NotImplementedError(
            f"producer {method!r} is declared in PRODUCERS but not implemented")
    return fn(merged_json, **params)


# ---------------------------------------------------------------------------
# producer: pause_aware -- anchor every speech run to its SOURCE onset
# ---------------------------------------------------------------------------

# A chunk shorter than this (in RAW clip seconds) cannot be rendered: the
# ffmpeg/rubberband analysis windows are ~10 ms and `backend.stretch.time_stretch`
# raises rather than emit an empty file. We merge such a run into its neighbour
# instead of handing the renderer a clip it will reject.
MIN_PIECE_SRC_S = 0.06


def _merge_tiny_runs(
    speech: List[Span], pauses: List[Span], raw: float,
) -> Tuple[List[Span], List[Optional[Span]], List[List[int]]]:
    """Merge speech runs whose proportional share of the clip is unrenderable.

    Returns (merged_runs, pause_before_each_merged_run, source_indices_per_run).
    `pause_before[0]` is always None. A merged run keeps the FIRST member's
    onset, because the onset is the thing this producer exists to preserve;
    the swallowed pause simply stops being reinserted.
    """
    total = sum(s.dur for s in speech)
    if total <= 0:
        return speech, [None] * len(speech), [[i] for i in range(len(speech))]

    out: List[Span] = []
    out_pause: List[Optional[Span]] = []
    out_idx: List[List[int]] = []
    for i, run in enumerate(speech):
        share = run.dur / total * raw
        if out and share < MIN_PIECE_SRC_S:
            # absorb into the previous run: extend its end, drop the pause
            out[-1] = Span(out[-1].start, max(out[-1].end, run.end))
            out_idx[-1].append(i)
            continue
        out.append(Span(run.start, run.end))
        out_pause.append(pauses[i - 1] if i > 0 and i - 1 < len(pauses) else None)
        out_idx.append([i])
    return out, out_pause, out_idx


def _cut_points_uniform(raw: float, weights: List[float]) -> List[float]:
    """Proportional cut points (clip-relative seconds), length len(weights)+1."""
    total = sum(weights) or 1.0
    cuts = [0.0]
    acc = 0.0
    for w in weights[:-1]:
        acc += w
        cuts.append(min(raw, acc / total * raw))
    cuts.append(raw)
    # monotone, and never a zero-length chunk
    for i in range(1, len(cuts)):
        cuts[i] = max(cuts[i], cuts[i - 1] + 1e-4)
    cuts[-1] = max(raw, cuts[-2] + 1e-4)
    return cuts


def _cut_points_char_mass(
    text: str, raw: float, weights: List[float],
) -> Optional[Tuple[List[float], str]]:
    """Cut the clip at TEXT word boundaries nearest each proportional target.

    Without TTS word timings the only signal about where words end is the text
    itself. Assuming a constant characters-per-second rate turns a character
    offset into a clip time; snapping to a token boundary first is what makes
    this better than a blind proportional cut, because it stops a chunk
    starting mid-word, which is audible as a click even before any stretching.
    """
    toks = (text or "").split()
    n = len(weights)
    if len(toks) < n or n < 2:
        return None
    lens = [len(t) for t in toks]
    total_chars = sum(lens) or 1
    cum = [0]
    for L in lens:
        cum.append(cum[-1] + L)

    total_w = sum(weights) or 1.0
    cuts = [0.0]
    used_tok = 0
    acc_w = 0.0
    for k in range(n - 1):
        acc_w += weights[k]
        target_chars = acc_w / total_w * total_chars
        # nearest token boundary strictly after the previous cut, and leaving
        # at least one token for each remaining chunk
        lo = used_tok + 1
        hi = len(toks) - (n - 1 - k)
        if lo > hi:
            return None
        best = min(range(lo, hi + 1), key=lambda j: abs(cum[j] - target_chars))
        used_tok = best
        cuts.append(min(raw, cum[best] / total_chars * raw))
    cuts.append(raw)
    for i in range(1, len(cuts)):
        cuts[i] = max(cuts[i], cuts[i - 1] + 1e-4)
    cuts[-1] = max(raw, cuts[-2] + 1e-4)
    return cuts, "char_mass"


def _cut_points_tts_words(
    tts_words, raw: float, weights: List[float],
) -> Optional[Tuple[List[float], str, List[Tuple[int, int]]]]:
    """Cut at the TTS word boundary nearest each proportional target.

    `tts_words[]` are CLIP-RELATIVE (see module docstring), which is exactly the
    coordinate system `Piece.src_span` wants, so no conversion happens here.
    """
    ws = []
    for i, w in enumerate(tts_words or []):
        try:
            s, e = float(w["start"]), float(w["end"])
        except Exception:
            return None
        if e < s:
            return None
        ws.append((s, e))
    n = len(weights)
    if len(ws) < n or n < 2:
        return None

    total_w = sum(weights) or 1.0
    cuts = [0.0]
    ranges: List[Tuple[int, int]] = []
    used = -1          # index of last word consumed
    acc_w = 0.0
    for k in range(n - 1):
        acc_w += weights[k]
        target_t = acc_w / total_w * raw
        lo = used + 1              # cut "after word j" needs j >= lo
        hi = len(ws) - (n - 1 - k) - 1
        if lo > hi:
            return None
        best = min(range(lo, hi + 1), key=lambda j: abs(ws[j][1] - target_t))
        ranges.append((used + 1, best + 1))
        used = best
        cuts.append(min(raw, ws[best][1]))
    ranges.append((used + 1, len(ws)))
    cuts.append(raw)
    for i in range(1, len(cuts)):
        cuts[i] = max(cuts[i], cuts[i - 1] + 1e-4)
    cuts[-1] = max(raw, cuts[-2] + 1e-4)
    return cuts, "tts_words", ranges


def _segment_costs(
    *,
    pieces: List[Piece],
    runs: List[Span],
    kept_pauses: List[float],
    src_pauses_used: List[Optional[Span]],
) -> dict:
    """Cost terms, defined once here so `plan_dp` can search over the same thing.

    duration -- |rendered speech seconds - SOURCE SPEECH seconds| / source speech
        seconds. Deliberately measured against the speech budget and not against
        the slot: optimising toward the slot is "TTS duration == source
        duration", which is the failure mode this module exists to avoid.
    pause    -- sum|reinserted pause - source pause| / sum(source pause). 0.0 when
        the segment has no internal pause, so a one-run segment is never
        penalised for a structure it does not have.
    rate     -- duration-weighted mean |local_tempo - 1|. Duration-weighted so a
        heavy stretch on a 4 s chunk outranks the same stretch on a 0.2 s one.
    onset    -- mean |piece.dst_start - source run onset| in seconds. This is the
        quantity the producer is actually trying to minimise; it is reported as
        a cost term so a DP can trade it against the other three.
    """
    src_speech_s = sum(r.dur for r in runs)
    dst_speech_s = sum(p.dst_dur for p in pieces)
    duration = abs(dst_speech_s - src_speech_s) / max(1e-6, src_speech_s)

    src_pause_s = sum(p.dur for p in src_pauses_used if p is not None)
    if src_pause_s > 1e-6 and kept_pauses:
        err = sum(abs(k - (p.dur if p is not None else 0.0))
                  for k, p in zip(kept_pauses, src_pauses_used[1:]))
        pause = err / src_pause_s
    else:
        pause = 0.0

    tot = sum(p.dst_dur for p in pieces) or 1.0
    rate = sum(abs(p.local_tempo - 1.0) * p.dst_dur for p in pieces) / tot

    onsets = [abs(p.dst_start - r.start) for p, r in zip(pieces, runs)]
    return {
        "duration": round(duration, 6),
        "pause": round(pause, 6),
        "rate": round(rate, 6),
        "onset": round(sum(onsets) / len(onsets), 6) if onsets else 0.0,
        "src_speech_s": round(src_speech_s, 4),
        "dst_speech_s": round(dst_speech_s, 4),
        "onset_err_max_s": round(max(onsets), 4) if onsets else 0.0,
    }


def plan_pause_aware(
    merged_json: Path,
    *,
    pause_min_s: float = DEFAULT_PAUSE_MIN_S,
    pause_scale: float = 1.0,
    pause_min_keep_s: float = 0.12,
    max_tempo: float = 1.15,
    min_tempo: float = 0.90,
    preserve: str = "onset",            # "onset" | "onset_offset" | "centroid"
    lead_in_policy: str = "source",     # "source" | "zero"
    shift_forward: bool = True,
    out_json: Optional[Path] = None,
) -> TimingPlan:
    """Place one Piece per SOURCE speech run, anchored to that run's onset.

    Why this is not `plan_naive_slot` with extra steps: naive targets
    `slot - margin`, and a slot is a diarization turn -- lead-in silence,
    trailing silence and every internal pause included. Fitting the TTS to that
    is the "total duration matches" objective the project rejects. Here the
    budget is `sum(src_speech[].dur)` and the pauses are put back where the
    source put them, so the thing being matched is WHERE SPEECH STARTS.

    The placement rule reduces exactly to `dst_start == src_speech[k].start`
    whenever the chunk physically fits, and only departs from the onset when it
    cannot -- at which point it degrades to the tightest legal pause rather than
    overlapping. `pause_scale < 1` caps each reinserted pause below its source
    length to buy room back when the target overruns; `pause_min_keep_s` floors
    it so a shrunk pause stays a pause instead of a glitch.
    """
    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    if preserve not in ("onset", "onset_offset", "centroid"):
        raise ValueError(f"preserve must be onset|onset_offset|centroid, got {preserve!r}")
    if lead_in_policy not in ("source", "zero"):
        raise ValueError(f"lead_in_policy must be source|zero, got {lead_in_policy!r}")

    plan_obj = TimingPlan(producer="pause_aware", params={
        "pause_min_s": pause_min_s,
        "pause_scale": pause_scale,
        "pause_min_keep_s": pause_min_keep_s,
        "max_tempo": max_tempo, "min_tempo": min_tempo,
        "preserve": preserve, "lead_in_policy": lead_in_policy,
        "shift_forward": shift_forward,
    })

    cursor = 0.0
    for seg in data.get("segments", []):
        sp = _base_segment_plan(seg, pause_min_s=pause_min_s)
        try:
            _plan_one_pause_aware(
                seg, sp,
                pause_scale=pause_scale, pause_min_keep_s=pause_min_keep_s,
                max_tempo=max_tempo, min_tempo=min_tempo,
                preserve=preserve, lead_in_policy=lead_in_policy,
            )
        except Exception as exc:                      # one bad segment != dead sweep
            log.warning("timing[pause_aware]: %s degraded: %s", sp.id, exc)
            sp.qc["error"] = str(exc)
            sp.pieces = []

        # cross-segment guard: honouring a source onset must never stack speech
        # on top of the previous segment's tail.
        if sp.pieces and shift_forward:
            first = sp.pieces[0].dst_start
            if first < cursor - 1e-6:
                shift = cursor - first
                for p in sp.pieces:
                    p.dst_start += shift
                sp.qc["shifted_s"] = round(shift, 4)
        if sp.pieces:
            cursor = max(cursor, sp.dst_end)
        plan_obj.segments.append(sp)

    plan_obj.summary = summarize(plan_obj)
    if out_json:
        plan_obj.write(Path(out_json))
    log.info("timing[pause_aware]: %d segments, %d pieces, %d pauses, drift_end=%.2fs",
             plan_obj.summary.get("segments", 0), plan_obj.summary.get("pieces", 0),
             plan_obj.summary.get("inserted_pauses", 0),
             plan_obj.summary.get("fit_duration_s", 0.0))
    return plan_obj


def _plan_one_pause_aware(
    seg: dict,
    sp: SegmentPlan,
    *,
    pause_scale: float,
    pause_min_keep_s: float,
    max_tempo: float,
    min_tempo: float,
    preserve: str,
    lead_in_policy: str,
) -> None:
    """Fill `sp.pieces/inserted_pauses/cost/qc` in place. Mutates nothing else."""
    raw = raw_clip_duration(seg)
    if raw is None or raw <= 0:
        sp.qc["no_audio"] = True
        return
    if not seg.get("words"):
        # no forced alignment -> source_speech_and_pauses returned one slot-wide
        # run, so this segment is naive placement wearing a pause_aware hat. Say
        # so loudly rather than letting the sweep read it as a working method.
        sp.qc["degraded_no_words"] = True

    runs, pause_before, run_map = _merge_tiny_runs(sp.src_speech, sp.src_pauses, raw)
    if len(runs) < len(sp.src_speech):
        sp.qc["merged_runs"] = len(sp.src_speech) - len(runs)
    n = len(runs)
    weights = [r.dur for r in runs]
    speech_budget = sum(weights)
    if speech_budget <= 0:                 # degenerate alignment; use the slot
        runs = [Span(sp.slot.start, sp.slot.end)]
        pause_before = [None]
        run_map = [[0]]
        n, weights, speech_budget = 1, [sp.slot.dur], max(0.01, sp.slot.dur)
        sp.qc["degenerate_speech"] = True

    # --- split the raw clip -------------------------------------------------
    word_ranges: List[Optional[Tuple[int, int]]] = [None] * n
    rule = "single"
    cuts = [0.0, raw]
    if n > 1:
        got = _cut_points_tts_words(seg.get("tts_words"), raw, weights)
        if got is not None:
            cuts, rule, wr = got
            word_ranges = list(wr)
        else:
            got2 = _cut_points_char_mass(
                seg.get("final_text") or seg.get("mt_text") or "", raw, weights)
            if got2 is not None:
                cuts, rule = got2
            else:
                cuts, rule = _cut_points_uniform(raw, weights), "uniform_time"
    sp.qc["split_rule"] = rule

    # --- tempo --------------------------------------------------------------
    # One tempo for the whole segment keeps the speaking rate steady inside a
    # turn, which listeners notice far more than a few tens of ms of onset drift.
    seg_tempo, capped = _clamp_tempo(raw / speech_budget, min_tempo, max_tempo)
    if capped:
        sp.qc["tempo_cap_hit"] = True

    chunk_src = [Span(cuts[i], cuts[i + 1]) for i in range(n)]
    tempos: List[float] = []
    for i in range(n):
        if preserve == "onset_offset":
            # also land this chunk's END near the source run's end
            t, c = _clamp_tempo(chunk_src[i].dur / max(1e-6, runs[i].dur),
                                min_tempo, max_tempo)
            if c:
                sp.qc["tempo_cap_hit"] = True
        else:
            t = seg_tempo
        tempos.append(t)

    # --- placement ----------------------------------------------------------
    offset = 0.0
    if lead_in_policy == "zero":
        # drop the source's lead-in silence; everything slides earlier by it
        offset = -(runs[0].start - sp.slot.start)

    pieces: List[Piece] = []
    kept_pauses: List[float] = []
    floored = 0
    text = seg.get("final_text") or seg.get("mt_text") or ""
    prev_end = None
    for i in range(n):
        dur = chunk_src[i].dur / max(1e-6, tempos[i])
        if preserve == "centroid":
            desired = (runs[i].start + runs[i].end) / 2.0 - dur / 2.0 + offset
        else:
            desired = runs[i].start + offset

        if prev_end is None:
            start = desired
        else:
            src_gap = pause_before[i].dur if pause_before[i] is not None else 0.0
            natural = desired - prev_end
            # pause_scale caps the reinserted pause; pause_min_keep_s floors it.
            # With pause_scale == 1 and a chunk that fits, hi == src_gap and
            # natural == src_gap, so `kept` IS the source pause and `start` IS
            # the source onset -- the cursor only bites when the chunk overruns.
            hi = max(pause_min_keep_s, src_gap * pause_scale)
            kept = min(max(natural, pause_min_keep_s), hi)
            if kept > natural + 1e-9:
                floored += 1
            start = prev_end + kept
            kept_pauses.append(kept)

        pieces.append(Piece(
            src_span=chunk_src[i], dst_start=start, local_tempo=tempos[i],
            anchor="whole" if n == 1 else ("word" if rule == "tts_words" else "phrase"),
            text=text if n == 1 else "",
            src_word_range=word_ranges[i],
        ))
        prev_end = start + dur

    sp.pieces = pieces
    sp.inserted_pauses = [(i, round(g, 4)) for i, g in enumerate(kept_pauses)]
    if floored:
        sp.qc["pauses_floored"] = floored
    sp.qc["run_map"] = run_map
    sp.cost = _segment_costs(pieces=pieces, runs=runs,
                             kept_pauses=kept_pauses, src_pauses_used=pause_before)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """python -m backend.timing plan --method pause_aware --merged X --out Y"""
    import argparse
    import inspect
    import sys

    ap = argparse.ArgumentParser(prog="backend.timing")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="build a timing plan from a merged MT json")
    p.add_argument("--method", default="pause_aware", choices=list(PRODUCERS))
    p.add_argument("--merged", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--pause-min-s", type=float, default=DEFAULT_PAUSE_MIN_S)
    p.add_argument("--pause-scale", type=float, default=1.0)
    p.add_argument("--pause-min-keep-s", type=float, default=0.12)
    p.add_argument("--max-tempo", type=float, default=None)
    p.add_argument("--min-tempo", type=float, default=None)
    p.add_argument("--preserve", default="onset",
                   choices=["onset", "onset_offset", "centroid"])
    p.add_argument("--lead-in-policy", default="source", choices=["source", "zero"])
    p.add_argument("--safety-margin-s", type=float, default=None)
    p.add_argument("--no-shift-forward", action="store_true")
    p.add_argument("--log-level", default="INFO")

    a = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s")

    wanted = {
        "pause_min_s": a.pause_min_s, "pause_scale": a.pause_scale,
        "pause_min_keep_s": a.pause_min_keep_s,
        "preserve": a.preserve, "lead_in_policy": a.lead_in_policy,
        "shift_forward": not a.no_shift_forward, "out_json": a.out,
    }
    for k, v in (("max_tempo", a.max_tempo), ("min_tempo", a.min_tempo),
                 ("safety_margin_s", a.safety_margin_s)):
        if v is not None:
            wanted[k] = v

    from . import timing as _self
    fn = getattr(_self, f"plan_{a.method}", None)
    if fn is None:
        print(f"producer {a.method!r} is not implemented yet", file=sys.stderr)
        return 2
    # Producers land independently, so pass only what this one actually accepts
    # instead of exploding on an argument another agent has not added.
    try:
        sig = inspect.signature(fn)
        params = {k: v for k, v in wanted.items() if k in sig.parameters}
    except (TypeError, ValueError):
        params = wanted
    try:
        result = plan(a.method, Path(a.merged), **params)
    except NotImplementedError as exc:
        print(f"producer {a.method!r} not available: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"producer": result.producer, "params": result.params,
                      "summary": result.summary}, indent=2, ensure_ascii=False))
    if a.out:
        print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
