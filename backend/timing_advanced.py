"""The two search-based timing producers: `word_anchored` (M4) and `dp` (M7).

Why this lives in its own module rather than in `backend.timing`
----------------------------------------------------------------
`backend/timing.py` holds the integration CONTRACT -- the dataclasses every
producer emits, the shared source/TTS helpers, and the dispatcher. That file is
being edited concurrently by other producers, and a contract file that three
agents append to at once is a merge conflict waiting to happen. So the contract
stays in one place and is imported, never copied, and the two heavyweight
producers live here. The bottom of this module assigns them onto the timing
module, so `timing.plan("dp", ...)` keeps working through the existing
dispatcher with no edit to it: importing this module is the registration.

What a reader should know before touching anything here
-------------------------------------------------------
Time conventions, restated because mixing them up is the bug this whole phase
is most likely to produce:

  * `slot`, `src_speech`, `src_pauses`, `dst_start`     ABSOLUTE output seconds
  * `Piece.src_span`                                    RELATIVE to the raw clip
  * `segments[].words[]`                                ABSOLUTE
  * `segments[].tts_words[]`                            CLIP-RELATIVE

Functions below name their units in the signature docstring wherever a float
could plausibly be either.

The cost model
--------------
The cost formulas live HERE (see `_COST_DEFINITIONS` and `segment_cost`), not
in `backend.timing`, and are written so the DP's two degenerate cases hold by
construction:

  lambda_pause = lambda_speechrate = lambda_phoneme = 0  ->  naive_slot placement
  lambda_duration = 0, lambda_pause = 1                  ->  pause_aware onsets

The second case is checked against the real `timing.plan_pause_aware`, which
`_resolve_fallback` looks up at call time. Note that it reproduces that
producer's SEMANTICS, not its output byte for byte: on a segment whose clip
overruns, pause_aware is held to one tempo per segment while the DP may re-cut,
so the DP hits onsets pause_aware misses. `tests/test_timing_plan.py` pins that
difference as a strict xfail rather than letting it look like a regression.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .timing import (
    DEFAULT_PAUSE_MIN_S,
    Piece,
    SegmentPlan,
    Span,
    TimingPlan,
    _base_segment_plan,
    _clamp_tempo,
    raw_clip_duration,
    source_speech_and_pauses,
    summarize,
)

log = logging.getLogger(__name__)

# ffmpeg's atempo/rubberband filters silently DROP an input shorter than their
# analysis window (~0.01 s), which is how Phase 0 found itself writing empty
# files. No producer here may emit a piece the renderer would swallow, so every
# cut candidate is held this far from its neighbours.
MIN_RENDERABLE_PIECE_S = 0.05

# Costs that are mathematically equal can differ in the last bits purely from
# the order the terms were summed in: in the undershoot case the slot-start and
# source-onset paths both score 1.5666666666666673 and 1.566666666666667, and
# a raw `min` took the second. Round before comparing so genuine ties fall
# through to the key tie-break instead of being decided by float association.
# 1e-9 s is nine orders of magnitude below anything audible.
COST_TIE_NDIGITS = 9

# Sentence/clause terminators. Devanagari danda and double danda first because
# the target side is Indic; the ASCII set covers the English source side and
# Indic text that kept Latin punctuation.
_CLAUSE_PUNCT = "।॥.!?;:,"
_HARD_PUNCT = "।॥.!?"

_COST_DEFINITIONS = {
    "D_duration": (
        "|placement - slot.start| + |segment dst_end - (slot.end - safety_margin_s)|. "
        "Anchors the segment AS A BLOCK to its slot. The onset term is what makes "
        "the pure-duration optimum unique (many tempo/onset pairs share an end time)."
    ),
    "D_pause": (
        "sum_k |dst_start_k - src_speech[k].start| over matched pieces, plus "
        "sum of src_speech[j].dur over source speech runs left with no piece. "
        "Anchors pieces WITHIN the segment to the source speech onsets, and makes "
        "one-piece-per-source-run the optimal piece count."
    ),
    "D_speechrate": (
        "sum_k src_dur_k * |ln(local_tempo_k)|. Seconds-weighted log tempo, so "
        "0.9x and 1.11x cost the same and a long distorted piece costs more than "
        "a short one."
    ),
    "D_phoneme": (
        "seconds of viseme-class disagreement between src_visemes[] (absolute) and "
        "tts_visemes[] (clip-relative, mapped through the piece placement), sampled "
        "at viseme_frame_s. Skipped entirely when lambda_phoneme == 0."
    ),
}


# ---------------------------------------------------------------------------
# shared small helpers
# ---------------------------------------------------------------------------

def _seg_text(seg: dict) -> str:
    return seg.get("final_text") or seg.get("mt_text") or seg.get("text") or ""


def _resolve_fallback(name: str) -> Optional[Callable]:
    """Look up a fallback producer on the timing module AT CALL TIME.

    Late-bound rather than imported at module scope because this module is
    itself imported FROM `backend.timing` (see the registration block at the
    bottom), so a top-level lookup would run mid-import and miss producers
    defined after that point.
    """
    from . import timing as _t

    return getattr(_t, f"plan_{name}", None)


def _nonoverlapping(pieces: List[Piece]) -> List[Piece]:
    """Push each piece forward so it starts no earlier than the previous one ends.

    Overlapping pieces from the SAME clip would render as the clip talking over
    itself, which no amount of mixing fixes. Cross-segment overlap is left to
    the caller's shift-forward cursor; this is the within-segment guard.
    """
    out: List[Piece] = []
    cursor = -math.inf
    for p in pieces:
        if p.dst_start < cursor:
            p.dst_start = cursor
        out.append(p)
        cursor = p.dst_end
    return out


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------

def _onset_targets(sp: SegmentPlan, n_pieces: int) -> List[float]:
    """Absolute seconds each piece WANTS to start at, one per piece.

    Beyond the last source speech run the target saturates at that run's onset;
    the penalty for having too many pieces then comes out of D_pause's
    unmatched-run term rather than from a silently wrapping index.
    """
    runs = sp.src_speech or [sp.slot]
    return [runs[min(k, len(runs) - 1)].start for k in range(n_pieces)]


def _d_pause(sp: SegmentPlan, pieces: Sequence[Piece]) -> float:
    runs = sp.src_speech or [sp.slot]
    targets = _onset_targets(sp, len(pieces))
    err = sum(abs(p.dst_start - t) for p, t in zip(pieces, targets))
    # A source speech run with no piece of its own is a pause we failed to
    # reinsert; charge its length so that K == len(runs) is the optimum.
    unmatched = sum(r.dur for r in runs[len(pieces):])
    return err + unmatched


def _d_duration(sp: SegmentPlan, pieces: Sequence[Piece], *, safety_margin_s: float) -> float:
    if not pieces:
        return 0.0
    placement = pieces[0].dst_start
    end = max(p.dst_end for p in pieces)
    target_end = sp.slot.end - safety_margin_s
    return abs(placement - sp.slot.start) + abs(end - target_end)


def _d_speechrate(pieces: Sequence[Piece]) -> float:
    return sum(p.src_span.dur * abs(math.log(max(1e-6, p.local_tempo))) for p in pieces)


def _parse_visemes(raw) -> List[Tuple[float, float, str]]:
    """Accept either [{start,end,viseme|phone|label}] or [[start,end,label]]."""
    out: List[Tuple[float, float, str]] = []
    for v in raw or []:
        try:
            if isinstance(v, dict):
                lab = v.get("viseme") or v.get("phone") or v.get("label") or ""
                out.append((float(v["start"]), float(v["end"]), str(lab)))
            else:
                out.append((float(v[0]), float(v[1]), str(v[2]) if len(v) > 2 else ""))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out


def _viseme_at(vis: Sequence[Tuple[float, float, str]], t: float) -> Optional[str]:
    # Linear scan is fine: a segment holds tens of visemes, not thousands, and
    # the whole term is skipped unless lambda_phoneme > 0.
    for s, e, lab in vis:
        if s <= t < e:
            return lab
    return None


def _d_phoneme(
    seg: dict,
    pieces: Sequence[Piece],
    *,
    frame_s: float = 0.02,
) -> Tuple[float, Optional[str]]:
    """Seconds of viseme disagreement; (0.0, reason) when it cannot be measured."""
    src = _parse_visemes(seg.get("src_visemes"))
    tts = _parse_visemes(seg.get("tts_visemes"))
    if not src or not tts:
        return 0.0, "no_visemes"
    bad = 0.0
    for p in pieces:
        n = max(1, int(p.dst_dur / frame_s))
        for i in range(n):
            dst_t = p.dst_start + i * frame_s
            # Walk back through the piece's own mapping: dst -> clip-relative.
            clip_t = p.src_span.start + (dst_t - p.dst_start) * p.local_tempo
            a = _viseme_at(tts, clip_t)
            b = _viseme_at(src, dst_t)
            if a is not None and b is not None and a != b:
                bad += frame_s
    return bad, None


def segment_cost(
    seg: dict,
    sp: SegmentPlan,
    pieces: Sequence[Piece],
    *,
    lambda_duration: float,
    lambda_pause: float,
    lambda_speechrate: float,
    lambda_phoneme: float,
    safety_margin_s: float,
    viseme_frame_s: float = 0.02,
) -> Tuple[float, dict]:
    """The one cost function. Returns (total, components-for-SegmentPlan.cost).

    `lambda_phoneme == 0` short-circuits the viseme term -- it is not computed
    and then multiplied by zero, because computing it means a per-frame scan
    inside the DP's innermost loop.
    """
    parts: dict = {}
    total = 0.0
    if lambda_duration:
        d = _d_duration(sp, pieces, safety_margin_s=safety_margin_s)
        parts["D_duration"] = round(d, 5)
        total += lambda_duration * d
    if lambda_pause:
        d = _d_pause(sp, pieces)
        parts["D_pause"] = round(d, 5)
        total += lambda_pause * d
    if lambda_speechrate:
        d = _d_speechrate(pieces)
        parts["D_speechrate"] = round(d, 5)
        total += lambda_speechrate * d
    if lambda_phoneme:
        d, why = _d_phoneme(seg, pieces, frame_s=viseme_frame_s)
        parts["D_phoneme"] = round(d, 5)
        if why:
            parts["D_phoneme_skipped"] = why
        total += lambda_phoneme * d
    parts["total"] = round(total, 5)
    return total, parts


# ---------------------------------------------------------------------------
# M4: word_anchored
# ---------------------------------------------------------------------------

def _tokenize_words(seg: dict, key: str) -> List[dict]:
    out = []
    for w in seg.get(key) or []:
        try:
            s, e = w.get("start"), w.get("end")
            if s is None or e is None:
                continue
            out.append({"text": str(w.get("text") or w.get("word") or ""),
                        "start": float(s), "end": float(e)})
        except Exception:
            continue
    out.sort(key=lambda w: w["start"])
    return out


def _synth_tts_words(seg: dict, raw: float) -> List[dict]:
    """Pseudo clip-relative word timings, spread over the clip by character mass.

    Used only when `require_tts_words=False`. It is NOT forced alignment -- it
    assumes a constant characters-per-second rate -- but it is monotone and
    lets clause cutting run before the TTS forced-aligner lands. Every segment
    built this way is flagged in `qc`, so no measurement can quietly treat it
    as real alignment.
    """
    toks = [t for t in re.split(r"\s+", _seg_text(seg)) if t]
    if not toks or raw <= 0:
        return []
    mass = [max(1, len(t)) for t in toks]
    total = float(sum(mass))
    out, acc = [], 0.0
    for t, m in zip(toks, mass):
        s = raw * (acc / total)
        acc += m
        e = raw * (acc / total)
        out.append({"text": t, "start": s, "end": e})
    return out


def _clause_groups(words: List[dict], *, max_gap_s: float) -> List[Tuple[int, int]]:
    """Group word indices into clauses -- [start, end) index pairs.

    Cuts after punctuation, and also after an unusually long inter-word gap,
    because a pause IS a clause boundary whether or not the text wrote one.
    """
    if not words:
        return []
    groups: List[Tuple[int, int]] = []
    start = 0
    for i, w in enumerate(words):
        last = i == len(words) - 1
        txt = (w.get("text") or "").strip()
        ends_clause = bool(txt) and txt[-1] in _CLAUSE_PUNCT
        gap_break = (not last) and (words[i + 1]["start"] - w["end"]) >= max_gap_s
        if last or ends_clause or gap_break:
            groups.append((start, i + 1))
            start = i + 1
    return groups


def _merge_short(
    groups: List[Tuple[int, int]],
    words: List[dict],
    *,
    min_piece_s: float,
    max_groups: int,
) -> List[Tuple[int, int]]:
    """Enforce min_piece_s then max_groups, both by merging neighbours.

    Short pieces go first because a sub-min piece is a correctness problem (the
    renderer may drop it); the cap is only a tidiness bound, so it merges the
    cheapest adjacent pair -- the one with the smallest gap between them.
    """
    def dur(g):
        return words[g[1] - 1]["end"] - words[g[0]]["start"]

    gs = list(groups)
    i = 0
    while i < len(gs) and len(gs) > 1:
        if dur(gs[i]) < min_piece_s:
            j = i - 1 if i > 0 else i + 1
            a, b = min(i, j), max(i, j)
            gs[a] = (gs[a][0], gs[b][1])
            del gs[b]
            i = max(0, a - 1)
        else:
            i += 1
    while len(gs) > max_groups:
        gaps = [words[gs[i + 1][0]]["start"] - words[gs[i][1] - 1]["end"]
                for i in range(len(gs) - 1)]
        i = min(range(len(gaps)), key=lambda k: gaps[k])
        gs[i] = (gs[i][0], gs[i + 1][1])
        del gs[i + 1]
    return gs


def _cum_bounds(
    words: List[dict], groups: List[Tuple[int, int]], *, by: str,
) -> Tuple[List[float], List[float]]:
    """(start, end) cumulative fraction in [0,1] for each group.

    BOTH ends, not just the end, because the two are used for different
    things and conflating them is what broke this producer: a clause is
    ANCHORED by where its content STARTS, and SIZED by how much of the stream
    it covers. `_cum_mass` used to return only the end fractions, so the
    anchor was derived from the end -- see `_match_proportional`.

    `by` picks what "how much content" means:

      "time"   each word contributes its OWN DURATION, and the gaps between
               words contribute nothing. This is the default, and it is the
               only basis that uses the input this producer is named after.
               Counting words or characters instead throws the word timings
               away at exactly the step that decides placement, and the two
               are not interchangeable: 40 source words spoken over 12 s of
               speech separated by four pauses do not distribute their
               content uniformly over the index. Measured over the four
               word-bearing merged JSONs on disk, moving from word count to
               duration took the median piece-onset error from 827/346/0 ms
               to 495/27/0 ms on the three that have room to differ.
      "chars"  character mass -- the fallback when the word timings are all
               zero-width (some aligners emit point timings), which would
               make "time" a division by zero.
      "words"  word count. Kept so the earlier behaviour is still reachable.
    """
    if by == "time":
        unit = [max(0.0, float(w.get("end", 0.0)) - float(w.get("start", 0.0)))
                for w in words]
        if sum(unit) <= 0:                       # point timings; see docstring
            by = "chars"
    if by == "chars":
        unit = [max(1, len((w.get("text") or "").strip())) for w in words]
    elif by != "time":
        unit = [1.0] * len(words)
    tot = float(sum(unit)) or 1.0
    starts: List[float] = []
    ends: List[float] = []
    acc = 0.0
    for a, b in groups:
        starts.append(acc / tot)
        acc += sum(unit[a:b])
        ends.append(acc / tot)
    return starts, ends


def _word_cum(words: List[dict], *, by: str) -> Tuple[List[float], List[float]]:
    """(start, end) cumulative fraction in [0,1] for each individual WORD."""
    return _cum_bounds(words, [(i, i + 1) for i in range(len(words))], by=by)


def _groups_from_runs(
    words: List[dict], runs: Sequence[Span],
) -> List[Tuple[int, int]]:
    """Index groups over `words` induced by the segment's SPEECH RUNS.

    The source anchor set must be the acoustic events, not the punctuation.
    `_clause_groups` cuts on a terminator as well as on a gap, and a comma the
    speaker did not pause at is not an onset: anchoring a piece there invents a
    speech start the source does not have, which is precisely what the
    governing constraint forbids. It also costs a piece out of
    `max_pieces_per_segment` that a real onset then does not get.

    Measured on `data/mt/2vxQaT0Fy-c_60to80s_720p.indictrans2.hi.mt.json`,
    whose single segment has 4 speech runs: `_clause_groups` found 5 source
    groups, the surplus piece landed mid-run, and the median piece-onset error
    was 1052 ms against pause_aware's 327 ms. Grouping by run instead is
    346 ms before the other two fixes in this function and 27 ms after.

    Deriving the groups from `sp.src_speech` rather than re-splitting on the
    gap keeps this EXACTLY consistent with `_onset_targets`, `_d_pause` and
    `plan_pause_aware`, which all key off the same runs -- including the case
    where `source_speech_and_pauses` REFUSED an off-timeline word stream and
    returned one slot-wide run. There every word falls in that one group, the
    segment gets one piece at the slot start, and the corrupt word times
    cannot steer placement. That is the right degradation, and the clause
    split did not have it.
    """
    if not words:
        return []
    if not runs:
        return [(0, len(words))]
    groups: List[Tuple[int, int]] = []
    i = 0
    for r in runs:
        a = i
        while i < len(words) and words[i]["start"] < r.end + 1e-9:
            i += 1
        if i > a:
            groups.append((a, i))
    if not groups:
        return [(0, len(words))]
    if i < len(words):          # words past the last run: keep them, do not drop
        groups[-1] = (groups[-1][0], len(words))
    return groups


def _source_clause_cuts(
    tts_words: List[dict], src_words: List[dict],
    src_groups: List[Tuple[int, int]], *, by: str,
) -> List[int]:
    """Target word indices that line up, in mass, with the SOURCE RUN boundaries.

    Why this is word_anchored's business and not a borrowed pause_aware trick:
    the target side only ever yields a clause boundary where its own text
    punctuates or its own word timings leave a gap, and Indic MT output
    punctuates sparsely -- on the dress-rehearsal clip one segment produced a
    SINGLE target clause against four source speech runs, which is naive
    placement wearing this producer's name. The source words[] stream is the
    input this method exists to use, so its run boundaries are allowed to
    choose cut points in the target, at word granularity. `_merge_short` then
    still enforces `min_piece_s` and `max_pieces_per_segment`, so this can
    only ever propose cuts, never force an unrenderable one.

    Returns `len(src_groups) - 1` indices at most. It CANNOT express "no cut
    at all", so the single-run case is handled by the caller, not here.
    """
    if not tts_words or len(src_groups) < 2:
        return []
    _ws, we = _word_cum(tts_words, by=by)
    _ss, se = _cum_bounds(src_words, src_groups, by=by)
    out: List[int] = []
    for c in se[:-1]:
        i = next((i for i in range(len(we)) if we[i] > c + 1e-9), None)
        if i is not None and i > 0:
            out.append(i)
    return out


def _match_proportional(
    tgt_start: List[float], src_start: List[float],
) -> List[int]:
    """Monotone target-group -> source-group map: NEAREST start mass.

    Target group k maps to the source group whose START mass is closest to
    target group k's start mass, subject to the map never going backwards.
    The caller anchors the piece at that group's onset, so the quantity
    matched is the one the anchor is read off: a start to a start, and the
    comparison is a distance, not a containment test.

    Two earlier rules both lost a clause here, in opposite directions:

      * comparing target END mass to source END mass and then anchoring on the
        matched group's START is an off-by-one in mass space; every piece after
        the first landed one source group LATE, so the error grew with group
        length across a segment.
      * "the first source group whose END mass is past the target's start mass"
        is off by one the other way whenever the two mass streams agree only
        approximately -- which is the normal case, because the target cut index
        chosen by `_source_clause_cuts` is the first target word that CROSSES
        the source boundary, so the target group's start mass sits just BELOW
        the source boundary it was derived from. Measured on the dress clip's
        SEG_0000, target starts [0, .161, .214, .321, .464, .607, .750, .857]
        against source starts [0, .163, .224, .327, .469, .612, .755, .857] --
        visibly the identity map -- came back as [0,0,1,2,3,4,5,7]. An exact
        tie resolved correctly and a 0.002-of-mass near-miss did not, which
        is why the bug survived the fixtures: the fixture masses tie exactly.

    Nearest-start has no such edge: it is insensitive to which side of the
    boundary the crossing index landed on. On these inputs it returns the same
    map as the global DP in `_match_monotone_dtw` (verified on all four
    word-bearing merged JSONs), at a fraction of the cost.

    Monotone by construction: `j` never decreases. Group counts need not agree.
    """
    out, j = [], 0
    n = len(src_start)
    if n == 0:
        return [0] * len(tgt_start)
    for c in tgt_start:
        # Strict `<` so an exact tie keeps the earlier group: skipping a source
        # onset is the one error that cannot be recovered downstream.
        while j + 1 < n and abs(src_start[j + 1] - c) < abs(src_start[j] - c):
            j += 1
        out.append(j)
    return out


def _match_monotone_dtw(tgt_cum: List[float], src_cum: List[float]) -> List[int]:
    """Monotone alignment minimising cumulative-mass disagreement.

    A real DP rather than a greedy walk: every target clause is consumed exactly
    once, source clauses may be consumed by several target clauses or skipped,
    and the mapping cannot go backwards. That last constraint is the point --
    a non-monotone match would reorder speech.
    """
    n, m = len(tgt_cum), len(src_cum)
    if n == 0 or m == 0:
        return [0] * n
    INF = float("inf")
    # cost[i][j] = best cost aligning target clauses 0..i with target i -> source j
    cost = [[INF] * m for _ in range(n)]
    back = [[-1] * m for _ in range(n)]
    for j in range(m):
        cost[0][j] = abs(tgt_cum[0] - src_cum[j])
    for i in range(1, n):
        best_prev, best_j = INF, -1
        for j in range(m):
            if cost[i - 1][j] < best_prev:
                best_prev, best_j = cost[i - 1][j], j
            # best_prev/best_j now cover source indices 0..j, enforcing j' <= j
            if best_prev < INF:
                cost[i][j] = best_prev + abs(tgt_cum[i] - src_cum[j])
                back[i][j] = best_j
    j = min(range(m), key=lambda k: cost[n - 1][k])
    out = [0] * n
    for i in range(n - 1, -1, -1):
        out[i] = j
        j = back[i][j] if i > 0 else j
    return out


def plan_word_anchored(
    merged_json: Path,
    *,
    unit: str = "clause",
    min_piece_s: float = 0.35,
    max_pieces_per_segment: int = 12,
    anchor: str = "proportional",
    max_tempo: float = 1.15,
    min_tempo: float = 0.90,
    require_tts_words: bool = True,
    fallback: str = "pause_aware",
    out_json: Optional[Path] = None,
    pause_min_s: float = DEFAULT_PAUSE_MIN_S,
    clause_gap_s: float = 0.25,
    shift_forward: bool = True,
    safety_margin_s: float = 0.10,
    split_on_source_clauses: bool = True,
    mass_basis: str = "time",
) -> TimingPlan:
    """Cut the TTS clip at clause boundaries and place each clause on its source.

    The premise: a dub is in sync when speech EVENTS line up, and a clause is
    the smallest unit whose boundary a viewer actually perceives. Cutting the
    clip at clause boundaries from `tts_words[]` (clip-relative) and placing
    each chunk on the SOURCE SPEECH RUN it corresponds to (absolute, from
    `words[]`) moves the onsets to where the source put them, instead of
    moving them all by one uniform tempo the way `naive_slot` does.

    Three choices make the placement, and all three were wrong at once in the
    dress rehearsal. Each is documented where it is implemented, because each
    was a separate measurable defect:

      WHAT THE ANCHORS ARE.  The source speech runs, via `_groups_from_runs` --
        not `_clause_groups`, which also cuts on punctuation and so invents
        onsets at commas the speaker did not pause at.
      HOW CONTENT IS MEASURED.  Seconds of word, on both sides, via
        `_cum_bounds(by="time")` -- not word or character count, which throws
        away the timings this producer is named after. `mass_basis` reaches
        the old bases ("words", "chars") for comparison.
      HOW THE TWO ARE MATCHED.  `anchor` picks the algorithm; both compare a
        target group's START mass against a source run's START mass:
          "proportional"  nearest start, greedy and monotone (`_match_proportional`)
          "monotone_dtw"  nearest start, globally optimal (`_match_monotone_dtw`)
        On all four word-bearing merged JSONs on disk the two agree exactly;
        the DP is the safety net for an input where the greedy is fooled.

    Either way the match is monotone -- never reordering -- which is the only
    property the renderer depends on.

    A segment missing word timings on EITHER side degrades to `fallback` and
    says so in `qc`; one unaligned segment must not kill a sweep.
    """
    if mass_basis not in ("time", "chars", "words"):
        raise ValueError(
            f"mass_basis must be time|chars|words, got {mass_basis!r}")
    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    params = {
        "unit": unit, "min_piece_s": min_piece_s,
        "max_pieces_per_segment": max_pieces_per_segment, "anchor": anchor,
        "max_tempo": max_tempo, "min_tempo": min_tempo,
        "require_tts_words": require_tts_words, "fallback": fallback,
        "pause_min_s": pause_min_s, "clause_gap_s": clause_gap_s,
        "shift_forward": shift_forward, "safety_margin_s": safety_margin_s,
        "split_on_source_clauses": split_on_source_clauses,
        "mass_basis": mass_basis,
    }
    plan = TimingPlan(producer="word_anchored", params=params)

    # The fallback producer is run ONCE over the whole file, and individual
    # degraded segments are lifted out of it. Running it per segment would mean
    # re-reading and re-probing the merged JSON for every bad segment.
    fb_by_id: Dict[str, SegmentPlan] = {}
    fb_fn = _resolve_fallback(fallback)
    if fb_fn is not None:
        try:
            # Forward the knobs the two producers share. Calling the fallback
            # bare let it run on ITS OWN defaults, so `shift_forward=False`
            # here silently produced a shift-forward plan -- word_anchored
            # scored 3.25 s median onset error against pause_aware's 0.00 on
            # the same file, purely because the kwarg was dropped. Only pass
            # what the callee actually accepts, so a fallback with a different
            # signature still works.
            import inspect
            shared = {"pause_min_s": pause_min_s, "max_tempo": max_tempo,
                      "min_tempo": min_tempo, "shift_forward": shift_forward}
            try:
                accepted = set(inspect.signature(fb_fn).parameters)
                kw = {k: v for k, v in shared.items() if k in accepted}
            except (TypeError, ValueError):       # builtins / C callables
                kw = {}
            fb_plan = fb_fn(merged_json, **kw)
            fb_by_id = {s.id: s for s in fb_plan.segments}
        except Exception as exc:      # a broken fallback must not kill the run
            log.warning("word_anchored: fallback %r failed (%s); "
                        "degraded segments will be left empty", fallback, exc)

    cursor = 0.0
    for seg in data.get("segments", []):
        sp = _base_segment_plan(seg, pause_min_s=pause_min_s)
        raw = raw_clip_duration(seg)

        def degrade(reason: str) -> None:
            nonlocal cursor
            sp.qc["degraded_to"] = fallback
            sp.qc["degrade_reason"] = reason
            fb = fb_by_id.get(sp.id)
            if fb is not None and fb.pieces:
                sp.pieces = [Piece(src_span=Span(p.src_span.start, p.src_span.end),
                                   dst_start=max(p.dst_start, cursor)
                                   if shift_forward else p.dst_start,
                                   local_tempo=p.local_tempo, anchor=p.anchor,
                                   text=p.text, src_word_range=p.src_word_range)
                             for p in fb.pieces]
                sp.pieces = _nonoverlapping(sp.pieces)
                sp.qc.update({k: v for k, v in fb.qc.items() if k != "degraded_to"})
                cursor = max(p.dst_end for p in sp.pieces)
            else:
                sp.qc["no_fallback"] = True

        if raw is None or raw <= 0:
            sp.qc["no_audio"] = True
            plan.segments.append(sp)
            continue

        src_words = _tokenize_words(seg, "words")
        tts_words = _tokenize_words(seg, "tts_words")
        if not tts_words:
            if require_tts_words:
                degrade("no_tts_words")
                plan.segments.append(sp)
                continue
            tts_words = _synth_tts_words(seg, raw)
            sp.qc["tts_words"] = "synthesized_uniform"
        if not tts_words or not src_words:
            degrade("no_tts_words" if not tts_words else "no_src_words")
            plan.segments.append(sp)
            continue

        # Cumulative mass is measured in SECONDS OF WORD on both sides. See
        # `_cum_bounds`; `mass_basis` exists only to get the earlier
        # count-based behaviour back for a comparison.
        by = mass_basis
        tgt_groups = _clause_groups(tts_words, max_gap_s=clause_gap_s)
        # Source anchors are the SPEECH RUNS, not the punctuation clauses --
        # see `_groups_from_runs`.
        src_groups = _groups_from_runs(src_words, sp.src_speech)
        if unit == "word":
            tgt_groups = [(i, i + 1) for i in range(len(tts_words))]
            src_groups = [(i, i + 1) for i in range(len(src_words))]
        tgt_groups = _merge_short(tgt_groups, tts_words, min_piece_s=min_piece_s,
                                  max_groups=max_pieces_per_segment)
        if (unit != "word" and split_on_source_clauses
                and len(src_groups) != len(tgt_groups)):
            # `!=`, not `>`: the right number of pieces is one per source
            # speech run in BOTH directions. Too few target clauses leaves a
            # source onset with no piece; too many is just as bad, and less
            # obviously so. A surplus target clause is matched to a run that
            # already has a piece, `j_hi = max(j_lo, match[k+1] - 1)` then
            # collapses to the single run `j_lo`, and the chunk is sized
            # against a region it does not cover. Measured on
            # 2vxQaT0Fy-c_60to80s_720p (4 speech runs): target punctuation
            # produced 5 clauses, and the median piece-onset error was
            # 1052 ms. Re-cutting to 4 makes it 27 ms.
            #
            # Compared AFTER `_merge_short`, not before: the raw clause split
            # is routinely finer than what survives `min_piece_s`, so a
            # pre-merge count says the target has plenty of boundaries when
            # the plan is about to be built from five. On the dress-rehearsal
            # clip SEG_0000 that is 9 raw target clauses collapsing to 5
            # against 8 source runs -- the pre-merge test read 8 > 9 and
            # declined to help.
            # REPLACE the target grouping rather than union the two. The right
            # number of pieces is one per source run: that is what gives every
            # source onset a piece of its own and leaves none over. Unioning
            # the two boundary sets overshoots it and the surplus pieces land
            # on no source run at all -- measured on the dress clip, union
            # emits 23 pieces against 19 source runs with 6 unmatched, replace
            # emits 18 with 1.
            regroup: Optional[List[Tuple[int, int]]] = None
            if len(src_groups) == 1:
                # One source run means one piece, and `_source_clause_cuts`
                # cannot say so: it proposes BOUNDARIES, and the right number
                # here is zero, which is indistinguishable from "found none".
                # This is the case the refused off-timeline word streams land
                # in (`source_speech_and_pauses` returns one slot-wide run),
                # and without it a corrupt stream still got 6 target clauses
                # cut against the 1 run it is allowed to know about.
                regroup = [(0, len(tts_words))]
            else:
                extra = _source_clause_cuts(tts_words, src_words, src_groups, by=by)
                if extra:
                    cuts = [0] + sorted(set(extra)) + [len(tts_words)]
                    regroup = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
                    sp.qc["src_run_cuts"] = len(extra)
            if regroup:
                tgt_groups = _merge_short(
                    regroup, tts_words, min_piece_s=min_piece_s,
                    max_groups=max_pieces_per_segment)
        if not tgt_groups or not src_groups:
            degrade("no_clauses")
            plan.segments.append(sp)
            continue

        tgt_start, _tgt_end = _cum_bounds(tts_words, tgt_groups, by=by)
        src_start_cum, _src_end_cum = _cum_bounds(src_words, src_groups, by=by)
        # Both modes match a target clause's START mass to a source clause's
        # START mass. monotone_dtw gets the start fractions for the same
        # reason proportional does -- what is being located is an onset.
        match = (_match_monotone_dtw(tgt_start, src_start_cum)
                 if anchor == "monotone_dtw"
                 else _match_proportional(tgt_start, src_start_cum))
        # Both producers above are monotone by construction; enforce it anyway
        # so a future matcher cannot silently reorder speech.
        for k in range(1, len(match)):
            if match[k] < match[k - 1]:
                match[k] = match[k - 1]

        n_src = len(src_groups)
        pieces: List[Piece] = []
        capped = False
        natural_rate = 0
        for k, (a, b) in enumerate(tgt_groups):
            # Clip-relative span. The first piece starts at 0.0 and the last
            # ends at `raw` so no audio is dropped on the floor.
            s = 0.0 if k == 0 else tts_words[a]["start"]
            e = raw if k == len(tgt_groups) - 1 else tts_words[b - 1]["end"]
            e = max(e, s + MIN_RENDERABLE_PIECE_S)
            # The source region this target clause covers runs from its own
            # matched clause up to the clause before the NEXT target clause's
            # anchor (and, for the final target clause, to the end of the
            # source speech). Sizing the piece against the matched clause
            # ALONE was the second half of the bug: when the clause counts
            # disagree -- 5 target clauses over 8 source clauses on the
            # dress-rehearsal clip -- a 5.2 s chunk was asked to fit a single
            # 1.58 s clause, demanded tempo 3.3, hit the 1.15 cap, and the
            # 2.9 s overrun was handed to the next piece by `_nonoverlapping`.
            # That is where the per-segment +3.4..+6.5 s slot overruns came
            # from, and why the lateness compounded along a segment.
            j_lo = match[k]
            j_hi = (n_src - 1 if k == len(tgt_groups) - 1
                    else max(j_lo, match[k + 1] - 1))
            src_start = src_words[src_groups[j_lo][0]]["start"]        # absolute
            src_dur = max(1e-3,
                          src_words[src_groups[j_hi][1] - 1]["end"] - src_start)
            want = (e - s) / src_dur
            tempo, was = _clamp_tempo(want, min_tempo, max_tempo)
            if want < tempo <= 1.0:
                # The chunk is SHORTER than the source region it covers. The
                # clamp would stretch it to 1/min_tempo to fill the room,
                # which smears speech across a silence the source actually
                # had. Play it at its natural rate instead and let the
                # remainder BE the pause -- preserving pauses is the point of
                # this producer, and `min_tempo` is a floor on deliberate
                # slowing, not an instruction to slow down.
                tempo, was = 1.0, False
                natural_rate += 1
            capped = capped or was
            pieces.append(Piece(src_span=Span(s, min(e, raw)), dst_start=src_start,
                                local_tempo=tempo, anchor=unit,
                                text=" ".join(w["text"] for w in tts_words[a:b]),
                                src_word_range=(a, b)))

        if shift_forward and pieces and pieces[0].dst_start < cursor:
            shift = cursor - pieces[0].dst_start
            for p in pieces:
                p.dst_start += shift
            sp.qc["shifted_s"] = round(shift, 4)
        sp.pieces = _nonoverlapping(pieces)
        if capped:
            sp.qc["tempo_cap_hit"] = True
        sp.qc["n_src_runs"] = len(src_groups)
        sp.qc["n_tgt_clauses"] = len(tgt_groups)
        sp.qc["mass_basis"] = by
        if natural_rate:
            sp.qc["natural_rate_pieces"] = natural_rate
        cursor = max(p.dst_end for p in sp.pieces)
        _, parts = segment_cost(seg, sp, sp.pieces, lambda_duration=1.0,
                                lambda_pause=1.0, lambda_speechrate=1.0,
                                lambda_phoneme=0.0, safety_margin_s=safety_margin_s)
        sp.cost = parts
        plan.segments.append(sp)

    plan.summary = summarize(plan)
    plan.summary["degraded_segments"] = sum(
        1 for s in plan.segments if s.qc.get("degraded_to"))
    if out_json:
        plan.write(Path(out_json))
    log.info("timing[word_anchored/%s]: %d segments, %d pieces, %d degraded",
             anchor, plan.summary.get("segments", 0),
             plan.summary.get("pieces", 0), plan.summary["degraded_segments"])
    return plan


# ---------------------------------------------------------------------------
# M7: dp
# ---------------------------------------------------------------------------

def _cut_candidates(
    seg: dict,
    sp: SegmentPlan,
    raw: float,
    *,
    allow_split: bool,
    max_cut_candidates: int,
) -> List[float]:
    """Clip-relative cut offsets the DP may split at, always including 0 and raw.

    Three sources, in priority order:
      1. `tts_words[]` gaps -- real silences in the synthesised audio, the only
         places a cut is inaudible.
      2. the SOURCE-speech-proportional cuts -- included unconditionally so the
         pause-aware optimum is actually inside the search space. Without these
         the DP could only approach onset anchoring, never reproduce it, and
         the second degenerate case would fail for a reason that is about the
         candidate set rather than about the cost.
      3. a uniform grid, as filler when neither of the above gives enough.
    """
    if not allow_split:
        return [0.0, raw]
    cands = {0.0, raw}

    runs = sp.src_speech or []
    total_run = sum(r.dur for r in runs)
    if total_run > 0:
        acc = 0.0
        for r in runs[:-1]:
            acc += r.dur
            cands.add(raw * (acc / total_run))

    tw = _tokenize_words(seg, "tts_words")
    if tw:
        gaps = sorted(
            ((tw[i + 1]["start"] - tw[i]["end"], 0.5 * (tw[i]["end"] + tw[i + 1]["start"]))
             for i in range(len(tw) - 1)),
            reverse=True)
        for _, t in gaps[:max_cut_candidates]:
            cands.add(t)

    n_fill = max(0, max_cut_candidates - len(cands))
    for i in range(1, n_fill + 1):
        cands.add(raw * i / (n_fill + 1))

    # Drop anything the renderer would swallow, keeping 0 and raw.
    out = []
    for c in sorted(cands):
        c = min(max(0.0, c), raw)
        if out and c - out[-1] < MIN_RENDERABLE_PIECE_S:
            continue
        out.append(c)
    if out[-1] < raw - 1e-9:
        if raw - out[-1] < MIN_RENDERABLE_PIECE_S and len(out) > 1:
            out.pop()
        out.append(raw)
    return out


def _tempo_candidates(
    src_dur: float, target_dur: float, grid: Sequence[float],
    lo: float, hi: float,
) -> List[Tuple[float, bool]]:
    """Grid tempos PLUS the exact-fit tempo for this piece, all clamped.

    The exact-fit candidate is why the first degenerate case can reproduce
    `naive_slot` exactly rather than to within half a grid step: naive_slot
    computes a continuous tempo, and a fixed grid can never contain it.
    """
    out: Dict[float, bool] = {}
    for t in list(grid) + [src_dur / max(1e-6, target_dur)]:
        c, was = _clamp_tempo(float(t), lo, hi)
        out[round(c, 9)] = out.get(round(c, 9), False) or was
    return sorted(out.items())


def _bucket(t: float, ref: float, step: float) -> int:
    return int(round((t - ref) / step))


def _inner_dp(
    seg: dict,
    sp: SegmentPlan,
    raw: float,
    cuts: List[float],
    entry_cursor: float,
    *,
    lambda_duration: float,
    lambda_pause: float,
    lambda_speechrate: float,
    lambda_phoneme: float,
    tempo_grid: Sequence[float],
    pause_grid_s: Sequence[float],
    min_tempo: float,
    max_tempo: float,
    max_pieces_per_segment: int,
    drift_bucket_s: float,
    safety_margin_s: float,
    shift_forward: bool,
    inner_beam: int,
    viseme_frame_s: float,
) -> Tuple[float, List[Piece], List[Tuple[int, float]], bool]:
    """Best (cost, pieces, inserted_pauses, tempo_cap_hit) for ONE segment.

    State is (cut index, pieces placed so far, dst-time bucket). The dst time is
    carried as an exact float inside the state VALUE and bucketed only to form
    the dedup key -- if the time itself were snapped to the grid, a piece could
    never land exactly on a source onset like 0.031 s, and onset anchoring would
    be quantised away. Bucketing keeps the state count finite; the exact float
    keeps the placements exact.
    """
    targets = _onset_targets(sp, max_pieces_per_segment)
    runs = sp.src_speech or [sp.slot]
    end_target = sp.slot.end - safety_margin_s

    # The first piece's onset: the shift-forward floor, and the first source
    # speech onset. Two candidates, because those are the only two things any
    # of the cost terms can want.
    floor = max(sp.slot.start, entry_cursor) if shift_forward else -math.inf
    onsets = {max(floor, sp.slot.start), max(floor, runs[0].start)}
    if not shift_forward:
        onsets = {sp.slot.start, runs[0].start}

    n_cuts = len(cuts)
    # state key -> (cost_so_far, dst_time, parent_key, (tempo, pause, capped))
    frontier: Dict[Tuple[int, int, int], Tuple[float, float, object, object]] = {}
    for o in sorted(onsets):
        key = (0, 0, _bucket(o, sp.slot.start, drift_bucket_s))
        # D_duration is |placement - slot.start| + |end - end_target|, but the
        # terminal branch only ever charged the second half (its `placement`
        # was marked "recovered on reconstruction" and never was). Seeding the
        # placement error here is where it belongs: it is fixed the moment the
        # first onset is chosen, and charging it at the start makes every path
        # through this state carry it.
        #
        # It only changed the argmin in the UNDERSHOOT regime, where the two
        # terms cancel and cost falls with a later start -- the DP then picked
        # the source onset over the slot start (measured: 0.2000 vs
        # naive_slot's 0.0000 for a 3 s clip in a 5 s slot). In the overrun
        # regime that real Indic TTS always lands in, the end term alone is
        # increasing in the start time, so the answer was right by accident,
        # and tie-broken by set-iteration order when both candidates scored 0.
        c0 = lambda_duration * abs(o - sp.slot.start) if lambda_duration else 0.0
        if key not in frontier or c0 < frontier[key][0]:
            frontier[key] = (c0, o, None, None)

    all_states: Dict[Tuple[int, int, int], Tuple[float, float, object, object]] = dict(frontier)
    terminal: Dict[Tuple[int, int, int], float] = {}
    n_states = 0
    n_transitions = 0

    # Breadth-first over piece count: every transition increments k by one, so
    # the layers are acyclic and one pass per layer is a complete DP.
    for k in range(max_pieces_per_segment):
        layer = {key: v for key, v in frontier.items() if key[1] == k}
        if not layer:
            break
        n_states += len(layer)
        nxt: Dict[Tuple[int, int, int], Tuple[float, float, object, object]] = {}
        for key, (cost, t_now, _p, _c) in layer.items():
            i = key[0]
            src_start = cuts[i]
            pieces_left = max_pieces_per_segment - k
            for j in range(i + 1, n_cuts):
                piece_src_dur = cuts[j] - src_start
                if piece_src_dur < MIN_RENDERABLE_PIECE_S:
                    continue
                # A path that cannot reach the clip's end within the remaining
                # piece budget is dead; pruning it here keeps the layer small.
                if j < n_cuts - 1 and pieces_left <= 1:
                    continue
                last = j == n_cuts - 1
                onset_target = targets[k]
                nxt_target = targets[min(k + 1, len(targets) - 1)]
                # What duration would this piece need for the NEXT piece to land
                # on its own onset target with no inserted pause?
                want = max(1e-3, (nxt_target - t_now) if not last
                           else (end_target - t_now))
                for tempo, was_capped in _tempo_candidates(
                        piece_src_dur, want, tempo_grid, min_tempo, max_tempo):
                    dst_dur = piece_src_dur / tempo
                    piece_end = t_now + dst_dur
                    # Per-piece cost contributions that do not need the whole
                    # segment: speech rate now, onset error now.
                    c_add = 0.0
                    if lambda_speechrate:
                        c_add += lambda_speechrate * piece_src_dur * abs(math.log(tempo))
                    if lambda_pause:
                        c_add += lambda_pause * abs(t_now - onset_target)
                    if lambda_phoneme:
                        d, _ = _d_phoneme(
                            seg,
                            [Piece(Span(src_start, cuts[j]), t_now, tempo)],
                            frame_s=viseme_frame_s)
                        c_add += lambda_phoneme * d
                    if last:
                        # Close out the segment: the whole-segment duration term
                        # is only knowable here.
                        tot = cost + c_add
                        if lambda_duration:
                            placement = None  # recovered on reconstruction
                            tot += lambda_duration * abs(piece_end - end_target)
                        if lambda_pause:
                            tot += lambda_pause * sum(r.dur for r in runs[k + 1:])
                        tkey = (j, k + 1, _bucket(piece_end, sp.slot.start, drift_bucket_s))
                        prev = terminal.get(tkey)
                        if prev is None or tot < prev:
                            terminal[tkey] = tot
                            all_states[("T",) + tkey] = (
                                tot, piece_end, key, (tempo, 0.0, was_capped, i, j))
                        continue
                    # Pause candidates: the grid, plus the exact pause that puts
                    # the next piece on its onset target (same reason as the
                    # exact-fit tempo above).
                    fit_pause = nxt_target - piece_end
                    for pause in list(pause_grid_s) + [fit_pause]:
                        pause = float(pause)
                        if pause < 0 or pause > max(pause_grid_s) + 1e-9:
                            if abs(pause - fit_pause) > 1e-12 or pause < 0:
                                continue
                        t_next = piece_end + pause
                        nkey = (j, k + 1, _bucket(t_next, sp.slot.start, drift_bucket_s))
                        ncost = cost + c_add
                        prev = nxt.get(nkey)
                        n_transitions += 1
                        if prev is None or ncost < prev[0]:
                            nxt[nkey] = (ncost, t_next, key,
                                         (tempo, pause, was_capped, i, j))
        if inner_beam and len(nxt) > inner_beam:
            keep = sorted(nxt.items(), key=lambda kv: kv[1][0])[:inner_beam]
            nxt = dict(keep)
        all_states.update(nxt)
        frontier.update(nxt)

    if not terminal:
        return math.inf, [], [], False

    # Ties are not hypothetical: with only lambda_duration active and a clip
    # that UNDERSHOOTS its slot, |o - slot.start| + |end_target - o - dur| is
    # genuinely constant in o, so the slot-start and source-onset candidates
    # score identically and dict order alone decided the answer. Break ties on
    # the state key, whose drift bucket is monotone in the piece end time, so
    # the earliest (= slot-start-anchored) path wins. That is what makes
    # degenerate case 1 reproduce `plan_naive_slot` deterministically rather
    # than by luck.
    best_key = min(terminal, key=lambda k: (round(terminal[k], COST_TIE_NDIGITS), k))
    # Walk back.
    chain = []
    key = ("T",) + best_key
    while key is not None:
        cost, t_at, parent, choice = all_states[key]
        if choice is None:
            chain.append((t_at, None))
            break
        chain.append((t_at, choice))
        key = parent
    chain.reverse()

    pieces: List[Piece] = []
    inserted: List[Tuple[int, float]] = []
    capped_any = False
    # chain[0] is the start state (t = onset, choice None); each later entry
    # carries the choice that PRODUCED it, so the piece it describes starts at
    # the previous entry's time.
    for idx in range(1, len(chain)):
        t_prev = chain[idx - 1][0]
        tempo, pause, capped, i, j = chain[idx][1]
        capped_any = capped_any or capped
        pieces.append(Piece(src_span=Span(cuts[i], cuts[j]), dst_start=t_prev,
                            local_tempo=tempo, anchor="phrase"))
        if pause > 1e-6:
            inserted.append((len(pieces) - 1, pause))
    return terminal[best_key], pieces, inserted, capped_any


def plan_dp(
    merged_json: Path,
    *,
    lambda_duration: float = 1.0,
    lambda_pause: float = 1.0,
    lambda_speechrate: float = 1.0,
    lambda_phoneme: float = 0.0,
    tempo_grid: Sequence[float] = (0.90, 0.95, 1.00, 1.05, 1.10, 1.15),
    pause_grid_s: Sequence[float] = (0.0, 0.15, 0.30, 0.50, 0.80),
    allow_split: bool = True,
    max_pieces_per_segment: int = 8,
    drift_bucket_s: float = 0.05,
    drift_max_s: float = 3.0,
    out_json: Optional[Path] = None,
    # --- additive knobs; the defaults reproduce the behaviour described above ---
    min_tempo: Optional[float] = None,
    max_tempo: Optional[float] = None,
    safety_margin_s: float = 0.10,
    shift_forward: bool = True,
    pause_min_s: float = DEFAULT_PAUSE_MIN_S,
    max_cut_candidates: int = 10,
    inner_beam: int = 64,
    outer_beam: int = 24,
    viseme_frame_s: float = 0.02,
) -> TimingPlan:
    """Choose splits, per-piece tempo and pause insertions by dynamic programming.

    Minimises
        L1*D_duration + L2*D_pause + L3*D_speechrate + L4*D_phoneme
    (see `_COST_DEFINITIONS`) over the joint choice of split points, a tempo per
    piece and a pause after each piece.

    Two levels, because the two decisions have different scopes:
      inner  per segment, over (split, tempo, pause) GIVEN an entry cursor;
      outer  across segments, over the accumulated drift that entry cursor
             encodes.
    The outer state is drift quantised to `drift_bucket_s` and clipped to
    +/-`drift_max_s`, which is what makes the state space finite: a segment that
    has already run three seconds long cannot be rescued by the next one, so
    tracking drift past that bound buys nothing and costs a dimension.

    Degenerate cases, which are the real specification of the cost function:
      * lambda_pause = lambda_speechrate = lambda_phoneme = 0 reproduces
        `plan_naive_slot`'s placement -- D_duration's onset term is minimised by
        the shift-forward floor, and the exact-fit tempo candidate reproduces
        naive_slot's continuous tempo.
      * lambda_duration = 0, lambda_pause = 1 reproduces pause-aware onset
        anchoring -- the source-proportional cut candidates and the exact-fit
        pause candidate put that solution inside the search space, and D_pause's
        unmatched-run term makes one-piece-per-source-run the optimal count.

        This matches pause-aware SEMANTICS, which is a stronger claim than
        matching `plan_pause_aware`'s OUTPUT, and the two part company on an
        overrunning clip. `plan_pause_aware` is constrained to one tempo for
        the whole segment and to cut points proportional to the source run
        durations, so its chunk can run past the next onset, at which point it
        degrades to `prev_end + pause_min_keep_s` and sets `qc.pauses_floored`.
        The DP is free to re-cut and does not need to, so it lands on every
        onset exactly. Measured on `tests/fixtures/mini.mt.json`: pause_aware
        is 433 ms and 277 ms late on two segments where the DP is exact.
        `tests/test_timing_plan.py` asserts equality only on the segments
        pause_aware did not degrade, and pins the rest as a known gap.
    """
    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    lo = float(min(tempo_grid)) if min_tempo is None else float(min_tempo)
    hi = float(max(tempo_grid)) if max_tempo is None else float(max_tempo)
    params = {
        "lambda_duration": lambda_duration, "lambda_pause": lambda_pause,
        "lambda_speechrate": lambda_speechrate, "lambda_phoneme": lambda_phoneme,
        "tempo_grid": list(tempo_grid), "pause_grid_s": list(pause_grid_s),
        "allow_split": allow_split,
        "max_pieces_per_segment": max_pieces_per_segment,
        "drift_bucket_s": drift_bucket_s, "drift_max_s": drift_max_s,
        "min_tempo": lo, "max_tempo": hi, "safety_margin_s": safety_margin_s,
        "shift_forward": shift_forward, "max_cut_candidates": max_cut_candidates,
        "inner_beam": inner_beam, "outer_beam": outer_beam,
        "cost_definitions": _COST_DEFINITIONS,
    }
    plan = TimingPlan(producer="dp", params=params)

    segs = list(data.get("segments", []))
    prepared = []
    for seg in segs:
        sp = _base_segment_plan(seg, pause_min_s=pause_min_s)
        raw = raw_clip_duration(seg)
        cuts = (_cut_candidates(seg, sp, raw, allow_split=allow_split,
                                max_cut_candidates=max_cut_candidates)
                if raw and raw > 0 else [])
        prepared.append((seg, sp, raw, cuts))

    n_drift = int(round(2 * drift_max_s / drift_bucket_s)) + 1
    # outer: drift bucket -> (cumulative cost, cursor, back-reference)
    outer: Dict[int, Tuple[float, float, object]] = {0: (0.0, 0.0, None)}
    chosen: List[Dict[int, Tuple[int, object]]] = []
    timings: List[float] = []
    inner_calls = 0

    for s_idx, (seg, sp, raw, cuts) in enumerate(prepared):
        t0 = time.perf_counter()
        nxt: Dict[int, Tuple[float, float, object]] = {}
        step: Dict[int, Tuple[int, object]] = {}
        for db, (cum, cursor, _bk) in outer.items():
            if raw is None or raw <= 0 or len(cuts) < 2:
                # Degrade, do not raise: carry the cursor through untouched.
                res = (0.0, [], [], False)
            else:
                res = _inner_dp(
                    seg, sp, raw, cuts, cursor,
                    lambda_duration=lambda_duration, lambda_pause=lambda_pause,
                    lambda_speechrate=lambda_speechrate,
                    lambda_phoneme=lambda_phoneme,
                    tempo_grid=tempo_grid, pause_grid_s=pause_grid_s,
                    min_tempo=lo, max_tempo=hi,
                    max_pieces_per_segment=max_pieces_per_segment,
                    drift_bucket_s=drift_bucket_s,
                    safety_margin_s=safety_margin_s,
                    shift_forward=shift_forward, inner_beam=inner_beam,
                    viseme_frame_s=viseme_frame_s)
                inner_calls += 1
            cost, pieces, inserted, capped = res
            if cost == math.inf:
                cost, pieces, inserted, capped = 0.0, [], [], False
            new_cursor = max((p.dst_end for p in pieces), default=cursor)
            # Drift the NEXT segment inherits, relative to its own slot start.
            nxt_slot = (prepared[s_idx + 1][1].slot.start
                        if s_idx + 1 < len(prepared) else new_cursor)
            d = max(-drift_max_s, min(drift_max_s, new_cursor - nxt_slot))
            nb = _bucket(d, 0.0, drift_bucket_s)
            tot = cum + cost
            if nb not in nxt or tot < nxt[nb][0]:
                nxt[nb] = (tot, new_cursor, db)
                step[nb] = (db, (pieces, inserted, capped, cost))
        if outer_beam and len(nxt) > outer_beam:
            keep = sorted(nxt.items(), key=lambda kv: kv[1][0])[:outer_beam]
            nxt = dict(keep)
            step = {k: step[k] for k in nxt}
        if not nxt:
            nxt = {0: (0.0, 0.0, None)}
            step = {0: (0, ([], [], False, 0.0))}
        outer = nxt
        chosen.append(step)
        timings.append(time.perf_counter() - t0)

    # Backtrack the outer DP.
    final_b = min(outer, key=lambda b: outer[b][0]) if outer else 0
    picks: List[object] = [None] * len(prepared)
    b = final_b
    for s_idx in range(len(prepared) - 1, -1, -1):
        prev_b, payload = chosen[s_idx][b]
        picks[s_idx] = payload
        b = prev_b

    for (seg, sp, raw, cuts), payload in zip(prepared, picks):
        pieces, inserted, capped, cost = payload
        if raw is None or raw <= 0:
            sp.qc["no_audio"] = True
        elif not pieces:
            sp.qc["dp_no_solution"] = True
        sp.pieces = _nonoverlapping(pieces)
        sp.inserted_pauses = inserted
        if capped:
            sp.qc["tempo_cap_hit"] = True
        if sp.pieces:
            for p in sp.pieces:
                p.text = ""
            sp.pieces[0].text = _seg_text(seg)[:120]
        _, parts = segment_cost(
            seg, sp, sp.pieces, lambda_duration=lambda_duration,
            lambda_pause=lambda_pause, lambda_speechrate=lambda_speechrate,
            lambda_phoneme=lambda_phoneme, safety_margin_s=safety_margin_s,
            viseme_frame_s=viseme_frame_s)
        parts["dp_objective"] = round(cost, 5)
        sp.cost = parts
        plan.segments.append(sp)

    plan.summary = summarize(plan)
    plan.summary["dp"] = {
        "objective": round(outer[final_b][0], 5) if outer else None,
        "inner_calls": inner_calls,
        "drift_buckets": n_drift,
        "outer_state_bound": min(n_drift, outer_beam) if outer_beam else n_drift,
        "inner_state_bound": (max_pieces_per_segment
                              * (max_cut_candidates + 2) * (inner_beam or n_drift)),
        "wall_clock_total_s": round(sum(timings), 4),
        "wall_clock_per_segment_s": round(sum(timings) / max(1, len(timings)), 4),
        "wall_clock_max_segment_s": round(max(timings), 4) if timings else 0.0,
    }
    if out_json:
        plan.write(Path(out_json))
    log.info("timing[dp]: %d segments, %d pieces, objective=%.3f, %.2fs",
             plan.summary.get("segments", 0), plan.summary.get("pieces", 0),
             plan.summary["dp"]["objective"] or 0.0,
             plan.summary["dp"]["wall_clock_total_s"])
    return plan


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
# `timing.plan()` dispatches with getattr(timing, f"plan_{method}"), so attaching
# the two heavyweight producers here is all the registration the dispatcher
# needs. They live in this module so that a plain `plan("pause_aware", ...)`
# never pays to import them.
from . import timing as _timing  # noqa: E402

_timing.plan_word_anchored = plan_word_anchored
_timing.plan_dp = plan_dp
