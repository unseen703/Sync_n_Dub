"""Timing metrics over already-built dubbing artifacts.

Everything here is a pure function over spans / masks / manifest segments —
nothing reads a model or renders audio. The house rule applies throughout:
a metric that cannot be computed is reported in `not_measured[]` with a
reason, never silently dropped and never guessed.

Two design decisions worth stating up front, because they are what makes
these numbers rankable:

1. **Every offset report also carries `n_matched` / `n_invented` /
   `n_missed`.** A method that emits three well-placed segments out of
   forty otherwise scores as near-perfect. The match counts are the
   denominator the reader needs.
2. **Median, p95 and the SIGNED mean are reported separately.** The signed
   mean is the bias: "uniformly 180 ms late" and "jitters +-180 ms" have
   the same median |offset| and rank completely differently. p95 is where
   the audible failures live.
"""
from __future__ import annotations

import logging
import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

# The sibling module is owned by another agent; these modules must stay
# importable (and unit-testable) on their own, so the shared types are
# optional and a local stand-in is used when they are not there yet.
try:  # pragma: no cover - exercised only by import order
    from .reference import Word  # type: ignore
except ImportError:  # pragma: no cover
    @dataclass
    class Word:  # type: ignore[no-redef]
        """Fallback stand-in for `reference.Word` (text + span, seconds)."""
        text: str
        start: float
        end: float


__all__ = [
    "Span",
    "as_spans",
    "match_segments",
    "offset_stats",
    "speech_iou",
    "drift_curve",
    "pause_f1",
    "gap_stats",
    "stretch_stats",
    "akshara_count",
    "rate_stats",
    "masks_from_spans",
]


# ---------------------------------------------------------------------------
# Span normalisation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Span:
    """A half-open time interval [start, end) in seconds."""
    start: float
    end: float

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


# Accepted key spellings, in priority order. The corpus carries all three:
# mt.json segments use start/end, golden/index.json clips use t0/t1, and a
# few writers emit start_s/end_s.
_START_KEYS = ("start", "start_s", "t0", "begin", "placed", "onset")
_END_KEYS = ("end", "end_s", "t1", "stop", "offset")
_DUR_KEYS = ("duration", "duration_s", "dur", "dur_s", "raw_duration_s")


def _as_span(obj: Any) -> Optional[Span]:
    """Coerce one segment-ish object to a `Span`, or None if it has no times."""
    if obj is None:
        return None
    if isinstance(obj, Span):
        return obj
    if isinstance(obj, dict):
        start = next((float(obj[k]) for k in _START_KEYS
                      if obj.get(k) is not None), None)
        if start is None:
            return None
        end = next((float(obj[k]) for k in _END_KEYS
                    if obj.get(k) is not None), None)
        if end is None:
            dur = next((float(obj[k]) for k in _DUR_KEYS
                        if obj.get(k) is not None), None)
            if dur is None:
                return None
            end = start + dur
        return Span(start, end)
    if isinstance(obj, (tuple, list)):
        nums = [x for x in obj if isinstance(x, (int, float))
                and not isinstance(x, bool)]
        if len(nums) >= 2:
            # mix.py's plan rows are (id, placed, wav, raw_dur, drift): the
            # first two numbers are placed + duration, not start + end. A
            # pair is read as (start, end); anything longer as (start, dur).
            if len(obj) == 2 and len(nums) == 2:
                return Span(float(nums[0]), float(nums[1]))
            return Span(float(nums[0]), float(nums[0]) + float(nums[1]))
        return None
    start = getattr(obj, "start", None)
    if start is None:
        return None
    end = getattr(obj, "end", None)
    if end is None:
        dur = getattr(obj, "duration", None)
        if dur is None:
            return None
        end = float(start) + float(dur)
    return Span(float(start), float(end))


def as_spans(items: Iterable[Any]) -> List[Span]:
    """Coerce an iterable of segment-ish objects to `Span`s, dropping untimed ones."""
    out: List[Span] = []
    for it in items or []:
        s = _as_span(it)
        if s is not None:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _pct(values: Sequence[float], q: float) -> Optional[float]:
    if not len(values):
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _median(values: Sequence[float]) -> Optional[float]:
    if not len(values):
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _mean(values: Sequence[float]) -> Optional[float]:
    if not len(values):
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


# ---------------------------------------------------------------------------
# Shared matcher
# ---------------------------------------------------------------------------

def match_segments(
    hyp: Iterable[Any],
    ref: Iterable[Any],
    *,
    min_iou: float = 0.0,
) -> List[Tuple[Optional[int], Optional[int]]]:
    """Greedy 1-1 assignment of hypothesis to reference spans by maximum IoU.

    Text is deliberately ignored: this is a timing matcher, and bringing text
    in would let a method launder bad timing through good wording.

    A pair is a candidate only when the spans actually overlap (IoU > 0) AND
    IoU >= `min_iou`. `min_iou=0.0` therefore still means "must overlap" —
    without the strict-positive guard, disjoint segments on opposite ends of
    the clip would be matched to each other and the offsets would be noise.

    Returns `(hyp_idx, ref_idx)` pairs: matched pairs first (ordered by hyp
    index), then `(i, None)` for invented hypotheses, then `(None, j)` for
    missed references.
    """
    h = as_spans(hyp)
    r = as_spans(ref)

    cands: List[Tuple[float, float, int, int]] = []
    for i, a in enumerate(h):
        for j, b in enumerate(r):
            inter = min(a.end, b.end) - max(a.start, b.start)
            if inter <= 0:
                continue
            union = max(a.end, b.end) - min(a.start, b.start)
            if union <= 0:
                continue
            iou = inter / union
            if iou < min_iou:
                continue
            # Tie-break on |dstart| so equal-IoU candidates resolve the same
            # way on every run; the trailing indices make it fully total.
            cands.append((iou, -abs(a.start - b.start), i, j))

    cands.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))

    used_h: set = set()
    used_r: set = set()
    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    for _iou, _tb, i, j in cands:
        if i in used_h or j in used_r:
            continue
        used_h.add(i)
        used_r.add(j)
        pairs.append((i, j))

    pairs.sort(key=lambda p: (p[0] if p[0] is not None else -1))
    pairs += [(i, None) for i in range(len(h)) if i not in used_h]
    pairs += [(None, j) for j in range(len(r)) if j not in used_r]
    return pairs


# ---------------------------------------------------------------------------
# Offsets
# ---------------------------------------------------------------------------

def _offset_block(deltas_s: Sequence[float], n_ref: int) -> Dict[str, Any]:
    """Median |d|, p95 |d| and SIGNED mean, plus tolerance hit rates."""
    if not len(deltas_s):
        return {
            "n": 0, "median_abs_ms": None, "p95_abs_ms": None,
            "mean_signed_ms": None, "median_signed_ms": None,
            "max_abs_ms": None,
            "within_20ms": None, "within_100ms": None,
            "within_20ms_of_ref": None, "within_100ms_of_ref": None,
        }
    d = np.asarray(deltas_s, dtype=float) * 1000.0
    a = np.abs(d)
    n = int(d.size)
    w20 = int(np.count_nonzero(a <= 20.0))
    w100 = int(np.count_nonzero(a <= 100.0))
    return {
        "n": n,
        "median_abs_ms": float(np.median(a)),
        "p95_abs_ms": float(np.percentile(a, 95)),
        # The bias term. Separated from the |.| stats on purpose: a method
        # that is uniformly late has a large signed mean and small spread.
        "mean_signed_ms": float(np.mean(d)),
        "median_signed_ms": float(np.median(d)),
        "max_abs_ms": float(np.max(a)),
        # Fraction of MATCHED segments...
        "within_20ms": w20 / n,
        "within_100ms": w100 / n,
        # ...and the same count over ALL references, which is the honest
        # figure when a method drops segments rather than mistiming them.
        "within_20ms_of_ref": (w20 / n_ref) if n_ref else None,
        "within_100ms_of_ref": (w100 / n_ref) if n_ref else None,
    }


def offset_stats(
    hyp: Iterable[Any],
    ref: Iterable[Any],
    *,
    min_iou: float = 0.0,
) -> Dict[str, Any]:
    """Start/end offset distribution between matched hypothesis and reference spans.

    Sign convention: `d = hyp - ref`, so positive means the hypothesis is LATE.
    """
    h = as_spans(hyp)
    r = as_spans(ref)
    pairs = match_segments(h, r, min_iou=min_iou)

    dstart: List[float] = []
    dend: List[float] = []
    n_matched = 0
    n_invented = 0
    n_missed = 0
    for i, j in pairs:
        if i is not None and j is not None:
            n_matched += 1
            dstart.append(h[i].start - r[j].start)
            dend.append(h[i].end - r[j].end)
        elif i is not None:
            n_invented += 1
        else:
            n_missed += 1

    out: Dict[str, Any] = {
        "n_hyp": len(h),
        "n_ref": len(r),
        "n_matched": n_matched,
        "n_invented": n_invented,
        "n_missed": n_missed,
        "match_recall": (n_matched / len(r)) if r else None,
        "match_precision": (n_matched / len(h)) if h else None,
        "min_iou": min_iou,
        "sign_convention": "d = hyp - ref; positive = hypothesis is late",
        "dstart": _offset_block(dstart, len(r)),
        "dend": _offset_block(dend, len(r)),
        "not_measured": [],
    }
    if n_matched == 0:
        out["not_measured"].append({
            "metric": "offset_stats",
            "reason": f"no hyp/ref pair overlapped at min_iou={min_iou} "
                      f"({len(h)} hyp, {len(r)} ref)",
        })
    return out


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def masks_from_spans(
    spans: Iterable[Any],
    total_s: float,
    frame_s: float = 0.01,
) -> np.ndarray:
    """Rasterise spans into a boolean speech mask of `ceil(total_s/frame_s)` frames.

    Frame k covers [k*frame_s, (k+1)*frame_s). Spans are clamped to the
    canvas; anything past `total_s` is dropped rather than growing the array,
    so two masks built with the same `total_s` are always comparable.
    """
    if frame_s <= 0:
        raise ValueError("frame_s must be > 0")
    n = int(math.ceil(max(0.0, float(total_s)) / frame_s))
    mask = np.zeros(n, dtype=bool)
    for sp in as_spans(spans):
        a = int(math.floor(max(0.0, sp.start) / frame_s))
        b = int(math.ceil(min(float(total_s), sp.end) / frame_s))
        if b > a:
            mask[max(0, a):min(n, b)] = True
    return mask


def _align_masks(hyp_mask, ref_mask) -> Tuple[np.ndarray, np.ndarray, int]:
    """Zero-pad the shorter mask so the pair is comparable; report the padding."""
    a = np.asarray(hyp_mask).astype(bool).ravel()
    b = np.asarray(ref_mask).astype(bool).ravel()
    n = max(a.size, b.size)
    pad = abs(a.size - b.size)
    if a.size < n:
        a = np.concatenate([a, np.zeros(n - a.size, dtype=bool)])
    if b.size < n:
        b = np.concatenate([b, np.zeros(n - b.size, dtype=bool)])
    return a, b, pad


def speech_iou(hyp_mask, ref_mask, *, frame_s: float = 0.01) -> Dict[str, Any]:
    """Frame-level speech-activity IoU, with precision and recall alongside.

    IoU alone hides which way an error goes: a dub that talks over the whole
    clip and one that drops half the lines can land on the same IoU. Precision
    (hyp speech that is really speech) and recall (ref speech that got spoken)
    separate them.
    """
    a, b, pad = _align_masks(hyp_mask, ref_mask)
    tp = int(np.count_nonzero(a & b))
    fp = int(np.count_nonzero(a & ~b))
    fn = int(np.count_nonzero(~a & b))
    union = tp + fp + fn

    out: Dict[str, Any] = {
        "frame_s": frame_s,
        "n_frames": int(a.size),
        "padded_frames": pad,
        "tp_frames": tp, "fp_frames": fp, "fn_frames": fn,
        "tp_s": tp * frame_s, "fp_s": fp * frame_s, "fn_s": fn * frame_s,
        "hyp_speech_s": int(np.count_nonzero(a)) * frame_s,
        "ref_speech_s": int(np.count_nonzero(b)) * frame_s,
        "iou": (tp / union) if union else None,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "recall": (tp / (tp + fn)) if (tp + fn) else None,
        "not_measured": [],
    }
    if union == 0:
        out["not_measured"].append({
            "metric": "speech_iou",
            "reason": "both masks are entirely silent (union = 0 frames)",
        })
    if out["precision"] is None:
        out["not_measured"].append({
            "metric": "precision", "reason": "hypothesis mask has no speech"})
    if out["recall"] is None:
        out["not_measured"].append({
            "metric": "recall", "reason": "reference mask has no speech"})
    return out


def drift_curve(
    hyp_mask,
    ref_mask,
    *,
    frame_s: float = 0.01,
    sample_s: float = 1.0,
) -> Dict[str, Any]:
    """Cumulative speech-time difference D(t) = hyp_speech(<=t) - ref_speech(<=t).

    `max_abs_s` is the headline, not the final value. A `global_tempo` mix can
    land within +0.7% at the end of the clip while being seconds out of phase
    in the middle — the final number says the budget was met, `max|D|` says
    the viewer heard the dub drift away and come back.
    """
    a, b, pad = _align_masks(hyp_mask, ref_mask)
    if a.size == 0:
        return {
            "frame_s": frame_s, "sample_s": sample_s, "n_frames": 0,
            "times_s": [], "drift_s": [],
            "d25_s": None, "d50_s": None, "d75_s": None, "d100_s": None,
            "max_abs_s": None, "max_abs_at_s": None, "final_s": None,
            "not_measured": [{"metric": "drift_curve",
                              "reason": "empty masks (0 frames)"}],
        }

    cum = (np.cumsum(a.astype(np.float64)) - np.cumsum(b.astype(np.float64))) * frame_s
    total_s = a.size * frame_s

    step = max(1, int(round(sample_s / frame_s)))
    idx = np.arange(step - 1, a.size, step)
    if idx.size == 0 or idx[-1] != a.size - 1:
        idx = np.append(idx, a.size - 1)
    times = (idx + 1) * frame_s
    values = cum[idx]

    def at(frac: float) -> float:
        k = min(a.size - 1, max(0, int(round(frac * a.size)) - 1))
        return float(cum[k])

    k_max = int(np.argmax(np.abs(cum)))
    return {
        "frame_s": frame_s,
        "sample_s": sample_s,
        "n_frames": int(a.size),
        "padded_frames": pad,
        "duration_s": total_s,
        "times_s": [float(t) for t in times],
        "drift_s": [float(v) for v in values],
        "d25_s": at(0.25), "d50_s": at(0.50),
        "d75_s": at(0.75), "d100_s": at(1.00),
        "final_s": float(cum[-1]),
        "max_abs_s": float(np.abs(cum[k_max])),
        "max_abs_at_s": float((k_max + 1) * frame_s),
        "max_abs_signed_s": float(cum[k_max]),
        "not_measured": [],
    }


# ---------------------------------------------------------------------------
# Pauses
# ---------------------------------------------------------------------------

def pause_f1(
    hyp_pauses: Iterable[Any],
    ref_pauses: Iterable[Any],
    *,
    tol_s: float = 0.20,
) -> Dict[str, Any]:
    """Precision / recall / F1 over pauses, plus what was swallowed or invented.

    A pause matches when BOTH boundaries land within `tol_s`. Midpoint-only
    matching would call a 0.2 s breath and a 2 s beat the same pause, which is
    exactly the confusion that makes a dub feel rushed.

    `swallowed` = reference pauses with no match (the dub talked through them).
    `invented` = hypothesis pauses with no match (dead air the original has not).
    Both are reported as a count AND as seconds, because one swallowed 3 s beat
    matters more than six swallowed 0.2 s breaths.
    """
    h = as_spans(hyp_pauses)
    r = as_spans(ref_pauses)

    cands: List[Tuple[float, int, int]] = []
    for i, a in enumerate(h):
        for j, b in enumerate(r):
            err = max(abs(a.start - b.start), abs(a.end - b.end))
            if err <= tol_s:
                cands.append((err, i, j))
    cands.sort(key=lambda c: (c[0], c[1], c[2]))

    used_h: set = set()
    used_r: set = set()
    errs: List[float] = []
    for err, i, j in cands:
        if i in used_h or j in used_r:
            continue
        used_h.add(i)
        used_r.add(j)
        errs.append(err)

    matched = len(errs)
    inv = [h[i] for i in range(len(h)) if i not in used_h]
    swa = [r[j] for j in range(len(r)) if j not in used_r]
    prec = (matched / len(h)) if h else None
    rec = (matched / len(r)) if r else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (
        0.0 if (prec is not None and rec is not None) else None)

    out: Dict[str, Any] = {
        "tol_s": tol_s,
        "n_hyp": len(h), "n_ref": len(r),
        "n_matched": matched,
        "n_invented": len(inv), "n_missed": len(swa),
        "precision": prec, "recall": rec, "f1": f1,
        "swallowed_n": len(swa),
        "swallowed_s": float(sum(s.dur for s in swa)),
        "invented_n": len(inv),
        "invented_s": float(sum(s.dur for s in inv)),
        "boundary_err_median_ms": (_median(errs) or 0.0) * 1000.0 if errs else None,
        "boundary_err_p95_ms": (_pct(errs, 95) * 1000.0) if errs else None,
        "not_measured": [],
    }
    if not h and not r:
        out["not_measured"].append({
            "metric": "pause_f1", "reason": "no pauses on either side"})
    return out


# ---------------------------------------------------------------------------
# Gaps and stretch
# ---------------------------------------------------------------------------

def gap_stats(segments: Iterable[Any]) -> Dict[str, Any]:
    """Inter-segment gap distribution, sorted by start time.

    `pct_negative` is the interesting one: a negative gap is one dubbed clip
    running into the next, which the shift-forward mixer hides by pushing the
    later clip and the global_tempo mixer hides by squeezing everything.
    """
    sp = sorted(as_spans(segments), key=lambda s: (s.start, s.end))
    gaps = [sp[i + 1].start - sp[i].end for i in range(len(sp) - 1)]
    neg = [g for g in gaps if g < 0]

    out: Dict[str, Any] = {
        "n_segments": len(sp),
        "n_gaps": len(gaps),
        "median_s": _median(gaps),
        "p10_s": _pct(gaps, 10),
        "p90_s": _pct(gaps, 90),
        "mean_s": _mean(gaps),
        "min_s": float(min(gaps)) if gaps else None,
        "max_s": float(max(gaps)) if gaps else None,
        "n_negative": len(neg),
        "pct_negative": (100.0 * len(neg) / len(gaps)) if gaps else None,
        "total_negative_s": float(-sum(neg)) if neg else 0.0,
        "not_measured": [],
    }
    if not gaps:
        out["not_measured"].append({
            "metric": "gap_stats",
            "reason": f"need >= 2 timed segments, got {len(sp)}"})
    return out


_STRETCH_KEYS = ("stretch_factor", "factor", "tempo", "stretch")


def stretch_stats(
    segments: Iterable[Any],
    *,
    max_stretch: float = 0.10,
    saturation_tol: float = 1e-6,
    cap_low: Optional[float] = None,
    cap_high: Optional[float] = None,
) -> Dict[str, Any]:
    """Distribution of per-segment time-stretch factors against the cap budget.

    `n_cap_saturated` is the honest measure of budget pressure. A segment sitting
    exactly at the cap was not "stretched by that much" — it needed more, the cap
    truncated it, and the remainder leaked out as drift or as a clipped tail.
    Counting only `pct_exceeding_cap` reports zero violations for a pipeline that
    is violating the budget on every single segment.

    Two cap models, because the callers have two:

      * `max_stretch` alone — a SYMMETRIC budget, `1 +- max_stretch`. This is
        the historical default and nothing about it changes.
      * `cap_low` / `cap_high` — the ASYMMETRIC caps a producer actually ran
        with (`backend.timing` clamps to `min_tempo`/`max_tempo`, typically
        0.90 and 1.15). Passing the symmetric model where the real cap is
        asymmetric mismeasures saturation in both directions: against
        `max_stretch=0.10` a piece pinned at 1.15 is counted as a cap BREACH
        rather than as saturation, and a floor of 0.90 is looked for at 0.85
        and never found. When either is given it replaces the symmetric
        bound on that side, and `cap_model` in the result says so.
    """
    factors: List[float] = []
    n_missing = 0
    n_total = 0
    for seg in segments or []:
        n_total += 1
        f = None
        if isinstance(seg, dict):
            for k in _STRETCH_KEYS:
                if seg.get(k) is not None:
                    f = float(seg[k])
                    break
        elif isinstance(seg, (int, float)) and not isinstance(seg, bool):
            f = float(seg)
        else:
            for k in _STRETCH_KEYS:
                v = getattr(seg, k, None)
                if v is not None:
                    f = float(v)
                    break
        if f is None or not math.isfinite(f) or f <= 0:
            n_missing += 1
            continue
        factors.append(f)

    lo = float(cap_low) if cap_low is not None else 1.0 - max_stretch
    hi = float(cap_high) if cap_high is not None else 1.0 + max_stretch
    asymmetric = cap_low is not None or cap_high is not None

    out: Dict[str, Any] = {
        "n_segments": n_total,
        "n_with_factor": len(factors),
        "n_missing_factor": n_missing,
        "max_stretch": max_stretch,
        "cap_low": lo,
        "cap_high": hi,
        "cap_model": ("explicit caps [cap_low, cap_high]" if asymmetric
                      else f"symmetric 1 +- max_stretch ({max_stretch})"),
        "not_measured": [],
    }
    if not factors:
        out.update({
            "median_factor": None, "p95_factor": None,
            "median_abs_dev": None, "p95_abs_dev": None,
            "min_factor": None, "max_factor": None,
            "n_exceeding_cap": 0, "pct_exceeding_cap": None,
            "n_cap_saturated": 0, "pct_cap_saturated": None,
            "n_saturated_up": 0, "n_saturated_down": 0,
        })
        out["not_measured"].append({
            "metric": "stretch_stats",
            "reason": f"no segment carried a stretch factor "
                      f"(looked for {_STRETCH_KEYS} on {n_total} segments)"})
        return out

    arr = np.asarray(factors, dtype=float)
    dev = np.abs(arr - 1.0)
    up = hi
    down = lo
    sat_up = int(np.count_nonzero(np.abs(arr - up) <= saturation_tol))
    sat_down = int(np.count_nonzero(np.abs(arr - down) <= saturation_tol))
    # Strictly beyond the cap, with the saturation tolerance added so a
    # float-rounded 1.1000000000000001 counts as saturated, not as a breach.
    n_over = int(np.count_nonzero((arr > up + saturation_tol)
                                  | (arr < down - saturation_tol)))

    out.update({
        "median_factor": float(np.median(arr)),
        "p95_factor": float(np.percentile(arr, 95)),
        "mean_factor": float(np.mean(arr)),
        "median_abs_dev": float(np.median(dev)),
        "p95_abs_dev": float(np.percentile(dev, 95)),
        "min_factor": float(arr.min()),
        "max_factor": float(arr.max()),
        "n_exceeding_cap": n_over,
        "pct_exceeding_cap": 100.0 * n_over / arr.size,
        "n_cap_saturated": sat_up + sat_down,
        "pct_cap_saturated": 100.0 * (sat_up + sat_down) / arr.size,
        "n_saturated_up": sat_up,
        "n_saturated_down": sat_down,
    })
    if n_missing:
        out["not_measured"].append({
            "metric": "stretch_factor",
            "reason": f"{n_missing}/{n_total} segments carried no usable "
                      f"stretch factor and were excluded"})
    return out


# ---------------------------------------------------------------------------
# Devanagari syllable proxy
# ---------------------------------------------------------------------------

# Independent vowels: U+0904..U+0914, plus the vocalic RR/LL at U+0960/0961
# and the modern additions U+0972..U+0977.
_DEVA_IND_VOWELS = (
    set(range(0x0904, 0x0915))
    | {0x0960, 0x0961}
    | set(range(0x0972, 0x0978))
)
# Consonants: the main block U+0915..U+0939, the nukta forms U+0958..U+095F,
# and the extras U+0979..U+097F.
_DEVA_CONSONANTS = (
    set(range(0x0915, 0x093A))
    | set(range(0x0958, 0x0960))
    | set(range(0x0979, 0x0980))
)
_NUKTA = 0x093C
_VIRAMA = 0x094D


def akshara_count(text: str) -> int:
    """Devanagari syllable proxy: independent vowels + non-virama-bound consonants.

    An akshara is a consonant cluster plus its vowel. A consonant followed by
    the virama (halant, U+094D) carries no vowel — it is the first half of a
    conjunct — so it is not counted. A nukta (U+093C) may sit between the two,
    so it is skipped when looking ahead.

    This is a proxy, not a phonological parser: it ignores schwa deletion, so
    it over-counts word-final inherent vowels the way written Hindi does. It is
    stable and monotone in text length, which is all a rate metric needs.

    Non-Devanagari characters are ignored entirely, so mixed-script text
    returns only its Devanagari syllable count.
    """
    s = unicodedata.normalize("NFC", text or "")
    cps = [ord(c) for c in s]
    n = 0
    for i, cp in enumerate(cps):
        if cp in _DEVA_IND_VOWELS:
            n += 1
        elif cp in _DEVA_CONSONANTS:
            j = i + 1
            while j < len(cps) and cps[j] == _NUKTA:
                j += 1
            if j < len(cps) and cps[j] == _VIRAMA:
                continue  # dead consonant: part of a conjunct, no vowel
            n += 1
    return n


# ---------------------------------------------------------------------------
# Speaking rate
# ---------------------------------------------------------------------------

_TEXT_KEYS = ("final_text", "mt_text", "text")
_HYP_DUR_KEYS = ("tts_duration_s", "target_duration_s", "est_duration_s")


def _seg_text(seg: Any, text_key: Optional[str]) -> str:
    if isinstance(seg, dict):
        if text_key and seg.get(text_key):
            return str(seg[text_key])
        for k in _TEXT_KEYS:
            if seg.get(k):
                return str(seg[k])
        return ""
    return str(getattr(seg, text_key or "text", "") or "")


def _seg_duration(seg: Any) -> Optional[float]:
    """Spoken duration of a hypothesis segment, preferring measured over planned."""
    if isinstance(seg, dict):
        for k in _HYP_DUR_KEYS:
            v = seg.get(k)
            if v is not None and float(v) > 0:
                return float(v)
    sp = _as_span(seg)
    if sp is not None and sp.dur > 0:
        return sp.dur
    return None


def rate_stats(
    segments: Iterable[Any],
    ref_words: Optional[Iterable[Any]] = None,
    *,
    text_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Speaking rate of the dub, and its ratio to the reference rate.

    Two rates per side, because they answer different questions:
      * words/s and aksharas/s over the SPAN  — how packed the timeline is.
      * the same over VOICED time only        — articulation rate, i.e. how
        fast the speaker is actually talking.

    `ref_words` is a sequence of `reference.Word` (or anything with
    text/start/end). When it is absent the reference block is reported as
    not_measured rather than defaulted to 1.0.
    """
    wps: List[float] = []
    aps: List[float] = []
    n_no_dur = 0
    n_segs = 0
    for seg in segments or []:
        n_segs += 1
        txt = _seg_text(seg, text_key)
        dur = _seg_duration(seg)
        if not dur or dur <= 0:
            n_no_dur += 1
            continue
        nw = len([w for w in txt.split() if w.strip()])
        na = akshara_count(txt)
        wps.append(nw / dur)
        aps.append(na / dur)

    hyp = {
        "n_rated": len(wps),
        "median_wps": _median(wps), "p95_wps": _pct(wps, 95),
        "median_aps": _median(aps), "p95_aps": _pct(aps, 95),
    }

    out: Dict[str, Any] = {
        "n_segments": n_segs,
        "n_without_duration": n_no_dur,
        "hyp": hyp,
        "ref": None,
        "ratio_wps": None,
        "ratio_aps": None,
        "not_measured": [],
    }
    if not wps:
        out["not_measured"].append({
            "metric": "rate_stats.hyp",
            "reason": f"no segment had both text and a positive duration "
                      f"(of {n_segs})"})

    rw = list(ref_words or [])
    if not rw:
        out["not_measured"].append({
            "metric": "rate_stats.ref",
            "reason": "no ref_words supplied; reference rate and the hyp/ref "
                      "ratios cannot be computed"})
        return out

    spans = as_spans(rw)
    if not spans:
        out["not_measured"].append({
            "metric": "rate_stats.ref",
            "reason": f"{len(rw)} ref_words carried no usable start/end"})
        return out

    span_s = max(s.end for s in spans) - min(s.start for s in spans)
    voiced_s = float(sum(s.dur for s in spans))
    texts = [getattr(w, "text", None) or (w.get("text") if isinstance(w, dict) else "")
             for w in rw]
    n_ak = sum(akshara_count(t or "") for t in texts)

    ref = {
        "n_words": len(spans),
        "span_s": span_s,
        "voiced_s": voiced_s,
        "wps_over_span": (len(spans) / span_s) if span_s > 0 else None,
        "wps_over_voiced": (len(spans) / voiced_s) if voiced_s > 0 else None,
        "aps_over_span": (n_ak / span_s) if span_s > 0 else None,
        "aps_over_voiced": (n_ak / voiced_s) if voiced_s > 0 else None,
        "n_aksharas": n_ak,
    }
    out["ref"] = ref
    if hyp["median_wps"] is not None and ref["wps_over_span"]:
        out["ratio_wps"] = hyp["median_wps"] / ref["wps_over_span"]
    if hyp["median_aps"] is not None and ref["aps_over_span"]:
        out["ratio_aps"] = hyp["median_aps"] / ref["aps_over_span"]
    if n_ak == 0:
        out["not_measured"].append({
            "metric": "rate_stats.ref.aps",
            "reason": "reference words contain no Devanagari; akshara rate "
                      "is 0 by construction and the aps ratio is meaningless"})
    return out
