"""Pause absorption: spend a cue's duration deficit on its SILENCE first.

`timing.plan_naive_slot` fits a TTS clip to its slot with one tempo over the
whole clip, charging the same compression to every syllable and to the
pauses the engine put at commas and sentence breaks. Compression spent on
silence is inaudible; compression spent on speech is what makes a dub sound
rushed.

The rule here: shrink the silence to its floor before touching the speech.
On any line with sentence breaks that is often the whole deficit, and the
speech renders at 1.0x. Slot 5.0 s:

    clip 6.5 s = 5.5 speech + 1.0 silence   blob 1.30x  ->  speech 1.18x
    clip 6.5 s = 4.7 speech + 1.8 silence   blob 1.30x  ->  speech 1.04x
    clip 6.0 s = 4.5 speech + 1.5 silence   blob 1.20x  ->  speech 1.00x

It runs the other way too: a clip SHORTER than its slot grows its pauses
first and leaves the speech alone, so a short line is padded with silence
rather than drawled at `--speed-floor`.

Three safety properties:

  * Every floor is a floor, not a target. A sentence boundary collapsed to
    nothing welds two sentences together and reads as breathless even at
    1.0x speech. Floors are tiered by gap length (`PAUSE_TIERS`).
  * `max_absorb_frac` caps what any single gap may give up even when its
    floor would allow more. A dub with every pause floored is relentless.
  * A failed VAD returns None, never a guess. The caller falls back to the
    blob tempo, so a missing backend costs the optimisation, not the run.

Dependency-free at import: the VAD is reached from inside a function body,
so `backend.timing` can import this without pulling pyannote/torch into a
process that may only be reading a plan back.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: Floors by gap length, longest first: `(gap_at_least_s, floor_s)`.
#:
#: Tiered by duration, not punctuation, because the VAD path has no text to
#: read; a gap past 0.35 s inside a synthesised clip is almost always a
#: sentence boundary. The clause floor is 0.12 s to agree with
#: `timing.plan_pause_aware`'s `pause_min_keep_s`, the same quantity under
#: another name.
PAUSE_TIERS: Tuple[Tuple[float, float], ...] = (
    (0.35, 0.25),     # sentence boundary
    (0.18, 0.12),     # clause / comma
    (0.00, 0.08),     # breath
)

#: Never take more than this fraction of any single gap, floor or no floor.
DEFAULT_MAX_ABSORB_FRAC = 0.60

#: A gap may grow by at most this fraction of itself when the clip is
#: shorter than its slot. Unbounded growth puts a three-second hole mid-cue.
DEFAULT_MAX_EXPAND_FRAC = 1.50

#: Leading / trailing silence in a TTS clip is a synthesis artifact, not
#: prosody -- the slot already decides where the cue starts. Trimmed to this.
EDGE_FLOOR_S = 0.02

#: Energy, not pyannote: a TTS clip is clean audio whose pauses are
#: near-digital silence, the case energy VAD is reliable on. pyannote would
#: also hold VRAM in a process that goes on to run stage 9h, where MuseTalk
#: has ~100 MiB of headroom. `auto` and `pyannote` remain reachable by flag.
DEFAULT_VAD_BACKEND = "energy"

#: Gap-detection threshold handed to the VAD. Much smaller than the 0.25 s
#: `timing.DEFAULT_PAUSE_MIN_S` used for structural source pauses: here every
#: gap is budget, including the 80 ms ones. A 0.25 s threshold would merge
#: the cheap gaps into the speech runs and hide most of the pool.
DEFAULT_MIN_GAP_S = 0.06

#: A speech run shorter than this cannot be rendered (`backend.stretch`
#: windows are ~10 ms, `mix._MIN_STRETCH_INPUT_S` is 0.05), so it is merged
#: into its neighbour.
MIN_RUN_S = 0.10

#: One ffmpeg input and filter chain per piece, against `mix`'s 400-piece
#: ceiling. 8 runs per cue leaves room for ~50 cues; past that the smallest
#: gaps are merged away first.
DEFAULT_MAX_RUNS = 8


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------

@dataclass
class ClipPauses:
    """VAD of one TTS clip. All times are CLIP-RELATIVE seconds."""

    speech: List[Tuple[float, float]] = field(default_factory=list)
    gaps: List[float] = field(default_factory=list)      # len == len(speech) - 1
    lead_s: float = 0.0
    tail_s: float = 0.0
    duration_s: float = 0.0
    backend: str = ""
    merged_runs: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.speech)

    @property
    def speech_s(self) -> float:
        return sum(b - a for a, b in self.speech)

    @property
    def pause_s(self) -> float:
        return sum(self.gaps)


def pause_floor(gap_s: float, tiers: Sequence[Tuple[float, float]] = PAUSE_TIERS) -> float:
    """The smallest this gap is allowed to become.

    Clamped to the gap itself, so a 0.05 s breath is not handed a 0.08 s
    floor it already sits under -- which would make it grow in order to be
    compressed.
    """
    for at_least, floor in tiers:
        if gap_s >= at_least:
            return min(floor, gap_s)
    return min(tiers[-1][1], gap_s)


def _merge_runs(
    speech: List[Tuple[float, float]],
    gaps: List[float],
    *,
    max_runs: int,
    min_run_s: float,
) -> Tuple[List[Tuple[float, float]], List[float], int]:
    """Merge across the SMALLEST gaps until the run count and length fit.

    Smallest-first, not left-to-right: the cheapest gap to lose is the one
    barely there, and merging a 0.4 s sentence boundary to save a filter
    instance throws away the structure the tiers protect. A merged run spans
    first start to last end, so the swallowed gap becomes speech.
    """
    speech = list(speech)
    gaps = list(gaps)
    merged = 0
    while speech and (
        len(speech) > max_runs
        or (len(speech) > 1 and min(b - a for a, b in speech) < min_run_s)
    ):
        if not gaps:
            break
        k = min(range(len(gaps)), key=lambda j: gaps[j])
        speech[k] = (speech[k][0], speech[k + 1][1])
        del speech[k + 1]
        del gaps[k]
        merged += 1
    return speech, gaps, merged


def detect_pauses(
    wav: Path,
    *,
    duration_s: Optional[float] = None,
    backend: str = DEFAULT_VAD_BACKEND,
    min_gap_s: float = DEFAULT_MIN_GAP_S,
    min_run_s: float = MIN_RUN_S,
    max_runs: int = DEFAULT_MAX_RUNS,
) -> ClipPauses:
    """VAD one TTS clip into speech runs and the gaps between them.

    Gaps come from consecutive speech spans, not `VadResult.pauses`, which
    lists only gaps at or above its own `min_pause_s` -- the ones under it
    are a real part of the pool.

    Never raises: an unusable answer has `ok == False` and an `error`, and
    the caller keeps its blob tempo.
    """
    out = ClipPauses(duration_s=float(duration_s or 0.0), backend=backend)
    try:
        from .eval.reference import vad as _vad
    except Exception as e:                                   # noqa: BLE001
        out.error = f"backend.eval.reference unavailable: {type(e).__name__}: {e}"
        return out

    try:
        res = _vad(Path(wav), min_pause_s=min_gap_s,
                   min_speech_s=min_run_s, backend=backend)
    except Exception as e:                                   # noqa: BLE001
        out.error = f"vad({Path(wav).name}) raised {type(e).__name__}: {e}"
        return out

    out.backend = getattr(res, "backend", backend) or backend
    if getattr(res, "error", None) and not getattr(res, "speech", None):
        out.error = str(res.error)
        return out

    dur = float(duration_s or getattr(res, "duration_s", 0.0) or 0.0)
    spans = [(max(0.0, float(a)), float(b)) for a, b in (res.speech or [])]
    spans = [(a, min(b, dur) if dur > 0 else b) for a, b in spans if b > a]
    if not spans:
        out.error = "VAD found no speech in this clip"
        return out
    spans.sort()
    out.duration_s = dur or spans[-1][1]

    gaps = [max(0.0, spans[i + 1][0] - spans[i][1]) for i in range(len(spans) - 1)]
    spans, gaps, merged = _merge_runs(spans, gaps,
                                      max_runs=max_runs, min_run_s=min_run_s)
    out.speech, out.gaps, out.merged_runs = spans, gaps, merged
    out.lead_s = max(0.0, spans[0][0])
    out.tail_s = max(0.0, out.duration_s - spans[-1][1]) if out.duration_s else 0.0
    return out


# ---------------------------------------------------------------------------
# allocation
# ---------------------------------------------------------------------------

@dataclass
class Allocation:
    """How one cue's deficit was split between its silence and its speech."""

    speech_tempo: float = 1.0
    kept_gaps: List[float] = field(default_factory=list)
    mode: str = "fit"              # "fit" | "absorb" | "expand"
    absorbed_s: float = 0.0        # silence given up (compression)
    added_s: float = 0.0           # silence gained (expansion)
    absorbed_frac: float = 0.0     # of the original pause pool
    speech_capped: bool = False
    unfilled_s: float = 0.0        # slot time neither gaps nor speech could fill


def allocate(
    speech_s: float,
    gaps: Sequence[float],
    target_s: float,
    *,
    max_speech_tempo: float,
    min_speech_tempo: float,
    max_absorb_frac: float = DEFAULT_MAX_ABSORB_FRAC,
    max_expand_frac: float = DEFAULT_MAX_EXPAND_FRAC,
    tiers: Sequence[Tuple[float, float]] = PAUSE_TIERS,
) -> Allocation:
    """Split a duration deficit between silence and speech, silence first.

    `speech_tempo == 1.0` on the way back means the gaps covered the whole
    deficit and no syllable was touched -- the common case on any line with
    sentence breaks in it, and the thing to assert in a test, since a
    refactor that re-introduces blob tempo still produces a plausible plan.

    Gaps give up time in proportion to what each can spare -- its distance to
    its own floor, capped by `max_absorb_frac` -- so every gap ends the same
    fraction of the way to its floor. Draining by LENGTH instead pushes the
    short gaps through their floors while the long ones still have room.
    """
    gaps = [max(0.0, float(g)) for g in gaps]
    pause_s = sum(gaps)
    need = (speech_s + pause_s) - target_s

    if abs(need) <= 1e-4:
        return Allocation(speech_tempo=1.0, kept_gaps=list(gaps), mode="fit")

    # ---- compress: the clip is longer than its slot -----------------------
    if need > 0:
        spare = [max(0.0, min(g - pause_floor(g, tiers), g * max_absorb_frac))
                 for g in gaps]
        pool = sum(spare)
        give = min(need, pool)
        if pool > 1e-9:
            kept = [g - give * (s / pool) for g, s in zip(gaps, spare)]
        else:
            kept = list(gaps)
        remaining = need - give

        alloc = Allocation(
            kept_gaps=kept, mode="absorb",
            absorbed_s=round(give, 6),
            absorbed_frac=round(give / pause_s, 6) if pause_s > 1e-9 else 0.0,
        )
        if remaining <= 1e-6:
            alloc.speech_tempo = 1.0            # silence did all the work
            return alloc

        speech_out = speech_s - remaining
        if speech_out <= 1e-6:
            alloc.speech_tempo = max_speech_tempo
            alloc.speech_capped = True
            return alloc
        t = speech_s / speech_out
        alloc.speech_tempo = min(t, max_speech_tempo)
        alloc.speech_capped = t > max_speech_tempo + 1e-9
        return alloc

    # ---- expand: the clip is shorter than its slot ------------------------
    #
    # Pauses grow BEFORE speech slows, so a short line is padded with silence
    # rather than drawled at 0.75x.
    want = -need
    room = [g * max_expand_frac for g in gaps]
    pool = sum(room)
    take = min(want, pool)
    if pool > 1e-9:
        kept = [g + take * (r / pool) for g, r in zip(gaps, room)]
    else:
        kept = list(gaps)
    remaining = want - take

    alloc = Allocation(kept_gaps=kept, mode="expand", added_s=round(take, 6))
    if remaining <= 1e-6:
        alloc.speech_tempo = 1.0
        return alloc

    speech_out = speech_s + remaining
    t = speech_s / speech_out if speech_out > 1e-6 else 1.0
    if t < min_speech_tempo:
        alloc.speech_tempo = min_speech_tempo
        alloc.speech_capped = True
        # What neither the gaps nor the floor could absorb stays a real
        # pause at the end of the slot, reported as `plan_naive_slot`'s
        # `unfilled_s` is.
        alloc.unfilled_s = round(remaining - (speech_s / min_speech_tempo - speech_s), 6)
        if alloc.unfilled_s < 0:
            alloc.unfilled_s = 0.0
    else:
        alloc.speech_tempo = t
    return alloc


# ---------------------------------------------------------------------------
# one cue, end to end
# ---------------------------------------------------------------------------

@dataclass
class SegmentAbsorb:
    """Pieces for one cue, as plain tuples plus the qc to record.

    Tuples rather than `timing.Piece`: `backend.timing` imports this module,
    so returning its dataclass would make the import circular. The caller
    maps the four floats onto a Piece.
    """

    pieces: List[Tuple[float, float, float, float]] = field(default_factory=list)
    #: (src_start, src_end, dst_start, local_tempo) -- clip-relative src,
    #: absolute output-timeline dst, matching `Piece`'s convention.
    qc: Dict[str, object] = field(default_factory=dict)


def plan_segment(
    wav: Path,
    *,
    raw_s: float,
    target_s: float,
    placed_s: float,
    max_speech_tempo: float,
    min_speech_tempo: float,
    max_absorb_frac: float = DEFAULT_MAX_ABSORB_FRAC,
    max_expand_frac: float = DEFAULT_MAX_EXPAND_FRAC,
    vad_backend: str = DEFAULT_VAD_BACKEND,
    max_runs: int = DEFAULT_MAX_RUNS,
    min_run_s: float = MIN_RUN_S,
    min_gap_s: float = DEFAULT_MIN_GAP_S,
    trim_edges: bool = True,
) -> Optional[SegmentAbsorb]:
    """VAD one clip, allocate its deficit, and lay the pieces on the timeline.

    Returns None when the clip cannot be used this way -- no VAD backend, no
    speech found, or a single run with no gaps to spend. The caller falls
    back to the blob tempo, so this is never the difference between a dub and
    no dub.

    A single run with no gaps is a refusal, not a one-piece plan: with no
    silence to absorb the answer equals the blob tempo anyway.
    """
    cp = detect_pauses(wav, duration_s=raw_s, backend=vad_backend,
                       min_gap_s=min_gap_s, min_run_s=min_run_s,
                       max_runs=max_runs)
    if not cp.ok:
        log.debug("pause_absorb: %s unusable (%s)", Path(wav).name, cp.error)
        return None
    if len(cp.speech) < 2 and not (trim_edges and (cp.lead_s + cp.tail_s) > 0.05):
        return None

    # Edge silence is recovered outright: the slot already decides where the
    # cue starts, so leaving it in pays speech compression for the engine's
    # run-up.
    #
    # The kept edges come OUT OF THE BUDGET before allocation. Handing
    # `allocate` the full target and prepending 20 ms afterwards overshoots
    # the slot on every cue, which accumulates into visible drift.
    spans = list(cp.speech)
    lead_keep = min(cp.lead_s, EDGE_FLOOR_S) if trim_edges else cp.lead_s
    tail_keep = min(cp.tail_s, EDGE_FLOOR_S) if trim_edges else cp.tail_s
    lead_trim = max(0.0, cp.lead_s - lead_keep)
    tail_trim = max(0.0, cp.tail_s - tail_keep)

    speech_s = sum(b - a for a, b in spans)
    alloc = allocate(
        speech_s, cp.gaps, max(0.01, target_s - lead_keep - tail_keep),
        max_speech_tempo=max_speech_tempo, min_speech_tempo=min_speech_tempo,
        max_absorb_frac=max_absorb_frac, max_expand_frac=max_expand_frac,
    )

    tempo = max(1e-6, alloc.speech_tempo)
    pieces: List[Tuple[float, float, float, float]] = []
    cursor = placed_s + lead_keep
    for i, (a, b) in enumerate(spans):
        pieces.append((a, b, cursor, tempo))
        cursor += (b - a) / tempo
        if i < len(alloc.kept_gaps):
            cursor += max(0.0, alloc.kept_gaps[i])

    qc: Dict[str, object] = {
        "absorb": alloc.mode,
        "speech_tempo": round(alloc.speech_tempo, 4),
        "vad_backend": cp.backend,
        "runs": len(spans),
        "pause_s_before": round(cp.pause_s, 4),
        "pause_s_after": round(sum(alloc.kept_gaps), 4),
    }
    if alloc.absorbed_s > 1e-4:
        qc["pause_absorbed_s"] = round(alloc.absorbed_s, 4)
        qc["pause_absorbed_frac"] = round(alloc.absorbed_frac, 4)
    if alloc.added_s > 1e-4:
        qc["pause_added_s"] = round(alloc.added_s, 4)
    if lead_trim > 1e-4 or tail_trim > 1e-4:
        qc["clip_edge_trim_s"] = round(lead_trim + tail_trim, 4)
    if cp.merged_runs:
        qc["merged_runs"] = cp.merged_runs
    if alloc.speech_capped:
        qc["tempo_cap_hit"] = True
    if alloc.unfilled_s > 1e-3:
        qc["unfilled_s"] = round(alloc.unfilled_s, 4)
    # The honest headline: did the silence carry it, or did the speech?
    qc["speech_untouched"] = abs(alloc.speech_tempo - 1.0) < 1e-6

    return SegmentAbsorb(pieces=pieces, qc=qc)


__all__ = [
    "Allocation",
    "ClipPauses",
    "DEFAULT_MAX_ABSORB_FRAC",
    "DEFAULT_MAX_EXPAND_FRAC",
    "DEFAULT_MAX_RUNS",
    "DEFAULT_VAD_BACKEND",
    "PAUSE_TIERS",
    "SegmentAbsorb",
    "allocate",
    "detect_pauses",
    "pause_floor",
    "plan_segment",
]
