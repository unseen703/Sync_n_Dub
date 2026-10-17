"""Run one scored evaluation of one dubbed variant, and sweep over many.

This module owns the *runner*: input validation, the result schema, the
reconstruction of where mix.py actually placed each dubbed clip, and the
dispatch into the metric modules. It owns no metric formulas — those live in
`backend.eval.{timing,audio,text}` and are imported lazily, inside the group
branch that needs them, for two reasons:

  1. `--groups timing,pauses` must stay cheap. `backend.eval.audio` pulls in
     torch and soundfile; importing it at module scope would make the
     placement-only path pay for a GPU stack it never touches.
  2. The metric modules are written by other hands and may not exist on disk
     yet. A missing one has to degrade into a `not_measured` entry, not an
     ImportError at startup that takes the whole sweep down.

House rule enforced throughout: a number that could not be computed is
recorded in `errors[]` or `not_measured[]` with a reason. It is never
silently omitted and never guessed.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

GROUPS = ("timing", "pauses", "rate", "audio", "intel", "mt")
CHEAP = ("timing", "pauses", "rate")     # placement-only: no audio decode, no torch

# Groups that need a decoded waveform. Gating on this set is what makes
# `--groups timing,pauses` genuinely cheap rather than cheap-looking.
AUDIO_GROUPS = frozenset({"audio", "intel"})

# Which VariantInputs field each group cannot run without.
_GROUP_REQUIRES: Dict[str, Tuple[str, ...]] = {
    "timing": ("mt_json",),
    "pauses": ("mt_json",),
    "rate": ("mt_json",),
    "audio": ("mt_json", "mix_wav"),
    "intel": ("mt_json", "mix_wav"),
    "mt": ("mt_json",),
}

_INPUT_FIELDS = ("mt_json", "mix_wav", "mix_mp4", "tts_only_wav")

DEFAULT_GOLDEN_ROOT = Path("data/golden")
DEFAULT_OUT_DIR = Path("data/eval")

# mix.py's own defaults, used only when a manifest predates the keys being
# written into `extra.mix`. Every replay records which of the two it used.
_MIX_DEFAULT_STRATEGY = "global_tempo"
_MIX_DEFAULT_TEMPO_CAP = 1.5
_MIX_DEFAULT_CLIP_SOURCE = "auto"

# ---------------------------------------------------------------------------
# Measured calibration of this corpus. Established once, off-line, against
# data/golden; carried into every result JSON so a reader never has to guess
# whether a number is good. These are NOT tuning knobs and nothing in this
# module may re-derive them.
# ---------------------------------------------------------------------------

# The SOURCE-side (English) word-level floor: median disagreement between
# whisperx forced-align and the `en-orig` caption word stream. Measured by
# refbuild on 12 of 18 golden clips. It is recorded for provenance and it does
# NOT bound the ranking -- see below.
GT_SOURCE_FLOOR_MS = 55.0
GT_SOURCE_FLOOR_RANGE_MS = (43.0, 79.0)

# The REF_DUB (Hindi) side -- the one every timing metric here is scored
# against -- has NO word-level floor on any clip. No video in the corpus
# carries an `hi-orig` auto-caption, so there is exactly one Hindi word stream
# (`whisperx forced-align (hi)`) and nothing independent to check it against;
# `meta.agreement_median_ms.ref_dub` says so verbatim on all 18 clips.
#
# The measurable substitute is cue-level: human-authored `.hi.vtt` cue onsets
# against the CTC onset whisperx gave that cue's first word. The two share
# their text but not their timing, so the disagreement is a real timing
# cross-check -- at CUE resolution, which is coarser than word resolution.
# Measured by scripts/hi_cue_floor.py over the 9 clips where >=90% of cues
# matched a whisperx token: median 185 ms, range 79-338 ms.
#
# Therefore 185 ms is an UPPER BOUND on the unknown Hindi word-level floor,
# not an estimate of it. No timing claim below it can be distinguished from
# reference noise with the data available, so differences smaller than it are
# reported as ties rather than ranks.
GT_NOISE_FLOOR_MS = 185.0
GT_NOISE_FLOOR_RANGE_MS = (79.0, 338.0)
GT_NOISE_FLOOR_BASIS = (
    "cue-level: manual .hi.vtt cue onset vs whisperx CTC onset, 9 clips "
    "(scripts/hi_cue_floor.py); word-level unmeasurable -- no hi-orig "
    "caption exists in this corpus, so the Hindi reference has only one "
    "word stream. This is an upper bound, not an estimate."
)

# The no-op baseline: the UNTOUCHED English source scored against the human
# Hindi dub. speech IoU already sits at 0.872 before anything is dubbed, so
# its headroom (0.128) is a third of its own clip-to-clip spread -- it is a
# GUARD against a catastrophically mistimed arm, never a ranker. pause F1 is
# the discriminator: the no-op scores exactly zero on it.
NOOP_SPEECH_IOU_MEDIAN = 0.872
NOOP_PAUSE_F1_AT_200MS_MEDIAN = 0.000

# The golden VAD's operating point (backend/eval/reference.vad defaults).
# Planned hypothesis spans are merged at the SAME min_pause_s so that a
# "pause" means the same thing on both sides of the comparison.
_GOLDEN_MIN_PAUSE_S = 0.25
_GOLDEN_MIN_SPEECH_S = 0.20

# Frame size for every rasterised mask in this module.
_FRAME_S = 0.01

# Golden artifacts written by `python -m backend.eval refbuild`.
_GOLDEN_REF_VAD = "ref_dub.vad.json"
_GOLDEN_REF_WORDS = "ref_dub.words.json"
_GOLDEN_SRC_VAD = "source.vad.json"
_GOLDEN_SRC_WORDS = "source.words.json"
_GOLDEN_META = "meta.json"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class VariantInputs:
    """The artifacts one variant produced. Only `mt_json` is ever mandatory."""
    mt_json: Path
    mix_wav: Optional[Path] = None
    mix_mp4: Optional[Path] = None
    tts_only_wav: Optional[Path] = None


@dataclass
class VariantSpec:
    """One cell of the sweep: a variant of a clip, plus where its files are."""
    clip_id: str
    variant_id: str
    inputs: VariantInputs


def _sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def validate_inputs(
    inputs: VariantInputs,
    *,
    groups: Sequence[str] = CHEAP,
    record: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Check every supplied artifact and return one reason string per problem.

    Fills `record` (if given) with the schema's `inputs` block: path, sha256,
    byte count and an `ok` flag per field. A file that cannot be opened,
    hashed or (for audio groups) decoded yields a reason — never an exception.
    `data/mix/2vxQaT0Fy-c_2min_720p.dubbed.wav` is a dangling symlink on this
    host, which is exactly the case this has to survive.
    """
    reasons: List[str] = []
    record = record if record is not None else {}

    needed: set[str] = set()
    for g in groups:
        needed.update(_GROUP_REQUIRES.get(g, ()))
    want_audio = bool(AUDIO_GROUPS.intersection(groups))

    for fieldname in _INPUT_FIELDS:
        raw = getattr(inputs, fieldname, None)
        rec: Dict[str, Any] = {"path": None, "sha256": None, "bytes": None,
                               "ok": False, "reason": None}
        record[fieldname] = rec

        if raw is None:
            rec["reason"] = "not supplied"
            if fieldname in needed:
                reasons.append(f"{fieldname}: not supplied")
            continue

        path = Path(raw)
        rec["path"] = str(path)

        try:
            # stat() follows symlinks, so a dangling link fails here rather
            # than three stages later inside a metric function.
            st = path.stat()
        except OSError as e:
            rec["reason"] = f"unreadable: {type(e).__name__}: {e}"
            if fieldname in needed:
                reasons.append(f"{fieldname}: {rec['reason']}")
            continue
        rec["bytes"] = st.st_size

        if st.st_size == 0:
            rec["reason"] = "zero bytes on disk"
            if fieldname in needed:
                reasons.append(f"{fieldname}: {rec['reason']}")
            continue

        try:
            rec["sha256"] = _sha256(path)
        except OSError as e:
            rec["reason"] = f"could not hash: {type(e).__name__}: {e}"
            if fieldname in needed:
                reasons.append(f"{fieldname}: {rec['reason']}")
            continue

        if fieldname == "mt_json":
            try:
                json.loads(path.read_text())
            except Exception as e:
                rec["reason"] = f"not valid JSON: {type(e).__name__}: {e}"
                reasons.append(f"{fieldname}: {rec['reason']}")
                continue
        elif want_audio and fieldname in ("mix_wav", "tts_only_wav"):
            # Header read only, and only when an audio group was asked for.
            probe = _probe_audio(path)
            if probe.get("error"):
                rec["reason"] = probe["error"]
                if fieldname in needed:
                    reasons.append(f"{fieldname}: {probe['error']}")
                continue
            rec.update({k: v for k, v in probe.items() if k != "error"})

        rec["ok"] = True

    return reasons


def _probe_audio(path: Path) -> Dict[str, Any]:
    """soundfile header probe. Any failure comes back as `{"error": ...}`."""
    try:
        import soundfile as sf       # lazy: not on the cheap path
    except Exception as e:
        return {"error": f"soundfile unavailable: {type(e).__name__}: {e}"}
    try:
        info = sf.info(str(path))
    except Exception as e:
        # soundfile.LibsndfileError among others. Never let it escape.
        return {"error": f"unreadable audio: {type(e).__name__}: {e}"}
    return {"duration_s": float(info.frames) / float(info.samplerate or 1),
            "sample_rate": int(info.samplerate), "channels": int(info.channels)}


# ---------------------------------------------------------------------------
# Replaying mix.py's placement
# ---------------------------------------------------------------------------

@dataclass
class PlacedSegment:
    seg_id: str
    src_start: float
    src_end: float
    placed_start: float
    placed_end: float
    shift_s: float            # placed_start - src_start, on the output timeline
    clip_duration_s: float
    stretch_factor: Optional[float]
    text: str


@dataclass
class PlacedPiece:
    """One `backend.timing.Piece` after rendering, on the output timeline."""
    seg_id: str
    index: int
    dst_start: float
    dst_end: float
    local_tempo: float
    anchor: str
    src_dur: float            # seconds of RAW clip consumed (pre-stretch)
    text: str

    @property
    def dst_dur_s(self) -> float:
        return max(0.0, self.dst_end - self.dst_start)


@dataclass
class MixPlan:
    segments: List[PlacedSegment]
    strategy: str
    strategy_source: str      # "manifest" | "assumed_mix_default"
    clip_source: str
    global_tempo: float
    tempo_source: str
    fit_duration_s: float
    source_duration_s: float
    final_duration_s: float
    skipped: List[Dict[str, str]]
    # --- timing_plan replay (empty for the other strategies) --------------
    pieces: List[PlacedPiece] = field(default_factory=list)
    placement_source: str = "shift_forward_cursor"
    plan_path: Optional[str] = None
    plan_producer: Optional[str] = None
    plan_summary: Dict[str, Any] = field(default_factory=dict)
    plan_params: Dict[str, Any] = field(default_factory=dict)
    plan_reasons: List[str] = field(default_factory=list)
    mix_tempo_cap: Optional[float] = None


def _seg_clip_duration(seg: dict, clip_source: str) -> Optional[float]:
    """Length of the clip mix.py would have placed for this segment.

    Read off the manifest rather than the WAV header, because the cheap path
    is not allowed to touch audio. `tts_duration_s` is the post-stretch
    length (backend/tts.py writes it that way), so the raw clip is recovered
    by dividing out the recorded stretch factor.
    """
    dur = seg.get("tts_duration_s")
    if dur is None:
        return None
    try:
        dur = float(dur)
    except (TypeError, ValueError):
        return None
    if dur <= 0:
        return None
    if clip_source == "raw":
        sf_ = seg.get("stretch_factor")
        try:
            sf_ = float(sf_)
        except (TypeError, ValueError):
            sf_ = None
        if sf_ and sf_ > 0:
            dur = dur / sf_
    return dur


def _resolve_clip_field(seg: dict, clip_source: str) -> Optional[str]:
    """Mirror of backend.mix._resolve_clip, on fields only (no disk access)."""
    if clip_source == "auto":
        if seg.get("vc_audio"):
            return "vc_audio"
        return "tts_audio" if seg.get("tts_audio") else None
    key = {"raw": "tts_raw_audio", "stretched": "tts_audio", "vc": "vc_audio"}.get(clip_source)
    return key if key and seg.get(key) else None


def _load_timing_plan(
    data: dict,
    mix: dict,
    *,
    mt_path: Optional[Path],
) -> Tuple[Optional[dict], Optional[str], List[str]]:
    """Find the TimingPlan that `strategy=timing_plan` actually rendered.

    Three places, in the order mix.py itself would have found it:
      1. `extra["timing_plan"]`, written by `timing.apply_plan_to_merged`;
      2. `extra.mix["timing_plan_source"]`, the path mix.py recorded;
      3. `<manifest dir>/timing/<producer>.plan.json`, the layout
         backend/experiment.py writes.

    Returns `(plan_dict, path_or_None, reasons)`. A plan that cannot be found
    is a measurement gap, never an exception.
    """
    reasons: List[str] = []

    embedded = (data.get("extra") or {}).get("timing_plan")
    if isinstance(embedded, dict) and embedded.get("segments"):
        return embedded, (str(mt_path) + ":extra.timing_plan") if mt_path else "extra.timing_plan", reasons

    cands: List[Path] = []
    src = mix.get("timing_plan_source")
    if src:
        cands.append(Path(str(src).split(":extra.")[0]))
    producer = mix.get("timing_producer")
    if mt_path is not None and producer:
        cands.append(Path(mt_path).parent / "timing" / f"{producer}.plan.json")
    if mt_path is not None:
        cands.extend(sorted((Path(mt_path).parent / "timing").glob("*.plan.json")))

    seen: set = set()
    for cand in cands:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if not cand.exists():
            continue
        try:
            obj = json.loads(cand.read_text())
        except Exception as e:
            reasons.append(f"timing plan {cand} unreadable: {type(e).__name__}: {e}")
            continue
        if isinstance(obj, dict) and obj.get("segments"):
            return obj, str(cand), reasons
        reasons.append(f"timing plan {cand} carried no segments[]")

    reasons.append(
        "strategy=timing_plan but no plan was found "
        f"(looked in extra.timing_plan, extra.mix.timing_plan_source"
        + (f", {Path(mt_path).parent / 'timing'}/*.plan.json" if mt_path else "")
        + "); fell back to the shift-forward replay, which is NOT what mix.py rendered"
    )
    return None, None, reasons


def _replay_timing_plan(
    plan_obj: dict,
    manifest: dict,
) -> Tuple[List[PlacedSegment], List[PlacedPiece]]:
    """Render a TimingPlan's Pieces onto the output timeline.

    `Piece.dst_dur == src_span.dur / local_tempo` (backend/timing.py:88), and
    `dst_start` is already absolute output seconds, so no cursor is involved:
    this reads the placements the renderer used rather than re-deriving them.
    """
    texts = {
        str(s.get("id")): str(s.get("final_text") or s.get("mt_text") or s.get("text") or "")
        for s in manifest.get("segments", [])
    }
    placed: List[PlacedSegment] = []
    pieces: List[PlacedPiece] = []

    for sp in plan_obj.get("segments", []):
        sid = str(sp.get("id") or f"SEG_{len(placed):04d}")
        text = texts.get(sid, "")
        slot = sp.get("slot") or {}
        try:
            src_start = float(slot.get("start", 0.0))
            src_end = float(slot.get("end", src_start))
        except (TypeError, ValueError):
            src_start = src_end = 0.0

        seg_pieces: List[PlacedPiece] = []
        for k, pc in enumerate(sp.get("pieces", []) or []):
            span = pc.get("src_span") or {}
            try:
                if isinstance(span, (list, tuple)):
                    a, b = float(span[0]), float(span[1])
                else:
                    a, b = float(span.get("start", 0.0)), float(span.get("end", 0.0))
                tempo = float(pc.get("local_tempo", 1.0)) or 1.0
                dst0 = float(pc.get("dst_start"))
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            src_dur = max(0.0, b - a)
            dst_dur = src_dur / max(1e-6, tempo)
            seg_pieces.append(PlacedPiece(
                seg_id=sid, index=k, dst_start=dst0, dst_end=dst0 + dst_dur,
                local_tempo=tempo, anchor=str(pc.get("anchor") or "whole"),
                src_dur=src_dur, text=str(pc.get("text") or ""),
            ))

        if not seg_pieces:
            continue
        pieces.extend(seg_pieces)
        p0 = min(p.dst_start for p in seg_pieces)
        p1 = max(p.dst_end for p in seg_pieces)
        # One representative tempo per segment, duration-weighted so a long
        # capped piece is not averaged away by a short un-capped one.
        tot = sum(p.dst_dur_s for p in seg_pieces)
        tempo = (sum(p.local_tempo * p.dst_dur_s for p in seg_pieces) / tot) if tot > 0 else None
        placed.append(PlacedSegment(
            seg_id=sid, src_start=src_start, src_end=src_end,
            placed_start=p0, placed_end=p1, shift_s=p0 - src_start,
            clip_duration_s=p1 - p0, stretch_factor=tempo, text=text,
        ))

    placed.sort(key=lambda p: p.placed_start)
    pieces.sort(key=lambda p: p.dst_start)
    return placed, pieces


def replay_mix_plan(data: dict, *, mt_path: Optional[Path] = None) -> MixPlan:
    """Reconstruct where each dubbed clip landed on the output timeline.

    Two placement rules, because mix.py has two:

      * `strategy=timing_plan` — mix.py renders a `backend.timing.TimingPlan`
        piece by piece at each `Piece.dst_start`. There is no cursor and no
        shift-forward, so the plan is LOADED and read, not re-derived. Before
        this was wired up the replay silently re-ran the shift-forward rule
        here and was out by up to 3.4 s against the rendered wav, while five
        differently-planned arms came back bit-identical.
      * every other strategy — mix.py has no plan, and its rule (sort by
        source start, shift-forward cursor, then one global atempo) is
        deterministic and replayed exactly from the manifest alone.

    `placement_source` on the returned MixPlan always says which one ran.
    """
    mix = (data.get("extra") or {}).get("mix") or {}

    clip_source = mix.get("clip_source") or _MIX_DEFAULT_CLIP_SOURCE
    try:
        mix_tempo_cap: Optional[float] = float(mix["tempo_cap"])
    except (KeyError, TypeError, ValueError):
        mix_tempo_cap = None
    strategy = mix.get("strategy")
    strategy_source = "manifest"
    if not strategy:
        # Manifests written before mix.py recorded its strategy. Say so rather
        # than let the assumption vanish into a number.
        strategy = _MIX_DEFAULT_STRATEGY
        strategy_source = "assumed_mix_default"

    placed: List[PlacedSegment] = []
    skipped: List[Dict[str, str]] = []
    rows: List[Tuple[str, float, float, float, Optional[float], str]] = []

    # --- strategy=timing_plan: read the placements, do not re-derive them --
    pieces: List[PlacedPiece] = []
    placement_source = "shift_forward_cursor"
    plan_path: Optional[str] = None
    plan_summary: Dict[str, Any] = {}
    plan_params: Dict[str, Any] = {}
    plan_reasons: List[str] = []
    if strategy == "timing_plan":
        plan_obj, plan_path, plan_reasons = _load_timing_plan(data, mix, mt_path=mt_path)
        if plan_obj is not None:
            placed, pieces = _replay_timing_plan(plan_obj, data)
            plan_summary = plan_obj.get("summary") or {}
            plan_params = plan_obj.get("params") or {}
            if placed:
                placement_source = "timing_plan_pieces"
            else:
                plan_reasons.append(
                    f"timing plan {plan_path} rendered 0 placeable pieces; "
                    "fell back to the shift-forward replay")

    if placement_source == "timing_plan_pieces":
        planned = {p.seg_id for p in placed}
        for seg in data.get("segments", []):
            sid = str(seg.get("id") or "")
            if sid and sid not in planned:
                skipped.append({"id": sid, "reason": "no pieces in the timing plan"})
        fit_duration = max((p.placed_end for p in placed), default=0.0)
        source_duration = mix.get("source_duration_s")
        try:
            source_duration = float(source_duration)
        except (TypeError, ValueError):
            source_duration = None
        if not source_duration or source_duration <= 0:
            ends = [float(s.get("end", s.get("end_s", 0.0)) or 0.0)
                    for s in data.get("segments", [])]
            source_duration = max(max(ends) if ends else 0.0, 1.0)
        gt = mix.get("global_tempo")
        try:
            gt = float(gt)
        except (TypeError, ValueError):
            gt = None
        # mix.py applies no second atempo under timing_plan: the plan already
        # owns every tempo decision, per piece.
        tempo_source = "manifest" if gt and gt > 0 else "timing_plan_owns_tempo"
        if not gt or gt <= 0:
            gt = 1.0
        return MixPlan(
            segments=placed, strategy=strategy, strategy_source=strategy_source,
            clip_source=clip_source, global_tempo=gt, tempo_source=tempo_source,
            fit_duration_s=fit_duration, source_duration_s=source_duration,
            final_duration_s=fit_duration, skipped=skipped,
            pieces=pieces, placement_source=placement_source,
            plan_path=plan_path, plan_producer=mix.get("timing_producer"),
            plan_summary=plan_summary, plan_params=plan_params,
            plan_reasons=plan_reasons, mix_tempo_cap=mix_tempo_cap,
        )

    for seg in data.get("segments", []):
        sid = str(seg.get("id") or f"SEG_{len(rows):04d}")
        if _resolve_clip_field(seg, clip_source) is None:
            skipped.append({"id": sid, "reason": f"no clip for clip_source={clip_source}"})
            continue
        dur = _seg_clip_duration(seg, clip_source)
        if dur is None:
            skipped.append({"id": sid, "reason": "no usable tts_duration_s"})
            continue
        try:
            start = float(seg.get("start", seg.get("start_s", 0.0)))
            end = float(seg.get("end", seg.get("end_s", start)))
        except (TypeError, ValueError):
            skipped.append({"id": sid, "reason": "unusable start/end"})
            continue
        sf_ = seg.get("stretch_factor")
        try:
            sf_ = float(sf_)
        except (TypeError, ValueError):
            sf_ = None
        text = str(seg.get("final_text") or seg.get("mt_text") or seg.get("text") or "")
        rows.append((sid, start, end, dur, sf_, text))

    rows.sort(key=lambda r: r[1])

    cursor = 0.0
    for sid, start, end, dur, sf_, text in rows:
        p0 = max(start, cursor)
        cursor = p0 + dur
        placed.append(PlacedSegment(sid, start, end, p0, p0 + dur, p0 - start, dur, sf_, text))
    fit_duration = cursor

    source_duration = mix.get("source_duration_s")
    try:
        source_duration = float(source_duration)
    except (TypeError, ValueError):
        source_duration = None
    if not source_duration or source_duration <= 0:
        ends = [float(s.get("end", s.get("end_s", 0.0)) or 0.0)
                for s in data.get("segments", [])]
        source_duration = max(ends) if ends else 0.0
        source_duration = max(source_duration, 1.0)

    gt = mix.get("global_tempo")
    try:
        gt = float(gt)
    except (TypeError, ValueError):
        gt = None
    if gt and gt > 0:
        tempo_source = "manifest"
    else:
        cap = mix.get("tempo_cap")
        try:
            cap = float(cap)
        except (TypeError, ValueError):
            cap = _MIX_DEFAULT_TEMPO_CAP
        if strategy == "global_tempo" and fit_duration > source_duration + 0.01:
            gt = min(cap, fit_duration / source_duration)
        else:
            gt = 1.0
        tempo_source = "recomputed_from_" + strategy_source

    if gt != 1.0:
        # atempo retimes the whole mixed canvas, so every placement moves.
        for p in placed:
            p.placed_start /= gt
            p.placed_end /= gt
            p.clip_duration_s /= gt
            p.shift_s = p.placed_start - p.src_start

    return MixPlan(
        segments=placed, strategy=strategy, strategy_source=strategy_source,
        clip_source=clip_source, global_tempo=gt, tempo_source=tempo_source,
        fit_duration_s=fit_duration, source_duration_s=source_duration,
        final_duration_s=(fit_duration / gt) if gt else fit_duration,
        skipped=skipped,
        pieces=[], placement_source=placement_source,
        plan_path=plan_path, plan_producer=mix.get("timing_producer"),
        plan_summary=plan_summary, plan_params=plan_params,
        plan_reasons=plan_reasons, mix_tempo_cap=mix_tempo_cap,
    )


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    """The human Hindi dub of this clip, as `backend.eval refbuild` wrote it.

    Two tiers of ground truth live here and they are NOT interchangeable:

      * `speech` / `pauses` come from `ref_dub.vad.json` and exist for every
        clip that has a dub (15 of the 18). They are the PRIMARY reference.
      * `words` comes from `ref_dub.words.json` and is trustworthy only where
        `meta["words"]["ref_dub"]["vad_coverage"]["usable_for_word_metrics"]`
        is true (7 clips). On the other four NPTEL clips the transcript is
        under-subtitled -- 0.96-1.45 words per second of VAD speech against
        2.51-5.15 on a healthy clip -- so word timings there describe the
        subtitle file, not the speech. `words_usable` gates every word-level
        metric; an ineligible clip is recorded as not_measured and is never
        pooled into a word-level number.
    """
    method: str                          # how the ground truth was obtained
    clip_duration_s: Optional[float]
    speech: List[Tuple[float, float]]
    pauses: List[Tuple[float, float]]
    words: List[Tuple[str, float, float]]
    words_usable: bool
    words_reason: Optional[str]
    text: str
    tier: Optional[str]
    vad_backend: Optional[str]
    min_pause_s: float
    min_speech_s: float
    source_speech: List[Tuple[float, float]]
    source_pauses: List[Tuple[float, float]]
    paths: Dict[str, Optional[str]]
    agreement_median_ms: Optional[float]
    agreement_note: Optional[str]
    reasons: List[str]

    # Back-compat for callers that predate the pause/word split.
    @property
    def spans(self) -> List[Tuple[float, float]]:
        return self.speech

    @property
    def path(self) -> Optional[str]:
        return self.paths.get("ref_dub_vad")


def _coerce_spans(obj: Any) -> List[Tuple[float, float]]:
    """Accept `[[a,b],...]` (what refbuild writes) or `[{start,end},...]`."""
    out: List[Tuple[float, float]] = []
    if not isinstance(obj, list):
        return out
    for item in obj:
        try:
            if isinstance(item, dict):
                a = float(item.get("start", item.get("t0")))
                b = float(item.get("end", item.get("t1")))
            else:
                a, b = float(item[0]), float(item[1])
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if b > a:
            out.append((a, b))
    return sorted(out)


def _coerce_words(obj: Any) -> List[Tuple[str, float, float]]:
    out: List[Tuple[str, float, float]] = []
    if not isinstance(obj, list):
        return out
    for item in obj:
        if not isinstance(item, dict):
            continue
        try:
            a = float(item.get("start"))
            b = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        out.append((str(item.get("text") or item.get("word") or ""), a, b))
    return out


def _read_json(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    if not path.exists():
        return None, f"{path} does not exist"
    try:
        return json.loads(path.read_text()), None
    except Exception as e:
        return None, f"{path} unreadable: {type(e).__name__}: {e}"


def _word_eligibility(meta: Optional[dict], track: str) -> Tuple[bool, Optional[str]]:
    """`usable_for_word_metrics` for one track, plus why when it is false."""
    if not isinstance(meta, dict):
        return False, "no golden meta.json; word eligibility is unknown"
    cov = (((meta.get("words") or {}).get(track) or {}).get("vad_coverage") or {})
    if not cov:
        return False, (f"meta.json carries no words.{track}.vad_coverage; "
                       f"word eligibility is unknown")
    usable = bool(cov.get("usable_for_word_metrics"))
    if usable:
        return True, None
    bits = []
    if cov.get("words_per_speech_s") is not None:
        bits.append(f"words_per_speech_s={cov['words_per_speech_s']} "
                    f"(min {cov.get('words_per_speech_s_min')})")
    if cov.get("coverage") is not None:
        bits.append(f"coverage={cov['coverage']} (min {cov.get('coverage_min')})")
    return False, (f"meta.json says words.{track}.vad_coverage."
                   f"usable_for_word_metrics is false"
                   + (f": {', '.join(bits)}" if bits else "")
                   + "; the transcript is under-subtitled, so its word times "
                     "describe the subtitle file and not the speech")


def load_reference(
    clip_id: str,
    *,
    golden_root: Path,
    manifest: dict,
    plan: MixPlan,
) -> Reference:
    """Load the golden reference for `clip_id`, or fall back to the source.

    Tier 1 is what `python -m backend.eval refbuild` wrote under
    `data/golden/<clip_id>/`: `ref_dub.vad.json` (speech + pauses of the human
    Hindi dub) and, where `meta.json` says it is usable, `ref_dub.words.json`.

    Tier 2, when no dub exists for the clip, is the source segment grid in the
    manifest. That is a real reference -- it is the timeline the dub is meant
    to hold -- but a much weaker one, so `timing_method` records which tier a
    row came through and no row can quietly cross between them.

    The previous version of this function looked for `reference.json`,
    `reference.words.json` or `ref.json`. Nothing in this repository has ever
    written any of those three names, so every clip fell through to tier 2 and
    the whole sweep scored each dub against its own source slot grid instead of
    against the human dub.
    """
    reasons: List[str] = []
    golden_root = Path(golden_root)
    clip_dir = golden_root / clip_id

    meta, meta_err = _read_json(clip_dir / _GOLDEN_META)
    if meta_err:
        reasons.append(f"golden meta: {meta_err}")

    agreement, agreement_note = _clip_agreement_ms(
        clip_id, golden_root=golden_root, meta=meta)

    clip_duration = None
    if isinstance(meta, dict):
        try:
            clip_duration = float(((meta.get("source") or {}).get("duration_s")))
        except (TypeError, ValueError):
            clip_duration = None
    tier = (meta or {}).get("tier")

    src_vad, _ = _read_json(clip_dir / _GOLDEN_SRC_VAD)
    source_speech = _coerce_spans((src_vad or {}).get("speech"))
    source_pauses = _coerce_spans((src_vad or {}).get("pauses"))

    ref_vad, ref_vad_err = _read_json(clip_dir / _GOLDEN_REF_VAD)
    if ref_vad_err:
        reasons.append(f"golden ref_dub VAD: {ref_vad_err}")

    speech = _coerce_spans((ref_vad or {}).get("speech"))
    pauses = _coerce_spans((ref_vad or {}).get("pauses"))
    min_pause_s = float((ref_vad or {}).get("min_pause_s") or _GOLDEN_MIN_PAUSE_S)
    min_speech_s = float((ref_vad or {}).get("min_speech_s") or _GOLDEN_MIN_SPEECH_S)
    if clip_duration is None:
        try:
            clip_duration = float((ref_vad or {}).get("duration_s"))
        except (TypeError, ValueError):
            clip_duration = None

    ref_words_doc, ref_words_err = _read_json(clip_dir / _GOLDEN_REF_WORDS)
    words = _coerce_words((ref_words_doc or {}).get("words"))
    words_usable, words_reason = _word_eligibility(meta, "ref_dub")
    if ref_words_err:
        words_usable = False
        words_reason = f"golden ref_dub words: {ref_words_err}"
    elif words_usable and not words:
        words_usable = False
        words_reason = (f"{clip_dir / _GOLDEN_REF_WORDS} carried no usable "
                        f"word timings")
    if not words_usable and words_reason:
        reasons.append(f"word-level metrics: {words_reason}")

    paths = {
        "golden_dir": str(clip_dir),
        "ref_dub_vad": str(clip_dir / _GOLDEN_REF_VAD) if ref_vad else None,
        "ref_dub_words": str(clip_dir / _GOLDEN_REF_WORDS) if ref_words_doc else None,
        "source_vad": str(clip_dir / _GOLDEN_SRC_VAD) if src_vad else None,
        "meta": str(clip_dir / _GOLDEN_META) if meta else None,
    }

    if speech:
        return Reference(
            method="golden_ref_dub_vad",
            clip_duration_s=clip_duration,
            speech=speech, pauses=pauses, words=words,
            words_usable=words_usable, words_reason=words_reason,
            text=" ".join(w[0] for w in words).strip(),
            tier=tier, vad_backend=(ref_vad or {}).get("backend"),
            min_pause_s=min_pause_s, min_speech_s=min_speech_s,
            source_speech=source_speech, source_pauses=source_pauses,
            paths=paths, agreement_median_ms=agreement,
            agreement_note=agreement_note, reasons=reasons,
        )

    reasons.append(
        f"no human dub reference under {clip_dir}/ "
        f"(need {_GOLDEN_REF_VAD} with a non-empty speech[]); "
        f"scoring against the manifest's own source slot grid instead, which "
        f"CANNOT measure fidelity to the human dub -- "
        f"run `python -m backend.eval refbuild` if this clip has a dub"
    )
    slots = sorted({(p.src_start, p.src_end) for p in plan.segments})
    slots = [(a, b) for a, b in slots if b > a]
    fallback_speech = _merge_spans(slots, gap=min_pause_s) if slots else []
    fallback_pauses = [
        (fallback_speech[i][1], fallback_speech[i + 1][0])
        for i in range(len(fallback_speech) - 1)
        if fallback_speech[i + 1][0] - fallback_speech[i][1] >= min_pause_s
    ]
    if clip_duration is None and slots:
        clip_duration = max(b for _, b in slots)
    return Reference(
        method="source_segments",
        clip_duration_s=clip_duration,
        speech=fallback_speech, pauses=fallback_pauses, words=[],
        words_usable=False,
        words_reason="no golden ref_dub words: this clip has no human dub "
                     "reference at all",
        text=" ".join(str(s.get("text") or "")
                      for s in manifest.get("segments", [])).strip(),
        tier=tier, vad_backend=None,
        min_pause_s=min_pause_s, min_speech_s=min_speech_s,
        source_speech=source_speech, source_pauses=source_pauses,
        paths=paths, agreement_median_ms=agreement,
        agreement_note=agreement_note, reasons=reasons,
    )


def _clip_agreement_ms(
    clip_id: str,
    *,
    golden_root: Path,
    meta: Optional[dict] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """This clip's own ground-truth timing floor, if anyone measured it.

    `meta["agreement_median_ms"]` is per-track -- refbuild writes
    `{"source": {...}, "ref_dub": {...}}`, where each side is either
    `{"median_ms": ...}` or `{"not_measured": "<why>"}`. The ref_dub track is
    the one this harness scores against, so that is the one read. A clip whose
    floor was never measured returns `(None, "<why>")` and the corpus-wide
    `GT_NOISE_FLOOR_MS` is carried in the result alongside it.
    """
    def _dig(obj: Any) -> Tuple[Optional[float], Optional[str]]:
        if obj is None:
            return None, None
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            return float(obj), None
        if isinstance(obj, dict):
            track = obj.get("ref_dub", obj)
            if isinstance(track, dict):
                if track.get("median_ms") is not None:
                    try:
                        return float(track["median_ms"]), None
                    except (TypeError, ValueError):
                        return None, "median_ms was not a number"
                if track.get("not_measured"):
                    return None, str(track["not_measured"])
            return None, None
        return None, None

    if meta is None:
        for cand in (Path(golden_root) / clip_id / _GOLDEN_META,
                     Path(golden_root) / f"{clip_id}.meta.json"):
            meta, _ = _read_json(cand)
            if meta:
                break
    val, why = _dig((meta or {}).get("agreement_median_ms"))
    if val is not None or why:
        return val, why

    for entry in load_golden_index(golden_root):
        if entry.get("clip_id") != clip_id:
            continue
        src = entry.get("source_dir")
        if not src:
            break
        obj, _ = _read_json(Path(src) / "meta.json")
        return _dig((obj or {}).get("agreement_median_ms"))
    return None, "no agreement_median_ms recorded for this clip"


def load_golden_index(golden_root: Path = DEFAULT_GOLDEN_ROOT) -> List[dict]:
    path = Path(golden_root) / "index.json"
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text())
    except Exception as e:
        log.warning("golden index %s unreadable: %s", path, e)
        return []
    return obj if isinstance(obj, list) else obj.get("clips", [])


# ---------------------------------------------------------------------------
# Small numeric helpers (harness-owned primitives only)
# ---------------------------------------------------------------------------

def _pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[int(pos)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return {"n": 0, "median": None, "p90": None, "max": None, "mean": None}
    return {"n": len(vals), "median": _pct(vals, 0.5), "p90": _pct(vals, 0.9),
            "max": max(vals), "mean": sum(vals) / len(vals)}


def _merge_spans(spans: Sequence[Tuple[float, float]], *, gap: float) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _gaps(spans: Sequence[Tuple[float, float]], *, min_gap: float = 0.20) -> List[float]:
    g: List[float] = []
    ordered = sorted(spans)
    for (_, prev_end), (nxt_start, _) in zip(ordered, ordered[1:]):
        d = nxt_start - prev_end
        if d >= min_gap:
            g.append(d)
    return g


def _pauses_between(
    speech: Sequence[Tuple[float, float]],
    *,
    min_pause_s: float,
) -> List[Tuple[float, float]]:
    """Internal gaps between consecutive speech runs, at the golden VAD's rule.

    Same definition `backend.eval.reference.vad` uses for `ref_dub.vad.json`:
    INTERNAL gaps only, and only those at least `min_pause_s` long. Leading
    and trailing silence are not pauses the dub has to reproduce, so they are
    excluded on both sides of the comparison.
    """
    sp = sorted(speech)
    return [(sp[i][1], sp[i + 1][0]) for i in range(len(sp) - 1)
            if sp[i + 1][0] - sp[i][1] >= min_pause_s]


def _hyp_from_plan(
    plan: MixPlan,
    *,
    min_pause_s: float,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Dict[str, Any]]:
    """PLANNED hypothesis spans, straight off the TimingPlan's Pieces.

    Each piece occupies `[dst_start, dst_start + src_span.dur / local_tempo]`.
    Consecutive pieces closer together than `min_pause_s` are one speech run
    -- the renderer butts them up against each other and a listener hears one
    utterance -- so they are merged at exactly the threshold the golden VAD
    used, and what is left between runs is the planned pause structure.

    These are the dub's INTENTION. They are cheap (no audio decode) and they
    are what the timing groups score. A divergence between them and the VAD of
    the rendered wav is itself a finding, so `source` says which one this is.
    """
    if plan.pieces:
        raw = [(p.dst_start, p.dst_end) for p in plan.pieces if p.dst_end > p.dst_start]
        unit = "timing_plan_piece"
    else:
        raw = [(p.placed_start, p.placed_end) for p in plan.segments
               if p.placed_end > p.placed_start]
        unit = "mix_replay_segment"
    speech = _merge_spans(raw, gap=min_pause_s)
    pauses = _pauses_between(speech, min_pause_s=min_pause_s)
    info = {
        "source": "planned",
        "derived_from": unit,
        "measured": False,
        "placement_source": plan.placement_source,
        "n_raw": len(raw),
        "n_speech_runs": len(speech),
        "n_pauses": len(pauses),
        "merged_at_min_pause_s": min_pause_s,
        "note": "PLANNED placements, not measured audio: these are where the "
                "plan says speech starts and stops, merged at the golden "
                "VAD's min_pause_s so a 'pause' means the same thing on both "
                "sides. The audio groups re-derive this from the rendered wav.",
    }
    return speech, pauses, info


def _hyp_from_audio(
    wav: Path,
    *,
    min_pause_s: float,
    min_speech_s: float,
    backend: str = "auto",
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Dict[str, Any]]:
    """MEASURED hypothesis spans: VAD of the wav that actually got rendered.

    Uses the same `backend.eval.reference.vad` entry point, at the same
    operating point, that built `ref_dub.vad.json` -- otherwise the hypothesis
    and the reference would sit at two different thresholds and the pause
    comparison would measure the VADs, not the dubs.
    """
    try:
        from . import reference as _ref
    except Exception as e:
        return [], [], {"source": "not_measured", "measured": False,
                        "error": f"backend.eval.reference unavailable: "
                                 f"{type(e).__name__}: {e}"}
    try:
        res = _ref.vad(Path(wav), min_pause_s=min_pause_s,
                       min_speech_s=min_speech_s, backend=backend)
    except Exception as e:
        return [], [], {"source": "not_measured", "measured": False,
                        "error": f"vad({Path(wav).name}) raised "
                                 f"{type(e).__name__}: {e}"}
    speech = [(float(a), float(b)) for a, b in res.speech]
    pauses = [(float(a), float(b)) for a, b in res.pauses]
    info = {
        "source": "measured",
        "derived_from": "vad(rendered mix wav)",
        "measured": True,
        "wav": str(wav),
        "vad_backend": res.backend,
        "min_pause_s": res.min_pause_s,
        "min_speech_s": res.min_speech_s,
        "duration_s": res.duration_s,
        "n_speech_runs": len(speech),
        "n_pauses": len(pauses),
        "error": res.error,
    }
    return speech, pauses, info


def _plan_tempos(plan: MixPlan) -> List[float]:
    """Every usable time-stretch factor this variant actually applied.

    Under `strategy=timing_plan` the tempo is per PIECE, in
    `Piece.local_tempo`, and nothing writes it back to
    `segments[].stretch_factor` -- so looking only at the manifest reported
    `n_cap_saturated = 0` for every plan-driven arm by construction, however
    hard the cap was biting. The pieces are the authority when they exist;
    `segments[].stretch_factor` is the authority for the other strategies.
    """
    if plan.pieces:
        return [p.local_tempo for p in plan.pieces
                if p.local_tempo and math.isfinite(p.local_tempo) and p.local_tempo > 0]
    return [p.stretch_factor for p in plan.segments
            if p.stretch_factor and math.isfinite(p.stretch_factor)
            and p.stretch_factor > 0]


def _plan_tempo_caps(plan: MixPlan) -> Tuple[Optional[float], Optional[float], str]:
    """The tempo caps this variant was actually planned under, and where from.

    `backend.timing` clamps every `local_tempo` into `[min_tempo, max_tempo]`
    and those two are written into the plan's `params` -- 0.90/1.15 for the
    pause-aware, word-anchored and dp producers, 0.90/1.10 for naive_slot.
    They are ASYMMETRIC, so scoring them through `stretch_stats`'s default
    symmetric `1 +- 0.10` budget looks for a floor at 0.85 that no producer
    has and calls a piece pinned at 1.15 a cap BREACH rather than saturation.
    That is the same failure as reading tempo off `segments[].stretch_factor`:
    a number that is zero by construction rather than by measurement.

    Falls back to `extra.mix.tempo_cap` (symmetric, mix.py's own global-atempo
    ceiling) and finally to `(None, None, why)` so the caller records a reason
    rather than a wrong cap.
    """
    params = plan.plan_params or {}
    lo = params.get("min_tempo")
    hi = params.get("max_tempo")
    try:
        lo = float(lo) if lo is not None else None
        hi = float(hi) if hi is not None else None
    except (TypeError, ValueError):
        lo = hi = None
    if lo is not None or hi is not None:
        return lo, hi, (f"timing plan params (producer={plan.plan_producer}): "
                        f"min_tempo={lo}, max_tempo={hi}")
    if plan.mix_tempo_cap and plan.mix_tempo_cap > 0:
        cap = plan.mix_tempo_cap
        return 1.0 / cap, cap, (f"extra.mix.tempo_cap={cap} (mix.py's global "
                                f"atempo ceiling; the plan carried no "
                                f"min_tempo/max_tempo)")
    return None, None, ("neither the timing plan params nor extra.mix.tempo_cap "
                        "recorded a tempo cap")


def _tempo_source(plan: MixPlan) -> str:
    if plan.pieces:
        return "timing_plan Piece.local_tempo"
    return "manifest segments[].stretch_factor"


def _word_runs(
    words: Sequence[Tuple[str, float, float]],
    *,
    min_pause_s: float,
) -> List[Tuple[float, float]]:
    """Speech runs implied by a word stream, merged at the VAD's pause rule."""
    spans = [(a, b) for _t, a, b in words if b > a]
    return _merge_spans(spans, gap=min_pause_s) if spans else []


def _iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _fallback_match(
    ref: Sequence[Tuple[float, float]],
    hyp: Sequence[Tuple[float, float]],
    *,
    min_iou: float = 0.10,
) -> Tuple[int, int, int]:
    """Greedy IoU matching. Used only when `backend.eval.timing` is absent.

    n_matched / n_invented / n_missed are top-level schema fields the runner
    has to fill even when the metric module is missing, so there is a local
    matcher — but the result records that it was used.
    """
    pairs = sorted(
        ((_iou(r, h), i, j) for i, r in enumerate(ref) for j, h in enumerate(hyp)),
        reverse=True,
    )
    used_r: set[int] = set()
    used_h: set[int] = set()
    matched = 0
    for score, i, j in pairs:
        if score < min_iou:
            break
        if i in used_r or j in used_h:
            continue
        used_r.add(i)
        used_h.add(j)
        matched += 1
    return matched, len(hyp) - matched, len(ref) - matched


# ---------------------------------------------------------------------------
# Metric-module plumbing
# ---------------------------------------------------------------------------

def _lazy_module(name: str) -> Tuple[Any, Optional[str]]:
    """Import a sibling metric module, returning (module, None) or (None, why)."""
    try:
        import importlib
        return importlib.import_module(name), None
    except Exception as e:
        return None, f"{name} unavailable: {type(e).__name__}: {e}"


class _Recorder:
    """Collects group payloads plus the errors / not_measured ledgers."""

    def __init__(self) -> None:
        self.groups: Dict[str, Dict[str, Any]] = {g: {} for g in GROUPS}
        self.errors: List[Dict[str, str]] = []
        self.not_measured: List[Dict[str, str]] = []

    def skip(self, key: str, reason: str) -> None:
        self.not_measured.append({"key": key, "reason": reason})

    def fail(self, key: str, reason: str) -> None:
        self.errors.append({"key": key, "error": reason})
        self.not_measured.append({"key": key, "reason": reason})

    def call_as(self, group: str, key_name: str, fn_name: str, mod: Any,
                mod_error: Optional[str], *args: Any, **kwargs: Any) -> Any:
        """Call `mod.fn_name(...)` but file the result under `key_name`.

        Several metrics are legitimately run more than once per variant --
        `gap_stats` on the hypothesis and on the reference, `speech_iou` on the
        planned spans and again on the VAD of the rendered wav. The function
        name is therefore not the result key, and conflating the two is what
        made the old runner pass whatever arguments happened to be in scope.
        """
        key = f"{group}.{key_name}"
        if mod is None:
            self.skip(key, mod_error or "metric module unavailable")
            return None
        fn = getattr(mod, fn_name, None)
        if fn is None:
            self.skip(key, f"{mod.__name__}.{fn_name} is not defined")
            return None
        try:
            value = fn(*args, **kwargs)
        except TypeError as e:
            self.skip(key, f"{mod.__name__}.{fn_name} signature mismatch: {e}")
            return None
        except Exception as e:
            self.fail(key, f"{mod.__name__}.{fn_name} raised {type(e).__name__}: {e}")
            return None
        self.groups[group][key_name] = value
        return value

    def call(self, group: str, name: str, mod: Any, mod_error: Optional[str],
             *args: Any, **kwargs: Any) -> None:
        key = f"{group}.{name}"
        if mod is None:
            self.skip(key, mod_error or "metric module unavailable")
            return
        fn = getattr(mod, name, None)
        if fn is None:
            self.skip(key, f"{mod.__name__}.{name} is not defined")
            return
        try:
            value = fn(*args, **kwargs)
        except TypeError as e:
            # Signature drift between the runner and the metric module is a
            # measurement gap, not a crash.
            self.skip(key, f"{mod.__name__}.{name} signature mismatch: {e}")
        except Exception as e:
            self.fail(key, f"{mod.__name__}.{name} raised {type(e).__name__}: {e}")
        else:
            self.groups[group][name] = value


# ---------------------------------------------------------------------------
# Scoring one variant
# ---------------------------------------------------------------------------

def score_variant(
    clip_id: str,
    variant_id: str,
    inputs: VariantInputs,
    *,
    golden_root: Path = DEFAULT_GOLDEN_ROOT,
    groups: Sequence[str] = CHEAP,
    out_dir: Path = DEFAULT_OUT_DIR,
    write: bool = True,
) -> dict:
    """Score one variant of one clip and write its result JSON.

    Returns the result dict. Never raises on bad input data: a problem that
    stops a metric becomes an entry in `errors[]` / `not_measured[]`.
    """
    groups = tuple(dict.fromkeys(groups))
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise ValueError(f"unknown group(s) {unknown}; choose from {list(GROUPS)}")

    t0 = time.time()
    rec = _Recorder()
    input_block: Dict[str, Any] = {}
    reasons = validate_inputs(inputs, groups=groups, record=input_block)

    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "variant_id": variant_id,
        "timing_method": None,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "groups_requested": list(groups),
        "inputs": input_block,
        "n_segments": 0,
        "n_matched": 0,
        "n_invented": 0,
        "n_missed": 0,
        "reference": {}, "hypothesis": {}, "calibration": {},
        "timing": {}, "pauses": {}, "rate": {}, "audio": {}, "intel": {}, "mt": {},
        "errors": rec.errors,
        "not_measured": rec.not_measured,
    }

    # Drop groups whose inputs did not survive validation.
    blocked: Dict[str, str] = {}
    for g in groups:
        for need in _GROUP_REQUIRES.get(g, ()):
            r = input_block.get(need) or {}
            if not r.get("ok"):
                blocked[g] = f"{need}: {r.get('reason') or 'unavailable'}"
                break
    runnable = [g for g in groups if g not in blocked]
    for g, why in blocked.items():
        rec.skip(g, why)

    if "mt_json" in blocked.get("timing", "") or not (input_block.get("mt_json") or {}).get("ok"):
        result["timing_method"] = "none (manifest unusable)"
        if write:
            _write_result(result, out_dir=out_dir)
        return result

    manifest = json.loads(Path(inputs.mt_json).read_text())
    plan = replay_mix_plan(manifest, mt_path=Path(inputs.mt_json))
    for why in plan.plan_reasons:
        rec.skip("mix_replay", why)
    reference = load_reference(clip_id, golden_root=golden_root,
                               manifest=manifest, plan=plan)
    for why in reference.reasons:
        rec.skip("reference", why)

    result["timing_method"] = (
        f"{reference.method}+mix_replay:{plan.strategy}:{plan.placement_source}")
    result["n_segments"] = len(manifest.get("segments", []))

    # --- the canvas --------------------------------------------------------
    # Both masks are rasterised on ONE canvas so they are comparable at all.
    # Its length is the golden clip duration off meta.json (60.0 s for every
    # clip in this corpus -- read, never hardcoded).
    canvas_s = reference.clip_duration_s
    canvas_source = "golden meta.json source.duration_s"
    if not canvas_s or canvas_s <= 0:
        ends = [b for _a, b in reference.speech] + \
               [p.placed_end for p in plan.segments] + [plan.source_duration_s]
        canvas_s = max([e for e in ends if e] or [0.0])
        canvas_source = ("no golden clip duration; fell back to the longest "
                         "end across reference, placements and mix source duration")
        rec.skip("canvas", canvas_source)

    min_pause_s = reference.min_pause_s

    # --- reference ---------------------------------------------------------
    ref_speech = list(reference.speech)
    ref_pauses = list(reference.pauses)
    ref_word_runs = (_word_runs(reference.words, min_pause_s=min_pause_s)
                     if reference.words_usable else [])

    # --- hypothesis --------------------------------------------------------
    # Cheap groups score the PLAN. Audio groups score the rendered wav, which
    # is the audio that actually exists. Both are recorded, and so is which
    # one produced each number: a divergence between them is a finding.
    hyp_speech, hyp_pauses, hyp_info = _hyp_from_plan(plan, min_pause_s=min_pause_s)

    audio_hyp: Optional[Tuple[List[Tuple[float, float]],
                              List[Tuple[float, float]], Dict[str, Any]]] = None
    if AUDIO_GROUPS.intersection(runnable) and inputs.mix_wav:
        audio_hyp = _hyp_from_audio(
            Path(inputs.mix_wav), min_pause_s=min_pause_s,
            min_speech_s=reference.min_speech_s,
            backend=reference.vad_backend or "auto")
        if audio_hyp[2].get("error") and not audio_hyp[0]:
            rec.skip("hypothesis.audio_vad", str(audio_hyp[2]["error"]))

    result["reference"] = {
        "method": reference.method,
        "tier": reference.tier,
        "paths": reference.paths,
        "clip_duration_s": canvas_s,
        "clip_duration_source": canvas_source,
        "vad_backend": reference.vad_backend,
        "min_pause_s": reference.min_pause_s,
        "min_speech_s": reference.min_speech_s,
        "n_speech": len(ref_speech),
        "n_pauses": len(ref_pauses),
        "speech_s": round(sum(b - a for a, b in ref_speech), 4),
        "n_words": len(reference.words),
        "words_usable_for_word_metrics": reference.words_usable,
        "words_not_usable_reason": reference.words_reason,
        "n_source_speech": len(reference.source_speech),
        "agreement_median_ms": reference.agreement_median_ms,
        "agreement_note": reference.agreement_note,
    }
    result["hypothesis"] = {
        "cheap_groups": hyp_info,
        "audio_groups": (audio_hyp[2] if audio_hyp else
                         {"source": "not_run",
                          "reason": "no audio group requested, or no mix_wav"}),
    }
    result["calibration"] = {
        "gt_noise_floor_ms": GT_NOISE_FLOOR_MS,
        "gt_noise_floor_range_ms": list(GT_NOISE_FLOOR_RANGE_MS),
        "gt_noise_floor_basis": GT_NOISE_FLOOR_BASIS,
        "gt_source_floor_ms": GT_SOURCE_FLOOR_MS,
        "gt_source_floor_note":
            "the ENGLISH source-side word-level floor. Provenance only -- the "
            "ranking is scored against the Hindi dub, which has no word-level "
            "floor. Do not quote this as the ground truth's noise floor.",
        "noop_speech_iou_median": NOOP_SPEECH_IOU_MEDIAN,
        "noop_pause_f1_at_200ms_median": NOOP_PAUSE_F1_AT_200MS_MEDIAN,
        "note": "speech IoU is a GUARD, not a ranker: the untouched English "
                "source already scores 0.872 against the human Hindi dub, so "
                "its headroom is a third of its own clip-to-clip spread. "
                "pause F1 is the discriminator (no-op floor 0.000). No timing "
                "difference below gt_noise_floor_ms (185 ms, cue-level upper "
                "bound) is real; smaller differences are ties, not ranks.",
    }

    # --- structural counts -------------------------------------------------
    timing_mod, timing_err = (None, None)
    if {"timing", "pauses", "rate"}.intersection(runnable):
        timing_mod, timing_err = _lazy_module("backend.eval.timing")

    matched = invented = missed = None
    if timing_mod is not None and hasattr(timing_mod, "match_segments"):
        try:
            # Argument order is (hyp, ref) and the return is a LIST OF PAIRS
            # `(hyp_idx, ref_idx)`, not a triple of counts.
            m = timing_mod.match_segments(hyp_speech, ref_speech)
        except Exception as e:
            rec.skip("match_segments",
                     f"backend.eval.timing.match_segments failed "
                     f"({type(e).__name__}: {e}); used harness fallback matcher")
        else:
            if isinstance(m, dict):
                matched = m.get("n_matched", m.get("matched"))
                invented = m.get("n_invented", m.get("invented"))
                missed = m.get("n_missed", m.get("missed"))
            elif isinstance(m, (list, tuple)):
                pairs = [q for q in m
                         if isinstance(q, (list, tuple)) and len(q) == 2]
                if len(pairs) == len(m):
                    matched = sum(1 for a, b in pairs
                                  if a is not None and b is not None)
                    invented = sum(1 for a, b in pairs if b is None)
                    missed = sum(1 for a, b in pairs if a is None)
                    result["timing"]["match_segments"] = [list(q) for q in pairs]
                else:
                    rec.skip("match_segments",
                             "backend.eval.timing.match_segments returned "
                             f"{type(m).__name__} of unexpected shape; "
                             "used harness fallback matcher")
    if matched is None:
        rec.skip("match_segments",
                 (timing_err or "backend.eval.timing.match_segments unavailable")
                 + "; used harness fallback greedy-IoU matcher for "
                   "n_matched/n_invented/n_missed")
        matched, invented, missed = _fallback_match(ref_speech, hyp_speech)
    result["n_matched"], result["n_invented"], result["n_missed"] = (
        int(matched), int(invented), int(missed))

    # --- harness-owned primitives -----------------------------------------
    # These are manifest arithmetic, not metric formulas, so they are computed
    # here and kept under "_harness" so they can never collide with a key the
    # metric modules own.
    if "timing" in runnable:
        result["timing"]["_harness"] = {
            "mix_replay": {
                "strategy": plan.strategy,
                "strategy_source": plan.strategy_source,
                "placement_source": plan.placement_source,
                "timing_plan_path": plan.plan_path,
                "timing_producer": plan.plan_producer,
                "timing_plan_summary": plan.plan_summary,
                "clip_source": plan.clip_source,
                "global_tempo": plan.global_tempo,
                "tempo_source": plan.tempo_source,
                "fit_duration_s": plan.fit_duration_s,
                "source_duration_s": plan.source_duration_s,
                "final_duration_s": plan.final_duration_s,
                "drift_pct": (100.0 * (plan.final_duration_s - plan.source_duration_s)
                              / plan.source_duration_s) if plan.source_duration_s else None,
                "n_placed": len(plan.segments),
                "n_pieces": len(plan.pieces),
                "n_skipped": len(plan.skipped),
                "skipped": plan.skipped[:20],
            },
            "placement_shift_s": _summary([abs(p.shift_s) for p in plan.segments]),
            "reference": {
                "method": reference.method,
                "path": reference.path,
                "n_spans": len(ref_speech),
                "n_words": len(reference.words),
                "words_usable_for_word_metrics": reference.words_usable,
                "agreement_median_ms": reference.agreement_median_ms,
                "gt_noise_floor_ms": GT_NOISE_FLOOR_MS,
            },
        }
    if "pauses" in runnable:
        result["pauses"]["_harness"] = {
            "min_pause_s": min_pause_s,
            "ref_pause_s": _summary([b - a for a, b in ref_pauses]),
            "hyp_pause_s": _summary([b - a for a, b in hyp_pauses]),
            "source_gaps_s": _summary(_gaps(ref_speech, min_gap=min_pause_s)),
            "dub_gaps_s": _summary(_gaps(hyp_speech, min_gap=min_pause_s)),
            "noop_pause_f1_at_200ms_median": NOOP_PAUSE_F1_AT_200MS_MEDIAN,
        }
    if "rate" in runnable:
        cps = [len(p.text) / p.clip_duration_s
               for p in plan.segments if p.clip_duration_s > 0 and p.text]
        slot = sum(max(0.0, p.src_end - p.src_start) for p in plan.segments)
        spoken = sum(p.clip_duration_s * plan.global_tempo for p in plan.segments)
        result["rate"]["_harness"] = {
            "chars_per_s": _summary(cps),
            "stretch_factor": _summary(_plan_tempos(plan)),
            "tempo_source": _tempo_source(plan),
            "slot_s": slot,
            "raw_tts_s": spoken,
            "overflow_ratio": (spoken / slot) if slot > 0 else None,
        }

    # --- metric modules ----------------------------------------------------
    # Every call below passes what the callee's signature actually takes.
    # Masks are MASKS (`masks_from_spans`), not Span objects; the order is
    # (hyp, ref) where the signature says so; pauses are PAUSES, not speech.
    if "timing" in runnable:
        masks = None
        if timing_mod is not None and hasattr(timing_mod, "masks_from_spans"):
            try:
                hyp_mask = timing_mod.masks_from_spans(
                    hyp_speech, canvas_s, frame_s=_FRAME_S)
                ref_mask = timing_mod.masks_from_spans(
                    ref_speech, canvas_s, frame_s=_FRAME_S)
            except Exception as e:
                rec.fail("timing.masks_from_spans",
                         f"backend.eval.timing.masks_from_spans raised "
                         f"{type(e).__name__}: {e}")
            else:
                masks = (hyp_mask, ref_mask)
                result["timing"]["masks"] = {
                    "frame_s": _FRAME_S,
                    "total_s": canvas_s,
                    "total_s_source": canvas_source,
                    "n_frames": int(len(ref_mask)),
                    "hyp_speech_frames": int(sum(1 for v in hyp_mask if v)),
                    "ref_speech_frames": int(sum(1 for v in ref_mask if v)),
                }
        elif timing_mod is None:
            rec.skip("timing.masks_from_spans",
                     timing_err or "backend.eval.timing unavailable")
        else:
            rec.skip("timing.masks_from_spans",
                     "backend.eval.timing.masks_from_spans is not defined")

        rec.call("timing", "offset_stats", timing_mod, timing_err,
                 hyp_speech, ref_speech)
        if masks is not None:
            rec.call("timing", "speech_iou", timing_mod, timing_err,
                     masks[0], masks[1], frame_s=_FRAME_S)
            rec.call("timing", "drift_curve", timing_mod, timing_err,
                     masks[0], masks[1], frame_s=_FRAME_S)
        else:
            for name in ("speech_iou", "drift_curve"):
                rec.skip(f"timing.{name}",
                         "no comparable mask pair could be built on the clip "
                         "canvas; both metrics need masks, not spans")
        blk = rec.groups["timing"].get("speech_iou")
        if isinstance(blk, dict):
            # Carried next to the value so a reader cannot mistake 0.9 for a
            # good score: the no-op baseline is already at 0.872.
            blk["noop_floor_iou_median"] = NOOP_SPEECH_IOU_MEDIAN
            blk["interpretation"] = (
                "GUARD, not a ranker: the untouched source already scores "
                f"{NOOP_SPEECH_IOU_MEDIAN} against the human dub.")
        blk = rec.groups["timing"].get("offset_stats")
        if isinstance(blk, dict):
            blk["gt_noise_floor_ms"] = GT_NOISE_FLOOR_MS

        # Word-level offsets, gated on this clip's own eligibility. The word
        # stream is merged into runs at the VAD's own pause rule and scored
        # through the same offset formula; it is NEVER pooled with a clip
        # whose transcript meta.json marked unusable.
        if reference.words_usable and ref_word_runs:
            blk = rec.call_as("timing", "offset_stats_words", "offset_stats",
                              timing_mod, timing_err, hyp_speech, ref_word_runs)
            if isinstance(blk, dict):
                blk["reference"] = (
                    f"{len(reference.words)} ref_dub words merged into "
                    f"{len(ref_word_runs)} runs at min_pause_s={min_pause_s}")
                blk["gt_noise_floor_ms"] = GT_NOISE_FLOOR_MS
        else:
            rec.skip("timing.offset_stats_words",
                     reference.words_reason or
                     "clip is not eligible for word-level metrics")

    if "pauses" in runnable:
        # pause_f1(hyp_pauses, ref_pauses): PAUSES, in that order. It used to
        # be handed (ref_speech, hyp_speech) -- speech spans as pauses, the
        # wrong way round -- which is why swallowed_s came back as the total
        # raw TTS duration and invented_s as the total slot duration.
        rec.call("pauses", "pause_f1", timing_mod, timing_err,
                 hyp_pauses, ref_pauses, tol_s=0.20)
        blk = rec.groups["pauses"].get("pause_f1")
        if isinstance(blk, dict):
            blk["noop_floor_f1_median"] = NOOP_PAUSE_F1_AT_200MS_MEDIAN
            blk["interpretation"] = (
                "the discriminator: the no-op baseline scores "
                f"{NOOP_PAUSE_F1_AT_200MS_MEDIAN} here.")
        # gap_stats takes ONE sequence of segments, so each side is measured
        # on its own and the two are reported side by side.
        rec.call_as("pauses", "gap_stats_hyp", "gap_stats",
                    timing_mod, timing_err, hyp_speech)
        rec.call_as("pauses", "gap_stats_ref", "gap_stats",
                    timing_mod, timing_err, ref_speech)

    if "rate" in runnable:
        # rate_stats(segments, ref_words): segments need text AND a positive
        # duration on the same object, and ref_words are the golden dub's.
        rate_segs = [{"text": p.text, "tts_duration_s": p.clip_duration_s}
                     for p in plan.segments if p.clip_duration_s > 0]
        if reference.words_usable and reference.words:
            ref_words_arg = [{"text": t, "start": a, "end": b}
                             for t, a, b in reference.words]
        else:
            ref_words_arg = None
            rec.skip("rate.rate_stats.ref",
                     reference.words_reason or
                     "clip is not eligible for word-level metrics, so the "
                     "reference speaking rate cannot be measured")
        rec.call("rate", "rate_stats", timing_mod, timing_err,
                 rate_segs, ref_words_arg)
        # Under strategy=timing_plan the tempo lives in Piece.local_tempo, not
        # on segments[].stretch_factor, so the factors are taken from whichever
        # one actually carries them.
        tempos = _plan_tempos(plan)
        # ... and the cap those tempos were clamped against is the producer's
        # own asymmetric [min_tempo, max_tempo], not stretch_stats's default
        # symmetric 1 +- 0.10. Measuring saturation against the wrong cap is
        # how `n_cap_saturated` stays 0 on an arm whose plan summary already
        # counted five cap hits.
        cap_lo, cap_hi, cap_src = _plan_tempo_caps(plan)
        cap_kwargs: Dict[str, Any] = {}
        if cap_lo is not None:
            cap_kwargs["cap_low"] = cap_lo
        if cap_hi is not None:
            cap_kwargs["cap_high"] = cap_hi
        if not cap_kwargs:
            rec.skip("rate.stretch_stats.cap",
                     cap_src + "; saturation is measured against "
                     "stretch_stats's default symmetric 1 +- 0.10 budget, "
                     "which is NOT the cap this variant ran under")
        rec.call("rate", "stretch_stats", timing_mod, timing_err, tempos,
                 **cap_kwargs)
        blk = rec.groups["rate"].get("stretch_stats")
        if isinstance(blk, dict):
            blk["factor_source"] = _tempo_source(plan)
            blk["factor_unit"] = ("timing-plan piece" if plan.pieces
                                  else "manifest segment")
            blk["cap_source"] = cap_src
            cap_hits = plan.plan_summary.get("tempo_cap_hits")
            blk["plan_tempo_cap_hits"] = cap_hits
            blk["plan_tempo_cap_hits_unit"] = (
                "SEGMENTS whose plan hit the tempo cap, as the TimingPlan "
                "summary counted them -- not comparable one-for-one with "
                "n_cap_saturated, which counts PIECES" if cap_hits is not None
                else None)
            if cap_hits is None and plan.placement_source == "timing_plan_pieces":
                rec.skip("rate.stretch_stats.tempo_cap_hits",
                         "the timing plan summary carried no tempo_cap_hits")
            for k in ("tempo_min", "tempo_max", "tempo_mean"):
                if plan.plan_summary.get(k) is not None:
                    blk[f"plan_{k}"] = plan.plan_summary[k]
        if not tempos:
            rec.skip("rate.stretch_stats",
                     f"neither the timing plan's Piece.local_tempo nor "
                     f"segments[].stretch_factor carried a usable tempo "
                     f"(strategy={plan.strategy}, "
                     f"placement_source={plan.placement_source})")

    if {"audio", "intel"}.intersection(runnable):
        audio_mod, audio_err = _lazy_module("backend.eval.audio")
        mix_wav = Path(inputs.mix_wav) if inputs.mix_wav else None
        a_speech, a_pauses, a_info = (audio_hyp if audio_hyp
                                      else ([], [], {"source": "not_measured"}))
        if "audio" in runnable:
            rec.call("audio", "clipping_stats", audio_mod, audio_err, mix_wav)
            # truncation_check(mix_wav, mix_mp4) compares the wav against the
            # muxed video, not against spans.
            if inputs.mix_mp4:
                rec.call("audio", "truncation_check", audio_mod, audio_err,
                         mix_wav, Path(inputs.mix_mp4))
            else:
                rec.skip("audio.truncation_check",
                         "no mix_mp4 supplied; the tail-drop check compares "
                         "the dubbed wav against the muxed video")
            # overlap_stats takes ONE sequence of placements.
            rec.call("audio", "overlap_stats", audio_mod, audio_err,
                     [{"start": p.dst_start, "end": p.dst_end}
                      for p in plan.pieces] or
                     [{"start": p.placed_start, "end": p.placed_end}
                      for p in plan.segments])
            if inputs.tts_only_wav:
                rec.call("audio", "pitch_shift_cents", audio_mod, audio_err,
                         Path(inputs.tts_only_wav), mix_wav)
            else:
                rec.skip("audio.pitch_shift_cents",
                         "no tts_only_wav supplied; pitch shift needs the "
                         "unstretched synth as its reference")
            # The audio path's own view of timing, measured not planned.
            if a_speech and timing_mod is None:
                timing_mod, timing_err = _lazy_module("backend.eval.timing")
            if a_speech and timing_mod is not None:
                try:
                    a_hyp_mask = timing_mod.masks_from_spans(
                        a_speech, canvas_s, frame_s=_FRAME_S)
                    a_ref_mask = timing_mod.masks_from_spans(
                        ref_speech, canvas_s, frame_s=_FRAME_S)
                except Exception as e:
                    rec.fail("audio.measured_masks",
                             f"masks_from_spans raised {type(e).__name__}: {e}")
                else:
                    blk = rec.call_as("audio", "speech_iou_measured",
                                      "speech_iou", timing_mod, None,
                                      a_hyp_mask, a_ref_mask, frame_s=_FRAME_S)
                    if isinstance(blk, dict):
                        blk["noop_floor_iou_median"] = NOOP_SPEECH_IOU_MEDIAN
                    blk = rec.call_as("audio", "pause_f1_measured", "pause_f1",
                                      timing_mod, None, a_pauses, ref_pauses,
                                      tol_s=0.20)
                    if isinstance(blk, dict):
                        blk["noop_floor_f1_median"] = NOOP_PAUSE_F1_AT_200MS_MEDIAN
                    rec.groups["audio"]["_measured_vs_planned"] = {
                        "note": "the audio path VADs the wav that was actually "
                                "rendered; the cheap path reads the plan. A "
                                "divergence between the two is a finding, not "
                                "a rounding error.",
                        "planned_speech_s": round(
                            sum(b - a for a, b in hyp_speech), 4),
                        "measured_speech_s": round(
                            sum(b - a for a, b in a_speech), 4),
                        "planned_n_runs": len(hyp_speech),
                        "measured_n_runs": len(a_speech),
                        "planned_n_pauses": len(hyp_pauses),
                        "measured_n_pauses": len(a_pauses),
                        "vad": a_info,
                    }
            elif not a_speech:
                rec.skip("audio.speech_iou_measured",
                         f"no measured speech spans from the rendered wav "
                         f"({a_info.get('error') or a_info.get('source')})")
        if "intel" in runnable:
            rec.call("intel", "roundtrip_wer", audio_mod, audio_err, mix_wav,
                     " ".join(p.text for p in plan.segments))
            # vad_xcorr(hyp_mask, ref_mask) takes MASKS, not a wav and spans.
            if a_speech and timing_mod is not None:
                try:
                    x_hyp = timing_mod.masks_from_spans(a_speech, canvas_s,
                                                        frame_s=_FRAME_S)
                    x_ref = timing_mod.masks_from_spans(ref_speech, canvas_s,
                                                        frame_s=_FRAME_S)
                except Exception as e:
                    rec.fail("intel.vad_xcorr",
                             f"masks_from_spans raised {type(e).__name__}: {e}")
                else:
                    rec.call("intel", "vad_xcorr", audio_mod, audio_err,
                             x_hyp, x_ref, frame_s=_FRAME_S)
            else:
                rec.skip("intel.vad_xcorr",
                         "vad_xcorr needs a MEASURED hypothesis mask from the "
                         "rendered wav and a reference mask; one of them is "
                         "unavailable")

    if "mt" in runnable:
        text_mod, text_err = _lazy_module("backend.eval.text")
        hyp_text = " ".join(p.text for p in plan.segments).strip()
        if not reference.text:
            rec.skip("mt", "no reference text for this clip "
                           f"(reference method: {reference.method})")
        else:
            rec.call("mt", "chrfpp_report", text_mod, text_err, reference.text, hyp_text)
            rec.call("mt", "length_ratio", text_mod, text_err, reference.text, hyp_text)
            rec.call("mt", "safe_cer", text_mod, text_err, reference.text, hyp_text)
            rec.call("mt", "roundtrip_mt_wer", text_mod, text_err, reference.text, hyp_text)
            rec.call("mt", "term_retention", text_mod, text_err,
                     [str(s.get("text") or "") for s in manifest.get("segments", [])],
                     [p.text for p in plan.segments])
            rec.call("mt", "term_consistency", text_mod, text_err,
                     [p.text for p in plan.segments])

    for g in GROUPS:
        if rec.groups[g]:
            result[g].update(rec.groups[g])

    for r in reasons:
        if not any(e["reason"] == r for e in rec.not_measured):
            rec.skip("inputs", r)

    result["elapsed_s"] = round(time.time() - t0, 3)
    if write:
        result["result_path"] = str(_write_result(result, out_dir=out_dir))
    return result


def _write_result(result: dict, *, out_dir: Path) -> Path:
    dest = Path(out_dir) / result["clip_id"] / f"{result['variant_id']}.timing_eval.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return dest


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep(
    clips: Optional[Sequence[str]],
    variants: Sequence[VariantSpec],
    *,
    groups: Sequence[str] = CHEAP,
    jobs: int = 1,
    golden_root: Path = DEFAULT_GOLDEN_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> List[dict]:
    """Score every variant (optionally filtered to `clips`) and return results.

    Threads rather than processes: the expensive groups spend their time in
    numpy/ffmpeg/torch, all of which drop the GIL, and the cheap groups are
    too small for process startup to pay for itself.
    """
    todo = [v for v in variants if not clips or v.clip_id in set(clips)]
    if not todo:
        log.warning("sweep: no variants matched clips=%s", clips)
        return []

    def _one(spec: VariantSpec) -> dict:
        try:
            return score_variant(spec.clip_id, spec.variant_id, spec.inputs,
                                 golden_root=golden_root, groups=groups,
                                 out_dir=out_dir)
        except Exception as e:                     # a bug here must not kill the sweep
            log.exception("sweep: %s/%s failed", spec.clip_id, spec.variant_id)
            return {
                "schema_version": SCHEMA_VERSION, "clip_id": spec.clip_id,
                "variant_id": spec.variant_id, "timing_method": None,
                "inputs": {}, "n_segments": 0, "n_matched": 0,
                "n_invented": 0, "n_missed": 0,
                "timing": {}, "pauses": {}, "rate": {}, "audio": {},
                "intel": {}, "mt": {},
                "errors": [{"key": "score_variant",
                            "error": f"{type(e).__name__}: {e}"}],
                "not_measured": [{"key": "all",
                                  "reason": f"runner raised {type(e).__name__}: {e}"}],
            }

    if jobs <= 1:
        return [_one(s) for s in todo]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_one, todo))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_MT_SUFFIX = ".mt.json"


def discover_variants(
    patterns: Sequence[str],
    *,
    require_tts: bool = True,
    golden_root: Path = DEFAULT_GOLDEN_ROOT,
) -> List[VariantSpec]:
    """Build VariantSpecs from merged MT manifests on disk.

    `require_tts` keeps only the runs that actually synthesized audio — on
    this corpus that is five manifests out of 142, and the other 137 are
    text-stage only, so scoring them would manufacture empty rows.
    """
    paths: List[Path] = []
    for pat in patterns:
        hits = sorted(Path().glob(pat)) if any(c in pat for c in "*?[") else [Path(pat)]
        for h in hits:
            if h.is_dir():
                paths.extend(sorted(h.glob(f"*{_MT_SUFFIX}")))
            elif h.exists():
                paths.append(h)
    seen: set[str] = set()
    index = {e.get("clip_id") for e in load_golden_index(golden_root)}

    specs: List[VariantSpec] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            log.warning("discover: %s is not readable JSON (%s); skipped", p, e)
            continue
        segs = data.get("segments", [])
        if require_tts and not any(s.get("tts_audio") or s.get("vc_audio") for s in segs):
            continue
        stem = p.name[: -len(_MT_SUFFIX)] if p.name.endswith(_MT_SUFFIX) else p.stem
        clip_id = stem.split(".", 1)[0]
        if clip_id not in index:
            # Not a golden clip; keep the manifest's own base as the clip id so
            # the result still files somewhere sane.
            clip_id = stem.split(".", 1)[0]
        variant_id = stem[len(clip_id):].lstrip(".") or "default"
        mix = (data.get("extra") or {}).get("mix") or {}
        specs.append(VariantSpec(
            clip_id=clip_id, variant_id=variant_id,
            inputs=VariantInputs(
                mt_json=p,
                mix_wav=Path(mix["out_wav"]) if mix.get("out_wav") else None,
                mix_mp4=Path(mix["out_mp4"]) if mix.get("out_mp4") else None,
                tts_only_wav=None,
            ),
        ))
    return specs
