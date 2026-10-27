"""The `warped` timing producer: one continuous rate curve per cue.

This is the fourth way the repo can pace a dub, and the first that does not
hand the mixer a constant tempo:

    naive_slot    one Piece per cue, one tempo over speech AND silence
    pause_aware   N Pieces anchored to the SOURCE onsets
    word_anchored / dp   the research arms
    warped        ONE Piece per cue, already stretched, local_tempo = 1.0

The chain:

    VAD the clip          backend.pause_absorb.detect_pauses
      -> speech seconds, internal gaps, edge silence
    smooth across cues    backend.tempo_plan.plan_tempo
      -> islands; per cue a slot (out_start, out_end) whose boundary change
         is bounded by `max_rel_step`, with island spans conserved exactly
    split inside the cue  backend.pause_absorb.allocate
      -> silence to its floor before the speech moves; expands the silence
         when the cue has room, instead of drawling
    one curve             backend.rate_ramp.rate_envelope + build_warp
      -> r(t): fast in the gaps, slow in the speech, smoothstep at the joints
    render                backend.rate_ramp.apply_warp
      -> `tts_warped_audio`, exactly `out_slot` long

WHY THE PRODUCER RENDERS AUDIO. A `TimingPlan` is `(file, trim, delay,
tempo)`; there is no way to spell a time-varying rate in it. So the plan this
producer emits -- one Piece, `local_tempo = 1.0` -- is only meaningful against
files that already exist. Keeping the render in a separate stage would mean
emitting a plan that cannot be rendered until something else has run, which
is a worse contract than a slow producer. Stage 9f2 is therefore minutes
rather than milliseconds on this path, and the log says so per cue.

WHY TWO PASSES AND NOT ONE. `plan_tempo` decides how time is shared BETWEEN
cues; `allocate` decides how one cue's slot is split between its speech and
its silence. Running the allocator second, against the slot the planner
fixed, means island conservation holds by construction -- `out_slot` is never
touched again -- while the realised speech rate is still chosen silence-first.
The cost is that the realised rate can drift off the planner's by a little,
so the slew limit is VERIFIED on the realised tempos and reported as
`tempo_step_max` rather than asserted.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .timing import (Piece, SegmentPlan, Span, TimingPlan, _base_segment_plan,
                     DEFAULT_PAUSE_MIN_S, summarize)

log = logging.getLogger(__name__)

__all__ = ["plan_warped"]


def _read_mono(path: Path):
    """(samples, sr) as float32 mono. Stereo is averaged, not left-channelled."""
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = np.mean(y, axis=1).astype("float32")
    return y, int(sr)


def _silence_spans(cp, duration_s: float, *, edge: bool = True
                   ) -> List[Tuple[float, float]]:
    """Every silent span in the clip: lead, internal gaps, tail.

    Edges are included as ordinary gaps. The engine's run-up is not prosody --
    the slot already decides where the cue starts -- so there is no reason to
    protect it more than a breath, and treating it uniformly keeps one code
    path instead of two.
    """
    spans: List[Tuple[float, float]] = []
    if not cp.speech:
        return spans
    if edge and cp.speech[0][0] > 1e-4:
        spans.append((0.0, float(cp.speech[0][0])))
    for a, b in zip(cp.speech, cp.speech[1:]):
        if b[0] - a[1] > 1e-4:
            spans.append((float(a[1]), float(b[0])))
    if edge and duration_s - cp.speech[-1][1] > 1e-4:
        spans.append((float(cp.speech[-1][1]), float(duration_s)))
    return spans


def plan_warped(
    merged_json: Path,
    *,
    # smoothness / bounds -- see backend.tempo_plan.TempoConfig
    max_rel_step: float = 0.20,
    tempo_min: float = 0.80,
    tempo_max: float = 1.45,
    tempo_hard_max: float = 1.80,
    pause_borrow_s: float = 0.30,
    pause_min_s: float = 0.08,
    island_gap_s: float = 1.00,
    break_on_speaker: bool = True,
    break_on_scene: bool = True,
    # ramp
    ramp_s: float = 0.18,
    ramp_prefer_silence: bool = True,
    max_boundary_drift: float = 0.02,
    # pause absorption
    absorb_max_frac: float = 0.60,
    vad_backend: str = "energy",
    # rendering
    warp_dir: Optional[Path] = None,
    warp_backend: str = "auto",
    out_json: Optional[Path] = None,
) -> TimingPlan:
    """Plan and render a warped dub. See the module docstring for the chain."""
    from . import pause_absorb as _absorb
    from . import rate_ramp as _ramp
    from .tempo_plan import Cue, TempoConfig, plan_tempo, validate

    import numpy as np
    import soundfile as sf

    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    segs = list(data.get("segments", []))

    # Preflight, before a single cue is planned. Finding out that nothing on
    # this host can render a time map AFTER the whole file has been planned
    # is the expensive way to learn it, and `backend/stretch.py` records that
    # pyrubberband has been missing here before.
    chosen = (_ramp.available_backend() if warp_backend == "auto"
              else warp_backend)
    if chosen is None:
        raise RuntimeError(
            "timing producer 'warped' needs a time-map stretcher.\n"
            + _ramp.install_hint())
    if chosen == "pytsmod":
        log.warning("warp: using pytsmod (WSOLA). rubberband gives better "
                    "quality on speech; see backend.rate_ramp.install_hint()")

    cfg = TempoConfig(
        max_rel_step=max_rel_step, tempo_min=tempo_min, tempo_max=tempo_max,
        tempo_hard_max=tempo_hard_max, pause_borrow_s=pause_borrow_s,
        pause_min_s=pause_min_s, island_gap_s=island_gap_s,
        break_on_speaker=break_on_speaker, break_on_scene=break_on_scene,
        ramp_s=ramp_s, ramp_prefer_silence=ramp_prefer_silence,
    )
    warp_dir = Path(warp_dir) if warp_dir else merged_json.parent.parent / "tts" / "warped"
    warp_dir.mkdir(parents=True, exist_ok=True)

    # ---- pass 0: measure every clip -------------------------------------
    cues: List[Cue] = []
    meta: Dict[str, dict] = {}
    for seg in segs:
        clip = seg.get("tts_raw_audio") or seg.get("tts_audio")
        sid = seg.get("id", "SEG_????")
        start = float(seg.get("start", seg.get("start_s", 0.0)))
        end = float(seg.get("end", seg.get("end_s", 0.0)))
        if not clip or not Path(clip).exists():
            meta[sid] = {"skip": "no_audio"}
            continue
        try:
            raw = float(sf.info(str(clip)).duration)
        except Exception as exc:                            # noqa: BLE001
            meta[sid] = {"skip": f"unprobeable: {type(exc).__name__}"}
            continue
        if raw <= 0:
            meta[sid] = {"skip": "empty_clip"}
            continue

        cp = _absorb.detect_pauses(Path(clip), duration_s=raw,
                                   backend=vad_backend,
                                   max_runs=10_000)   # no merging: gaps are rates
        if cp.ok:
            spans = _silence_spans(cp, raw)
            speech_s = max(1e-3, raw - sum(e - s for s, e in spans))
        else:
            spans, speech_s = [], raw
        # Both bounds apply, so the floor is the LARGER of them: a gap may
        # not fall below its tier floor, and it may not give up more than
        # `absorb_max_frac` of itself. `allocate` enforces the same pair.
        floors = [max(_absorb.pause_floor(e - s),
                      (e - s) * (1.0 - absorb_max_frac)) for s, e in spans]
        # `fixed_s` is the TIGHTEST the silence may be, which is what gives
        # the planner the most room. Whatever slot it then hands back is
        # re-split by `allocate` below, so nothing is lost by being greedy.
        fixed_s = float(sum(max(f, 0.0) for f in floors))

        meta[sid] = {"clip": str(clip), "raw": raw, "spans": spans,
                     "speech_s": speech_s, "vad": cp.backend,
                     "vad_error": cp.error}
        cues.append(Cue(
            id=sid, start=start, end=end, tts_duration=raw,
            pinned=bool(seg.get("on_screen")), speaker=seg.get("speaker"),
            scene=int(seg.get("scene") or 0),
            speech_s=speech_s, fixed_s=min(fixed_s, 0.9 * max(raw, 1e-6)),
        ))

    plan_obj = TimingPlan(producer="warped", params={
        "max_rel_step": max_rel_step, "tempo_min": tempo_min,
        "tempo_max": tempo_max, "tempo_hard_max": tempo_hard_max,
        "pause_borrow_s": pause_borrow_s, "pause_min_s": pause_min_s,
        "island_gap_s": island_gap_s, "break_on_speaker": break_on_speaker,
        "break_on_scene": break_on_scene, "ramp_s": ramp_s,
        "ramp_prefer_silence": ramp_prefer_silence,
        "absorb_max_frac": absorb_max_frac, "vad_backend": vad_backend,
        "warp_backend": chosen, "max_boundary_drift": max_boundary_drift,
    })
    if not cues:
        log.warning("warp: no segment carries a usable clip; nothing planned")
        for seg in segs:
            sp = _base_segment_plan(seg, pause_min_s=DEFAULT_PAUSE_MIN_S)
            sp.qc.update(meta.get(sp.id, {"skip": "no_audio"}))
            plan_obj.segments.append(sp)
        plan_obj.summary = summarize(plan_obj)
        return plan_obj

    # ---- pass 1: smooth across cues --------------------------------------
    cue_plans = plan_tempo(cues, cfg)
    by_id = {p.id: p for p in cue_plans}
    report = validate(cues, cue_plans, cfg)
    log.info("warp: %d cues in %d island(s); planned max step %.3f "
             "(limit %.2f, naive %.3f), island error %.1e s",
             len(cues), report["n_islands"],
             report["planned"]["max_rel_step"], max_rel_step,
             report["naive"]["max_rel_step"], report["max_island_dur_err_s"])

    # ---- pass 2: split each slot silence-first, then realise the tempos ---
    realised: Dict[str, float] = {}
    allocs: Dict[str, object] = {}
    for c in cues:
        cp_plan = by_id[c.id]
        spans = meta[c.id]["spans"]
        gaps = [e - s for s, e in spans]
        a = _absorb.allocate(
            meta[c.id]["speech_s"], gaps, cp_plan.out_slot,
            max_speech_tempo=tempo_hard_max, min_speech_tempo=tempo_min,
            max_absorb_frac=absorb_max_frac)
        realised[c.id] = a.speech_tempo
        allocs[c.id] = a

    # lead/trail must follow the REALISED rates, or the joints are matched
    # against numbers nothing rendered.
    #
    # Neighbours are ISLAND-LOCAL. Taking them in global order would couple
    # the last cue of one island to the first of the next -- across a scene
    # cut, a speaker change or a second of silence, which are precisely the
    # places the planner refused to couple. The ramp would then ease toward a
    # rate the next cue does not have.
    from .tempo_plan import _islands

    for group in _islands(cues, cfg):
        ids = [cues[i].id for i in group]
        for j, cid in enumerate(ids):
            p = by_id[cid]
            prev_t = realised[ids[j - 1]] if j > 0 else realised[cid]
            next_t = realised[ids[j + 1]] if j + 1 < len(ids) else realised[cid]
            p.tempo = realised[cid]
            p.lead_tempo = math.sqrt(realised[cid] * prev_t)
            p.trail_tempo = math.sqrt(realised[cid] * next_t)

    # ---- pass 3: envelope, warp, render ----------------------------------
    n_drift = n_failed = 0
    for seg in segs:
        sid = seg.get("id", "SEG_????")
        sp = _base_segment_plan(seg, pause_min_s=DEFAULT_PAUSE_MIN_S)
        info = meta.get(sid, {})
        if sid not in by_id or "skip" in info:
            sp.qc["skip"] = info.get("skip", "not_planned")
            plan_obj.segments.append(sp)
            continue

        cp_plan = by_id[sid]
        a = allocs[sid]
        spans = info["spans"]
        gap_rates = [
            (s, e, max(1e-3, (e - s) / max(kept, 1e-4)))
            for (s, e), kept in zip(spans, a.kept_gaps)
        ]
        try:
            y, sr = _read_mono(Path(info["clip"]))
            wr = _ramp.build_warp(
                cp_plan, info["raw"], cfg, sr, len(y),
                silence_spans=spans, gap_rates=gap_rates,
                max_boundary_drift=max_boundary_drift, strict=False)
            out = _ramp.apply_warp(y, sr, wr.time_map, backend=chosen)
            dest = warp_dir / f"{sid}.warped.wav"
            sf.write(str(dest), np.asarray(out, dtype="float32"), sr)
            got = float(len(out)) / sr
        except Exception as exc:                            # noqa: BLE001
            # One bad cue must not cost the dub. Fall back to this cue's
            # steady tempo over the raw clip -- exactly what `naive_slot`
            # would have rendered.
            #
            # `tts_warped_audio` is POINTED AT THE RAW FILE rather than left
            # unset, and that is load-bearing: the mix runs with
            # `clip_source="warped"`, and `mix._resolve_clip` raises on a
            # segment that carries other clips but not the requested one --
            # which `_build_plan_pieces` turns into a SKIP. An unset field
            # here would drop the line from the dub entirely, silently, over
            # a stretcher hiccup. With the field set the mix reads the raw
            # clip and applies `local_tempo` in its own graph.
            n_failed += 1
            log.warning("warp: %s failed (%s: %s); falling back to a constant "
                        "%.3fx over the raw clip", sid, type(exc).__name__,
                        exc, cp_plan.tempo)
            seg["tts_warped_audio"] = str(info["clip"])
            sp.qc.update({"warp_failed": f"{type(exc).__name__}: {exc}",
                          "warp_fallback": "constant_tempo_over_raw",
                          "speech_tempo": round(cp_plan.tempo, 4)})
            sp.pieces = [Piece(src_span=Span(0.0, info["raw"]),
                               dst_start=cp_plan.out_start,
                               local_tempo=cp_plan.tempo, anchor="whole",
                               text=seg.get("final_text") or seg.get("mt_text") or "")]
            plan_obj.segments.append(sp)
            continue

        seg["tts_warped_audio"] = str(dest)
        if wr.over_drift:
            n_drift += 1
        sp.pieces = [Piece(src_span=Span(0.0, got), dst_start=cp_plan.out_start,
                           local_tempo=1.0, anchor="whole",
                           text=seg.get("final_text") or seg.get("mt_text") or "")]
        sp.qc.update({
            "producer": "warped",
            "speech_tempo": round(cp_plan.tempo, 4),
            "lead_tempo": round(cp_plan.lead_tempo, 4),
            "trail_tempo": round(cp_plan.trail_tempo, 4),
            "planned_slot_s": round(cp_plan.out_slot, 4),
            "rendered_s": round(got, 4),
            "onset_shift_s": round(cp_plan.out_start - sp.slot.start, 4),
            "vad_backend": info.get("vad"),
            "silent_spans": len(spans),
            "pause_s_before": round(sum(e - s for s, e in spans), 4),
            "pause_s_after": round(sum(a.kept_gaps), 4),
            "boundary_drift": round(wr.boundary_drift, 5),
            "warp_anchors": int(len(wr.src_sec)),
            "speech_untouched": abs(cp_plan.tempo - 1.0) < 1e-6,
        })
        if a.absorbed_s > 1e-4:
            sp.qc["pause_absorbed_s"] = round(a.absorbed_s, 4)
        if a.added_s > 1e-4:
            sp.qc["pause_added_s"] = round(a.added_s, 4)
        if cp_plan.flags:
            sp.qc["plan_flags"] = list(cp_plan.flags)
        plan_obj.segments.append(sp)

    # ---- verify the constraint on what was ACTUALLY rendered -------------
    tempos = [by_id[c.id].tempo for c in cues]
    steps = [abs(math.log(max(b, 1e-9)) - math.log(max(a_, 1e-9)))
             for a_, b in zip(tempos, tempos[1:])]
    step_max = float(np.exp(max(steps)) - 1.0) if steps else 0.0

    # `summarize` already derives speech_tempo_max/mean and
    # speech_untouched_segments from `qc["speech_tempo"]`, and its version is
    # the better one -- it includes cues that fell back to a constant tempo,
    # which `tempos` (the cue plans) does not.
    plan_obj.summary = summarize(plan_obj)
    plan_obj.summary.update({
        # The headline for this producer: the realised boundary step, against
        # the limit asked for, beside what the naive fit would have done.
        "tempo_step_max": round(step_max, 4),
        "tempo_step_limit": max_rel_step,
        "tempo_step_ok": bool(step_max <= max_rel_step + 1e-3),
        "naive_tempo_step_max": round(report["naive"]["max_rel_step"], 4),
        "max_onset_drift_s": round(report["planned"]["max_onset_drift_s"], 4),
        "max_island_dur_err_s": report["max_island_dur_err_s"],
        "n_islands": report["n_islands"],
        "plan_flags": report["flag_counts"],
        "warp_backend": chosen,
        "cues_over_drift": n_drift,
        "cues_failed": n_failed,
    })
    data["segments"] = segs                    # carry tts_warped_audio back
    if out_json:
        plan_obj.write(Path(out_json))
    # This producer REWRITES the merged JSON in place (it adds
    # `tts_warped_audio` per segment), so it owes a version -- CLAUDE.md §2,
    # and `tests/test_mt_versions.py::test_every_mutating_stage_snapshots`
    # greps for exactly this. The snapshot is taken of what is about to be
    # written, never of whatever is on disk afterwards.
    merged_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    try:
        from .mt_versions import snapshot as _mt_snapshot
        _mt_snapshot(merged_json, "warp", data=data)
    except Exception as exc:                                # noqa: BLE001
        log.warning("warp: version snapshot failed (%s); the dub is unaffected",
                    exc)
    log.info("warp: rendered %d/%d cues via %s; speech tempo max %.3f, "
             "boundary step %.3f (limit %.2f, naive %.3f), %d over-drift, %d failed",
             len(cues) - n_failed, len(cues), chosen,
             max(tempos) if tempos else 0.0, step_max, max_rel_step,
             report["naive"]["max_rel_step"], n_drift, n_failed)
    return plan_obj
