"""The `warped` timing producer plus the rate-ramp warp machinery it uses.

Two layers in one file:

* The low-level warp machinery (formerly `backend/rate_ramp.py`): one
  continuous rate curve per cue, applied via a time-map stretcher.
  `backend.tempo_plan` hands each cue a steady-state SPEECH rate plus the
  rates at its two joints. Applying one factor piecewise-constant STEPS the
  rate at every cue boundary, which is audible -- the just-noticeable
  difference is around 10% (Quene 2007) and a 1.2x -> 1.8x boundary is 50%.
  Here each cue's rate is a continuous curve: in at `lead_tempo`, eased to
  `tempo`, held, eased out to `trail_tempo`. The planner sets
  `lead_tempo[i+1] == trail_tempo[i]`, so the rate is continuous along the
  whole chain even though each cue is stretched as its own file.
  `gap_rates` also carries the pauses -- fast inside the engine's own
  silence, slow inside the speech:

      r(t)  3.0x        ---,        ---,              <- inside pauses
           1.05x  --\\______/_________/______/--      <- inside speech
                    ^ lead ramp        trail ramp ^

  That is `backend.pause_absorb` expressed as a rate rather than a cut,
  which keeps the cue ONE piece (so `mix.DEFAULT_MAX_PIECES` is not a
  ceiling), leaves no seams to click, and makes the realised duration exact
  by construction.

  Two facts about the duration correction `k`:
  1. No root find is needed. Scaling the envelope scales the realised
     duration by exactly `1/k`, since INT dv/(k r) == (1/k) INT dv/r, so
     `k = I / T`.
  2. Scaling by `k` scales the BOUNDARY rates too, so chain continuity
     breaks by `|k - 1|`. That is not bounded by 2% -- a short cue with
     wide ramps and a big lead/tempo ratio reaches 5-8% -- so it is
     measured and reported.

  Past `max_boundary_drift` this records the drift rather than raising;
  `strict=True` restores the raise. Real invariants raise rather than
  assert, because `python -O` strips asserts and `_integerize`'s
  monotonicity is what pyrubberband validates.

  Ported from `third_party/rate_ramp.py`; deliberate differences are marked
  PORT: in the code.

* `plan_warped`, the timing producer. This is the fourth way the repo can
  pace a dub, and the first that does not hand the mixer a constant tempo:

      naive_slot    one Piece per cue, one tempo over speech AND silence
      pause_aware   N Pieces anchored to the SOURCE onsets
      word_anchored / dp   the research arms
      warped        ONE Piece per cue, already stretched, local_tempo = 1.0

  The chain:

      VAD the clip          backend.pause_absorb.detect_pauses
        -> speech seconds, internal gaps, edge silence
      smooth across cues    backend.tempo_plan.plan_tempo
        -> islands; per cue a slot (out_start, out_end) whose boundary
           change is bounded by `max_rel_step`, with island spans
           conserved exactly
      split inside the cue  backend.pause_absorb.allocate
        -> silence to its floor before the speech moves; expands the
           silence when the cue has room, instead of drawling
      one curve             rate_envelope + build_warp (below)
        -> r(t): fast in the gaps, slow in the speech, smoothstep at joints
      render                apply_warp (below)
        -> `tts_warped_audio`, exactly `out_slot` long

  THIS PRODUCER RENDERS AUDIO. A `TimingPlan` is `(file, trim, delay,
  tempo)` and cannot spell a time-varying rate, so the plan it emits --
  one Piece, `local_tempo = 1.0` -- is only meaningful against files that
  already exist. Stage 9f2 is therefore minutes rather than milliseconds
  here, and the log says so per cue.

  TWO PASSES, NOT ONE. `plan_tempo` shares time BETWEEN cues; `allocate`
  splits one cue's slot between its speech and its silence. Running the
  allocator second, against the slot the planner fixed, keeps island
  conservation true by construction (`out_slot` is never touched again)
  while the realised speech rate is still chosen silence-first. The
  realised rate can then drift off the planner's, so the slew limit is
  VERIFIED on the realised tempos and reported as `tempo_step_max` rather
  than asserted.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from .tempo_plan import CuePlan, TempoConfig
from .timing import (DEFAULT_PAUSE_MIN_S, Piece, SegmentPlan, Span, TimingPlan,
                     _base_segment_plan, summarize)

log = logging.getLogger(__name__)

__all__ = [
    "WarpResult", "apply_warp", "available_backend", "build_warp",
    "ffmpeg_fallback_note", "plan_warped", "ramp_windows", "rate_envelope",
    "warp_map",
]

_SpanTuple = Tuple[float, float]
GapRate = Tuple[float, float, float]          # (start_s, end_s, rate)
_EPS = 1e-12

#: Width of the smoothstep in and out of a gap's own rate: short, since a gap
#: is short, but long enough that entering 3x is not itself a step.
DEFAULT_GAP_EDGE_S = 0.04


# --------------------------------------------------------------------------- #
# envelope
# --------------------------------------------------------------------------- #
def _smoothstep(u):
    """C1 Hermite step 3u^2 - 2u^3, zero slope at both ends.

    Zero slope keeps d(rate)/dt continuous where a ramp meets the plateau
    beside it; a linear ramp kinks there, and the kink reads as a click.
    """
    import numpy as np
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def ramp_windows(tts_duration: float, cfg: TempoConfig,
                 silence_spans: Sequence[_SpanTuple] | None = None):
    """`(a0, b0, a1, b1, w)`: the lead ramp [a0,b0] and trail ramp [a1,b1].

    Width is `cfg.ramp_s` clamped to 40% of the cue, which also keeps the two
    from overlapping. With `ramp_prefer_silence` each window slides toward
    the nearest silence within `ramp_s` of its edge -- a rate change inside a
    pause is far less audible than mid-phoneme.
    """
    import numpy as np

    d = float(max(tts_duration, 0.0))
    w = min(float(cfg.ramp_s), 0.4 * d)
    if w <= 0.0:
        return 0.0, 0.0, d, d, 0.0

    a0, a1 = 0.0, d - w
    if cfg.ramp_prefer_silence and silence_spans:
        spans = [(float(s), float(e)) for s, e in silence_spans if e > s]
        reach = float(cfg.ramp_s)

        lead = [sp for sp in spans if sp[0] <= reach and sp[1] > 0.0]
        if lead:
            s, e = min(lead, key=lambda sp: sp[0])
            want = 0.5 * (s + e) - 0.5 * w
            if e - s >= w:
                want = min(max(want, s), e - w)
            a0 = float(np.clip(want, 0.0, max(0.0, min(reach, d - 2.0 * w))))
        b0 = a0 + w

        trail = [sp for sp in spans if sp[1] >= d - reach and sp[0] < d]
        if trail:
            s, e = min(trail, key=lambda sp: d - sp[1])
            want = 0.5 * (s + e) - 0.5 * w
            if e - s >= w:
                want = min(max(want, s), e - w)
            a1 = float(np.clip(want, max(d - w - reach, b0), d - w))

    b0, b1 = a0 + w, a1 + w
    if not (0.0 <= a0 <= b0 <= a1 <= b1 <= d + 1e-9):
        raise ValueError(f"degenerate ramp windows {(a0, b0, a1, b1)} for d={d}")
    return a0, b0, a1, b1, w


def rate_envelope(
    plan: CuePlan,
    tts_duration: float,
    cfg: TempoConfig,
    silence_spans: Sequence[_SpanTuple] | None = None,
    gap_rates: Sequence[GapRate] | None = None,
    gap_edge_s: float = DEFAULT_GAP_EDGE_S,
) -> Callable:
    """Instantaneous rate `r(t)` on the SOURCE (unstretched) clip clock.

    Composed in LOG RATE: a rate factor is a ratio, so in log space 0.8x and
    1.25x are mirror images, a transition's midpoint is the geometric mean,
    `r` stays positive for any input, and `d(log r)/dt` is what the ear
    tracks.

    PORT: BASE + CORRECTION, not `np.where`. The base is the speech tempo
    overridden by each gap's own rate; the two boundary ramps are added on
    top as corrections decaying to zero. That way a ramp slid INTO a gap by
    `ramp_prefer_silence` scales the rate locally in force instead of
    overwriting it.
    """
    import numpy as np

    for name in ("tempo", "lead_tempo", "trail_tempo"):
        v = float(getattr(plan, name))
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"{plan.id}: {name} must be finite and > 0, got {v!r}")

    d = float(max(tts_duration, 0.0))
    a0, b0, a1, _b1, w = ramp_windows(d, cfg, silence_spans)
    lt = math.log(plan.tempo)
    d_lead = math.log(plan.lead_tempo) - lt
    d_trail = math.log(plan.trail_tempo) - lt

    def r(tau):
        t = np.clip(np.asarray(tau, dtype=float), 0.0, d)
        logr = np.full(t.shape, lt)

        for gs, ge, gr in (gap_rates or []):
            if ge <= gs or gr <= 0:
                continue
            edge = min(float(gap_edge_s), 0.5 * (ge - gs))
            dg = math.log(gr) - lt
            if edge <= 0:
                logr = np.where((t >= gs) & (t <= ge), lt + dg, logr)
                continue
            up = _smoothstep((t - gs) / edge)
            down = 1.0 - _smoothstep((t - (ge - edge)) / edge)
            inside = np.clip(np.minimum(up, down), 0.0, 1.0)
            inside = np.where((t >= gs - edge) & (t <= ge + edge), inside, 0.0)
            logr = logr + dg * inside

        if w > 0.0:
            logr = logr + d_lead * (1.0 - _smoothstep((t - a0) / w))
            logr = logr + d_trail * _smoothstep((t - a1) / w)
        return np.exp(logr)

    return r


# --------------------------------------------------------------------------- #
# warp map
# --------------------------------------------------------------------------- #
class WarpResult(NamedTuple):
    time_map: object                # (K,2) int64, columns (src_sample, dst_sample)
    src_sec: object
    tgt_sec: object
    k: float                        # envelope scale that hits the duration exactly
    realized_lead: float
    realized_trail: float
    boundary_drift: float           # |k - 1|: continuity error at the joints
    target_duration: float
    realized_duration: float
    max_anchor_error_s: float
    over_drift: bool = False        # PORT: drift exceeded the tolerance


def _solve_scale(integral: float, target: float) -> float:
    """`k` such that INT dv/(k r) == target. Closed form `k = I / T`."""
    if integral <= 0.0 or target <= 0.0:
        raise ValueError(f"degenerate duration solve: I={integral}, T={target}")
    return float(integral / target)


def _pick_anchors(tau, t, rate, n_anchors: int, tol_s: float):
    """Density-weighted seed where the map bends, then adaptive bisection."""
    import numpy as np

    n = tau.size
    n_anchors = int(np.clip(n_anchors, 2, n))
    curv = np.abs(np.gradient(1.0 / rate, tau))
    wt = 1.0 + curv / max(curv.mean(), _EPS)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (wt[1:] + wt[:-1]) * np.diff(tau))])
    if cdf[-1] <= 0:
        idx = np.linspace(0, n - 1, n_anchors)
    else:
        idx = np.interp(np.linspace(0.0, cdf[-1], n_anchors), cdf, np.arange(n))
    keep = np.unique(np.concatenate([[0], np.round(idx), [n - 1]])).astype(int)

    for _ in range(24):
        err = np.abs(np.interp(tau, tau[keep], t[keep]) - t)
        if err.max() <= tol_s:
            break
        add = []
        for lo, hi in zip(keep[:-1], keep[1:]):
            if hi - lo < 2:
                continue
            j = lo + 1 + int(np.argmax(err[lo + 1:hi]))
            if err[j] > tol_s:
                add.append(j)
        if not add:
            break
        keep = np.unique(np.concatenate([keep, add])).astype(int)
    return keep


def build_warp(
    plan: CuePlan,
    tts_duration: float,
    cfg: TempoConfig,
    sr: int,
    n_samples: int,
    silence_spans: Sequence[_SpanTuple] | None = None,
    gap_rates: Sequence[GapRate] | None = None,
    n_grid: int = 2048,
    n_anchors: int = 64,
    anchor_tol_s: float = 1e-3,
    max_boundary_drift: float = 0.02,
    strict: bool = False,
) -> WarpResult:
    """Integrate the envelope, rescale to hit `out_slot` exactly, emit anchors.

    Output time is `t_out(tau) = INT_0^tau dv / r(v)`: the rate lives on the
    source clock, so output time accumulates as its reciprocal.
    """
    import numpy as np
    from scipy.integrate import cumulative_trapezoid

    d = float(max(tts_duration, 0.0))
    n_samples = int(n_samples)
    target = float(plan.out_slot) if plan.out_slot > 0 else d / float(plan.tempo)
    if d <= 0.0 or n_samples <= 1 or target <= 0.0:
        n_t = max(1, int(round(target * sr)))
        tm = np.array([[0, 0], [max(1, n_samples), n_t]], dtype=np.int64)
        return WarpResult(tm, np.array([0.0, d]), np.array([0.0, target]), 1.0,
                          float(plan.lead_tempo), float(plan.trail_tempo), 0.0,
                          target, target, 0.0, False)

    tau = np.linspace(0.0, d, int(max(n_grid, 16)))
    rate = rate_envelope(plan, d, cfg, silence_spans, gap_rates)(tau)
    if not np.all(np.isfinite(rate)) or np.any(rate <= 0):
        raise ValueError(f"{plan.id}: envelope produced a non-positive or NaN rate")

    integral = float(cumulative_trapezoid(1.0 / rate, tau, initial=0.0)[-1])
    k = _solve_scale(integral, target)

    t = cumulative_trapezoid(1.0 / (k * rate), tau, initial=0.0)
    realized = float(t[-1])
    if not math.isfinite(realized) or realized <= 0:
        raise ValueError(f"{plan.id}: duration integral came back {realized!r}")
    t = t * (target / realized)                      # pin the endpoint exactly

    drift = abs(k - 1.0)
    over = drift > max_boundary_drift
    if over:
        msg = (f"{plan.id}: the duration correction scales the envelope by "
               f"k={k:.4f}, so the boundary rates drift {drift * 100:.2f}% "
               f"(> {max_boundary_drift * 100:.2f}%). Narrow cfg.ramp_s, widen "
               f"the slot, or raise max_boundary_drift.")
        if strict:
            raise ValueError(msg)
        # PORT: a warning, not a dead run. The cue renders at the right
        # LENGTH; only the joint-continuity promise is weaker.
        log.warning("rate_ramp: %s", msg)

    keep = _pick_anchors(tau, t, k * rate, n_anchors, anchor_tol_s)
    src_sec, tgt_sec = tau[keep], t[keep]
    max_err = float(np.abs(np.interp(tau, src_sec, tgt_sec) - t).max())

    n_target = int(round(target * sr))
    tm = _integerize(src_sec, tgt_sec, sr, n_samples, max(n_target, 1))
    return WarpResult(tm, src_sec, tgt_sec, k, k * plan.lead_tempo,
                      k * plan.trail_tempo, drift, target, float(t[-1]),
                      max_err, over)


def _integerize(src_sec, tgt_sec, sr: int, n_samples: int, n_target: int):
    """Round to samples, keeping BOTH columns strictly increasing.

    pyrubberband requires a time map that rises in both columns and ends at
    `(len(y), target_len)`, rejecting equal consecutive entries. Duplicates
    are dropped rather than nudged: nudging moves an anchor off the curve,
    dropping coarsens it by under a sample.
    """
    import numpy as np

    s = np.round(np.asarray(src_sec) * sr).astype(np.int64)
    g = np.round(np.asarray(tgt_sec) * sr).astype(np.int64)
    out = [(0, 0)]
    for a, b in zip(s[1:-1], g[1:-1]):
        if a > out[-1][0] and b > out[-1][1] and a < n_samples and b < n_target:
            out.append((int(a), int(b)))
    while out and (out[-1][0] >= n_samples or out[-1][1] >= n_target):
        out.pop()
    out.append((int(n_samples), int(n_target)))
    if len(out) < 2 or out[0] != (0, 0):
        out = [(0, 0), (int(n_samples), int(n_target))]
    tm = np.asarray(out, dtype=np.int64)
    # PORT: raise, not assert. `python -O` strips asserts and this is the
    # exact invariant the backend validates; failing here names the cue.
    if not (np.all(np.diff(tm[:, 0]) > 0) and np.all(np.diff(tm[:, 1]) > 0)):
        raise ValueError("time map is not strictly increasing in both columns")
    return tm


def warp_map(plan: CuePlan, tts_duration: float, cfg: TempoConfig, sr: int,
             n_samples: int, silence_spans=None, gap_rates=None, **kw):
    """Just the `(K,2)` map. Use `build_warp` when the drift matters."""
    return build_warp(plan, tts_duration, cfg, sr, n_samples,
                      silence_spans, gap_rates, **kw).time_map


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def _has_rubberband() -> bool:
    return (importlib.util.find_spec("pyrubberband") is not None
            and shutil.which("rubberband") is not None)


def available_backend() -> str | None:
    """Which time-map stretcher this host has, or None.

    Separate from `apply_warp` so a producer can PREFLIGHT. Discovering that
    nothing can render a warp after planning every cue is the expensive way
    to find out, and `backend/stretch.py` records that pyrubberband has been
    absent here before.
    """
    if _has_rubberband():
        return "rubberband"
    if importlib.util.find_spec("pytsmod") is not None:
        return "pytsmod"
    return None


def install_hint() -> str:
    return (
        "No time-map stretcher is installed. Either:\n"
        "  sudo apt-get install rubberband-cli && pip install pyrubberband\n"
        "      (preferred: phase vocoder with transient and formant handling;\n"
        "       note the ffmpeg `rubberband` FILTER in the imageio build is a\n"
        "       different thing and cannot take a time map)\n"
        "  pip install pytsmod\n"
        "      (pure-Python WSOLA, no system package, lower quality)"
    )


def apply_warp(y, sr: int, time_map: Iterable[Sequence[int]],
               backend: str = "auto"):
    """Resample `y` along `time_map`. Backend errors propagate untouched."""
    import numpy as np

    tm = np.asarray(time_map)
    if backend == "auto":
        backend = available_backend()
        if backend is None:
            raise RuntimeError(install_hint())
    if backend == "rubberband":
        import pyrubberband
        return pyrubberband.pyrb.timemap_stretch(
            y, sr, [(int(a), int(b)) for a, b in tm])
    if backend == "pytsmod":
        import pytsmod
        return pytsmod.wsola(y, tm.T.astype(float))
    raise ValueError(f"unknown backend {backend!r}")


def ffmpeg_fallback_note() -> str:
    """Why ffmpeg cannot stand in for a time map."""
    return (
        "ffmpeg's atempo filter takes a single constant factor per instance "
        "and does not accept an expression or a per-sample rate, so it cannot "
        "express a time-varying rate. Approximating a ramp would mean "
        "splitting the clip into many constant-atempo chunks and "
        "concatenating, which reintroduces exactly the rate steps the ramp "
        "exists to remove, plus a seam per chunk."
    )


# --------------------------------------------------------------------------- #
# producer: plan + render in one call
# --------------------------------------------------------------------------- #
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

    Edges count as ordinary gaps: the slot already decides where the cue
    starts, so the engine's run-up is not prosody worth protecting.
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
    from .tempo_plan import Cue, TempoConfig, plan_tempo, validate

    import numpy as np
    import soundfile as sf

    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    segs = list(data.get("segments", []))

    # Preflight before any cue is planned: discovering that nothing here can
    # render a time map after the whole file is planned is the expensive way
    # to find out.
    chosen = (available_backend() if warp_backend == "auto"
              else warp_backend)
    if chosen is None:
        raise RuntimeError(
            "timing producer 'warped' needs a time-map stretcher.\n"
            + install_hint())
    if chosen == "pytsmod":
        log.warning("warp: using pytsmod (WSOLA). rubberband gives better "
                    "quality on speech; see install_hint() in this module")

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
        # Both bounds apply, so the floor is the LARGER: a gap may not fall
        # below its tier floor, nor give up more than `absorb_max_frac` of
        # itself. `allocate` enforces the same pair.
        floors = [max(_absorb.pause_floor(e - s),
                      (e - s) * (1.0 - absorb_max_frac)) for s, e in spans]
        # `fixed_s` is the TIGHTEST the silence may be, giving the planner
        # the most room. `allocate` re-splits whatever slot comes back, so
        # being greedy here costs nothing.
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

    # lead/trail follow the REALISED rates, or the joints match numbers
    # nothing rendered.
    #
    # Neighbours are ISLAND-LOCAL. Global order would couple the last cue of
    # one island to the first of the next -- across a cut, a speaker change
    # or a second of silence, exactly where the planner refused to couple --
    # and the ramp would ease toward a rate the next cue does not have.
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
            wr = build_warp(
                cp_plan, info["raw"], cfg, sr, len(y),
                silence_spans=spans, gap_rates=gap_rates,
                max_boundary_drift=max_boundary_drift, strict=False)
            out = apply_warp(y, sr, wr.time_map, backend=chosen)
            dest = warp_dir / f"{sid}.warped.wav"
            sf.write(str(dest), np.asarray(out, dtype="float32"), sr)
            got = float(len(out)) / sr
        except Exception as exc:                            # noqa: BLE001
            # One bad cue must not cost the dub: fall back to this cue's
            # steady tempo over the raw clip, what `naive_slot` would render.
            #
            # Point `tts_warped_audio` AT THE RAW FILE rather than leaving it
            # unset. The mix runs with `clip_source="warped"`, and
            # `mix._resolve_clip` raises on a segment carrying other clips but
            # not the requested one, which `_build_plan_pieces` turns into a
            # SKIP -- so an unset field drops the line from the dub silently.
            # With it set, the mix reads the raw clip and applies
            # `local_tempo` in its own graph.
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

    # `summarize` derives speech_tempo_max/mean and
    # speech_untouched_segments from `qc["speech_tempo"]`, and includes cues
    # that fell back to a constant tempo, which `tempos` does not.
    plan_obj.summary = summarize(plan_obj)
    plan_obj.summary.update({
        # The headline: the realised boundary step, the limit asked for, and
        # what the naive fit would have done.
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
    # This producer rewrites the merged JSON in place (adding
    # `tts_warped_audio` per segment), so it owes a version.
    # Snapshot what is about to be written, not what lands on disk.
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
