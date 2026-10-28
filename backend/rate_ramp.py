"""Smooth speech-rate ramps and pause absorption, as one resampling time map.

`backend.tempo_plan` hands each cue a steady-state SPEECH rate plus the rates
at its two joints. Applying one factor piecewise-constant STEPS the rate at
every cue boundary, which is audible -- the just-noticeable difference is
around 10% (Quene 2007) and a 1.2x -> 1.8x boundary is 50%. Here each cue's
rate is a continuous curve: in at `lead_tempo`, eased to `tempo`, held, eased
out to `trail_tempo`. The planner sets `lead_tempo[i+1] == trail_tempo[i]`,
so the rate is continuous along the whole chain even though each cue is
stretched as its own file.

`gap_rates` also carries the pauses -- fast inside the engine's own silence,
slow inside the speech:

    r(t)  3.0x        ---,        ---,              <- inside pauses
         1.05x  --\______/_________/______/--       <- inside speech
                  ^ lead ramp        trail ramp ^

That is `backend.pause_absorb` expressed as a rate rather than a cut, which
keeps the cue ONE piece (so `mix.DEFAULT_MAX_PIECES` is not a ceiling),
leaves no seams to click, and makes the realised duration exact by
construction.

Two facts about the duration correction `k`:

1. No root find is needed. Scaling the envelope scales the realised duration
   by exactly `1/k`, since INT dv/(k r) == (1/k) INT dv/r, so `k = I / T`.
2. Scaling by `k` scales the BOUNDARY rates too, so chain continuity breaks
   by `|k - 1|`. That is not bounded by 2% -- a short cue with wide ramps and
   a big lead/tempo ratio reaches 5-8% -- so it is measured and reported.

Past `max_boundary_drift` this records the drift rather than raising;
`strict=True` restores the raise. Real invariants raise rather than assert,
because `python -O` strips asserts and `_integerize`'s monotonicity is what
pyrubberband validates.

Ported from `third_party/rate_ramp.py`; deliberate differences are marked
PORT: in the code.
"""
from __future__ import annotations

import importlib.util
import logging
import math
import shutil
from typing import Callable, Iterable, NamedTuple, Sequence

from .tempo_plan import CuePlan, TempoConfig

log = logging.getLogger(__name__)

__all__ = [
    "WarpResult", "apply_warp", "available_backend", "build_warp",
    "ffmpeg_fallback_note", "ramp_windows", "rate_envelope", "warp_map",
]

Span = tuple[float, float]
GapRate = tuple[float, float, float]          # (start_s, end_s, rate)
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
                 silence_spans: Sequence[Span] | None = None):
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
    silence_spans: Sequence[Span] | None = None,
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

        # Base: each gap at its own rate, eased in and out so entering 3x is
        # not itself a step.
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

        # Corrections: full at the cue edge, decayed to nothing by the end
        # of each ramp window.
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
    silence_spans: Sequence[Span] | None = None,
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
