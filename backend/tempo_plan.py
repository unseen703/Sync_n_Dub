"""Slew-limited speaking-rate planner over a sequence of cues.

A naive dub assigns each cue `tempo_i = tts_duration_i / slot_i`: every cue
fits and the chain lurches, 1.2x -> 1.9x -> 0.95x, each step several times
the ~10% just-noticeable difference for speech tempo (Quene 2007). This
solves one constrained problem per ISLAND of cues, picking tempos and pauses
that keep sync, bound the per-boundary change and keep onsets near their
original times.

Enforced vs preferred:

  HARD  no tempo outside [tempo_min, tempo_max] (relaxed to tempo_hard_max
        only on cues flagged `budget_violation`)
  HARD  adjacent cues in an island differ by at most `max_rel_step`,
        symmetric in the log domain -- dropped last-resort as `slew_relaxed`
  HARD  island total duration conserved; sync at island edges is sacred
  HARD  a `pinned` cue keeps its onset exactly
  SOFT  tempos near 1.0, pauses near the originals, onsets near the originals

ISLANDS ARE THE SYNC CONTRACT. Where a long pause, a speaker change or a
scene cut makes neighbour-coupling meaningless, the chain splits and each
side solves independently, so drift exists only INSIDE an island and only up
to `pause_borrow_s`.

Ported from `third_party/mixing methods..zip` (`cue_types.py` +
`tempo_plan.py`). Four deliberate differences, marked PORT: in the code --
the 0.30 `max_rel_step` default, `speech_s`/`fixed_s` making the optimiser's
variable the SPEECH rate rather than the blob rate, the single-cue
early return, and the end-pin adjusting the last tempo rather than only
`out_end`.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace

log = logging.getLogger(__name__)

__all__ = ["Cue", "CuePlan", "TempoConfig", "plan_tempo", "validate", "naive_plan"]

_ONSET_TAU = 0.25        # seconds; typical tolerance for a felt onset drift
_SOLVE_FTOL = 1e-7
_SOLVE_MAXITER = 200


# --------------------------------------------------------------------------- #
# types
# --------------------------------------------------------------------------- #
@dataclass
class Cue:
    """One synthesised dialogue segment, as the planner sees it."""

    id: str
    start: float                 # original slot start on the video timeline (s)
    end: float                   # original slot end (s)
    tts_duration: float          # natural-rate synthesised duration (s)
    pinned: bool = False         # on-screen: the onset must not move
    speaker: str | None = None
    scene: int = 0               # shot index; tempo never smooths across a change

    #: PORT: seconds of actual SPEECH in the clip (VAD), the optimiser's `d`.
    #: None means not measured and falls back to `tts_duration`.
    speech_s: float | None = None
    #: PORT: internal silence surviving absorption at its floor. Added to
    #: every duration term, never scaled by tempo -- silence has no rate.
    fixed_s: float = 0.0

    @property
    def slot(self) -> float:
        return self.end - self.start

    @property
    def rate_duration(self) -> float:
        """The part of the clip the tempo actually applies to."""
        return float(self.speech_s if self.speech_s is not None else self.tts_duration)

    @property
    def required_tempo(self) -> float:
        """Rate needed to fit the original slot with no redistribution."""
        room = max(self.slot - self.fixed_s, 1e-6)
        return self.rate_duration / room


@dataclass
class CuePlan:
    id: str
    tempo: float                 # steady-state SPEECH rate factor (>1 = faster)
    out_start: float
    out_end: float
    lead_tempo: float = 1.0      # rate at the joint with the previous cue
    trail_tempo: float = 1.0     # rate at the joint with the next cue
    fixed_s: float = 0.0         # internal silence included in out_slot
    flags: list[str] = field(default_factory=list)

    @property
    def out_slot(self) -> float:
        return self.out_end - self.out_start


@dataclass
class TempoConfig:
    # --- slew / smoothness ---
    max_rel_step: float = 0.30   # PORT: now 0.30 (was 0.20; see module docstring)
    use_log_slew: bool = True
    # --- absolute bounds ---
    tempo_min: float = 0.80
    tempo_max: float = 1.45
    tempo_hard_max: float = 1.80  # beyond this -> flag; shorten the text upstream
    tempo_target: float = 1.0
    # --- pause borrowing (the Virkar et al. boundary relaxation) ---
    pause_borrow_s: float = 0.30
    pause_min_s: float = 0.08
    # --- island segmentation ---
    island_gap_s: float = 1.00
    # False lets a speaker change still share an island, so the planner has
    # a neighbour to ramp into. True produces many single-cue islands with no
    # shape, which is the failure this knob exists to avoid.
    break_on_speaker: bool = False
    break_on_scene: bool = True
    #: PORT: cost is ~O(n^2) per SLSQP iteration, since the onset loop is
    #: numerically differentiated over 2n-1 variables. Islands are usually
    #: small, but pyannote under-clustering makes them bigger, so this is a
    #: hard split.
    max_island_cues: int = 40
    # --- objective weights ---
    w_rate: float = 1.0
    w_onset: float = 4.0
    w_pause: float = 0.5
    # --- ramping (consumed by backend.warp_plan) ---
    ramp_s: float = 0.18
    ramp_prefer_silence: bool = True


# --------------------------------------------------------------------------- #
# island segmentation
# --------------------------------------------------------------------------- #
def _islands(cues: list[Cue], cfg: TempoConfig) -> list[list[int]]:
    """Indices grouped into chains that share a tempo-smoothing constraint."""
    if not cues:
        return []
    islands: list[list[int]] = [[0]]
    for i in range(1, len(cues)):
        a, b = cues[i - 1], cues[i]
        split = (
            (b.start - a.end) >= cfg.island_gap_s
            or (cfg.break_on_speaker and a.speaker != b.speaker)
            or (cfg.break_on_scene and a.scene != b.scene)
            or len(islands[-1]) >= max(1, cfg.max_island_cues)
        )
        islands.append([i]) if split else islands[-1].append(i)
    return islands


# --------------------------------------------------------------------------- #
# naive baseline (seeds the solver, and is the comparison arm)
# --------------------------------------------------------------------------- #
def naive_plan(cues: list[Cue], cfg: TempoConfig) -> list[CuePlan]:
    """Independent per-cue fit: tempo = rate_duration / room, slot unchanged."""
    import numpy as np

    out: list[CuePlan] = []
    for c in cues:
        t = c.required_tempo
        t_clip = float(np.clip(t, cfg.tempo_min, cfg.tempo_hard_max))
        out.append(CuePlan(
            id=c.id, tempo=t_clip, out_start=c.start, out_end=c.end,
            lead_tempo=t_clip, trail_tempo=t_clip, fixed_s=c.fixed_s,
            flags=["naive"] + (["budget_violation"] if t > cfg.tempo_max else []),
        ))
    return out


# --------------------------------------------------------------------------- #
# per-island solve
# --------------------------------------------------------------------------- #
def _solve_island(cues, cfg, tempo_max_vec, slew_cap):
    """Return (x = log tempos, p = pauses, success), or None on hard failure."""
    import numpy as np
    from scipy.optimize import minimize

    n = len(cues)
    T = cues[-1].end - cues[0].start                          # island span (sacred)
    d = np.array([c.rate_duration for c in cues], dtype=float)
    fx = np.array([c.fixed_s for c in cues], dtype=float)     # PORT
    a = np.array([c.start - cues[0].start for c in cues])
    g = np.array([cues[i + 1].start - cues[i].end for i in range(n - 1)])
    pinned = np.array([c.pinned for c in cues], dtype=bool)
    pinned[0] = True                                          # island edge

    log_tmin = math.log(cfg.tempo_min)
    log_tmax = np.log(tempo_max_vec)
    log_ttarget = math.log(cfg.tempo_target)
    pause_hi = g + cfg.pause_borrow_s
    pause_lo = np.minimum(np.maximum(cfg.pause_min_s, g - cfg.pause_borrow_s),
                          pause_hi)

    # PORT (3): a one-cue island has no neighbour and no choice to make;
    # solving it costs an SLSQP run whose answer is discarded.
    if n == 1:
        room = max(cues[0].slot - cues[0].fixed_s, 1e-6)
        t = float(np.clip(d[0] / room, cfg.tempo_min, tempo_max_vec[0]))
        return np.array([math.log(t)]), np.array([]), True

    mean_d = max(d.mean(), 1e-6)
    w_d = d / mean_d                                          # content weighting

    def unpack(v):
        return v[:n], v[n:]

    def onsets(x, p):
        durs = d * np.exp(-x) + fx                            # PORT (2)
        o = np.zeros(n)
        for i in range(1, n):
            o[i] = o[i - 1] + durs[i - 1] + p[i - 1]
        return o

    def objective(v):
        x, p = unpack(v)
        o = onsets(x, p)
        rate = float(np.sum(w_d * (x - log_ttarget) ** 2))
        onset = float(np.sum((o - a) ** 2) / _ONSET_TAU ** 2) * cfg.w_onset
        pause = (float(np.sum((p - g) ** 2))
                 / max(cfg.pause_borrow_s ** 2, 1e-6)) * cfg.w_pause
        return cfg.w_rate * rate + onset + pause

    def pin_resid(v):
        x, p = unpack(v)
        o = onsets(x, p)
        return np.asarray([o[i] - a[i] for i in range(n) if pinned[i] and i > 0])

    def total_resid(v):
        x, p = unpack(v)
        return float(np.sum(d * np.exp(-x) + fx) + np.sum(p) - T)

    constraints = []
    if any(pinned[1:]):
        constraints.append({"type": "eq", "fun": pin_resid})
    constraints.append({"type": "eq", "fun": total_resid})
    if slew_cap is not None:
        def slew(v):
            x, _ = unpack(v)
            diffs = np.diff(x)
            return np.concatenate([slew_cap - diffs, slew_cap + diffs])
        constraints.append({"type": "ineq", "fun": slew})

    bounds = ([(log_tmin, float(log_tmax[i])) for i in range(n)]
              + [(float(pause_lo[j]), float(pause_hi[j])) for j in range(n - 1)])

    # feasibility: can the chain fit with every cue at its max and pauses tight?
    if float(np.sum(d / np.exp(log_tmax)) + np.sum(fx) + np.sum(pause_lo)) > T + 1e-6:
        return None

    x0 = np.clip(np.log(np.maximum(
        [c.required_tempo for c in cues], 1e-6)), log_tmin, log_tmax)
    p0 = np.clip(g, pause_lo, pause_hi)
    tot = float(np.sum(d * np.exp(-x0) + fx) + np.sum(p0))
    if tot < T:
        p0 = np.clip(p0 + (T - tot) / (n - 1), pause_lo, pause_hi)
    elif tot > T:
        slack = float(np.sum(p0 - pause_lo))
        take = min(tot - T, slack)
        if slack > 1e-9 and take > 0:
            p0 = p0 - (p0 - pause_lo) * (take / slack)

    try:
        res = minimize(objective, np.concatenate([x0, p0]), method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": _SOLVE_FTOL, "maxiter": _SOLVE_MAXITER})
    except Exception as exc:                                   # noqa: BLE001
        log.debug("tempo_plan: SLSQP raised %s: %s", type(exc).__name__, exc)
        return None
    x, p = unpack(res.x)
    return x, p, bool(res.success)


# --------------------------------------------------------------------------- #
# deterministic fallback -- never fails to return something valid
# --------------------------------------------------------------------------- #
def _fallback(cues, cfg, tempo_max_vec, slew_cap):
    """Clip + forward/backward slew sweep + residue absorbed into pauses.

    Always returns (x, p) inside the tempo bounds and conserving the island
    span. Where pauses would have to exceed `pause_hi` the extra is accepted
    as surplus silence after the last cue: uglier than borrowing, but the
    island edge is sacred and this is the fallback, not the planner.
    """
    import numpy as np

    n = len(cues)
    d = np.array([c.rate_duration for c in cues], dtype=float)
    fx = np.array([c.fixed_s for c in cues], dtype=float)
    g = (np.array([cues[i + 1].start - cues[i].end for i in range(n - 1)])
         if n > 1 else np.array([]))
    T = cues[-1].end - cues[0].start

    log_tmin, log_tmax = math.log(cfg.tempo_min), np.log(tempo_max_vec)
    pause_hi = g + cfg.pause_borrow_s if n > 1 else g
    pause_lo = (np.minimum(np.maximum(cfg.pause_min_s, g - cfg.pause_borrow_s),
                           pause_hi) if n > 1 else g)

    x = np.clip(np.log(np.clip([c.required_tempo for c in cues], 1e-6, None)),
                log_tmin, log_tmax)
    if slew_cap is not None and n > 1:
        for _ in range(50):
            prev = x.copy()
            for i in range(1, n):
                x[i] = min(max(x[i], x[i - 1] - slew_cap), x[i - 1] + slew_cap)
            for i in range(n - 2, -1, -1):
                x[i] = min(max(x[i], x[i + 1] - slew_cap), x[i + 1] + slew_cap)
            x = np.clip(x, log_tmin, log_tmax)
            if np.allclose(prev, x, atol=1e-9):
                break

    p = np.clip(g, pause_lo, pause_hi) if n > 1 else np.array([])
    deficit = T - float(np.sum(d * np.exp(-x) + fx) + np.sum(p))
    if n > 1 and deficit > 1e-9:
        room = pause_hi - p
        r = float(np.sum(room))
        if r > 0:
            add = min(deficit, r)
            p = p + room * (add / r)
            deficit -= add
        if deficit > 1e-9:
            p[-1] += deficit                      # surplus silence; span preserved
    elif n > 1 and deficit < -1e-9:
        room = p - pause_lo
        r = float(np.sum(room))
        if r > 0:
            take = min(-deficit, r)
            p = p - room * (take / r)
            deficit += take
        if deficit < -1e-9:
            room = log_tmax - x
            headroom = float(np.sum(d * np.exp(-x) * room))
            if headroom > 0:
                x = np.minimum(x + (-deficit) / headroom, log_tmax)
    return x, p


# --------------------------------------------------------------------------- #
# island -> CuePlan
# --------------------------------------------------------------------------- #
def _plans_from_solution(cues, cfg, x, p, flags_per_cue):
    import numpy as np

    n = len(cues)
    d = np.array([c.rate_duration for c in cues], dtype=float)
    fx = np.array([c.fixed_s for c in cues], dtype=float)
    tempos = np.exp(x)

    # PORT (4): pin the island END by adjusting the LAST CUE'S TEMPO, so
    # `out_slot == rate_duration/tempo + fixed_s` keeps holding. Moving
    # `out_end` alone leaves the plan claiming a length the renderer will
    # not produce.
    durs = d / tempos + fx
    span = float(np.sum(durs) + np.sum(p))
    T = cues[-1].end - cues[0].start
    err = span - T
    if abs(err) > 1e-9 and durs[-1] - fx[-1] > 1e-6:
        want = max(1e-6, durs[-1] - err - fx[-1])
        tempos[-1] = float(np.clip(d[-1] / want, cfg.tempo_min, cfg.tempo_hard_max))
        durs[-1] = d[-1] / tempos[-1] + fx[-1]

    plans: list[CuePlan] = []
    cursor = cues[0].start
    for i in range(n):
        lead = tempos[i] if i == 0 else math.sqrt(tempos[i] * tempos[i - 1])
        trail = tempos[i] if i == n - 1 else math.sqrt(tempos[i] * tempos[i + 1])
        plans.append(CuePlan(
            id=cues[i].id, tempo=float(tempos[i]),
            out_start=float(cursor), out_end=float(cursor + durs[i]),
            lead_tempo=float(lead), trail_tempo=float(trail),
            fixed_s=float(fx[i]), flags=list(flags_per_cue[i]),
        ))
        cursor = plans[-1].out_end + (float(p[i]) if i < n - 1 else 0.0)
    return plans


def _solve_one_island(cues: list[Cue], cfg: TempoConfig) -> list[CuePlan]:
    import numpy as np

    n = len(cues)
    if n == 0:
        return []
    slew = math.log1p(cfg.max_rel_step) if cfg.use_log_slew else cfg.max_rel_step
    tempo_max_vec = np.full(n, cfg.tempo_max, dtype=float)
    flags: list[list[str]] = [[] for _ in range(n)]

    sol = _solve_island(cues, cfg, tempo_max_vec, slew)
    if sol is None:                                   # let the hot cues run hotter
        req = np.array([c.required_tempo for c in cues])
        hot = req > cfg.tempo_max
        tempo_max_vec = np.where(hot, cfg.tempo_hard_max, cfg.tempo_max)
        for i in np.where(hot)[0]:
            flags[i].append("budget_violation")
        sol = _solve_island(cues, cfg, tempo_max_vec, slew)
    if sol is None:                                   # drop the slew limit
        for f in flags:
            f.append("slew_relaxed")
        sol = _solve_island(cues, cfg, tempo_max_vec, None)
    if sol is None or not sol[2]:
        x, p = _fallback(cues, cfg, tempo_max_vec, slew)
        for f in flags:
            if "slew_relaxed" not in f:
                f.append("slew_relaxed")
            f.append("fallback")
    else:
        x, p, _ = sol

    x = np.clip(x, math.log(cfg.tempo_min), np.log(tempo_max_vec))
    plans = _plans_from_solution(cues, cfg, x, p, flags)

    # Verify rather than assume: the end-pin above can nudge the last tempo
    # out of the slew band.
    if n >= 2:
        for i in range(1, n):
            step = abs(math.log(plans[i].tempo) - math.log(plans[i - 1].tempo))
            if step > slew + 1e-6 and "slew_relaxed" not in plans[i].flags:
                plans[i].flags.append("slew_relaxed")
    return plans


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def plan_tempo(cues: list[Cue], cfg: TempoConfig) -> list[CuePlan]:
    """Plan speaking rates across `cues`. One solve per island."""
    if not cues:
        return []
    out: list[CuePlan] = []
    for group in _islands(cues, cfg):
        out.extend(_solve_one_island([cues[i] for i in group], cfg))
    return out


def validate(cues: list[Cue], plans: list[CuePlan], cfg: TempoConfig) -> dict:
    """Numerical summary, with the naive baseline alongside for comparison."""
    import numpy as np

    naive = naive_plan(cues, cfg)
    log_slew = math.log1p(cfg.max_rel_step)

    def stats(ps):
        if not ps:
            return {"n": 0}
        t = np.array([p.tempo for p in ps])
        steps = np.abs(np.diff(np.log(np.maximum(t, 1e-9))))
        drift = np.array([abs(p.out_start - c.start) for p, c in zip(ps, cues)])
        return {
            "n": len(ps),
            "max_tempo": float(t.max()), "min_tempo": float(t.min()),
            "mean_abs_log_tempo": float(np.mean(np.abs(np.log(t)))),
            "max_rel_step": float(np.exp(steps.max()) - 1) if steps.size else 0.0,
            "slew_ok": bool(steps.size == 0 or steps.max() <= log_slew + 1e-6),
            "max_onset_drift_s": float(drift.max()),
            "mean_onset_drift_s": float(drift.mean()),
        }

    island_err = []
    for group in _islands(cues, cfg):
        island_err.append(abs((cues[group[-1]].end - cues[group[0]].start)
                              - (plans[group[-1]].out_end - plans[group[0]].out_start)))
    counts: dict[str, int] = {}
    for p in plans:
        for f in p.flags:
            counts[f] = counts.get(f, 0) + 1
    return {
        "planned": stats(plans), "naive": stats(naive),
        "max_island_dur_err_s": float(max(island_err)) if island_err else 0.0,
        "n_islands": len(island_err), "flag_counts": counts,
    }
