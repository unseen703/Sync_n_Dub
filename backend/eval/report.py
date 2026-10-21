"""Aggregate `*.timing_eval.json` results into a markdown report.

Three decisions are baked in here, and all three are about not over-claiming.

**Bootstrap over clips, not segments.** mix.py places clips with a
shift-forward cursor, so segment *k*'s start depends on every segment before
it inside the same clip. Resampling segments would treat those as independent
draws and return confidence intervals several times tighter than the evidence
supports. The resampling unit is therefore the clip, and what is resampled is
the *paired* difference against a named baseline arm.

**Noise floors are printed, and ties are called ties.** Every metric carries a
floor below which this corpus cannot resolve a difference. A difference under
its floor renders as `tie` — never as a rank.

**MT metrics live in their own 2-arm table.** Only the MT stage changes the
text, so chrF++ across eight timing arms would be eight copies of one number
dressed up as a comparison.
"""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

DEFAULT_RESULT_ROOT = Path("data/eval")
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20251007

# Smallest difference this corpus can resolve, per metric. Printed in the
# header of every report so a reader never has to go looking for them.
NOISE_FLOORS: Dict[str, float] = {
    "chrf_pp": 1.5,                 # chrF++ points
    "rt_wer": 2.0,                  # round-trip WER points (percent)
    "pause_f1": 0.06,
    "speech_iou": 0.05,
    "median_abs_start_ms": 80.0,    # milliseconds -- bootstrap only; the GT
                                    # floor (185 ms) dominates it, see
                                    # resolved_floor()
    "p95_abs_start_ms": 80.0,
    "mean_signed_start_ms": 80.0,
    "term_retention": 0.04,
}


# A median over one or two points is not a median. When an arm places its
# pieces back to back -- which `shift_forward` does by construction -- the
# hypothesis merges into ONE speech span, the IoU matcher pairs that span with
# whichever reference run it overlaps most, and `median_abs_start_ms` becomes
# the distance to that arbitrary run. Measured on the tier-A sweep: 42 of 110
# cells collapsed to a single span, and their median |Δstart| was 21,322 ms
# against 2,953 ms for the multi-span cells. Those five-figure numbers are an
# artefact of the collapse, not a placement error anyone can act on.
#
# So offset statistics require at least three matched pairs. The collapse
# itself is not thrown away -- it is reported by `speech_run_recall` and
# `n_hyp_runs`, where it belongs and where it is genuinely discriminating.
MIN_MATCHED_FOR_OFFSET = 3


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    group: str
    paths: Tuple[str, ...]          # candidate dotted paths; first hit wins
    unit: str = ""
    scale: float = 1.0
    lower_is_better: bool = True
    digits: int = 3
    is_mt: bool = False
    #: Minimum `timing.offset_stats.n_matched` before this cell is allowed to
    #: carry a number. See MIN_MATCHED_FOR_OFFSET.
    min_matched: int = 0


METRICS: Tuple[MetricSpec, ...] = (
    # `offset_stats` nests per boundary -- `dstart` / `dend` (backend/eval/timing.py
    # `_offset_block`) -- and both are ALREADY in milliseconds, so scale stays 1.0.
    # The flat `...offset_stats.median_abs_ms` path this used to name has never
    # existed. It fell through to `timing._harness.placement_shift_s.median`, which
    # is displacement against the clip's OWN SOURCE grid rather than against the
    # human dub -- a different quantity, in seconds, printed as if it were this one.
    # Measured on the dress-2 run it collapsed seven distinct arms onto two values
    # (m3 and m4 both printed 80.3 ms while differing by 125 ms in truth). That
    # fallback is deliberately NOT retained: a wrong number ranks the arms, an
    # empty cell with a reason does not.
    MetricSpec("median_abs_start_ms", "median |Δstart|", "timing",
               ("timing.offset_stats.dstart.median_abs_ms",),
               unit="ms", scale=1.0, digits=0,
               min_matched=MIN_MATCHED_FOR_OFFSET),
    MetricSpec("p95_abs_start_ms", "p95 |Δstart|", "timing",
               ("timing.offset_stats.dstart.p95_abs_ms",),
               unit="ms", scale=1.0, digits=0,
               min_matched=MIN_MATCHED_FOR_OFFSET),
    MetricSpec("mean_signed_start_ms", "signed mean Δstart", "timing",
               ("timing.offset_stats.dstart.mean_signed_ms",),
               unit="ms", scale=1.0, digits=0,
               min_matched=MIN_MATCHED_FOR_OFFSET),
    # Structure, not placement. These are what the collapsed cells actually
    # have to say, and they need no matched-pair gate because they ARE the
    # match counts. `speech_run_recall` answers "what fraction of the human
    # dub's speech runs did this arm put anything against"; `n_hyp_runs`
    # answers "how many distinct runs did it produce at all" -- an arm that
    # emits one 60-second blob scores 1 here however well it fits the slot,
    # which is exactly the user's constraint made measurable.
    MetricSpec("speech_run_recall", "speech-run recall", "timing",
               ("timing.offset_stats.match_recall",),
               lower_is_better=False, digits=3),
    MetricSpec("n_hyp_runs", "hyp speech runs", "timing",
               ("timing.offset_stats.n_hyp",),
               lower_is_better=False, digits=0),
    MetricSpec("speech_iou", "speech IoU", "timing",
               ("timing.speech_iou.iou", "timing.speech_iou"),
               lower_is_better=False),
    MetricSpec("drift_pct", "duration drift", "timing",
               ("timing._harness.mix_replay.drift_pct",), unit="%", digits=2),
    MetricSpec("pause_f1", "pause F1", "pauses",
               ("pauses.pause_f1.f1", "pauses.pause_f1"),
               lower_is_better=False),
    MetricSpec("chars_per_s", "chars/s (median)", "rate",
               ("rate.rate_stats.chars_per_s_median",
                "rate._harness.chars_per_s.median"), digits=2),
    MetricSpec("overflow_ratio", "TTS overflow", "rate",
               ("rate._harness.overflow_ratio",), unit="x", digits=3),
    MetricSpec("rt_wer", "round-trip WER", "intel",
               ("intel.roundtrip_wer.wer", "intel.roundtrip_wer"), digits=3),
    MetricSpec("chrf_pp", "chrF++", "mt",
               ("mt.chrfpp_report.chrf_pp", "mt.chrfpp_report.score"),
               lower_is_better=False, digits=2, is_mt=True),
    MetricSpec("term_retention", "term retention", "mt",
               ("mt.term_retention.retention", "mt.term_retention.rate"),
               lower_is_better=False, is_mt=True),
    MetricSpec("length_ratio", "length ratio", "mt",
               ("mt.length_ratio.ratio", "mt.length_ratio"),
               digits=3, is_mt=True),
)

# Metrics shown in the main (many-arm) placement table.
_PLACEMENT_KEYS = ("median_abs_start_ms", "p95_abs_start_ms",
                   "mean_signed_start_ms", "n_hyp_runs", "speech_run_recall",
                   "speech_iou", "pause_f1",
                   "drift_pct", "chars_per_s", "overflow_ratio", "rt_wer")
_MT_KEYS = tuple(m.key for m in METRICS if m.is_mt)

_SPEC_BY_KEY = {m.key: m for m in METRICS}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(
    root: Path = DEFAULT_RESULT_ROOT,
    *,
    clip_ids: Optional[Sequence[str]] = None,
    variant_ids: Optional[Sequence[str]] = None,
) -> List[dict]:
    """Read every `*.timing_eval.json` under `root`, newest schema first."""
    out: List[dict] = []
    for path in sorted(Path(root).rglob("*.timing_eval.json")):
        try:
            obj = json.loads(path.read_text())
        except Exception as e:
            log.warning("report: %s unreadable (%s); skipped", path, e)
            continue
        if clip_ids and obj.get("clip_id") not in set(clip_ids):
            continue
        if variant_ids and obj.get("variant_id") not in set(variant_ids):
            continue
        obj["_path"] = str(path)
        out.append(obj)
    return out


def _dig(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _n_matched(result: dict) -> Optional[int]:
    v = _dig(result, "timing.offset_stats.n_matched")
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def metric_value(result: dict, spec: MetricSpec) -> Optional[float]:
    if spec.min_matched:
        nm = _n_matched(result)
        if nm is None or nm < spec.min_matched:
            return None
    for path in spec.paths:
        val = _dig(result, path)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)) and not math.isnan(float(val)):
            return float(val) * spec.scale
    return None


def metric_reason(result: dict, spec: MetricSpec) -> str:
    """Why a cell is empty, taken from the result's own ledger."""
    if spec.min_matched:
        nm = _n_matched(result)
        if nm is None or nm < spec.min_matched:
            nr = _dig(result, "timing.offset_stats.n_ref")
            nh = _dig(result, "timing.offset_stats.n_hyp")
            return (f"only {nm if nm is not None else 0} matched pair(s) "
                    f"({nh} hyp run(s) vs {nr} ref run(s)); a median needs "
                    f"{spec.min_matched} — see speech-run recall")
    ledger = {e.get("key"): e.get("reason") for e in result.get("not_measured", [])}
    for path in spec.paths:
        parts = path.split(".")
        for depth in range(len(parts), 0, -1):
            key = ".".join(parts[:depth])
            if key in ledger:
                return str(ledger[key])
    if spec.group not in (result.get("groups_requested") or []):
        return f"group '{spec.group}' not requested"
    return "not measured"


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_paired(
    pairs: Sequence[Tuple[float, float]],
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Median and 95% CI of the paired difference (arm − baseline).

    `pairs` is one (baseline, arm) tuple per CLIP. Clips are the resampling
    unit: see the module docstring for why segments are not.
    """
    diffs = [float(a) - float(b) for b, a in pairs]
    n = len(diffs)
    if n == 0:
        return {"n_clips": 0, "median": None, "ci_lo": None, "ci_hi": None,
                "draws": draws, "reason": "no clip has both arms"}
    if n == 1:
        return {"n_clips": 1, "median": diffs[0], "ci_lo": None, "ci_hi": None,
                "draws": draws,
                "reason": "one paired clip: a CI needs at least two"}

    rng = random.Random(seed)
    meds: List[float] = []
    for _ in range(draws):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        sample.sort()
        mid = n // 2
        meds.append(sample[mid] if n % 2 else 0.5 * (sample[mid - 1] + sample[mid]))
    meds.sort()
    lo = meds[int(0.025 * (draws - 1))]
    hi = meds[int(0.975 * (draws - 1))]
    s = sorted(diffs)
    mid = n // 2
    point = s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])
    return {"n_clips": n, "median": point, "ci_lo": lo, "ci_hi": hi, "draws": draws}


# Offset metrics in milliseconds. The ground-truth floor binds on all of them:
# it is a property of the reference, not of which statistic is taken over it.
_MS_OFFSET_KEYS = ("median_abs_start_ms", "p95_abs_start_ms",
                   "mean_signed_start_ms")


def _corpus_floor_ms(results: Sequence[dict]) -> Optional[float]:
    """The corpus-wide GT floor the harness stamped into every result."""
    for r in results:
        for path in ("calibration.gt_noise_floor_ms",
                     "timing._harness.reference.gt_noise_floor_ms"):
            v = _dig(r, path)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return None


def _corpus_floor_basis(results: Sequence[dict]) -> str:
    for r in results:
        v = _dig(r, "calibration.gt_noise_floor_basis")
        if isinstance(v, str) and v:
            return v
    return "basis not recorded by the harness"


def resolved_floor(key: str, results: Sequence[dict]) -> Tuple[float, str]:
    """max(declared bootstrap floor, ground-truth floor) for a metric.

    The GT floor only binds on timing metrics expressed in milliseconds: it is
    the median disagreement between a clip's two references, and no method can
    be credited with beating it.
    """
    base = NOISE_FLOORS.get(key)
    if base is None:
        return (0.0, "no floor declared")
    if key not in _MS_OFFSET_KEYS:
        return (base, "bootstrap floor")
    gts = [v for v in (_dig(r, "timing._harness.reference.agreement_median_ms")
                       for r in results) if isinstance(v, (int, float))]
    if not gts:
        # Not "nobody measured it" -- every clip measured it and recorded that
        # the Hindi side CANNOT be measured (one word stream, no second
        # reference). The corpus-wide cue-level bound stands in, and it is
        # larger than the bootstrap floor, so it is what binds.
        corpus = _corpus_floor_ms(results)
        if corpus is None:
            return (base, "bootstrap floor; no per-clip GT floor was "
                          "measurable and no corpus-wide floor was recorded")
        why = _corpus_floor_basis(results)
        if corpus > base:
            return (corpus, f"GT floor {corpus:g} ms (corpus-wide) exceeds the "
                            f"{base:g} ms bootstrap floor — {why}")
        return (base, f"bootstrap floor; the corpus-wide GT floor "
                      f"({corpus:g} ms) is smaller — {why}")
    gts.sort()
    gt = gts[len(gts) // 2]
    if gt > base:
        return (float(gt), f"GT floor (median agreement_median_ms over "
                           f"{len(gts)} clip(s)) exceeds the {base:g} ms "
                           f"bootstrap floor")
    return (base, f"bootstrap floor (GT floor {gt:g} ms is smaller)")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _fmt(val: Optional[float], spec: MetricSpec) -> str:
    if val is None:
        return "—"
    return f"{val:.{spec.digits}f}{spec.unit}"


def _cell(result: dict, spec: MetricSpec) -> str:
    val = metric_value(result, spec)
    if val is None:
        return f"n/a ({metric_reason(result, spec)})"
    return _fmt(val, spec)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def _short(sha: Optional[str]) -> str:
    return (sha[:12] if sha else "n/a (not hashed)")


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def render_report(
    results: Sequence[dict],
    *,
    baseline: Optional[str] = None,
    mt_arms: Optional[Sequence[str]] = None,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> str:
    results = list(results)
    if not results:
        return ("# Timing-alignment evaluation\n\n"
                "No `*.timing_eval.json` results found. Run "
                "`python -m backend.eval sweep` first.\n")

    arms = sorted({r.get("variant_id", "?") for r in results})
    clips = sorted({r.get("clip_id", "?") for r in results})
    by: Dict[Tuple[str, str], dict] = {
        (r.get("clip_id"), r.get("variant_id")): r for r in results}

    gen = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L: List[str] = []
    L.append("# Timing-alignment evaluation — measured results\n")
    L.append(f"{len(results)} scored result(s): {len(arms)} arm(s) over "
             f"{len(clips)} clip(s). Generated {gen} by "
             f"`python -m backend.eval report`.\n")

    # --- noise floors --------------------------------------------------
    L.append("## 0. Noise floors — read these before ranking anything\n")
    L.append("A difference smaller than its floor is printed **tie**. It is not "
             "a rank, and it must not be quoted as one.\n")
    floor_rows = []
    for key in ("median_abs_start_ms", "speech_iou", "pause_f1",
                "chrf_pp", "rt_wer", "term_retention"):
        f, why = resolved_floor(key, results)
        spec = _SPEC_BY_KEY.get(key)
        unit = spec.unit if spec else ""
        floor_rows.append([f"`{key}`", f"{f:g}{unit or ''}", why])
    L.append(_table(["Metric", "Resolution", "Where the floor comes from"], floor_rows))
    L.append("Confidence intervals are bootstrapped over **clips** "
             f"({draws:,} draws, seed {seed}), never over segments: mix.py's "
             "shift-forward cursor couples the segments inside a clip, so a "
             "segment-level bootstrap returns fake-tight intervals.\n")

    # --- 1. runs scored --------------------------------------------------
    L.append("## 1. Runs scored\n")
    rows = []
    for r in results:
        rows.append([
            f"`{r.get('clip_id')}`", f"`{r.get('variant_id')}`",
            f"`{r.get('timing_method') or 'n/a (no manifest)'}`",
            r.get("n_segments", 0), r.get("n_matched", 0),
            r.get("n_invented", 0), r.get("n_missed", 0),
            ",".join(r.get("groups_requested") or []) or "—",
            len(r.get("errors") or []),
        ])
    L.append(_table(["Clip", "Variant", "Timing method", "Segs", "Matched",
                     "Invented", "Missed", "Groups", "Errors"], rows))

    # --- 2. placement table ---------------------------------------------
    L.append("## 2. Placement, pause and rate metrics\n")
    keys = [k for k in _PLACEMENT_KEYS if k in _SPEC_BY_KEY]
    header = ["Clip", "Variant"] + [_SPEC_BY_KEY[k].label for k in keys]
    rows = []
    for clip in clips:
        for arm in arms:
            r = by.get((clip, arm))
            if r is None:
                continue
            rows.append([f"`{clip}`", f"`{arm}`"] +
                        [_cell(r, _SPEC_BY_KEY[k]) for k in keys])
    L.append(_table(header, rows))

    # --- 3. paired difference -------------------------------------------
    L.append("## 3. Paired difference vs baseline\n")
    if baseline is None and len(arms) > 1:
        # The arm present on the most clips is the only defensible default.
        counts = {a: sum(1 for c in clips if (c, a) in by) for a in arms}
        baseline = max(sorted(counts), key=lambda a: counts[a])
        L.append(f"No `--baseline` given; using `{baseline}` "
                 f"(present on the most clips: {counts[baseline]}).\n")
    if baseline is None or len(arms) < 2:
        L.append("n/a (a paired comparison needs at least two arms; "
                 f"this run has {len(arms)}).\n")
    else:
        rows = []
        for arm in arms:
            if arm == baseline:
                continue
            for key in keys:
                spec = _SPEC_BY_KEY[key]
                pairs: List[Tuple[float, float]] = []
                for clip in clips:
                    rb, ra = by.get((clip, baseline)), by.get((clip, arm))
                    if rb is None or ra is None:
                        continue
                    vb, va = metric_value(rb, spec), metric_value(ra, spec)
                    if vb is None or va is None:
                        continue
                    pairs.append((vb, va))
                bs = bootstrap_paired(pairs, draws=draws, seed=seed)
                if bs["median"] is None:
                    rows.append([f"`{arm}`", spec.label, "n/a (" + bs["reason"] + ")",
                                 "n/a", "n/a"])
                    continue
                floor, _ = resolved_floor(key, results)
                ci = ("n/a (" + bs.get("reason", "one clip") + ")"
                      if bs["ci_lo"] is None
                      else f"[{bs['ci_lo']:+.{spec.digits}f}, "
                           f"{bs['ci_hi']:+.{spec.digits}f}]")
                if floor and abs(bs["median"]) < floor:
                    verdict = "**tie**"
                else:
                    better = (bs["median"] < 0) == spec.lower_is_better
                    verdict = ("better" if better else "worse") + f" ({arm})"
                    if bs["ci_lo"] is not None and bs["ci_lo"] <= 0 <= bs["ci_hi"]:
                        verdict += ", CI spans 0"
                rows.append([f"`{arm}`", spec.label,
                             f"{bs['median']:+.{spec.digits}f}{spec.unit}",
                             ci, verdict])
        L.append(f"Baseline arm: `{baseline}`. Δ is *arm − baseline*, "
                 "median over clips with the 95% bootstrap CI.\n")
        L.append(_table(["Arm", "Metric", "Δ median", "95% CI", "Verdict"], rows))

    # --- 4. drift --------------------------------------------------------
    L.append("## 4. Duration drift, per clip\n")
    L.append("Dubbed length vs source length, replayed from each manifest's "
             "`extra.mix` block.\n")
    spec = _SPEC_BY_KEY["drift_pct"]
    rows = []
    for clip in clips:
        row = [f"`{clip}`"]
        src = None
        for arm in arms:
            r = by.get((clip, arm))
            if r is None:
                row.append("—")
                continue
            src = src or _dig(r, "timing._harness.mix_replay.source_duration_s")
            fin = _dig(r, "timing._harness.mix_replay.final_duration_s")
            d = metric_value(r, spec)
            row.append("n/a (" + metric_reason(r, spec) + ")" if d is None
                       else f"{fin:.2f} s · {d:+.1f}%")
        rows.append([row[0], f"{src:.2f} s" if src else "n/a (no source duration)"]
                    + row[1:])
    L.append(_table(["Clip", "Source"] + [f"`{a}`" for a in arms], rows))

    # --- 5. MT -----------------------------------------------------------
    L.append("## 5. MT text metrics — separate, 2-arm only\n")
    mt_capable = [a for a in arms
                  if any((by.get((c, a)) or {}).get("mt") for c in clips)]
    L.append("Only the MT stage changes the text. Reporting chrF++ across "
             "every timing arm would be the same number repeated, so this "
             "table is restricted to two arms by construction.\n")
    chosen = list(mt_arms) if mt_arms else mt_capable
    if not mt_capable:
        L.append("n/a (no result carries an `mt` group; run the sweep with "
                 "`--groups mt`).\n")
    elif len(chosen) > 2:
        L.append(f"n/a ({len(chosen)} arms carry MT metrics: "
                 + ", ".join(f"`{a}`" for a in chosen)
                 + ". Name the two that differ in the MT stage with "
                   "`--mt-arms A,B`.)\n")
    else:
        header = ["Clip"] + [f"`{a}`" for a in chosen]
        for key in _MT_KEYS:
            s = _SPEC_BY_KEY[key]
            rows = []
            for clip in clips:
                row = [f"`{clip}`"]
                for a in chosen:
                    r = by.get((clip, a))
                    row.append("—" if r is None else _cell(r, s))
                rows.append(row)
            L.append(f"**{s.label}**\n")
            L.append(_table(header, rows))

    # --- 9. caveats ------------------------------------------------------
    L.append("## 9. Caveats — every number that could not be measured\n")
    tally: Dict[Tuple[str, str], int] = {}
    for r in results:
        for e in r.get("not_measured") or []:
            tally[(str(e.get("key")), str(e.get("reason")))] = \
                tally.get((str(e.get("key")), str(e.get("reason"))), 0) + 1
    if tally:
        rows = [[n, f"`{k}`", reason]
                for (k, reason), n in sorted(tally.items(), key=lambda kv: -kv[1])]
        L.append(_table(["Results", "Key", "Reason"], rows))
    else:
        L.append("None: every requested metric was computed.\n")

    err_rows = []
    for r in results:
        for e in r.get("errors") or []:
            err_rows.append([f"`{r.get('clip_id')}`", f"`{r.get('variant_id')}`",
                             f"`{e.get('key')}`", str(e.get("error"))])
    L.append("\n**Hard errors**\n")
    L.append(_table(["Clip", "Variant", "Key", "Error"], err_rows)
             if err_rows else "None.\n")

    # --- 12. provenance --------------------------------------------------
    L.append("## 12. Provenance — exactly which bytes produced each row\n")
    rows = []
    for r in results:
        for fieldname, rec in (r.get("inputs") or {}).items():
            if not isinstance(rec, dict):
                continue
            if rec.get("path") is None and rec.get("reason") == "not supplied":
                continue
            rows.append([
                f"`{r.get('clip_id')}`", f"`{r.get('variant_id')}`", fieldname,
                f"`{rec.get('path')}`", f"`{_short(rec.get('sha256'))}`",
                rec.get("bytes") if rec.get("bytes") is not None else "n/a",
                "ok" if rec.get("ok") else f"n/a ({rec.get('reason')})",
            ])
    L.append(_table(["Clip", "Variant", "Input", "Path", "sha256 (12)",
                     "Bytes", "State"], rows))
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m backend.eval report",
        description="Aggregate timing_eval results into markdown.",
    )
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULT_ROOT,
                    help="Root to scan for *.timing_eval.json.")
    ap.add_argument("--clip", action="append", dest="clips", default=None)
    ap.add_argument("--variant", action="append", dest="variants", default=None)
    ap.add_argument("--baseline", default=None,
                    help="Arm every paired difference is taken against.")
    ap.add_argument("--mt-arms", default=None,
                    help="Comma-separated pair of arms for the MT table.")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=None,
                    help="Write the markdown here as well as to stdout.")
    args = ap.parse_args(argv)

    results = load_results(args.results, clip_ids=args.clips,
                           variant_ids=args.variants)
    md = render_report(results, baseline=args.baseline,
                       mt_arms=args.mt_arms.split(",") if args.mt_arms else None,
                       draws=args.draws, seed=args.seed)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        log.info("report: wrote %s", args.out)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
