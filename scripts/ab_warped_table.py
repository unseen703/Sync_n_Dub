"""Per-segment tempo breakdown: naive vs warped, one row per cue.

Reads the merged JSONs the A/B rerun left under each job's `ab_warped/<arm>/`
and prints a side-by-side table. The naive arm has one tempo per cue; the
warped arm has a speech tempo plus two joint rates and may absorb silence.

The three questions a reader is actually asking per cue, and the column
that answers each:

  HOW HARD IS THE SPEECH PUSHED  Naive's `tempo` is the ONE tempo it applies
    to the whole clip -- speech and silence alike. Warped's `speech_tempo`
    is just the speech; its pauses move independently and often harder.

  WHAT HAPPENED AT THE JOINT  `step` is this cue's relative change from the
    previous one. Naive has no concept of this; the number can be anything.
    Warped's should be at or under `--speed-step`, with the exceptions the
    plan flags out loud.

  WHO PAID FOR THE DEFICIT  `absorb (s)` is seconds taken from the clip's
    own silence. Positive means the silence shrank (compression); negative
    means it grew (expansion to fill a short slot). The headline case is a
    cue with speech=1.0x and absorb > 0 -- the silence covered the whole
    deficit and the speech was not touched at all.

No audio access, no re-planning: this reads what the A/B rerun wrote and
prints a table. Nothing is modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/data1/dipak_data/dubbingstudio")

JOBS = [
    ("L3oOldViIgY",
     "data/jobs/20261019-023946__L3oOldViIgY__pyannote.bodhan.mms_fa.bodhan.hi"),
    ("svwrTOm55y4",
     "data/jobs/20261022-070010__svwrTOm55y4__pyannote.bodhan.mms_fa.bodhan.hi"),
    ("x901PWoBO4A",
     "data/jobs/20261023-011027__x901PWoBO4A__pyannote.bodhan.mms_fa.bodhan.hi"),
]


def load_segments(arm_dir: Path) -> dict:
    """{id: row} for every segment in this arm's merged JSON.

    Returns {} when the arm has not been run yet -- the caller prints a
    hint rather than a wall of empty tables.
    """
    try:
        mt = next((arm_dir / "mt").glob("*.mt.json"))
    except StopIteration:
        return {}
    data = json.loads(mt.read_text())
    plan = ((data.get("extra") or {}).get("timing_plan") or {})
    rows: dict[str, dict] = {}
    for seg_src, seg_plan in zip(data.get("segments", []),
                                 plan.get("segments", [])):
        sid = seg_src.get("id")
        if sid is None:
            continue
        pieces = seg_plan.get("pieces", []) or []
        qc = seg_plan.get("qc") or {}
        rows[sid] = {
            # the source cue
            "src_start": float(seg_src.get("start", 0.0)),
            "src_end": float(seg_src.get("end", 0.0)),
            "target_s": float(seg_src.get("target_duration_s") or 0.0),
            "pieces": len(pieces),
            # the tempos. `tempo` is the one-size naive_slot number; the
            # warped producer records `speech_tempo` on top of it.
            "tempo": (float(pieces[0]["local_tempo"])
                      if pieces and "local_tempo" in pieces[0] else None),
            "speech_tempo": qc.get("speech_tempo"),
            "lead": qc.get("lead_tempo"),
            "trail": qc.get("trail_tempo"),
            # what the clip's own silence did
            "absorb_s": qc.get("pause_absorbed_s") or qc.get("pause_added_s"),
            "pause_before": qc.get("pause_s_before"),
            "pause_after": qc.get("pause_s_after"),
            "untouched": bool(qc.get("speech_untouched")),
            # anomalies worth seeing in the flags column
            "flags": list(qc.get("plan_flags") or []),
            "onset_shift": qc.get("onset_shift_s"),
            "boundary_drift": qc.get("boundary_drift"),
            "warp_failed": qc.get("warp_failed"),
        }
    return rows


def fmt_tempo(t) -> str:
    """One-cell tempo, with a visible marker for 'no number here'."""
    return "    -  " if t is None else f"{t:5.3f}x"


def arrow(prev, cur) -> str:
    """Relative step from `prev` to `cur` with a sign; '-' if either missing."""
    if prev is None or cur is None or prev <= 0:
        return "   -  "
    rel = abs(cur - prev) / prev
    sign = "+" if cur > prev else ("-" if cur < prev else " ")
    return f"{sign}{rel * 100:4.1f}%"


def collect_flags(row: dict) -> str:
    """Comma-joined flags, with derived annotations prepended.

    The derived ones (`1.0x`, `shift=`, `drift=`) are not plan flags per se;
    they are properties of the row worth reading at a glance beside the
    real flags. Keeping them in the same column saves horizontal space.
    """
    parts: list[str] = []
    if row.get("untouched"):
        parts.append("1.0x")
    if row.get("warp_failed"):
        parts.append("WARP_FAILED")
    shift = row.get("onset_shift") or 0.0
    if abs(shift) > 0.05:
        parts.append(f"shift={shift:+.2f}s")
    drift = row.get("boundary_drift") or 0.0
    if drift > 0.02:
        parts.append(f"drift={drift * 100:.1f}%")
    parts.extend(row.get("flags") or [])
    return " ".join(parts)


def summarise(arm: str, rows: dict) -> str:
    """A one-liner at the end of each job, from the per-row data only.

    Deliberately reconstructs the same numbers `plan.summary` carries so
    this script does not depend on the summary keys drifting -- a plan read
    from an older run would then still produce a comparable readout.
    """
    if not rows:
        return ""
    tempos = [r.get("speech_tempo") or r.get("tempo") for r in rows.values()]
    tempos = [t for t in tempos if t]
    if not tempos:
        return ""
    steps = [abs(b - a) / a for a, b in zip(tempos, tempos[1:]) if a > 0]
    untouched = sum(1 for r in rows.values() if r.get("untouched"))
    absorbed = sum(r.get("absorb_s") or 0.0 for r in rows.values()
                   if (r.get("absorb_s") or 0) > 0)
    added = sum(-(r.get("absorb_s") or 0.0) for r in rows.values()
                if (r.get("absorb_s") or 0) < 0)
    return (f"  [{arm:6s}] mean={sum(tempos) / len(tempos):.3f}x  "
            f"max={max(tempos):.3f}x  "
            f"max_step={(max(steps) * 100 if steps else 0.0):.1f}%  "
            f"untouched={untouched}/{len(rows)}  "
            f"absorbed={absorbed:.2f}s  added={added:.2f}s")


def print_job_table(tag: str, naive: dict, warped: dict) -> None:
    print(f"\n{'=' * 110}")
    print(f"  {tag}   (naive={len(naive)} segs, warped={len(warped)} segs)")
    print(f"{'=' * 110}")

    headers = (f"  {'id':<10} {'src slot (s)':<14} {'target':>6}  "
               f"| {'NAIVE tempo':<13} {'step':>7}   "
               f"| {'WARPED speech':<14} {'step':>7}  "
               f"{'lead/trail':<17} {'absorb (s)':<12}  flags")
    print(headers)
    print(f"  {'-' * 10} {'-' * 14} {'-' * 6}  "
          f"| {'-' * 13} {'-' * 7}   | {'-' * 14} {'-' * 7}  "
          f"{'-' * 17} {'-' * 12}  {'-' * 5}")

    prev_n = prev_w = None
    for sid in sorted(set(naive) | set(warped)):
        n = naive.get(sid, {})
        w = warped.get(sid, {})
        src = n or w
        slot = f"{src.get('src_start', 0):5.2f}-{src.get('src_end', 0):5.2f}"
        tgt = src.get("target_s") or 0.0

        n_t = n.get("tempo")
        w_t = w.get("speech_tempo") or w.get("tempo")
        lead = w.get("lead")
        trail = w.get("trail")

        pb, pa = w.get("pause_before"), w.get("pause_after")
        absorb = w.get("absorb_s")
        if pb is not None and pa is not None:
            pause_cell = f"{pb:.2f}->{pa:.2f}"
        elif absorb:
            pause_cell = f"{absorb:+.2f}"
        else:
            pause_cell = ""

        joints = f"{fmt_tempo(lead)}/{fmt_tempo(trail)}"
        print(f"  {sid:<10} {slot:<14} {tgt:6.2f}  "
              f"| {fmt_tempo(n_t):<13} {arrow(prev_n, n_t):>7}   "
              f"| {fmt_tempo(w_t):<14} {arrow(prev_w, w_t):>7}  "
              f"{joints:<17} {pause_cell:<12}  {collect_flags(w)}")
        prev_n, prev_w = n_t, w_t

    for arm, rows in (("NAIVE", naive), ("WARPED", warped)):
        line = summarise(arm, rows)
        if line:
            print(line)


def main() -> int:
    print(f"\n  per-segment tempo: NAIVE_SLOT vs WARPED\n")
    any_data = False
    for tag, job_rel in JOBS:
        job = ROOT / job_rel
        naive = load_segments(job / "ab_warped" / "naive")
        warped = load_segments(job / "ab_warped" / "warped")
        if not naive and not warped:
            print(f"\n[{tag}] no ab_warped/ output on disk -- run "
                  f"scripts/ab_warped_rerun.py first")
            continue
        any_data = True
        print_job_table(tag, naive, warped)
    if not any_data:
        print("\nnothing to show.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
