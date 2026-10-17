"""Cue-level timing cross-check for the Hindi reference.

The word-level floor (whisperx vs caption word stream) is unmeasurable in Hindi:
no video in the corpus carries an `hi-orig` auto-caption, so there is exactly one
Hindi word stream and nothing to compare it against.

This computes the next-best independent check: HUMAN-AUTHORED cue onsets from the
manual .hi.vtt vs the CTC word onset whisperx assigned to that cue's first word.
The two share text but not timing, so the disagreement is a genuine timing floor --
at cue resolution, not word resolution. It is an UPPER bound on the word-level
floor's unknown value, not a substitute for it.
"""
from __future__ import annotations
import json, pathlib, statistics as st, sys
sys.path.insert(0, "/data1/dipak_data/dubbingstudio")
from backend.transcripts import parse_vtt

GOLD = pathlib.Path("data/golden")
VID = pathlib.Path("data/videos")

def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

rows = []
for d in sorted(GOLD.iterdir()):
    if not d.is_dir():
        continue
    meta = json.loads((d / "meta.json").read_text())
    wfile = d / "ref_dub.words.json"
    if not wfile.exists():
        rows.append((d.name, meta["tier"], None, None, 0, 0.0, "no ref_dub.words.json"))
        continue
    src = meta["source"]
    vtt = VID / pathlib.Path(src["dir"]).name / src["ref_vtt"]
    if not vtt.exists():
        rows.append((d.name, meta["tier"], None, None, 0, 0.0, f"missing {vtt.name}"))
        continue

    t0, t1 = float(src["t0"]), float(src["t1"])
    cues = [c for c in parse_vtt(vtt) if t0 <= c.start < t1]
    words = json.loads(wfile.read_text())
    words = words.get("words", words) if isinstance(words, dict) else words
    # golden artifacts are CLIP-RELATIVE; cues are absolute
    wl = [(float(w["start"]), norm(str(w.get("text", "")))) for w in words
          if w.get("start") is not None]
    wl.sort()

    deltas = []
    for c in cues:
        first = norm((c.text or "").split()[0]) if (c.text or "").split() else ""
        if not first:
            continue
        cue_rel = c.start - t0
        # nearest whisperx word MATCHING the cue's first token, within +-2.0s
        cands = [(abs(ws - cue_rel), ws) for ws, tok in wl
                 if tok and (tok == first or tok.startswith(first) or first.startswith(tok))
                 and abs(ws - cue_rel) <= 2.0]
        if cands:
            deltas.append(min(cands)[0] * 1000.0)

    n_cues = sum(1 for c in cues if (c.text or "").split())
    if deltas:
        xs = sorted(deltas)
        k = 0.9 * (len(xs) - 1)
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        p90 = xs[lo] + (k - lo) * (xs[hi] - xs[lo])
        rate = len(deltas) / n_cues if n_cues else 0.0
        rows.append((d.name, meta["tier"], round(st.median(xs), 1), round(p90, 1),
                     len(deltas), rate, f"{len(deltas)}/{n_cues} cues matched"))
    else:
        rows.append((d.name, meta["tier"], None, None, 0, 0.0,
                     f"0/{n_cues} cues matched a whisperx token"))

MIN_MATCH_RATE = 0.90   # below this the median is computed on too few cues to mean anything

print(f"{'clip':34s} {'T':2s} {'med_ms':>8s} {'p90_ms':>8s} {'n':>4s} {'rate':>5s}  note")
for name, t, m, p, k, rate, note in rows:
    gate = "" if rate >= MIN_MATCH_RATE else "   <- excluded (match rate)"
    print(f"{name:34s} {t:2s} {str(m):>8s} {str(p):>8s} {k:>4d} {rate:>5.2f}  {note}{gate}")

used = [(r[0], r[2]) for r in rows if r[2] is not None and r[5] >= MIN_MATCH_RATE]
print()
if used:
    meds = [m for _n, m in used]
    print(f"Hindi CUE-LEVEL floor (match rate >= {MIN_MATCH_RATE:.0%}): n={len(used)} clips")
    print(f"  min={min(meds)}ms  median={st.median(meds)}ms  max={max(meds)}ms")
    by_vid = {}
    for n, m in used:
        by_vid.setdefault(n.split("_")[0], []).append(m)
    for v, ms in sorted(by_vid.items()):
        print(f"  {v:14s} n={len(ms)} median={st.median(ms)}ms  range={min(ms)}-{max(ms)}ms")
else:
    print("Hindi cue-level floor: not measured on any clip")
