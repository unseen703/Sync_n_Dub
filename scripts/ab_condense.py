"""A/B the 9c2 shortening agent: the UI's own run, with and without it.

    $PY scripts/ab_condense.py --without <job_id>        # run arm B + compare
    $PY scripts/ab_condense.py --without <job_id> --compare-only

Arm A is a REAL UI run that has already happened -- its manifest records
`condense_overlong_text: False`, so it is the "without" arm as it stands.
Arm B re-runs the SAME clip through `ui.presets.build_kwargs`, the UI's own
kwargs builder, with exactly one thing added: `condense_overlong_text=True`.

Going through `build_kwargs` rather than hand-mapping the manifest onto CLI
flags is deliberate. The UI calls `run_pipeline` directly and sets
combinations the CLI cannot express -- the live job carries both
`timing_producer='naive_slot'` and `tts_force_align=True`, and no single
`--timing-method` produces that pair. Rebuilding it by hand would quietly
compare two different configurations and attribute the difference to the
agent.

The LLM is PREFLIGHTED before anything is spent. A wrong model slug does
not raise in `backend/condense.py` -- it logs, keeps the original text and
carries on -- so arm B would silently become a second copy of arm A and
the comparison would read "the agent does nothing". That is the failure
this script exists to make impossible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/data1/dipak_data/dubbingstudio")
sys.path.insert(0, str(ROOT))

JOBS = ROOT / "data/jobs"


def preflight_llm(model: str | None) -> tuple[bool, str]:
    """One tiny round trip. Returns (ok, message); prints no secrets."""
    try:
        from backend.llm import OpenRouterClient
        from backend.condense import DEFAULT_MODEL
        client = OpenRouterClient(model=model or DEFAULT_MODEL)
        if not client.available():
            return False, "OPENROUTER_API_KEY is not set"
        reply = client.chat_json(
            [{"role": "system",
              "content": 'Reply with only this JSON: {"text": "ok"}'},
             {"role": "user", "content": "ping"}],
            temperature=0.0)
        got = (reply or {}).get("text") if isinstance(reply, dict) else None
        if not got:
            return False, f"{client.describe()} answered but gave no usable JSON"
        return True, f"{client.describe()} -> {got!r}"
    except Exception as e:                               # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def plan_summary(job: Path) -> dict:
    for p in sorted(job.glob("timing/*.plan.json")):
        try:
            return json.loads(p.read_text()).get("summary", {})
        except Exception:                                # noqa: BLE001
            pass
    return {}


def text_stats(job: Path) -> dict:
    mt = max(job.glob("mt/*.json"),
             key=lambda p: len(json.loads(p.read_text()).get("segments") or []),
             default=None)
    if not mt:
        return {}
    d = json.loads(mt.read_text())
    segs = d.get("segments") or []
    chars = sum(len((s.get("final_text") or s.get("mt_text") or "")) for s in segs)
    est = sum(float(s.get("est_duration_s") or 0) for s in segs)
    return {"segments": len(segs), "chars": chars,
            "est_speech_s": round(est, 2),
            "condense": (d.get("extra") or {}).get("condense") or {}}


def report(a: Path, b: Path) -> None:
    pa, pb = plan_summary(a), plan_summary(b)
    ta, tb = text_stats(a), text_stats(b)
    from backend.media_io import probe_duration

    def line(label, x, y, fmt="{}"):
        fa = "-" if x is None else fmt.format(x)
        fb = "-" if y is None else fmt.format(y)
        print(f"  {label:24} {fa:>12}  {fb:>12}")

    print("\n" + "=" * 56)
    print(f"  {'':24} {'WITHOUT':>12}  {'WITH':>12}")
    line("segments", ta.get("segments"), tb.get("segments"))
    line("translated chars", ta.get("chars"), tb.get("chars"))
    line("estimated speech s", ta.get("est_speech_s"), tb.get("est_speech_s"))
    for k in ("tempo_mean", "tempo_max", "tempo_cap_hits",
              "fit_duration_s", "unfilled_slot_s"):
        line(k, pa.get(k), pb.get(k))
    for tag, j in (("without", a), ("with", b)):
        w = next(iter(sorted(j.glob("mix/*.dubbed.wav"))), None)
        if w:
            print(f"  dub length ({tag:7}): {probe_duration(w):.2f}s")
    c = tb.get("condense") or {}
    if c:
        print(f"\n  agent: model={c.get('model')}  considered={c.get('considered')} "
              f"rewritten={c.get('rewritten')} rejected={c.get('rejected')} "
              f"failed={c.get('failed')}  saved={c.get('saved_s')}s")
        for s in (c.get("segments") or [])[:8]:
            print(f"    {s['id']}: {s['before_s']}s -> {s['after_s']}s "
                  f"(slot {s['target_s']}s, {s['before_chars']}->{s['after_chars']} chars)"
                  f"{'' if s.get('fits') else '   still over'}")
    else:
        print("\n  agent: NO RECORD in the with-arm -- it did not run")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--without", required=True, help="the existing UI job id")
    ap.add_argument("--model", default=None, help="override CONDENSE_MODEL")
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-24s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S")

    job_a = JOBS / a.without
    if not job_a.is_dir():
        print(f"no such job: {job_a}")
        return 2
    cfg = json.loads((job_a / "manifest.json").read_text()).get("config", {})
    if cfg.get("condense_overlong_text"):
        print(f"{a.without} already RAN the agent; it is not a 'without' arm")
        return 2
    job_b = JOBS / f"{a.without}__condense"

    if a.compare_only:
        report(job_a, job_b)
        return 0

    from dotenv import load_dotenv
    load_dotenv()

    ok, msg = preflight_llm(a.model)
    print(f"\nLLM preflight: {'OK' if ok else 'FAILED'} -- {msg}")
    if not ok:
        print(
            "\nRefusing to run the 'with' arm. A dead model slug does not\n"
            "raise -- condense keeps the original text and the run looks\n"
            "normal -- so arm B would be a copy of arm A and the result\n"
            "would read 'the agent does nothing'.\n\n"
            "Find the real slug with your own key:\n"
            "  curl -s https://openrouter.ai/api/v1/models \\\n"
            '    -H "Authorization: Bearer $OPENROUTER_API_KEY" \\\n'
            "    | python3 -c \"import json,sys;[print(m['id']) for m in "
            "json.load(sys.stdin)['data'] if 'gemini' in m['id']]\"\n\n"
            "then re-run with --model <that id>.")
        return 3

    clip = next(iter(sorted(job_a.glob("videos/*.mp4"))), None)
    if not clip:
        print(f"no clip under {job_a}/videos")
        return 2

    from ui import presets
    from backend.pipeline import run_pipeline

    # The UI's own builder, so arm B differs by exactly one setting.
    kwargs = presets.build_kwargs(
        presets.resolve_preset(None),
        target_language=cfg.get("target_language") or "hi",
        tts=cfg.get("tts") or "bodhan",
        lipsync_model="off" if not cfg.get("lipsync") else cfg["lipsync"],
        skip_stems=bool(cfg.get("skip_stems")),
        source_language=cfg.get("asr_language") or None,
    )
    kwargs["condense_overlong_text"] = True
    if a.model:
        kwargs["condense_model"] = a.model
        os.environ["CONDENSE_MODEL"] = a.model

    print(f"\nclip   : {clip.name}")
    print(f"arm B  : {job_b.name}")
    print(f"delta  : condense_overlong_text=True"
          f"{'  model=' + a.model if a.model else ''}\n")

    t0 = time.time()
    run_pipeline(str(clip), job_id=job_b.name, **kwargs)
    print(f"\narm B finished in {time.time() - t0:.0f}s")
    report(job_a, job_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
