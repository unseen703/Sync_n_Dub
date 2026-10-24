"""Re-fit a finished dub with ENGINE-SIDE speed control, and compare.

    $PY scripts/test_tts_speed_control.py -j <job_id> [--mode oneshot]
                                          [--bounds 0.85,1.40]

Replays stages 9d..9g from the job's EXISTING merged JSON, so the
translation, the segmentation and the cast are byte-identical to the
original run and the requested TTS rate is the only thing that changed.
Re-running the whole pipeline would also re-translate, and a different
Hindi sentence is a different length -- which is the thing being measured.

Writes to `<job>__spd<mode>` and never touches the original job.

Why this exists: on the 2026-10-21 svwrTOm55y4 run the timing planner was
already at full stretch and still overran. 50.9s of Hindi had to fit 30.6s
of slots -- 1.66x -- against a 1.5x cap, so 3 of 5 segments were clamped
and the dub ran ~8s long. `atempo` cannot close that gap. Engine-side rate
attacks it earlier: the TTS returns shorter audio, so the planner starts
from a smaller number instead of clamping.

**The rate field is an inference and this is its first real test.** Bodhan's
docs list only model/input/voice/instructions -- no speed parameter -- so
the adapter sends `speed`, borrowed from the OpenAI /v1/audio/speech shape
this endpoint otherwise matches. If the name is wrong the API answers 422
and the adapter says so by name. A 422 here is a RESULT, not a crash: it
tells you to try `BODHAN_TTS_RATE_FIELD=rate` or `instructions.speed`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/data1/dipak_data/dubbingstudio")
sys.path.insert(0, str(ROOT))


def plan_summary(path: Path) -> dict:
    try:
        return json.loads(path.read_text()).get("summary", {})
    except Exception:                                    # noqa: BLE001
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--job", required=True)
    ap.add_argument("--mode", default="oneshot",
                    choices=("oneshot", "closed_loop"))
    ap.add_argument("--bounds", default="0.85,1.40",
                    help="LO,HI the planner may ask the engine for. The "
                         "default 0.85,1.15 is too narrow for this clip: it "
                         "needs 1.66x overall, so a 1.15 ceiling leaves most "
                         "of the gap to atempo and changes little.")
    a = ap.parse_args()

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-26s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S")
    log = logging.getLogger("spdtest")

    from dotenv import load_dotenv
    load_dotenv()                      # BODHAN_TTS_KEY; see CLAUDE.md
    from backend.pipeline import run_post_mt_stages
    from backend.media_io import probe_duration

    src_job = ROOT / "data/jobs" / a.job
    if not src_job.is_dir():
        log.error("no such job: %s", src_job)
        return 2
    cfg = json.loads((src_job / "manifest.json").read_text()).get("config", {})
    # Pick the file with the most SEGMENTS, not the alphabetically first.
    # `mt/` accumulates stubs: a lip-sync-only replay drops a
    # `lipsync_only.json` carrying `{"segments": []}` there, and `l` sorts
    # before `s`, so a plain sorted()[0] silently chose the empty one and
    # the run died at "no usable segments" before reaching the API.
    def _n_segments(p: Path) -> int:
        try:
            return len(json.loads(p.read_text()).get("segments") or [])
        except Exception:                                # noqa: BLE001
            return -1

    candidates = sorted(src_job.glob("mt/*.json"), key=_n_segments, reverse=True)
    mt_json = next((p for p in candidates if _n_segments(p) > 0), None)
    video = next(iter(sorted(src_job.glob("videos/*.mp4"))), None)
    me_stem = next(iter(sorted(src_job.glob("stems/**/no_vocals.wav"))), None)
    if not (mt_json and video):
        log.error("job is missing mt/ or videos/: mt=%s video=%s",
                  mt_json, video)
        return 2

    lo, hi = (float(x) for x in a.bounds.split(","))
    out_root = ROOT / "data/jobs" / f"{a.job}__spd{a.mode}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "mt").mkdir(exist_ok=True)
    work_mt = out_root / "mt" / mt_json.name
    shutil.copy2(mt_json, work_mt)     # never annotate the original

    merged = json.loads(work_mt.read_text())
    # Pin the SAME cast. Stage 9d would otherwise re-derive it, and a
    # different voice speaks at a different natural pace -- which would
    # contaminate exactly the measurement being made.
    overrides = (merged.get("extra", {}) or {}).get("speaker_voice_map") or {}

    # Stage 9d needs the diarization turns and the audio to cut speaker
    # reference clips from. Both are REQUIRED even when the cast is pinned:
    # `run_post_mt_stages` still walks `turns` to total per-speaker speech
    # and still calls `extract_speaker_references`. Passing None for either
    # fails inside soundfile with `Error opening 'None'`, which names
    # neither the stage nor the argument.
    #
    # Turns come from the merged JSON's own segments rather than from one of
    # the three RTTMs on disk (raw / resegmented / refined). Those differ,
    # and picking the wrong one would re-synthesise against a segmentation
    # the text was never aligned to. The merged JSON is by definition the
    # one the dub was built from.
    from models.diarization.base import Turn
    turns = [Turn(start=float(s["start"]), end=float(s["end"]),
                  speaker=str(s.get("speaker") or "SPK_00"))
             for s in merged.get("segments", [])
             if s.get("start") is not None and s.get("end") is not None]
    if not turns:
        log.error("no usable segments in %s", work_mt)
        return 2

    # The dialogue stem when stems ran, else the raw 16 kHz mix -- the same
    # order `pipeline.py` uses for `dia_input`, which is what it passes here.
    ref_audio = next(iter(sorted(src_job.glob("audio/*.dialogue.16k.mono.wav"))), None) \
        or next(iter(sorted(src_job.glob("audio/*.16k.mono.wav"))), None)
    if not ref_audio:
        log.error("no reference audio under %s/audio", src_job)
        return 2
    log.info("turns      : %d from the merged JSON", len(turns))
    log.info("ref audio  : %s", ref_audio.name)

    base_plan = plan_summary(
        src_job / "timing" / f"{cfg.get('timing_producer', 'naive_slot')}.plan.json")

    log.info("job        : %s", a.job)
    log.info("mode       : %s, bounds %.2f-%.2f", a.mode, lo, hi)
    log.info("out_root   : %s", out_root)
    log.info("voices pinned from the original: %s", list(overrides) or "(none)")

    t0 = time.time()
    post = run_post_mt_stages(
        mt_json=work_mt,
        source_video=video,
        me_stem=me_stem,
        reference_audio_src=ref_audio,
        turns=turns,
        out_root=out_root,
        target_language=cfg.get("target_language") or "hi",
        target_script=cfg.get("target_script"),
        manifest=None,
        duration_adjust=bool(cfg.get("duration_adjust")),
        voice_engine=cfg.get("voice_engine") or "indic-speak",
        speaker_voice_overrides=overrides or None,
        tts_model=cfg.get("tts") or "bodhan",
        tts_style=cfg.get("tts_style"),
        tts_force_align=bool(cfg.get("tts_force_align")),
        timing_producer=cfg.get("timing_producer"),
        # the one variable
        tts_speed_control=a.mode,
        tts_speed_bounds=(lo, hi),
        mix=True,
    )
    wall = time.time() - t0

    new_plan = plan_summary(
        out_root / "timing" / f"{cfg.get('timing_producer', 'naive_slot')}.plan.json")

    def row(name, b, n):
        bb = b.get(name); nn = n.get(name)
        f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
        return f"  {name:22} {f(bb):>10} -> {f(nn):>10}"

    print("\n" + "=" * 60)
    print(f"  wall {wall:.0f}s   mode={a.mode}  bounds={lo}-{hi}\n")
    print(f"  {'metric':22} {'BEFORE':>10}    {'AFTER':>10}")
    for k in ("tempo_mean", "tempo_max", "tempo_cap_hits",
              "fit_duration_s", "unfilled_slot_s"):
        print(row(k, base_plan, new_plan))

    old_wav = next(iter(sorted(src_job.glob("mix/*.dubbed.wav"))), None)
    new_wav = post.mix_wav
    if old_wav and new_wav and Path(new_wav).is_file():
        print(f"\n  dub length  {probe_duration(old_wav):.2f}s -> "
              f"{probe_duration(Path(new_wav)):.2f}s "
              f"(video {probe_duration(video):.2f}s)")
    print(f"\n  out: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
