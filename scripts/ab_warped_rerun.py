"""A/B naive_slot vs warped on three jobs, mixed to listenable dubs.

Writes INSIDE each job folder under `ab_warped/<arm>/` so outputs are
visible in the workspace (not /tmp) and sit beside the real dubbed files
for direct comparison. The source `mt/*.mt.json`, `stems/*.me.wav` and
video are READ; nothing in the source job is modified.

What each arm produces, per job:

  ab_warped/naive/   mt/*.mt.json + *.dubbed.wav + *.dubbed.mp4
  ab_warped/warped/  mt/*.mt.json + *.dubbed.wav + *.dubbed.mp4
                   + tts/warped/SEG_*.warped.wav (per-cue pre-stretched clips)

The warped arm uses retuned bounds: step 0.30, cap 1.60, ramp 0.12 s. The
0.20/1.45 defaults were ported straight from Amazon's TACL-corpus work and
proved infeasible on 60 s Indic clips with 1.3-1.8x compression ratios --
every cue bounced into the `fallback` path, so the slew limit was not
actually binding anything. These are a one-step loosening whose values
should be re-checked after a listening pass.

Delete outputs with  `rm -rf data/jobs/*/ab_warped/`.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data1/dipak_data/dubbingstudio")
from backend.mix import mix_dubbed_audio
from backend.timing import apply_plan_to_merged, plan, plan_naive_slot

ROOT = Path("/data1/dipak_data/dubbingstudio")

JOBS = [
    ("L3oOldViIgY",
     "data/jobs/20261019-023946__L3oOldViIgY__pyannote.bodhan.mms_fa.bodhan.hi"),
    ("svwrTOm55y4",
     "data/jobs/20261022-070010__svwrTOm55y4__pyannote.bodhan.mms_fa.bodhan.hi"),
    ("x901PWoBO4A",
     "data/jobs/20261023-011027__x901PWoBO4A__pyannote.bodhan.mms_fa.bodhan.hi"),
]

#: naive_slot bounds, matching what `--speed-mode segment` runs today so the
#: comparison is honest: this arm is EXACTLY what the current pipeline would
#: produce. If the arm looks different from the dub in the job's own `mix/`
#: folder, that is a signal the pipeline has drifted since the job ran.
SEG = dict(safety_margin_s=0.0, max_tempo=1.8, min_tempo=0.75,
           use_target_duration=True, overlap_policy="keep", gap_barrier_s=1.0)

#: warped bounds. Two things changed from the previous 0.20 run:
#:
#: `max_rel_step=0.30` because 0.20 fell straight through to the fallback on
#: every job (realised step 0.55/0.56/0.80 against the 0.20 limit asked for).
#: 0.30 is still tighter than the ear's "lurch" threshold and sits above the
#: material's natural tempo spread, so the solver converges instead of giving
#: up -- which is the thing that gives the warped arm an audible shape.
#:
#: `break_on_speaker=False` is the structural change. With it True, every
#: diarized label starts a new island and the planner has nothing to smooth
#: ACROSS. The previous table showed many single-cue islands with flat
#: lead/trail (1.600x/1.600x) -- a ramp needs two cues. False lets A at 1.07x
#: and B at 1.80x be in one island even when a different speaker says each,
#: so the ripple reaches across the diarization boundary and the warped arm
#: actually gets to do its job.
WARP = dict(max_rel_step=0.30, tempo_max=1.60, tempo_hard_max=1.80,
            ramp_s=0.12, pause_borrow_s=0.30,
            break_on_speaker=False)


def find_video(job: Path) -> Path | None:
    """The source mp4 under this job. Four conventions, in order of likelihood.

    `data/DATA_LAYOUT.md` names `source/` but older jobs put the download at
    the job root. A loud None here beats a silent fall-through to no mux.
    """
    for pat in ("source/*.mp4", "video/*.mp4", "download/*.mp4", "*.mp4"):
        found = sorted(job.glob(pat))
        if found:
            return found[0]
    return None


def run_arm(job: Path, arm: str, src_mt: Path, me: Path | None,
            video: Path) -> None:
    """One arm: copy the merged JSON, plan, apply, mix, mux. Loud on failure."""
    work = job / "ab_warped" / arm
    work.mkdir(parents=True, exist_ok=True)
    # Per-arm `mt/` so the warp producer's `tts/warped/` lands at
    # `work/tts/warped/` -- it writes next to the merged JSON's parent.parent,
    # so two arms using the same tts/ directory would overwrite each other's
    # SEG_NNN.warped.wav (which bit the first run of this script, hence the
    # per-arm dir).
    mt_dir = work / "mt"
    mt_dir.mkdir(exist_ok=True)
    mt = mt_dir / src_mt.name
    shutil.copy(src_mt, mt)

    t_plan_0 = time.time()
    if arm == "naive":
        p = plan_naive_slot(mt, **SEG)
        clip_source = "raw"
    else:
        p = plan("warped", mt, **WARP)
        clip_source = "warped"
    apply_plan_to_merged(p, mt)
    t_plan = time.time() - t_plan_0

    t_mix_0 = time.time()
    try:
        r = mix_dubbed_audio(mt, me_stem=me, source_video=video, out_dir=work,
                             strategy="timing_plan", clip_source=clip_source,
                             mux_video=True)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  {arm:7s} MIX FAILED {type(exc).__name__}: {exc}")
        return
    t_mix = time.time() - t_mix_0

    s = p.summary
    step_note = ""
    if arm == "warped":
        step_note = (f"step={s.get('tempo_step_max')}/"
                     f"{s.get('tempo_step_limit')} ok={s.get('tempo_step_ok')} ")

    out = r.video_mp4 or r.audio_wav
    rel = out.relative_to(ROOT) if out else "(no output?)"
    speech_max = s.get("speech_tempo_max") or s.get("tempo_max")
    print(f"  {arm:7s} plan={t_plan:5.2f}s mix={t_mix:5.2f}s  "
          f"speech_max={speech_max} {step_note}-> {rel}")


def main() -> int:
    for tag, job_rel in JOBS:
        job = ROOT / job_rel
        mts = sorted(job.glob("mt/*.mt.json"))
        if not mts:
            print(f"\nskip {tag}: no mt/*.mt.json under {job}")
            continue
        src_mt = mts[0]
        me_candidates = sorted((job / "stems").glob("*.me.wav"))
        me = me_candidates[0] if me_candidates else None
        video = find_video(job)
        if video is None:
            print(f"\nskip {tag}: no source video under {job}")
            continue

        print(f"\n{'=' * 72}\n{tag}  video={video.name}  "
              f"me={'yes' if me else 'NO (silent bed)'}"
              f"\n  outputs -> {job.relative_to(ROOT)}/ab_warped/\n{'=' * 72}")
        for arm in ("naive", "warped"):
            run_arm(job, arm, src_mt, me, video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
