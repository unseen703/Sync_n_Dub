# MuseTalk frame deletion → freeze → mouth desync

Status: **pending approval**
Evidence job: `data/jobs/20261018-005205__L3oOldViIgY__pyannote.bodhan.mms_fa.bodhan.hi`

## Requirements

1. The rendered video plays continuously for the full source duration. No freeze, no truncation.
2. A frame MuseTalk cannot place a face in is **copied from the source unchanged**, never dropped.
3. Audio is embedded where it exists; where there is no dub, the picture still advances.
4. Output frame count == source frame count, exactly.

## Root cause — measured, not inferred

A single frame with no detected face truncates the rest of the shard. Chain:

| # | Where | What happens |
|---|---|---|
| 1 | `preprocessing.py:69,110` | no face → `coord_placeholder = (0.0,0.0,0.0,0.0)` |
| 2 | `inference.py:167` | `if bbox == coord_placeholder: continue` — frame never becomes a latent, so `input_latent_list` is N−k |
| 3 | `inference.py:178-180` | `frame_list_cycle`/`coord_list_cycle` are 2N but `input_latent_list_cycle` is 2(N−k) — **misaligned** |
| 4 | `inference.py:211` vs `datagen` | blend indexes frames by `i % 2N`, generator indexes latents by `i % 2(N−k)` → after the first placeholder every mouth lands on a **shifted frame** |
| 5 | `inference.py:217-220` | placeholder bbox → `cv2.resize(..., (0,0))` raises → `except: continue` → that index's PNG is never written |
| 6 | `inference.py:231` | `ffmpeg -f image2 -i %08d.png` **stops at the first gap** → video encodes only up to the first missing index |
| 7 | `inference.py:235` | `ffmpeg -i audio -i video out.mp4`, no `-shortest` → container length = longest stream = **audio** |
| 8 | `backend/gpu.py:185` | `concat_videos` re-encodes both shards into the final file |

Measured on shard 0 (`part00.lipsync.mp4`):

```
video stream : 10.635625 s / 255 frames      <- stops at the first missing PNG
audio stream : 20.020000 s / 313 packets
=> player holds the last frame for 9.38 s    <- the reported "pause from 0:11"
```

| | frames in | frames out | duration |
|---|---|---|---|
| part00 | 480 | **255** | 20.020 s |
| part01 | 480 | 468 | 20.354 s |
| final  | 960 | **950** | 40.499 s (source 40.040 s) |

Independent corroboration: stage-5 retinaface reports **44 of 959 frames with no face**, clustered near second 11 — where shard 0 truncates (255/23.976 = 10.64 s).

**Two defects, not one.** The freeze (steps 5-7) is visible. The mouth/frame desync (steps 2-4) is silent and corrupts frames that *were* written.

## Options

### A. Keep index alignment in the vendored loop, write originals for placeholders — CHOSEN
For a placeholder frame, reuse the last valid bbox so a latent is still produced (list stays length N), and in the blend loop write the **original frame unchanged** instead of compositing.
- Pros: fixes both defects at the source; ~6 lines; every index keeps a PNG so the image2 reader never hits a gap; no extra MuseTalk startups.
- Cons: edits vendored `third_party/MuseTalk`; one wasted inference per placeholder frame; a re-clone would revert it.

### B. Supply a complete coord list via `--use_saved_coord`
Compute bboxes ourselves (stage-5 retinaface already runs) and interpolate across no-face frames so no placeholder ever reaches MuseTalk.
- Pros: no vendored edit; reuses detections already paid for.
- Cons: must reproduce MuseTalk's landmark-derived box maths (`face_land_mark[28..30]`, iBUG-68 ordering **unverified** against insightface's layout); MuseTalk deletes the pickle at `inference.py:245`; does not fix the desync if a placeholder ever slips through.

### C. Span-based rendering at our layer
Render only spans with speech + face via `run_lipsync(window=…)`, splice untouched source frames around them.
- Pros: no vendored edit; `_splice_window` exists; matches `scripts/clip_asd_musetalk_run.py`.
- Cons: one MuseTalk startup (60-90 s) **per span** — this clip has 8 segments; does not fix the desync inside a span.

### D. Post-hoc conform
Detect the short video stream and pad from the source.
- Rejected: hides a corrupt render behind a cosmetic repair, and leaves the desync untouched.

**Why A:** it is the only option that fixes the silent desync as well as the visible freeze, and it is the smallest change. B and C both leave steps 2-4 intact. The vendored-edit risk is mitigated by a test that fails if the patch is reverted.

## Implementation steps

1. **`third_party/MuseTalk/scripts/inference.py` (~line 166)** — never skip; keep lists the same length.
   Track placeholders in a parallel `is_placeholder: list[bool]`, and for a placeholder reuse the last valid bbox (or the first valid one if none seen yet) so `input_latent_list` stays length N.
2. **Same file (~line 211)** — in the blend loop, if `is_placeholder[i % N]`, `cv2.imwrite` the untouched `ori_frame` and `continue` **before** any resize. Remove the bare `except: continue` so a genuine resize failure is no longer silent.
3. **`models/lipsync/musetalk_adapter.py`** — add `keep_work_dir: bool = False`. When set, copy the PNG dir and `<basename>.pkl` out of the `TemporaryDirectory` ([line 94](models/lipsync/musetalk_adapter.py#L94)) before it is destroyed, so placeholder indices are inspectable after the fact.
4. **`backend/lipsync.py`** — extend the frame-parity guard (added this session) to run **per shard** inside `_run_lipsync_sharded`, not only on the final concat. The concat masked 723 real frames as 950.
5. **`backend/lipsync.py`** — add a **stream-duration** check alongside frame parity: fail when the video stream is shorter than the audio stream by more than one frame. That is the signature that actually produced the freeze.

## Acceptance criteria

1. On the evidence clip, output frame count **== 960** and video-stream duration within **1 frame (0.042 s)** of 40.040 s.
2. No two consecutive output frames are byte-identical where the corresponding source frames differ by `mean|diff| > 1.0` (the 12→20 s window currently reads 0.00).
3. For each frame index MuseTalk marked placeholder, the output frame equals the source frame at that index with `PSNR > 50 dB`.
4. On a clip with **zero** placeholder frames, output is unchanged versus pre-patch (face-crop PSNR > 40 dB) — the patch is inert when no face is missing.
5. `_frame_count(src) == _frame_count(out)` holds per shard, asserted inside `_run_lipsync_sharded`.
6. Full suite green (currently 547 passed, 2 skipped, 1 xfailed).

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Vendored patch lost on re-clone | Test asserts the patched markers are present in `inference.py`; fails loudly if reverted |
| Reused bbox is wrong for the placeholder frame | Irrelevant — that frame's generated output is **discarded** and the original written; the latent exists only to hold index alignment |
| Wasted inference per placeholder frame | Bounded: 44 of 959 frames here (4.6%); measure before/after wall time |
| Removing `except: continue` surfaces a real crash | Intended — that silence is the defect; the error now reaches `result.error` |
| 68-point topology assumption (option B) | Not taken; A needs no landmark remapping |

## Verification

1. Re-render the evidence clip (stages 1-9g reusable via `--post-mt-only-from`) and compare frame counts and per-stream durations.
2. Frame-diff the 12→20 s window against the source; both must move together.
3. PSNR source-vs-output on the placeholder indices reported by `keep_work_dir`.
4. Confirm `extra.lipsync.produced` is True and `lipsync_mp4` is set.
5. Re-run the full suite.
