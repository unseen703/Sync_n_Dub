# Lip-Sync Track — Aggregated Chat History

Consolidated from the Claude Code session transcripts for this project
(`~/.claude/projects/-data1-dipak-data-dubbingstudio/`, 56 sessions, 126 MB).
18 of those sessions carry the lip-sync thread; the rest are audio-pipeline work.

Span: **2026-10-02 21:17 → 2026-10-06 11:55**, with infrastructure follow-ups on
10-08 and 10-11. Everything below is drawn from what was actually said and
measured in those sessions — no numbers are reconstructed from memory.

---

## 1. Session index

| # | Session | When | What it was about |
|---|---|---|---|
| 1 | `406d4e73` | 10-02 21:17 → 10-03 20:38 | **Origin.** "use Wav2Lip, MuseTalk 1.5, LatentSync 1.6 — provide support for three models." Adapters built; Bodhan TTS outage forced the Edge-TTS fallback; first dubbed MP4s. |
| 2 | `5f0b63fd` | 10-02 22:22 → 10-11 02:15 | VS Code inspection UI (video/transcript flash cards); later job-manifest browsing. |
| 3 | `9b4753bf` | 10-02 23:10 → 23:24 | "How long to lipsync a 2-min video?" → **multi-GPU time-sharding** built. |
| 4 | `17d595a5` | 10-02 23:29 → 10-03 13:43 | Full no-Bodhan run, all three models, 1 min of `2vxQaT0Fy-c`. MuseTalk succeeded; Wav2Lip and LatentSync both failed on face-less frames. |
| 5 | `22df6f2c` | 10-03 20:41 → 10-04 13:28 | Continuation from a pasted session summary. The 3-model × 3-clip matrix; tmux `dubsync` persistence. |
| 6 | `aaed53b5` | 10-03 20:48 | "add support for LipGAN" → 4th adapter registered. |
| 7 | `e2f2bf22` | 10-04 13:30 → 17:37 | 30 s NAG clip, all four variants, Bodhan-only ASR/MT/TTS. LipGAN stuck on CPU for 2 h. |
| 8 | `53ec3173` | 10-04 14:05 | LatentSync resource requirements (VRAM, scaling, minimum rigs). |
| 9 | `8322982d` | 10-04 17:39 → 18:13 | `lipgan-tf2.zip` port review; the MFCC-vs-mel-spectrogram feature question. Port rejected and deleted. |
| 10 | `8f4dc746` | 10-04 18:47 → 18:53 | First **downloaded-dub** run: YouTube Hindi audio track → Wav2Lip, MT/TTS skipped entirely. |
| 11 | `88c96c97` | 10-04 18:49 → 20:22 | Audio-joining bugs; "how does fit-3a + speed-up affect lip-sync?" |
| 12 | `12a9a387` | 10-04 20:26 → 10-05 08:53 | Which LipGAN checkpoint, and a line-by-line review of `lipgan-tf2_1.zip` (v2). |
| 13 | `3fdca646` | 10-05 08:56 → 11:48 | **LipGAN ported to PyTorch for Blackwell**, verified numerically identical to Keras. |
| 14 | `3a1aec7a` | 10-05 12:30 → 14:02 | Wav2Lip and MuseTalk on the NAG clip with global-tempo mixing; full time/memory profile of both. |
| 15 | `207007aa` | 10-05 13:37 → 13:46 | Complete model inventory, including every auxiliary model each lip-sync backend loads. RetinaFace/MediaPipe swap question. |
| 16 | `ec42c56f` | 10-05 14:15 → 10-06 01:39 | MuseTalk `bbox_shift` sweep that produced **no visible difference** — diagnosed down to the landmark level. |
| 17 | `e161959d` | 10-06 01:38 | "Which model handles a very small face?" — the PIP lecture-capture case. |
| 18 | `d18386db` | 10-06 01:44 → 10:47 | 2-GPU MuseTalk, LatentSync on Hindi, Whisper-size question, and the architectural walkthrough of both. |
| 19 | `7a64184e` | 10-06 11:51 → 11:55 | Last lip-sync run: Wav2Lip, NAG 1:20–1:40, raw downloaded Hindi. |

Later sessions touched lip-sync only indirectly: `52546545` (duration matching),
`53d4c6f6` (job-folder refactor + data archive), `467db3cd` (download provenance).

---

## 2. How the thread developed

### Phase 1 — Build the stage (10-02 → 10-03)

The opening ask was three backends plus an end-to-end run on two 5-minute clips.
What got built:

- `models/lipsync/{base,__init__,wav2lip_adapter,musetalk_adapter,latentsync_adapter}.py`
  — a registry of subprocess wrappers around each upstream repo under `third_party/`.
- `backend/lipsync.py` — `run_lipsync()`, plus `_needs_remux()` / `_remux_audio()`
  to force 48 kHz stereo AAC. This existed because Wav2Lip emits 16 kHz mono AAC
  and several players silently refuse it — which is what the "no audio is playing"
  complaint turned out to be. The audio was there (RMS 0.09); the container was the problem.
- `backend/pipeline.py` — lip-sync as stage 9h, after mix; `--lipsync` guarded to require `--mix`.

Two things derailed the original run and are worth remembering:

1. **Bodhan TTS returned HTTP 500 on every segment, ASR intermittently 502.** This
   was confirmed as an upstream `internal_error`, *not* rate limiting — the failures
   carried no `x-ratelimit-*` headers while a successful MT call showed 119/120
   requests remaining. The fix was a new **Edge TTS adapter** as a free fallback,
   plus fail-fast on two identical consecutive 5xx (previously it burned 5 min/segment).
2. **`librosa.filters.mel()` positional-arg crash** in `third_party/Wav2Lip/audio.py`
   under modern librosa — patched to keyword args.

### Phase 2 — The face-less-frame wall (10-03)

Running all three models on the 1-minute clip exposed the same root cause twice:

- **Wav2Lip** hard-raises `ValueError('Face not detected! Ensure the video contains
  a face in all the frames.')` — the source is lecture content with slide-only shots.
- **LatentSync** hit the same thing, plus silent OOM: v1.6 wants ~18 GB at 512×512
  and the cards here are 16 GB. A patch to `affine_transform_video` + `restore_video`
  made it carry forward the previous face and leave face-less frames untouched.
- **MuseTalk** was the only one that completed unattended (674 s for 60 s of 720p).

The eventual workaround was structural rather than algorithmic: run on clips that
are continuously face-bearing, and accept that slide-only lectures have nothing to animate.
`nsIMQP0QhpE` (NPTEL thermodynamics) was dropped from the matrix for exactly this
reason — 0 faces across sampled frames, and Wav2Lip's output there came from
false-positive detections.

### Phase 3 — Multi-GPU and persistence (10-02 → 10-04)

`backend/gpu.py` added `visible_cuda_devices()`, `pick_devices()`, `split_video_audio()`
and `concat_videos()`. `run_lipsync(..., num_gpus=N)` splits video + driving audio
into N equal-duration chunks, runs one adapter per GPU in a thread pool, and
concatenates. `--lipsync-num-gpus` exposes it; `DUBBING_GPUS=0,1` pins physical devices.

Two caveats recorded at the time, both still true:
- Chunk boundaries can show a one-frame mouth discontinuity, because each shard
  does its own face-detection smoothing. Use `--lipsync-num-gpus 1` for final renders.
- Clips shorter than `N × 15 s` fall back to single-GPU automatically.

Separately, runs were moved into a tmux session (`dubsync`) so they survive a
Claude session hitting its limit — this came directly from the request to "run in
persistent terminal so that it doesn't die when claude session ends due to limit."

### Phase 4 — The LipGAN saga (10-03 → 10-05)

This is the longest sub-thread and it resolved well.

1. Adapter added (`lipgan_adapter.py`), shelling out to upstream `batch_inference.py`.
2. First real run: upstream LipGAN pins **TF 1.15**, which cannot use modern CUDA,
   so it ran **CPU-only** — over 2 hours on a 30 s / 750-frame clip and still not done.
3. `lipgan-tf2.zip` (a supplied TF2 port) was reviewed. The key technical finding:
   LipGAN's `mfccs.mat` files actually hold **real HTK-style MFCCs** — 16 kHz,
   25/10 ms frames, pre-emphasis 0.97, 40 mel filters over 300–3700 Hz, 13 DCT
   coefficients with C₀ dropped, L=22 lifter → `(12, 35)` per 350 ms clip.
   The "LipGAN was trained on mel-specs" claim confuses it with **Wav2Lip**, which
   did switch to 80-band log-mel. Feeding librosa's default MFCCs (Slaney mel scale)
   gives a different feature distribution and garbage mouths.
4. v1 of the port was deleted. v2 (`lipgan-tf2_1.zip`) reviewed as a genuine drop-in
   for the adapter's CLI, but with a live risk: `lipgan_residual_mel.h5` has "mel"
   in its name while the master-branch port expects MFCC, and `load_weights(skip_mismatch=True)`
   would silently leave the audio encoder randomly initialized rather than fail loudly.
5. **Resolution:** instead of the TF2 port, LipGAN was reimplemented in **PyTorch**
   (`third_party/LipGAN/lipgan_torch.py` + `inference_torch.py`), reading the original
   Keras `.h5` and rebuilding the layer DAG. This is what `lipgan_adapter.py` calls today.

   Verification: **max abs diff 0.00000** against `tf.keras.load_model` on the same
   `.h5` over 4 samples (≤3e-5 at intermediate convs, pure float32 rounding). GPU vs
   CPU on a 75-frame video: per-frame mean diff 0.902/255, 100 % of frames within 3 LSB.

   One real bug was found during that verification: TF's `"same"` padding is
   **asymmetric for strided convs** (k=5,s=2 → 1 top / 2 bottom), so the initial
   symmetric `(k-1)//2` shortcut was correct at stride 1 and wrong at stride 2 —
   24 % output error, localized to `add_1` by a layer-by-layer diff.

### Phase 5 — Driving from downloaded dubs (10-04 onward)

A change of approach that shortened the loop considerably: for videos that already
carry an official YouTube dub track, **skip MT/TTS/indic-speak entirely** and drive
lip-sync from the downloaded Hindi audio. `scripts/fetch_indic_audio.py --langs hi`
→ `yt-dlp --download-sections` → ffmpeg trim/mux → `scripts/run_lipsync_only.py`.
Most runs from 10-04 on use this path.

### Phase 6 — The small-face problem (10-05 → 10-06)

The NAG lecture has the speaker as a **~70×58 px picture-in-picture tile in a
1920×1080 frame** — about 0.2 % of frame area. Three sessions went into it.

The `bbox_shift` sweep (2, 3, 4 px) produced **no visible difference at all**, and
the debug images explained why:

- mmpose places landmark 8 ("chin bottom") on the **lower lip** on a face this small —
  the real chin is 10–20 px lower.
- MuseTalk derives the bbox bottom from `max_y` over those landmarks, so the box
  stops at the lower lip and misses the mouth-to-chin region entirely.
- **`bbox_shift` only moves the top edge.** The bottom edge — where the actual
  problem was — never changes, no matter the value.
- The cyan `extra_margin = 10 px` extension was the only thing reaching the mouth at all.

The recommended fix was a sweep over **`extra_margin` ∈ {15, 25, 35}` with `bbox_shift = 0`**.
That was proposed at 10-06 01:39 and **never run** — `extra_margin` is still not
plumbed through `models/lipsync/musetalk_adapter.py`, which exposes only `bbox_shift`.

---

## 3. Measured results

All wall times are from the runs themselves; sources are the `logs/*.log` files
still present in the repo.

### 3-model matrix (10-04)

| Clip | Model | Wall | Output | Note |
|---|---|---|---|---|
| `2vxQaT0Fy-c_60to80s_720p` (20 s) | wav2lip | 1 m 44 s | 4.4 MB | |
| | musetalk | 5 m 37 s | 2.5 MB | |
| | latentsync | 27 m 44 s | 6.2 MB | |
| `2vxQaT0Fy-c_2min_720p` (120 s) | wav2lip | 6 m 27 s | 25.8 MB | |
| | musetalk | 16 m 50 s | 10 MB | |
| | latentsync | killed | — | ETA ~2.8 h, stopped on request |
| `nsIMQP0QhpE_60to80s` | all | — | — | 0 faces in source; not applicable |
| `2vxQaT0Fy-c_1min720` (60 s) | musetalk | 674 s | 8.7 MB | only model to survive the 10-03 run |

### 4-variant run, NAG 30 s clip (10-04)

| Model | Status | Wall |
|---|---|---|
| wav2lip | done | 117 s |
| musetalk | done | 588 s |
| latentsync | done (retry after OOM) | 2220 s |
| lipgan | **never finished** | >2 h on CPU, TF 1.15, 101 % CPU, 6.9 GB RSS |

### Wav2Lip vs MuseTalk, identical inputs (10-05, NAG 30 s @ 1920×1080)

| | Wav2Lip | MuseTalk v1.5 | Ratio |
|---|---|---|---|
| Wall time | 141.9 s | 634.4 s | **4.5× slower** |
| Realtime factor | 4.7× | 21.1× | |
| CPU user time | 165 s | 2562 s | 15.5× |
| CPU utilization | ~1.25 cores | ~4.2 cores | |
| Peak CPU RSS | 3.21 GiB | 7.49 GiB | 2.3× |
| VRAM delta (p90) | ~1.24 GiB | ~11.0 GiB | **8.9×** |
| Output resolution | 960×540 (resize_factor=2) | 1920×1080 (full) | |

MuseTalk peaked at 15 334 MiB on a 16 GiB card — 99 MiB free at worst.
`MUSETALK_BATCH=10` was the recorded mitigation for longer clips.

### LatentSync

- Measured peak: **~13.6 GB VRAM / ~10.6 GB RSS** on a 20 s clip.
- Documented floor: 8 GB at 256×256 (v1.5), **18 GB at 512×512** (v1.6 default).
- Blocker hit repeatedly: the shipped UNet checkpoint **locks cross-attention dim at 384**,
  so swapping Whisper-tiny for a larger Whisper is not possible without retraining.
- Rough throughput on a 24 GB card at defaults: v1.5 ≈ 2–4× realtime, v1.6 ≈ 0.5–1×.

### LipGAN (PyTorch port, 3 s / 75 frames / 640×360)

| Config | Inference | Wall |
|---|---|---|
| Blackwell GPU 0 (RTX PRO 2000, sm_120) | 7.6 s @ 9.88 it/s | 1 m 14 s |
| CPU, same code | 21 s @ 3.41 it/s | 3 m 01 s |

---

## 4. Model inventory (as recorded 10-05)

Each backend loads far more than its generator.

**MuseTalk runs six models at inference:**

| Component | Model | Size |
|---|---|---|
| Lip-sync UNet v1.5 | `models/musetalkV15/unet.pth` | ~1.3 GB |
| Audio encoder | Whisper **tiny** | ~150 MB |
| Latent VAE | `sd-vae-ft-mse` | ~335 MB |
| Face parsing | BiSeNet (`79999_iter.pth` + resnet18) | ~95 MB |
| Pose/landmarks | DWPose / RTMPose-L | ~130 MB |
| Face detection | S3FD | ~86 MB |
| *(SyncNet shipped but unused at inference)* | | ~1.1 GB |

**LatentSync:**

| Component | Model | Size |
|---|---|---|
| Lip-sync UNet3D | `latentsync_unet.pt` | ~5 GB (~1.3B params) |
| Audio encoder | Whisper tiny/small | ~150/460 MB |
| Latent VAE | `sd-vae-ft-mse` | ~335 MB |
| Detection + 106-pt landmarks | InsightFace buffalo_l (SCRFD-10GF) | ~325 MB |
| *(SyncNet, training/eval only)* | | ~1.2 GB |

Note LatentSync **already uses SCRFD/RetinaFace** — there is no detector swap to
make there. The swap question only applies to Wav2Lip's S3FD.

### How the two actually work

- **MuseTalk** = latent-space *inpainting*, one UNet forward pass per batch.
  Face crop 256 → VAE encode → mask lower face → cross-attend to a ±2-frame
  Whisper window → UNet (1 pass) → VAE decode → composite. Temporally local,
  which is exactly why time-sharding across GPUs works.
- **LatentSync** = audio-conditioned latent *diffusion*. 512×512 crop, 16-frame
  sliding window, 20 DDIM steps per window, classifier-free guidance at 2.0
  (so 2 UNet calls per step), plus 3D motion modules. Trained with reconstruction
  + perceptual + **SyncNet** + TREPA losses.

The ~100× gap between them is 20 iterative steps × 2 CFG passes × 4× the pixels ×
a heavier 3D UNet — not an implementation inefficiency.

---

## 5. Open items

Carried forward from the chat and still unresolved in the repo:

1. **`extra_margin` is not plumbed through the MuseTalk adapter.** This is the one
   knob the bbox diagnosis identified as actually controlling the bottom edge, and
   the proposed {15, 25, 35} sweep was never run. `musetalk_adapter.py` exposes
   `bbox_shift` only — which the debugging showed cannot fix this symptom.
2. **No VideoReTalking adapter.** Recommended twice as the best fit for the PIP
   lecture case (Wav2Lip-grade sync plus an enhancement stage and better paste-back seams).
3. **No pre-upscale option.** The recommended combo for tiny faces was
   GFPGAN/CodeFormer on the crop → lip-sync → downscale → paste, with an aggressively
   feathered paste mask. Never implemented.
4. **Wav2Lip still uses its own S3FD.** The suggested patch was to have `face_detect()`
   accept pre-computed boxes from `backend/faces.py` — a quality win *and* it removes
   duplicate per-frame detection cost, since the pipeline already detects faces at stage 5.
5. **No bbox tracker.** Per-frame detection jitter was called "visually fatal" at 70 px;
   ByteTrack on face detections or a landmark-anchored Kalman filter was the suggestion.
6. **LatentSync at 512 does not fit** the 16 GiB cards. Either stay at 256, or move to a 24 GB card.
7. **No sync metric was ever computed.** Every comparison in this history is wall-time,
   memory and visual inspection. LSE-C / LSE-D (SyncNet) scores were never run, even
   though both MuseTalk and LatentSync ship a SyncNet checkpoint.

## 6. Recorded guidance worth keeping

- **For small/PIP faces, Wav2Lip is counterintuitively the best of the four.** Its
  96×96 input is close to the native 70×58 crop, so nothing is upsampled; the mouth
  is low-res but correct. MuseTalk at 256 upsamples 3–4× and hallucinates drifting
  lip texture; LatentSync at 512 is the worst fit.
- **For full-face footage the ranking inverts** — LatentSync > MuseTalk > Wav2Lip on fidelity.
- Changing the face *detector* does not fix Wav2Lip's blur, teeth artifacts, or
  temporal flicker. Those are generator limits. Swapping the *generator* is the
  only large quality move.
- Lip-sync is deliberately **out of scope for `comparison-results.md`**, which covers
  the audio half only.

---

## 7. State of the artifacts

As of 2026-10-11, `data/lipsync/` is **empty** and `data/jobs/` holds no manifests —
the data folders were archived and the bulk media permanently deleted during the
job-layout refactor. Every output MP4 referenced above is gone; what survives is:

- the adapters and `backend/lipsync.py` (all four backends still registered),
- `third_party/{Wav2Lip,MuseTalk,LatentSync,LipGAN}/` including the PyTorch LipGAN port,
- **37 lip-sync run logs** in `logs/` (`lipsync_*`, `ls_*`, `nag_*`, `musetalk_*`,
  `latentsync_*`, `wav2lip_*`), which are the only first-hand record of the runs left on disk,
- `scripts/run_lipsync_only.py` and `scripts/run_lipsync.sh`. The four shell
  runners that drove the runs above (`run_musetalk.sh`,
  `retry_{latentsync,wav2lip}.sh`, `persistent_lipsync_runner.sh`) were removed
  on 2026-10-17: all four pinned the stem `2vxQaT0Fy-c_1min`, whose media the
  archive deleted, and three of them were the same script with the model name
  changed. `run_lipsync.sh <model> <stem>` is the merged replacement.
