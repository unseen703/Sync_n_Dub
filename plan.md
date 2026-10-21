# Indic Video Dubbing Pipeline — Implementation & Benchmarking Plan

**Project:** LipSyncing pipeline · RamanLab, IIT Madras
**Date:** 2026-09-30
**Type:** Engineering project — integration, benchmarking and model selection using pretrained models and APIs only
**Provider:** Bodhan AI (Indic-Transcribe, Indic-Translate, Indic-Transliterate, Indic-Speak)

---

## 1. Framing

This project **integrates existing pretrained models and APIs**. It does not train, fine-tune or distil anything, and it does not build datasets for training.

The question every experiment answers is:

> **Which existing model works best for our pipeline and our data?**

Not: *can we train a better one?*

The one research-flavoured question that survives, reframed to fit:

> **Evaluate existing zero-shot voice conversion and expressive TTS methods for preserving speaker identity and speaking style during multilingual Indic dubbing.**

That is a measurement question about published models, answerable with a benchmark and a listening study. It is the study most likely to be publishable *as an evaluation paper*, and it needs no training run.

### Explicitly out of scope

Training or fine-tuning any diarization, ASR, MT, TTS, VC or lip-sync model · training speaker or emotion classifiers · RL / reward optimisation · building a custom end-to-end model · creating datasets for model training.

Annotating a small evaluation set is **not** dataset creation for training — it is how you measure anything at all, and it stays.

### Decisions taken

| Question | Decision |
| --- | --- |
| API provider | **Bodhan AI** — separate endpoint and access key per service (ASR, MT, TTS) |
| Code-mixing | Retain English terms, rendered in the **target Indic script** |
| Voice | Indic-Speak stock voices assigned per speaker; zero-shot cloning and VC benchmarked as alternatives |
| Languages | All 22 scheduled languages; **Hindi first**, then Tamil, then the rest |
| Lip sync | MuseTalk 1.5 primary, LatentSync 1.6 benchmarked (see §7) |

### Non-goals for v1

Overlapped speech (falls back to source audio) · singing and shouted speech · real-time streaming.

---

## 2. Pipeline

```text
Video
  ↓
Media I/O — demux, audio extraction              (FFmpeg / PyAV)
  ↓
Stem separation — dialogue vs music & effects    (HTDemucs)
  ↓
Speaker diarization                              (pyannote 3.1)
  ↓
Face detection                                   (RetinaFace / SCRFD)
  ↓
Face tracking                                    (ByteTrack)
  ↓
Active speaker detection                         (TalkNet / TS-TalkNet)
  ↓
Speaker ↔ face assignment                        (Hungarian over ASD scores)
  ↓
ASR                                              (Bodhan Indic-Transcribe)
  ↓
Forced alignment — word boundaries               (CTC aligner)
  ↓
Speaker-attributed timed transcript              ← the central representation
  ↓
Machine translation                              (Bodhan Indic-Translate)
  ↓
Code-mixing / script normalisation               (Bodhan Indic-Transliterate + termbase)
  ↓
Duration-aware text adjustment                   (candidate generation + ranking)
  ↓
Voice assignment per speaker                     (attributes + screen time)
  ↓
Speaker-specific TTS                             (Bodhan Indic-Speak)
  ↓
Optional voice conversion / style transfer       (Seed-VC / OpenVoice v2)
  ↓
Audio mixing over the M&E stem                   (FFmpeg)
  ↓
Lip sync                                         (MuseTalk 1.5)
  ↓
Mux                                              (FFmpeg)
  ↓
Final dubbed video
  ↓
QC — round-trip ASR, sync and timing metrics
```

---

## 3. The central representation

**A speaker-aware timed dialogue JSON — the manifest — is the single contract every stage reads and writes.**

```
Stage: (manifest, media_dir, config) → manifest'
```

No stage talks to another directly. No stage holds state. This is the design decision that makes the whole benchmarking plan possible:

1. **Independent evaluation.** Freeze a manifest at any stage boundary; run one implementation against it; score it. No upstream model needs loading.
2. **Model swapping.** Every implementation is an adapter behind one interface. `--asr bodhan` vs `--asr whisperx` vs `--asr indicconformer` is a flag.
3. **Incremental re-runs.** Cache key = `hash(stage, impl, config, the manifest slice the stage declares it reads)`. Change the TTS model and only TTS onward recomputes.

```jsonc
{
  "schema_version": "1.0",
  "job_id": "iitm_lec_042",
  "source": { "video": "source.mp4", "duration_s": 612.4, "fps": 25.0, "language": "en" },
  "target": { "language": "hi", "script": "Deva",
              "code_mix": "retain_in_target_script", "render_mode": "native" },
  "audio": { "original": "…", "dialogue_stem": "…", "me_stem": "…", "dubbed": "…" },

  "speakers": [
    { "id": "SPK_00", "total_speech_s": 384.1,
      "predicted_gender": "male", "gender_confidence": 0.97,
      "embedding_path": "emb/SPK_00.npy",
      "face_track_ids": ["FACE_01"], "on_screen_ratio": 0.82,
      "assigned_voice": { "engine": "indic-speak", "voice": "<voice id>" },
      "reference_audio": "audio/ref/SPK_00.wav" }
  ],

  "face_tracks": [
    { "id": "FACE_01", "frames": [[120, 480], [900, 1450]],
      "bbox_path": "tracks/FACE_01.npy",
      "mean_pose_yaw": 6.2, "occlusion_ratio": 0.03 }
  ],

  "segments": [
    { "id": "SEG_0007", "speaker": "SPK_00",
      "start_s": 42.180, "end_s": 46.940,
      "overlap": false, "on_screen": true,
      "face_track": "FACE_01", "asd_score": 0.91,

      "asr_text": "So the gradient descent step updates the weight matrix.",
      "words": [ { "w": "So", "s": 42.180, "e": 42.310 } ],
      "pause_mask": [[42.180, 43.010], [43.180, 46.940]],

      "retain_spans": [ { "text": "gradient descent", "start": 11, "end": 27,
                          "type": "technical" } ],
      "mt_path": "A",
      "mt_text": "…",
      "mt_candidates": [ { "text": "…", "est_duration_s": 4.88, "chrf": 0.69 } ],
      "transliterations": { "gradient descent": "ग्रेडिएंट डिसेंट" },
      "final_text": "तो ग्रेडिएंट डिसेंट स्टेप वेट मैट्रिक्स को अपडेट करता है।",

      "target_duration_s": 4.760, "est_duration_s": 4.88,
      "tts_audio": "audio/seg/SEG_0007.wav", "tts_duration_s": 5.02,
      "stretch_factor": 0.948,

      "qc": { "roundtrip_wer": 0.07, "terms_retained": 2, "terms_expected": 2 } }
  ],

  "stage_log": [
    { "stage": "asr", "impl": "bodhan/indic-transcribe-core",
      "wall_s": 31.2, "cost_inr": 0.08, "cache_hit": false } ]
}
```

Notes that matter:

- `pause_mask` holds the speech intervals inside a segment. Preserving these matters more than matching total duration — a line that matches in length but redistributes its pauses still reads out of sync.
- `asd_score` records how confidently the active speaker detector bound this segment to that face track. Low scores route to off-screen handling rather than producing a wrong render.
- `retain_spans` uses character offsets into `asr_text`, so it survives re-translation.
- `mt_path` records which code-mix strategy produced the text (§6), which makes the A/B measurable after the fact.
- `cost_inr` per stage — Bodhan bills per unit and the eval suite's cost should be visible, not discovered on an invoice.

A `validate_references()` pass runs after every stage. It catches the failure that actually bites: a stage renames a speaker or drops a face track, and the dangling reference only surfaces at lip-sync time.

---

## 4. Face tracking and speaker ↔ face assignment

This is the part most open dubbing tools skip, and it is what makes multi-speaker lip-sync possible at all.

**Detection → tracking → ASD → assignment.**

1. **Detect** faces per frame. RetinaFace or SCRFD for accuracy; MediaPipe when speed matters more.
2. **Track** across frames into stable identities. ByteTrack is the default — it associates on detection confidence including low-score boxes, which handles the brief occlusions and profile turns that break simpler trackers. DeepSORT as the alternative when appearance re-identification is needed across long gaps.
3. **Detect the active speaker.** [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD) scores, per frame per face track, whether that face is speaking, from audio-visual cues. [TS-TalkNet](https://github.com/Jiang-Yidi/TS-TalkNet) (Interspeech 2023) extends it with a target-speaker enrolment embedding — useful precisely when several faces are visible and the audio is mixed.
4. **Assign.** For each diarized turn, average the ASD score of every face track over that interval. Solve the turn × track matrix with Hungarian matching rather than independent argmax, so two turns cannot claim the same face in overlapping time. Below a score threshold, the segment is marked `on_screen: false`, which keeps narration and off-camera speech out of the animated set.

The output is `segment.face_track`, which is the only thing the lip-sync stage needs and the only way it can be correct in a panel or interview.

**Speaker embeddings** (SpeechBrain ECAPA-TDNN, or pyannote's own) serve two jobs here: consistency checks on diarization output, and the similarity matrix behind voice assignment (§8). Note that verification embeddings encode channel and language, not perceptual similarity — they tell you *same person or not*, not *sounds alike*. Use them for the first and treat the second as a rough prior.

---

## 5. Bodhan AI integration

OpenAI-compatible endpoints, bearer auth, one call per task. **Credentials are per service** — ASR, MT and TTS each have their own endpoint and key.

| Task | Endpoint | Model |
| --- | --- | --- |
| Speech to text | `POST /v1/audio/transcriptions` | Indic-Transcribe (Core / Flex) |
| Translation | `POST /translate` | Indic-Translate |
| Transliteration | `POST /transliterate` | Indic-Transliterate |
| Text to speech | `POST /v1/audio/speech` | Indic-Speak |

```bash
# .env — gitignored
BODHAN_ASR_URL= ;  BODHAN_ASR_KEY= ;  BODHAN_ASR_MODEL=indic-transcribe-core
BODHAN_MT_URL=  ;  BODHAN_MT_KEY=  ;  BODHAN_MT_MODEL=indic-translate
BODHAN_TTS_URL= ;  BODHAN_TTS_KEY= ;  BODHAN_TTS_MODEL=indic-speak
BODHAN_TRANSLIT_URL= ; BODHAN_TRANSLIT_KEY=   # blank → MT endpoint serves it
```

The client separates **transport** (auth, retries, rate-limit headers, cost accounting — shared) from **payload mapping** (one small `encode`/`decode` pair per service). If an endpoint's shape differs from the public console API, only that pair changes and no stage code moves.

Rate limits come back as `x-ratelimit-limit-requests` and `x-ratelimit-limit-parallel-requests`, with 429 on exceed. The client reads those headers and adapts concurrency rather than fixing a worker count — ablation sweeps are exactly the workload that trips them.

### The models

| Model | Size / architecture | Coverage |
| --- | --- | --- |
| **Indic-Transcribe Core / Flex** | 1.2B each — NVIDIA Canary: FastConformer encoder (32L, 811M) + Transformer decoder (24L, 419M) | 27 languages: Indian-accented English, 22 scheduled, Hindi dialects, Bhili, Bhojpuri. 8.7 OIWER (Core) / 11.1 (Flex) on Voice of India |
| **Indic-Translate** | 7.94B, Gemma 4 E4B architecture, bf16, 32K context | English ↔ 22 languages, 44 directions, 25 language-script combinations, native and romanized. 58.97 dBLEU vs 47.44 baseline |
| **Indic-Speak** | 3.36B stack | 22 languages across 12 scripts, **45 voices**. ~93% of output in the top content-fidelity band |

**Indic-Transcribe transcribes code-switched speech as spoken** and offers three rendering modes — native script, mixed script (English and numerals in Latin), fully romanized. That makes the render mode a knob in the code-mix study (§6) rather than a fixed upstream.

### What Bodhan does not provide

Diarization · word-level timestamps · duration or length control on MT or TTS · voice cloning · speaking-style control.

Those gaps are what the rest of this pipeline supplies. Weights are also on Hugging Face under [`bodhan-ai`](https://huggingface.co/bodhan-ai), so self-hosting is available for reproducible sweeps — no rate limits, no network variance, byte-identical reruns.

---

## 6. Code-mixing — a selection question, not a modelling one

Target: retain English terms an Indian speaker would actually say in English, rendered in the target Indic script.

Bodhan may give this for free. Two paths; the experiment decides.

**Path A — native.** Indic-Transcribe already returns code-switched source text; ask Indic-Translate for code-mixed target output in native script and take what comes back. Two API calls per segment, no termbase, no masking.

**Path B — explicit control.** Mask the spans to retain, translate around them, transliterate them back:

```
"So the [gradient descent] step updates the [weight matrix]."
   ↓ mask       "So the <t0> step updates the <t1>."
   ↓ MT         "तो <t0> स्टेप <t1> को अपडेट करता है।"
   ↓ translit   <t0> → "ग्रेडिएंट डिसेंट",  <t1> → "वेट मैट्रिक्स"
   ↓ substitute "तो ग्रेडिएंट डिसेंट स्टेप वेट मैट्रिक्स को अपडेट करता है।"
```

Span selection, highest confidence first: termbase hits → acronyms and units (regex) → proper nouns (NER) → numbers with units → LLM judgement for the remainder, cached by text hash.

**The experiment.** Both paths on the same 200 segments. Measure term-retention rate, transliteration consistency (does "gradient descent" render identically every time), human naturalness (2 native speakers, blind to path), and cost/latency (Path B is 2–3× the calls).

**Decision rule:** if Path A lands within ~5 points of Path B on naturalness, ship Path A and keep the termbase only as an override for terms Path A gets wrong. The masking machinery is real complexity and a real bug surface — buy it only if the data says so.

Either way `termbase/{domain}.tsv` stays as the override layer. It is the cheapest quality lever in the pipeline and the only one a domain expert can drive without touching code.

---

## 7. Duration-aware text adjustment

Indic translations of English run **1.1× to 1.4× longer**. Indic-Translate has no length control, so: generate and rank.

1. **Budget** — `target_duration_s` from the diarized segment, minus a safety margin.
2. **Candidates** — 3–5 per segment: Indic-Translate baseline; a more aggressive retain-list (English terms are usually *shorter* than their Indic translations, so retaining more shortens the output **and** sounds more natural); Indic-Translate prompted for a shorter rendering, 2 variants.
3. **Estimate duration without synthesising** — fit `duration ≈ a·chars + b·words + c` per (language, voice) from a one-off calibration of ~500 sentences through Indic-Speak. This is a measurement, not a training run: a three-parameter least-squares fit that takes half a day and avoids synthesising every candidate.
4. **Rank** on `w₁·|est_dur − target| + w₂·(1 − chrf) + w₃·term_retention_penalty`.
5. **Synthesise** the winner; measure actual duration.
6. **Correct residual** with WSOLA time-stretch, **hard-capped at ±10%**. Beyond that, log `budget_violation` and absorb the overrun into a following pause or shorten the text further.

Headline metrics: `mean |actual − target|` and `% of segments needing >10% stretch`.

---

## 8. Model inventory

Primary choice plus the alternatives each experiment compares against. Licence notes are for the **weights**, listed for reference.

| Component | Primary | Alternatives | Licence notes |
| --- | --- | --- | --- |
| Media I/O | FFmpeg | PyAV | LGPL/GPL depending on build |
| Stem separation | HTDemucs | MossFormer2, none | MIT |
| Diarization | [pyannote 3.1](https://github.com/pyannote/pyannote-audio) | WhisperX (wraps pyannote), DiariZen, Sortformer | pyannote MIT · DiariZen and Sortformer weights **CC-BY-NC** |
| Face detection | RetinaFace / SCRFD | MediaPipe | InsightFace non-commercial for some models — check |
| Face tracking | ByteTrack | DeepSORT | MIT |
| Active speaker detection | [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD) | [TS-TalkNet](https://github.com/Jiang-Yidi/TS-TalkNet) | check repo licences |
| Speaker embedding | SpeechBrain ECAPA-TDNN | pyannote embeddings | Apache-2.0 |
| ASR | **Bodhan Indic-Transcribe** | WhisperX, IndicConformer, Whisper large-v3 | Indic Open Model v1.0 · Whisper MIT |
| Translation | **Bodhan Indic-Translate** | IndicTrans2, NLLB-200 | Indic Open Model v1.0 · NLLB **CC-BY-NC** |
| Code mixing | **Bodhan** (native or masked) | LLM with termbase | — |
| Transliteration | **Bodhan Indic-Transliterate** | IndicXlit | — |
| TTS | **Bodhan Indic-Speak** | IndicF5, CosyVoice 3, OpenVoice v2 | CosyVoice Apache-2.0 · OpenVoice v2 **MIT, commercial OK** |
| Voice conversion | [Seed-VC](https://github.com/Plachtaa/seed-vc) | [OpenVoice v2](https://github.com/myshell-ai/OpenVoice), CosyVoice, Vevo | Seed-VC **GPL-3.0** (copyleft) · OpenVoice MIT · Vevo **CC-BY-NC** |
| Expressive S2ST reference | — | SeamlessExpressive | gated research licence; **reportedly no Indic coverage** — verify before planning around it |
| Lip sync | [MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk) | [LatentSync 1.6](https://github.com/bytedance/LatentSync), Wav2Lip, TBDub | MuseTalk MIT code, weights **"available for any purpose, even commercially"** · LatentSync Apache-2.0 · **Wav2Lip non-commercial** |

### On MuseTalk vs LatentSync

Worth stating plainly, because the choice is not obvious:

- **MuseTalk 1.5** is latent-space *inpainting* (not diffusion): frozen VAE, frozen Whisper-tiny audio encoder, SD-style UNet with cross-attention. **256×256** face region, **30fps+ on a V100** — genuine real-time. v1.5 (March 2025) added perceptual, GAN and sync losses over v1.0's L1-only training.
- **LatentSync 1.6** is audio-conditioned latent *diffusion* at **512×512**, with SyncNet supervision in pixel space and TREPA for temporal consistency. Slower, higher fidelity.

For offline dubbing, output resolution matters more than throughput. The sensible split: **MuseTalk during development** because a 30fps render gives you a fast feedback loop, **LatentSync 1.6 for final renders and for the quality number in the results table**, and benchmark both on the golden set rather than assuming. Wav2Lip stays in as a historical baseline.

### Commercial reference baselines

Run the golden set through **ElevenLabs Dubbing** and **HeyGen** and score them with the same metrics. Two reasons: they set a realistic quality target, and when a reviewer asks "how close is this to what you can buy", you have the number. Treat them as an upper bound to close on, not as competitors — they are closed cascades over roughly the same component types, with their advantage in data, tuning and human review rather than architecture.

---

## 9. Experiment matrix

Every row: freeze the upstream fixture, vary one stage, score with one metric.

| # | Stage | Compare | Primary metric | Secondary |
| --- | --- | --- | --- | --- |
| E1 | Diarization | pyannote 3.1, WhisperX, DiariZen, Sortformer, **oracle** | DER, JER | speaker-turn boundary error (median ms) |
| E2 | Face detection + tracking | RetinaFace+ByteTrack, SCRFD+ByteTrack, MediaPipe+DeepSORT | track ID-switch rate | detection recall on profile/occluded faces |
| E3 | Active speaker detection | TalkNet, TS-TalkNet | frame-level ASD F1 | accuracy on 3+ visible faces |
| E4 | Speaker ↔ face assignment | Hungarian vs per-turn argmax | turn-level assignment accuracy | off-screen precision/recall |
| E5 | ASR | Indic-Transcribe Core, Flex, WhisperX, IndicConformer | WER, CER | **code-mix WER**; hallucination rate on music/silence |
| E6 | ASR render mode | native / mixed / romanized | downstream chrF++ after MT | term-retention rate |
| E7 | Forced alignment | MMS CTC, WhisperX | median + p95 boundary error | % within 20 ms |
| E8 | Translation | Indic-Translate, IndicTrans2, NLLB-200 | chrF++, COMET | term-retention rate |
| E9 | Code-mix path | Path A vs Path B | human naturalness (5-pt) | term retention, transliteration consistency, cost |
| E10 | Transliteration | Indic-Transliterate, IndicXlit, inline via MT | top-1 accuracy on 200 terms | cross-segment consistency |
| E11 | Duration model | fitted vs naive chars-per-second | MAE as % of target | violation rate at ±10% |
| E12 | TTS | Indic-Speak, IndicF5, CosyVoice 3, OpenVoice v2 | **round-trip WER** | duration accuracy, attribute match, MOS |
| E13 | Voice assignment | attribute+screen-time, embedding match, random | listener speaker-confusion rate | perceived cast diversity |
| E14 | **Voice conversion / expressive** | Seed-VC, OpenVoice v2, CosyVoice, none | speaker similarity **and** EER vs source | style/emotion retention, round-trip WER |
| E15 | Lip sync | MuseTalk 1.5, LatentSync 1.6, no-op | LSE-D, LSE-C | ArcFace identity, FVD, human sync MOS |
| E16 | End-to-end | best-of-each vs ElevenLabs vs HeyGen | human MOS (naturalness, sync, comprehension) | speech overlap ratio, round-trip WER |
| E17 | Oracle ablation | each stage replaced by ground truth in turn | Δ end-to-end MOS | tells you where the error actually lives |

**E14 is the study.** It is where the reframed research question lives: given a source performance and a target-language utterance, which published zero-shot method best preserves identity and speaking style? Report speaker similarity *and* EER against the source speaker — the second is more informative, because it measures identity **leakage** rather than identity match. Pair with an emotion-classifier score and F0 correlation for style, and round-trip WER for intelligibility. Expect the honest result: disentanglement is never complete, and cross-lingual conversion degrades intelligibility in the low-resource direction.

**E17 is the table that explains everything else.** Swap each stage for ground truth, measure the end-to-end gain, and you know where the next month should go.

---

## 10. Evaluation

### Golden set — build this first

**20 clips, 30–90 s each**, hand-annotated. Nothing in §9 is measurable without it.

| Axis | Values |
| --- | --- |
| Speakers | 1 / 2 / 3+ |
| Setting | talking head, lecture with slides, interview, panel |
| Audio | clean, background music, room noise |
| Source | monolingual English, already code-mixed English–Hindi |
| Faces | full-frontal, profile, occluded, off-screen narration |

Annotate: speaker turns with timestamps; verbatim transcript; word boundaries on a 5-clip subset; face boxes and active-speaker labels on a 5-clip subset. Roughly 3 days. Check inter-annotator agreement on 3 clips before doing the rest.

### Metric definitions

| Metric | What it measures | Note |
| --- | --- | --- |
| DER, JER | diarization error | standard; report both |
| Speaker-turn boundary error | median \|predicted − true\| at turn changes | predicts dub quality better than DER |
| ASD F1 | active speaker detection, frame level | report separately for 1, 2, 3+ visible faces |
| WER / CER | recognition | plus a separate code-mix WER |
| Boundary error | alignment | median and p95 in ms |
| chrF++, COMET | translation quality | against reference translations |
| **Term-retention rate** | of spans tagged `retain`, fraction present in output | custom; no standard MT metric captures it |
| **Round-trip WER** | re-ASR the synthesised audio, compare to intended text | best automatic TTS metric; catches G2P and transliteration failures |
| Speaker similarity | WavLM-TDNN cosine | identity match |
| **EER vs source speaker** | identity *leakage* after conversion | more informative than similarity alone |
| LSE-D, LSE-C | lip-sync, SyncNet | partly gamed — models train against this family |
| ArcFace cosine | identity retention through lip-sync | |
| Human MOS | naturalness, sync, comprehension | the arbiter |

**Standalone MOS predictors are not a gate here.** UTMOS and its relatives did not track human preference in the COMPASS benchmark, so they serve as a smoke test only.

**E15 includes the no-op baseline** — original video with new audio. A lip-sync model that does not beat it on human MOS is adding artefacts, not value.

---

## 11. Repository structure

```
dubbing/
├── manifest.py              # schema + validate_references()
├── pipeline.py              # stage runner, caching, resume, ablate()
├── registry.py              # adapter registration with licence metadata
├── stages/
│   ├── base.py
│   ├── media.py             # ffmpeg / pyav demux, mux
│   ├── separate.py          # htdemucs
│   ├── diarize/            {pyannote.py, whisperx.py, diarizen.py, sortformer.py, oracle.py}
│   ├── faces/
│   │   ├── detect/         {retinaface.py, scrfd.py, mediapipe.py}
│   │   ├── track/          {bytetrack.py, deepsort.py}
│   │   └── asd/            {talknet.py, ts_talknet.py}
│   ├── assign_faces.py      # hungarian over ASD scores
│   ├── embed/              {ecapa.py, pyannote_emb.py}
│   ├── asr/                {bodhan.py, whisperx.py, indicconformer.py}
│   ├── align/              {ctc_mms.py, whisperx.py}
│   ├── terms/              {termbase.py, ner.py, llm_tagger.py}
│   ├── translate/          {bodhan.py, indictrans2.py, nllb.py}
│   ├── translit/           {bodhan.py, indicxlit.py}
│   ├── duration.py
│   ├── voices.py            # voice assignment
│   ├── tts/                {bodhan.py, indicf5.py, cosyvoice.py, openvoice.py}
│   ├── vc/                 {seedvc.py, openvoice.py, none.py}
│   ├── mix.py
│   ├── lipsync/            {musetalk.py, latentsync.py, wav2lip.py, noop.py}
│   └── qc.py
├── clients/
│   ├── bodhan.py            # per-service config, shared transport, cost accounting
│   └── ratelimit.py
├── baselines/              {elevenlabs.py, heygen.py}   # reference comparisons
├── termbase/               {ml_lecture.tsv, general.tsv}
├── eval/                   {metrics.py, harness.py, report.py}
├── fixtures/                # frozen manifests at every stage boundary
├── configs/                {default.yaml, hindi.yaml, tamil.yaml, ablations/}
└── tests/
```

### Interfaces

```python
class Stage(Protocol):
    name: str
    reads:  list[str]   # manifest paths consumed
    writes: list[str]   # manifest paths produced
    def run(self, m: Manifest, media_dir: Path, cfg: dict) -> Manifest: ...
```

`reads` / `writes` are load-bearing: the runner builds cache keys from them and validates the whole stage DAG before spending a GPU-hour discovering that lip-sync needed face tracks nobody produced.

The registry carries a **licence field per adapter**, which each adapter writes into the manifest it produces.

```bash
python -m dubbing run input.mp4 --target hi --config configs/hindi.yaml
python -m dubbing stage tts --manifest fixtures/after_translate_hi.json --impl bodhan
python -m dubbing ablate lipsync --impls musetalk,latentsync,noop --set eval/golden/
```

---

## 12. Phased implementation

| Phase | Deliverable | Gate |
| --- | --- | --- |
| **P0** | Manifest, runner, caching, registry, Bodhan client, fixtures | tests green; no-op pipeline round-trips a manifest |
| **P1** | Vertical slice: single speaker, English → Hindi, no faces, no lip-sync | a 60 s clip plays intelligibly in Hindi |
| **P2** | Golden set annotated (20 clips) | inter-annotator agreement checked on 3 |
| **P3** | Diarization + speaker-attributed transcript · **E1, E5, E7** | multi-speaker transcript correct on the golden set |
| **P4** | Face detection, tracking, ASD, speaker↔face assignment · **E2, E3, E4** | correct face bound to each turn in panel clips |
| **P5** | Code-mix path decision · **E6, E8, E9, E10** | term retention >90%, path chosen with numbers |
| **P6** | Duration model + candidate ranking · **E11** | <15% of segments need >10% stretch |
| **P7** | TTS + voice assignment selection · **E12, E13** | best TTS chosen; voices distinguishable |
| **P8** | Lip sync integrated · **E15** | beats the no-op baseline on human sync MOS |
| **P9** | Voice conversion / expressive study · **E14** | the evaluation paper's main result |
| **P10** | Language expansion: Tamil, then Bengali/Marathi/Gujarati/Telugu | same thresholds per language |
| **P11** | End-to-end + commercial baselines · **E16, E17** | final results table |

P0–P1 run in parallel with P2 — annotation is human time, implementation is yours.

---

## 13. Risks

| Risk | Mitigation |
| --- | --- |
| Rate limits throttle ablation sweeps | Client adapts concurrency from `x-ratelimit-*` headers; self-host Bodhan weights for sweeps |
| Bodhan gives no word timestamps | The local CTC aligner supplies them; API timestamps are not a dependency |
| Diarization error puts the wrong voice on a line **and** animates the wrong face | E17 oracle ablation quantifies it; low-confidence segments route to review |
| ASD fails in panels with 3+ faces | E3 measures this explicitly by face count; fall back to `on_screen: false` rather than guessing |
| Path A code-mixing right on average, wrong on the terms that matter | Termbase override exists under both paths; E9 measures rather than assumes |
| Indic expansion overruns every duration budget | Expected, not an edge case — §7 is built for it; report violation rate every run |
| Lip-sync identity drift on profile and occluded faces | Tracker reports pose and occlusion; skip lip-sync on those segments rather than shipping an artefact |
| SeamlessExpressive turns out to have no Indic coverage | Verify at P0; if so, drop it from E14 and say so in the writeup — a negative finding worth recording |

---

## 14. Questions the benchmark answers

Not research proposals — the questions this project will have numbers for at the end:

1. **Which ASR is best per language**, and does Indic-Transcribe's code-switching advantage survive on real lecture audio with music underneath?
2. **Does the ASR render mode change downstream translation quality?** Native vs mixed vs romanized is a free knob nobody has characterised.
3. **Is explicit code-mix control worth its complexity**, or does asking the model directly get within noise?
4. **How far off is a fitted duration model**, and does candidate ranking beat naive time-stretching on human sync MOS?
5. **Which zero-shot VC method best preserves identity and speaking style for Indic dubbing** — the E14 study.
6. **Does lip-sync beat the no-op baseline** on real content, and at what resolution does it stop being worth the compute?
7. **How close does an open pipeline get to ElevenLabs and HeyGen**, and which stage accounts for the remaining gap?

---

## Appendix — references

| Component | Link |
| --- | --- |
| Bodhan console & API docs | https://console.bodhan.ai/ · https://console.bodhan.ai/api-docs/ |
| Bodhan open weights | https://huggingface.co/bodhan-ai |
| pyannote-audio | https://github.com/pyannote/pyannote-audio |
| WhisperX | https://github.com/m-bain/whisperX |
| DiariZen | https://github.com/BUTSpeechFIT/DiariZen |
| NeMo Sortformer | https://huggingface.co/nvidia/diar_sortformer_4spk-v1 |
| TalkNet-ASD | https://github.com/TaoRuijie/TalkNet-ASD |
| TS-TalkNet | https://github.com/Jiang-Yidi/TS-TalkNet |
| SpeechBrain | https://github.com/speechbrain/speechbrain |
| AI4Bharat (IndicTrans2, IndicXlit, IndicConformer) | https://github.com/AI4Bharat |
| Whisper | https://github.com/openai/whisper |
| CosyVoice | https://github.com/FunAudioLLM/CosyVoice |
| OpenVoice | https://github.com/myshell-ai/OpenVoice |
| Seed-VC | https://github.com/Plachtaa/seed-vc |
| Amphion / Vevo | https://github.com/open-mmlab/Amphion |
| MuseTalk | https://github.com/TMElyralab/MuseTalk |
| LatentSync | https://github.com/bytedance/LatentSync |
| Wav2Lip (non-commercial) | https://github.com/Rudrabha/Wav2Lip |
| Demucs | https://github.com/adefossez/demucs |
| open-dubbing (closest reference implementation) | https://github.com/Softcatala/open-dubbing |
| Methods survey for this project | https://claude.ai/code/artifact/2180d02f-eab4-4a10-be60-6238d0f63243 |