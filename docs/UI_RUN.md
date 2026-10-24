# A UI run, stage by stage

What `POST /api/jobs` actually sets, and which stage each setting governs.
Reference for anyone wondering why a run did — or did not — do something.

Generated from `ui/presets.py` and `ui/progress.py` on **2026-10-20** by
resolving a real submission (`preset=standard`, `target_language=hi`). The
tables are the code's own output, not a transcription.

---

## 1. How a submission becomes a run

```
    browser  ──POST /api/jobs──▶  ui/routes_create.py
                                        │  validates, allocates a job folder
                                        ▼
                                  ui/presets.build_kwargs(preset, …)
                                        │  ~30 kwargs; everything else keeps
                                        │  the pipeline's OWN default
                                        ▼
                                  scripts/ui_run_job.sh   (DETACHED)
                                        │  the only launcher exporting the
                                        │  libstdc++ preload — a bare Popen
                                        │  walks stage 9h into the cv2 trap
                                        ▼
                                  ui/runner.py ──▶ backend.pipeline.run_pipeline
                                        │
                   ProgressWriter on the ROOT logger scrapes the
                   `=== stage N ===` lines into <job>/progress.json
```

The UI never calls `run_pipeline` in-process. The job is detached, so closing
the browser — or restarting the server — does not stop it.

## 2. The presets

Three named points on a ~60-argument surface. They differ in **two** things
only: whether stems are separated, and what happens at lip-sync.

| preset | label | stages planned | stage 3 (stems) | lip-sync model |
|---|---|---|---|---|
| **draft** | Draft — audio only | 20 | skipped | — |
| **standard** | Standard — lip-synced | 22 | run | deferred to the face probe |
| **best** | Best — MuseTalk | 23 | run | musetalk |

`standard` plans 22 and runs 23: `lipsync_model` is deliberately **absent**
rather than None, because the model is chosen from the face size after stage
5. `plan_stages` therefore cannot see it, and `9h` is appended live when its
marker arrives. Writing `None` there would read as "lip-sync is off" to both
the planner and the pipeline.

> **Out of date from here down in two ways.** Submissions no longer choose a
> lip-sync model at all (2026-10-20): every preset ends at the mix and
> lip-sync is a separate `POST /api/jobs/{id}/lipsync` the operator starts
> after reviewing the dub. And since 2026-10-21 `plan_stages` appends `9h`
> unconditionally, carrying `deferred: True` when no model was configured,
> so the bar always shows all seven phases and the Lip sync node sits
> `pending` until that second submission opens it. The stage-by-stage table
> below is still correct about which marker lands in which group.

## 3. The stage plan for a default run

`preset=standard`, 22 planned. The right column is the progress bar's group —
the UI shows **seven** phases, not 22 rows.

| key | what the UI calls it | progress group |
|---|---|---|
| `fetch` | Fetching the video | Download video |
| `cut` | Trimming to the chosen part | Download video |
| `1` | Fetching the video | Download video |
| `2` | Extracting the audio | Download video |
| `3` | Separating music from speech | Download video |
| `4` | Working out who speaks when | Speaker annotation |
| `4b` | Tidying up the segments | Speaker annotation |
| `4c` | Telling the speakers apart | Speaker annotation |
| `5` | Finding faces | Speaker annotation |
| `6` | Working out who is on screen | Speaker annotation |
| `7` | Transcribing | Transcribe |
| `8` | Timing the words | Transcribe |
| `9` | Translating | Translate |
| `9b1` | Working out the tone | Speech generation |
| `9b2` | Listening for emotion | Speech generation |
| `9c` | Fitting the text to the timing | Translate |
| `9d` | Casting voices | Speech generation |
| `9e` | Speaking the translation | Speech generation |
| `9f1` | Timing the new speech | Speech generation |
| `9f1b` | Reading mouth shapes | Speech generation |
| `9f2` | Planning the timing | Speech generation |
| `9g` | Mixing the new soundtrack | Mix |

Stage `9b` (transliteration) and `9f` (voice conversion) are **not** in a UI
run. `9b` is absent because code-mixing is now the MT call's own job; `9f`
because neither upstream repo is installed here, and it is an *alternative*
to stage 9d rather than an addition — it would override the voice just cast.

## 4. Every setting a UI run sets

Everything not listed here keeps `backend/pipeline.py`'s own default, which
is deliberate: the form sets only what it means.

| kwarg | stage | value | note |
|---|---|---|---|
| `align_model` | 8 | `mms_fa` | forced alignment, regardless of ASR |
| `asd_model` | 6 | `talknet` | active-speaker detection |
| `asd_on_screen_threshold` | 6 | `0.3` | below this the turn is off-screen |
| `asr_language` | 7 | `en` | 'en' — NOT auto-detect |
| `asr_model` | 7 | `bodhan` | needs BODHAN_ASR_KEY |
| `attach_visemes_stage` | 9f1b | `True` | mouth shapes |
| `diarization_model` | 4 | `pyannote` | pyannote speaker-diarization-community-1 |
| `duration_adjust` | 9c | `True` | fit text length to the slot |
| `emotion_model` | 9b2 | `emotion2vec` | source emotion |
| `face_detector` | 5 | `retinaface` | only set when lip-sync is wanted |
| `lipsync_batch_size` | 9h | `10` | 10, not MuseTalk's 20 — VRAM headroom |
| `mix` | 9g | `True` |  |
| `mix_clip_source` | 9g | `raw` | 'raw' — avoids the double stretch |
| `mix_strategy` | 9g | `timing_plan` |  |
| `mt_model` | 9 | `bodhan` | needs BODHAN_MT_KEY |
| `mt_target_language` | 9 | `hi` | from the form |
| `mt_target_script` | 9 | `codemix` | code-mix is the MT call's own job |
| `mux_video` | 9g | `True` | write the dubbed audio back onto the video |
| `num_speakers` | 4 | None | None = let pyannote decide |
| `refine_speaker_labels` | 4c | `True` | split labels holding >1 person |
| `resegment_turns` | 4b | `True` | merge/split turns on sentence boundaries |
| `skip_stems` | 3 | `False` | False on standard/best, True on draft |
| `timing_params` | 9f2 | `max_tempo=1.5, min_tempo=0.6666666666666666, safety_margin_s=0.0, use_target_duration=True` | tempo 0.67–1.5, no safety margin |
| `timing_producer` | 9f2 | `naive_slot` | per-clip slot fitting |
| `tts_align_model` | 9f1 | `mms_fa` |  |
| `tts_apply_stretch` | 9f2 | `False` | False — the timing plan fits instead |
| `tts_force_align` | 9f1 | `True` | re-time the synthesised speech |
| `tts_model` | 9e | `bodhan` | needs BODHAN_TTS_KEY |
| `tts_style` | 9b1 | `auto` | 'auto' = classify register from metadata |
| `viseme_backend` | 9f1b | `mms_char` |  |
| `voice_engine` | 9d | `indic-speak` | the voice CATALOG, not the engine |

## 5. What the operator actually chooses

| field | default | effect |
|---|---|---|
| `preset` | `standard` | the table in §2 |
| `target_language` | *required* | `mt_target_language` |
| `source_language` | `en` | `asr_language`. **Not** auto-detect — `DUBBING_UI_SOURCE_LANGUAGE=auto` restores per-turn detection, which once transcribed one lecture into four different Indic scripts |
| `style` | None | an explicit register REPLACES `tts_style="auto"`, so stage 9b1 is not planned and not run |
| `tts` | install default (`bodhan`) | also picks `voice_engine` via `voice_engine_for` |
| `num_speakers` | None | let pyannote decide |
| `lipsync_model` | preset's | `"off"`, `None` and `"auto"` are three different answers |
| clip window | whole video | `start_s`/`end_s`, cut before the pipeline sees it |

## 6. Things that surprise people

**Every hosted stage is one vendor.** ASR, MT and TTS are all Bodhan, so an
outage takes the whole run rather than most of it — and these endpoints answer
`internal_error` 5xx, not 429. There is no automatic fallback;
`DUBBING_UI_ASR_MODEL=whisperx` is the manual one. ASR also needs its own
`BODHAN_ASR_KEY`, separate from the MT and TTS keys.

**A TTS outage still reports green.** `bodhan_adapter.synthesize` catches per
request, so every segment comes back with `tts_error` set and stage 9e
succeeds. The signal is `extra.tts.errors == len(segments)`, not the exit code.

**`asr_language` is pinned to `en`,** because this project dubs English into
Indic languages and per-turn detection produced one lecture in four scripts.

**`tts_apply_stretch=False` is not optional.** `backend/tts.py` fits each clip
to its slot and the timing plan fits it again; leaving it on stretches every
clip twice to the same target. `mix_clip_source="raw"` is the other half.

**`lipsync_batch_size=10`, not MuseTalk's own 20.** 20 measured a 15496 MiB
peak on a 16311 MiB card — 815 MiB of headroom, and an OOM as soon as anything
else is resident, such as the ~460 MiB the onnxruntime CUDA EP holds after
face detection. Batch size moves throughput and VRAM only, not output pixels.

**The clip is cut BEFORE the pipeline runs,** so `download_youtube` is never
reached and `source_metadata` would be empty — which used to make stage 9b1
abstain on exactly the runs with the richest metadata.
`_recover_source_metadata` picks the sidecar `.meta.json` up instead.
