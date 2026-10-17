# Component-Combination Results — Translated Audio Generation

Scope: the **audio** half of the dubbing pipeline — diarization → ASR → forced
alignment → MT → transliteration/code-mix → duration planning → TTS → mix.
Lip-sync (Wav2Lip / MuseTalk / LatentSync) is a separate track and is **not**
covered here.

All numbers below were re-derived from the artifacts on disk
(`data/mt/*.mt.json`, `data/asr/*.asr.json`, `data/eval/*.json`, `data/mix/*.wav`,
`logs/*.log`) rather than copied from any earlier summary. Where a number could
not be measured, the cell says so instead of guessing.

---

## 1. The component grid

Registered adapters (`models/<stage>/_REGISTRY_SPEC`):

| Stage | Options available | Exercised end-to-end |
|---|---|---|
| Diarization | `pyannote`, `nemotron`, `diarizen` | `pyannote`, `nemotron` (text-only) |
| ASR | `whisperx` (small/medium/large-v3), `bodhan` (indic-transcribe), `indicconformer` | `whisperx-large-v3`, `bodhan` |
| Alignment | `whisperx`, `mms_fa` | both |
| MT | `indictrans2` (200M dist), `nllb` (600M dist), `bodhan` (indic-translate) | `indictrans2`, `bodhan` |
| Transliteration | `bodhan`, `indicxlit` | `bodhan` only |
| TTS | `edge`, `bodhan` (indic-speak), `indicf5` | `edge`, `bodhan` |
| Voice conversion | `openvoice`, `vevo` | **none** — `data/vc/` is empty |
| Mix strategy | `global_tempo`, `no_tempo` (+ experimental fit1/fit2/fit3a/fit3b) | all |

Target languages attempted: **hi** and **ta**. Only **hi** ever reached audio;
every Tamil run stops at the MT-text stage.

---

## 2. Coverage: what actually produced translated audio

142 job manifests exist under `data/jobs/`, but they are not 142 dubs.

| Depth reached | Count | Notes |
|---|---|---|
| Full grid sweep, **MT text only** | 128 | `*.sweep.*` — no TTS, no mix. Transcripts are corrupted (§7). |
| Clean-path MT text only | 8 | `nsIMQP0QhpE_60s.{nllb,indictrans2,bodhan,whisperx_bodhan}.{hi,ta}` |
| Reached TTS + mix (**dubbed audio**) | **5** | table in §3 |
| Reached duration planning but TTS failed | 1 | `2vxQaT0Fy-c_5min.bodhan.hi` |

So the honest headline: **five component combinations have ever produced a
translated audio track.** Everything else is text-stage evidence.

---

## 3. End-to-end runs that produced dubbed audio

| Clip | Diar | ASR | Align | MT | Translit | TTS | Segs | Slot (s) | Raw TTS (s) | Overflow | Budget violations | Mean stretch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `2vxQaT0Fy-c_1min` | pyannote | whisperx L-v3 | whisperx | indictrans2 | — | edge | 4 | 58.34 | 77.16 | **1.32×** | 3/4 | 1.098 |
| `2vxQaT0Fy-c_2min` | pyannote | bodhan | mms_fa | bodhan | bodhan (10 terms) | edge | 5 | 117.59 | 134.81 | 1.15× | 4/5 | 1.077 |
| `2vxQaT0Fy-c_60to80s` | pyannote | whisperx L-v3 | whisperx | indictrans2 | — | edge | 1 | 20.05 | 27.62 | **1.38×** | 1/1 | 1.100 |
| `nsIMQP0QhpE_60to80s` | pyannote | whisperx L-v3 | whisperx | indictrans2 | — | edge | 4 | 17.60 | 20.59 | 1.17× | 3/4 | 1.063 |
| `NAG_deeplearning_course` | pyannote | bodhan | — | bodhan | — | **bodhan** | 7 | 27.56 | 39.08 | **1.42×** | 7/7 | 1.071 |

*Slot* = sum of diarization-turn durations. *Raw TTS* = sum of un-stretched
synthesized clip lengths (measured from `data/tts/*/raw/*.wav`). *Overflow* =
raw TTS ÷ slot. *Budget violation* = segment that still exceeds its slot after
the ±10 % WSOLA stretch cap.

### What this says

1. **Every combination overflows its time budget.** The best case is 1.15×, the
   worst 1.42×. The ±10 % stretch cap recovers at most 10 points of that, so
   **18/21 segments across all five runs ended up over budget.** Duration fit is
   the binding constraint on this pipeline, not translation quality.
2. **TTS backend dominates the overflow, not MT.** The `bodhan` / indic-speak run
   has the *lowest* text expansion of the five (0.963 char, 1.052 word) yet the
   *highest* audio overflow (1.42×) — indic-speak simply speaks slower
   (≈11–13 chars/s) than its text length predicts. Edge-TTS runs land at
   1.15–1.38× from comparable text.
3. **Character expansion is a poor predictor of audio overflow.** Across the five
   runs, char expansion spans 0.906–1.137 while overflow spans 1.13–1.42, and the
   ranking does not match. The duration model in `backend/duration.py`
   (`a·chars + b·words + c`) is still running on **unfitted default coefficients**
   (`n_fit: 0, mae_s: 0.0` in every manifest), which is the direct cause.
4. **Short single-segment clips are the worst case.** `2vxQaT0Fy-c_60to80s` is one
   20 s turn that synthesizes to 27.6 s — there is no neighbouring silence to
   absorb the excess, so the whole error lands on screen.

### Stage timings (wall clock, from logs)

| Stage | 2 min clip (bodhan ASR+MT) | 1 min clip (whisperx+indictrans2) | 20 s clip |
|---|---|---|---|
| Stem separation (HTDemucs) | 7 s | 0 s (cached) | 8 s |
| Diarization (pyannote) | 12 s | 11 s | 23 s |
| ASR | 18 s (bodhan API) | 8 s (whisperx) | 26 s (whisperx, cold) |
| Alignment | 4 s (mms_fa) | 2 s (whisperx) | 1 s |
| MT | 7 s (bodhan API) | 4 s (indictrans2) | 7 s |
| Code-mix/translit | 6 s | — | — |
| TTS | 18 s (edge) | 12 s (edge) | 4 s |
| Mix | 3 s | 1 s | 1 s |
| **Audio total** | **~75 s** | **~38 s** | **~70 s** |

Audio generation is cheap. For reference, the lip-sync stage on the same 2 min
clip took **3316 s** — 44× the entire audio pipeline.

---

## 4. Mix / duration-fit strategy — the one comparison with a clear winner

Six strategies were rendered over the same TTS output. Drift = dubbed length vs
source video length (measured directly from the WAV headers).

| Clip (src len) | fit3a (shift, no tempo) | fit3b (tempo then shift) | **global_tempo** |
|---|---|---|---|
| `2vxQaT0Fy-c_1min` (60.08 s) | 77.69 s · **+29.3 %** | 61.79 s · +2.8 % | 60.48 s · **+0.7 %** |
| `2vxQaT0Fy-c_2min` (120.00 s) | 135.34 s · +12.8 % | 120.58 s · +0.5 % | 120.51 s · **+0.4 %** |
| `2vxQaT0Fy-c_60to80s` (20.00 s) | 28.16 s · +40.8 % | 27.65 s · +38.3 % | 20.43 s · **+2.2 %** |
| `nsIMQP0QhpE_60to80s` (20.00 s) | 22.32 s · +11.6 % | 21.05 s · +5.2 % | 20.54 s · **+2.7 %** |
| `NAG_deeplearning_course` (30.00 s) | 40.42 s · +34.7 % | 32.44 s · +8.1 % | 30.29 s · **+1.0 %** |
| `2vxQaT0Fy-c_5min` (300.00 s) | 15.29 s · **−94.9 %** ✗ | 300.55 s · +0.2 % | 15.29 s · **−94.9 %** ✗ |

Four earlier variants on `NAG_deeplearning_course` (30.00 s source):

| Variant | Length | Drift |
|---|---|---|
| `fit1_hardclamp` | 30.43 s | +1.4 % |
| `fit2_cascade` | 30.24 s | +0.8 % |
| `fit3_shift_notempo` | 40.42 s | +34.7 % |
| `fit3b_tempo_then_shift` | 31.34 s | +4.5 % |

**Verdict: `global_tempo` wins on every clip that has complete TTS input**, holding
drift to +0.4 % … +2.7 % versus +11.6 % … +40.8 % for pure shift-forward. The
earlier `fit1_hardclamp` and `fit2_cascade` were also within ~1.5 %, but they clamp
per segment rather than re-pacing globally, so they pay for it in intra-segment
distortion. `global_tempo` is already the CLI default (`--mix-strategy`).

The `2vxQaT0Fy-c_5min` row is **not** a strategy failure: that run's TTS stage
died (§6), so only ~15 s of segments existed to mix. `fit3b` on that clip is a
300.55 s file of mostly silence, not a 300 s dub. Treat the whole row as void.

---

## 5. MT model comparison

Same clip, same ASR, same 11 segments (`nsIMQP0QhpE_60s`, clean non-sweep path):

| MT model | → hi char exp | → hi word exp | → ta char exp | → ta word exp | Smoke latency (11 segs) |
|---|---|---|---|---|---|
| `nllb` (nllb-200-distilled-600M) | **0.965** | **1.031** | 1.199 | 0.681 | 8.20 s (hi) / 4.49 s (ta) |
| `indictrans2` (en-indic-dist-200M) | 0.990 | 1.055 | 1.170 | 0.638 | **3.81 s (hi) / 3.35 s (ta)** |
| `bodhan` (indic-translate, API) | 0.989 | 1.049 | **1.127** | 0.638 | not instrumented |
| `bodhan` on whisperx transcript | 0.959 | 1.005 | 1.125 | 0.643 | not instrumented |

- For **Hindi**, all three models land within 2.5 % of each other on character
  length. There is no length-based reason to prefer one.
- For **Tamil**, the spread matters: NLLB expands 20 % by characters, bodhan only
  12.7 %. Since Tamil never reached TTS, this is a prediction, not a measurement.
- `indictrans2` is the fastest local model and is the only one with both low
  latency and no API dependency.
- All MT adapters produced **11/11 non-empty translations** on the smoke test
  (`data/mt/_smoke_report.json`) — no silent dropouts.

First-segment output for the same English input, by way of qualitative sample:

| Model | Hindi output |
|---|---|
| nllb | एक बंद प्रणाली की आंतरिक ऊर्जा एक बंद प्रणाली के बराबर होने जा रहा है |
| indictrans2 | एक बंद प्रणाली की आंतरिक ऊर्जा में एक प्रणाली के लिए समान होने जा रहा है |
| bodhan | किसी प्रणाली की आंतरिक ऊर्जा में, एक बंद प्रणाली की, के बराबर होने जा रहा है। |
| bodhan (whisperx src) | एक बंद प्रणाली की आंतरिक ऊर्जा ऊष्मा के बराबर होने वाली है। |

All four carry the source's own truncation ("…is going to be equal to the") into
the target; none hallucinates a completion. bodhan is the only one emitting
sentence-final danda punctuation, which matters for TTS prosody.

---

## 6. ASR model comparison

Same 60 s clip, same 11 diarization turns:

| ASR | Words | Chars | Note |
|---|---|---|---|
| `whisperx` large-v3 | 163 | 832 | baseline, mixed case + punctuation |
| `bodhan` (indic-transcribe) | 163 | 818 | identical word count; inconsistent casing (some segments ALL CAPS) |
| `indicconformer` 600M multilingual | 158 | 701 | **transcribes English speech into Devanagari script** (इन द इंटरनल एनर्जी…) |

`indicconformer` is unusable for an English→Hindi pipeline as configured: it
romanizes-in-reverse, producing Devanagari phonetic English that the MT stage then
has to translate from the wrong script.

Whisper model size (from the sweep ASR, deduplicated):

| Clip | small | medium | large-v3 |
|---|---|---|---|
| `nsIMQP0QhpE_60s` | 123 words | 108 words | **163 words** |
| `nsIMQP0QhpE_10min` | 1635 words | 610 words | **1687 words** |

**`whisperx-medium` is a sharp regression, not a midpoint** — it recovers 34 %
fewer words than large-v3 on the 60 s clip and 64 % fewer on the 10 min clip, and
is *worse than small*. Whatever is wrong with the medium checkpoint in this
environment, it should not be offered as a quality/speed tradeoff until it is
investigated. Use large-v3; fall back to small, never medium.

---

## 7. Components that made no difference

### Forced alignment: zero effect on the translated audio

Across all 44 (clip × diarizer × ASR × MT × language) groups in the sweep, the
three aligner settings `whisperx`, `mms_fa`, and `none` produced **byte-identical
MT output** — same segment count, same character count, same expansion ratio.
**0 of 44 groups differed.**

This is expected from the architecture: alignment writes word-level timings, and
nothing downstream of MT currently reads them. The word timings are produced and
then discarded before TTS. Either wire them into duration planning or stop paying
for the stage (it costs 1–4 s per clip, so this is a correctness question, not a
cost one).

### Diarization backend: changes turn count, not transcript

On `nsIMQP0QhpE_60s`, `pyannote` yields 11 turns and `nemotron` yields 26 — but the
deduplicated transcripts are character-identical (163 / 108 / 123 words for
large-v3 / medium / small under *both* backends). Diarization changes how the
audio is sliced for TTS, not what gets said. Its real effect is on per-segment
slot length and therefore on overflow, which no completed run measured for
`nemotron`.

---

## 8. Failure modes by component

Recorded in the logs, with the exact error:

| Component | Failure | Evidence |
|---|---|---|
| `bodhan` TTS | `Bodhan TTS failed after 10 retries` on 4 of 20 segments; run abandoned | `logs/run_2vxQ_20261003T025957.log` |
| `bodhan` ASR | `Bodhan ASR failed after 1 retries` / `after 5 retries` — two consecutive runs lost at stage 7 | `logs/run_2vxQ_20261003T032942.log`, `…T033503.log` |
| `pyannote` | `huggingface-hub==2.1.1` incompatible (needs `>=0.34,<1.0`) | `logs/run_nobodhan_1min_…T143006.log` |
| `indictrans2` | `ModuleNotFoundError: IndicTransToolkit` | `…T143154.log` |
| `indicxlit` | `ImportError: requires ai4bharat-transliteration` | `…T143447.log` |
| `retinaface` | missing `insightface` / `retina-face` | `…T143059.log` |
| torchaudio FFmpeg | `libavutil.so.56/57/58` not found — non-fatal, logged on every run | all runs |

The two bodhan API stages (ASR and TTS) are the only components that failed
*intermittently* rather than deterministically. The remaining failures are
one-time environment gaps that were subsequently fixed.

---

## 9. Data-quality caveats — read before quoting any number above

### 9a. The 128-run sweep is built on duplicated transcripts

Every `data/asr/*.sweep.*.asr.json` repeats the same text across consecutive
diarization turns:

| Sweep ASR file | Segments | Unique texts | Word inflation |
|---|---|---|---|
| `nsIMQP0QhpE_60s.sweep.pyannote.whisperx-largev3` | 11 | 3 | 4.06× |
| `nsIMQP0QhpE_60s.sweep.nemotron.whisperx-medium` | 26 | 3 | 10.68× |
| `nsIMQP0QhpE_10min.sweep.pyannote.whisperx-largev3` | 119 | 27 | 4.33× |
| `nsIMQP0QhpE.sweep.pyannote.whisperx-largev3` | 440 | 85 | 4.98× |

The clean (non-sweep) path has **zero** duplicate segment texts on the same
clips, so this is a defect in the sweep harness — it appears to broadcast a
whole-audio transcript across every turn instead of slicing per turn. All 128
sweep MT outputs are therefore translations of 4–10× duplicated input. Their
*expansion ratios* are still internally comparable (numerator and denominator are
inflated together), which is why §5–§7 use them, but their absolute segment and
word counts are meaningless and none of them should be synthesized.

### 9b. The ASR/MT eval scores in `data/eval/` are measured against a broken reference

| Eval | Score | Verdict |
|---|---|---|
| `nsIMQP0QhpE_60s.whisperx.en.asr_eval` | WER 3.304 (330 %), 119 insertions vs 46 ref words | **invalid** |
| `nsIMQP0QhpE_60s.bodhan.en.asr_eval` | identical figures to whisperx | **invalid** (and suspiciously identical — likely the wrong `asr_json` was scored) |
| `nsIMQP0QhpE_60s.indicconformer.hi.asr_eval` | WER 1.340, CER 0.934 | invalid, and cross-script anyway |
| `nsIMQP0QhpE_60s.bodhan.hi.mt_eval` | BLEU 16.63, chrF2 32.66 | partially valid, see below |
| `nsIMQP0QhpE_60s.bodhan.ta.mt_eval` | BLEU 24.31, chrF2 42.65 | partially valid |

The reference `nsIMQP0QhpE_60s.en.vtt` holds **3 cues / 46 words** timed
00:30.3 → 01:12.6, while the clip's actual audio is 60 s of different content
(the first ASR segment begins at 0.03 s with "in the internal energy of a
system…"). The reference is the untrimmed, un-offset head of the full video's
subtitle file. The parent `nsIMQP0QhpE.en.vtt` is itself degenerate — 30 cues for
a 1955 s video, with one cue spanning 10:35 → 32:31.

Consequence: **WER/CER/BLEU here measure subtitle misalignment, not model
quality.** The Hindi MT evals score higher only because the Hindi subtitle track
is denser (181 cues) and its third cue happens to overlap the clip's content. No
MT eval exists at all for `nllb` or `indictrans2` — only `bodhan` was ever scored.

### 9c. Other gaps

- **No perceptual or intelligibility metric was ever computed** on the dubbed
  audio — no MOS, no round-trip WER on the synthesized Hindi, no speaker-similarity
  score. Every "quality" claim above is a proxy (length, violations, drift).
- **Voice conversion never ran.** `openvoice` and `vevo` are registered but
  `data/vc/` is empty, so all dubs use a generic TTS voice regardless of the
  source speaker. Speaker references *were* extracted (`data/voices/*.refs`) and
  then unused.
- **`indicf5` TTS never ran.** Only `edge` and `bodhan` have output directories.
- **Tamil never reached audio.** All `ta` evidence is text-stage.

### 9d. An oracle baseline exists and should be used

`scripts/fetch_indic_audio.py` pulls the official YouTube Hindi dub track, and
several artifacts compare against it (`data/audio/NAG_deeplearning_course.downloaded_hi.wav`,
`data/mix/NAG_dIYDPtHWBNA_80to100_rawhi.dubbed.wav`,
`data/indic_audio/{dIYDPtHWBNA,svwrTOm55y4,uL2SzOMvzUY,fsTRJeWpZ_o}.hi.*`). This is
a professionally-dubbed reference for the *same* video — the right reference for
both MT scoring and TTS naturalness. No scored comparison against it has been run
yet; the artifacts so far feed the lip-sync track only.

---

## 10. Recommended stack, on current evidence

| Stage | Pick | Why |
|---|---|---|
| Diarization | `pyannote` | only backend with a completed audio run; `nemotron` over-segments 2.4× with no transcript gain |
| ASR | `whisperx` **large-v3** | +34 % word recall over medium, +33 % over small; local, no API flakiness |
| Alignment | either (or skip) | provably no effect downstream (§7) |
| MT | `indictrans2` for hi | fastest local model, length parity with nllb/bodhan, no API dependency |
| MT (ta, when it ships) | `bodhan` | lowest Tamil char expansion (1.127 vs 1.199 nllb) |
| TTS | `edge` | 1.15–1.38× overflow vs 1.42× for indic-speak, and no API retry failures |
| Mix | `global_tempo` | +0.4 %…+2.7 % drift vs +11.6 %…+40.8 % for shift-only |

Avoid: `whisperx-medium` (§6), `indicconformer` for English source (§6),
`bodhan` ASR/TTS where a run must not fail (§8).

---

## 11. What to run next, in priority order

1. **Fix the sweep ASR duplication bug** (§9a) and re-run the grid. 128 of the 142
   jobs are currently non-evidence.
2. **Rebuild the eval references** (§9b) — clip the VTT to the extracted window and
   apply the time offset, or drop the YouTube subtitles and score against the
   official Hindi dub track instead (§9d).
3. **Fit the duration model.** Every manifest shows `n_fit: 0`. One calibration run
   per (language, voice) over the existing `data/tts/*/raw/*.wav` pairs would give
   real coefficients and should move overflow materially — this is the single
   highest-leverage fix for translated audio.
4. **Take one combination all the way to Tamil**, to convert §5's Tamil predictions
   into measurements.
5. **Run voice conversion once.** The speaker references already exist; the stage
   has simply never been invoked.
6. **Add one perceptual metric** — round-trip ASR WER on the dubbed Hindi is the
   cheapest honest proxy and needs no human raters.

---

## 12. Regenerating these numbers

| Table | Source |
|---|---|
| §3 run table, §5 expansion | `data/mt/*.mt.json` → `segments[].{text,mt_text,final_text,tts_duration_s,stretch_factor}`, `extra.{tts,duration,codemix}` |
| §3 raw TTS seconds | WAV headers of `data/tts/<base>.<tts>/raw/SEG_*.wav` |
| §3 stage timings | `=== stage N: … ===` timestamps in `logs/run_*.log`, `logs/dub_*.log` |
| §4 drift | WAV headers of `data/mix/*.wav` vs `ffprobe` duration of `data/videos/*.mp4` |
| §5 latency | `data/mt/_smoke_report.json` |
| §6, §7, §9a | `data/asr/*.asr.json`, deduplicated on `segments[].text` |
| §8 | `grep -iE "ERROR|Traceback|failed" logs/*.log` |
| §9b | `data/eval/*.json` vs `data/videos/*.vtt` cue timings |

---

# 13. Timing-alignment methods — the eight-method sweep

Everything in §1–§12 above measured *which models* to use. This section measures *how to place
the synthesized audio on the timeline*, against the human Hindi dub as timestamped ground truth.

**The governing constraint is not "does the dub fit the slot".** It is **where speech starts and
stops, and where the pauses are.** Final-duration delta is therefore not a ranking metric — several
arms force it to ~0 by construction.

## 13.1 What was run

| | |
|---|---|
| Clips | 15 (11 tier-A ranking, 4 tier-B generalization). Tier-C (3 TED clips) excluded: no Hindi dub. |
| Arms | 10 per clip = **150 pipeline runs, 150 rendered mixes, 0 failures** |
| Scored | 150 arm-clip results (`data/eval/sweepA`, `data/eval/sweepB`) |
| Wall clock | ~95 s/clip incl. prefix; 2 TTS passes per clip, not 10 |
| Suite | `experiments/suites/eight_methods.json` |

Arms: control (`naive_slot`), baseline (`global_tempo`, what ships), M1 engine speed, M2 post-TTS
stretch, and M3/M4/M7 each **paired with and without M5** (post-TTS forced alignment).

## 13.2 The headline result: the tempo cap is binding, and it decides everything

Measured over every piece in every plan on disk:

| arm | median applied tempo | cap | % of pieces pinned AT the cap |
|---|---|---|---|
| `m3_pause_aware` | 1.150 | 1.15 | **100 %** |
| `m4_word_anchored` | 1.150 | 1.15 | 94 % |
| `ctl_naive_slot` | 1.100 | 1.10 | 92 % |
| `m1_engine_speed` | 1.100 | 1.10 | 88 % |
| `m4_word_anchored_m5` | 1.150 | 1.15 | 87 % |
| `m2_post_tts_stretch` | 1.500 | **1.50** | 68 % |
| `m7_dp` | 1.150 | 1.15 | 66 % |

**Every arm's median tempo equals its own cap.** The planners have no freedom left: the Hindi TTS is
physically longer than the English speech it must cover (1.21–1.65× measured on real segments), so
every piece is compressed as hard as it is allowed and is *still* too long. Once the cap binds, the
no-overlap cursor sets the onsets, not the anchor.

The consequence is visible in two independent places:

**(a) Slot overrun.** Median over 15 clips, dub duration vs the 60 s source:

| arm | overrun | | arm | overrun |
|---|---|---|---|---|
| `base_global_tempo` | **+0.6 %** | | `m4_word_anchored` | +20.6 % |
| `m2_post_tts_stretch` | **+2.3 %** | | `m7_dp` | +20.8 % |
| `m1_engine_speed` | +10.4 % | | `m3_pause_aware` | +22.2 % |
| `m4_word_anchored_m5` | +16.7 % | | `ctl_naive_slot` | +23.8 % |

The only two arms that stay inside the slot are the only two permitted to compress hard — the
baseline (one global `atempo` up to 1.5×) and M2 (per-segment rubberband up to 1.5×).

**(b) Pause structure.** Only **7 of 150** cells score a non-zero pause F1 — and **M2 holds the top
four**:

| pause F1 | clip | arm | hyp pauses / ref pauses |
|---|---|---|---|
| 0.421 | `ZPUtA3W-7_I_09880to09940` | `m2_post_tts_stretch` | 6 / 13 |
| 0.211 | `ZPUtA3W-7_I_03800to03860` | `m2_post_tts_stretch` | 5 / 14 |
| 0.133 | `ZPUtA3W-7_I_07700to07760` | `m2_post_tts_stretch` | 5 / 10 |
| 0.111 | `qjPH9njnaVU_01180to01240` | `m2_post_tts_stretch` | 2 / 16 |
| 0.105 | `6-NRiom8K9Y_00005to00065` | `m7_dp` | 5 / 14 |
| 0.091 | `ZPUtA3W-7_I_07160to07220` | `ctl_naive_slot` | 2 / 20 |
| 0.077 | `6-NRiom8K9Y_00005to00065` | `m4_word_anchored_m5` | 12 / 14 |

**M2 is the crudest method in the set — it just stretches harder — and it is the only one that
reliably reproduces any pause structure at all.** Not because it places better (it uses the control's
naive placement), but because compressing to 1.5× is the only thing that leaves room for a gap.
Sophisticated placement capped at 1.15× cannot leave room that does not exist.

## 13.3 Structure metrics (the ones that survive)

Median over 11 tier-A clips. `runs` = distinct speech runs the arm emits; the human dub has a
median of **11** (range 6–21). `recall` = fraction of reference speech runs the arm placed anything
against.

| arm | runs | recall | speech IoU | pause F1 |
|---|---|---|---|---|
| `m2_post_tts_stretch` | 3 | **0.333** | 0.874 | 0.000 |
| `m4_word_anchored_m5` | 4 | 0.214 | 0.837 | 0.000 |
| `m3_pause_aware` | 4 | 0.190 | 0.832 | 0.000 |
| `m4_word_anchored` | 4 | 0.190 | 0.847 | 0.000 |
| `m7_dp_m5` | 3 | 0.190 | 0.847 | 0.000 |
| `m7_dp` | 2 | 0.182 | 0.866 | 0.000 |
| `m3_pause_aware_m5` | 3 | 0.167 | 0.842 | 0.000 |
| `ctl_naive_slot` | 1 | 0.143 | 0.880 | 0.000 |
| `m1_engine_speed` | 2 | 0.143 | 0.880 | 0.000 |
| `base_global_tempo` | **1** | **0.095** | 0.884 | 0.000 |

**What ships today emits one continuous speech blob** and places something against 9.5 % of the
human dub's speech runs. Every timing-plan method beats it on structure. Note that speech IoU runs
*backwards* to recall — the no-op floor is 0.872 and every arm sits near it, confirming IoU is a
guard, not a ranker.

Tier-B (4 Durov clips, never pooled into the ranking) reproduces the ordering: `m4_word_anchored_m5`
0.254 and `m2_post_tts_stretch` 0.231 lead; `base_global_tempo`, `ctl_naive_slot` and `m7_dp` tie at
0.081.

## 13.4 Paired bootstrap vs the shipping baseline

10 000 draws, resampled over **clips** (never segments — `mix.py`'s shift-forward cursor couples
segments within a clip). `*` = 95 % CI excludes zero.

| arm | Δ speech-run recall | Δ speech runs |
|---|---|---|
| `m2_post_tts_stretch` | +0.143 `[0.000, 0.333]` | +2 `[0, 5]` |
| `m4_word_anchored_m5` | +0.143 `[0.000, 0.333]` | +3 `[0, 4]` |
| `m3_pause_aware` | +0.095 `[0.000, 0.333]` | +2 `[0, 5]` |
| `m4_word_anchored` | +0.095 `[0.000, 0.273]` | +2 `[0, 5]` |
| `m7_dp_m5` | +0.091 `[0.000, 0.167]` | **+2 `[1, 3]` \*** |
| `m7_dp` | +0.071 `[0.000, 0.133]` | +1 `[0, 4]` |
| `ctl_naive_slot` / `m1_engine_speed` | +0.000 | +0 |

**At n=11, exactly one comparison is statistically significant**: `m7_dp_m5` emits more speech runs
than the baseline. Every other arm's CI touches zero. The honest reading is that several of these
methods are indistinguishable at this sample size, which is what the plan predicted.

## 13.5 The primary ranking metric could not be computed

`median |Δstart|` — the plan's designated primary ranker — **has n=1 paired clip against the
baseline and is therefore not reported.**

Why: when an arm places pieces back to back, the hypothesis merges into one speech span, the IoU
matcher pairs that blob with whichever reference run it most overlaps, and the "offset" becomes the
distance to an arbitrary run. 42 of 110 tier-A cells collapsed this way, with a median apparent
offset of **21,322 ms** against 2,953 ms for multi-span cells. Five-figure numbers that mean nothing.

`backend/eval/report.py` now requires **≥3 matched pairs** before an offset statistic may carry a
number (`MIN_MATCHED_FOR_OFFSET`); below that the cell states why. The baseline clears that bar on
only 1 of 11 clips — *because it collapses to one span on 7 of them*. So the metric is not broken;
it is correctly refusing to rank a degenerate hypothesis. The collapse is reported instead, as
`n_hyp_runs` and `speech_run_recall`, where it is genuinely discriminating (§13.3).

## 13.6 M5, M6, M8 — the three methods that had no call site

All three were built, individually tested, and **unreachable from the pipeline**. All three now have
call sites. Two of them produce negative results.

**M5 (post-TTS forced alignment) — ships as a paired factor, not a row.** Stage 9f1. Fires at 5/5
segments, 323 words, token coverage 1.0. Word-anchored goes from 5/5 segments degraded to 0/5 and
its onset p95 falls 6.86 s → 4.59 s. It also moves **M3**, which the original design did not
anticipate: `plan_pause_aware` reads `tts_words` too, so pairing only M4 and M7 would have silently
mixed two M5 states into the M3-vs-M4 comparison. In the sweep its effect is real but small —
`m4_word_anchored_m5` is the best M5 pair, `m3_pause_aware_m5` is *worse* than `m3_pause_aware`.
It cannot help an asset with no source `words[]`.

**M6 (viseme/phoneme alignment) — excluded, with evidence.** The plumbing works: `D_phoneme` returns
2.980 instead of `(0.0, 'no_visemes')`, λ>0 provably re-cuts the DP (2.46 → 0.90 → 0.60), and λ=0
with keys attached is byte-identical to M7, so an M6 arm would be genuinely distinct. Excluded for
two measured reasons: match_rate at zero desync is only 0.228–0.312, so most of the cost is skip
penalty rather than measured desync; and λ>0 makes `plan_dp` **23–24× slower** (a 5-segment file was
killed after 36 min). The shipped viseme cost was also **non-monotone** — a 2 s desync scored better
than a 100 ms one — and is now repaired.

**M8 (duration-aware translation) — excluded, with evidence. This is the clearest negative result.**
Fully live: 40/40 segments carry 3–4 candidates, top-1 preserved, `candidates_text` absent at
n_best=1 and present at 4, `final_text != mt_text` on 15/40. And it cannot do the job: 39/40 segments
are already over target, and **the candidate pool offers a median 1 character of shortening where 24
are needed** — 3.9 % of the gap, fully closing it on 1 of 39 segments. Widening `length_penalties`
50× moved the spread by 0.0 points. Diverse beam search is unreachable on this host
(`transformers` 4.57.6 requires `custom_generate` + network). M8 needs a different generator —
length-constrained decoding or a target-length logit processor — not more plumbing.

## 13.7 Recommendation on current evidence

1. **The binding constraint is the TTS length budget, not placement sophistication.** Every arm is
   pinned at its tempo cap. Until the synthesized Hindi is shorter, no placement method can leave
   room for a pause. This reverses the plan's working assumption.
2. **Raise the tempo cap on the placement methods before ranking them again.** M2 differs from the
   others almost entirely by being allowed 1.5× instead of 1.15×, and it leads on both structure
   metrics. M3/M4/M7 have never been measured with enough headroom to express what they do.
3. **Ship nothing from this sweep yet.** Every timing-plan arm overruns the slot by 17–24 %, which is
   worse than today's behaviour on the one axis users notice immediately.
4. **M4+M5 is the most promising placement method** (best structure metrics among the capped arms,
   reproduced on tier-B), but it is not separable from M3/M7 at n=11.

## 13.8 Caveats that bound every number above

- **Noise floor is 185 ms, not 80 ms.** The Hindi reference has **no word-level floor on any clip**:
  no video in the corpus carries an `hi-orig` auto-caption, so every Hindi word stream comes from a
  single producer (whisperx) with nothing to cross-check. 185 ms is the measurable substitute —
  cue-level disagreement between the manual `.hi.vtt` and the CTC onsets, 9 clips
  (`scripts/hi_cue_floor.py`). It is an **upper bound**, not an estimate. The 55 ms figure quoted
  elsewhere is the **English source** side and does not bound this ranking.
- **n = 11 tier-A clips for VAD metrics, 7 for word-level metrics.** Not the 12 the plan assumed.
- **Pause F1, the designated discriminator, is saturated at its no-op floor on 143 of 150 cells.**
  It discriminates only where an arm is allowed to compress enough to leave gaps.
- **One tier-A clip (`6-NRiom8K9Y_00265to00325`) has a single segment** and cannot rank anything.
  Six of 12 clips carry 1–2 degenerate (<0.1 s, empty-text) segments from diarization.
- **No intelligibility (round-trip WER) or MT metrics were run** in this sweep — `--groups
  timing,pauses,rate` only. The `intel` group needs whisperx over every mix.

## 13.9 Where to listen

`data/listen/<clip_id>/` — hard links, no extra disk. `01_GROUND_TRUTH_human_hindi_dub.wav` against
`02_SHIPS_TODAY_global_tempo.wav` is the gap this section is about. See `data/listen/README.md`.
