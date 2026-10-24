# Known rough edges

Moved out of CLAUDE.md on 2026-10-24. CLAUDE.md keeps the pointer; this file
holds the incident history.

---

## 4. Known rough edges

- **Three commits now exist.** `8c77641` ("Commit full pipeline work…")
  brought `backend/`, `models/`, `tests/` and `ui/` under version control, so
  the long-standing "only two commits; everything lives on this disk" warning
  no longer applies. `git status` is still busy — `data/` and a number of
  `scripts/` remain untracked — so check it rather than assuming either way.
- `extra_margin` is not plumbed through `models/lipsync/musetalk_adapter.py` —
  that is the knob controlling the bottom of the face crop, and `bbox_shift`
  (which is exposed) only moves the top edge.
- `data/jobs/` was emptied by the 2026-10-11 archive, which deleted the bulk
  media; `logs/` is the surviving record of older runs.
- **Stage 9c2 rewrites a line the stretch cannot fit, and it is ON for
  every UI run.** Turned on 2026-10-22 by request; the CLI still needs
  `--condense-overlong`. It reads 9c's per-segment estimates, so it needs
  `--duration-adjust`, and it needs `OPENROUTER_API_KEY` — without one it
  logs and leaves every line alone rather than failing the dub.

  **The gate is the speed-up the MIX can apply**, `est > target *
  --condense-speed-cap` (default **1.5**), not a flat overrun. A line
  needing 1.2x is left alone because `atempo` fits it; only one past the cap
  is worth a call. The previous 1.10x gate rewrote 4 of 5 segments on the
  svwrTOm55y4 run where only 3 ever clamped — rewriting text that already
  renders correctly is a translation risk for nothing. **An ordinary run
  therefore costs ZERO LLM calls**: the gate is evaluated from 9c's
  estimates before any request is made.

  **1.5 here is deliberately NOT the 1.8 `--speed-cap` now defaults to.**
  They answer different questions and were asked for separately: 1.8 is how
  hard the mixer is *allowed* to push, 1.5 is the point past which the
  author would rather re-word the line. Hindi TTS already runs 1.21–1.65x
  longer than the English it covers, so a gate at 1.8 would almost never
  fire. `--condense-speed-cap 1.8` makes them agree.

  **Segments go up in batches of 2, and the number is small on purpose.**
  Added 2026-10-23. The system prompt is 10.7 KB — ~2.9k tokens — and was
  re-sent once per segment, so N over-long lines paid for it N times.
  `--condense-batch-size` / `CONDENSE_BATCH_SIZE` / `batch_size=` (default
  **2**) sends several per request; the chunk is `min(batch_size,
  overlong)`, so one over-long line is still one call and the ordinary
  case is unchanged. `1` is exactly the pre-batch behaviour.

  **2 rather than 5 because the dangerous failure is the invisible one.**
  A dropped or duplicated segment is caught by the id mapping, and a
  rewrite that is too long is caught by the duration re-score. Neither can
  see a line that borrowed its NEIGHBOUR's wording: it is short, fluent,
  and in the wrong place. The prompt's "Segment isolation and identity"
  section forbids it outright, but a model attends less to each line the
  more of them it holds, so the batch stays small until a run has been
  read against its source.

  Two mechanics follow from that. **Mapping is by the echoed
  `segment_id`**, never by position — positional matching is allowed only
  when NO row carries an id and the count is exactly right; anything
  half-labelled raises `_BatchMismatch`. And **an unmappable, failed or
  partial batch is re-asked one segment at a time**, so batching can cost
  an extra call but never a segment. `extra.condense.calls` beside
  `considered` is the checkable form of "batching happened".

  **`max_tokens` is scaled with the chunk** (`900 × n`, floored at the
  client's 2048). Devanagari costs 2–3 tokens per akshara and the QC
  schema is 16 keys, so the flat default truncates a multi-segment reply
  into unparseable JSON — which loses the whole chunk rather than one
  line.

  **The prompt is `translate_agetn.prompt.txt`, read from disk, not a copy
  in the module.** The author wrote it; `_system_prompt()` fills in
  `{{target_language}}` and sends it verbatim, and
  `tests/test_condense.py` asserts distinctive lines from both halves of
  the file still appear, so the two cannot drift. A short fallback exists
  only for a checkout that lost the file. It takes `{"segments": [...]}`
  and answers in the same shape — **always an array, even for one
  segment**, because a request whose shape changes with the batch size is
  two contracts and the rare one breaks unnoticed.

  **It rewrites `mt_text`; the English is CONTEXT.** The prompt is explicit
  — "Rewrite `mt_text`, not `english_asr`... Use `english_asr` to verify the
  intended meaning, resolve mistranslations, and identify possible ASR
  errors." A first cut of this module retranslated from the source instead,
  which throws away a translation that is usually correct and merely long,
  and re-opens every choice the MT already made about register and
  code-mixing.

  **The segment goes as the prompt's own input contract** — `segment_id`,
  `english_asr`, `mt_text`, `target_duration_s`, `est_duration_s`,
  `protected_terms` (from `seg["retain_spans"]`, i.e. the code-mix terms)
  and `stretch_cap_pct`. That last one is **derived from the speed cap, 50,
  not the prompt's default of 10**: the prompt computes `fits` as
  `est <= target * (1 + pct/100)`, and here a line fits when the mix can
  stretch it. Left at 10 the model reports `fits: false` for lines that
  render perfectly well and then cuts them far harder than needed.

  **Its metrics are reported, never trusted.** The prompt says so of itself
  ("a proxy, not a guarantee"), so `duration.predict` re-scores the returned
  text and that is what decides. The QC fields it returns — `adequacy`,
  `meaning_loss`, `techniques`, `preserved_terms`, `asr_suspect`, `note` —
  are recorded per segment, and a `major_issue` or `asr_suspect` is logged
  at WARNING and collected in `extra.condense.flagged`: a translation fault
  the length pass noticed and could not fix would otherwise vanish.
  `infeasible` (the model's own "required cut exceeds 40%") ends the rounds
  for that segment, since a second ask cannot make an incompressible line
  compressible.

  Two safety properties are unchanged and load-bearing: the same duration
  model **re-scores** the rewrite and discards one that is not actually
  shorter (a model asked for brevity will cheerfully return something
  longer), and the original wording is kept in `seg["condense_backup"]`.
  `extra.condense` records the cap used, the QC fields per segment and
  `rate_before`/`rate_after`.

  `CONDENSE_MODEL` defaults to **`google/gemini-3.1-flash-lite`** (set by
  the author 2026-10-22, from `google/gemini-2.5-flash-lite`). **No slug in
  this family has been confirmed against the live catalogue from this
  host** — two attempts on 2026-10-22 returned a malformed id
  (`anthropic/google/gemini-3.8-flash[1m]`) and then a page with no Gemini
  rows. A wrong slug degrades to "no shortening", so it cannot break a run
  — which is exactly why it can stay wrong unnoticed. `extra.condense.error`
  and the Langfuse trace are where it shows. Confirm once with the curl in
  `backend/condense.py`'s header before trusting a dub's timing to it.

- **The stretch cap is ±0.20 (0.8x-1.2x) and widening it widened the
  DOUBLE-fit too.** Raised from 0.10 on 2026-10-20 by request. It is a
  per-clip cap, not a global speed: a clip already near natural pace is
  untouched, and only one that misses its slot by more than 20% gets pinned
  at the bound. The thing to watch is that the two layers below now each
  have twice the room, so a clip the TTS stage stretches to 1.2x can still
  meet a timing plan that fits it again.

- **Two layers can fit the same clip to the same slot, and both default on
  is a double stretch.** `backend/tts.py` stretches each clip to
  `target_duration_s` (±`--tts-max-stretch`, **default 0.20 since
  2026-10-20** — a ± FRACTION, so 0.20 clamps every clip to 0.8x-1.2x speed
  via `backend/stretch.py:378`'s `max(1-cap, min(1+cap, ratio))`; it was 0.10
  and was widened by request), and a timing plan
  fits it again in the mix. `--speed-mode segment` and `--timing-method`
  therefore both set `tts_apply_stretch=False` and `mix_clip_source=raw`.
  The default `global` path has its own version of this: `global_tempo` +
  `clip_source=auto` reads the already-stretched `tts_audio` and then applies
  a global `atempo` on top. That is long-standing behaviour, documented in
  `backend/mix.py`'s docstring, and `--mix-clip-source raw` is the way out.
- **Per-segment TTS overrides exist now, and only some engines honour them.**
  `backend/edits.py` writes `tts_voice`, `tts_style`, `tts_rate_override`,
  `tts_max_stretch` and `tts_reference_audio` onto individual segments, and
  `backend/tts.py` reads all five. But `style` reaches only the Bodhan adapter
  (`models/tts/bodhan_adapter.py` puts it in `instructions`), and `rate` only
  engines with `supports_rate = True` — since Edge was removed on 2026-10-17
  that is Bodhan alone, and IndicF5 reads neither. A tone edit on
  an IndicF5 run therefore changes the wording and nothing about the delivery.

  **Bodhan's rate field is an inference, and a checkable one.** The hosted
  docs (console.bodhan.ai/api-docs) list only `model`, `input`, `voice` and
  `instructions` — no speed parameter — so the adapter sends `speed`, from
  the OpenAI `/v1/audio/speech` shape this endpoint otherwise matches
  exactly. It is sent ONLY when the rate differs from 1.0, so an ordinary
  run's payload is unchanged and cannot be broken by a wrong guess. If the
  name is wrong the API answers 422 and the adapter says so by name,
  pointing at `BODHAN_TTS_RATE_FIELD` (alternatives: `rate`, or
  `instructions.speed` to nest it). `--tts-speed-control` output counts as
  validated once one rated call has returned 200 on this host.

  **Reachable from the CLI since 2026-10-17** — `--tts-speed-control
  {off,oneshot,closed_loop}` plus `--tts-speed-bounds LO,HI`,
  `--tts-speed-tolerance` and `--tts-speed-max-rounds`. It was
  `experiment.py`-only before, so the adapter's rate support had no pipeline
  caller at all. Default stays `off`, and at rate 1.0 the payload is
  byte-identical to a pre-rate run (verified: keys `input, instructions,
  model, voice`), so the undocumented field name cannot break a run that
  does not ask for a rate.

  Two bounds exist and they are not the same thing: `--tts-speed-bounds`
  (default 0.85–1.15) is what the *planner* is allowed to ask for, while
  `BODHAN_TTS_RATE_RANGE` (default 0.25–4.0) is the adapter's own clamp on
  whatever it is handed. A rate outside the second is clamped with a
  warning; the first never reaches it.

  `--tts-speed-control` does not disable the post-hoc WSOLA stretch, and
  that is deliberate — unlike the double-fit above, these converge on one
  target rather than compounding: the engine gets close, the stretch mops up
  the residual within `--tts-max-stretch`. `--no-tts-stretch` gives a clean
  method-1-vs-method-2 comparison, and the agent says so in its plan rather
  than letting it pass silently.
  `tests/test_edit_agent.py` asserts `backend/tts.py` still reads every key
  `backend/edits.py` writes — the cheap guard against this pair drifting
  back into the dead-code shape that put a female voice on a male lecture.
- **The default output resolution is 1080p, and `resize_factor` is resolved
  per source rather than fixed.** Changed 2026-10-22 by request, from the
  flat `resize_factor=2` of 2026-10-18. The rule, in
  `wav2lip_adapter.resolve_resize_factor`: a source at or below 1080p renders
  at its **native** resolution (factor 1); anything above it is **halved**
  (factor 2). 2160/2 is exactly 1080, which is why the ceiling is 1080 and
  why 2 is the only factor applied.

  **Why the old default was over-cautious.** It existed for one case — a 4K
  source, where `third_party/Wav2Lip/inference.py:118` raises `Image too big
  to run face detection on GPU` at stage 9h, after ASR, MT, TTS and the mix
  have been paid for (measured 2026-10-18: shard 1 of a 2-GPU split died
  there while shard 2 succeeded). But **that raise is an OOM fallback, not a
  pixel limit**: the loop above it halves `face_det_batch_size` on every
  `RuntimeError` and only gives up at a batch of 1. So the constraint is VRAM
  against frame size, and the answer is a ceiling rather than a constant —
  the old default was paying half the resolution of every 720p and 1080p
  source to insure against 4K.

  Wav2Lip repaints a 96x96 mouth crop at any factor, so the mouth was never
  what the old default cost; it was the other 99% of the frame. **VRAM is the
  thing to watch now**: ~1.2 GiB at factor 2 versus ~4.0 GiB at 1 on 720p,
  and 1080p is 2.25x the pixels of 720p. The OOM fallback absorbs an
  overshoot by halving the detection batch, so the failure mode is a slower
  render rather than a dead one.

  Three details that are load-bearing:

  1. **A failed probe returns 2, not 1.** `_video_dims` answers None when
     ffprobe is missing or the file is unreadable, and None is a silent
     answer in every caller here (§1). The two guesses are not symmetric:
     guessing native risks the OOM after the whole dub is paid for, guessing
     halved costs resolution on one file.
  2. **The RESOLVED factor keys the coord cache**, not `self.resize_factor`.
     Keying on the attribute would file every auto-resolved run under the
     same `None`, so a 4K render's half-size boxes would load for a 720p clip
     and be painted at the wrong scale — at exit code 0, since upstream only
     checks the box COUNT against the frame count.
  3. **`wav2lip_hd` restores to the cap, not to the source.** Stage 2 targets
     `cap_dims(source)`, so a 4K source is 1080p out and a 720p source stays
     720p. Restoring the full 4K would make it the one model that ignores the
     ceiling, and would do it by ESRGAN-upscaling a 1080p render.

  Still no CLI flag; it travels as an adapter kwarg, so the adapter default
  is the only place that reaches the CLI, the UI runner, the edit agent and
  `experiment.py` at once. `backend/cli.py` only ever puts `batch_size` and
  `inference_steps` into `adapter_kwargs`, and the UI's replay
  (`edit_agent._rerun_lipsync_only`) only `batch_size` — so **every UI render
  takes the ceiling**. An explicit `resize_factor=N` still wins untouched;
  `WAV2LIP_MAX_HEIGHT` moves the ceiling itself.

  **`wav2lip_hd` inherits the ceiling and restores to it.** It subclasses
  `Wav2LipSync`, so stage 1 obeys the same rule; stage 2 then passes
  `--target-width/--target-height` from `cap_dims(_video_dims(request.video))`
  and resizes to exactly that. So both models now land on the same size and
  the ceiling means one thing across the pair:

  | source | `wav2lip` | `wav2lip_hd` |
  |---|---|---|
  | 3840x2160 | 1920x1080 (factor 2) | 1920x1080 (restored) |
  | 1920x1080 | 1920x1080 (factor 1) | 1920x1080 (detail pass) |
  | 1280x720  | 1280x720 (factor 1)  | 1280x720 (never upscaled) |

  The difference between the two is now sharpness, not size — which is the
  honest comparison, and was not possible to make while one halved and the
  other did not.

- **A registered lip-sync model used to reach the UI as a bare slug with no
  numbers.** Fixed 2026-10-22. `wav2lip_hd` was registered on 10-19 and was
  selectable from the day it landed — `GET .../lipsync/readiness` serves
  `available_models()` and the POST validates against it — but it was in
  none of the four tables that *describe* a model, and the failures were all
  silent:

  1. `presets.estimate_runtime` adds nothing for a model it has no band for
     and returned the dub-only estimate as an ordinary "about X to Y". So the
     one model that runs a second per-frame GAN pass advertised as the
     CHEAPEST row on the list. It now returns `lipsync_measured` and says so
     in the text; `ui/cost.py` had always warned by name, and the two
     disagreeing was the tell.
  2. **`"wav2lip_hd".startswith("wav2lip")` is True**, and `wav2lip` came
     first in all three prefix loops — `spec.py`, `server.py::_job_config`
     and `edit_agent._regex_ops`. An HD run displayed as "On — Wav2Lip",
     `_job_config` REWROTE `cfg["lipsync"]` to `"wav2lip"` (the truth went to
     `lipsync_variant`, which nothing read), and "lip-sync it with
     wav2lip_hd" parsed to the plain model in the regex fallback — the parser
     that runs when there is no `OPENROUTER_API_KEY`, i.e. when it is the
     only one.
  3. `/api/options.lipsync_models` was three hand-written rows against five
     registered adapters, and `edits.py`'s op catalogue advertised three to
     the LLM, so the agent was never told HD or LipGAN existed.

  `ui/lipsync_models.py` is now the single source: labels, measured bands,
  per-model caveats, and one `normalise()` (longest key first) replacing all
  three loops. `presets.LIPSYNC_FACTORS`, `cost.LIPSYNC_RATES`,
  `spec.LIPSYNC_WORDS` and `/api/options` all derive from it, and
  `tests/test_ui_lipsync_models.py` fails on drift in either direction.
  Note `lipgan`'s cost midpoint moved 4.0 → 10.0: `cost.py` and `presets.py`
  had carried two different guesses for it, and deriving one from the other
  had to pick the band.

  HD's band (6–24x) is measured from the three runs in
  `data/gender_bench/cuts/hd_runs/logs/` — stage 9h to the conform step, 2
  GPUs: 40s→246s, 40s→550s, 100s→2292s. The spread is not noise; stage 2 is
  per-PIXEL, so it tracks frame size as well as length.

- **LatentSync runs 1.5 as well as 1.6, and a version is a PAIR.** Added
  2026-10-23. `latentsync15` is its own registry id rather than a kwarg,
  because the UI replay (`edit_agent._rerun_lipsync_only`) forwards only
  `batch_size` — a version passed as an adapter kwarg could not be reached
  from the one place the model is chosen. Same shape as `wav2lip_hd`
  subclassing `wav2lip`.

  | id | config | checkpoint | res | inference |
  |---|---|---|---|---|
  | `latentsync` | `stage2_512.yaml` | `latentsync_unet.pt` | 512 | ~18 GB |
  | `latentsync15` | `stage2.yaml` | `latentsync_unet_1.5.pt` | 256 | ~8 GB |

  Upstream's 1.6 changelog: the architecture is unchanged, so "you only need
  to load the corresponding checkpoint and modify the `resolution`
  parameter". **Identical architecture is exactly why this needs a guard** —
  a 1.5 checkpoint loads without complaint under a 512 config and simply
  generates a soft mouth, with no exception and no warning, after ~80x the
  clip length of GPU time. `_ensure_loaded` reads `data.resolution` out of
  the config and refuses a mismatched pair; an unreadable config warns
  instead of blocking. The two checkpoints get **different filenames**
  because upstream ships both as `latentsync_unet.pt` from different HF
  repos, so dropping 1.5's weights in place would silently convert every 1.6
  run into a mismatch. On these 16 GiB cards 1.5 is the one that fits with
  room; 1.6 exists because 1.5's teeth and lips are blurrier.

- **The lip-sync upstreams live under `models/lipsync/vendor/`, not
  `third_party/`.** Moved 2026-10-23 by request — ~21 GB (LatentSync 12G,
  MuseTalk 8.7G, Wav2Lip 517M) — so nothing under `models/` reaches outside
  it. `scripts/move_lipsync_vendor.sh` performs it; `--dry-run` reports
  without touching anything.

  **The code resolves either location, by EXISTENCE rather than
  preference** (`models/lipsync/base.py::resolve_vendor_repo`: explicit →
  env → `models/lipsync/vendor/<name>` → `third_party/<Name>`). That is
  deliberate rather than transitional tidiness: the move is a shell command
  an operator runs, so between the code change and the move — and on any
  checkout where it never happened — the old path is still the right
  answer. Both present logs a warning naming each, because two copies of a
  patched upstream is the state where a fix lands in the one nobody runs.

  Three things this move breaks if done naively:

  1. **`*.pt` was not gitignored.** `third_party/` was ignored wholesale, so
     LatentSync's 4.7 GB `latentsync_unet.pt` escaped git by location, not
     by pattern. Under `models/` it became trackable for the first time —
     one `git add -A` from a 4.7 GB blob that cannot be removed without
     rewriting history. `*.pt` and `models/lipsync/vendor/` were added to
     `.gitignore` first, and the move script **refuses to run** if that
     ignore is not live.
  2. **Wav2Lip's checkpoint was a second hardcoded path.** `DEFAULT_CKPT`
     named `third_party/Wav2Lip/checkpoints/...` outright, so the repo would
     have resolved to its new home while the weights still pointed at the
     old one — raising at stage 9h, after ASR, MT, TTS and the mix are paid
     for. It is derived from `self.repo_path` now.
  3. **A `*_REPO` env var PINS one location and defeats the fallback.**
     `.env.example` therefore ships them blank; the live `.env` never set
     them, which is why the move needs no `.env` edit here.

- **MuseTalk and LatentSync resolve their own interpreter, and a lip-sync
  failure can no longer read as a success.** Both fixed 2026-10-17 after one
  run demonstrated all three halves at once.

  `<MODEL>_PYTHON` was absent from `.env.example` (it documented "Four
  adapters" and listed two), so the adapters fell back to `sys.executable` —
  the `dubbing` env, where `import diffusers` fails. A blank looks harmless
  because `LIPGAN_PYTHON` / `WAV2LIP_PYTHON` legitimately are: those two DO
  run in `dubbing`. `models/lipsync/base.py::resolve_env_python` now tries the
  explicit argument, then `<MODEL>_PYTHON`, then the **sibling conda env**
  (`<prefix>/envs/<name>/bin/python`, derived from `sys.executable` the same
  way `media_io._ffmpeg_bin()` finds the running env), then `sys.executable`
  with a warning. Resolving it in the adapter covers the CLI, `ui/runner.py`,
  the edit agent and `experiment.py` at once; `.env` is now optional. All four
  variables are in `.env.example`.

  The reporting was the expensive part. `run_lipsync` logged
  `lipsync[musetalk]: done → <path>` at INFO with `(err=… rc=1)` as a trailing
  parenthetical after a screen of upstream TensorFlow banner noise, and the
  pipeline recorded `result.lipsync_mp4 = ls.out_video` regardless — the path
  the adapter was ASKED to write. A 40 s run executed all 15 stages, exited 0,
  and left a manifest naming an mp4 that was never created. Now: **the file is
  the arbiter**, not the error field. `produced` (size > 0) is computed, rc=0
  with no output is promoted to an error — the Wav2Lip shape, where
  `inference.py` muxes with `subprocess.call` and discards the return code —
  a failure logs at ERROR with the verdict first, "done" appears only for a
  real render, and `out.lipsync_mp4` is set only when the file exists.

- **The adapters do not choose the SPEAKING face, and the trap is in both of
  them.** Wav2Lip takes the first S3FD detection (`d = d[0]`,
  `third_party/Wav2Lip/face_detection/api.py:73`) — and so does MuseTalk,
  which vendors the same face_detection lineage at
  `musetalk/utils/face_detection/api.py:80`. This file used to record it for
  Wav2Lip alone, which read as though MuseTalk were safe. It is not: grep the
  whole MuseTalk tree for ASD and the only hit is a GPT-2 vocabulary file.

  MuseTalk is in fact worse, because its crop is landmark-derived. The 68
  points come from DWPose `keypoints[0]` — instance 0 of a **whole-frame**
  pose estimate — and are never matched to the S3FD box they are paired with,
  so on a multi-face frame the box and the landmarks can describe different
  people.

  **`run_lipsync(..., coord_list=…)` is the fix, added 2026-10-19 — and
  NOTHING CALLED IT until 2026-10-22.** `backend/lipsync_targets.py::
  build_coord_list` turns the tracker's face tracks plus TalkNet's per-frame
  scores into one box per frame and reaches MuseTalk through its own
  `--use_saved_coord` pickle, so nothing upstream is forked. Measured on the
  Dark Knight mob clip (132–137 s, 10 face tracks): 125 frames, 37 with a
  confirmed speaker across **three different tracks**, all 37 landmarked with
  no fallback, 88 left untouched.

  That measurement came from the tests and `scripts/lipsync_asd_gated.py`,
  never from a pipeline run, and this entry used to stop at the paragraph
  above — which read as "solved". Two things kept it unwired: the script
  says it was "kept as a script rather than wired into the pipeline **so the
  comparison can be made before the default changes**" and that comparison
  was never acted on; and **stage 6 persisted the wrong shape**, since
  `assignments.json` is the Hungarian output (one mean score per diarized
  turn) while the coord list needs per track, per FRAME.

  Now: stage 6 writes `asd/<stem>.<model>.scores.npz`
  (`backend/asd.py::save_scores_npz`) and `pipeline._asd_coord_list` reads
  it. A job with no npz logs a warning and degrades to the adapter's own
  choice; `scripts/lipsync_asd_gated.py` re-scores for those.

  **Two call sites, not one.** Stage 9h in `run_post_mt_stages`, and
  `edit_agent._rerun_lipsync_only` — which bypasses the post-MT chain
  because the mix is still valid, and is how the UI's "Start lip-sync"
  button finishes a draft. Wiring only the first would have fixed the
  pipeline while the button everyone presses carried on as before.

  **The window slice is load-bearing, not a nicety.** A whole-dub lip-sync
  edit resolves to `Scope(all=True)`, and `_h_lipsync_window` turns that
  into a real window (first cue start to last cue end) — so the UI path
  always passes one. `window` cuts a sub-clip that renumbers frames from
  zero, exactly like the sharding and segmenting paths that already refuse
  a whole-video list, and it had no guard at all. `run_lipsync` now slices
  the list to the window. The fps probe is cv2, not ffprobe: §1 records
  `_ffprobe_bin()` answering `None` as a silent `fps=0.0`.

  Four things decided that design, each checked rather than assumed:

  1. **`(0,0,0,0)` now means "write this frame unchanged".** It is MuseTalk's
     `coord_placeholder`, and the 2026-10-18 patch at
     `scripts/inference.py:253` writes such a frame through instead of
     dropping it — the earlier behaviour truncated a 40 s render at 10.64 s.
     This module is only safe against a MuseTalk carrying that fix.
  2. **InsightFace's `landmark_3d_68` is iBUG-68, verified on this host** —
     points 27→30 descend the nose bridge, 8 is the chin, 48/54 are the mouth
     corners. So MuseTalk's `face_land_mark[29]` is InsightFace index 29 and
     the box math transfers 1:1, with no DWPose dependency. `2d106det` is a
     different topology and is deliberately not used.
  3. **Gap-filling is not optional.** A two-frame ASD dip mid-sentence would
     otherwise leave un-animated frames between animated ones, which reads as
     a mouth stutter — worse than the wrong-face problem being fixed.
     `MAX_GAP_FILL` (12 frames) and `MIN_SPEAK_RUN` (5) smooth the runs.
  4. **A supplied list is never written into the coord cache**, and forces
     the single-GPU path. It is gated on one run's ASD rather than being a
     property of the file, and the shards are cut by time with their own
     frame numbering — a whole-video list applied to shard 2 animates the
     wrong frames at exit code 0.

  `bbox_shift` is a no-op on this path: it is consumed inside the detection
  function being skipped, so the shift goes into `build_coord_list`'s own
  argument instead. Wav2Lip and LatentSync ignore `coord_list` rather than
  half-honouring it. `scripts/clip_asd_musetalk_run.py` still does the older
  shot-boundary workaround, and that gating still lives in the script.
- **The manifest is the only durable record of a run, and it used to be
  incomplete.** `_run_pipeline` calls `_jobs.allocate(config=…)` at startup,
  which OVERWRITES whatever the submitter wrote — so the UI's richer config
  survived only until the pipeline began. That dict carried 15 keys and
  none of the optional-stage switches, while `ui/progress.plan_stages`
  branches on 22. A finished job therefore re-planned to 15 stages against
  a progress doc holding 25, and `ui/spec.py`, `ui/cost.py`, `_list_jobs`
  and `edit_agent._config` all described a run with no resegmentation, no
  speaker refinement, no register, no emotion, no visemes and no face
  detector — none of which was true. Fixed 2026-10-17: the allocate config
  now records every switch under the key name `plan_stages` reads, and
  `tests/test_ui_backend_sync.py` greps both sides and fails on the
  difference, so the two cannot drift apart again in silence.

- **An edit replay used to re-pace the dub from argparse defaults, and on a
  UI job that silently cancelled the edit.** Fixed 2026-10-22.

  `edit_agent.rerun` recovers a run's timing setup by re-parsing its argv
  through `cli.resolve_speed_setup`, which is right for a CLI run and
  impossible for a UI one: `ui/runner.py` is launched with the **job id
  alone**, so `recover_args` hands the pipeline parser `['<job_id>']`, it
  exits on the missing `--input`, the exception is caught, and `ns` is None.
  Every `nsget(...)` then returned its argparse default and `setup` stayed
  None. A UI submission runs `naive_slot` with `use_target_duration=True`,
  `tts_apply_stretch=False`, `mix_strategy=timing_plan`,
  `mix_clip_source=raw`; the defaults are `timing_params=None`, stretch ON,
  `global_tempo`, `auto`. So **every** edit replayed on a UI job came back
  paced differently from the dub it was editing.

  `set_speed` is where that becomes a wrong answer rather than a different
  one. The op speeds nothing up by itself — `_h_set_speed` SHORTENS
  `segments[].target_duration_s` and lets the timing plan fit the same clip
  into the smaller slot — and `naive_slot` reads that field only when
  `use_target_duration` is set, which **defaults to False**.
  `backend/timing.py` says so in its own docstring. So the replay ignored
  the shortened slot, re-fitted every clip to its untouched `end - start`,
  and the dub came back at exactly the original speed, at exit code 0, with
  the edit journalled as applied.

  Job `20261022-044302` carries the evidence as two files:
  `timing/naive_slot.cap18.plan.json` (`use_target_duration: true,
  max_tempo: 1.8` — the run) beside `timing/naive_slot.plan.json`
  (`use_target_duration: false, max_tempo: 1.1` — the replay).

  Three sources now, most authoritative first: an explicit `speed_setup=`
  from the caller, then the argv, then `speed_setup_from_config(cfg)` off the
  manifest. `ui/runner.py` passes the first, read from **`request.json`'s
  `pipeline` block** rather than the manifest — the resolved kwargs are
  recorded there verbatim for every job this UI has ever submitted, so jobs
  that predate the fix replay correctly too. `ui/presets.plan_config` also
  records the pacing now, for everyone else.
  `tests/test_ui_backend_sync.py` greps `SpeedSetup._fields` against
  `plan_config`, so a field added to one and not the other fails loudly
  instead of becoming a default on every replay.

- **A re-run used to duplicate a stage row rather than reuse it.**
  `ProgressWriter._open` matched only a **pending** entry, so a second
  attempt over a stage the first had passed or skipped appended a SECOND
  row with the same key. Every reader takes the first match, so the UI
  reported the previous attempt's verdict: stage 9b2 read `skipped` on a
  run whose own log shows it executing for 11.9 s. `_open` now reuses an
  existing row for the key and appends only when the key is genuinely
  absent. `--rerun` and the edit agent's replay both take this path.

- **Bodhan is the only TTS engine, and the UI no longer offers a choice.**
  Changed 2026-10-17. `ui/presets.py::default_tts()` returns `bodhan`, the
  new-job form states the engine instead of picking it, and `/api/options`
  returns a one-row `tts_engines` list. `POST /api/jobs` validates the id
  against the registry and 400s on an unknown one — it used to allocate,
  queue and launch the job, then die at stage 9e with `Unknown TTS model`.

  Two things follow that are easy to miss:

  1. **`--tts bodhan` now implies `--voice-engine indic-speak`**
     (`pipeline._default_voice_engine`). Those were separate flags, and the
     gap was a silent wrong output: without `--voice-engine`, stage 9d
     skipped casting, every speaker got `voice=None`, and
     `bodhan_adapter._post_one`'s `"voice": voice or "Amit"` read the whole
     dub — every speaker, every line — in one Hindi male voice, with the
     predicted genders computed and discarded. An explicit `--voice-engine`
     overrides it.
  2. **Edge is gone, not deprecated** — `models/tts/edge_adapter.py` is
     deleted and `edge` is out of the registry. It spoke 11 of the 23
     languages offered here and had one voice per gender per language, so it
     could neither cover the catalog nor tell two same-gender speakers
     apart. Its table also carried `pa-IN-OjasNeural`, which does not exist
     — verified absent from all 322 voices `edge_tts.list_voices()` returns
     — and the adapter answered an unknown language by silently substituting
     Hindi, so every Punjabi Edge run produced fluent Hindi at exit code 0.
     `ui/spec.py` and `ui/server.py` still carry `edge` as a *display label*,
     deliberately: jobs recorded before 10-17 name it on disk.

- **The TTS register is chosen at stage 9b1, and a UI run used to miss it
  entirely.** `--tts-style auto` (the UI default) runs
  `backend/classify.py::classify_register` over the uploader's metadata and
  picks one of Bodhan's 8 document-level styles, which rides to the engine
  in `instructions.style`. Four tiers, cheapest first: YouTube `categories`
  → channel map → description patterns → an LLM over the transcript. It is
  resolved at `pipeline.py:1676`, before `run_post_mt_stages`, so it reaches
  TTS; a None answer means the voice's neutral reading, which is the correct
  answer for a local file with no metadata.

  **The trap, fixed 2026-10-17: `source_metadata` is only filled by the
  pipeline's OWN download.** The UI pre-cuts the clip and hands the pipeline
  a local path, so `download_youtube` is never reached there and the dict
  stayed empty — on exactly the runs that have the richest metadata. Tiers
  1–3 all abstained, the LLM tier read the transcript and answered
  "unknown", and the run recorded `register = None` while
  `categories: ["Education"]` sat unused in `<job>/source/<stem>.meta.json`.
  Nothing errored; the dub was merely read neutral. `_recover_source_metadata`
  now picks that file up when `source_metadata` is empty, which took the
  W3L9 lecture from `None (via none)` to `educational lecture (via
  categories)`.

  All 8 `REGISTER_STYLES` are members of the engine's 14
  `INDIC_SPEAK_STYLES` — they are declared in different modules and an
  unknown style is a 422 on every segment, so `tests/test_register_to_tts.py`
  asserts the subset along with the whole five-hop chain from metadata to
  payload. **`--tts-style` defaults to None on the CLI**, and 9b1 fires only
  on the literal `"auto"`, so CLI runs do not classify unless asked.

- **Gender prediction is a THREE-WAY MAJORITY VOTE as of 2026-10-18.**
  `backend/gender.py` runs all three and takes the majority:

  1. **wav2vec2-XLSR-53** (`alefiury/wav2vec2-large-xlsr-53-gender-
     recognition-librispeech`) over the speaker's reference clip.
  2. **F0 median pitch** (librosa.pyin, `F0_THRESHOLD_HZ = 165`), restored
     after a spell deleted. It now ABSTAINS on an unvoiced clip instead of
     answering "male" — with a vote at stake, a guess behind no evidence can
     carry a majority.
  3. **insightface `genderage.onnx`** over the face track ASD attributed to
     this speaker. **CPU only** (`FACE_GENDER_DEVICE`), by request and
     because the CUDA EP holds ~460 MiB for the life of the process (§1)
     upstream of a lip-sync stage with ~100 MiB of headroom.

  The ECAPA logistic head stays deleted (44.4%, 0% female recall, no class
  separation). `DUBBING_GENDER_METHOD` and `DUBBING_ECAPA_THRESHOLD` are
  gone; `DUBBING_GENDER_MODEL` swaps the wav2vec2 checkpoint.

  **The vote helps and does not fix the hard case, which is worth knowing
  before trusting it.** Measured end to end on 25 clips from five videos with
  known truth (`data/gender_bench/eval_5s_*_2026-10-18/`): wav2vec2 alone
  **16/25 (64%)**, the three-way majority **18/25 (72%)**. The 7 survivors
  are all one speaker — the W3L9/W4L10 GAN lecturer — and they survive
  because **the two audio methods fail together on him**: his median F0
  really is 150–207 Hz, above the line, and wav2vec2 moves with it. Two
  correlated voters outvote the one correct face 2-1. The face was right on
  **23 of the 23 clips where it answered**, every one of those seven
  included. A face veto would score 25/25 here; that is a departure from
  majority rule rather than an oversight, and it is not what runs.

  Three more things that are easy to get wrong:

  1. **The wav2vec2 label map is `{0: female, 1: male}`** — the opposite of
     the alphabetical guess — and it is read off `model.config.id2label`,
     never assumed. Hard-coding it inverts the catalog filter for every
     speaker at once with no error anywhere. A checkpoint whose labels are
     not male/female is REFUSED at load.
  2. **Abstention is not disagreement.** A 2-0 with the face silent logs at
     INFO; only voters contradicting each other log at WARNING. Conflating
     them fires on every audio-only run and trains the reader to skip the
     line that matters. `unanimous` additionally requires that all three
     voted, so the artifact cannot claim three-way support it lacks.
  3. **Ties go to the face** (`TIEBREAK_ORDER`), reachable only when one
     voter abstains and the other two split. Justified by the 23/23 above.

  **No votes at all → `FALLBACK_GENDER` ("male") at WARNING, and then a
  TARGET-LANGUAGE male voice.** `pipeline.py` marks that speaker
  `gender_basis="fallback"` and `assign_voices` switches its sort to
  language-first, recording `rationale.basis = "language+gender (no gender
  evidence)"`. Ranking a placeholder gender by ECAPA cosine would dress a
  guess up as a timbre match.

  **Stage 6's `assignments.json` finally has a reader.** It recorded which
  face belongs to which speaker and nothing consumed it, which is why the
  face signal had no caller. `backend/assign_faces.py::speaker_face_tracks`
  reads it back: the winning track is the one with the most ON-SCREEN
  SECONDS (not the most turns — six reaction shots must not outvote one
  close-up), off-screen turns are skipped, and a speaker under
  `min_on_screen_s` is omitted rather than guessed at. Without stage 5 + 6
  the face abstains and the vote is between the two audio methods.

  `GenderVote.to_jsonable()` is written per speaker into the merged JSON
  (`speakers[].gender_vote`) and into `voices/<stem>.speaker_voice_map.json`,
  so a finished run says whether its label came from a 3-0 sweep, a 1-1
  tiebreak, an edit override or a no-evidence fallback without re-running
  three models.

- **The edit agent runs Gemma 3 12B, its catalogue lives in the SYSTEM
  prompt, and OpenRouter JSON mode is OFF because it corrupted output.**
  All three changed 2026-10-19.

  `OPENROUTER_MODEL` now defaults to `google/gemma-3-12b-it` (verified live:
  131k context, $0.05/M in, $0.15/M out). The operation catalogue moved out
  of the per-request user turn into `_SYSTEM`, cutting the user message from
  ~3400 to ~1100 characters.

  **That did NOT make requests cheaper, and it is worth being exact about:**
  measured on Langfuse either side, input tokens per parse went from 1653 to
  **2156, up 30%** — this prompt is 1772 tokens against the old
  `op_catalogue()`'s 635, so the system half grew by more than the user half
  fell. The win is correctness, plus a fixed prefix a cache could reuse.

  The cost is drift risk, since `OPS` is generated and the prompt is
  hand-written, so `tests/test_edit_agent.py` now greps every registered op
  out of `OPS` and fails if the prompt omits it, and fails the other way if
  the prompt advertises an op that does not exist.

  That guard immediately paid: `set_speaker` has been registered in
  `backend/edits.py` all along, but `op_catalogue()` advertised only eight
  of the nine ops, so the model was never told it could reattribute a line.

  **The JSON-mode finding is the one worth remembering.**
  `response_format={"type":"json_object"}` — the flag whose entire purpose
  is to guarantee parseable JSON — is what was breaking it. On this model
  and provider it answered:

      {"ops":[{}]}
      {".ops":[{".op":".lipsync_window",".scope":{"start_s":"0:20.00", ...

  an empty op, and every key prefixed with a full stop. Both are **valid
  JSON**, so `extract_json` parses them without complaint and the caller
  silently receives `ops=None` or a nameless op; the symptom at the CLI was
  `unknown op ''`. The same requests with json mode OFF came back correct
  inside ```json fences, which `extract_json` already strips. So
  `OPENROUTER_JSON_MODE` defaults to `auto`, which means off;
  `OPENROUTER_JSON_MODE=on` restores it for a model that needs it.

  Measured after the change, 8 representative requests: 7 resolve correctly
  (including `set_speaker`, `code_mix` term extraction and a correct refusal
  on an unknown speaker). The eighth, "make it a little faster", returns
  `set_speed` with no `factor` where the prompt asks it to return `{"ops":
  []}` — `backend/edits.py` rejects that with "set_speed needs a positive
  `factor`", so it fails loudly rather than guessing a speed.

- **Voice assignment falls back to target language + gender, and says so.**
  `assign_voices` ranks the same-gender pool by ECAPA cosine, but that needs
  a clip of the speaker, a clip of the catalog voice, and `speechbrain`. Any
  one missing and every candidate ties at 0.0. That used to resolve by
  accident — a stable sort over a catalog that happens to be ordered
  native-language-first. It is now explicit: a voice **recorded in the target
  language, of the speaker's gender**, with `rationale.basis` set to
  `"embedding"` or `"language+gender"` so a finished run says which ran. A
  `score: 0.0` with no `basis` is a pre-10-17 file.

  Gender stays the hard filter: when the target language's voice of that
  gender is taken, a cross-language voice of the right gender wins over the
  right language in the wrong gender. English always lands here — it is a
  valid `lang` for the engine but no voice is *recorded* in it.

  The catalog itself was re-verified against Bodhan's own API docs on
  2026-10-17: all 45 names, languages and genders in `DEFAULT_INDIC_SPEAK`
  match, so the table needs no maintenance, only the clips do.

- **Voice reference clips are RENDERED BY THE ENGINE, never downloaded.**
  `backend/voices.py::assign_voices` ranks candidate voices by ECAPA cosine
  between the real speaker's clip and the catalog voice's own clip, found by
  convention at `data/voice_refs/<engine>/<voice>.wav`. All 45 `indic-speak`
  clips were rendered on 2026-10-17 and the term is now live — verified by
  feeding a catalog voice's own clip back in as a speaker reference and
  getting that same voice at cosine 1.0, with the next-nearest male voice at
  0.75 and the pool spanning 0.12–1.00. Before that the directory was empty,
  `attach_voice_references` attached 0, and the branch scored nothing: every
  candidate tied at 0.0 and the stable sort silently fell back to catalog
  order. Two behaviours worth knowing, in order of cost:
  1. **A clip of some other human filed under `Amit.wav` is worse than no
     clip at all.** The cosine then compares the speaker against the wrong
     timbre and returns a confident similarity, so the log looks like the
     feature is working. `scripts/build_voice_refs.py` re-renders them; the
     clips are not sourced from the internet. (A zero-shot engine whose
     "voices" ARE reference clips — IndicF5 takes `ref_audio_path` +
     `ref_text` — is a different catalog with a different provenance story,
     not more files in this directory. `load_voice_catalog("indicf5")`
     returns 0 voices today.)
  2. `scripts/build_voice_refs.py` had no `load_dotenv()` call, so run
     directly it saw no `BODHAN_TTS_KEY` and every render failed on auth —
     which reads like an upstream outage rather than a missing line. Fixed
     2026-10-17. `backend/cli.py`, `edit_agent.py` and `experiment.py` each
     call it for themselves, and a new entry point under `scripts/` needs
     the same.
  Expect 429s throughout a full 45-voice run; the adapter backs off and the
  script is idempotent, so a re-run fills only the gaps.

- **Bodhan endpoints return `internal_error` 5xx during upstream outages** —
  this is not rate limiting, and since Edge was removed there is no second
  engine to fall back to. `bodhan_adapter.synthesize` catches per request, so
  an outage does not raise: every segment comes back with `tts_error` set and
  the stage returns green with `extra.tts.errors == len(segments)`. That count
  is the signal, not the exit code. A missing `BODHAN_TTS_KEY` is the louder
  case — `_ensure_loaded` raises before the first request.
- **ASD scores produced before 2026-10-11 are worthless.** `talknet_adapter.py`
  fed `soundfile` floats in [-1, 1] into an MFCC front-end that upstream trains
  and infers on int16 PCM, which shifted every log-energy cepstral term and
  pinned all scores near zero. Measured on one 3 s single-speaker close-up:
  mean 0.044 / max 0.129 / 0 of 75 frames above 0.5 before the fix, versus
  mean 0.411 / max 0.798 / 31 of 75 after. The stage never errored — it just
  answered "nobody is speaking" every time, so any archived `assignments.json`
  or on-screen count from an earlier run is stale rather than reusable.

---

