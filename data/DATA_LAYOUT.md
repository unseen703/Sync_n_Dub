> **Status (2026-10-11): this document describes the retired per-stage layout.**
>
> Every pipeline run is now **self-contained under `data/jobs/<job_id>/`**. The shared
> `data/golden/`, `data/exp/`, `data/mt/` … trees this document walks through no longer
> exist on disk; see [The job layout](#the-job-layout) below for what replaced them, and
> `data/_archive/*.tar.zst` for what was kept. The experiment and ground-truth *method*
> described here is still accurate and is why the file is kept.

# The job layout

One run = one folder. Nothing a run produces is written outside it.

```
data/
  jobs/<job_id>/            one end-to-end run
    manifest.json           status, config, argv, and an inventory of every file below
    videos/                 the input (downloaded, or hard-linked in from a local path)
    audio/ stems/           extracted audio, HTDemucs stems
    diarization/ faces/ asd/
    asr/ align/ mt/         transcripts, word alignment, translation
    mt/versions/            every version of the merged JSON (see below)
    voices/ tts/            speaker refs, synthesized clips
    mix/ lipsync/           final dubbed audio, muxed video (history in versions/ beside each)
    versions/index.json     which edit produced which files; `current` = latest (see below)
    eval/ exp/              scores; per-variant subtrees for a sweep
  _archive/*.tar.zst        the retired per-stage folders
```

`<job_id>` is `<timestamp>__<input-slug>__<diar>.<asr>.<align>.<mt>.<lang>`, so `ls` sorts
chronologically and re-running the same video with the same settings never overwrites the
previous run.

## `mt/` and `mt/versions/`

The merged ASR+MT JSON is the document the whole post-MT half of the pipeline
rewrites **in place**. One live file, read and written by every stage:

```
mt/<stem>.<engine>.<lang>.mt.json          ← the live document; stages read THIS
mt/<stem>.<engine>.<lang>.transcript.txt
```

Every writer also snapshots, so no version is lost (`backend/mt_versions.py`):

```
mt/versions/<stem>.<engine>.<lang>.v000.original.json    straight out of MT
mt/versions/<stem>.<engine>.<lang>.v001.codemix.json     stage 9b
mt/versions/<stem>.<engine>.<lang>.v002.condense.json    stage 9c2
mt/versions/<stem>.<engine>.<lang>.v003.emotion.json     stage 9b2
mt/versions/<stem>.<engine>.<lang>.v004.tts.json         stage 9e
mt/versions/<stem>.<engine>.<lang>.v005.edit_000.json    an applied edit
mt/versions/index.json                                   producer, bytes, segments, time
```

Four naming rules, each defensive rather than cosmetic:

1. **A subdirectory, not `mt/`.** `edit_agent._find_merged_json` globs
   `mt/*.mt.json` and, when several match, takes **the newest by mtime** with
   only a log line. A snapshot is newer than the live file by construction, so
   in `mt/` it would win — silently becoming the document every later edit is
   applied to. `glob` is not recursive, so a subdirectory is invisible to it.
2. **No version ends in `.mt.json`.** Belt and braces for (1): the suffix is
   `.<producer>.json`, and `mt` is a *reserved* producer name.
3. **Zero-padded to three digits.** `v10` sorts before `v9` unpadded, which is
   how a history silently starts reading in the wrong order past ten entries.
4. **The producer is in the name**, so the history reads without opening a file.
   An applied edit carries the same `NNN` as its `edits/edit_NNN.json` journal
   entry, so the request and the resulting document line up by name.

A snapshot never overwrites: the same producer twice (a re-run, a replay) takes
the next free number. Failures are logged and swallowed — a job that cannot
write a snapshot still renders, it just has a thinner history.

## `versions/index.json` — which files belong to which version

`mt/versions/` and `mix/versions/` keep every transcript and every mix, each
numbered on its own. `versions/index.json` (`backend/job_versions.py`) is the
map over them: one row per version, saying which edit produced it and which
files it links to. It copies nothing — an artifact is a relative path plus a
sha256 into the snapshot folders.

```
{"schema": 1, "current": 7, "versions": [
  {"v": 7, "parent": 6, "kind": "edit", "edit_id": "edit_003",
   "nlq": "...", "ops": [...], "dirty_stages": ["tts", "mix"],
   "status": "rendered",
   "artifacts": {"transcript": {"path", "sha256", "bytes"},
                 "mix_wav": {...}, "mix_mp4": {...}, "lipsync_mp4": {...}}}]}
```

- `kind`: `original` | `cast` | `edit` | `lipsync` | `revert`.
  `status`: `applied` → `rendering` → `rendered` | `failed`.
- **`current` is what "latest" means.** The UI keeps reading the live files;
  use `job_versions.latest(job, "mix_mp4")` rather than globbing by mtime.
- **A version number is never reused**, even across the UI server and the
  detached runner: writes are locked and atomic.
- **Revert is a new version** (`kind=revert`, `restores=<v>`), never a
  deletion. It records a transcript only, because the audio on disk keeps the
  undone edit until a re-render; `latest("mix_wav")` walks `parent` to the mix
  that is really there.
- **Lip-sync is the one artifact nothing else snapshots.** Before a new render
  the previous one is *renamed* into `lipsync/versions/<stem>.vNNN.mp4` (no
  extra disk), and renamed back if the render fails.
- Per-cue TTS clips are not versioned; they are a hash cache rebuilt from the
  transcript.
- Older jobs are back-filled on first read from `edits/edit_*.json`,
  `mt/versions/` and `mix/versions/`.

```bash
python -m backend.job_versions list -j <job_id>
python -m backend.job_versions show 3 -j <job_id>
python -m backend.job_versions restore 0 -j <job_id>   # new version; re-render after
# HTTP: GET /api/jobs/<job_id>/versions   (backend record; the UI does not call it)
```

```bash
python -m backend.cli --input <url-or-file> ...   # one input per run; creates a job
python -m backend.jobs list                       # every run, newest first
python -m backend.jobs show <job_id>              # one run's manifest
```

`manifest.json` carries `status`: `running` while the run is in flight, then `ok`, or
`failed` with the error — a crashed run closes out its own folder rather than leaving it
indistinguishable from one still going.

Two things deliberately live *outside* a job, because they are inputs rather than outputs:
`models/duration/<lang>.<voice>.json` (fitting one costs a full TTS calibration pass) and
`experiments/suites/*.json` (sweep definitions, which are config).

---

# What is on disk: the ground truth, and the sweep runs

Two trees. `data/golden/` is the **reference** — derived from the source videos, never from the
pipeline. `data/exp/sweep1/` is the **hypothesis** — what the pipeline produced. Nothing in the
first is written by a pipeline run; that separation is what makes the comparison meaningful.

---

# Part 1 — Ground truth (`data/golden/`)

## 1.1 Where it comes from

Four YouTube videos, chosen because all three Hindi tracks are **human dubs** (`dubbed`, not
`dubbed-auto`) and all four carry **manual** (human-authored) Hindi subtitles:

| Video | Title | Hindi audio | Role |
|---|---|---|---|
| `6-NRiom8K9Y` | NPTEL design-thinking lecture | human dub | tier A, domain-matched |
| `ZPUtA3W-7_I` | Lex Fridman × Narendra Modi | human dub + `hi-Latn` | tier A, multi-speaker |
| `qjPH9njnaVU` | Lex Fridman × Pavel Durov | human dub | tier B, generalization |
| `YrtANPtnhyg` | TED, Pranav Mistry | **none** | tier C, source-side only |

18 clips of ~60 s were cut from these. **Tiers decide what a clip is allowed to be used for:**

- **A — 11 clips.** Rank the methods.
- **B — 4 clips.** Generalization check, reported in a separate table, **never pooled** into the ranking.
- **C — 3 clips.** No Hindi dub exists, so no timing ground truth. Source-side component tests only;
  excluded from the sweep entirely.

## 1.2 What a clip directory contains

Example `data/golden/ZPUtA3W-7_I_09880to09940/` — the window 9880–9940 s of the source video:

| File | What it is | How it was derived |
|---|---|---|
| `clip.mp4` | the 60 s source clip, video+English audio | ffmpeg cut |
| `source.wav` | English audio, 16 kHz mono | ffmpeg from the `en-US` track |
| `ref_dub.wav` | **the reference** — the human Hindi dub of the same 60 s | ffmpeg from the `hi` track |
| `source.vad.json` | English speech/pause spans | pyannote VAD |
| `ref_dub.vad.json` | **reference speech/pause spans** | pyannote VAD on the human dub |
| `source.words.json` | English word times | `en-orig` caption inline word stream |
| `source.words.whisperx.json` | the same, independently | whisperx forced-align |
| `ref_dub.words.json` | **reference word times** | whisperx forced-align of the manual Hindi subtitle text onto the dub audio |
| `segments.json` | diarized turns | pyannote |
| `reference_qc.json` | the §1.3 gate: every rule's measured value + verdict | — |
| `meta.json` | provenance, sha256 per artifact, QC, coverage gates, agreement | — |

**Time base, stated in every file.** `meta.json` carries the sentence *"ALL times in every artifact
in this directory are CLIP-RELATIVE (0.0 == source.t0)"*. Only `source.t0` / `source.t1` are absolute
positions in the original video. This is the single most error-prone thing in the project and it is
written into the data rather than left to convention.

A real `ref_dub.vad.json`:

    backend pyannote, min_pause_s 0.25, min_speech_s 0.2, duration_s 60.0
    n_speech 14, n_pauses 13, speech_s 49.33 (82.2%), lead_silence 0.031, tail_silence 0.029
    speech: [[0.031, 3.1191], ...]      pauses: [[3.1191, 4.5197], ...]

**That `speech` / `pauses` list is what the methods are scored against.** Not the subtitle text, not
the duration — where the human dub actually talks and stops.

## 1.3 Two gates that decide what each clip may measure

**(a) Reference QC** (`reference_qc.json`) — cue density, max cue span, temporal coverage,
rolling-caption duplication, cross-language parity, media readability. Verdict `pass` /
`pass_with_warnings` / `reject` per track, with every rule's measured value.

**(b) Word-metric eligibility** (`meta.words.<track>.vad_coverage`) — a clip may carry VAD-level
metrics but *not* word-level ones. The gate is **≥2.0 words per second of VAD speech**:

    words_per_speech_s 3.125 (min 2.0), coverage 0.787 (min 0.55) → usable_for_word_metrics: true

This is why **n = 11 tier-A clips for VAD metrics but only 7 for word-level metrics**. Four NPTEL
clips are under-subtitled (0.96–1.45 words/s): their Hindi word times describe the subtitle file,
not the speech. Those cells report `not_measured` with that sentence, rather than a number.

## 1.4 The honest limit of this ground truth

The reference's own accuracy is measured and published, per clip, in
`meta.agreement_median_ms` — and the result bounds every claim:

| Side | Clips measured | Median |
|---|---|---|
| Source (English), word-level | 12 of 18 | 55 ms |
| **Reference dub (Hindi), word-level** | **0 of 18** | **not measurable** |

No video in the corpus carries an `hi-orig` auto-caption, so every Hindi word stream comes from a
single producer (whisperx) with nothing independent to check it against. The measurable substitute
is cue-level — manual `.hi.vtt` cue onsets against the CTC onset whisperx gave that cue's first word
(shared text, independent timing): **185 ms median over 9 clips**.

**So 185 ms is an upper bound on the Hindi floor, and differences smaller than it are ties, not
ranks.** The 55 ms figure is the English side, and is not the ranking's floor.
Full detail: `data/golden_reference_provenance.md`.

---

# Part 2 — The sweep runs (`data/exp/sweep1/`)

## 2.1 Shape

**15 clips × 10 arms = 150 pipeline runs, 150 rendered mixes, 0 failures**, ~95 s per clip.

    data/exp/sweep1/
      audio/ stems/ diarization/ asr/ align/ mt/   ← upstream stages, shared
      exp/<clip_id>/
        _shared/prefix.mt.json      ← stages 1..9a, run ONCE per clip, copied never mutated
        <arm_id>/
          variant.json              ← fully-resolved config + every output path
          plan.mt.json              ← the merged JSON THIS arm mutates
          timing/<producer>.plan.json
          tts/<clip>.edge/{raw,stretched}/
          tts_align/work/           ← M5 arms only: 16 kHz resamples for forced alignment
          mix/<clip>.dubbed.wav     ← the rendered dub
          metrics.json

**Isolation is the point.** Every arm gets its own copy of the merged JSON and writes only inside
its own directory — the trees are meant to stay disjoint, with no arm's
config referencing another arm's path except `source_json` pointing at its declared TTS donor.
That protects the 8.2 GB of irreplaceable artifacts in `data/mt`, `data/tts`, `data/mix`.

**Two TTS passes, not ten.** Only `ctl_naive_slot` and `m1_engine_speed` change what is
*synthesised*; the other eight reuse the control's raw clips via `reuse_tts_from`. Within an
M5 pair the audio is bit-identical and only the plan differs — which is what makes the pairing clean.

## 2.2 The ten arms

| Arm | What it does |
|---|---|
| `ctl_naive_slot` | **Control.** One clip per slot, one tempo, cap 1.10×. Today's placement, re-rendered through the timing-plan renderer so control and treatments share a renderer. |
| `base_global_tempo` | **Baseline — what ships today.** No timing plan: shift-forward placement plus one global `atempo` over the whole dub, cap 1.5×. |
| `m1_engine_speed` | M1. Ask the TTS engine for the rate that fits (closed loop, 2 rounds, 0.85–1.15×), then place naively. |
| `m2_post_tts_stretch` | M2. Control placement, but rubberband may stretch to 1.50×/0.667×. |
| `m3_pause_aware` / `_m5` | M3. Anchor each speech run to its source onset, reinsert source pauses. **Paired on M5.** |
| `m4_word_anchored` / `_m5` | M4. Word-level alignment. Without M5 the target word times are synthesised; with M5 they are real. **Paired.** |
| `m7_dp` / `_m5` | M7. DP over splits, per-piece tempi and pause lengths. **Paired.** |

**M5 is a paired factor, not a row** — it is post-TTS forced alignment, an *input* three methods
read, not a placement method. It is paired on M3 as well as M4/M7 because `plan_pause_aware` reads
`tts_words` too. **M6 and M8 are excluded with measured evidence**, recorded in the suite under
`notes.excluded`.

## 2.3 What a run produces

`variant.json` is the provenance record — the fully-resolved config including defaults, so a result
can be reproduced without the suite file:

    variant: {id, label, method, reuse_tts_from, config{pipeline, timing{producer, params}}}
    resolved: every knob of every section, defaults filled in
    source_json: the prefix (or the donor's merged JSON)
    paths: dir, plan_json, timing_dir, tts_dir, mix_dir, metrics_json

`timing/<producer>.plan.json` is the **TimingPlan** — the one contract every method emits and the
single renderer consumes:

    summary: {segments 6, pieces 24, inserted_pauses 0, overlap_s 0.0,
              tempo_min 0.963, tempo_max 1.15, tempo_mean 1.110,
              tempo_cap_hits 4, fit_duration_s 67.466, degraded_segments 0}

and per segment:

    id SEG_0002, speaker SPK_00, slot {0.774 → 10.426}
    src_speech  [6 spans]        ← where the SOURCE talks  (absolute)
    src_pauses  [5 spans]        ← where it stops          (absolute)
    pieces      [4]              ← each: src_span (clip-relative!), dst_start (absolute),
                                   local_tempo, anchor, text, src_word_range
    cost {D_duration 0.910, D_pause 5.787, D_speechrate 0.322, total 7.020}
    qc   {src_run_cuts 5, tempo_cap_hit true, n_src_runs 6, n_tgt_clauses 4,
          mass_basis "time", natural_rate_pieces 1}

**Note the mixed time bases** — `src_span` is clip-relative while `dst_start`, `slot`, `src_speech`
and `src_pauses` are absolute. That asymmetry is deliberate (a piece cuts *into* a raw TTS clip and
places it *on* the timeline) and is the most error-prone part of the schema.

## 2.4 Scoring

`data/eval/sweepA/` (11 tier-A clips) and `data/eval/sweepB/` (4 tier-B) — one
`<arm>.timing_eval.json` per arm-clip, **150 total**, from
`python -m backend.eval run --groups timing,pauses,rate`.

Each result records what it was scored against and how, not just the numbers:

    timing_method    golden_ref_dub_vad+mix_replay:timing_plan:timing_plan_pieces
    reference        ref_dub.vad.json / ref_dub.words.json / source.vad.json / meta.json
    calibration      gt_noise_floor_ms 185, noop_speech_iou_median 0.872,
                     noop_pause_f1_at_200ms_median 0.000
    hypothesis       planned | measured   (cheap groups score the PLAN and say so;
                                           audio groups VAD the rendered wav and say so)
    not_measured[]   every skipped metric with the reason, in words

`groups timing,pauses,rate` decode no audio and import no torch — seconds per clip. The `intel`
group (round-trip WER) and `mt` group were **not run** in this sweep.

## 2.5 Listening

`data/listen/<clip_id>/` — hard links, no extra disk. Numbered in listening order:
`00_SOURCE_*`, `01_GROUND_TRUTH_human_hindi_dub.wav`, `02_SHIPS_TODAY_global_tempo.wav`,
`03_control_naive_slot.wav`, then `04`–`11` for the method arms. See `data/listen/README.md`.

## 2.6 Reproducing

    python -m backend.eval run --clip-id <id> --variant-id <arm> \
        --mt-json data/exp/sweep1/exp/<id>/<arm>/plan.mt.json --groups timing,pauses,rate
    python -m backend.eval report --results data/eval/sweepA --baseline base_global_tempo

Clips are fed to the runner as **hard links** at `data/exp/clips/<clip_id>.mp4`, because every
golden clip's video is literally named `clip.mp4` — passing those paths directly would derive the
base name `clip` and collide all 18 clips into one output tree. A symlink does not fix it (the
media stage resolves it); a hard link has nothing to resolve.
