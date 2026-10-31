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
