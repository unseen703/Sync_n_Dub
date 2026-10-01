# Dubbing Studio — Pipeline Scaffold

First-cut pipeline for the Indic Video Dubbing project ([plan.md](plan.md)).
This scaffold covers the four earliest stages end-to-end so the rest of the
pipeline can be built on top of it.

## Pipeline (this scaffold)

```
YouTube URL or video file
        │
        ▼
┌───────────────────────┐
│ 1. download_youtube   │   yt-dlp
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 2. extract_audio      │   ffmpeg — mono 16 kHz WAV
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 3. separate_stems     │   HTDemucs — vocals (dialogue) + no_vocals (M&E)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 4. diarize            │   pyannote-community-1 · Sortformer · DiariZen
└───────────────────────┘
```

## Layout

```
dubbingstudio/
├── backend/                    # pipeline orchestration + stage functions
│   ├── downloader.py           # download_youtube(url, out_dir)
│   ├── media_io.py             # extract_audio(video, out_wav)
│   ├── stem_separation.py      # separate_stems(audio, out_dir)
│   ├── diarize.py              # diarize(audio, model=..., ...)
│   ├── faces.py                # detect_and_track(video, detector=..., tracker=...)
│   ├── asd.py                  # score_asd(...) + Hungarian speaker↔face assignment
│   ├── assign_faces.py         # SegmentAssignment, assign_faces_to_turns(...)
│   ├── asr.py                  # transcribe(audio, model=..., turns=...)
│   ├── pipeline.py             # run_pipeline(input, ...)
│   └── cli.py                  # `python -m backend.cli --input ... [--asd ...] [--asr ...]`
├── frontend/
│   └── app.py                  # gradio UI  — `python frontend/app.py`
├── models/
│   ├── diarization/            # swappable diarization adapters
│   │   ├── base.py             # BaseDiarizer, DiarizationResult, Turn
│   │   ├── pyannote_adapter.py
│   │   └── diarizen_adapter.py
│   ├── faces/
│   │   ├── base.py             # FaceBox, FaceTrack, FaceTracks, bases
│   │   ├── detect/             # retinaface_adapter.py, mediapipe_adapter.py
│   │   └── track/              # bytetrack_adapter.py
│   ├── asd/                    # active-speaker detection
│   │   ├── base.py             # ASDResult, ASDScore
│   │   ├── talknet_adapter.py
│   │   └── ts_talknet_adapter.py
│   └── asr/
│       ├── base.py             # ASRResult, ASRSegment, Word
│       ├── bodhan_adapter.py
│       ├── whisperx_adapter.py
│       └── indicconformer_adapter.py
├── notebooks/
│   └── pipeline.ipynb          # run each stage cell-by-cell
├── data/                       # gitignored — pipeline artefacts land here
│   ├── videos/  audio/  stems/  diarization/
├── audio_browser.ipynb         # pre-existing — inline player over outputs
├── plan.md                     # the full plan
├── requirements.txt
└── .env.example
```

## Setup

One-shot (recommended) — creates the venv, installs torch + requirements,
fetches the TalkNet-ASD repo + checkpoint, and runs a 60-second dry-run
pipeline on the first reference video:

```bash
sudo apt install ffmpeg           # or: brew install ffmpeg — must be on PATH
./setup.sh                        # see ./setup.sh --help for flags

# useful variants:
./setup.sh --check-only           # report state, change nothing (doctor mode)
./setup.sh --no-dry-run           # install + fetch, skip the 60s pipeline run
./setup.sh --no-talknet           # pyannote-only; skip TalkNet repo+ckpt
```

The script is idempotent — re-run it any time to re-verify the project
is still set up correctly. Model-download guidelines (what gets fetched
automatically, what needs a token, what has to be installed by hand) live
at the top of [scripts/setup.py](scripts/setup.py).

Credentials still need to be filled in manually in `.env` (the setup
script copies `.env.example` → `.env` with mode 600 on first run):

- `HF_TOKEN` — accept the gate at https://hf.co/pyannote/speaker-diarization-community-1
- `BODHAN_*_URL` / `BODHAN_*_KEY` — only needed for the `bodhan` ASR adapter

Manual install (if you prefer step-by-step):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cp .env.example .env && chmod 600 .env   # fill HF_TOKEN
python scripts/setup.py --no-dry-run     # fetch TalkNet, verify folder structure
```

## Run — three entry points

**CLI:**
```bash
python -m backend.cli \
  --input "https://www.youtube.com/watch?v=nsIMQP0QhpE" \
  --model pyannote \
  --face-detector retinaface --asd talknet \
  --asr bodhan --asr-language hi
# --model {pyannote, diarizen}
# --face-detector {retinaface, mediapipe}  (none → skip faces/ASD entirely)
# --face-tracker  {bytetrack}
# --asd           {talknet, ts_talknet}    (needs --face-detector)
# --asr           {bodhan, whisperx, indicconformer}
# --asr-language  ISO-639 code (en, hi, ta, …); omit to let the model detect.
# --render-mode   {native, mixed, roman}   (Bodhan Indic-Transcribe only)
# --hf-token / --checkpoint / --batch-size are forwarded to the adapter that
#   accepts them; unknown-for-adapter kwargs are dropped with a warning.
# --skip-stems runs diarization on the raw extracted audio instead of the
#   dialogue stem. dialogue_stem/me_stem are then None in the manifest, not
#   the mixed audio — do not treat a skipped stem as an M&E bed.
```

**Gradio:**
```bash
python frontend/app.py                    # binds 127.0.0.1 only
python frontend/app.py --host 0.0.0.0     # LAN — requires DUBBING_UI_USER/PASS
# opens http://localhost:7860
```

The default bind is loopback because the UI runs yt-dlp against arbitrary
user-supplied URLs — an unauthenticated LAN-exposed instance is an SSRF /
GPU-abuse vector. LAN mode requires HTTP basic auth via
`DUBBING_UI_USER` / `DUBBING_UI_PASS`.

**Notebook:**
```bash
jupyter lab notebooks/pipeline.ipynb
```

## Security notes

- `.env` contains an HF token. `backend.cli` refuses to start if `.env` is
  group- or world-readable (mode > 0600). Run `chmod 600 .env` before first
  launch. Override with `--allow-insecure-env` at your own risk.
- If a token was previously stored world-readable, **revoke and rotate it** at
  https://hf.co/settings/tokens — assume it leaked.
- `.env` is in `.gitignore`; `.env.example` is the tracked template.

## Reference videos

The three videos wired into the notebook by default:

- https://www.youtube.com/watch?v=nsIMQP0QhpE
- https://www.youtube.com/watch?v=P26AE7NLx4Q
- https://www.youtube.com/watch?v=2vxQaT0Fy-c

## Model notes

### Diarization

| Adapter | HF repo | Gate | License |
| --- | --- | --- | --- |
| `pyannote` | `pyannote/speaker-diarization-community-1` | yes — accept model card | MIT (code) / CC-BY (weights) |
| `diarizen` | `BUT-FIT/diarizen-wavlm-large-s80-md` | no | CC-BY-NC-4.0 — non-commercial |

### Face detection + tracking

| Adapter | Weights | Notes |
| --- | --- | --- |
| `retinaface` | InsightFace `buffalo_l` (SCRFD-10GF) | 5-point landmarks; downloads ~250 MB on first run |
| `mediapipe` | bundled | CPU-friendly, bbox only (projects 6-pt landmarks to 5-pt) |
| `bytetrack` | — | pure-Python ByteTrack via `supervision` |

### Active-speaker detection

Both adapters need an external repo clone + weights (not pip-installable):

| Adapter | Repo | Env vars |
| --- | --- | --- |
| `talknet` | https://github.com/TaoRuijie/TalkNet-ASD | `TALKNET_REPO`, `TALKNET_CHECKPOINT` |
| `ts_talknet` | https://github.com/Jiang-Yidi/TS-TalkNet | `TS_TALKNET_REPO`, `TS_TALKNET_CHECKPOINT`, `TS_TALKNET_SPEAKER_ENCODER` |

`ts_talknet` uses target-speaker enrolment embeddings; without reference
audio it degenerates to plain TalkNet behaviour.

### ASR

| Adapter | Model | Timestamps | License |
| --- | --- | --- | --- |
| `bodhan` | Bodhan Indic-Transcribe Core/Flex (REST) | segment-level | Indic Open Model v1.0 |
| `whisperx` | Whisper (CT2) + wav2vec2 CTC align | word-level | MIT |
| `indicconformer` | AI4Bharat Conformer-CTC 600M multilingual | segment-level | Apache-2.0 |

Bodhan credentials are per-service — `BODHAN_ASR_URL`, `BODHAN_ASR_KEY`,
`BODHAN_ASR_MODEL`. See `.env.example`.

## Adding a new diarization model

1. Drop `models/diarization/my_model_adapter.py` implementing `BaseDiarizer`.
2. Register the name in `models/diarization/__init__.py`.
3. Call it via `--model my_model` or the frontend dropdown.
