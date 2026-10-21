# Pending structural changes

Buffer written by `scripts/repo_checkup.py`. Entries sit here until they
are folded into the prose sections of [README.md](README.md), after which
this file is cleared with `python scripts/repo_checkup.py --clear`.

The generated structure block in README.md is already current — it is
rewritten on every checkup. What lives here is the *narrative* follow-up:
what changed, so someone can say why it changed and what it means.

---
## 2026-10-17 14:25

- **backend/ modules** added: `speaker_id`
- **Pipeline stages** added: `4c: speaker label refinement`
- **CLI flags** added: `--refine-speaker-labels`, `--speaker-min-speech-s`, `--speaker-purity-floor`
- **scripts/** added: `benchmark_gender.py`
- **tests/** added: `test_speaker_id`

## 2026-10-17 14:53

- **CLI flags** removed: `--restore`, `--restore-fidelity`, `--restore-upscale`
- **scripts/** removed: `compare_restore.py`, `run_nag_wav2lip_codeformer.sh`
- **tests/** removed: `test_restore_registry`
- **restore adapters** removed: `codeformer`

## 2026-10-17 15:31

- **tests/** added: `test_ui_backend_sync`
- **third_party/ clones** removed: `CodeFormer`

## 2026-10-17 21:03

- **tests/** added: `test_register_to_tts`

## 2026-10-17 23:42

- **Top-level entries** added: `run_w1l2_cli.sh`
- **tests/** added: `test_lipsync_env_python`, `test_lipsync_failure_honesty`, `test_ui_register_row`

## 2026-10-20 00:09

- **Top-level entries** added: `docs/`, `gfpgan/`, `next_implementation_steps.py`
- **backend/ modules** added: `caption_asr`, `lipsync_targets`, `tracing`
- **Pipeline stages** added: `7: transcript from captions`
- **CLI flags** added: `--captions-allow-auto`, `--no-captions-transcript`
- **scripts/** added: `gpu_watcher.py`, `langfuse_usage.py`, `run_wav2liphd_clips.sh`, `setup_wav2liphd.py`, `wav2liphd_enhance.py`
- **tests/** added: `test_caption_asr`, `test_coord_cache`, `test_gpu_pick`, `test_lipsync_targets`, `test_musetalk_frame_parity`, `test_speaker_face_tracks`, `test_staged_review`, `test_talking_head`, `test_tracing`
- **third_party/ clones** added: `Wav2Lip-HD`, `langfuse`
- **Environments** added: `wav2liphd`
- **lipsync adapters** registered: `wav2lip_hd`
