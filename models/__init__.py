"""Models package. Each submodule hosts a family of swappable adapters.

Currently:
  - models.diarization: speaker diarization (pyannote, diarizen)
  - models.faces      : face detection (retinaface, mediapipe) + tracking (bytetrack)
  - models.asd        : active-speaker detection (talknet, ts_talknet)
  - models.asr        : speech recognition (bodhan, whisperx, indicconformer)

Future subfolders (per plan.md) will follow the same adapter pattern:
  models/translate, models/tts, models/vc, models/lipsync, ...
"""
