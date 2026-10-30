"""Models package. Each submodule hosts a family of swappable adapters.

  - models.diarization: speaker diarization (pyannote)
  - models.faces      : face detection (retinaface, mediapipe) + tracking (bytetrack)
  - models.asd        : active-speaker detection (loconet)
  - models.asr        : speech recognition (bodhan)
  - models.align      : forced alignment (mms_fa)
  - models.mt         : translation (bodhan)
  - models.translit   : transliteration (bodhan)
  - models.tts        : speech synthesis (bodhan)
  - models.emotion    : speech emotion (emotion2vec)
  - models.lipsync    : lip-sync (wav2lip, musetalk, latentsync15)
  - models.restore    : face restoration (gfpgan)
"""
