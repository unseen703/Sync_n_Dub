"""Pipeline backend: download, media I/O, stem separation, diarization, orchestration.

Public entry points:
  - backend.pipeline.run_pipeline(input_, diarization_model, ...)
  - backend.downloader.download_youtube(url, out_dir)
  - backend.media_io.extract_audio(video, out_wav)
  - backend.stem_separation.separate_stems(audio, out_dir)
  - backend.diarize.diarize(audio, model, ...)
"""
