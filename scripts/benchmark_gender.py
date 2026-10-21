"""Benchmark the voice-gender classifier against the FACE label.

Design, and why each step is there:

  1. Cut a 90 s clip. Full videos would be hours of GPU for a number a clip
     answers; the clip is taken from the middle, away from titles and credits.
  2. Face detect + track  -> candidate people, as `(frames, bboxes)` tracks.
  3. `predict_gender_face` on each track -> the REFERENCE label. This is
     itself a model (insightface genderage.onnx), not truth, so the result
     below is agreement with it, not accuracy.
  4. ASD (TalkNet) scores each track per frame for "is this face speaking".
     This is what pairs a face with a VOICE: without it, the audio under a
     face track is whoever happened to be talking, and the whole comparison
     measures nothing. Only tracks that actually speak are kept.
  5. Cut that track's speaking span out of the audio -> run the wav2vec2
     classifier on exactly the audio that face produced.
  6. Compare against the face label.

Last run 2026-10-18, after F0 and ECAPA were removed from `backend/gender.py`:
the classifier agreed with the face on 9 of 11 tracks (81.8%), against 72.7%
for the F0 threshold and 45.5% for the ECAPA head it replaced. Both of its
disagreements are John Wick tracks where F0 ALSO said male against a face
label of female, so the reference is at least as likely to be wrong there.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# TALKNET_REPO and TALKNET_CHECKPOINT live in .env and nothing puts them in
# the environment on its own; without this the ASD stage dies with "TalkNet
# repo not found", which reads like a missing clone rather than an unloaded
# file. The path is explicit because bare `load_dotenv()` searches upward
# from the CALLING FILE — run from outside the repo it silently finds
# nothing and the failure looks identical. Same convention as
# `scripts/clip_dub_audio.py`.
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np

from backend.asd import score_asd
from backend.faces import detect_and_track
from backend.gender import predict_gender_face, predict_gender_w2v2_scores
from backend.media_io import _ffmpeg_bin, probe_duration

OUT = ROOT / "data/gender_bench"
CLIP_S = 90.0
MIN_SPEAK_FRAMES = 25          # ~1 s at 25 fps: shorter is not a voice sample
ASD_SPEAK_THRESHOLD = 0.5      # talknet score above which a face is speaking


def cut(src: Path, dst: Path, start: float, dur: float, *, audio_only=False):
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_ffmpeg_bin(), "-y", "-v", "error", "-ss", f"{start:.3f}",
           "-t", f"{dur:.3f}", "-i", str(src)]
    # `-r 25` is not cosmetic. TalkNet pairs 100 fps audio features with
    # video frames 4:1; on a variable-frame-rate source the frame count
    # drifts from the audio length and the model dies with
    # "size of tensor a (111) must match tensor b (106)". The repo's own
    # clip runner forces a constant rate for the same reason, and strips
    # audio from the video so there is exactly one audio timeline.
    cmd += (["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]
            if audio_only else
            ["-an", "-r", "25", "-c:v", "libx264", "-preset", "fast",
             "-crf", "20", "-pix_fmt", "yuv420p"])
    subprocess.run(cmd + [str(dst)], check=True, capture_output=True)
    return dst


def run_one(name: str, src: Path) -> list:
    dur = probe_duration(src)
    start = max(0.0, (dur - CLIP_S) / 2) if dur > CLIP_S else 0.0
    span = min(CLIP_S, dur)
    work = OUT / name
    clip = cut(src, work / "clip.mp4", start, span)
    # The audio is cut LONGER than the video on purpose. The TalkNet adapter
    # takes `audio[track_start : track_start + n_frames * 640]` and clamps the
    # end to `len(audio)`, so a face track running to the last video frame
    # gets a short slice whose MFCC falls a few frames under the 4:1 the
    # model's frontend expects — and the whole call dies with "size of tensor
    # a (19) must match the size of tensor b (20)". A tail of spare audio
    # means the clamp never bites. Nothing reads past the track's own span.
    wav = cut(src, work / "clip.16k.wav", start, span + 2.0, audio_only=True)
    print(f"\n=== {name}  ({dur:.0f}s source, clip {start:.0f}-{start+span:.0f}s)",
          flush=True)

    npz = work / "faces.npz"
    if npz.exists():
        from models.faces import FaceTracks
        tracks = FaceTracks.load_npz(npz)     # the first run already paid for these
    else:
        tracks = detect_and_track(clip, detector="retinaface",
                                  tracker="bytetrack", every_n=1,
                                  min_score=0.5, tracks_out=npz)
    print(f"  tracks: {len(tracks.tracks)}", flush=True)
    asd = score_asd(clip, wav, tracks, model="talknet")

    rows = []
    for tr in tracks.tracks:
        tid = tr.track_id
        frames = np.asarray(tr.frames)
        bboxes = np.asarray(tr.bboxes)
        scores = asd.per_track.get(tid)
        if scores is None or frames.size == 0:
            continue
        s = np.asarray(scores.scores, dtype=float)
        n = min(len(s), len(frames))
        speaking = s[:n] >= ASD_SPEAK_THRESHOLD
        if speaking.sum() < MIN_SPEAK_FRAMES:
            continue

        face_g, face_conf, votes = predict_gender_face(clip, frames, bboxes)
        if face_g is None:
            continue

        spk_frames = frames[:n][speaking]
        t0 = float(spk_frames.min()) / tracks.fps
        t1 = float(spk_frames.max() + 1) / tracks.fps
        seg = cut(wav, work / f"{tid}.speech.wav", t0, max(1.0, t1 - t0),
                  audio_only=True)

        scores = predict_gender_w2v2_scores(seg)
        voice_g = max(scores, key=scores.__getitem__) if scores else None
        # The winning probability, kept because the two catalog errors this
        # model makes are 0.94 and 0.999 confident -- a low number here is
        # informative, a high one is not evidence of anything.
        voice_p = round(max(scores.values()), 3) if scores else None
        rows.append({
            "video": name, "track": tid,
            # Where this voice sample came from. `clip_*` is relative to the
            # 90 s cut; `src_*` is the timestamp in the original video, which
            # is the one you can actually scrub to and check by ear.
            "clip_start_s": round(start, 1),
            "clip_t0": round(t0, 1), "clip_t1": round(t1, 1),
            "src_t0": round(start + t0, 1), "src_t1": round(start + t1, 1),
            "speech_s": round(t1 - t0, 1),
            "speaking_frames": int(speaking.sum()),
            # Two predictions of the same thing, from two signals.
            "face": face_g, "face_conf": round(face_conf, 2), "votes": votes,
            "voice": voice_g, "voice_p": voice_p,
        })
        print(f"  {tid:9s} {start+t0:6.1f}-{start+t1:6.1f}s  "
              f"face={face_g:6s}({face_conf:.2f})  "
              f"voice={voice_g}({voice_p})", flush=True)
    return rows


VIDEOS = {
    "dark_knight_L3oOldViIgY": next(
        iter(sorted(ROOT.glob("data/jobs/*L3oOldViIgY*/videos/*.mp4"))),
        ROOT / "data/gender_bench/src/L3oOldViIgY.mp4"),
    # Per the `youtube-media-fetch` skill, source media lives in
    # `data/videos/<slug>_<video_id>/` with a slug naming the real video and
    # a `meta.json` beside it. Globbed by id so a re-slug does not break this.
    "john_wick_51t2_o_H_Rw": next(
        iter(sorted(ROOT.glob("data/videos/*51t2_o_H_Rw/*.mp4"))), None),
    "matrix_GjOWdF1flYE": next(
        iter(sorted(ROOT.glob("data/videos/*GjOWdF1flYE/*.mp4"))), None),
    "iit_profs_svwrTOm55y4":    ROOT / "data/clips/IIT_svwrTOm55y4_240to270.mp4",
}

if __name__ == "__main__":
    all_rows = []
    for name, src in VIDEOS.items():
        if src is None or not src.exists():
            print(f"!! missing {name}: {src}", flush=True)
            continue
        try:
            all_rows += run_one(name, src)
        except Exception as e:
            print(f"!! {name} failed: {type(e).__name__}: {e}", flush=True)
    (OUT / "results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\nwrote {len(all_rows)} rows -> {OUT/'results.json'}", flush=True)
