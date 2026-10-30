"""Warm-model Wav2Lip driver: load the net once, render many pieces.

Runs under the env `Wav2LipSync` resolves, with `--repo` at the vendored clone.
The vendored `inference.py` parses argv at IMPORT and builds the model inside
its render loop, so a segmented render reloads it per piece. This imports the
module's internals instead of editing the clone: a minimal `sys.argv` lets the
import-time `parse_args()` succeed, the net loads once, then each piece
overrides `inference.args.*` and runs the same render body.

Manifest: a JSON list of objects
  {"face": ..., "audio": ..., "out": ...,
   "pads": [t,b,l,r], "resize_factor": int,
   "wav2lip_batch_size": int, "face_det_batch_size": int,
   "use_saved_coord": bool, "saved_coord": bool, "coord_path": ... | null}
Per piece it prints `[worker] ok <out>` or `[worker] fail <out> :: <reason>`.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import traceback


def _render_piece(inference, w2l_audio, model, device, scratch, p) -> None:
    """One (face, audio) -> out render reusing the already-loaded `model`.

    A faithful copy of the vendored `inference.main()` body, with the model
    hoisted out and the hardcoded `temp/...` paths moved into `scratch` so
    sequential pieces cannot collide. Reads `inference.args`, which the caller
    sets per piece.
    """
    import cv2
    import numpy as np
    import torch

    args = inference.args
    face = p["face"]
    audio_path = p["audio"]
    outfile = p["out"]

    # ---- read frames ------------------------------------------------------
    if face.split(".")[-1].lower() in ("jpg", "png", "jpeg"):
        full_frames = [cv2.imread(face)]
        fps = args.fps
    else:
        cap = cv2.VideoCapture(face)
        fps = cap.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(
                    frame,
                    (frame.shape[1] // args.resize_factor,
                     frame.shape[0] // args.resize_factor))
            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]
            full_frames.append(frame[y1:y2, x1:x2])

    # ---- audio -> mel chunks ----------------------------------------------
    wav_path = audio_path
    if not audio_path.endswith(".wav"):
        wav_path = os.path.join(scratch, "temp.wav")
        subprocess.call(
            "ffmpeg -y -i {} -strict -2 {}".format(audio_path, wav_path),
            shell=True)
    wav = w2l_audio.load_wav(wav_path, 16000)
    mel = w2l_audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError("Mel contains nan -- add a small epsilon noise to the "
                         "driving wav and retry")

    mel_chunks = []
    mel_idx_multiplier = 80.0 / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + inference.mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - inference.mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx:start_idx + inference.mel_step_size])
        i += 1

    full_frames = full_frames[:len(mel_chunks)]

    # ---- batch inference --------------------------------------------------
    batch_size = args.wav2lip_batch_size
    gen = inference.datagen(full_frames.copy(), mel_chunks)
    avi = os.path.join(scratch, "result.avi")
    out_writer = None
    try:
        for img_batch, mel_batch, frames, coords in gen:
            if out_writer is None:
                h, w = full_frames[0].shape[:-1]
                out_writer = cv2.VideoWriter(
                    avi, cv2.VideoWriter_fourcc(*"DIVX"), fps, (w, h))
            img_t = torch.FloatTensor(
                np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_t = torch.FloatTensor(
                np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            with torch.no_grad():
                pred = model(mel_t, img_t)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            for pr, fr, co in zip(pred, frames, coords):
                y1, y2, x1, x2 = co
                fr[y1:y2, x1:x2] = cv2.resize(
                    pr.astype(np.uint8), (x2 - x1, y2 - y1))
                out_writer.write(fr)
    finally:
        if out_writer is not None:
            out_writer.release()

    # ---- mux the driving audio onto the rendered video --------------------
    rc = subprocess.call(
        "ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}".format(
            wav_path, avi, outfile),
        shell=True)
    if rc != 0 or not (os.path.isfile(outfile) and os.path.getsize(outfile)):
        raise RuntimeError(f"ffmpeg mux produced no output (rc={rc})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="vendored Wav2Lip repo")
    ap.add_argument("--manifest", required=True, help="JSON list of pieces")
    ap.add_argument("--checkpoint_path", required=True)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    with open(args.manifest) as fh:
        pieces = json.load(fh)

    # The vendored module parses argv AT IMPORT, so hand it a minimal valid
    # set; every real value is set on `inference.args` per piece.
    saved_argv = sys.argv
    sys.argv = ["inference.py",
                "--checkpoint_path", args.checkpoint_path,
                "--face", "placeholder", "--audio", "placeholder"]
    try:
        import inference            # vendored Wav2Lip/inference.py
        import audio as w2l_audio   # vendored Wav2Lip/audio.py
    finally:
        sys.argv = saved_argv

    import torch

    model = inference.load_model(args.checkpoint_path)
    device = inference.device
    print(f"[worker] model loaded once for {len(pieces)} piece(s)", flush=True)

    scratch = os.path.join(repo, "_warm_scratch")
    os.makedirs(scratch, exist_ok=True)

    rc = 0
    for i, p in enumerate(pieces):
        out = p["out"]
        print(f"[worker] start {i + 1}/{len(pieces)} {out}", flush=True)
        # Per-piece knobs onto the globals datagen/face_detect read.
        a = inference.args
        a.face = p["face"]
        a.audio = p["audio"]
        a.outfile = out
        a.pads = list(p.get("pads", [0, 10, 0, 0]))
        a.resize_factor = int(p.get("resize_factor", 1))
        a.wav2lip_batch_size = int(p.get("wav2lip_batch_size", 16))
        a.face_det_batch_size = int(p.get("face_det_batch_size", 4))
        a.use_saved_coord = bool(p.get("use_saved_coord", False))
        a.saved_coord = bool(p.get("saved_coord", False))
        a.coord_path = p.get("coord_path")
        a.nosmooth = True
        a.box = [-1, -1, -1, -1]
        a.static = False
        a.crop = [0, -1, 0, -1]
        a.rotate = False
        a.img_size = 96
        a.fps = getattr(a, "fps", 25.0)
        try:
            _render_piece(inference, w2l_audio, model, device, scratch, p)
            print(f"[worker] ok   {out}", flush=True)
        except Exception as e:                                # noqa: BLE001
            rc = 1
            print(f"[worker] fail {out} :: {e}", flush=True)
            traceback.print_exc()
        finally:
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
