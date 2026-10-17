#!/usr/bin/env python
"""ASD-gated clip run: faces -> ASD -> mix -> MuseTalk.

Takes one time window of one video, drives it with an already-downloaded
YouTube dub track (so MT and TTS are skipped entirely), and lip-syncs only the
shots that ASD says contain a single speaking face.

What the run always writes is what the lip-sync depends on -- face tracks, the
per-track ASD numbers, the shot gating decision, and the cut segments:

    faces/clip.<detector>_<tracker>.npz   tracked boxes
    faces/face_tracks.json                the same, readable
    asd/asd_scores.json                   per-track speaking summary
    asd/shot_gating.json                  which shots get lip-synced, and why
    lipsync/shot<N>.src.mp4               the segments handed to MuseTalk

`--bbox-preview` adds the diagnostic renders on top: boxed-face video and
stills, an ASD-coloured video and stills, per-speaker crops, the score plot,
and MuseTalk's own crop box dumped before inference. MuseTalk reads none of
them -- it is handed shot<N>.src.mp4 and redoes its own detection -- so they
are for a human checking which face the gate picked, and they are not cheap:
the crop dump alone averaged 262 s/run on the 2026-10-15 indic15 batch.

Why gating matters here: `backend/lipsync.py::run_lipsync` hands the whole
video to the adapter, and MuseTalk (like Wav2Lip) animates one face per frame
with no identity tracking. On a multi-face wide shot that is an arbitrary face.
Rather than animate the wrong person, this driver lip-syncs only single-face
shots and passes the rest through untouched.

Run it detached -- see scripts/run_clip_asd_musetalk.sh.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FF = "/data1/dipak_data/miniconda3/envs/dubbing/bin/ffmpeg"
FFPROBE = "/data1/dipak_data/miniconda3/envs/dubbing/bin/ffprobe"
MUSETALK_PY = "/data1/dipak_data/miniconda3/envs/musetalk/bin/python"
MUSETALK_REPO = ROOT / "third_party" / "MuseTalk"

log = logging.getLogger("clip_run")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log.info("$ %s", " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if p.returncode != 0:
        log.error("rc=%d stderr:\n%s", p.returncode, p.stderr[-4000:])
        raise RuntimeError(f"command failed rc={p.returncode}: {cmd[0]}")
    return p


def probe(path: Path) -> dict:
    p = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,"
         "nb_read_packets,sample_rate,channels",
         "-count_packets", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(p.stdout)


def nframes(path: Path) -> int:
    for s in probe(path).get("streams", []):
        if s.get("codec_type") == "video":
            return int(s.get("nb_read_packets", 0))
    return 0


def duration(path: Path) -> float:
    return float(probe(path)["format"]["duration"])


class VramSampler(threading.Thread):
    """Peak VRAM is what decides whether the next clip fits, so sample it
    during the run rather than reading the final value."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak: dict[int, int] = {}
        self._stop = threading.Event()
        self.marks: list[tuple[str, dict[int, int]]] = []

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout
                for line in out.strip().splitlines():
                    idx, used = (x.strip() for x in line.split(","))
                    i, u = int(idx), int(used)
                    self.peak[i] = max(self.peak.get(i, 0), u)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def mark(self, label: str) -> None:
        self.marks.append((label, dict(self.peak)))

    def stop(self) -> None:
        self._stop.set()


@dataclass
class Phase:
    name: str
    t0: float
    t1: Optional[float] = None

    @property
    def wall(self) -> float:
        return (self.t1 or time.time()) - self.t0


class Run:
    def __init__(self, job_dir: Path, status_path: Path):
        self.job_dir = job_dir
        self.status_path = status_path
        self.phases: list[Phase] = []
        self.notes: dict = {}

    def phase(self, name: str) -> Phase:
        if self.phases and self.phases[-1].t1 is None:
            self.phases[-1].t1 = time.time()
        p = Phase(name, time.time())
        self.phases.append(p)
        log.info("=" * 70)
        log.info("PHASE %s", name)
        log.info("=" * 70)
        self.write_status(f"running {name}")
        return p

    def write_status(self, state: str) -> None:
        """One line on disk so a cold session does not parse the whole log."""
        payload = {
            "state": state,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "phases": [{"name": p.name, "wall_s": round(p.wall, 1),
                        "done": p.t1 is not None} for p in self.phases],
            "notes": self.notes,
        }
        self.status_path.write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------
# P1 -- cut the window
# --------------------------------------------------------------------------

def phase_cut(run: Run, src: Path, dub: Path, en_audio: Path,
              start: float, dur: float, width: int, height: int,
              fps: int) -> dict:
    d = run.job_dir / "videos"; d.mkdir(parents=True, exist_ok=True)
    a = run.job_dir / "audio"; a.mkdir(parents=True, exist_ok=True)

    clip = d / "clip.mp4"
    sh([FF, "-y", "-v", "error", "-ss", start, "-i", src, "-t", dur,
        "-an", "-vf", f"scale={width}:{height}", "-r", fps,
        "-c:v", "libx264", "-crf", "16", "-preset", "medium",
        "-pix_fmt", "yuv420p", clip])

    en16k = a / "clip.en.16k.mono.wav"
    sh([FF, "-y", "-v", "error", "-ss", start, "-i", en_audio, "-t", dur,
        "-ar", "16000", "-ac", "1", en16k])

    hi48k = a / "clip.dub.48k.stereo.wav"
    sh([FF, "-y", "-v", "error", "-ss", start, "-i", dub, "-t", dur,
        "-ar", "48000", "-ac", "2", hi48k])

    info = {
        "clip_mp4": str(clip), "clip_frames": nframes(clip),
        "clip_duration_s": round(duration(clip), 3),
        "en_16k": str(en16k), "en_duration_s": round(duration(en16k), 3),
        "dub_48k": str(hi48k), "dub_duration_s": round(duration(hi48k), 3),
    }
    expected = int(round(dur * fps))
    if info["clip_frames"] != expected:
        log.warning("clip has %d frames, expected %d", info["clip_frames"], expected)
    log.info("cut: %s", json.dumps(info, indent=2))
    return info


def detect_shots(clip: Path, threshold: float = 0.2) -> list[float]:
    """Shot-cut times in clip-relative seconds. Lip-sync is applied per shot:
    a face box is only meaningful within one continuous shot."""
    p = subprocess.run(
        [FF, "-v", "info", "-i", str(clip), "-an",
         "-vf", f"select=gt(scene\\,{threshold}),metadata=print", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    times: list[float] = []
    for line in p.stderr.splitlines():
        if "pts_time:" in line:
            try:
                times.append(float(line.split("pts_time:")[1].split()[0]))
            except (IndexError, ValueError):
                continue
    return sorted(times)


# --------------------------------------------------------------------------
# P2 -- face detect + track, and render what was detected
# --------------------------------------------------------------------------

def phase_faces(run: Run, clip: Path, detector: str, tracker: str,
                device: str, min_score: float):
    from backend.faces import detect_and_track

    out = run.job_dir / "faces"; out.mkdir(parents=True, exist_ok=True)
    tracks = detect_and_track(
        clip, detector=detector, tracker=tracker, device=device,
        every_n=1, min_score=min_score,
        tracks_out=out / f"clip.{detector}_{tracker}.npz",
    )
    summary = {
        "detector": detector, "tracker": tracker,
        "fps": tracks.fps, "num_frames": tracks.num_frames,
        "num_tracks": len(tracks.tracks),
        "tracks": [t.to_jsonable() for t in tracks.tracks],
    }
    (out / "face_tracks.json").write_text(json.dumps(summary, indent=2))
    log.info("faces: %d tracks over %d frames",
             len(tracks.tracks), tracks.num_frames)
    for t in tracks.tracks:
        fr0, fr1 = t.frame_range
        log.info("  track %-6s frames %4d-%4d (%3d) mean_bbox=%s",
                 t.track_id, fr0, fr1, t.num_frames,
                 [round(v, 1) for v in t.mean_bbox().tolist()])
    return tracks, summary


def render_boxes(clip: Path, tracks, out_video: Path, out_stills: Path,
                 asd=None, still_times: Optional[list[float]] = None,
                 fps: int = 25) -> None:
    """Draw every tracked face box on every frame. When `asd` is given the box
    is coloured by that track's ASD score at that frame, which is the whole
    point of the render: you can see which face the model thinks is talking."""
    import cv2
    import numpy as np

    out_stills.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(clip))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = out_video.with_suffix(".tmp.avi")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    by_frame: dict[int, list[tuple]] = {}
    for t in tracks.tracks:
        for i, f in enumerate(t.frames.tolist()):
            by_frame.setdefault(int(f), []).append(
                (t.track_id, t.bboxes[i], float(t.scores[i])))

    asd_lookup: dict[tuple[str, int], float] = {}
    if asd is not None:
        for tid, sc in asd.per_track.items():
            for f, s in zip(sc.frames.tolist(), sc.scores.tolist()):
                asd_lookup[(tid, int(f))] = float(s)

    want = set()
    if still_times:
        want = {int(round(t * fps)) for t in still_times}

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, bbox, det in by_frame.get(idx, []):
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])
            label = f"{tid} det={det:.2f}"
            colour = (0, 200, 255)          # amber: detected, no ASD opinion
            if asd is not None:
                s = asd_lookup.get((tid, idx))
                if s is not None:
                    label = f"{tid} asd={s:.2f}"
                    colour = (0, 220, 0) if s >= 0.5 else (60, 60, 230)
            thick = max(2, int(round(w / 960)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)
            fs = max(0.5, w / 1920 * 0.8)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)),
                          (x1 + tw + 6, y1), colour, -1)
            cv2.putText(frame, label, (x1 + 3, max(th, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thick)
        stamp = f"t={idx / fps:6.2f}s  f={idx:4d}"
        cv2.putText(frame, stamp, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.6, w / 1920), (255, 255, 255), 2)
        vw.write(frame)
        if idx in want:
            cv2.imwrite(str(out_stills / f"t{idx / fps:05.2f}s_f{idx:04d}.png"), frame)
        idx += 1
    cap.release(); vw.release()
    sh([FF, "-y", "-v", "error", "-i", tmp, "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", out_video])
    tmp.unlink(missing_ok=True)
    log.info("rendered %s (%d frames) + %d stills", out_video, idx, len(want))


def speaker_summary(clip: Path, tracks, asd, out_dir: Path, fps: int = 25,
                    write_crops: bool = True) -> list:
    """Per-track ASD summary rows, plus -- only when `write_crops` -- the face
    crop at each track's highest-ASD frame.

    The rows are the load-bearing half: they land in `asd/asd_scores.json` and
    describe who the model thinks is speaking. The PNGs are for a human to
    look at. The rows must come out identical either way, so a frame that
    cannot be decoded costs its crop, never its row -- otherwise the numbers
    would silently depend on whether pictures were asked for.

    With `write_crops=False` the video is never opened at all.
    """
    import numpy as np

    rows = []
    cap = None
    if write_crops:
        import cv2
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(clip))
    for t in tracks.tracks:
        sc = asd.per_track.get(t.track_id)
        if sc is None or sc.scores.size == 0:
            continue
        best_i = int(np.argmax(sc.scores))
        best_frame = int(sc.frames[best_i])
        best_score = float(sc.scores[best_i])
        where = np.where(t.frames == best_frame)[0]
        if where.size == 0:
            continue
        bbox = t.bboxes[int(where[0])]
        row = {
            "track_id": t.track_id,
            "best_frame": best_frame,
            "best_time_s": round(best_frame / fps, 2),
            "best_asd_score": round(best_score, 4),
            "mean_asd_score": round(float(sc.scores.mean()), 4),
            "frames_speaking_ge_0.5": int((sc.scores >= 0.5).sum()),
            "num_frames": int(sc.frames.size),
            "bbox_xyxy_at_best": [round(float(v), 1) for v in bbox[:4]],
        }
        if cap is not None:
            import cv2
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = (int(v) for v in bbox[:4])
                mx, my = int((x2 - x1) * 0.35), int((y2 - y1) * 0.35)
                cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
                cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
                crop = frame[cy1:cy2, cx1:cx2].copy()
                cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1),
                              (0, 220, 0) if best_score >= 0.5 else (60, 60, 230), 3)
                name = f"{t.track_id}_asd{best_score:.2f}_f{best_frame:04d}.png"
                cv2.imwrite(str(out_dir / name), crop)
                row["crop_png"] = str(out_dir / name)
            else:
                log.warning("  speaker crop: %s frame %d would not decode; "
                            "row kept, crop skipped", t.track_id, best_frame)
        rows.append(row)
        log.info("  speaker: %s best=%.3f mean=%.3f at t=%.2fs",
                 t.track_id, best_score, float(sc.scores.mean()),
                 best_frame / fps)
    if cap is not None:
        cap.release()
    return rows


def plot_asd(asd, out_png: Path, fps: int = 25, shots: Optional[list] = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                       # plotting is a nicety
        log.warning("skipping ASD plot: %s", e)
        return
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for tid, sc in sorted(asd.per_track.items()):
        if sc.frames.size == 0:
            continue
        ax.plot(sc.frames / fps, sc.scores, label=tid, linewidth=1.4)
    ax.axhline(0.5, ls="--", c="k", lw=0.8, alpha=0.6)
    for s in (shots or []):
        ax.axvline(s, c="magenta", lw=1.0, alpha=0.7)
    ax.set_xlabel("clip time (s)"); ax.set_ylabel("TalkNet ASD score")
    ax.set_title("Active-speaker score per face track (dashed = 0.5; magenta = shot cut)")
    ax.set_ylim(-0.05, 1.05); ax.legend(ncol=4, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    log.info("asd plot -> %s", out_png)


# --------------------------------------------------------------------------
# P5 -- MuseTalk's own crop box, dumped before inference
# --------------------------------------------------------------------------

MUSETALK_PREVIEW = r'''
import json, sys, os
import cv2
sys.path.insert(0, os.getcwd())
from musetalk.utils.preprocessing import get_landmark_and_bbox

frames_dir, out_json, out_dir, bbox_shift = sys.argv[1:5]
bbox_shift = int(bbox_shift)
os.makedirs(out_dir, exist_ok=True)
imgs = sorted(
    os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
    if f.endswith(".png")
)
coord_list, frame_list = get_landmark_and_bbox(imgs, bbox_shift)
rows = []
for i, (coord, img) in enumerate(zip(coord_list, frame_list)):
    x1, y1, x2, y2 = [int(v) for v in coord]
    rows.append({"frame": i, "bbox_xyxy": [x1, y1, x2, y2],
                 "w": x2 - x1, "h": y2 - y1,
                 "placeholder": (x1, y1, x2, y2) == (0, 0, 0, 0)})
    if i % 20 == 0 and (x1, y1, x2, y2) != (0, 0, 0, 0):
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 255), 3)
        cv2.putText(vis, f"musetalk crop {x2-x1}x{y2-y1}", (x1, max(24, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
        cv2.imwrite(os.path.join(out_dir, "musetalk_bbox_f%04d.png" % i), vis)
json.dump({"bbox_shift": bbox_shift, "num_frames": len(rows), "boxes": rows},
          open(out_json, "w"), indent=2)
print("wrote", out_json, len(rows), "boxes")
'''


def phase_musetalk_bbox(run: Run, seg_video: Path, bbox_shift: int) -> dict:
    """Dump the crop MuseTalk will actually use, before it runs.

    `bbox_shift` moves the *top* edge of the crop, not the bottom -- see
    third_party/MuseTalk/musetalk/utils/preprocessing.py:119
    (`half_face_coord[1] = upperbondrange + half_face_coord[1]`, where index 1
    is y1). The bottom edge is `extra_margin`, which this adapter does not
    plumb through. Left at 0 here; the dump is for inspection, not tuning.
    """
    out = run.job_dir / "lipsync_preview"; out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"; frames_dir.mkdir(exist_ok=True)
    sh([FF, "-y", "-v", "error", "-i", seg_video,
        "-start_number", "0", str(frames_dir / "%08d.png")])

    script = out / "_preview.py"
    script.write_text(MUSETALK_PREVIEW)
    out_json = out / "musetalk_boxes.json"
    env = dict(os.environ)
    env["LD_PRELOAD"] = "/data1/dipak_data/miniconda3/envs/musetalk/lib/libstdc++.so.6"
    p = subprocess.run(
        [MUSETALK_PY, str(script), str(frames_dir), str(out_json),
         str(out), str(bbox_shift)],
        cwd=str(MUSETALK_REPO), env=env, capture_output=True, text=True,
    )
    log.info("musetalk preview stdout:\n%s", p.stdout[-2000:])
    if p.returncode != 0:
        log.error("musetalk preview stderr:\n%s", p.stderr[-4000:])
        raise RuntimeError("musetalk bbox preview failed")
    data = json.loads(out_json.read_text())
    boxes = [b for b in data["boxes"] if not b["placeholder"]]
    log.info("musetalk crop: %d/%d frames have a face box",
             len(boxes), data["num_frames"])
    if boxes:
        ws = [b["w"] for b in boxes]; hs = [b["h"] for b in boxes]
        log.info("  crop size min/mean/max: %dx%d / %dx%d / %dx%d",
                 min(ws), min(hs), sum(ws) // len(ws), sum(hs) // len(hs),
                 max(ws), max(hs))
    shutil.rmtree(frames_dir, ignore_errors=True)
    return data


# --------------------------------------------------------------------------
# P6/P7 -- lip-sync the gated shot, then splice it back
# --------------------------------------------------------------------------

def cut_segment(src: Path, out: Path, start: float, end: float,
                fps: int, audio: Optional[Path] = None) -> Path:
    args = [FF, "-y", "-v", "error", "-ss", f"{start:.6f}", "-i", str(src),
            "-t", f"{end - start:.6f}"]
    if audio is not None:
        args += ["-ss", f"{start:.6f}", "-i", str(audio), "-t", f"{end - start:.6f}",
                 "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
    else:
        args += ["-an"]
    args += ["-r", str(fps), "-c:v", "libx264", "-crf", "16", "-preset", "medium",
             "-pix_fmt", "yuv420p", str(out)]
    sh(args)
    return out


def phase_lipsync(run: Run, seg_video: Path, seg_audio: Path, out: Path,
                  batch_size: int, gpu: int, version: str, bbox_shift: int,
                  use_float16: bool = False):
    from models.lipsync import LipSyncRequest, get_lipsync

    adapter = get_lipsync(
        "musetalk", python_bin=MUSETALK_PY, repo_path=str(MUSETALK_REPO),
        version=version, batch_size=batch_size, fps=25,
        bbox_shift=bbox_shift, use_float16=use_float16, cuda_device=gpu,
    )
    t0 = time.time()
    res = adapter.sync(LipSyncRequest(video=seg_video, audio=seg_audio, out_path=out))
    wall = time.time() - t0
    log.info("musetalk: model=%s out=%s fps=%.2f dur=%.2fs wall=%.1fs err=%r",
             res.model_name, res.out_video, res.fps, res.duration_s, wall, res.error)
    if res.error:
        raise RuntimeError(f"musetalk failed: {res.error}")
    return res, wall


def concat_video(parts: list[Path], out: Path, fps: int) -> Path:
    """Concat with the filter (one re-encode) rather than the demuxer: the
    MuseTalk output does not share encoder params with the pass-through cuts."""
    args = [FF, "-y", "-v", "error"]
    for p in parts:
        args += ["-i", str(p)]
    streams = "".join(f"[{i}:v:0]" for i in range(len(parts)))
    args += ["-filter_complex", f"{streams}concat=n={len(parts)}:v=1:a=0[v]",
             "-map", "[v]", "-r", str(fps), "-c:v", "libx264", "-crf", "16",
             "-preset", "medium", "-pix_fmt", "yuv420p", str(out)]
    sh(args)
    return out


def mux(video: Path, audio: Path, out: Path) -> Path:
    """48 kHz stereo AAC. Wav2Lip's 16 kHz mono AAC is silently refused by
    several players -- not a mistake worth repeating here."""
    sh([FF, "-y", "-v", "error", "-i", video, "-i", audio,
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest", out])
    return out


def frame_diff(a: Path, b: Path, start: float, dur: float) -> float:
    """Mean abs pixel difference between two videos over a window. A lip-sync
    that 'ran green' but changed nothing shows up here as ~0."""
    import cv2
    import numpy as np
    ca, cb = cv2.VideoCapture(str(a)), cv2.VideoCapture(str(b))
    fps = ca.get(cv2.CAP_PROP_FPS) or 25.0
    f0, f1 = int(start * fps), int((start + dur) * fps)
    ca.set(cv2.CAP_PROP_POS_FRAMES, f0); cb.set(cv2.CAP_PROP_POS_FRAMES, f0)
    tot, n = 0.0, 0
    for _ in range(max(0, f1 - f0)):
        oka, fa = ca.read(); okb, fb = cb.read()
        if not (oka and okb):
            break
        if fa.shape != fb.shape:
            fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]))
        tot += float(np.abs(fa.astype(np.int16) - fb.astype(np.int16)).mean())
        n += 1
    ca.release(); cb.release()
    return tot / n if n else 0.0


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dub-audio", required=True, type=Path)
    ap.add_argument("--en-audio", required=True, type=Path)
    ap.add_argument("--start", required=True, type=float)
    ap.add_argument("--duration", required=True, type=float)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--detector", default="retinaface")
    ap.add_argument("--tracker", default="bytetrack")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--asd-threshold", type=float, default=0.5,
                    help="Per-frame TalkNet score above which a frame counts "
                         "as speaking.")
    ap.add_argument("--asd-min-speaking-frac", type=float, default=0.25,
                    help="Fraction of a shot's frames a track must be "
                         "speaking in to count as the speaker of that shot.")
    ap.add_argument("--bbox-preview", action="store_true",
                    help="Render the diagnostic overlays: boxed face video + "
                         "stills, ASD-coloured video + stills, per-speaker "
                         "crops, the score plot, and MuseTalk's own crop dump "
                         "(P5). None of it feeds lip-sync -- MuseTalk redoes "
                         "its own detection from the segment mp4 -- and P5 "
                         "alone cost 262 s/run on the 2026-10-15 indic15 "
                         "batch, 61%% of lip-sync's own time. Off by default; "
                         "turn it on when a human needs to check which face "
                         "the gate picked.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--bbox-shift", type=int, default=0)
    ap.add_argument("--musetalk-version", default="v15")
    ap.add_argument("--use-float16", action="store_true",
                    help="Run MuseTalk in half precision (upstream "
                         "--use_float16).")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--talknet-repo", type=Path,
                    default=ROOT / "third_party" / "TalkNet-ASD",
                    help="Passed explicitly so the run never reads .env.")
    ap.add_argument("--talknet-checkpoint", type=Path,
                    default=ROOT / "third_party" / "TalkNet-ASD" / "pretrain_TalkSet.model")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    job_dir = args.data_dir / "jobs" / args.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    status_path = job_dir / "STATUS.json"
    run = Run(job_dir, status_path)

    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(name)-22s %(levelname)-7s %(message)s",
    )

    cfg = {
        "src": str(args.src), "dub_audio": str(args.dub_audio),
        "en_audio": str(args.en_audio),
        "window_abs_s": [args.start, args.start + args.duration],
        "window_hhmmss": [time.strftime("%H:%M:%S", time.gmtime(args.start)),
                          time.strftime("%H:%M:%S", time.gmtime(args.start + args.duration))],
        "language": args.lang, "working_resolution": f"{args.width}x{args.height}",
        "fps": args.fps, "detector": args.detector, "tracker": args.tracker,
        "face_min_score": args.min_score, "asd_model": "talknet",
        "asd_threshold": args.asd_threshold,
        "asd_min_speaking_frac": args.asd_min_speaking_frac,
        "lipsync_model": f"musetalk-{args.musetalk_version}",
        "lipsync_batch_size": args.batch_size, "bbox_shift": args.bbox_shift,
        "lipsync_precision": "fp16" if args.use_float16 else "fp32",
        "gpu": args.gpu, "mt_tts": "SKIPPED (downloaded YouTube dub drives audio)",
        "face_policy": "ASD-gated: lip-sync only single-face shots",
        "bbox_preview": args.bbox_preview,
        "talknet_repo": str(args.talknet_repo),
        "talknet_checkpoint": str(args.talknet_checkpoint),
    }
    log.info("CONFIG\n%s", json.dumps(cfg, indent=2))
    (job_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    run.notes["config"] = cfg

    vram = VramSampler(); vram.start()
    t_start = time.time()
    try:
        # ---- P1 cut -------------------------------------------------------
        run.phase("P1_CUT")
        cut = phase_cut(run, args.src, args.dub_audio, args.en_audio,
                        args.start, args.duration, args.width, args.height, args.fps)
        clip = Path(cut["clip_mp4"])
        en16k = Path(cut["en_16k"]); dub48k = Path(cut["dub_48k"])
        shots = detect_shots(clip)
        bounds = [0.0] + shots + [round(cut["clip_duration_s"], 3)]
        log.info("shot cuts at %s -> %d shots", shots, len(bounds) - 1)
        run.notes["cut"] = cut
        run.notes["shot_cuts_s"] = shots

        # ---- P2 faces -----------------------------------------------------
        run.phase("P2_FACES")
        tracks, face_summary = phase_faces(run, clip, args.detector, args.tracker,
                                           f"cuda:{args.gpu}", args.min_score)
        vram.mark("after_faces")
        still_times = [b + 0.6 for b in bounds[:-1]]
        if args.bbox_preview:
            render_boxes(clip, tracks,
                         job_dir / "faces" / "faces_detected.mp4",
                         job_dir / "faces" / "stills",
                         asd=None, still_times=still_times, fps=args.fps)

        # ---- P3 ASD -------------------------------------------------------
        # TalkNet is driven by the ORIGINAL English audio: it correlates speech
        # with lip motion, and the YouTube machine dub is not lip-synchronous
        # with this footage. Scoring it against the dub would measure nothing.
        run.phase("P3_ASD")
        from backend.asd import score_asd
        asd = score_asd(clip, en16k, tracks, model="talknet",
                        device=f"cuda:{args.gpu}",
                        repo_path=args.talknet_repo,
                        checkpoint=args.talknet_checkpoint)
        vram.mark("after_asd")
        # The asd/ dir used to be created as a side effect of writing crops.
        # It has to exist whether or not any picture is written.
        (job_dir / "asd").mkdir(parents=True, exist_ok=True)
        per_track = speaker_summary(clip, tracks, asd,
                                    job_dir / "asd" / "speaker_crops", args.fps,
                                    write_crops=args.bbox_preview)
        (job_dir / "asd" / "asd_scores.json").write_text(json.dumps(
            {"model": asd.model_name, "fps": asd.fps,
             "driving_audio": str(en16k),
             "driving_audio_note": "original English, not the dub",
             "per_track": per_track}, indent=2))
        if args.bbox_preview:
            plot_asd(asd, job_dir / "asd" / "asd_scores.png", args.fps, shots)
            render_boxes(clip, tracks,
                         job_dir / "asd" / "asd_speaking.mp4",
                         job_dir / "asd" / "stills",
                         asd=asd, still_times=still_times, fps=args.fps)

        # ---- gate: which shots get lip-synced? -----------------------------
        import numpy as np
        gated = []
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            f0, f1 = int(round(s * args.fps)), int(round(e * args.fps))
            present = [t for t in tracks.tracks
                       if ((t.frames >= f0) & (t.frames < f1)).sum() >= (f1 - f0) * 0.5]
            # A speaker pauses mid-shot, so the shot MEAN is the wrong
            # statistic -- it buries a clearly-speaking track under its own
            # silences. Gate on the fraction of frames scored as speaking.
            speaking, stats = [], []
            for t in present:
                sc = asd.per_track.get(t.track_id)
                if sc is None:
                    continue
                m = (sc.frames >= f0) & (sc.frames < f1)
                if not m.sum():
                    continue
                win = sc.scores[m]
                frac = float((win >= args.asd_threshold).mean())
                stats.append({"track_id": t.track_id,
                              "speaking_frac": round(frac, 3),
                              "mean": round(float(win.mean()), 3),
                              "max": round(float(win.max()), 3)})
                if frac >= args.asd_min_speaking_frac:
                    speaking.append((t.track_id, round(frac, 3)))
            ok = len(present) == 1 and len(speaking) == 1
            shot = {"shot": i, "start_s": round(s, 3), "end_s": round(e, 3),
                    "frames": [f0, f1], "num_faces_present": len(present),
                    "face_ids": [t.track_id for t in present],
                    "asd_stats": stats,
                    "speaking": speaking, "lipsync": ok,
                    "reason": ("single speaking face" if ok else
                               f"{len(present)} faces present, {len(speaking)} "
                               f"with >={args.asd_min_speaking_frac:.0%} of frames "
                               f"above ASD {args.asd_threshold}")}
            gated.append(shot)
            log.info("shot %d [%6.2f-%6.2f] faces=%d %s -> %s (%s)",
                     i, s, e, len(present), [t.track_id for t in present],
                     "LIPSYNC" if ok else "pass-through", shot["reason"])
            for st in stats:
                log.info("    %-8s speaking_frac=%.3f mean=%.3f max=%.3f",
                         st["track_id"], st["speaking_frac"], st["mean"], st["max"])
        (job_dir / "asd" / "shot_gating.json").write_text(json.dumps(gated, indent=2))
        run.notes["gating"] = gated

        targets = [s for s in gated if s["lipsync"]]
        if not targets:
            log.error("no shot passed the ASD gate; nothing to lip-sync")
            run.write_status("failed: no gated shot")
            return 3

        # ---- P4 mix -------------------------------------------------------
        run.phase("P4_MIX")
        mix_dir = job_dir / "mix"; mix_dir.mkdir(parents=True, exist_ok=True)
        dubbed_wav = mix_dir / f"clip.dubbed.{args.lang}.wav"
        shutil.copy2(dub48k, dubbed_wav)
        dubbed_mp4 = mux(clip, dubbed_wav, mix_dir / f"clip.dubbed.{args.lang}.mp4")
        log.info("mix: %s (%.2fs) + %s", dubbed_mp4, duration(dubbed_mp4), dubbed_wav)

        # ---- P5 cut the gated shots (+ optional crop preview) --------------
        # Cutting shot<N>.src.mp4 is load-bearing: P6 reads those files back.
        # The bbox dump beside it is not -- it re-runs get_landmark_and_bbox,
        # which MuseTalk calls again itself seconds later.
        run.phase("P5_SEGMENT")
        seg_dir = job_dir / "lipsync"; seg_dir.mkdir(parents=True, exist_ok=True)
        previews = {}
        for s in targets:
            segv = cut_segment(clip, seg_dir / f"shot{s['shot']}.src.mp4",
                               s["start_s"], s["end_s"], args.fps)
            if args.bbox_preview:
                previews[s["shot"]] = phase_musetalk_bbox(run, segv, args.bbox_shift)
        if previews:
            run.notes["musetalk_boxes"] = {
                k: {"num_frames": v["num_frames"],
                    "with_face": sum(1 for b in v["boxes"] if not b["placeholder"])}
                for k, v in previews.items()}

        # ---- P6 lipsync ---------------------------------------------------
        run.phase("P6_LIPSYNC")
        synced: dict[int, Path] = {}
        lip_wall = 0.0
        for s in targets:
            segv = seg_dir / f"shot{s['shot']}.src.mp4"
            sega = seg_dir / f"shot{s['shot']}.dub.wav"
            sh([FF, "-y", "-v", "error", "-ss", f"{s['start_s']:.6f}",
                "-i", dubbed_wav, "-t", f"{s['end_s'] - s['start_s']:.6f}",
                "-ar", "48000", "-ac", "2", sega])
            out = seg_dir / f"shot{s['shot']}.musetalk.mp4"
            res, wall = phase_lipsync(run, segv, sega, out, args.batch_size,
                                      args.gpu, args.musetalk_version, args.bbox_shift,
                                      use_float16=args.use_float16)
            vram.mark(f"after_lipsync_shot{s['shot']}")
            lip_wall += wall
            got, want = nframes(out), s["frames"][1] - s["frames"][0]
            log.info("shot %d lipsync frames: got %d want %d", s["shot"], got, want)
            synced[s["shot"]] = out
        run.notes["lipsync_wall_s"] = round(lip_wall, 1)

        # ---- P7 splice ----------------------------------------------------
        run.phase("P7_SPLICE")
        parts = []
        for s in gated:
            if s["shot"] in synced:
                parts.append(synced[s["shot"]])
            else:
                parts.append(cut_segment(
                    clip, seg_dir / f"shot{s['shot']}.passthrough.mp4",
                    s["start_s"], s["end_s"], args.fps))
        video_only = concat_video(parts, seg_dir / "spliced.noaudio.mp4", args.fps)
        final = mux(video_only, dubbed_wav,
                    job_dir / f"final.{args.lang}.musetalk.asdgated.mp4")
        log.info("final: %s (%.3fs, %d frames)",
                 final, duration(final), nframes(final))

        # ---- P8 verify ----------------------------------------------------
        run.phase("P8_VERIFY")
        checks = {
            "final_duration_s": round(duration(final), 3),
            "final_frames": nframes(final),
            "source_clip_frames": cut["clip_frames"],
            "duration_matches_request": abs(duration(final) - args.duration) < 0.15,
        }
        for s in gated:
            d = frame_diff(clip, final, s["start_s"], s["end_s"] - s["start_s"])
            checks[f"shot{s['shot']}_mean_abs_pixel_diff"] = round(d, 3)
            checks[f"shot{s['shot']}_lipsynced"] = s["lipsync"]
            log.info("shot %d (%s): mean abs pixel diff vs source = %.3f",
                     s["shot"], "lipsynced" if s["lipsync"] else "pass-through", d)
        a = probe(final)
        checks["final_audio"] = [
            {"codec": s.get("codec_name"), "sr": s.get("sample_rate"),
             "ch": s.get("channels")}
            for s in a["streams"] if s.get("codec_type") == "audio"]
        vram.stop(); time.sleep(0.2)
        checks["peak_vram_mib"] = vram.peak
        checks["vram_marks"] = [{"at": m, "peak": p} for m, p in vram.marks]
        checks["total_wall_s"] = round(time.time() - t_start, 1)
        checks["phases"] = [{"name": p.name, "wall_s": round(p.wall, 1)}
                            for p in run.phases]
        (job_dir / "VERIFY.json").write_text(json.dumps(checks, indent=2))
        log.info("VERIFY\n%s", json.dumps(checks, indent=2))

        run.phases[-1].t1 = time.time()
        run.notes["verify"] = checks
        run.notes["final"] = str(final)
        run.write_status("done")
        log.info("=== DONE in %.1fs -> %s ===", time.time() - t_start, final)
        return 0

    except Exception as e:
        log.exception("run failed: %s", e)
        run.notes["error"] = str(e)
        run.write_status(f"failed: {e}")
        return 1
    finally:
        vram.stop()


if __name__ == "__main__":
    sys.exit(main())
