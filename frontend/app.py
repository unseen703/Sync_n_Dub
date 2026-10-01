"""Gradio frontend for the dubbing pipeline scaffold.

Run:
    python frontend/app.py                    # binds 127.0.0.1 only
    python frontend/app.py --host 0.0.0.0     # LAN — requires DUBBING_UI_USER/PASS
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add repo root to sys.path so `backend` / `models` import when run directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio as gr
from dotenv import load_dotenv

from backend.pipeline import DEFAULT_DATA_DIR, run_pipeline
from models.diarization import available_models

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

REFERENCE_URLS = [
    "https://www.youtube.com/watch?v=nsIMQP0QhpE",
    "https://www.youtube.com/watch?v=P26AE7NLx4Q",
    "https://www.youtube.com/watch?v=2vxQaT0Fy-c",
]


def _run(
    url: str,
    uploaded_video,          # gr.File payload — Path-like or None
    model: str,
    num_speakers: int,
    skip_stems: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not url and not uploaded_video:
        raise gr.Error("Provide a YouTube URL or upload a video file.")
    if url and uploaded_video:
        raise gr.Error("Provide either a URL or a file, not both.")

    input_ = url.strip() if url else str(uploaded_video)
    num_speakers = None if num_speakers <= 0 else int(num_speakers)

    progress(0.0, desc=f"Running pipeline with {model} ...")
    result = run_pipeline(
        input_,
        diarization_model=model,
        data_dir=DEFAULT_DATA_DIR,
        num_speakers=num_speakers,
        skip_stems=skip_stems,
    )

    stems_note = " · **stem separation skipped**" if result.stems_skipped else ""
    summary = (
        f"**Model:** `{result.diarization_model}`  \n"
        f"**Speakers detected:** {result.num_speakers}  \n"
        f"**Turns:** {result.num_turns}{stems_note}  \n"
        f"**Video:** `{result.video}`  \n"
        f"**Manifest:** `{result.manifest}`"
    )
    rttm_text = Path(result.rttm).read_text()

    return (
        str(result.video),
        str(result.audio_16k),
        str(result.dialogue_stem) if result.dialogue_stem else None,
        str(result.me_stem) if result.me_stem else None,
        summary,
        rttm_text,
        str(result.rttm),
    )


with gr.Blocks(title="Dubbing Studio — Pipeline Scaffold") as demo:
    gr.Markdown("# Dubbing Studio — Pipeline Scaffold")
    gr.Markdown(
        "Download → extract audio → separate stems → diarize. "
        "Pick a YouTube URL *or* upload a local video."
    )

    with gr.Row():
        with gr.Column():
            url = gr.Textbox(label="YouTube URL", placeholder=REFERENCE_URLS[0])
            gr.Examples(examples=REFERENCE_URLS, inputs=url, label="Reference clips")
            uploaded = gr.File(label="… or upload a video", file_types=["video"])
            model = gr.Dropdown(
                choices=available_models(),
                value="pyannote",
                label="Diarization model",
                info="pyannote community-1 · diarizen (non-commercial)",
            )
            num_speakers = gr.Number(
                value=0, precision=0,
                label="Force number of speakers (0 = auto)",
            )
            skip_stems = gr.Checkbox(
                value=False,
                label="Skip stem separation (diarize raw audio)",
            )
            run_btn = gr.Button("Run pipeline", variant="primary")

        with gr.Column():
            out_summary = gr.Markdown(label="Summary")
            out_video = gr.Video(label="Input video", interactive=False)
            out_audio = gr.Audio(label="Extracted 16 kHz mono", interactive=False)
            out_dialogue = gr.Audio(label="Dialogue stem (vocals)", interactive=False)
            out_me = gr.Audio(label="Music & effects stem", interactive=False)
            out_rttm_text = gr.Code(label="RTTM (diarization)", language="markdown")
            out_rttm_file = gr.File(label="RTTM download")

    run_btn.click(
        _run,
        inputs=[url, uploaded, model, num_speakers, skip_stems],
        outputs=[out_video, out_audio, out_dialogue, out_me, out_summary, out_rttm_text, out_rttm_file],
    )


def _launch() -> None:
    ap = argparse.ArgumentParser(description="Gradio UI for the dubbing pipeline.")
    ap.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address. Default 127.0.0.1 (loopback only). Setting anything "
             "else (e.g. 0.0.0.0) requires DUBBING_UI_USER and DUBBING_UI_PASS "
             "in the environment, because this UI runs yt-dlp against arbitrary "
             "user-supplied URLs — an unauthenticated LAN-exposed instance is "
             "an SSRF / GPU-abuse vector.",
    )
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    auth = None
    if args.host != "127.0.0.1":
        user, pw = os.environ.get("DUBBING_UI_USER"), os.environ.get("DUBBING_UI_PASS")
        if not user or not pw:
            raise SystemExit(
                "Refusing to bind to a non-loopback address without auth. "
                "Set DUBBING_UI_USER and DUBBING_UI_PASS in the environment, "
                "or drop the --host flag to bind loopback only."
            )
        auth = (user, pw)
        logging.warning(
            "Gradio bound to %s:%d with HTTP basic auth. This UI runs yt-dlp "
            "against arbitrary URLs — do not expose to untrusted networks.",
            args.host, args.port,
        )

    demo.launch(server_name=args.host, server_port=args.port, auth=auth)


if __name__ == "__main__":
    _launch()
