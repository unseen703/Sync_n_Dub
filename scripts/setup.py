"""Dubbing Studio — doctor, model fetcher, and dry-run.

SUPERSEDED as of the portability pass that added `envs/*.yml`,
`models/assets.lock.json`, `scripts/fetch_assets.py` and `scripts/doctor.py`.
`./setup.sh` now execs `scripts/install.sh` and no longer calls this file at
all -- it predates the 4-conda-env reality (CLAUDE.md), targets a single
`.venv`, and its "Only ONE weights blob is fetched explicitly" design is
exactly what `models/assets.lock.json` replaces. Left in place rather than
deleted because nothing has been asked to remove it yet; it is not on any
live entry point's path. Prefer `scripts/doctor.py` (checks) and
`scripts/fetch_assets.py` (weights) to anything below.

Entry point for `./setup.sh`; can also be run standalone inside the venv:

    python scripts/setup.py                 # fix anything missing + 60s dry run
    python scripts/setup.py --check-only    # report state, change nothing
    python scripts/setup.py --no-dry-run    # fix only, skip pipeline run

Model download guidelines
-------------------------
No weights are fetched here. The vendored upstreams and their checkpoints
are pinned in `models/assets.lock.json` and fetched by
`scripts/fetch_assets.py`, which verifies sha256; everything else is pulled
on first use by the adapter that needs it:

* pyannote/speaker-diarization-community-1 — HF-gated. Accept the model
  card at https://hf.co/pyannote/speaker-diarization-community-1 then put
  the token in `.env` as HF_TOKEN=…
* InsightFace `buffalo_l` (RetinaFace) — auto-downloaded (~250 MB) on
  first `--face-detector retinaface` run.
* MediaPipe face detector — bundled with the pip wheel.
* Bodhan ASR / MT / TTS — hosted service, no local weights. Fill the
  BODHAN_* keys in `.env`.
* LoCoNet (ASD) and the lip-sync upstreams — vendored clones plus
  checkpoints, pinned in `models/assets.lock.json`. Fetch with
  `python scripts/fetch_assets.py`.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Pipeline artefact directories (regenerable; all gitignored).
DATA_SUBDIRS = ("videos", "audio", "stems", "diarization", "faces", "asd",
                "asr", "align", "mt")

# ---------- .env helpers ----------
def _patch_env_var(text: str, key: str, value: str) -> str:
    """Set key=value in dotenv text, replacing any existing (possibly empty) line."""
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    replacement = f"{key}={value}"
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text.rstrip() + f"\n{replacement}\n"


# ---------- dry run ----------
def dry_run(url: str, seconds: int, *, with_asd: bool) -> bool:
    """Download the first `seconds` of `url` and run the pipeline on it.

    No ASR — that would need Bodhan credentials. Diarization is pyannote;
    face+ASD are added when the LoCoNet clone is present locally.
    """
    _say(f"dry run: first {seconds}s of {url}")

    clip = REPO_ROOT / "data" / "videos" / f"_dryrun_{seconds}s.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    if clip.exists():
        clip.unlink()

    yt_cmd = [
        "yt-dlp",
        "-f", "bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/best[ext=mp4]/best",
        "--download-sections", f"*0-{seconds}",
        "--force-keyframes-at-cuts",
        "-o", str(clip),
        url,
    ]
    try:
        subprocess.check_call(yt_cmd)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        _fail(f"yt-dlp failed: {e}")
        return False
    if not clip.is_file():
        _fail(f"clip not written: {clip}")
        return False

    cli = [
        sys.executable, "-m", "backend.cli",
        "--input", str(clip),
        "--model", "pyannote",
        "--data-dir", str(REPO_ROOT / "data"),
        "-v",
    ]
    if with_asd:
        cli += ["--face-detector", "retinaface", "--asd", "loconet"]

    _say("running: " + " ".join(cli))
    rc = subprocess.call(cli, cwd=REPO_ROOT)
    if rc != 0:
        _fail(f"backend.cli exited with code {rc}")
        return False
    _ok("dry run complete")
    return True


# ---------- entry point ----------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dubbing Studio doctor + model fetcher + dry run.")
    ap.add_argument("--check-only", action="store_true",
                    help="Report state only; make no changes.")
    ap.add_argument("--no-dry-run", action="store_true",
                    help="Skip the 60s pipeline run at the end.")
    ap.add_argument("--no-asd", action="store_true",
                    help="Skip face+ASD in the dry run (pyannote only).")
    ap.add_argument("--video-url", default=DRY_RUN_URL,
                    help=f"URL for the dry run (default: {DRY_RUN_URL}).")
    ap.add_argument("--seconds", type=int, default=DRY_RUN_SECONDS,
                    help=f"Seconds to pull for the dry run (default: {DRY_RUN_SECONDS}).")
    args = ap.parse_args(argv)

    fix = not args.check_only
    mode = "CHECK" if args.check_only else "SETUP"
    print(f"\n=== Dubbing Studio {mode} ===\n")

    hard = [
        check_python(),
        check_ffmpeg(),
        ensure_data_dirs(fix=fix),
        ensure_env_file(fix=fix),
    ]
    check_cuda()
    hf_ok = check_hf_token()

    asd_ok = (not args.no_asd) and (
        REPO_ROOT / "models" / "asd" / "vendor" / "loconet").is_dir()
    if not args.no_asd and not asd_ok:
        _warn("models/asd/vendor/loconet is missing; run "
              "`python scripts/fetch_assets.py --feature asd:loconet`. "
              "The dry run will skip face+ASD.")

    if not all(hard):
        print(f"\n{C_RED}Required checks failed.{C_RESET} Fix the ✗ items and re-run.")
        return 1

    if args.check_only:
        print(f"\n{C_GREEN}Doctor finished.{C_RESET} "
              f"{C_DIM}(HF_TOKEN: {'ok' if hf_ok else 'missing'}, "
              f"LoCoNet: {'ok' if asd_ok else 'missing'}){C_RESET}")
        return 0

    if args.no_dry_run:
        print(f"\n{C_GREEN}Setup complete.{C_RESET} Skipped dry run (--no-dry-run).")
        return 0

    if not hf_ok:
        print(f"\n{C_YELLOW}Skipping dry run{C_RESET} — pyannote needs HF_TOKEN. "
              f"Set it in .env, then re-run `./setup.sh` (or `python scripts/setup.py`).")
        return 0

    ok = dry_run(args.video_url, args.seconds, with_asd=asd_ok)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
