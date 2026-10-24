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
    python scripts/setup.py --no-talknet    # skip the TalkNet repo+ckpt fetch

Model download guidelines
-------------------------
Only ONE weights blob is fetched explicitly here: the TalkNet-ASD
pretrained checkpoint, because its repo is not pip-installable. Everything
else is pulled on first use by the adapter that needs it:

* pyannote/speaker-diarization-community-1 — HF-gated. Accept the model
  card at https://hf.co/pyannote/speaker-diarization-community-1 then put
  the token in `.env` as HF_TOKEN=…
* BUT-FIT/diarizen-wavlm-large-s80-md — HF, non-commercial. Dropped from
  the default requirements due to dep conflicts; install into a separate
  venv (`pip install git+https://github.com/BUTSpeechFIT/DiariZen.git`)
  if you need it.
* InsightFace `buffalo_l` (RetinaFace) — auto-downloaded (~250 MB) on
  first `--face-detector retinaface` run.
* WhisperX / faster-whisper — auto-downloaded on first `--asr whisperx` run.
* AI4Bharat IndicConformer 600M — auto-downloaded on first
  `--asr indicconformer` run.
* MediaPipe face detector — bundled with the pip wheel.
* Bodhan ASR / MT / TTS — hosted service, no local weights. Fill the
  BODHAN_* keys in `.env`.
* TS-TalkNet — optional. Clone from https://github.com/Jiang-Yidi/TS-TalkNet
  manually and point TS_TALKNET_* at the clone + its checkpoint + its
  speaker-encoder checkpoint (upstream README has the links).
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

# TalkNet-ASD — not pip-installable; cloned under third_party/ (gitignored).
# Checkpoint ID is published in the TalkNet-ASD README.
THIRD_PARTY = REPO_ROOT / "third_party"
TALKNET_REPO_DIR = THIRD_PARTY / "TalkNet-ASD"
TALKNET_REPO_URL = "https://github.com/TaoRuijie/TalkNet-ASD.git"
TALKNET_CKPT_NAME = "pretrain_TalkSet.model"
TALKNET_CKPT_GDRIVE_ID = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"

# Dry-run defaults — one of the three reference videos from README, first 60s.
DRY_RUN_URL = "https://www.youtube.com/watch?v=nsIMQP0QhpE"
DRY_RUN_SECONDS = 60

# ---------- pretty printing ----------
C_GREEN = "\033[32m"; C_YELLOW = "\033[33m"; C_RED = "\033[31m"
C_CYAN = "\033[36m"; C_DIM = "\033[2m"; C_RESET = "\033[0m"

def _ok(msg: str) -> None:   print(f"{C_GREEN}✓{C_RESET} {msg}")
def _warn(msg: str) -> None: print(f"{C_YELLOW}!{C_RESET} {msg}")
def _fail(msg: str) -> None: print(f"{C_RED}✗{C_RESET} {msg}")
def _say(msg: str) -> None:  print(f"{C_CYAN}==>{C_RESET} {msg}")


# ---------- checks ----------
def check_python() -> bool:
    v = sys.version_info
    if (v.major, v.minor) < (3, 10):
        _fail(f"python {v.major}.{v.minor} < 3.10 — the pipeline targets 3.10+")
        return False
    _ok(f"python {v.major}.{v.minor}.{v.micro}")
    return True


def check_ffmpeg() -> bool:
    if shutil.which("ffmpeg"):
        _ok("ffmpeg on PATH")
        return True
    _fail("ffmpeg not on PATH — apt install ffmpeg / brew install ffmpeg")
    return False


def check_cuda() -> None:
    """Informational — no return value, never fails."""
    try:
        import torch  # type: ignore
    except ImportError:
        _warn("torch not importable from this interpreter")
        return
    if torch.cuda.is_available():
        _ok(f"CUDA available ({torch.cuda.device_count()}x {torch.cuda.get_device_name(0)})")
    else:
        _warn("no CUDA — pipeline will run on CPU and be very slow")


def ensure_data_dirs(*, fix: bool) -> bool:
    data = REPO_ROOT / "data"
    missing = [data / d for d in DATA_SUBDIRS if not (data / d).is_dir()]
    if not missing:
        _ok("data/ subdirs present")
        return True
    if not fix:
        for p in missing:
            _fail(f"missing dir: {p.relative_to(REPO_ROOT)}")
        return False
    for p in missing:
        p.mkdir(parents=True, exist_ok=True)
    _ok(f"created {len(missing)} data/ subdirs")
    return True


def ensure_env_file(*, fix: bool) -> bool:
    env = REPO_ROOT / ".env"
    example = REPO_ROOT / ".env.example"
    if not env.exists():
        if not fix:
            _fail(".env missing (cp .env.example .env && chmod 600 .env)")
            return False
        if not example.exists():
            _fail(".env.example missing — cannot create .env")
            return False
        shutil.copy(example, env)
        env.chmod(0o600)
        _warn("copied .env.example → .env (0600) — fill HF_TOKEN before running pyannote")
        return True
    mode = env.stat().st_mode & 0o777
    if mode & 0o077:
        if not fix:
            _fail(f".env mode {oct(mode)} is group/world-readable (chmod 600 .env)")
            return False
        env.chmod(0o600)
        _ok(f".env chmod {oct(mode)} → 600")
    else:
        _ok(".env present and 0600")
    return True


def check_hf_token() -> bool:
    """Returns True iff a non-empty HF_TOKEN is set in env or .env."""
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        env = REPO_ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                m = re.match(r"^\s*HF_TOKEN\s*=\s*(.+?)\s*$", line)
                if m and m.group(1):
                    _ok("HF_TOKEN set in .env")
                    return True
        _warn("HF_TOKEN not set — pyannote diarizer and HF-gated models will fail")
        return False
    _ok("HF_TOKEN set in environment")
    return True


# ---------- TalkNet fetch ----------
def _patch_env_var(text: str, key: str, value: str) -> str:
    """Set key=value in dotenv text, replacing any existing (possibly empty) line."""
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    replacement = f"{key}={value}"
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text.rstrip() + f"\n{replacement}\n"


def _write_talknet_env_paths() -> None:
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    text = env.read_text()
    text = _patch_env_var(text, "TALKNET_REPO", str(TALKNET_REPO_DIR))
    text = _patch_env_var(text, "TALKNET_CHECKPOINT", str(TALKNET_REPO_DIR / TALKNET_CKPT_NAME))
    env.write_text(text)
    env.chmod(0o600)
    _ok("TALKNET_REPO / TALKNET_CHECKPOINT written into .env")


def ensure_talknet(*, fix: bool) -> bool:
    repo_ok = TALKNET_REPO_DIR.is_dir() and (TALKNET_REPO_DIR / "demoTalkNet.py").is_file()
    ckpt = TALKNET_REPO_DIR / TALKNET_CKPT_NAME
    ckpt_ok = ckpt.is_file() and ckpt.stat().st_size > 10_000_000  # >10 MB sanity floor

    if repo_ok and ckpt_ok:
        _ok(f"TalkNet-ASD ready at {TALKNET_REPO_DIR.relative_to(REPO_ROOT)}")
        return True

    if not fix:
        if not repo_ok: _warn(f"TalkNet-ASD repo missing at {TALKNET_REPO_DIR.relative_to(REPO_ROOT)}")
        if not ckpt_ok: _warn(f"TalkNet checkpoint missing: {ckpt.relative_to(REPO_ROOT)}")
        return False

    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    if not repo_ok:
        _say(f"cloning TalkNet-ASD → {TALKNET_REPO_DIR.relative_to(REPO_ROOT)}")
        if TALKNET_REPO_DIR.exists():
            shutil.rmtree(TALKNET_REPO_DIR)
        subprocess.check_call(
            ["git", "clone", "--depth", "1", TALKNET_REPO_URL, str(TALKNET_REPO_DIR)]
        )

    if not ckpt_ok:
        _say(f"downloading {TALKNET_CKPT_NAME} from Google Drive (upstream link)")
        try:
            import gdown  # type: ignore
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "gdown>=5.0"])
            import gdown  # type: ignore
        gdown.download(id=TALKNET_CKPT_GDRIVE_ID, output=str(ckpt), quiet=False)
        if not ckpt.is_file() or ckpt.stat().st_size < 10_000_000:
            _fail(
                f"TalkNet checkpoint download failed. Fetch manually from the "
                f"TalkNet-ASD README and place it at {ckpt}"
            )
            return False

    _write_talknet_env_paths()
    _ok("TalkNet-ASD ready")
    return True


# ---------- dry run ----------
def dry_run(url: str, seconds: int, *, with_asd: bool) -> bool:
    """Download the first `seconds` of `url` and run the pipeline on it.

    No ASR — that would need Bodhan credentials. Diarization is pyannote;
    face+ASD are added when TalkNet is available locally.
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
        cli += ["--face-detector", "retinaface", "--asd", "talknet"]

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
    ap.add_argument("--no-talknet", action="store_true",
                    help="Skip the TalkNet repo+checkpoint fetch (pyannote-only run).")
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

    talknet_ok = True
    if not args.no_talknet:
        talknet_ok = ensure_talknet(fix=fix)

    if not all(hard):
        print(f"\n{C_RED}Required checks failed.{C_RESET} Fix the ✗ items and re-run.")
        return 1

    if args.check_only:
        print(f"\n{C_GREEN}Doctor finished.{C_RESET} "
              f"{C_DIM}(HF_TOKEN: {'ok' if hf_ok else 'missing'}, "
              f"TalkNet: {'ok' if talknet_ok else 'missing'}){C_RESET}")
        return 0

    if args.no_dry_run:
        print(f"\n{C_GREEN}Setup complete.{C_RESET} Skipped dry run (--no-dry-run).")
        return 0

    if not hf_ok:
        print(f"\n{C_YELLOW}Skipping dry run{C_RESET} — pyannote needs HF_TOKEN. "
              f"Set it in .env, then re-run `./setup.sh` (or `python scripts/setup.py`).")
        return 0

    ok = dry_run(args.video_url, args.seconds, with_asd=talknet_ok)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
