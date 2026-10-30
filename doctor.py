"""One doctor, one row per feature.

    $PY ./doctor.py                  # everything
    $PY ./doctor.py --quick          # skip the slow/import-heavy checks
    $PY ./doctor.py --json           # machine-readable, for the UI's
                                            # /api/system/readiness endpoint

Every check is read-only: nothing here installs, downloads, writes `.env`,
or starts a job. PASS/WARN/FAIL per row, and a row names the fix rather than
just the symptom — the whole point is that a missing dependency shows up
here, once, in plain language, instead of as a job that dies at stage 9h
after ASR/MT/TTS/mix have already run, or as "The job failed (unrecognised)"
in the UI.

Reuses rather than re-derives: `./fetch_assets.py`'s manifest for
vendor/weight presence, `models.lipsync.base.resolve_env_python` for
interpreter resolution, `backend.gpu.ensure_onnxruntime_cuda` for the CUDA
EP check, `models.restore.gfpgan_adapter.GfpganRestore.blockers()` for the
restore stage. A check added here for something already checked elsewhere
should call that code, not copy it (CLAUDE.md's canonical-implementations
rule applies to diagnostics too).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import fetch_assets as FA  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
Row = Tuple[str, str, str]  # (status, name, detail)


def _row(status: str, name: str, detail: str) -> Row:
    return (status, name, detail)


# ---------------------------------------------------------------------------
# host basics
# ---------------------------------------------------------------------------

def check_interpreter() -> Row:
    return _row(PASS, "interpreter", sys.executable)


def check_env_file() -> Row:
    env = REPO_ROOT / ".env"
    if not env.is_file():
        return _row(WARN, ".env", "missing — copy .env.example to .env and fill it in")
    mode = oct(env.stat().st_mode)[-3:]
    if mode != "600":
        return _row(WARN, ".env", f"mode is {mode}, not 600 — `chmod 600 .env`")
    return _row(PASS, ".env", "present, mode 600")


def check_secret(var: str, *, required_group: Optional[List[str]] = None) -> Row:
    """Presence only — never the value, never logged, never in --json either."""
    val = os.environ.get(var, "")
    present = bool(val.strip())
    if required_group:
        others = [os.environ.get(v, "").strip() for v in required_group if v != var]
        if present and not all(others):
            missing = [v for v in required_group if v != var and not os.environ.get(v, "").strip()]
            return _row(WARN, var, f"set, but {', '.join(missing)} is not — both or neither")
    return _row(PASS if present else WARN, var, "set" if present else "not set")


def check_gpus() -> Row:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return _row(WARN, "GPUs", f"nvidia-smi unavailable ({e}) — CPU-only run")
    if out.returncode != 0:
        return _row(WARN, "GPUs", f"nvidia-smi failed: {out.stderr.strip()}")
    lines = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return _row(WARN, "GPUs", "nvidia-smi returned no devices")
    return _row(PASS, "GPUs", f"{len(lines)} device(s): " + " | ".join(lines))


def check_ffmpeg() -> Row:
    from backend.media_io import _ffmpeg_bin, _ffprobe_bin
    ff = _ffmpeg_bin()
    fp = _ffprobe_bin()
    if not ff or not fp:
        return _row(FAIL, "ffmpeg/ffprobe",
                    f"ffmpeg={ff or 'NOT FOUND'} ffprobe={fp or 'NOT FOUND'}")
    filters = subprocess.run([ff, "-hide_banner", "-filters"],
                              capture_output=True, text=True).stdout
    has_atempo = "atempo" in filters
    has_rubberband = "rubberband" in filters
    note = f"atempo={'yes' if has_atempo else 'NO'} rubberband={'yes' if has_rubberband else 'no (expected — see docs/HOST.md)'}"
    if not has_atempo:
        return _row(FAIL, "ffmpeg/ffprobe", f"{ff}: {note}")
    return _row(PASS, "ffmpeg/ffprobe", f"{ff} — {note}")


def check_onnxruntime_cuda() -> Row:
    try:
        from backend.gpu import ensure_onnxruntime_cuda
        ok = ensure_onnxruntime_cuda()
    except Exception as e:
        return _row(WARN, "onnxruntime CUDA EP", f"could not check: {e}")
    return _row(PASS if ok else WARN, "onnxruntime CUDA EP",
                "active" if ok else "fell back to CPU — see docs/HOST.md "
                "(onnxruntime/onnxruntime-gpu conflict, or missing CUDA libs)")


def check_tmux() -> Row:
    import shutil
    return _row(PASS if shutil.which("tmux") else WARN, "tmux",
                "found — detached UI jobs use it" if shutil.which("tmux")
                else "not found — UI jobs fall back to a plain detached process (fine, but not attachable)")


# ---------------------------------------------------------------------------
# per-env interpreter checks
# ---------------------------------------------------------------------------

#: (label, env_name, model_arg for resolve_env_python, modules to import-test)
_ENV_CHECKS = [
    ("musetalk",   "musetalk",   "musetalk",   ["torch", "diffusers", "mmcv", "mmdet", "mmpose", "chumpy"]),
    ("latentsync", "latentsync", "latentsync", ["torch", "diffusers", "insightface", "decord"]),
    ("restore",    "restore",    "restore",    ["torch", "basicsr", "facexlib", "gfpgan", "realesrgan"]),
]


def check_sidecar_envs(*, quick: bool) -> List[Row]:
    from models.lipsync.base import resolve_env_python
    rows = []
    for label, env_name, model_arg, modules in _ENV_CHECKS:
        py = resolve_env_python(model_arg, env_name)
        if py == sys.executable:
            rows.append(_row(FAIL, f"env:{label}",
                              f"no sibling '{env_name}' conda env found — resolved to this "
                              f"process's own interpreter, which will fail on `import "
                              f"{modules[1] if len(modules) > 1 else modules[0]}`"))
            continue
        if not Path(py).is_file():
            rows.append(_row(FAIL, f"env:{label}", f"resolved to {py}, which does not exist"))
            continue
        if quick:
            rows.append(_row(PASS, f"env:{label}", f"{py} (import check skipped, --quick)"))
            continue
        check = "; ".join(f"import {m}" for m in modules)
        proc = subprocess.run([py, "-c", check], capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            rows.append(_row(PASS, f"env:{label}", f"{py} — {', '.join(modules)} all import clean"))
        else:
            err = (proc.stderr or proc.stdout).strip().splitlines()[-1] if (proc.stderr or proc.stdout) else "unknown error"
            rows.append(_row(FAIL, f"env:{label}", f"{py}: {err}"))
    return rows


def check_dubui() -> Row:
    from models.lipsync.base import resolve_env_python  # noqa: F401 (consistency of import site)
    base = Path(sys.executable).resolve().parent.parent.parent
    node = base / "dubui" / "bin" / "node"
    if not node.is_file():
        return _row(WARN, "env:dubui",
                     f"no sibling 'dubui' env at {node} — UI build needs "
                     f"`conda create -n dubui nodejs=20` (see envs/dubui.yml)")
    proc = subprocess.run([str(node), "-v"], capture_output=True, text=True)
    return _row(PASS, "env:dubui", f"{node} — {proc.stdout.strip()}")


def check_ui_build() -> Row:
    web_dist = REPO_ROOT / "ui" / "web-dist" / "index.html"
    web_src = REPO_ROOT / "ui" / "web" / "src"
    if not web_dist.is_file():
        return _row(WARN, "ui build",
                     "ui/web-dist/ missing — server is API-only, no UI. Build it: conda run -n dubui npm --prefix ui/web ci "
                     "&& conda run -n dubui npm --prefix ui/web run build")
    if web_src.is_dir():
        newest_src = max((p.stat().st_mtime for p in web_src.rglob("*") if p.is_file()), default=0)
        if newest_src > web_dist.stat().st_mtime:
            return _row(WARN, "ui build", "ui/web-dist/ is OLDER than ui/web/src/ — rebuild it")
    return _row(PASS, "ui build", str(web_dist))


# ---------------------------------------------------------------------------
# assets (vendor repos + weights), via the fetch_assets manifest
# ---------------------------------------------------------------------------

def check_assets(*, quick: bool) -> List[Row]:
    manifest = FA.load_manifest()
    rows = []
    for v in manifest["vendor_repos"]:
        state = FA.status_vendor(v)
        status = PASS if state == "present" else FAIL
        rows.append(_row(status, f"vendor:{v['name']}",
                          state if state == "present" else
                          f"{state} — `$PY ./fetch_assets.py --feature {v['feature']}`"))
    required = [w for w in manifest["weights"] if not w.get("optional")]
    missing = []
    for w in required:
        state = FA.status_weight(w) if quick else FA.check_weight(w)
        ok = state in ("present", "ok")
        if not ok:
            missing.append(w["id"])
    if missing:
        rows.append(_row(FAIL, "weights (required)",
                          f"{len(missing)}/{len(required)} missing or mismatched: "
                          f"{', '.join(missing[:6])}{'…' if len(missing) > 6 else ''} — "
                          f"`$PY ./fetch_assets.py --all`"))
    else:
        rows.append(_row(PASS, "weights (required)", f"all {len(required)} present"))
    return rows


def check_restore_blockers() -> Row:
    try:
        from models.restore.gfpgan_adapter import GFPGANRestore
        blockers = GFPGANRestore().blockers()
    except Exception as e:
        return _row(WARN, "restore (gfpgan)", f"could not check: {e}")
    if blockers:
        return _row(FAIL, "restore (gfpgan)", "; ".join(blockers))
    return _row(PASS, "restore (gfpgan)", "ready")


# ---------------------------------------------------------------------------
# small data sets
# ---------------------------------------------------------------------------

def check_voice_refs() -> Row:
    from backend.voices import VOICE_REF_DIR
    d = VOICE_REF_DIR / "indic-speak"
    n = len(list(d.glob("*.wav"))) if d.is_dir() else 0
    if n == 0:
        return _row(WARN, "voice reference clips", f"0 at {d} — voice casting falls "
                    "back from timbre-matched to language+gender only.")
    if n < 45:
        return _row(WARN, "voice reference clips", f"{n}/45 at {d} — add the missing reference WAVs")
    return _row(PASS, "voice reference clips", f"{n}/45 at {d}")


def check_talking_head_audio() -> Row:
    catalogue = REPO_ROOT / "data" / "talking_head" / "dialogues.json"
    audio_dir = REPO_ROOT / "data" / "talking_head" / "audio"
    if not catalogue.is_file():
        return _row(WARN, "talking-head catalogue", f"{catalogue} missing")
    from ui.talking_head import dialogues_for, languages_with_dialogues
    want = sum(1 for lang in languages_with_dialogues()
               for e in dialogues_for(lang) if e.get("audio_url"))
    have = len(list(audio_dir.glob("*.wav"))) if audio_dir.is_dir() else 0
    if have < want:
        return _row(WARN, "talking-head audio",
                    f"{have}/{want} clips at {audio_dir} — every missing one shows as "
                    f"playable but refuses at render.")
    return _row(PASS, "talking-head audio", f"{have}/{want} clips")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_all(*, quick: bool) -> List[Row]:
    rows: List[Row] = []
    rows.append(check_interpreter())
    rows.append(check_env_file())
    rows.append(check_gpus())
    rows.append(check_ffmpeg())
    if not quick:
        rows.append(check_onnxruntime_cuda())
    rows.append(check_tmux())

    rows.append(check_secret("HF_TOKEN"))
    rows.append(check_secret("BODHAN_ASR_KEY"))
    rows.append(check_secret("BODHAN_MT_KEY"))
    rows.append(check_secret("BODHAN_TTS_KEY"))
    rows.append(check_secret("OPENROUTER_API_KEY"))
    rows.append(check_secret("LANGFUSE_PUBLIC_KEY", required_group=["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]))
    rows.append(check_secret("LANGFUSE_SECRET_KEY", required_group=["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]))

    rows.extend(check_sidecar_envs(quick=quick))
    rows.append(check_dubui())
    rows.append(check_ui_build())

    rows.extend(check_assets(quick=quick))
    rows.append(check_restore_blockers())

    rows.append(check_voice_refs())
    rows.append(check_talking_head_audio())
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                     help="skip sha256 verification and sidecar-env import tests "
                          "(size/existence only — fast, for a pre-job sanity check)")
    ap.add_argument("--json", action="store_true", help="machine-readable output, no secret values ever")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    rows = run_all(quick=args.quick)

    if args.json:
        print(json.dumps([{"status": s, "name": n, "detail": d} for s, n, d in rows], indent=2))
    else:
        width = max(len(n) for _, n, _ in rows)
        for status, name, detail in rows:
            print(f"[{status:4}] {name:<{width}}  {detail}")
        n_fail = sum(1 for s, _, _ in rows if s == FAIL)
        n_warn = sum(1 for s, _, _ in rows if s == WARN)
        print(f"\n{n_fail} FAIL, {n_warn} WARN, {len(rows) - n_fail - n_warn} PASS")

    return 1 if any(s == FAIL for s, _, _ in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
