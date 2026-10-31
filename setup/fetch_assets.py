"""Fetch the vendored lip-sync/ASD repos and their weights onto a fresh clone.

Reads `setup/assets.lock.json` -- every vendor repo (pinned commit + patch)
and weight file (pinned source + sha256) this project needs beyond what pip
and conda can install. Nothing here is guessed: every commit, revision and
hash in that manifest was either read off a live checkout, cross-checked
against huggingface_hub's own download-cache metadata, or computed with
sha256sum and verified to match that metadata (see the manifest's own
`_comment`).

    $PY ./fetch_assets.py --list                 # status table, fetches nothing
    $PY ./fetch_assets.py --check                 # verify existing files' hashes
    $PY ./fetch_assets.py --feature lipsync:wav2lip
    $PY ./fetch_assets.py --all                   # every REQUIRED item
    $PY ./fetch_assets.py --all --include-optional

Idempotent: a file already on disk with the right size and sha256 is left
alone. `--dry-run` prints what each run would do without writing anything.

Three weight sources, by `source` in the manifest:
  hf      -- huggingface_hub.hf_hub_download(repo_id, revision, filename),
             which does its own integrity check against the hub; this script
             re-verifies the local sha256 afterwards against the manifest.
  url     -- a plain HTTPS GET (GitHub release assets), streamed to disk,
             sha256-verified after.
  gdrive  -- `gdown --id <id>`, Google Drive. Quota-limited rather than
             reliably scriptable -- a 429 here is an upstream limit, not a
             bug in this script. The manifest flags both gdrive-sourced
             files as worth mirroring to a project-controlled host.

Vendor repos: `git clone` at the pinned commit, then `git apply --check`
the patch before `git apply` for real -- refusing rather than leaving a
half-applied tree if the patch no longer fits the pinned commit (it always
should; the commit is pinned specifically so it always does).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = Path(__file__).resolve().parent / "assets.lock.json"
#: Must match backend/restore/gfpgan_adapter.py's DEFAULT_WEIGHTS_DIR --
#: both read the same RESTORE_WEIGHTS override.
DEFAULT_RESTORE_WEIGHTS = "backend/faces/vendor/gfpgan"

log = logging.getLogger("fetch_assets")


def _torch_home() -> Path:
    """Same default torch itself uses, since nothing here imports torch."""
    env = os.environ.get("TORCH_HOME")
    if env:
        return Path(env).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "torch"


def _resolve_dest(dest: str) -> Path:
    if dest.startswith("<TORCH_HOME>/"):
        return _torch_home() / dest[len("<TORCH_HOME>/"):]
    p = Path(dest)
    return p if p.is_absolute() else REPO_ROOT / p


def load_manifest() -> Dict[str, Any]:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _matches_feature(item_feature: str, wanted: List[str]) -> bool:
    if "all" in wanted:
        return True
    return item_feature in wanted or item_feature.split(":")[0] in wanted


# ---------------------------------------------------------------------------
# status / check
# ---------------------------------------------------------------------------

def status_weight(w: Dict[str, Any]) -> str:
    """'ok' | 'size-mismatch' | 'missing', without hashing (cheap for --list)."""
    dest = _resolve_dest(w["dest"])
    if not dest.is_file():
        return "missing"
    if dest.stat().st_size != w["bytes"]:
        return "size-mismatch"
    return "present"


def check_weight(w: Dict[str, Any]) -> str:
    """Full sha256 check -- 'ok' | 'hash-mismatch' | 'missing'."""
    dest = _resolve_dest(w["dest"])
    if not dest.is_file():
        return "missing"
    got = sha256_of(dest)
    return "ok" if got == w["sha256"] else "hash-mismatch"


def status_vendor(v: Dict[str, Any]) -> str:
    dest = REPO_ROOT / v["dest"]
    if not (dest / ".git").exists() and not (dest / ".git").is_file():
        return "missing"
    try:
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unreadable"
    if head != v["commit"]:
        return f"wrong-commit ({head[:12]})"
    if v.get("patch"):
        diff = subprocess.run(["git", "-C", str(dest), "diff"],
                               capture_output=True, text=True).stdout
        patch_text = (REPO_ROOT / v["patch"]).read_text()
        if diff.strip() != patch_text.strip():
            return "patch-not-applied"
    return "present"


def cmd_list(manifest: Dict[str, Any], features: List[str]) -> None:
    print("--- vendor repos ---")
    for v in manifest["vendor_repos"]:
        if not _matches_feature(v["feature"], features):
            continue
        print(f"  [{status_vendor(v):18}] {v['name']:12} {v['feature']:20} {v['dest']}")
    print("--- weights ---")
    for w in manifest["weights"]:
        if not _matches_feature(w["feature"], features):
            continue
        opt = " (optional)" if w.get("optional") else ""
        size_gb = w["bytes"] / 1e9
        print(f"  [{status_weight(w):13}] {w['id']:28} {size_gb:6.2f} GB  {w['feature']}{opt}")


def cmd_check(manifest: Dict[str, Any], features: List[str]) -> int:
    bad = 0
    for w in manifest["weights"]:
        if not _matches_feature(w["feature"], features):
            continue
        result = check_weight(w)
        mark = "OK  " if result == "ok" else "FAIL"
        print(f"[{mark}] {w['id']:28} {result}")
        if result != "ok" and result != "missing":
            bad += 1
        elif result == "missing" and not w.get("optional"):
            bad += 1
    for v in manifest["vendor_repos"]:
        if not _matches_feature(v["feature"], features):
            continue
        result = status_vendor(v)
        mark = "OK  " if result == "present" else "FAIL"
        print(f"[{mark}] {v['name']:28} {result}")
        if result not in ("present",):
            bad += 1
    if bad:
        print(f"\n{bad} problem(s) found.")
    else:
        print("\neverything checked matches the manifest.")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def fetch_vendor(v: Dict[str, Any], *, dry_run: bool) -> bool:
    dest = REPO_ROOT / v["dest"]
    state = status_vendor(v)
    if state == "present":
        log.info("%s: already at the pinned commit with its patch applied", v["name"])
        return True
    if dry_run:
        log.info("[dry-run] would clone %s @ %s -> %s%s", v["url"], v["commit"][:12],
                  dest, f" + apply {v['patch']}" if v.get("patch") else "")
        return True

    if dest.exists():
        log.warning("%s: %s exists but is %s; leaving it alone -- remove it "
                     "by hand and re-run if you want it replaced", v["name"], dest, state)
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("%s: cloning %s", v["name"], v["url"])
    subprocess.run(["git", "clone", "--no-checkout", v["url"], str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", v["commit"]], check=True)

    if v.get("patch"):
        patch_path = REPO_ROOT / v["patch"]
        check = subprocess.run(["git", "-C", str(dest), "apply", "--check", str(patch_path)],
                                capture_output=True, text=True)
        if check.returncode != 0:
            log.error("%s: %s does not apply at the pinned commit -- refusing "
                       "to leave a half-patched checkout.\n%s",
                       v["name"], v["patch"], check.stderr)
            return False
        subprocess.run(["git", "apply", str(patch_path)], cwd=str(dest), check=True)
        log.info("%s: applied %s", v["name"], v["patch"])
    return True


def _download_hf(w: Dict[str, Any], dest: Path) -> None:
    from huggingface_hub import hf_hub_download
    got = hf_hub_download(repo_id=w["repo"], filename=w["filename"],
                           revision=w.get("revision"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(got, dest)


def _download_url(w: Dict[str, Any], dest: Path) -> None:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(w["url"], stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    tmp.rename(dest)


def _download_gdrive(w: Dict[str, Any], dest: Path) -> None:
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=w["gdrive_id"], output=str(dest), quiet=False)


def fetch_weight(w: Dict[str, Any], *, dry_run: bool, force: bool) -> bool:
    dest = _resolve_dest(w["dest"])
    if not force and dest.is_file() and dest.stat().st_size == w["bytes"]:
        log.info("%s: already present at the expected size", w["id"])
        return True
    if dry_run:
        log.info("[dry-run] would fetch %s (%s, %.2f GB) -> %s",
                  w["id"], w["source"], w["bytes"] / 1e9, dest)
        return True

    log.info("%s: fetching from %s (%.2f GB) -> %s", w["id"], w["source"],
              w["bytes"] / 1e9, dest)
    try:
        if w["source"] == "hf":
            _download_hf(w, dest)
        elif w["source"] == "url":
            _download_url(w, dest)
        elif w["source"] == "gdrive":
            _download_gdrive(w, dest)
        else:
            log.error("%s: unknown source %r", w["id"], w["source"])
            return False
    except Exception as e:
        log.error("%s: download failed: %s", w["id"], e)
        return False

    got = sha256_of(dest)
    if got != w["sha256"]:
        log.error("%s: sha256 mismatch after download (got %s, expected %s) "
                   "-- removing the bad file", w["id"], got[:16], w["sha256"][:16])
        dest.unlink(missing_ok=True)
        return False
    log.info("%s: verified", w["id"])
    return True


def place_face_restore_weights() -> int:
    """Copy facexlib/gfpgan/realesrgan weights from backend/faces/vendor/gfpgan
    into the running interpreter's site-packages, so the libraries do not
    download them from GitHub on first use mid-render. Idempotent; a size
    match counts as "already placed".

    Called after a fetch run that matched the `restore` feature. Must run
    inside the `dubbing` env (basicsr / gfpgan / facexlib / realesrgan were
    folded in on 2026-10-30).
    """
    import sysconfig

    weights = _resolve_dest(os.getenv("RESTORE_WEIGHTS", DEFAULT_RESTORE_WEIGHTS))
    sp = Path(sysconfig.get_paths()["purelib"])
    targets = {
        "detection_Resnet50_Final.pth": [sp / "facexlib/weights"],
        "parsing_parsenet.pth": [sp / "facexlib/weights"],
        "GFPGANv1.4.pth": [sp / "gfpgan/weights"],
        "RealESRGAN_x4plus.pth": [sp / "realesrgan/weights"],
    }
    placed = 0
    for name, dests in targets.items():
        src = weights / name
        if not src.is_file():
            log.info("place: %s not in %s (skipping)", name, weights)
            continue
        for d in dests:
            d.mkdir(parents=True, exist_ok=True)
            dst = d / name
            if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                continue
            try:
                shutil.copy2(src, dst)
            except OSError as e:
                log.warning("place: %s -> %s failed (%s)", name, dst, e)
                continue
            log.info("place: %s -> %s", name, dst.relative_to(sp))
            placed += 1
    return placed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature", action="append", default=[],
                     help="e.g. lipsync:wav2lip, or just 'lipsync' for every "
                          "lipsync:* row, or 'restore', 'asd'. Repeatable.")
    ap.add_argument("--all", action="store_true", help="shorthand for --feature all")
    ap.add_argument("--include-optional", action="store_true",
                     help="also fetch rows marked optional (large, usually unneeded)")
    ap.add_argument("--list", action="store_true", help="print status, fetch nothing")
    ap.add_argument("--check", action="store_true",
                     help="sha256-verify files already on disk, fetch nothing")
    ap.add_argument("--dry-run", action="store_true", help="print what would happen")
    ap.add_argument("--force", action="store_true",
                     help="re-fetch even if a file already looks right")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    manifest = load_manifest()
    features = ["all"] if args.all else args.feature

    if args.list:
        cmd_list(manifest, features or ["all"])
        return 0
    if args.check:
        return cmd_check(manifest, features or ["all"])
    if not features:
        print("Nothing to do -- pass --feature <name>, --all, --list or --check.\n"
              "Run with --list to see every available feature name.", file=sys.stderr)
        return 2

    ok = True
    restore_matched = False
    for v in manifest["vendor_repos"]:
        if _matches_feature(v["feature"], features):
            ok &= fetch_vendor(v, dry_run=args.dry_run)
    for w in manifest["weights"]:
        if not _matches_feature(w["feature"], features):
            continue
        if w.get("optional") and not args.include_optional:
            log.info("%s: skipping optional (pass --include-optional to fetch it)", w["id"])
            continue
        if w["feature"].startswith("restore"):
            restore_matched = True
        ok &= fetch_weight(w, dry_run=args.dry_run, force=args.force)

    # Weight placement for face restoration: facexlib/gfpgan/realesrgan expect
    # their weights inside their own package dirs, else they phone home at
    # render time. Only runs when a `restore` weight was in scope and the run
    # is live (not --dry-run).
    if restore_matched and not args.dry_run:
        placed = place_face_restore_weights()
        log.info("place: %d face-restore weight(s) copied into site-packages", placed)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
