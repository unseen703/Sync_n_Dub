"""Make the `restore` env usable: shim basicsr, place the facexlib weights.

Run once after `pip install basicsr facexlib gfpgan realesrgan` into that
env. Idempotent -- safe to re-run, and re-running is the intended fix after
any upgrade of those four packages puts the import back.

Two things are wrong out of the box, and both fail at IMPORT time, which is
the good case: they fail before a render rather than after one.

1. **`torchvision.transforms.functional_tensor` no longer exists.** basicsr
   (and facexlib) import `rgb_to_grayscale` from it. It moved to
   `torchvision.transforms.functional` in 0.17. The usual advice is to pin
   torchvision below 0.17 -- which cannot work here, because these are
   Blackwell sm_120 cards and torch < 2.7 has no kernels for them. So the
   import is rewritten instead, in this env only.

2. **facexlib downloads its detection and parsing weights on first use**,
   into its own package directory, from GitHub at render time. On a shared
   box mid-run that is a surprise network dependency inside stage 9h. The
   two files are already in `models/weights/restore/`, so they are copied
   where facexlib and gfpgan look.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

#: Renamed from `wav2liphd` on 2026-10-28. The old path is still accepted
#: so this runs on a host where `conda rename` has not happened yet.
_ENVS = Path("/data1/dipak_data/miniconda3/envs")
ENV = _ENVS / "restore" if (_ENVS / "restore").is_dir() else _ENVS / "wav2liphd"
PY = ENV / "bin" / "python"
WEIGHTS = Path(__file__).resolve().parent.parent / "models/weights/restore"

OLD = "torchvision.transforms.functional_tensor"
NEW = "torchvision.transforms.functional"


def site_packages() -> Path:
    out = subprocess.run(
        [str(PY), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, check=True).stdout.strip()
    return Path(out)


def shim_imports(sp: Path) -> int:
    """Rewrite the dead torchvision import wherever it appears."""
    fixed = 0
    for pkg in ("basicsr", "facexlib", "gfpgan", "realesrgan"):
        root = sp / pkg
        if not root.is_dir():
            continue
        for py in root.rglob("*.py"):
            try:
                text = py.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if OLD not in text:
                continue
            py.write_text(text.replace(OLD, NEW))
            print(f"  shimmed {py.relative_to(sp)}")
            fixed += 1
    return fixed


def place_weights(sp: Path) -> int:
    """Put the facexlib/gfpgan weights where the libraries look for them."""
    placed = 0
    targets = {
        "detection_Resnet50_Final.pth": [sp / "facexlib/weights"],
        "parsing_parsenet.pth": [sp / "facexlib/weights"],
        "GFPGANv1.4.pth": [sp / "gfpgan/weights"],
        "RealESRGAN_x4plus.pth": [sp / "realesrgan/weights"],
    }
    for name, dests in targets.items():
        src = WEIGHTS / name
        if not src.is_file():
            print(f"  !! {name} not in {WEIGHTS}")
            continue
        for d in dests:
            d.mkdir(parents=True, exist_ok=True)
            dst = d / name
            if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                continue
            shutil.copy2(src, dst)
            print(f"  placed {name} -> {dst.relative_to(sp)}")
            placed += 1
    return placed


def smoke(sp: Path) -> bool:
    code = (
        "import torch;"
        "from basicsr.archs.rrdbnet_arch import RRDBNet;"
        "from realesrgan import RealESRGANer;"
        "from gfpgan import GFPGANer;"
        "print('imports ok; cuda', torch.cuda.is_available(),"
        "      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    )
    p = subprocess.run([str(PY), "-c", code], capture_output=True, text=True)
    print((p.stdout or p.stderr).strip()[-900:])
    return p.returncode == 0


def main() -> int:
    if not PY.is_file():
        print(f"env missing: {PY}")
        return 2
    sp = site_packages()
    print(f"site-packages: {sp}")
    print("- shimming the dead torchvision import")
    print(f"  {shim_imports(sp)} file(s) changed")
    print("- placing weights")
    print(f"  {place_weights(sp)} file(s) placed")
    print("- smoke test")
    return 0 if smoke(sp) else 1


if __name__ == "__main__":
    raise SystemExit(main())
