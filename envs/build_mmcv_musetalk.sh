#!/usr/bin/env bash
# Build mmcv 2.1.0 for the musetalk env, against torch 2.8.0+cu128 / sm_120.
#
# Why this exists: no prebuilt mmcv wheel exists for this torch/CUDA/GPU-arch
# combination (verified: mmcv's own wheel index tops out well short of
# cu128+sm_120), so the live musetalk env on this host has an mmcv compiled
# from source. The build directory it was compiled in was under /tmp and no
# longer exists, so this script is RECONSTRUCTED from mmcv's documented build
# process plus what is verifiably installed here (envs/musetalk.pip.txt,
# the mmcv/mmdet/mmpose/mmengine/chumpy versions, and this env's CUDA
# toolchain in envs/musetalk.yml) -- not a recovered transcript. Read it
# before running it.
#
# Run AFTER: `conda env create -f envs/musetalk.yml` and
# `pip install -r envs/musetalk.pip.txt --no-deps --extra-index-url ...`
# (see envs/musetalk.yml for the exact command).
#
#   envs/build_mmcv_musetalk.sh [path to musetalk python]
#
set -euo pipefail

PY="${1:-}"
if [ -z "$PY" ]; then
  # Same sibling-env convention backend/models resolve_env_python uses.
  BASE="$(conda info --base 2>/dev/null || true)"
  PY="${BASE:+$BASE/envs/musetalk/bin/python}"
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "usage: $0 <path to the musetalk env's python>" >&2
  exit 1
fi

"$PY" -c "import torch" || { echo "torch not installed in $PY yet -- run the pip install step from envs/musetalk.yml first" >&2; exit 1; }

say() { printf '\033[36m==>\033[0m %s\n' "$*"; }

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

say "mmcv 2.1.0 from source into $BUILD_DIR"
git clone --branch v2.1.0 --depth 1 https://github.com/open-mmlab/mmcv.git "$BUILD_DIR/mmcv"
cd "$BUILD_DIR/mmcv"

# sm_120 (Blackwell, compute capability 12.0) -- this is the GPU arch this
# whole repo targets (CLAUDE.md). Add other arches here (e.g. 8.6 8.9) if
# the same wheel needs to run on different cards too.
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$PY")")}"   # <env prefix>, where cuda-nvcc etc. were installed

say "building the wheel (this is the ~20+ minute step)"
"$PY" -m pip install --no-build-isolation wheel
"$PY" -m pip wheel . --no-build-isolation --no-deps -w "$BUILD_DIR/dist"

WHEEL="$(ls "$BUILD_DIR"/dist/mmcv-2.1.0-*.whl)"
say "installing $WHEEL"
"$PY" -m pip install --no-deps "$WHEEL"

# mmdet 3.1.0 / mmpose 1.1.0 both declare an upper mmcv bound below 2.1.0
# (mmdet's own METADATA: `mmcv<2.1.0,>=2.0.0rc4`) and ENFORCE it again at
# import time in their own __init__.py via a hardcoded
# `mmcv_maximum_version`. mmcv 2.1.0 is the only version with a working
# build for this torch/CUDA pair, so the ceiling is raised to admit it --
# this is a hand edit to installed files, recorded here rather than left as
# an undocumented site-packages diff.
say "raising mmdet/mmpose's mmcv_maximum_version to admit 2.1.0"
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
for f in "$SITE/mmdet/__init__.py" "$SITE/mmpose/__init__.py"; do
  if [ -f "$f" ]; then
    sed -i "s/mmcv_maximum_version = '2\.[0-9]*\.[0-9]*'/mmcv_maximum_version = '2.3.0'/" "$f"
    grep -n "mmcv_maximum_version" "$f"
  else
    echo "warning: $f not found -- install mmdet/mmpose from musetalk.pip.txt first" >&2
  fi
done

# chumpy: PyPI's 0.71 release has known numpy-compatibility issues (deprecated
# aliases removed in modern numpy). This env's chumpy came from a specific
# post-release git commit instead -- mmpose's own dependency, not loaded by
# musetalk's code directly.
say "installing chumpy from the pinned commit"
"$PY" -m pip install --no-deps \
  "git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da"

say "smoke test"
"$PY" -c "
import mmcv, mmengine, mmdet, mmpose, chumpy
print('mmcv', mmcv.__version__)
print('mmengine', mmengine.__version__)
print('mmdet', mmdet.__version__)
print('mmpose', mmpose.__version__)
print('chumpy', chumpy.__version__)
print('all import clean')
"
