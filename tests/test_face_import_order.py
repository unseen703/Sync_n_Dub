"""Stage 5 depends on an import that happens in stage 3's neighbourhood.

`pyannote.audio` loads a libjpeg without the 12-bit symbols that cv2's bundled
`libtiff.so.6` needs, so whichever lands first wins for the life of the
process:

    import pyannote.audio
    from insightface.app import FaceAnalysis
    ImportError: .../libtiff.so.6: undefined symbol: jpeg12_write_raw_data

`LD_PRELOAD` does not cover this one -- that handles the libstdc++ pair in
CLAUDE.md section 1, a different clash.

Nothing in a passing run points at the ordering, which is the problem. The
symptom is stage 5 raising "RetinaFace needs either `insightface` ... or the
`retina-face` TF package" on a host where insightface is installed and imports
perfectly on its own. It stayed hidden for weeks because `--lipsync auto` runs
the face-size probe before diarization and warms the import by accident; name
a model explicitly and the probe short-circuits.

These tests are static -- they import neither library, so they cost nothing
and run anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_the_face_stack_is_warmed_before_diarization():
    """The call must sit ABOVE the stage 4 marker, not merely exist."""
    src = (REPO_ROOT / "backend" / "pipeline.py").read_text()

    assert "preimport_face_stack()" in src, \
        "the face-stack pre-import is gone; stage 5 will fail on explicit --lipsync"

    warm = src.index("preimport_face_stack()")
    stage4 = src.index('"=== stage 4: diarization')
    assert warm < stage4, (
        "preimport_face_stack() moved below stage 4 — pyannote now imports "
        "first and stage 5 dies claiming insightface is not installed"
    )


def test_the_warmup_never_raises():
    """A host genuinely without insightface must still reach the real error.

    The adapter's own message is the one that explains the fix, so swallowing
    the import here must not turn into swallowing the diagnosis there.
    """
    import backend.faces as faces

    src = (REPO_ROOT / "backend" / "faces.py").read_text()
    block = src[src.index("def preimport_face_stack"):src.index("def detect_faces")]
    assert "except Exception" in block, "a failed pre-import must not kill the run"
    assert "return False" in block

    assert faces.preimport_face_stack() in (True, False)


def test_the_adapter_still_reports_why_insightface_failed():
    """A bare "insightface unavailable" reads as "not installed".

    It is usually this import-order clash instead, and only the exception
    text tells the two apart — a 2026-10-17 session lost time to `pip install`
    on a package that was already there.
    """
    src = (REPO_ROOT / "models" / "faces" / "detect" / "retinaface_adapter.py").read_text()
    assert "except ImportError as e:" in src, \
        "the insightface failure reason is being discarded again"
    assert "insightface did not load" in src
