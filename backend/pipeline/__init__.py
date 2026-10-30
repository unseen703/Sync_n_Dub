"""Pipeline, jobs, versioning and editing.

Re-exports the orchestration entry points so callers can keep writing
`from backend.pipeline import run_pipeline` after the carve-out.
"""
from .pipeline import (
    DEFAULT_DATA_DIR,
    REPO_ROOT,
    _speaker_face_tracks,
    run_pipeline,
)

__all__ = [
    "DEFAULT_DATA_DIR",
    "REPO_ROOT",
    "_speaker_face_tracks",
    "run_pipeline",
]
