"""Compat shim for the documented `python -m backend.edit_agent` entry point.

The implementation moved to `backend.pipeline.edit_agent` in the batch-9
carve-out; this module keeps the CLI invocation working.
"""
from backend.pipeline.edit_agent import *  # noqa: F401,F403
from backend.pipeline.edit_agent import main

if __name__ == "__main__":
    raise SystemExit(main())
