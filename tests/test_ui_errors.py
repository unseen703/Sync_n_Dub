"""Does a failure tell the user something they can act on?

The UI is user-facing, so it does not show the runner's log. That raises the
bar on the failure message: "Failed" with nothing else is worse than a
traceback, because the user cannot tell whether they asked for something
impossible, whether a credential is missing, or whether the machine is busy.

Every rule here corresponds to a failure this pipeline actually produces —
most of them documented in CLAUDE.md, and the HF_TOKEN one hit for real while
building this UI.
"""
from __future__ import annotations

import pytest

from ui.errors import explain


def test_no_error_means_no_failure_block():
    assert explain(None) is None
    assert explain("") is None


def test_failed_with_no_reason_still_says_something():
    """A job marked failed with an empty error must not render blank."""
    got = explain(None, state="failed")
    assert got and got["title"]
    assert got["recognised"] is False


@pytest.mark.parametrize("raw,expect_in_title", [
    ("RuntimeError: pyannote community-1 requires HF_TOKEN in the environment.",
     "Hugging Face"),
    ("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
     "GPU memory"),
    ("HTTPError: 429 Client Error: Too Many Requests", "rate-limit"),
    ("bodhan returned internal_error", "outage"),
    ("ERROR: [youtube] x: Private video. Sign in if you have been granted access",
     "could not be downloaded"),
    ("OSError: [Errno 28] No space left on device", "disk is full"),
])
def test_known_failures_are_explained(raw, expect_in_title):
    got = explain(raw, state="failed")
    assert got["recognised"] is True
    assert expect_in_title.lower() in got["title"].lower()


@pytest.mark.parametrize("raw", [
    "RuntimeError: pyannote community-1 requires HF_TOKEN in the environment.",
    "torch.cuda.OutOfMemoryError: CUDA out of memory.",
    "HTTPError: 429 Too Many Requests",
])
def test_actionable_failures_say_what_to_do(raw):
    """Naming the problem without naming the fix just relocates the
    helplessness."""
    assert explain(raw, state="failed")["action"]


def test_an_unknown_failure_is_marked_as_such():
    """Pretending to recognise something is worse than admitting we don't —
    the UI uses this flag to decide whether to offer the raw line."""
    got = explain("ValueError: something nobody has seen before", state="failed")
    assert got["recognised"] is False
    assert "something nobody has seen" in got["detail"]


def test_the_exception_line_is_preferred_over_the_traceback_header():
    """Python puts the exception LAST. Taking the first line would show
    every user `Traceback (most recent call last):`, forever."""
    raw = ('Traceback (most recent call last):\n'
           '  File "/x/y.py", line 3, in <module>\n'
           '    boom()\n'
           'ZeroDivisionError: division by zero')
    assert "ZeroDivisionError" in explain(raw, state="failed")["detail"]


def test_raw_is_one_line_not_a_log():
    """`raw` exists for a one-line "technical details" disclosure. If it
    carried the whole traceback the UI would be showing a log again, which
    is the thing this module exists to avoid."""
    raw = "\n".join(f"line {i}" for i in range(500)) + "\nRuntimeError: boom"
    got = explain(raw, state="failed")
    assert "\n" not in got["raw"]
    assert len(got["raw"]) <= 300


@pytest.mark.parametrize("raw", [
    # The message `ui/runner.py`'s SIGTERM handler ACTUALLY raises. This
    # test used to pass only "KeyboardInterrupt" — a string the runner never
    # writes — so it went green while every real Stop click on disk was
    # being shown as "The job failed". Pin the real text first.
    "Stopped on signal 15.",
    "Stopped on signal 2.",
    "KeyboardInterrupt",
    "Traceback...\nui.runner.Cancelled: Stopped on signal 15.",
])
def test_cancellation_is_not_dressed_up_as_an_error(raw):
    got = explain(raw, state="failed")
    assert "stopped" in got["title"].lower(), got
    assert got["recognised"] is True


@pytest.mark.parametrize("raw", [
    "RuntimeError: RetinaFace needs either `insightface` (preferred: "
    "`pip install insightface onnxruntime-gpu`) or the `retina-face` package.",
    "ModuleNotFoundError: No module named 'retinaface'",
    "ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version "
    "`CXXABI_1.3.15' not found (required by .../cv2/../../libtbb.so.12)",
])
def test_face_detection_load_failure_is_recognised(raw):
    """All three spellings of one fault reach the same remedy.

    The adapter's own message says `pip install insightface`, and on this
    host insightface is already installed and loads on CUDA and CPU — so
    following that text is a dead end. The cause is the libstdc++ import
    order (CLAUDE.md section 1), and the remedy names the launcher that
    exports the preload.
    """
    got = explain(raw, state="failed")
    assert got["recognised"] is True, got
    assert "face detection" in got["title"].lower()
    assert "LD_PRELOAD" in (got["action"] or "")


def test_the_retinaface_rule_matches_the_adapter_s_real_message():
    """Pin the rule to the adapter's literal, not to a paraphrase of it.

    The cancellation rule drifted from the runner exactly this way: the
    pattern named a word the code never wrote, and the test fed it a string
    the code never produced, so both stayed green while real jobs showed
    an unrecognised failure.
    """
    import inspect

    from models.faces.detect import retinaface_adapter as RA

    src = inspect.getsource(RA)
    assert "RetinaFace needs either" in src, "the adapter's message changed"
    assert explain("RuntimeError: RetinaFace needs either `insightface`",
                   state="failed")["recognised"]


def test_a_cancelled_state_outranks_whatever_the_error_text_says():
    """The operator's own recorded intent beats a string match.

    A job the user stopped is not a job that failed, even when the
    exception that carried out the stop looks like any other error.
    """
    got = explain("RuntimeError: something unrelated", state="cancelled")
    assert got["title"] == "The job was stopped"
    assert got["recognised"] is True


def test_runner_cancel_message_matches_the_rule_that_catches_it():
    """Guards the pair, not either half.

    The rule and the runner drifted apart once already: the pattern listed
    `SIGTERM`, the handler wrote `Stopped on signal 15`, and nothing failed.
    Grep the handler's literal out of the source so renaming it breaks here.
    """
    import inspect

    from ui import runner as R

    src = inspect.getsource(R._install_sigterm)
    assert "Stopped on signal" in src, "the handler's message changed"
    assert explain("Stopped on signal 15.", state="failed")["recognised"]


def test_progress_payload_carries_the_explanation(tmp_path):
    """The UI reads `progress.failure`, so the wiring matters as much as the
    rules."""
    import json

    from ui.progress import read_progress, write_atomic
    write_atomic(tmp_path / "progress.json", {
        "version": 1, "job_id": "j", "state": "failed", "pid": None,
        "error": "RuntimeError: pyannote community-1 requires HF_TOKEN",
        "stages": [], "n_done": 0, "n_total": 1})
    doc = read_progress(tmp_path)
    assert doc["failure"]["recognised"] is True
    assert "Hugging Face" in doc["failure"]["title"]


def test_a_healthy_job_has_no_failure_block(tmp_path):
    from ui.progress import read_progress, write_atomic
    write_atomic(tmp_path / "progress.json", {
        "version": 1, "job_id": "j", "state": "done", "pid": None,
        "error": None, "stages": [], "n_done": 1, "n_total": 1})
    assert read_progress(tmp_path)["failure"] is None
