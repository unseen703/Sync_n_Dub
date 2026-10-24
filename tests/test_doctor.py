"""scripts/doctor.py — the logic that doesn't need a live GPU/conda host.

The per-env import checks, GPU query and sha256 verification are exercised
by actually running the script (see the deploy notes) rather than here —
they need the real sidecar envs and weights this suite's CI checkout won't
have. What's tested here is the host-independent plumbing: secret
presence/pairing, env-file mode, and the row shape every check returns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import doctor as D  # noqa: E402


def test_row_shape():
    row = D._row(D.PASS, "thing", "detail")
    assert row == (D.PASS, "thing", "detail")


# ---------------------------------------------------------------------------
# secrets — presence only, and the paired-keys rule
# ---------------------------------------------------------------------------

def test_secret_absent_is_warn(monkeypatch):
    monkeypatch.delenv("FAKE_KEY", raising=False)
    status, name, detail = D.check_secret("FAKE_KEY")
    assert status == D.WARN
    assert detail == "not set"


def test_secret_present_is_pass(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-something")
    status, name, detail = D.check_secret("FAKE_KEY")
    assert status == D.PASS
    assert detail == "set"
    assert "sk-something" not in detail, "a check must never echo the secret's value"


def test_secret_never_leaks_its_own_value_even_on_failure_paths(monkeypatch):
    """Every branch of check_secret, not just the happy one."""
    monkeypatch.setenv("FAKE_PUB", "pk-abc123")
    monkeypatch.delenv("FAKE_SEC", raising=False)
    for var in ("FAKE_PUB", "FAKE_SEC"):
        _, _, detail = D.check_secret(var, required_group=["FAKE_PUB", "FAKE_SEC"])
        assert "abc123" not in detail


def test_paired_secret_both_set_is_clean_pass(monkeypatch):
    monkeypatch.setenv("FAKE_PUB", "x")
    monkeypatch.setenv("FAKE_SEC", "y")
    status, _, detail = D.check_secret("FAKE_PUB", required_group=["FAKE_PUB", "FAKE_SEC"])
    assert status == D.PASS
    assert detail == "set"


def test_paired_secret_one_missing_is_warn_naming_the_other(monkeypatch):
    """Half-configured Langfuse: looks switched on, traces and reads nothing."""
    monkeypatch.setenv("FAKE_PUB", "x")
    monkeypatch.delenv("FAKE_SEC", raising=False)
    status, _, detail = D.check_secret("FAKE_PUB", required_group=["FAKE_PUB", "FAKE_SEC"])
    assert status == D.WARN
    assert "FAKE_SEC" in detail


def test_paired_secret_neither_set_is_plain_warn_not_pairing_warn(monkeypatch):
    """Neither set is the normal 'tracing is off' case, not a half-config warning."""
    monkeypatch.delenv("FAKE_PUB", raising=False)
    monkeypatch.delenv("FAKE_SEC", raising=False)
    status, _, detail = D.check_secret("FAKE_PUB", required_group=["FAKE_PUB", "FAKE_SEC"])
    assert status == D.WARN
    assert detail == "not set"


# ---------------------------------------------------------------------------
# .env file mode
# ---------------------------------------------------------------------------

def test_env_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "REPO_ROOT", tmp_path)
    status, _, detail = D.check_env_file()
    assert status == D.WARN
    assert "missing" in detail


def test_env_file_wrong_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "REPO_ROOT", tmp_path)
    f = tmp_path / ".env"
    f.write_text("X=1\n")
    f.chmod(0o644)
    status, _, detail = D.check_env_file()
    assert status == D.WARN
    assert "600" in detail


def test_env_file_correct_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "REPO_ROOT", tmp_path)
    f = tmp_path / ".env"
    f.write_text("X=1\n")
    f.chmod(0o600)
    status, _, detail = D.check_env_file()
    assert status == D.PASS


# ---------------------------------------------------------------------------
# exit code
# ---------------------------------------------------------------------------

def test_any_fail_means_nonzero_overall(monkeypatch):
    monkeypatch.setattr(D, "run_all", lambda quick: [
        (D.PASS, "a", ""), (D.FAIL, "b", "broken"), (D.WARN, "c", "")])
    monkeypatch.setattr(sys, "argv", ["doctor.py", "--quick"])
    assert D.main() == 1


def test_only_warn_and_pass_means_zero(monkeypatch):
    monkeypatch.setattr(D, "run_all", lambda quick: [
        (D.PASS, "a", ""), (D.WARN, "c", "")])
    monkeypatch.setattr(sys, "argv", ["doctor.py", "--quick"])
    assert D.main() == 0
