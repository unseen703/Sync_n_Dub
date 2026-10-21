"""Tracing must never be the thing that breaks a dub.

`backend/tracing.py` sits inside `backend/llm.py`, which sits inside the edit
agent and the register classifier — both on the path of a fifteen-stage GPU
run. An observability layer that can raise there is worse than no
observability, so most of these tests are about what happens when Langfuse
is absent, half-configured, or actively broken.

The other half pins the two things that are silently wrong rather than
loudly wrong: the usage/cost mapping (get it wrong and the dashboard shows
zero spend forever, which looks like "we spend nothing") and the content
switch (get it wrong and the dub's dialogue goes to a vendor that was never
meant to see it).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.tracing as T                                    # noqa: E402


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    """The client is a module-level singleton built once per process."""
    monkeypatch.setattr(T, "_client", None)
    monkeypatch.setattr(T, "_tried", False)
    for var in (T.PUBLIC_KEY_ENV, T.SECRET_KEY_ENV, T.HOST_ENV, T.CAPTURE_ENV):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Off is a supported state
# ---------------------------------------------------------------------------

def test_unconfigured_is_a_silent_no_op(caplog):
    """No keys is the DEFAULT. It must not warn on the hot path — a warning
    per LLM call would be noise on every run that never opted in."""
    with caplog.at_level(logging.DEBUG, logger="backend.tracing"):
        assert T.configured() is False
        assert T.client() is None
        with T.generation("x", model="m", input=["hi"]) as rec:
            rec.succeed("out", usage={"prompt_tokens": 1}, cost=0.1)
        with T.trace("req") as span:
            assert span is None
        T.flush()

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_half_configured_warns_and_stays_off(monkeypatch, caplog):
    """THE reported case: only the private key in hand. Langfuse is Basic
    auth as public_key:secret_key, so a lone secret sends nothing — and
    `.env` looks switched on, which is why silence here is unacceptable.
    """
    monkeypatch.setenv(T.SECRET_KEY_ENV, "sk-lf-x")

    with caplog.at_level(logging.WARNING, logger="backend.tracing"):
        assert T.configured() is False
        assert T.client() is None

    msg = " ".join(r.getMessage() for r in caplog.records)
    assert T.PUBLIC_KEY_ENV in msg
    assert "not a secret" in msg, "must say the public key is retrievable"


def test_a_broken_client_does_not_raise(monkeypatch, caplog):
    """An unreachable host, a bad SDK version, a DNS failure — all of it has
    to degrade to "no tracing", never to an exception in a dub."""
    monkeypatch.setenv(T.PUBLIC_KEY_ENV, "pk-lf-x")
    monkeypatch.setenv(T.SECRET_KEY_ENV, "sk-lf-x")

    import langfuse
    monkeypatch.setattr(langfuse, "Langfuse",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    with caplog.at_level(logging.WARNING, logger="backend.tracing"):
        assert T.client() is None
        with T.generation("x", model="m") as rec:
            rec.succeed("out")

    assert any("tracing is OFF" in r.getMessage() for r in caplog.records)


def test_a_generation_that_fails_mid_block_still_ends(monkeypatch):
    """A caller raising inside the `with` must not leave a span open."""
    ended, failed = [], []

    class _Obs:
        def update(self, **kw):
            failed.append(kw)

        def end(self):
            ended.append(True)

    class _C:
        def start_observation(self, **kw):
            return _Obs()

    monkeypatch.setattr(T, "_client", _C())
    monkeypatch.setattr(T, "_tried", True)

    with pytest.raises(ValueError):
        with T.generation("x", model="m"):
            raise ValueError("caller blew up")

    assert ended, "the span was never ended"
    assert failed and failed[0].get("level") == "ERROR"


# ---------------------------------------------------------------------------
# Usage and cost
# ---------------------------------------------------------------------------

class _Obs:
    def __init__(self):
        self.updates = []
        self.ended = 0

    def update(self, **kw):
        self.updates.append(kw)

    def end(self):
        self.ended += 1


def _recorder(monkeypatch):
    obs = _Obs()
    monkeypatch.setattr(T, "_client", type("C", (), {
        "start_observation": lambda self, **kw: obs})())
    monkeypatch.setattr(T, "_tried", True)
    return obs


def test_openrouter_usage_maps_onto_langfuse_fields(monkeypatch):
    """Langfuse wants `usage_details` / `cost_details`; OpenRouter answers
    with `prompt_tokens` / `completion_tokens` / `cost`. Get this mapping
    wrong and the dashboard reports zero spend forever, which reads as
    "this costs nothing" rather than as a bug.
    """
    obs = _recorder(monkeypatch)
    with T.generation("g", model="m") as rec:
        rec.succeed("hello", usage={"prompt_tokens": 11, "completion_tokens": 5,
                                    "total_tokens": 16}, cost=1.24e-06)

    u = obs.updates[0]
    assert u["usage_details"] == {"input": 11, "output": 5, "total": 16}
    assert u["cost_details"] == {"total": pytest.approx(1.24e-06)}
    assert u["output"] == "hello"


def test_a_missing_usage_block_is_not_a_crash(monkeypatch):
    """Not every provider returns usage, and a 200 with none must still
    close the span cleanly."""
    obs = _recorder(monkeypatch)
    with T.generation("g", model="m") as rec:
        rec.succeed("hello", usage={}, cost=None)

    assert "usage_details" not in obs.updates[0]
    assert "cost_details" not in obs.updates[0]
    assert obs.ended >= 1


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def test_content_capture_can_be_switched_off(monkeypatch):
    """The prompts carry the cue table — the dub's actual dialogue. On
    Langfuse Cloud that text leaves this host, so there has to be a way to
    keep the numbers and drop the words.
    """
    monkeypatch.setenv(T.CAPTURE_ENV, "0")
    assert T.capture_content() is False

    obs = _recorder(monkeypatch)
    with T.generation("g", model="m", input=[{"role": "user", "content": "secret"}]) as rec:
        rec.succeed("also secret", usage={"prompt_tokens": 3})

    assert "output" not in obs.updates[0], "completion text leaked"
    assert obs.updates[0]["usage_details"] == {"input": 3}, "numbers must survive"


def test_content_capture_is_on_by_default(monkeypatch):
    assert T.capture_content() is True


# ---------------------------------------------------------------------------
# The llm.py side
# ---------------------------------------------------------------------------

def test_chat_asks_openrouter_for_usage():
    """`usage: {"include": true}` is how a PLAIN api key gets token counts
    and cost back — the account-level Analytics API needs a management key
    and 403s otherwise. Without this the cost column is empty.
    """
    src = (REPO_ROOT / "backend" / "llm.py").read_text()
    assert '"usage": {"include": True}' in src


def test_every_llm_call_site_names_its_trace():
    """An unnamed generation shows up as `openrouter.chat` for all of them,
    which makes the per-operation cost breakdown useless."""
    agent = (REPO_ROOT / "backend" / "edit_agent.py").read_text()
    classify = (REPO_ROOT / "backend" / "classify.py").read_text()

    assert 'trace_name="edit_agent.parse"' in agent
    assert 'trace_name="edit_agent.rewrite"' in agent
    assert 'trace_name="classify.register"' in classify


def test_the_agent_groups_a_request_into_one_trace():
    """One request is a parse plus N rewrites. As N+1 loose traces it does
    not say which rewrite served which request."""
    src = (REPO_ROOT / "backend" / "edit_agent.py").read_text()
    assert '_tracing.trace("edit_agent.request"' in src


def test_a_span_is_ended_exactly_once(monkeypatch):
    """`generation()` ends the span in `finally` as a backstop, so a normal
    `succeed()` path would end it twice — and the SDK logs
    "Calling end() on an ended span." every time, three lines per agent
    request. Not fatal, but it is noise that trains you to ignore the log.
    """
    obs = _recorder(monkeypatch)
    with T.generation("g", model="m") as rec:
        rec.succeed("out")

    assert obs.ended == 1, f"span ended {obs.ended} times"


def test_a_double_close_by_the_caller_is_also_safe(monkeypatch):
    obs = _recorder(monkeypatch)
    with T.generation("g", model="m") as rec:
        rec.succeed("out")
        rec.fail("changed my mind")

    assert obs.ended == 1


# ---------------------------------------------------------------------------
# JSON mode
# ---------------------------------------------------------------------------
#
# Not tracing, but it lives in the same module and it is the subtlest failure
# found so far: `response_format={"type":"json_object"}` on gemma-3-12b-it
# returned VALID JSON of the wrong shape — `{"ops":[{}]}` and keys prefixed
# with a full stop (`".op"`). Valid JSON means `extract_json` parses it, so
# nothing downstream can tell. The flag meant to guarantee parseable output
# was the thing corrupting it.

def test_json_mode_is_off_under_auto(monkeypatch):
    from backend.llm import _json_mode_enabled, JSON_MODE_ENV

    monkeypatch.delenv(JSON_MODE_ENV, raising=False)
    assert _json_mode_enabled(True) is False, "auto must resolve to OFF"
    monkeypatch.setenv(JSON_MODE_ENV, "auto")
    assert _json_mode_enabled(True) is False


def test_json_mode_can_be_forced_back_on(monkeypatch):
    """The escape hatch for a model that genuinely needs the constraint."""
    from backend.llm import _json_mode_enabled, JSON_MODE_ENV

    monkeypatch.setenv(JSON_MODE_ENV, "on")
    assert _json_mode_enabled(True) is True
    monkeypatch.setenv(JSON_MODE_ENV, "off")
    assert _json_mode_enabled(True) is False


def test_a_caller_that_never_asked_for_json_mode_never_gets_it(monkeypatch):
    from backend.llm import _json_mode_enabled, JSON_MODE_ENV

    monkeypatch.setenv(JSON_MODE_ENV, "on")
    assert _json_mode_enabled(False) is False


def test_an_exception_inside_a_trace_propagates_unchanged(monkeypatch):
    """`trace()` used to catch around its own `yield` and then yield again,
    which `@contextmanager` turns into `RuntimeError: generator didn't stop
    after throw()`. Every EditError raised inside `make_plan` surfaced as
    that RuntimeError instead of itself — tracing swallowing the error it
    exists to record.
    """
    class _Span:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False          # do not suppress

    monkeypatch.setattr(T, "_client", type("C", (), {
        "start_as_current_observation": lambda self, **kw: _Span()})())
    monkeypatch.setattr(T, "_tried", True)

    with pytest.raises(ValueError, match="the real error"):
        with T.trace("req"):
            raise ValueError("the real error")


def test_a_trace_that_cannot_start_still_runs_the_block(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("langfuse down")

    monkeypatch.setattr(T, "_client", type("C", (), {
        "start_as_current_observation": lambda self, **kw: _boom()})())
    monkeypatch.setattr(T, "_tried", True)

    ran = []
    with T.trace("req") as span:
        ran.append(span)
    assert ran == [None], "the block must still run with no span"
