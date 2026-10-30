"""Langfuse tracing for `backend/llm.py`, the only LLM caller here.

Wraps the calls `backend/edit_agent.py` and `backend/classify.py` make, so
the raw model reply is recorded alongside the parsed ops that end up in
`<job>/edits/edit_NNN.json`.

Three rules this module keeps:

1. It never breaks a dub. Every public function swallows its own exceptions
   and logs; `generation()` yields a no-op recorder when Langfuse is
   unreachable.
2. Unconfigured is a supported state. No keys means every call is a cheap
   no-op and nothing warns on the hot path.
3. Short-lived processes must flush. The SDK batches on a background thread,
   so `flush()` is registered with `atexit` and the CLI entry points call it
   explicitly.

Auth needs BOTH keys: Langfuse uses HTTP Basic with the public key as
username and the secret as password, so a secret key alone sends nothing.
The public key is not a secret and can be re-read from Langfuse project
settings at any time.

The prompts carry the cue table, i.e. the dialogue being dubbed, so on
Langfuse Cloud that text leaves this host.
`DUBBING_LANGFUSE_CAPTURE_CONTENT=0` keeps timings, tokens, cost and model
parameters while dropping prompt and completion text.
"""
from __future__ import annotations

import atexit
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
HOST_ENV = "LANGFUSE_HOST"
CAPTURE_ENV = "DUBBING_LANGFUSE_CAPTURE_CONTENT"

DEFAULT_HOST = "https://cloud.langfuse.com"

_client = None
_tried = False
_atexit_registered = False


def capture_content() -> bool:
    """False drops prompt/completion TEXT and keeps everything else."""
    return (os.getenv(CAPTURE_ENV) or "1").strip().lower() not in (
        "0", "false", "no", "off")


def configured() -> bool:
    """True when both keys are present. Does not touch the network."""
    return bool((os.getenv(PUBLIC_KEY_ENV) or "").strip()
                and (os.getenv(SECRET_KEY_ENV) or "").strip())


def client():
    """The shared `Langfuse` client, or None. Built once, never raises.

    A secret key with no public key looks like "tracing is on" in `.env` and
    silently sends nothing, so that case says what is missing.
    """
    global _client, _tried, _atexit_registered
    if _tried:
        return _client
    _tried = True

    pub = (os.getenv(PUBLIC_KEY_ENV) or "").strip()
    sec = (os.getenv(SECRET_KEY_ENV) or "").strip()
    if not pub and not sec:
        return None                       # off, deliberately; stay quiet
    if not pub or not sec:
        missing = PUBLIC_KEY_ENV if not pub else SECRET_KEY_ENV
        log.warning(
            "langfuse: %s is set but %s is not, so tracing is OFF. Langfuse "
            "signs in with HTTP Basic as public_key:secret_key and needs "
            "BOTH. The public key is not a secret — Langfuse → Project "
            "Settings → API Keys shows it any time (only the sk-lf-… is "
            "one-time).",
            SECRET_KEY_ENV if not pub else PUBLIC_KEY_ENV, missing)
        return None

    try:
        from langfuse import Langfuse
        _client = Langfuse(public_key=pub, secret_key=sec,
                           host=(os.getenv(HOST_ENV) or DEFAULT_HOST).strip())
    except Exception as exc:                                  # noqa: BLE001
        log.warning("langfuse: client unavailable (%s: %s); tracing is OFF",
                    type(exc).__name__, exc)
        _client = None
        return None

    if not _atexit_registered:
        # The SDK batches on a background thread, so a CLI process that
        # exits with a full queue drops the trace it just made.
        atexit.register(flush)
        _atexit_registered = True
    log.info("langfuse: tracing ON (%s, content=%s)",
             os.getenv(HOST_ENV) or DEFAULT_HOST,
             "on" if capture_content() else "metadata only")
    return _client


def flush() -> None:
    """Push anything queued. Safe to call when tracing is off."""
    if _client is None:
        return
    try:
        _client.flush()
    except Exception as exc:                                  # noqa: BLE001
        log.debug("langfuse: flush failed (%s: %s)", type(exc).__name__, exc)


def check() -> tuple:
    """`(ok, message)` for a `--check-langfuse` style probe. One round trip."""
    if not configured():
        return False, (f"{PUBLIC_KEY_ENV} / {SECRET_KEY_ENV} not both set — "
                       "tracing is off")
    c = client()
    if c is None:
        return False, "client could not be built (see the warning above)"
    try:
        ok = bool(c.auth_check())
    except Exception as exc:                                  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    host = os.getenv(HOST_ENV) or DEFAULT_HOST
    return (ok, f"OK — keys accepted by {host}" if ok else
            f"{host} rejected the key pair")


class _NullRecorder:
    """Stands in for a generation when tracing is off. Same shape, no cost."""

    def succeed(self, output=None, *, usage=None, cost=None, **_kw) -> None:
        pass

    def fail(self, message: str) -> None:
        pass


class _Recorder:
    def __init__(self, obs):
        self._obs = obs
        # `generation()` ends the span in its `finally` as a backstop, and
        # ending twice is not idempotent -- the SDK logs "Calling end() on
        # an ended span." every time.
        self._ended = False

    def succeed(self, output=None, *, usage=None, cost=None, **extra) -> None:
        """Close the generation as a success.

        `usage` is OpenRouter's `usage` block and `cost` its dollar figure,
        both present only because `backend/llm.py` asks for
        `usage: {"include": true}`. They are mapped onto Langfuse's
        `usage_details` / `cost_details`; without that mapping Langfuse
        re-derives cost from its model table, which lacks this model's
        price.
        """
        try:
            fields: Dict[str, Any] = dict(extra)
            if capture_content() and output is not None:
                fields["output"] = output
            if usage:
                details = {k: int(v) for k, v in (
                    ("input", usage.get("prompt_tokens")),
                    ("output", usage.get("completion_tokens")),
                    ("total", usage.get("total_tokens")),
                ) if isinstance(v, (int, float))}
                if details:
                    fields["usage_details"] = details
            if isinstance(cost, (int, float)):
                fields["cost_details"] = {"total": float(cost)}
            if fields:
                self._obs.update(**fields)
        except Exception as exc:                              # noqa: BLE001
            log.debug("langfuse: update failed (%s: %s)",
                      type(exc).__name__, exc)
        finally:
            self._end()

    def fail(self, message: str) -> None:
        try:
            self._obs.update(level="ERROR", status_message=str(message)[:2000])
        except Exception:                                     # noqa: BLE001
            pass
        finally:
            self._end()

    def _end(self) -> None:
        if self._ended:
            return
        self._ended = True
        try:
            self._obs.end()
        except Exception:                                     # noqa: BLE001
            pass


@contextmanager
def generation(name: str, *, model: str, input: Any = None,
               metadata: Optional[Dict[str, Any]] = None,
               model_parameters: Optional[Dict[str, Any]] = None):
    """One LLM call. Yields a recorder; always yields something.

    The caller closes it with `.succeed(...)` or `.fail(...)`. If the caller
    raises instead, the generation is still ended and marked ERROR.
    """
    c = client()
    if c is None:
        yield _NullRecorder()
        return

    obs = None
    try:
        obs = c.start_observation(
            name=name, as_type="generation",
            input=(input if capture_content() else None),
            metadata=metadata, model=model, model_parameters=model_parameters,
        )
    except Exception as exc:                                  # noqa: BLE001
        log.debug("langfuse: could not start a generation (%s: %s)",
                  type(exc).__name__, exc)
        yield _NullRecorder()
        return

    rec = _Recorder(obs)
    try:
        yield rec
    except BaseException as exc:
        rec.fail(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        rec._end()        # no-op when succeed()/fail() already ended it


@contextmanager
def trace(name: str, *, metadata: Optional[Dict[str, Any]] = None,
          input: Any = None):
    """Group the calls one user request makes into a single trace.

    An edit agent request is one parse call plus up to N rewrite calls.
    Nesting is by OTel context, so generations opened inside this block
    attach themselves -- no ids to thread through.
    """
    c = client()
    span_cm = None
    if c is not None:
        try:
            span_cm = c.start_as_current_observation(
                name=name, as_type="span", metadata=metadata,
                input=(input if capture_content() else None))
        except Exception as exc:                              # noqa: BLE001
            log.debug("langfuse: trace %r could not start (%s: %s)", name,
                      type(exc).__name__, exc)
            span_cm = None

    if span_cm is None:
        yield None
        return

    # Not wrapped in try/except around the yield: catching there and
    # yielding again makes `@contextmanager` raise "generator didn't stop
    # after throw()", which masks every exception raised inside the block.
    # Failure to CREATE the span is handled above.
    with span_cm as span:
        yield span
