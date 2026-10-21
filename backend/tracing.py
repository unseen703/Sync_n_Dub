"""Langfuse tracing for the one module here that talks to an LLM.

`backend/llm.py` is the only LLM caller (on behalf of `backend/edit_agent.py`
and `backend/classify.py`), so this is the only place that needs wiring. Added
2026-10-19 to close a specific gap: the parsed ops survive in
`<job>/edits/edit_NNN.json` and `extra.edits.history`, but the RAW model reply
was never stored anywhere, so a model that misbehaves — a hallucinated voice
id, a degenerate scope — left no record of what it actually said.

THREE RULES, in order of importance.

1. **It must never break a dub.** Observability that can take down a
   fifteen-stage GPU run is worse than no observability. Every public
   function here swallows its own exceptions and logs; `generation()` yields
   a no-op recorder rather than raising when Langfuse is unreachable.

2. **Unconfigured is a supported state**, exactly as an empty
   `OPENROUTER_API_KEY` is (CLAUDE.md §1). No keys → every call is a cheap
   no-op and nothing warns on the hot path.

3. **Short-lived processes must flush.** The SDK batches over a background
   thread, so `python -m backend.edit_agent` can exit with a full queue and
   lose the trace it just made. `flush()` is registered with `atexit`, and
   the CLI entry points call it explicitly.

AUTH NEEDS BOTH KEYS. Langfuse authenticates with HTTP Basic where the
PUBLIC key is the username and the SECRET key is the password — the SDK and
the REST API alike. A secret key on its own cannot send or read anything.
The public key is **not** a secret: it is shown in Langfuse → Project
Settings → API Keys any time you look. Only the secret key is one-time. So
"I only have the private key" is recoverable — open project settings and
copy the `pk-lf-…` next to it.

CONTENT. The prompts carry the cue table, i.e. the actual translated
dialogue of whatever is being dubbed. On Langfuse Cloud that text leaves
this host. `DUBBING_LANGFUSE_CAPTURE_CONTENT=0` keeps the timings, token
counts, costs and model parameters while dropping prompt and completion
text, which is the setting to use if the material is not yours to send.
Self-hosting (`LANGFUSE_HOST=http://localhost:3000`) avoids the question.
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

    A half-configured install is the case worth being loud about: a secret
    key with no public key looks like "tracing is on" in `.env` and silently
    sends nothing, so it says what is missing and where to find it.
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
        # The SDK batches on a background thread. A CLI process that exits
        # with a full queue drops the trace it just made, which looks like
        # "tracing does not work" rather than "the process ended early".
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
        # `generation()` ends the span in its `finally` as a backstop for a
        # caller that returns without calling either method. Ending twice is
        # NOT idempotent — the SDK logs "Calling end() on an ended span." for
        # every call — so the guard lives here rather than in the comment
        # that used to claim it was harmless.
        self._ended = False

    def succeed(self, output=None, *, usage=None, cost=None, **extra) -> None:
        """Close the generation as a success.

        `usage` is OpenRouter's `usage` block and `cost` its dollar figure;
        both arrive only because `backend/llm.py` asks for
        `usage: {"include": true}`. They are mapped onto Langfuse's own
        `usage_details` / `cost_details` so the dashboard's token and spend
        columns populate — without that mapping Langfuse would re-derive
        cost from its model table, which does not know this model's price.
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
    raises instead, the generation is still ended on the way out and marked
    ERROR, so a crashed run does not leave a span open forever.
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

    An edit agent request is one parse call plus up to N rewrite calls, and
    read as N+1 unrelated traces it tells you nothing about which rewrite
    belonged to which request. Nesting is by OTel context, so generations
    opened inside this block attach themselves — no ids to thread through.
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

    # NOT wrapped in try/except. An earlier version caught around the yield
    # and then yielded AGAIN, which `@contextmanager` answers with
    # `RuntimeError: generator didn't stop after throw()` — so every
    # exception raised inside a traced block, every EditError the agent
    # relies on, surfaced as that RuntimeError instead of itself. Tracing
    # that swallows the error it was meant to record is worse than none.
    # Failure to CREATE the span is handled above; once it exists the SDK's
    # own context manager ends it and lets the exception through.
    with span_cm as span:
        yield span
