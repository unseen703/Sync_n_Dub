"""OpenRouter chat client — the one place in this repo that talks to an LLM.

Shape copied from `models/mt/bodhan_adapter.py` on purpose: both endpoints are
OpenAI-chat-compatible, both are rate limited, and both are read from `.env`
through `os.getenv` after `backend.cli` (or `backend.edit_agent`) has called
`load_dotenv()` once. Nothing here ever logs a key.

Env
---
  OPENROUTER_API_KEY   required. Without it `available()` returns False and
                       every caller in this repo falls back to a deterministic
                       path rather than crashing — an NLQ edit still parses,
                       it just loses the LLM's reading of the sentence.
  OPENROUTER_URL       default https://openrouter.ai/api/v1
  OPENROUTER_MODEL     default google/gemma-3-4b-it

On the default model
--------------------
The request was "gemma-3b". OpenRouter has no such slug; the Gemma 3
instruction-tuned family there is `google/gemma-3-{1b,4b,12b,27b}-it` (plus
`:free` variants). `google/gemma-3-4b-it` is the nearest thing to what was
asked for, so that is the default — override with OPENROUTER_MODEL or
`--llm-model` without touching code.

A 4B model is a weak structured-output generator. Everything here is built
for that: the JSON schema goes in the prompt AND in `response_format`, the
reply is run through a fence-stripping/brace-balancing extractor, a malformed
reply is retried once with the parser error fed back, and the caller is
expected to have a rule-based fallback (`backend.edit_agent._rule_parse`).
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

DEFAULT_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemma-3-4b-it"

#: Sent so usage shows up under a recognisable name in the OpenRouter
#: dashboard. Both headers are optional for the API and carry no secrets.
_REFERER = "https://github.com/unseen703/Sync_n_Dub"
_TITLE = "Dubbing Studio"

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _env_num(name: str, default):
    """Read a numeric env var, tolerating the blank that a copied template
    leaves behind. `int(os.getenv("X", 4))` raises on `X=`, which turns a
    harmlessly empty line in .env into a crash at client construction."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %r", name, raw, default)
        return default


class LLMUnavailable(RuntimeError):
    """No API key, or the endpoint refused every attempt."""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of a chat reply.

    Small models wrap JSON in prose, in ``` fences, or in both, and they like
    to append a closing sentence. Plain `json.loads` on the whole reply fails
    on all three, so: fences first, then a brace/bracket scan that respects
    string literals and escapes.

    Raises ValueError when there is no parseable JSON value at all.
    """
    if not text or not text.strip():
        raise ValueError("empty reply")

    candidates: List[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1))
    candidates.append(text)

    for blob in candidates:
        blob = blob.strip()
        if not blob:
            continue
        try:
            return json.loads(blob)
        except ValueError:
            pass
        span = _first_json_span(blob)
        if span is None:
            continue
        try:
            return json.loads(span)
        except ValueError:
            continue
    raise ValueError(f"no JSON value in reply: {text[:300]!r}")


def _first_json_span(s: str) -> Optional[str]:
    """Return the first balanced {...} or [...] substring, or None."""
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    opener = s[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------

class OpenRouterClient:
    """Minimal chat client. One `requests.Session`, retries, no streaming."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 90.0,
        max_retries: int = 4,
        min_interval_s: float = 0.0,
        max_backoff_s: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.base_url = (base_url or os.getenv("OPENROUTER_URL")
                         or DEFAULT_URL).rstrip("/")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or None
        self.model = model or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL
        self.timeout_s = float(timeout_s)
        self.max_retries = int(_env_num("OPENROUTER_MAX_RETRIES", max_retries))
        self.min_interval_s = float(
            _env_num("OPENROUTER_MIN_INTERVAL_S", min_interval_s))
        self.max_backoff_s = float(max_backoff_s)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._session = None
        self._last_post_ts = 0.0

    # -- availability ------------------------------------------------------

    def available(self) -> bool:
        """True when a call could plausibly succeed (key present)."""
        return bool(self.api_key)

    def describe(self) -> str:
        """Model + endpoint, never the key."""
        return f"{self.model} @ {self.base_url}" + (
            "" if self.api_key else "  [NO KEY — OPENROUTER_API_KEY unset]")

    # -- transport ---------------------------------------------------------

    def _ensure_session(self):
        if self._session is not None:
            return
        if not self.api_key:
            raise LLMUnavailable(
                "OPENROUTER_API_KEY is not set. Add it to .env (see "
                ".env.example) or pass api_key=…"
            )
        try:
            import requests
        except ImportError as e:            # pragma: no cover - env fault
            raise LLMUnavailable("`pip install requests` for the LLM client.") from e
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "HTTP-Referer": _REFERER,
            "X-Title": _TITLE,
        })
        log.info("openrouter: %s", self.describe())

    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        wait = self.min_interval_s - (time.monotonic() - self._last_post_ts)
        if wait > 0:
            time.sleep(wait)

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_object: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Post one chat completion, return the assistant's text.

        `json_object=True` asks for `response_format={"type":"json_object"}`.
        Providers that do not honour it answer normally, which is why the
        caller still runs `extract_json` over the result.
        """
        self._ensure_session()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else float(temperature),
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        if extra_body:
            payload.update(extra_body)

        url = f"{self.base_url}/chat/completions"
        last_err: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout_s)
            except Exception as e:                  # network-level
                last_err = f"{type(e).__name__}: {e}"
                resp = None
            finally:
                self._last_post_ts = time.monotonic()

            if resp is not None and resp.status_code == 200:
                try:
                    body = resp.json()
                except ValueError as e:
                    raise LLMUnavailable(f"non-JSON 200 from {url}: {e}") from e
                return _content_of(body)

            if resp is not None:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in _RETRY_STATUS:
                    raise LLMUnavailable(f"openrouter refused: {last_err}")
                retry_after = _retry_after_s(resp)
            else:
                retry_after = None

            if attempt >= self.max_retries:
                break
            delay = retry_after if retry_after is not None else min(
                self.max_backoff_s, (2 ** attempt) * 1.0 + random.uniform(0, 0.5))
            log.warning("openrouter: attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1, self.max_retries + 1, last_err, delay)
            time.sleep(delay)

        raise LLMUnavailable(
            f"openrouter failed after {self.max_retries + 1} attempts: {last_err}")

    def chat_json(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repair_rounds: int = 1,
    ) -> Any:
        """`chat`, but parse the reply as JSON, re-asking once on garbage.

        The repair turn feeds the parser's own error back as a user message —
        cheaper and more reliable on a small model than raising at the caller.
        """
        convo = list(messages)
        err: Optional[Exception] = None
        for round_ in range(repair_rounds + 1):
            text = self.chat(convo, temperature=temperature,
                             max_tokens=max_tokens, json_object=True)
            try:
                return extract_json(text)
            except ValueError as e:
                err = e
                log.warning("openrouter: unparseable JSON (round %d): %s",
                            round_ + 1, e)
                if round_ >= repair_rounds:
                    break
                convo = list(messages) + [
                    {"role": "assistant", "content": text[:2000]},
                    {"role": "user", "content":
                        f"That was not valid JSON ({e}). Reply with the JSON "
                        f"object ONLY — no prose, no code fences."},
                ]
        raise LLMUnavailable(f"model did not return parseable JSON: {err}")


def _content_of(body: Dict[str, Any]) -> str:
    """Pull `choices[0].message.content` out of an OpenAI-shaped reply."""
    choices = body.get("choices") or []
    if not choices:
        err = body.get("error") or {}
        raise LLMUnavailable(
            f"no choices in reply: {err.get('message') or str(body)[:300]}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):        # some providers return content parts
        content = "".join(p.get("text", "") for p in content
                          if isinstance(p, dict))
    return (content or "").strip()


def _retry_after_s(resp) -> Optional[float]:
    raw = resp.headers.get("Retry-After") if resp.headers else None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


