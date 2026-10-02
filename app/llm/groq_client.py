"""Async Groq client (OpenAI-compatible chat completions) over a shared httpx.AsyncClient.

The shared client is opened/closed by the FastAPI lifespan (init_client / close_client).
Logs model, latency and token usage per call; never logs prompt or completion text.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger("learnai.llm")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 1
RETRY_STATUS = {500, 502, 503, 504}
MAX_RATE_LIMIT_WAIT = 3.0  # on 429, wait+retry only if Retry-After <= this; else fail fast
MAX_CONCURRENT_CALLS = 2   # in-process limiter to avoid bursting Groq's per-minute limits
# gpt-oss models count hidden reasoning tokens against max_tokens; callers pass the budget for
# the visible answer and we add headroom for reasoning so answers are not truncated.
REASONING_HEADROOM = {"low": 512, "medium": 1536, "high": 3072}

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)

_client: httpx.AsyncClient | None = None
_limiter: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None


def _semaphore() -> asyncio.Semaphore:
    """One semaphore per running event loop (tests run many short-lived loops)."""
    global _limiter
    loop = asyncio.get_running_loop()
    if _limiter is None or _limiter[0] is not loop:
        _limiter = (loop, asyncio.Semaphore(MAX_CONCURRENT_CALLS))
    return _limiter[1]


class LLMError(Exception):
    def __init__(self, code: str, message: str, status_code: int | None = None, details: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


# --------------------------------------------------------------------------- client lifecycle

def init_client(client: httpx.AsyncClient | None = None) -> httpx.AsyncClient:
    """Create (or inject, for tests) the shared client."""
    global _client
    _client = client or httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_SECONDS))
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _get_client() -> httpx.AsyncClient:
    # Normally created in the lifespan; lazily created for scripts that skip it.
    return _client if _client is not None else init_client()


# --------------------------------------------------------------------------- parsing

# A backslash that does not start a valid JSON escape, e.g. LaTeX "\\frac" or "\\(" written raw.
_BAD_ESCAPE = re.compile(r'(?<!\\)\\(?!["\\/bfnrtu])')
# LaTeX commands that happen to start with a valid JSON escape (\n, \b, \f, \t, \r) and would
# otherwise silently become newline/tab/... characters. Only used in the lenient fallback path.
_LATEX_ESCAPE = re.compile(
    r"(?<!\\)\\(?=(?:nabla|neq|nu|beta|bar|bm|boldsymbol|bf|frac|forall|theta|tau|times|text|textbf|tilde|top|"
    r"rho|right|rightarrow|rangle)(?![A-Za-z]))"
)


def _loads_lenient(text: str):
    """json.loads, then a second try with invalid backslash escapes doubled (common with LaTeX)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_BAD_ESCAPE.sub(r"\\\\", _LATEX_ESCAPE.sub(r"\\\\", text)))
    except json.JSONDecodeError:
        return None


def parse_json_content(content: str) -> dict:
    """Parse model output as a JSON object, tolerating ``` fences and surrounding prose."""
    text = content.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    parsed = _loads_lenient(text)
    if parsed is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            parsed = _loads_lenient(text[start:end + 1])
    if parsed is None:
        raise LLMError("MODEL_ERROR", "Model did not return valid JSON.", details={"reason": "bad_json"})
    if not isinstance(parsed, dict):
        raise LLMError("MODEL_ERROR", "Model JSON was not an object.", details={"reason": "bad_json"})
    return parsed


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers["retry-after"])
    except (KeyError, ValueError):
        return None


# --------------------------------------------------------------------------- request

async def _post_with_retry(client: httpx.AsyncClient, body: dict, headers: dict) -> httpx.Response:
    """POST with one retry on 5xx/timeout/network errors.

    429: retry once only when Retry-After <= 3s; otherwise return immediately so the caller
    raises MODEL_RATE_LIMIT and the app can fall back without stalling the request.
    """
    resp: httpx.Response | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.post(GROQ_CHAT_URL, json=body, headers=headers, timeout=TIMEOUT_SECONDS)
        except httpx.TimeoutException as e:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1.0)
                continue
            raise LLMError("MODEL_TIMEOUT", "The AI model took too long to respond.", details=type(e).__name__)
        except httpx.HTTPError as e:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1.0)
                continue
            raise LLMError("MODEL_ERROR", "Could not reach the AI model provider.", details=type(e).__name__)
        if resp.status_code == 429 and attempt < MAX_RETRIES:
            wait = _retry_after(resp)
            if wait is not None and wait <= MAX_RATE_LIMIT_WAIT:
                await asyncio.sleep(wait)
                continue
            return resp
        if resp.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
            await asyncio.sleep(1.0)
            continue
        return resp
    assert resp is not None
    return resp


def _error_detail(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error", {})
        return f"{err.get('code') or err.get('type') or ''}: {err.get('message', '')}"[:300]
    except ValueError:
        return resp.text[:300]


async def chat_completion(
    messages: list[dict[str, str]],
    model: str,
    json_mode: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1200,
    reasoning_effort: str | None = None,
) -> str | dict:
    """Return message.content (str), or a parsed dict when json_mode=True.

    `reasoning_effort` defaults to GROQ_REASONING_EFFORT; pass "" to disable.
    """
    settings = get_settings()
    api_key = settings.groq_api_key.strip()
    if not api_key:
        raise LLMError("LLM_NOT_CONFIGURED", "GROQ_API_KEY is not set.")
    effort = settings.groq_reasoning_effort.strip() if reasoning_effort is None else reasoning_effort.strip()

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens + (REASONING_HEADROOM.get(effort, 1024) if effort else 0),
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if effort:
        body["reasoning_effort"] = effort
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    client = _get_client()
    started = time.perf_counter()
    async with _semaphore():
        resp = await _post_with_retry(client, body, headers)

        if resp.status_code == 400 and "reasoning_effort" in body and "json_validate_failed" not in resp.text:
            logger.warning("Groq rejected reasoning_effort=%s for model=%s (%s); retrying without it",
                           effort, model, _error_detail(resp))
            body.pop("reasoning_effort")
            body["max_tokens"] = max_tokens
            resp = await _post_with_retry(client, body, headers)

    latency_ms = round((time.perf_counter() - started) * 1000)

    if resp.status_code == 429:
        logger.warning("Groq rate limit model=%s latency_ms=%d", model, latency_ms)
        raise LLMError("MODEL_RATE_LIMIT", "The AI service is experiencing high load.", 429, _error_detail(resp))
    if resp.status_code == 400 and "json_validate_failed" in resp.text:
        # Groq rejects output that is not strict JSON but returns it as failed_generation;
        # salvage it when the only problem is something we can repair (e.g. LaTeX backslashes).
        try:
            failed = (resp.json().get("error") or {}).get("failed_generation") or ""
        except ValueError:
            failed = ""
        if json_mode and failed:
            try:
                salvaged = parse_json_content(failed)
                logger.warning("Groq JSON validation failed model=%s latency_ms=%d; salvaged failed_generation",
                               model, latency_ms)
                _record_latency(latency_ms)
                return salvaged
            except LLMError:
                pass
        logger.warning("Groq JSON validation failed model=%s latency_ms=%d", model, latency_ms)
        raise LLMError("MODEL_ERROR", "Model did not return valid JSON.", 400, {"reason": "bad_json"})
    if resp.status_code >= 400:
        logger.warning("Groq HTTP %d model=%s latency_ms=%d", resp.status_code, model, latency_ms)
        raise LLMError("MODEL_ERROR", f"AI model provider returned HTTP {resp.status_code}.",
                       resp.status_code, _error_detail(resp))

    try:
        data = resp.json()
        message = data["choices"][0]["message"]
        content = message.get("content") or ""  # any `reasoning` field is deliberately ignored
        usage = data.get("usage") or {}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
        raise LLMError("MODEL_ERROR", "Unexpected response shape from the AI model.", resp.status_code,
                       type(e).__name__)

    logger.info(
        "Groq call model=%s latency_ms=%d prompt_tokens=%s completion_tokens=%s total_tokens=%s",
        model, latency_ms, usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"),
    )
    _record_latency(latency_ms)

    if not content.strip():
        raise LLMError("MODEL_ERROR", "The AI model returned an empty response.", resp.status_code,
                       {"reason": "empty_content", "finish_reason": data["choices"][0].get("finish_reason")})
    return parse_json_content(content) if json_mode else content


# --------------------------------------------------------------------------- stats

_stats = {"calls": 0, "total_latency_ms": 0}


def _record_latency(ms: int) -> None:
    _stats["calls"] += 1
    _stats["total_latency_ms"] += ms


def latency_stats() -> dict:
    calls = _stats["calls"]
    return {"calls": calls, "avg_latency_ms": round(_stats["total_latency_ms"] / calls) if calls else None}
