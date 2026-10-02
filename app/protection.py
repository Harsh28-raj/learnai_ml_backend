"""Abuse protection without extra dependencies: per-IP rate limits and a request body cap.

Rate limits are a sliding 60-second window per (client IP, bucket), kept in memory (one
uvicorn worker). Memory is bounded: at most MAX_TRACKED_KEYS keys, least recently used evicted.
"""

import json
import math
import threading
import time
from collections import OrderedDict, deque

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

LLM_ACTIONS = frozenset({"tutor_chat", "generate_questions", "assessment"})
WINDOW_SECONDS = 60.0
MAX_TRACKED_KEYS = 5000
MAX_BODY_BYTES = 32 * 1024


def client_ip(request: Request) -> str:
    """Render sits behind a proxy: the first X-Forwarded-For entry is the real client."""
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    if first:
        return first
    return request.client.host if request.client else "unknown"


class RateLimiter:
    def __init__(self, max_keys: int = MAX_TRACKED_KEYS):
        self.max_keys = max_keys
        self._hits: OrderedDict[tuple[str, str], deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, ip: str, bucket: str, limit: int, now: float | None = None) -> int | None:
        """Record a hit. Returns None if allowed, else seconds until the next slot frees up."""
        if limit <= 0:
            return None
        now = time.monotonic() if now is None else now
        key = (ip, bucket)
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = self._hits[key] = deque()
            self._hits.move_to_end(key)
            while hits and now - hits[0] >= WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= limit:
                return max(1, math.ceil(WINDOW_SECONDS - (now - hits[0])))
            hits.append(now)
            while len(self._hits) > self.max_keys:
                self._hits.popitem(last=False)
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def __len__(self) -> int:
        return len(self._hits)


limiter = RateLimiter()


def bucket_for(action: str) -> str:
    return "llm" if action in LLM_ACTIONS else "other"


class BodySizeLimitMiddleware:
    """Reject request bodies over max_bytes with HTTP 413 in the standard error envelope."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return

        # No (or small) Content-Length: read the body ourselves, counting bytes, then replay it.
        chunks: list[Message] = []
        size = 0
        while True:
            message = await receive()
            chunks.append(message)
            if message["type"] == "http.request":
                size += len(message.get("body", b""))
                if size > self.max_bytes:
                    await self._reject(send)
                    return
                if not message.get("more_body", False):
                    break
            else:
                break

        async def replay() -> Message:
            return chunks.pop(0) if chunks else {"type": "http.disconnect"}

        await self.app(scope, replay, send)

    async def _reject(self, send: Send) -> None:
        body = json.dumps({
            "success": False, "action": "", "data": None, "fallback_data": None,
            "error": {"code": "PAYLOAD_TOO_LARGE", "message": f"Request body exceeds {self.max_bytes // 1024} KB.",
                      "details": {"max_bytes": self.max_bytes}},
        }).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
