"""Async HTTP client with timeouts, retry/backoff, and response caching."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from bioretrieval_mcp import __version__
from bioretrieval_mcp.cache import TTLCache, make_key

log = logging.getLogger(__name__)

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
USER_AGENT = f"bioretrieval-mcp/{__version__} (+https://github.com/axle962/bioretrieval-mcp)"


class UpstreamError(RuntimeError):
    """An upstream API failed in a way the caller should report to the agent."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class NotFoundError(UpstreamError):
    pass


def _retry_after_s(resp: httpx.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


class HttpClient:
    def __init__(
        self,
        cache: TTLCache | None = None,
        *,
        timeout_s: float = 20.0,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
        min_interval_s: dict[str, float] | None = None,
    ):
        self.cache = cache
        # Per-host politeness spacing, e.g. {"eutils.ncbi.nlm.nih.gov": 0.34} for NCBI's 3 req/s.
        self.min_interval_s = min_interval_s or {}
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._host_last: dict[str, float] = {}
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            transport=transport,
        )

    async def _pace(self, url: str) -> None:
        host = httpx.URL(url).host
        interval = self.min_interval_s.get(host)
        if not interval:
            return
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._host_last.get(host, 0.0) + interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._host_last[host] = time.monotonic()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get(self, url: str, params: dict[str, Any] | None = None, *, as_json: bool = True) -> Any:
        """GET ``url`` and return parsed JSON (or text). Cached by (url, params).

        Retries transient failures (timeouts, connection errors, 429, 5xx) with
        exponential backoff and jitter, honouring ``Retry-After``. A 404 raises
        :class:`NotFoundError` immediately; other 4xx raise :class:`UpstreamError`.
        """
        params = {k: v for k, v in (params or {}).items() if v is not None}
        key = make_key("GET", url, params, as_json)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return hit

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            await self._pace(url)
            try:
                resp = await self._client.get(url, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                delay = None
            else:
                if resp.status_code == 404:
                    raise NotFoundError(f"Not found: {resp.request.url}", status=404)
                if resp.status_code in RETRY_STATUS:
                    last_exc = UpstreamError(f"HTTP {resp.status_code} from {url}", status=resp.status_code)
                    delay = _retry_after_s(resp)
                elif resp.status_code >= 400:
                    raise UpstreamError(
                        f"HTTP {resp.status_code} from {url}: {resp.text[:200]}", status=resp.status_code
                    )
                else:
                    try:
                        data = resp.json() if as_json else resp.text
                    except ValueError as exc:
                        raise UpstreamError(f"Invalid JSON from {url}") from exc
                    if self.cache is not None:
                        self.cache.set(key, data)
                    return data

            if attempt == self.max_retries:
                break
            if delay is None:
                delay = self.backoff_s * (2**attempt) + random.uniform(0, self.backoff_s)
            log.warning("GET %s failed (%s); retry %d/%d in %.2fs", url, last_exc, attempt + 1, self.max_retries, delay)
            await asyncio.sleep(min(delay, 30.0))

        if isinstance(last_exc, UpstreamError):
            raise last_exc
        raise UpstreamError(f"Request to {url} failed after {self.max_retries + 1} attempts: {last_exc!r}")
