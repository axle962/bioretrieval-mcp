import httpx
import pytest

from bioretrieval_mcp import http as http_mod
from bioretrieval_mcp.cache import TTLCache
from bioretrieval_mcp.http import HttpClient, NotFoundError, UpstreamError

pytestmark = pytest.mark.anyio


@pytest.fixture
def sleeps(monkeypatch):
    calls: list[float] = []

    async def fake_sleep(delay):
        calls.append(delay)

    monkeypatch.setattr(http_mod.asyncio, "sleep", fake_sleep)
    return calls


def sequence_handler(*responses):
    """Serve responses in order; an Exception instance is raised instead of returned."""
    it = iter(responses)
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    handler.seen = seen
    return handler


async def test_retries_transient_status_then_succeeds(make_http, sleeps):
    h = sequence_handler(httpx.Response(503), httpx.Response(502), httpx.Response(200, json={"ok": 1}))
    assert await make_http(h).get("https://x.test/a") == {"ok": 1}
    assert len(h.seen) == 3
    assert len(sleeps) == 2


async def test_honours_retry_after(make_http, sleeps):
    h = sequence_handler(httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json={}))
    await make_http(h).get("https://x.test/a")
    assert sleeps == [7.0]


async def test_retries_timeouts_and_gives_up(make_http, sleeps):
    h = sequence_handler(*[httpx.ReadTimeout("slow")] * 3)
    with pytest.raises(UpstreamError, match="failed after 3 attempts"):
        await make_http(h, retries=2).get("https://x.test/a")
    assert len(h.seen) == 3


async def test_exhausted_5xx_reports_status(make_http, sleeps):
    h = sequence_handler(*[httpx.Response(500)] * 3)
    with pytest.raises(UpstreamError) as info:
        await make_http(h, retries=2).get("https://x.test/a")
    assert info.value.status == 500


async def test_404_is_not_retried(make_http, sleeps):
    h = sequence_handler(httpx.Response(404))
    with pytest.raises(NotFoundError):
        await make_http(h).get("https://x.test/a")
    assert len(h.seen) == 1 and sleeps == []


async def test_other_4xx_not_retried(make_http, sleeps):
    h = sequence_handler(httpx.Response(400, text="bad query"))
    with pytest.raises(UpstreamError, match="bad query"):
        await make_http(h).get("https://x.test/a")
    assert len(h.seen) == 1


async def test_cache_serves_repeat_requests(make_http):
    h = sequence_handler(httpx.Response(200, json={"n": 1}))
    client = make_http(h)
    assert await client.get("https://x.test/a", {"q": "1"}) == {"n": 1}
    assert await client.get("https://x.test/a", {"q": "1"}) == {"n": 1}
    assert len(h.seen) == 1
    assert client.cache.hits == 1


async def test_cache_key_includes_params(make_http):
    h = sequence_handler(httpx.Response(200, json=1), httpx.Response(200, json=2))
    client = make_http(h)
    assert await client.get("https://x.test/a", {"q": "1"}) == 1
    assert await client.get("https://x.test/a", {"q": "2"}) == 2


async def test_none_params_dropped(make_http):
    h = sequence_handler(httpx.Response(200, json={}))
    await make_http(h).get("https://x.test/a", {"api_key": None, "q": "x"})
    assert "api_key" not in str(h.seen[0].url)


async def test_errors_are_not_cached(make_http, sleeps):
    h = sequence_handler(httpx.Response(404), httpx.Response(200, json={"ok": True}))
    client = make_http(h)
    with pytest.raises(NotFoundError):
        await client.get("https://x.test/a")
    assert await client.get("https://x.test/a") == {"ok": True}


async def test_text_mode_and_user_agent(make_http):
    h = sequence_handler(httpx.Response(200, text="a,b\n1,2\n"))
    assert await make_http(h).get("https://x.test/a", as_json=False) == "a,b\n1,2\n"
    assert h.seen[0].headers["user-agent"].startswith("bioretrieval-mcp/")


async def test_invalid_json_raises(make_http):
    h = sequence_handler(httpx.Response(200, text="<html>"))
    with pytest.raises(UpstreamError, match="Invalid JSON"):
        await make_http(h).get("https://x.test/a")


def test_ttl_cache_expiry(tmp_path, monkeypatch):
    cache = TTLCache(tmp_path / "c.sqlite", ttl_s=10)
    now = [1000.0]
    monkeypatch.setattr("bioretrieval_mcp.cache.time.time", lambda: now[0])
    cache.set("k", {"v": 1})
    assert cache.get("k") == {"v": 1}
    now[0] += 11
    assert cache.get("k") is None


def test_ttl_cache_persists(tmp_path):
    TTLCache(tmp_path / "c.sqlite", ttl_s=60).set("k", [1, 2])
    assert TTLCache(tmp_path / "c.sqlite", ttl_s=60).get("k") == [1, 2]


async def test_client_closes():
    client = HttpClient(None, transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    await client.aclose()


async def test_per_host_pacing(sleeps):
    client = HttpClient(
        None,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        min_interval_s={"slow.test": 1.0},
    )
    await client.get("https://slow.test/a")
    await client.get("https://slow.test/b")
    await client.get("https://fast.test/c")
    assert len(sleeps) == 1 and 0 < sleeps[0] <= 1.0
