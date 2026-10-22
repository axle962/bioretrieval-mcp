from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import pytest

from bioretrieval_mcp.cache import TTLCache
from bioretrieval_mcp.config import Settings
from bioretrieval_mcp.http import HttpClient

FIXTURES = Path(__file__).parent / "fixtures"

FERRITIN_SEQ = (
    "MTTASTSQVRQNYHQDSEAAINRQINLELYASYVYLSMSYYFDRDDVALKNFAKYFLHQSHEEREHAEKLMKLQNQRGGRIFLQDIKKPDCDDWESGLNAM"
    "ECALHLEKNVNQSLLELHKLATDKNDPHLCDFIETHYLNEQVKAIKELGDHVTNLRKMGAPESGLAEYLFDKHTLGDSDNES"
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Keep the developer's BIORETRIEVAL_* / NCBI_* settings out of the tests."""
    import os

    for name in list(os.environ):
        if name.startswith(("BIORETRIEVAL_", "NCBI_")):
            monkeypatch.delenv(name)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def fixture_json(name: str):
    return json.loads(fixture_text(name))


Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    """MockTransport wrapper that records every request it serves."""

    def __init__(self, handler: Handler):
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)


def default_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "ebi.ac.uk/emdb/api/search" in url:
        return httpx.Response(200, text=fixture_text("emdb_search.csv"))
    if "rest.uniprot.org/uniprotkb/P02794.json" in url:
        return httpx.Response(200, json=fixture_json("uniprot_P02794.json"))
    if "rest.uniprot.org/uniprotkb/" in url:
        return httpx.Response(404, json={"messages": ["not found"]})
    if "esearch.fcgi" in url:
        return httpx.Response(200, json=fixture_json("geo_esearch.json"))
    if "esummary.fcgi" in url:
        return httpx.Response(200, json=fixture_json("geo_esummary.json"))
    return httpx.Response(500, text=f"unmocked {url}")


@pytest.fixture
def recorder() -> Recorder:
    return Recorder(default_handler)


@pytest.fixture
def make_http():
    clients: list[HttpClient] = []

    def _make(handler: Handler, *, cache: bool = True, retries: int = 2) -> HttpClient:
        client = HttpClient(
            TTLCache(None, 60) if cache else None,
            max_retries=retries,
            backoff_s=0.0,
            transport=httpx.MockTransport(handler),
        )
        clients.append(client)
        return client

    return _make


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(cache_dir=tmp_path, http_cache_ttl_s=60)


class FakeEmbedder:
    """Deterministic stand-in for ESM-2: a hash-seeded 1280-d vector."""

    model_name = "fake-esm"
    cache_namespace = "fake-esm"

    def __init__(self):
        self.calls = 0
        self._seen: set[str] = set()

    def embed(self, sequence: str):
        self.calls += 1
        cached = sequence in self._seen
        self._seen.add(sequence)
        rng = np.random.default_rng(int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16))
        return rng.standard_normal(1280).astype(np.float32), cached


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
