"""Boot the real server process over stdio and streamable HTTP (no network calls made)."""

import os
import socket
import subprocess
import sys
import time

import anyio
import pytest
from mcp import Client, StdioServerParameters

pytestmark = pytest.mark.anyio

EXPECTED = {"search_emdb", "get_uniprot", "search_geo_methylation"}


def _env(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("BIORETRIEVAL_")}
    env["BIORETRIEVAL_CACHE_DIR"] = str(tmp_path)
    return env


async def test_stdio_transport(tmp_path):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "bioretrieval_mcp.server", "--transport", "stdio"], env=_env(tmp_path)
    )
    async with Client(params, read_timeout_seconds=60) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert names >= EXPECTED
        r = await c.call_tool("get_uniprot", {"accession": "not-an-accession"})
        assert r.is_error


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_streamable_http_transport(tmp_path):
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "bioretrieval_mcp.server", "--transport", "streamable-http", "--port", str(port)],
        env=_env(tmp_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            if proc.poll() is not None:
                pytest.fail(proc.stderr.read().decode())
            await anyio.sleep(0.2)
        async with Client(f"http://127.0.0.1:{port}/mcp", read_timeout_seconds=60) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert names >= EXPECTED
            r = await c.call_tool("search_geo_methylation", {"query": "x", "platform": "bogus"})
            assert r.is_error
    finally:
        proc.terminate()
        proc.wait(timeout=10)
