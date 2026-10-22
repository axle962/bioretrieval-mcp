"""End-to-end tool tests through a real MCP client session (in-process), with mocked HTTP."""

import httpx
import numpy as np
import pytest
from mcp import Client

from bioretrieval_mcp.ml.cpgpt import CpgptStore
from bioretrieval_mcp.ml.retrieval import MapIndex, Retriever
from bioretrieval_mcp.server import create_server
from tests.conftest import FERRITIN_SEQ

pytestmark = pytest.mark.anyio


def baseline_retriever(embedder, n_decoys=5):
    """Index where the ferritin map is the embedder's own vector, so it must rank first."""
    target, _ = embedder.embed(FERRITIN_SEQ)
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.standard_normal((n_decoys, 1280)), target, target * 0.9 + 0.1]).astype(np.float32)
    ids = [f"EMD-{1000 + i}" for i in range(n_decoys)] + ["EMD-11668", "EMD-11668"]
    index = MapIndex(
        emdb_ids=np.array(ids),
        embeddings=emb / np.linalg.norm(emb, axis=1, keepdims=True),
        uniprot_ids=np.array(["Q00000"] * n_decoys + ["P02794", "P02794"]),
        resolutions=np.array([3.0] * n_decoys + [1.15, np.nan]),
        titles=np.array(["decoy"] * n_decoys + ["apoferritin", "apoferritin"]),
        kind="esm_seq_baseline",
        model="test-baseline",
    )
    return Retriever(index, embedder)


@pytest.fixture
def server(settings, make_http, recorder, fake_embedder, tmp_path):
    np.savez(tmp_path / "cpgpt.npz", sample_ids=np.array(["GSM1", "GSM2"]), embeddings=np.ones((2, 4)), model=np.array("cpgpt-test"))
    return create_server(
        settings,
        http=make_http(recorder),
        embedder=fake_embedder,
        retriever=baseline_retriever(fake_embedder),
        cpgpt=CpgptStore(tmp_path / "cpgpt.npz"),
    )


async def call(server, name, args):
    async with Client(server) as c:
        return await c.call_tool(name, args)


async def test_lists_all_tools_with_schemas(server):
    async with Client(server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
        assert c.instructions and "retrieve_maps" in c.instructions
    assert set(tools) == {
        "search_emdb", "get_uniprot", "embed_sequence", "retrieve_maps", "search_geo_methylation", "cpgpt_embed",
    }
    for t in tools.values():
        assert t.description and t.output_schema and t.annotations.read_only_hint
    geo_props = tools["search_geo_methylation"].input_schema["properties"]
    assert geo_props["platform"]["enum"] == ["27K", "450K", "EPIC", "EPICv2"]
    assert tools["search_geo_methylation"].input_schema["required"] == ["query", "platform"]
    assert tools["retrieve_maps"].input_schema["properties"]["k"]["maximum"] == 50
    # Agents need to know which part of a long sequence is kept (q31 in the eval had to guess).
    for name in ("embed_sequence", "retrieve_maps"):
        assert "first 1022 residues" in tools[name].description
    assert tools["search_emdb"].input_schema["properties"]["resolution_max"]["anyOf"][0]["exclusiveMinimum"] == 0


async def test_optional_tools_absent_without_artifacts(settings, make_http, recorder):
    srv = create_server(settings, http=make_http(recorder), embedder=None, retriever=None, cpgpt=None)
    async with Client(srv) as c:
        names = {t.name for t in (await c.list_tools()).tools}
    assert "cpgpt_embed" not in names and "retrieve_maps" not in names


async def test_search_emdb(server, recorder):
    r = await call(server, "search_emdb", {"query": "apoferritin", "resolution_max": 2, "limit": 2})
    assert not r.is_error
    assert r.structured_content["query"] == "(apoferritin) AND resolution:[0 TO 2]"
    assert r.structured_content["entries"][0]["emdb_id"] == "EMD-57470"


@pytest.mark.parametrize(
    "args",
    [{"query": ""}, {"query": "x", "resolution_max": 0}, {"query": "x", "limit": 51}, {"query": "a{b}"}],
)
async def test_search_emdb_rejects_bad_input(server, recorder, args):
    r = await call(server, "search_emdb", args)
    assert r.is_error
    assert recorder.requests == []


async def test_get_uniprot(server):
    r = await call(server, "get_uniprot", {"accession": "p02794"})
    sc = r.structured_content
    assert not r.is_error and sc["accession"] == "P02794" and sc["pfam"][0]["pfam_id"] == "PF00210"


async def test_get_uniprot_not_found(server):
    r = await call(server, "get_uniprot", {"accession": "Q99999"})
    assert r.is_error and "No UniProtKB entry" in r.content[0].text


async def test_get_uniprot_invalid(server, recorder):
    r = await call(server, "get_uniprot", {"accession": "ferritin"})
    assert r.is_error and "not a valid UniProt accession" in r.content[0].text
    assert recorder.requests == []


async def test_upstream_failure_is_reported_not_raised(settings, make_http, fake_embedder, monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("bioretrieval_mcp.http.asyncio.sleep", no_sleep)
    srv = create_server(settings, http=make_http(lambda req: httpx.Response(503)), embedder=fake_embedder)
    r = await call(srv, "get_uniprot", {"accession": "P02794"})
    assert r.is_error and "safe to retry" in r.content[0].text


async def test_search_geo(server):
    r = await call(server, "search_geo_methylation", {"query": "Alzheimer", "platform": "450K", "limit": 2})
    sc = r.structured_content
    assert not r.is_error and sc["platform_gpls"] == ["GPL13534", "GPL16304"] and sc["series"][0]["accession"] == "GSE144858"


async def test_search_geo_bad_platform(server, recorder):
    r = await call(server, "search_geo_methylation", {"query": "x", "platform": "850K"})
    assert r.is_error and recorder.requests == []


async def test_embed_sequence(server, fake_embedder):
    r = await call(server, "embed_sequence", {"sequence": ">x\n" + FERRITIN_SEQ.lower()})
    sc = r.structured_content
    assert not r.is_error
    assert sc["dim"] == 1280 and sc["length"] == 183 and sc["embedding"] is None and len(sc["preview"]) == 8
    r = await call(server, "embed_sequence", {"sequence": FERRITIN_SEQ, "include_vector": True})
    assert len(r.structured_content["embedding"]) == 1280


async def test_embed_sequence_reports_truncation(server):
    r = await call(server, "embed_sequence", {"sequence": "M" * 1500})
    assert r.structured_content["truncated"] is True and r.structured_content["length"] == 1022


async def test_embed_rejects_dna(server, fake_embedder):
    before = fake_embedder.calls
    r = await call(server, "embed_sequence", {"sequence": "ACGT" * 20})
    assert r.is_error and fake_embedder.calls == before


async def test_retrieve_maps_ranks_and_dedupes(server):
    r = await call(server, "retrieve_maps", {"sequence": FERRITIN_SEQ, "k": 3})
    sc = r.structured_content
    assert not r.is_error
    assert sc["index_size"] == 7 and sc["model"] == "test-baseline"
    hits = sc["hits"]
    assert [h["rank"] for h in hits] == [1, 2, 3]
    assert hits[0]["emdb_id"] == "EMD-11668" and hits[0]["score"] == pytest.approx(1.0, abs=1e-3)
    assert hits[0]["uniprot_accession"] == "P02794" and hits[0]["resolution_angstrom"] == 1.15
    assert [h["emdb_id"] for h in hits].count("EMD-11668") == 1
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]


async def test_cpgpt_embed(server):
    r = await call(server, "cpgpt_embed", {"sample_id": "GSM2"})
    assert r.structured_content == {"sample_id": "GSM2", "model": "cpgpt-test", "dim": 4, "embedding": [1.0] * 4}
    r = await call(server, "cpgpt_embed", {"sample_id": "GSM404"})
    assert r.is_error and "not in the CpGPT embedding store" in r.content[0].text
