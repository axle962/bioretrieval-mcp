"""Checkpoint loading for both trainer formats (needs torch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from bioretrieval_mcp.ml.retrieval import MapIndex, Retriever, _unwrap_state_dict, build_sequence_head  # noqa: E402
from tests.conftest import FERRITIN_SEQ  # noqa: E402

pytestmark = pytest.mark.ml


def make_head(act, in_dim=1280, hidden=1280, out_dim=512):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), act(), nn.Linear(hidden, out_dim)).eval()


@pytest.mark.parametrize(
    "fmt,prefix,act,hidden,out_dim",
    [
        ("node2", "sequence_projection.net.", nn.GELU, 1280, 512),  # {"model": ...}
        ("mmprotein", "seq_projector.net.", nn.ReLU, 1280, 256),  # {"model_state_dict": ...}
    ],
)
def test_head_reconstruction_matches_original(fmt, prefix, act, hidden, out_dim):
    ref = make_head(act, hidden=hidden, out_dim=out_dim)
    state = {prefix + k: v for k, v in ref.state_dict().items()}
    state["volume_encoder.net.0.weight"] = torch.zeros(1)  # unrelated keys are ignored
    ckpt = {"model": state} if fmt == "node2" else {"model_state_dict": state, "config": {}}
    head, found = build_sequence_head(_unwrap_state_dict(ckpt))
    assert found == prefix
    x = torch.randn(3, 1280)
    with torch.no_grad():
        assert torch.allclose(head(x), ref(x), atol=1e-6)


def test_unwrap_ddp_prefix():
    assert _unwrap_state_dict({"state_dict": {"module.a": 1, "module.b": 2}}) == {"a": 1, "b": 2}


def test_missing_head_errors():
    with pytest.raises(ValueError, match="No sequence projection head"):
        build_sequence_head({"foo.weight": torch.zeros(1)})


def test_from_paths_end_to_end(tmp_path, fake_embedder):
    head = make_head(nn.GELU)
    torch.save({"model": {"sequence_projection.net." + k: v for k, v in head.state_dict().items()}}, tmp_path / "ckpt.pt")
    q, _ = fake_embedder.embed(FERRITIN_SEQ)
    with torch.no_grad():
        target = head(torch.from_numpy(q)[None]).numpy()[0]
    rng = np.random.default_rng(1)
    emb = np.vstack([rng.standard_normal((20, 512)), target]).astype(np.float32)
    np.savez(tmp_path / "idx.npz", emdb_ids=np.array([f"EMD-{i}" for i in range(21)]), embeddings=emb, model=np.array("tt"))

    r = Retriever.from_paths(tmp_path / "idx.npz", tmp_path / "ckpt.pt", fake_embedder)
    hits = r.search(FERRITIN_SEQ, 5)
    assert hits[0]["emdb_id"] == "EMD-20" and hits[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert hits[0]["uniprot_accession"] is None and len(hits) == 5
    assert r.model_name == "tt (ckpt.pt)"


def test_two_tower_index_requires_head(tmp_path, fake_embedder):
    np.savez(tmp_path / "idx.npz", emdb_ids=np.array(["EMD-1"]), embeddings=np.ones((1, 512)))
    with pytest.raises(ValueError, match="needs the checkpoint"):
        Retriever(MapIndex.load(tmp_path / "idx.npz"), fake_embedder)


def test_dim_mismatch(tmp_path, fake_embedder):
    np.savez(tmp_path / "idx.npz", emdb_ids=np.array(["EMD-1"]), embeddings=np.ones((1, 64)), kind=np.array("esm_seq_baseline"))
    r = Retriever(MapIndex.load(tmp_path / "idx.npz"), fake_embedder)
    with pytest.raises(ValueError, match="does not match index dim"):
        r.search(FERRITIN_SEQ, 1)


def test_index_length_mismatch(tmp_path):
    np.savez(tmp_path / "idx.npz", emdb_ids=np.array(["EMD-1", "EMD-2"]), embeddings=np.ones((1, 8)))
    with pytest.raises(ValueError, match="2 ids but 1 embeddings"):
        MapIndex.load(tmp_path / "idx.npz")
