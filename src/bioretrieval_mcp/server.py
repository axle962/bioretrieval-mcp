"""bioretrieval-mcp server: tool definitions and entry point."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from typing import Annotated

import anyio
import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from bioretrieval_mcp import __version__
from bioretrieval_mcp.cache import TTLCache
from bioretrieval_mcp.config import Settings, get_settings
from bioretrieval_mcp.http import HttpClient, NotFoundError, UpstreamError
from bioretrieval_mcp.schemas import (
    CpgptEmbedding,
    EmdbSearchResult,
    GeoSearchResult,
    MapHit,
    MapRetrievalResult,
    MethylationPlatform,
    SequenceEmbedding,
    UniProtEntry,
)
from bioretrieval_mcp.sources import emdb, geo, uniprot
from bioretrieval_mcp.validation import (
    ESM_MAX_RESIDUES,
    normalize_accession,
    normalize_emdb_query,
    normalize_free_text,
    normalize_sequence,
)

log = logging.getLogger("bioretrieval_mcp")

INSTRUCTIONS = """\
Tools for structural biology and epigenomics research.
- search_emdb: find cryo-EM entries (maps) by keyword/Solr query, optionally capped by resolution.
- get_uniprot: sequence, curated function, and Pfam families for a UniProt accession.
- embed_sequence: ESM-2 650M mean-pooled embedding of a protein sequence.
- retrieve_maps: rank cryo-EM maps by learned similarity to a protein *sequence* (two-tower model).
  Use it when the user has a sequence and wants structurally matching maps; use search_emdb for
  keyword/metadata lookups.
- search_geo_methylation: GEO series profiled on an Illumina methylation array (27K/450K/EPIC/EPICv2).
Typical chain: get_uniprot(accession) -> retrieve_maps(sequence) -> search_emdb for details.
"""

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True, idempotentHint=True)
LOCAL_COMPUTE = ToolAnnotations(readOnlyHint=True, openWorldHint=False, idempotentHint=True)


def _ml_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "esm"))


def _as_tool_error(exc: Exception) -> ToolError:
    if isinstance(exc, NotFoundError):
        return ToolError(f"Not found upstream: {exc}")
    if isinstance(exc, UpstreamError):
        return ToolError(f"Upstream service error (safe to retry later): {exc}")
    return ToolError(str(exc))


def create_server(
    settings: Settings | None = None,
    *,
    http: HttpClient | None = None,
    embedder=None,
    retriever=None,
    cpgpt=None,
) -> MCPServer:
    """Build the server. Collaborators can be injected (tests); otherwise they come from settings."""
    s = settings or get_settings()
    if http is None:
        http = HttpClient(
            TTLCache(s.http_cache_path, s.http_cache_ttl_s),
            timeout_s=s.http_timeout_s,
            max_retries=s.http_max_retries,
            backoff_s=s.http_backoff_s,
            min_interval_s={"eutils.ncbi.nlm.nih.gov": 0.11 if s.ncbi_api_key else 0.34},
        )
    if embedder is None and _ml_available():
        from bioretrieval_mcp.ml.esm import EsmEmbedder

        embedder = EsmEmbedder(s.esm_model, s.embedding_cache_dir, s.device)
    if retriever is None and embedder is not None and s.map_index is not None:
        from bioretrieval_mcp.ml.retrieval import Retriever

        retriever = Retriever.from_paths(s.map_index, s.retrieval_checkpoint, embedder)
    if cpgpt is None and s.cpgpt_embeddings is not None:
        from bioretrieval_mcp.ml.cpgpt import CpgptStore

        cpgpt = CpgptStore(s.cpgpt_embeddings)

    mcp = MCPServer("bioretrieval-mcp", version=__version__, instructions=INSTRUCTIONS)

    @mcp.tool(title="Search EMDB", annotations=READ_ONLY)
    async def search_emdb(
        query: Annotated[
            str,
            Field(
                description="Keywords or EMDB Solr syntax, e.g. 'apoferritin', 'spike AND Homo sapiens', "
                "'fitted_pdbs:7a6a', 'xref_UNIPROTKB:P02794'."
            ),
        ],
        resolution_max: Annotated[
            float | None, Field(gt=0, le=100, description="Only return maps at or better than this resolution (Å).")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=50, description="Maximum entries to return.")] = 10,
    ) -> EmdbSearchResult:
        """Search the Electron Microscopy Data Bank for cryo-EM entries.

        Returns per-entry metadata: title, resolution (Å), method, source organisms,
        fitted PDB models, UniProt cross-references, Pfam names, microscope, release
        date and primary citation. Results are relevance-ranked by EMDB.
        """
        try:
            return await emdb.search(http, normalize_emdb_query(query), resolution_max, limit)
        except Exception as exc:
            raise _as_tool_error(exc) from exc

    @mcp.tool(title="Get UniProt entry", annotations=READ_ONLY)
    async def get_uniprot(
        accession: Annotated[str, Field(description="UniProtKB accession, e.g. P02794 or P0DTC2.")],
    ) -> UniProtEntry:
        """Fetch a UniProtKB entry: amino-acid sequence, curated function, Pfam families,
        gene names, organism, review status, and EMDB cross-references."""
        try:
            return await uniprot.get_entry(http, normalize_accession(accession))
        except NotFoundError as exc:
            raise ToolError(f"No UniProtKB entry for accession '{accession}'.") from exc
        except Exception as exc:
            raise _as_tool_error(exc) from exc

    @mcp.tool(title="Search GEO methylation series", annotations=READ_ONLY)
    async def search_geo_methylation(
        query: Annotated[
            str,
            Field(description="Free-text GEO query, e.g. 'Alzheimer brain', 'aging blood', 'colorectal cancer'."),
        ],
        platform: Annotated[
            MethylationPlatform,
            Field(description="Illumina array: 27K (GPL8490), 450K (GPL13534), EPIC (GPL21145), EPICv2 (GPL33022)."),
        ],
        limit: Annotated[int, Field(ge=1, le=50, description="Maximum series to return.")] = 10,
    ) -> GeoSearchResult:
        """Search NCBI GEO for DNA-methylation series (GSE) run on a given Illumina array.

        Returns total hit count plus, per series: accession, title, summary (truncated),
        platforms, sample count, organisms, submission date and linked PubMed IDs.
        """
        try:
            return await geo.search(
                http,
                normalize_free_text(query),
                platform,
                limit,
                api_key=s.ncbi_api_key,
                email=s.ncbi_email,
            )
        except Exception as exc:
            raise _as_tool_error(exc) from exc

    if embedder is not None:

        @mcp.tool(title="Embed protein sequence (ESM-2)", annotations=LOCAL_COMPUTE)
        async def embed_sequence(
            sequence: Annotated[
                str, Field(description="Protein sequence (one-letter codes; FASTA header allowed).")
            ],
            include_vector: Annotated[
                bool, Field(description="Return the full 1280-d vector (large). Default returns summary only.")
            ] = False,
        ) -> SequenceEmbedding:
            """Compute an ESM-2 650M (esm2_t33_650M_UR50D) embedding: final-layer residue
            representations mean-pooled to 1280 dims. Sequences longer than 1022 residues are
            truncated. Results are cached on disk by sequence hash."""
            from bioretrieval_mcp.ml.esm import sequence_key

            try:
                seq = normalize_sequence(sequence)
                vec, cached = await anyio.to_thread.run_sync(embedder.embed, seq)
            except Exception as exc:
                raise _as_tool_error(exc) from exc
            return SequenceEmbedding(
                model=embedder.model_name,
                length=min(len(seq), ESM_MAX_RESIDUES),
                truncated=len(seq) > ESM_MAX_RESIDUES,
                dim=int(vec.shape[0]),
                pooling="mean over residue tokens, final layer",
                cached=cached,
                sha256=sequence_key(embedder.cache_namespace, seq),
                norm=round(float(np.linalg.norm(vec)), 4),
                embedding=[round(float(x), 5) for x in vec] if include_vector else None,
                preview=[round(float(x), 4) for x in vec[:8]],
            )

    if retriever is not None:

        @mcp.tool(title="Retrieve cryo-EM maps for a sequence", annotations=LOCAL_COMPUTE)
        async def retrieve_maps(
            sequence: Annotated[
                str, Field(description="Query protein sequence (one-letter codes; FASTA header allowed).")
            ],
            k: Annotated[int, Field(ge=1, le=50, description="Number of maps to return.")] = 10,
        ) -> MapRetrievalResult:
            """Rank indexed cryo-EM density maps by similarity to a protein sequence using the
            two-tower sequence/map retrieval model (ESM-2 sequence tower, 3D-CNN volume tower,
            shared latent space). Scores are cosine similarities; higher is better. Each hit
            includes the UniProt accession the map is paired with in the index."""
            try:
                seq = normalize_sequence(sequence)
                hits = await anyio.to_thread.run_sync(retriever.search, seq, k)
            except Exception as exc:
                raise _as_tool_error(exc) from exc
            return MapRetrievalResult(
                model=retriever.model_name,
                index_size=len(retriever.index),
                query_length=len(seq),
                hits=[MapHit(**h) for h in hits],
            )

    if cpgpt is not None:

        @mcp.tool(title="CpGPT sample embedding", annotations=LOCAL_COMPUTE)
        async def cpgpt_embed(
            sample_id: Annotated[str, Field(min_length=1, max_length=64, description="Sample ID, e.g. a GSM accession.")],
        ) -> CpgptEmbedding:
            """Return the precomputed CpGPT methylation-foundation-model embedding for a sample."""
            try:
                vec = cpgpt.get(sample_id.strip())
            except Exception as exc:
                raise _as_tool_error(exc) from exc
            return CpgptEmbedding(
                sample_id=sample_id.strip(),
                model=cpgpt.model,
                dim=int(vec.shape[0]),
                embedding=[round(float(x), 5) for x in vec],
            )

    return mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bioretrieval-mcp", description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default="/mcp", help="HTTP mount path for streamable-http")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    # stdout carries the protocol on stdio; logs must go to stderr.
    logging.basicConfig(level=args.log_level.upper(), stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    server = create_server()
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port, streamable_http_path=args.path)


if __name__ == "__main__":
    main()
