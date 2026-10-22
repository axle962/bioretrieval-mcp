"""Pydantic output models. These become each tool's JSON ``outputSchema``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MethylationPlatform = Literal["27K", "450K", "EPIC", "EPICv2"]


class EmdbEntry(BaseModel):
    emdb_id: str = Field(description="EMDB accession, e.g. EMD-11668")
    title: str
    resolution_angstrom: float | None = Field(description="Reported resolution in Å (lower is better)")
    method: str | None = Field(description="singleParticle, subtomogramAveraging, helical, tomography, ...")
    organisms: list[str]
    fitted_pdbs: list[str]
    uniprot_accessions: list[str]
    pfam_names: list[str]
    microscope: str | None
    release_date: str | None
    citation_title: str | None
    pubmed_ids: list[str]
    url: str


class EmdbSearchResult(BaseModel):
    query: str = Field(description="Exact Solr query sent to EMDB")
    returned: int
    entries: list[EmdbEntry]


class PfamFamily(BaseModel):
    pfam_id: str
    name: str | None


class UniProtEntry(BaseModel):
    accession: str
    entry_name: str
    reviewed: bool
    protein_name: str | None
    genes: list[str]
    organism: str | None
    taxon_id: int | None
    length: int
    sequence: str
    function: str | None = Field(description="UniProt 'FUNCTION' comment, if curated")
    pfam: list[PfamFamily]
    emdb_cross_refs: list[str] = Field(description="EMDB entries cross-referenced by UniProt")
    url: str


class SequenceEmbedding(BaseModel):
    model: str
    length: int = Field(description="Residues embedded (after truncation)")
    truncated: bool = Field(description="True if the sequence exceeded ESM-2's 1022-residue window")
    dim: int
    pooling: str
    cached: bool
    sha256: str = Field(description="Hash of the normalized sequence; the embedding cache key")
    norm: float
    embedding: list[float] | None = Field(description="Omitted unless include_vector=true (1280 floats)")
    preview: list[float] = Field(description="First 8 dimensions, for a quick sanity check")


class MapHit(BaseModel):
    rank: int
    emdb_id: str
    score: float = Field(description="Cosine similarity in the shared latent space, in [-1, 1]")
    uniprot_accession: str | None = Field(description="UniProt accession paired with this map in the index")
    resolution_angstrom: float | None
    title: str | None
    url: str


class MapRetrievalResult(BaseModel):
    model: str
    index_size: int
    query_length: int
    hits: list[MapHit]


class GeoSeries(BaseModel):
    accession: str
    title: str
    summary: str
    platforms: list[str] = Field(description="GPL accessions in the series")
    n_samples: int | None
    organisms: list[str]
    gds_type: str | None
    submission_date: str | None
    pubmed_ids: list[str]
    url: str


class GeoSearchResult(BaseModel):
    query: str
    platform: MethylationPlatform
    platform_gpls: list[str]
    total_count: int = Field(description="Total matching series in GEO (results are paged)")
    returned: int
    series: list[GeoSeries]


class CpgptEmbedding(BaseModel):
    sample_id: str
    model: str
    dim: int
    embedding: list[float]
