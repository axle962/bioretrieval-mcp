"""EMDB search via the EBI Solr-backed API (https://www.ebi.ac.uk/emdb/api)."""

from __future__ import annotations

import csv
import io

from bioretrieval_mcp.http import HttpClient
from bioretrieval_mcp.schemas import EmdbEntry, EmdbSearchResult

SEARCH_URL = "https://www.ebi.ac.uk/emdb/api/search/{query}"
FIELDS = [
    "emdb_id",
    "title",
    "resolution",
    "structure_determination_method",
    "natural_source_organism",
    "fitted_pdbs",
    "xref_UNIPROTKB",
    "pfam_name",
    "microscope_name",
    "map_release_date",
    "primary_citation_title",
    "xref_PUBMED",
]


def _split(value: str) -> list[str]:
    # Multi-valued Solr fields come back comma-joined in CSV mode.
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_query(query: str, resolution_max: float | None) -> str:
    q = f"({query})"
    if resolution_max is not None:
        q += f" AND resolution:[0 TO {resolution_max:g}]"
    return q


def parse_csv(text: str) -> list[EmdbEntry]:
    entries = []
    for row in csv.DictReader(io.StringIO(text)):
        emdb_id = row.get("emdb_id", "").strip()
        if not emdb_id:
            continue
        entries.append(
            EmdbEntry(
                emdb_id=emdb_id,
                title=row.get("title", "").strip(),
                resolution_angstrom=_float(row.get("resolution", "")),
                method=row.get("structure_determination_method") or None,
                organisms=_split(row.get("natural_source_organism", "")),
                fitted_pdbs=[p.upper() for p in _split(row.get("fitted_pdbs", ""))],
                uniprot_accessions=_split(row.get("xref_UNIPROTKB", "")),
                pfam_names=_split(row.get("pfam_name", "")),
                microscope=row.get("microscope_name") or None,
                release_date=(row.get("map_release_date") or "")[:10] or None,
                citation_title=row.get("primary_citation_title") or None,
                pubmed_ids=_split(row.get("xref_PUBMED", "")),
                url=f"https://www.ebi.ac.uk/emdb/{emdb_id}",
            )
        )
    return entries


async def search(http: HttpClient, query: str, resolution_max: float | None, limit: int) -> EmdbSearchResult:
    solr = build_query(query, resolution_max)
    # The query is a path segment; httpx percent-encodes it.
    text = await http.get(
        SEARCH_URL.format(query=solr),
        params={"rows": limit, "wt": "csv", "fl": ",".join(FIELDS)},
        as_json=False,
    )
    entries = parse_csv(text)[:limit]
    return EmdbSearchResult(query=solr, returned=len(entries), entries=entries)
