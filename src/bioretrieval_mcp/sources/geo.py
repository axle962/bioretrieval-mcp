"""GEO DataSets (db=gds) search through NCBI E-utilities, scoped to Illumina methylation arrays."""

from __future__ import annotations

from typing import Any

from bioretrieval_mcp.http import HttpClient
from bioretrieval_mcp.schemas import GeoSearchResult, GeoSeries, MethylationPlatform

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Each array has a canonical GPL plus re-annotated / alternate-manifest twins that
# submitters also use. Verified against GEO platform records.
PLATFORM_GPLS: dict[str, list[str]] = {
    "27K": ["GPL8490"],  # Illumina HumanMethylation27 BeadChip
    "450K": ["GPL13534", "GPL16304"],  # HumanMethylation450 (+ UBC enhanced annotation)
    "EPIC": ["GPL21145", "GPL23976"],  # Infinium MethylationEPIC v1 (+ HumanMethylation850 listing)
    "EPICv2": ["GPL33022"],  # Infinium MethylationEPIC v2.0
}


def build_term(query: str, platform: MethylationPlatform) -> str:
    gpls = " OR ".join(f"{g}[ACCN]" for g in PLATFORM_GPLS[platform])
    return f"({query}) AND ({gpls}) AND gse[ETYP]"


def _ids(value: str) -> list[str]:
    return [v.strip() for v in value.split(";") if v.strip()] if value else []


def parse_summary(doc: dict[str, Any]) -> GeoSeries:
    acc = doc.get("accession", "")
    return GeoSeries(
        accession=acc,
        title=doc.get("title", ""),
        summary=doc.get("summary", ""),
        platforms=[f"GPL{g}" for g in _ids(doc.get("gpl", ""))],
        n_samples=doc.get("n_samples"),
        organisms=_ids(doc.get("taxon", "")),
        gds_type=doc.get("gdstype") or None,
        submission_date=(doc.get("pdat") or "").replace("/", "-") or None,
        pubmed_ids=[str(p) for p in doc.get("pubmedids", [])],
        url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}",
    )


async def search(
    http: HttpClient,
    query: str,
    platform: MethylationPlatform,
    limit: int,
    *,
    api_key: str | None = None,
    email: str | None = None,
    summary_chars: int = 600,
) -> GeoSearchResult:
    common = {"retmode": "json", "tool": "bioretrieval-mcp", "email": email, "api_key": api_key}
    term = build_term(query, platform)
    found = await http.get(f"{EUTILS}/esearch.fcgi", params={"db": "gds", "term": term, "retmax": limit, **common})
    result = found.get("esearchresult", {})
    if "ERROR" in result:
        raise ValueError(f"GEO rejected the query: {result['ERROR']}")
    ids = result.get("idlist", [])
    series: list[GeoSeries] = []
    if ids:
        summ = await http.get(f"{EUTILS}/esummary.fcgi", params={"db": "gds", "id": ",".join(ids), **common})
        docs = summ.get("result", {})
        for uid in docs.get("uids", ids):
            if uid in docs:
                s = parse_summary(docs[uid])
                if len(s.summary) > summary_chars:
                    s.summary = s.summary[:summary_chars].rsplit(" ", 1)[0] + " ..."
                series.append(s)
    return GeoSearchResult(
        query=term,
        platform=platform,
        platform_gpls=PLATFORM_GPLS[platform],
        total_count=int(result.get("count", len(series))),
        returned=len(series),
        series=series,
    )
