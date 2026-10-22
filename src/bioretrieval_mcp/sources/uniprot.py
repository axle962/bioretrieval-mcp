"""UniProtKB REST client (https://rest.uniprot.org)."""

from __future__ import annotations

from typing import Any

from bioretrieval_mcp.http import HttpClient
from bioretrieval_mcp.schemas import PfamFamily, UniProtEntry

ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"


def _protein_name(desc: dict[str, Any]) -> str | None:
    for key in ("recommendedName", "submissionNames"):
        node = desc.get(key)
        if isinstance(node, list):
            node = node[0] if node else None
        if node and (name := node.get("fullName", {}).get("value")):
            return name
    return None


def parse_entry(data: dict[str, Any]) -> UniProtEntry:
    acc = data["primaryAccession"]
    functions = [
        t["value"]
        for c in data.get("comments", [])
        if c.get("commentType") == "FUNCTION"
        for t in c.get("texts", [])
    ]
    xrefs = data.get("uniProtKBCrossReferences", [])
    pfam = [
        PfamFamily(
            pfam_id=x["id"],
            name=next((p["value"] for p in x.get("properties", []) if p.get("key") == "EntryName"), None),
        )
        for x in xrefs
        if x.get("database") == "Pfam"
    ]
    genes = [g["geneName"]["value"] for g in data.get("genes", []) if "geneName" in g]
    organism = data.get("organism", {})
    seq = data.get("sequence", {})
    return UniProtEntry(
        accession=acc,
        entry_name=data.get("uniProtkbId", acc),
        reviewed="Swiss-Prot" in data.get("entryType", ""),
        protein_name=_protein_name(data.get("proteinDescription", {})),
        genes=genes,
        organism=organism.get("scientificName"),
        taxon_id=organism.get("taxonId"),
        length=seq.get("length", len(seq.get("value", ""))),
        sequence=seq.get("value", ""),
        function=" ".join(functions) or None,
        pfam=pfam,
        emdb_cross_refs=[x["id"] for x in xrefs if x.get("database") == "EMDB"],
        url=f"https://www.uniprot.org/uniprotkb/{acc}/entry",
    )


async def get_entry(http: HttpClient, accession: str) -> UniProtEntry:
    data = await http.get(ENTRY_URL.format(accession=accession))
    if data.get("entryType") == "Inactive":
        reason = data.get("inactiveReason", {}).get("inactiveReasonType", "inactive")
        merged = data.get("inactiveReason", {}).get("mergeDemergeTo", [])
        hint = f" It was merged into {', '.join(merged)}." if merged else ""
        raise LookupError(f"UniProt entry {accession} is {reason.lower()}.{hint}")
    return parse_entry(data)
