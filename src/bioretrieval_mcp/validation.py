"""Input validation shared by tools. Raise ``ValueError`` with an agent-readable message."""

from __future__ import annotations

import re

# UniProt accession format, from https://www.uniprot.org/help/accession_numbers
UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?$"
)
# 20 canonical amino acids plus the ambiguity / rare codes ESM-2's alphabet accepts.
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
ESM_MAX_RESIDUES = 1022  # ESM-2 context is 1024 tokens including BOS/EOS.
MAX_SEQUENCE_LEN = 5000
# Lucene/Solr syntax we allow through to EMDB: words, quotes, field:value, ranges, boolean ops.
_SOLR_FORBIDDEN = re.compile(r"[{}\\;<>]|&&|\|\|")


def normalize_accession(accession: str) -> str:
    acc = accession.strip().upper()
    if not UNIPROT_ACCESSION_RE.match(acc):
        raise ValueError(
            f"'{accession}' is not a valid UniProt accession (expected e.g. P02794, Q9Y6K9, A0A024V5I6)."
        )
    return acc


def normalize_sequence(sequence: str) -> str:
    """Strip FASTA header/whitespace, uppercase, and check the amino-acid alphabet."""
    lines = [ln for ln in sequence.strip().splitlines() if not ln.startswith(">")]
    seq = re.sub(r"\s+", "", "".join(lines)).upper().rstrip("*")
    if not seq:
        raise ValueError("Sequence is empty.")
    if len(seq) > MAX_SEQUENCE_LEN:
        raise ValueError(f"Sequence has {len(seq)} residues; the limit is {MAX_SEQUENCE_LEN}.")
    bad = sorted(set(seq) - AMINO_ACIDS)
    if bad:
        raise ValueError(f"Sequence contains non amino-acid characters: {''.join(bad)}.")
    if set(seq) <= set("ACGTUN") and len(seq) >= 30:
        raise ValueError("Sequence looks like DNA/RNA (only A/C/G/T/U/N). Provide a protein sequence.")
    return seq


def normalize_emdb_query(query: str) -> str:
    q = query.strip()
    if not q:
        raise ValueError("Query is empty.")
    if len(q) > 500:
        raise ValueError("Query is longer than 500 characters.")
    if _SOLR_FORBIDDEN.search(q):
        raise ValueError("Query contains unsupported characters ({ } \\ ; < > && ||). Use AND / OR / NOT.")
    if q.count('"') % 2:
        raise ValueError("Query has an unbalanced double quote.")
    return q


def normalize_free_text(query: str, *, max_len: int = 300) -> str:
    q = " ".join(query.split())
    if not q:
        raise ValueError("Query is empty.")
    if len(q) > max_len:
        raise ValueError(f"Query is longer than {max_len} characters.")
    return q
