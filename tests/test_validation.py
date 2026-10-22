import pytest

from bioretrieval_mcp.validation import (
    normalize_accession,
    normalize_emdb_query,
    normalize_free_text,
    normalize_sequence,
)


@pytest.mark.parametrize("acc", ["P02794", "p02794", " Q9Y6K9 ", "A0A024V5I6", "P0DTC2", "P02794-2"])
def test_valid_accessions(acc):
    assert normalize_accession(acc) == acc.strip().upper()


@pytest.mark.parametrize("acc", ["", "12345", "EMD-11668", "P0279", "FRIH_HUMAN", "P02794;DROP"])
def test_invalid_accessions(acc):
    with pytest.raises(ValueError, match="not a valid UniProt accession"):
        normalize_accession(acc)


def test_sequence_strips_fasta_and_whitespace():
    assert normalize_sequence(">sp|P1|X\nmkt ayi\nAKQR*\n") == "MKTAYIAKQR"


@pytest.mark.parametrize(
    "seq,msg",
    [
        ("", "empty"),
        (">header only\n", "empty"),
        ("MKT1AY", "non amino-acid"),
        ("ACGT" * 10, "DNA/RNA"),
        ("M" * 5001, "limit"),
    ],
)
def test_invalid_sequences(seq, msg):
    with pytest.raises(ValueError, match=msg):
        normalize_sequence(seq)


def test_short_acgt_peptide_is_allowed():
    # Short peptides made only of A/C/G/T letters are legitimate protein sequences.
    assert normalize_sequence("GATTACA") == "GATTACA"


@pytest.mark.parametrize("q", ["apoferritin", 'title:"spike protein" AND resolution:[0 TO 3]', "fitted_pdbs:7a6a"])
def test_emdb_query_ok(q):
    assert normalize_emdb_query(q) == q


@pytest.mark.parametrize("q", ["", "   ", "a{b}", "x; drop", 'unbalanced "quote', "a && b", "x" * 501])
def test_emdb_query_rejected(q):
    with pytest.raises(ValueError):
        normalize_emdb_query(q)


def test_free_text_collapses_whitespace():
    assert normalize_free_text("  aging \n blood  ") == "aging blood"
    with pytest.raises(ValueError):
        normalize_free_text(" ")
