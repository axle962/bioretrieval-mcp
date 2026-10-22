from urllib.parse import parse_qs, unquote, urlparse

import pytest

from bioretrieval_mcp.sources import emdb, geo, uniprot
from tests.conftest import fixture_json, fixture_text

pytestmark = pytest.mark.anyio


# ---- EMDB -------------------------------------------------------------------


def test_emdb_build_query():
    assert emdb.build_query("apoferritin", None) == "(apoferritin)"
    assert emdb.build_query("spike", 3.0) == "(spike) AND resolution:[0 TO 3]"
    assert emdb.build_query("spike", 2.5) == "(spike) AND resolution:[0 TO 2.5]"


def test_emdb_parse_csv():
    entries = {e.emdb_id: e for e in emdb.parse_csv(fixture_text("emdb_search.csv"))}
    e = entries["EMD-11668"]
    assert e.resolution_angstrom == 1.15
    assert e.fitted_pdbs == ["7A6A"]
    assert e.uniprot_accessions == ["P02794"]
    assert e.pfam_names == ["Ferritin"]
    assert e.release_date == "2020-09-02"
    assert e.url == "https://www.ebi.ac.uk/emdb/EMD-11668"
    # quoted multi-valued field
    assert entries["EMD-14438"].organisms == ["Homo sapiens", "Plasmodium falciparum"]
    assert entries["EMD-57470"].fitted_pdbs == []


def test_emdb_parse_tolerates_missing_values():
    text = "emdb_id,title,resolution\nEMD-1,t,\n,skip me,1\n"
    (e,) = emdb.parse_csv(text)
    assert e.resolution_angstrom is None and e.organisms == []


async def test_emdb_search_request(make_http, recorder):
    result = await emdb.search(make_http(recorder), "apoferritin", 2.0, limit=2)
    assert result.returned == 2
    url = urlparse(str(recorder.requests[0].url))
    assert unquote(url.path).endswith("/search/(apoferritin) AND resolution:[0 TO 2]")
    qs = parse_qs(url.query)
    assert qs["wt"] == ["csv"] and qs["rows"] == ["2"]
    assert "xref_UNIPROTKB" in qs["fl"][0]


# ---- UniProt ----------------------------------------------------------------


def test_uniprot_parse_entry():
    e = uniprot.parse_entry(fixture_json("uniprot_P02794.json"))
    assert e.accession == "P02794"
    assert e.entry_name == "FRIH_HUMAN"
    assert e.reviewed is True
    assert e.genes == ["FTH1"]
    assert e.organism == "Homo sapiens" and e.taxon_id == 9606
    assert e.length == len(e.sequence) == 183
    assert e.function.startswith("Stores iron")
    assert [(p.pfam_id, p.name) for p in e.pfam] == [("PF00210", "Ferritin")]
    assert "EMD-11668" in e.emdb_cross_refs


def test_uniprot_unreviewed_and_submission_name():
    data = {
        "primaryAccession": "A0A000",
        "entryType": "UniProtKB unreviewed (TrEMBL)",
        "proteinDescription": {"submissionNames": [{"fullName": {"value": "Putative thing"}}]},
        "sequence": {"value": "MKT", "length": 3},
    }
    e = uniprot.parse_entry(data)
    assert e.reviewed is False and e.protein_name == "Putative thing" and e.pfam == [] and e.function is None


async def test_uniprot_inactive_entry(make_http):
    import httpx

    def handler(request):
        return httpx.Response(
            200,
            json={
                "primaryAccession": "Q00001",
                "entryType": "Inactive",
                "inactiveReason": {"inactiveReasonType": "MERGED", "mergeDemergeTo": ["P12345"]},
            },
        )

    with pytest.raises(LookupError, match="merged.*P12345"):
        await uniprot.get_entry(make_http(handler), "Q00001")


# ---- GEO --------------------------------------------------------------------


@pytest.mark.parametrize(
    "platform,gpls",
    [("27K", ["GPL8490"]), ("450K", ["GPL13534", "GPL16304"]), ("EPIC", ["GPL21145", "GPL23976"]), ("EPICv2", ["GPL33022"])],
)
def test_geo_term(platform, gpls):
    term = geo.build_term("aging", platform)
    assert term.startswith("(aging) AND (") and term.endswith("AND gse[ETYP]")
    for g in gpls:
        assert f"{g}[ACCN]" in term


def test_geo_parse_summary():
    doc = fixture_json("geo_esummary.json")["result"]["200144858"]
    s = geo.parse_summary(doc)
    assert s.accession == "GSE144858"
    assert s.platforms == ["GPL13534"]
    assert s.n_samples == 300
    assert s.organisms == ["Homo sapiens"]
    assert s.submission_date == "2020-11-18"
    assert s.pubmed_ids == ["32745807"]


async def test_geo_search_two_calls(make_http, recorder):
    r = await geo.search(make_http(recorder), "Alzheimer", "450K", 2, api_key="KEY", email="a@b.c", summary_chars=50)
    assert [str(x.url).split("?")[0].rsplit("/", 1)[1] for x in recorder.requests] == ["esearch.fcgi", "esummary.fcgi"]
    q = parse_qs(urlparse(str(recorder.requests[0].url)).query)
    assert q["db"] == ["gds"] and q["api_key"] == ["KEY"] and q["tool"] == ["bioretrieval-mcp"]
    assert r.total_count == 20 and r.returned == 2
    assert all(len(s.summary) <= 54 for s in r.series)


async def test_geo_no_hits_skips_summary(make_http):
    import httpx

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"esearchresult": {"count": "0", "idlist": []}})

    r = await geo.search(make_http(handler), "zzzz", "27K", 5)
    assert r.total_count == 0 and r.series == [] and len(calls) == 1


async def test_geo_error_payload(make_http):
    import httpx

    handler = lambda request: httpx.Response(200, json={"esearchresult": {"ERROR": "bad term"}})  # noqa: E731
    with pytest.raises(ValueError, match="bad term"):
        await geo.search(make_http(handler), "x", "EPIC", 5)
