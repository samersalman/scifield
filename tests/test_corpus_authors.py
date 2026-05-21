from __future__ import annotations

from scifield.corpus.authors import (
    AuthorsConfig,
    disambiguate_authorships,
    first_initial,
    heuristic_id,
    normalize_last_name,
)


def test_normalize_last_name_handles_accents():
    assert normalize_last_name("Müller") == "muller"
    assert normalize_last_name("O'Brien") == "obrien"
    assert normalize_last_name("VAN DER BERG") == "vanderberg"
    assert normalize_last_name("") == ""
    assert normalize_last_name("José-Maria") == "josemaria"


def test_first_initial():
    assert first_initial("Jane Smith") == "j"
    assert first_initial("") == ""
    assert first_initial("Élise Dubois") == "e"
    assert first_initial("   Bob") == "b"
    # leading non-alpha tokens should be skipped to find first alpha char
    assert first_initial("J. Smith") == "j"


def test_heuristic_id_stable():
    a = heuristic_id("scifield_v1", "smith", "j", "I123")
    b = heuristic_id("scifield_v1", "smith", "j", "I123")
    assert a == b
    # Different salt → different output
    c = heuristic_id("other_salt", "smith", "j", "I123")
    assert c != a
    # Different signal → different output
    d = heuristic_id("scifield_v1", "smith", "j", "I999")
    assert d != a
    # Length is consistent (prefix "H" + 16 hex chars)
    assert len(a) == len(b) == len(c) == len(d)
    assert a.startswith("H")


def test_layer1_openalex_wins_when_no_orcid():
    cfg = AuthorsConfig()
    rows = [
        {
            "pmid": "1",
            "author_position": 1,
            "author_position_label": "first",
            "author_oa_id": "A123",
            "author_orcid": "",
            "author_display_name": "Jane Smith",
            "institutions": [],
        }
    ]
    out = disambiguate_authorships(rows, cfg)
    assert len(out) == 1
    assert out[0]["disambiguation_method"] == "openalex"
    assert out[0]["author_canonical_id"] == "OA:A123"


def test_layer2_orcid_overrides_openalex():
    cfg = AuthorsConfig()
    rows = [
        {
            "pmid": "1",
            "author_position": 1,
            "author_position_label": "first",
            "author_oa_id": "A123",
            "author_orcid": "0000-0001-2345-6789",
            "author_display_name": "Jane Smith",
            "institutions": [],
        }
    ]
    out = disambiguate_authorships(rows, cfg)
    assert out[0]["disambiguation_method"] == "orcid"
    assert out[0]["author_canonical_id"].startswith("ORCID:")
    assert out[0]["author_canonical_id"] == "ORCID:0000-0001-2345-6789"


def test_layer3_heuristic_fallback():
    cfg = AuthorsConfig()
    rows = [
        {
            "pmid": "1",
            "author_position": 1,
            "author_position_label": "first",
            "author_oa_id": "",
            "author_orcid": "",
            "author_display_name": "Jane Smith",
            "institutions": [{"oa_id": "I100", "country_code": "US"}],
        }
    ]
    out = disambiguate_authorships(rows, cfg)
    assert out[0]["disambiguation_method"] == "heuristic"
    assert out[0]["author_canonical_id"].startswith("H:")


def test_heuristic_groups_across_rows():
    cfg = AuthorsConfig()
    rows = [
        {
            "pmid": "1",
            "author_position": 1,
            "author_position_label": "first",
            "author_oa_id": "",
            "author_orcid": "",
            "author_display_name": "Jane Smith",
            "institutions": [{"oa_id": "I1", "country_code": "US"}],
        },
        {
            "pmid": "2",
            "author_position": 1,
            "author_position_label": "first",
            "author_oa_id": "",
            "author_orcid": "",
            "author_display_name": "Jane Smith",
            "institutions": [],
        },
    ]
    out = disambiguate_authorships(rows, cfg)
    assert out[0]["disambiguation_method"] == "heuristic"
    assert out[1]["disambiguation_method"] == "heuristic"
    # Same canonical id because signal is pooled across the group
    assert out[0]["author_canonical_id"] == out[1]["author_canonical_id"]


def test_is_first_is_last_from_label_first():
    cfg = AuthorsConfig()
    rows = [
        {
            "pmid": "1",
            "author_position": 99,
            "author_position_label": "first",
            "author_oa_id": "A1",
            "author_orcid": "",
            "author_display_name": "Jane Smith",
            "institutions": [],
        },
        {
            "pmid": "1",
            "author_position": 1,
            "author_position_label": "last",
            "author_oa_id": "A2",
            "author_orcid": "",
            "author_display_name": "Bob Lee",
            "institutions": [],
        },
    ]
    out = disambiguate_authorships(rows, cfg)
    by_oa = {r["author_oa_id"]: r for r in out}
    assert by_oa["A1"]["is_first"] is True
    assert by_oa["A1"]["is_last"] is False
    assert by_oa["A2"]["is_last"] is True
    assert by_oa["A2"]["is_first"] is False


def test_is_first_is_last_fallback_to_position():
    cfg = AuthorsConfig()
    rows = [
        {
            "pmid": "1",
            "author_position": 1,
            "author_position_label": "",
            "author_oa_id": "A1",
            "author_orcid": "",
            "author_display_name": "Alpha One",
            "institutions": [],
        },
        {
            "pmid": "1",
            "author_position": 2,
            "author_position_label": "",
            "author_oa_id": "A2",
            "author_orcid": "",
            "author_display_name": "Beta Two",
            "institutions": [],
        },
        {
            "pmid": "1",
            "author_position": 3,
            "author_position_label": "",
            "author_oa_id": "A3",
            "author_orcid": "",
            "author_display_name": "Gamma Three",
            "institutions": [],
        },
    ]
    out = disambiguate_authorships(rows, cfg)
    by_oa = {r["author_oa_id"]: r for r in out}
    assert by_oa["A1"]["is_first"] is True
    assert by_oa["A1"]["is_last"] is False
    assert by_oa["A2"]["is_first"] is False
    assert by_oa["A2"]["is_last"] is False
    assert by_oa["A3"]["is_first"] is False
    assert by_oa["A3"]["is_last"] is True


def test_empty_input_returns_empty():
    cfg = AuthorsConfig()
    assert disambiguate_authorships([], cfg) == []
