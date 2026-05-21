"""Tests for the `scifield enrich` Typer subcommand (V1-S04 CLI smoke).

All external HTTP traffic is respx-mocked. The CLI loads ``conf/corpus/v1.yaml``
whose output paths are relative (``data/v1/...``), so we ``monkeypatch.chdir``
into ``tmp_path`` for every test that touches the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import httpx
import pyarrow.parquet as pq
import pytest
import respx
from typer.testing import CliRunner

from scifield.cli import app
from scifield.corpus import build_duckdb, write_bucket_parquet
from scifield.corpus.enrich_store import SEMANTIC_SCHOLAR_SCHEMA
from tests.test_corpus_store import _sample_row

FIXTURE_DIR = Path(__file__).parent / "fixtures"
OPENALEX_FIXTURE = FIXTURE_DIR / "openalex" / "sample_work.json"
ROR_FIXTURE = FIXTURE_DIR / "ror" / "sample_response.json"

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_corpus_duckdb(tmp_path: Path, pmids: list[str]) -> Path:
    """Materialize a minimal V1-S03 papers.duckdb under tmp_path/data/v1.

    Writes one Parquet bucket holding `pmids` rows, then calls build_duckdb
    so the `papers` view exists for `load_pmids_from_corpus`.
    """
    parquet_dir = tmp_path / "data" / "v1" / "parquet"
    duckdb_path = tmp_path / "data" / "v1" / "papers.duckdb"
    rows = [_sample_row(pmid, "arthroscopy", 2024, n_mesh=1) for pmid in pmids]
    write_bucket_parquet(rows, slug="arthroscopy", year=2024, parquet_dir=parquet_dir)
    build_duckdb(parquet_dir=parquet_dir, duckdb_path=duckdb_path)
    return duckdb_path


def _install_mocks(router: respx.MockRouter) -> None:
    """Install respx routes for OpenAlex + ROR (Semantic Scholar is unused)."""
    openalex_payload = OPENALEX_FIXTURE.read_bytes()
    ror_payload = ROR_FIXTURE.read_bytes()

    router.get(url__regex=r"https://api\.openalex\.org/works.*").mock(
        return_value=httpx.Response(
            200,
            content=openalex_payload,
            headers={"Content-Type": "application/json"},
        )
    )
    router.get(url__regex=r"https://api\.ror\.org/organizations.*").mock(
        return_value=httpx.Response(
            200,
            content=ror_payload,
            headers={"Content-Type": "application/json"},
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_enrich_help() -> None:
    result = runner.invoke(app, ["enrich", "--help"])
    assert result.exit_code == 0, result.stdout
    for flag in ("--config", "--only", "--skip", "--limit"):
        assert flag in result.stdout


def test_enrich_loud_failure_no_openalex_email(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing OPENALEX_EMAIL must abort with a loud, actionable message.

    The CLI checks OPENALEX_EMAIL BEFORE the duckdb-exists check (cli.py
    line ~253 vs ~310), so no duckdb seeding is required.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENALEX_EMAIL", raising=False)

    result = runner.invoke(app, ["enrich", "--config", "v1"])
    assert result.exit_code != 0
    assert "OPENALEX_EMAIL" in result.stdout


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_enrich_smoke_all_sources_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full CLI smoke run with every external source respx-mocked."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_EMAIL", "test@example.com")
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)

    pmids = ["40000001", "40000002", "40000003"]
    _seed_corpus_duckdb(tmp_path, pmids)

    with respx.mock(assert_all_called=False) as router:
        _install_mocks(router)
        result = runner.invoke(
            app,
            ["enrich", "--config", "v1", "--limit", "3"],
        )

    assert result.exit_code == 0, result.stdout

    enrichment_dir = tmp_path / "data" / "v1" / "enrichment"
    expected_files_with_sidecar = [
        "openalex_works.parquet",
        "references_out.parquet",
        "authorships.parquet",
        "institutions.parquet",
        "paper_institutions.parquet",
        "semantic_scholar.parquet",
        "citation_intents.parquet",
    ]
    for name in expected_files_with_sidecar:
        parquet_path = enrichment_dir / name
        assert parquet_path.exists(), f"missing {parquet_path}"
        sidecar = Path(str(parquet_path) + ".run.json")
        assert sidecar.exists(), f"missing sidecar {sidecar}"

    # enrichment_failed.parquet is written by OpenAlex; sidecar emitted too.
    failed_path = enrichment_dir / "enrichment_failed.parquet"
    assert failed_path.exists()

    # SS Parquet must exist, have 0 rows, and match the documented schema.
    ss_table = pq.read_table(enrichment_dir / "semantic_scholar.parquet")
    assert ss_table.num_rows == 0
    assert ss_table.schema.equals(SEMANTIC_SCHOLAR_SCHEMA)

    # DuckDB views for every enrichment table that exists on disk.
    duckdb_path = tmp_path / "data" / "v1" / "papers.duckdb"
    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        rows = conn.execute(
            "SELECT view_name FROM duckdb_views() WHERE internal = false"
        ).fetchall()
    finally:
        conn.close()
    view_names = {r[0] for r in rows}
    for expected_view in (
        "openalex_works",
        "authorships",
        "references_out",
        "institutions",
        "paper_institutions",
        "semantic_scholar",
        "citation_intents",
        "enrichment_failed",
    ):
        assert expected_view in view_names, f"missing view {expected_view}; got {view_names}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_enrich_only_openalex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--only openalex must skip Authors, ROR, and Semantic Scholar."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_EMAIL", "test@example.com")
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)

    pmids = ["40000001", "40000002", "40000003"]
    _seed_corpus_duckdb(tmp_path, pmids)

    with respx.mock(assert_all_called=False) as router:
        _install_mocks(router)
        result = runner.invoke(
            app,
            ["enrich", "--config", "v1", "--only", "openalex", "--limit", "3"],
        )

    assert result.exit_code == 0, result.stdout

    enrichment_dir = tmp_path / "data" / "v1" / "enrichment"

    # OpenAlex outputs must exist.
    assert (enrichment_dir / "openalex_works.parquet").exists()
    assert (enrichment_dir / "references_out.parquet").exists()

    # Authors/ROR/SS outputs must NOT exist (or, if they do for some reason,
    # they must be schema-only 0-row tables).
    for missing in (
        "authorships.parquet",
        "institutions.parquet",
        "paper_institutions.parquet",
        "semantic_scholar.parquet",
        "citation_intents.parquet",
    ):
        path = enrichment_dir / missing
        if path.exists():
            table = pq.read_table(path)
            assert table.num_rows == 0, f"{missing} should be absent or empty"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_enrich_skip_ror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--skip ror must not produce institutions.parquet."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_EMAIL", "test@example.com")
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)

    pmids = ["40000001", "40000002", "40000003"]
    _seed_corpus_duckdb(tmp_path, pmids)

    with respx.mock(assert_all_called=False) as router:
        _install_mocks(router)
        result = runner.invoke(
            app,
            ["enrich", "--config", "v1", "--skip", "ror", "--limit", "3"],
        )

    assert result.exit_code == 0, result.stdout

    enrichment_dir = tmp_path / "data" / "v1" / "enrichment"
    assert not (enrichment_dir / "institutions.parquet").exists()
    assert not (enrichment_dir / "paper_institutions.parquet").exists()
    # OpenAlex still runs.
    assert (enrichment_dir / "openalex_works.parquet").exists()
