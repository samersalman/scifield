"""Tests for the `scifield harvest` Typer subcommand (V1-S03 Task 7)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from scifield.cli import app

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pubmed_xml"

runner = CliRunner()


def _minimal_esearch_xml(pmids: list[str]) -> bytes:
    id_xml = "".join(f"<Id>{pmid}</Id>" for pmid in pmids)
    return (
        f'<?xml version="1.0"?>'
        f"<eSearchResult>"
        f"<Count>{len(pmids)}</Count>"
        f"<RetMax>{len(pmids)}</RetMax>"
        f"<RetStart>0</RetStart>"
        f"<IdList>{id_xml}</IdList>"
        f"</eSearchResult>"
    ).encode()


def test_harvest_help() -> None:
    result = runner.invoke(app, ["harvest", "--help"])
    assert result.exit_code == 0
    for flag in ("--config", "--journal", "--year", "--refresh", "--max-papers-per-bucket"):
        assert flag in result.stdout


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_harvest_dry_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end CLI smoke run with respx-mocked NCBI.

    The CLI loads `conf/corpus/v1.yaml` (whose output paths are relative:
    `data/v1/parquet`, `data/v1/papers.duckdb`, …), so by chdir-ing into
    tmp_path we make those relative paths resolve to a clean tmp tree.
    """
    monkeypatch.chdir(tmp_path)
    # Make sure no API key is read; this forces the no-key rate limit branch.
    monkeypatch.delenv("NCBI_API_KEY", raising=False)

    pmids = ["40000001", "40000002"]
    efetch_body = (FIXTURE_DIR / "efetch_arthroscopy_2024.xml").read_bytes()

    with respx.mock(assert_all_called=False) as router:
        router.get(url__regex=r".*/esearch\.fcgi.*").mock(
            return_value=httpx.Response(200, content=_minimal_esearch_xml(pmids))
        )
        router.post(url__regex=r".*/efetch\.fcgi.*").mock(
            return_value=httpx.Response(200, content=efetch_body)
        )

        result = runner.invoke(
            app,
            [
                "harvest",
                "--config",
                "v1",
                "--journal",
                "arthroscopy",
                "--year",
                "2024",
                "--max-papers-per-bucket",
                "2",
            ],
        )

    assert result.exit_code == 0, result.stdout

    parquet_path = tmp_path / "data" / "v1" / "parquet" / "arthroscopy" / "2024.parquet"
    assert parquet_path.exists()

    sidecar = Path(str(parquet_path) + ".run.json")
    assert sidecar.exists()

    duckdb_path = tmp_path / "data" / "v1" / "papers.duckdb"
    assert duckdb_path.exists()
    duckdb_sidecar = Path(str(duckdb_path) + ".run.json")
    assert duckdb_sidecar.exists()

    manifest_path = tmp_path / "data" / "v1" / "manifests" / "arthroscopy" / "2024.json"
    assert manifest_path.exists()
