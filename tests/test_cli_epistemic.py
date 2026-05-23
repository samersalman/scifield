"""Smoke + round-trip tests for the `scifield epistemic` Typer sub-app (V1-S07).

These tests cover the CLI plumbing only — sampling correctness lives in
``test_epistemic_sampling.py``, extractor correctness in
``test_epistemic_extract.py``, etc. We patch the heavy pieces
(`stratified_sample`, `extract_one`) and exercise the four commands end
to end with tmp_path fixtures.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from openpyxl import Workbook
from typer.testing import CliRunner

import scifield.cli as cli_mod
from scifield.cli import app
from scifield.epistemic.labeling import LABELS_HEADER
from scifield.epistemic.schema import EpistemicExtraction, EpistemicLabel

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synth_duckdb(path: Path) -> None:
    """Plant a tiny ``papers`` table sufficient for ``ensure_papers_distinct_view``."""
    con = duckdb.connect(str(path))
    try:
        con.execute(
            """
            CREATE TABLE papers (
                pmid BIGINT,
                journal_slug VARCHAR,
                title VARCHAR,
                abstract VARCHAR,
                year INTEGER,
                fetched_at VARCHAR
            )
            """
        )
        rows = []
        for i in range(5):
            rows.append(
                (
                    1_000_000 + i,
                    "ann_surg",
                    f"title-{i}",
                    "A" * 80 + f" synthetic abstract {i}",
                    2015,
                    "2026-01-01T00:00:00Z",
                )
            )
        con.executemany(
            "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    finally:
        con.close()


def _make_synth_topics(path: Path) -> None:
    df = pd.DataFrame(
        {
            "pmid": [1_000_000 + i for i in range(5)],
            "topic_id": [0, 1, 2, 3, 4],
            "is_noise": [False] * 5,
        }
    )
    df.to_parquet(path, index=False)


def _make_synth_sample_parquet(path: Path, n_rows: int = 3) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "pmid": [1_000_000 + i for i in range(n_rows)],
            "journal": ["ann_surg"] * n_rows,
            "year": [2015] * n_rows,
            "era": ["2010-2019"] * n_rows,
            "topic_id": pd.array([0, 1, 2][:n_rows], dtype="Int64"),
            "title": [f"title-{i}" for i in range(n_rows)],
            "abstract": ["A" * 80 + f" synthetic abstract {i}" for i in range(n_rows)],
        }
    )
    df.to_parquet(path, index=False)
    return df


def _build_synth_cfg(tmp_path: Path) -> DictConfig:
    duckdb_path = tmp_path / "papers.duckdb"
    topics_parquet = tmp_path / "topics.parquet"
    sample_path = tmp_path / "handlabel_sample.parquet"
    handlabel_parquet = tmp_path / "epistemic_handlabel.parquet"
    pilot_path = tmp_path / "epistemic_pilot.parquet"
    pilot_failed_path = tmp_path / "epistemic_pilot_failed.parquet"
    labels_xlsx_dir = tmp_path
    return OmegaConf.create(
        {
            "input": {
                "duckdb_path": str(duckdb_path),
                "topics_parquet": str(topics_parquet),
            },
            "output": {
                "sample_path": str(sample_path),
                "handlabel_parquet": str(handlabel_parquet),
                "pilot_path": str(pilot_path),
                "pilot_failed_path": str(pilot_failed_path),
                "labels_xlsx_dir": str(labels_xlsx_dir),
            },
            "sampling": {
                "n_sample": 3,
                "seed": 42,
                "eras": ["pre2000", "2000-2009", "2010-2019", "2020+"],
                "topic_coverage_min": 1,
            },
            "pilot": {
                "n_pilot": 3,
                "claude_cmd": ["claude", "--print"],
            },
            "model": {"id": "claude-via-claude-code"},
            "prompt": {"version": "v0.1"},
        }
    )


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


def test_help_works() -> None:
    result = runner.invoke(app, ["epistemic", "--help"])
    assert result.exit_code == 0, result.stdout
    for cmd in ("sample", "export-labels", "import-labels", "pilot"):
        assert cmd in result.stdout, f"missing {cmd!r} in help output"


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


def test_sample_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """End-to-end the `sample` command with `stratified_sample` patched.

    The CLI is responsible for: loading config, ensuring the
    `papers_distinct` view exists, calling the sampler, writing parquet,
    and recording the run sidecar. We stub `stratified_sample` so this
    test exercises only the CLI plumbing.
    """
    cfg = _build_synth_cfg(tmp_path)
    _make_synth_duckdb(Path(str(cfg.input.duckdb_path)))
    _make_synth_topics(Path(str(cfg.input.topics_parquet)))

    monkeypatch.setattr(cli_mod, "_load_epistemic_config", lambda name: cfg)

    fake_df = pd.DataFrame(
        {
            "pmid": [1_000_000, 1_000_001, 1_000_002],
            "journal": ["ann_surg", "ann_surg", "ann_surg"],
            "year": [2015, 2015, 2015],
            "era": ["2010-2019", "2010-2019", "2010-2019"],
            "topic_id": pd.array([0, 1, 2], dtype="Int64"),
            "title": ["t0", "t1", "t2"],
            "abstract": ["a0", "a1", "a2"],
        }
    )

    def _fake_stratified_sample(con: Any, sampling_cfg: Any) -> pd.DataFrame:
        return fake_df

    # The CLI body does `from scifield.epistemic.sampling import ... stratified_sample`,
    # so we patch the symbol on that module.
    import scifield.epistemic.sampling as sampling_mod

    monkeypatch.setattr(sampling_mod, "stratified_sample", _fake_stratified_sample)

    result = runner.invoke(
        app,
        ["epistemic", "sample", "--n-sample", "3", "--seed", "42"],
    )
    assert result.exit_code == 0, result.stdout

    sample_path = Path(str(cfg.output.sample_path))
    assert sample_path.exists(), "sample parquet was not written"
    sidecar = Path(str(sample_path) + ".run.json")
    assert sidecar.exists(), "run.json sidecar was not written"
    written = pd.read_parquet(sample_path)
    assert len(written) == 3
    assert set(written.columns) >= {"pmid", "journal", "year", "era", "title", "abstract"}


def test_sample_errors_on_missing_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _build_synth_cfg(tmp_path)
    # Do NOT create the duckdb; only topics.parquet.
    _make_synth_topics(Path(str(cfg.input.topics_parquet)))
    monkeypatch.setattr(cli_mod, "_load_epistemic_config", lambda name: cfg)

    result = runner.invoke(app, ["epistemic", "sample"])
    assert result.exit_code == 1
    assert "papers DuckDB not found" in result.stdout


# ---------------------------------------------------------------------------
# export-labels
# ---------------------------------------------------------------------------


def test_export_labels_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _build_synth_cfg(tmp_path)
    sample_path = Path(str(cfg.output.sample_path))
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    _make_synth_sample_parquet(sample_path, n_rows=3)

    monkeypatch.setattr(cli_mod, "_load_epistemic_config", lambda name: cfg)

    result = runner.invoke(app, ["epistemic", "export-labels", "--rater", "test"])
    assert result.exit_code == 0, result.stdout

    expected = Path(str(cfg.output.labels_xlsx_dir)) / "labels_test.xlsx"
    assert expected.exists(), f"expected xlsx at {expected} was not written"
    assert f"wrote {expected}" in result.stdout


def test_export_labels_errors_on_missing_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _build_synth_cfg(tmp_path)
    # Do NOT create the sample parquet.
    monkeypatch.setattr(cli_mod, "_load_epistemic_config", lambda name: cfg)

    result = runner.invoke(app, ["epistemic", "export-labels", "--rater", "test"])
    assert result.exit_code == 1
    assert "sample parquet not found" in result.stdout


# ---------------------------------------------------------------------------
# import-labels
# ---------------------------------------------------------------------------


def _build_labels_xlsx(path: Path, pmid: int = 1_000_000) -> None:
    """Fabricate a one-row labeled workbook against ``LABELS_HEADER``."""
    wb = Workbook()
    # First sheet becomes Instructions; we just need a Labels sheet.
    wb.active.title = "Instructions"
    ws = wb.create_sheet("Labels")
    ws.append(list(LABELS_HEADER))
    ws.append(
        [
            pmid,
            "ann_surg",
            2015,
            "title-x",
            "abstract-x" * 8,
            "RCT",  # study_design
            120,  # sample_size
            "TRUE",  # has_control
            "positive",  # effect_direction
            "TRUE",  # statistical_claim_present
            "FALSE",  # coi_disclosed_in_abstract
        ]
    )
    wb.save(path)


def test_import_labels_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _build_synth_cfg(tmp_path)
    xlsx_path = tmp_path / "labels_test.xlsx"
    _build_labels_xlsx(xlsx_path, pmid=1_000_000)

    monkeypatch.setattr(cli_mod, "_load_epistemic_config", lambda name: cfg)

    result = runner.invoke(
        app,
        ["epistemic", "import-labels", "--rater", "test", "--file", str(xlsx_path)],
    )
    assert result.exit_code == 0, result.stdout

    parquet_out = Path(str(cfg.output.handlabel_parquet))
    assert parquet_out.exists()
    df = pd.read_parquet(parquet_out)
    # 6 long-form rows per labeled abstract (one per RATER_FILL field).
    assert len(df) == 6
    assert set(df["pmid"].unique()) == {1_000_000}
    assert set(df["rater"].unique()) == {"test"}


def test_import_labels_exits_1_on_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _build_synth_cfg(tmp_path)
    monkeypatch.setattr(cli_mod, "_load_epistemic_config", lambda name: cfg)

    # Workbook with one row that has a bad enum -> import collects an error.
    xlsx_path = tmp_path / "labels_bad.xlsx"
    wb = Workbook()
    wb.active.title = "Instructions"
    ws = wb.create_sheet("Labels")
    ws.append(list(LABELS_HEADER))
    ws.append(
        [
            1_000_000,
            "ann_surg",
            2015,
            "t",
            "abstract" * 8,
            "NOT_A_VALID_DESIGN",  # bad enum
            120,
            "TRUE",
            "positive",
            "TRUE",
            "FALSE",
        ]
    )
    wb.save(xlsx_path)

    result = runner.invoke(
        app,
        ["epistemic", "import-labels", "--rater", "test", "--file", str(xlsx_path)],
    )
    assert result.exit_code == 1, result.stdout
    assert "n_errors=1" in result.stdout


# ---------------------------------------------------------------------------
# pilot
# ---------------------------------------------------------------------------


def _install_fake_pilot_module(
    monkeypatch: pytest.MonkeyPatch,
    canned: EpistemicExtraction,
) -> None:
    """Install a minimal `scifield.epistemic.pilot` module for the CLI to import.

    Batch 4A owns the real implementation; this stub is just enough for
    the CLI command body to do its lazy import and run.
    """
    from dataclasses import dataclass

    fake = types.ModuleType("scifield.epistemic.pilot")

    @dataclass(frozen=True)
    class PilotConfig:  # noqa: D401 - stub
        sample_path: Path
        pilot_path: Path
        pilot_failed_path: Path
        n_pilot: int
        extract_cfg: Any

    def run_pilot(cfg: PilotConfig, progress: Any = None) -> dict:
        df = pd.read_parquet(cfg.sample_path)
        n = min(int(cfg.n_pilot), len(df))
        ok = 0
        for i in range(n):
            if progress is not None:
                progress(i + 1, n, "ok")
            ok += 1
        # Write a minimal pilot parquet so the assertion below has
        # something to find.
        pd.DataFrame(
            {
                "pmid": df["pmid"].iloc[:n].tolist(),
                "model_id": [canned.model_id] * n,
                "prompt_version": [canned.prompt_version] * n,
            }
        ).to_parquet(cfg.pilot_path, index=False)
        return {
            "n_ok": ok,
            "n_failed": 0,
            "pilot_path": str(cfg.pilot_path),
            "pilot_failed_path": str(cfg.pilot_failed_path),
        }

    fake.PilotConfig = PilotConfig  # type: ignore[attr-defined]
    fake.run_pilot = run_pilot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scifield.epistemic.pilot", fake)


def test_pilot_runs_with_mocked_extract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _build_synth_cfg(tmp_path)
    sample_path = Path(str(cfg.output.sample_path))
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    _make_synth_sample_parquet(sample_path, n_rows=3)

    canned = EpistemicExtraction(
        pmid=1_000_000,
        label=EpistemicLabel(
            study_design="RCT",
            sample_size=120,
            has_control=True,
            effect_direction="positive",
            statistical_claim_present=True,
            coi_disclosed_in_abstract=False,
        ),
        model_id="claude-via-claude-code",
        prompt_version="v0.1",
        raw_response='{"study_design":"RCT"}',
    )

    _install_fake_pilot_module(monkeypatch, canned)
    monkeypatch.setattr(cli_mod, "_load_epistemic_config", lambda name: cfg)

    result = runner.invoke(app, ["epistemic", "pilot", "--n", "3"])
    assert result.exit_code == 0, result.stdout

    pilot_path = Path(str(cfg.output.pilot_path))
    assert pilot_path.exists(), "pilot parquet was not written"
    df = pd.read_parquet(pilot_path)
    assert len(df) == 3
    assert "pilot done" in result.stdout
