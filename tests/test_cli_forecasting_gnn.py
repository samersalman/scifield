"""Smoke + sidecar tests for the `scifield forecasting` GNN subcommands (V1-S13).

These tests cover the CLI plumbing ONLY — the real graph build lives in
``test_forecasting_gnn_loader.py``, the trainer in ``test_forecasting_gnn_train.py``,
and the Optuna sweep in ``test_forecasting_gnn_sweep.py``. Here we patch the heavy
compute (``build_year_snapshots`` / ``_load_gnn_snapshots`` / ``train_hgt`` /
``run_sweep``) and exercise the three commands end to end with tmp_path fixtures,
asserting exit codes, that the per-year snapshot counts are echoed, that the
``gnn-train`` checkpoint sidecar (``hgt_train_best.pt.run.json``) is written and
— per the plan's Verification — carries the ``preregistration`` block, that
``--resume`` flows through into ``train_hgt``, and that ``gnn-sweep`` does NOT
itself double-write the sweep's own sidecars.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from typer.testing import CliRunner

import scifield.cli as cli_mod
from scifield.cli import app
from scifield.forecasting.train import TrainResult

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _touch(path: Path, content: bytes = b"x") -> None:
    """Plant a tiny readable file so ``record_run``'s ``_hash_file`` succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _build_synth_cfg(tmp_path: Path) -> DictConfig:
    """A synthetic forecasting config with every key the GNN commands read."""
    return OmegaConf.create(
        {
            "input": {
                "archetypes_parquet": str(tmp_path / "archetypes.parquet"),
                "topics_parquet": str(tmp_path / "topics.parquet"),
                "duckdb_path": str(tmp_path / "papers.duckdb"),
            },
            "output": {
                "features_parquet": str(tmp_path / "forecasting_features.parquet"),
                "metrics_parquet": str(tmp_path / "forecasting_baselines.parquet"),
                "snapshots_dir": str(tmp_path / "gnn_snapshots"),
                "sweep_parquet": str(tmp_path / "forecasting_sweep.parquet"),
                "checkpoint": str(tmp_path / "models" / "hgt_best.pt"),
                "study_db": str(tmp_path / "models" / "hgt_study.db"),
            },
            "split": {
                "train": [1998, 2017],
                "val": [2018, 2020],
                "test": [2021, 2022],
                "horizon": 3,
                "trailing_window": 3,
            },
            "label": {"gamma": 1.5, "v_min": 30, "mode": "multiplicative", "delta": 0.002},
            "noise": {"denominator": "leaf_only", "noise_topic_id": -1},
            "gnn": {
                "conv_type": "hgt",
                "hidden": 64,
                "n_layers": 2,
                "heads": 2,
                "dropout": 0.2,
                "lr": 0.001,
                "epochs": 200,
                "patience": 25,
                "seed": 1729,
                "device": "cpu",
                "pos_weight": "auto",
                "sweep": {
                    "n_trials": 40,
                    "search": {
                        "lr": [1.0e-4, 1.0e-2],
                        "hidden": [32, 64, 128],
                        "n_layers": [1, 2, 3],
                        "dropout": [0.0, 0.5],
                        "heads": [1, 2, 4],
                    },
                },
            },
            "preregistration": {
                "osf_url": "https://doi.org/10.17605/OSF.IO/XP94F",
                "pr_doc": "docs/preregistrations/PR2_forecasting.md",
            },
        }
    )


def _fake_features_df() -> pd.DataFrame:
    """A tiny frame standing in for the materialized features table."""
    return pd.DataFrame(
        {
            "topic_id": [0, 1, 2],
            "origin_year": [2018, 2019, 2020],
            "share_3yr_mean": [0.1, 0.2, 0.3],
            "emergent": [0, 1, 0],
            "forward_share": [0.12, 0.25, 0.18],
        }
    )


def _fake_metas() -> dict[int, dict[str, Any]]:
    """``{t -> meta}`` shaped like ``build_year_snapshots``'s return value."""
    out: dict[int, dict[str, Any]] = {}
    for i, t in enumerate((1998, 1999, 2000)):
        out[t] = {
            "t": t,
            "path": f"/tmp/snapshot_{t}.pt",
            "n_nodes": {
                "Paper": 100 + i,
                "Author": 50,
                "Institution": 10,
                "Journal": 5,
                "Topic": 42,
            },
            "n_edges": {
                "CITES": 200 + i,
                "AUTHORED_BY": 80,
                "AFFILIATED_WITH": 40,
                "PUBLISHED_IN": 30,
                "ASSIGNED_TO": 20,
            },
            "n_future_citation_artifacts": i,
        }
    return out


def _fake_snapshots() -> dict[int, Any]:
    """A stand-in ``{origin_year -> HeteroData}`` mapping (opaque to the CLI)."""
    return {1998: object(), 1999: object(), 2000: object()}


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


def test_forecasting_gnn_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["forecasting", "--help"])
    assert result.exit_code == 0, result.stdout
    for cmd in ("gnn-snapshots", "gnn-train", "gnn-sweep"):
        assert cmd in result.stdout, f"missing {cmd!r} in help output"


# ---------------------------------------------------------------------------
# gnn-snapshots
# ---------------------------------------------------------------------------


def test_gnn_snapshots_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`gnn-snapshots` echoes per-year counts with `build_year_snapshots` patched."""
    cfg = _build_synth_cfg(tmp_path)
    _touch(Path(str(cfg.input.duckdb_path)))
    _touch(Path(str(cfg.input.topics_parquet)))
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    monkeypatch.setattr(
        "scifield.forecasting.gnn.loader.build_year_snapshots",
        lambda cfg_in: _fake_metas(),
    )

    result = runner.invoke(app, ["forecasting", "gnn-snapshots"])
    assert result.exit_code == 0, result.stdout

    # Every built year and its Paper-node count surface in the summary.
    assert "3 years" in result.stdout
    for t in (1998, 1999, 2000):
        assert str(t) in result.stdout
    assert "Paper/Topic=100/42" in result.stdout
    # L5 diagnostic total = 0 + 1 + 2.
    assert "total L5 future-citation-artifacts" in result.stdout
    assert "=3" in result.stdout


def test_gnn_snapshots_errors_on_missing_duckdb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing DuckDB => exit 1 before `build_year_snapshots` is reached."""
    cfg = _build_synth_cfg(tmp_path)
    # Plant topics but NOT the duckdb -> the first existence check fails.
    _touch(Path(str(cfg.input.topics_parquet)))
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    def _boom(*_a: Any, **_k: Any) -> dict[int, dict[str, Any]]:
        raise AssertionError("build_year_snapshots must not run when DuckDB is missing")

    monkeypatch.setattr("scifield.forecasting.gnn.loader.build_year_snapshots", _boom)

    result = runner.invoke(app, ["forecasting", "gnn-snapshots"])
    assert result.exit_code == 1, result.stdout
    assert "papers DuckDB not found" in result.stdout


def test_gnn_snapshots_errors_on_missing_topics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing topics parquet => exit 1 (DuckDB present, topics absent)."""
    cfg = _build_synth_cfg(tmp_path)
    _touch(Path(str(cfg.input.duckdb_path)))
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    def _boom(*_a: Any, **_k: Any) -> dict[int, dict[str, Any]]:
        raise AssertionError("build_year_snapshots must not run when topics are missing")

    monkeypatch.setattr("scifield.forecasting.gnn.loader.build_year_snapshots", _boom)

    result = runner.invoke(app, ["forecasting", "gnn-snapshots"])
    assert result.exit_code == 1, result.stdout
    assert "topics parquet not found" in result.stdout


# ---------------------------------------------------------------------------
# gnn-train
# ---------------------------------------------------------------------------


def test_gnn_train_smoke_writes_best_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`gnn-train` records the best-checkpoint sidecar carrying the prereg block.

    Verification: the CLI loads features, builds snapshots, trains, and stamps a
    ``hgt_train_best.pt.run.json`` sidecar referencing the OSF pre-registration.
    The sweep's own ``hgt_best.pt`` must NOT be the artifact the trainer writes.
    """
    cfg = _build_synth_cfg(tmp_path)
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    features_path = Path(str(cfg.output.features_parquet))
    features_path.parent.mkdir(parents=True, exist_ok=True)
    _fake_features_df().to_parquet(features_path, index=False)

    monkeypatch.setattr(cli_mod, "_load_gnn_snapshots", lambda cfg_in, *, build: _fake_snapshots())

    captured: dict[str, Any] = {}

    def fake_train_hgt(**kwargs: Any) -> TrainResult:
        captured.update(kwargs)
        best_path = Path(kwargs["best_path"])
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_bytes(b"ckpt")  # so record_run's _hash_file(best_path) succeeds
        return TrainResult(
            best_val_auc=0.73,
            best_epoch=11,
            last_epoch=36,
            best_path=best_path,
            history=[],
        )

    monkeypatch.setattr("scifield.forecasting.train.train_hgt", fake_train_hgt)

    result = runner.invoke(app, ["forecasting", "gnn-train"])
    assert result.exit_code == 0, result.stdout

    # TRAIN-specific paths (the sweep's hgt_best.pt is left alone).
    models_dir = Path(str(cfg.output.checkpoint)).parent
    best_path = models_dir / "hgt_train_best.pt"
    assert Path(captured["best_path"]) == best_path
    assert Path(captured["checkpoint_path"]) == models_dir / "hgt_latest.pt"
    assert not Path(str(cfg.output.checkpoint)).exists(), "sweep's hgt_best.pt was clobbered"

    # The best-checkpoint sidecar is written and carries the preregistration block.
    sidecar = Path(str(best_path) + ".run.json")
    assert sidecar.exists(), "hgt_train_best.pt.run.json sidecar was not written"
    payload = json.loads(sidecar.read_text())
    prereg = payload["config"]["preregistration"]
    assert prereg["osf_url"] == "https://doi.org/10.17605/OSF.IO/XP94F"
    assert payload["config"]["best_val_auc"] == 0.73
    assert payload["config"]["resumed"] is False

    # Default invocation does not resume; the trainer got the right epoch budget.
    assert captured["resume"] is False
    assert captured["max_epochs"] == int(cfg.gnn.epochs)
    assert captured["patience"] == int(cfg.gnn.patience)

    assert "best_val_auc=0.7300" in result.stdout
    assert "best_epoch=11" in result.stdout


def test_gnn_train_resume_and_epochs_flow_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--resume` and `--epochs N` reach `train_hgt` (captured via the patch)."""
    cfg = _build_synth_cfg(tmp_path)
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    features_path = Path(str(cfg.output.features_parquet))
    features_path.parent.mkdir(parents=True, exist_ok=True)
    _fake_features_df().to_parquet(features_path, index=False)

    monkeypatch.setattr(cli_mod, "_load_gnn_snapshots", lambda cfg_in, *, build: _fake_snapshots())

    captured: dict[str, Any] = {}

    def fake_train_hgt(**kwargs: Any) -> TrainResult:
        captured.update(kwargs)
        best_path = Path(kwargs["best_path"])
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_bytes(b"ckpt")
        return TrainResult(
            best_val_auc=float("nan"),  # exercise the nan-safe echo path
            best_epoch=-1,
            last_epoch=4,
            best_path=best_path,
            history=[],
        )

    monkeypatch.setattr("scifield.forecasting.train.train_hgt", fake_train_hgt)

    result = runner.invoke(app, ["forecasting", "gnn-train", "--resume", "--epochs", "5"])
    assert result.exit_code == 0, result.stdout

    assert captured["resume"] is True
    assert captured["max_epochs"] == 5  # the override wins over cfg.gnn.epochs

    # The sidecar records the resume flag too.
    best_path = Path(str(cfg.output.checkpoint)).parent / "hgt_train_best.pt"
    payload = json.loads(Path(str(best_path) + ".run.json").read_text())
    assert payload["config"]["resumed"] is True

    assert "best_val_auc=nan" in result.stdout
    assert "resumed=True" in result.stdout


def test_gnn_train_errors_on_missing_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`gnn-train` without a features parquet => exit 1 with a 'run … first' hint."""
    cfg = _build_synth_cfg(tmp_path)
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    def _boom_snap(*_a: Any, **_k: Any) -> dict[int, Any]:
        raise AssertionError("snapshots must not be built when features are missing")

    def _boom_train(**_k: Any) -> TrainResult:
        raise AssertionError("train_hgt must not run when features are missing")

    monkeypatch.setattr(cli_mod, "_load_gnn_snapshots", _boom_snap)
    monkeypatch.setattr("scifield.forecasting.train.train_hgt", _boom_train)

    result = runner.invoke(app, ["forecasting", "gnn-train"])
    assert result.exit_code == 1, result.stdout
    assert "forecasting features parquet not found" in result.stdout
    assert "run `scifield forecasting features` first" in result.stdout


# ---------------------------------------------------------------------------
# gnn-sweep
# ---------------------------------------------------------------------------


def test_gnn_sweep_smoke_no_double_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`gnn-sweep` calls `run_sweep` (which writes everything) and only echoes.

    CRITICAL: the CLI must NOT itself write a second sweep sidecar — ``run_sweep``
    owns the sweep parquet + ``hgt_best.pt`` + both ``record_run`` sidecars. The
    fake ``run_sweep`` writes the parquet + its sidecar; we assert the CLI passes
    the right cfg/inputs through and echoes the summary, and that the CLI did not
    overwrite the sidecar ``run_sweep`` already wrote.
    """
    from scifield.forecasting.sweep import SweepResult

    cfg = _build_synth_cfg(tmp_path)
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    features_path = Path(str(cfg.output.features_parquet))
    features_path.parent.mkdir(parents=True, exist_ok=True)
    _fake_features_df().to_parquet(features_path, index=False)

    monkeypatch.setattr(cli_mod, "_load_gnn_snapshots", lambda cfg_in, *, build: _fake_snapshots())

    captured: dict[str, Any] = {}
    sentinel_sidecar = "written-by-run_sweep"

    def fake_run_sweep(**kwargs: Any) -> SweepResult:
        captured.update(kwargs)
        sweep_path = Path(str(cfg.output.sweep_parquet))
        ckpt_path = Path(str(cfg.output.checkpoint))
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trial_number": [0]}).to_parquet(sweep_path, index=False)
        # Stand in for run_sweep's own record_run sidecar; the CLI must NOT touch it.
        Path(str(sweep_path) + ".run.json").write_text(sentinel_sidecar)
        return SweepResult(
            best_params={"lr": 0.001, "hidden": 64, "n_layers": 2, "dropout": 0.2, "heads": 2},
            best_val_auc=0.71,
            best_val_mape=0.38,
            n_trials=40,
            sweep_parquet_path=sweep_path,
            checkpoint_path=ckpt_path,
            trials=[],
        )

    monkeypatch.setattr("scifield.forecasting.sweep.run_sweep", fake_run_sweep)

    result = runner.invoke(app, ["forecasting", "gnn-sweep", "--n-trials", "7"])
    assert result.exit_code == 0, result.stdout

    # The override + cfg + inputs flowed into run_sweep.
    assert captured["n_trials"] == 7
    assert captured["cfg"] is cfg
    assert captured["inputs"] == {"forecasting_features": features_path}

    # The CLI did NOT overwrite the sidecar run_sweep itself wrote.
    sidecar = Path(str(cfg.output.sweep_parquet) + ".run.json")
    assert sidecar.read_text() == sentinel_sidecar, "CLI double-wrote the sweep sidecar"

    assert "gnn-sweep done" in result.stdout
    assert "best_val_auc=0.7100" in result.stdout
    assert "best_val_mape=0.3800" in result.stdout
    assert "n_trials=40" in result.stdout
    assert "best_params=" in result.stdout


def test_gnn_sweep_errors_on_missing_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`gnn-sweep` without a features parquet => exit 1 before `run_sweep` runs."""
    cfg = _build_synth_cfg(tmp_path)
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    def _boom_snap(*_a: Any, **_k: Any) -> dict[int, Any]:
        raise AssertionError("snapshots must not be built when features are missing")

    def _boom_sweep(**_k: Any) -> Any:
        raise AssertionError("run_sweep must not run when features are missing")

    monkeypatch.setattr(cli_mod, "_load_gnn_snapshots", _boom_snap)
    monkeypatch.setattr("scifield.forecasting.sweep.run_sweep", _boom_sweep)

    result = runner.invoke(app, ["forecasting", "gnn-sweep"])
    assert result.exit_code == 1, result.stdout
    assert "forecasting features parquet not found" in result.stdout
