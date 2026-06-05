"""Smoke + sidecar tests for the `scifield forecasting` GNN subcommands (V1-S13).

These tests cover the CLI plumbing ONLY — the real graph build lives in
``test_forecasting_gnn_loader.py``, the trainer in ``test_forecasting_gnn_train.py``,
and the Optuna sweep in ``test_forecasting_gnn_sweep.py``. Here we patch the heavy
compute (``build_year_snapshots`` / ``_load_gnn_snapshots`` / ``train_hgt`` /
``run_sweep`` / ``materialize`` / ``load_forecaster``) and exercise the four
commands end to end with tmp_path fixtures, asserting exit codes, that the per-year
snapshot counts are echoed, that the ``gnn-train`` checkpoint sidecar
(``hgt_train_best.pt.run.json``) is written and — per the plan's Verification —
carries the ``preregistration`` block, that ``--resume`` flows through into
``train_hgt``, and that ``gnn-sweep`` does NOT itself double-write the sweep's own
sidecars. The V1-S14 ``gnn-eval`` smoke test additionally proves the frozen
checkpoint is loaded (never retrained / clobbered), that ``materialize`` is called
with ``allow_test=True``, and that all five sealed-test artifacts + their sidecars
land with the pre-registered verdict keys.
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
                "test_metrics_parquet": str(tmp_path / "forecasting_test_metrics.parquet"),
                "test_per_unit_parquet": str(tmp_path / "forecasting_test_per_unit.parquet"),
                "test_calibration_parquet": str(tmp_path / "forecasting_test_calibration.parquet"),
                "test_sensitivity_parquet": str(tmp_path / "forecasting_test_sensitivity.parquet"),
                "test_wilcoxon_json": str(tmp_path / "forecasting_test_wilcoxon.json"),
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
            # Mirrors v1.yaml: gnn-eval's _build_predictors reads features.* +
            # baselines.* to fit the four baselines on the TRAIN slice.
            "features": {
                "mlp_columns": list(_NODE_FEATURE_COLS[:9]),
                "node_columns": list(_NODE_FEATURE_COLS),
            },
            "baselines": {
                "naive": {},
                "arima": {"order": [1, 1, 0]},
                "mlp": {"hidden": 8, "epochs": 2, "lr": 0.01, "batch_size": 256, "seed": 1729},
                "no_graph": {"hidden": 8, "epochs": 2, "lr": 0.01, "batch_size": 256, "seed": 1729},
            },
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


# All 16 GNN node features (= MLP_FEATURES + 7 trailing cols) the materialized
# frame carries; the four baselines + the calibration/sensitivity paths read a
# subset, so we plant the full set with finite values.
_NODE_FEATURE_COLS = (
    "count_3yr",
    "share_3yr_mean",
    "share_last",
    "share_growth_3yr",
    "share_momentum",
    "share_accel",
    "share_volatility_3yr",
    "topic_age",
    "share_of_max",
    "sem_nov_mean_3yr",
    "cd5_3yr",
    "cd10_3yr",
    "cited_by_pctile_3yr",
    "rct_share_3yr",
    "review_share_3yr",
    "n_journals_3yr",
)


def _fake_materialized_df() -> pd.DataFrame:
    """A tiny materialized frame WITH train AND test rows for the eval harness.

    The four baselines fit on the TRAIN labeled slice and every model is scored on
    the TEST labeled slice, so we need rows in BOTH splits (all ``volume_ok`` &
    ``label_complete``) and BOTH emergent classes in the test slice (so AUC, the
    paired tests, and calibration are all well defined). All NODE_FEATURES columns
    are present with finite values plus the three label variants + ``forward_share``.
    """
    import numpy as np

    rng = np.random.default_rng(1729)
    rows = []
    # 6 train rows (origins 2015-2017) + 6 test rows (origins 2021-2022), mixed labels.
    plan = [
        ("train", 2015),
        ("train", 2016),
        ("train", 2017),
        ("train", 2015),
        ("train", 2016),
        ("train", 2017),
        ("test", 2021),
        ("test", 2022),
        ("test", 2021),
        ("test", 2022),
        ("test", 2021),
        ("test", 2022),
    ]
    for i, (split, year) in enumerate(plan):
        emergent = i % 2  # guarantees both classes appear in train AND test
        row: dict[str, Any] = {
            "topic_id": i,
            "origin_year": year,
            "split": split,
            "volume_ok": True,
            "label_complete": True,
            "emergent": emergent,
            "emergent_additive": (i // 2) % 2,
            "emergent_count_surge": (i + 1) % 2,
            "forward_share": float(0.05 + 0.01 * i),
        }
        for c in _NODE_FEATURE_COLS:
            row[c] = float(rng.uniform(0.01, 0.99))
        rows.append(row)
    return pd.DataFrame(rows)


class _FakeForecaster:
    """A stand-in restored HGT: scores from a feature column; ``fit`` must NOT run."""

    name = "hgt"

    def predict(self, features: pd.DataFrame) -> Any:
        import numpy as np

        from scifield.forecasting.baselines.base import ForecastPrediction

        n = len(features)
        # Derive a deterministic probability in [0, 1] from a present feature column.
        base = features["share_3yr_mean"].to_numpy(dtype=float) if n else np.empty(0)
        emergence_score = np.clip(base, 0.0, 1.0)
        share_forecast = np.clip(
            features["share_last"].to_numpy(dtype=float) if n else np.empty(0), 0.0, None
        )
        return ForecastPrediction(emergence_score=emergence_score, share_forecast=share_forecast)

    def fit(self, *_a: Any, **_k: Any) -> Any:  # pragma: no cover - must never run
        raise AssertionError("load_forecaster result must NOT be retrained in gnn-eval")


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


def test_forecasting_gnn_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["forecasting", "--help"])
    assert result.exit_code == 0, result.stdout
    for cmd in ("gnn-snapshots", "gnn-train", "gnn-sweep", "gnn-eval"):
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


# ---------------------------------------------------------------------------
# gnn-eval (V1-S14 / Gate G4 — the one controlled sealed-test-set touch)
# ---------------------------------------------------------------------------


def test_gnn_eval_smoke_writes_all_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`gnn-eval` scores the frozen HGT + baselines once and writes 5 artifacts.

    Verification: the checkpoint is LOADED (never retrained — the fake forecaster's
    ``fit`` raises) and byte-UNCHANGED after the run; ``materialize`` is called with
    ``allow_test=True`` (the only way the test slice exists); all five sealed-test
    artifacts and their ``record_run`` sidecars land; the Wilcoxon JSON carries the
    pre-registered verdict keys; and a sidecar references the OSF pre-registration.
    """
    cfg = _build_synth_cfg(tmp_path)
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    # Plant the guarded inputs + the frozen checkpoint (so guards + record_run pass).
    _touch(Path(str(cfg.input.duckdb_path)))
    _touch(Path(str(cfg.input.topics_parquet)))
    _touch(Path(str(cfg.input.archetypes_parquet)))
    checkpoint_path = Path(str(cfg.output.checkpoint))
    _touch(checkpoint_path, content=b"FROZEN-HGT-CHECKPOINT-BYTES")
    checkpoint_bytes_before = checkpoint_path.read_bytes()

    # The real graph build is mocked; assert_snapshot_no_leakage is unit-tested in
    # test_forecasting_gnn_loader.py, so we do NOT build a heavy DuckDB fixture here.
    build_calls: dict[str, Any] = {}

    def fake_build_year_snapshots(cfg_in: Any, *, years: Any = None) -> dict[int, dict[str, Any]]:
        build_calls["years"] = years
        return {int(y): {"path": f"/tmp/snapshot_{y}.pt"} for y in (years or [])}

    monkeypatch.setattr(
        "scifield.forecasting.gnn.loader.build_year_snapshots", fake_build_year_snapshots
    )

    snap_calls: dict[str, Any] = {}

    def fake_load_snapshots(
        cfg_in: Any, *, build: bool, include_test: bool = False
    ) -> dict[int, Any]:
        snap_calls["build"] = build
        snap_calls["include_test"] = include_test
        return _fake_snapshots()

    monkeypatch.setattr(cli_mod, "_load_gnn_snapshots", fake_load_snapshots)

    # materialize MUST be called with allow_test=True — capture its kwargs.
    materialize_calls: dict[str, Any] = {}

    def fake_materialize(cfg_in: Any, *, allow_test: bool = False) -> tuple[pd.DataFrame, dict]:
        materialize_calls["allow_test"] = allow_test
        return _fake_materialized_df(), {}

    monkeypatch.setattr("scifield.forecasting.data.materialize", fake_materialize)

    # load_forecaster returns the frozen, predict-ready forecaster; its fit raises.
    loader_calls: dict[str, Any] = {}

    def fake_load_forecaster(ckpt_path: Any, snapshots: Any) -> _FakeForecaster:
        loader_calls["ckpt_path"] = Path(str(ckpt_path))
        loader_calls["snapshots"] = snapshots
        return _FakeForecaster()

    monkeypatch.setattr("scifield.forecasting.train.load_forecaster", fake_load_forecaster)

    result = runner.invoke(app, ["forecasting", "gnn-eval"])
    assert result.exit_code == 0, result.stdout

    # The sealed-test snapshots were built for the TEST origin years (2021, 2022),
    # then the full range loaded with include_test=True.
    assert build_calls["years"] == [2021, 2022]
    assert snap_calls == {"build": False, "include_test": True}

    # materialize was opened with the S14 escape hatch.
    assert materialize_calls["allow_test"] is True

    # The frozen checkpoint was loaded from the configured path...
    assert loader_calls["ckpt_path"] == checkpoint_path
    # ...and is byte-UNCHANGED afterwards (no retrain / clobber).
    assert checkpoint_path.read_bytes() == checkpoint_bytes_before

    # All five artifacts + their record_run sidecars exist.
    artifacts = {
        "metrics": Path(str(cfg.output.test_metrics_parquet)),
        "per_unit": Path(str(cfg.output.test_per_unit_parquet)),
        "calibration": Path(str(cfg.output.test_calibration_parquet)),
        "sensitivity": Path(str(cfg.output.test_sensitivity_parquet)),
        "wilcoxon": Path(str(cfg.output.test_wilcoxon_json)),
    }
    for name, path in artifacts.items():
        assert path.exists(), f"missing {name} artifact at {path}"
        assert Path(str(path) + ".run.json").exists(), f"missing {name} sidecar"

    # The metrics parquet has the five models in the locked row order + columns.
    metrics = pd.read_parquet(artifacts["metrics"])
    assert list(metrics["model"]) == ["naive", "arima", "mlp", "no_graph", "hgt"]
    assert list(metrics.columns) == [
        "model",
        "emergence_auc",
        "share_mape",
        "n_train",
        "n_test",
        "n_test_pos",
        "fallback_frac",
    ]

    # The per-unit parquet is tidy/long: one block per model, locked columns.
    per_unit = pd.read_parquet(artifacts["per_unit"])
    assert list(per_unit.columns) == [
        "model",
        "topic_id",
        "origin_year",
        "emergence_score",
        "emergent",
        "emergent_additive",
        "emergent_count_surge",
        "share_forecast",
        "forward_share",
    ]
    assert set(per_unit["model"].unique()) == {"naive", "arima", "mlp", "no_graph", "hgt"}

    # Calibration carries the leading model column for hgt + no_graph.
    calib = pd.read_parquet(artifacts["calibration"])
    assert calib.columns[0] == "model"
    assert set(calib["model"].unique()) == {"hgt", "no_graph"}

    # Sensitivity has the three pre-registered label variants.
    sens = pd.read_parquet(artifacts["sensitivity"])
    assert list(sens["variant"]) == ["primary", "additive_jump", "count_surge"]

    # A sidecar references the OSF pre-registration.
    sidecar = json.loads(Path(str(artifacts["metrics"]) + ".run.json").read_text())
    assert (
        sidecar["config"]["preregistration"]["osf_url"] == "https://doi.org/10.17605/OSF.IO/XP94F"
    )
    assert sidecar["config"]["comparator"] == "no_graph"
    assert sidecar["config"]["margin_pp"] == 5.0
    assert sidecar["config"]["alpha"] == 0.05

    # The Wilcoxon JSON has the three top-level keys + the verdict sub-keys.
    wil = json.loads(artifacts["wilcoxon"].read_text())
    assert set(wil) == {"primary_brier", "secondary_raw_score", "verdict"}
    for key in ("overall_pass", "mechanical_recommendation", "h2_direction"):
        assert key in wil["verdict"], f"verdict missing {key!r}"
    assert "pvalue" in wil["primary_brier"]
    assert "pvalue" in wil["secondary_raw_score"]

    # The summary echoes the gate verdict line.
    assert "gnn-eval done" in result.stdout
    assert "GATE G4" in result.stdout
    assert "overall_pass=" in result.stdout


def test_gnn_eval_errors_on_missing_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`gnn-eval` with inputs present but NO frozen checkpoint => exit 1, no build."""
    cfg = _build_synth_cfg(tmp_path)
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    # Inputs present, checkpoint absent -> the checkpoint guard trips last.
    _touch(Path(str(cfg.input.duckdb_path)))
    _touch(Path(str(cfg.input.topics_parquet)))
    _touch(Path(str(cfg.input.archetypes_parquet)))

    def _boom_build(*_a: Any, **_k: Any) -> dict[int, dict[str, Any]]:
        raise AssertionError("build_year_snapshots must not run when the checkpoint is missing")

    def _boom_load(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("load_forecaster must not run when the checkpoint is missing")

    monkeypatch.setattr("scifield.forecasting.gnn.loader.build_year_snapshots", _boom_build)
    monkeypatch.setattr("scifield.forecasting.train.load_forecaster", _boom_load)

    result = runner.invoke(app, ["forecasting", "gnn-eval"])
    assert result.exit_code == 1, result.stdout
    assert "frozen HGT checkpoint not found" in result.stdout
