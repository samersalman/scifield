"""Smoke + sidecar tests for the `scifield forecasting` Typer sub-app (V1-S12).

These tests cover the CLI plumbing only — the leakage-safe feature
``materialize`` lives in ``test_forecasting_data.py`` and the baseline
fit/score correctness in ``test_forecasting_baselines.py``. Here we patch
the heavy compute (``materialize``, ``run_baselines``, ``per_topic_scores``)
and exercise the two commands end to end with tmp_path fixtures, asserting
that parquet + ``.run.json`` sidecars are written, exit codes are right, and
— per plan Verification #6 — the baseline sidecar carries the
``preregistration`` block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from typer.testing import CliRunner

import scifield.cli as cli_mod
from scifield.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _touch(path: Path, content: bytes = b"x") -> None:
    """Plant a tiny readable file so ``record_run``'s ``_hash_file`` succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _build_synth_cfg(tmp_path: Path) -> DictConfig:
    """A synthetic forecasting config with every key the commands read."""
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
            },
            "split": {
                "train": [1998, 2017],
                "val": [2018, 2020],
                "test": [2021, 2022],
                "horizon": 3,
                "trailing_window": 3,
            },
            "label": {
                "gamma": 1.5,
                "v_min": 30,
                "mode": "multiplicative",
                "delta": 0.002,
            },
            "noise": {
                "denominator": "leaf_only",
                "noise_topic_id": -1,
            },
            "features": {
                "mlp_columns": [
                    "count_3yr",
                    "share_3yr_mean",
                    "share_last",
                    "share_growth_3yr",
                    "share_momentum",
                    "share_accel",
                    "share_volatility_3yr",
                    "topic_age",
                    "share_of_max",
                ],
                "node_columns": [
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
                ],
            },
            "baselines": {
                "naive": {},
                "arima": {"order": [1, 1, 0]},
                "mlp": {
                    "hidden": 4,
                    "epochs": 1,
                    "lr": 0.01,
                    "batch_size": 8,
                    "seed": 1729,
                },
                "no_graph": {
                    "hidden": 4,
                    "epochs": 1,
                    "lr": 0.01,
                    "batch_size": 8,
                    "seed": 1729,
                },
            },
            "preregistration": {
                "osf_url": "PENDING_OSF_SUBMISSION",
                "pr_doc": "docs/preregistrations/PR2_forecasting.md",
            },
        }
    )


def _make_feature_inputs(cfg: DictConfig) -> None:
    """Plant the three input files the `features` command stats + hashes."""
    _touch(Path(str(cfg.input.archetypes_parquet)))
    _touch(Path(str(cfg.input.topics_parquet)))
    _touch(Path(str(cfg.input.duckdb_path)))


BASELINE_NAMES = ("naive", "arima", "mlp", "no_graph")


def _fake_info() -> dict[str, Any]:
    """An info dict shaped like ``materialize``'s second return value.

    numpy scalars are used deliberately to exercise the CLI's ``_to_jsonable``
    coercion before ``record_run`` json-dumps the config.
    """
    return {
        "per_split": {
            "train": {"n": 3, "positive_rate": np.float64(0.33)},
            "val": {"n": 2, "positive_rate": np.float64(0.5)},
        },
        "n_rows": 5,
        "n_labeled": 5,
        "n_excluded_volume": 1,
        "n_noise_total": 12,
        "noise_frac": np.float64(0.24),
        "denominator": "leaf_only",
        "thresholds": {"gamma": 1.5, "v_min": 30, "horizon": 3},
    }


def _fake_features_df() -> pd.DataFrame:
    """A tiny frame standing in for the materialized features table."""
    return pd.DataFrame(
        {
            "topic_id": [0, 1, 2],
            "origin_year": [2018, 2019, 2020],
            "share_3yr_mean": [0.1, 0.2, 0.3],
            "emergent": [0, 1, 0],
        }
    )


def _fake_metrics_df() -> pd.DataFrame:
    """A metrics table with the LOCKED columns, one row per baseline."""
    return pd.DataFrame(
        {
            "baseline": list(BASELINE_NAMES),
            "emergence_auc": [0.6, 0.55, 0.7, 0.72],
            "share_mape": [0.4, 0.42, 0.38, 0.39],
            "n_train": [3, 3, 3, 3],
            "n_val": [2, 2, 2, 2],
            "n_val_pos": [1, 1, 1, 1],
            # arima carries a real fallback_frac; the others are NaN by contract.
            "fallback_frac": [float("nan"), 0.25, float("nan"), float("nan")],
        }
    )


def _fake_per_topic() -> dict[str, dict[str, np.ndarray]]:
    """Per-baseline staged arrays matching the shape the command consumes."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in BASELINE_NAMES:
        out[name] = {
            "topic_id": np.array([0, 1], dtype=np.int64),
            "origin_year": np.array([2018, 2019], dtype=np.int64),
            "emergence_score": np.array([0.3, 0.8], dtype=np.float64),
            "emergent": np.array([0, 1], dtype=np.int64),
            "share_forecast": np.array([0.11, 0.22], dtype=np.float64),
            "forward_share": np.array([0.12, 0.25], dtype=np.float64),
        }
    return out


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


def test_forecasting_help_works() -> None:
    result = runner.invoke(app, ["forecasting", "--help"])
    assert result.exit_code == 0, result.stdout
    for cmd in ("features", "baselines"):
        assert cmd in result.stdout, f"missing {cmd!r} in help output"


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def test_features_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`features` writes the parquet + sidecar with `materialize` patched.

    The CLI is responsible for: loading config, checking inputs exist, calling
    `materialize`, writing the features parquet, and recording the run sidecar
    (folding the preregistration block in). We stub `materialize` so this test
    exercises only the CLI plumbing.
    """
    cfg = _build_synth_cfg(tmp_path)
    _make_feature_inputs(cfg)
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    captured: dict[str, Any] = {}

    def fake_materialize(cfg_in: Any, *, allow_test: bool = False) -> tuple[pd.DataFrame, dict]:
        captured["allow_test"] = allow_test
        return _fake_features_df(), _fake_info()

    monkeypatch.setattr("scifield.forecasting.data.materialize", fake_materialize)

    result = runner.invoke(app, ["forecasting", "features"])
    assert result.exit_code == 0, result.stdout

    # S12 must never materialize test-origin rows.
    assert captured["allow_test"] is False

    out_path = Path(str(cfg.output.features_parquet))
    assert out_path.exists(), "features parquet was not written"
    sidecar = Path(str(out_path) + ".run.json")
    assert sidecar.exists(), "features run.json sidecar was not written"

    written = pd.read_parquet(out_path)
    assert len(written) == 3
    # The echo surfaces class balance / positive rate for Samer to sanity-check.
    assert "class_balance" in result.stdout
    assert "pos_rate=" in result.stdout

    # The features sidecar also folds in the preregistration block.
    payload = json.loads(sidecar.read_text())
    assert payload["config"]["preregistration"]["osf_url"] == "PENDING_OSF_SUBMISSION"


def test_features_errors_on_missing_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A missing input parquet => exit 1 before `materialize` is ever reached."""
    cfg = _build_synth_cfg(tmp_path)
    # Plant topics + duckdb but NOT archetypes -> the first existence check fails.
    _touch(Path(str(cfg.input.topics_parquet)))
    _touch(Path(str(cfg.input.duckdb_path)))
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    def _boom(*_a: Any, **_k: Any) -> tuple[pd.DataFrame, dict]:
        raise AssertionError("materialize should not be reached when an input is missing")

    monkeypatch.setattr("scifield.forecasting.data.materialize", _boom)

    result = runner.invoke(app, ["forecasting", "features"])
    assert result.exit_code == 1, result.stdout
    assert "archetypes parquet not found" in result.stdout


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def test_baselines_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`baselines` writes metrics + per-topic companions and stamps the OSF block.

    CRITICAL (plan Verification #6): the metrics sidecar's
    ``config.preregistration.osf_url`` must be ``PENDING_OSF_SUBMISSION``.
    """
    cfg = _build_synth_cfg(tmp_path)
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    # The command reads the features parquet with pandas before doing anything,
    # so it must exist and be loadable.
    features_path = Path(str(cfg.output.features_parquet))
    features_path.parent.mkdir(parents=True, exist_ok=True)
    _fake_features_df().to_parquet(features_path, index=False)

    monkeypatch.setattr(
        "scifield.forecasting.baselines.evaluate.run_baselines",
        lambda df, cfg_in: _fake_metrics_df(),
    )
    monkeypatch.setattr(
        "scifield.forecasting.baselines.evaluate.per_topic_scores",
        lambda df, cfg_in: _fake_per_topic(),
    )

    result = runner.invoke(app, ["forecasting", "baselines"])
    assert result.exit_code == 0, result.stdout

    metrics_path = Path(str(cfg.output.metrics_parquet))
    assert metrics_path.exists(), "metrics parquet was not written"
    metrics_sidecar = Path(str(metrics_path) + ".run.json")
    assert metrics_sidecar.exists(), "metrics run.json sidecar was not written"

    # Per-topic companion lives next to the metrics parquet, with its own sidecar.
    per_topic_path = metrics_path.parent / "forecasting_per_topic_scores.parquet"
    assert per_topic_path.exists(), "per-topic companion parquet was not written"
    assert Path(
        str(per_topic_path) + ".run.json"
    ).exists(), "per-topic run.json sidecar was not written"

    # Long-form conversion: 4 baselines x 2 staged rows each = 8 rows.
    per_topic_df = pd.read_parquet(per_topic_path)
    assert len(per_topic_df) == 8
    assert set(per_topic_df["baseline"].unique()) == set(BASELINE_NAMES)

    # Verification #6 — the preregistration block lands in the metrics sidecar.
    payload = json.loads(metrics_sidecar.read_text())
    assert payload["config"]["preregistration"]["osf_url"] == "PENDING_OSF_SUBMISSION"


def test_baselines_errors_on_missing_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`baselines` without a features parquet => exit 1 with a 'run … first' hint."""
    cfg = _build_synth_cfg(tmp_path)
    monkeypatch.setattr(cli_mod, "_load_forecasting_config", lambda name: cfg)

    # Do NOT write cfg.output.features_parquet.
    def _boom(*_a: Any, **_k: Any) -> pd.DataFrame:
        raise AssertionError("run_baselines should not run when features are missing")

    monkeypatch.setattr("scifield.forecasting.baselines.evaluate.run_baselines", _boom)

    result = runner.invoke(app, ["forecasting", "baselines"])
    assert result.exit_code == 1, result.stdout
    assert "forecasting features parquet not found" in result.stdout
    assert "run `scifield forecasting features` first" in result.stdout
