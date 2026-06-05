"""Unit tests for the V1-S13 Optuna HGT hyperparameter sweep.

Hermetic and fast: tiny synthetic snapshots are assembled with
:func:`scifield.forecasting.gnn.loader.build_snapshot_from_frames` (no DuckDB, no
real corpus, no network), a small ``(features, labels)`` panel spans a couple of
TRAIN years plus a VAL year with BOTH classes present (so emergence AUC is
defined), and the sweep runs with a TINY budget (``n_trials=2``, 2 epochs,
``patience=2``) over single-option search spaces. ``tmp_path`` backs every
artifact (study sqlite, sweep parquet, checkpoint).

The five behaviours under test (see the HANDOFF NOTES in ``sweep.py``):

1. **Sweep runs end-to-end** — :func:`run_sweep` completes and returns a
   :class:`SweepResult` with a finite ``best_val_auc``.
2. **Sweep parquet shape** — ``forecasting_sweep.parquet`` exists with exactly
   ``n_trials`` rows and the required columns.
3. **Best checkpoint persisted + loadable** — ``hgt_best.pt`` exists and
   :func:`load_forecaster` rebuilds a predict-ready forecaster from it whose VAL
   predictions are finite and aligned to the VAL rows.
4. **Sidecars reference the OSF DOI** — both ``*.run.json`` sidecars contain the
   pre-registration DOI.
5. **Study resumes (``load_if_exists``)** — running ``n_trials=1`` then again
   with ``n_trials=2`` against the SAME db yields 2 total completed trials (the
   second run added only the 1 missing trial, it did not restart).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch_geometric")

import optuna  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from scifield.forecasting.data import NODE_FEATURES  # noqa: E402
from scifield.forecasting.gnn.loader import build_snapshot_from_frames  # noqa: E402
from scifield.forecasting.sweep import SweepResult, run_sweep  # noqa: E402
from scifield.forecasting.train import load_checkpoint, load_forecaster  # noqa: E402

torch.set_num_threads(1)

#: OSF DOI from the pre-registration (must surface in both sidecars).
_OSF_DOI = "10.17605/OSF.IO/XP94F"

#: Topic ids shared by every synthetic snapshot (5/8 get papers; 12 never does).
_TOPIC_IDS = [5, 8, 12]

#: Two TRAIN years + one VAL year for the panel.
_TRAIN_YEARS = (2001, 2002)
_VAL_YEAR = 2003
_ALL_YEARS = (*_TRAIN_YEARS, _VAL_YEAR)


def _corpus_frames() -> dict[str, pd.DataFrame]:
    """Tiny full-corpus frames shaped like the canonical kuzu queries' output."""
    papers = pd.DataFrame(
        {"pmid": ["p1", "p2", "p3", "p4", "p5"], "year": [2000, 2001, 2002, 2003, 2004]}
    )
    topics = pd.DataFrame({"topic_id": _TOPIC_IDS})
    cites = pd.DataFrame(
        {"from_pmid": ["p2", "p3", "p4", "p5"], "to_pmid": ["p1", "p1", "p2", "p4"]}
    )
    authored_by = pd.DataFrame(
        {
            "pmid": ["p1", "p2", "p3", "p4", "p5"],
            "author_canonical_id": ["a1", "a1", "a2", "a2", "a2"],
        }
    )
    affiliated_with = pd.DataFrame(
        {"author_canonical_id": ["a1", "a2"], "institution_canonical_id": ["i1", "i2"]}
    )
    published_in = pd.DataFrame(
        {"pmid": ["p1", "p2", "p3", "p4", "p5"], "journal_slug": ["j1", "j1", "j2", "j2", "j2"]}
    )
    assigned_to = pd.DataFrame(
        {"pmid": ["p1", "p2", "p3", "p4", "p5"], "topic_id": [5, 5, 8, 8, 8]}
    )
    return {
        "papers": papers,
        "topics": topics,
        "cites": cites,
        "authored_by": authored_by,
        "affiliated_with": affiliated_with,
        "published_in": published_in,
        "assigned_to": assigned_to,
    }


def _snapshots(years: tuple[int, ...] = _ALL_YEARS) -> dict[int, object]:
    """Build forward-edge-only snapshots for ``years`` from the synthetic corpus."""
    frames = _corpus_frames()
    return {t: build_snapshot_from_frames(t, **frames)[0] for t in years}


def _panel(*, seed: int = 0) -> pd.DataFrame:
    """Materialized-style frame: TRAIN years + a VAL year, all masks set.

    One row per ``(topic_id in _TOPIC_IDS, origin_year)`` with the 16 NODE_FEATURES
    filled reproducibly, the ``split`` / ``volume_ok`` / ``label_complete`` masks,
    plus ``emergent`` / ``forward_share`` supervision. The label alternates within
    each year so neither split is single-class (``emergence_auc`` stays defined).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for year in _ALL_YEARS:
        split = "val" if year == _VAL_YEAR else "train"
        for k, gid in enumerate(_TOPIC_IDS):
            row: dict[str, object] = {
                "topic_id": gid,
                "origin_year": year,
                "split": split,
                "volume_ok": True,
                "label_complete": True,
                "emergent": int(k % 2),
                "forward_share": 0.05 + 0.01 * k,
            }
            for col in NODE_FEATURES:
                row[col] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


def _cfg(tmp_path: Path) -> DictConfig:
    """A TINY forecasting-shaped config: SAGE conv, 2 epochs, 1-option search.

    Single-option search spaces keep every trial's sampled config identical and
    valid (``heads=2`` divides ``hidden=8``), so the only variation is the noise
    of the fit — exactly what we want for a fast, deterministic sweep test.
    Output paths point under ``tmp_path``.
    """
    return OmegaConf.create(
        {
            "gnn": {
                "conv_type": "sage",
                "hidden": 8,
                "n_layers": 2,
                "heads": 2,
                "dropout": 0.2,
                "lr": 0.01,
                "epochs": 2,
                "patience": 2,
                "seed": 1729,
                "device": "cpu",
                "pos_weight": "auto",
                "sweep": {
                    "n_trials": 2,
                    "search": {
                        "lr": [1.0e-3, 1.0e-2],
                        "hidden": [8],
                        "n_layers": [2],
                        "dropout": [0.0, 0.3],
                        "heads": [2],
                    },
                },
            },
            "output": {
                "sweep_parquet": str(tmp_path / "forecasting_sweep.parquet"),
                "checkpoint": str(tmp_path / "hgt_best.pt"),
                "study_db": str(tmp_path / "hgt_study.db"),
            },
            "preregistration": {
                "osf_url": f"https://doi.org/{_OSF_DOI}",
                "pr_doc": "docs/preregistrations/PR2_forecasting.md",
            },
        }
    )


# --------------------------------------------------------------------------- #
# (1) Sweep runs end-to-end -> finite best_val_auc
# --------------------------------------------------------------------------- #


def test_sweep_runs_end_to_end(tmp_path: Path) -> None:
    features = _panel()
    result = run_sweep(
        features=features,
        labels=features[["emergent", "forward_share"]],
        snapshots=_snapshots(),
        cfg=_cfg(tmp_path),
        n_trials=2,
    )
    assert isinstance(result, SweepResult)
    assert result.n_trials == 2
    assert np.isfinite(result.best_val_auc)
    # best_params carries the winning forecaster kwargs (the searched + fixed set).
    assert result.best_params["conv_type"] == "sage"
    assert result.best_params["hidden"] == 8
    assert result.best_params["seed"] == 1729


# --------------------------------------------------------------------------- #
# (2) Sweep parquet: exactly n_trials rows + required columns
# --------------------------------------------------------------------------- #


def test_sweep_parquet_shape(tmp_path: Path) -> None:
    features = _panel()
    result = run_sweep(
        features=features,
        labels=features[["emergent", "forward_share"]],
        snapshots=_snapshots(),
        cfg=_cfg(tmp_path),
        n_trials=2,
    )
    assert result.sweep_parquet_path.exists()
    df = pd.read_parquet(result.sweep_parquet_path)
    assert len(df) == 2  # one row per trial
    required = {
        "trial_number",
        "lr",
        "hidden",
        "n_layers",
        "dropout",
        "heads",
        "conv_type",
        "seed",
        "val_auc",
        "val_mape",
        "best_epoch",
        "n_epochs_run",
        "state",
    }
    assert required.issubset(df.columns)
    # The searched params landed in the (single-option) ranges.
    assert set(df["hidden"]) == {8}
    assert set(df["heads"]) == {2}
    assert df["lr"].between(1.0e-3, 1.0e-2).all()


# --------------------------------------------------------------------------- #
# (3) Best checkpoint persisted + loadable into a fresh predict
# --------------------------------------------------------------------------- #


def test_best_checkpoint_persisted_and_loadable(tmp_path: Path) -> None:
    features = _panel()
    result = run_sweep(
        features=features,
        labels=features[["emergent", "forward_share"]],
        snapshots=_snapshots(),
        cfg=_cfg(tmp_path),
        n_trials=2,
    )
    assert result.checkpoint_path.exists()

    # Rebuild a FRESH forecaster from hgt_best.pt (no training data) and predict
    # the VAL slice; the prediction must align 1:1 with the VAL rows and be finite.
    val = features[features["split"] == "val"].reset_index(drop=True)
    fc = load_forecaster(result.checkpoint_path, _snapshots())
    assert fc.net_ is not None and fc.pipeline_ is not None
    pred = fc.predict(val)
    assert pred.emergence_score.shape == (len(val),)
    assert np.all(np.isfinite(pred.emergence_score))
    assert np.all(pred.share_forecast >= 0.0)


# --------------------------------------------------------------------------- #
# (4) Both sidecars reference the OSF DOI
# --------------------------------------------------------------------------- #


def test_sidecars_reference_osf_doi(tmp_path: Path) -> None:
    features = _panel()
    result = run_sweep(
        features=features,
        labels=features[["emergent", "forward_share"]],
        snapshots=_snapshots(),
        cfg=_cfg(tmp_path),
        n_trials=2,
        inputs={},
    )
    for artifact in (result.sweep_parquet_path, result.checkpoint_path):
        sidecar = artifact.with_name(artifact.name + ".run.json")
        assert sidecar.exists(), f"missing sidecar for {artifact}"
        text = sidecar.read_text()
        assert _OSF_DOI in text, f"OSF DOI not found in {sidecar}"
        # And it is well-formed JSON carrying the prereg block + sweep summary.
        payload = json.loads(text)
        assert "preregistration" in payload["config"]
        assert "best_params" in payload["config"]


# --------------------------------------------------------------------------- #
# (5) Study resumes via load_if_exists (1 trial, then 2 -> 2 total)
# --------------------------------------------------------------------------- #


def test_study_resumes_load_if_exists(tmp_path: Path) -> None:
    features = _panel()
    cfg = _cfg(tmp_path)
    snaps = _snapshots()

    # First run: a single trial lands in the sqlite db.
    first = run_sweep(
        features=features,
        labels=features[["emergent", "forward_share"]],
        snapshots=snaps,
        cfg=cfg,
        n_trials=1,
    )
    assert first.n_trials == 1

    # Second run against the SAME study_db with a target of 2: it must ADD only
    # the 1 missing trial (resume), not restart -> 2 total completed trials.
    second = run_sweep(
        features=features,
        labels=features[["emergent", "forward_share"]],
        snapshots=snaps,
        cfg=cfg,
        n_trials=2,
    )
    assert second.n_trials == 2

    # Cross-check directly against the persisted study.
    study = optuna.load_study(study_name="hgt_v1s13", storage=f"sqlite:///{cfg.output.study_db}")
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert n_complete == 2


# --------------------------------------------------------------------------- #
# (6) hgt_best.pt holds the BEST-val-AUC weights, not the early-stop terminal
# --------------------------------------------------------------------------- #


def test_best_checkpoint_holds_best_epoch_not_terminal(tmp_path: Path) -> None:
    """Regression guard: the retrain must persist the *val-selected* weights.

    ``train_hgt`` writes its ``checkpoint_path`` every epoch but ``best_path`` only
    on val-AUC improvement. If the sweep pointed BOTH at ``hgt_best.pt`` the terminal
    (early-stop) epoch would overwrite the best one — leaving non-val-selected
    weights for V1-S14's sealed test eval. The retrain therefore sends the per-epoch
    "latest" to a DISTINCT sibling so ``hgt_best.pt`` keeps the best epoch.

    Two deterministic checks (independent of whether early-stop fires here):
    1. the distinct ``hgt_best.latest.pt`` sibling exists — the buggy single-path
       wiring would never create it (it wrote everything to ``hgt_best.pt``);
    2. the persisted ``hgt_best.pt`` records ``epoch == best_epoch`` — the best
       checkpoint is always written with ``epoch`` set to the improving epoch, so
       the two agree iff hgt_best.pt is the best (not the terminal) checkpoint.
    """
    features = _panel()
    result = run_sweep(
        features=features,
        labels=features[["emergent", "forward_share"]],
        snapshots=_snapshots(),
        cfg=_cfg(tmp_path),
        n_trials=2,
    )
    checkpoint = result.checkpoint_path
    latest = checkpoint.with_name(f"{checkpoint.stem}.latest.pt")
    assert latest.exists(), "retrain per-epoch checkpoint must use a path distinct from hgt_best.pt"

    ckpt = load_checkpoint(checkpoint)
    assert ckpt["epoch"] == ckpt["best_epoch"], (
        "hgt_best.pt must hold the best-val-AUC epoch's weights, not the terminal epoch "
        f"(epoch={ckpt['epoch']} best_epoch={ckpt['best_epoch']})"
    )
