"""Unit tests for the V1-S13 HGT checkpoint-resume training loop.

Hermetic and fast: tiny synthetic snapshots are assembled with
:func:`scifield.forecasting.gnn.loader.build_snapshot_from_frames` (no DuckDB, no
real corpus, no network) and a small ``(features, labels)`` panel spans a couple
of TRAIN years plus a VAL year. The five invariants under test (see the HANDOFF
NOTES in ``train.py``):

1. **Checkpoint round-trip** — :func:`load_forecaster` rebuilds a predict-ready
   forecaster from a checkpoint into FRESH objects (no training data), the saved
   torch RNG state round-trips, and the loaded weights match the trained net.
2. **resume == straight-through** — training ``N`` epochs straight gives the same
   predictions as training ``N//2`` then resuming for the rest (RNG restore makes
   resume byte-identical to an uninterrupted run).
3. **Determinism** — two same-seed/params runs give identical ``best_val_auc`` +
   predictions.
4. **best-by-VAL-AUC + early-stop patience** — with a scripted per-epoch AUC peak
   and a small ``patience``, training stops within ``patience`` of the best, and
   ``best_path`` carries the best-epoch (not last-epoch) model.
5. **fit unchanged** — ``train_hgt`` for ``N`` epochs (no early stop) reproduces
   ``HGTForecaster(epochs=N).fit().predict()`` bit-for-bit, proving the shared
   ``_setup_training`` / ``_train_one_epoch`` seams did not change ``fit``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from scifield.forecasting import train as train_mod  # noqa: E402
from scifield.forecasting.data import NODE_FEATURES  # noqa: E402
from scifield.forecasting.gnn.hgt import HGTForecaster  # noqa: E402
from scifield.forecasting.gnn.loader import build_snapshot_from_frames  # noqa: E402
from scifield.forecasting.train import (  # noqa: E402
    TrainResult,
    load_checkpoint,
    load_forecaster,
    save_checkpoint,
    train_hgt,
)

torch.set_num_threads(1)

#: Topic ids shared by every synthetic snapshot (stable across years). 5/8 get
#: papers; 12 never does but still reserves a node every year.
_TOPIC_IDS = [5, 8, 12]

#: Years used for the panel: two TRAIN years + one VAL year.
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
    """Build a materialized-style frame: TRAIN years + a VAL year, all masks set.

    One row per ``(topic_id in _TOPIC_IDS, origin_year)`` with the 16 NODE_FEATURES
    filled reproducibly, the ``split`` / ``volume_ok`` / ``label_complete`` masking
    columns the labeled-slice convention needs, plus ``emergent`` /
    ``forward_share`` supervision. The label pattern keeps BOTH classes present in
    every split so ``emergence_auc`` is defined (not single-class).
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
                # Alternate the label within each year so neither split is
                # single-class (emergence_auc would be nan otherwise).
                "emergent": int(k % 2),
                "forward_share": 0.05 + 0.01 * k,
            }
            for col in NODE_FEATURES:
                row[col] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


def _params(**overrides: object) -> dict[str, object]:
    """Small, fast HGTForecaster kwargs for the trainer (CPU, deterministic)."""
    base: dict[str, object] = {
        "conv_type": "sage",
        "hidden": 8,
        "n_layers": 2,
        "heads": 2,
        "dropout": 0.2,
        "lr": 1e-2,
        "seed": 1729,
        "device": "cpu",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# (1) Checkpoint round-trip: RNG restore + fresh-process predict
# --------------------------------------------------------------------------- #


def test_checkpoint_roundtrip_rng_and_fresh_predict(tmp_path: Path) -> None:
    features = _panel()
    snaps = _snapshots()
    ckpt_path = tmp_path / "ckpt.pt"
    best_path = tmp_path / "best.pt"

    result = train_hgt(
        features=features,
        labels=features[["emergent", "forward_share"]],
        snapshots=snaps,
        params=_params(),
        checkpoint_path=ckpt_path,
        best_path=best_path,
        max_epochs=5,
        patience=100,
    )
    assert isinstance(result, TrainResult)
    assert ckpt_path.exists() and best_path.exists()

    # The latest checkpoint's saved torch RNG state is the live RNG after the last
    # trained epoch (captured immediately post-step, before the eval-mode predict).
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt["epoch"] == result.last_epoch
    assert set(ckpt["rng"]) == {"torch", "numpy", "python"}

    # Rebuild a FRESH forecaster from the checkpoint (no training data) and predict.
    fresh_snaps = _snapshots()
    fc = load_forecaster(ckpt_path, fresh_snaps)
    assert fc.net_ is not None and fc.pipeline_ is not None
    val = features[features["split"] == "val"].reset_index(drop=True)
    pred = fc.predict(val)
    assert pred.emergence_score.shape == (len(val),)
    assert np.all(np.isfinite(pred.emergence_score))
    assert np.all(pred.share_forecast >= 0.0)

    # The restored weights equal the saved weights (round-trip is lossless).
    for key, saved in ckpt["model"].items():
        assert torch.equal(fc.net_.state_dict()[key], saved)


# --------------------------------------------------------------------------- #
# (2) resume == straight-through (bit-identical)
# --------------------------------------------------------------------------- #


def test_resume_matches_straight_through(tmp_path: Path) -> None:
    features = _panel()
    val = features[features["split"] == "val"].reset_index(drop=True)
    n_epochs = 8

    # --- Straight run: n_epochs in one shot (patience high -> no early stop). ---
    ck_a = tmp_path / "a_ckpt.pt"
    best_a = tmp_path / "a_best.pt"
    train_hgt(
        features=features,
        labels=features[["emergent", "forward_share"]],
        snapshots=_snapshots(),
        params=_params(),
        checkpoint_path=ck_a,
        best_path=best_a,
        max_epochs=n_epochs,
        patience=1000,
    )
    pred_a = load_forecaster(ck_a, _snapshots()).predict(val)

    # --- Split run: train n_epochs//2, then resume for the rest. ---
    ck_b = tmp_path / "b_ckpt.pt"
    best_b = tmp_path / "b_best.pt"
    train_hgt(
        features=features,
        labels=features[["emergent", "forward_share"]],
        snapshots=_snapshots(),
        params=_params(),
        checkpoint_path=ck_b,
        best_path=best_b,
        max_epochs=n_epochs // 2,
        patience=1000,
    )
    res_b = train_hgt(
        features=features,
        labels=features[["emergent", "forward_share"]],
        snapshots=_snapshots(),
        params=_params(),
        checkpoint_path=ck_b,
        best_path=best_b,
        resume=True,
        max_epochs=n_epochs,
        patience=1000,
    )
    assert res_b.last_epoch == n_epochs - 1
    pred_b = load_forecaster(ck_b, _snapshots()).predict(val)

    # RNG restore -> resume trajectory is byte-identical to the uninterrupted run.
    assert np.array_equal(pred_a.emergence_score, pred_b.emergence_score)
    assert np.array_equal(pred_a.share_forecast, pred_b.share_forecast)


# --------------------------------------------------------------------------- #
# (3) Determinism: same seed/params -> identical AUC + predictions
# --------------------------------------------------------------------------- #


def test_determinism_same_seed(tmp_path: Path) -> None:
    features = _panel()
    val = features[features["split"] == "val"].reset_index(drop=True)

    def _run(tag: str) -> tuple[float, np.ndarray]:
        ck = tmp_path / f"{tag}_ckpt.pt"
        best = tmp_path / f"{tag}_best.pt"
        res = train_hgt(
            features=features,
            labels=features[["emergent", "forward_share"]],
            snapshots=_snapshots(),
            params=_params(),
            checkpoint_path=ck,
            best_path=best,
            max_epochs=6,
            patience=1000,
        )
        pred = load_forecaster(best, _snapshots()).predict(val)
        return res.best_val_auc, pred.emergence_score

    auc1, scores1 = _run("r1")
    auc2, scores2 = _run("r2")

    assert auc1 == auc2 or (np.isnan(auc1) and np.isnan(auc2))
    assert np.array_equal(scores1, scores2)


# --------------------------------------------------------------------------- #
# (4) best-by-VAL-AUC + early-stop patience (scripted AUC peak)
# --------------------------------------------------------------------------- #


def test_best_by_val_auc_and_early_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    features = _panel()
    val = features[features["split"] == "val"].reset_index(drop=True)

    # Script a per-epoch AUC that peaks at epoch index 2 then declines, so the best
    # epoch and the early-stop point are deterministic regardless of the (flat)
    # synthetic signal. emergence_auc is called once per epoch (val_eval=True).
    scripted = iter([0.50, 0.60, 0.90, 0.70, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60])

    def _fake_auc(scores: np.ndarray, labels: np.ndarray) -> float:
        return float(next(scripted))

    monkeypatch.setattr(train_mod, "emergence_auc", _fake_auc)

    patience = 3
    ck = tmp_path / "ckpt.pt"
    best = tmp_path / "best.pt"
    res = train_hgt(
        features=features,
        labels=features[["emergent", "forward_share"]],
        snapshots=_snapshots(),
        params=_params(),
        checkpoint_path=ck,
        best_path=best,
        max_epochs=20,
        patience=patience,
    )

    # Peak (0.90) is epoch index 2; with patience=3 we stop 3 epochs later (idx 5).
    assert res.best_epoch == 2
    assert res.best_val_auc == pytest.approx(0.90)
    assert res.last_epoch == 2 + patience  # stopped within `patience` of the best
    assert res.last_epoch < 19  # genuinely early-stopped (not the full budget)

    # best_path holds the BEST epoch's checkpoint, not the last epoch's.
    best_ckpt = load_checkpoint(best)
    assert best_ckpt["epoch"] == 2
    assert best_ckpt["best_epoch"] == 2
    last_ckpt = load_checkpoint(ck)
    assert last_ckpt["epoch"] == res.last_epoch
    assert last_ckpt["epoch"] != best_ckpt["epoch"]

    # load_forecaster(best_path) gives the best-epoch model, whose weights equal
    # the best checkpoint's saved weights (NOT the later, last-epoch weights).
    fc_best = load_forecaster(best, _snapshots())
    assert fc_best.net_ is not None
    for key, saved in best_ckpt["model"].items():
        assert torch.equal(fc_best.net_.state_dict()[key], saved)
    # Sanity: the last checkpoint's weights differ from the best's (training moved).
    differ = any(
        not torch.equal(last_ckpt["model"][k], best_ckpt["model"][k]) for k in best_ckpt["model"]
    )
    assert differ
    # Predict works off the best-epoch forecaster.
    assert np.all(np.isfinite(fc_best.predict(val).emergence_score))


# --------------------------------------------------------------------------- #
# (5) fit unchanged: train_hgt == HGTForecaster.fit (shared epoch step)
# --------------------------------------------------------------------------- #


def test_train_hgt_matches_plain_fit(tmp_path: Path) -> None:
    features = _panel()
    n_epochs = 7

    # Reference: a plain HGTForecaster.fit over the TRAIN labeled slice. Build the
    # slice with the SAME masking train_hgt uses so the two see identical rows.
    train_slice = features[features["split"] == "train"].reset_index(drop=True)
    ref_params: dict[str, Any] = _params()
    ref = HGTForecaster(_snapshots(), epochs=n_epochs, **ref_params)
    ref.fit(train_slice, train_slice[["emergent", "forward_share"]])

    # train_hgt with val_eval=False runs exactly _train_one_epoch n_epochs times
    # (no interleaved eval-mode predict, no early stop) -> same trajectory as fit.
    ck = tmp_path / "ckpt.pt"
    best = tmp_path / "best.pt"
    train_hgt(
        features=features,
        labels=features[["emergent", "forward_share"]],
        snapshots=_snapshots(),
        params=ref_params,
        checkpoint_path=ck,
        best_path=best,
        max_epochs=n_epochs,
        val_eval=False,
    )
    loaded = load_forecaster(ck, _snapshots())

    # Score the FULL panel with both; predictions must be bit-identical.
    ref_pred = ref.predict(features)
    got_pred = loaded.predict(features)
    assert np.array_equal(ref_pred.emergence_score, got_pred.emergence_score)
    assert np.array_equal(ref_pred.share_forecast, got_pred.share_forecast)

    # And the trained weights themselves match the reference fit exactly.
    assert ref.net_ is not None and loaded.net_ is not None
    for key, ref_w in ref.net_.state_dict().items():
        assert torch.equal(loaded.net_.state_dict()[key], ref_w)


# --------------------------------------------------------------------------- #
# Atomic save/load smoke (no orphan tmp files)
# --------------------------------------------------------------------------- #


def test_save_checkpoint_atomic_no_tmp_left(tmp_path: Path) -> None:
    path = tmp_path / "x.pt"
    save_checkpoint(path, a=1, b=torch.zeros(3))
    assert path.exists()
    loaded = load_checkpoint(path)
    assert loaded["a"] == 1
    assert torch.equal(loaded["b"], torch.zeros(3))
    # No leftover tmp files in the directory.
    assert [p.name for p in tmp_path.iterdir()] == ["x.pt"]
