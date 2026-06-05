"""V1-S13 HGT checkpoint-resume training loop — the crash-safe trainer.

This module is the **training driver** of the V1-S13 GNN slice. It wraps the
:class:`~scifield.forecasting.gnn.hgt.HGTForecaster` (the model + per-epoch step
from :mod:`scifield.forecasting.gnn.hgt`) in a loop that trains one
hyperparameter configuration to convergence with **bit-reproducible,
crash-safe** checkpoints. The Optuna sweep (next slice) calls :func:`train_hgt`
once per trial; the CLI's ``gnn-train --resume`` calls it to continue an
interrupted run. The pre-registration (OSF DOI 10.17605/OSF.IO/XP94F) constrains
nothing about checkpointing — this is pure engineering on top of the locked
model.

What "checkpoint-resume" buys (and why each design choice)
----------------------------------------------------------
The training loop is **full-graph, full-batch with NO minibatch shuffle**: the
only stochastic operation inside a step is dropout (see
:meth:`scifield.forecasting.gnn.hgt.HGTForecaster._train_one_epoch`). So an
interrupted run can be resumed to a **byte-identical** trajectory iff we restore
*all four* pieces of mutable state:

* **model weights** — ``net_.state_dict()``;
* **optimizer state** — Adam's per-parameter momentum buffers
  (``optimizer.state_dict()``); dropping these would restart Adam's moments and
  change every subsequent update;
* **the RNG state** — ``torch.get_rng_state()`` (the dropout masks); numpy +
  python RNG are also saved for parity even though this loop does not consume
  them, so a future stochastic addition resumes correctly;
* **the epoch counter** — to continue from the right step.

We checkpoint **every epoch** because the run is cheap (Mac CPU, $0) and a crash
mid-sweep must lose at most one epoch. The write is **atomic** (write a tmp file
in the *same directory*, then :func:`os.replace`) so a crash mid-write can never
leave a half-written, un-loadable checkpoint: ``os.replace`` is atomic on the
same filesystem, and the tmp-in-same-dir rule guarantees we never cross a
filesystem boundary (which would silently degrade ``os.replace`` to a
copy-then-delete).

Why the impute+scale pipeline is *pickled and restored*, never refit
--------------------------------------------------------------------
The forecaster's :class:`~sklearn.pipeline.Pipeline` (mean-imputer +
standard-scaler) is fit on the **TRAIN** feature distribution only. Refitting it
at predict time — or rebuilding it from scratch in a fresh process — would fold
the prediction slice's own mean/variance into the features, leaking that slice's
distribution into the model input (the exact leak the train-only pipeline exists
to prevent, and the reason it is held byte-identical to the ``no_graph``
baseline). So the fitted pipeline is serialised into the checkpoint and restored
verbatim; :func:`load_forecaster` can therefore produce a working ``.predict()``
in a *fresh process with no training data at all*.

Public surface (kept stable for the sweep + CLI that depend on it)
------------------------------------------------------------------
* :func:`train_hgt` — train one config (optionally resuming); returns a
  :class:`TrainResult`.
* :func:`save_checkpoint` / :func:`load_checkpoint` — atomic torch I/O.
* :func:`load_forecaster` — rebuild a predict-ready forecaster from a checkpoint.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from scifield.forecasting.baselines.base import emergence_auc
from scifield.forecasting.baselines.evaluate import _labeled_slice, _pin_torch_single_threaded
from scifield.forecasting.gnn.hgt import HGTForecaster

if TYPE_CHECKING:
    import pandas as pd
    from torch_geometric.data import HeteroData

__all__ = [
    "TrainResult",
    "load_checkpoint",
    "load_forecaster",
    "save_checkpoint",
    "train_hgt",
]


@dataclass
class TrainResult:
    """Outcome of one :func:`train_hgt` run.

    Attributes
    ----------
    best_val_auc:
        The best (highest) validation emergence ROC-AUC seen across epochs, or
        ``nan`` if ``val_eval`` was off / the VAL slice was single-class every
        epoch. The Optuna sweep maximises this.
    best_epoch:
        0-based index of the epoch that produced ``best_val_auc`` (the epoch whose
        weights were written to ``best_path``); ``-1`` if no epoch ever improved.
    last_epoch:
        0-based index of the last epoch actually run (``< max_epochs - 1`` if early
        stopping fired or a resume started past the budget).
    best_path:
        Path the best-epoch checkpoint was written to (the artifact
        :func:`load_forecaster` should be pointed at for the selected model).
    history:
        One dict per epoch with at least ``epoch`` (int), ``train_loss`` (float),
        ``val_auc`` (float, ``nan`` if not evaluated) and ``is_best`` (bool).
    """

    best_val_auc: float
    best_epoch: int
    last_epoch: int
    best_path: Path
    history: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Atomic checkpoint I/O
# --------------------------------------------------------------------------- #


def save_checkpoint(path: str | Path, **state: Any) -> Path:
    """Atomically write a checkpoint ``state`` dict to ``path`` via ``torch.save``.

    Writes to a temporary file **in the same directory** as ``path`` and then
    :func:`os.replace`-s it into place. Both halves matter:

    * ``os.replace`` is an atomic rename on a single filesystem, so a reader (or a
      crash) never observes a partially-written file — the destination is either
      the old contents or the complete new contents, never a torn mix.
    * keeping the tmp file in the *same directory* guarantees source and
      destination share a filesystem; ``os.replace`` across filesystems raises (or
      silently degrades), defeating atomicity.

    The payload is pickled with :func:`torch.save`, which handles both the tensor
    state-dicts and the small sklearn ``Pipeline`` object embedded under the
    ``"pipeline"`` key.

    Parameters
    ----------
    path:
        Destination checkpoint path (its parent directory must exist).
    **state:
        Arbitrary picklable checkpoint contents (see :func:`train_hgt` for the
        schema written by the trainer).

    Returns
    -------
    pathlib.Path
        The destination ``path`` (now containing the checkpoint).
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(state), tmp)
        os.replace(tmp, path)
    finally:
        # On a failure before/at os.replace the tmp may linger; clean it up so a
        # crashed write never leaves orphan tmp files next to the checkpoint.
        if tmp.exists():
            tmp.unlink()
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint` (``weights_only=False``).

    ``weights_only=False`` is required because the checkpoint is NOT a bare
    state-dict: it embeds the pickled sklearn ``Pipeline`` (under ``"pipeline"``),
    the RNG ``ByteTensor``/tuples, and the ``params`` dict. These are our own,
    locally-produced artifacts, so loading them with full unpickling is safe here.

    Parameters
    ----------
    path:
        Path to a checkpoint file.

    Returns
    -------
    dict
        The deserialised checkpoint state dict.
    """
    # weights_only=False: the checkpoint carries a pickled sklearn Pipeline + RNG
    # state + a params dict, not just tensors. The annotated assignment pins the
    # type (torch.load is loosely typed) without a blanket ignore.
    loaded: dict[str, Any] = torch.load(path, weights_only=False)
    return loaded


# --------------------------------------------------------------------------- #
# RNG capture / restore (the resume-determinism core)
# --------------------------------------------------------------------------- #


def _capture_rng() -> dict[str, Any]:
    """Snapshot torch + numpy + python RNG state for the checkpoint.

    Only ``torch`` is actually consumed by the training loop (dropout); numpy and
    python are captured for parity so a future stochastic op (e.g. a minibatch
    shuffle) resumes deterministically without a checkpoint-format change.
    """
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(rng: dict[str, Any]) -> None:
    """Restore the RNG state captured by :func:`_capture_rng`.

    Restoring ``torch.set_rng_state`` is what makes a resumed run bit-identical to
    an uninterrupted one: the dropout masks drawn in the remaining epochs continue
    the exact same pseudo-random sequence the straight-through run would have used.
    """
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


# --------------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------------- #


def train_hgt(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    snapshots: Mapping[int, HeteroData],
    params: dict[str, Any],
    train_split: str = "train",
    val_split: str = "val",
    checkpoint_path: Path,
    best_path: Path,
    resume: bool = False,
    patience: int = 25,
    max_epochs: int = 200,
    cfg_hash: str = "",
    val_eval: bool = True,
) -> TrainResult:
    """Train one HGT config to convergence with crash-safe, resumable checkpoints.

    Slices the TRAIN/VAL labeled rows out of ``features`` using the LOCKED
    convention (``split & volume_ok & label_complete``, reusing
    :func:`scifield.forecasting.baselines.evaluate._labeled_slice`), builds an
    :class:`~scifield.forecasting.gnn.hgt.HGTForecaster` from ``params``, and runs
    its shared per-epoch step (:meth:`HGTForecaster._train_one_epoch`) for up to
    ``max_epochs`` epochs. Every epoch it (optionally) computes the VAL emergence
    AUC, appends to ``history``, **atomically** writes the full resumable state to
    ``checkpoint_path``, and — whenever the VAL AUC *strictly* improves — also
    atomically writes ``best_path``. Training early-stops once the VAL AUC has not
    improved for ``patience`` consecutive epochs.

    Determinism / resume
        ``params["seed"]`` (forwarded to the forecaster) seeds torch+numpy once in
        ``_setup_training``; the only per-epoch stochastic op is dropout, so
        checkpointing ``torch.get_rng_state()`` (plus numpy/python for parity)
        makes ``resume=True`` produce a trajectory **byte-identical** to an
        uninterrupted run. On resume the forecaster is rebuilt and re-set-up
        (which re-seeds), then the saved model/optimizer/RNG/epoch/best state is
        restored and training **continues from the saved epoch + 1**.

    Checkpoint dict schema (exact keys, written every epoch)
        ``model`` (net state_dict), ``optimizer`` (Adam state_dict), ``epoch``
        (int, last completed 0-based epoch), ``best_val_auc`` (float),
        ``best_epoch`` (int), ``rng`` (``{"torch","numpy","python"}``),
        ``pipeline`` (the fitted sklearn impute+scale Pipeline — restored, never
        refit), ``pos_weight`` (float), ``params`` (dict), ``feature_columns``
        (list[str]), ``cfg_hash`` (str). ``best_path`` carries the same schema for
        the best epoch.

    Parameters
    ----------
    features:
        The materialized ``(topic_id, origin_year, NODE_FEATURES..., label cols,
        split, volume_ok, label_complete)`` frame.
    labels:
        Ignored for masking (kept for signature symmetry with the baselines);
        supervision is read off the TRAIN/VAL slices' own ``emergent`` /
        ``forward_share`` columns. May be ``None``-like; only the slices are used.
    snapshots:
        ``{origin_year -> HeteroData}`` forward-edge-only snapshots covering every
        TRAIN (and, when ``val_eval``, VAL) origin year.
    params:
        Constructor kwargs for :class:`HGTForecaster` (e.g. ``conv_type``,
        ``hidden``, ``n_layers``, ``heads``, ``dropout``, ``lr``, ``seed``,
        ``device``, ``pos_weight``). ``epochs`` is overridden by the loop and may
        be omitted.
    train_split, val_split:
        Split names to slice (default ``"train"`` / ``"val"``).
    checkpoint_path:
        Where the per-epoch resumable checkpoint is written (atomically).
    best_path:
        Where the best-by-VAL-AUC checkpoint is written (atomically).
    resume:
        If ``True`` and ``checkpoint_path`` exists, restore and continue.
    patience:
        Early-stop after this many epochs with no strict VAL-AUC improvement.
    max_epochs:
        Hard epoch budget.
    cfg_hash:
        Opaque config-hash string recorded into the checkpoint (provenance).
    val_eval:
        If ``False``, skip VAL scoring (``val_auc`` is ``nan`` every epoch, no
        early stopping, ``best_path`` tracks the *last* epoch). Used by the
        retrain-at-best pass where the trainer just runs a fixed epoch budget.

    Returns
    -------
    TrainResult
        ``best_val_auc``, ``best_epoch``, ``last_epoch``, ``best_path``, and the
        per-epoch ``history``.
    """
    _pin_torch_single_threaded()
    # Deterministic kernels where available; warn (don't raise) on ops without a
    # deterministic impl so CPU PyG convs still run.
    torch.use_deterministic_algorithms(True, warn_only=True)

    train_slice = _labeled_slice(features, train_split)
    val_slice = _labeled_slice(features, val_split)
    train_labels = train_slice[["emergent", "forward_share"]]

    # epochs in params is irrelevant here (the loop owns the budget); pop it so the
    # forecaster constructor does not also try to honour a stale value.
    fc_params = {k: v for k, v in params.items() if k != "epochs"}
    forecaster = HGTForecaster(snapshots, **fc_params)

    # _setup_training seeds torch+numpy, fits the train-only pipeline, resolves
    # pos_weight_, builds net_ + optimizer + criteria + the per-year cache. After
    # this, net_/pipeline_/pos_weight_ are live so predict() works for VAL scoring.
    state = forecaster._setup_training(train_slice, train_labels)

    start_epoch = 0
    best_val_auc = float("nan")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    epochs_since_improve = 0

    if resume and Path(checkpoint_path).exists():
        ckpt = load_checkpoint(checkpoint_path)
        _restore_from_checkpoint(forecaster, state, ckpt)
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_auc = float(ckpt["best_val_auc"])
        best_epoch = int(ckpt["best_epoch"])
        history = list(ckpt.get("history", []))
        # Rebuild the patience counter from how far the saved run had drifted past
        # its best, so resume early-stops at exactly the same epoch as a straight
        # run would have.
        epochs_since_improve = start_epoch - 1 - best_epoch if best_epoch >= 0 else start_epoch

    feature_columns = list(forecaster.feature_columns)
    pos_weight_val = float(forecaster.pos_weight_) if forecaster.pos_weight_ is not None else 1.0

    val_emergent = val_slice["emergent"].to_numpy() if len(val_slice) else np.empty(0)

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs):
        train_loss = forecaster._train_one_epoch(state)
        # Snapshot RNG immediately after the (dropout-consuming) step, BEFORE the
        # eval-mode predict (which draws no dropout), so the saved state is exactly
        # what the next epoch must continue from.
        rng_state = _capture_rng()

        val_auc = float("nan")
        if val_eval and len(val_slice):
            # predict() flips net_ to eval(); flip back to train() after so the next
            # epoch's dropout is active again (the epoch step also calls train()).
            pred = forecaster.predict(val_slice)
            val_auc = emergence_auc(pred.emergence_score, val_emergent)

        improved = val_auc == val_auc and (best_epoch < 0 or val_auc > best_val_auc)  # NaN-safe
        if not val_eval:
            # No VAL signal: treat every epoch as the "best so far" so best_path
            # tracks the final epoch of a fixed-budget retrain.
            improved = True

        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": val_auc,
                "is_best": improved,
            }
        )

        ckpt_state = _checkpoint_state(
            forecaster=forecaster,
            optimizer_state=state.optimizer.state_dict(),
            epoch=epoch,
            best_val_auc=best_val_auc,
            best_epoch=best_epoch,
            rng_state=rng_state,
            pos_weight=pos_weight_val,
            params=params,
            feature_columns=feature_columns,
            cfg_hash=cfg_hash,
            history=history,
        )
        save_checkpoint(checkpoint_path, **ckpt_state)
        if improved:
            save_checkpoint(best_path, **ckpt_state)

        last_epoch = epoch
        if val_eval and epochs_since_improve >= patience:
            break

    return TrainResult(
        best_val_auc=best_val_auc,
        best_epoch=best_epoch,
        last_epoch=last_epoch,
        best_path=Path(best_path),
        history=history,
    )


# --------------------------------------------------------------------------- #
# Checkpoint <-> forecaster glue
# --------------------------------------------------------------------------- #


def _checkpoint_state(
    *,
    forecaster: HGTForecaster,
    optimizer_state: dict[str, Any],
    epoch: int,
    best_val_auc: float,
    best_epoch: int,
    rng_state: dict[str, Any],
    pos_weight: float,
    params: dict[str, Any],
    feature_columns: list[str],
    cfg_hash: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the exact checkpoint dict schema (see :func:`train_hgt`).

    The fitted sklearn ``pipeline`` is embedded directly (torch.save pickles it);
    it is RESTORED verbatim on load — never refit — so the prediction-time feature
    scaling exactly reproduces the train-only statistics (no distribution leak).
    """
    assert forecaster.net_ is not None  # set by _setup_training
    return {
        "model": forecaster.net_.state_dict(),
        "optimizer": optimizer_state,
        "epoch": int(epoch),
        "best_val_auc": float(best_val_auc),
        "best_epoch": int(best_epoch),
        "rng": rng_state,
        "pipeline": forecaster.pipeline_,
        "pos_weight": float(pos_weight),
        "params": dict(params),
        "feature_columns": list(feature_columns),
        "cfg_hash": str(cfg_hash),
        "history": list(history),
    }


def _restore_from_checkpoint(forecaster: HGTForecaster, state: Any, ckpt: dict[str, Any]) -> None:
    """Restore model+optimizer+RNG+pipeline+pos_weight into a live, set-up run.

    Assumes ``forecaster`` has already been through ``_setup_training`` (so
    ``net_`` / ``optimizer`` exist with the right shapes) and overwrites their
    parameters in place. The pipeline + ``pos_weight_`` are restored (not refit /
    recomputed) for the leakage reason in the module docstring.
    """
    assert forecaster.net_ is not None  # set by _setup_training
    forecaster.net_.load_state_dict(ckpt["model"])
    state.optimizer.load_state_dict(ckpt["optimizer"])
    forecaster.pipeline_ = ckpt["pipeline"]
    forecaster.pos_weight_ = float(ckpt["pos_weight"])
    _restore_rng(ckpt["rng"])


def load_forecaster(
    checkpoint_path: str | Path, snapshots: Mapping[int, HeteroData]
) -> HGTForecaster:
    """Rebuild a predict-ready :class:`HGTForecaster` from a checkpoint path.

    Reconstructs the forecaster from the checkpoint's ``params`` + ``snapshots``,
    rebuilds the architecturally-identical ``net_`` via ``_build_net``, loads the
    saved weights, restores the **fitted** impute+scale ``pipeline`` and the
    resolved ``pos_weight_``, and puts the net in ``eval()`` mode. The result has
    everything ``predict`` needs and NOTHING from training — so it works in a
    **fresh process with no training data** (plan Verification #5). The pipeline
    is restored, never refit, so prediction-time scaling reproduces the train-only
    statistics exactly (no leakage).

    Parameters
    ----------
    checkpoint_path:
        Path to a checkpoint written by :func:`train_hgt` (typically the
        ``best_path``).
    snapshots:
        ``{origin_year -> HeteroData}`` covering every origin year that will be
        scored (the model needs each year's graph to forward).

    Returns
    -------
    HGTForecaster
        A fitted-and-restored forecaster ready for ``.predict()``.
    """
    _pin_torch_single_threaded()
    ckpt = load_checkpoint(checkpoint_path)
    params = {k: v for k, v in dict(ckpt["params"]).items() if k != "epochs"}
    forecaster = HGTForecaster(snapshots, **params)
    # Build the architecturally-identical net, then load the saved weights into it.
    net = forecaster._build_net()
    net.load_state_dict(ckpt["model"])
    net.eval()
    forecaster.net_ = net
    forecaster.pipeline_ = ckpt["pipeline"]
    forecaster.pos_weight_ = float(ckpt["pos_weight"])
    return forecaster
