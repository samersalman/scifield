"""V1-S13 Optuna hyperparameter sweep over the pre-registered HGT forecaster.

This module is the **tuning driver** of the V1-S13 GNN slice. The architecture,
metrics, and temporal split are frozen by the pre-registration (OSF DOI
10.17605/OSF.IO/XP94F); what is NOT frozen is the handful of hyperparameters
(learning rate, hidden width, conv depth, dropout, attention heads). This sweep
explores that small grid with Optuna's TPE sampler, scores every trial by the
**validation emergence ROC-AUC** (the locked objective from
:func:`scifield.forecasting.baselines.base.emergence_auc`), selects the single
best configuration, and retrains it once to disk as ``hgt_best.pt``. It runs
LOCALLY on Mac CPU at $0 — no network, no external API, no test-set access (the
test set stays sealed until V1-S14).

Why the snapshots are built ONCE, outside the objective
-------------------------------------------------------
A :class:`~torch_geometric.data.HeteroData` snapshot per origin year is the
data the model fits on; building (or ``ToUndirected``-symmetrising) it is the
expensive part, and it is **identical across every trial** (hyperparameters
change the model, never the graph). So the caller injects the already-built
``{origin_year -> HeteroData}`` mapping and we hand the *same* dict to every
trial's forecaster + the final retrain. Rebuilding per trial would multiply the
graph cost by ``n_trials`` for zero benefit and risk a per-trial graph drift.

Why the seed is FIXED across trials (not a search dimension)
------------------------------------------------------------
The validation slice has only ~17 positives, so the AUC of a single fit is a
noisy estimate. Adding the seed as a search dimension would let TPE chase that
noise — picking a "lucky" seed rather than a genuinely better architecture, an
overfit to the tiny VAL slice. The pre-registered design therefore holds the
seed fixed (``cfg.gnn.seed``): trials differ ONLY by hyperparameters, the run is
deterministic, and the retrain-at-best uses that same seed. (Seed-averaging /
seed-robustness is a separate, out-of-scope question for a later slice.)

Why a lightweight per-trial epoch loop (and no per-trial checkpoint)
--------------------------------------------------------------------
Each trial drives the SAME shared per-epoch step the crash-safe trainer uses
(:meth:`scifield.forecasting.gnn.hgt.HGTForecaster._train_one_epoch` via
:meth:`~scifield.forecasting.gnn.hgt.HGTForecaster._setup_training`), so the
trial-time loss is byte-identical to :func:`scifield.forecasting.train.train_hgt`
— there is no second, drifting copy of the optimisation logic. But trials do
NOT write a checkpoint every epoch: a sweep runs dozens of configurations and
per-trial disk churn would dominate the wall-clock for no gain (only the
*winner* needs to be persisted). Instead, after each epoch we score VAL AUC and
**report it to Optuna's :class:`~optuna.pruners.MedianPruner`** as an
intermediate value; clearly-losing trials are pruned early. Selection is by the
best per-trial VAL AUC (``study.best_trial``), and ONLY that configuration is
retrained — this time through :func:`~scifield.forecasting.train.train_hgt`, so
``hgt_best.pt`` carries a full, resumable, predict-ready checkpoint.

Why the sweep itself resumes (``load_if_exists``)
-------------------------------------------------
The study is stored in a SQLite db (``study_db``) under a fixed ``study_name``
with ``load_if_exists=True``: re-running :func:`run_sweep` against the same db
*adds* trials toward ``n_trials`` rather than restarting, so an interrupted
sweep continues exactly where it stopped (and a fully-completed sweep runs zero
new trials). The provenance sidecars written next to both artifacts fold in
``cfg.preregistration`` (the OSF DOI), exactly like the V1-S12 baselines, so
every output is traceable to the registration that constrains it.

Public surface (kept stable for the CLI that depends on it)
-----------------------------------------------------------
* :func:`run_sweep` — run/resume the study, retrain the winner, write the sweep
  parquet + ``hgt_best.pt`` + both ``record_run`` sidecars; returns a
  :class:`SweepResult`.
* :class:`SweepResult` — the typed summary the CLI echoes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from scifield.forecasting.baselines.base import emergence_auc, share_mape
from scifield.forecasting.baselines.evaluate import _labeled_slice, _pin_torch_single_threaded
from scifield.forecasting.gnn.hgt import HGTForecaster
from scifield.forecasting.train import load_forecaster, train_hgt
from scifield.repro import record_run

if TYPE_CHECKING:
    import optuna
    from torch_geometric.data import HeteroData

__all__ = ["SweepResult", "run_sweep"]

#: Fixed Optuna study name — the same name is reused across resumed runs so
#: ``load_if_exists=True`` re-opens (rather than forks) the on-disk study.
_STUDY_NAME = "hgt_v1s13"

#: Default trial budget when neither the ``n_trials`` arg nor ``cfg.gnn.sweep``
#: supplies one (mirrors ``conf/forecasting/v1.yaml``'s ``gnn.sweep.n_trials``).
_DEFAULT_N_TRIALS = 40

#: The 5 searched hyperparameters, in the order they appear in the sweep parquet.
_SEARCH_PARAMS = ("lr", "hidden", "n_layers", "dropout", "heads")

#: The per-trial parquet columns (one row per trial), in order.
_SWEEP_COLUMNS = (
    "trial_number",
    *_SEARCH_PARAMS,
    "conv_type",
    "seed",
    "val_auc",
    "val_mape",
    "best_epoch",
    "n_epochs_run",
    "state",
)


@dataclass
class SweepResult:
    """Outcome of one :func:`run_sweep` run.

    Attributes
    ----------
    best_params:
        The winning :class:`HGTForecaster` constructor kwargs (the 5 searched
        hyperparameters plus the fixed ``conv_type`` / ``seed`` / ``device`` /
        ``pos_weight``) — exactly what was passed to the retrain-at-best.
    best_val_auc:
        The best (highest) validation emergence ROC-AUC over the study, taken
        from ``study.best_value``. May be ``nan`` only in the degenerate case
        where every trial's VAL slice was single-class (guarded, never poisons
        the objective on the real ~17-positive slice).
    best_val_mape:
        The validation share MAPE recorded for the winning trial (an
        accompanying diagnostic; selection is by AUC, never MAPE).
    n_trials:
        Total number of COMPLETE trials in the study after this run (including
        any inherited from a resumed db) — NOT just the trials added this call.
    sweep_parquet_path:
        Path the one-row-per-trial sweep table was written to.
    checkpoint_path:
        Path the retrained best model (``hgt_best.pt``) was written to.
    trials:
        One dict per trial (the rows that became the sweep parquet): the searched
        params, ``val_auc`` / ``val_mape``, ``best_epoch``, ``n_epochs_run`` and
        ``state`` (``"complete"`` / ``"pruned"``).
    """

    best_params: dict[str, Any]
    best_val_auc: float
    best_val_mape: float
    n_trials: int
    sweep_parquet_path: Path
    checkpoint_path: Path
    trials: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #


def _cfg_hash(cfg: Any) -> str:
    """Stable hash of the resolved ``cfg.gnn`` block (checkpoint provenance).

    Hashes a JSON dump of the (numpy-coerced) ``gnn`` sub-config so the value
    recorded into the retrain checkpoint pins exactly which architecture/sweep
    settings produced it. Best-effort: if the block cannot be resolved we fall
    back to the empty string (the checkpoint just carries no hash).
    """
    try:
        from omegaconf import OmegaConf

        gnn = OmegaConf.to_container(cfg.gnn, resolve=True)
    except Exception:  # pragma: no cover - defensive; cfg.gnn is always present
        return ""
    return hashlib.sha256(json.dumps(_to_jsonable(gnn), sort_keys=True).encode()).hexdigest()


def _to_jsonable(obj: Any) -> Any:
    """Recursively coerce numpy scalars/arrays to native Python for ``json.dumps``.

    A tiny local equivalent of ``scifield.cli._to_jsonable`` (kept here so this
    module does not import the CLI): ``record_run`` ``json.dumps`` the config
    dict, which chokes on numpy floats/ints (e.g. a ``np.float64`` AUC).
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    return obj


def _prereg_dict(cfg: Any) -> dict[str, Any]:
    """Resolve ``cfg.preregistration`` to a plain dict (the OSF DOI block).

    Folded into both sidecar configs so each artifact references the
    pre-registration (#2) that constrains it, exactly like
    ``cli.forecasting_baselines``.
    """
    from typing import cast

    from omegaconf import OmegaConf

    block = cast("dict[str, Any]", OmegaConf.to_container(cfg.preregistration, resolve=True))
    return cast("dict[str, Any]", _to_jsonable(block))


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #


def run_sweep(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    snapshots: Mapping[int, HeteroData],
    cfg: Any,
    n_trials: int | None = None,
    study_db: Path | None = None,
    sweep_parquet_path: Path | None = None,
    checkpoint_path: Path | None = None,
    inputs: dict[str, Path] | None = None,
) -> SweepResult:
    """Run/resume the HGT Optuna sweep, retrain the winner, and persist everything.

    Drives a TPE-sampled, median-pruned search over the 5 pre-registered-free
    hyperparameters (``lr, hidden, n_layers, dropout, heads`` from
    ``cfg.gnn.sweep.search``), scoring each trial by the validation emergence
    ROC-AUC. The study lives in a SQLite db under a fixed name with
    ``load_if_exists=True``, so re-invoking this function against the same
    ``study_db`` *adds* trials toward ``n_trials`` (resume) instead of
    restarting. After the search the single best configuration
    (``study.best_trial``) is retrained via
    :func:`scifield.forecasting.train.train_hgt` straight onto ``hgt_best.pt`` (a
    full, predict-ready checkpoint), the one-row-per-trial sweep table is written
    to ``sweep_parquet_path``, and a :func:`scifield.repro.record_run` provenance
    sidecar — folding in the OSF DOI from ``cfg.preregistration`` — is written
    next to BOTH artifacts.

    Determinism / honesty
        The forecaster seed is held FIXED at ``cfg.gnn.seed`` for every trial and
        the retrain (see the module docstring): trials differ only by
        hyperparameters, the run is reproducible, and we make no attempt to chase
        the tiny VAL slice's seed noise. Selection is strictly by VAL AUC; the
        VAL MAPE is recorded per trial as a diagnostic only.

    Snapshots are built ONCE
        ``snapshots`` is the caller-injected ``{origin_year -> HeteroData}``
        mapping and the SAME object is reused by every trial and the retrain —
        never rebuilt inside the objective (the graph is identical across
        hyperparameter settings).

    Parameters
    ----------
    features:
        The materialized ``(topic_id, origin_year, NODE_FEATURES..., label cols,
        split, volume_ok, label_complete)`` frame. The TRAIN/VAL labeled slices
        are cut with the LOCKED ``split & volume_ok & label_complete`` masking
        (:func:`scifield.forecasting.baselines.evaluate._labeled_slice`).
    labels:
        Kept for signature symmetry with the trainer/baselines; supervision is
        read off each slice's own ``emergent`` / ``forward_share`` columns.
    snapshots:
        ``{origin_year -> HeteroData}`` forward-edge-only snapshots covering every
        TRAIN and VAL origin year (``ToUndirected`` is applied internally by the
        forecaster). Built once by the caller and reused throughout.
    cfg:
        The forecasting ``DictConfig`` (``conf/forecasting/v1.yaml``). Reads
        ``gnn.sweep.{n_trials, search.*}``, ``gnn.{conv_type, epochs, patience,
        seed, device, pos_weight}``, ``output.{sweep_parquet, checkpoint,
        study_db}`` and ``preregistration``.
    n_trials:
        Trial budget; ``None`` falls back to ``cfg.gnn.sweep.n_trials`` then
        :data:`_DEFAULT_N_TRIALS`. This is the *target total* completed trials —
        a resumed study only runs ``max(0, n_trials - already_complete)`` more.
    study_db, sweep_parquet_path, checkpoint_path:
        Output path overrides; ``None`` falls back to ``cfg.output.study_db`` /
        ``cfg.output.sweep_parquet`` / ``cfg.output.checkpoint``.
    inputs:
        ``{name -> path}`` of upstream artifacts to hash into the sidecars (e.g.
        ``{"forecasting_features": <parquet>}``); defaults to ``{}``.

    Returns
    -------
    SweepResult
        ``best_params``, ``best_val_auc``, ``best_val_mape``, ``n_trials`` (total
        completed), the two artifact paths, and the per-trial ``trials`` rows.
    """
    # Lazy import: keeps optuna out of the module-import path (mirrors evaluate.py
    # lazy-importing the torch baselines) and defers its (slow) import to call time.
    import optuna

    _pin_torch_single_threaded()

    # --- Resolve the trial budget + all output paths (arg overrides cfg). ------
    if n_trials is None:
        n_trials = int(getattr(cfg.gnn.sweep, "n_trials", _DEFAULT_N_TRIALS))
    n_trials = int(n_trials)

    study_db = Path(study_db) if study_db is not None else Path(str(cfg.output.study_db))
    sweep_parquet_path = (
        Path(sweep_parquet_path)
        if sweep_parquet_path is not None
        else Path(str(cfg.output.sweep_parquet))
    )
    checkpoint_path = (
        Path(checkpoint_path) if checkpoint_path is not None else Path(str(cfg.output.checkpoint))
    )
    inputs = inputs or {}

    # Fixed, non-searched forecaster settings (the pre-registered constants).
    seed = int(cfg.gnn.seed)
    conv_type = str(cfg.gnn.conv_type)
    device = str(cfg.gnn.device)
    pos_weight = cfg.gnn.pos_weight
    pos_weight = str(pos_weight) if isinstance(pos_weight, str) else float(pos_weight)
    epochs = int(cfg.gnn.epochs)
    patience = int(cfg.gnn.patience)

    # Search-space bounds (read once, outside the objective).
    search = cfg.gnn.sweep.search
    lr_lo, lr_hi = float(search.lr[0]), float(search.lr[1])
    hidden_choices = [int(h) for h in search.hidden]
    n_layers_choices = [int(n) for n in search.n_layers]
    dropout_lo, dropout_hi = float(search.dropout[0]), float(search.dropout[1])
    heads_choices = [int(h) for h in search.heads]

    # --- TRAIN/VAL labeled slices (LOCKED masking) + VAL targets, built once. --
    train_slice = _labeled_slice(features, "train")
    val_slice = _labeled_slice(features, "val")
    train_labels = train_slice[["emergent", "forward_share"]]
    val_emergent = val_slice["emergent"].to_numpy() if len(val_slice) else np.empty(0)
    val_forward = val_slice["forward_share"].to_numpy() if len(val_slice) else np.empty(0)

    snapshots = dict(snapshots)  # detach from the caller's mapping; reused for all trials

    def objective(trial: optuna.Trial) -> float:
        """Train one sampled config; report VAL AUC each epoch; return the best.

        Drives the SAME shared per-epoch step as ``train_hgt`` (via
        ``_setup_training`` / ``_train_one_epoch``) so the trial loss never drifts
        from the trainer's, but skips per-epoch checkpointing (only the winner is
        persisted, after the study). Each epoch's VAL AUC is reported to the
        MedianPruner; clearly-losing trials raise :class:`optuna.TrialPruned`.
        The fixed seed makes the trajectory depend only on the sampled
        hyperparameters.
        """
        lr = trial.suggest_float("lr", lr_lo, lr_hi, log=True)
        hidden = trial.suggest_categorical("hidden", hidden_choices)
        n_layers = trial.suggest_categorical("n_layers", n_layers_choices)
        dropout = trial.suggest_float("dropout", dropout_lo, dropout_hi)
        heads = trial.suggest_categorical("heads", heads_choices)

        # Defensive guard: heads must divide hidden (HGTConv requirement). Every
        # configured combo satisfies this ({1,2,4} | {32,64,128}), but a future
        # search-space edit could violate it — prune rather than crash the study.
        if hidden % heads != 0:
            raise optuna.TrialPruned()

        params: dict[str, Any] = {
            "conv_type": conv_type,
            "hidden": int(hidden),
            "n_layers": int(n_layers),
            "heads": int(heads),
            "dropout": float(dropout),
            "lr": float(lr),
            "seed": seed,
            "device": device,
            "pos_weight": pos_weight,
        }

        # Lightweight per-trial loop over the shared seams (no disk churn).
        fc = HGTForecaster(snapshots, **params, epochs=epochs)
        state = fc._setup_training(train_slice, train_labels)

        best_auc = float("nan")
        best_epoch = -1
        epochs_since_improve = 0
        n_epochs_run = 0
        for epoch in range(epochs):
            fc._train_one_epoch(state)
            n_epochs_run = epoch + 1

            val_auc = float("nan")
            if len(val_slice):
                val_auc = emergence_auc(fc.predict(val_slice).emergence_score, val_emergent)

            # NaN-safe best tracking: a single-class VAL slice yields nan AUC; never
            # let it become the "best" or get reported (it would poison the pruner).
            improved = val_auc == val_auc and (best_epoch < 0 or val_auc > best_auc)
            if improved:
                best_auc = val_auc
                best_epoch = epoch
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if val_auc == val_auc:  # finite AUC only -> safe to report/prune
                trial.report(val_auc, epoch)
                if trial.should_prune():
                    # Record what we have so the parquet row is complete even when
                    # the trial is pruned mid-run.
                    _set_trial_attrs(
                        trial, params, best_auc, float("nan"), best_epoch, n_epochs_run
                    )
                    raise optuna.TrialPruned()

            if epochs_since_improve >= patience:
                break

        # VAL share MAPE at the final state (a recorded diagnostic, not selected on).
        val_mape = float("nan")
        if len(val_slice):
            val_mape = share_mape(fc.predict(val_slice).share_forecast, val_forward)

        _set_trial_attrs(trial, params, best_auc, val_mape, best_epoch, n_epochs_run)
        # If the VAL slice was single-class every epoch, best_auc is nan; return
        # -inf so Optuna's maximize never selects it (it cannot be the best_trial).
        return best_auc if best_auc == best_auc else float("-inf")

    # --- Create-or-resume the study, then run only the remaining trials. -------
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()
    study_db.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{study_db}",
        study_name=_STUDY_NAME,
        load_if_exists=True,
    )
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, n_trials - n_complete)
    if remaining:
        study.optimize(objective, n_trials=remaining)

    # --- Assemble the one-row-per-trial sweep table. ---------------------------
    trial_rows = [_trial_row(t) for t in study.trials]
    sweep_df = pd.DataFrame(trial_rows, columns=list(_SWEEP_COLUMNS))
    sweep_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_parquet(sweep_parquet_path, index=False)

    # --- Select best by VAL AUC + retrain it onto hgt_best.pt (full checkpoint).
    best_trial = study.best_trial
    best_params = dict(best_trial.user_attrs.get("params", {}))
    # best_trial is COMPLETE, so _set_trial_attrs stored its metrics; fall back to
    # the (maximised) trial value, then nan, never to None (mypy + runtime safe).
    best_val_auc = float(
        best_trial.user_attrs.get(
            "best_val_auc",
            best_trial.value if best_trial.value is not None else float("nan"),
        )
    )
    best_val_mape = float(best_trial.user_attrs.get("val_mape", float("nan")))

    # ``checkpoint_path`` (hgt_best.pt) MUST receive the BEST-val-AUC weights, since
    # V1-S14 loads exactly this file to evaluate on the sealed test set for Gate G4.
    # ``train_hgt`` writes its ``checkpoint_path`` EVERY epoch but ``best_path`` only
    # on val-AUC improvement; pointing both at hgt_best.pt would let the final
    # (early-stop terminal) epoch overwrite the best one, persisting NON-val-selected
    # weights. So the per-epoch "latest" goes to a distinct sibling and the best goes
    # to hgt_best.pt — keeping train/select consistent with the pre-registered protocol.
    retrain_latest = checkpoint_path.with_name(f"{checkpoint_path.stem}.latest.pt")
    train_hgt(
        features=features,
        labels=labels,
        snapshots=snapshots,
        params=best_params,
        checkpoint_path=retrain_latest,
        best_path=checkpoint_path,
        max_epochs=epochs,
        patience=patience,
        val_eval=True,
        cfg_hash=_cfg_hash(cfg),
    )
    # On real data the first epoch's finite val-AUC always counts as an improvement,
    # so hgt_best.pt is written. Guard the degenerate single-class-VAL case (val-AUC
    # nan every epoch -> no "best" ever recorded) by falling back to the latest, so
    # the artifact always exists for the load-and-predict guarantee below.
    if not checkpoint_path.exists():
        shutil.copyfile(retrain_latest, checkpoint_path)
    # Confirm the persisted checkpoint is predict-ready in a fresh forecaster
    # (the plan's load-and-predict guarantee for hgt_best.pt).
    if len(val_slice):
        load_forecaster(checkpoint_path, snapshots).predict(val_slice)

    # --- Provenance sidecars (both artifacts reference the OSF DOI). -----------
    n_trials_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    summary_config: dict[str, Any] = {
        "preregistration": _prereg_dict(cfg),
        "study_name": _STUDY_NAME,
        "n_trials": int(n_trials_complete),
        "best_params": _to_jsonable(best_params),
        "best_val_auc": best_val_auc,
        "best_val_mape": best_val_mape,
    }
    record_run(
        artifact_path=sweep_parquet_path,
        inputs=inputs,
        config=_to_jsonable(summary_config),
    )
    record_run(
        artifact_path=checkpoint_path,
        inputs=inputs,
        config=_to_jsonable(summary_config),
    )

    return SweepResult(
        best_params=best_params,
        best_val_auc=best_val_auc,
        best_val_mape=best_val_mape,
        n_trials=int(n_trials_complete),
        sweep_parquet_path=sweep_parquet_path,
        checkpoint_path=checkpoint_path,
        trials=trial_rows,
    )


# --------------------------------------------------------------------------- #
# Trial <-> parquet glue
# --------------------------------------------------------------------------- #


def _set_trial_attrs(
    trial: optuna.Trial,
    params: dict[str, Any],
    best_val_auc: float,
    val_mape: float,
    best_epoch: int,
    n_epochs_run: int,
) -> None:
    """Stash the assembled per-trial result on ``trial`` for the parquet pass.

    Stored as Optuna ``user_attrs`` (which persist in the SQLite db) so the
    sweep table can be reassembled even from a resumed study where the in-memory
    objective locals are gone — and so the winning trial's full ``params`` dict
    is recoverable for the retrain-at-best.
    """
    trial.set_user_attr("params", _to_jsonable(params))
    trial.set_user_attr("best_val_auc", float(best_val_auc))
    trial.set_user_attr("val_mape", float(val_mape))
    trial.set_user_attr("best_epoch", int(best_epoch))
    trial.set_user_attr("n_epochs_run", int(n_epochs_run))


def _trial_row(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    """Build one sweep-parquet row from a (possibly resumed) frozen trial.

    Reads the searched params straight off ``trial.params`` and the recorded
    metrics off ``trial.user_attrs`` (set by :func:`_set_trial_attrs`), so a row
    is complete for COMPLETE and PRUNED trials alike. ``val_auc`` is the best VAL
    AUC seen that trial; ``state`` is the lower-cased Optuna trial state.
    """
    import optuna

    attrs = trial.user_attrs
    p = trial.params
    val_auc = attrs.get("best_val_auc")
    if val_auc is None and trial.value is not None and trial.value != float("-inf"):
        val_auc = float(trial.value)
    return {
        "trial_number": int(trial.number),
        "lr": float(p["lr"]) if "lr" in p else float("nan"),
        "hidden": int(p["hidden"]) if "hidden" in p else -1,
        "n_layers": int(p["n_layers"]) if "n_layers" in p else -1,
        "dropout": float(p["dropout"]) if "dropout" in p else float("nan"),
        "heads": int(p["heads"]) if "heads" in p else -1,
        "conv_type": str(attrs.get("params", {}).get("conv_type", "")),
        "seed": int(attrs.get("params", {}).get("seed", -1)),
        "val_auc": float(val_auc) if val_auc is not None else float("nan"),
        "val_mape": float(attrs.get("val_mape", float("nan"))),
        "best_epoch": int(attrs.get("best_epoch", -1)),
        "n_epochs_run": int(attrs.get("n_epochs_run", 0)),
        "state": (
            "pruned" if trial.state == optuna.trial.TrialState.PRUNED else trial.state.name.lower()
        ),
    }
