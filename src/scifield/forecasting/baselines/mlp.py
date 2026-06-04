"""V1-S12 torch-CPU MLP forecaster + the shared two-head trainer.

This module owns the *shared* learned-baseline trainer that both ``mlp`` and
``no_graph`` (the GNN-minus-graph ablation) reuse. The only thing that differs
between the two baselines is the **feature set**: ``MlpForecaster`` consumes the
9 block-A structural/temporal features (:data:`MLP_FEATURES`), while
``NoGraphForecaster`` (in :mod:`scifield.forecasting.baselines.no_graph`)
consumes all 16 GNN node features (:data:`NODE_FEATURES`). Both import the
trainer base class :class:`_TorchForecaster` from here so the architecture,
optimisation, NaN handling, and determinism are byte-identical.

Architecture
------------
A single shared trunk (``Linear(in_dim, hidden) -> ReLU``) feeds TWO heads:

``emergence head``
    ``Linear(hidden, 1)``, trained with ``BCEWithLogitsLoss`` against the binary
    ``emergent`` label. At predict time the logit is passed through ``sigmoid``
    to produce ``emergence_score`` (a probability in ``[0, 1]``; AUC is
    rank-based so the exact scale is immaterial).

``log-share head``
    ``Linear(hidden, 1)``, trained with MSE against ``log(forward_share + eps)``.
    Regressing in log space keeps the heavily right-skewed share target
    well-conditioned. At predict time the prediction is mapped back with
    ``exp(.) - eps`` and clipped to ``>= 0`` so ``share_forecast`` lives in the
    original share space (the MAPE target space).

The joint loss is the unweighted sum ``BCE + MSE``.

NaN handling
------------
The block-B trailing-novelty features (``sem_nov_mean_3yr``, ``cd5_3yr``,
``cd10_3yr``, ``cited_by_pctile_3yr``, ``rct_share_3yr``, ``review_share_3yr``)
can be NaN for empty trailing windows, ``n_journals_3yr`` can be 0.0, and some
block-A columns may be NaN too. Torch must never see a NaN, so every feature
matrix is passed through an sklearn pipeline
``make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())`` whose
imputation means and standardisation statistics are fit on the **training**
matrix only and then reused (transform-only) at predict time. A feature column
that is entirely NaN on the training slice imputes to 0 after standardisation
(``SimpleImputer`` drops it, ``StandardScaler`` cannot reintroduce it), which is
the desired "no information" behaviour rather than a crash.

Determinism
-----------
``fit`` seeds both ``torch`` (``torch.manual_seed(seed)``) and ``numpy``
(``np.random.seed(seed)``) before constructing the network and iterating, and
shuffles minibatches with a seeded ``numpy.random.Generator``. With a fixed
``seed`` and identical inputs, two ``fit().predict()`` runs produce bit-identical
arrays on CPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from scifield.forecasting.baselines.base import ForecastPrediction
from scifield.forecasting.data import MLP_FEATURES

__all__ = ["MlpForecaster", "_TorchForecaster"]

#: Small floor so the log-share target ``log(forward_share + _LOG_EPS)`` is finite
#: for zero-share rows; the inverse map subtracts it back before the ``>= 0`` clip.
_LOG_EPS: float = 1e-6


class _TwoHeadNet(nn.Module):
    """Shared trunk MLP with an emergence logit head and a log-share head."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.emergence_head = nn.Linear(hidden, 1)
        self.log_share_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(emergence_logit, log_share)``, each shape ``(n, 1)``."""
        h = self.trunk(x)
        return self.emergence_head(h), self.log_share_head(h)


class _TorchForecaster:
    """Reusable two-head torch-CPU forecaster parameterised by feature columns.

    Both :class:`MlpForecaster` and
    :class:`scifield.forecasting.baselines.no_graph.NoGraphForecaster` subclass
    this, differing only in the default ``feature_columns`` they pass up. It
    implements the :class:`scifield.forecasting.baselines.base.Predictor`
    protocol (it carries a ``name`` and the ``fit`` / ``predict`` pair).

    Parameters
    ----------
    name:
        The stable baseline name used as the metrics-table row key.
    feature_columns:
        Ordered feature column names this baseline reads from the input frames.
    hidden:
        Width of the shared trunk hidden layer. Default 32.
    epochs:
        Number of full passes over the training rows. Default 50.
    lr:
        Adam learning rate. Default 0.01.
    batch_size:
        Minibatch size; full-batch when the training slice is smaller. Default
        256.
    seed:
        RNG seed for ``torch`` + ``numpy`` (determinism). Default 1729.
    """

    name: str

    def __init__(
        self,
        name: str,
        feature_columns: tuple[str, ...],
        *,
        hidden: int = 32,
        epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 256,
        seed: int = 1729,
    ) -> None:
        self.name = name
        self.feature_columns = tuple(feature_columns)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        # Populated by ``fit``.
        self.pipeline_: Pipeline | None = None
        self.net_: _TwoHeadNet | None = None

    def _feature_matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Select the configured feature columns as a float64 numpy matrix.

        Missing values are left as NaN here — imputation is the pipeline's job,
        fit on the training slice only.
        """
        cols = list(self.feature_columns)
        return features.loc[:, cols].to_numpy(dtype=np.float64)

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> _TorchForecaster:
        """Fit the impute+scale pipeline and train the two-head net.

        Reads supervision from the LOCKED label columns: the binary emergence
        label from ``labels["emergent"]`` and the share target from
        ``labels["forward_share"]``. The imputation means and standardisation
        statistics are fit on the **training** feature matrix only.

        Parameters
        ----------
        features:
            Training feature rows (one per ``(topic_id, origin_year)``).
        labels:
            Aligned label frame carrying ``emergent`` and ``forward_share``.

        Returns
        -------
        _TorchForecaster
            ``self`` (for chaining).
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        x_raw = self._feature_matrix(features)

        # Impute (TRAIN means) then standardise — fit once, here, on TRAIN only.
        pipeline = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
        x_np = np.ascontiguousarray(pipeline.fit_transform(x_raw), dtype=np.float64)

        y_emergent = labels["emergent"].to_numpy(dtype=np.float64)
        forward_share = labels["forward_share"].to_numpy(dtype=np.float64)
        y_log_share = np.log(np.clip(forward_share, 0.0, None) + _LOG_EPS)

        x = torch.from_numpy(x_np).float()
        emergent_t = torch.from_numpy(y_emergent).float().unsqueeze(1)
        log_share_t = torch.from_numpy(y_log_share).float().unsqueeze(1)

        net = _TwoHeadNet(in_dim=x.shape[1], hidden=self.hidden)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        n = x.shape[0]
        batch = min(self.batch_size, n) if n else 0
        rng = np.random.default_rng(self.seed)

        net.train()
        for _ in range(self.epochs):
            if n == 0:
                break
            order = rng.permutation(n)
            for start in range(0, n, batch):
                idx = order[start : start + batch]
                xb = x[idx]
                logit, log_share_pred = net(xb)
                loss = bce(logit, emergent_t[idx]) + mse(log_share_pred, log_share_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.pipeline_ = pipeline
        self.net_ = net
        return self

    def predict(self, features: pd.DataFrame) -> ForecastPrediction:
        """Score ``features``; arrays align 1:1 with the input rows.

        Transforms the features through the stored (train-fit) impute+scale
        pipeline, runs the net, and returns ``emergence_score = sigmoid(logit)``
        and ``share_forecast = exp(log_share) - eps`` clipped to ``>= 0`` — both
        finite float arrays, same length and order as ``features``.

        Parameters
        ----------
        features:
            Feature rows to score.

        Returns
        -------
        ForecastPrediction
            ``emergence_score`` and ``share_forecast`` arrays.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.pipeline_ is None or self.net_ is None:
            raise RuntimeError(f"{self.name!r} predict() called before fit()")

        x_raw = self._feature_matrix(features)
        x_np = np.ascontiguousarray(self.pipeline_.transform(x_raw), dtype=np.float64)
        x = torch.from_numpy(x_np).float()

        self.net_.eval()
        with torch.no_grad():
            logit, log_share_pred = self.net_(x)

        emergence_score = torch.sigmoid(logit).squeeze(1).numpy().astype(np.float64)
        share_forecast = np.exp(log_share_pred.squeeze(1).numpy().astype(np.float64)) - _LOG_EPS
        share_forecast = np.clip(share_forecast, 0.0, None)
        return ForecastPrediction(emergence_score=emergence_score, share_forecast=share_forecast)


class MlpForecaster(_TorchForecaster):
    """Two-head torch-CPU MLP over the 9 block-A features (``name = "mlp"``).

    The plain learned baseline: shared trunk + emergence (BCE) and log-share
    (MSE) heads over :data:`MLP_FEATURES`. See :class:`_TorchForecaster` for the
    architecture, NaN handling, and determinism guarantees.

    Parameters
    ----------
    feature_columns:
        Override for the consumed feature columns; defaults to
        ``tuple(MLP_FEATURES)`` (so ``evaluate.py`` can pass
        ``cfg.features.mlp_columns``).
    hidden, epochs, lr, batch_size, seed:
        Hyperparameters from ``cfg.baselines.mlp`` (defaults 32 / 50 / 0.01 /
        256 / 1729).
    """

    name = "mlp"

    def __init__(
        self,
        feature_columns: tuple[str, ...] | None = None,
        *,
        hidden: int = 32,
        epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 256,
        seed: int = 1729,
    ) -> None:
        super().__init__(
            name="mlp",
            feature_columns=(
                tuple(MLP_FEATURES) if feature_columns is None else tuple(feature_columns)
            ),
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
        )
