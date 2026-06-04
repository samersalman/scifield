"""V1-S12 ``no_graph`` baseline — the honest "GNN minus the graph" ablation.

:class:`NoGraphForecaster` reuses the EXACT shared trainer
(:class:`scifield.forecasting.baselines.mlp._TorchForecaster`) that
:class:`scifield.forecasting.baselines.mlp.MlpForecaster` uses — same shared
trunk, same emergence (BCE) + log-share (MSE) heads, same train-only
impute+scale pipeline, same determinism. The ONLY difference is the feature set:
this baseline consumes all 16 GNN node features (:data:`NODE_FEATURES`) — i.e.
every input the V1-S13 GNN's nodes will carry, but with no message passing over
edges. It is therefore the controlled comparison that isolates the contribution
of the graph structure: if the GNN cannot beat ``no_graph``, the edges add
nothing.
"""

from __future__ import annotations

from scifield.forecasting.baselines.mlp import _TorchForecaster
from scifield.forecasting.data import NODE_FEATURES

__all__ = ["NoGraphForecaster"]


class NoGraphForecaster(_TorchForecaster):
    """Two-head torch-CPU MLP over all 16 node features (``name = "no_graph"``).

    Identical trainer to :class:`scifield.forecasting.baselines.mlp.MlpForecaster`
    (imported :class:`_TorchForecaster`); differs only in consuming
    :data:`NODE_FEATURES` instead of ``MLP_FEATURES``. See
    :class:`_TorchForecaster` for the architecture, NaN handling, and determinism
    guarantees.

    Parameters
    ----------
    feature_columns:
        Override for the consumed feature columns; defaults to
        ``tuple(NODE_FEATURES)`` (so ``evaluate.py`` can pass
        ``cfg.features.node_columns``).
    hidden, epochs, lr, batch_size, seed:
        Hyperparameters from ``cfg.baselines.no_graph`` (defaults 32 / 50 / 0.01
        / 256 / 1729).
    """

    name = "no_graph"

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
            name="no_graph",
            feature_columns=(
                tuple(NODE_FEATURES) if feature_columns is None else tuple(feature_columns)
            ),
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
        )
