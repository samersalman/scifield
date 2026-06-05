"""V1-S13 heterogeneous graph transformer (HGT) emergence/share forecaster.

This module is the **model** half of the V1-S13 GNN slice. It sits on top of the
leakage-safe snapshots from :mod:`scifield.forecasting.gnn.loader` and adds
citation / authorship / topic STRUCTURE to the *exact* 16 topic features the
``no_graph`` baseline (val AUC 0.701) already consumes. The pre-registered
V1-S14 ">5pp over no_graph" test (OSF DOI 10.17605/OSF.IO/XP94F) is then a clean
attribution: any lift over ``no_graph`` is attributable to message passing over
the graph, because everything else (the 16 Topic features, the two-head
emergence(BCE)+log-share(MSE) objective, the train-only impute+scale pipeline,
the determinism) is held byte-identical to the baseline.

Two pieces live here
--------------------
:class:`HGTNet`
    The :class:`torch.nn.Module`. A per-node-type input projection lifts the
    ragged per-type feature dims (Topic 16, Paper 2, entities 1) to a common
    ``hidden`` width, then ``n_layers`` of :class:`~torch_geometric.nn.HGTConv`
    (or a pure-Python :class:`~torch_geometric.nn.SAGEConv`-based
    :class:`~torch_geometric.nn.HeteroConv` fallback) propagate information over
    the 9 (post-``ToUndirected``) relations, each wrapped in
    ``residual + LayerNorm + ReLU + dropout``. Two linear heads read out of the
    ``Topic`` embeddings: an emergence logit (BCE) and a log-share (MSE).

:class:`HGTForecaster`
    The :class:`scifield.forecasting.baselines.base.Predictor` adapter. It owns
    the train-only impute+scale pipeline, the ``pos_weight`` for the imbalanced
    BCE, the per-origin-year snapshot bookkeeping, and the ``fit`` / ``predict``
    pair that plugs into the V1-S12 evaluation harness unchanged.

Why a per-type input ``Linear`` and NO per-node embeddings
----------------------------------------------------------
The five node types carry different input widths (Topic=16, Paper=2,
Author/Institution/Journal=1; see :mod:`scifield.forecasting.gnn.loader`), so
``HGTConv`` — which wants a single ``hidden`` width — needs a per-type
``Linear(in_dim_τ -> hidden)`` front end. We deliberately do **not** add
per-node (id) embeddings: with only ~55 positive training rows, learned node
identities would let the model memorise specific topics/papers (an identity
leak / overfit) rather than learn a transferable structural signal. The entity
nodes therefore carry a constant ``ones(n, 1)`` feature (capacity comes purely
from their input ``Linear``).

The ablation identity (critical for Gate G4 honesty)
----------------------------------------------------
``n_layers == 0`` (equivalently ``ablate_edges=True``) collapses the model to a
two-head MLP over ONLY the 16 ``Topic`` features: it skips the entire
``HGTConv`` stack AND every non-``Topic`` node/edge, so the forward pass is
``Linear(16 -> hidden) -> ReLU -> {emergence head, log-share head}`` —
architecturally equivalent to ``no_graph``'s ``_TwoHeadNet``. This is enforced
to *genuinely* ignore the graph: perturbing edges or other node types must not
change the ablation's output (the rigorous edge-invariance test in
``tests/test_forecasting_gnn_model.py``). Without this guarantee the V1-S14
comparison would be confounded — the "graph" arm and the "no graph" arm would
not share an architecture.

Snapshot-injection contract
----------------------------
:class:`HGTForecaster` is constructed with an *injected* mapping
``{origin_year -> HeteroData}`` rather than a directory, so it is testable with
tiny hand-built snapshots (see :func:`build_snapshot_from_frames`) and decoupled
from disk. The CLI / hyperparameter sweep loads snapshots via
:func:`scifield.forecasting.gnn.loader.load_snapshot` into a dict and passes it
in. The snapshots arrive **forward-edge-only** (the loader does not symmetrise);
this module applies :class:`~torch_geometric.transforms.ToUndirected` itself and
caches the transformed result per year so a snapshot is never double-transformed.

Determinism & threading
------------------------
``fit`` seeds ``torch`` and ``numpy`` and pins torch to a single intra-op thread
(:func:`scifield.forecasting.baselines.evaluate._pin_torch_single_threaded`'s
``torch.set_num_threads(1)`` — the Darwin libomp segfault guard). With a fixed
``seed`` and identical injected snapshots, two ``fit().predict()`` runs produce
bit-identical arrays on CPU.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.nn import HeteroConv, HGTConv, SAGEConv
from torch_geometric.transforms import ToUndirected

from scifield.forecasting.baselines.base import ForecastPrediction
from scifield.forecasting.data import NODE_FEATURES
from scifield.forecasting.gnn.loader import topic_id_to_index

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

__all__ = ["HGTForecaster", "HGTNet", "TrainingState"]

#: Small floor so the log-share target ``log(forward_share + _LOG_EPS)`` is finite
#: for zero-share rows; the inverse map subtracts it back before the ``>= 0`` clip.
#: Identical to :data:`scifield.forecasting.baselines.mlp._LOG_EPS` so the share
#: head behaves byte-for-byte like the baseline's.
_LOG_EPS: float = 1e-6

#: The single node type the model reads out of (and, in the ablation, the only
#: node type it touches at all). Its input width is ``len(NODE_FEATURES) == 16``.
_TOPIC: str = "Topic"


def _pin_torch_single_threaded() -> None:
    """Pin torch to one intra-op thread (Darwin libomp segfault guard).

    Mirrors :func:`scifield.forecasting.baselines.evaluate._pin_torch_single_threaded`
    and :func:`scifield.forecasting.gnn.loader._pin_torch_single_threaded`: the
    default multi-threaded intra-op pool races libomp on this macOS box, so every
    entry point that touches torch must pin to one thread. Idempotent.
    """
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)


class HGTNet(nn.Module):
    """Heterogeneous graph transformer with two readout heads on ``Topic`` nodes.

    The forward pass has three stages:

    1. **Per-type input projection.** A ``Linear(in_dim_τ -> hidden) -> ReLU`` per
       node type lifts the ragged per-type widths to a common ``hidden`` size
       (so :class:`~torch_geometric.nn.HGTConv` sees one width). ``in_dim_τ`` is
       read from the ``metadata``-derived ``in_channels`` mapping.
    2. **Message passing.** ``n_layers`` heterogeneous convs over the 9
       (post-``ToUndirected``) relations, each wrapped per node type in
       ``h = LayerNorm(h + dropout(ReLU(conv(h))))`` — a pre-residual block that
       keeps the signal stable as depth grows on the tiny training set. With
       ``conv_type="hgt"`` the conv is :class:`~torch_geometric.nn.HGTConv`; with
       ``"sage"`` it is a :class:`~torch_geometric.nn.HeteroConv` of one
       :class:`~torch_geometric.nn.SAGEConv` per relation (a guaranteed
       pure-Python fallback with the identical residual/LN/ReLU/dropout wrapper).
    3. **Readout.** Two ``Linear(hidden, 1)`` heads on the ``Topic`` embeddings:
       ``emergence_head`` (a logit for ``BCEWithLogitsLoss``) and
       ``log_share_head`` (a regression onto ``log(forward_share + eps)``).

    Ablation (``n_layers == 0``)
        The constructor builds ONLY the ``Topic`` input projection and the two
        heads. :meth:`forward` then ignores ``edge_index_dict`` and every
        non-``Topic`` entry of ``x_dict`` entirely — the model is exactly
        ``Linear(16 -> hidden) -> ReLU -> 2 heads``, the same shape as
        ``no_graph``'s shared-trunk ``_TwoHeadNet``. This makes the GNN-minus-graph
        ablation architecturally honest: it cannot read the graph even in
        principle.

    Parameters
    ----------
    metadata:
        The PyG ``(node_types, edge_types)`` tuple, taken from
        ``data.metadata()`` **after** ``ToUndirected`` (so ``edge_types`` is the
        9-relation set, not the 5 forward triplets).
    in_channels:
        ``{node_type -> input feature width}``. ``Topic`` must be 16.
    hidden:
        Common hidden width. ``heads`` must divide it.
    n_layers:
        Number of conv layers. ``0`` selects the MLP-over-``Topic`` ablation.
    heads:
        Attention heads for ``HGTConv`` (ignored by the SAGE fallback). Must
        divide ``hidden``.
    dropout:
        Dropout probability inside each conv's residual block.
    conv_type:
        ``"hgt"`` (default) or ``"sage"`` (pure-Python fallback).
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels: Mapping[str, int],
        *,
        hidden: int = 64,
        n_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.2,
        conv_type: str = "hgt",
    ) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"heads ({heads}) must divide hidden ({hidden})")
        if conv_type not in ("hgt", "sage"):
            raise ValueError(f"conv_type must be 'hgt' or 'sage', got {conv_type!r}")

        self.metadata = (list(metadata[0]), list(metadata[1]))
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.conv_type = conv_type
        self.ablate = self.n_layers <= 0

        if _TOPIC not in in_channels:
            raise ValueError(f"in_channels must contain {_TOPIC!r}; got {sorted(in_channels)}")

        if self.ablate:
            # Ablation: ONLY the Topic input projection + the two heads. No conv
            # stack, no other node types — architecturally == no_graph's trunk.
            self.input_proj = nn.ModuleDict(
                {_TOPIC: nn.Linear(int(in_channels[_TOPIC]), self.hidden)}
            )
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
        else:
            # Full model: one input Linear per node type, then the conv stack.
            self.input_proj = nn.ModuleDict(
                {nt: nn.Linear(int(in_channels[nt]), self.hidden) for nt in self.metadata[0]}
            )
            self.convs = nn.ModuleList([self._make_conv() for _ in range(self.n_layers)])
            # A per-(layer, node_type) LayerNorm for the residual blocks.
            self.norms = nn.ModuleList(
                [
                    nn.ModuleDict({nt: nn.LayerNorm(self.hidden) for nt in self.metadata[0]})
                    for _ in range(self.n_layers)
                ]
            )

        self.dropout_layer = nn.Dropout(self.dropout)
        self.emergence_head = nn.Linear(self.hidden, 1)
        self.log_share_head = nn.Linear(self.hidden, 1)

    def _make_conv(self) -> nn.Module:
        """Build one heterogeneous conv layer of the configured ``conv_type``.

        Both branches map ``hidden -> hidden`` over every relation in
        ``self.metadata`` so the residual add in :meth:`forward` is shape-safe.
        """
        # PyG's conv constructors are loosely typed (mypy infers ``Any``), but
        # every conv IS an ``nn.Module``; the cast records that without a blanket
        # ignore.
        conv: nn.Module
        if self.conv_type == "hgt":
            conv = HGTConv(self.hidden, self.hidden, self.metadata, heads=self.heads)
        else:
            # SAGE fallback: one SAGEConv per relation, summed at the destination.
            conv = HeteroConv(
                {rel: SAGEConv(self.hidden, self.hidden) for rel in self.metadata[1]},
                aggr="sum",
            )
        return conv

    def _topic_embedding(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        """Return the ``Topic`` node embeddings after input projection + convs.

        In the ablation this is just ``ReLU(Linear(Topic.x))`` — ``edge_index_dict``
        and all non-``Topic`` features are ignored. In the full model it is the
        ``Topic`` slice of the conv stack's output.
        """
        if self.ablate:
            # Touch ONLY Topic.x: the graph is invisible to the ablation.
            return torch.relu(self.input_proj[_TOPIC](x_dict[_TOPIC]))

        # Project every node type up to `hidden`, then run the residual conv stack.
        h_dict = {nt: torch.relu(self.input_proj[nt](x_dict[nt])) for nt in self.metadata[0]}
        for conv, norm in zip(self.convs, self.norms, strict=True):
            out = conv(h_dict, edge_index_dict)
            new_h: dict[str, torch.Tensor] = {}
            for nt, h in h_dict.items():
                # A conv may omit a node type with no incoming edges; keep its
                # previous representation (residual identity) in that case.
                delta = out.get(nt)
                if delta is None:
                    new_h[nt] = h
                    continue
                # Pre-residual block: h <- LayerNorm(h + dropout(ReLU(conv(h)))).
                # ``norm`` is yielded by iterating a ``ModuleList``, so it is
                # statically the base ``Module`` (not subscriptable); cast back to
                # the ``ModuleDict`` it is at runtime so ``[nt]`` type-checks under
                # both the dev (mypy 2.1) and pinned pre-commit (mypy 1.11) versions.
                norm_dict = cast(nn.ModuleDict, norm)
                new_h[nt] = norm_dict[nt](h + self.dropout_layer(torch.relu(delta)))
            h_dict = new_h
        return h_dict[_TOPIC]

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(emergence_logit, log_share)`` for every ``Topic`` node.

        Both outputs have shape ``(n_topic, 1)``; the caller gathers the rows for
        the topics present in a given snapshot year.
        """
        topic_h = self._topic_embedding(x_dict, edge_index_dict)
        return self.emergence_head(topic_h), self.log_share_head(topic_h)


@dataclass
class TrainingState:
    """Per-fit handles shared by :meth:`HGTForecaster._setup_training` and
    :meth:`HGTForecaster._train_one_epoch`.

    Extracting the epoch body into :meth:`HGTForecaster._train_one_epoch` lets
    BOTH the all-in-one :meth:`HGTForecaster.fit` *and* the external
    checkpoint-resume training loop (``scifield.forecasting.train``) drive the
    *identical* optimisation step, so a model trained by the Optuna sweep and one
    rebuilt from a checkpoint cannot silently diverge. This struct carries exactly
    the (non-RNG) handles that step needs; the RNG itself lives on torch/numpy
    global state (the only stochastic op per epoch is dropout, so the resume loop
    checkpoints ``torch.get_rng_state()`` to make a resumed trajectory bitwise
    identical to an uninterrupted one).

    Attributes
    ----------
    optimizer:
        The Adam optimiser over ``net_.parameters()`` (its momentum buffers are
        part of the resumable state — saved/restored alongside the weights).
    bce:
        The emergence ``BCEWithLogitsLoss`` carrying the resolved ``pos_weight_``.
    mse:
        The log-share regression ``MSELoss``.
    x_scaled:
        ``(n_rows, 16)`` train-only pipeline-transformed feature matrix, aligned
        positionally to ``year_groups`` / the label arrays.
    y_emergent:
        ``(n_rows,)`` binary emergence targets (float), positionally aligned.
    y_log_share:
        ``(n_rows,)`` ``log(forward_share + eps)`` regression targets, aligned.
    topic_id:
        ``(n_rows,)`` topic id per row (to scatter onto ``Topic`` nodes).
    year_groups:
        ``{origin_year -> positional row indices}`` (see
        :meth:`HGTForecaster._group_positions`).
    """

    optimizer: torch.optim.Optimizer
    bce: nn.Module
    mse: nn.Module
    x_scaled: np.ndarray
    y_emergent: np.ndarray
    y_log_share: np.ndarray
    topic_id: np.ndarray
    year_groups: dict[int, np.ndarray]


class HGTForecaster:
    """HGT emergence/share forecaster implementing the :class:`Predictor` protocol.

    Drops into the V1-S12 evaluation harness exactly like ``no_graph``: it carries
    a stable ``name`` (``"hgt"``) and the ``fit`` / ``predict`` pair, reads
    supervision from ``labels["emergent"]`` / ``labels["forward_share"]``, and
    returns a :class:`ForecastPrediction` aligned **1:1 with the input rows**.
    The only structural difference from the baseline is that scoring a
    ``(topic_id, origin_year)`` row requires that year's graph snapshot, so the
    forecaster is constructed with an injected ``{origin_year -> HeteroData}``
    mapping (see the module docstring's snapshot-injection contract).

    Training proceeds per origin year: for each year present in the TRAIN frame
    the corresponding snapshot's ``Topic.x`` is overwritten with the scaled
    16-feature panel rows (placed at the right ``Topic`` nodes via
    :func:`topic_id_to_index`), the whole snapshot is forwarded once, the
    ``Topic`` outputs for the rows present that year are gathered, and the
    per-snapshot ``BCE(pos_weight) + MSE`` readout loss is backpropagated to
    accumulate gradients; a single optimizer step per epoch then applies the
    joint update over all years (gradient accumulation — equivalent to summing
    the per-year losses and backpropagating once, but with peak memory bounded
    by one snapshot's graph). ``pos_weight`` defaults to ``"auto"``
    (``n_neg / n_pos`` from the TRAIN labels, ≈22.6 on real data) so the heavily
    imbalanced emergence head is not swamped by the negative class.

    Parameters
    ----------
    snapshots:
        ``{origin_year -> HeteroData}`` forward-edge-only snapshots (the loader's
        output). ``ToUndirected`` is applied internally and cached per year.
    conv_type:
        ``"hgt"`` (default) or ``"sage"`` (pure-Python fallback).
    hidden:
        Hidden width (default 64). ``heads`` must divide it.
    n_layers:
        Conv depth (default 2). ``0`` selects the MLP-over-``Topic`` ablation.
    heads:
        ``HGTConv`` attention heads (default 2).
    dropout:
        Dropout inside the residual blocks (default 0.2).
    lr:
        Adam learning rate (default 1e-3).
    epochs:
        Full-graph optimisation steps (default 200).
    ablate_edges:
        Force the ablation regardless of ``n_layers`` (default ``False``).
    pos_weight:
        ``"auto"`` (default, ``n_neg/n_pos`` from TRAIN) or an explicit float for
        the emergence ``BCEWithLogitsLoss``.
    seed:
        RNG seed for ``torch`` + ``numpy`` (determinism; default 1729).
    device:
        Torch device string (default ``"cpu"``; ``"mps"`` is accepted but
        untested).
    feature_columns:
        Ordered ``Topic`` feature columns; defaults to ``tuple(NODE_FEATURES)``
        (the 16, in order — must match the loader's ``Topic`` width).
    """

    name: str = "hgt"

    def __init__(
        self,
        snapshots: Mapping[int, HeteroData],
        *,
        conv_type: str = "hgt",
        hidden: int = 64,
        n_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 200,
        ablate_edges: bool = False,
        pos_weight: float | str = "auto",
        seed: int = 1729,
        device: str = "cpu",
        feature_columns: tuple[str, ...] = tuple(NODE_FEATURES),
    ) -> None:
        self.snapshots = dict(snapshots)
        self.conv_type = conv_type
        self.hidden = int(hidden)
        # ablate_edges OR n_layers==0 both collapse to the MLP ablation.
        self.n_layers = 0 if ablate_edges else int(n_layers)
        self.ablate_edges = bool(ablate_edges) or self.n_layers <= 0
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.pos_weight = pos_weight
        self.seed = int(seed)
        self.device = torch.device(device)
        self.feature_columns = tuple(feature_columns)

        # Cache of ToUndirected-transformed snapshots, keyed by origin year, so a
        # snapshot is symmetrised at most once across fit + every predict call.
        self._undirected: dict[int, HeteroData] = {}

        # Populated by ``fit``.
        self.pipeline_: Pipeline | None = None
        self.net_: HGTNet | None = None
        self.pos_weight_: float | None = None

    # ----------------------------------------------------------------- helpers

    def _feature_matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Select the configured ``Topic`` feature columns as a float64 matrix.

        Missing values are left as NaN — imputation is the train-fit pipeline's
        job (mirrors :meth:`scifield.forecasting.baselines.mlp._TorchForecaster._feature_matrix`).
        """
        matrix = features.loc[:, list(self.feature_columns)].to_numpy(dtype=np.float64)
        return np.asarray(matrix, dtype=np.float64)

    def _undirected_snapshot(self, year: int) -> HeteroData:
        """Return the ``ToUndirected`` snapshot for ``year`` (built once, cached).

        The loader emits forward-edge-only snapshots; ``ToUndirected`` adds the
        ``rev_*`` relations (and symmetrises ``CITES`` in place) that message
        passing into ``Topic`` / ``Author`` / ``Institution`` / ``Journal``
        needs. We ``copy.copy`` before transforming so the caller's injected
        snapshot is never mutated. Idempotent per year via the cache.
        """
        cached = self._undirected.get(year)
        if cached is not None:
            return cached
        if year not in self.snapshots:
            raise KeyError(f"no snapshot injected for origin_year {year}")
        data = ToUndirected()(copy.copy(self.snapshots[year]))
        self._undirected[year] = data
        return data

    def _build_net(self) -> HGTNet:
        """Construct the :class:`HGTNet` from a representative snapshot's metadata.

        The per-type ``in_channels`` and the (post-``ToUndirected``) ``metadata``
        are read from any one injected snapshot — they are identical across years
        (same node types, same widths, same relation set). Exposed so a future
        train-loop owner (T4) can rebuild an architecturally-identical net for
        checkpoint resume.

        Raises
        ------
        ValueError
            If no snapshots were injected, or a snapshot's ``Topic`` width does
            not match ``len(feature_columns)``.
        """
        if not self.snapshots:
            raise ValueError("HGTForecaster requires at least one injected snapshot")
        any_year = next(iter(sorted(self.snapshots)))
        data = self._undirected_snapshot(any_year)
        in_channels = {nt: int(data[nt].x.shape[1]) for nt in data.node_types}
        if in_channels.get(_TOPIC) != len(self.feature_columns):
            raise ValueError(
                f"snapshot Topic width {in_channels.get(_TOPIC)} != "
                f"len(feature_columns) {len(self.feature_columns)}"
            )
        net = HGTNet(
            data.metadata(),
            in_channels,
            hidden=self.hidden,
            n_layers=self.n_layers,
            heads=self.heads,
            dropout=self.dropout,
            conv_type=self.conv_type,
        )
        # ``.to`` returns ``Self`` at runtime but is stub-typed as ``Any``; move in
        # place and return the typed local so the ``-> HGTNet`` contract is exact.
        net.to(self.device)
        return net

    def _scaled_topic_x(
        self, data: HeteroData, scaled_rows: np.ndarray, topic_ids: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Place ``scaled_rows`` onto ``data``'s ``Topic`` nodes; return ``(x, gather)``.

        Builds a ``(n_topic, 16)`` ``Topic.x`` tensor that is zero everywhere
        except the nodes named by ``topic_ids`` (mapped to node indices via
        :func:`topic_id_to_index`), where it holds the matching scaled feature
        row. Topics not present in this year's rows keep the loader's all-zeros
        placeholder (they contribute structure but no readout). Also returns the
        ``gather`` index tensor — the ``Topic`` node index of each input row, in
        row order — so the caller can pull the readout back in the right order.

        Parameters
        ----------
        data:
            The (ToUndirected) snapshot whose ``Topic`` nodes to fill.
        scaled_rows:
            ``(n_rows, 16)`` pipeline-transformed feature rows for this year.
        topic_ids:
            ``(n_rows,)`` topic id per row (aligned to ``scaled_rows``).
        """
        id_to_idx = topic_id_to_index(data)
        n_topic = int(data[_TOPIC].num_nodes or len(id_to_idx))
        topic_x = torch.zeros(n_topic, scaled_rows.shape[1], dtype=torch.float32)
        gather = np.empty(len(topic_ids), dtype=np.int64)
        for r, gid in enumerate(topic_ids):
            node = id_to_idx[int(gid)]
            topic_x[node] = torch.from_numpy(scaled_rows[r]).float()
            gather[r] = node
        return topic_x.to(self.device), torch.from_numpy(gather).to(self.device)

    def _x_dict(self, data: HeteroData, topic_x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build the ``x_dict`` for a forward pass: ``Topic`` overwritten, rest as-is.

        Non-``Topic`` features come straight off the snapshot (constant for
        entities, ``[year, log1p(in-degree)]`` for papers); only ``Topic.x`` is
        replaced with the scaled panel. In the ablation the net ignores every key
        but ``Topic``, but we still pass the full dict so the same code path
        serves both modes.
        """
        x_dict = {nt: data[nt].x.to(self.device) for nt in data.node_types}
        x_dict[_TOPIC] = topic_x
        return x_dict

    # --------------------------------------------------------------------- fit

    def _setup_training(self, features: pd.DataFrame, labels: pd.DataFrame) -> TrainingState:
        """Do everything :meth:`fit` does BEFORE its epoch loop; return the handles.

        Pins torch to one thread, seeds ``torch`` + ``numpy`` ONCE (the per-epoch
        loop's only stochastic op is dropout, so this single seeding fully
        determines the trajectory), fits the train-only impute+scale pipeline,
        resolves and stores ``self.pos_weight_``, builds ``self.net_`` (an
        architecturally-fixed :class:`HGTNet`) and the Adam optimiser, the
        BCE(pos_weight)+MSE criteria, and the per-year scaled-tensor/gather cache.
        Assigning ``self.net_`` / ``self.pipeline_`` / ``self.pos_weight_`` HERE
        (rather than after the loop, as the monolithic ``fit`` once did) is what
        lets the external checkpoint-resume loop call ``predict`` for a per-epoch
        validation AUC mid-training and lets it persist/restore the pipeline —
        which **must be restored, never refit at predict time**, because refitting
        the imputer means / standardisation statistics on the prediction slice
        would leak that slice's distribution into the features (the exact leak the
        train-only pipeline exists to prevent).

        This is the single source of truth for fit's pre-loop setup: ``fit`` and
        ``scifield.forecasting.train.train_hgt`` both call it, so the model the
        Optuna sweep trains and the one rebuilt from a checkpoint share identical
        initialisation, optimiser, and supervision wiring.

        Parameters
        ----------
        features:
            Training feature rows; must carry ``topic_id``, ``origin_year`` and
            the configured ``feature_columns``.
        labels:
            Aligned label frame carrying ``emergent`` and ``forward_share``.

        Returns
        -------
        TrainingState
            The (non-RNG) handles :meth:`_train_one_epoch` consumes.
        """
        _pin_torch_single_threaded()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # --- Train-only impute+scale pipeline (identical to the baseline's). ---
        x_raw = self._feature_matrix(features)
        pipeline = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
        x_scaled = np.ascontiguousarray(pipeline.fit_transform(x_raw), dtype=np.float64)

        y_emergent = labels["emergent"].to_numpy(dtype=np.float64)
        forward_share = labels["forward_share"].to_numpy(dtype=np.float64)
        y_log_share = np.log(np.clip(forward_share, 0.0, None) + _LOG_EPS)

        # --- pos_weight = n_neg / n_pos from TRAIN (auto), else the given float. ---
        if isinstance(self.pos_weight, str):
            if self.pos_weight != "auto":
                raise ValueError(f"pos_weight str must be 'auto', got {self.pos_weight!r}")
            n_pos = float((y_emergent == 1).sum())
            n_neg = float((y_emergent == 0).sum())
            # Guard a degenerate single-class TRAIN slice (no positives -> weight 1).
            self.pos_weight_ = (n_neg / n_pos) if n_pos > 0 else 1.0
        else:
            self.pos_weight_ = float(self.pos_weight)

        net = self._build_net()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        pos_weight_t = torch.tensor([self.pos_weight_], dtype=torch.float32, device=self.device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        mse = nn.MSELoss()

        # Group the row positions by origin year ONCE (values stay aligned to the
        # scaled matrix / label arrays by positional index).
        origin_year = features["origin_year"].to_numpy()
        topic_id = features["topic_id"].to_numpy()
        year_groups = self._group_positions(origin_year)

        self.pipeline_ = pipeline
        self.net_ = net
        return TrainingState(
            optimizer=optimizer,
            bce=bce,
            mse=mse,
            x_scaled=x_scaled,
            y_emergent=y_emergent,
            y_log_share=y_log_share,
            topic_id=topic_id,
            year_groups=year_groups,
        )

    def _train_one_epoch(self, state: TrainingState) -> float:
        """Run ONE full-graph optimisation step; return the scalar train loss.

        The body of the original ``for _ in range(self.epochs)`` loop: one
        ``zero_grad`` → for each origin-year snapshot, a single forward pass and
        its ``BCE(pos_weight) + MSE`` readout loss is backpropagated immediately
        (gradient accumulation) → one ``step``. Accumulating per-snapshot rather
        than summing all years into one tensor before a single ``backward`` gives
        the identical joint gradient (∂/∂w Σ Lᵧ = Σ ∂Lᵧ/∂w) but frees each year's
        autograd graph as it goes, bounding peak memory to the single largest
        snapshot (the joint-backward form OOMs on the full corpus). It puts
        ``self.net_`` into ``train()`` mode first so dropout is
        active (the lone stochastic op, hence the RNG-state checkpoint in the
        resume loop). Sharing this exact step between :meth:`fit` and
        ``train_hgt`` is what guarantees a sweep-trained and a checkpoint-trained
        model optimise identically.

        Parameters
        ----------
        state:
            The handles from :meth:`_setup_training` (optimizer, criteria, the
            scaled matrix, targets, and the per-year grouping).

        Returns
        -------
        float
            The summed train loss for this epoch (for history / logging).
        """
        assert self.net_ is not None  # set by _setup_training
        net = self.net_
        net.train()
        state.optimizer.zero_grad()
        total_loss = 0.0
        for year, pos in state.year_groups.items():
            data = self._undirected_snapshot(year)
            topic_x, gather = self._scaled_topic_x(data, state.x_scaled[pos], state.topic_id[pos])
            x_dict = self._x_dict(data, topic_x)
            logit_all, log_share_all = net(x_dict, data.edge_index_dict)
            logit = logit_all[gather]
            log_share_pred = log_share_all[gather]
            emergent_t = (
                torch.from_numpy(state.y_emergent[pos]).float().unsqueeze(1).to(self.device)
            )
            log_share_t = (
                torch.from_numpy(state.y_log_share[pos]).float().unsqueeze(1).to(self.device)
            )
            loss = state.bce(logit, emergent_t) + state.mse(log_share_pred, log_share_t)
            # Backprop per snapshot and ACCUMULATE into ``.grad`` rather than summing
            # all years into one tensor and calling ``backward`` once. The summed
            # gradient is identical either way (∂/∂w Σᵧ Lᵧ = Σᵧ ∂Lᵧ/∂w), but doing it
            # per-snapshot frees each year's (large) autograd graph immediately, so
            # peak memory is bounded by the single largest snapshot's forward graph
            # instead of the sum over all ~20 train years. On the full corpus the
            # largest year is ~82k Paper nodes / ~2.4M edges (post-ToUndirected);
            # the joint-backward form OOM-kills a Mac, the accumulating form does not.
            # Only the optimizer ``step`` happens once per epoch, so the update is
            # the same joint step over all years.
            loss.backward()
            total_loss += float(loss.detach().cpu().item())
        state.optimizer.step()
        return total_loss

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> HGTForecaster:
        """Fit the train-only impute+scale pipeline and train the HGT jointly.

        Reads supervision from the LOCKED label columns ``labels["emergent"]``
        (binary) and ``labels["forward_share"]`` (share target). The impute means
        and standardisation statistics are fit on the **training** ``Topic``
        feature matrix only; ``pos_weight`` (when ``"auto"``) is ``n_neg/n_pos``
        over the TRAIN ``emergent`` column and is stored on ``self.pos_weight_``.
        Optimisation is full-graph per year: every TRAIN origin year contributes
        one snapshot forward pass per epoch whose readout loss is backpropagated
        immediately to accumulate gradients, with a single optimizer step per
        epoch applying the joint update (gradient accumulation — same gradient as
        summing the per-year losses, but peak memory is bounded by one snapshot).

        Implemented as :meth:`_setup_training` once then :meth:`_train_one_epoch`
        ``self.epochs`` times — the SAME two seams the external checkpoint-resume
        loop drives — so this method's observable behaviour is byte-identical to
        the historical monolithic loop (two same-seed fits still produce
        bit-identical predictions).

        Parameters
        ----------
        features:
            Training feature rows; must carry ``topic_id``, ``origin_year`` and
            the configured ``feature_columns``.
        labels:
            Aligned label frame carrying ``emergent`` and ``forward_share``.

        Returns
        -------
        HGTForecaster
            ``self`` (for chaining).
        """
        state = self._setup_training(features, labels)
        for _ in range(self.epochs):
            self._train_one_epoch(state)
        return self

    @staticmethod
    def _group_positions(origin_year: np.ndarray) -> dict[int, np.ndarray]:
        """Map ``origin_year -> the positional row indices`` with that year.

        Built off an explicit ``np.arange`` so the returned positions index the
        scaled feature matrix / label arrays directly — never relying on pandas
        groupby preserving order. The same primitive backs the row-alignment
        guarantee in :meth:`predict`.
        """
        order = np.arange(len(origin_year))
        groups: dict[int, list[int]] = {}
        for pos, year in zip(order, origin_year, strict=True):
            groups.setdefault(int(year), []).append(int(pos))
        return {year: np.asarray(p, dtype=np.int64) for year, p in groups.items()}

    # ----------------------------------------------------------------- predict

    def predict(self, features: pd.DataFrame) -> ForecastPrediction:
        """Score ``features``; arrays align 1:1 with the input rows (same order).

        For each origin year the corresponding snapshot is forwarded once, the
        ``Topic`` outputs for that year's rows are gathered, and the results are
        **scattered back into the exact input row positions** using an explicit
        positional index (``np.arange`` grouped by year) — never assuming groupby
        preserves order. Returns ``emergence_score = sigmoid(logit)`` and
        ``share_forecast = clip(exp(log_share) - eps, 0, None)``, both finite
        float arrays of length ``len(features)``.

        Parameters
        ----------
        features:
            Feature rows to score; must carry ``topic_id`` and ``origin_year``.

        Returns
        -------
        ForecastPrediction
            ``emergence_score`` and ``share_forecast``, aligned to ``features``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.pipeline_ is None or self.net_ is None:
            raise RuntimeError("'hgt' predict() called before fit()")
        _pin_torch_single_threaded()

        n = len(features)
        x_raw = self._feature_matrix(features)
        x_scaled = np.ascontiguousarray(self.pipeline_.transform(x_raw), dtype=np.float64)
        origin_year = features["origin_year"].to_numpy()
        topic_id = features["topic_id"].to_numpy()
        year_groups = self._group_positions(origin_year)

        emergence_score = np.empty(n, dtype=np.float64)
        share_forecast = np.empty(n, dtype=np.float64)

        self.net_.eval()
        with torch.no_grad():
            for year, pos in year_groups.items():
                data = self._undirected_snapshot(year)
                topic_x, gather = self._scaled_topic_x(data, x_scaled[pos], topic_id[pos])
                x_dict = self._x_dict(data, topic_x)
                logit_all, log_share_all = self.net_(x_dict, data.edge_index_dict)
                logit = logit_all[gather].squeeze(1).cpu().numpy().astype(np.float64)
                log_share = log_share_all[gather].squeeze(1).cpu().numpy().astype(np.float64)
                # Scatter into the EXACT input row positions for this year.
                emergence_score[pos] = 1.0 / (1.0 + np.exp(-logit))
                share_forecast[pos] = np.clip(np.exp(log_share) - _LOG_EPS, 0.0, None)

        return ForecastPrediction(emergence_score=emergence_score, share_forecast=share_forecast)


def _assert_predictor_protocol() -> None:
    """Static reminder that :class:`HGTForecaster` satisfies :class:`Predictor`.

    Not executed at import; exists so a type checker / reader can see the
    intended runtime ``isinstance(HGTForecaster(...), Predictor)`` contract that
    the test suite asserts. (Kept as a no-op function rather than module-level
    code so importing this module never constructs anything.)
    """
    from scifield.forecasting.baselines.base import Predictor

    # A static structural check: mypy accepts this assignment iff HGTForecaster
    # implements the Predictor protocol (the runtime ``isinstance`` is asserted in
    # the test suite).
    _: type[Predictor] = HGTForecaster
