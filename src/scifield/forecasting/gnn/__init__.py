"""V1-S13 HGT forecasting — the heterogeneous graph layer over the V1-S12 split.

This package adds citation/authorship/topic STRUCTURE on top of the same 16
topic features the ``no_graph`` baseline (val AUC 0.701) already consumes, so the
pre-registered V1-S14 >5pp test (OSF DOI 10.17605/OSF.IO/XP94F) isolates exactly
what the graph buys. It has two layers:

* :mod:`scifield.forecasting.gnn.loader` — the **leakage-safe graph snapshot
  loader** (this slice): one :class:`~torch_geometric.data.HeteroData` per
  forecast-origin year ``t`` over all papers with ``year <= t`` plus their
  induced authors/institutions/journals and ALL leaf topics. It is the data
  layer the HGT model fits on.
* ``hgt`` — the HGT (heterogeneous graph transformer) model itself (added by
  the next slice; ``HGTForecaster`` is appended to this package's exports there).

See :func:`scifield.forecasting.gnn.loader.build_year_snapshots` for the build
contract and the per-snapshot ``node_types`` / ``edge_types`` metadata the model
needs to construct its ``HGTConv`` layers.
"""

from scifield.forecasting.gnn.hgt import HGTForecaster
from scifield.forecasting.gnn.loader import (
    EDGE_TYPES,
    NODE_TYPES,
    assert_snapshot_no_leakage,
    build_snapshot_from_frames,
    build_year_snapshots,
    load_snapshot,
    topic_id_to_index,
)

__all__ = [
    # --- snapshot loader (V1-S13 data layer) ---
    "EDGE_TYPES",
    "NODE_TYPES",
    "assert_snapshot_no_leakage",
    "build_snapshot_from_frames",
    "build_year_snapshots",
    "load_snapshot",
    "topic_id_to_index",
    # --- HGT model (appended by the next slice) ---
    "HGTForecaster",
]
