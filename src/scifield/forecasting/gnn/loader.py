"""V1-S13 leakage-safe heterogeneous graph snapshots — the HGT data layer.

This module materialises **one** :class:`~torch_geometric.data.HeteroData`
snapshot per forecast-origin year ``t``. Each snapshot is the citation /
authorship / topic graph as it stood *at or before* ``t``: all papers with
``year <= t`` plus the authors / institutions / journals induced from them, ALL
leaf topics, and the induced edges. The HGT model (next slice) fits on these
snapshots; this module guarantees they carry no information from after ``t``.

Why one graph over ALL topics per ``t`` (not one subgraph per topic)
-------------------------------------------------------------------
A topic does not predict its own future in isolation — adjacent fields cite each
other, and that cross-topic citation flow is precisely the signal a graph model
can exploit that the ``no_graph`` baseline cannot. So every snapshot is a single
graph spanning every leaf topic, and ``Paper -CITES-> Paper`` edges let
information cross topic boundaries during message passing. The model reads out
the per-topic prediction at the corresponding ``Topic`` node.

Why ALL leaf topics are present in every snapshot (even with zero papers ≤ t)
----------------------------------------------------------------------------
Reserving a node for every leaf topic in every year guarantees (a) every
requested ``(topic_id, t)`` readout resolves to a node — assertion **L4** — and
(b) the ``topic_id -> node_idx`` map is **stable across all years**, so the model
can scatter the scaled per-(topic, t) feature panel onto the graph by a single
fixed index map. ``Topic`` nodes are ordered by ascending ``topic_id`` and the
``topic_id`` per node index is stored on the snapshot (``data["Topic"].topic_id``)
so the model can rebuild the inverse map.

Leakage safety (the core deliverable)
-------------------------------------
Every feature is reconstructable from ``year <= t`` data only:

* ``Paper.x = [normalized_year, log1p(in-snapshot CITES in-degree)]`` — the
  in-degree counts only edges *inside this snapshot* (both endpoints ≤ t).
* ``Topic.x`` is **left for the model** to fill from the scaled panel at fit
  time (only the node set + ``topic_id`` are reserved here), so no future share
  ever leaks through the graph.
* ``Author`` / ``Institution`` / ``Journal`` carry a constant ``ones(n, 1)``; a
  per-type input ``Linear`` in the model supplies capacity, and deliberately
  avoiding per-node embeddings prevents an identity/overfit leak.

:func:`assert_snapshot_no_leakage` enforces four guards (L1–L4) with SHORT
keyword messages (``leakage:max_year`` etc.) mirroring
:func:`scifield.forecasting.data.assert_no_leakage`, plus a **diagnostic** L5:
the count of ``CITES`` edges whose citing paper predates the cited paper. That
is a pre-existing data artifact (OpenAlex back-resolution), NOT a leak — both
endpoints are still ≤ t — so L5 is counted and recorded, never asserted on.

Edges are built **forward-direction only** (mirroring the Kùzu schema triplets);
the model applies :class:`torch_geometric.transforms.ToUndirected` itself, which
adds the ``rev_*`` reverse relations needed for message passing into ``Topic`` /
``Author`` / ``Institution`` / ``Journal``.

Reuse
-----
The DuckDB ``SELECT`` strings are imported verbatim from
:mod:`scifield.novelty.kuzu_loader` (``_NODE_QUERIES`` / ``_REL_QUERIES``) so the
node/edge definitions stay byte-identical to the canonical Kùzu graph — the
single source of truth for what an edge *is*. This module adds only the
per-year ``year <= t`` induction and the leakage guards on top.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from scifield.novelty.kuzu_loader import _NODE_QUERIES, _REL_QUERIES
from scifield.thematic.dedup import ensure_papers_distinct_view

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

__all__ = [
    "EDGE_TYPES",
    "NODE_TYPES",
    "assert_snapshot_no_leakage",
    "build_snapshot_from_frames",
    "build_year_snapshots",
    "load_snapshot",
    "topic_id_to_index",
]

#: Node types present in every snapshot, in a fixed order (the model builds its
#: ``HGTConv`` metadata from this + :data:`EDGE_TYPES`). ``ToUndirected`` does
#: not change the node set.
NODE_TYPES: tuple[str, ...] = ("Paper", "Author", "Institution", "Journal", "Topic")

#: Forward edge triplets present in every snapshot, mirroring the Kùzu schema.
#: The model applies ``ToUndirected``, which adds a ``("dst", "rev_<rel>", "src")``
#: relation for each of the FOUR heterogeneous edges but symmetrises the
#: ``Paper--CITES-->Paper`` self-relation IN PLACE (no separate ``rev_CITES``) —
#: so the model's metadata has 9 relations, not 5. Build ``HGTConv`` metadata
#: from ``data.metadata()`` after ``ToUndirected``, never from this tuple directly.
EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("Paper", "CITES", "Paper"),
    ("Paper", "AUTHORED_BY", "Author"),
    ("Author", "AFFILIATED_WITH", "Institution"),
    ("Paper", "PUBLISHED_IN", "Journal"),
    ("Paper", "ASSIGNED_TO", "Topic"),
)

#: Per-type input feature dimensions. ``Topic`` is reserved-only here (the model
#: overwrites ``data["Topic"].x`` with the 16-dim scaled panel at fit time).
TOPIC_FEATURE_DIM = 16
PAPER_FEATURE_DIM = 2  # [normalized_year, log1p(in-snapshot CITES in-degree)]
ENTITY_FEATURE_DIM = 1  # Author / Institution / Journal constant ones(n, 1)

#: Year origin used to normalise ``Paper`` years. The corpus floor is 1995, so
#: ``(year - 1995) / (t - 1995 + 1)`` is bounded in ``(0, 1]`` and depends only
#: on ``year <= t`` data — no future leak.
_YEAR_FLOOR = 1995


# --------------------------------------------------------------------------- #
# Pure assembly core (pandas in, HeteroData out — unit-testable without DuckDB)
# --------------------------------------------------------------------------- #


def _index_map(values: list[Any]) -> dict[Any, int]:
    """Map a list of already-sorted unique ids to contiguous ``0..n-1`` ints.

    Deterministic by construction (the caller sorts first); used to turn opaque
    string/int ids into the row positions PyG edge_index tensors require.
    """
    return {v: i for i, v in enumerate(values)}


def build_snapshot_from_frames(
    t: int,
    *,
    papers: pd.DataFrame,
    topics: pd.DataFrame,
    cites: pd.DataFrame,
    authored_by: pd.DataFrame,
    affiliated_with: pd.DataFrame,
    published_in: pd.DataFrame,
    assigned_to: pd.DataFrame,
) -> tuple[HeteroData, dict[str, Any]]:
    """Assemble the leakage-safe :class:`HeteroData` snapshot for origin year ``t``.

    This is the pure, I/O-free heart of the loader: every input is a pandas frame
    holding the FULL-corpus node/edge rows (exactly the columns the canonical
    :mod:`scifield.novelty.kuzu_loader` queries emit), and the ``year <= t``
    induction + tensor assembly happen here so the logic is unit-testable without
    DuckDB. :func:`build_year_snapshots` is the thin DuckDB wrapper around it.

    Induction order (the only correct order — AFFILIATED_WITH has no pmid)
        1. ``Paper`` = rows with ``year <= t`` (sorted by pmid).
        2. ``CITES`` kept iff BOTH endpoints are eligible papers.
        3. ``Author`` induced from ``AUTHORED_BY`` edges off eligible papers.
        4. ``Institution`` induced from ``AFFILIATED_WITH`` edges off *induced*
           authors (an institution reachable only via an ineligible author is
           dropped — it has no provenance to an eligible paper).
        5. ``Journal`` induced from ``PUBLISHED_IN`` edges off eligible papers.
        6. ``Topic`` = ALL leaf topics (NOT induced — see module docstring).

    Parameters
    ----------
    t:
        Forecast-origin year (the inclusive feature cutoff).
    papers:
        Columns ``pmid`` (str), ``year`` (int); one row per corpus paper.
    topics:
        Column ``topic_id`` (int); one row per leaf topic (full set).
    cites:
        Columns ``from_pmid``, ``to_pmid`` (str); corpus-internal citations.
    authored_by:
        Columns ``pmid``, ``author_canonical_id``.
    affiliated_with:
        Columns ``author_canonical_id``, ``institution_canonical_id``.
    published_in:
        Columns ``pmid``, ``journal_slug``.
    assigned_to:
        Columns ``pmid``, ``topic_id`` (paper→leaf-topic membership).

    Returns
    -------
    tuple[HeteroData, dict]
        ``(data, meta)``. ``data`` carries the five node types, five forward edge
        types, ``Paper.x`` / ``Paper.year`` / ``Topic.topic_id`` (and a zeros
        ``Topic.x`` placeholder the model overwrites). ``meta`` carries per-type
        node/edge counts, the L5 future-citation-artifact count, and ``t``.
    """
    import torch
    from torch_geometric.data import HeteroData

    # --- 1. Eligible papers (year <= t), deterministic pmid order. ----------
    elig = papers.loc[papers["year"].astype(int) <= int(t)].copy()
    elig["pmid"] = elig["pmid"].astype(str)
    elig = elig.drop_duplicates(subset="pmid").sort_values("pmid").reset_index(drop=True)
    paper_ids: list[str] = elig["pmid"].tolist()
    paper_idx = _index_map(paper_ids)
    # np.array(copy=True) so the tensor owns writable memory (to_numpy may view).
    paper_years = np.array(elig["year"].astype(np.int64), dtype=np.int64)
    eligible_pmids = set(paper_ids)
    n_paper = len(paper_ids)

    # --- 2. CITES: keep only edges whose BOTH endpoints are eligible. -------
    c = cites.copy()
    c["from_pmid"] = c["from_pmid"].astype(str)
    c["to_pmid"] = c["to_pmid"].astype(str)
    c = c[c["from_pmid"].isin(eligible_pmids) & c["to_pmid"].isin(eligible_pmids)]
    cite_src = np.array([paper_idx[p] for p in c["from_pmid"]], dtype=np.int64)
    cite_dst = np.array([paper_idx[p] for p in c["to_pmid"]], dtype=np.int64)

    # --- 3. AUTHORED_BY off eligible papers -> induced Authors. -------------
    ab = authored_by.copy()
    ab["pmid"] = ab["pmid"].astype(str)
    ab["author_canonical_id"] = ab["author_canonical_id"].astype(str)
    ab = ab[ab["pmid"].isin(eligible_pmids)].drop_duplicates()
    author_ids = sorted(set(ab["author_canonical_id"]))
    author_idx = _index_map(author_ids)
    n_author = len(author_ids)

    # --- 4. AFFILIATED_WITH off induced Authors -> induced Institutions. ----
    aw = affiliated_with.copy()
    aw["author_canonical_id"] = aw["author_canonical_id"].astype(str)
    aw["institution_canonical_id"] = aw["institution_canonical_id"].astype(str)
    aw = aw[aw["author_canonical_id"].isin(author_idx)].drop_duplicates()
    inst_ids = sorted(set(aw["institution_canonical_id"]))
    inst_idx = _index_map(inst_ids)
    n_inst = len(inst_ids)

    # --- 5. PUBLISHED_IN off eligible papers -> induced Journals. -----------
    pin = published_in.copy()
    pin["pmid"] = pin["pmid"].astype(str)
    pin["journal_slug"] = pin["journal_slug"].astype(str)
    pin = pin[pin["pmid"].isin(eligible_pmids)].drop_duplicates()
    journal_ids = sorted(set(pin["journal_slug"]))
    journal_idx = _index_map(journal_ids)
    n_journal = len(journal_ids)

    # --- 6. Topic: ALL leaf topics, ascending topic_id (stable across t). ---
    topic_ids = sorted(int(x) for x in topics["topic_id"].unique())
    topic_idx = _index_map(topic_ids)
    n_topic = len(topic_ids)

    # ASSIGNED_TO off eligible papers + present topics.
    at = assigned_to.copy()
    at["pmid"] = at["pmid"].astype(str)
    at["topic_id"] = at["topic_id"].astype(np.int64)
    at = at[at["pmid"].isin(eligible_pmids) & at["topic_id"].isin(topic_idx)].drop_duplicates()

    # --- Build edge_index tensors (forward direction only). -----------------
    aff_src = np.array([author_idx[a] for a in aw["author_canonical_id"]], dtype=np.int64)
    aff_dst = np.array([inst_idx[i] for i in aw["institution_canonical_id"]], dtype=np.int64)
    auth_src = np.array([paper_idx[p] for p in ab["pmid"]], dtype=np.int64)
    auth_dst = np.array([author_idx[a] for a in ab["author_canonical_id"]], dtype=np.int64)
    pub_src = np.array([paper_idx[p] for p in pin["pmid"]], dtype=np.int64)
    pub_dst = np.array([journal_idx[j] for j in pin["journal_slug"]], dtype=np.int64)
    asg_src = np.array([paper_idx[p] for p in at["pmid"]], dtype=np.int64)
    asg_dst = np.array([topic_idx[int(g)] for g in at["topic_id"]], dtype=np.int64)

    # --- Paper features: [normalized_year, log1p(in-snapshot in-degree)]. ----
    indeg = np.zeros(n_paper, dtype=np.float64)
    if cite_dst.size:
        np.add.at(indeg, cite_dst, 1.0)
    denom = float(int(t) - _YEAR_FLOOR + 1)  # >= 1 for any t >= 1995
    norm_year = (paper_years.astype(np.float64) - _YEAR_FLOOR) / denom
    paper_x = np.stack([norm_year, np.log1p(indeg)], axis=1).astype(np.float32)

    # --- Assemble HeteroData (forward edges only; model adds reverses). -----
    data = HeteroData()
    data["Paper"].x = torch.from_numpy(paper_x)
    data["Paper"].year = torch.from_numpy(paper_years)
    data["Author"].x = torch.ones(n_author, ENTITY_FEATURE_DIM, dtype=torch.float32)
    data["Institution"].x = torch.ones(n_inst, ENTITY_FEATURE_DIM, dtype=torch.float32)
    data["Journal"].x = torch.ones(n_journal, ENTITY_FEATURE_DIM, dtype=torch.float32)
    # Topic: reserve nodes + store topic_id; zeros placeholder (model overwrites).
    data["Topic"].num_nodes = n_topic
    data["Topic"].x = torch.zeros(n_topic, TOPIC_FEATURE_DIM, dtype=torch.float32)
    data["Topic"].topic_id = torch.tensor(topic_ids, dtype=torch.long)

    data["Paper", "CITES", "Paper"].edge_index = torch.from_numpy(np.stack([cite_src, cite_dst]))
    data["Paper", "AUTHORED_BY", "Author"].edge_index = torch.from_numpy(
        np.stack([auth_src, auth_dst])
    )
    data["Author", "AFFILIATED_WITH", "Institution"].edge_index = torch.from_numpy(
        np.stack([aff_src, aff_dst])
    )
    data["Paper", "PUBLISHED_IN", "Journal"].edge_index = torch.from_numpy(
        np.stack([pub_src, pub_dst])
    )
    data["Paper", "ASSIGNED_TO", "Topic"].edge_index = torch.from_numpy(
        np.stack([asg_src, asg_dst])
    )

    # --- L5 diagnostic: CITES edges with citing.year < cited.year. ----------
    n_future_citation_artifacts = 0
    if cite_src.size:
        n_future_citation_artifacts = int(np.sum(paper_years[cite_src] < paper_years[cite_dst]))
    data.n_future_citation_artifacts = n_future_citation_artifacts
    data.origin_year = int(t)

    meta: dict[str, Any] = {
        "t": int(t),
        "n_nodes": {
            "Paper": n_paper,
            "Author": n_author,
            "Institution": n_inst,
            "Journal": n_journal,
            "Topic": n_topic,
        },
        "n_edges": {
            "CITES": int(cite_src.size),
            "AUTHORED_BY": int(auth_src.size),
            "AFFILIATED_WITH": int(aff_src.size),
            "PUBLISHED_IN": int(pub_src.size),
            "ASSIGNED_TO": int(asg_src.size),
        },
        "n_future_citation_artifacts": n_future_citation_artifacts,
    }
    return data, meta


# --------------------------------------------------------------------------- #
# Leakage filter — mirrors data.py's keyword-message guard style
# --------------------------------------------------------------------------- #


def assert_snapshot_no_leakage(
    data: HeteroData,
    t: int,
    requested_topic_ids: list[int] | set[int] | None = None,
) -> int:
    """Assert the snapshot leakage invariants L1–L4; record the L5 diagnostic.

    Each failure carries a SHORT keyword so tests/callers can assert on it,
    matching the convention in :func:`scifield.forecasting.data.assert_no_leakage`.

    L1 (``leakage:max_year``)
        Every ``Paper`` has ``year <= t`` (no future paper in the snapshot).
    L2 (``leakage:cites_endpoint``)
        Every endpoint of every ``CITES`` edge has ``year <= t`` — checked by
        indexing ``Paper.year`` with both edge-index rows. **L2 is evaluated
        before L1** so a future paper that participates in a citation gets the
        specific ``cites_endpoint`` message, while a future paper with no CITES
        edge falls through to the general ``max_year`` guard.
    L3 (``leakage:node_provenance``)
        Every ``Author`` / ``Institution`` / ``Journal`` node connects, via its
        edge type, to at least one eligible ``Paper`` (no orphan non-Paper,
        non-Topic node). ``Author`` provenance is the union of ``AUTHORED_BY``
        targets and ``AFFILIATED_WITH`` sources.
    L4 (``leakage:topic_resolves``)
        Every requested ``topic_id`` resolves to a ``Topic`` node. Since the
        builder reserves a node for every leaf topic this always holds for the
        full set; the guard catches a caller asking for a topic the snapshot was
        not built with (default ``requested_topic_ids`` = all present).

    L5 (DIAGNOSTIC, not an assertion)
        The count of ``CITES`` edges whose citing paper predates the cited paper
        — a pre-existing OpenAlex back-resolution artifact (both endpoints still
        ≤ t, so NO leak). Read off ``data.n_future_citation_artifacts`` (or
        recomputed from ``Paper.year`` if absent) and returned for the sidecar.

    Parameters
    ----------
    data:
        The snapshot to validate. Must carry ``Paper.year`` and ``Topic.topic_id``
        (the builder always stores them).
    t:
        The origin year the snapshot was built for.
    requested_topic_ids:
        Optional iterable of topic ids the caller intends to read out. Defaults
        to every present ``Topic.topic_id``.

    Returns
    -------
    int
        The L5 future-citation-artifact count (recorded, never raised on).
    """
    import torch

    t = int(t)
    paper_year = data["Paper"].year
    if not isinstance(paper_year, torch.Tensor):  # pragma: no cover - builder guarantees tensor
        paper_year = torch.as_tensor(paper_year)

    # L2 is checked BEFORE L1 on purpose: a future paper that participates in a
    # CITES edge is the more specific failure (its citation structure is leaking),
    # so it gets the precise ``cites_endpoint`` message; a future paper with no
    # CITES edge falls through to the general ``max_year`` guard below.
    cites = data["Paper", "CITES", "Paper"].edge_index
    if cites.numel():
        endpoint_years = paper_year[cites.reshape(-1)]
        bad = int((endpoint_years > t).sum().item())
        assert bad == 0, f"leakage:cites_endpoint {bad} CITES endpoint(s) with year > {t}"

    # L1: max Paper.year <= t (catches any future paper, CITES-linked or not).
    if paper_year.numel():
        max_year = int(paper_year.max().item())
        assert max_year <= t, f"leakage:max_year Paper.year max {max_year} > origin year {t}"

    # L3: every Author/Institution/Journal node has provenance to an eligible Paper.
    auth_ei = data["Paper", "AUTHORED_BY", "Author"].edge_index
    aff_ei = data["Author", "AFFILIATED_WITH", "Institution"].edge_index
    pub_ei = data["Paper", "PUBLISHED_IN", "Journal"].edge_index

    n_author = int(data["Author"].num_nodes or 0)
    # An author is grounded by being an AUTHORED_BY target OR an AFFILIATED_WITH source.
    author_seen = set(auth_ei[1].tolist()) | set(aff_ei[0].tolist())
    author_orphans = sorted(set(range(n_author)) - author_seen)
    assert not author_orphans, (
        f"leakage:node_provenance Author node(s) {author_orphans[:5]} not linked to any "
        f"eligible Paper (year <= {t})"
    )

    n_inst = int(data["Institution"].num_nodes or 0)
    inst_seen = set(aff_ei[1].tolist())
    inst_orphans = sorted(set(range(n_inst)) - inst_seen)
    assert not inst_orphans, (
        f"leakage:node_provenance Institution node(s) {inst_orphans[:5]} not linked to any "
        f"induced Author"
    )

    n_journal = int(data["Journal"].num_nodes or 0)
    journal_seen = set(pub_ei[1].tolist())
    journal_orphans = sorted(set(range(n_journal)) - journal_seen)
    assert not journal_orphans, (
        f"leakage:node_provenance Journal node(s) {journal_orphans[:5]} not linked to any "
        f"eligible Paper (year <= {t})"
    )

    # L4: every requested topic_id resolves to a Topic node.
    present_topics = {int(g) for g in data["Topic"].topic_id.tolist()}
    if requested_topic_ids is None:
        requested = present_topics
    else:
        requested = {int(g) for g in requested_topic_ids}
    missing = sorted(requested - present_topics)
    assert (
        not missing
    ), f"leakage:topic_resolves requested topic_id(s) {missing[:5]} not in snapshot"

    # L5: diagnostic only.
    n_artifacts = int(getattr(data, "n_future_citation_artifacts", -1))
    if n_artifacts < 0:  # recompute defensively if a hand-built snapshot omitted it
        n_artifacts = 0
        if cites.numel():
            src_years = paper_year[cites[0]]
            dst_years = paper_year[cites[1]]
            n_artifacts = int((src_years < dst_years).sum().item())
    return n_artifacts


# --------------------------------------------------------------------------- #
# Helpers the model needs
# --------------------------------------------------------------------------- #


def topic_id_to_index(data: HeteroData) -> dict[int, int]:
    """Return the ``{topic_id -> Topic node index}`` map for a loaded snapshot.

    The inverse of ``data["Topic"].topic_id`` (which is the LongTensor of
    ``topic_id`` per node index). The model uses this to scatter its scaled
    per-(topic, t) feature panel onto the ``Topic`` nodes and to gather the
    per-topic readout. The map is identical across all snapshots because the
    builder reserves a node for every leaf topic in ascending ``topic_id`` order.
    """
    return {int(g): i for i, g in enumerate(data["Topic"].topic_id.tolist())}


# --------------------------------------------------------------------------- #
# I/O — single-thread pin, cache I/O, and the DuckDB build entry point
# --------------------------------------------------------------------------- #


def _pin_torch_single_threaded() -> None:
    """Force torch onto a single intra-op thread (Darwin libomp segfault guard).

    Mirrors :func:`scifield.forecasting.baselines.evaluate._pin_torch_single_threaded`:
    the default multi-threaded intra-op pool races libomp on this macOS box. Every
    entry point that touches torch must pin to one thread; idempotent.
    """
    import torch

    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)


def load_snapshot(path: str | Path) -> HeteroData:
    """Load a cached snapshot from ``path`` and re-run the L1/L2 guards.

    Re-running the *year* guards on load is the staleness net: if a cache file is
    ever out of date with the corpus or the build code, the L1 (``max_year``) /
    L2 (``cites_endpoint``) assertions fire here rather than letting a leaky
    snapshot silently reach the model.

    Parameters
    ----------
    path:
        Path to a ``snapshot_{t}.pt`` file written by :func:`build_year_snapshots`.

    Returns
    -------
    torch_geometric.data.HeteroData
        The deserialised snapshot.
    """
    import torch

    _pin_torch_single_threaded()
    path = Path(path)
    # weights_only=False: a HeteroData is a structured object, not a state-dict.
    data: HeteroData = torch.load(path, weights_only=False)
    t = int(getattr(data, "origin_year", 0))
    # Re-run the year guards only (L1/L2); pass all present topics for L4 = no-op.
    assert_snapshot_no_leakage(data, t)
    return data


def _origin_years(cfg: DictConfig) -> list[int]:
    """Origin years to build: the inclusive ``train.lo .. val.hi`` range (≈1998..2020).

    These are the only years the model trains/validates on (test is sealed), so
    building exactly this range avoids materialising snapshots no fitting run can
    legally consume.
    """
    train_lo = int(cfg.split.train[0])
    val_hi = int(cfg.split.val[1])
    return list(range(train_lo, val_hi + 1))


def _load_corpus_frames(duck: Any, topics_posix: str) -> dict[str, pd.DataFrame]:
    """Run the canonical Kùzu node/edge SELECTs once into full-corpus frames.

    The frames are loaded ONCE (not per year): per-year induction is a cheap
    pandas filter in :func:`build_snapshot_from_frames`, so re-hitting DuckDB for
    every origin year would be wasted I/O. Reuses the exact
    :mod:`scifield.novelty.kuzu_loader` queries so the node/edge definitions stay
    identical to the canonical graph.
    """
    papers = duck.execute(_NODE_QUERIES["Paper"].format(topics=topics_posix)).df()
    papers = papers[["pmid", "year"]]
    topics = duck.execute(_NODE_QUERIES["Topic"].format(topics=topics_posix)).df()
    cites = duck.execute(_REL_QUERIES["CITES"].format(topics=topics_posix)).df()
    authored_by = duck.execute(_REL_QUERIES["AUTHORED_BY"].format(topics=topics_posix)).df()
    affiliated_with = duck.execute(_REL_QUERIES["AFFILIATED_WITH"].format(topics=topics_posix)).df()
    published_in = duck.execute(_REL_QUERIES["PUBLISHED_IN"].format(topics=topics_posix)).df()
    assigned_to = duck.execute(_REL_QUERIES["ASSIGNED_TO"].format(topics=topics_posix)).df()
    return {
        "papers": papers,
        "topics": topics,
        "cites": cites,
        "authored_by": authored_by,
        "affiliated_with": affiliated_with,
        "published_in": published_in,
        "assigned_to": assigned_to,
    }


def build_year_snapshots(
    cfg: DictConfig, *, years: list[int] | None = None
) -> dict[int, dict[str, Any]]:
    """Build, assert, and cache one leakage-safe snapshot per origin year.

    Opens ``cfg.input.duckdb_path`` (read-write, to (re)create ``papers_distinct``
    via :func:`scifield.thematic.dedup.ensure_papers_distinct_view`), runs the
    canonical node/edge queries once, then for every requested origin year ``t``
    builds the snapshot (:func:`build_snapshot_from_frames`), runs
    :func:`assert_snapshot_no_leakage`, and caches it to
    ``cfg.output.snapshots_dir/snapshot_{t}.pt`` via :func:`torch.save`. Pins torch
    to a single thread first. Deterministic (sorted node ordering; no RNG).

    Parameters
    ----------
    cfg:
        The forecasting ``DictConfig`` (``conf/forecasting/v1.yaml``). Reads
        ``input.duckdb_path``, ``input.topics_parquet``, ``output.snapshots_dir``,
        and the ``split`` ranges.
    years:
        Explicit origin years to build; defaults to ``train.lo .. val.hi``
        inclusive (the train∪val range; ≈1998..2020).

    Returns
    -------
    dict[int, dict]
        ``{t: meta}`` where each ``meta`` carries per-type node/edge counts, the
        L5 future-citation-artifact count, and the cache ``path``.
    """
    import duckdb
    import torch

    _pin_torch_single_threaded()

    duckdb_path = Path(str(cfg.input.duckdb_path))
    topics_posix = Path(str(cfg.input.topics_parquet)).as_posix()
    snapshots_dir = Path(str(cfg.output.snapshots_dir))
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    build_years = _origin_years(cfg) if years is None else [int(y) for y in years]

    out: dict[int, dict[str, Any]] = {}
    # Read-write: ensure_papers_distinct_view creates a VIEW (kuzu_loader does the same).
    duck = duckdb.connect(str(duckdb_path), read_only=False)
    try:
        ensure_papers_distinct_view(duck)
        frames = _load_corpus_frames(duck, topics_posix)
    finally:
        duck.close()

    for t in build_years:
        data, meta = build_snapshot_from_frames(t, **frames)
        n_artifacts = assert_snapshot_no_leakage(data, t)
        meta["n_future_citation_artifacts"] = n_artifacts

        path = snapshots_dir / f"snapshot_{t}.pt"
        torch.save(data, path)
        meta["path"] = str(path)
        out[t] = meta
        logger.info(
            "snapshot t=%d: nodes=%s edges=%s future_cite_artifacts=%d -> %s",
            t,
            meta["n_nodes"],
            meta["n_edges"],
            n_artifacts,
            path,
        )
    return out
