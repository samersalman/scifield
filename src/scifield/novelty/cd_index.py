"""Funk--Owen-Smith disruption index (CD_t) computed from scratch.

This module implements the Funk & Owen-Smith (2017) / Wu, Wang & Evans (2019)
CD index (a.k.a. the "disruption" or D index) for a directed citation graph,
plus a corpus-level driver that builds the graph from the SciField DuckDB store
and the sibling ``cited_by`` forward-citation harvest.

The CD index for a focal work ``F`` published in year ``y_F`` is computed over
the set of works ``j`` that, within the window ``(y_F, y_F + t]``, cite ``F``
and/or cite at least one of ``F``'s references ``R``:

* type-i  -- cites ``F`` but **none** of ``R``           -> ``+1``
* type-j  -- cites ``F`` **and** at least one of ``R``    -> ``-1``
* type-k  -- cites at least one of ``R`` but **not** ``F`` ->  ``0``

``CD_t = (n_i - n_j) / (n_i + n_j + n_k)``, bounded ``[-1, 1]``; ``NaN`` when the
window contains no qualifying citers.

The time-window convention is pinned to the installed ``cdindex`` C extension:
a citer is *in window* iff ``y_F < y_citer <= y_F + t`` (strictly after the
focal year, inclusive of the upper bound). This was verified empirically against
``cdindex.Graph.cdindex`` (see ``tests/test_novelty_cd_index.py``); the package's
documented toy graph (focal ``4Z``, ``t=5``) yields ``1/6``.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "CitationGraph",
    "cd_index",
    "classify_citers",
    "compute_corpus_cd",
]


@dataclass(frozen=True)
class CitationGraph:
    """A directed citation graph.

    An edge ``u -> v`` means "work ``u`` cites work ``v``". Every node carries an
    integer time (publication year). The graph stores both out-adjacency (who a
    node cites) and in-adjacency (who cites a node) for O(1) lookups during CD
    computation.

    Parameters
    ----------
    times :
        Mapping from node id to its integer publication year/time.
    out_edges :
        Mapping from node id to the set of node ids it cites (its references).
    in_edges :
        Mapping from node id to the set of node ids that cite it (forward
        citations).

    Notes
    -----
    Construct via :meth:`from_edges` rather than directly. The dataclass is
    frozen and the adjacency dicts are not deep-copied, so do not mutate the
    inputs after construction.
    """

    times: Mapping[Hashable, int]
    out_edges: Mapping[Hashable, frozenset[Hashable]]
    in_edges: Mapping[Hashable, frozenset[Hashable]]
    _empty: frozenset[Hashable] = field(default_factory=frozenset, repr=False)

    @classmethod
    def from_edges(
        cls,
        times: Mapping[Any, int],
        edges: Iterable[tuple[Hashable, Hashable]],
    ) -> CitationGraph:
        """Build a :class:`CitationGraph` from node times and directed edges.

        Parameters
        ----------
        times :
            Mapping from node id to integer publication year. Defines the node
            set; edges referencing ids absent from ``times`` are ignored (this
            mirrors the corpus path, where some referenced works are outside the
            corpus and have unknown years).
        edges :
            Iterable of ``(u, v)`` tuples meaning ``u`` cites ``v``.

        Returns
        -------
        CitationGraph
            The constructed graph.
        """
        out: dict[Hashable, set[Hashable]] = {n: set() for n in times}
        inc: dict[Hashable, set[Hashable]] = {n: set() for n in times}
        for u, v in edges:
            if u not in times or v not in times:
                continue
            if u == v:
                continue
            out[u].add(v)
            inc[v].add(u)
        return cls(
            times=dict(times),
            out_edges={n: frozenset(s) for n, s in out.items()},
            in_edges={n: frozenset(s) for n, s in inc.items()},
        )

    def citers_of(self, node: Hashable) -> frozenset[Hashable]:
        """Return the set of nodes that cite ``node`` (forward citations)."""
        return self.in_edges.get(node, self._empty)

    def references_of(self, node: Hashable) -> frozenset[Hashable]:
        """Return the set of nodes that ``node`` cites (its references)."""
        return self.out_edges.get(node, self._empty)

    def time_of(self, node: Hashable) -> int:
        """Return the integer publication time of ``node``."""
        return self.times[node]


def classify_citers(
    graph: CitationGraph,
    focal: Hashable,
    t: int,
) -> tuple[int, int, int]:
    """Count type-i / type-j / type-k citers of ``focal`` in window ``(y_F, y_F+t]``.

    Parameters
    ----------
    graph :
        The citation graph.
    focal :
        Id of the focal node.
    t :
        Window width in time units (years). A citer published in year ``y`` is
        in window iff ``y_F < y <= y_F + t``.

    Returns
    -------
    tuple[int, int, int]
        ``(n_i, n_j, n_k)`` where ``n_i`` cites focal only, ``n_j`` cites focal
        and >=1 reference of focal, ``n_k`` cites >=1 reference of focal but not
        focal.

    Notes
    -----
    The candidate set is ``citers(focal) `` union `` { citers of each reference of
    focal }``, restricted to the time window. This matches the ``cdindex`` C
    extension exactly.
    """
    y_f = graph.time_of(focal)
    lo, hi = y_f, y_f + t

    refs = graph.references_of(focal)
    focal_citers = graph.citers_of(focal)

    # Candidate citers: anything that cites focal, or cites any reference of focal.
    candidates: set[Hashable] = set(focal_citers)
    for r in refs:
        candidates.update(graph.citers_of(r))

    n_i = n_j = n_k = 0
    for c in candidates:
        yc = graph.times.get(c)
        if yc is None or yc <= lo or yc > hi:
            continue
        cites_focal = c in focal_citers
        cites_ref = not refs.isdisjoint(graph.references_of(c))
        if cites_focal and cites_ref:
            n_j += 1
        elif cites_focal:
            n_i += 1
        elif cites_ref:
            n_k += 1
    return n_i, n_j, n_k


def cd_index(graph: CitationGraph, focal: Hashable, t: int) -> float:
    """Compute the Funk--Owen-Smith CD index for ``focal`` at window ``t``.

    Parameters
    ----------
    graph :
        The citation graph (edge ``u -> v`` means ``u`` cites ``v``).
    focal :
        Id of the focal node.
    t :
        Window width in time units (years), e.g. ``5`` for CD_5.

    Returns
    -------
    float
        ``(n_i - n_j) / (n_i + n_j + n_k)`` in ``[-1, 1]``, or ``float('nan')``
        when no qualifying citers exist in the window.
    """
    n_i, n_j, n_k = classify_citers(graph, focal, t)
    denom = n_i + n_j + n_k
    if denom == 0:
        return math.nan
    return (n_i - n_j) / denom


# --------------------------------------------------------------------------- #
# Corpus driver
# --------------------------------------------------------------------------- #


def _normalise_oa_id(value: str | None) -> str | None:
    """Strip an OpenAlex URL prefix, returning a bare ``W...`` id (or ``None``)."""
    if value is None:
        return None
    v = value.rsplit("/", 1)[-1]
    return v or None


def compute_corpus_cd(
    *,
    duckdb_path: Path,
    cited_by_parquet: Path,
    windows: Sequence[int] = (5, 10),
) -> pd.DataFrame:
    """Compute CD_5 / CD_10 for every corpus work with forward-citation data.

    Builds the directed citation graph from two sources and runs
    :func:`cd_index` for each focal corpus work:

    * **out-edges** (focal -> its references): ``references_out`` joined to
      ``openalex_works`` for the focal work's OpenAlex id and publication year.
      Edge ``citing_pmid``'s OpenAlex id -> ``ref_openalex_id``.
    * **in-edges** (citer -> focal): ``cited_by_parquet`` produced by the
      ``cited_by`` harvester, with columns ``focal_oa_id`` (str),
      ``citing_oa_id`` (str), ``citing_year`` (int), ``cites_focal_ref`` (bool).

    Because the harvest already records ``cites_focal_ref`` per citer, the
    n_i/n_j split is taken directly from that flag rather than re-derived from
    the reference edges -- this lets the function score focal works whose
    references' forward citations were not separately harvested.

    Parameters
    ----------
    duckdb_path :
        Path to ``papers.duckdb`` (read-only).
    cited_by_parquet :
        Path to the forward-citation parquet from the ``cited_by`` harvester.
    windows :
        CD window widths to compute. Only ``(5, 10)`` populate the ``cd5`` /
        ``cd10`` output columns; others are ignored for column emission.

    Returns
    -------
    pandas.DataFrame
        Columns: ``pmid, openalex_id, cd5, cd10, n_citers, n_window_5,
        n_window_10``. One row per focal corpus work present in the cited_by
        parquet.

    Notes
    -----
    This is intended to run only after the gated cited_by harvest. It is safe to
    import without the parquet present; a missing parquet raises at call time.
    """
    import duckdb
    import pandas as pd

    duckdb_path = Path(duckdb_path)
    cited_by_parquet = Path(cited_by_parquet)
    if not cited_by_parquet.exists():
        raise FileNotFoundError(f"cited_by parquet not found: {cited_by_parquet}")

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        # Focal corpus works: bare OpenAlex id + publication year, keyed by pmid.
        works = con.execute(
            """
            SELECT pmid, openalex_id, publication_year
            FROM openalex_works
            WHERE openalex_id IS NOT NULL
            """
        ).fetchdf()

        # Forward citations (in-edges), already flagged for shared references.
        cited_by = pd.read_parquet(cited_by_parquet)
    finally:
        con.close()

    works["openalex_id"] = works["openalex_id"].map(_normalise_oa_id)
    works = works.dropna(subset=["openalex_id"])
    # Year may be 0/NULL for some works; coerce to nullable int.
    works["publication_year"] = pd.to_numeric(works["publication_year"], errors="coerce").astype(
        "Int64"
    )

    oa_to_year: dict[str, int] = {
        oa: int(yr)
        for oa, yr in zip(works["openalex_id"], works["publication_year"], strict=True)
        if pd.notna(yr) and int(yr) > 0
    }
    # Map to integer pmid (nullable) so cd_index.parquet joins cleanly to
    # novelty_semantic.parquet, which keys on int64 pmid. Focal OA ids absent
    # from the corpus map to None -> <NA> in the output.
    oa_to_pmid: dict[str, int | None] = {
        oa: (int(p) if pd.notna(p) and str(p).strip() else None)
        for oa, p in zip(works["openalex_id"], works["pmid"], strict=True)
    }

    cb = cited_by.copy()
    cb["focal_oa_id"] = cb["focal_oa_id"].map(_normalise_oa_id)
    cb["citing_oa_id"] = cb["citing_oa_id"].map(_normalise_oa_id)
    cb = cb.dropna(subset=["focal_oa_id", "citing_oa_id"])
    cb["citing_year"] = pd.to_numeric(cb["citing_year"], errors="coerce").astype("Int64")
    cb["cites_focal_ref"] = cb["cites_focal_ref"].astype(bool)

    rows: list[dict[str, object]] = []
    for focal_oa, grp in cb.groupby("focal_oa_id", sort=False):
        y_f = oa_to_year.get(str(focal_oa))
        pmid = oa_to_pmid.get(str(focal_oa))
        if y_f is None:
            # Focal year unknown -> cannot window; emit NaNs.
            rows.append(
                {
                    "pmid": pmid,
                    "openalex_id": focal_oa,
                    "cd5": math.nan,
                    "cd10": math.nan,
                    "n_citers": int(grp["citing_oa_id"].nunique()),
                    "n_window_5": 0,
                    "n_window_10": 0,
                }
            )
            continue

        # Deduplicate citers, keeping ref-flag = OR over rows for that citer.
        per_citer = grp.groupby("citing_oa_id").agg(
            citing_year=("citing_year", "max"),
            cites_focal_ref=("cites_focal_ref", "max"),
        )

        result: dict[int, float] = {}
        n_window: dict[int, int] = {}
        for t in windows:
            lo, hi = y_f, y_f + t
            in_win = per_citer[
                per_citer["citing_year"].notna()
                & (per_citer["citing_year"] > lo)
                & (per_citer["citing_year"] <= hi)
            ]
            n_window[t] = int(len(in_win))
            n_j = int(in_win["cites_focal_ref"].astype(bool).sum())
            n_i = int(len(in_win)) - n_j
            denom = n_i + n_j  # type-k citers (cite only refs) are not in cited_by
            result[t] = (n_i - n_j) / denom if denom else math.nan

        rows.append(
            {
                "pmid": pmid,
                "openalex_id": focal_oa,
                "cd5": result.get(5, math.nan),
                "cd10": result.get(10, math.nan),
                "n_citers": int(len(per_citer)),
                "n_window_5": n_window.get(5, 0),
                "n_window_10": n_window.get(10, 0),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "pmid",
            "openalex_id",
            "cd5",
            "cd10",
            "n_citers",
            "n_window_5",
            "n_window_10",
        ],
    )
    # int64 (nullable) pmid to match novelty_semantic.parquet's join key.
    df["pmid"] = df["pmid"].astype("Int64")
    return df
