"""V2 cartography — the canonical *journal-flow* data model.

This module owns the per-``(topic, journal, year)`` **flow table**: the tidy
substrate every downstream cartography layer (cascade engine V2-S03, cascade
validation V2-S04, roles & velocity V2-S05) reads. A "flow" cell records, for one
research topic in one canonical journal in one year, the publication *volume*
(``n_papers``), whether that year is the topic's *first appearance* in that journal
(the cascade anchor), and — when a citation edge list is supplied — the citational
``in_flow`` / ``out_flow`` crossing that cell.

Canonical journal entity (the one bug this module exists to prevent)
--------------------------------------------------------------------
The master ``archetypes.parquet`` carries a ``journal`` *display* string with
**eleven** distinct values, because the journal that is today *JAMA surgery* also
appears under its pre-2013 name *Archives of surgery (Chicago, Ill. : 1960)*. Both
are the single canonical journal whose slug is ``jama_surg``. Keying the flow on the
display string would invent a phantom 11th journal and split ``jama_surg``'s history
across two rows. **The flow table therefore keys journals on ``journal_slug`` — the
clean ten — never on the display name.** The builder script resolves the slug by
joining ``pmid → papers_distinct.journal_slug`` from the papers DuckDB (see
``V2/CONTEXT.md`` §3); this module simply *requires* its input frame to already
carry ``journal_slug`` and operates on it. :func:`assert_canonical_journals` is a
cheap guard the builder (or a test) can call to fail loudly if more than the ten
expected slugs ever appear.

Topic granularity contract (``grain``)
---------------------------------------
The same logic runs at either topic grain. The *caller* selects the grain and
passes the matching topic-key column:

* ``grain="leaf"`` (default) — the input frame's topic key is ``topic_id`` (149
  leaf topics).
* ``grain="mid"`` — the caller has joined ``topic_hierarchy`` so the input frame
  carries ``mid_level_id`` (96 mid-level topics); pass that as the topic key.

:func:`topic_key_for_grain` maps a ``grain`` string to its expected column name, so
both the builder and the tests agree on the contract. The returned flow table names
its topic column exactly that key (``topic_id`` or ``mid_level_id``), so a consumer
can branch on ``grain`` without re-deriving anything.

Purity
------
Every function here is **pure and I/O-free**: it takes in-memory DataFrames and
returns a new DataFrame. Reading parquet / DuckDB, traversing Kuzu, and writing the
output live in ``V2/scripts/build_flow.py``. We reuse — and generalize — the
lead/follow logic in :mod:`scifield.findings.seeding`: :func:`first_appearance`
delegates to :func:`scifield.findings.seeding.first_publication_year` so the
"first year a (topic, entity) reached ``min_papers``" definition is shared, not
duplicated.

Conventions: ``from __future__ import annotations``; numpy-style docstrings; pandas
+ numpy only; ruff/black line-length 100. Real counts are COMPUTED, never hardcoded.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "CANONICAL_JOURNAL_SLUGS",
    "GRAIN_TOPIC_KEYS",
    "add_citation_flows",
    "assert_canonical_journals",
    "build_flow_table",
    "first_appearance",
    "topic_key_for_grain",
]

# The clean ten canonical journal slugs (V2/CONTEXT.md §3). The flow table keys on
# these; an 11th value means the jama_surg display split leaked in.
CANONICAL_JOURNAL_SLUGS: tuple[str, ...] = (
    "ann_surg",
    "arthroscopy",
    "br_j_surg",
    "clin_orthop_relat_res",
    "j_am_coll_surg",
    "j_arthroplasty",
    "j_bone_joint_surg_am",
    "jama_surg",
    "spine",
    "surgery",
)

# grain -> the topic-key column the caller must supply (and that the flow names).
GRAIN_TOPIC_KEYS: dict[str, str] = {"leaf": "topic_id", "mid": "mid_level_id"}


def topic_key_for_grain(grain: str) -> str:
    """Topic-key column name expected for a given granularity.

    Parameters
    ----------
    grain :
        ``"leaf"`` (key ``topic_id``, 149 leaf topics) or ``"mid"`` (key
        ``mid_level_id``, 96 mid-level topics). See :data:`GRAIN_TOPIC_KEYS`.

    Returns
    -------
    str
        The column name the input frame must carry as its topic key and that the
        flow table will name its topic column.

    Raises
    ------
    ValueError
        If ``grain`` is neither ``"leaf"`` nor ``"mid"``.
    """
    try:
        return GRAIN_TOPIC_KEYS[grain]
    except KeyError as exc:
        valid = ", ".join(sorted(GRAIN_TOPIC_KEYS))
        raise ValueError(f"unknown grain {grain!r}; expected one of {{{valid}}}") from exc


def assert_canonical_journals(journals: object) -> None:
    """Fail loudly if any journal outside the canonical ten appears.

    A guard the builder (or a test) calls on the input frame's ``journal_slug``
    values to catch the jama_surg display-name split early — if a display string
    ("JAMA surgery", "Archives of surgery ...") ever reaches the flow as a journal
    key, the count exceeds ten and this raises.

    Parameters
    ----------
    journals :
        Any iterable of journal identifiers (typically a ``journal_slug`` Series or
        its unique values). ``NaN`` / ``None`` entries are ignored.

    Raises
    ------
    ValueError
        If the distinct non-missing values are not a subset of
        :data:`CANONICAL_JOURNAL_SLUGS`.

    Notes
    -----
    Pure / no I/O. Only the *membership* is checked; the panel need not be complete
    (a topic-restricted frame may legitimately omit some slugs).
    """
    seen = {
        j
        for j in cast(Iterable[object], journals)
        if j is not None and not (isinstance(j, float) and j != j)
    }
    extra = sorted(str(j) for j in seen - set(CANONICAL_JOURNAL_SLUGS))
    if extra:
        raise ValueError(
            "non-canonical journal(s) in flow input "
            f"(expected only journal_slug from the clean ten): {extra}. "
            "Likely the jama_surg display-name split leaked in — key on journal_slug."
        )


def first_appearance(
    papers_topics: pd.DataFrame,
    *,
    topic_key: str = "topic_id",
    min_papers: int = 1,
) -> pd.DataFrame:
    """Earliest year each ``(topic, journal_slug)`` reached ``min_papers`` papers.

    The *first-appearance year* is the cascade anchor: "topic T first appeared in
    journal J in year Y" (panel-conditional on the 10-journal corpus). This is a
    thin generalisation of
    :func:`scifield.findings.seeding.first_publication_year` to an arbitrary topic
    key and the journal-slug entity grain — the threshold semantics (per-*year*
    count, not cumulative) are inherited unchanged so the two layers agree.

    Parameters
    ----------
    papers_topics :
        Tidy frame, one row per paper, carrying at least ``[topic_key,
        "journal_slug", "year"]``. Extra columns are ignored; rows missing any of
        those three are dropped.
    topic_key :
        The topic-grain column, ``"topic_id"`` (leaf) or ``"mid_level_id"`` (mid).
        Default ``"topic_id"``.
    min_papers :
        Per-year publication threshold. ``first_year`` is the earliest year the
        (topic, journal) pair published **at least** ``min_papers`` papers in that
        single year. Default ``1``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[topic_key, "journal_slug", "first_year"]``, one row per
        (topic, journal) pair that ever met the threshold, sorted by ``topic_key``
        then ``journal_slug``. ``first_year`` is ``int64``.
    """
    from scifield.findings.seeding import first_publication_year

    work = papers_topics
    # first_publication_year expects the topic column to be named "topic_id"; rename
    # only if we are operating at a non-leaf grain, then rename back on the way out.
    rename_back: dict[str, str] = {}
    if topic_key != "topic_id":
        work = work.rename(columns={topic_key: "topic_id"})
        rename_back = {"topic_id": topic_key}

    first = first_publication_year(work, entity_col="journal_slug", min_papers=min_papers)
    if rename_back:
        first = first.rename(columns=rename_back)
    return first[[topic_key, "journal_slug", "first_year"]]


def add_citation_flows(
    flow: pd.DataFrame,
    papers_topics: pd.DataFrame,
    citation_edges: pd.DataFrame,
    *,
    topic_key: str = "topic_id",
) -> pd.DataFrame:
    """Attach corpus-internal citation ``in_flow`` / ``out_flow`` to a flow table.

    Each citation edge ``citing_pmid -> cited_pmid`` is attributed to a
    ``(topic, journal_slug, year)`` cell via the ``papers_topics`` mapping at *both*
    endpoints:

    * **out_flow** of a cell = number of edges whose *citing* paper falls in that
      cell (citations *made by* the cell's papers — references it sends out).
    * **in_flow** of a cell = number of edges whose *cited* paper falls in that cell
      (citations *received by* the cell's papers).

    Edges are corpus-internal (both endpoints in the 10-journal panel; see
    ``V2/CONTEXT.md`` §3 — Kuzu ``CITES`` has both endpoints in-corpus). Citation
    year follows the endpoint's own ``(topic, journal, year)`` row, so a cell's
    ``out_flow`` is anchored to the citing paper's publication year.

    Parameters
    ----------
    flow :
        A flow table from :func:`build_flow_table` (without citation columns), keyed
        on ``[topic_key, "journal_slug", "year"]``.
    papers_topics :
        The same per-paper frame used to build ``flow``, carrying ``["pmid",
        topic_key, "journal_slug", "year"]``. Provides the
        ``pmid -> (topic, journal, year)`` attribution for each citation endpoint.
    citation_edges :
        Edge list with at least ``["citing_pmid", "cited_pmid"]`` (extra columns
        ignored). ``pmid`` dtype need only be join-compatible with
        ``papers_topics["pmid"]``; the builder casts both to string.
    topic_key :
        Topic-grain column. Default ``"topic_id"``.

    Returns
    -------
    pandas.DataFrame
        ``flow`` with two added integer columns ``out_flow`` and ``in_flow``
        (``int64``, zero-filled for cells no edge touches). Row order and all other
        columns are preserved.

    Notes
    -----
    Only edges whose *relevant* endpoint resolves to a cell present in ``flow`` are
    counted; an edge whose citing paper is a noise-topic paper absent from ``flow``
    contributes no ``out_flow`` (and likewise for ``in_flow``). The total
    ``out_flow`` and total ``in_flow`` across all cells are therefore both ``<=``
    the edge count and need not be equal (some endpoints are unattributed).
    """
    import pandas as pd

    cell_cols = [topic_key, "journal_slug", "year"]
    attrib = papers_topics.loc[:, ["pmid", *cell_cols]].dropna(subset=["pmid"]).copy()
    attrib["pmid"] = attrib["pmid"].astype("string")

    edges = citation_edges.loc[:, ["citing_pmid", "cited_pmid"]].copy()
    edges["citing_pmid"] = edges["citing_pmid"].astype("string")
    edges["cited_pmid"] = edges["cited_pmid"].astype("string")

    # out_flow: attribute the CITING endpoint to a cell, then count edges per cell.
    out_attr = edges.merge(
        attrib.rename(columns={"pmid": "citing_pmid"}), on="citing_pmid", how="inner"
    )
    out_counts = (
        out_attr.groupby(cell_cols, dropna=False).size().reset_index(name="out_flow")
        if not out_attr.empty
        else pd.DataFrame({**{c: [] for c in cell_cols}, "out_flow": []})
    )

    # in_flow: attribute the CITED endpoint to a cell, then count edges per cell.
    in_attr = edges.merge(
        attrib.rename(columns={"pmid": "cited_pmid"}), on="cited_pmid", how="inner"
    )
    in_counts = (
        in_attr.groupby(cell_cols, dropna=False).size().reset_index(name="in_flow")
        if not in_attr.empty
        else pd.DataFrame({**{c: [] for c in cell_cols}, "in_flow": []})
    )

    out = flow.merge(out_counts, on=cell_cols, how="left")
    out = out.merge(in_counts, on=cell_cols, how="left")
    out["out_flow"] = out["out_flow"].fillna(0).astype("int64")
    out["in_flow"] = out["in_flow"].fillna(0).astype("int64")
    return out


def build_flow_table(
    papers_topics: pd.DataFrame,
    *,
    citation_edges: pd.DataFrame | None = None,
    grain: str = "leaf",
    min_papers: int = 1,
) -> pd.DataFrame:
    """Build the per-``(topic, journal_slug, year)`` flow table.

    Parameters
    ----------
    papers_topics :
        Tidy frame, one row per paper, already carrying the canonical journal slug:
        at least ``["pmid", "journal_slug", <topic_key>, "year"]`` where
        ``<topic_key>`` is ``"topic_id"`` for ``grain="leaf"`` or ``"mid_level_id"``
        for ``grain="mid"`` (see :func:`topic_key_for_grain`). The builder resolves
        ``journal_slug`` by the ``pmid -> papers_distinct`` join; this function
        trusts it (and :func:`assert_canonical_journals` guards it). Extra columns
        (e.g. novelty ``sem_nov_mean``) are ignored — volume is a row count, so
        novelty aggregation is left to downstream layers. Rows missing the topic
        key, ``journal_slug``, or ``year`` are dropped.
    citation_edges :
        Optional corpus-internal edge list ``["citing_pmid", "cited_pmid"]``. If
        given, :func:`add_citation_flows` attaches integer ``in_flow`` / ``out_flow``
        columns. If ``None`` (default) those columns are still emitted but as
        ``out_flow = 0`` and ``in_flow = NaN`` (float) — documented sentinel meaning
        **"citation flows not computed for this build"** (distinct from a genuine
        zero, which only appears when edges were supplied). Consumers should treat a
        ``NaN`` ``in_flow`` as "unknown / not requested".
    grain :
        ``"leaf"`` (default) or ``"mid"``; selects the topic key per
        :func:`topic_key_for_grain`. The caller must supply the matching column.
    min_papers :
        Per-year threshold for the first-appearance anchor (forwarded to
        :func:`first_appearance`). The ``n_papers`` volume itself is the raw
        per-year count and is unaffected by ``min_papers``. Default ``1``.

    Returns
    -------
    pandas.DataFrame
        One row per ``(topic, journal_slug, year)`` cell, sorted by topic key then
        ``journal_slug`` then ``year``, with columns:

        ``<topic_key>`` : int64
            ``"topic_id"`` (leaf) or ``"mid_level_id"`` (mid).
        ``journal_slug`` : object/str
            The canonical journal (one of the clean ten).
        ``year`` : int64
            Publication year.
        ``n_papers`` : int64
            Volume — papers in this cell.
        ``is_first_appearance`` : bool
            ``True`` iff ``year`` is the earliest year this (topic, journal) reached
            ``min_papers`` papers in a single year (the cascade anchor).
        ``out_flow`` : int64 or 0
            Corpus-internal citations *made by* the cell (0/NaN sentinel if no
            ``citation_edges``).
        ``in_flow`` : int64 or NaN
            Corpus-internal citations *received by* the cell (NaN sentinel if no
            ``citation_edges``).

    Raises
    ------
    ValueError
        If ``grain`` is unknown, if the required topic key / ``journal_slug`` /
        ``year`` columns are absent, or if a non-canonical journal slug appears.

    Notes
    -----
    *Panel-conditional caveat:* ``is_first_appearance`` means "first within our 10
    journals", never "first in the world" — the true origin may be a journal never
    harvested (``V2/CONTEXT.md`` §0.2). Volume counts every paper in the cell; the
    first-appearance flag uses the same per-year ``min_papers`` rule as
    :func:`scifield.findings.seeding.first_publication_year`.
    """
    import numpy as np
    import pandas as pd

    topic_key = topic_key_for_grain(grain)
    required = [topic_key, "journal_slug", "year"]
    missing = [c for c in required if c not in papers_topics.columns]
    if missing:
        raise ValueError(
            f"papers_topics is missing required column(s) for grain={grain!r}: {missing}"
        )

    work = papers_topics.loc[:, required].dropna(subset=required).copy()
    assert_canonical_journals(work["journal_slug"])

    if work.empty:
        empty = pd.DataFrame(
            {
                topic_key: pd.Series([], dtype="int64"),
                "journal_slug": pd.Series([], dtype="object"),
                "year": pd.Series([], dtype="int64"),
                "n_papers": pd.Series([], dtype="int64"),
                "is_first_appearance": pd.Series([], dtype="bool"),
                "out_flow": pd.Series([], dtype="int64"),
                "in_flow": pd.Series([], dtype="float64"),
            }
        )
        return empty

    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.dropna(subset=["year"])
    work["year"] = work["year"].astype("int64")

    # Volume: papers per (topic, journal, year).
    flow = (
        work.groupby(required, dropna=False)
        .size()
        .reset_index(name="n_papers")
        .reset_index(drop=True)
    )
    flow["n_papers"] = flow["n_papers"].astype("int64")

    # First-appearance anchor: earliest qualifying year per (topic, journal).
    first = first_appearance(work, topic_key=topic_key, min_papers=min_papers)
    first = first.rename(columns={"first_year": "_first_year"})
    flow = flow.merge(first, on=[topic_key, "journal_slug"], how="left")
    flow["is_first_appearance"] = (flow["year"] == flow["_first_year"]).fillna(False)
    flow = flow.drop(columns=["_first_year"])

    flow = flow.sort_values([topic_key, "journal_slug", "year"]).reset_index(drop=True)

    if citation_edges is None:
        # Documented sentinel: citation flows not computed for this build.
        flow["out_flow"] = 0
        flow["in_flow"] = np.nan
        flow["in_flow"] = flow["in_flow"].astype("float64")
        return flow

    return add_citation_flows(
        flow,
        _work_with_pmid(papers_topics, topic_key),
        citation_edges,
        topic_key=topic_key,
    )


def _work_with_pmid(papers_topics: pd.DataFrame, topic_key: str) -> pd.DataFrame:
    """Project ``papers_topics`` to the columns :func:`add_citation_flows` needs.

    Internal helper: keeps ``["pmid", topic_key, "journal_slug", "year"]`` with
    ``year`` coerced to ``int64`` and the cell columns non-null, matching the
    filtering applied to the volume frame so citation attribution lands on the same
    cells.

    Parameters
    ----------
    papers_topics :
        The per-paper input frame (must include ``pmid``).
    topic_key :
        Topic-grain column.

    Returns
    -------
    pandas.DataFrame
        The projected, cleaned frame.

    Raises
    ------
    ValueError
        If ``pmid`` is absent (required to attribute citation endpoints).
    """
    import pandas as pd

    if "pmid" not in papers_topics.columns:
        raise ValueError("papers_topics must carry 'pmid' to compute citation flows")
    cols = ["pmid", topic_key, "journal_slug", "year"]
    out = papers_topics.loc[:, cols].dropna(subset=[topic_key, "journal_slug", "year"]).copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out.dropna(subset=["year"])
    out["year"] = out["year"].astype("int64")
    return out
