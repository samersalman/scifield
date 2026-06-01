"""Build the canonical Kùzu property graph for SciField novelty (V1-S10).

This module materialises a `Kùzu <https://kuzudb.com>`_ property graph from
the curated corpus in ``papers.duckdb`` plus the BERTopic assignment table in
``topics.parquet``. The graph is the structural substrate for Phase-4 novelty
scoring (citation structure, co-authorship, topic membership).

Schema
------
Node tables:

``Paper``
    Keyed by ``pmid`` (STRING). Carries ``title``, ``year`` and ``journal_slug``.
``Author``
    Keyed by ``author_canonical_id`` (STRING), the disambiguated author id.
    Carries ``display_name``.
``Journal``
    Keyed by ``journal_slug`` (STRING). Carries ``journal`` (display name).
``Institution``
    Keyed by ``institution_canonical_id`` (STRING). Carries ``display_name``
    and ``country_code``.
``Topic``
    Keyed by ``topic_id`` (INT64).

Relationship tables:

``CITES`` (Paper -> Paper)
    Corpus-internal citations only: a ``references_out`` row contributes an
    edge **only** when ``ref_openalex_id`` resolves, via
    ``openalex_works.openalex_id``, to a PMID that is itself in the corpus.
``AUTHORED_BY`` (Paper -> Author)
``AFFILIATED_WITH`` (Author -> Institution)
``PUBLISHED_IN`` (Paper -> Journal)
``ASSIGNED_TO`` (Paper -> Topic)

Design decisions
----------------
* **AFFILIATED_WITH endpoint = Author -> Institution.** ``authorships``
  exposes the disambiguated ``author_canonical_id`` (the ``Author`` primary
  key) but no institution; ``paper_institutions`` exposes
  ``institution_canonical_id`` (the ``Institution`` primary key) but no
  canonical author id. The two are joinable only on ``(pmid,
  author_position)``. Joining them yields the disambiguated author-to-
  institution relationship, which is the semantically correct edge and the
  only one that connects both canonical-id node tables. A paper-level edge
  would discard the author entity the graph is organised around.

* **Idempotency.** :func:`build_kuzu_graph` is destructive-by-default: it
  opens (or creates) the database at ``kuzu_dir`` and drops any pre-existing
  graph tables before recreating them, so re-running yields a clean rebuild
  rather than failing or duplicating data.

* **Bulk load.** Each node/edge set is computed in DuckDB, exported to a
  temporary Parquet file, then ingested with Kùzu's ``COPY <Table> FROM
  '<parquet>'``. This is the fastest ingestion path for Kùzu 0.11.x.

* **Topics.** ``ASSIGNED_TO`` excludes BERTopic noise (``is_noise = true`` /
  ``topic_id = -1``); noise is not a real topic. ``topics.parquet`` stores
  ``pmid`` as INT64, so it is cast to STRING to match ``Paper.pmid``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import duckdb
import kuzu

from scifield.thematic.dedup import ensure_papers_distinct_view

__all__ = ["build_kuzu_graph", "NODE_DDL", "REL_DDL"]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Schema DDL (Kùzu 0.11.x syntax).
# --------------------------------------------------------------------------- #
NODE_DDL: dict[str, str] = {
    "Paper": (
        "CREATE NODE TABLE Paper("
        "pmid STRING, title STRING, year INT64, journal_slug STRING, "
        "PRIMARY KEY(pmid))"
    ),
    "Author": (
        "CREATE NODE TABLE Author("
        "author_canonical_id STRING, display_name STRING, "
        "PRIMARY KEY(author_canonical_id))"
    ),
    "Journal": (
        "CREATE NODE TABLE Journal(journal_slug STRING, journal STRING, PRIMARY KEY(journal_slug))"
    ),
    "Institution": (
        "CREATE NODE TABLE Institution("
        "institution_canonical_id STRING, display_name STRING, "
        "country_code STRING, PRIMARY KEY(institution_canonical_id))"
    ),
    "Topic": "CREATE NODE TABLE Topic(topic_id INT64, PRIMARY KEY(topic_id))",
}

REL_DDL: dict[str, str] = {
    "CITES": "CREATE REL TABLE CITES(FROM Paper TO Paper)",
    "AUTHORED_BY": "CREATE REL TABLE AUTHORED_BY(FROM Paper TO Author)",
    "AFFILIATED_WITH": "CREATE REL TABLE AFFILIATED_WITH(FROM Author TO Institution)",
    "PUBLISHED_IN": "CREATE REL TABLE PUBLISHED_IN(FROM Paper TO Journal)",
    "ASSIGNED_TO": "CREATE REL TABLE ASSIGNED_TO(FROM Paper TO Topic)",
}

# SELECT statements producing exactly the columns each COPY target expects, in
# order. Node selects emit primary-key + property columns; rel selects emit the
# FROM key then the TO key. ``{topics}`` is substituted with the parquet path.
_NODE_QUERIES: dict[str, str] = {
    "Paper": ("SELECT pmid, title, year, journal_slug FROM papers_distinct WHERE pmid IS NOT NULL"),
    "Author": (
        "SELECT author_canonical_id, any_value(author_display_name) AS display_name "
        "FROM authorships WHERE author_canonical_id IS NOT NULL "
        "GROUP BY author_canonical_id"
    ),
    "Journal": (
        "SELECT journal_slug, any_value(journal) AS journal "
        "FROM papers_distinct WHERE journal_slug IS NOT NULL "
        "GROUP BY journal_slug"
    ),
    "Institution": (
        "SELECT institution_canonical_id, display_name, country_code "
        "FROM institutions WHERE institution_canonical_id IS NOT NULL"
    ),
    "Topic": (
        "SELECT DISTINCT topic_id FROM read_parquet('{topics}') "
        "WHERE is_noise = false AND topic_id <> -1"
    ),
}

_REL_QUERIES: dict[str, str] = {
    # Corpus-internal citations: resolve ref_openalex_id -> cited corpus pmid.
    "CITES": (
        "SELECT DISTINCT r.citing_pmid AS from_pmid, ow.pmid AS to_pmid "
        "FROM references_out r "
        "JOIN openalex_works ow ON r.ref_openalex_id = ow.openalex_id "
        "WHERE ow.pmid IS NOT NULL "
        "AND r.citing_pmid IN (SELECT pmid FROM papers_distinct) "
        "AND ow.pmid IN (SELECT pmid FROM papers_distinct)"
    ),
    "AUTHORED_BY": (
        "SELECT DISTINCT pmid, author_canonical_id FROM authorships "
        "WHERE author_canonical_id IS NOT NULL "
        "AND pmid IN (SELECT pmid FROM papers_distinct)"
    ),
    # Author -> Institution via the (pmid, author_position) join.
    "AFFILIATED_WITH": (
        "SELECT DISTINCT a.author_canonical_id, pi.institution_canonical_id "
        "FROM authorships a "
        "JOIN paper_institutions pi "
        "ON a.pmid = pi.pmid AND a.author_position = pi.author_position "
        "WHERE a.author_canonical_id IS NOT NULL "
        "AND pi.institution_canonical_id IS NOT NULL"
    ),
    "PUBLISHED_IN": (
        "SELECT DISTINCT pmid, journal_slug FROM papers_distinct "
        "WHERE pmid IS NOT NULL AND journal_slug IS NOT NULL"
    ),
    "ASSIGNED_TO": (
        "SELECT CAST(pmid AS VARCHAR) AS pmid, topic_id "
        "FROM read_parquet('{topics}') "
        "WHERE is_noise = false AND topic_id <> -1 "
        "AND CAST(pmid AS VARCHAR) IN (SELECT pmid FROM papers_distinct)"
    ),
}


def _reset_schema(con: kuzu.Connection) -> None:
    """Drop existing graph tables (rels first) then recreate the full schema.

    Parameters
    ----------
    con
        An open Kùzu connection.
    """
    for name in REL_DDL:
        con.execute(f"DROP TABLE IF EXISTS {name}")
    for name in NODE_DDL:
        con.execute(f"DROP TABLE IF EXISTS {name}")
    for ddl in NODE_DDL.values():
        con.execute(ddl)
    for ddl in REL_DDL.values():
        con.execute(ddl)


def _bulk_copy(
    *,
    duck: duckdb.DuckDBPyConnection,
    kcon: kuzu.Connection,
    table: str,
    query: str,
    tmp_dir: Path,
) -> int:
    """Export a DuckDB query to Parquet and bulk-``COPY`` it into Kùzu.

    Parameters
    ----------
    duck
        Read-side DuckDB connection (the corpus database).
    kcon
        Target Kùzu connection.
    table
        Name of the Kùzu node or rel table to load.
    query
        DuckDB ``SELECT`` producing the columns ``table`` expects, in order.
    tmp_dir
        Directory for the intermediate Parquet file.

    Returns
    -------
    int
        Number of rows copied into ``table``.
    """
    out = (tmp_dir / f"{table}.parquet").as_posix()
    duck.execute(f"COPY ({query}) TO '{out}' (FORMAT PARQUET)")
    row = duck.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()
    n = int(row[0]) if row is not None else 0
    if n:
        kcon.execute(f"COPY {table} FROM '{out}'")
    return int(n)


def build_kuzu_graph(
    *,
    duckdb_path: Path,
    topics_parquet: Path,
    kuzu_dir: Path,
) -> dict[str, int]:
    """Build the canonical Kùzu property graph from the corpus.

    Constructs the full Paper/Author/Journal/Institution/Topic graph (see the
    module docstring for the schema) at ``kuzu_dir``, reading nodes and edges
    from ``duckdb_path`` and ``topics_parquet``. The operation is offline (no
    network) and idempotent: any pre-existing graph tables at ``kuzu_dir`` are
    dropped and rebuilt.

    Parameters
    ----------
    duckdb_path
        Path to ``papers.duckdb``. Opened read-write because the
        ``papers_distinct`` view may need to be (re)created via
        :func:`scifield.thematic.dedup.ensure_papers_distinct_view`.
    topics_parquet
        Path to ``topics.parquet`` (columns ``pmid``, ``topic_id``,
        ``is_noise``).
    kuzu_dir
        Destination directory for the persistent Kùzu database.

    Returns
    -------
    dict[str, int]
        Row counts keyed by table name, e.g. ``{"Paper": ..., "Author": ...,
        "CITES": ..., "AUTHORED_BY": ..., ...}``.
    """
    duckdb_path = Path(duckdb_path)
    topics_parquet = Path(topics_parquet)
    kuzu_dir = Path(kuzu_dir)
    kuzu_dir.parent.mkdir(parents=True, exist_ok=True)

    topics_posix = topics_parquet.as_posix()
    counts: dict[str, int] = {}

    # papers_distinct may be a view that must be (re)created -> read-write.
    duck = duckdb.connect(str(duckdb_path), read_only=False)
    try:
        ensure_papers_distinct_view(duck)

        db = kuzu.Database(str(kuzu_dir))
        kcon = kuzu.Connection(db)
        _reset_schema(kcon)

        with tempfile.TemporaryDirectory(prefix="scifield_kuzu_") as td:
            tmp_dir = Path(td)
            for table, query in _NODE_QUERIES.items():
                q = query.format(topics=topics_posix)
                counts[table] = _bulk_copy(
                    duck=duck, kcon=kcon, table=table, query=q, tmp_dir=tmp_dir
                )
            for table, query in _REL_QUERIES.items():
                q = query.format(topics=topics_posix)
                counts[table] = _bulk_copy(
                    duck=duck, kcon=kcon, table=table, query=q, tmp_dir=tmp_dir
                )

        kcon.close()
        db.close()
    finally:
        duck.close()

    logger.info("Kùzu graph built at %s: %s", kuzu_dir, counts)
    return counts
