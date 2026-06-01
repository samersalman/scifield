"""Tests for :mod:`scifield.novelty.kuzu_loader` on a tiny synthetic corpus.

No real data and no network: a minimal ``papers.duckdb`` and ``topics.parquet``
are built in ``tmp_path``, the graph is materialised with
:func:`build_kuzu_graph`, and node/edge counts are asserted both from the
returned dict and via Cypher ``count`` queries against the resulting Kùzu DB.

The synthetic data is hand-constructed so every expected count is known,
including the key property that ``CITES`` is corpus-internal only: a reference
to a non-corpus OpenAlex id must not create an edge.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import kuzu

from scifield.novelty.kuzu_loader import build_kuzu_graph


def _build_synthetic_duckdb(path: Path) -> None:
    """Create a tiny corpus database with the columns the loader reads.

    A ``papers`` table is created (not ``papers_distinct``); the loader calls
    ``ensure_papers_distinct_view`` which builds the ``papers_distinct`` view
    from ``papers``. ``papers`` therefore carries the ``abstract`` column the
    dedup tiebreak orders on. PMIDs are numeric strings ('1','2','3') so the
    integer pmids in ``topics.parquet`` join after the loader's cast.

    Corpus papers: 1, 2, 3 (3 papers). Paper 1 has a duplicate row (different
    abstract length) to exercise dedup -> still 3 distinct papers.

    OpenAlex map: 1->W1, 2->W2, 3->W3.

    References:
      * 1 -> W2   (internal: cited paper 2 is in corpus)  -> CITES edge
      * 1 -> W3   (internal: cited paper 3 is in corpus)  -> CITES edge
      * 2 -> W1   (internal)                              -> CITES edge
      * 2 -> W999 (external: not a corpus work)           -> NO edge

    Authors (authorships):
      * 1 pos0 -> A1, 1 pos1 -> A2
      * 2 pos0 -> A1
      * 3 pos0 -> A3
    => 4 distinct (pmid, author) AUTHORED_BY edges; 3 distinct authors.

    Institutions (paper_institutions), joined on (pmid, author_position):
      * 1 pos0 -> I1, 1 pos1 -> I2
      * 2 pos0 -> I1
      * 3 pos0 -> I2
    => Author->Institution distinct: A1->I1, A2->I2, A3->I2 = 3 edges.
       (A1 appears at 1/pos0/I1 and 2/pos0/I1 -> deduped to one edge.)

    Journals: slug j1 used by papers 1, 2; slug j2 used by paper 3.
    """
    con = duckdb.connect(str(path))

    con.execute(
        """
        CREATE TABLE papers (
            pmid VARCHAR, title VARCHAR, year INTEGER,
            journal_slug VARCHAR, journal VARCHAR, abstract VARCHAR
        )
        """
    )
    con.execute(
        """
        INSERT INTO papers VALUES
            ('1', 'Paper One',   2020, 'j1', 'Journal One',  'short'),
            ('1', 'Paper One',   2020, 'j1', 'Journal One',  'a much longer abstract'),
            ('2', 'Paper Two',   2021, 'j1', 'Journal One',  'abstract two'),
            ('3', 'Paper Three', 2022, 'j2', 'Journal Two',  'abstract three')
        """
    )

    con.execute("CREATE TABLE openalex_works (pmid VARCHAR, openalex_id VARCHAR)")
    con.execute(
        """
        INSERT INTO openalex_works VALUES
            ('1', 'W1'), ('2', 'W2'), ('3', 'W3')
        """
    )

    con.execute(
        """
        CREATE TABLE references_out (
            citing_pmid VARCHAR, ref_openalex_id VARCHAR,
            ref_pmid_if_known VARCHAR
        )
        """
    )
    con.execute(
        """
        INSERT INTO references_out VALUES
            ('1', 'W2',   NULL),
            ('1', 'W3',   NULL),
            ('2', 'W1',   NULL),
            ('2', 'W999', NULL)
        """
    )

    con.execute(
        """
        CREATE TABLE authorships (
            pmid VARCHAR, author_position INTEGER,
            author_canonical_id VARCHAR, author_display_name VARCHAR
        )
        """
    )
    con.execute(
        """
        INSERT INTO authorships VALUES
            ('1', 0, 'A1', 'Alice'),
            ('1', 1, 'A2', 'Bob'),
            ('2', 0, 'A1', 'Alice'),
            ('3', 0, 'A3', 'Carol')
        """
    )

    con.execute(
        """
        CREATE TABLE paper_institutions (
            pmid VARCHAR, author_position INTEGER,
            institution_canonical_id VARCHAR
        )
        """
    )
    con.execute(
        """
        INSERT INTO paper_institutions VALUES
            ('1', 0, 'I1'),
            ('1', 1, 'I2'),
            ('2', 0, 'I1'),
            ('3', 0, 'I2')
        """
    )

    con.execute(
        """
        CREATE TABLE institutions (
            institution_canonical_id VARCHAR, display_name VARCHAR,
            country_code VARCHAR
        )
        """
    )
    con.execute(
        """
        INSERT INTO institutions VALUES
            ('I1', 'Inst One', 'US'),
            ('I2', 'Inst Two', 'GB')
        """
    )

    con.close()


def _build_synthetic_topics(path: Path) -> None:
    """Write a tiny topics parquet via DuckDB.

    Assignments (pmids are integers, matching the numeric corpus pmids after
    the loader's cast-to-string): 1->10, 2->10, 3->20, plus one noise row
    (3->-1, noise). => 2 distinct non-noise topics (10, 20); 3 non-noise
    ASSIGNED_TO edges. The noise row must be excluded from both Topic nodes
    and edges.
    """
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT * FROM (VALUES
                (CAST(1 AS BIGINT), CAST(10 AS BIGINT), FALSE),
                (CAST(2 AS BIGINT), CAST(10 AS BIGINT), FALSE),
                (CAST(3 AS BIGINT), CAST(20 AS BIGINT), FALSE),
                (CAST(3 AS BIGINT), CAST(-1 AS BIGINT), TRUE)
            ) AS t(pmid, topic_id, is_noise)
        ) TO '{path.as_posix()}' (FORMAT PARQUET)
        """
    )
    con.close()


def _count(kcon: kuzu.Connection, cypher: str) -> int:
    return int(kcon.execute(cypher).get_next()[0])


def test_build_kuzu_graph_counts(tmp_path: Path) -> None:
    """Synthetic end-to-end build yields the hand-computed node/edge counts."""
    duckdb_path = tmp_path / "papers.duckdb"
    topics_parquet = tmp_path / "topics.parquet"
    kuzu_dir = tmp_path / "graph.kuzu"

    _build_synthetic_duckdb(duckdb_path)
    _build_synthetic_topics(topics_parquet)

    counts = build_kuzu_graph(
        duckdb_path=duckdb_path,
        topics_parquet=topics_parquet,
        kuzu_dir=kuzu_dir,
    )

    # Node counts.
    assert counts["Paper"] == 3
    assert counts["Author"] == 3
    assert counts["Journal"] == 2
    assert counts["Institution"] == 2
    assert counts["Topic"] == 2  # noise topic excluded

    # Edge counts.
    assert counts["CITES"] == 3  # W999 external reference excluded
    assert counts["AUTHORED_BY"] == 4
    assert counts["AFFILIATED_WITH"] == 3
    assert counts["PUBLISHED_IN"] == 3
    assert counts["ASSIGNED_TO"] == 3  # 3 non-noise assignments; noise excluded

    # Re-open the persisted DB and confirm the same counts via Cypher.
    db = kuzu.Database(str(kuzu_dir))
    kcon = kuzu.Connection(db)
    try:
        assert _count(kcon, "MATCH (n:Paper) RETURN count(n)") == 3
        assert _count(kcon, "MATCH (n:Author) RETURN count(n)") == 3
        assert _count(kcon, "MATCH (n:Journal) RETURN count(n)") == 2
        assert _count(kcon, "MATCH (n:Institution) RETURN count(n)") == 2
        assert _count(kcon, "MATCH (n:Topic) RETURN count(n)") == 2

        assert _count(kcon, "MATCH (:Paper)-[r:CITES]->(:Paper) RETURN count(r)") == 3
        assert _count(kcon, "MATCH (:Paper)-[r:AUTHORED_BY]->(:Author) RETURN count(r)") == 4
        assert (
            _count(
                kcon,
                "MATCH (:Author)-[r:AFFILIATED_WITH]->(:Institution) RETURN count(r)",
            )
            == 3
        )
        assert _count(kcon, "MATCH (:Paper)-[r:PUBLISHED_IN]->(:Journal) RETURN count(r)") == 3
        assert _count(kcon, "MATCH (:Paper)-[r:ASSIGNED_TO]->(:Topic) RETURN count(r)") == 3

        # Specific structural check: the external W999 reference produced no edge.
        # Paper 2 cites only paper 1 (W1); the W999 reference must NOT add an edge.
        p2_out = _count(
            kcon,
            "MATCH (a:Paper {pmid: '2'})-[r:CITES]->(:Paper) RETURN count(r)",
        )
        assert p2_out == 1
    finally:
        kcon.close()
        db.close()


def test_build_kuzu_graph_idempotent_rebuild(tmp_path: Path) -> None:
    """Running the build twice into the same dir rebuilds cleanly (no dupes)."""
    duckdb_path = tmp_path / "papers.duckdb"
    topics_parquet = tmp_path / "topics.parquet"
    kuzu_dir = tmp_path / "graph.kuzu"

    _build_synthetic_duckdb(duckdb_path)
    _build_synthetic_topics(topics_parquet)

    first = build_kuzu_graph(
        duckdb_path=duckdb_path,
        topics_parquet=topics_parquet,
        kuzu_dir=kuzu_dir,
    )
    second = build_kuzu_graph(
        duckdb_path=duckdb_path,
        topics_parquet=topics_parquet,
        kuzu_dir=kuzu_dir,
    )

    assert first == second

    db = kuzu.Database(str(kuzu_dir))
    kcon = kuzu.Connection(db)
    try:
        assert _count(kcon, "MATCH (n:Paper) RETURN count(n)") == 3
        assert _count(kcon, "MATCH (:Paper)-[r:CITES]->(:Paper) RETURN count(r)") == 3
    finally:
        kcon.close()
        db.close()
