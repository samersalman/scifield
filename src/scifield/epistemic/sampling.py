"""Stratified hand-labeling sampler for V1-S07 epistemic extraction.

Builds the 500-row ``handlabel_sample.parquet`` that drives the V1-S08
hand-labeling sprint and the 50-abstract Claude-Code pilot. The pool is
``papers_distinct`` filtered to abstract-bearing rows (canonical filter
``abstract IS NOT NULL AND length(abstract) > 50`` — matches the embed
step's filter in ``conf/thematic/embed.yaml``, NOT the precomputed
``has_abstract`` column, for cross-stage consistency).

Stratification is **journal_slug × era**: 4 era buckets (``pre2000``,
``2000-2009``, ``2010-2019``, ``2020+``) × the 10 journal slugs in the
V1 corpus = 40 cells. (The plan text mentions 41 cells with floor 689 —
both numbers are slightly off versus the actual snapshot, which has 40
cells with a per-cell floor of 624 abstract-bearing papers. The plan is
not updated to avoid churn; this comment is the authoritative correction
for future readers.)

Allocation uses **largest-remainder (Hare)** rounding so the cell
targets sum to exactly ``cfg.n_sample``. Within-cell draws are uniform
without replacement, seeded per cell via a deterministic ``(seed,
journal, era)`` hash so the global ``cfg.seed`` reproduces the sample
exactly.

Topic IDs are attached via a left join on ``data/v1/topics.parquet``
(``pmid, topic_id, is_noise``); papers without a topic row are kept
with ``topic_id`` set to a nullable Int64 ``<NA>``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from scifield.thematic.dedup import ensure_papers_distinct_view

__all__ = ["SamplingConfig", "stratified_sample"]


_DEFAULT_ERAS: tuple[str, ...] = ("pre2000", "2000-2009", "2010-2019", "2020+")


@dataclass(frozen=True)
class SamplingConfig:
    """Frozen config for :func:`stratified_sample`.

    Field names mirror ``conf/epistemic/v1.yaml`` so a CLI loader can
    splat the YAML's ``input`` + ``sampling`` blocks directly.
    """

    duckdb_path: Path
    topics_parquet: Path
    n_sample: int = 500
    seed: int = 20260522
    eras: tuple[str, ...] = field(default=_DEFAULT_ERAS)
    topic_coverage_min: int = 80


def _largest_remainder(counts: np.ndarray, n_total: int) -> np.ndarray:
    """Hare/largest-remainder allocation summing to exactly ``n_total``.

    Each cell's ideal share is ``n_total * counts[i] / sum(counts)``;
    we floor each, then distribute the remaining seats to the cells
    with the largest fractional remainders (ties broken by index order,
    which is stable as long as the caller orders cells deterministically).
    Cells with zero pool count get zero allocation by construction.
    """
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros_like(counts)

    exact = counts.astype(np.float64) * (n_total / total)
    floors = np.floor(exact).astype(np.int64)
    remainder = int(n_total - floors.sum())
    if remainder > 0:
        frac = exact - floors
        # Tiebreak: larger pool first, then earlier index — both stable.
        order = np.lexsort((np.arange(len(counts)), -counts, -frac))
        winners = order[:remainder]
        floors[winners] += 1

    # Cap to pool size in case Hare overshoots a small cell (shouldn't
    # happen at 500/40 with floor ~600, but guards future config shifts).
    over = floors > counts
    if over.any():
        spare = int((floors[over] - counts[over]).sum())
        floors = np.minimum(floors, counts)
        # Redistribute spare to cells with remaining headroom, largest
        # remainder first.
        while spare > 0:
            headroom = counts - floors
            if headroom.sum() == 0:
                break
            frac = exact - np.floor(exact)
            order = np.lexsort((np.arange(len(counts)), -headroom, -frac))
            placed = False
            for i in order:
                if headroom[i] > 0:
                    floors[i] += 1
                    spare -= 1
                    placed = True
                    break
            if not placed:
                break
    return floors


def _cell_seed(global_seed: int, journal: str, era: str) -> int:
    """Stable per-cell seed derived from the global seed + cell key.

    Uses BLAKE2b truncated to 8 bytes so the result fits in a 64-bit
    unsigned int that ``numpy.random.default_rng`` accepts. Deterministic
    under fixed ``global_seed`` regardless of Python hash randomization.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(global_seed).encode("utf-8"))
    h.update(b"\x00")
    h.update(journal.encode("utf-8"))
    h.update(b"\x00")
    h.update(era.encode("utf-8"))
    return int.from_bytes(h.digest(), "big", signed=False)


def _load_pool(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Read the abstract-bearing pool from ``papers_distinct``.

    Uses the canonical ``abstract IS NOT NULL AND length(abstract) > 50``
    filter (matches ``scifield embed``) rather than the precomputed
    ``has_abstract`` column, so the pool is consistent with the corpus
    that produced the topic model.
    """
    sql = """
        SELECT CAST(pmid AS BIGINT) AS pmid,
               journal_slug AS journal,
               year,
               CASE
                 WHEN year < 2000 THEN 'pre2000'
                 WHEN year < 2010 THEN '2000-2009'
                 WHEN year < 2020 THEN '2010-2019'
                 ELSE '2020+'
               END AS era,
               title,
               abstract
        FROM papers_distinct
        WHERE abstract IS NOT NULL AND length(abstract) > 50
    """
    return con.execute(sql).fetch_df()


def _attach_topics(pool: pd.DataFrame, topics_parquet: Path) -> pd.DataFrame:
    """Left-join ``topic_id`` from the V1-S06 topics parquet.

    Papers absent from the topics table keep ``topic_id`` as ``<NA>``
    under :class:`pandas.Int64Dtype`. We deliberately do not drop them —
    the labeling sample is meant to cover the corpus, not just the
    topic-clustered slice.
    """
    topics = pd.read_parquet(topics_parquet, columns=["pmid", "topic_id"])
    topics["pmid"] = topics["pmid"].astype("int64")
    topics["topic_id"] = topics["topic_id"].astype("Int64")
    merged = pool.merge(topics, on="pmid", how="left")
    # Ensure nullable int dtype even if the merge produced object/float NaN.
    merged["topic_id"] = merged["topic_id"].astype("Int64")
    return merged


def stratified_sample(con: duckdb.DuckDBPyConnection, cfg: SamplingConfig) -> pd.DataFrame:
    """Draw the V1-S07 hand-labeling sample.

    Returns a DataFrame with columns
    ``pmid, journal, year, era, topic_id, title, abstract`` (in that
    order), sorted by ``pmid`` ascending for stable downstream ordering.
    """
    ensure_papers_distinct_view(con)
    pool = _load_pool(con)
    if pool.empty:
        raise AssertionError("abstract-bearing pool is empty; cannot sample")

    pool = _attach_topics(pool, Path(cfg.topics_parquet))

    # Restrict to configured eras and to non-null journal slugs (defensive;
    # the harvest stage should already have populated journal_slug).
    pool = pool[pool["era"].isin(list(cfg.eras))]
    pool = pool[pool["journal"].notna()]
    if pool.empty:
        raise AssertionError("pool empty after era/journal filtering")

    # Build a deterministic ordering of (journal, era) cells. Sorting by
    # (journal, era) gives a stable index order for the allocator and
    # downstream debugging; the era list is iterated in cfg.eras order.
    journals = sorted(pool["journal"].unique())
    cells: list[tuple[str, str]] = [(j, e) for j in journals for e in cfg.eras]

    # Sort the pool by pmid first so cell membership lists have a
    # deterministic order regardless of DuckDB's underlying row order
    # (papers_distinct's ROW_NUMBER tiebreaks can shuffle equal-length
    # abstracts between runs).
    pool = pool.sort_values("pmid", kind="mergesort").reset_index(drop=True)

    cell_to_idx = {c: pool[(pool["journal"] == c[0]) & (pool["era"] == c[1])].index for c in cells}
    counts = np.array([len(cell_to_idx[c]) for c in cells], dtype=np.int64)

    if counts.sum() == 0:
        raise AssertionError("no rows in any configured (journal, era) cell")

    targets = _largest_remainder(counts, cfg.n_sample)
    if int(targets.sum()) != cfg.n_sample:
        # Should be unreachable given _largest_remainder's contract; assert
        # rather than silently continuing.
        raise AssertionError(
            f"allocator produced sum={int(targets.sum())} != n_sample={cfg.n_sample}"
        )

    sampled_frames: list[pd.DataFrame] = []
    exhausted: list[tuple[str, str]] = []
    for (journal, era), target in zip(cells, targets, strict=True):
        idx = cell_to_idx[(journal, era)]
        if len(idx) == 0:
            exhausted.append((journal, era))
            continue
        if target <= 0:
            continue
        rng = np.random.default_rng(_cell_seed(cfg.seed, journal, era))
        # Sample without replacement; target is already capped to pool size.
        chosen = rng.choice(idx.to_numpy(), size=int(target), replace=False)
        sampled_frames.append(pool.loc[chosen])

    if not sampled_frames:
        raise AssertionError("no cells produced any rows")

    out = pd.concat(sampled_frames, axis=0, ignore_index=True)
    out = out[["pmid", "journal", "year", "era", "topic_id", "title", "abstract"]]
    out = out.sort_values("pmid", kind="mergesort").reset_index(drop=True)

    # --- Post-sample assertions ---------------------------------------------
    if len(out) != cfg.n_sample:
        raise AssertionError(f"sample size {len(out)} != cfg.n_sample {cfg.n_sample}")

    expected_cells = len(cells) - len(exhausted)
    seen_cells = out[["journal", "era"]].drop_duplicates().shape[0]
    if seen_cells < expected_cells:
        raise AssertionError(
            f"sample covers {seen_cells} cells; expected {expected_cells} (exhausted={exhausted})"
        )

    n_topics = int(out["topic_id"].dropna().nunique())
    if n_topics < cfg.topic_coverage_min:
        raise AssertionError(
            f"topic coverage {n_topics} < topic_coverage_min {cfg.topic_coverage_min}"
        )

    return out
