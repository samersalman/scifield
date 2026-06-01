"""V1-S10 per-paper SEMANTIC novelty (embedding distance to prior literature).

Semantic novelty asks: *how far is this paper, in embedding space, from the
work that came before it in its own field?* We compute two complementary
scores per paper, both bounded in ``[0, 2]`` but in practice ``[0, ~1]`` for
SPECTER-style biomedical embeddings:

``sem_nov_mean``
    Distance from the focal vector to the **centroid** of all strictly-prior
    same-field vectors: ``1 - f . centroid_prior``. Because every vector is
    L2-normalised, ``mean_j cos(f, p_j) == f . (mean of prior vectors)``, so
    the mean cosine collapses to a single dot product against a running
    centroid — exact and O(N), no pairwise blowup.

``sem_nov_min``
    Distance to the **nearest** strictly-prior same-field neighbour:
    ``1 - max_j cos(f, prior_j)``. A paper that closely re-states a prior
    paper scores near 0; a paper unlike anything before it scores high.

Field partition
---------------
The "field" of a paper is its leaf ``topic_id`` from the BERTopic backbone.
Noise papers (``topic_id == noise_topic_id``, default ``-1``) have no leaf
field, so their field is the **whole corpus**: for ``sem_nov_min`` we query
the global FAISS index and post-filter to strictly-prior papers. (Their
``sem_nov_mean`` would require a corpus-wide running centroid; we provide it
too, computed over the noise partition as a degenerate "field" so the column
is never silently dropped — see :func:`_mean_novelty_by_field`, which treats
``-1`` as just another field id.)

Prior tie rule
--------------
A paper is a *prior* of the focal paper iff its publication ``year`` is
**strictly less** than the focal paper's year (``year < focal.year``). Papers
published in the SAME year are NOT priors of each other. This is deliberate:
within-year ordering is unobservable at year granularity, so we refuse to
assert that one same-year paper preceded another.

Earliest-in-field
------------------
A paper with no strictly-prior same-field paper has ``n_prior == 0`` and
``NaN`` for both novelty scores (you cannot measure distance to a literature
that does not yet exist). Downstream code should treat ``n_prior == 0`` as the
authoritative "undefined" flag rather than thresholding the NaNs.

Normalisation
-------------
:func:`scifield.thematic.dedup.load_deduped_embeddings` returns vectors that
are **not** guaranteed unit-norm. All cosine math here requires unit vectors,
so :func:`compute_semantic_novelty` L2-normalises every row up front; after
that, ``cos(a, b) == a . b``. The FAISS index was built over L2-normalised
vectors with an inner-product metric, so its returned scores are already
cosine similarities and need no re-normalisation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["compute_semantic_novelty"]


# --------------------------------------------------------------------------
# Pure numeric core (numpy in, numpy out — unit-testable without I/O)
# --------------------------------------------------------------------------


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalised copy of ``vectors`` (float32).

    Zero-norm rows are left as zeros (their cosine against anything is 0),
    which keeps the function total instead of producing NaNs.

    Parameters
    ----------
    vectors:
        Array of shape ``(n, d)``.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(n, d)`` with unit-norm rows (except
        all-zero rows, left unchanged).
    """
    v = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normed: np.ndarray = (v / norms).astype(np.float32, copy=False)
    return normed


def _mean_novelty_by_field(
    vectors: np.ndarray,
    topic_ids: np.ndarray,
    years: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact O(N) mean-novelty via running per-field prior centroids.

    For each field (unique ``topic_id``) we process papers in ascending year
    order, maintaining a running SUM and COUNT of vectors from *strictly
    earlier* years. The focal paper's mean cosine to its priors is then
    ``f . (running_sum / running_count)`` because the inputs are unit-norm.

    Same-year papers are flushed into the running totals only AFTER every
    paper of that year has been scored, enforcing the strict-prior tie rule.

    Parameters
    ----------
    vectors:
        L2-normalised array of shape ``(n, d)``.
    topic_ids:
        Field id per row, shape ``(n,)``.
    years:
        Publication year per row, shape ``(n,)``.

    Returns
    -------
    sem_nov_mean : numpy.ndarray
        Shape ``(n,)`` float64; ``NaN`` where ``n_prior == 0``.
    n_prior : numpy.ndarray
        Shape ``(n,)`` int64; count of strictly-prior same-field papers.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    topic_ids = np.asarray(topic_ids)
    years = np.asarray(years, dtype=np.int64)
    n, d = vectors.shape

    sem_nov_mean = np.full(n, np.nan, dtype=np.float64)
    n_prior = np.zeros(n, dtype=np.int64)

    for field in np.unique(topic_ids):
        idx = np.flatnonzero(topic_ids == field)
        # Stable sort on year so equal years stay grouped contiguously.
        order = idx[np.argsort(years[idx], kind="stable")]
        field_years = years[order]

        running_sum = np.zeros(d, dtype=np.float64)
        running_count = 0

        # Walk year-blocks: score everyone in the block against the centroid
        # of strictly-earlier years, THEN fold the whole block into totals.
        start = 0
        m = order.shape[0]
        while start < m:
            end = start
            while end < m and field_years[end] == field_years[start]:
                end += 1
            block = order[start:end]

            if running_count > 0:
                centroid = running_sum / running_count
                # f . centroid for every paper in the block.
                dots = vectors[block].astype(np.float64) @ centroid
                sem_nov_mean[block] = 1.0 - dots
                n_prior[block] = running_count

            running_sum += vectors[block].astype(np.float64).sum(axis=0)
            running_count += block.shape[0]
            start = end

    return sem_nov_mean, n_prior


def _min_novelty_within_field(
    vectors: np.ndarray,
    topic_ids: np.ndarray,
    years: np.ndarray,
    *,
    fields: np.ndarray | None = None,
) -> np.ndarray:
    """Brute-force min-novelty within each field (distance to nearest prior).

    For each focal paper, find the maximum cosine to any strictly-prior
    same-field paper and return ``1 - that max``. Used directly for leaf
    topics (avg ~600 papers/topic, so the per-field ``O(k^2 d)`` is cheap).
    Noise papers are handled by FAISS in :func:`compute_semantic_novelty`
    and should be excluded via ``fields`` here.

    Parameters
    ----------
    vectors:
        L2-normalised array of shape ``(n, d)``.
    topic_ids:
        Field id per row, shape ``(n,)``.
    years:
        Publication year per row, shape ``(n,)``.
    fields:
        Optional explicit list of field ids to process. Rows whose
        ``topic_id`` is not in ``fields`` keep ``NaN``. Defaults to every
        unique field.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)`` float64; ``NaN`` where the paper has no prior.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    topic_ids = np.asarray(topic_ids)
    years = np.asarray(years, dtype=np.int64)
    n = vectors.shape[0]

    sem_nov_min = np.full(n, np.nan, dtype=np.float64)
    field_iter = np.unique(topic_ids) if fields is None else np.asarray(fields)

    for field in field_iter:
        idx = np.flatnonzero(topic_ids == field)
        if idx.shape[0] < 2:
            continue
        fy = years[idx]
        fv = vectors[idx]
        # Full cosine matrix for this field (small k); mask non-priors.
        sims = fv @ fv.T  # (k, k)
        prior_mask = fy[None, :] < fy[:, None]  # [i, j] True iff j is prior of i
        sims_masked = np.where(prior_mask, sims, -np.inf)
        best = sims_masked.max(axis=1)  # (k,)
        has_prior = np.isfinite(best)
        sem_nov_min[idx[has_prior]] = 1.0 - best[has_prior].astype(np.float64)

    return sem_nov_min


def _min_novelty_noise_via_faiss(
    vectors: np.ndarray,
    years: np.ndarray,
    noise_rows: np.ndarray,
    *,
    index,
    index_years: np.ndarray,
    k: int,
) -> np.ndarray:
    """Min-novelty for corpus-wide (noise) papers via the global FAISS index.

    For each noise paper we ask FAISS for the top-``k`` nearest neighbours over
    the whole corpus (inner product == cosine, since the index is built on
    unit vectors), then post-filter the returned neighbours to those with
    ``year < focal.year`` and keep the nearest survivor. ``k`` must be large
    enough that at least one strictly-prior neighbour survives the filter for
    almost every paper; rows that exhaust their ``k`` candidates without a
    prior keep ``NaN`` (treated as no-prior).

    Parameters
    ----------
    vectors:
        L2-normalised array of shape ``(n, d)`` aligned to ``years``.
    years:
        Publication year per row, shape ``(n,)``.
    noise_rows:
        Row indices (into ``vectors``/``years``) of the noise papers.
    index:
        A FAISS index over the full corpus, with the SAME row order as
        ``index_years`` (i.e. the FAISS pmid map aligned to ``vectors``).
    index_years:
        Year per FAISS row position, shape ``(index.ntotal,)``.
    k:
        Neighbour fan-out per query.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)`` float64; defined only at ``noise_rows``, ``NaN``
        elsewhere and where no prior neighbour was found within ``k``.
    """
    n = vectors.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if noise_rows.shape[0] == 0:
        return out

    queries = np.ascontiguousarray(vectors[noise_rows], dtype=np.float32)
    # +1 because the focal paper itself is in the index and will be its own
    # top hit at cosine ~1; we discard self below via the strict-year filter
    # (self has year == focal.year, so it is never a valid prior anyway).
    sims, neigh = index.search(queries, k)

    focal_years = years[noise_rows]
    for q in range(noise_rows.shape[0]):
        cand = neigh[q]
        cand_sims = sims[q]
        valid = cand >= 0
        cand = cand[valid]
        cand_sims = cand_sims[valid]
        prior = index_years[cand] < focal_years[q]
        if not np.any(prior):
            continue
        best = float(cand_sims[prior].max())
        out[noise_rows[q]] = 1.0 - best
    return out


# --------------------------------------------------------------------------
# Public entry point (I/O + orchestration)
# --------------------------------------------------------------------------


def compute_semantic_novelty(
    *,
    embeddings_parquet: Path,
    topics_parquet: Path,
    duckdb_path: Path,
    faiss_index_path: Path,
    faiss_pmid_map_path: Path,
    noise_topic_id: int = -1,
    faiss_k: int = 256,
) -> pd.DataFrame:
    """Compute per-paper semantic novelty against prior same-field literature.

    Joins deduped embeddings, leaf topic assignments, and publication years,
    then computes :data:`sem_nov_mean` (distance to prior-field centroid) and
    :data:`sem_nov_min` (distance to nearest prior-field neighbour). Leaf
    topics use exact brute force; noise papers (``topic_id == noise_topic_id``)
    use the global FAISS index for the min score over the whole corpus.

    No network access. Reads ``papers_distinct`` read-only; if the view is
    missing it is created on a read-write connection (mirrors the topics CLI).

    Parameters
    ----------
    embeddings_parquet:
        V1-S05 ``embeddings.parquet`` (``pmid`` + ``embedding``). Vectors are
        L2-normalised internally.
    topics_parquet:
        ``topics.parquet`` with columns ``pmid, topic_id, is_noise``.
    duckdb_path:
        ``papers.duckdb`` providing ``papers_distinct`` (year per pmid).
    faiss_index_path:
        FAISS HNSW inner-product index over the corpus embeddings.
    faiss_pmid_map_path:
        Parquet mapping FAISS row positions to pmids.
    noise_topic_id:
        Topic id flagging corpus-wide (noise) papers. Default ``-1``.
    faiss_k:
        Neighbour fan-out for noise-paper min-novelty queries. Must be large
        enough that a strictly-prior neighbour survives the year filter for
        nearly every noise paper.

    Returns
    -------
    pandas.DataFrame
        Columns exactly ``pmid, topic_id, year, n_prior, sem_nov_mean,
        sem_nov_min``. Rows with ``n_prior == 0`` have ``NaN`` novelty scores.
    """
    from scifield.thematic.dedup import ensure_papers_distinct_view, load_deduped_embeddings
    from scifield.thematic.faiss_index import read_index, read_pmid_map

    # --- Load embeddings (one row per pmid) and normalise. ----------------
    emb_pmids, emb_raw = load_deduped_embeddings(Path(embeddings_parquet))
    vectors = _l2_normalize(emb_raw)

    # --- Topic assignments. -----------------------------------------------
    topics_df = pd.read_parquet(topics_parquet, columns=["pmid", "topic_id"])
    topics_df["pmid"] = topics_df["pmid"].astype(np.int64)
    topics_df["topic_id"] = topics_df["topic_id"].astype(np.int64)

    # --- Years from papers_distinct (read-only; create view if absent). ---
    years_df = _load_years(Path(duckdb_path), ensure_papers_distinct_view)

    # --- Assemble aligned frame keyed by embedding pmids. -----------------
    base = pd.DataFrame({"pmid": emb_pmids.astype(np.int64), "_row": np.arange(emb_pmids.shape[0])})
    merged = base.merge(topics_df, on="pmid", how="inner").merge(years_df, on="pmid", how="inner")
    merged = merged.sort_values("_row").reset_index(drop=True)

    rows = merged["_row"].to_numpy()
    vecs = vectors[rows]
    topic_ids = merged["topic_id"].to_numpy(dtype=np.int64)
    years = merged["year"].to_numpy(dtype=np.int64)

    # --- Mean novelty (all fields, incl. noise treated as one field). -----
    sem_nov_mean, n_prior = _mean_novelty_by_field(vecs, topic_ids, years)

    # --- Min novelty: leaf topics brute-force, noise via FAISS. -----------
    leaf_fields = np.array([f for f in np.unique(topic_ids) if f != noise_topic_id])
    sem_nov_min = _min_novelty_within_field(vecs, topic_ids, years, fields=leaf_fields)

    noise_rows = np.flatnonzero(topic_ids == noise_topic_id)
    if noise_rows.shape[0] > 0:
        index = read_index(Path(faiss_index_path))
        _, faiss_pmids = read_pmid_map(Path(faiss_pmid_map_path))
        index_years = _faiss_row_years(faiss_pmids, merged)
        noise_min = _min_novelty_noise_via_faiss(
            vecs,
            years,
            noise_rows,
            index=index,
            index_years=index_years,
            k=faiss_k,
        )
        sem_nov_min[noise_rows] = noise_min[noise_rows]

    # n_prior is the single source of truth for "undefined" — force both
    # scores to NaN wherever there is genuinely no prior.
    no_prior = n_prior == 0
    sem_nov_mean[no_prior] = np.nan
    sem_nov_min[no_prior] = np.nan

    return pd.DataFrame(
        {
            "pmid": merged["pmid"].to_numpy(dtype=np.int64),
            "topic_id": topic_ids,
            "year": years,
            "n_prior": n_prior,
            "sem_nov_mean": sem_nov_mean,
            "sem_nov_min": sem_nov_min,
        }
    )


def _load_years(duckdb_path: Path, ensure_view) -> pd.DataFrame:
    """Return a ``pmid, year`` frame from ``papers_distinct``.

    Tries a read-only connection first; if the view does not yet exist, opens
    a read-write connection to create it (mirrors the topics CLI command),
    then reads.
    """
    import duckdb

    query = "SELECT CAST(pmid AS BIGINT) AS pmid, year FROM papers_distinct"
    try:
        con = duckdb.connect(str(duckdb_path), read_only=True)
        try:
            df = con.execute(query).df()
        finally:
            con.close()
    except duckdb.CatalogException:
        con = duckdb.connect(str(duckdb_path), read_only=False)
        try:
            ensure_view(con)
            df = con.execute(query).df()
        finally:
            con.close()
    df["pmid"] = df["pmid"].astype(np.int64)
    df["year"] = df["year"].astype(np.int64)
    return df


def _faiss_row_years(faiss_pmids: np.ndarray, merged: pd.DataFrame) -> np.ndarray:
    """Map each FAISS row position to a publication year.

    FAISS rows whose pmid is absent from ``merged`` (e.g. dropped during the
    topic/year join) get a sentinel year of ``np.iinfo(np.int64).max`` so they
    can never qualify as a strict prior of any real paper.
    """
    merged_pmids = merged["pmid"].to_numpy(dtype=np.int64)
    merged_years = merged["year"].to_numpy(dtype=np.int64)
    lookup = dict(zip(merged_pmids, merged_years, strict=True))
    sentinel = np.iinfo(np.int64).max
    out = np.fromiter(
        (lookup.get(int(p), sentinel) for p in faiss_pmids),
        dtype=np.int64,
        count=faiss_pmids.shape[0],
    )
    return out
