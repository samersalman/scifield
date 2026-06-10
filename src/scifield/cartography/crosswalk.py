"""V2 cartography — the v1↔v2 **topic crosswalk** (pure, I/O-free).

Why this module exists
----------------------
V1's findings (F1 level-cascade, F2 sector novelty, the 2×2 role archetypes) are all
indexed by v1 *leaf topic id*. When the corpus is re-clustered at v2 scale, BERTopic
runs HDBSCAN afresh: topic ids do **not** carry over, and a v1 topic may **split**
into several v2 topics, **merge** with neighbours, or dissolve into noise. So a v1
finding stated as "topic 42 behaves like X" is uninterpretable against v2 unless we
know which v2 topic(s) *are* old-topic-42. This module builds that bridge: a tidy
``v1_topic → {v2_topic: overlap}`` map. We expect — and the design embraces —
splits/merges, **not** a clean permutation.

Two independent overlap signals
-------------------------------
Both v1 and v2 embed with the *same* model (``all-mpnet-base-v2``, 768-d), so the two
embedding spaces are directly comparable and we can measure topic overlap two ways:

1. **Centroid cosine** — cosine similarity between the (L2-normalized) mean embedding
   of a v1 topic and that of a v2 topic. A *semantic* signal: it sees "these two
   clusters sit in the same neighbourhood" even when their member papers barely
   overlap (e.g. v2 added new papers to the same theme).
2. **PMID Jaccard** — ``|A ∩ B| / |A ∪ B|`` over the topics' member PMID sets. A
   *membership* signal: it sees "literally the same papers", robust to any embedding
   drift, but blind to v2-only growth.

The two disagree in informative ways (high cosine + low Jaccard ⇒ a theme that grew;
high Jaccard + low cosine would flag an embedding problem), so we **report both** and
let the consumer adjudicate, rather than collapsing them into one opaque score.

Combination / ranking rule (documented contract)
-------------------------------------------------
For each v1 topic we emit its top-``k`` candidate v2 topics, ranked by **cosine**
similarity (the dense, always-defined signal — Jaccard is 0 for the many
non-overlapping pairs and would not order them). For every emitted pair we *also*
report its Jaccard so a split/merge is visible in the numbers. A candidate pair is
**kept** if it clears **either** threshold — ``cosine ≥ min_cosine`` **or**
(``jaccard > 0`` **and** ``jaccard ≥ min_jaccard``) — so a pair that shares papers but
sits slightly off-centroid (or vice-versa) is not silently dropped. The ``jaccard > 0``
guard on the Jaccard arm matters: without it, the default ``min_jaccard = 0`` would
match *every* pair (Jaccard is non-negative) and the cosine floor could never filter
anything; with it, ``min_jaccard = 0`` means "keep any pair that genuinely shares
papers", while non-overlapping pairs are judged on cosine alone. Among kept pairs the
top-``k`` by cosine
survive; ties in cosine break by descending Jaccard, then by ascending ``v2_topic`` so
the output is fully deterministic. A v1 topic with no kept candidate contributes no
rows (it has no v2 home — itself a finding worth surfacing downstream).

Splits and merges read straight off the long table:

* **split** — one ``v1_topic`` has several rows (several v2 topics with comparable
  cosine/Jaccard).
* **merge** — one ``v2_topic`` appears under several ``v1_topic`` values.

Purity
------
Every function takes in-memory structures (numpy arrays, dict, DataFrame) and returns
a new value; no parquet, DuckDB, or filesystem access. Reading the topic/embedding
parquets and writing the crosswalk live in ``V2/scripts/build_crosswalk.py``.

Conventions: ``from __future__ import annotations``; numpy-style docstrings; numpy +
pandas only; ruff/black line-length 100. The noise topic (label ``-1``) is excluded
from every centroid and member set. Real values are COMPUTED, never hardcoded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Topic label used by HDBSCAN/BERTopic for unclustered ("noise") papers. Excluded
#: from every centroid and member set on both sides.
NOISE_LABEL = -1

#: Column order of the tidy crosswalk frame returned by :func:`build_crosswalk`.
CROSSWALK_COLUMNS = [
    "v1_topic",
    "v2_topic",
    "cosine",
    "jaccard",
    "n_v1",
    "n_v2",
    "n_overlap",
    "rank",
]


def topic_centroids(
    embeddings: np.ndarray,
    topic_labels: np.ndarray,
    *,
    normalize: bool = True,
) -> dict[int, np.ndarray]:
    """Mean embedding per topic id, skipping the noise label.

    Parameters
    ----------
    embeddings :
        ``(n_papers, dim)`` float array, one row per paper.
    topic_labels :
        ``(n_papers,)`` integer topic ids aligned row-for-row with ``embeddings``.
        Rows whose label is :data:`NOISE_LABEL` (``-1``) are ignored.
    normalize :
        If ``True`` (default), L2-normalize each centroid so a later dot product is a
        cosine similarity. A zero-vector centroid (degenerate) is left as zeros.

    Returns
    -------
    dict of int to numpy.ndarray
        ``{topic_id: centroid}`` with one ``(dim,)`` float64 vector per non-noise
        topic, deterministically ordered by ascending ``topic_id``.

    Raises
    ------
    ValueError
        If ``embeddings`` is not 2-D or the label length does not match the row count.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(topic_labels)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2-D; got shape {emb.shape}")
    if labels.shape[0] != emb.shape[0]:
        raise ValueError(
            f"row/label mismatch: {emb.shape[0]} embeddings vs {labels.shape[0]} labels"
        )

    centroids: dict[int, np.ndarray] = {}
    for tid in sorted({int(t) for t in labels.tolist()}):
        if tid == NOISE_LABEL:
            continue
        mean_vec = emb[labels == tid].mean(axis=0)
        if normalize:
            norm = float(np.linalg.norm(mean_vec))
            if norm > 0.0:
                mean_vec = mean_vec / norm
        centroids[tid] = mean_vec
    return centroids


def _stack_centroids(
    centroids: dict[int, np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    """Stack a centroid dict into a ``(n_topics, dim)`` matrix + its id ordering.

    Ids are sorted ascending so the row/column ordering is deterministic. An empty
    dict yields a ``(0, 0)`` array and an empty id list.
    """
    ids = sorted(centroids)
    if not ids:
        return np.zeros((0, 0), dtype=np.float64), []
    matrix = np.vstack([np.asarray(centroids[i], dtype=np.float64) for i in ids])
    return matrix, ids


def centroid_cosine_matrix(
    v1_centroids: dict[int, np.ndarray],
    v2_centroids: dict[int, np.ndarray],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Cosine-similarity matrix between v1 and v2 topic centroids.

    Parameters
    ----------
    v1_centroids, v2_centroids :
        ``{topic_id: centroid}`` maps (e.g. from :func:`topic_centroids`). Centroids
        need not be pre-normalized — this function normalizes internally so the result
        is a true cosine regardless of how the centroids were built.

    Returns
    -------
    matrix : numpy.ndarray
        ``(n_v1, n_v2)`` float64 cosine similarities; ``matrix[i, j]`` is the cosine
        between ``v1_ids[i]`` and ``v2_ids[j]``. A degenerate (zero-norm) centroid
        yields a row/column of zeros.
    v1_ids : list of int
        Row ordering (ascending v1 topic id).
    v2_ids : list of int
        Column ordering (ascending v2 topic id).
    """
    v1_mat, v1_ids = _stack_centroids(v1_centroids)
    v2_mat, v2_ids = _stack_centroids(v2_centroids)
    if not v1_ids or not v2_ids:
        return np.zeros((len(v1_ids), len(v2_ids)), dtype=np.float64), v1_ids, v2_ids

    v1_unit = _l2_normalize_rows(v1_mat)
    v2_unit = _l2_normalize_rows(v2_mat)
    matrix = v1_unit @ v2_unit.T
    # Floating-point can nudge a self-identical cosine to 1+eps; clip to [-1, 1].
    return np.clip(matrix, -1.0, 1.0), v1_ids, v2_ids


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` with each row scaled to unit L2 norm (zeros left as zeros)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    return matrix / safe


def pmid_jaccard(
    v1_members: dict[int, set],
    v2_members: dict[int, set],
) -> dict[tuple[int, int], float]:
    """Jaccard overlap for every v1/v2 topic pair that shares ≥1 PMID.

    Only pairs with a non-empty intersection are returned (the vast majority of pairs
    share nothing, so materializing them would be wasteful and uninformative). The
    inverted-index walk over the smaller-cardinality side keeps this near-linear in the
    number of shared memberships rather than quadratic in the number of topics.

    Parameters
    ----------
    v1_members, v2_members :
        ``{topic_id: set_of_pmids}`` for each side. PMIDs may be any hashable type as
        long as the two sides use the *same* type (the build script casts both to a
        common dtype before calling).

    Returns
    -------
    dict of (int, int) to float
        ``{(v1_topic, v2_topic): jaccard}`` for overlapping pairs only;
        ``jaccard = |A ∩ B| / |A ∪ B|`` in ``(0, 1]``.
    """
    # Invert v2 once: pmid -> list of v2 topics containing it.
    pmid_to_v2: dict[object, list[int]] = {}
    for v2_tid, members in v2_members.items():
        for pmid in members:
            pmid_to_v2.setdefault(pmid, []).append(v2_tid)

    # Accumulate intersection sizes per (v1, v2) pair by walking v1 members.
    inter: dict[tuple[int, int], int] = {}
    for v1_tid, members in v1_members.items():
        for pmid in members:
            for v2_tid in pmid_to_v2.get(pmid, ()):  # empty tuple if no overlap
                key = (v1_tid, v2_tid)
                inter[key] = inter.get(key, 0) + 1

    sizes_v1 = {tid: len(m) for tid, m in v1_members.items()}
    sizes_v2 = {tid: len(m) for tid, m in v2_members.items()}
    out: dict[tuple[int, int], float] = {}
    for (v1_tid, v2_tid), n_int in inter.items():
        union = sizes_v1[v1_tid] + sizes_v2[v2_tid] - n_int
        out[(v1_tid, v2_tid)] = (n_int / union) if union > 0 else 0.0
    return out


def build_crosswalk(
    v1_centroids: dict[int, np.ndarray],
    v2_centroids: dict[int, np.ndarray],
    v1_members: dict[int, set],
    v2_members: dict[int, set],
    *,
    top_k: int = 3,
    min_cosine: float = 0.3,
    min_jaccard: float = 0.0,
) -> pd.DataFrame:
    """Build the tidy long v1↔v2 topic crosswalk.

    For each v1 topic, find its candidate v2 topics, keep those clearing **either**
    threshold (``cosine ≥ min_cosine`` OR ``jaccard ≥ min_jaccard``), and emit the
    top-``k`` by cosine. See the module docstring for the full combination/ranking
    contract. The result captures splits (one v1 topic → many rows) and merges (one v2
    topic under many v1 topics) directly.

    Parameters
    ----------
    v1_centroids, v2_centroids :
        ``{topic_id: centroid}`` maps; passed to :func:`centroid_cosine_matrix`.
    v1_members, v2_members :
        ``{topic_id: set_of_pmids}`` maps; ``n_v1`` / ``n_v2`` are their sizes and the
        intersection drives ``jaccard`` / ``n_overlap``.
    top_k :
        Max v2 candidates emitted per v1 topic (default 3).
    min_cosine :
        Cosine threshold for the "kept" disjunction (default 0.3).
    min_jaccard :
        Jaccard threshold for the "kept" disjunction (default 0.0). The Jaccard arm
        requires ``jaccard > 0`` *and* ``jaccard >= min_jaccard``, so the default keeps
        any pair that genuinely shares papers while leaving non-overlapping pairs to be
        judged on cosine alone (see the module docstring for why the ``> 0`` guard is
        needed).

    Returns
    -------
    pandas.DataFrame
        Columns :data:`CROSSWALK_COLUMNS` —
        ``[v1_topic, v2_topic, cosine, jaccard, n_v1, n_v2, n_overlap, rank]``. One row
        per kept (v1, v2) candidate; ``rank`` is 1-based per v1 topic (best cosine =
        rank 1). Sorted by ``(v1_topic, rank)``. Always non-noise; an empty input
        yields the empty frame with this exact schema and dtypes.
    """
    cos_matrix, v1_ids, v2_ids = centroid_cosine_matrix(v1_centroids, v2_centroids)
    jaccard = pmid_jaccard(v1_members, v2_members)
    v2_pos = {tid: j for j, tid in enumerate(v2_ids)}
    sizes_v1 = {tid: len(m) for tid, m in v1_members.items()}
    sizes_v2 = {tid: len(m) for tid, m in v2_members.items()}

    rows: list[dict[str, object]] = []
    for i, v1_tid in enumerate(v1_ids):
        # Candidate v2 topics for this v1 topic: any v2 topic that has a centroid
        # (a cosine column) OR shares ≥1 PMID. Union both so a pair that overlaps in
        # membership but lacks a v2 centroid (degenerate) is still considered.
        cos_row = cos_matrix[i] if cos_matrix.size else np.zeros(0)
        candidate_v2 = set(v2_ids)
        candidate_v2.update(v2_tid for (a, v2_tid) in jaccard if a == v1_tid)

        scored: list[tuple[float, float, int]] = []  # (cosine, jaccard, v2_topic)
        for v2_tid in candidate_v2:
            cos = float(cos_row[v2_pos[v2_tid]]) if v2_tid in v2_pos else 0.0
            jac = float(jaccard.get((v1_tid, v2_tid), 0.0))
            # Either-threshold keep. The Jaccard arm requires *actual* overlap
            # (jac > 0): otherwise `min_jaccard=0.0` would match every pair (Jaccard is
            # non-negative) and the cosine floor could never filter anything. So a pair
            # survives iff it is semantically close (cosine) OR genuinely shares papers
            # at the requested level (jaccard > 0 and >= min_jaccard).
            keep_cos = cos >= min_cosine
            keep_jac = jac > 0.0 and jac >= min_jaccard
            if keep_cos or keep_jac:
                scored.append((cos, jac, v2_tid))

        # Rank: cosine desc, then jaccard desc, then v2_topic asc (full determinism).
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))

        for rank, (cos, jac, v2_tid) in enumerate(scored[:top_k], start=1):
            n_v1 = sizes_v1.get(v1_tid, 0)
            n_v2 = sizes_v2.get(v2_tid, 0)
            # n_overlap is exact from the membership sets (more reliable than inverting
            # jaccard, which would round); compute it directly when sets are present.
            n_overlap = (
                len(v1_members[v1_tid] & v2_members[v2_tid])
                if v1_tid in v1_members and v2_tid in v2_members
                else 0
            )
            rows.append(
                {
                    "v1_topic": int(v1_tid),
                    "v2_topic": int(v2_tid),
                    "cosine": cos,
                    "jaccard": jac,
                    "n_v1": int(n_v1),
                    "n_v2": int(n_v2),
                    "n_overlap": int(n_overlap),
                    "rank": int(rank),
                }
            )

    if not rows:
        return _empty_crosswalk()

    out = pd.DataFrame(rows, columns=CROSSWALK_COLUMNS)
    out = out.sort_values(["v1_topic", "rank"]).reset_index(drop=True)
    return _coerce_dtypes(out)


def _empty_crosswalk() -> pd.DataFrame:
    """Return a well-formed empty crosswalk with the canonical columns and dtypes."""
    return _coerce_dtypes(pd.DataFrame({c: [] for c in CROSSWALK_COLUMNS}))


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Pin integer/float dtypes so empty and non-empty frames share one schema."""
    int_cols = ["v1_topic", "v2_topic", "n_v1", "n_v2", "n_overlap", "rank"]
    float_cols = ["cosine", "jaccard"]
    for col in int_cols:
        df[col] = df[col].astype("int64")
    for col in float_cols:
        df[col] = df[col].astype("float64")
    return df[CROSSWALK_COLUMNS]
