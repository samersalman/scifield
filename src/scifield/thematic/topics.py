"""BERTopic pipeline for V1-S06 — precomputed embeddings → UMAP → HDBSCAN.

Public API
----------
* :class:`TopicConfig` — frozen hyperparameter dataclass.
* :func:`make_bertopic_model` — build an unfit :class:`bertopic.BERTopic`
  with ``embedding_model=None`` so caller-supplied embeddings are used.
* :func:`fit_topics` — run the full pipeline on precomputed vectors.
* :func:`build_hierarchy` — derive per-leaf-topic mid/top level IDs from
  the BERTopic hierarchical_topics linkage via union-find, *without*
  mutating the underlying model (so we can write both levels at once).
* :class:`SweepRow` / :func:`sweep` — grid harness; per-config exceptions
  are captured into the row rather than bubbling, so a single failing
  config doesn't sink the whole sweep.

Heavy deps (bertopic, umap-learn, hdbscan, scikit-learn) are imported
lazily inside the functions so this module can be imported in
environments that haven't yet ``uv sync``'d those packages — fast unit
tests can ``pytest.importorskip`` on demand.

Determinism caveat: even with ``UMAP(random_state=...)`` and
``HDBSCAN(core_dist_n_jobs=1)``, UMAP can produce slightly different
projections under different BLAS thread counts. Topic *contents* stay
stable, but topic IDs and exact NPMI scores can shift by ~1e-3. We
record the thread count + library versions in the run sidecar so a
reviewer can match the environment.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - import-only branch for type checkers
    from bertopic import BERTopic

__all__ = [
    "TopicConfig",
    "SweepRow",
    "make_bertopic_model",
    "fit_topics",
    "build_hierarchy",
    "sweep",
]


@dataclass(frozen=True)
class TopicConfig:
    """Hyperparameters for a single BERTopic configuration.

    Frozen so a config is hashable and safe to drop into a sidecar JSON
    after the sweep picks a winner.
    """

    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    hdbscan_min_cluster_size: int = 50
    hdbscan_min_samples: int | None = None
    hdbscan_cluster_selection_method: str = "eom"
    nr_topics: str | int = "auto"
    random_state: int = 42
    vectorizer_min_df: int = 10
    vectorizer_ngram_max: int = 2


@dataclass
class SweepRow:
    """One row of the parameter-sweep output table."""

    config: dict[str, Any]
    n_leaf_topics: int
    noise_fraction: float
    npmi_top10: float
    cv_top10: float
    wall_seconds: float
    error: str | None = None


def make_bertopic_model(cfg: TopicConfig) -> BERTopic:
    """Construct an unfit :class:`bertopic.BERTopic` per ``cfg``.

    ``embedding_model=None`` is the contract that says "I will supply
    pre-computed embeddings at fit time". The vectorizer is built from
    ``cfg.vectorizer_*`` so c-TF-IDF runs on the same biomedical text the
    topics were learned on, without re-encoding.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=cfg.vectorizer_min_df,
        ngram_range=(1, cfg.vectorizer_ngram_max),
    )
    umap_model = UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        n_components=cfg.umap_n_components,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=cfg.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        cluster_selection_method=cfg.hdbscan_cluster_selection_method,
        core_dist_n_jobs=1,
        prediction_data=True,
    )
    return BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=cfg.nr_topics,
        calculate_probabilities=False,
    )


def fit_topics(
    embeddings: np.ndarray,
    documents: list[str],
    cfg: TopicConfig,
) -> BERTopic:
    """Fit a BERTopic model on ``documents`` with precomputed ``embeddings``.

    Embeddings are cast to ``float32`` because UMAP rejects fp16 inputs;
    V1-S05 stores them in fp16 on disk for size, and the dedup helper
    already up-casts on load, so this is belt-and-braces.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D; got shape {embeddings.shape}")
    if embeddings.shape[0] != len(documents):
        raise ValueError(
            f"row/document mismatch: {embeddings.shape[0]} embeddings vs {len(documents)} docs"
        )
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    model = make_bertopic_model(cfg)
    model.fit(documents, embeddings=emb)
    return model


# --- Hierarchy via union-find on hierarchical_topics merges -----------------


class _UnionFind:
    """Minimal union-find used to roll up leaf topics into mid/top groups.

    BERTopic's :meth:`bertopic.BERTopic.hierarchical_topics` returns a
    merge DataFrame; rather than calling :meth:`reduce_topics` (which
    mutates the model and would force two separate fits to get both
    mid- and top-level labels), we replay the merges greedily up to the
    target cluster count, using union-find to track membership.
    """

    def __init__(self, items: list[int]) -> None:
        self.parent: dict[int, int] = {x: x for x in items}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True

    def n_clusters(self) -> int:
        return len({self.find(x) for x in self.parent})


def _collect_leaf_topic_ids(merge_row, leaf_ids: set[int]) -> tuple[list[int], list[int]]:
    """Return ``(left_leaves, right_leaves)`` for a BERTopic merge row.

    BERTopic encodes hierarchical merges with ``Topics`` columns that
    list the leaf topic IDs contained in each child sub-tree. We use
    those directly when available; otherwise we fall back to the
    ``Child_Left_ID`` / ``Child_Right_ID`` (which are leaf IDs for the
    first merge layer).
    """

    def _resolve(side: str) -> list[int]:
        topics_col = f"{side}_Children"
        # Different BERTopic versions name the leaf-id list column
        # variously; try the common ones, then fall back to the single ID.
        for name in (topics_col, f"{side}_Topic", f"{side}_ID"):
            if name in merge_row.index:
                val = merge_row[name]
                if isinstance(val, list | tuple | np.ndarray):
                    return [int(x) for x in val if int(x) in leaf_ids]
                if val is not None and not isinstance(val, float):
                    iv = int(val)
                    if iv in leaf_ids:
                        return [iv]
        return []

    return _resolve("Child_Left"), _resolve("Child_Right")


def _cut_to_target(
    leaf_ids: list[int],
    merges: pd.DataFrame,
    target: int,
) -> dict[int, int]:
    """Apply hierarchical merges greedily until cluster count <= target.

    Returns a mapping ``leaf_topic_id -> group_root_id``. ``group_root_id``
    is just the union-find root, used as a stable group label.

    If the merge frame doesn't carry enough information to reach the
    target (e.g. small synthetic corpora with few topics), we stop at
    whatever cluster count we land on — caller writes the actual count
    into the sidecar.
    """
    uf = _UnionFind(leaf_ids)
    leaf_set = set(leaf_ids)
    target = max(1, int(target))

    # BERTopic returns merges with a "Distance" column; ascending = merge
    # closest sub-trees first, which is what we want for a clean cut.
    if "Distance" in merges.columns:
        ordered = merges.sort_values("Distance", ascending=True)
    else:
        ordered = merges

    for _, row in ordered.iterrows():
        if uf.n_clusters() <= target:
            break
        left, right = _collect_leaf_topic_ids(row, leaf_set)
        if not left or not right:
            continue
        # Merge all leaves on each side into a single root, then unite roots.
        a0 = left[0]
        for x in left[1:]:
            uf.union(a0, x)
        b0 = right[0]
        for x in right[1:]:
            uf.union(b0, x)
        uf.union(a0, b0)

    return {lid: uf.find(lid) for lid in leaf_ids}


def _topic_words(model: BERTopic, topic_id: int, top_n: int = 10) -> list[str]:
    """Return up to ``top_n`` words for ``topic_id``; empty list on miss."""
    words = model.get_topic(topic_id)
    if not words:
        return []
    return [str(w) for w, _ in words[:top_n]]


def _representative_docs(
    model: BERTopic,
    topic_id: int,
    documents: list[str],
    max_n: int = 3,
) -> list[str]:
    """Return up to ``max_n`` representative documents for ``topic_id``.

    Uses BERTopic's API when available; otherwise samples a few documents
    whose assigned topic equals ``topic_id``. Falls back gracefully so
    the synthetic test corpus (which BERTopic may not provide reps for
    on very small inputs) still produces some text.
    """
    try:
        reps = model.get_representative_docs(topic_id)
    except Exception:
        reps = None
    if reps:
        return [str(d) for d in reps[:max_n]]

    topics_arr = np.asarray(model.topics_)
    idxs = np.where(topics_arr == topic_id)[0]
    return [documents[i] for i in idxs[:max_n].tolist()]


def build_hierarchy(
    model: BERTopic,
    documents: list[str],
    *,
    target_mid_levels: int = 20,
    target_top_levels: int = 6,
) -> pd.DataFrame:
    """Roll up leaf topics into mid + top groups; return per-leaf metadata.

    Returned DataFrame columns:
        ``topic_id`` (int), ``top_words`` (list[str], up to 10),
        ``size`` (int — assignment count), ``mid_level_id`` (int),
        ``top_level_id`` (int), ``representative_docs`` (list[str], up to 3).

    The mapping is derived without mutating ``model`` — we replay the
    :meth:`hierarchical_topics` merge graph via union-find. This lets the
    caller write both mid- and top-level columns from a single fit
    instead of calling :meth:`reduce_topics` twice with destructive side
    effects on the model state.
    """
    info = model.get_topic_info()
    # "Topic" is the BERTopic-canonical column for leaf IDs; -1 is the
    # noise topic, which we strip before hierarchy rollup.
    leaf_ids: list[int] = [int(t) for t in info["Topic"].tolist() if int(t) != -1]
    if not leaf_ids:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "top_words",
                "size",
                "mid_level_id",
                "top_level_id",
                "representative_docs",
            ]
        )

    merges = model.hierarchical_topics(documents)

    mid_map = _cut_to_target(leaf_ids, merges, target_mid_levels)
    top_map = _cut_to_target(leaf_ids, merges, target_top_levels)

    mid_root_to_id = {r: i for i, r in enumerate(sorted(set(mid_map.values())))}
    top_root_to_id = {r: i for i, r in enumerate(sorted(set(top_map.values())))}

    topics_arr = np.asarray(model.topics_)
    sizes = {int(t): int((topics_arr == t).sum()) for t in leaf_ids}

    rows = []
    for tid in leaf_ids:
        rows.append(
            {
                "topic_id": tid,
                "top_words": _topic_words(model, tid, top_n=10),
                "size": sizes[tid],
                "mid_level_id": mid_root_to_id[mid_map[tid]],
                "top_level_id": top_root_to_id[top_map[tid]],
                "representative_docs": _representative_docs(model, tid, documents, max_n=3),
            }
        )
    return pd.DataFrame(rows)


# --- Sweep harness ----------------------------------------------------------


def _evaluate_config(
    embeddings: np.ndarray,
    documents: list[str],
    cfg: TopicConfig,
    coherence_texts: list[list[str]],
) -> SweepRow:
    """Run one config end-to-end and return a populated :class:`SweepRow`."""
    from scifield.thematic.coherence import compute_coherence

    cfg_dict = asdict(cfg)
    start = time.perf_counter()
    try:
        model = fit_topics(embeddings, documents, cfg)
        topics_arr = np.asarray(model.topics_)
        n_total = int(len(topics_arr))
        noise_count = int((topics_arr == -1).sum())
        unique_non_noise = sorted({int(t) for t in topics_arr.tolist() if int(t) != -1})
        n_leaf = len(unique_non_noise)
        word_lists = [_topic_words(model, t, top_n=10) for t in unique_non_noise]
        coh = compute_coherence(word_lists, coherence_texts, top_n=10)
        elapsed = time.perf_counter() - start
        return SweepRow(
            config=cfg_dict,
            n_leaf_topics=n_leaf,
            noise_fraction=(noise_count / n_total) if n_total else float("nan"),
            npmi_top10=float(coh.get("c_npmi", float("nan"))),
            cv_top10=float(coh.get("c_v", float("nan"))),
            wall_seconds=elapsed,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 — sweep must continue past per-config failures
        elapsed = time.perf_counter() - start
        return SweepRow(
            config=cfg_dict,
            n_leaf_topics=0,
            noise_fraction=float("nan"),
            npmi_top10=float("nan"),
            cv_top10=float("nan"),
            wall_seconds=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )


def sweep(
    embeddings: np.ndarray,
    documents: list[str],
    grid: list[TopicConfig],
    coherence_texts: list[list[str]],
    *,
    per_config_timeout_s: float = 900.0,  # noqa: ARG001 - reserved; see docstring
) -> pd.DataFrame:
    """Run the parameter sweep; one DataFrame row per config.

    A per-config exception is captured in the ``error`` column rather
    than re-raised — the whole point of a sweep is to enumerate winners
    and losers in one shot. We deliberately do NOT enforce
    ``per_config_timeout_s`` via signals: SIGALRM is POSIX-only and
    interferes with libraries that catch signals internally. The
    argument is kept in the signature for forward compatibility with a
    future subprocess-based runner; for now, configs run to natural
    completion and the caller is expected to size the grid accordingly.
    """
    rows: list[SweepRow] = []
    for cfg in grid:
        rows.append(_evaluate_config(embeddings, documents, cfg, coherence_texts))
    return pd.DataFrame([asdict(r) for r in rows])
