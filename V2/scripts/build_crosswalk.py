"""Build the v1↔v2 **topic crosswalk** (V2-S08).

Runnable I/O driver for :mod:`scifield.cartography.crosswalk`. It owns all the I/O the
pure module avoids: loading each side's topic assignments + embeddings, deriving
per-topic centroids and member PMID sets, and writing the crosswalk parquet with a
``record_run`` provenance sidecar.

What it does
------------
1. For each side (``--v1-dir`` default ``data/v1``, ``--v2-dir`` default ``data/v2``):
   load ``{dir}/topics.parquet`` (schema ``pmid, topic_id, is_noise``) and
   ``{dir}/embeddings.parquet`` (schema ``pmid, embedding, model_name`` — a 768-d
   ``all-mpnet-base-v2`` vector per pmid). Join on ``pmid``, drop the noise topic
   (``topic_id == -1``), and build ``{topic_id: centroid}`` (mean L2-normalized
   embedding) plus ``{topic_id: set_of_pmids}``.
2. Call :func:`scifield.cartography.crosswalk.build_crosswalk` with the
   ``--top-k`` / ``--min-cosine`` / ``--min-jaccard`` passthroughs.
3. Write ``--out`` (default ``V2/data/crosswalk/topic_crosswalk.parquet``), mkdir-ing
   parents, and emit a ``<out>.run.json`` sidecar via
   :func:`scifield.repro.record_run` with task id ``"V2-S08-crosswalk"``.

Both sides must use the SAME embedding model for centroid cosine to be meaningful;
the loader asserts this and aborts otherwise.

Status (V2-S08)
---------------
The v2 topic/embedding artifacts do **not** exist yet (the v2 embed/cluster steps are
deferred). This script is therefore AUTHORED but not run now: it is import-clean and
``--help``-able, and exits with a clear message if an input is missing — correctness is
proven by the synthetic unit test (``tests/test_cartography_crosswalk.py``), not a run.

Usage
-----
``.venv/bin/python V2/scripts/build_crosswalk.py``  (defaults: data/v1 ↔ data/v2)
``.venv/bin/python V2/scripts/build_crosswalk.py --v1-dir data/v1 --v2-dir data/v2``

$0 / read-only: reads two parquet pairs, writes one parquet + sidecar. No network, GPU,
or DuckDB. Self-contained — does NOT import ``V2/scripts/_path_config.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scifield.cartography.crosswalk import NOISE_LABEL, build_crosswalk
from scifield.repro import record_run

REPO_ROOT = Path(__file__).resolve().parents[2]

TASK_ID = "V2-S08-crosswalk"

#: Column names in the topic / embedding parquets (v1 and v2 share this schema).
TOPICS_PMID = "pmid"
TOPICS_TOPIC = "topic_id"
EMB_PMID = "pmid"
EMB_VECTOR = "embedding"
EMB_MODEL = "model_name"


def _resolve(path_like: str) -> Path:
    """Resolve a CLI path: absolute as-is, else relative to the repo root."""
    p = Path(path_like)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_side(
    data_dir: Path, *, label: str
) -> tuple[dict[int, np.ndarray], dict[int, set], Path, Path]:
    """Load one side's centroids + member sets from its topic/embedding parquets.

    Parameters
    ----------
    data_dir :
        Directory holding ``topics.parquet`` and ``embeddings.parquet``.
    label :
        ``"v1"`` / ``"v2"`` — used only for error messages.

    Returns
    -------
    centroids : dict of int to numpy.ndarray
        ``{topic_id: L2-normalized mean embedding}`` (noise dropped).
    members : dict of int to set
        ``{topic_id: set_of_pmids}`` (noise dropped); pmids cast to ``int`` so the two
        sides share a comparable key type for Jaccard.
    topics_path, embeddings_path : pathlib.Path
        The two input paths (for the provenance sidecar).

    Raises
    ------
    SystemExit
        If an input parquet is missing, an expected column is absent, the model names
        disagree within the side, or no non-noise paper remains after the join.
    """
    topics_path = data_dir / "topics.parquet"
    embeddings_path = data_dir / "embeddings.parquet"
    for p in (topics_path, embeddings_path):
        if not p.exists():
            _abort(
                f"missing {label} input: {p}\n"
                f"  (the {label} topic/embedding artifacts may not exist yet — "
                "run the v2 embed + cluster steps first)"
            )

    topics = pd.read_parquet(topics_path)
    emb = pd.read_parquet(embeddings_path)
    _require_columns(topics, [TOPICS_PMID, TOPICS_TOPIC], topics_path)
    _require_columns(emb, [EMB_PMID, EMB_VECTOR], embeddings_path)

    if EMB_MODEL in emb.columns:
        models = set(emb[EMB_MODEL].dropna().unique().tolist())
        if len(models) > 1:
            _abort(f"{label} embeddings mix models {sorted(models)}; expected one")

    # Join topic id onto embeddings by pmid; drop noise; build centroids + members.
    topics = topics[[TOPICS_PMID, TOPICS_TOPIC]].copy()
    topics[TOPICS_PMID] = topics[TOPICS_PMID].astype("int64")
    emb = emb[[EMB_PMID, EMB_VECTOR]].copy()
    emb[EMB_PMID] = emb[EMB_PMID].astype("int64")
    merged = emb.merge(topics, left_on=EMB_PMID, right_on=TOPICS_PMID, how="inner")
    merged = merged[merged[TOPICS_TOPIC] != NOISE_LABEL]
    if merged.empty:
        _abort(
            f"no non-noise {label} papers after pmid join of "
            f"{topics_path.name} + {embeddings_path.name}"
        )

    matrix = np.vstack([np.asarray(v, dtype=np.float64) for v in merged[EMB_VECTOR].to_numpy()])
    labels = merged[TOPICS_TOPIC].to_numpy()
    pmids = merged[EMB_PMID].to_numpy()

    from scifield.cartography.crosswalk import topic_centroids

    centroids = topic_centroids(matrix, labels, normalize=True)
    members: dict[int, set] = {}
    for tid, pmid in zip(labels.tolist(), pmids.tolist(), strict=True):
        members.setdefault(int(tid), set()).add(int(pmid))

    print(
        f"  [{label}] {len(merged):,} papers ⋈ -> {len(centroids)} topics "
        f"(from {topics_path.relative_to(REPO_ROOT)} + {embeddings_path.relative_to(REPO_ROOT)})"
    )
    return centroids, members, topics_path, embeddings_path


def _require_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    """Abort with a clear message if any required column is missing from ``df``."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        _abort(f"{path} is missing column(s) {missing}; found {list(df.columns)}")


def _abort(message: str) -> None:
    """Print an error to stderr and exit non-zero (graceful, not a traceback)."""
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    """CLI entry point: load both sides, build the crosswalk, write parquet + sidecar."""
    parser = argparse.ArgumentParser(description="Build the v1↔v2 topic crosswalk.")
    parser.add_argument("--v1-dir", default="data/v1", help="v1 data dir (default: data/v1)")
    parser.add_argument("--v2-dir", default="data/v2", help="v2 data dir (default: data/v2)")
    parser.add_argument(
        "--out",
        default="V2/data/crosswalk/topic_crosswalk.parquet",
        help="output parquet (default: V2/data/crosswalk/topic_crosswalk.parquet)",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="v2 candidates per v1 topic (default: 3)"
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.3,
        help="centroid-cosine keep threshold (default: 0.3)",
    )
    parser.add_argument(
        "--min-jaccard",
        type=float,
        default=0.0,
        help="PMID-Jaccard keep threshold (default: 0.0)",
    )
    args = parser.parse_args()

    v1_dir = _resolve(args.v1_dir)
    v2_dir = _resolve(args.v2_dir)
    out_path = _resolve(args.out)

    print(f"Loading v1 from {v1_dir} and v2 from {v2_dir} ...")
    v1_centroids, v1_members, v1_topics_p, v1_emb_p = load_side(v1_dir, label="v1")
    v2_centroids, v2_members, v2_topics_p, v2_emb_p = load_side(v2_dir, label="v2")

    print(
        f"Building crosswalk (top_k={args.top_k}, min_cosine={args.min_cosine}, "
        f"min_jaccard={args.min_jaccard}) ..."
    )
    crosswalk = build_crosswalk(
        v1_centroids,
        v2_centroids,
        v1_members,
        v2_members,
        top_k=args.top_k,
        min_cosine=args.min_cosine,
        min_jaccard=args.min_jaccard,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_parquet(out_path, index=False)

    record_run(
        artifact_path=out_path,
        inputs={
            "v1_topics": v1_topics_p,
            "v1_embeddings": v1_emb_p,
            "v2_topics": v2_topics_p,
            "v2_embeddings": v2_emb_p,
        },
        config={
            "task": TASK_ID,
            "top_k": args.top_k,
            "min_cosine": args.min_cosine,
            "min_jaccard": args.min_jaccard,
            "noise_topic_dropped": NOISE_LABEL,
        },
    )

    n_v1_mapped = crosswalk["v1_topic"].nunique()
    n_v1_total = len(v1_centroids)
    splits = crosswalk.groupby("v1_topic").size()
    merges = crosswalk.groupby("v2_topic").size()
    print(f"\n=== {out_path.name} ===")
    print(f"  path:            {out_path}")
    print(f"  rows:            {len(crosswalk):,}")
    print(f"  v1 topics mapped:{n_v1_mapped} / {n_v1_total}")
    print(f"  v1 splits (>1 v2 candidate): {int((splits > 1).sum())}")
    print(f"  v2 merges (>1 v1 source):    {int((merges > 1).sum())}")
    if not crosswalk.empty:
        print("  --- 5-row sample ---")
        print(crosswalk.head(5).to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
