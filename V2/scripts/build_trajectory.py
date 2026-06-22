"""Build the V2 per-topic trajectory / forecast tables (V2-S10).

Runnable I/O driver for :mod:`scifield.cartography.trajectory`. The trajectory module
is pure (in-memory frames in, frames out); this script owns all the I/O the module
avoids:

1. Load the per-paper ``[topic_id, year]`` slice from ``archetypes.parquet`` (the only
   two columns the time series needs — a minimal read of the 802k-row master).
2. Build the long per-(topic, year) count + corpus-share series via
   :func:`~scifield.cartography.trajectory.build_topic_year_series` (BERTopic noise
   ``-1`` dropped from numerator AND denominator; the partial trailing year dropped).
3. Fit + project each leaf topic's **share** (headline) and **volume** (context)
   ``horizon`` years ahead via
   :func:`~scifield.cartography.trajectory.fit_trajectories` — a local-linear-trend
   state-space model with a log-linear OLS fallback, each back-transformed so the bands
   stay in range (share ∈ ``[0, 1]``, volume ``>= 0``).
4. Join the real ``label`` (comma-joined ``top_words``) + ``size`` from the topic
   hierarchy onto the per-topic ``summary`` (the pure layer returns those two columns as
   placeholders), then write ``{trajectory_series, trajectory_summary}.parquet`` — each
   with a :func:`scifield.repro.record_run` sidecar (config ``task="V2-S10"``) — and
   print the fit-ok / model / direction audit plus the top-10 rising and falling topics.

This layer is **leaf-only**: the ``grain`` column is carried as the constant ``"leaf"``
to match the roles/cascade table convention.

Usage
-----
``.venv/bin/python V2/scripts/build_trajectory.py``                  # v1 frozen
``KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python V2/scripts/build_trajectory.py --data-version v2``

$0 / read-only: reads ``archetypes.parquet`` + ``topic_hierarchy.parquet`` only; no
network, no GPU, no DeepSeek.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _path_config import get_paths, resolve_data_version  # V2/scripts sibling helper

from scifield.cartography.mapio import topic_labels
from scifield.cartography.trajectory import build_topic_year_series, fit_trajectories
from scifield.repro import record_run

REPO_ROOT = Path(__file__).resolve().parents[2]

# Data-version-derived paths (default "v1" at import; re-resolved in `main()`).
# get_paths has no trajectory key, so the output dir is derived from OUT_ROOT here.
_PATHS = get_paths(REPO_ROOT, resolve_data_version())
ARCHETYPES = _PATHS["ARCHETYPES"]
TOPIC_HIERARCHY = _PATHS["TOPIC_HIERARCHY"]
OUT_DIR = _PATHS["OUT_ROOT"] / "trajectory"

MAX_COMPLETE_YEAR = 2025  # last fully-observed year (the corpus's 2026 is partial)


def _set_paths(version: str) -> None:
    """Rebind module-level path constants for a resolved data version (see main)."""
    global ARCHETYPES, TOPIC_HIERARCHY, OUT_DIR
    p = get_paths(REPO_ROOT, version)
    ARCHETYPES = p["ARCHETYPES"]
    TOPIC_HIERARCHY = p["TOPIC_HIERARCHY"]
    OUT_DIR = p["OUT_ROOT"] / "trajectory"


def write_table(name: str, frame: pd.DataFrame, *, config: dict) -> Path:
    """Write a parquet + a record_run sidecar.

    Parameters
    ----------
    name :
        Output stem (``trajectory_series`` / ``trajectory_summary``).
    frame :
        The frame to write.
    config :
        Sidecar config keys (the ``"table"`` key is added here).

    Returns
    -------
    pathlib.Path
        The written parquet path.
    """
    out_path = OUT_DIR / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)

    record_run(
        artifact_path=out_path,
        inputs={"archetypes": ARCHETYPES, "topic_hierarchy": TOPIC_HIERARCHY},
        config={"table": name, **config},
    )
    return out_path


def print_trajectory_audit(summary: pd.DataFrame) -> None:
    """Print the fit-ok / model-path / direction counts for the per-topic summary."""
    print("\n=== TRAJECTORY FIT AUDIT ===")
    print(f"  {int(summary['fit_ok'].sum())}/{len(summary)} topics fit_ok")
    print("  model (share path):")
    for model, n in summary["model"].value_counts().items():
        print(f"    {model}: {n}")
    print("  direction (sign of observed share slope):")
    for direction, n in summary["direction"].value_counts().items():
        print(f"    {direction}: {n}")


def print_top_movers(summary: pd.DataFrame, *, k: int = 10) -> None:
    """Print the top-``k`` rising and top-``k`` falling topics by share slope."""
    cols = [
        "topic_id",
        "label",
        "slope_share_per_yr",
        "last_obs_share",
        "proj_share",
        "proj_share_lo",
        "proj_share_hi",
        "direction",
        "model",
    ]

    def _print(sub: pd.DataFrame) -> None:
        s = sub[cols].copy()
        s["slope_share_per_yr"] = s["slope_share_per_yr"].astype("float64").round(6)
        for c in ("last_obs_share", "proj_share", "proj_share_lo", "proj_share_hi"):
            s[c] = s[c].astype("float64").round(5)
        print(s.to_string(index=False))

    rising = summary.sort_values("slope_share_per_yr", ascending=False).head(k)
    falling = summary.sort_values("slope_share_per_yr", ascending=True).head(k)
    print(f"\n=== TOP {k} RISING (by slope_share_per_yr, descending) ===")
    _print(rising)
    print(f"\n=== TOP {k} FALLING (by slope_share_per_yr, ascending) ===")
    _print(falling)


def main() -> None:
    """CLI entry point: build the trajectory tables and print the audit."""
    parser = argparse.ArgumentParser(description="Build V2 per-topic trajectory / forecast tables.")
    parser.add_argument(
        "--grain",
        choices=["leaf"],
        default="leaf",
        help="topic granularity (leaf-only layer; the grain column is set to 'leaf')",
    )
    parser.add_argument(
        "--data-version",
        default=None,
        help="data version (default: $SCIFIELD_DATA_VERSION or 'v1'); "
        "v1 reads/writes V2/data (frozen); v2 reads data/v2 + writes V2/data_v2",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="number of future years to project (default: 5 -> 2026..2030)",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.80,
        help="central probability of the projection band (default: 0.80)",
    )
    args = parser.parse_args()
    version = resolve_data_version(args.data_version)
    _set_paths(version)
    if args.grain != "leaf":  # pragma: no cover - argparse already constrains the choice
        raise ValueError(f"trajectory layer is leaf-only; got grain={args.grain!r}")

    print(f"Building trajectories (data-version={version!r}, grain='leaf')...")
    print(f"  reading {ARCHETYPES.relative_to(REPO_ROOT)} [topic_id, year]")
    papers = pd.read_parquet(ARCHETYPES, columns=["topic_id", "year"])
    print(f"  {len(papers)} papers loaded")

    series = build_topic_year_series(papers, max_complete_year=MAX_COMPLETE_YEAR)
    series_out, summary = fit_trajectories(series, horizon=args.horizon, ci_level=args.ci_level)
    print(
        f"  {summary['topic_id'].nunique()} topics fit | "
        f"{len(series_out)} series rows (observed + projected)"
    )

    # Join the REAL label + size from the topic hierarchy, overwriting the pure layer's
    # placeholders ("" / <NA>). Touch no other summary column or the column order.
    labels = topic_labels(repo_root=REPO_ROOT, data_version=version)
    label_map = labels.set_index("topic_id")
    summary["label"] = summary["topic_id"].map(label_map["label"]).fillna("").astype("object")
    summary["size"] = summary["topic_id"].map(label_map["size"]).astype("Int64")

    config = {
        "task": "V2-S10",
        "grain": "leaf",
        "horizon": args.horizon,
        "ci_level": args.ci_level,
        "metric": "share+volume",
        "model": "state_space_llt+loglinear_fallback",
        "max_complete_year": MAX_COMPLETE_YEAR,
    }
    series_path = write_table("trajectory_series", series_out, config=config)
    summary_path = write_table("trajectory_summary", summary, config=config)

    print_trajectory_audit(summary)
    print_top_movers(summary)

    print("\nDone. Outputs + sidecars:")
    for path in (series_path, summary_path):
        sidecar = Path(str(path) + ".run.json")
        ok = path.exists() and sidecar.exists()
        print(f"  [{'ok' if ok else 'XX'}] {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
