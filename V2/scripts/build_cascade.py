"""Build the V2 topic-cascade tables (lag matrix, origins, diffusion curves).

Runnable I/O driver for :mod:`scifield.cartography.cascade`. The cascade module is
pure; this script owns all the I/O the module avoids:

1. Load the shipped flow tables ``V2/data/flow/flow_leaf.parquet`` (149 leaf
   topics × 10 journals) and ``V2/data/flow/flow_mid.parquet`` (96 mid topics).
   These already carry the canonical ``journal_slug`` (the jama_surg collapse) and
   per-(topic, journal, year) ``is_first_appearance`` — we consume ``journal_slug``
   from them and never re-derive it from the archetypes display name.
2. For each grain call the cascade functions:
   :func:`~scifield.cartography.cascade.lead_lag_matrix`,
   :func:`~scifield.cartography.cascade.origin_attribution`,
   :func:`~scifield.cartography.cascade.diffusion_curve`.
3. Write ``V2/data/cascade/{lag_matrix,origin_attribution,diffusion_curves}.parquet``
   — both grains stacked with a ``grain`` column ("leaf" / "mid") — each with a
   :func:`scifield.repro.record_run` provenance sidecar (config ``task="V2-S03"``).
4. Print row counts + a 5–10 topic spot-check (origin journal + the journals that
   followed) so the run is auditable.

Panel-conditional caveat (carry into every consumer): an "origin journal" is the
*first within our ten-journal panel*, never "first in the world" — the true origin
may be a journal never harvested (``V2/CONTEXT.md`` §0.2).

Usage
-----
``.venv/bin/python V2/scripts/build_cascade.py``            # both grains (default)
``.venv/bin/python V2/scripts/build_cascade.py --grain leaf``
``.venv/bin/python V2/scripts/build_cascade.py --spot-check 8``

$0 / read-only: reads only the V2 flow parquets + topic_hierarchy; no network, no
GPU, no DeepSeek.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _path_config import get_paths, resolve_data_version  # V2/scripts sibling helper

from scifield.cartography.cascade import diffusion_curve, lead_lag_matrix, origin_attribution
from scifield.cartography.flow import topic_key_for_grain
from scifield.repro import record_run

REPO_ROOT = Path(__file__).resolve().parents[2]

# Data-version-derived paths (default "v1" at import; re-resolved in `main()`).
_PATHS = get_paths(REPO_ROOT, resolve_data_version())
FLOW_DIR = _PATHS["FLOW_DIR"]
TOPIC_HIERARCHY = _PATHS["TOPIC_HIERARCHY"]
OUT_DIR = _PATHS["CASCADE_DIR"]


def _set_paths(version: str) -> None:
    """Rebind module-level path constants for a resolved data version (see main)."""
    global FLOW_DIR, TOPIC_HIERARCHY, OUT_DIR
    p = get_paths(REPO_ROOT, version)
    FLOW_DIR = p["FLOW_DIR"]
    TOPIC_HIERARCHY = p["TOPIC_HIERARCHY"]
    OUT_DIR = p["CASCADE_DIR"]


def load_flow(grain: str) -> pd.DataFrame:
    """Load the shipped flow table for a grain.

    Parameters
    ----------
    grain :
        ``"leaf"`` or ``"mid"``.

    Returns
    -------
    pandas.DataFrame
        The flow table from ``V2/data/flow/flow_{grain}.parquet``.

    Raises
    ------
    FileNotFoundError
        If the flow table has not been built (run ``build_flow.py`` first).
    """
    path = FLOW_DIR / f"flow_{grain}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"missing flow table {path} — run `.venv/bin/python V2/scripts/build_flow.py` first"
        )
    return pd.read_parquet(path)


def load_top_words() -> dict[int, str]:
    """Map leaf ``topic_id`` -> a short top-words label for the spot-check.

    Returns
    -------
    dict
        ``topic_id -> "word1, word2, ..."`` (first few BERTopic top words).
    """
    th = pd.read_parquet(TOPIC_HIERARCHY, columns=["topic_id", "top_words"])
    labels: dict[int, str] = {}
    for row in th.itertuples():
        words = list(row.top_words)[:5] if row.top_words is not None else []
        labels[int(row.topic_id)] = ", ".join(str(w) for w in words)
    return labels


def build_grain(grain: str) -> dict[str, pd.DataFrame]:
    """Compute the three cascade tables for one grain.

    Parameters
    ----------
    grain :
        ``"leaf"`` or ``"mid"``.

    Returns
    -------
    dict
        ``{"lag_matrix", "origin_attribution", "diffusion_curves"}`` -> frame,
        each with a leading ``grain`` column.
    """
    topic_key = topic_key_for_grain(grain)
    flow = load_flow(grain)

    lag = lead_lag_matrix(flow, topic_key=topic_key)
    origins = origin_attribution(flow, topic_key=topic_key)
    curves = diffusion_curve(flow, topic_key=topic_key)

    for frame in (lag, origins, curves):
        frame.insert(0, "grain", grain)
    return {
        "lag_matrix": lag,
        "origin_attribution": origins,
        "diffusion_curves": curves,
    }


def write_table(name: str, frames: list[pd.DataFrame], grains: list[str]) -> Path:
    """Concatenate per-grain frames, write parquet + a record_run sidecar.

    Parameters
    ----------
    name :
        Output stem (``lag_matrix`` / ``origin_attribution`` / ``diffusion_curves``).
    frames :
        Per-grain frames (already carrying a ``grain`` column).
    grains :
        The grains built, for the sidecar config.

    Returns
    -------
    pathlib.Path
        The written parquet path.
    """
    combined = pd.concat(frames, ignore_index=True)
    out_path = OUT_DIR / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)

    inputs = {f"flow_{g}": FLOW_DIR / f"flow_{g}.parquet" for g in grains}
    record_run(
        artifact_path=out_path,
        inputs=inputs,
        config={
            "task": "V2-S03",
            "table": name,
            "grains": grains,
            "tie_break": "alphabetically-first co-earliest journal_slug",
            "panel_conditional": (
                "origin = first within the 10-journal panel, not first in the world"
            ),
        },
    )
    return out_path


def spot_check(grain: str, origins: pd.DataFrame, flow: pd.DataFrame, n: int) -> None:
    """Print an auditable origin + followers spot-check for ``n`` leaf topics.

    For each sampled topic, print its top-words label, the origin journal +
    origin year, and the journals that followed (with their lag in years).

    Parameters
    ----------
    grain :
        The grain (only leaf carries top-words labels; mid is labelled by id).
    origins :
        The grain's origin-attribution frame (with the ``grain`` column).
    flow :
        The grain's flow table (to recover per-journal first years).
    n :
        Number of topics to display.
    """
    from scifield.cartography.cascade import _first_appearance_frame

    topic_key = topic_key_for_grain(grain)
    labels = load_top_words() if grain == "leaf" else {}
    first = _first_appearance_frame(flow, topic_key=topic_key)

    # Pick a spread: the n topics reaching the most journals (richest cascades).
    ranked = origins.sort_values(["n_journals", topic_key], ascending=[False, True])
    sample = ranked.head(n)

    print(f"\n=== SPOT-CHECK ({grain}): {n} richest-cascade topics ===")
    for row in sample.itertuples():
        tid = getattr(row, topic_key)
        label = labels.get(int(tid), f"<{topic_key}={tid}>")
        topic_first = first[first[topic_key] == tid].sort_values("first_year")
        followers = topic_first[topic_first["journal_slug"] != row.origin_journal_slug]
        follow_str = ", ".join(
            f"{r.journal_slug}(+{int(r.first_year) - int(row.origin_year)}y)"
            for r in followers.itertuples()
        )
        tie_note = "  [TIE]" if row.tie else ""
        print(
            f"  topic {tid} [{label}]{tie_note}\n"
            f"    ORIGIN: {row.origin_journal_slug} ({row.origin_year}) "
            f"-> reached {row.n_journals}/10 journals\n"
            f"    FOLLOWERS: {follow_str or '(none)'}"
        )


def main() -> None:
    """CLI entry point: build the cascade tables and print an audit."""
    parser = argparse.ArgumentParser(description="Build V2 topic-cascade tables.")
    parser.add_argument(
        "--grain",
        choices=["leaf", "mid", "both"],
        default="both",
        help="topic granularity to build (default: both)",
    )
    parser.add_argument(
        "--spot-check",
        type=int,
        default=8,
        help="number of topics to spot-check (default: 8)",
    )
    parser.add_argument(
        "--data-version",
        default=None,
        help="data version (default: $SCIFIELD_DATA_VERSION or 'v1'); "
        "v1 reads/writes V2/data (frozen); v2 reads data/v2 + writes V2/data_v2",
    )
    args = parser.parse_args()

    _set_paths(resolve_data_version(args.data_version))

    grains = ["leaf", "mid"] if args.grain == "both" else [args.grain]

    per_table: dict[str, list[pd.DataFrame]] = {
        "lag_matrix": [],
        "origin_attribution": [],
        "diffusion_curves": [],
    }
    flows: dict[str, pd.DataFrame] = {}
    origins_by_grain: dict[str, pd.DataFrame] = {}

    for grain in grains:
        print(f"\nBuilding cascade tables for grain={grain!r}...")
        flows[grain] = load_flow(grain)
        tables = build_grain(grain)
        origins_by_grain[grain] = tables["origin_attribution"]
        for name, frame in tables.items():
            per_table[name].append(frame)
        topic_key = topic_key_for_grain(grain)
        print(f"  lag_matrix:        {len(tables['lag_matrix']):>6,} ordered journal pairs")
        print(
            f"  origin_attribution:{len(tables['origin_attribution']):>6,} topics "
            f"(key={topic_key}, {int(tables['origin_attribution']['tie'].sum())} ties)"
        )
        n_fit = int(tables["diffusion_curves"]["curve_fitted"].sum())
        n_single = int((tables["diffusion_curves"]["n_journals"] == 1).sum())
        print(
            f"  diffusion_curves:  {len(tables['diffusion_curves']):>6,} topics "
            f"({n_fit} logistic-fitted, {n_single} single-journal/no-spread)"
        )

    written = []
    for name, frames in per_table.items():
        path = write_table(name, frames, grains)
        written.append(path)
        print(f"\nwrote {path}  ({len(pd.concat(frames)):,} rows across {len(grains)} grain(s))")

    # Spot-check on the leaf grain (top-words labelled) if built, else first grain.
    sc_grain = "leaf" if "leaf" in grains else grains[0]
    spot_check(sc_grain, origins_by_grain[sc_grain], flows[sc_grain], args.spot_check)

    print("\nDone. Outputs + sidecars:")
    for path in written:
        sidecar = Path(str(path) + ".run.json")
        print(f"  [{'ok' if path.exists() and sidecar.exists() else 'XX'}] {path.name}")


if __name__ == "__main__":
    main()
