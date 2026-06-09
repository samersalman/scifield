"""Run the V2-S04 cascade validation (Methods A–D) and write the verdict tables.

Runnable I/O driver for :mod:`scifield.cartography.cascade_validation` (which is
pure). This script owns the I/O the module avoids:

1. Load the shipped flow tables ``V2/data/flow/flow_leaf.parquet`` (149 leaf
   topics × 10 journals) and ``V2/data/flow/flow_mid.parquet`` (96 mid topics).
   They already carry the canonical ``journal_slug`` (the jama_surg collapse) and a
   ``year`` column, so the cascade-validation statistics can be recomputed on any
   year/topic subset.
2. Run :func:`~scifield.cartography.cascade_validation.run_all_validations` at BOTH
   grains, for BOTH the all-topics and resolvable subsets (the 1995-censoring
   sensitivity), with the LOCKED constants (``n_perm=1000``, ``seed=20260609``).
3. Flatten the nested verdict to three tidy tables and write them with
   :func:`scifield.repro.record_run` sidecars (config ``task="V2-S04"``):
   ``V2/data/cascade/null_results.parquet`` (Method B, one row per
   grain × subset × stat × null), ``jackknife_results.parquet`` (Method C, one row
   per grain × subset), ``granularity_results.parquet`` (Method D + Method A
   supporting, one row per metric).
4. Print the verdict table and the elapsed time.

The make-or-break, gate-critical cell is **S1 under Null-1 at the leaf grain on
both the all-topics and resolvable subsets** — that always runs at the full
``n_perm=1000``. The full all-cells run is ~80 s on the current corpus, so no
secondary capping is needed.

$0 / read-only: reads only the V2 flow parquets; no network, no GPU, no DeepSeek.

Usage
-----
``.venv/bin/python V2/scripts/run_cascade_validation.py``
``.venv/bin/python V2/scripts/run_cascade_validation.py --n-perm 1000``
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from scifield.cartography import cascade_validation as cv
from scifield.cartography.flow import topic_key_for_grain
from scifield.repro import record_run

REPO_ROOT = Path(__file__).resolve().parents[2]
FLOW_DIR = REPO_ROOT / "V2/data/flow"
OUT_DIR = REPO_ROOT / "V2/data/cascade"


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


def flatten_null_results(method_b: dict) -> pd.DataFrame:
    """Flatten the Method-B nested verdict to one row per cell + a verdict row.

    Parameters
    ----------
    method_b :
        ``run_all_validations(...)["method_b"]``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[grain, subset, stat, null, observed, null_mean, null_std, z,
        p_value, percentile, n_perm, n_topics, cell_verdict, method_b_verdict]``,
        one row per (grain, subset, stat, null).
    """
    rows: list[dict[str, object]] = []
    for grain, subsets in method_b.items():
        for subset, block in subsets.items():
            for (stat, null), cell in block["cells"].items():
                rows.append(
                    {
                        "grain": grain,
                        "subset": subset,
                        "stat": stat,
                        "null": null,
                        "observed": cell["observed"],
                        "null_mean": cell["null_mean"],
                        "null_std": cell["null_std"],
                        "z": cell["z"],
                        "p_value": cell["p_value"],
                        "percentile": cell["percentile"],
                        "n_perm": cell["n_perm"],
                        "n_topics": cell["n_topics"],
                        "cell_verdict": cell["verdict"],
                        "method_b_verdict": block["verdict"],
                    }
                )
    return pd.DataFrame(rows)


def flatten_jackknife_results(method_c: dict) -> pd.DataFrame:
    """Flatten the Method-C jackknife verdict to one row per grain × subset.

    Parameters
    ----------
    method_c :
        ``run_all_validations(...)["method_c"]``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[grain, subset, attribution_flip_rate, forced_reattribution_count,
        n_eligible_cells, n_genuine_flips, n_topics_flipped, n_topics, verdict]``.
    """
    rows: list[dict[str, object]] = []
    for grain, subsets in method_c.items():
        for subset, summary in subsets.items():
            rows.append({"grain": grain, "subset": subset, **summary})
    return pd.DataFrame(rows)


def flatten_granularity_results(method_d: dict, method_a: dict) -> pd.DataFrame:
    """Flatten Method D (granularity) + Method A (held-out, supporting) to a table.

    Parameters
    ----------
    method_d :
        ``run_all_validations(...)["method_d"]``.
    method_a :
        ``run_all_validations(...)["method_a"]`` (keyed by grain).

    Returns
    -------
    pandas.DataFrame
        One row for the granularity (leaf-vs-mid) comparison and one per grain for
        the held-out comparison, columns ``[method, grain, spearman_rho, bar,
        n_compared, n_excluded, supporting_only, verdict]``.
    """
    rows: list[dict[str, object]] = [
        {
            "method": "D_granularity",
            "grain": "leaf_vs_mid",
            "spearman_rho": method_d["spearman_rho"],
            "bar": method_d["rho_bar"],
            "n_compared": method_d["n_journals"],
            "n_excluded": 0,
            "supporting_only": False,
            "verdict": method_d["verdict"],
        }
    ]
    for grain, a in method_a.items():
        rows.append(
            {
                "method": "A_heldout",
                "grain": grain,
                "spearman_rho": a["spearman_rho"],
                "bar": 0.5,
                "n_compared": a["n_journals_paired"],
                "n_excluded": a["n_topics_excluded"],
                "supporting_only": a["supporting_only"],
                "verdict": a["verdict"],
            }
        )
    return pd.DataFrame(rows)


def write_table(name: str, frame: pd.DataFrame, config: dict) -> Path:
    """Write a verdict table parquet + a record_run sidecar.

    Parameters
    ----------
    name :
        Output stem (``null_results`` / ``jackknife_results`` /
        ``granularity_results``).
    frame :
        The tidy verdict frame.
    config :
        Extra config recorded in the sidecar (merged with the common keys).

    Returns
    -------
    pathlib.Path
        The written parquet path.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.parquet"
    frame.to_parquet(out_path, index=False)
    record_run(
        artifact_path=out_path,
        inputs={
            "flow_leaf": FLOW_DIR / "flow_leaf.parquet",
            "flow_mid": FLOW_DIR / "flow_mid.parquet",
        },
        config={
            "task": "V2-S04",
            "table": name,
            "n_perm": cv.N_PERM,
            "seed": cv.SEED,
            "null_1": "journal-label shuffle within topic (primary)",
            "null_2": "first-appearance-year shuffle within topic (secondary)",
            "stat_s1": "seeding-score spread (std across 10 journals, primary)",
            "stat_s2": "net directed lead-lag asymmetry (secondary)",
            "censoring": (
                "all-topics vs resolvable subset (>=2 distinct first-years) reported; "
                "1995 origin = present at panel start, not born 1995"
            ),
            **config,
        },
    )
    return out_path


def print_verdict_table(out: dict) -> None:
    """Print the four-method verdict table to stdout for the audit log."""
    print("\n" + "=" * 78)
    print("V2-S04 CASCADE VALIDATION — VERDICT TABLE (locked constants)")
    print("=" * 78)

    print("\nMethod B — null / permutation (S1 primary; Null-1 primary; gate-critical)")
    print(
        f"  {'grain':<5} {'subset':<11} {'stat':<3} {'null':<13} "
        f"{'obs':>7} {'z':>6} {'p':>7} {'pct':>6}  verdict"
    )
    for grain, subsets in out["method_b"].items():
        for subset, block in subsets.items():
            for (stat, null), c in block["cells"].items():
                print(
                    f"  {grain:<5} {subset:<11} {stat:<3} {null:<13} "
                    f"{c['observed']:>7.4f} {c['z']:>+6.2f} {c['p_value']:>7.4f} "
                    f"{c['percentile']:>6.3f}  {c['verdict']}"
                )
            print(f"    -> Method-B [{grain}/{subset}] = {block['verdict']}")

    print("\nMethod C — drop-one-journal jackknife of ORIGIN attributions")
    print(f"  {'grain':<5} {'subset':<11} {'flip_rate':>9} {'forced':>7} {'flips':>6}  verdict")
    for grain, subsets in out["method_c"].items():
        for subset, s in subsets.items():
            print(
                f"  {grain:<5} {subset:<11} {s['attribution_flip_rate']:>9.4f} "
                f"{s['forced_reattribution_count']:>7} {s['n_genuine_flips']:>6}  {s['verdict']}"
            )

    d = out["method_d"]
    print("\nMethod D — topic-granularity sensitivity (leaf vs mid)")
    print(
        f"  seeding-rank Spearman rho = {d['spearman_rho']:.4f} "
        f"(bar {d['rho_bar']})  {d['verdict']}"
    )

    print("\nMethod A — held-out years (1995-2018 vs 2019-2025) [SUPPORTING ONLY]")
    for grain, a in out["method_a"].items():
        print(
            f"  {grain:<5} rho = {a['spearman_rho']:.4f}  "
            f"(excluded {a['n_topics_excluded']} post-2018 topics)  {a['verdict']}"
        )

    o = out["overall"]
    print("\n" + "-" * 78)
    print(f"OVERALL CASCADE VERDICT: {o['cascade_verdict']}")
    print(f"  rationale: {o['rationale']}")
    print(f"  primary S1/Null-1 (grain, subset): {o['primary_s1_null1_verdicts']}")
    print(f"  Method C all PASS: {o['method_c_pass']};  Method D: {o['method_d_verdict']}")
    print("-" * 78)


def main() -> None:
    """CLI entry point: run validations at both grains/subsets and write tables."""
    parser = argparse.ArgumentParser(description="Run V2-S04 cascade validation.")
    parser.add_argument(
        "--n-perm",
        type=int,
        default=cv.N_PERM,
        help=f"permutation count (default LOCKED {cv.N_PERM})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cv.SEED,
        help=f"RNG seed (default LOCKED {cv.SEED})",
    )
    parser.add_argument(
        "--secondary-n-perm",
        type=int,
        default=None,
        help="optional smaller perm count for secondary Method-B cells (default: same as --n-perm)",
    )
    args = parser.parse_args()

    if args.n_perm != cv.N_PERM or args.seed != cv.SEED:
        print(
            f"!! WARNING: running with non-locked constants (n_perm={args.n_perm}, "
            f"seed={args.seed}); the verdict tables will NOT match the protocol lock "
            f"(n_perm={cv.N_PERM}, seed={cv.SEED})."
        )

    print("Loading flow tables (leaf + mid)...")
    flow_leaf = load_flow("leaf")
    flow_mid = load_flow("mid")
    leaf_key = topic_key_for_grain("leaf")
    mid_key = topic_key_for_grain("mid")
    print(
        f"  leaf: {flow_leaf[leaf_key].nunique()} topics, {len(flow_leaf):,} cells; "
        f"mid: {flow_mid[mid_key].nunique()} topics, {len(flow_mid):,} cells"
    )

    t0 = time.time()
    out = cv.run_all_validations(
        flow_leaf,
        flow_mid,
        leaf_key=leaf_key,
        mid_key=mid_key,
        n_perm=args.n_perm,
        seed=args.seed,
        secondary_n_perm=args.secondary_n_perm,
    )
    elapsed = time.time() - t0

    null_tbl = flatten_null_results(out["method_b"])
    jk_tbl = flatten_jackknife_results(out["method_c"])
    gran_tbl = flatten_granularity_results(out["method_d"], out["method_a"])

    # JSON-safe overall verdict for the sidecar (tuple keys -> "grain/subset").
    overall = out["overall"]
    overall_safe = {
        "cascade_verdict": overall["cascade_verdict"],
        "rationale": overall["rationale"],
        "primary_s1_null1_verdicts": {
            f"{g}/{s}": v for (g, s), v in overall["primary_s1_null1_verdicts"].items()
        },
        "method_c_pass": overall["method_c_pass"],
        "method_d_verdict": overall["method_d_verdict"],
    }

    written = [
        write_table("null_results", null_tbl, {"overall_verdict": overall_safe}),
        write_table("jackknife_results", jk_tbl, {"overall_verdict": overall_safe}),
        write_table("granularity_results", gran_tbl, {"overall_verdict": overall_safe}),
    ]

    print_verdict_table(out)

    print(f"\nElapsed: {elapsed:.1f} s")
    print("\nOutputs + sidecars:")
    for path in written:
        sidecar = Path(str(path) + ".run.json")
        ok = path.exists() and sidecar.exists()
        print(f"  [{'ok' if ok else 'XX'}] {path}")


if __name__ == "__main__":
    main()
