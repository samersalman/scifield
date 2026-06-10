"""Build the V2-S06b robustness tables for the per-institution novelty origins.

The S06 novelty-origin findings are keyed on author-institution ``country_code``
(geography) and ``type`` (sector). The protocol's Method-D (leaf-vs-mid topic
granularity) is **N/A by construction** for these: a country / sector key does not
change when the topic grain changes. This script supplies the grain-independent
robustness number instead:

* **Geography** → split-half stability (:func:`scifield.cartography.origins.
  split_half_stability`): the per-country mean-novelty ranking is computed in two
  independent halves of the papers and correlated (Spearman ρ), in both ``random``
  (sampling robustness) and ``temporal`` (early-vs-late, out-of-period) modes. A
  finding that reproduces in both halves is robust.
* **Sector** → bootstrap delta CI (:func:`scifield.cartography.origins.
  bootstrap_delta_ci`): the company-minus-non-company mean-novelty delta is
  bootstrapped (1000 reps, 95% percentile CI). A CI that spans zero confirms the
  near-null; one that excludes zero would mean a real sector effect.

Both are run for ``sem_nov_mean`` (semantic novelty) and ``cd5`` (disruption).

Join recipe (reused from ``build_origins.py``, the S06 builder)
---------------------------------------------------------------
``archetypes.pmid`` (BIGINT, noise topic dropped, carries novelty + ``year``) →
``paper_institutions.pmid`` (VARCHAR, cast) → ``institutions.country_code`` /
``type``. Any-author tagging: a paper contributes one observation to every distinct
country / sector among its authors; blank ``country_code`` / ``type`` are dropped,
never imputed. This produces the same *exploded long frame* whose per-key means feed
the S06 headline tables, so the robustness numbers describe the headline finding.

Outputs (under ``V2/data/origins/``, each with a ``record_run`` sidecar, task
``V2-S06b``):

* ``geo_robustness.parquet`` — one row per (novelty_col × mode): ``spearman_rho``,
  ``n_keys``, ``mode``, ``novelty_col``, ``n_min``.
* ``sector_robustness.parquet`` — one row per novelty_col: ``delta``, ``ci_low``,
  ``ci_high``, ``spans_zero``, ``n_boot``, ``novelty_col``.

Usage
-----
``.venv/bin/python V2/scripts/build_origins_robustness.py``

$0 / read-only: reads ``data/v1/`` parquet only; no network, no GPU, no DeepSeek.
Locked session seed ``20260609`` is used for both the random split and the bootstrap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _path_config import get_paths, resolve_data_version  # V2/scripts sibling helper

from scifield.cartography.origins import (
    _attach_paper_attribute,
    bootstrap_delta_ci,
    paper_sector_tags,
    split_half_stability,
)
from scifield.repro import record_run

REPO_ROOT = Path(__file__).resolve().parents[2]

# Data-version-derived paths (default "v1" at import; re-resolved in `main()`).
_PATHS = get_paths(REPO_ROOT, resolve_data_version())
ARCHETYPES = _PATHS["ARCHETYPES"]
PAPER_INST = _PATHS["PAPER_INST"]
INSTITUTIONS = _PATHS["INSTITUTIONS"]
OUT_DIR = _PATHS["ORIGINS_DIR"]

NOISE_TOPIC_ID = -1
NOVELTY_COLS = ["sem_nov_mean", "cd5"]
SEED = 20260609
N_MIN = 30
N_BOOT = 1000
CI = 0.95


def _set_paths(version: str) -> None:
    """Rebind module-level path constants for a resolved data version (see main)."""
    global ARCHETYPES, PAPER_INST, INSTITUTIONS, OUT_DIR
    p = get_paths(REPO_ROOT, version)
    ARCHETYPES = p["ARCHETYPES"]
    PAPER_INST = p["PAPER_INST"]
    INSTITUTIONS = p["INSTITUTIONS"]
    OUT_DIR = p["ORIGINS_DIR"]


def load_paper_novelty() -> pd.DataFrame:
    """Per-paper novelty + year frame (noise topic dropped).

    Returns
    -------
    pandas.DataFrame
        ``["pmid"(str), "year"(int)] + NOVELTY_COLS``. ``pmid`` is a string to join
        ``paper_institutions``; the noise topic is excluded.
    """
    cols = ["pmid", "year", "topic_id", *NOVELTY_COLS]
    df = pd.read_parquet(ARCHETYPES, columns=cols)
    df = df.loc[df["topic_id"] != NOISE_TOPIC_ID].copy()
    df["pmid"] = df["pmid"].astype("string")
    return df.loc[:, ["pmid", "year", *NOVELTY_COLS]].reset_index(drop=True)


def build_geo_long(
    novelty: pd.DataFrame, paper_inst: pd.DataFrame, institutions: pd.DataFrame
) -> pd.DataFrame:
    """Exploded per-(paper, country) long frame: ``[country_code, year, *novelty]``.

    Uses the S06 any-author rule (:func:`_attach_paper_attribute`): blank country
    codes dropped, a multi-country paper appears once per country.
    """
    tags = _attach_paper_attribute(
        paper_inst, institutions, attr_col="country_code", inst_key="institution_canonical_id"
    )
    merged = tags.merge(novelty, on="pmid", how="inner")
    return merged.loc[:, ["country_code", "year", *NOVELTY_COLS]]


def build_sector_flag(
    novelty: pd.DataFrame, paper_inst: pd.DataFrame, institutions: pd.DataFrame
) -> pd.DataFrame:
    """Per-paper ``is_company`` flag + novelty frame for the bootstrap.

    One row per distinct paper (NOT exploded): the bootstrap contrasts company vs
    non-company *papers*, so each paper appears once with its paper-level
    ``is_company`` flag (any author at a ``type=='company'`` institution).
    """
    tags = paper_sector_tags(paper_inst, institutions)
    flag = (
        tags.loc[:, ["pmid", "is_company"]].drop_duplicates(subset=["pmid"]).reset_index(drop=True)
    )
    merged = flag.merge(novelty, on="pmid", how="inner")
    return merged.loc[:, ["pmid", "is_company", *NOVELTY_COLS]]


def _write(df: pd.DataFrame, name: str, inputs: dict[str, Path], config: dict) -> Path:
    """Write one robustness table + a record_run sidecar; return its path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    record_run(artifact_path=out_path, inputs=inputs, config=config)
    return out_path


def main() -> None:
    """Compute + write the geography and sector robustness tables; print them."""
    parser = argparse.ArgumentParser(description="Build V2-S06b novelty-origin robustness tables.")
    parser.add_argument(
        "--data-version",
        default=None,
        help="data version (default: $SCIFIELD_DATA_VERSION or 'v1'); "
        "v1 reads data/v1 + writes V2/data (frozen); v2 reads data/v2 + writes V2/data_v2",
    )
    args = parser.parse_args()
    _set_paths(resolve_data_version(args.data_version))

    print("Loading per-paper novelty + year (archetypes, noise topic dropped)...")
    novelty = load_paper_novelty()
    print(f"  {len(novelty):,} papers; novelty cols = {NOVELTY_COLS}")

    print("Loading paper_institutions + institutions...")
    paper_inst = pd.read_parquet(PAPER_INST, columns=["pmid", "institution_canonical_id"])
    paper_inst["pmid"] = paper_inst["pmid"].astype("string")
    institutions = pd.read_parquet(
        INSTITUTIONS, columns=["institution_canonical_id", "type", "country_code"]
    )
    print(f"  {len(paper_inst):,} paper-institution rows; {len(institutions):,} institutions")

    inputs = {
        "archetypes": ARCHETYPES,
        "paper_institutions": PAPER_INST,
        "institutions": INSTITUTIONS,
    }

    # --- Geography split-half stability (random + temporal) -----------------
    print("\nComputing geography split-half stability (random + temporal)...")
    geo_long = build_geo_long(novelty, paper_inst, institutions)
    geo_rows: list[dict[str, object]] = []
    for nov_col in NOVELTY_COLS:
        for mode in ("random", "temporal"):
            res = split_half_stability(
                geo_long,
                key_col="country_code",
                novelty_col=nov_col,
                seed=SEED,
                n_min=N_MIN,
                mode=mode,
            )
            geo_rows.append(
                {
                    "novelty_col": nov_col,
                    "mode": res["mode"],
                    "spearman_rho": res["spearman_rho"],
                    "n_keys": res["n_keys"],
                    "n_min": N_MIN,
                    "seed": SEED,
                }
            )
    geo_robust = pd.DataFrame(geo_rows)
    geo_path = _write(
        geo_robust,
        "geo_robustness",
        inputs,
        {
            "task": "V2-S06b",
            "axis": "geography",
            "method": "split_half_stability",
            "novelty_cols": NOVELTY_COLS,
            "n_min": N_MIN,
            "seed": SEED,
        },
    )

    # --- Sector bootstrap delta CI ------------------------------------------
    print("Computing sector company-vs-non-company bootstrap delta CI...")
    sector_flag = build_sector_flag(novelty, paper_inst, institutions)
    n_co = int(sector_flag["is_company"].sum())
    n_nc = int((~sector_flag["is_company"]).sum())
    print(f"  company papers: {n_co:,}; non-company papers: {n_nc:,}")
    sector_rows: list[dict[str, object]] = []
    for nov_col in NOVELTY_COLS:
        res = bootstrap_delta_ci(
            sector_flag,
            flag_col="is_company",
            novelty_col=nov_col,
            seed=SEED,
            n_boot=N_BOOT,
            ci=CI,
        )
        sector_rows.append(
            {
                "novelty_col": nov_col,
                "delta": res["delta"],
                "ci_low": res["ci_low"],
                "ci_high": res["ci_high"],
                "spans_zero": res["spans_zero"],
                "n_boot": res["n_boot"],
                "ci": CI,
                "n_company": n_co,
                "n_noncompany": n_nc,
            }
        )
    sector_robust = pd.DataFrame(sector_rows)
    sector_path = _write(
        sector_robust,
        "sector_robustness",
        inputs,
        {
            "task": "V2-S06b",
            "axis": "sector",
            "method": "bootstrap_delta_ci",
            "flag": "is_company",
            "novelty_cols": NOVELTY_COLS,
            "n_boot": N_BOOT,
            "ci": CI,
            "seed": SEED,
        },
    )

    # --- Prints -------------------------------------------------------------
    print("\n" + "=" * 72)
    print("GEOGRAPHY split-half stability  (Spearman ρ of per-country novelty ranking)")
    print(f"  path: {geo_path}")
    print(f"  (grain-independent Method-D analogue; keys with >= {N_MIN} papers in BOTH halves)")
    print(geo_robust.to_string(index=False))

    print("\n" + "=" * 72)
    print("SECTOR company-vs-non-company bootstrap delta CI  (company − non-company)")
    print(f"  path: {sector_path}")
    print(f"  ({N_BOOT} reps, {int(CI * 100)}% percentile CI; spans_zero ⇒ genuine near-null)")
    print(sector_robust.to_string(index=False))

    print("\nVERDICT (read the actual numbers above):")
    for _, r in geo_robust.iterrows():
        print(
            f"  geo {r['novelty_col']:<13} {r['mode']:<8} ρ={r['spearman_rho']:+.3f} "
            f"(n_keys={int(r['n_keys'])})"
        )
    for _, r in sector_robust.iterrows():
        verdict = "NEAR-NULL (CI spans 0)" if r["spans_zero"] else "REAL gap (CI excludes 0)"
        print(
            f"  sector {r['novelty_col']:<13} delta={r['delta']:+.4f} "
            f"CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] -> {verdict}"
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
