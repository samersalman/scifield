"""Build the V2 journal-role + citational-velocity tables (V2-S05).

Runnable I/O driver for :mod:`scifield.cartography.roles`. The roles module is pure;
this script owns all the I/O the module avoids:

1. Load the shipped flow tables ``V2/data/flow/flow_{leaf,mid}.parquet`` — already
   carrying the canonical ``journal_slug`` (the jama_surg collapse), the per-cell
   ``is_first_appearance`` anchor, and the corpus-internal citation ``out_flow`` /
   ``in_flow``. We consume ``journal_slug`` from them and never re-derive it.
2. For each grain build the directed seeding network (the topic-flow graph) from the
   flow slice via :func:`scifield.findings.seeding.directed_seeding_network`, then
   compute :func:`~scifield.cartography.roles.role_scores` (source / bridge /
   terminal + label).
3. Run the **drop-one-journal jackknife** of the role scores
   (:func:`~scifield.cartography.roles.role_scores_jackknife`) and feed it to
   :func:`scifield.cartography.nulls.rank_stability` to report the protocol's
   ``role_rank_correlation`` (Method C; Gate G6 roles input) at BOTH grains.
4. Compute per-journal **citational velocity** (years from publication to citation)
   by joining ``cited_by.parquet`` (external inbound citations) to the focal paper's
   ``journal_slug`` + publication year, resolved via
   ``archetypes.openalex_id → papers_distinct.journal_slug`` (read-only DuckDB).
5. Write ``V2/data/roles/{role_scores,velocity}.parquet`` (both grains stacked for
   role_scores) — each with a :func:`scifield.repro.record_run` sidecar
   (config ``task="V2-S05"``) — and print the role table, the velocity table, and the
   jackknife ``role_rank_correlation``.

Within-corpus prior (carry into the doc): the true source generalists (Nature / NEJM
/ Lancet) are NOT in the corpus, so roles are validated against the WITHIN-corpus
expectation that the generalist *surgery* journals (Ann Surg / Br J Surg / JAMA Surg /
J Am Coll Surg / Surgery) behave more source/bridge-like and the subspecialty journals
(Spine / Arthroscopy / J Arthroplasty / CORR / J Bone Joint Surg) more terminal-like.

1995 left-censoring caveat: the seeding sub-signal of the source/terminal scores is
biased by the panel's 1995 start (S03 hand-off); the citation-flow and betweenness
sub-signals are not, and the script prints the seeding-vs-flow decomposition so a
reader can see neither alone drives the label.

Usage
-----
``.venv/bin/python V2/scripts/build_roles.py``            # both grains (default)
``.venv/bin/python V2/scripts/build_roles.py --grain leaf``

$0 / read-only: reads the V2 flow parquets + archetypes + cited_by + papers DuckDB
(read_only=True); no network, no GPU, no DeepSeek.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from scifield.cartography.nulls import rank_stability
from scifield.cartography.roles import (
    citational_velocity,
    role_scores,
    role_scores_jackknife,
)
from scifield.findings.seeding import directed_seeding_network, specialty_of
from scifield.repro import record_run

REPO_ROOT = Path(__file__).resolve().parents[2]
FLOW_DIR = REPO_ROOT / "V2/data/flow"
ARCHETYPES = REPO_ROOT / "data/v1/archetypes.parquet"
CITED_BY = REPO_ROOT / "data/v1/enrichment/cited_by.parquet"
PAPERS_DUCKDB = REPO_ROOT / "data/v1/papers.duckdb"
OUT_DIR = REPO_ROOT / "V2/data/roles"

GRAIN_KEYS = {"leaf": "topic_id", "mid": "mid_level_id"}
WEIGHT_THRESHOLD = 0.5  # min lead→follow weight for the betweenness graph


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


def build_role_grain(grain: str) -> tuple[pd.DataFrame, float]:
    """Compute role scores + the jackknife ``role_rank_correlation`` for one grain.

    Parameters
    ----------
    grain :
        ``"leaf"`` or ``"mid"``.

    Returns
    -------
    scores : pandas.DataFrame
        :func:`~scifield.cartography.roles.role_scores` output with a leading
        ``grain`` column and a joined ``specialty`` column (for prior validation).
    role_rank_correlation : float
        The mean Spearman rank-correlation of the source-axis ranking across the ten
        drop-one-journal runs (protocol Method C metric).
    """
    topic_key = GRAIN_KEYS[grain]
    flow = load_flow(grain)

    seed_in = flow[[topic_key, "journal_slug", "year"]].rename(columns={topic_key: "topic_id"})
    net = directed_seeding_network(seed_in, entity_col="journal_slug")
    scores = role_scores(flow, net, topic_key=topic_key, weight_threshold=WEIGHT_THRESHOLD)
    scores.insert(0, "grain", grain)
    scores["specialty"] = scores["journal_slug"].map(specialty_of)

    jk = role_scores_jackknife(
        flow, topic_key=topic_key, weight_threshold=WEIGHT_THRESHOLD, score_col="source_z"
    )
    rrc = rank_stability(jk, key_col="journal_slug", score_col="score", run_col="held_out")
    return scores, float(rrc)


def load_paper_meta() -> pd.DataFrame:
    """Per-paper ``[openalex_id, journal_slug, year]`` for the velocity join.

    Resolves the canonical ``journal_slug`` via
    ``archetypes.pmid → papers_distinct.pmid`` (DuckDB read-only), NOT by parsing the
    archetypes display ``journal`` string.

    Returns
    -------
    pandas.DataFrame
        One row per focal paper with a non-null ``openalex_id``: ``openalex_id``,
        ``journal_slug``, ``year``.
    """
    arch = pd.read_parquet(ARCHETYPES, columns=["pmid", "openalex_id", "year"])
    arch = arch.dropna(subset=["openalex_id"]).copy()
    arch["pmid"] = arch["pmid"].astype("string")

    con = duckdb.connect(str(PAPERS_DUCKDB), read_only=True)
    try:
        pj = con.execute("SELECT pmid, journal_slug FROM papers_distinct").df()
    finally:
        con.close()
    pj["pmid"] = pj["pmid"].astype("string")

    meta = arch.merge(pj, on="pmid", how="inner")
    return meta[["openalex_id", "journal_slug", "year"]]


def build_velocity() -> pd.DataFrame:
    """Compute per-journal citational velocity from external inbound citations.

    Returns
    -------
    pandas.DataFrame
        :func:`~scifield.cartography.roles.citational_velocity` output (one row per
        journal), with a joined ``specialty`` column.
    """
    meta = load_paper_meta()
    cb = pd.read_parquet(CITED_BY, columns=["focal_oa_id", "citing_year"])
    cb = cb.rename(columns={"focal_oa_id": "openalex_id"})
    vel = citational_velocity(cb, meta, journal_col="journal_slug")
    vel["specialty"] = vel["journal_slug"].map(specialty_of)
    return vel


def write_table(name: str, frame: pd.DataFrame, *, grains: list[str], extra: dict) -> Path:
    """Write a parquet + a record_run sidecar.

    Parameters
    ----------
    name :
        Output stem (``role_scores`` / ``velocity``).
    frame :
        The frame to write.
    grains :
        Grains built (for the sidecar config).
    extra :
        Extra config keys for the sidecar (e.g. the role_rank_correlation values).

    Returns
    -------
    pathlib.Path
        The written parquet path.
    """
    out_path = OUT_DIR / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)

    inputs = {f"flow_{g}": FLOW_DIR / f"flow_{g}.parquet" for g in grains}
    if name == "velocity":
        inputs |= {"archetypes": ARCHETYPES, "cited_by": CITED_BY}
    record_run(
        artifact_path=out_path,
        inputs=inputs,
        config={
            "task": "V2-S05",
            "table": name,
            "grains": grains,
            "weight_threshold": WEIGHT_THRESHOLD,
            "role_rule": (
                "argmax of z-scored {source, bridge, terminal}; " "ties source>bridge>terminal"
            ),
            "source": "mean z(seeding_score) + z(net_outflow_share)",
            "bridge": "betweenness centrality on the directed seeding network",
            "terminal": "mean z(-net_outflow_share) + z(1-seeding_score)",
            "velocity": "years from publication to citation (citing_year - pub_year)",
            "panel_conditional": (
                "roles are within-panel; true source generalists (Nature/NEJM/Lancet) "
                "are not in the corpus"
            ),
            "left_censoring": (
                "seeding sub-signal biased by 1995 panel start; flow + betweenness "
                "sub-signals are not"
            ),
            **extra,
        },
    )
    return out_path


def print_role_table(scores: pd.DataFrame, grain: str) -> None:
    """Print the per-journal role table for one grain (scores + label + prior)."""
    sub = scores[scores["grain"] == grain].sort_values("role")
    print(f"\n=== ROLE SCORES ({grain}) ===")
    cols = [
        "journal_slug",
        "specialty",
        "seeding_score",
        "net_outflow_share",
        "betweenness",
        "source",
        "bridge",
        "terminal",
        "role",
    ]
    show = sub[cols].copy()
    for c in cols[2:-1]:
        show[c] = show[c].astype("float64").round(3)
    print(show.to_string(index=False))


def print_velocity_table(vel: pd.DataFrame) -> None:
    """Print the per-journal citational-velocity table."""
    print("\n=== CITATIONAL VELOCITY (years publication -> citation) ===")
    cols = [
        "journal_slug",
        "specialty",
        "n_citations",
        "n_papers_cited",
        "median_lag",
        "mean_lag",
        "iqr_lag",
        "frac_within_2y",
        "velocity",
    ]
    show = vel[cols].copy()
    for c in ("mean_lag", "iqr_lag", "frac_within_2y"):
        show[c] = show[c].astype("float64").round(3)
    print(show.to_string(index=False))


def main() -> None:
    """CLI entry point: build the role + velocity tables and print the audit."""
    parser = argparse.ArgumentParser(description="Build V2 journal-role + velocity tables.")
    parser.add_argument(
        "--grain",
        choices=["leaf", "mid", "both"],
        default="both",
        help="topic granularity to build (default: both)",
    )
    args = parser.parse_args()
    grains = ["leaf", "mid"] if args.grain == "both" else [args.grain]

    role_frames: list[pd.DataFrame] = []
    rrc_by_grain: dict[str, float] = {}
    for grain in grains:
        print(f"\nBuilding roles for grain={grain!r}...")
        scores, rrc = build_role_grain(grain)
        role_frames.append(scores)
        rrc_by_grain[grain] = rrc
        print(
            f"  {len(scores)} journals scored | "
            f"role_rank_correlation (drop-one jackknife) = {rrc:.4f}"
        )

    all_scores = pd.concat(role_frames, ignore_index=True)

    # Velocity is grain-independent (paper-level), computed once.
    print("\nComputing citational velocity (whole-graph inbound citations)...")
    vel = build_velocity()

    role_path = write_table(
        "role_scores",
        all_scores,
        grains=grains,
        extra={f"role_rank_correlation_{g}": rrc_by_grain[g] for g in grains},
    )
    vel_path = write_table("velocity", vel, grains=grains, extra={})

    for grain in grains:
        print_role_table(all_scores, grain)
    print_velocity_table(vel)

    print("\n=== JACKKNIFE role_rank_correlation (Method C; Gate G6 roles input) ===")
    bar = 0.7
    for grain in grains:
        rrc = rrc_by_grain[grain]
        verdict = "PASS" if rrc >= bar else "BELOW BAR"
        print(f"  {grain}: role_rank_correlation = {rrc:.4f}  (bar >= {bar} -> {verdict})")

    print("\nDone. Outputs + sidecars:")
    for path in (role_path, vel_path):
        sidecar = Path(str(path) + ".run.json")
        ok = path.exists() and sidecar.exists()
        print(f"  [{'ok' if ok else 'XX'}] {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
