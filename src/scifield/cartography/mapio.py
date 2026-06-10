"""Loader / prep layer for the V2 cartography map v0 (V2-S07).

This module is the **data-prep seam** between the parquet tables produced by the
earlier cartography batches (cascade, roles, origins, flow) and the figure-assembly
that lives in ``V2/scripts/build_map.py``. It deliberately keeps *plotting out* — every
function here reads a parquet (or a small set of them) and returns a tidy, figure-ready
``pandas.DataFrame`` (or a small dict of frames). Because the I/O is trivial and the
shaping is pure, the loaders are individually testable against the real artifacts.

Design contract
---------------
* **Read-only.** Loaders only ever *read* the V2 ``data/`` parquets and the V1
  ``topic_hierarchy.parquet``; they never write.
* **Grain-aware.** The cascade and role tables carry a ``grain`` column
  (``"leaf"`` / ``"mid"``); loaders that touch them take a ``grain`` argument and
  filter to it (default ``"leaf"`` — the map's primary grain).
* **Canonical journals only.** Journal columns are the 10 canonical slugs (the
  ``jama_surg`` collapse already done upstream); :func:`journal_display` maps a slug
  to a short human label for axis ticks / hover, and :data:`ROLE_COLOR` is the locked
  source/bridge/terminal palette carried from ``v2_02_roles_velocity.ipynb``.
* **Graceful absence.** Each loader resolves its path under a ``repo_root`` (the repo
  root inferred from this file by default) and raises a clear ``FileNotFoundError`` if
  the artifact is missing, so the builder/tests can skip rather than crash cryptically.

Conventions
-----------
``from __future__ import annotations``; numpy-style docstrings; pandas only; ruff/black
line-length 100. Real values are read from the artifacts, never hardcoded.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pandas as pd

# Repo root inferred from this file: src/scifield/cartography/mapio.py -> repo root.
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Env var consulted when a loader gets no explicit ``data_version`` (see
#: ``V2/scripts/_path_config.py`` — this is the library-side mirror of that logic;
#: kept duplicated rather than imported so this module stays free of any
#: ``V2/scripts`` dependency).
_ENV_VAR = "SCIFIELD_DATA_VERSION"
_DEFAULT_DATA_VERSION = "v1"

#: Locked source / bridge / terminal palette (carried from v2_02_roles_velocity.ipynb).
ROLE_COLOR: dict[str, str] = {
    "source": "#34a853",
    "bridge": "#fbbc04",
    "terminal": "#ea4335",
}

#: Short human-readable labels for the 10 canonical journal slugs (display only —
#: always key on the slug; this is for axis ticks and hover text).
JOURNAL_DISPLAY: dict[str, str] = {
    "ann_surg": "Ann Surg",
    "arthroscopy": "Arthroscopy",
    "br_j_surg": "Br J Surg",
    "clin_orthop_relat_res": "CORR",
    "j_am_coll_surg": "J Am Coll Surg",
    "j_arthroplasty": "J Arthroplasty",
    "j_bone_joint_surg_am": "J Bone Joint Surg",
    "jama_surg": "JAMA Surg",
    "spine": "Spine",
    "surgery": "Surgery",
}

#: Valid grain tokens and the topic key each one is keyed on in the cascade tables.
GRAIN_TOPIC_KEYS: dict[str, str] = {"leaf": "topic_id", "mid": "mid_level_id"}


def journal_display(slug: str) -> str:
    """Map a canonical journal slug to a short human label.

    Parameters
    ----------
    slug :
        A canonical journal slug (one of the 10 in :data:`JOURNAL_DISPLAY`).

    Returns
    -------
    str
        The short display label, or the slug unchanged if it is not a known slug
        (so an unexpected value is surfaced verbatim rather than silently dropped).
    """
    return JOURNAL_DISPLAY.get(slug, slug)


def _resolve(repo_root: Path | None, *parts: str) -> Path:
    """Resolve a path under ``repo_root`` (or the inferred default) and require it.

    Parameters
    ----------
    repo_root :
        Repo root; if ``None`` the root inferred from this module's location is used.
    *parts :
        Path components joined under the root.

    Returns
    -------
    pathlib.Path
        The resolved, existing path.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    """
    root = repo_root if repo_root is not None else _DEFAULT_REPO_ROOT
    path = root.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"missing cartography artifact: {path}")
    return path


def _resolve_version(data_version: str | None) -> str:
    """Resolve the active data version: explicit > ``$SCIFIELD_DATA_VERSION`` > ``v1``.

    Mirrors ``V2/scripts/_path_config.resolve_data_version`` (duplicated to keep this
    library module independent of the build-script package).
    """
    return data_version or os.environ.get(_ENV_VAR) or _DEFAULT_DATA_VERSION


def _out_rel(version: str) -> str:
    """Cartography output root, relative to the repo root, for a data version.

    ``v1`` keeps its historical UNVERSIONED location (``V2/data``) so the frozen
    artifacts stay byte-identical; any other version is namespaced ``V2/data_<v>``.
    Mirrors ``V2/scripts/_path_config.out_root``.
    """
    return "V2/data" if version == "v1" else f"V2/data_{version}"


def _resolve_out(repo_root: Path | None, version: str, *parts: str) -> Path:
    """Resolve a built-artifact path (under the version's ``V2/data[_v]`` root)."""
    return _resolve(repo_root, _out_rel(version), *parts)


def _resolve_data(repo_root: Path | None, version: str, *parts: str) -> Path:
    """Resolve an input-data path (under the version's ``data/<version>`` dir)."""
    return _resolve(repo_root, "data", version, *parts)


def _check_grain(grain: str) -> str:
    """Validate a grain token, returning the topic key it maps to.

    Parameters
    ----------
    grain :
        ``"leaf"`` or ``"mid"``.

    Returns
    -------
    str
        The topic key (``"topic_id"`` for leaf, ``"mid_level_id"`` for mid).

    Raises
    ------
    ValueError
        If ``grain`` is not a recognised token.
    """
    if grain not in GRAIN_TOPIC_KEYS:
        raise ValueError(f"grain must be one of {sorted(GRAIN_TOPIC_KEYS)}, got {grain!r}")
    return GRAIN_TOPIC_KEYS[grain]


def topic_labels(*, repo_root: Path | None = None, data_version: str | None = None) -> pd.DataFrame:
    """Load leaf-topic labels (``top_words``) and sizes from the topic hierarchy.

    Parameters
    ----------
    repo_root :
        Repo root override (mainly for tests). Defaults to the inferred root.
    data_version :
        Data version selecting the ``data/<version>`` input dir. Defaults to
        ``$SCIFIELD_DATA_VERSION`` then ``"v1"`` (the frozen default → ``data/v1``).

    Returns
    -------
    pandas.DataFrame
        Columns ``[topic_id, label, size, mid_level_id]`` where ``label`` is the
        comma-joined ``top_words`` list (a compact topic name for hover text). One
        row per leaf topic.
    """
    version = _resolve_version(data_version)
    path = _resolve_data(repo_root, version, "topic_hierarchy.parquet")
    hier = pd.read_parquet(path, columns=["topic_id", "top_words", "size", "mid_level_id"])
    out = hier.copy()
    out["label"] = out["top_words"].apply(_join_words)
    return out[["topic_id", "label", "size", "mid_level_id"]].reset_index(drop=True)


def _join_words(words: object, *, n: int = 6) -> str:
    """Join the first ``n`` top-words of a topic into a compact label string.

    Parameters
    ----------
    words :
        The ``top_words`` cell — a list/array of strings, or NaN.
    n :
        Maximum number of words to keep.

    Returns
    -------
    str
        ``"word1, word2, ..."`` (up to ``n`` words), or ``""`` if unavailable.
    """
    if words is None:
        return ""
    try:
        seq = list(cast(Iterable[object], words))
    except TypeError:
        return str(words)
    return ", ".join(str(w) for w in seq[:n])


def load_lag_matrix(
    grain: str = "leaf", *, repo_root: Path | None = None, data_version: str | None = None
) -> pd.DataFrame:
    """Load the inter-journal lead-lag matrix for a grain (tidy long form).

    Parameters
    ----------
    grain :
        ``"leaf"`` or ``"mid"``.
    repo_root :
        Repo root override.
    data_version :
        Data version selecting the artifact root. Defaults to
        ``$SCIFIELD_DATA_VERSION`` then ``"v1"`` (→ frozen ``V2/data``).

    Returns
    -------
    pandas.DataFrame
        Columns ``[journal_i, journal_j, mean_lag, median_lag, n_shared, n_i_leads]``
        for the requested grain. ``mean_lag > 0`` ⇒ ``journal_i`` LEADS ``journal_j``
        (lag = first_year[j] − first_year[i]).
    """
    _check_grain(grain)
    version = _resolve_version(data_version)
    path = _resolve_out(repo_root, version, "cascade", "lag_matrix.parquet")
    df = pd.read_parquet(path)
    return df[df["grain"] == grain].drop(columns=["grain"]).reset_index(drop=True)


def lag_matrix_wide(
    grain: str = "leaf", *, repo_root: Path | None = None, data_version: str | None = None
) -> pd.DataFrame:
    """Load the lead-lag matrix as a square journal×journal frame of ``mean_lag``.

    The diagonal is filled with ``0.0`` (a journal does not lead itself); cell
    ``[i, j]`` is the signed mean lag with ``> 0`` meaning row journal ``i`` leads
    column journal ``j``.

    Parameters
    ----------
    grain :
        ``"leaf"`` or ``"mid"``.
    repo_root :
        Repo root override.
    data_version :
        Data version (threaded to :func:`load_lag_matrix`). Defaults to
        ``$SCIFIELD_DATA_VERSION`` then ``"v1"``.

    Returns
    -------
    pandas.DataFrame
        A square frame indexed and columned by canonical journal slug (sorted), values
        = ``mean_lag``, diagonal 0.0.
    """
    long = load_lag_matrix(grain, repo_root=repo_root, data_version=data_version)
    journals = sorted(set(long["journal_i"]) | set(long["journal_j"]))
    wide = (
        long.pivot(index="journal_i", columns="journal_j", values="mean_lag")
        .reindex(index=journals, columns=journals)
        .fillna(0.0)
    )
    wide.index.name = "journal_i"
    wide.columns.name = "journal_j"
    return wide


def load_origin_attribution(
    grain: str = "leaf", *, repo_root: Path | None = None, data_version: str | None = None
) -> pd.DataFrame:
    """Load the per-topic origin attribution for a grain.

    Parameters
    ----------
    grain :
        ``"leaf"`` or ``"mid"``.
    repo_root :
        Repo root override.
    data_version :
        Data version selecting the artifact root. Defaults to
        ``$SCIFIELD_DATA_VERSION`` then ``"v1"`` (→ frozen ``V2/data``).

    Returns
    -------
    pandas.DataFrame
        Columns ``[topic_id, origin_journal_slug, origin_year, n_journals, tie,
        n_co_earliest]`` (mid grain re-keys on ``topic_id`` carrying the mid id). One
        row per topic.
    """
    key = _check_grain(grain)
    version = _resolve_version(data_version)
    path = _resolve_out(repo_root, version, "cascade", "origin_attribution.parquet")
    df = pd.read_parquet(path)
    sub = df[df["grain"] == grain].copy()
    if key == "mid_level_id":
        sub["topic_id"] = sub["mid_level_id"]
    cols = [
        "topic_id",
        "origin_journal_slug",
        "origin_year",
        "n_journals",
        "tie",
        "n_co_earliest",
    ]
    return sub[cols].reset_index(drop=True)


def load_diffusion_curves(
    grain: str = "leaf", *, repo_root: Path | None = None, data_version: str | None = None
) -> pd.DataFrame:
    """Load the per-topic diffusion / adoption-curve parameters for a grain.

    Parameters
    ----------
    grain :
        ``"leaf"`` or ``"mid"``.
    repo_root :
        Repo root override.
    data_version :
        Data version selecting the artifact root. Defaults to
        ``$SCIFIELD_DATA_VERSION`` then ``"v1"`` (→ frozen ``V2/data``).

    Returns
    -------
    pandas.DataFrame
        Columns ``[topic_id, origin_year, n_journals, reach_fraction, span_years,
        t50_empirical, t50_logistic, logistic_rate, curve_fitted]``. One row per topic.
    """
    key = _check_grain(grain)
    version = _resolve_version(data_version)
    path = _resolve_out(repo_root, version, "cascade", "diffusion_curves.parquet")
    df = pd.read_parquet(path)
    sub = df[df["grain"] == grain].copy()
    if key == "mid_level_id":
        sub["topic_id"] = sub["mid_level_id"]
    cols = [
        "topic_id",
        "origin_year",
        "n_journals",
        "reach_fraction",
        "span_years",
        "t50_empirical",
        "t50_logistic",
        "logistic_rate",
        "curve_fitted",
    ]
    return sub[cols].reset_index(drop=True)


def adoption_curve(
    grain: str = "leaf", *, repo_root: Path | None = None, data_version: str | None = None
) -> pd.DataFrame:
    """Build the corpus-level mean adoption curve from per-topic diffusion data.

    For each integer offset ``k = 0 .. 9`` (number of *additional* journals beyond the
    origin), the fraction of topics that have reached at least ``k + 1`` journals by
    that offset is approximated from the per-topic ``n_journals`` reach. This gives a
    monotone non-increasing "how many topics reach this breadth" curve — a compact,
    honest summary of cross-journal spread without re-fitting per-topic curves here.

    Parameters
    ----------
    grain :
        ``"leaf"`` or ``"mid"``.
    repo_root :
        Repo root override.
    data_version :
        Data version (threaded to :func:`load_diffusion_curves`). Defaults to
        ``$SCIFIELD_DATA_VERSION`` then ``"v1"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[n_journals_reached, frac_topics]`` where ``frac_topics`` is the
        fraction of topics reaching at least ``n_journals_reached`` journals.
    """
    dc = load_diffusion_curves(grain, repo_root=repo_root, data_version=data_version)
    n_topics = len(dc)
    rows = []
    for k in range(1, 11):
        frac = float((dc["n_journals"] >= k).mean()) if n_topics else 0.0
        rows.append({"n_journals_reached": k, "frac_topics": frac})
    return pd.DataFrame(rows)


def load_roles(
    grain: str = "leaf", *, repo_root: Path | None = None, data_version: str | None = None
) -> pd.DataFrame:
    """Load the per-journal role scores for a grain.

    Parameters
    ----------
    grain :
        ``"leaf"`` or ``"mid"``.
    repo_root :
        Repo root override.
    data_version :
        Data version selecting the artifact root. Defaults to
        ``$SCIFIELD_DATA_VERSION`` then ``"v1"`` (→ frozen ``V2/data``).

    Returns
    -------
    pandas.DataFrame
        Columns ``[journal_slug, specialty, seeding_score, net_outflow_share,
        betweenness, source, bridge, terminal, source_z, bridge_z, terminal_z, role]``
        for the requested grain (one row per journal), with a ``display`` column added.
    """
    _check_grain(grain)
    version = _resolve_version(data_version)
    path = _resolve_out(repo_root, version, "roles", "role_scores.parquet")
    df = pd.read_parquet(path)
    sub = df[df["grain"] == grain].drop(columns=["grain"]).reset_index(drop=True)
    sub["display"] = sub["journal_slug"].map(journal_display)
    return sub


def load_velocity(
    *, repo_root: Path | None = None, data_version: str | None = None
) -> pd.DataFrame:
    """Load the per-journal citational-velocity table (grain-independent).

    Parameters
    ----------
    repo_root :
        Repo root override.
    data_version :
        Data version selecting the artifact root. Defaults to
        ``$SCIFIELD_DATA_VERSION`` then ``"v1"`` (→ frozen ``V2/data``).

    Returns
    -------
    pandas.DataFrame
        Columns ``[journal_slug, specialty, n_citations, median_lag, mean_lag, iqr_lag,
        q25_lag, q75_lag, frac_within_2y, velocity]`` plus a ``display`` column. One row
        per journal.
    """
    version = _resolve_version(data_version)
    path = _resolve_out(repo_root, version, "roles", "velocity.parquet")
    df = pd.read_parquet(path)
    out = df.copy()
    out["display"] = out["journal_slug"].map(journal_display)
    return out


def load_origins(
    *, repo_root: Path | None = None, data_version: str | None = None
) -> dict[str, pd.DataFrame]:
    """Load the four novelty-origin tables as a dict of frames.

    Parameters
    ----------
    repo_root :
        Repo root override.
    data_version :
        Data version selecting the artifact root. Defaults to
        ``$SCIFIELD_DATA_VERSION`` then ``"v1"`` (→ frozen ``V2/data``).

    Returns
    -------
    dict of str to pandas.DataFrame
        Keys ``"sector"``, ``"geo"``, ``"recombination"``, ``"recombination_by_topic"``
        mapping to the respective ``V2/data[_v]/origins/*.parquet`` frames.
    """
    version = _resolve_version(data_version)
    sector = pd.read_parquet(_resolve_out(repo_root, version, "origins", "sector_novelty.parquet"))
    geo = pd.read_parquet(_resolve_out(repo_root, version, "origins", "geo_novelty.parquet"))
    recomb = pd.read_parquet(_resolve_out(repo_root, version, "origins", "recombination.parquet"))
    recomb_topic = pd.read_parquet(
        _resolve_out(repo_root, version, "origins", "recombination_by_topic.parquet")
    )
    return {
        "sector": sector,
        "geo": geo,
        "recombination": recomb,
        "recombination_by_topic": recomb_topic,
    }


def geo_ranked(
    *,
    repo_root: Path | None = None,
    data_version: str | None = None,
    novelty_col: str = "sem_nov_mean_mean",
    well_sampled: bool = True,
) -> pd.DataFrame:
    """Return countries ranked by a novelty measure (most-differentiated origin axis).

    Parameters
    ----------
    repo_root :
        Repo root override.
    data_version :
        Data version (threaded to :func:`load_origins`). Defaults to
        ``$SCIFIELD_DATA_VERSION`` then ``"v1"``.
    novelty_col :
        The novelty column to rank by (default semantic-novelty mean).
    well_sampled :
        If True (default), drop low-n countries (``low_n == True``) so the ranking is
        not dominated by 1-paper countries.

    Returns
    -------
    pandas.DataFrame
        Columns ``[country_code, n_papers, <novelty_col>]`` sorted descending by the
        novelty column.
    """
    geo = load_origins(repo_root=repo_root, data_version=data_version)["geo"]
    sub = geo[~geo["low_n"]] if well_sampled else geo
    cols = ["country_code", "n_papers", novelty_col]
    return sub[cols].sort_values(novelty_col, ascending=False).reset_index(drop=True)


def topic_landscape_frame(
    grain: str = "leaf", *, repo_root: Path | None = None, data_version: str | None = None
) -> pd.DataFrame:
    """Assemble the topic-landscape scatter frame (labels + size + origin + recombination).

    Joins :func:`topic_labels` to the leaf origin attribution and the per-topic
    recombination rate, producing one row per leaf topic ready for a sized, coloured
    scatter (size = topic size, colour = origin journal, hover = words + recombination).

    Parameters
    ----------
    grain :
        Currently only ``"leaf"`` is supported for the landscape (the topic-words live
        at leaf grain); passing ``"mid"`` raises ``ValueError``.
    repo_root :
        Repo root override.
    data_version :
        Data version (threaded to the underlying loaders). Defaults to
        ``$SCIFIELD_DATA_VERSION`` then ``"v1"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[topic_id, label, size, origin_journal_slug, origin_display,
        origin_year, n_journals, recombination_rate]``.
    """
    if grain != "leaf":
        raise ValueError("topic_landscape_frame supports the leaf grain only (topic words)")
    labels = topic_labels(repo_root=repo_root, data_version=data_version)
    origin = load_origin_attribution("leaf", repo_root=repo_root, data_version=data_version)
    recomb = load_origins(repo_root=repo_root, data_version=data_version)["recombination_by_topic"]

    df = labels.merge(origin, on="topic_id", how="left")
    df = df.merge(recomb[["topic_id", "recombination_rate"]], on="topic_id", how="left")
    df["origin_display"] = df["origin_journal_slug"].map(journal_display)
    cols = [
        "topic_id",
        "label",
        "size",
        "origin_journal_slug",
        "origin_display",
        "origin_year",
        "n_journals",
        "recombination_rate",
    ]
    return df[cols].reset_index(drop=True)
