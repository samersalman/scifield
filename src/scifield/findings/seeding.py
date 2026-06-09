"""V1-S15 cross-journal *seeding* — exploratory, NON-gating bonus layer.

This module asks a deliberately simple lead/follow question: within a research
topic, *which entity tends to publish first* and which tends to follow? An entity
that consistently shows up early across many topics is said to **seed** that
literature; one that consistently arrives late is a **follower**. The signal is
purely temporal (first-publication year per topic) — it makes no causal claim and
is clearly labelled exploratory, so it NEVER contributes to the Gate G5 confirmed
finding count.

Every function here is **pure and I/O-free**: it receives an in-memory tidy frame
(one row per paper, at least ``[topic_id, <entity>, year]``) and returns a new
frame. The notebook ``notebooks/14_bonus_cross_journal.ipynb`` does all the I/O
(reading ``archetypes.parquet`` + the enrichment tables) and draws the network
with matplotlib; this module owns only the logic. There is intentionally **no
networkx dependency** — the directed seeding network is returned as a pandas
edge-list frame.

Entity grain
------------
The same logic runs at any grain via ``entity_col``:

* ``"journal"`` — the 10 corpus journals (slug *or* display name; see
  :data:`SPECIALTY_GROUPS` / :func:`specialty_of`).
* ``"institution_canonical_id"`` — ROR-canonicalised institutions.
* ``"country_code"`` — institution country (blank for ~44% of institutions).

:func:`seeding_by_group` is the grain-agnostic convenience wrapper; it **skips
blank / NaN entities** (e.g. the missing country codes) rather than imputing them,
so an absent country never invents a phantom seeder.

The normalized-lead score
--------------------------
Per topic the participating entities are ordered by ``first_year`` ascending and
assigned a rank ``r ∈ {0 .. m-1}`` with **ties resolved to the minimum rank**
(co-earliest entities share rank 0). The per-topic *normalized lead* is

    normalized_lead = 1 - r / (m - 1)        if m > 1
    normalized_lead = 1.0                     if m == 1

so the earliest entity scores ``1.0``, the latest ``0.0``, and a lone entity in a
topic scores ``1.0`` (it is trivially first). An entity's :func:`seeding_score` is
the **mean** of its normalized leads over the topics it appears in — higher means
it more consistently publishes early.

Specialty grouping
------------------
:data:`SPECIALTY_GROUPS` records the locked 5-vs-5 anatomical split of the corpus
journals (orthopedic vs general_surgery), keyed by BOTH duckdb ``journal_slug``
AND archetypes ``journal`` display names because callers hold one or the other.
:func:`specialty_of` normalises an input (lower / strip / spaces+hyphens →
underscore) and resolves it to ``"orthopedic"`` | ``"general_surgery"`` | ``None``.

Conventions
-----------
``from __future__ import annotations``; numpy-style docstrings; numpy + pandas +
scipy only (no networkx); ruff/black line-length 100. Real counts are COMPUTED,
never hardcoded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "SPECIALTY_GROUPS",
    "directed_seeding_network",
    "first_publication_year",
    "normalize_journal",
    "seeding_by_group",
    "seeding_score",
    "specialty_of",
]


# --------------------------------------------------------------------------- #
# Locked specialty grouping (Samer 2026-06-08): 5-vs-5 anatomical split.
# Keyed by BOTH the duckdb journal_slug AND the archetypes display name, since
# callers hold one or the other. Matching is also case-insensitive / by slug via
# normalize_journal (lower, strip, spaces+hyphens -> underscore).
# --------------------------------------------------------------------------- #
SPECIALTY_GROUPS: dict[str, tuple[str, ...]] = {
    "orthopedic": (
        # slugs
        "spine",
        "j_arthroplasty",
        "clin_orthop_relat_res",
        "j_bone_joint_surg_am",
        "arthroscopy",
        # display names
        "Spine",
        "J Arthroplasty",
        "Clin Orthop Relat Res",
        "J Bone Joint Surg Am",
        "Arthroscopy",
    ),
    "general_surgery": (
        # slugs
        "surgery",
        "ann_surg",
        "br_j_surg",
        "j_am_coll_surg",
        "jama_surg",
        # display names
        "Surgery",
        "Ann Surg",
        "Br J Surg",
        "J Am Coll Surg",
        "JAMA Surg",
    ),
}


def normalize_journal(journal: object) -> str:
    """Normalise a journal slug or display name to a canonical lookup key.

    The corpus identifies a journal either by its duckdb ``journal_slug``
    (``"j_bone_joint_surg_am"``) or by the archetypes display ``journal`` string
    (``"J Bone Joint Surg Am"``). Both collapse to the same key by lower-casing,
    stripping surrounding whitespace, and mapping every run of spaces / hyphens to
    a single underscore.

    Parameters
    ----------
    journal :
        A slug or display name. Non-string / missing values (``None``, ``NaN``)
        normalise to the empty string ``""``.

    Returns
    -------
    str
        The canonical key, e.g. both ``"J Bone Joint Surg Am"`` and
        ``"j_bone_joint_surg_am"`` -> ``"j_bone_joint_surg_am"``. A missing input
        yields ``""``.

    Notes
    -----
    Pure and deterministic. Internal hyphens/spaces collapse so ``"Clin Orthop
    Relat Res"`` and ``"clin_orthop_relat_res"`` agree.
    """
    if journal is None:
        return ""
    # Treat pandas/NumPy NaN (a float that is not equal to itself) as missing.
    if isinstance(journal, float) and journal != journal:
        return ""
    text = str(journal).strip().lower()
    out_chars: list[str] = []
    prev_underscore = False
    for ch in text:
        if ch in " -_\t":
            if not prev_underscore:
                out_chars.append("_")
                prev_underscore = True
        else:
            out_chars.append(ch)
            prev_underscore = False
    return "".join(out_chars).strip("_")


# Precomputed reverse index: normalised key -> specialty group. Built once at
# import from SPECIALTY_GROUPS so specialty_of is a dict lookup.
_NORMALIZED_TO_GROUP: dict[str, str] = {
    normalize_journal(name): group for group, names in SPECIALTY_GROUPS.items() for name in names
}


def specialty_of(journal: object) -> str | None:
    """Resolve a journal slug or display name to its specialty group.

    Parameters
    ----------
    journal :
        A duckdb ``journal_slug`` *or* an archetypes display ``journal`` name.
        Normalised via :func:`normalize_journal` before lookup, so case,
        surrounding whitespace, and spaces-vs-hyphens-vs-underscores do not matter.

    Returns
    -------
    str or None
        ``"orthopedic"`` or ``"general_surgery"`` for a known corpus journal,
        else ``None`` (unknown journal, blank, or missing input).

    Examples
    --------
    >>> specialty_of("j_bone_joint_surg_am")
    'orthopedic'
    >>> specialty_of("Ann Surg")
    'general_surgery'
    >>> specialty_of("Nature") is None
    True
    """
    key = normalize_journal(journal)
    if not key:
        return None
    return _NORMALIZED_TO_GROUP.get(key)


# --------------------------------------------------------------------------- #
# Pure logic core (no I/O -- unit-testable on in-memory data).
# --------------------------------------------------------------------------- #


def first_publication_year(
    df: pd.DataFrame,
    *,
    entity_col: str = "journal",
    min_papers: int = 1,
) -> pd.DataFrame:
    """First year each entity reached ``min_papers`` papers in a topic.

    Parameters
    ----------
    df :
        Tidy frame with one row per paper carrying at least ``["topic_id",
        entity_col, "year"]``. Extra columns are ignored. Rows whose ``topic_id``,
        ``entity_col``, or ``year`` is missing are dropped (a paper with no entity
        or no year cannot anchor a first-publication year).
    entity_col :
        Column naming the entity grain — ``"journal"`` (slug or display name),
        ``"institution_canonical_id"``, or ``"country_code"``. Default
        ``"journal"``.
    min_papers :
        The publication-count threshold. The returned ``first_year`` is the
        earliest year in which the (topic, entity) pair accumulated **at least**
        ``min_papers`` papers *in that single year*. Default ``1`` (the first year
        the entity appears in the topic at all).

    Returns
    -------
    pandas.DataFrame
        Columns ``["topic_id", entity_col, "first_year"]``, one row per
        (topic, entity) pair that ever met the threshold, sorted by ``topic_id``
        then ``entity_col``. ``first_year`` is integer-valued. Pairs that never hit
        ``min_papers`` in any single year are omitted.

    Notes
    -----
    The threshold is evaluated **per year**, not cumulatively: with
    ``min_papers=2`` a (topic, entity) that published one paper a year forever is
    excluded, whereas one that published two papers in some year qualifies, its
    ``first_year`` being that year.
    """
    import pandas as pd

    needed = ["topic_id", entity_col, "year"]
    work = df.loc[:, needed].copy()
    work = work.dropna(subset=needed)
    if work.empty:
        return pd.DataFrame({"topic_id": [], entity_col: [], "first_year": []})

    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.dropna(subset=["year"])
    work["year"] = work["year"].astype("int64")

    # Papers per (topic, entity, year); keep only year-buckets meeting threshold.
    counts = (
        work.groupby(["topic_id", entity_col, "year"], dropna=False)
        .size()
        .reset_index(name="n_papers")
    )
    qualifying = counts.loc[counts["n_papers"] >= int(min_papers)]
    if qualifying.empty:
        return pd.DataFrame({"topic_id": [], entity_col: [], "first_year": []})

    first = (
        qualifying.groupby(["topic_id", entity_col], dropna=False)["year"]
        .min()
        .reset_index(name="first_year")
    )
    first["first_year"] = first["first_year"].astype("int64")
    first = first.sort_values(["topic_id", entity_col]).reset_index(drop=True)
    return first


def seeding_score(
    df: pd.DataFrame,
    *,
    entity_col: str = "journal",
    min_papers: int = 1,
) -> pd.DataFrame:
    """Mean normalized-lead "seeding" score per entity across its topics.

    Within each topic the entities are ranked by :func:`first_publication_year`
    ascending (``r = 0`` earliest), ties taking the **minimum** rank. The per-topic
    *normalized lead* of an entity with rank ``r`` among ``m`` entities is
    ``1 - r / (m - 1)`` when ``m > 1`` else ``1.0`` — earliest maps to ``1.0``,
    latest to ``0.0``. An entity's ``seeding_score`` is the mean of its normalized
    leads over the topics it participates in.

    Parameters
    ----------
    df :
        Tidy paper frame; see :func:`first_publication_year`.
    entity_col :
        Entity grain column. Default ``"journal"``.
    min_papers :
        Per-year publication threshold passed to :func:`first_publication_year`.
        Default ``1``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[entity_col, "seeding_score", "n_topics"]``, one row per entity,
        sorted by ``seeding_score`` descending then ``entity_col`` ascending.
        ``seeding_score`` is a float in ``[0, 1]`` (higher = consistently early =
        "seeds"); ``n_topics`` is the integer number of topics the entity took part
        in. Empty input yields an empty frame with these columns.

    Notes
    -----
    The rank base is ``m - 1`` where ``m`` is the count of entities **present in
    that topic** (after the ``min_papers`` filter), so topics with more
    participants spread the lead more finely. A single-entity topic contributes a
    perfect ``1.0`` to that entity.
    """
    import pandas as pd

    first = first_publication_year(df, entity_col=entity_col, min_papers=min_papers)
    if first.empty:
        return pd.DataFrame({entity_col: [], "seeding_score": [], "n_topics": []})

    leads = _normalized_leads(first, entity_col=entity_col)

    agg = (
        leads.groupby(entity_col, dropna=False)["normalized_lead"]
        .agg(seeding_score="mean", n_topics="size")
        .reset_index()
    )
    agg["n_topics"] = agg["n_topics"].astype("int64")
    agg = agg.sort_values(["seeding_score", entity_col], ascending=[False, True]).reset_index(
        drop=True
    )
    return agg


def directed_seeding_network(
    df: pd.DataFrame,
    *,
    entity_col: str = "journal",
    min_papers: int = 1,
) -> pd.DataFrame:
    """Directed lead/follow edge list between every co-occurring entity pair.

    For each **ordered** pair ``(src, dst)`` that shares ``n_shared`` topics,
    ``n_precedes`` counts the shared topics where ``first_year[src] <
    first_year[dst]`` (a strict lead — equal years count for neither direction),
    and ``weight = n_precedes / n_shared``. A high-weight edge ``src -> dst`` means
    ``src`` tends to publish before ``dst`` (``src`` seeds ``dst``).

    Parameters
    ----------
    df :
        Tidy paper frame; see :func:`first_publication_year`.
    entity_col :
        Entity grain column. Default ``"journal"``.
    min_papers :
        Per-year publication threshold passed to :func:`first_publication_year`.
        Default ``1``.

    Returns
    -------
    pandas.DataFrame
        Edge list with columns ``["src", "dst", "n_precedes", "n_shared",
        "weight"]``, one row per ordered pair of distinct entities sharing ``>= 1``
        topic, sorted by ``src`` then ``dst``. ``n_precedes``/``n_shared`` are
        integers; ``weight`` is ``n_precedes / n_shared`` in ``[0, 1]``. Both
        directions of a sharing pair appear; their ``weight`` values sum to ``1``
        only when the pair never ties (tied topics deduct from both). **No
        networkx** — the notebook draws this with matplotlib.

    Notes
    -----
    The frame is symmetric in ``n_shared`` (``src->dst`` and ``dst->src`` share the
    same value) but generally asymmetric in ``n_precedes``/``weight``. An empty
    input — or one where no two entities ever co-occur — yields an empty edge list
    with these columns.
    """
    import pandas as pd

    first = first_publication_year(df, entity_col=entity_col, min_papers=min_papers)
    cols = ["src", "dst", "n_precedes", "n_shared", "weight"]
    if first.empty:
        return pd.DataFrame({c: [] for c in cols})

    # Self-join on topic_id to enumerate co-occurring entity pairs per topic.
    left = first.rename(columns={entity_col: "src", "first_year": "src_year"})
    right = first.rename(columns={entity_col: "dst", "first_year": "dst_year"})
    pairs = left.merge(right, on="topic_id")
    # Drop the diagonal (an entity paired with itself).
    pairs = pairs.loc[pairs["src"] != pairs["dst"]].copy()
    if pairs.empty:
        return pd.DataFrame({c: [] for c in cols})

    pairs["precedes"] = (pairs["src_year"] < pairs["dst_year"]).astype("int64")

    edges = (
        pairs.groupby(["src", "dst"], dropna=False)
        .agg(n_precedes=("precedes", "sum"), n_shared=("precedes", "size"))
        .reset_index()
    )
    edges["n_precedes"] = edges["n_precedes"].astype("int64")
    edges["n_shared"] = edges["n_shared"].astype("int64")
    edges["weight"] = edges["n_precedes"] / edges["n_shared"]
    edges = edges.sort_values(["src", "dst"]).reset_index(drop=True)
    return edges[cols]


def seeding_by_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    min_papers: int = 1,
) -> pd.DataFrame:
    """Seeding scores at an arbitrary entity grain, skipping blank entities.

    A thin grain-agnostic wrapper over :func:`seeding_score` for grains other than
    ``"journal"`` — typically ``group_col="institution_canonical_id"`` or
    ``group_col="country_code"``. Rows whose ``group_col`` is missing **or blank**
    (``NaN``, ``None``, ``""``, or whitespace-only — e.g. the ~44% of institutions
    with no harvested country) are dropped before scoring and never imputed, so an
    absent entity cannot masquerade as a seeder.

    Parameters
    ----------
    df :
        Tidy paper frame carrying at least ``["topic_id", group_col, "year"]``.
    group_col :
        The entity grain column to score on.
    min_papers :
        Per-year publication threshold forwarded to :func:`seeding_score`. Default
        ``1``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[group_col, "seeding_score", "n_topics"]`` exactly as
        :func:`seeding_score`, computed only over rows with a non-blank
        ``group_col``. Sorted by ``seeding_score`` descending then ``group_col``.
        If every entity is blank, an empty frame with these columns is returned.

    Notes
    -----
    "Blank" is defined on the string representation: a value is dropped when it is
    NaN/None, or when ``str(value).strip()`` is empty. Numeric ROR ids therefore
    survive; only truly empty / whitespace tokens are excluded.
    """
    import pandas as pd

    if group_col not in df.columns:
        return pd.DataFrame({group_col: [], "seeding_score": [], "n_topics": []})

    work = df.copy()
    entity = work[group_col]
    # Blank iff NaN/None or the stripped string form is empty.
    not_null = entity.notna()
    stripped_nonempty = entity.astype("string").str.strip().fillna("") != ""
    keep = (not_null & stripped_nonempty).to_numpy()
    work = work.loc[keep]
    if work.empty:
        return pd.DataFrame({group_col: [], "seeding_score": [], "n_topics": []})

    return seeding_score(work, entity_col=group_col, min_papers=min_papers)


def _normalized_leads(first: pd.DataFrame, *, entity_col: str) -> pd.DataFrame:
    """Per-(topic, entity) normalized lead from a first-publication-year frame.

    Ranks entities within each topic by ``first_year`` ascending with ties taking
    the **minimum** rank, then maps rank ``r`` among ``m`` topic participants to
    ``1 - r / (m - 1)`` (``1.0`` when ``m == 1``).

    Parameters
    ----------
    first :
        Output of :func:`first_publication_year` — ``["topic_id", entity_col,
        "first_year"]``.
    entity_col :
        Entity grain column.

    Returns
    -------
    pandas.DataFrame
        ``first`` with an added float ``"normalized_lead"`` column in ``[0, 1]``.

    Notes
    -----
    ``rank(method="min")`` is 1-based, so it is shifted to a 0-based ``r`` before
    the formula. ``m`` is the per-topic participant count (``transform("size")``).
    """
    import numpy as np

    out = first.copy()
    grp = out.groupby("topic_id", dropna=False)["first_year"]
    # 1-based min-rank -> 0-based r.
    rank0 = grp.rank(method="min").astype("float64") - 1.0
    m = out.groupby("topic_id", dropna=False)["first_year"].transform("size").astype("float64")

    denom = m - 1.0
    # m == 1 -> lone entity in topic -> normalized_lead 1.0 (trivially first).
    normalized = np.where(denom > 0, 1.0 - rank0 / denom, 1.0)
    out["normalized_lead"] = normalized.astype("float64")
    return out
