"""Single source of truth for the cartography corpus journal slug sets (V2-S08).

Historically the canonical journal roster lived as three hand-maintained literals:
``flow.CANONICAL_JOURNAL_SLUGS`` (the v1 ten), ``cascade_validation.PANEL_JOURNALS``
(a byte-duplicate of the same ten), and ``thematic.dedup._GENERALIST_SLUGS`` (the
eight Tier-1 generalists). V2 grows the corpus to ~78 journals (``conf/corpus/v2.yaml``,
Tier-0 = the v1 ten kept verbatim), so a literal-per-module roster no longer scales.

This module reads the roster **from the active corpus config** so every consumer
agrees on one definition that extends automatically when a version's YAML changes:

* :func:`get_canonical_journal_slugs` — the journal slugs of one corpus version, in
  file order. ``"v1"`` returns exactly the historical ten (verified set-equal to
  ``flow.CANONICAL_JOURNAL_SLUGS``); ``"v2"`` returns the 78-journal roster.
* :func:`get_generalist_slugs` — the eight Tier-1 generalist slugs (the demotion set
  used by dedup). The tier is **not** encoded in the YAML schema, so this list is held
  here as the canonical definition (kept in sync with the Tier-1 block of
  ``conf/corpus/v2.yaml``).
* :func:`all_known_slugs` — the union across known versions; used as the permissive
  default for the flow guard so a single unchanged call site validates both a v1 and a
  v2 cartography run while still catching display-name leaks / bogus slugs.

Dependency discipline
---------------------
This module imports **only** the standard library plus ``OmegaConf`` (the repo's
config-loading idiom) — never its
``cartography``/``thematic`` siblings — so ``thematic.dedup`` (and anyone else) can
safely import *from* it without risking an import cycle.

Conventions: ``from __future__ import annotations``; numpy-style docstrings; stdlib +
OmegaConf only; ruff/black line-length 100.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

__all__ = [
    "DEFAULT_DATA_VERSION",
    "ENV_VAR",
    "all_known_slugs",
    "get_canonical_journal_slugs",
    "get_generalist_slugs",
    "resolve_data_version",
]

#: Frozen default corpus/data version (the byte-frozen v1 ship).
DEFAULT_DATA_VERSION = "v1"

#: Env var consulted when no explicit version is passed (mirrors
#: ``V2/scripts/_path_config.py``; duplicated, not imported — that module is not on
#: the importable library path).
ENV_VAR = "SCIFIELD_DATA_VERSION"

#: Repo root: ``src/scifield/cartography/corpus_config.py`` -> parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]

# The eight Tier-1 generalist slugs (``conf/corpus/v2.yaml`` Tier-1 block). The tier
# is not encoded in the YAML schema, so this is the canonical definition; the dedup
# generalist-demotion set is sourced from here.
_GENERALIST_SLUGS: frozenset[str] = frozenset(
    {
        "n_engl_j_med",
        "lancet",
        "jama",
        "bmj",
        "ann_intern_med",
        "nature",
        "science",
        "proc_natl_acad_sci_usa",
    }
)

#: Corpus versions whose rosters contribute to :func:`all_known_slugs`. v2 ⊇ v1
#: (Tier-0 kept), but the union is taken explicitly so the helper stays correct if
#: that ever changes.
_KNOWN_VERSIONS: tuple[str, ...] = ("v1", "v2")


def resolve_data_version(explicit: str | None = None) -> str:
    """Resolve the active data/corpus version.

    Mirrors ``V2/scripts/_path_config.resolve_data_version`` (duplicated rather than
    imported — ``V2/scripts`` is not on the importable library path).

    Parameters
    ----------
    explicit :
        A version passed explicitly (e.g. from ``--data-version``); takes priority.

    Returns
    -------
    str
        ``explicit`` if given, else ``$SCIFIELD_DATA_VERSION``, else
        :data:`DEFAULT_DATA_VERSION` (``"v1"``).
    """
    return explicit or os.environ.get(ENV_VAR) or DEFAULT_DATA_VERSION


@cache
def get_canonical_journal_slugs(version: str = "v1") -> tuple[str, ...]:
    """Canonical journal slugs for a corpus version, in YAML file order.

    Reads ``conf/corpus/{version}.yaml`` and returns the ``slug`` of each entry under
    ``journals``, preserving file order (so the order is stable and matches the YAML).

    Parameters
    ----------
    version :
        Corpus version key (``"v1"`` / ``"v2"``). Default ``"v1"`` — the frozen ten.

    Returns
    -------
    tuple of str
        The journal slugs in file order. ``get_canonical_journal_slugs("v1")`` is
        exactly the historical ten and set-equals ``flow.CANONICAL_JOURNAL_SLUGS``.

    Raises
    ------
    FileNotFoundError
        If ``conf/corpus/{version}.yaml`` does not exist.
    ValueError
        If the config has no ``journals`` list or an entry lacks a ``slug``.

    Notes
    -----
    Cached (:func:`functools.cache`) — the config is read once per version per
    process. Pure / read-only otherwise.
    """
    path = _REPO_ROOT / "conf" / "corpus" / f"{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"corpus config not found for version {version!r}: {path} "
            f"(expected conf/corpus/{version}.yaml)"
        )
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    cfg = cast(dict[str, Any], raw or {})

    journals = cfg.get("journals")
    if not journals:
        raise ValueError(f"corpus config {path} has no non-empty 'journals' list")

    slugs: list[str] = []
    for entry in journals:
        slug = entry.get("slug") if isinstance(entry, dict) else None
        if not slug:
            raise ValueError(
                f"corpus config {path} has a journal entry without a 'slug': {entry!r}"
            )
        slugs.append(str(slug))
    return tuple(slugs)


def get_generalist_slugs() -> frozenset[str]:
    """The eight Tier-1 generalist journal slugs (the dedup demotion set).

    Returns
    -------
    frozenset of str
        ``{n_engl_j_med, lancet, jama, bmj, ann_intern_med, nature, science,
        proc_natl_acad_sci_usa}`` — the Tier-1 block of ``conf/corpus/v2.yaml``. Held
        as a literal here because the tier is not encoded in the YAML schema.
    """
    return _GENERALIST_SLUGS


@cache
def all_known_slugs() -> frozenset[str]:
    """Union of canonical journal slugs across all known corpus versions.

    Since v2 ⊇ v1 (Tier-0 kept) this equals the v2 roster (78 slugs), but the union
    is taken explicitly over :data:`_KNOWN_VERSIONS` so it stays correct if a future
    version drops or renames a slug.

    Returns
    -------
    frozenset of str
        Every journal slug that appears in any known corpus version's config. Used as
        the permissive default membership for
        :func:`scifield.cartography.flow.assert_canonical_journals` so one unchanged
        call site validates both v1 and v2 runs.

    Notes
    -----
    Cached. Pure (delegates to :func:`get_canonical_journal_slugs`, itself cached).
    """
    union: set[str] = set()
    for version in _KNOWN_VERSIONS:
        union.update(get_canonical_journal_slugs(version))
    return frozenset(union)
