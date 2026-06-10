"""Single source of truth for data-version path resolution (V2-S08).

Resolution order for the data version: explicit arg > ``$SCIFIELD_DATA_VERSION`` >
``"v1"``. ``v1`` is the frozen default and keeps the historical UNVERSIONED output
dir (``V2/data``) so every v1 artifact stays byte-identical to the V2 phase-1 ship;
any other version writes to ``V2/data_{version}`` and reads from ``data/{version}``.

The build scripts under ``V2/scripts/`` each grew their own ``REPO_ROOT`` +
hardcoded path block; this helper collapses that into one place. Library code that
must not depend on ``V2/scripts`` (e.g. ``scifield.cartography.mapio``) duplicates the
three-line ``out_root`` / ``data_dir`` logic with a comment cross-referencing here
rather than importing from this module.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_VERSION = "v1"

#: Env var consulted when no explicit version is passed.
ENV_VAR = "SCIFIELD_DATA_VERSION"


def resolve_data_version(explicit: str | None = None) -> str:
    """Resolve the active data version.

    Parameters
    ----------
    explicit :
        A version passed on the command line (``--data-version``); takes priority.

    Returns
    -------
    str
        ``explicit`` if given, else ``$SCIFIELD_DATA_VERSION``, else
        :data:`DEFAULT_DATA_VERSION` (``"v1"``).
    """
    return explicit or os.environ.get(ENV_VAR) or DEFAULT_DATA_VERSION


def data_dir(repo_root: Path, version: str) -> Path:
    """Input data directory for a version (``<repo>/data/<version>``)."""
    return repo_root / f"data/{version}"


def out_root(repo_root: Path, version: str) -> Path:
    """Cartography output root for a version.

    ``v1`` keeps its historical UNVERSIONED location (``V2/data``) to stay
    byte-frozen; any other version is namespaced as ``V2/data_<version>``.
    """
    return repo_root / ("V2/data" if version == "v1" else f"V2/data_{version}")


def get_paths(repo_root: Path, version: str) -> dict[str, Path]:
    """All cartography input/output paths for a version.

    Parameters
    ----------
    repo_root :
        Repository root (``Path(__file__).resolve().parents[2]`` in the builders).
    version :
        The resolved data version (see :func:`resolve_data_version`).

    Returns
    -------
    dict of str to pathlib.Path
        Keyed by the constant names the build scripts use. ``ARCHETYPES`` /
        ``PAPER_INST`` / ``INSTITUTIONS`` / ``REFERENCES_OUT`` are derived from the
        version's ``data/<version>`` dir (for v2 the archetypes master is rebuilt
        before the cartography re-run, per the V2-S08 plan).
    """
    d, o = data_dir(repo_root, version), out_root(repo_root, version)
    return {
        "DATA_DIR": d,
        "OUT_ROOT": o,
        "ARCHETYPES": d / "archetypes.parquet",
        "PAPERS_DUCKDB": d / "papers.duckdb",
        "TOPIC_HIERARCHY": d / "topic_hierarchy.parquet",
        "KUZU_GRAPH": d / "kuzu_graph",
        "CITED_BY": d / "enrichment/cited_by.parquet",
        "PAPER_INST": d / "enrichment/paper_institutions.parquet",
        "INSTITUTIONS": d / "enrichment/institutions.parquet",
        "REFERENCES_OUT": d / "enrichment/references_out.parquet",
        "FLOW_DIR": o / "flow",
        "CASCADE_DIR": o / "cascade",
        "ROLES_DIR": o / "roles",
        "ORIGINS_DIR": o / "origins",
    }
