"""Light smoke test for the V2 cartography **map builder** (``V2/scripts/build_map.py``).

This is intentionally a *structural* smoke test, not a render/golden test: it confirms the
new V2-S11 trajectory figure builders return Plotly figures and that ``build_html`` wires
the 5th "Trajectories" tab into the v2 page (and omits it for v1). It never writes a file
and never calls ``main()`` — the only side effect is reading the version parquets through
the ``mapio`` loaders, which the builder reaches via the ``SCIFIELD_DATA_VERSION`` env var
(set per-test with ``monkeypatch`` so nothing leaks between tests).

``build_map`` is a script under ``V2/scripts`` (not an installed package) and there is no
conftest, so the import shim below puts that directory on ``sys.path`` before importing it.
Each test skip-guards on the relevant version's data artifacts being present, so the suite
still runs green on a fresh checkout before the data builders have produced the parquets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

from scifield.cartography import mapio

# build_map lives under V2/scripts (a script dir, not an installed package); put it on the
# path so it can be imported. parents[1] of tests/test_build_map.py is the repo root.
_SCRIPTS = Path(__file__).resolve().parents[1] / "V2" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import build_map  # noqa: E402

# v2 trajectory artifacts (only materialised for v2) and the v1 role table (gates the v1
# regression — v1 has no trajectory layer, so its absence would skip the v1 build entirely).
_V2_TRAJ = (
    "V2/data_v2/trajectory/trajectory_summary.parquet",
    "V2/data_v2/trajectory/trajectory_series.parquet",
)
_V2_ROLES = "V2/data_v2/roles/role_scores.parquet"
_V1_ROLES = "V2/data/roles/role_scores.parquet"


def _has(*parts: str) -> bool:
    """True if a path under the inferred repo root exists (mirrors test_cartography_mapio)."""
    return mapio._DEFAULT_REPO_ROOT.joinpath(*parts).exists()


def _v2_data_present() -> bool:
    """The v2 build needs the v2 role table plus both trajectory parquets."""
    return _has(*_V2_ROLES.split("/")) and all(_has(*p.split("/")) for p in _V2_TRAJ)


# --------------------------------------------------------------------------- figure builders


def test_fig_trajectory_directions_returns_figure(monkeypatch) -> None:
    """The direction-count bar builder returns a Plotly Figure (reads v2 via the env var)."""
    if not _v2_data_present():
        pytest.skip("v2 trajectory artifacts not present")
    monkeypatch.setenv("SCIFIELD_DATA_VERSION", "v2")
    fig = build_map.fig_trajectory_directions()
    assert isinstance(fig, go.Figure)


def test_fig_trajectory_movers_returns_figure(monkeypatch) -> None:
    """The biggest-movers fan-chart builder returns a Plotly Figure."""
    if not _v2_data_present():
        pytest.skip("v2 trajectory artifacts not present")
    monkeypatch.setenv("SCIFIELD_DATA_VERSION", "v2")
    fig = build_map.fig_trajectory_movers()
    assert isinstance(fig, go.Figure)


def test_fig_trajectory_explorer_returns_figure_with_updatemenus(monkeypatch) -> None:
    """The all-topic explorer returns a Figure carrying the dropdown (updatemenus)."""
    if not _v2_data_present():
        pytest.skip("v2 trajectory artifacts not present")
    monkeypatch.setenv("SCIFIELD_DATA_VERSION", "v2")
    fig = build_map.fig_trajectory_explorer()
    assert isinstance(fig, go.Figure)
    assert fig.layout.updatemenus  # non-empty → the topic dropdown is wired in


# --------------------------------------------------------------------------- build_html wiring


def test_build_html_v2_includes_trajectory_tab(monkeypatch) -> None:
    """The v2 page carries the 5th trajectory tab + its markers and 5 tab buttons."""
    if not _v2_data_present():
        pytest.skip("v2 trajectory artifacts not present")
    monkeypatch.setenv("SCIFIELD_DATA_VERSION", "v2")
    html = build_map.build_html("v2")
    assert isinstance(html, str)
    assert "t-trajectory" in html
    assert "e. Trajectories" in html
    assert "updatemenus" in html
    # Five nav buttons (a..e). Count the button markup, not the bare token: the tab-switch
    # JS also references ``.tab-btn`` twice, so a raw ``count("tab-btn")`` would be 7.
    assert html.count('class="tab-btn') == 5


def test_build_html_v1_omits_trajectory_tab(monkeypatch) -> None:
    """The v1 (prove-phase) page has no trajectory layer → 4 tabs, no t-trajectory."""
    if not _has(*_V1_ROLES.split("/")):
        pytest.skip("v1 role_scores.parquet not present")
    monkeypatch.setenv("SCIFIELD_DATA_VERSION", "v1")
    html = build_map.build_html("v1")
    assert isinstance(html, str)
    assert "t-trajectory" not in html
    assert html.count('class="tab-btn') == 4
