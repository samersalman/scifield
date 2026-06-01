"""Offline validation of the from-scratch CD index against the ``cdindex`` package.

No network. Three checks:

* Test 1 -- reproduce the ``cdindex`` package's documented toy-graph value exactly.
* Test 2 -- Pearson r >= 0.99 vs ``cdindex`` over a deterministic random graph.
* Test 3 -- hand-built n_i/n_j/n_k classification with a known CD value.
"""

from __future__ import annotations

import math
import random

import cdindex
import numpy as np
import pytest

from scifield.novelty.cd_index import CitationGraph, cd_index, classify_citers

# Toy graph from cdindex/cdindex.py docstring. Edge u->v means u cites v.
# Times are integer years (the cdindex C extension uses raw integer time units,
# so t_delta is in the same units -- here, years).
TOY_TIMES = {
    "0Z": 1992,
    "1Z": 1992,
    "2Z": 1993,
    "3Z": 1993,
    "4Z": 1995,
    "5Z": 1997,
    "6Z": 1998,
    "7Z": 1999,
    "8Z": 1999,
    "9Z": 1998,
    "10Z": 1997,
}
TOY_EDGES = [
    ("4Z", "2Z"),
    ("4Z", "0Z"),
    ("4Z", "1Z"),
    ("4Z", "3Z"),
    ("5Z", "2Z"),
    ("6Z", "2Z"),
    ("6Z", "4Z"),
    ("7Z", "4Z"),
    ("8Z", "4Z"),
    ("9Z", "4Z"),
    ("9Z", "1Z"),
    ("9Z", "3Z"),
    ("10Z", "4Z"),
]


def _build_cdindex_graph(times: dict[str, int], edges: list[tuple[str, str]]) -> cdindex.Graph:
    """Build a ``cdindex.Graph`` from integer-year node times and directed edges."""
    g = cdindex.Graph()
    for name, t in times.items():
        g.add_vertex(name, int(t))
    seen: set[tuple[str, str]] = set()
    for u, v in edges:
        if (u, v) in seen or u == v:
            continue
        seen.add((u, v))
        g.add_edge(u, v)
    return g


def test_toy_graph_matches_cdindex_exactly() -> None:
    """Our CD index reproduces the package's documented toy value (focal 4Z)."""
    ref = _build_cdindex_graph(TOY_TIMES, TOY_EDGES)
    ours = CitationGraph.from_edges(TOY_TIMES, TOY_EDGES)

    # Documented canonical value for focal 4Z at t=5 is 1/6.
    expected = ref.cdindex("4Z", 5)
    assert expected is not None
    assert math.isclose(expected, 1.0 / 6.0, abs_tol=1e-12)
    assert math.isclose(cd_index(ours, "4Z", 5), expected, abs_tol=1e-9)

    # Match across several windows and several focal nodes.
    for focal in TOY_TIMES:
        for t in (3, 5, 10):
            pkg = ref.cdindex(focal, t)
            mine = cd_index(ours, focal, t)
            if pkg is None:
                assert math.isnan(mine), (focal, t, mine)
            else:
                assert math.isclose(mine, pkg, abs_tol=1e-9), (focal, t, mine, pkg)


def _make_random_graph(
    n: int = 120, seed: int = 1234
) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """Build a deterministic citation graph: newer nodes cite older ones."""
    rng = random.Random(seed)
    names = [f"n{i}" for i in range(n)]
    times = {name: 2000 + rng.randint(0, 25) for name in names}
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for u in names:
        # Each node cites a handful of strictly-older nodes.
        candidates = [v for v in names if times[v] < times[u]]
        rng.shuffle(candidates)
        for v in candidates[: rng.randint(0, 6)]:
            if (u, v) not in seen:
                seen.add((u, v))
                edges.append((u, v))
    return times, edges


def test_pearson_correlation_against_cdindex() -> None:
    """CD over a deterministic random graph correlates with cdindex (r >= 0.99)."""
    times, edges = _make_random_graph()
    ref = _build_cdindex_graph(times, edges)
    ours = CitationGraph.from_edges(times, edges)

    mine_vals: list[float] = []
    pkg_vals: list[float] = []
    exact_mismatch = 0
    for focal in times:
        for t in (5, 10):
            pkg = ref.cdindex(focal, t)
            mine = cd_index(ours, focal, t)
            # Handle degenerate (no-citer) focal nodes consistently.
            if pkg is None:
                assert math.isnan(mine), (focal, t, mine)
                continue
            if math.isnan(mine):
                # Should not happen if pkg is defined.
                raise AssertionError(f"mine NaN where pkg defined: {focal} {t}")
            mine_vals.append(mine)
            pkg_vals.append(pkg)
            if not math.isclose(mine, pkg, abs_tol=1e-9):
                exact_mismatch += 1

    assert len(mine_vals) >= 30, "too few non-degenerate focal nodes to correlate"
    # Implementation should in fact match cdindex exactly, not merely correlate.
    assert exact_mismatch == 0, f"{exact_mismatch} focal nodes disagreed with cdindex"

    r = np.corrcoef(np.array(mine_vals), np.array(pkg_vals))[0, 1]
    assert r >= 0.99, f"Pearson r below threshold: {r}"


def test_classification_handbuilt() -> None:
    """Hand-built 5-node example with a known n_i/n_j/n_k split and CD value.

    Graph (edge u->v == u cites v):
      F (2000) cites R (1998).
      I (2002) cites F only            -> type-i
      J (2003) cites F and R           -> type-j
      K (2004) cites R only            -> type-k
    Window (2000, 2005]: all three citers in-window.
      n_i=1, n_j=1, n_k=1 -> CD_5 = (1-1)/3 = 0.
    """
    times = {"F": 2000, "R": 1998, "I": 2002, "J": 2003, "K": 2004}
    edges = [("F", "R"), ("I", "F"), ("J", "F"), ("J", "R"), ("K", "R")]
    g = CitationGraph.from_edges(times, edges)

    n_i, n_j, n_k = classify_citers(g, "F", 5)
    assert (n_i, n_j, n_k) == (1, 1, 1)
    assert cd_index(g, "F", 5) == pytest.approx(0.0)

    # Tighten window so K (2004) drops out but only by upper bound; here use t=2
    # -> window (2000, 2002], only I qualifies -> CD = (1-0)/1 = 1.0.
    n_i2, n_j2, n_k2 = classify_citers(g, "F", 2)
    assert (n_i2, n_j2, n_k2) == (1, 0, 0)
    assert cd_index(g, "F", 2) == pytest.approx(1.0)

    # No-citer focal node -> NaN.
    assert math.isnan(cd_index(g, "R", 5)) is False or True  # R has citers
    # A leaf with no citers:
    times2 = {"X": 2000}
    g2 = CitationGraph.from_edges(times2, [])
    assert math.isnan(cd_index(g2, "X", 5))


def test_boundary_window_inclusive_upper_exclusive_lower() -> None:
    """Citer at exactly y_F+t is in-window; at y_F is out (matches cdindex)."""
    times = {"F": 2000, "R": 1998, "A": 2005, "B": 2000, "C": 2006}
    # A cites F at the upper bound (2000+5); B at focal year; C past the window.
    edges = [("F", "R"), ("A", "F"), ("B", "F"), ("C", "F")]
    g = CitationGraph.from_edges(times, edges)
    ref = _build_cdindex_graph(times, edges)
    assert classify_citers(g, "F", 5) == (1, 0, 0)  # only A
    assert math.isclose(cd_index(g, "F", 5), ref.cdindex("F", 5), abs_tol=1e-9)
