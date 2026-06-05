"""Unit tests for the V1-S13 HGT model + :class:`HGTForecaster`.

Hermetic and fast: tiny synthetic snapshots are assembled with
:func:`scifield.forecasting.gnn.loader.build_snapshot_from_frames` (no DuckDB, no
real corpus, no network), and the forecaster is trained for a couple of epochs.
Mirrors the pure-fixture conventions of ``tests/test_forecasting_gnn_loader.py``.

The four invariants under test (see the HANDOFF NOTES in ``hgt.py``):
- forward/fit/predict produce finite, correctly-shaped, ``>= 0``-share outputs
  for BOTH ``conv_type="hgt"`` and ``conv_type="sage"``;
- ``HGTForecaster`` satisfies the runtime-checkable ``Predictor`` protocol;
- ``pos_weight`` is computed as ``n_neg/n_pos`` from the TRAIN labels;
- ``predict`` outputs track input ROWS, not row order (shuffle-invariance);
- the ``n_layers=0`` / ``ablate_edges=True`` ablation genuinely ignores the
  graph (edge/Paper perturbations leave predictions bit-identical).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from scifield.forecasting.baselines.base import ForecastPrediction, Predictor  # noqa: E402
from scifield.forecasting.data import NODE_FEATURES  # noqa: E402
from scifield.forecasting.gnn.hgt import HGTForecaster, HGTNet  # noqa: E402
from scifield.forecasting.gnn.loader import build_snapshot_from_frames  # noqa: E402

torch.set_num_threads(1)

#: Topic ids shared by every synthetic snapshot (stable across years, like the
#: real loader). Two of these (5, 8) get papers; 12 is never assigned but still
#: reserves a node every year.
_TOPIC_IDS = [5, 8, 12]


def _corpus_frames() -> dict[str, pd.DataFrame]:
    """Tiny full-corpus frames shaped like the canonical kuzu queries' output.

    5 papers across 2000..2004 with a backward-in-time citation chain (no L5
    artifacts), 2 authors, 2 institutions, 2 journals, and the 3 leaf topics in
    :data:`_TOPIC_IDS`. Reused to build several per-year snapshots.
    """
    papers = pd.DataFrame(
        {"pmid": ["p1", "p2", "p3", "p4", "p5"], "year": [2000, 2001, 2002, 2003, 2004]}
    )
    topics = pd.DataFrame({"topic_id": _TOPIC_IDS})
    cites = pd.DataFrame(
        {"from_pmid": ["p2", "p3", "p4", "p5"], "to_pmid": ["p1", "p1", "p2", "p4"]}
    )
    authored_by = pd.DataFrame(
        {
            "pmid": ["p1", "p2", "p3", "p4", "p5"],
            "author_canonical_id": ["a1", "a1", "a2", "a2", "a2"],
        }
    )
    affiliated_with = pd.DataFrame(
        {"author_canonical_id": ["a1", "a2"], "institution_canonical_id": ["i1", "i2"]}
    )
    published_in = pd.DataFrame(
        {"pmid": ["p1", "p2", "p3", "p4", "p5"], "journal_slug": ["j1", "j1", "j2", "j2", "j2"]}
    )
    assigned_to = pd.DataFrame(
        {"pmid": ["p1", "p2", "p3", "p4", "p5"], "topic_id": [5, 5, 8, 8, 8]}
    )
    return {
        "papers": papers,
        "topics": topics,
        "cites": cites,
        "authored_by": authored_by,
        "affiliated_with": affiliated_with,
        "published_in": published_in,
        "assigned_to": assigned_to,
    }


def _snapshots(years: tuple[int, ...]) -> dict[int, object]:
    """Build forward-edge-only snapshots for ``years`` from the synthetic corpus."""
    frames = _corpus_frames()
    out: dict[int, object] = {}
    for t in years:
        data, _ = build_snapshot_from_frames(t, **frames)
        out[t] = data
    return out


def _panel(years: tuple[int, ...], *, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an aligned ``(features, labels)`` panel over ``(topic_id, year)`` rows.

    One row per ``(topic_id in _TOPIC_IDS, origin_year in years)``. The 16
    NODE_FEATURES are filled with reproducible pseudo-random values; emergence
    labels are a fixed pattern (so a known n_pos/n_neg is available for the
    ``pos_weight`` test) and ``forward_share`` is a small positive value.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for gid in _TOPIC_IDS:
            row: dict[str, object] = {"topic_id": gid, "origin_year": year}
            for col in NODE_FEATURES:
                row[col] = float(rng.normal())
            rows.append(row)
    features = pd.DataFrame(rows)
    n = len(features)
    # Deterministic label pattern: ~1/3 positive.
    emergent = np.array([1 if (i % 3 == 0) else 0 for i in range(n)], dtype=np.int64)
    labels = pd.DataFrame(
        {
            "emergent": emergent,
            "forward_share": np.full(n, 0.05, dtype=np.float64),
        }
    )
    return features, labels


# --------------------------------------------------------------------------- #
# (1) Forward / fit / predict smoke for both conv types
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("conv_type", ["hgt", "sage"])
def test_fit_predict_smoke(conv_type: str) -> None:
    years = (2001, 2002, 2003)
    snaps = _snapshots(years)
    features, labels = _panel(years)

    fc = HGTForecaster(
        snaps, conv_type=conv_type, hidden=8, n_layers=2, heads=2, epochs=2, seed=1729
    )
    out = fc.fit(features, labels).predict(features)

    assert isinstance(out, ForecastPrediction)
    assert out.emergence_score.shape == (len(features),)
    assert out.share_forecast.shape == (len(features),)
    assert np.all(np.isfinite(out.emergence_score))
    assert np.all(np.isfinite(out.share_forecast))
    assert np.all(out.share_forecast >= 0.0)
    # emergence_score is a sigmoid -> in [0, 1].
    assert np.all((out.emergence_score >= 0.0) & (out.emergence_score <= 1.0))


def test_predict_before_fit_raises() -> None:
    snaps = _snapshots((2001,))
    features, _ = _panel((2001,))
    with pytest.raises(RuntimeError, match="before fit"):
        HGTForecaster(snaps, hidden=8, n_layers=1).predict(features)


# --------------------------------------------------------------------------- #
# (2) Predictor protocol
# --------------------------------------------------------------------------- #


def test_satisfies_predictor_protocol() -> None:
    snaps = _snapshots((2001,))
    fc = HGTForecaster(snaps, hidden=8, n_layers=1)
    assert isinstance(fc, Predictor)
    assert fc.name == "hgt"


# --------------------------------------------------------------------------- #
# (3) pos_weight computed from the TRAIN labels
# --------------------------------------------------------------------------- #


def test_pos_weight_from_train() -> None:
    years = (2001, 2002)
    snaps = _snapshots(years)
    features, labels = _panel(years)

    n_pos = int((labels["emergent"] == 1).sum())
    n_neg = int((labels["emergent"] == 0).sum())
    assert n_pos > 0 and n_neg > 0  # guard the fixture

    fc = HGTForecaster(snaps, hidden=8, n_layers=1, epochs=1, pos_weight="auto")
    fc.fit(features, labels)
    assert fc.pos_weight_ == pytest.approx(n_neg / n_pos)


def test_pos_weight_explicit_float() -> None:
    years = (2001,)
    snaps = _snapshots(years)
    features, labels = _panel(years)
    fc = HGTForecaster(snaps, hidden=8, n_layers=1, epochs=1, pos_weight=3.5)
    fc.fit(features, labels)
    assert fc.pos_weight_ == pytest.approx(3.5)


# --------------------------------------------------------------------------- #
# (4) Row-alignment under shuffle (the critical one)
# --------------------------------------------------------------------------- #


def test_predict_row_alignment_under_shuffle() -> None:
    years = (2001, 2002, 2003)
    snaps = _snapshots(years)
    features, labels = _panel(years)

    # Enough epochs that the per-row outputs are non-degenerate / distinguishable.
    fc = HGTForecaster(snaps, conv_type="sage", hidden=8, n_layers=2, epochs=50, seed=7)
    fc.fit(features, labels)

    pred_ordered = fc.predict(features)

    # Shuffle the SAME rows; predictions must follow the rows, not the position.
    perm = np.random.default_rng(123).permutation(len(features))
    shuffled = features.iloc[perm].reset_index(drop=True)
    pred_shuffled = fc.predict(shuffled)

    # Output i of the shuffled run must equal output perm[i] of the ordered run.
    assert np.allclose(pred_shuffled.emergence_score, pred_ordered.emergence_score[perm])
    assert np.allclose(pred_shuffled.share_forecast, pred_ordered.share_forecast[perm])

    # And the outputs are genuinely row-varying (not a constant the test couldn't
    # distinguish from a positional bug).
    assert np.unique(np.round(pred_ordered.emergence_score, 6)).size > 1


# --------------------------------------------------------------------------- #
# (5) Ablation truly ignores the graph
# --------------------------------------------------------------------------- #


def _frames_dense_cites() -> dict[str, pd.DataFrame]:
    """A SECOND corpus identical in topics but with very different edges/Papers.

    Same 3 leaf topics and the same per-(topic, year) feature panel will be used,
    but this corpus has more papers and a denser citation structure, so the full
    model would read a different graph. The ablation must NOT.
    """
    papers = pd.DataFrame(
        {
            "pmid": [f"q{i}" for i in range(1, 9)],
            "year": [2000, 2000, 2001, 2001, 2002, 2002, 2003, 2003],
        }
    )
    topics = pd.DataFrame({"topic_id": _TOPIC_IDS})
    # Dense backward-in-time citations.
    cites = pd.DataFrame(
        {
            "from_pmid": ["q3", "q4", "q5", "q6", "q7", "q8", "q5", "q7"],
            "to_pmid": ["q1", "q2", "q3", "q4", "q5", "q6", "q1", "q2"],
        }
    )
    authored_by = pd.DataFrame(
        {
            "pmid": [f"q{i}" for i in range(1, 9)],
            "author_canonical_id": ["a1", "a2", "a3", "a1", "a2", "a3", "a1", "a2"],
        }
    )
    affiliated_with = pd.DataFrame(
        {
            "author_canonical_id": ["a1", "a2", "a3"],
            "institution_canonical_id": ["i1", "i2", "i3"],
        }
    )
    published_in = pd.DataFrame(
        {
            "pmid": [f"q{i}" for i in range(1, 9)],
            "journal_slug": ["j1", "j2", "j3", "j1", "j2", "j3", "j1", "j2"],
        }
    )
    assigned_to = pd.DataFrame(
        {
            "pmid": [f"q{i}" for i in range(1, 9)],
            "topic_id": [5, 8, 5, 8, 5, 8, 12, 12],
        }
    )
    return {
        "papers": papers,
        "topics": topics,
        "cites": cites,
        "authored_by": authored_by,
        "affiliated_with": affiliated_with,
        "published_in": published_in,
        "assigned_to": assigned_to,
    }


@pytest.mark.parametrize("kwargs", [{"n_layers": 0}, {"n_layers": 2, "ablate_edges": True}])
def test_ablation_ignores_graph(kwargs: dict[str, Any]) -> None:
    years = (2001, 2002, 2003)
    features, labels = _panel(years)

    # Snapshot set A: the sparse synthetic corpus.
    snaps_a = _snapshots(years)
    # Snapshot set B: the dense corpus — identical Topic node set, totally
    # different edges / Paper nodes.
    frames_b = _frames_dense_cites()
    snaps_b = {t: build_snapshot_from_frames(t, **frames_b)[0] for t in years}

    common: dict[str, Any] = dict(conv_type="hgt", hidden=8, epochs=20, seed=2024, **kwargs)
    fc_a = HGTForecaster(snaps_a, **common)
    fc_b = HGTForecaster(snaps_b, **common)

    pred_a = fc_a.fit(features, labels).predict(features)
    pred_b = fc_b.fit(features, labels).predict(features)

    # Identical Topic.x + labels + seed, different graph -> identical outputs,
    # because the ablation reduces to an MLP over the 16 Topic features.
    assert np.allclose(pred_a.emergence_score, pred_b.emergence_score)
    assert np.allclose(pred_a.share_forecast, pred_b.share_forecast)


def test_ablation_architecture_is_mlp_over_topic() -> None:
    """The ablation builds ONLY a Topic input Linear + the two heads (no convs)."""
    snaps = _snapshots((2001,))
    fc = HGTForecaster(snaps, hidden=8, n_layers=0)
    net = fc._build_net()
    assert isinstance(net, HGTNet)
    assert net.ablate is True
    assert len(net.convs) == 0
    # Only the Topic input projection exists.
    assert list(net.input_proj.keys()) == ["Topic"]
    # Topic input Linear maps 16 -> hidden; heads map hidden -> 1.
    assert net.input_proj["Topic"].in_features == len(NODE_FEATURES)
    assert net.emergence_head.out_features == 1
    assert net.log_share_head.out_features == 1
