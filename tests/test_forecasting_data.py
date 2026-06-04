"""Unit tests for the pure data layer of V1-S12 forecasting (no real I/O).

These exercise the temporal feature/label math, the split mapping, and the seven
leakage assertions on tiny hand-built pandas frames — no real archetypes parquet,
no DuckDB. ``materialize`` is tested with its disk readers monkeypatched to
synthetic frames so the no-test-read / test-seal behaviour is verified without
ever touching the corpus (and a parallel pure test asserts the split-drop logic
directly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scifield.forecasting import data as fdata
from scifield.forecasting.data import (
    MLP_FEATURES,
    NODE_FEATURES,
    assert_no_leakage,
    assign_split,
    build_features,
    build_labels,
    check_share_sum,
    compute_yearly_topic_counts,
    materialize,
)

# A split config shaped like the OmegaConf one (plain dict — assign_split reads
# items as a fallback when attribute access fails).
SPLIT_CFG = {"train": [1998, 2017], "val": [2018, 2020], "test": [2021, 2022]}


# --------------------------------------------------------------------------- #
# Feature/label column contract
# --------------------------------------------------------------------------- #


def test_feature_tuples_match_contract() -> None:
    assert MLP_FEATURES == (
        "count_3yr",
        "share_3yr_mean",
        "share_last",
        "share_growth_3yr",
        "share_momentum",
        "share_accel",
        "share_volatility_3yr",
        "topic_age",
        "share_of_max",
    )
    # NODE = MLP (block A, first, in order) + 7 block-B extras.
    assert NODE_FEATURES[:9] == MLP_FEATURES
    assert NODE_FEATURES[9:] == (
        "sem_nov_mean_3yr",
        "cd5_3yr",
        "cd10_3yr",
        "cited_by_pctile_3yr",
        "rct_share_3yr",
        "review_share_3yr",
        "n_journals_3yr",
    )
    assert set(MLP_FEATURES) <= set(NODE_FEATURES)


# --------------------------------------------------------------------------- #
# compute_yearly_topic_counts — leaf-only denominator, shares sum to 1
# --------------------------------------------------------------------------- #


def _two_topic_papers() -> pd.DataFrame:
    # Topic 0 and 1 each one paper/year 2000-2002; two NOISE papers must NOT
    # enter the denominator.
    return pd.DataFrame(
        {
            "topic_id": [0, 0, 0, 1, 1, 1, -1, -1],
            "year": [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2002],
        }
    )


def test_counts_leaf_only_denominator_and_shares_sum_to_one() -> None:
    counts = compute_yearly_topic_counts(_two_topic_papers(), noise_topic_id=-1)
    assert set(counts.columns) == {"topic_id", "year", "n", "N", "share"}
    # Noise excluded: N(2000) counts only the two leaf papers, not the noise one.
    n2000 = counts.loc[counts["year"] == 2000, "N"].unique()
    assert list(n2000) == [2]
    # Every year's leaf shares sum to 1.
    per_year = counts.groupby("year")["share"].sum()
    assert np.allclose(per_year.to_numpy(), 1.0)
    # No noise topic in the output.
    assert -1 not in set(counts["topic_id"])
    # check_share_sum agrees and does not raise.
    check_share_sum(counts)


def test_counts_empty_input_returns_typed_empty() -> None:
    empty = pd.DataFrame({"topic_id": [-1, -1], "year": [2000, 2001]})  # all noise
    counts = compute_yearly_topic_counts(empty, noise_topic_id=-1)
    assert list(counts.columns) == ["topic_id", "year", "n", "N", "share"]
    assert counts.empty


# --------------------------------------------------------------------------- #
# build_features — trailing-only, NODE schema, NaN-skip
# --------------------------------------------------------------------------- #


def _counts_for_features() -> pd.DataFrame:
    # Single topic, growing share, plus a constant filler so shares sum to 1.
    rows = []
    for y in range(2000, 2005):
        share = 0.2
        rows.append({"topic_id": 0, "year": y, "n": 2, "N": 10, "share": share})
        rows.append({"topic_id": 1, "year": y, "n": 8, "N": 10, "share": 0.8})
    return pd.DataFrame(rows)


def test_build_features_schema_and_keys() -> None:
    counts = _counts_for_features()
    nov = pd.DataFrame(
        {
            "pmid": range(5),
            "topic_id": [0] * 5,
            "year": list(range(2000, 2005)),
            "sem_nov_mean": [0.1, np.nan, 0.3, 0.4, 0.5],
            "cd5": [0.0] * 5,
            "cd10": [0.0] * 5,
            "cited_by_pctile_within_year": [0.5] * 5,
        }
    )
    epi = pd.DataFrame(
        {
            "pmid": range(5),
            "topic_id": [0] * 5,
            "year": list(range(2000, 2005)),
            "is_rct": [True, False, False, False, False],
            "is_review": [False] * 5,
            "journal": ["A", "B", "A", "C", "A"],
        }
    )
    feats = build_features(counts, nov, epi, trailing_window=3)
    expected_cols = ["topic_id", "origin_year", *NODE_FEATURES, "trailing_3yr_volume"]
    assert list(feats.columns) == expected_cols
    # One row per (topic, observed year).
    assert len(feats) == len(counts)


def test_build_features_trailing_window_and_nan_skip() -> None:
    counts = _counts_for_features()
    nov = pd.DataFrame(
        {
            "pmid": range(5),
            "topic_id": [0] * 5,
            "year": list(range(2000, 2005)),
            # 2001 is NaN; the [1999..2001] window should average only 0.1 (2000).
            "sem_nov_mean": [0.1, np.nan, 0.3, 0.4, 0.5],
            "cd5": [0.0] * 5,
            "cd10": [0.0] * 5,
            "cited_by_pctile_within_year": [0.5] * 5,
        }
    )
    epi = pd.DataFrame(
        {
            "pmid": range(5),
            "topic_id": [0] * 5,
            "year": list(range(2000, 2005)),
            "is_rct": [True, True, False, False, False],
            "is_review": [False] * 5,
            "journal": ["A", "A", "B", "C", "A"],
        }
    )
    feats_all = build_features(counts, nov, epi, trailing_window=3)
    feats = feats_all[feats_all["topic_id"] == 0].set_index("origin_year")

    # NaN-skip: origin 2001 window {2000,2001} has values {0.1, NaN} -> mean 0.1.
    assert feats.loc[2001, "sem_nov_mean_3yr"] == pytest.approx(0.1)
    # count_3yr at 2002 = sum n over [2000,2001,2002] = 6; trailing volume mirrors it.
    assert feats.loc[2002, "count_3yr"] == pytest.approx(6.0)
    assert feats.loc[2002, "trailing_3yr_volume"] == pytest.approx(6.0)
    # rct_share over [2000,2001,2002] papers = 2 RCT of 3 = 2/3.
    assert feats.loc[2002, "rct_share_3yr"] == pytest.approx(2.0 / 3.0)
    # n_journals over [2000,2001,2002] = {A, B} = 2.
    assert feats.loc[2002, "n_journals_3yr"] == pytest.approx(2.0)
    # topic_age = origin - first active year (2000).
    assert feats.loc[2004, "topic_age"] == pytest.approx(4.0)
    # share_last is the share at t (0.2 by construction).
    assert feats.loc[2003, "share_last"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# build_labels — closed-form emergent / non-emergent, guards
# --------------------------------------------------------------------------- #


def _counts_known_emergence() -> pd.DataFrame:
    # Topic 0: share 0.2 through 2002 then 0.4 from 2003 (a clean 2x jump).
    # n = 20/year so trailing 3yr volume = 60 (>= v_min 30).
    rows = []
    for y in range(2000, 2006):
        share = 0.2 if y <= 2002 else 0.4
        n = 20
        rows.append({"topic_id": 0, "year": y, "n": n, "N": int(round(n / share)), "share": share})
    return pd.DataFrame(rows)


def test_build_labels_emergent_closed_form() -> None:
    counts = _counts_known_emergence()
    labels = build_labels(counts, horizon=3, gamma=1.5, v_min=30, delta=0.002).set_index(
        "origin_year"
    )

    # Origin 2002: trailing {2000,2001,2002}=0.2 mean; forward {2003,2004,2005}=0.4 mean.
    assert labels.loc[2002, "trailing_3yr_mean_share"] == pytest.approx(0.2)
    assert labels.loc[2002, "forward_3yr_mean_share"] == pytest.approx(0.4)
    # forward_share is the MAPE target == forward_3yr_mean_share.
    assert labels.loc[2002, "forward_share"] == pytest.approx(0.4)
    assert labels.loc[2002, "trailing_3yr_volume"] == pytest.approx(60.0)
    assert bool(labels.loc[2002, "volume_ok"]) is True
    assert bool(labels.loc[2002, "label_complete"]) is True
    # 0.4 >= 1.5 * 0.2 (= 0.3) -> emergent.
    assert int(labels.loc[2002, "emergent"]) == 1
    # Sensitivity variants present and sane: additive jump 0.4-0.2=0.2 >= delta.
    assert int(labels.loc[2002, "emergent_additive"]) == 1


def test_build_labels_non_emergent_and_volume_guard() -> None:
    # Flat topic with tiny volume: not emergent, fails volume_ok.
    rows = []
    for y in range(2000, 2006):
        rows.append({"topic_id": 0, "year": y, "n": 3, "N": 10, "share": 0.3})
    counts = pd.DataFrame(rows)
    labels = build_labels(counts, horizon=3, gamma=1.5, v_min=30).set_index("origin_year")
    # Flat share: forward == trailing -> not >= 1.5x -> emergent 0.
    assert int(labels.loc[2002, "emergent"]) == 0
    # trailing volume = 9 < 30 -> volume_ok False (recorded, not dropped).
    assert bool(labels.loc[2002, "volume_ok"]) is False
    # Row still present in the frame.
    assert 2002 in labels.index


def test_build_labels_label_complete_excludes_past_corpus_max() -> None:
    counts = _counts_known_emergence()  # max year 2005
    labels = build_labels(counts, horizon=3, gamma=1.5, v_min=30).set_index("origin_year")
    # 2002 + 3 = 2005 <= 2005 -> complete.
    assert bool(labels.loc[2002, "label_complete"]) is True
    # 2003 + 3 = 2006 > 2005 -> incomplete (forward window runs off the corpus).
    assert bool(labels.loc[2003, "label_complete"]) is False
    assert bool(labels.loc[2005, "label_complete"]) is False


def test_build_labels_columns_locked() -> None:
    labels = build_labels(_counts_known_emergence())
    assert list(labels.columns) == [
        "topic_id",
        "origin_year",
        "trailing_3yr_mean_share",
        "forward_3yr_mean_share",
        "forward_share",
        "trailing_3yr_volume",
        "volume_ok",
        "label_complete",
        "emergent",
        "emergent_additive",
        "emergent_count_surge",
    ]


# --------------------------------------------------------------------------- #
# assign_split — boundary mapping (scalar and vector)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2017, "train"),
        (2018, "val"),
        (2020, "val"),
        (2021, "test"),
        (1990, "excluded"),
        (2030, "excluded"),
        (1998, "train"),  # lower train boundary inclusive
        (2022, "test"),  # upper test boundary inclusive
    ],
)
def test_assign_split_scalar_boundaries(year: int, expected: str) -> None:
    assert assign_split(year, SPLIT_CFG) == expected


def test_assign_split_vector_returns_aligned_series() -> None:
    s = assign_split(pd.Series([2017, 2018, 2021, 1990], index=[10, 11, 12, 13]), SPLIT_CFG)
    assert isinstance(s, pd.Series)
    assert list(s) == ["train", "val", "test", "excluded"]
    # Index preserved.
    assert list(s.index) == [10, 11, 12, 13]


# --------------------------------------------------------------------------- #
# assert_no_leakage — the seven keyword assertions
# --------------------------------------------------------------------------- #


def _mk_features(keys: list[tuple[int, int]]) -> pd.DataFrame:
    d: dict[str, list] = {
        "topic_id": [k[0] for k in keys],
        "origin_year": [k[1] for k in keys],
    }
    for c in NODE_FEATURES:
        d[c] = [0.0] * len(keys)
    return pd.DataFrame(d)


def _mk_labels(
    keys: list[tuple[int, int]],
    volume_ok: list[bool] | None = None,
    label_complete: list[bool] | None = None,
) -> pd.DataFrame:
    n = len(keys)
    return pd.DataFrame(
        {
            "topic_id": [k[0] for k in keys],
            "origin_year": [k[1] for k in keys],
            "volume_ok": volume_ok if volume_ok is not None else [True] * n,
            "label_complete": label_complete if label_complete is not None else [True] * n,
            "forward_share": [0.1] * n,
            "trailing_3yr_mean_share": [0.1] * n,
            "forward_3yr_mean_share": [0.1] * n,
        }
    )


def test_assert_no_leakage_clean_frame_passes() -> None:
    keys = [(0, 2017), (0, 2018)]
    assert_no_leakage(
        _mk_features(keys),
        _mk_labels(keys),
        pd.Series(["train", "val"]),
        horizon=3,
        allowed_origins={2017, 2018},
    )


def test_leakage_allowed_origins() -> None:
    with pytest.raises(AssertionError, match="leakage:allowed_origins"):
        assert_no_leakage(
            _mk_features([(0, 2099)]),
            _mk_labels([(0, 2099)]),
            pd.Series(["excluded"]),
            horizon=3,
            allowed_origins={2017, 2018},
        )


def test_leakage_test_present_rejects_test_rows() -> None:
    with pytest.raises(AssertionError, match="leakage:test_present"):
        assert_no_leakage(
            _mk_features([(0, 2021)]),
            _mk_labels([(0, 2021)]),
            pd.Series(["test"]),
            horizon=3,
            allowed_origins={2021},
        )


def test_leakage_test_present_rejects_unknown_label() -> None:
    with pytest.raises(AssertionError, match="leakage:test_present"):
        assert_no_leakage(
            _mk_features([(0, 2017)]),
            _mk_labels([(0, 2017)]),
            pd.Series(["bogus"]),
            horizon=3,
            allowed_origins={2017},
        )


def test_leakage_disjoint_same_year_two_splits() -> None:
    # Same origin year labeled train AND val -> not disjoint.
    keys = [(0, 2017), (1, 2017)]
    with pytest.raises(AssertionError, match="leakage:disjoint"):
        assert_no_leakage(
            _mk_features(keys),
            _mk_labels(keys),
            pd.Series(["train", "val"]),
            horizon=3,
            allowed_origins={2017},
        )


def test_leakage_boundary_train_after_val() -> None:
    # A train origin (2019) lands after a val origin (2018) -> ordering broken.
    keys = [(0, 2019), (1, 2018)]
    with pytest.raises(AssertionError, match="leakage:boundary"):
        assert_no_leakage(
            _mk_features(keys),
            _mk_labels(keys),
            pd.Series(["train", "val"]),
            horizon=3,
            allowed_origins={2018, 2019},
        )


def test_leakage_label_complete_volume_ok_but_incomplete() -> None:
    keys = [(0, 2017)]
    with pytest.raises(AssertionError, match="leakage:label_complete"):
        assert_no_leakage(
            _mk_features(keys),
            _mk_labels(keys, volume_ok=[True], label_complete=[False]),
            pd.Series(["train"]),
            horizon=3,
            allowed_origins={2017},
        )


def test_leakage_key_align_mismatched_keys() -> None:
    with pytest.raises(AssertionError, match="leakage:key_align"):
        assert_no_leakage(
            _mk_features([(0, 2017)]),
            _mk_labels([(0, 2018)]),
            pd.Series(["train"]),
            horizon=3,
            allowed_origins={2017, 2018},
        )


def test_leakage_share_sum_via_assert_no_leakage() -> None:
    # When labels carry a per-year `share`, assert_no_leakage checks check #7.
    keys = [(0, 2017), (1, 2017)]
    labels = _mk_labels(keys)
    labels["share"] = [0.3, 0.3]  # sums to 0.6 per year -> trips
    with pytest.raises(AssertionError, match="leakage:share_sum"):
        assert_no_leakage(
            _mk_features(keys),
            labels,
            pd.Series(["train", "train"]),
            horizon=3,
            allowed_origins={2017},
        )


def test_check_share_sum_trips_on_bad_counts() -> None:
    bad = pd.DataFrame({"topic_id": [0], "year": [2000], "n": [1], "N": [2], "share": [0.5]})
    with pytest.raises(AssertionError, match="leakage:share_sum"):
        check_share_sum(bad)


# --------------------------------------------------------------------------- #
# materialize — no real I/O: monkeypatch the disk readers, verify test-seal
# --------------------------------------------------------------------------- #


class _Cfg:
    """Tiny attribute-access config mimicking the OmegaConf DictConfig fields."""

    def __init__(self) -> None:
        self.input = _NS(
            archetypes_parquet="UNUSED.parquet",
            topics_parquet="UNUSED.parquet",
            duckdb_path="UNUSED.duckdb",
        )
        self.output = _NS(features_parquet="o.parquet", metrics_parquet="m.parquet")
        self.split = _NS(
            train=[2000, 2010],
            val=[2011, 2013],
            test=[2014, 2015],
            horizon=3,
            trailing_window=3,
        )
        self.label = _NS(gamma=1.5, v_min=2, mode="multiplicative", delta=0.002)
        self.noise = _NS(denominator="leaf_only", noise_topic_id=-1)


class _NS:
    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)

    def __getitem__(self, key: str) -> object:  # assign_split's item fallback
        return self.__dict__[key]


def _synthetic_papers() -> pd.DataFrame:
    # Two leaf topics + noise across 2000..2016 so every split range has rows
    # (incl. the sealed test years 2014-2015, which must NOT survive).
    rows: list[dict[str, object]] = []
    for y in range(2000, 2017):
        for tid in (0, 1):
            for _ in range(5):  # 5 papers/topic/year so volume_ok is easy
                rows.append(
                    {
                        "pmid": len(rows),
                        "year": y,
                        "journal": "J",
                        "topic_id": tid,
                        "sem_nov_mean": 0.5,
                        "cd5": 0.1,
                        "cd10": 0.1,
                        "cited_by_pctile_within_year": 0.5,
                    }
                )
        rows.append(
            {
                "pmid": len(rows),
                "year": y,
                "journal": "J",
                "topic_id": -1,  # noise
                "sem_nov_mean": np.nan,
                "cd5": np.nan,
                "cd10": np.nan,
                "cited_by_pctile_within_year": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _patch_readers(monkeypatch: pytest.MonkeyPatch, papers: pd.DataFrame) -> None:
    def fake_read_parquet(path: object, columns: list[str] | None = None) -> pd.DataFrame:
        return papers[columns].copy() if columns else papers.copy()

    def fake_pub_types(_path: object) -> pd.DataFrame:
        pmids = papers["pmid"].astype(np.int64)
        return pd.DataFrame(
            {
                "pmid": pmids,
                "is_rct": [False] * len(pmids),
                "is_review": [False] * len(pmids),
            }
        )

    monkeypatch.setattr(fdata.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(fdata, "_load_publication_types", fake_pub_types)


def test_materialize_seals_test_rows_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_readers(monkeypatch, _synthetic_papers())
    df, info = materialize(_Cfg(), allow_test=False)

    # No test rows present, no excluded rows present.
    assert set(df["split"]) <= {"train", "val"}
    assert "test" not in set(df["split"])
    # Test origin years (2014, 2015) must be entirely absent.
    assert df.loc[df["origin_year"].isin([2014, 2015])].empty

    # Schema: NODE_FEATURES + label cols + split.
    for c in NODE_FEATURES:
        assert c in df.columns
    for c in ("emergent", "volume_ok", "label_complete", "forward_share", "split"):
        assert c in df.columns

    # info diagnostics for the notebook / class-balance.
    assert info["denominator"] == "leaf_only"
    assert info["n_noise_total"] == 17  # one noise paper per year, 2000..2016
    assert info["noise_frac"] == pytest.approx(17 / len(_synthetic_papers()))
    assert set(info["per_split"]) == {"train", "val", "test"}
    assert info["per_split"]["test"]["n"] == 0  # sealed
    assert info["thresholds"]["gamma"] == 1.5
    assert info["thresholds"]["v_min"] == 2
    assert "n_excluded_volume" in info


def test_materialize_allow_test_includes_test_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    # Guardrail check: when explicitly allowed, test rows DO appear (S12 never
    # passes this, but the flag must work).
    _patch_readers(monkeypatch, _synthetic_papers())
    df, _ = materialize(_Cfg(), allow_test=True)
    # With allow_test the seal is lifted: test-origin rows (2014/2015) appear
    # (they are flagged label_complete=False since 2014+3 > corpus max 2016, but
    # are NOT dropped). Excluded years (2016) are still dropped.
    assert set(df["split"]) <= {"train", "val", "test"}
    assert "test" in set(df["split"])
    assert not df.loc[df["origin_year"] == 2014].empty


def test_merge_features_labels_single_trailing_volume_no_suffixes() -> None:
    # trailing_3yr_volume is a LOCKED output of BOTH build_features and
    # build_labels; the materialize merge must dedupe it to ONE clean column with
    # no _x/_y suffixes (the §4 output contract). Build tiny real feature/label
    # frames and join them through the same helper materialize uses.
    counts = _counts_known_emergence()
    nov = pd.DataFrame(
        {
            "pmid": range(len(counts)),
            "topic_id": [0] * len(counts),
            "year": list(counts["year"]),
            "sem_nov_mean": [0.5] * len(counts),
            "cd5": [0.1] * len(counts),
            "cd10": [0.1] * len(counts),
            "cited_by_pctile_within_year": [0.5] * len(counts),
        }
    )
    epi = pd.DataFrame(
        {
            "pmid": range(len(counts)),
            "topic_id": [0] * len(counts),
            "year": list(counts["year"]),
            "is_rct": [False] * len(counts),
            "is_review": [False] * len(counts),
            "journal": ["J"] * len(counts),
        }
    )
    features = build_features(counts, nov, epi, trailing_window=3)
    labels = build_labels(counts, horizon=3, gamma=1.5, v_min=30, delta=0.002)
    # Both sides really do carry the colliding column.
    assert "trailing_3yr_volume" in features.columns
    assert "trailing_3yr_volume" in labels.columns

    df = fdata._merge_features_labels(features, labels)

    # Exactly one clean trailing_3yr_volume, and NO suffixed survivors.
    assert "trailing_3yr_volume" in df.columns
    assert "trailing_3yr_volume_x" not in df.columns
    assert "trailing_3yr_volume_y" not in df.columns
    assert list(df.columns).count("trailing_3yr_volume") == 1
    # Canonical value is the LABELS-side [t-2,t] volume (the v_min guard input);
    # origin 2002 trailing volume = 3 * 20 = 60.
    assert df.set_index("origin_year").loc[2002, "trailing_3yr_volume"] == pytest.approx(60.0)


def test_materialize_split_drop_logic_pure() -> None:
    # Pure mirror of materialize's seal step, no readers at all: build the joined
    # frame by hand and confirm dropping non-{train,val} rows removes test.
    keys = [(0, 2010), (0, 2012), (0, 2014)]  # train, val, test under _Cfg ranges
    df = _mk_features(keys)
    split_col = assign_split(df["origin_year"], _Cfg().split)
    assert isinstance(split_col, pd.Series)
    df["split"] = split_col.to_numpy()
    kept = df[df["split"].isin({"train", "val"})].reset_index(drop=True)
    assert list(kept["origin_year"]) == [2010, 2012]
    assert "test" not in set(kept["split"])
