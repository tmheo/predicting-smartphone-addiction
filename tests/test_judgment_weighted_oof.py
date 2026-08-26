"""가중 OOF 지표의 단위 테스트. (#383)

- 결측 패턴 가중치는 `P_test(패턴) / P_train(패턴)`이고, test에 없는 패턴은 0이다.
- 가중치가 전부 1이면 가중 OOF AUC는 기존 OOF AUC와 같다(추가 눈금 불변식).
- 유효 표본 수는 재채점한 행에서 다시 센다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline.data import ID, TARGET
from pipeline.judgment import (
    WEIGHTED_OOF_AUC_METRIC,
    JudgmentError,
    MissingnessReweighting,
    missingness_reweighting,
    weighted_oof_auc,
)

# train 8행, test 8행. 결측 패턴 a=(둘 다 채움), b=(x만 빔), c=(y만 빔), d=(둘 다 빔).
# train 구성비 a 1/2, b 1/4, c 1/4. test 구성비 a 1/4, b 1/2, d 1/4.
TRAIN_ROWS = [
    (0, 1.0, 1.0),  # a
    (1, 1.0, 1.0),  # a
    (2, 1.0, 1.0),  # a
    (3, 1.0, 1.0),  # a
    (4, None, 1.0),  # b
    (5, None, 1.0),  # b
    (6, 1.0, None),  # c
    (7, 1.0, None),  # c
]
TEST_ROWS = [
    (100, 1.0, 1.0),  # a
    (101, 1.0, 1.0),  # a
    (102, None, 1.0),  # b
    (103, None, 1.0),  # b
    (104, None, 1.0),  # b
    (105, None, 1.0),  # b
    (106, None, None),  # d: train에 없다.
    (107, None, None),  # d
]


def write_frames(tmp_path):
    train = pd.DataFrame(
        [
            {ID: i, "x": x, "y": y, TARGET: int(i % 2)}
            for i, x, y in TRAIN_ROWS
        ]
    )
    test = pd.DataFrame([{ID: i, "x": x, "y": y} for i, x, y in TEST_ROWS])
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def test_reweighting_moves_train_shares_onto_test_shares(tmp_path):
    train_path, test_path = write_frames(tmp_path)

    reweighting = missingness_reweighting(train_path, test_path)

    # a: (1/4)/(1/2)=0.5, b: (1/2)/(1/4)=2.0, c: 0/(1/4)=0.0.
    expected = pd.Series(
        [0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 0.0, 0.0],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7], name=ID),
        name="weight",
    )
    pd.testing.assert_series_equal(reweighting.weight, expected)
    assert reweighting.train_pattern_count == 3
    assert reweighting.test_pattern_count == 3
    assert reweighting.test_only_pattern_count == 1  # 패턴 d는 실을 train 행이 없다.
    assert reweighting.zero_weight_rows == 2  # 패턴 c는 test에 없다.
    # (Σw)²/Σw² = (4·0.5 + 2·2)² / (4·0.25 + 2·4) = 36/9 = 4.
    assert reweighting.effective_sample_size == pytest.approx(4.0)


def test_pattern_table_is_the_weight_provenance(tmp_path):
    train_path, test_path = write_frames(tmp_path)

    patterns = missingness_reweighting(train_path, test_path).patterns

    assert list(patterns.columns) == [
        "pattern",
        "missing_columns",
        "train_rows",
        "train_share",
        "test_share",
        "weight",
    ]
    # train에 있는 패턴만 담는다: a, b, c. 패턴 d는 실을 train 행이 없다.
    # "-"는 빈 칸이 없는 패턴이다(빈 문자열은 CSV 왕복에서 결측이 된다).
    assert list(patterns["missing_columns"]) == ["-", "x", "y"]
    assert list(patterns["train_rows"]) == [4, 2, 2]
    assert list(patterns["train_share"]) == [0.5, 0.25, 0.25]
    assert list(patterns["test_share"]) == [0.25, 0.5, 0.0]
    assert list(patterns["weight"]) == [0.5, 2.0, 0.0]


def test_reweighting_rejects_mismatched_feature_columns(tmp_path):
    train_path, test_path = write_frames(tmp_path)
    pd.read_csv(test_path).rename(columns={"y": "z"}).to_csv(test_path, index=False)

    with pytest.raises(JudgmentError, match="결측 패턴 대상 열"):
        missingness_reweighting(train_path, test_path)


def test_reweighting_cache_distinguishes_relative_paths_across_workspaces(
    tmp_path,
    monkeypatch,
):
    indexes = ([1, 2], [101, 102])
    roots = [tmp_path / "first", tmp_path / "second"]
    for root, ids in zip(roots, indexes, strict=True):
        root.mkdir()
        pd.DataFrame(
            {
                ID: ids,
                "x": [1.0, np.nan],
                TARGET: [0, 1],
            }
        ).to_csv(root / "train.csv", index=False)
        pd.DataFrame({ID: [ids[-1] + 1], "x": [1.0]}).to_csv(
            root / "test.csv",
            index=False,
        )

    observed = []
    for root in roots:
        monkeypatch.chdir(root)
        observed.append(
            list(missingness_reweighting(Path("train.csv"), Path("test.csv")).weight.index)
        )

    assert observed == [list(indexes[0]), list(indexes[1])]


def uniform_reweighting(index: pd.Index) -> MissingnessReweighting:
    return MissingnessReweighting(
        weight=pd.Series(np.ones(len(index)), index=index, name="weight"),
        train_pattern_count=1,
        test_pattern_count=1,
        test_only_pattern_count=0,
        zero_weight_rows=0,
    )


def test_all_ones_weight_reproduces_plain_oof_auc():
    rng = np.random.default_rng(0)
    index = pd.Index(np.arange(200), name=ID)
    y = pd.Series(rng.integers(0, 2, len(index)), index=index)
    prediction = pd.Series(
        rng.random(len(index)) + 0.3 * y.to_numpy(), index=index
    )

    weighted = weighted_oof_auc(prediction, y, uniform_reweighting(index))

    assert weighted.auc == pytest.approx(
        roc_auc_score(y.to_numpy(), prediction.to_numpy())
    )
    assert weighted.effective_sample_size == pytest.approx(len(index))
    assert weighted.effective_sample_fraction == pytest.approx(1.0)
    assert weighted.zero_weight_rows == 0


def test_zero_weight_rows_drop_out_of_the_score():
    index = pd.Index(np.arange(6), name=ID)
    y = pd.Series([0, 1, 0, 1, 1, 0], index=index)
    # 마지막 두 행은 예측을 뒤집어 놓았지만 가중치가 0이라 점수에 들어오면 안 된다.
    prediction = pd.Series([0.1, 0.9, 0.2, 0.8, 0.0, 1.0], index=index)
    reweighting = MissingnessReweighting(
        weight=pd.Series(
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0], index=index, name="weight"
        ),
        train_pattern_count=2,
        test_pattern_count=1,
        test_only_pattern_count=0,
        zero_weight_rows=2,
    )

    weighted = weighted_oof_auc(prediction, y, reweighting)

    assert weighted.auc == pytest.approx(1.0)
    assert weighted.zero_weight_rows == 2
    assert weighted.effective_sample_size == pytest.approx(4.0)
    assert weighted.effective_sample_fraction == pytest.approx(4.0 / 6.0)


def test_metrics_carry_the_auc_and_the_sample_provenance():
    index = pd.Index(np.arange(10), name=ID)
    y = pd.Series([0, 1] * 5, index=index)
    prediction = pd.Series(np.linspace(0, 1, 10), index=index)

    metrics = weighted_oof_auc(prediction, y, uniform_reweighting(index)).metrics()

    assert set(metrics) == {
        WEIGHTED_OOF_AUC_METRIC,
        "weighted_oof_effective_sample_size",
        "weighted_oof_effective_sample_fraction",
        "weighted_oof_zero_weight_rows",
        "weighted_oof_test_only_patterns",
    }
    assert all(isinstance(value, float) for value in metrics.values())


def test_missing_weight_for_an_id_is_a_judgment_error():
    index = pd.Index(np.arange(4), name=ID)
    y = pd.Series([0, 1, 0, 1], index=index)
    prediction = pd.Series([0.1, 0.9, 0.2, 0.8], index=index)
    short = uniform_reweighting(pd.Index([0, 1], name=ID))

    with pytest.raises(JudgmentError, match="가중치가 없는 id"):
        weighted_oof_auc(prediction, y, short)
