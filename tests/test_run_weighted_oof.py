"""단독 실행이 두 눈금을 함께 남기는지 본다. (#383)

CV 결과의 oof는 train 행 순서 그대로이므로(cv_seed_execution의 반환 규약), 재채점은
그 순서 위에서 id와 목표값을 맞춘다. 여기서 보는 것은 그 정렬과 metric 이름이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline.config import DataConfig
from pipeline.data import ID, TARGET
from pipeline.run import _weighted_oof_metrics

# 결측 패턴은 x 하나로 갈린다. train은 채움:빔이 1:1, test는 3:1이다.
TRAIN = pd.DataFrame(
    {
        ID: [10, 11, 12, 13, 14, 15, 16, 17],
        "x": [1.0, 1.0, 1.0, 1.0, None, None, None, None],
        TARGET: [0, 1, 0, 1, 1, 0, 1, 0],
    }
)
TEST = pd.DataFrame({ID: [20, 21, 22, 23], "x": [1.0, 1.0, 1.0, None]})
# 채운 쪽은 완벽히 맞히고 빈 쪽은 완전히 틀린 예측. 두 눈금이 확실히 갈린다.
PREDICTION = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])


def make_config(tmp_path) -> DataConfig:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    TRAIN.to_csv(train_path, index=False)
    TEST.to_csv(test_path, index=False)
    data = DataConfig(
        train=train_path,
        test=test_path,
        sample_submission=tmp_path / "sample_submission.csv",
        folds=tmp_path / "folds.parquet",
    )
    return type("Config", (), {"data": data})()


def test_weighted_oof_metrics_rescore_the_seed_mean_oof(tmp_path):
    cfg = make_config(tmp_path)
    oof = pd.DataFrame(
        {ID: TRAIN[ID], "fold": np.arange(len(TRAIN)) % 2, "pred": PREDICTION}
    )

    metrics = _weighted_oof_metrics(cfg, oof, TRAIN[TARGET])

    # 채운 패턴 w=(3/4)/(1/2)=1.5, 빈 패턴 w=(1/4)/(1/2)=0.5.
    weight = np.array([1.5] * 4 + [0.5] * 4)
    assert metrics["auc_oof_weighted"] == pytest.approx(
        roc_auc_score(TRAIN[TARGET].to_numpy(), PREDICTION, sample_weight=weight)
    )
    # 판정에 쓰는 auc_oof와 다른 값이어야 재가중이 실제로 걸린 것이다.
    assert metrics["auc_oof_weighted"] != pytest.approx(
        roc_auc_score(TRAIN[TARGET].to_numpy(), PREDICTION)
    )
    assert metrics["weighted_oof_effective_sample_size"] == pytest.approx(
        weight.sum() ** 2 / np.square(weight).sum()
    )
    assert metrics["weighted_oof_zero_weight_rows"] == 0.0
    assert metrics["weighted_oof_test_only_patterns"] == 0.0


def test_weighted_oof_metrics_follow_the_oof_row_order(tmp_path):
    """oof와 목표값이 어긋나면 값이 달라지도록, 순서를 섞은 판과 비교한다."""
    cfg = make_config(tmp_path)
    order = [3, 0, 5, 2, 7, 4, 1, 6]
    shuffled = TRAIN.iloc[order].reset_index(drop=True)
    oof = pd.DataFrame(
        {
            ID: shuffled[ID],
            "fold": np.arange(len(shuffled)) % 2,
            "pred": PREDICTION[order],
        }
    )

    metrics = _weighted_oof_metrics(cfg, oof, shuffled[TARGET])

    straight = _weighted_oof_metrics(
        cfg,
        pd.DataFrame(
            {ID: TRAIN[ID], "fold": np.arange(len(TRAIN)) % 2, "pred": PREDICTION}
        ),
        TRAIN[TARGET],
    )
    assert metrics["auc_oof_weighted"] == pytest.approx(straight["auc_oof_weighted"])
