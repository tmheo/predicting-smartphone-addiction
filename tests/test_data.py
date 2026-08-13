"""data.labels: train 라벨을 요청한 id 순서로 정렬해 돌려주는 공개 함수. (#pool에서 승격)"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.data import labels


@pytest.fixture
def train_path(tmp_path):
    train = pd.DataFrame({"id": [1, 2, 3], "addicted_label": [0, 1, 0], "age": [20, 30, 40]})
    path = tmp_path / "train.csv"
    train.to_csv(path, index=False)
    return path


def test_labels_follow_requested_id_order(train_path):
    y = labels(pd.Index([3, 1], name="id"), train_path=train_path)
    assert list(y.index) == [3, 1]
    assert list(y) == [0, 0]


def test_labels_reject_unknown_id(train_path):
    with pytest.raises(AssertionError):
        labels(pd.Index([1, 99], name="id"), train_path=train_path)
