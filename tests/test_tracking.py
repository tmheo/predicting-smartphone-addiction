"""MLflow 실행 저장소 초기화의 동시 접근 회귀 검사와 metric 키 상수의 규약 일치. (#549)"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from pipeline import tracking


def test_fresh_sqlite_initialization_is_serialized(tmp_path, monkeypatch) -> None:
    active = 0
    maximum_active = 0
    experiment_id: str | None = None
    state_lock = threading.Lock()
    start = threading.Barrier(4)

    class FakeClient:
        def __init__(self, *, tracking_uri: str) -> None:
            nonlocal active, maximum_active
            assert tracking_uri.endswith("fresh.db")
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1

        def get_experiment_by_name(self, name: str):
            assert name == tracking.EXPERIMENT_NAME
            with state_lock:
                return (
                    None
                    if experiment_id is None
                    else SimpleNamespace(experiment_id=experiment_id)
                )

        def create_experiment(self, name: str) -> str:
            nonlocal experiment_id
            assert name == tracking.EXPERIMENT_NAME
            with state_lock:
                assert experiment_id is None
                experiment_id = "1"
                return experiment_id

    monkeypatch.setattr("mlflow.tracking.MlflowClient", FakeClient)
    uri = f"sqlite:///{tmp_path / 'fresh.db'}"

    def initialize() -> str:
        start.wait()
        _, found = tracking.mlflow_client(uri)
        return found

    with ThreadPoolExecutor(max_workers=4) as executor:
        found = list(executor.map(lambda _: initialize(), range(4)))

    assert found == ["1"] * 4
    assert maximum_active == 1
    assert (tmp_path / "fresh.db.init.lock").exists()


def test_sqlite_lock_does_not_mask_backend_os_error(tmp_path, monkeypatch) -> None:
    class BrokenClient:
        def __init__(self, *, tracking_uri: str) -> None:
            raise OSError(f"backend failure: {tracking_uri}")

    monkeypatch.setattr("mlflow.tracking.MlflowClient", BrokenClient)

    with pytest.raises(OSError, match="backend failure"):
        tracking.mlflow_client(f"sqlite:///{tmp_path / 'fresh.db'}")


def test_metric_key_constants_match_the_recorded_convention() -> None:
    """상수가 기존 기록 키와 어긋나면 과거 실행 전체를 못 읽게 되므로 철자를 고정한다."""
    assert tracking.METRIC_OOF_AUC == "auc_oof"
    assert tracking.METRIC_PUBLIC_AUC == "public_auc"
    assert tracking.metric_fold_auc(0) == "auc_fold_0"
    assert tracking.metric_seed_auc(42) == "auc_oof_seed_42"


def test_cv_scoring_records_keys_built_from_the_constants() -> None:
    import numpy as np
    import pandas as pd

    from pipeline.cv import score_predictions

    y = pd.Series([0, 1, 0, 1])
    folds = pd.Series([0, 0, 1, 1])
    pred = np.array([0.1, 0.9, 0.2, 0.8])

    keys = set(score_predictions(y, folds, pred))

    assert keys == {
        tracking.METRIC_OOF_AUC,
        tracking.metric_fold_auc(0),
        tracking.metric_fold_auc(1),
    }


def test_judgment_delegates_metric_key_spelling_to_tracking() -> None:
    from pipeline import judgment

    assert judgment.seed_auc_metric(43) == tracking.metric_seed_auc(43)
    assert judgment.seed_aucs_of({tracking.metric_seed_auc(42): 0.9}) == {42: 0.9}
    assert judgment.fold_aucs_of({tracking.metric_fold_auc(3): 0.8}) == {3: 0.8}
