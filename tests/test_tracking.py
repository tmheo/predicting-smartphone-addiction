"""MLflow 실행 저장소 초기화의 동시 접근 회귀 검사."""

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
