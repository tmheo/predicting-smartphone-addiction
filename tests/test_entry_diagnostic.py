"""모델 진입 진단 공통 경로 회귀 시험. (#140)"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import model as model_mod
from pipeline.config import DataConfig, ExperimentConfig, FeatureConfig, ModelConfig
from pipeline.entry_diagnostic import (
    IMPORTANCE_NAME,
    PREDICTIONS_NAME,
    RESULT_NAME,
    run_fold_diagnostic,
    write_diagnostic,
)
from pipeline.ledger import CHAMPION_PATH, POOL_PATH
from pipeline.plan import FeaturePlan


class DiagnosticFakeAdapter:
    """학습 없이 결정적인 예측과 모델별 진단을 돌려주는 adapter."""

    def __init__(self, params: dict, fit: dict, seed: int) -> None:
        self.seed = seed
        self.validation_index: pd.Index | None = None
        self.features: list[str] = []

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        initial_score_tr: pd.Series | None = None,
        initial_score_va: pd.Series | None = None,
    ) -> np.ndarray:
        self.validation_index = X_va.index.copy()
        self.features = list(X_tr.columns)
        return np.full(len(X_va), 0.5, dtype="float64")

    def predict(
        self, X: pd.DataFrame, initial_score: pd.Series | None = None
    ) -> np.ndarray:
        return np.full(len(X), 0.5, dtype="float64")

    def importance(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.features, "gain": np.ones(len(self.features))})

    def entry_diagnostics(self) -> model_mod.AdapterDiagnostics:
        return model_mod.AdapterDiagnostics(
            assertions={
                model_mod.ASSERT_CANDIDATE_STORE_TRAIN_ONLY: True,
                model_mod.ASSERT_VALIDATION_LABELS_EXCLUDED: True,
                model_mod.ASSERT_SELF_ROWS_EXCLUDED: True,
            },
            observations={"candidate_rows": 40, "model_note": "fake"},
        )


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="entry_diagnostic_fake",
        data=DataConfig(
            train=Path("unused"),
            test=Path("unused"),
            sample_submission=Path("unused"),
            folds=Path("artifacts/folds.parquet"),
        ),
        features=FeatureConfig(base="raw", categorical=[], providers=[]),
        model=ModelConfig(kind="diagnostic_fake", params={}, fit={}),
        initial_score=None,
        seeds=[42],
        stage="screen",
        source_path=Path("configs/fake.yaml"),
    )


def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    n = 60
    train = pd.DataFrame(
        {
            "id": np.arange(n),
            "daily_screen_time_hours": np.linspace(1, 10, n),
            "social_media_hours": np.linspace(0, 5, n),
            "addicted_label": np.tile([0, 1], n // 2),
            "fold": np.arange(n) % 3,
        }
    )
    test = train.drop(columns=["addicted_label", "fold"]).iloc[:12].copy()
    test["id"] += n
    return train, test


def test_fold_zero_is_deterministic_and_does_not_modify_ledgers(monkeypatch, tmp_path):
    made: list[DiagnosticFakeAdapter] = []

    def factory(params: dict, fit: dict, seed: int) -> DiagnosticFakeAdapter:
        adapter = DiagnosticFakeAdapter(params, fit, seed)
        made.append(adapter)
        return adapter

    monkeypatch.setitem(model_mod.MODEL_REGISTRY, "diagnostic_fake", factory)
    cfg = _config()
    plan = FeaturePlan.from_config(cfg.features)
    train, test = _data()
    train, test = plan.apply_dataset_wide(train, test)
    champion_before = CHAMPION_PATH.read_bytes()
    pool_before = POOL_PATH.read_bytes()

    first = run_fold_diagnostic(
        cfg, plan, train, test, champion_fold_auc=0.5, limit_hours=24
    )
    assert len(made) == 1
    assert made[0].seed == 42
    assert made[0].validation_index.equals(train.index[train["fold"] == 0])
    assert set(first.predictions["fold"]) == {0}

    second = run_fold_diagnostic(
        cfg, plan, train, test, champion_fold_auc=0.5, limit_hours=24
    )
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert first.result["validation"] == second.result["validation"]
    assert first.result["adapter"] == second.result["adapter"]
    assert first.result["decision"]["checks"] == second.result["decision"]["checks"]
    assert first.result["decision"]["passed"] is True

    out_dir = tmp_path / "entry"
    write_diagnostic(first, out_dir)
    assert (out_dir / RESULT_NAME).is_file()
    stored = pd.read_parquet(out_dir / PREDICTIONS_NAME)
    pd.testing.assert_frame_equal(stored, first.predictions)
    assert (out_dir / IMPORTANCE_NAME).is_file()

    assert CHAMPION_PATH.read_bytes() == champion_before
    assert POOL_PATH.read_bytes() == pool_before


def test_failed_adapter_assertion_stops_promotion(monkeypatch):
    class UnsafeAdapter(DiagnosticFakeAdapter):
        def entry_diagnostics(self) -> model_mod.AdapterDiagnostics:
            return model_mod.AdapterDiagnostics(
                assertions={model_mod.ASSERT_SELF_ROWS_EXCLUDED: False},
                observations={"self_matches": 1},
            )

    monkeypatch.setitem(model_mod.MODEL_REGISTRY, "diagnostic_fake", UnsafeAdapter)
    cfg = _config()
    plan = FeaturePlan.from_config(cfg.features)
    train, test = _data()
    train, test = plan.apply_dataset_wide(train, test)

    run = run_fold_diagnostic(
        cfg,
        plan,
        train,
        test,
        champion_fold_auc=0.5,
    )

    assert run.result["decision"]["status"] == "stop"
    assert run.result["decision"]["checks"]["adapter_assertions"] is False
    assert "모델 assertion 실패" in run.result["decision"]["reasons"][0]


def test_wrong_validation_row_count_is_saved_as_stop_result(monkeypatch):
    class ShortAdapter(DiagnosticFakeAdapter):
        def fit(
            self,
            X_tr: pd.DataFrame,
            y_tr: pd.Series,
            X_va: pd.DataFrame,
            y_va: pd.Series,
            initial_score_tr: pd.Series | None = None,
            initial_score_va: pd.Series | None = None,
        ) -> np.ndarray:
            self.features = list(X_tr.columns)
            return np.full(len(X_va) - 1, 0.5, dtype="float64")

    monkeypatch.setitem(model_mod.MODEL_REGISTRY, "diagnostic_fake", ShortAdapter)
    cfg = _config()
    plan = FeaturePlan.from_config(cfg.features)
    train, test = _data()
    train, test = plan.apply_dataset_wide(train, test)

    run = run_fold_diagnostic(
        cfg, plan, train, test, champion_fold_auc=0.5
    )

    assert run.result["rows"]["validation_expected"] == 20
    assert run.result["rows"]["validation_predictions"] == 19
    assert run.result["validation"]["row_count_ok"] is False
    assert run.result["validation"]["auc"] is None
    assert run.result["decision"]["status"] == "stop"


def test_tabr_s_cuda_memory_above_ninety_percent_stops_promotion(monkeypatch):
    from pipeline import entry_diagnostic as diagnostic_mod

    monkeypatch.setitem(model_mod.MODEL_REGISTRY, "tabr_s", DiagnosticFakeAdapter)
    monkeypatch.setattr(diagnostic_mod, "_reset_cuda_peak", lambda: True)
    monkeypatch.setattr(
        diagnostic_mod,
        "_cuda_peak",
        lambda enabled: {
            "available": True,
            "source": "test",
            "max_allocated_bytes": 94,
            "max_reserved_bytes": 91,
            "device_total_bytes": 100,
        },
    )
    cfg = replace(_config(), model=ModelConfig(kind="tabr_s", params={}, fit={}))
    plan = FeaturePlan.from_config(cfg.features)
    train, test = _data()
    train, test = plan.apply_dataset_wide(train, test)

    run = run_fold_diagnostic(cfg, plan, train, test, champion_fold_auc=0.5)

    assert run.result["decision"]["checks"]["cuda_memory_limit"] is False
    assert run.result["projection"]["cuda_memory_fraction"] == 0.91
    assert run.result["decision"]["status"] == "stop"
