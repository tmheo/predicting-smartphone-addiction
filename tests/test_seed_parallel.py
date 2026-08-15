"""시드 병렬 실행 테스트. (#99)

- 순차 경로: PIPELINE_SEED_GPUS가 없으면 기존 시드 루프 그대로다(가짜 adapter로 검증).
- 병렬 경로: 스폰된 워커는 monkeypatch를 못 보므로 실제 등록된 logistic_onehot을
  소형 데이터로 학습해, 순차 실행과 결과가 동일하고 fold 완료 통지가 전부
  부모 recorder에 도착하는지 검증한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_model import FakeAdapter, SpyRecorder, toy_train_test

from pipeline import model as model_mod
from pipeline import seed_parallel
from pipeline.config import DataConfig, ExperimentConfig, FeatureConfig, ModelConfig
from pipeline.plan import FeaturePlan

N_FOLDS = 3
SEEDS = [7, 11, 13]


def experiment_config(kind: str, params: dict) -> ExperimentConfig:
    from pathlib import Path

    return ExperimentConfig(
        name="seed_parallel_test",
        data=DataConfig(
            train=Path("unused"), test=Path("unused"),
            sample_submission=Path("unused"), folds=Path("unused"),
        ),
        features=FeatureConfig(base="raw", categorical=[], providers=[]),
        model=ModelConfig(kind=kind, params=params, fit={}),
        initial_score=None,
        seeds=SEEDS,
        stage="confirm",
        source_path=Path("unused"),
    )


def prepared_inputs(cfg: ExperimentConfig) -> tuple[FeaturePlan, pd.DataFrame, pd.DataFrame]:
    plan = FeaturePlan.from_config(cfg.features)
    train, test = toy_train_test()
    train, test = plan.apply_dataset_wide(train, test)
    train["fold"] = np.arange(len(train)) % N_FOLDS
    return plan, train, test


def test_without_env_runs_sequentially_with_per_seed_stages(monkeypatch):
    monkeypatch.delenv(seed_parallel.ENV_GPUS, raising=False)
    monkeypatch.setitem(
        model_mod.MODEL_REGISTRY,
        "fake",
        lambda params, fit, seed: FakeAdapter(params, fit, seed, fold_value=0.5),
    )
    cfg = experiment_config("fake", {})
    plan, train, test = prepared_inputs(cfg)
    recorder = SpyRecorder()

    results = seed_parallel.run_seeds(cfg, plan, train, test, recorder=recorder)

    assert len(results) == len(SEEDS)
    # 순차 경로는 시드마다 feature_build/training 단계를 그대로 통지한다.
    assert recorder.stages == ["feature_build", "training"] * len(SEEDS)
    assert recorder.folds == [
        (seed_index, fold, pytest.approx(0.5))
        for seed_index in range(len(SEEDS))
        for fold in range(N_FOLDS)
    ]


def test_parallel_execution_matches_sequential_and_forwards_folds(monkeypatch):
    cfg = experiment_config("logistic_onehot", {"onehot_max_card": 10})
    plan, train, test = prepared_inputs(cfg)

    monkeypatch.delenv(seed_parallel.ENV_GPUS, raising=False)
    sequential = seed_parallel.run_seeds(cfg, plan, train, test)

    monkeypatch.setenv(seed_parallel.ENV_GPUS, "0,1")
    recorder = SpyRecorder()
    parallel = seed_parallel.run_seeds(cfg, plan, train, test, recorder=recorder)

    # 시드 순서와 예측·지표·중요도가 순차 실행과 동일하다(같은 시드 재심기 계약).
    assert len(parallel) == len(sequential) == len(SEEDS)
    for seq, par in zip(sequential, parallel):
        pd.testing.assert_frame_equal(seq.oof, par.oof)
        pd.testing.assert_frame_equal(seq.test_pred, par.test_pred)
        pd.testing.assert_frame_equal(seq.importance, par.importance)
        assert seq.fold_aucs == par.fold_aucs
        assert seq.feature_names == par.feature_names

    # 병렬 경로의 단계 기록은 training 하나로 묶이고, fold 완료는 전부 도착한다
    # (시드 간 순서는 비결정적이라 집합으로 비교).
    assert recorder.stages == ["training"]
    assert sorted((s, f) for s, f, _ in recorder.folds) == sorted(
        (seed_index, fold)
        for seed_index in range(len(SEEDS))
        for fold in range(N_FOLDS)
    )
