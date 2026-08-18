"""모델 진입 진단 공통 경로 회귀 시험. (#140)"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import model as model_mod
from pipeline.config import DataConfig, ExperimentConfig, FeatureConfig, ModelConfig
from pipeline.entry_diagnostic import (
    CHAMPION_IMPROVEMENT_MODE,
    IMPORTANCE_NAME,
    NEW_MODEL_FAMILY_MODE,
    PREDICTIONS_NAME,
    RESULT_NAME,
    BaselineEvidence,
    DiagnosticEvidenceError,
    DiagnosticRun,
    apply_paired_comparison,
    apply_reference_reproduction_check,
    build_execution_identity,
    load_baseline_evidence,
    parse_args,
    run_fold_diagnostic,
    validate_baseline_rows,
    validate_pairing_identity,
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


def _identity(
    cfg: ExperimentConfig, plan: FeaturePlan, *, fold: int = 0, seed: int = 42
) -> dict[str, object]:
    return build_execution_identity(
        cfg,
        plan,
        {"train": "train-sha", "test": "test-sha", "folds": "folds-sha"},
        fold=fold,
        seed=seed,
        model_dependencies={
            "python_version": "3.13.1",
            "uv_lock_sha256": "lock-sha",
            "project_packages": {"numpy": "2.2.0"},
        },
    )


def _reference_run(monkeypatch) -> tuple[DiagnosticRun, pd.DataFrame, dict[str, object]]:
    monkeypatch.setitem(model_mod.MODEL_REGISTRY, "diagnostic_fake", DiagnosticFakeAdapter)
    cfg = _config()
    plan = FeaturePlan.from_config(cfg.features)
    train, test = _data()
    train, test = plan.apply_dataset_wide(train, test)
    identity = _identity(cfg, plan)
    run = run_fold_diagnostic(
        cfg, plan, train, test, execution_identity=identity, limit_hours=24
    )
    apply_reference_reproduction_check(run, 0.5)
    return run, train, identity


def test_cli_requires_explicit_reference_or_baseline_artifacts(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "entry_diagnostic",
            "configs/exp085.yaml",
            "--out-dir",
            "artifacts/entry-exp085",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()

    monkeypatch.setattr(
        "sys.argv",
        [
            "entry_diagnostic",
            "configs/exp067.yaml",
            "--out-dir",
            "artifacts/entry-exp067",
            "--reference",
            "--expected-baseline-auc",
            "0.968294911389327",
            "--reference-auc-tolerance",
            "0.0002",
        ],
    )
    args = parse_args()
    assert args.reference is True
    assert args.expected_baseline_auc == 0.968294911389327
    assert args.reference_auc_tolerance == 0.0002


def test_cli_accepts_explicit_paired_comparison_inputs(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "entry_diagnostic",
            "configs/exp085.yaml",
            "--out-dir",
            "artifacts/entry-exp085",
            "--baseline-diagnostic",
            "artifacts/entry-exp067/entry_diagnostic.json",
            "--baseline-predictions",
            "artifacts/entry-exp067/validation_predictions.parquet",
            "--comparison-mode",
            CHAMPION_IMPROVEMENT_MODE,
            "--allow-model-diff",
            "params.learning_rate",
        ],
    )
    args = parse_args()
    assert args.comparison_mode == CHAMPION_IMPROVEMENT_MODE
    assert args.allow_model_diff == ["params.learning_rate"]


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
    identity = _identity(cfg, plan)
    champion_before = CHAMPION_PATH.read_bytes()
    pool_before = POOL_PATH.read_bytes()

    first = run_fold_diagnostic(
        cfg, plan, train, test, execution_identity=identity, limit_hours=24
    )
    assert len(made) == 1
    assert made[0].seed == 42
    assert made[0].validation_index.equals(train.index[train["fold"] == 0])
    assert set(first.predictions["fold"]) == {0}

    second = run_fold_diagnostic(
        cfg, plan, train, test, execution_identity=identity, limit_hours=24
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


def test_reference_artifacts_recompute_stored_auc_and_reject_three_seed_fold_mix(
    monkeypatch, tmp_path
):
    run, _, _ = _reference_run(monkeypatch)
    out_dir = tmp_path / "baseline"
    write_diagnostic(run, out_dir)

    evidence = load_baseline_evidence(
        out_dir / RESULT_NAME, out_dir / PREDICTIONS_NAME
    )
    assert evidence.auc == 0.5

    result_path = out_dir / RESULT_NAME
    result = json.loads(result_path.read_text())
    result["validation"]["auc"] = 0.51
    result_path.write_text(json.dumps(result))
    with pytest.raises(DiagnosticEvidenceError, match="저장된 같은 단계 champion"):
        load_baseline_evidence(result_path, out_dir / PREDICTIONS_NAME)


def test_reference_reproduction_stops_outside_stored_champion_tolerance(monkeypatch):
    run, _, _ = _reference_run(monkeypatch)

    apply_reference_reproduction_check(run, 0.51)

    assert run.result["reference_reproduction"]["matches"] is False
    assert (
        run.result["decision"]["checks"]["same_stage_champion_auc_reproduced"]
        is False
    )
    assert run.result["decision"]["status"] == "stop"


def test_reference_reproduction_accepts_explicit_hardware_tolerance(monkeypatch):
    run, _, _ = _reference_run(monkeypatch)

    apply_reference_reproduction_check(run, 0.50006, tolerance=2e-4)

    reproduction = run.result["reference_reproduction"]
    assert reproduction["matches"] is True
    assert reproduction["tolerance"] == 2e-4
    assert reproduction["tolerance_source"] == "explicit_hardware_environment"
    assert run.result["decision"]["status"] == "pass"


@pytest.mark.parametrize("tolerance", [0.0, -1e-9, 2.00001e-4, float("inf")])
def test_reference_reproduction_rejects_unsafe_tolerance(monkeypatch, tolerance):
    run, _, _ = _reference_run(monkeypatch)

    with pytest.raises(ValueError, match="기준 AUC 허용 범위"):
        apply_reference_reproduction_check(run, 0.5, tolerance=tolerance)


def test_baseline_json_and_prediction_file_are_content_bound(monkeypatch, tmp_path):
    run, _, _ = _reference_run(monkeypatch)
    out_dir = tmp_path / "baseline"
    write_diagnostic(run, out_dir)
    predictions_path = out_dir / PREDICTIONS_NAME
    predictions = pd.read_parquet(predictions_path)
    predictions.loc[0, "pred"] = 0.9
    predictions.to_parquet(predictions_path, index=False)

    with pytest.raises(DiagnosticEvidenceError, match="내용 해시"):
        load_baseline_evidence(out_dir / RESULT_NAME, predictions_path)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("input_sha256", "train"), "other", "입력 내용 해시"),
        (("input_sha256", "folds"), "other", "입력 내용 해시"),
        (("folds_sha256",), "other", "fold 파일"),
        (("fold",), 1, "fold 번호"),
        (("seed",), 43, "시드"),
        (("feature_plan", "columns"), ["other"], "피처 계획"),
        (("model_dependencies", "uv_lock_sha256"), "other", "실행 의존성 판본"),
    ],
)
def test_pairing_rejects_mismatched_execution_identity(
    monkeypatch, path, value, message
):
    run, _, identity = _reference_run(monkeypatch)
    baseline = BaselineEvidence(run.result, run.predictions, 0.5)
    challenger = copy.deepcopy(identity)
    cursor = challenger
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value

    with pytest.raises(DiagnosticEvidenceError, match=message):
        validate_pairing_identity(baseline, challenger, [])


def test_pairing_records_only_explicitly_allowed_model_differences(monkeypatch):
    run, _, identity = _reference_run(monkeypatch)
    baseline = BaselineEvidence(run.result, run.predictions, 0.5)
    challenger = copy.deepcopy(identity)
    challenger["model"]["params"]["learning_rate"] = 0.01

    with pytest.raises(DiagnosticEvidenceError, match="허용하지 않은 모델 설정 차이"):
        validate_pairing_identity(baseline, challenger, [])

    pairing = validate_pairing_identity(
        baseline, challenger, ["params.learning_rate"]
    )
    assert pairing["allowed_model_axes"] == ["params.learning_rate"]
    assert pairing["model_differences"] == [
        {
            "path": "params.learning_rate",
            "baseline": None,
            "challenger": 0.01,
            "baseline_present": False,
            "challenger_present": True,
        }
    ]


def test_pairing_rejects_wrong_validation_id_order_and_targets(monkeypatch):
    run, train, _ = _reference_run(monkeypatch)
    reversed_predictions = run.predictions.iloc[::-1].reset_index(drop=True)
    baseline = BaselineEvidence(run.result, reversed_predictions, 0.5)
    with pytest.raises(DiagnosticEvidenceError, match="id 순서"):
        validate_baseline_rows(baseline, train, fold=0)

    wrong_target = run.predictions.copy()
    wrong_target.loc[0, "addicted_label"] = 1 - wrong_target.loc[0, "addicted_label"]
    baseline = BaselineEvidence(run.result, wrong_target, 0.5)
    with pytest.raises(DiagnosticEvidenceError, match="목표값"):
        validate_baseline_rows(baseline, train, fold=0)


def test_champion_improvement_uses_paired_nonnegative_auc_delta(monkeypatch):
    reference, train, identity = _reference_run(monkeypatch)
    baseline = BaselineEvidence(reference.result, reference.predictions, 0.5)
    pairing = validate_pairing_identity(baseline, identity, [])
    validate_baseline_rows(baseline, train, fold=0)
    challenger, _, _ = _reference_run(monkeypatch)

    apply_paired_comparison(
        challenger,
        baseline,
        mode=CHAMPION_IMPROVEMENT_MODE,
        pairing=pairing,
        baseline_diagnostic_path=Path("baseline/entry_diagnostic.json"),
        baseline_predictions_path=Path("baseline/validation_predictions.parquet"),
    )

    assert challenger.result["comparison"]["auc_delta"] == 0.0
    assert challenger.result["comparison"]["auc_threshold"] == 0.5
    assert challenger.result["decision"]["checks"]["paired_auc_threshold"] is True
    assert challenger.result["decision"]["status"] == "pass"


@pytest.mark.parametrize(
    ("mode", "expected_status"),
    [
        (CHAMPION_IMPROVEMENT_MODE, "stop"),
        (NEW_MODEL_FAMILY_MODE, "pass"),
    ],
)
def test_comparison_modes_keep_separate_promotion_thresholds(mode, expected_status):
    target = np.array([0] * 10 + [1] * 10)
    baseline_predictions = pd.DataFrame(
        {
            "id": np.arange(20),
            "fold": 0,
            "addicted_label": target,
            "pred": np.arange(20, dtype="float64"),
        }
    )
    challenger_predictions = baseline_predictions.copy()
    challenger_predictions.loc[[9, 10], "pred"] = [10.0, 9.0]
    baseline = BaselineEvidence(
        {"experiment": "champion"}, baseline_predictions, 1.0
    )
    run = DiagnosticRun(
        result={
            "validation": {"auc": 0.99},
            "decision": {"checks": {"common": True}, "reasons": []},
        },
        predictions=challenger_predictions,
        importance=pd.DataFrame(),
    )

    apply_paired_comparison(
        run,
        baseline,
        mode=mode,
        pairing={"checks": {}},
        baseline_diagnostic_path=Path("baseline.json"),
        baseline_predictions_path=Path("baseline.parquet"),
    )

    assert run.result["comparison"]["auc_delta"] == pytest.approx(-0.01)
    expected_threshold = 1.0 if mode == CHAMPION_IMPROVEMENT_MODE else 0.99
    assert run.result["comparison"]["auc_threshold"] == expected_threshold
    assert run.result["decision"]["status"] == expected_status


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
    identity = _identity(cfg, plan)

    run = run_fold_diagnostic(
        cfg,
        plan,
        train,
        test,
        execution_identity=identity,
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
    identity = _identity(cfg, plan)

    run = run_fold_diagnostic(cfg, plan, train, test, execution_identity=identity)

    assert run.result["rows"]["validation_expected"] == 20
    assert run.result["rows"]["validation_predictions"] == 19
    assert run.result["validation"]["row_count_ok"] is False
    assert run.result["validation"]["auc"] is None
    assert run.result["decision"]["status"] == "stop"


def test_adapter_runtime_abort_skips_expensive_followup_steps(monkeypatch):
    class RuntimeAbortAdapter(DiagnosticFakeAdapter):
        def predict(
            self, X: pd.DataFrame, initial_score: pd.Series | None = None
        ) -> np.ndarray:
            raise AssertionError("진입 중단 뒤 test 예측을 호출하면 안 된다.")

        def importance(self) -> pd.DataFrame:
            raise AssertionError("진입 중단 뒤 중요도 계산을 호출하면 안 된다.")

        def entry_diagnostics(self) -> model_mod.AdapterDiagnostics:
            return model_mod.AdapterDiagnostics(
                observations={"projected_5fold_training_seconds": 25 * 3600}
            )

        def entry_abort_reason(self) -> str:
            return "5-fold 예상 시간이 24시간을 넘는다."

    monkeypatch.setitem(
        model_mod.MODEL_REGISTRY, "diagnostic_fake", RuntimeAbortAdapter
    )
    cfg = _config()
    plan = FeaturePlan.from_config(cfg.features)
    train, test = _data()
    train, test = plan.apply_dataset_wide(train, test)
    identity = _identity(cfg, plan)

    run = run_fold_diagnostic(
        cfg, plan, train, test, execution_identity=identity, limit_hours=24
    )

    assert run.result["validation"]["auc"] == 0.5
    assert run.result["decision"]["checks"]["adapter_entry_abort"] is False
    assert run.result["decision"]["checks"]["projected_time_limit"] is False
    assert run.result["projection"]["seed_5fold_seconds"] == 25 * 3600
    assert run.result["adapter"]["abort_reason"] is not None
    assert run.importance.empty


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
    identity = _identity(cfg, plan)

    run = run_fold_diagnostic(cfg, plan, train, test, execution_identity=identity)

    assert run.result["decision"]["checks"]["cuda_memory_limit"] is False
    assert run.result["projection"]["cuda_memory_fraction"] == 0.91
    assert run.result["decision"]["status"] == "stop"


def test_trompt_cuda_memory_above_absolute_limit_stops_promotion(monkeypatch):
    from pipeline import entry_diagnostic as diagnostic_mod

    gib = 1024**3
    monkeypatch.setitem(model_mod.MODEL_REGISTRY, "trompt", DiagnosticFakeAdapter)
    monkeypatch.setattr(diagnostic_mod, "_reset_cuda_peak", lambda: True)
    monkeypatch.setattr(
        diagnostic_mod,
        "_cuda_peak",
        lambda enabled: {
            "available": True,
            "source": "test",
            "max_allocated_bytes": 14 * gib,
            "max_reserved_bytes": 15 * gib,
            "device_total_bytes": 16 * gib,
        },
    )
    cfg = replace(_config(), model=ModelConfig(kind="trompt", params={}, fit={}))
    plan = FeaturePlan.from_config(cfg.features)
    train, test = _data()
    train, test = plan.apply_dataset_wide(train, test)
    identity = _identity(cfg, plan)

    run = run_fold_diagnostic(cfg, plan, train, test, execution_identity=identity)

    assert run.result["decision"]["checks"]["cuda_memory_limit"] is False
    assert run.result["projection"]["cuda_memory_limit_bytes"] == 14 * gib
    assert run.result["decision"]["status"] == "stop"
