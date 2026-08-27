"""자체 구성원 순위 사분위 범위 전용 선형 결합을 판정한다.

이슈 467의 동결 계약만 실행한다.
대조군과 후보군은 같은 바깥쪽 분할 반복에서 계산하고, 결과를 본 뒤 후보나
확인 구간, 연결 방식 또는 판정 문턱을 바꾸는 선택지는 제공하지 않는다.

사용법:
    uv run python scripts/judge_own_iqr_specialized_stack.py \
        --input-root /path/to/input/repository

입력 실행 기록이 현재 작업 트리에 없으면 ``--input-root``로 원본 실행 저장소를
지정한다.
산출물은 임시 폴더에서 모두 만든 뒤 최종 경로에 한 번에 확정한다.
기존 산출물이나 잠금이 있으면 덮어쓰지 않고 중단한다.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import sklearn
from sklearn.metrics import roc_auc_score

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import analyze_oof_row_difficulty as difficulty  # noqa: E402
from pipeline import ensemble  # noqa: E402

ISSUE = 467
ID = "id"
TARGET = "addicted_label"
LOWER_IQR = 0.023383523535395911
UPPER_IQR = 0.036234818789745993
BASELINE_NESTED_AUC = 0.9702876097776773
BASELINE_FOLD_AUCS = {
    0: 0.9697332661529522,
    1: 0.9704381694607899,
    2: 0.9703637806523540,
    3: 0.9708882903449186,
    4: 0.9700145394936390,
}
BASELINE_PREDICTION_SHA256 = (
    "006ca009b1b8f115939747ca55c908755f3d999426f35cf90b8d00b66d7367cc"
)
BASELINE_TOLERANCE = 1e-10
MINIMUM_NESTED_DELTA = 0.00002
EXPECTED_INPUT_HASHES = {
    "train": "f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c",
    "test": "8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e",
    "folds": "5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4",
    "pool": "caa1b90769720a4accbe07074dbc7efe0335ab6657fea80c6839b60121dc39d3",
    "champion": "aa012114107c06532cf51c0fa9c741f5949146428cf266cf4bedded783d20e09",
    "extended_evidence": (
        "9893c49fa3e39306713ff6fa99e69af78dd0cb1c557cbf03ead16cb239c3b0b3"
    ),
    "assembly_manifest": (
        "cb442519ea3120385f71c201b8fc2b313abcdb6994f2476f06d589b478aea480"
    ),
    "pool_oof_ledger": (
        "4e2ec4a8a030a448962dbdc4a889ffc32853c6b836828c5b569e0b9e70dc308e"
    ),
    "external_cache": (
        "f5ded7caa700804031a2cfc2901aa7be53a61b1f30d549cb792737a140862d47"
    ),
    "uv_lock": "27527bf7a3094af0c9fa85613216f097a9d18b4cf75ab4c5b4c73b2ad3af25b4",
    "external_ledger": (
        "5dca2d01acc320299ae41d396a1cc6a2e5777614ec665c4b039eed4efd036d3c"
    ),
    "submission_record": (
        "2de63c967efdbbb794e6858161caf582ad750971882ef9809d7c8e865a9b18ac"
    ),
}
OUTPUT_DIR = REPOSITORY_ROOT / "run-logs/issue467"
EVIDENCE_PATH = (
    REPOSITORY_ROOT
    / "docs/research/own-member-iqr-specialized-stack-evidence.json"
)
REPORT_PATH = REPOSITORY_ROOT / "docs/research/own-member-iqr-specialized-stack.md"


class ContractError(RuntimeError):
    """동결 계약을 지킬 수 없어 판정 불가로 중단해야 한다."""


@dataclass(frozen=True)
class NestedComparison:
    baseline_prediction: np.ndarray
    candidate_prediction: np.ndarray
    condition: np.ndarray
    own_iqr: np.ndarray
    fold_records: list[dict[str, Any]]
    elapsed_seconds: float


@dataclass(frozen=True)
class FullFitResult:
    candidate_prediction: np.ndarray
    condition: np.ndarray
    own_iqr: np.ndarray
    record: dict[str, Any]
    input_record: dict[str, Any]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def counts_by_target(mask: np.ndarray, target: pd.Series) -> dict[str, int]:
    values = target.to_numpy(dtype=np.int8)
    return {
        "total": int(mask.sum()),
        "target_0": int(np.sum(mask & (values == 0))),
        "target_1": int(np.sum(mask & (values == 1))),
    }


def own_member_iqr(block: pd.DataFrame, own_columns: list[str]) -> np.ndarray:
    """처리 블록 안에서 자체 구성원 35개의 평균 동순위 사분위 범위를 계산한다."""
    require(len(own_columns) == 35, f"자체 구성원이 35개가 아니다: {len(own_columns)}")
    require(len(block) > 0, "순위 사분위 범위를 계산할 블록이 비었다.")
    own = block[own_columns]
    require(
        np.isfinite(own.to_numpy(dtype=np.float64)).all(),
        "자체 구성원 예측에 유한하지 않은 값이 있다.",
    )
    ranks = own.rank(method="average", axis=0).to_numpy(dtype=np.float64)
    ranks = (ranks - 0.5) / len(block)
    q25, q75 = np.quantile(
        ranks,
        (0.25, 0.75),
        axis=1,
        method="linear",
    )
    result = np.asarray(q75 - q25, dtype=np.float64)
    require(np.isfinite(result).all(), "자체 구성원 순위 사분위 범위가 유한하지 않다.")
    return result


def condition_mask(iqr: np.ndarray) -> np.ndarray:
    return (iqr >= LOWER_IQR) & (iqr < UPPER_IQR)


def rank_percentile(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    return (ranks - 0.5) / len(ranks)


def text_list_sha256(values: list[str]) -> str:
    payload = (
        json.dumps(
            values,
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def fit_specialized(
    inner: pd.DataFrame,
    target: pd.Series,
    mask: np.ndarray,
) -> ensemble.FittedLogisticLinear:
    selected_target = target[mask]
    target_values = selected_target.to_numpy(dtype=np.int8)
    require(mask.any(), "확인 구간 전용 결합기의 학습 행이 없다.")
    require(
        set(np.unique(target_values)) == {0, 1},
        "확인 구간 전용 결합기의 학습 행에 두 목표값이 모두 있지 않다.",
    )
    model = ensemble.LogisticLinearCombiner(
        "own_iqr_specialized_rank_logit",
        "rank_logit",
        c=1.0,
        max_iter=1_000,
    )
    try:
        return model.fit(inner[mask], selected_target)
    except ensemble.CombinerConvergenceError as exc:
        raise ContractError(f"확인 구간 전용 결합기가 수렴하지 않았다: {exc}") from exc


def evaluate_nested_comparison(
    inputs: difficulty.Inputs,
) -> NestedComparison:
    started = time.monotonic()
    row_count = len(inputs.members)
    baseline = np.full(row_count, np.nan, dtype=np.float64)
    candidate = np.full(row_count, np.nan, dtype=np.float64)
    condition = np.zeros(row_count, dtype=bool)
    own_iqr = np.full(row_count, np.nan, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []

    for fold in sorted(int(value) for value in inputs.folds.unique()):
        print(f"  바깥쪽 분할 {fold}: 전역 결합기와 구간 전용 결합기를 맞춘다.", flush=True)
        inner_mask = (inputs.folds != fold).to_numpy()
        outer_mask = (inputs.folds == fold).to_numpy()
        require(inner_mask.any() and outer_mask.any(), f"바깥쪽 분할 {fold}이 비었다.")
        inner = inputs.members[inner_mask]
        outer = inputs.members[outer_mask]
        inner_target = inputs.target[inner_mask]
        outer_target = inputs.target[outer_mask]

        global_combiner = ensemble.ShrunkRankLogitCombiner(fold_of=inputs.folds)
        try:
            fitted_global = global_combiner.fit(inner, inner_target)
        except ensemble.CombinerConvergenceError as exc:
            raise ContractError(f"바깥쪽 분할 {fold}의 전역 결합기가 수렴하지 않았다: {exc}") from exc

        inner_iqr = own_member_iqr(inner, inputs.own_columns)
        inner_condition = condition_mask(inner_iqr)
        fitted_specialized = fit_specialized(inner, inner_target, inner_condition)

        outer_iqr = own_member_iqr(outer, inputs.own_columns)
        outer_condition = condition_mask(outer_iqr)
        global_prediction = np.asarray(fitted_global.predict(outer), dtype=np.float64)
        specialized_prediction = np.asarray(
            fitted_specialized.predict(outer), dtype=np.float64
        )
        require(
            np.isfinite(global_prediction).all()
            and np.isfinite(specialized_prediction).all(),
            f"바깥쪽 분할 {fold}의 결합 예측이 유한하지 않다.",
        )

        global_rank = rank_percentile(global_prediction)
        specialized_rank = rank_percentile(specialized_prediction)
        candidate_prediction = global_rank.copy()
        candidate_prediction[outer_condition] = specialized_rank[outer_condition]

        baseline[outer_mask] = global_prediction
        candidate[outer_mask] = candidate_prediction
        condition[outer_mask] = outer_condition
        own_iqr[outer_mask] = outer_iqr
        baseline_auc = float(
            roc_auc_score(outer_target.to_numpy(), global_prediction)
        )
        candidate_auc = float(
            roc_auc_score(outer_target.to_numpy(), candidate_prediction)
        )
        fold_records.append(
            {
                "fold": fold,
                "inner_rows": int(inner_mask.sum()),
                "outer_rows": int(outer_mask.sum()),
                "inner_condition_counts": counts_by_target(
                    inner_condition, inner_target
                ),
                "outer_condition_counts": counts_by_target(
                    outer_condition, outer_target
                ),
                "global_shrinkage_lambda": float(
                    fitted_global.shrinkage_lambda
                ),
                "baseline_auc": baseline_auc,
                "candidate_auc": candidate_auc,
                "delta": candidate_auc - baseline_auc,
            }
        )
        print(
            f"    기준 {baseline_auc:.16f}, 후보 {candidate_auc:.16f}, "
            f"차이 {candidate_auc - baseline_auc:+.16f}",
            flush=True,
        )
        del inner, outer, fitted_global, fitted_specialized
        gc.collect()

    require(
        np.isfinite(baseline).all() and np.isfinite(candidate).all(),
        "바깥쪽 분할이 전체 행의 유한 예측을 만들지 못했다.",
    )
    require(np.isfinite(own_iqr).all(), "바깥쪽 분할의 순위 사분위 범위가 완전하지 않다.")
    return NestedComparison(
        baseline_prediction=baseline,
        candidate_prediction=candidate,
        condition=condition,
        own_iqr=own_iqr,
        fold_records=fold_records,
        elapsed_seconds=time.monotonic() - started,
    )


def verify_input_hashes(inputs: difficulty.Inputs) -> None:
    actual = inputs.input_manifest["sha256"]
    require(set(EXPECTED_INPUT_HASHES) <= set(actual), "동결 입력 해시 항목이 빠졌다.")
    for name, expected in EXPECTED_INPUT_HASHES.items():
        require(
            actual[name] == expected,
            f"{name} SHA-256 불일치: {actual[name]}",
        )
    require(inputs.members.shape[1] == 242, "결합 구성원이 242개가 아니다.")
    require(len(inputs.own_columns) == 35, "자체 구성원이 35개가 아니다.")
    require(len(inputs.external_columns) == 207, "외부 구성원이 207개가 아니다.")
    require(
        list(inputs.members.columns)
        == [entry["column"] for entry in inputs.assembly["members"]],
        "구성원 열과 순서가 조립 manifest와 다르다.",
    )


def verify_baseline(
    comparison: NestedComparison,
    inputs: difficulty.Inputs,
) -> dict[str, Any]:
    target = inputs.target.to_numpy(dtype=np.int8)
    nested_auc = float(roc_auc_score(target, comparison.baseline_prediction))
    absolute_delta = abs(nested_auc - BASELINE_NESTED_AUC)
    require(
        absolute_delta <= BASELINE_TOLERANCE,
        f"기준 nested OOF AUC가 재현되지 않았다: {nested_auc}",
    )
    prediction_hash = difficulty.array_sha256(comparison.baseline_prediction)
    require(
        prediction_hash == BASELINE_PREDICTION_SHA256,
        f"기준 nested OOF 예측 해시가 다르다: {prediction_hash}",
    )
    for record in comparison.fold_records:
        fold = int(record["fold"])
        delta = abs(float(record["baseline_auc"]) - BASELINE_FOLD_AUCS[fold])
        require(
            delta <= BASELINE_TOLERANCE,
            f"기준 분할 {fold} AUC가 재현되지 않았다: {record['baseline_auc']}",
        )
    return {
        "nested_auc": nested_auc,
        "expected_nested_auc": BASELINE_NESTED_AUC,
        "absolute_delta": absolute_delta,
        "prediction_sha256": prediction_hash,
        "fold_aucs": {
            str(record["fold"]): record["baseline_auc"]
            for record in comparison.fold_records
        },
        "reproduced": True,
    }


def burden_summary(
    comparison: NestedComparison,
    inputs: difficulty.Inputs,
) -> dict[str, Any]:
    target = inputs.target.to_numpy(dtype=np.int8)
    weight = inputs.missing_weight.to_numpy(dtype=np.float64)
    baseline_standard, baseline_weighted, baseline_checks = difficulty.rank_burdens(
        comparison.baseline_prediction,
        target,
        weight,
    )
    candidate_standard, candidate_weighted, candidate_checks = difficulty.rank_burdens(
        comparison.candidate_prediction,
        target,
        weight,
    )

    scales = {
        "standard": (baseline_standard, candidate_standard),
        "missingness_reweighted": (baseline_weighted, candidate_weighted),
    }
    burden: dict[str, Any] = {}
    for name, (baseline_values, candidate_values) in scales.items():
        entry: dict[str, Any] = {}
        for target_value in (None, 0, 1):
            mask = comparison.condition.copy()
            key = "all_targets"
            if target_value is not None:
                mask &= target == target_value
                key = f"target_{target_value}"
            require(mask.any(), f"설명 지표의 {name} {key} 확인 구간이 비었다.")
            baseline_mean = float(baseline_values[mask].mean())
            candidate_mean = float(candidate_values[mask].mean())
            entry[key] = {
                "rows": int(mask.sum()),
                "baseline_mean_burden": baseline_mean,
                "candidate_mean_burden": candidate_mean,
                "burden_reduction": baseline_mean - candidate_mean,
            }
        burden[name] = entry

    return {
        "condition_counts": counts_by_target(comparison.condition, inputs.target),
        "weighted_oof_auc": {
            "baseline": float(
                roc_auc_score(
                    target,
                    comparison.baseline_prediction,
                    sample_weight=weight,
                )
            ),
            "candidate": float(
                roc_auc_score(
                    target,
                    comparison.candidate_prediction,
                    sample_weight=weight,
                )
            ),
        },
        "condition_burden": burden,
        "burden_integrity": {
            "baseline": baseline_checks,
            "candidate": candidate_checks,
        },
    }


def load_test_members(
    input_root: Path,
    inputs: difficulty.Inputs,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    test_ids = inputs.test[ID]
    own_path = input_root / "artifacts/full-refit/member_test_cv_full.parquet"
    require(own_path.is_file(), f"자체 시험 예측 파일이 없다: {own_path}")
    own = pd.read_parquet(own_path)
    require(ID in own.columns, "자체 시험 예측에 id 열이 없다.")
    require(own[ID].equals(test_ids), "자체 시험 예측 id 순서가 test와 다르다.")
    own = own.set_index(ID)

    columns: dict[str, np.ndarray] = {}
    source_records: list[dict[str, Any]] = []
    for entry in inputs.assembly["members"]:
        name = str(entry["column"])
        test_record = entry["test"]
        if entry["origin"] == "own":
            require(name in own.columns, f"자체 시험 예측 열이 없다: {name}")
            values = own[name].to_numpy(dtype=np.float64)
        else:
            values = difficulty._load_ledger_array(
                input_root, str(test_record["test_path"])
            )
        require(values.shape == (len(test_ids),), f"{name} 시험 예측 행 수가 다르다.")
        require(np.isfinite(values).all(), f"{name} 시험 예측이 유한하지 않다.")
        prediction_hash = difficulty.array_sha256(values)
        require(
            prediction_hash == test_record["prediction_sha256"],
            f"{name} 시험 예측 해시가 조립 manifest와 다르다: {prediction_hash}",
        )
        columns[name] = values
        source_records.append(
            {
                "column": name,
                "kind": test_record["kind"],
                "test_path": test_record["test_path"],
                "prediction_sha256": prediction_hash,
            }
        )

    matrix = pd.DataFrame(columns, index=pd.Index(test_ids, name=ID), dtype=np.float64)
    require(
        list(matrix.columns) == list(inputs.members.columns),
        "시험 예측 구성원 열 순서가 OOF와 다르다.",
    )
    return matrix, {
        "own_test_path": str(own_path.relative_to(input_root)),
        "own_test_file_sha256": difficulty.sha256_file(own_path),
        "member_count": int(matrix.shape[1]),
        "member_prediction_hashes": source_records,
    }


def fit_full_and_predict_test(
    input_root: Path,
    inputs: difficulty.Inputs,
) -> FullFitResult:
    started = time.monotonic()
    print("  전체 OOF로 전역 결합기와 구간 전용 결합기를 맞춘다.", flush=True)
    global_combiner = ensemble.ShrunkRankLogitCombiner(fold_of=inputs.folds)
    try:
        fitted_global = global_combiner.fit(inputs.members, inputs.target)
    except ensemble.CombinerConvergenceError as exc:
        raise ContractError(f"전체 자료 전역 결합기가 수렴하지 않았다: {exc}") from exc
    train_iqr = own_member_iqr(inputs.members, inputs.own_columns)
    train_condition = condition_mask(train_iqr)
    fitted_specialized = fit_specialized(
        inputs.members,
        inputs.target,
        train_condition,
    )

    print("  동결한 242개 시험 예측을 읽고 열별 해시를 검사한다.", flush=True)
    test_members, test_input_record = load_test_members(input_root, inputs)
    test_iqr = own_member_iqr(test_members, inputs.own_columns)
    test_condition = condition_mask(test_iqr)
    global_prediction = np.asarray(
        fitted_global.predict(test_members), dtype=np.float64
    )
    specialized_prediction = np.asarray(
        fitted_specialized.predict(test_members), dtype=np.float64
    )
    require(
        np.isfinite(global_prediction).all()
        and np.isfinite(specialized_prediction).all(),
        "전체 자료 적합의 시험 예측이 유한하지 않다.",
    )
    candidate = rank_percentile(global_prediction)
    specialized_rank = rank_percentile(specialized_prediction)
    candidate[test_condition] = specialized_rank[test_condition]
    require(np.isfinite(candidate).all(), "최종 후보 시험 예측이 유한하지 않다.")
    global_shrinkage_lambda = float(fitted_global.shrinkage_lambda)
    del test_members, fitted_global, fitted_specialized
    gc.collect()
    return FullFitResult(
        candidate_prediction=candidate,
        condition=test_condition,
        own_iqr=test_iqr,
        record={
            "train_condition_counts": counts_by_target(
                train_condition, inputs.target
            ),
            "test_condition_rows": int(test_condition.sum()),
            "test_rows": int(len(test_condition)),
            "global_shrinkage_lambda": global_shrinkage_lambda,
            "candidate_prediction_sha256": difficulty.array_sha256(candidate),
            "elapsed_seconds": time.monotonic() - started,
        },
        input_record=test_input_record,
    )


def runtime_record() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
    }


def markdown_report(evidence: dict[str, Any]) -> str:
    result = evidence["result"]
    baseline = evidence["baseline"]
    candidate = evidence["candidate"]
    explanation = evidence["explanatory_metrics"]
    verdict = "통과" if result["passed"] else "기각"
    fold_lines = [
        "| 바깥쪽 분할 | 대조군 AUC | 후보군 AUC | 차이 | 개선 |",
        "| ---: | ---: | ---: | ---: | :---: |",
    ]
    for record in candidate["folds"]:
        fold_lines.append(
            f"| {record['fold']} | {record['baseline_auc']:.16f} | "
            f"{record['candidate_auc']:.16f} | {record['delta']:+.16f} | "
            f"{'예' if record['delta'] > 0.0 else '아니요'} |"
        )
    standard = explanation["condition_burden"]["standard"]["all_targets"]
    weighted = explanation["condition_burden"]["missingness_reweighted"][
        "all_targets"
    ]
    counts = explanation["condition_counts"]
    weighted_auc = explanation["weighted_oof_auc"]
    test = evidence["full_fit_test"]
    lines = [
        "# 자체 구성원 순위 사분위 범위 전용 선형 결합 판정",
        "",
        "## 결론",
        "",
        f"동결 후보는 **{verdict}**이다.",
        f"후보 nested OOF AUC는 `{candidate['nested_auc']:.16f}`이고 대조군 대비 차이는 `{candidate['delta']:+.16f}`다.",
        f"전체 개선 문턱 `+{MINIMUM_NESTED_DELTA:.5f}` 통과 여부는 `{result['passes_overall_delta']}`이고 바깥쪽 분할 5개 전부 개선 여부는 `{result['passes_all_folds']}`다.",
        "후보가 두 조건을 모두 통과하지 못하면 현행 최종 242개 결합을 유지한다.",
        "",
        "## 기준 재현",
        "",
        f"대조군 nested OOF AUC는 `{baseline['nested_auc']:.16f}`로 동결값과 절대 차이 `{baseline['absolute_delta']:.3e}`다.",
        f"대조군 예측 내용 해시는 `{baseline['prediction_sha256']}`로 동결 해시와 같다.",
        "",
        "## 분할별 결과",
        "",
        *fold_lines,
        "",
        "## 설명 지표",
        "",
        f"확인 구간에는 전체 `{counts['total']}`행, 목표값 0 `{counts['target_0']}`행, 목표값 1 `{counts['target_1']}`행이 들었다.",
        f"확인 구간의 표준 순위 손실 부담 평균 감소는 `{standard['burden_reduction']:+.16f}`다.",
        f"확인 구간의 결측 보정 순위 손실 부담 평균 감소는 `{weighted['burden_reduction']:+.16f}`다.",
        f"결측 보정 OOF AUC는 대조군 `{weighted_auc['baseline']:.16f}`, 후보군 `{weighted_auc['candidate']:.16f}`다.",
        "이 지표들은 설명 자료이며 채택 판정을 바꾸지 않는다.",
        "",
        "## 전체 자료 적합과 시험 자료 적용",
        "",
        f"전체 OOF의 확인 구간 학습 행은 `{test['train_condition_counts']['total']}`개다.",
        f"시험 자료에서는 `{test['test_condition_rows']}`행에 구간 전용 결합 순위를 적용했다.",
        f"후보 시험 예측 내용 해시는 `{test['candidate_prediction_sha256']}`다.",
        "제출 파일은 만들지 않았고 MLflow, 후보 풀과 champion도 바꾸지 않았다.",
        "",
        "## 산출물",
        "",
        f"후보 nested OOF 예측 내용 해시는 `{candidate['prediction_sha256']}`다.",
        "대용량 OOF와 시험 예측은 `run-logs/issue467`에 두고 커밋하지 않는다.",
        "기계 판독 근거는 `docs/research/own-member-iqr-specialized-stack-evidence.json`이다.",
        "",
    ]
    return "\n".join(lines)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )


def persist_outputs(
    inputs: difficulty.Inputs,
    comparison: NestedComparison,
    full_fit: FullFitResult,
    evidence: dict[str, Any],
) -> None:
    for path in (OUTPUT_DIR, EVIDENCE_PATH, REPORT_PATH):
        require(not path.exists(), f"기존 산출물이 있어 중단한다: {path}")
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".issue467-", dir=OUTPUT_DIR.parent)
    )
    temporary_output = temporary_root / "issue467"
    temporary_output.mkdir()
    try:
        oof_path = temporary_output / "candidate-oof.parquet"
        test_path = temporary_output / "candidate-test-prediction.parquet"
        manifest_path = temporary_output / "manifest.json"
        report_path = temporary_root / REPORT_PATH.name
        evidence_path = temporary_root / EVIDENCE_PATH.name

        pd.DataFrame(
            {
                ID: inputs.train[ID].to_numpy(),
                "fold": inputs.folds.to_numpy(dtype=np.int8),
                "baseline_prediction": comparison.baseline_prediction,
                "candidate_prediction": comparison.candidate_prediction,
                "in_condition": comparison.condition,
                "iqr_own_members": comparison.own_iqr,
            }
        ).to_parquet(oof_path, index=False, compression="zstd")
        pd.DataFrame(
            {
                ID: inputs.test[ID].to_numpy(),
                "candidate_prediction": full_fit.candidate_prediction,
                "in_condition": full_fit.condition,
                "iqr_own_members": full_fit.own_iqr,
            }
        ).to_parquet(test_path, index=False, compression="zstd")

        output_hashes = {
            "candidate-oof.parquet": difficulty.sha256_file(oof_path),
            "candidate-test-prediction.parquet": difficulty.sha256_file(test_path),
        }
        manifest = {
            "schema_version": 1,
            "issue": ISSUE,
            "contract": evidence["contract"],
            "inputs": {
                "sha256": inputs.input_manifest["sha256"],
                "member_order": list(inputs.members.columns),
                "test_members": full_fit.input_record,
            },
            "baseline": evidence["baseline"],
            "candidate": evidence["candidate"],
            "explanatory_metrics": evidence["explanatory_metrics"],
            "full_fit_test": evidence["full_fit_test"],
            "software": evidence["software"],
            "outputs": {
                "candidate-oof.parquet": {
                    "file_sha256": output_hashes["candidate-oof.parquet"],
                    "rows": int(len(inputs.train)),
                    "prediction_sha256": evidence["candidate"][
                        "prediction_sha256"
                    ],
                },
                "candidate-test-prediction.parquet": {
                    "file_sha256": output_hashes[
                        "candidate-test-prediction.parquet"
                    ],
                    "rows": int(len(inputs.test)),
                    "prediction_sha256": evidence["full_fit_test"][
                        "candidate_prediction_sha256"
                    ],
                },
            },
            "contract_violations": [],
        }
        write_json(manifest_path, manifest)
        report_path.write_text(markdown_report(evidence))
        evidence["artifacts"] = [
            {
                "path": "run-logs/issue467/candidate-oof.parquet",
                "sha256": output_hashes["candidate-oof.parquet"],
            },
            {
                "path": "run-logs/issue467/candidate-test-prediction.parquet",
                "sha256": output_hashes["candidate-test-prediction.parquet"],
            },
            {
                "path": "run-logs/issue467/manifest.json",
                "sha256": difficulty.sha256_file(manifest_path),
            },
            {
                "path": "docs/research/own-member-iqr-specialized-stack.md",
                "sha256": difficulty.sha256_file(report_path),
            },
        ]
        write_json(evidence_path, evidence)

        os.replace(temporary_output, OUTPUT_DIR)
        os.replace(report_path, REPORT_PATH)
        os.replace(evidence_path, EVIDENCE_PATH)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def execute(input_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    for path in (OUTPUT_DIR, EVIDENCE_PATH, REPORT_PATH):
        require(not path.exists(), f"기존 산출물이 있어 중단한다: {path}")
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT_DIR.parent / ".issue467.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ContractError(f"실험 잠금이 이미 있다: {lock_path}") from exc
    os.close(lock_fd)
    try:
        print("[1/5] 동결 입력과 242개 구성원 해시를 검사한다.", flush=True)
        inputs = difficulty.load_inputs(input_root)
        verify_input_hashes(inputs)

        print("[2/5] 대조군과 후보군 nested OOF를 함께 계산한다.", flush=True)
        comparison = evaluate_nested_comparison(inputs)
        baseline = verify_baseline(comparison, inputs)
        target = inputs.target.to_numpy(dtype=np.int8)
        candidate_auc = float(
            roc_auc_score(target, comparison.candidate_prediction)
        )
        candidate_delta = candidate_auc - BASELINE_NESTED_AUC
        passes_delta = candidate_delta >= MINIMUM_NESTED_DELTA
        passes_folds = all(record["delta"] > 0.0 for record in comparison.fold_records)
        passed = passes_delta and passes_folds

        print("[3/5] 설명 지표를 계산한다.", flush=True)
        explanatory = burden_summary(comparison, inputs)

        print("[4/5] 전체 자료로 맞추고 시험 예측을 계산한다.", flush=True)
        full_fit = fit_full_and_predict_test(input_root, inputs)

        candidate = {
            "name": "own_member_iqr_specialized_rank_logit",
            "nested_auc": candidate_auc,
            "delta": candidate_delta,
            "prediction_sha256": difficulty.array_sha256(
                comparison.candidate_prediction
            ),
            "folds": comparison.fold_records,
            "elapsed_seconds": comparison.elapsed_seconds,
        }
        evidence = {
            "schema_version": 1,
            "issue": ISSUE,
            "result": {
                "passed": passed,
                "verdict": (
                    "adopt_candidate"
                    if passed
                    else "reject_candidate_keep_current_242"
                ),
                "passes_overall_delta": passes_delta,
                "passes_all_folds": passes_folds,
            },
            "contract": {
                "baseline_run_id": difficulty.FINAL_RUN_ID,
                "member_count": 242,
                "own_member_count": 35,
                "external_member_count": 207,
                "condition": {
                    "metric": "iqr_own_members",
                    "lower_inclusive": LOWER_IQR,
                    "upper_exclusive": UPPER_IQR,
                    "rank_method": "average_ties_(rank_minus_0.5)_over_block_rows",
                    "quantile_method": "linear",
                },
                "global_combiner": "shrunk_rank_logit_logistic",
                "specialized_combiner": {
                    "representation": "rank_logit",
                    "standard_scaler": True,
                    "logistic_regression": {
                        "C": 1.0,
                        "solver": "lbfgs",
                        "max_iter": 1_000,
                        "random_state": 0,
                        "logit_clip": 1e-6,
                    },
                },
                "minimum_nested_delta": MINIMUM_NESTED_DELTA,
                "required_positive_outer_folds": 5,
                "baseline_tolerance": BASELINE_TOLERANCE,
            },
            "inputs": {
                "root": str(input_root.resolve()),
                "sha256": inputs.input_manifest["sha256"],
                "member_order_sha256": text_list_sha256(
                    list(inputs.members.columns)
                ),
            },
            "baseline": baseline,
            "candidate": candidate,
            "explanatory_metrics": explanatory,
            "full_fit_test": full_fit.record,
            "software": runtime_record(),
            "script_sha256": difficulty.sha256_file(Path(__file__).resolve()),
            "execution": {
                "total_seconds_before_writes": time.monotonic() - started,
            },
            "contract_violations": [],
        }

        print("[5/5] 기계 판독 근거와 연구 문서를 확정한다.", flush=True)
        persist_outputs(inputs, comparison, full_fit, evidence)
        print(
            f"완료: 대조군 {baseline['nested_auc']:.16f}, "
            f"후보 {candidate_auc:.16f}, 차이 {candidate_delta:+.16f}, "
            f"판정 {'통과' if passed else '기각'}",
            flush=True,
        )
        return evidence
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="자체 구성원 순위 사분위 범위 전용 선형 결합 판정"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="동결 자료와 MLflow 실행 기록이 있는 저장소 루트",
    )
    args = parser.parse_args()
    try:
        execute(args.input_root.resolve())
    except (ContractError, difficulty.ContractError) as exc:
        raise SystemExit(f"판정 불가: {exc}") from exc


if __name__ == "__main__":
    main()
