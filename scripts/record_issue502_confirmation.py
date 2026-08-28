"""이슈 502의 세 시드 결측 증강 확인 결과를 재계산해 JSON과 Markdown으로 기록한다."""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.data import ID, TARGET, file_sha256
from pipeline.judgment import check_canaries, mean_gain_of
from pipeline.runs import MlflowRunStore


RUN_IDS = {
    "original": "4f4023e8fd35428fb2a5f07421768e02",
    "tripled": "7109552edcda4dbd97c7e68f87266aa2",
    "missingness_augmented": "e46d1ca38e0746209e049970d3dd2ab6",
}
FAILED_ATTEMPT_RUN_ID = "ee17ca6cf05647129b56d5761816e084"
SEEDS = (42, 43, 44)
FOLDS = (0, 1, 2, 3, 4)
ISSUE_URL = "https://github.com/tmheo/predicting-smartphone-addiction/issues/502"
NEXT_ISSUE_URL = "https://github.com/tmheo/predicting-smartphone-addiction/issues/503"
ARTIFACT_NAMES = (
    "feature_importance.parquet",
    "fold_feature_reuse.json",
    "fold_recovery.json",
    "model_training_diagnostics.json",
    "observability/fold_execution.jsonl.gz",
    "oof.parquet",
    "oof_seed_42.parquet",
    "oof_seed_43.parquet",
    "oof_seed_44.parquet",
    "seed_reuse.json",
    "submission.csv",
    "test_pred.parquet",
    "training_row_evidence.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("artifacts/issue502-three-seed-missingness-confirmation.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("docs/research/missingness-augmentation-three-seed-confirmation.md"),
    )
    return parser.parse_args()


def _json_artifact(store: MlflowRunStore, run_id: str, name: str) -> Any:
    return json.loads(store.artifact_bytes_of(run_id, name))


def _parquet_artifact(store: MlflowRunStore, run_id: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(store.artifact_bytes_of(run_id, name)))


def _assert_close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > 1e-12:
        raise AssertionError(f"{label} 재계산값 {actual}이 기록값 {expected}와 다르다.")


def _with_target(oof: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    return oof.merge(targets, on=ID, validate="one_to_one")


def _run_record(
    store: MlflowRunStore,
    arm: str,
    run_id: str,
    targets: pd.DataFrame,
) -> dict[str, Any]:
    meta = store.facts_of(run_id)
    if meta.status != "FINISHED":
        raise AssertionError(f"{arm} 실행 {run_id}의 상태가 FINISHED가 아니다: {meta.status}")
    seeds = tuple(int(seed) for seed in meta.params["seeds"].split(","))
    if seeds != SEEDS:
        raise AssertionError(f"{arm} 시드가 고정값과 다르다: {seeds}")
    if meta.tags["git_dirty"] != "False":
        raise AssertionError(f"{arm} 실행의 작업 트리가 깨끗하지 않다.")

    averaged_oof = _with_target(
        _parquet_artifact(store, run_id, "oof.parquet"), targets
    )
    recomputed_auc = float(roc_auc_score(averaged_oof[TARGET], averaged_oof["pred"]))
    _assert_close(recomputed_auc, meta.metrics["auc_oof"], f"{arm} 전체 OOF AUC")
    fold_auc = {}
    for fold in FOLDS:
        part = averaged_oof.loc[averaged_oof["fold"] == fold]
        score = float(roc_auc_score(part[TARGET], part["pred"]))
        _assert_close(score, meta.metrics[f"auc_fold_{fold}"], f"{arm} 분할 {fold} AUC")
        fold_auc[str(fold)] = score

    seed_auc = {}
    seed_oof_sha256 = {}
    for seed in SEEDS:
        artifact_name = f"oof_seed_{seed}.parquet"
        seed_oof = _with_target(
            _parquet_artifact(store, run_id, artifact_name), targets
        )
        score = float(roc_auc_score(seed_oof[TARGET], seed_oof["pred"]))
        _assert_close(score, meta.metrics[f"auc_oof_seed_{seed}"], f"{arm} 시드 {seed} AUC")
        seed_auc[str(seed)] = score
        seed_oof_sha256[str(seed)] = store.artifact_sha256_of(run_id, artifact_name)

    config_name = f"{meta.params['experiment']}.yaml"
    artifact_sha256 = {
        name: store.artifact_sha256_of(run_id, name)
        for name in (config_name, *ARTIFACT_NAMES)
    }
    evidence = _json_artifact(store, run_id, "training_row_evidence.json")["entries"]
    diagnostics = _json_artifact(store, run_id, "model_training_diagnostics.json")
    reuse = _json_artifact(store, run_id, "seed_reuse.json")

    features = set(meta.params["features"].split(","))
    canary = check_canaries(features, mean_gain_of(store.importance_of(run_id)))
    canary_record = {
        "ok": canary.ok,
        "placebo_gain": canary.placebo_gain,
        "checks": [
            {"feature": check.feature, "gain": check.gain, "ok": check.ok}
            for check in canary.checks
        ],
    }
    if not canary.ok:
        raise AssertionError(f"{arm} 목표 통계 인코딩 카나리아 검사가 실패했다.")

    training_length = {
        f"seed_{item['seed']}_fold_{item['fold']}": item["training_length_evidence"][
            "observations"
        ][0]["observed_training_length"]
        for item in diagnostics
    }
    return {
        "run_id": run_id,
        "status": meta.status,
        "experiment": meta.params["experiment"],
        "git_commit": meta.tags["git_commit"],
        "git_dirty": False,
        "auc_oof": recomputed_auc,
        "auc_oof_weighted": meta.metrics["auc_oof_weighted"],
        "seed_auc": seed_auc,
        "fold_auc": fold_auc,
        "input_sha256": {
            "train": meta.tags["sha256.train"],
            "test": meta.tags["sha256.test"],
            "folds": meta.tags["sha256.folds"],
        },
        "artifact_sha256": artifact_sha256,
        "seed_oof_sha256": seed_oof_sha256,
        "seed_reuse": reuse,
        "training_length": training_length,
        "training_row_evidence": evidence,
        "target_encoding_importance_canary": canary_record,
    }


def _assert_common_run_identity(runs: dict[str, dict[str, Any]]) -> None:
    commits = {run["git_commit"] for run in runs.values()}
    inputs = {json.dumps(run["input_sha256"], sort_keys=True) for run in runs.values()}
    if len(commits) != 1:
        raise AssertionError(f"세 학습군의 실행 커밋이 다르다: {sorted(commits)}")
    if len(inputs) != 1:
        raise AssertionError("세 학습군의 입력 해시가 다르다.")


def _comparison(runs: dict[str, dict[str, Any]], train_path: Path) -> dict[str, Any]:
    direct_delta = (
        runs["missingness_augmented"]["auc_oof"] - runs["tripled"]["auc_oof"]
    )
    seed_delta = {
        str(seed): runs["missingness_augmented"]["seed_auc"][str(seed)]
        - runs["tripled"]["seed_auc"][str(seed)]
        for seed in SEEDS
    }
    fold_delta = {
        str(fold): runs["missingness_augmented"]["fold_auc"][str(fold)]
        - runs["tripled"]["fold_auc"][str(fold)]
        for fold in FOLDS
    }

    store = MlflowRunStore()
    oof = {
        arm: _parquet_artifact(store, run["run_id"], "oof.parquet")
        for arm, run in runs.items()
    }
    reference = oof["original"][[ID, "fold"]].sort_values(ID).reset_index(drop=True)
    for arm, frame in oof.items():
        identity = frame[[ID, "fold"]].sort_values(ID).reset_index(drop=True)
        if not identity.equals(reference):
            raise AssertionError(f"{arm} OOF 행·분할이 다른 학습군과 다르다.")

    missing_evidence = runs["missingness_augmented"]["training_row_evidence"]
    raw_columns = missing_evidence[0]["raw_columns"]
    train = pd.read_csv(train_path, usecols=[ID, TARGET, *raw_columns])
    train["missing_count"] = train[raw_columns].isna().sum(axis=1)
    joined = train[[ID, TARGET, "missing_count"]].copy()
    for arm, frame in oof.items():
        joined = joined.merge(
            frame[[ID, "pred"]].rename(columns={"pred": f"pred_{arm}"}),
            on=ID,
            validate="one_to_one",
        )

    buckets = []
    for label, mask in (
        ("0", joined["missing_count"] == 0),
        ("1", joined["missing_count"] == 1),
        ("2", joined["missing_count"] == 2),
        ("3", joined["missing_count"] == 3),
        ("4", joined["missing_count"] == 4),
        ("5+", joined["missing_count"] >= 5),
    ):
        part = joined.loc[mask]
        auc = {
            arm: float(roc_auc_score(part[TARGET], part[f"pred_{arm}"]))
            for arm in runs
        }
        buckets.append(
            {
                "bucket": label,
                "row_count": int(len(part)),
                "share": float(len(part) / len(joined)),
                "auc": auc,
                "missingness_augmented_minus_tripled": (
                    auc["missingness_augmented"] - auc["tripled"]
                ),
                "missingness_augmented_minus_original": (
                    auc["missingness_augmented"] - auc["original"]
                ),
            }
        )

    return {
        "auc_oof": {arm: run["auc_oof"] for arm, run in runs.items()},
        "auc_oof_weighted": {
            arm: run["auc_oof_weighted"] for arm, run in runs.items()
        },
        "delta_vs_original": {
            "tripled": runs["tripled"]["auc_oof"] - runs["original"]["auc_oof"],
            "missingness_augmented": (
                runs["missingness_augmented"]["auc_oof"] - runs["original"]["auc_oof"]
            ),
        },
        "missingness_augmented_minus_tripled": direct_delta,
        "seed_missingness_augmented_minus_tripled": seed_delta,
        "fold_missingness_augmented_minus_tripled": fold_delta,
        "missing_count_buckets": buckets,
    }


def _training_row_integrity(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence = {
        arm: run["training_row_evidence"] for arm, run in runs.items()
    }
    for arm, entries in evidence.items():
        if len(entries) != len(SEEDS) * len(FOLDS):
            raise AssertionError(f"{arm} 학습 행 증거가 15개가 아니다: {len(entries)}")
        if not all(all(item["assertions"].values()) for item in entries):
            raise AssertionError(f"{arm} 학습 행 단언 가운데 실패가 있다.")
        if any(item["validation_augmented"] or item["test_augmented"] for item in entries):
            raise AssertionError(f"{arm} 검증 또는 시험 자료가 증강됐다.")

    tripled = {(item["seed"], item["outer_fold"]): item for item in evidence["tripled"]}
    missingness = {
        (item["seed"], item["outer_fold"]): item
        for item in evidence["missingness_augmented"]
    }
    identity_fields = (
        "parent_id_order",
        "parent_source_index_order",
        "target_order",
        "row_id_order",
        "training_row_count",
        "original_row_count",
        "replica_row_count",
    )
    pair_identity_matches = all(
        all(tripled[key][field] == missingness[key][field] for field in identity_fields)
        for key in tripled
    )
    if not pair_identity_matches:
        raise AssertionError("3배 행 대조군과 결측 증강군의 부모 행·목표값 순서가 다르다.")

    mask_records = []
    for item in evidence["missingness_augmented"]:
        canary = item["placebo_canary"]
        if not all(
            (
                canary["actual_missing_mask_matches_source"],
                canary["missing_mask_rebuilt_from_augmented_raw_source"],
                canary["parent_noise_is_shared_by_replica"],
            )
        ):
            raise AssertionError("목표 통계 인코딩 결측 카나리아가 실패했다.")
        for replica in item["replicas"]:
            mask_records.append(
                {
                    "seed": item["seed"],
                    "fold": item["outer_fold"],
                    "replica": replica["replica_index"],
                    "mask_seed": replica["mask_seed"],
                    "mask_sha256": replica["mask_sha256"],
                    "eligible_observed_cells": replica["eligible_observed_cells"],
                    "added_missing_cells": replica["added_missing_cells"],
                    "actual_added_missing_rate": replica["actual_added_missing_rate"],
                }
            )

    hashes = [item["mask_sha256"] for item in mask_records]
    all_masks_unique = len(set(hashes)) == len(hashes)
    if not all_masks_unique:
        raise AssertionError("결측 증강 마스크 해시가 모두 고유하지 않다.")
    eligible = sum(item["eligible_observed_cells"] for item in mask_records)
    added = sum(item["added_missing_cells"] for item in mask_records)

    return {
        "all_assertions_passed": True,
        "pair_identity_matches": pair_identity_matches,
        "validation_and_test_not_augmented": True,
        "target_encoding_canary_valid": True,
        "target_encoding_importance_canary": {
            arm: run["target_encoding_importance_canary"] for arm, run in runs.items()
        },
        "mask_count": len(mask_records),
        "all_mask_hashes_unique": all_masks_unique,
        "mask_records": mask_records,
        "missingness_augmented_eligible_observed_cells": eligible,
        "missingness_augmented_added_missing_cells": added,
        "missingness_augmented_added_missing_rate": added / eligible,
    }


def _decision(comparison: dict[str, Any]) -> dict[str, Any]:
    direct_delta = comparison["missingness_augmented_minus_tripled"]
    seed_wins = sum(
        delta > 0
        for delta in comparison["seed_missingness_augmented_minus_tripled"].values()
    )
    fold_wins = sum(
        delta > 0
        for delta in comparison["fold_missingness_augmented_minus_tripled"].values()
    )
    mean_gate = direct_delta >= 0.00002
    seed_gate = seed_wins >= 2
    fold_gate_required = direct_delta < 0.0002
    fold_gate = not fold_gate_required or fold_wins >= 3
    passed = mean_gate and seed_gate and fold_gate
    return {
        "status": "pass" if passed else "fail",
        "rule": {
            "mean_delta_minimum": 0.00002,
            "seed_wins_minimum": 2,
            "fold_gate_trigger_below": 0.0002,
            "fold_wins_minimum_when_triggered": 3,
        },
        "observed": {
            "mean_delta": direct_delta,
            "seed_wins": seed_wins,
            "fold_gate_required": fold_gate_required,
            "fold_wins": fold_wins,
        },
        "gates": {
            "mean_delta": mean_gate,
            "seed_wins": seed_gate,
            "fold_wins": fold_gate,
        },
        "next_step": (
            "build_pre_registered_candidate_pool_in_issue_503"
            if passed
            else "close_issue_503_as_not_applicable"
        ),
    }


def _failed_attempt(store: MlflowRunStore) -> dict[str, Any]:
    meta = store.facts_of(FAILED_ATTEMPT_RUN_ID)
    if meta.status != "FAILED":
        raise AssertionError("선언한 실패 실행의 상태가 FAILED가 아니다.")
    return {
        "run_id": FAILED_ATTEMPT_RUN_ID,
        "status": meta.status,
        "git_commit": meta.tags["git_commit"],
        "error_stage": meta.tags.get("error.stage"),
        "error_type": meta.tags.get("error.type"),
        "cause": "재사용 시드의 정렬된 피처 이름이 현재 피처 계획의 선언 순서와 달라 최종 평가 단언에서 중단됐다.",
        "resolution": "재사용 자료의 피처 집합은 검증하되 현재 피처 계획의 선언 순서를 보존하도록 수정했다.",
        "results_used": False,
    }


def _report(record: dict[str, Any]) -> str:
    runs = record["runs"]
    comparison = record["comparison"]
    decision = record["decision"]
    integrity = record["training_row_integrity"]
    lines = [
        "# 결측 증강 세 시드 확인",
        "",
        f"이 문서는 [세 시드 평균으로 결측 증강 효과를 확인한다]({ISSUE_URL})의 실행 결과와 판정을 기록한다.",
        "",
        "## 결론",
        "",
        f"결측 증강군은 세 시드 평균 OOF AUC에서 직접 대조군인 3배 행 대조군보다 `{comparison['missingness_augmented_minus_tripled']:+.10f}` 높아 사전 고정 관문을 통과했다.",
        "평균 차이 관문의 최소값은 `+0.0000200000`이었다.",
        f"시드별 승수는 `{decision['observed']['seed_wins']}/3`, 평균 분할별 승수는 `{decision['observed']['fold_wins']}/5`로 두 보조 관문도 통과했다.",
        f"따라서 다음 단계는 [사전 등록 후보군을 구성한다]({NEXT_ISSUE_URL})이다.",
        "이 판정은 결측 증강 후보군 구성을 허용하며, 아직 최종 채택이나 제출을 뜻하지 않는다.",
        "",
        "## 전체 결과",
        "",
        "| 학습군 | 세 시드 OOF AUC | 원본 기준 차이 | 가중 OOF AUC |",
        "| --- | ---: | ---: | ---: |",
        f"| 일반 기준군 | `{runs['original']['auc_oof']:.10f}` | 기준 | `{runs['original']['auc_oof_weighted']:.10f}` |",
        f"| 3배 행 대조군 | `{runs['tripled']['auc_oof']:.10f}` | `{comparison['delta_vs_original']['tripled']:+.10f}` | `{runs['tripled']['auc_oof_weighted']:.10f}` |",
        f"| 결측 증강군 | `{runs['missingness_augmented']['auc_oof']:.10f}` | `{comparison['delta_vs_original']['missingness_augmented']:+.10f}` | `{runs['missingness_augmented']['auc_oof_weighted']:.10f}` |",
        "",
        "결측 증강군과 3배 행 대조군의 직접 차이는 단순 행 복제 효과와 분리된 결측 증강 효과를 나타낸다.",
        "",
        "## 시드별 짝차이",
        "",
        "| 시드 | 일반 기준군 | 3배 행 대조군 | 결측 증강군 | 결측 증강군 - 3배 행 대조군 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        key = str(seed)
        lines.append(
            f"| {seed} | `{runs['original']['seed_auc'][key]:.10f}` | `{runs['tripled']['seed_auc'][key]:.10f}` | `{runs['missingness_augmented']['seed_auc'][key]:.10f}` | `{comparison['seed_missingness_augmented_minus_tripled'][key]:+.10f}` |"
        )
    lines.extend(
        [
            "",
            "세 시드가 모두 양수여서 평균 개선이 특정 시드 하나에 의존하지 않았다.",
            "",
            "## 평균 분할별 짝차이",
            "",
            "| 분할 | 결측 증강군 - 3배 행 대조군 |",
            "| ---: | ---: |",
        ]
    )
    for fold in FOLDS:
        key = str(fold)
        lines.append(
            f"| {fold} | `{comparison['fold_missingness_augmented_minus_tripled'][key]:+.10f}` |"
        )
    lines.extend(
        [
            "",
            "직접 차이가 `0.0002`보다 작아 분할 승수 관문이 적용됐고, 다섯 분할이 모두 양수여서 요구한 3/5를 넘었다.",
            "",
            "## 검증 행의 기존 결측 수별 결과",
            "",
            "| 기존 결측 수 | 행 수 | 전체 비중 | 결측 증강군 - 3배 행 대조군 | 결측 증강군 - 일반 기준군 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for bucket in comparison["missing_count_buckets"]:
        lines.append(
            f"| {bucket['bucket']} | {bucket['row_count']:,} | `{bucket['share']:.2%}` | `{bucket['missingness_augmented_minus_tripled']:+.10f}` | `{bucket['missingness_augmented_minus_original']:+.10f}` |"
        )
    lines.extend(
        [
            "",
            "결측 수 구간은 원자료의 12개 학습 피처에서 검증 행이 원래 가지고 있던 결측 개수로 나눴다.",
            "",
            "## 실행 무결성",
            "",
            f"세 실행은 커밋 `{runs['original']['git_commit']}`의 깨끗한 작업 트리에서 같은 로컬 기계로 순차 실행했다.",
            f"원본 학습 자료 SHA-256은 `{runs['original']['input_sha256']['train']}`이고 분할 파일 SHA-256은 `{runs['original']['input_sha256']['folds']}`이다.",
            "시드 42는 같은 설정·입력·의존성·OOF·분할·시험 행을 재검증한 뒤 기존 선별 실행에서 재사용했고, 시드 43과 44는 새로 학습했다.",
            f"결측 증강군은 원래 관측된 학습 셀 {integrity['missingness_augmented_eligible_observed_cells']:,}개 중 {integrity['missingness_augmented_added_missing_cells']:,}개를 추가로 비워 실측 비율 `{integrity['missingness_augmented_added_missing_rate']:.8%}`를 만들었다.",
            f"시드·분할·복제본별 마스크 {integrity['mask_count']}개는 모두 서로 다른 결정적 해시를 가졌다.",
            "3배 행 대조군과 결측 증강군은 부모 행 순서, 원본 행 번호 순서, 목표값 순서, 전체 행 수가 일치했다.",
            "검증 및 시험 자료에는 증강을 적용하지 않았다.",
            "목표 통계 인코딩은 복제본을 통계 산출에서 제외하고 부모 분할·목표값·위약 잡음을 물려받으며 증강 원자료에서 결측 마스크만 다시 계산한다는 검사를 통과했다.",
            "피처 중요도에서도 목표 통계 인코딩 위약 카나리아가 원본 위약 피처보다 낮아 누수 관문을 통과했다.",
            "",
            "| 학습군 | 중앙 MLflow 실행 ID | OOF SHA-256 | 시험 예측 SHA-256 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for arm, label in (
        ("original", "일반 기준군"),
        ("tripled", "3배 행 대조군"),
        ("missingness_augmented", "결측 증강군"),
    ):
        lines.append(
            f"| {label} | `{runs[arm]['run_id']}` | `{runs[arm]['artifact_sha256']['oof.parquet']}` | `{runs[arm]['artifact_sha256']['test_pred.parquet']}` |"
        )
    failed = record["failed_attempt"]
    lines.extend(
        [
            "",
            f"첫 원본 실행 `{failed['run_id']}`은 재사용 시드의 피처 이름 순서를 잘못 보존해 최종 평가에서 실패했다.",
            "실패 원인을 실제 실행으로 재현한 뒤 현재 피처 계획의 선언 순서를 보존하도록 고쳤으며, 실패 실행의 결과는 판정에 사용하지 않았다.",
            "수치, 시드·분할별 결과, 결측 수 구간, 입력·산출물 해시, 마스크 기록, 재사용 검증 내용은 `artifacts/issue502-three-seed-missingness-confirmation.json`에 기계 판독 가능한 형태로 보존한다.",
            "",
            "## 학습 길이 관찰",
            "",
            "세 배 행 대조군과 결측 증강군의 다수 분할이 20,000회 상한에 닿거나 매우 가까이 갔다.",
            "현재 지도 판정은 학습 길이를 포함해 사전 고정한 조건을 유지했으므로 이 관찰 때문에 설정을 바꾸지 않았다.",
            "학습 길이 확대 효과는 현재 지도 밖의 별도 실험으로 다뤄야 한다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    store = MlflowRunStore()
    train_path = Path(store.config_of(RUN_IDS["original"])["data"]["train"])
    targets = pd.read_csv(train_path, usecols=[ID, TARGET])
    runs = {
        arm: _run_record(store, arm, run_id, targets)
        for arm, run_id in RUN_IDS.items()
    }
    _assert_common_run_identity(runs)
    comparison = _comparison(runs, train_path)
    integrity = _training_row_integrity(runs)
    decision = _decision(comparison)
    if decision["status"] != "pass":
        raise AssertionError(f"이슈 502의 사전 고정 관문을 통과하지 못했다: {decision}")

    source_reuse = runs["original"]["seed_reuse"]["records"][0]
    input_sha256 = runs["original"]["input_sha256"]
    if input_sha256["train"] != file_sha256(train_path):
        raise AssertionError("현재 학습 자료 해시가 실행 기록과 다르다.")
    record = {
        "schema_version": 1,
        "issue": {"number": 502, "url": ISSUE_URL},
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "freeze": {
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
            "execution_commit": runs["original"]["git_commit"],
            "execution_tree_clean": True,
            "source_seed_42_commit": source_reuse["source_git_commit"],
            "source_seed_42_stage": source_reuse["source_stage"],
            "input_sha256": input_sha256,
            "model_dependencies": source_reuse["model_dependencies"],
            "execution_environment": {
                "source": "local_measured",
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "remote_provider": None,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "settings_changed_after_results": False,
            "public_repleaf_used": False,
            "kaggle_submission_uploaded": False,
        },
        "runs": runs,
        "comparison": comparison,
        "training_row_integrity": integrity,
        "decision": decision,
        "failed_attempt": _failed_attempt(store),
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    args.report_output.write_text(_report(record))
    print(f"status={decision['status']}")
    print(
        "direct_delta="
        f"{comparison['missingness_augmented_minus_tripled']:+.10f} "
        f"seed_wins={decision['observed']['seed_wins']}/3 "
        f"fold_wins={decision['observed']['fold_wins']}/5"
    )
    print(f"json={args.json_output}")
    print(f"report={args.report_output}")


if __name__ == "__main__":
    main()
