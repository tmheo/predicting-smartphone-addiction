"""이슈 #623 실행 기록 묶음(로컬 나무 12개, 원격 RealMLP 4개)을 main MLflow에 반입하고 예측 무결성을 감사한다.

#483의 import-and-audit을 여러 묶음으로 일반화했다. 같은 출처 실행이 이미 반입돼 있으면 재사용한다.

main 작업 폴더에서 실행한다:
    uv run --frozen python scripts/issue623/import_and_audit.py \\
        --bundle run-logs/issue623/bundles/exp117_ag25_gbm_r21.bundle.zip ... \\
        --expected-commit <커밋> --out run-logs/issue623/import-audit-local.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.bundle import MANIFEST_NAME, import_bundle
from pipeline.data import ID, TARGET, file_sha256
from pipeline.tracking import mlflow_client


EXPECTED_INPUT_SHA = {
    "train": "f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c",
    "test": "8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e",
    "folds": "5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4",
}
EXPECTED_SAMPLE_SHA = "206763fe5786fb9c80d4e9289a3b812030d3dbb36450c6eb63348098154ce63e"
REQUIRED_ARTIFACTS = {
    "oof.parquet",
    "oof_seed_42.parquet",
    "oof_seed_43.parquet",
    "oof_seed_44.parquet",
    "test_pred.parquet",
    "submission.csv",
    "feature_importance.parquet",
    "fold_feature_reuse.json",
    "model_training_diagnostics.json",
    "feature_importance_summary.csv",
    "stage_durations.csv",
    "summary.html",
}
SEEDS = (42, 43, 44)


def bundle_manifest(bundle: Path) -> dict:
    with zipfile.ZipFile(bundle) as archive:
        return json.loads(archive.read(MANIFEST_NAME))


def artifact_paths(client, run_id: str, root: str = "") -> set[str]:
    paths: set[str] = set()
    for item in client.list_artifacts(run_id, root):
        if item.is_dir:
            paths.update(artifact_paths(client, run_id, item.path))
        else:
            paths.add(item.path)
    return paths


def artifact_path(client, run_id: str, name: str) -> Path:
    return Path(client.download_artifacts(run_id, name))


def import_or_reuse(client, experiment_id: str, bundle: Path) -> tuple[str, bool]:
    source_run_id = bundle_manifest(bundle).get("source_run_id")
    if not isinstance(source_run_id, str):
        raise RuntimeError(f"{bundle}의 출처 실행 식별자가 문자열이 아니다.")
    existing = client.search_runs(
        [experiment_id], filter_string=f"tags.`source.run_id` = '{source_run_id}'"
    )
    if len(existing) > 1:
        raise RuntimeError(f"출처 실행 {source_run_id}의 로컬 반입본이 여러 개다.")
    if existing:
        run = existing[0]
        if run.data.tags.get("import.bundle_sha256") != file_sha256(bundle):
            raise RuntimeError(f"기존 반입 실행 {run.info.run_id}의 묶음 해시가 현재 파일과 다르다.")
        return run.info.run_id, True
    return import_bundle(bundle), False


def aligned_oof_prediction(frame, expected_ids, expected_folds, label):
    if list(frame.columns) != [ID, "fold", "pred"]:
        raise RuntimeError(f"{label} 열 구성이 다르다: {list(frame.columns)}")
    if frame[ID].duplicated().any() or len(frame) != len(expected_ids):
        raise RuntimeError(f"{label} 식별자가 중복됐거나 행 수가 다르다.")
    indexed = frame.set_index(ID).reindex(expected_ids)
    if indexed[["fold", "pred"]].isna().any().any():
        raise RuntimeError(f"{label} 식별자 집합이 입력과 다르다.")
    if not np.array_equal(indexed["fold"].to_numpy(), expected_folds.to_numpy()):
        raise RuntimeError(f"{label} fold 배정이 고정 입력과 다르다.")
    prediction = indexed["pred"].to_numpy(dtype=float)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{label} 예측에 유한하지 않은 값이 있다.")
    return prediction


def aligned_test_prediction(frame, expected_ids, label):
    if list(frame.columns) != [ID, "pred"]:
        raise RuntimeError(f"{label} 열 구성이 다르다: {list(frame.columns)}")
    if frame[ID].duplicated().any() or len(frame) != len(expected_ids):
        raise RuntimeError(f"{label} 식별자가 중복됐거나 행 수가 다르다.")
    indexed = frame.set_index(ID).reindex(expected_ids)
    if indexed["pred"].isna().any():
        raise RuntimeError(f"{label} 식별자 집합이 입력과 다르다.")
    prediction = indexed["pred"].to_numpy(dtype=float)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{label} 예측에 유한하지 않은 값이 있다.")
    return prediction


def audit_one(client, experiment_id, bundle: Path, inputs: dict, args) -> dict:
    train, test, sample, expected_folds = inputs["train"], inputs["test"], inputs["sample"], inputs["folds"]
    manifest = bundle_manifest(bundle)
    experiment = manifest.get("params", {}).get("experiment")
    if not experiment:
        raise RuntimeError(f"{bundle} manifest에 실험 이름이 없다.")
    run_id, reused = import_or_reuse(client, experiment_id, bundle)
    run = client.get_run(run_id)
    tags = dict(run.data.tags)
    params = dict(run.data.params)
    if run.info.status != "FINISHED":
        raise RuntimeError(f"{run_id} 상태가 FINISHED가 아니다: {run.info.status}")
    expected_tags = {"git_commit": args.expected_commit, "git_dirty": "False"}
    remote = experiment in set(args.remote_experiments)
    if remote:
        if args.remote_job_id:
            expected_tags["remote.job_id"] = args.remote_job_id
        if args.remote_provider:
            expected_tags["remote.provider"] = args.remote_provider
    for key, expected in expected_tags.items():
        if tags.get(key) != expected:
            raise RuntimeError(f"{run_id} 태그 {key} 불일치: {tags.get(key)!r} != {expected!r}")
    for name, digest in EXPECTED_INPUT_SHA.items():
        if tags.get(f"sha256.{name}") != digest:
            raise RuntimeError(f"{run_id} 입력 해시 {name}이 다르다.")
    if params.get("experiment") != experiment:
        raise RuntimeError(f"{run_id} 실험 이름이 다르다: {params.get('experiment')}")
    if params.get("stage") != "confirm" or params.get("seeds") != ",".join(map(str, SEEDS)):
        raise RuntimeError(f"{run_id} 확정 단계 또는 난수 목록이 다르다.")
    missing = sorted(REQUIRED_ARTIFACTS - artifact_paths(client, run_id))
    if missing:
        raise RuntimeError(f"{run_id} 필수 산출물 누락: {missing}")

    oof = pd.read_parquet(artifact_path(client, run_id, "oof.parquet"))
    test_pred = pd.read_parquet(artifact_path(client, run_id, "test_pred.parquet"))
    oof_by_id = aligned_oof_prediction(oof, train[ID], expected_folds, "OOF")
    test_by_id = aligned_test_prediction(test_pred, test[ID], "시험 예측")
    auc = float(roc_auc_score(train[TARGET].to_numpy(), oof_by_id))
    stored_auc = float(run.data.metrics["auc_oof"])
    if not math.isclose(auc, stored_auc, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"{run_id} OOF 재채점 불일치: {auc} != {stored_auc}")
    seed_predictions = []
    seed_aucs = {}
    seed_fold_aucs = {}
    fold_values = expected_folds.to_numpy()
    labels = train[TARGET].to_numpy()
    for seed in SEEDS:
        frame = pd.read_parquet(artifact_path(client, run_id, f"oof_seed_{seed}.parquet"))
        prediction = aligned_oof_prediction(frame, train[ID], expected_folds, f"난수 {seed} OOF")
        seed_predictions.append(prediction)
        seed_aucs[seed] = float(roc_auc_score(labels, prediction))
        stored = float(run.data.metrics[f"auc_oof_seed_{seed}"])
        if not math.isclose(seed_aucs[seed], stored, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f"{run_id} 난수 {seed} OOF 재채점 불일치: {seed_aucs[seed]} != {stored}")
        seed_fold_aucs[seed] = {
            int(fold): float(roc_auc_score(labels[fold_values == fold], prediction[fold_values == fold]))
            for fold in sorted(np.unique(fold_values))
        }
    if not np.allclose(np.mean(seed_predictions, axis=0), oof_by_id, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{run_id} 대표 OOF가 세 난수 평균과 다르다.")
    submission = pd.read_csv(artifact_path(client, run_id, "submission.csv"))
    if list(submission.columns) != [ID, TARGET] or list(submission[ID]) != list(sample[ID]):
        raise RuntimeError(f"{run_id} 제출 파일 열 또는 식별자 순서가 다르다.")
    submitted = submission[TARGET].to_numpy(dtype=float)
    if not np.isfinite(submitted).all() or not np.allclose(submitted, test_by_id, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{run_id} 제출 예측이 시험 예측과 다르다.")
    importance = pd.read_parquet(artifact_path(client, run_id, "feature_importance.parquet"))
    if (
        importance.empty
        or not {"feature", "fold", "seed", "gain"}.issubset(importance.columns)
        or not np.isfinite(importance["gain"].to_numpy(dtype=float)).all()
    ):
        raise RuntimeError(f"{run_id} 중요도 자료가 비었거나 유한하지 않다.")
    if manifest.get("source_run_id") != tags.get("source.run_id", ""):
        raise RuntimeError(f"{run_id} 묶음 출처와 반입 계보가 다르다.")

    return {
        "experiment": experiment,
        "git_commit": args.expected_commit,
        "provider": tags.get("remote.provider", "local"),
        "remote_job_id": tags.get("remote.job_id"),
        "bundle": str(bundle),
        "bundle_sha256": file_sha256(bundle),
        "source_run_id": tags.get("source.run_id"),
        "main_run_id": run_id,
        "reused_existing_import": reused,
        "auc_oof": auc,
        "auc_oof_seed": seed_aucs,
        "auc_oof_weighted": float(run.data.metrics.get("auc_oof_weighted", float("nan"))),
        "fold_aucs": {
            key: float(value) for key, value in run.data.metrics.items() if key.startswith("auc_fold_")
        },
        "seed_fold_aucs": seed_fold_aucs,
        "prediction_integrity_pass": True,
        "oof_content_sha256": hashlib.sha256(np.ascontiguousarray(oof_by_id).tobytes()).hexdigest(),
        "test_content_sha256": hashlib.sha256(np.ascontiguousarray(test_by_id).tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--remote-job-id", default=None, help="원격 실행 묶음의 remote.job_id 태그")
    parser.add_argument("--remote-provider", default=None, help="원격 실행 묶음의 remote.provider 태그")
    parser.add_argument(
        "--remote-experiments",
        default="exp139_realmlp_reference_qnormal_train_test,cdv2_realmlp_raw4,cdv2_realmlp_cats_te,cdv2_realmlp_ratio_round",
        help="원격 태그를 검사할 실험 이름(쉼표 구분)",
    )
    args = parser.parse_args()
    args.remote_experiments = [item for item in args.remote_experiments.split(",") if item]

    train_path, test_path = Path("data/train.csv"), Path("data/test.csv")
    sample_path, folds_path = Path("data/sample_submission.csv"), Path("artifacts/folds.parquet")
    pool_path = Path("artifacts/pool.yaml")
    for name, path in {"train": train_path, "test": test_path, "folds": folds_path}.items():
        if file_sha256(path) != EXPECTED_INPUT_SHA[name]:
            raise RuntimeError(f"main {name} 입력 해시가 실행 입력과 다르다.")
    if file_sha256(sample_path) != EXPECTED_SAMPLE_SHA:
        raise RuntimeError("main sample_submission 입력 해시가 실행 입력과 다르다.")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    folds = pd.read_parquet(folds_path)
    if list(test[ID]) != list(sample[ID]):
        raise RuntimeError("test와 sample_submission 식별자 순서가 다르다.")
    expected_folds = folds.set_index(ID).reindex(train[ID])["fold"]
    if expected_folds.isna().any():
        raise RuntimeError("고정 fold 입력을 train 순서로 정렬할 수 없다.")
    inputs = {"train": train, "test": test, "sample": sample, "folds": expected_folds}
    for bundle in args.bundle:
        if not bundle.is_file():
            raise RuntimeError(f"반입할 묶음이 없다: {bundle}")

    pool_sha256_before = file_sha256(pool_path)
    client, experiment_id = mlflow_client()
    runs = []
    for bundle in args.bundle:
        runs.append(audit_one(client, experiment_id, bundle, inputs, args))
        print(f"[import] {runs[-1]['experiment']} -> {runs[-1]['main_run_id']} auc={runs[-1]['auc_oof']:.7f} reused={runs[-1]['reused_existing_import']}", flush=True)
    if file_sha256(pool_path) != pool_sha256_before:
        raise RuntimeError("묶음 반입이 후보 풀을 바꿨다.")

    audit = {
        "schema_version": 1,
        "issue": 623,
        "git_commit": args.expected_commit,
        "pool_sha256": pool_sha256_before,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"runs": [(r["experiment"], r["main_run_id"], r["auc_oof"]) for r in runs]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
