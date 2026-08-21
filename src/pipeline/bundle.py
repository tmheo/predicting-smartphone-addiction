"""실행 기록 묶음: 로컬 밖에서 끝난 실험 실행을 로컬 실행 저장소로 반입한다. (#98, #58 소비)

사용법:
    uv run python -m pipeline.bundle export <run_id> [--out PATH]
    uv run python -m pipeline.bundle import <zip>

export는 완료된 실행의 기록 원형(params·metrics·tags)과 산출물 전체를
manifest.json과 함께 zip 하나로 사영한다. Kaggle GPU 실행(#58)의 노트북이
`pipeline.run`을 그대로 돌린 뒤 마지막 셀에서 부른다(절차: docs/kaggle-gpu-run.md).

import(묶음 반입)의 검증 게이트. 하나라도 실패하면 반입하지 않는다:
1. 입력 해시: manifest의 sha256(train·test·folds)이 로컬 파일과 일치해야 한다.
   같은 자료와 같은 5-fold 규율을 강제한다.
2. 출처: 실행의 git_commit이 로컬 git에 존재하고, 그 커밋의 config 파일과 묶음의
   config가 sha256 동일해야 하며, 원격 실행이 git_dirty=False여야 한다.
3. 재채점이 진실: 시드별 OOF(oof_seed_*.parquet)와 로컬 라벨로 auc_fold_*·auc_oof·
   auc_oof_seed_*를 전부 재계산해 기록한다. 시드 평균 예측이 묶음의 oof.parquet과
   다르거나 주장 지표와 부동소수점 허용 오차를 넘게 다르면 중단한다.
   원격이 주장한 지표는 기록하지 않는다.

통과하면 로컬 MLflow에 정상 run으로 재생되므로(산출물 전체 포함) compare·pool·
diagnose·submit은 반입 실행을 로컬 실행과 구별하지 않는다. 같은 묶음의 중복 반입은
묶음 sha256 태그로 거부한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import tracking
from .cv import score_predictions
from .data import ID, file_sha256, labels
from .fold_observability import (
    ARTIFACT_PATH as OBSERVABILITY_ARTIFACT_PATH,
    MLFLOW_METRICS as OBSERVABILITY_MLFLOW_METRICS,
    ObservabilitySchemaError,
    read_fold_observability,
)
from .judgment import seed_auc_metric
from .runs import TRACKING_URI, MlflowRunStore, RunStoreError

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
ARTIFACTS_DIR = "artifacts"
# roc_auc_score는 같은 float64 예측에 결정적이므로 재채점은 사실상 정확히 일치해야 한다.
METRIC_TOLERANCE = 1e-9
PRED_TOLERANCE = 1e-12
CORE_INPUTS = ("train", "test", "folds")  # 반입이 로컬 파일과 대조하는 입력 해시.


class BundleError(Exception):
    """묶음 생성·반입 불가. CLI가 sys.exit로 번역한다."""


def _seeds_of(params: dict[str, str]) -> list[int]:
    return [int(s) for s in params["seeds"].split(",")]


def _observability_metric_history(client, run_id: str) -> dict[str, list[dict[str, object]]]:
    history: dict[str, list[dict[str, object]]] = {}
    for name in sorted(OBSERVABILITY_MLFLOW_METRICS):
        values = client.get_metric_history(run_id, name)
        if values:
            history[name] = [
                {
                    "value": float(metric.value),
                    "step": int(metric.step),
                    "timestamp": int(metric.timestamp),
                }
                for metric in values
            ]
    return history


# ---------------------------------------------------------------------- export


def export_bundle(
    run_id: str, out_path: Path | None = None, tracking_uri: str = TRACKING_URI
) -> Path:
    """완료된 실행 하나를 실행 기록 묶음 zip으로 사영한다."""
    store = MlflowRunStore(tracking_uri)
    meta = store.facts_of(run_id)
    if "auc_oof" not in meta.metrics or "features" not in meta.params:
        raise BundleError(
            f"run {run_id}는 최종 기록(auc_oof, features)이 없는 미완료 실행이라 export할 수 없다."
        )

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    with tempfile.TemporaryDirectory() as tmp:
        art_dir = Path(client.download_artifacts(run_id, "", tmp))
        config_names = [p.name for p in art_dir.iterdir() if p.suffix in (".yaml", ".yml")]
        if len(config_names) != 1:
            raise BundleError(f"run {run_id}의 루트에서 설정 YAML 하나를 찾지 못했다: {config_names}")
        missing = [
            tracking.oof_seed_artifact(s)
            for s in _seeds_of(meta.params)
            if not (art_dir / tracking.oof_seed_artifact(s)).exists()
        ]
        if missing:
            raise BundleError(
                f"run {run_id}에 시드별 OOF 산출물({', '.join(missing)})이 없다. "
                "기록 규약 확장(#98) 이전 실행은 export할 수 없다. 재실행할 것."
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source_run_id": run_id,
            "run_name": meta.run_name,
            "params": meta.params,
            "metrics": meta.metrics,  # 주장값. 반입은 재채점과 대조만 하고 기록하지 않는다.
            "tags": meta.tags,
            "observability_metric_history": _observability_metric_history(client, run_id),
            "config_artifact": config_names[0],
            "config_path": f"configs/{config_names[0]}",
            "config_sha256": file_sha256(art_dir / config_names[0]),
            "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "environment": {
                "hostname": socket.gethostname(),
                "kaggle": "KAGGLE_KERNEL_RUN_TYPE" in os.environ,
            },
        }

        experiment = meta.params["experiment"]
        out = out_path or Path(f"{experiment}_{run_id[:8]}.bundle.zip")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
            for path in sorted(art_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, f"{ARTIFACTS_DIR}/{path.relative_to(art_dir)}")
    return out


# ---------------------------------------------------------------------- import


def _git_commit_exists(commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _git_file_sha256(commit: str, path: str) -> str | None:
    """커밋 시점 파일 내용의 sha256. 그 커밋에 파일이 없으면 None."""
    shown = subprocess.run(
        ["git", "show", f"{commit}:{path}"], capture_output=True, check=False
    )
    if shown.returncode != 0:
        return None
    return hashlib.sha256(shown.stdout).hexdigest()


def _verify_provenance(manifest: dict, bundle_dir: Path) -> None:
    tags = manifest["tags"]
    if tags.get("git_dirty") != "False":
        raise BundleError("반입 거부: git_dirty 실행이다. 커밋 고정 환경에서 재실행할 것.")
    commit = tags["git_commit"]
    if not _git_commit_exists(commit):
        raise BundleError(f"반입 거부: 실행의 git_commit {commit}이 로컬 git에 없다.")
    committed = _git_file_sha256(commit, manifest["config_path"])
    bundled = file_sha256(bundle_dir / ARTIFACTS_DIR / manifest["config_artifact"])
    if bundled != manifest["config_sha256"]:
        raise BundleError("반입 거부: 묶음 안의 config가 manifest의 config_sha256과 다르다.")
    if committed != manifest["config_sha256"]:
        raise BundleError(
            f"반입 거부: 커밋 {commit[:8]}의 {manifest['config_path']}와 묶음의 config가 다르다."
        )


def _verify_input_hashes(manifest: dict, bundle_dir: Path) -> dict[str, Path]:
    """입력 해시를 로컬 파일과 대조하고, 검증된 로컬 경로들을 돌려준다."""
    with (bundle_dir / ARTIFACTS_DIR / manifest["config_artifact"]).open() as f:
        config = yaml.safe_load(f)
    local_paths = {
        "train": Path(config["data"]["train"]),
        "test": Path(config["data"]["test"]),
        "folds": Path(config["data"]["folds"]),
    }
    for name in CORE_INPUTS:
        claimed = manifest["tags"].get(f"sha256.{name}")
        if claimed is None:
            raise BundleError(f"반입 거부: manifest에 sha256.{name} 태그가 없다.")
        path = local_paths[name]
        if not path.exists():
            raise BundleError(f"반입 거부: 로컬에 {path}가 없어 입력 해시를 대조할 수 없다.")
        if file_sha256(path) != claimed:
            raise BundleError(
                f"반입 거부: {name}의 sha256이 로컬 {path}와 다르다. 같은 자료·fold가 아니다."
            )
    return local_paths


def _verify_observability(manifest: dict, bundle_dir: Path) -> None:
    """관측 원본이 있으면 원격 내용 해시 그대로인지 반입 전에 확인한다."""
    artifact = bundle_dir / ARTIFACTS_DIR / OBSERVABILITY_ARTIFACT_PATH
    claimed = manifest["tags"].get("sha256.observability.fold_execution")
    if claimed is None and not artifact.exists():
        return  # 관측 계약 이전 실행은 observability_unavailable이다.
    if claimed is None or not artifact.is_file():
        raise BundleError("반입 거부: fold 관측 원본과 내용 해시가 함께 있지 않다.")
    if file_sha256(artifact) != claimed:
        raise BundleError("반입 거부: fold 관측 원본의 내용 해시가 manifest와 다르다.")
    try:
        read_fold_observability(artifact)
    except ObservabilitySchemaError as exc:
        raise BundleError(f"반입 거부: fold 관측 원본 형식이 잘못됐다: {exc}") from exc


def _validated_observability_metric_history(
    manifest: dict,
) -> dict[str, list[dict[str, object]]]:
    raw = manifest.get("observability_metric_history", {})
    if not isinstance(raw, dict):
        raise BundleError("반입 거부: fold 관측 MLflow 요약이 객체가 아니다.")
    if raw and "sha256.observability.fold_execution" not in manifest["tags"]:
        raise BundleError("반입 거부: fold 관측 원본 없이 MLflow 요약만 있다.")
    unknown = set(raw) - set(OBSERVABILITY_MLFLOW_METRICS)
    if unknown:
        raise BundleError(f"반입 거부: 알 수 없는 fold 관측 MLflow 지표다: {sorted(unknown)}")
    validated: dict[str, list[dict[str, object]]] = {}
    for name, values in raw.items():
        if not isinstance(values, list):
            raise BundleError(f"반입 거부: {name} 지표 이력이 목록이 아니다.")
        rows: list[dict[str, object]] = []
        for value in values:
            if not isinstance(value, dict):
                raise BundleError(f"반입 거부: {name} 지표 이력 항목이 객체가 아니다.")
            metric = value.get("value")
            step = value.get("step")
            timestamp = value.get("timestamp")
            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(metric)
                or isinstance(step, bool)
                or not isinstance(step, int)
                or step < 0
                or isinstance(timestamp, bool)
                or not isinstance(timestamp, int)
                or timestamp < 0
            ):
                raise BundleError(f"반입 거부: {name} 지표 이력 값이 잘못됐다.")
            rows.append(
                {"value": float(metric), "step": step, "timestamp": timestamp}
            )
        validated[name] = rows
    return validated


def _rescore(manifest: dict, bundle_dir: Path, inputs: dict[str, Path]) -> dict[str, float]:
    """시드별 OOF를 로컬 라벨로 재채점한다. 재채점이 기록될 유일한 지표다."""
    art = bundle_dir / ARTIFACTS_DIR
    base = pd.read_parquet(art / "oof.parquet")
    ids = pd.Index(base[ID], name=ID)
    y = labels(ids, train_path=inputs["train"])
    local_folds = pd.read_parquet(inputs["folds"]).set_index(ID)["fold"].reindex(ids)
    if not np.array_equal(local_folds.to_numpy(), base["fold"].to_numpy()):
        raise BundleError("반입 거부: 묶음 OOF의 fold 배정이 커밋된 folds.parquet과 다르다.")

    seeds = _seeds_of(manifest["params"])
    metrics: dict[str, float] = {}
    seed_preds = []
    for seed in seeds:
        oof = pd.read_parquet(art / tracking.oof_seed_artifact(seed))
        pred = oof.set_index(ID)["pred"].reindex(ids)
        if not pred.notna().all():
            raise BundleError(f"반입 거부: seed {seed} OOF의 id가 oof.parquet과 다르다.")
        seed_preds.append(pred.to_numpy())
        metrics[seed_auc_metric(seed)] = float(
            score_predictions(y, local_folds, pred.to_numpy())["auc_oof"]
        )

    mean_pred = np.mean(seed_preds, axis=0)
    if np.abs(mean_pred - base["pred"].to_numpy()).max() > PRED_TOLERANCE:
        raise BundleError("반입 거부: 시드별 OOF의 평균이 묶음의 oof.parquet과 다르다.")
    metrics.update(
        {k: float(v) for k, v in score_predictions(y, local_folds, mean_pred).items()}
    )

    for name, value in metrics.items():
        claimed = manifest["metrics"].get(name)
        if claimed is None or abs(claimed - value) > METRIC_TOLERANCE:
            raise BundleError(
                f"반입 거부: {name} 재채점 {value:.9f}이 주장값 {claimed}과 다르다. "
                "실행 환경이나 자료가 어긋난 실행이다."
            )
    return metrics


def import_bundle(zip_path: Path, tracking_uri: str = TRACKING_URI) -> str:
    """실행 기록 묶음을 검증·재채점해 로컬 실행 저장소에 정상 run으로 재생한다."""
    bundle_sha = file_sha256(zip_path)
    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(bundle_dir)
        manifest_path = bundle_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise BundleError(f"{zip_path}에 {MANIFEST_NAME}이 없다. 실행 기록 묶음이 아니다.")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise BundleError(
                f"지원하지 않는 묶음 스키마 버전이다: {manifest.get('schema_version')} "
                f"(지원: {SCHEMA_VERSION})"
            )

        inputs = _verify_input_hashes(manifest, bundle_dir)
        _verify_provenance(manifest, bundle_dir)
        _verify_observability(manifest, bundle_dir)
        observability_metrics = _validated_observability_metric_history(manifest)
        metrics = _rescore(manifest, bundle_dir, inputs)

        client, experiment_id = tracking.mlflow_client(tracking_uri)
        already = client.search_runs(
            [experiment_id], filter_string=f"tags.`import.bundle_sha256` = '{bundle_sha}'"
        )
        if already:
            raise BundleError(
                f"반입 거부: 같은 묶음이 이미 run {already[0].info.run_id}로 반입돼 있다."
            )

        run = client.create_run(experiment_id, run_name=manifest["run_name"])
        run_id = run.info.run_id
        for key, value in manifest["params"].items():
            client.log_param(run_id, key, value)
        for name, value in metrics.items():
            client.log_metric(run_id, name, value)
        for name, history in observability_metrics.items():
            for metric in history:
                client.log_metric(
                    run_id,
                    name,
                    metric["value"],
                    timestamp=metric["timestamp"],
                    step=metric["step"],
                )
        for key in ("git_commit", "git_dirty"):
            client.set_tag(run_id, key, manifest["tags"][key])
        for key, value in manifest["tags"].items():
            if key.startswith("sha256."):
                client.set_tag(run_id, key, value)
        client.set_tag(run_id, "source.kind", "bundle")
        client.set_tag(run_id, "source.run_id", manifest["source_run_id"])
        client.set_tag(run_id, "source.exported_at", manifest["exported_at"])
        client.set_tag(run_id, "import.bundle_sha256", bundle_sha)
        client.set_tag(
            run_id,
            "import.imported_at",
            datetime.now(UTC).isoformat(timespec="seconds"),
        )
        client.log_artifacts(run_id, str(bundle_dir / ARTIFACTS_DIR))
        client.log_artifact(run_id, str(manifest_path), artifact_path="bundle")
        client.set_terminated(run_id, status="FINISHED")
    return run_id


# ------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description="실행 기록 묶음 export / 묶음 반입 (#98)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_export = sub.add_parser("export", help="완료된 실행을 묶음 zip으로 사영")
    p_export.add_argument("run_id", help="export할 MLflow run_id")
    p_export.add_argument("--out", type=Path, help="출력 zip 경로 (기본: <experiment>_<run8>.bundle.zip)")
    p_import = sub.add_parser("import", help="묶음을 검증·재채점해 로컬 저장소로 반입")
    p_import.add_argument("zip", type=Path, help="반입할 묶음 zip 경로")
    args = parser.parse_args()

    try:
        if args.command == "export":
            out = export_bundle(args.run_id, args.out)
            print(f"묶음 생성: {out}")
        else:
            run_id = import_bundle(args.zip)
            print(f"반입 완료: run_id={run_id}. compare·pool에 이 run_id를 그대로 쓴다.")
    except (BundleError, RunStoreError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
