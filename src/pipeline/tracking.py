# PROTOTYPE (issue #17): 구조 확인용 뼈대.
"""실험 기록. MLflow 로컬 file store(mlruns/, gitignore 대상)에 남긴다. (#14)

실행당 기록 규약:
- params: 실험 이름, feature 목록(정렬), 모델 파라미터, 시드.
- metrics: auc_fold_0..4, auc_oof. 시드 반복 시 시드 평균본이 대표 metric.
- artifacts: 설정 원본(toml), oof.parquet, test_pred.parquet, submission.csv.
- tags: git_commit, git_dirty, 입력 파일 sha256. dirty 실행은 앙상블 후보에서 제외하는 관행. (#14)
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .config import ExperimentConfig
from .cv import CVResult


def git_state() -> dict[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    return {"git_commit": commit, "git_dirty": str(dirty)}


def log_run(cfg: ExperimentConfig, result: CVResult, input_hashes: dict[str, str]) -> str:
    """CV 결과 하나를 MLflow run 하나로 기록하고 run_id를 돌려준다."""
    import mlflow

    mlflow.set_experiment("predicting-smartphone-addiction")
    with mlflow.start_run(run_name=cfg.name) as run:
        mlflow.log_params(
            {
                "experiment": cfg.name,
                "features": ",".join(sorted(result.feature_names)),
                "seeds": ",".join(map(str, cfg.seeds)),
                **{f"model.{k}": v for k, v in cfg.model.params.items()},
            }
        )
        mlflow.log_metrics(result.fold_aucs)
        mlflow.set_tags({**git_state(), **{f"sha256.{k}": v for k, v in input_hashes.items()}})
        mlflow.log_artifact(str(cfg.source_path))
        with tempfile.TemporaryDirectory() as tmp:
            oof_path = Path(tmp) / "oof.parquet"
            test_path = Path(tmp) / "test_pred.parquet"
            result.oof.to_parquet(oof_path, index=False)
            result.test_pred.to_parquet(test_path, index=False)
            mlflow.log_artifact(str(oof_path))
            mlflow.log_artifact(str(test_path))
            # submission.csv도 여기서 만들어 artifact로 남긴다. TODO: sample_submission 컬럼 규약 확인.
        return run.info.run_id
