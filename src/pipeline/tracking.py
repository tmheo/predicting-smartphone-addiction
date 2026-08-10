"""실험 기록. MLflow 로컬 SQLite 백엔드(mlflow.db, gitignore 대상)에 남긴다. (#14)

#14는 file store를 권장했지만 mlflow 3.15부터 file store가 유지보수 모드로 내려가
기본 차단되므로, #14가 업그레이드 경로로 언급한 sqlite:///mlflow.db를 처음부터 쓴다.
artifact는 로컬 mlartifacts/ 아래 파일로 남으므로 소비 방식은 달라지지 않는다.

실행당 기록 규약:
- params: 실험 이름, feature 목록(정렬), 모델 파라미터, 시드.
- metrics: auc_fold_0..4, auc_oof. 시드 반복 시 시드 평균본이 대표 metric.
- artifacts: 설정 원본(yaml), oof.parquet, test_pred.parquet, submission.csv.
- tags: git_commit, git_dirty, 입력 파일 sha256. dirty 실행은 앙상블 후보에서 제외하는 관행. (#14)
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .cv import CVResult
from .data import ID, TARGET


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


def build_submission(cfg: ExperimentConfig, test_pred: pd.DataFrame) -> pd.DataFrame:
    """sample_submission의 id 순서를 따라 제출 파일(id, addicted_label)을 만든다.

    id 집합이 어긋나면 merge 검증이 즉시 실패한다.
    """
    sample = pd.read_csv(cfg.data.sample_submission, usecols=[ID])
    pred = test_pred.rename(columns={"pred": TARGET})
    return sample.merge(pred, on=ID, how="left", validate="one_to_one")


def log_run(cfg: ExperimentConfig, result: CVResult, input_hashes: dict[str, str]) -> str:
    """CV 결과 하나를 MLflow run 하나로 기록하고 run_id를 돌려준다."""
    import mlflow

    submission = build_submission(cfg, result.test_pred)
    assert submission[TARGET].notna().all(), "제출 파일에 예측이 없는 id가 있다."

    # 상대 경로 URI이므로 저장소 루트에서 실행하는 것이 전제다.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
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
            tmp_dir = Path(tmp)
            result.oof.to_parquet(tmp_dir / "oof.parquet", index=False)
            result.test_pred.to_parquet(tmp_dir / "test_pred.parquet", index=False)
            submission.to_csv(tmp_dir / "submission.csv", index=False)
            for name in ("oof.parquet", "test_pred.parquet", "submission.csv"):
                mlflow.log_artifact(str(tmp_dir / name))
        return run.info.run_id
