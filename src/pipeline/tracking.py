"""실험 기록의 내용 규약: 무엇을 기록하는가. MLflow 로컬 SQLite 백엔드에 남긴다. (#14)

#14는 file store를 권장했지만 mlflow 3.15부터 file store가 유지보수 모드로 내려가
기본 차단되므로, #14가 업그레이드 경로로 언급한 sqlite:///mlflow.db를 처음부터 쓴다.
artifact는 로컬 mlartifacts/ 아래 파일로 남으므로 소비 방식은 달라지지 않는다.

실행 수명주기(생성, 진행 기록, 종료)는 observe.RunObserver가 소유하고(#43),
이 모듈은 활성 실행 안에서 위임 호출되는 기록 헬퍼만 남긴다.

실행당 기록 규약:
- params: 실험 이름, 시드, 모델 파라미터(시작 시점), feature 목록(CV 후 확정되므로 최종 시점).
- metrics: auc_fold_0..4, auc_oof, auc_oof_seed_*. 시드 반복 시 시드 평균본이 대표 metric이고
  auc_oof_seed_*가 시드별 OOF AUC다(확정 재검증의 시드별 비교 근거, ADR 0001).
- artifacts: 설정 원본(yaml, 시작 시점), oof.parquet, oof_seed_<seed>.parquet(시드별 OOF,
  묶음 반입의 시드별 재채점 근거, #98), test_pred.parquet, submission.csv,
  feature_importance.parquet(feature, fold, seed, gain 스키마의 fold별 gain importance). (#19)
  test_pred는 라벨이 없어 재채점 가치가 없으므로 시드 평균본만 남긴다. (#98)
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
from .features import PLACEBO
from .judgment import mean_gain_of, placebo_gain_of

# 실행이 어디 있는가는 실행 저장소(runs)의 지식이다. 기록기는 그 위치에 쓴다.
from .runs import TRACKING_URI

EXPERIMENT_NAME = "predicting-smartphone-addiction"


def mlflow_client(tracking_uri: str = TRACKING_URI):
    """(MlflowClient, experiment_id)를 돌려준다. 실험이 없으면 만든다."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT_NAME)
    )
    return client, experiment_id


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


def log_start_records(client, run_id: str, cfg: ExperimentConfig) -> None:
    """실행 생성 직후 기록: params(실험, 시드, 모델), git 태그, 설정 원본. (#43 기록 시점 분배)

    feature 목록 param은 fold-fit 컬럼이 CV에서 확정되므로 log_final_records가 남긴다.
    """
    client.log_param(run_id, "experiment", cfg.name)
    client.log_param(run_id, "seeds", ",".join(map(str, cfg.seeds)))
    client.log_param(run_id, "model.kind", cfg.model.kind)
    for key, value in cfg.model.params.items():
        client.log_param(run_id, f"model.{key}", value)
    if cfg.initial_score is not None:
        client.log_param(run_id, "initial_score.kind", cfg.initial_score.kind)
        for key, value in cfg.initial_score.params.items():
            client.log_param(run_id, f"initial_score.{key}", value)
    for key, value in git_state().items():
        client.set_tag(run_id, key, value)
    client.log_artifact(run_id, str(cfg.source_path))


def log_input_hashes(client, run_id: str, input_hashes: dict[str, str]) -> None:
    """setup 단계에서 입력 파일 해시 계산 완료 직후 기록. (#43 기록 시점 분배)"""
    for name, digest in input_hashes.items():
        client.set_tag(run_id, f"sha256.{name}", digest)


def build_submission(cfg: ExperimentConfig, test_pred: pd.DataFrame) -> pd.DataFrame:
    """sample_submission의 id 순서를 따라 제출 파일(id, addicted_label)을 만든다.

    id 집합이 어긋나면 merge 검증이 즉시 실패한다.
    """
    sample = pd.read_csv(cfg.data.sample_submission, usecols=[ID])
    pred = test_pred.rename(columns={"pred": TARGET})
    return sample.merge(pred, on=ID, how="left", validate="one_to_one")


def oof_seed_artifact(seed: int) -> str:
    """시드별 OOF 산출물 이름. 묶음 반입의 재채점이 같은 이름으로 읽는다. (#98)"""
    return f"oof_seed_{seed}.parquet"


def log_final_records(
    client,
    run_id: str,
    cfg: ExperimentConfig,
    result: CVResult,
    seed_oofs: dict[int, pd.DataFrame],
) -> None:
    """종료 직전 기록: feature 목록 param, 최종 지표, 원본 산출물.

    seed_oofs는 시드 평균 전의 시드별 OOF(id, fold, pred)다. 시드 평균에서
    auc_oof_seed_*를 역산할 수 없으므로 시드별 예측을 산출물로 보존한다. (#98)
    """
    submission = build_submission(cfg, result.test_pred)
    assert submission[TARGET].notna().all(), "제출 파일에 예측이 없는 id가 있다."

    client.log_param(run_id, "features", ",".join(sorted(result.feature_names)))
    for name, value in result.fold_aucs.items():
        client.log_metric(run_id, name, value)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        names = ["oof.parquet", "test_pred.parquet", "feature_importance.parquet", "submission.csv"]
        result.oof.to_parquet(tmp_dir / "oof.parquet", index=False)
        result.test_pred.to_parquet(tmp_dir / "test_pred.parquet", index=False)
        result.importance.to_parquet(tmp_dir / "feature_importance.parquet", index=False)
        submission.to_csv(tmp_dir / "submission.csv", index=False)
        for seed, oof in seed_oofs.items():
            names.append(oof_seed_artifact(seed))
            oof.to_parquet(tmp_dir / oof_seed_artifact(seed), index=False)
        for name in names:
            client.log_artifact(run_id, str(tmp_dir / name))


def warn_below_placebo(importance: pd.DataFrame) -> None:
    """평균 gain이 플라시보보다 낮은 피처를 콘솔 경고로 알린다. (#19)

    이 경고는 판정이 아니라 관찰이다. 채택 판정은 pipeline.compare가 새 피처에만 묻는다.
    """
    mean_gain = mean_gain_of(importance)
    placebo_gain = placebo_gain_of(mean_gain)
    if placebo_gain is None:
        return
    below = mean_gain[mean_gain < placebo_gain].drop(PLACEBO, errors="ignore")
    for feature, gain in below.sort_values().items():
        print(
            f"경고: {feature}의 평균 gain importance({gain:.1f})가 "
            f"플라시보({placebo_gain:.1f})보다 낮다."
        )
