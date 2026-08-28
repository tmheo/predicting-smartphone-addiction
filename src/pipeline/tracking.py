"""실험 기록의 내용 규약: 무엇을 기록하는가. MLflow 로컬 SQLite 백엔드에 남긴다. (#14)

#14는 file store를 권장했지만 mlflow 3.15부터 file store가 유지보수 모드로 내려가
기본 차단되므로, #14가 업그레이드 경로로 언급한 sqlite:///mlflow.db를 처음부터 쓴다.
artifact는 로컬 mlartifacts/ 아래 파일로 남으므로 소비 방식은 달라지지 않는다.

실행 수명주기(생성, 진행 기록, 종료)는 observe.RunObserver가 소유하고(#43),
이 모듈은 활성 실행 안에서 위임 호출되는 기록 헬퍼만 남긴다.

실행당 기록 규약:
- params: 실험 이름, 시드, 단계(stage, 관찰용 - 판정은 seeds param 추론이 진실, #103),
  모델 파라미터(시작 시점), feature 목록(CV 후 확정되므로 최종 시점).
- metrics: auc_fold_0..4, auc_oof, auc_oof_seed_*. 시드 반복 시 시드 평균본이 대표 metric이고
  auc_oof_seed_*가 시드별 OOF AUC다(확정 재검증의 시드별 비교 근거, ADR 0001).
- artifacts: 설정 원본(yaml, 시작 시점), oof.parquet, oof_seed_<seed>.parquet(시드별 OOF,
  묶음 반입의 시드별 재채점 근거, #98), test_pred.parquet, submission.csv,
  feature_importance.parquet(feature, fold, seed, gain 스키마의 fold별 gain importance),
  fold_recovery.json(완료 fold의 해시와 재사용 여부),
  model_training_diagnostics.json(모델별 구조화 학습 관측과 반복형 계열의
  training_length_evidence 관측 학습 길이 근거), training_row_evidence.json(명시한
  학습 행 구성의 부모 대응, 마스크와 누출 방지 불변식). (#19, #141, #160, #372, #500)
  test_pred는 라벨이 없어 재채점 가치가 없으므로 시드 평균본만 남긴다. (#98)
- tags: git_commit, git_dirty, 입력 파일 sha256, 원격 실행 환경과 작업 식별자.
  dirty 실행은 앙상블 후보에서 제외하는 관행. (#14, #414)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from .config import ExperimentConfig
from .cv import CVResult
from .data import ID, TARGET, file_sha256
from .features import PLACEBO
from .fold_fit_reuse import EVIDENCE_NAME as FOLD_FIT_REUSE_EVIDENCE_NAME
from .fold_fit_reuse import SCHEMA_VERSION as FOLD_FIT_REUSE_SCHEMA_VERSION
from .fold_fit_reuse import canonical_json_bytes
from .judgment import mean_gain_of, placebo_gain_of
from .recovery import EVIDENCE_NAME, recovery_evidence
from .training_rows import (
    EVIDENCE_NAME as TRAINING_ROW_EVIDENCE_NAME,
    EVIDENCE_SCHEMA_VERSION as TRAINING_ROW_EVIDENCE_SCHEMA_VERSION,
)

# 실행이 어디 있는가는 실행 저장소(runs)의 지식이다. 기록기는 그 위치에 쓴다.
from .runs import TRACKING_URI

EXPERIMENT_NAME = "predicting-smartphone-addiction"


def _sqlite_database_path(tracking_uri: str) -> Path | None:
    """SQLite URI의 자료 파일 경로. 메모리·다른 저장소 URI면 None이다."""
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return None
    database = unquote(tracking_uri[len(prefix) :].partition("?")[0])
    if not database or database == ":memory:":
        return None
    return Path(database).resolve()


@contextmanager
def _sqlite_initialization_lock(tracking_uri: str) -> Iterator[None]:
    """새 SQLite MLflow 저장소의 스키마·실험 초기화를 프로세스 사이에서 직렬화한다."""
    database = _sqlite_database_path(tracking_uri)
    if database is None:
        yield
        return

    import fcntl

    lock_path = database.with_name(f"{database.name}.init.lock")
    try:
        stream = lock_path.open("a+b")
    except OSError as exc:
        raise RuntimeError(
            f"MLflow SQLite 초기화 잠금 파일을 열 수 없다: {database}"
        ) from exc
    with stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            raise RuntimeError(
                f"MLflow SQLite 초기화 잠금을 얻을 수 없다: {database}"
            ) from exc
        try:
            yield
        finally:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except OSError as exc:
                raise RuntimeError(
                    f"MLflow SQLite 초기화 잠금을 해제할 수 없다: {database}"
                ) from exc


def mlflow_client(tracking_uri: str = TRACKING_URI):
    """(MlflowClient, experiment_id)를 돌려준다. 실험이 없으면 만든다."""
    from mlflow.tracking import MlflowClient

    with _sqlite_initialization_lock(tracking_uri):
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(EXPERIMENT_NAME)
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


def log_start_records(
    client,
    run_id: str,
    cfg: ExperimentConfig,
    *,
    fixed_git_state: dict[str, str] | None = None,
    fixed_config_artifact: tuple[str, bytes] | None = None,
) -> None:
    """실행 생성 직후 기록: params(실험, 시드, 모델), git 태그, 설정 원본. (#43 기록 시점 분배)

    feature 목록 param은 fold-fit 컬럼이 CV에서 확정되므로 log_final_records가 남긴다.
    """
    client.log_param(run_id, "experiment", cfg.name)
    client.log_param(run_id, "seeds", ",".join(map(str, cfg.seeds)))
    client.log_param(run_id, "stage", cfg.stage)
    client.log_param(run_id, "model.kind", cfg.model.kind)
    for key, value in cfg.model.params.items():
        client.log_param(run_id, f"model.{key}", value)
    if cfg.initial_score is not None:
        client.log_param(run_id, "initial_score.kind", cfg.initial_score.kind)
        for key, value in cfg.initial_score.params.items():
            client.log_param(run_id, f"initial_score.{key}", value)
    if cfg.training_rows is not None:
        client.log_param(run_id, "training_rows.arm", cfg.training_rows.arm)
        client.log_param(
            run_id, "training_rows.replica_count", cfg.training_rows.replica_count
        )
        client.log_param(
            run_id,
            "training_rows.observed_cell_mask_probability",
            cfg.training_rows.observed_cell_mask_probability,
        )
    for key, value in (fixed_git_state or git_state()).items():
        client.set_tag(run_id, key, value)
    for key, environment_name in (
        ("remote.provider", "REMOTE_RUN_PROVIDER"),
        ("remote.job_id", "REMOTE_RUN_JOB_ID"),
    ):
        value = os.environ.get(environment_name)
        if value:
            client.set_tag(run_id, key, value)
    if fixed_config_artifact is None:
        client.log_artifact(run_id, str(cfg.source_path))
    else:
        name, payload = fixed_config_artifact
        if Path(name).name != name or Path(name).suffix not in {".yaml", ".yml"}:
            raise ValueError(f"고정 config 산출물 이름이 안전한 YAML 파일 이름이 아니다: {name}")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(payload)
            client.log_artifact(run_id, str(path))


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
        evidence_name = result.recovery_evidence_name
        if evidence_name == EVIDENCE_NAME:
            evidence_payload = recovery_evidence(result.recovery_evidence)
        else:
            from .training_state_recovery import (
                EVIDENCE_NAME as TRAINING_STATE_EVIDENCE_NAME,
            )
            from .training_state_recovery import training_state_recovery_evidence

            if evidence_name != TRAINING_STATE_EVIDENCE_NAME:
                raise ValueError(f"알 수 없는 복구 근거 산출물 이름이다: {evidence_name}")
            evidence_payload = training_state_recovery_evidence(
                result.recovery_evidence
            )
        (tmp_dir / evidence_name).write_text(
            json.dumps(
                evidence_payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if evidence_name == "training_state_recovery.json":
            client.set_tag(
                run_id,
                "sha256.training_state_recovery",
                file_sha256(tmp_dir / evidence_name),
            )
        names.append(evidence_name)
        reuse_evidence_path = tmp_dir / FOLD_FIT_REUSE_EVIDENCE_NAME
        reuse_evidence_path.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": FOLD_FIT_REUSE_SCHEMA_VERSION,
                    "entries": result.fold_feature_reuse_evidence,
                }
            )
            + b"\n"
        )
        client.set_tag(
            run_id,
            "sha256.fold_feature_reuse",
            file_sha256(reuse_evidence_path),
        )
        names.append(FOLD_FIT_REUSE_EVIDENCE_NAME)
        diagnostics_name = "model_training_diagnostics.json"
        (tmp_dir / diagnostics_name).write_text(
            json.dumps(
                result.model_training_diagnostics,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        client.set_tag(
            run_id,
            "sha256.model_training_diagnostics",
            file_sha256(tmp_dir / diagnostics_name),
        )
        names.append(diagnostics_name)
        if cfg.training_rows is not None:
            training_row_evidence_path = tmp_dir / TRAINING_ROW_EVIDENCE_NAME
            training_row_evidence_path.write_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": TRAINING_ROW_EVIDENCE_SCHEMA_VERSION,
                        "entries": result.training_row_evidence,
                    }
                )
                + b"\n"
            )
            client.set_tag(
                run_id,
                "sha256.training_row_evidence",
                file_sha256(training_row_evidence_path),
            )
            names.append(TRAINING_ROW_EVIDENCE_NAME)
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
