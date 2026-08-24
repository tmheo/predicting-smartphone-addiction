"""실행 저장소: 완료된 실험 실행의 사실·산출물 읽기와 뒤늦은 주석. (CONTEXT.md 용어)

compare·pool·submit과 진단 스크립트가 완료된 실행을 아는 유일한 통로다.
MLflow는 이 저장소의 adapter 하나일 뿐이며(MlflowRunStore), 테스트는
InMemoryRunStore로 같은 계약을 통과한다(tests/test_runs.py의 계약 테스트).

interface 규약:
- facts_of는 기록 원형(params/metrics/tags)을 그대로 돌려준다.
  metric 이름 규약(auc_oof_seed_* 등)의 의미 해석은 판정 쪽 소관이다.
- oof_of는 id 인덱스의 예측 Series를 돌려준다. 모든 소비자가 원하는 최종 형태다.
- artifact_bytes_of·artifact_sha256_of는 이름 있는 산출물 하나를 원본 바이트로 읽고
  그 내용 해시를 돌려준다. 계보 검증처럼 형식을 모르는 산출물을 다루는 소비자가
  MLflow 내부 경로를 직접 열지 않도록 이 계약 하나만 쓴다.
- 실행 중 기록(생성·진행·종료)은 observe.RunObserver의 소관이고 이 module이 아니다.
- 오류 모드: 없는 run은 RunNotFound, 있는 run의 없는 산출물은 ArtifactNotFound.
  둘 다 RunStoreError의 하위이므로 CLI는 RunStoreError 하나만 잡아 sys.exit로 번역한다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd
import yaml

from .data import ID

# 상대 경로 URI이므로 저장소 루트에서 실행하는 것이 전제다.
TRACKING_URI = "sqlite:///mlflow.db"


def sha256_of(payload: bytes) -> str:
    """산출물 내용 해시. 계보 기록의 파일 해시(data.file_sha256)와 같은 규약이다."""
    return hashlib.sha256(payload).hexdigest()


class RunStoreError(Exception):
    """실행 저장소 오류의 공통 뿌리. CLI는 이것 하나만 잡는다."""


class RunNotFound(RunStoreError):
    pass


class ArtifactNotFound(RunStoreError):
    pass


@dataclass(frozen=True)
class RunMeta:
    """실행의 기록 원형. 무엇이 기록돼 있는가만 담고, 그게 무슨 뜻인가는 소비자가 해석한다."""

    run_id: str
    run_name: str
    params: dict[str, str]
    metrics: dict[str, float]
    tags: dict[str, str]


class RunStore(Protocol):
    def facts_of(self, run_id: str) -> RunMeta: ...

    def oof_of(self, run_id: str) -> pd.Series:
        """OOF 예측을 id 인덱스 Series로 돌려준다."""
        ...

    def importance_of(self, run_id: str) -> pd.DataFrame:
        """feature, fold, seed, gain 스키마의 fold별 gain importance."""
        ...

    def config_of(self, run_id: str) -> dict:
        """실행에 남긴 설정 YAML의 파싱본."""
        ...

    def artifact_bytes_of(self, run_id: str, name: str) -> bytes:
        """이름 있는 산출물 하나를 원본 바이트로 읽는다."""
        ...

    def artifact_sha256_of(self, run_id: str, name: str) -> str:
        """이름 있는 산출물 하나의 내용 SHA-256."""
        ...

    def submission_path_of(self, run_id: str) -> Path: ...

    def annotate(
        self,
        run_id: str,
        *,
        tags: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """완료된 실행에 뒤늦은 주석(태그·지표)을 남긴다."""
        ...


class MlflowRunStore:
    """운영 adapter. mlflow.db의 완료된 실행을 읽고 주석을 남긴다."""

    def __init__(self, tracking_uri: str = TRACKING_URI) -> None:
        from mlflow.tracking import MlflowClient

        self._client = MlflowClient(tracking_uri=tracking_uri)

    def _run(self, run_id: str):
        from mlflow.exceptions import MlflowException

        try:
            return self._client.get_run(run_id)
        except MlflowException as exc:
            raise RunNotFound(f"run {run_id}를 실행 저장소에서 찾지 못했다.") from exc

    def _artifact(self, run_id: str, name: str) -> Path:
        from mlflow.exceptions import MlflowException

        self._run(run_id)
        try:
            return Path(self._client.download_artifacts(run_id, name))
        except (MlflowException, OSError) as exc:
            raise ArtifactNotFound(f"run {run_id}에 산출물 {name}이 없다.") from exc

    def facts_of(self, run_id: str) -> RunMeta:
        run = self._run(run_id)
        return RunMeta(
            run_id=run_id,
            run_name=run.info.run_name,
            params=dict(run.data.params),
            metrics=dict(run.data.metrics),
            tags=dict(run.data.tags),
        )

    def oof_of(self, run_id: str) -> pd.Series:
        oof = pd.read_parquet(self._artifact(run_id, "oof.parquet"))
        return oof.set_index(ID)["pred"]

    def importance_of(self, run_id: str) -> pd.DataFrame:
        return pd.read_parquet(self._artifact(run_id, "feature_importance.parquet"))

    def config_of(self, run_id: str) -> dict:
        run = self._run(run_id)
        names = [
            item.path
            for item in self._client.list_artifacts(run.info.run_id)
            if item.path.endswith((".yaml", ".yml"))
        ]
        if len(names) != 1:
            raise ArtifactNotFound(
                f"run {run_id}의 루트에서 설정 YAML 하나를 찾지 못했다: {names}"
            )
        with self._artifact(run_id, names[0]).open() as f:
            return yaml.safe_load(f)

    def artifact_bytes_of(self, run_id: str, name: str) -> bytes:
        path = self._artifact(run_id, name)
        if not path.is_file():
            raise ArtifactNotFound(f"run {run_id}의 산출물 {name}이 파일이 아니다.")
        return path.read_bytes()

    def artifact_sha256_of(self, run_id: str, name: str) -> str:
        return sha256_of(self.artifact_bytes_of(run_id, name))

    def submission_path_of(self, run_id: str) -> Path:
        return self._artifact(run_id, "submission.csv")

    def annotate(
        self,
        run_id: str,
        *,
        tags: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        self._run(run_id)
        for key, value in (tags or {}).items():
            self._client.set_tag(run_id, key, value)
        for key, value in (metrics or {}).items():
            self._client.log_metric(run_id, key, float(value))


@dataclass
class _StoredRun:
    run_name: str
    params: dict[str, str]
    metrics: dict[str, float]
    tags: dict[str, str]
    oof: pd.DataFrame | None
    importance: pd.DataFrame | None
    config: dict | None
    submission_path: Path | None
    artifacts: dict[str, bytes]


@dataclass
class InMemoryRunStore:
    """테스트 adapter. add_run으로 완료된 실행을 심고, MlflowRunStore와 같은 계약을 지킨다."""

    _runs: dict[str, _StoredRun] = field(default_factory=dict)

    def add_run(
        self,
        run_id: str,
        *,
        run_name: str = "",
        params: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        oof: pd.DataFrame | None = None,
        importance: pd.DataFrame | None = None,
        config: dict | None = None,
        submission_path: Path | None = None,
        artifacts: dict[str, bytes] | None = None,
    ) -> str:
        self._runs[run_id] = _StoredRun(
            run_name=run_name,
            params=dict(params or {}),
            metrics=dict(metrics or {}),
            tags=dict(tags or {}),
            oof=oof,
            importance=importance,
            config=config,
            submission_path=submission_path,
            artifacts=dict(artifacts or {}),
        )
        return run_id

    def _run(self, run_id: str) -> _StoredRun:
        if run_id not in self._runs:
            raise RunNotFound(f"run {run_id}를 실행 저장소에서 찾지 못했다.")
        return self._runs[run_id]

    def _artifact(self, run_id: str, name: str, value):
        if value is None:
            raise ArtifactNotFound(f"run {run_id}에 산출물 {name}이 없다.")
        return value

    def facts_of(self, run_id: str) -> RunMeta:
        run = self._run(run_id)
        return RunMeta(
            run_id=run_id,
            run_name=run.run_name,
            params=dict(run.params),
            metrics=dict(run.metrics),
            tags=dict(run.tags),
        )

    def oof_of(self, run_id: str) -> pd.Series:
        oof = self._artifact(run_id, "oof.parquet", self._run(run_id).oof)
        return oof.set_index(ID)["pred"]

    def importance_of(self, run_id: str) -> pd.DataFrame:
        return self._artifact(
            run_id, "feature_importance.parquet", self._run(run_id).importance
        )

    def config_of(self, run_id: str) -> dict:
        return self._artifact(run_id, "설정 YAML", self._run(run_id).config)

    def artifact_bytes_of(self, run_id: str, name: str) -> bytes:
        return self._artifact(run_id, name, self._run(run_id).artifacts.get(name))

    def artifact_sha256_of(self, run_id: str, name: str) -> str:
        return sha256_of(self.artifact_bytes_of(run_id, name))

    def submission_path_of(self, run_id: str) -> Path:
        return self._artifact(run_id, "submission.csv", self._run(run_id).submission_path)

    def annotate(
        self,
        run_id: str,
        *,
        tags: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        run = self._run(run_id)
        run.tags.update(tags or {})
        run.metrics.update({k: float(v) for k, v in (metrics or {}).items()})
