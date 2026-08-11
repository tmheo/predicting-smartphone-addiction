"""설정 파일 로딩. 형식은 YAML(실험 설정 관례를 따름)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    train: Path
    test: Path
    sample_submission: Path
    folds: Path


@dataclass(frozen=True)
class FeatureConfig:
    include: str  # "raw" 또는 명시적 컬럼 목록(후속 실험에서 확장)
    categorical: list[str]
    placebo: bool
    # 수치 컬럼을 유지한 채 <col>_cat 범주형 복제 컬럼을 추가할 대상. (#31 변형 b)
    categorical_copies: list[str] = field(default_factory=list)
    # 타깃 없이 행 단위로 계산하는 파생 컬럼 이름 목록. features 모듈 DERIVED_REGISTRY의 키. (#46)
    derived: list[str] = field(default_factory=list)
    # fold 루프 안에서 fit/transform하는 fold-fit 트랜스포머 목록.
    # 항목은 {kind: ..., 그 외 생성자 인자} 형태로, kind는 features 모듈 레지스트리의 키. (#35)
    fold_fit: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class ModelConfig:
    kind: str
    params: dict
    fit: dict


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    seeds: list[int]
    source_path: Path  # 설정 원본 경로. run artifact로 그대로 복사해 남긴다.


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(
        name=raw["name"],
        data=DataConfig(**{k: Path(v) for k, v in raw["data"].items()}),
        features=FeatureConfig(**raw["features"]),
        model=ModelConfig(
            kind=raw["model"]["kind"],
            params=raw["model"]["params"],
            fit=raw["model"]["fit"],
        ),
        seeds=raw["cv"]["seeds"],
        source_path=path,
    )
