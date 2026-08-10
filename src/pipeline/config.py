# PROTOTYPE (issue #17): 구조 확인용 뼈대.
"""설정 파일 로딩.

TOML을 쓰는 이유: 파이썬 3.11+ 표준 라이브러리(tomllib)로 파싱되므로 의존성이 늘지 않는다.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


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
    with path.open("rb") as f:
        raw = tomllib.load(f)
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
