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
    # "raw": 원시 CSV 컬럼 전부(ID·타깃 제외)라는 결정적 정의. 명시적 목록은 지원하지 않는다. (#71)
    base: str
    # LightGBM 내장 범주형 처리를 쓸 컬럼. 컬럼을 만들지 않는 표현 규칙이라 제공자가 아니다.
    categorical: list[str]
    # 순서 있는 컬럼 제공자 목록. 항목은 {kind: ..., 그 외 팩토리 인자}.
    # kind와 적용 단계는 plan.REGISTRY가 소유하고, 같은 단계 안에서는 이 목록 순서가 컬럼 순서다.
    providers: list[dict] = field(default_factory=list)
    # base에서 뺄 raw 컬럼. 학습 행렬에서만 빠지고 제공자 입력(원본 frame)에는 남으므로
    # 제외한 raw 컬럼의 파생 표현(예: age 제외 + age_te 유지)이 가능하다. (#79)
    exclude: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelConfig:
    kind: str
    params: dict
    fit: dict


@dataclass(frozen=True)
class InitialScoreConfig:
    kind: str
    params: dict


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    initial_score: InitialScoreConfig | None
    seeds: list[int]  # 출처는 stage → judgment 시드 상수 매핑이다. config 파일에는 없다. (#103)
    stage: str  # 실행 단계(screen|confirm). 관찰용 param 기록에 쓴다. (#103)
    source_path: Path  # 설정 원본 경로. run artifact로 그대로 복사해 남긴다.


# #71 이전 features 스키마의 키. 종결 실험 config 16개는 역사 기록으로 보존하되
# 재실행은 명확한 오류로 거부한다(재현이 필요하면 MLflow 보존본을 쓴다).
_LEGACY_FEATURE_KEYS = {"include", "placebo", "categorical_copies", "derived", "fold_fit", "pair_ce"}

# --stage 인자의 유효값. 시드 매핑의 유일 출처는 judgment의 시드 상수다(ADR 0001, #103).
STAGES = ("screen", "confirm")


def stage_seeds(stage: str) -> list[int]:
    """stage → 시드. 순환 import 방지를 위해 지연 import(FeaturePlan과 같은 패턴)."""
    from .judgment import CONFIRM_SEEDS, SCREENING_SEEDS

    if stage not in STAGES:
        raise ValueError(f"알 수 없는 stage: {stage!r}. --stage는 {'|'.join(STAGES)}만 받는다.")
    return list({"screen": SCREENING_SEEDS, "confirm": CONFIRM_SEEDS}[stage])


def load_config(path: str | Path, stage: str) -> ExperimentConfig:
    path = Path(path)
    seeds = stage_seeds(stage)
    with path.open() as f:
        raw = yaml.safe_load(f)
    legacy = sorted(_LEGACY_FEATURE_KEYS & raw["features"].keys())
    if legacy:
        raise ValueError(
            f"{path}: features 스키마가 #71 이전 형식이다(옛 키: {', '.join(legacy)}). "
            "base/categorical/providers 형식으로 재작성할 것. "
            "종결 실험의 재현이 필요하면 MLflow 보존본(설정 yaml artifact)을 쓴다."
        )
    if "cv" in raw:
        raise ValueError(
            f"{path}: cv 블록(cv.seeds)은 폐기됐다(#103). 단계는 --stage screen|confirm으로 "
            "지정하고 시드는 판정 계약(judgment)이 정한다. config에서 cv 블록을 제거할 것."
        )
    features = FeatureConfig(**raw["features"])
    # 누출 규율과 선언 충돌은 설정 적재 시점에 검증한다. 순환 import 방지를 위해 지연 import. (#71)
    from .plan import FeaturePlan

    FeaturePlan.from_config(features)
    return ExperimentConfig(
        name=raw["name"],
        data=DataConfig(**{k: Path(v) for k, v in raw["data"].items()}),
        features=features,
        model=ModelConfig(
            kind=raw["model"]["kind"],
            params=raw["model"]["params"],
            fit=raw["model"]["fit"],
        ),
        initial_score=(
            InitialScoreConfig(**raw["initial_score"])
            if raw.get("initial_score") is not None
            else None
        ),
        seeds=seeds,
        stage=stage,
        source_path=path,
    )
