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
class TrainingStateConfig:
    """결과 확인 전에 고정한 한 학습 궤적의 후보 시점 계약."""

    trajectory: str
    candidates: tuple[int, ...]
    selected: int
    schedule_horizon_epochs: int
    trajectory_end_epochs: int
    state_kind: str
    selection_rule: str


@dataclass(frozen=True)
class TrainingRowsConfig:
    """바깥쪽 분할의 학습 행을 구성하는 사전 고정 계약."""

    arm: str
    replica_count: int
    observed_cell_mask_probability: float


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
    # 여러 학습 시점 후보만 쓰는 선택 계약이다. None이면 기존 단일 시점 실행이다.
    training_state: TrainingStateConfig | None = None
    # 명시한 실험만 바깥쪽 학습 행 구성을 바꾼다. None이면 기존 원본 행 경로다.
    training_rows: TrainingRowsConfig | None = None


# #71 이전 features 스키마의 키. 종결 실험 config 16개는 역사 기록으로 보존하되
# 재실행은 명확한 오류로 거부한다(재현이 필요하면 MLflow 보존본을 쓴다).
_LEGACY_FEATURE_KEYS = {"include", "placebo", "categorical_copies", "derived", "fold_fit", "pair_ce"}

# --stage 인자의 유효값. 시드 매핑의 유일 출처는 judgment의 시드 상수다(ADR 0001, #103).
STAGES = ("screen", "confirm")

_TRAINING_STATE_FIELDS = {
    "trajectory",
    "candidates",
    "selected",
    "schedule_horizon_epochs",
    "trajectory_end_epochs",
    "state_kind",
    "selection_rule",
}
_TRAINING_STATE_MODELS = {"lookup_transformer", "realmlp"}
_TARGET_FREE_LOOKUP_SCHEDULES = {
    "one_cycle",
    "one_cycle_fixed_momentum",
    "warmup_cosine",
    "warmup_linear",
    "warmup_constant",
}

_TRAINING_ROW_ARMS = {
    "original": (0, 0.0),
    "tripled": (2, 0.0),
    "missingness_augmented": (2, None),
}


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
    model = ModelConfig(
        kind=raw["model"]["kind"],
        params=raw["model"]["params"],
        fit=raw["model"]["fit"],
    )
    training_state = _load_training_state(raw.get("training_state"), model, path)
    training_rows = _load_training_rows(raw.get("training_rows"), path)
    if training_state is not None and training_rows is not None:
        raise ValueError(f"{path}: training_state와 training_rows를 한 실행에서 함께 쓸 수 없다.")
    initial_score = (
        InitialScoreConfig(**raw["initial_score"])
        if raw.get("initial_score") is not None
        else None
    )
    if training_rows is not None and training_rows.replica_count and initial_score is not None:
        raise ValueError(
            f"{path}: 복제 학습 행과 initial_score의 부모 행 상속 계약은 정의되지 않았다."
        )
    return ExperimentConfig(
        name=raw["name"],
        data=DataConfig(**{k: Path(v) for k, v in raw["data"].items()}),
        features=features,
        model=model,
        initial_score=initial_score,
        seeds=seeds,
        stage=stage,
        source_path=path,
        training_state=training_state,
        training_rows=training_rows,
    )


def _load_training_rows(raw: object, path: Path) -> TrainingRowsConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: training_rows는 객체여야 한다.")
    expected = {"arm", "replica_count", "observed_cell_mask_probability"}
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown or missing:
        raise ValueError(
            f"{path}: training_rows 필드가 정확하지 않다. "
            f"누락={missing}, 알 수 없음={unknown}"
        )
    arm = raw["arm"]
    if arm not in _TRAINING_ROW_ARMS:
        raise ValueError(
            f"{path}: training_rows.arm은 {sorted(_TRAINING_ROW_ARMS)} 중 하나여야 한다: "
            f"{arm!r}"
        )
    replica_count = raw["replica_count"]
    if isinstance(replica_count, bool) or not isinstance(replica_count, int):
        raise ValueError(f"{path}: training_rows.replica_count는 정수여야 한다.")
    probability = raw["observed_cell_mask_probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ValueError(
            f"{path}: training_rows.observed_cell_mask_probability는 수여야 한다."
        )
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{path}: training_rows.observed_cell_mask_probability는 0 이상 1 이하여야 한다."
        )
    expected_replicas, expected_probability = _TRAINING_ROW_ARMS[arm]
    if replica_count != expected_replicas:
        raise ValueError(
            f"{path}: {arm} 팔의 replica_count는 {expected_replicas}여야 한다."
        )
    if expected_probability is not None and probability != expected_probability:
        raise ValueError(
            f"{path}: {arm} 팔의 observed_cell_mask_probability는 "
            f"{expected_probability}이어야 한다."
        )
    return TrainingRowsConfig(
        arm=arm,
        replica_count=replica_count,
        observed_cell_mask_probability=probability,
    )


def _positive_integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where}는 1 이상의 정수여야 한다: {value!r}")
    return value


def _load_training_state(
    raw: object, model: ModelConfig, path: Path
) -> TrainingStateConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: training_state는 객체여야 한다.")
    unknown = sorted(set(raw) - _TRAINING_STATE_FIELDS)
    missing = sorted(_TRAINING_STATE_FIELDS - set(raw))
    if unknown or missing:
        raise ValueError(
            f"{path}: training_state 필드가 정확하지 않다. "
            f"누락={missing}, 알 수 없음={unknown}"
        )
    trajectory = raw["trajectory"]
    if not isinstance(trajectory, str) or not trajectory.strip():
        raise ValueError(f"{path}: training_state.trajectory는 비어 있지 않은 문자열이어야 한다.")
    candidates_raw = raw["candidates"]
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError(f"{path}: training_state.candidates는 비어 있지 않은 목록이어야 한다.")
    candidates = tuple(
        _positive_integer(value, f"{path}: training_state.candidates[{index}]")
        for index, value in enumerate(candidates_raw)
    )
    if tuple(sorted(set(candidates))) != candidates:
        raise ValueError(
            f"{path}: training_state.candidates는 중복 없는 오름차순이어야 한다: "
            f"{list(candidates)}"
        )
    selected = _positive_integer(raw["selected"], f"{path}: training_state.selected")
    if selected not in candidates:
        raise ValueError(
            f"{path}: training_state.selected {selected}가 candidates에 없다."
        )
    horizon = _positive_integer(
        raw["schedule_horizon_epochs"],
        f"{path}: training_state.schedule_horizon_epochs",
    )
    trajectory_end = _positive_integer(
        raw["trajectory_end_epochs"],
        f"{path}: training_state.trajectory_end_epochs",
    )
    if max(candidates) > trajectory_end or trajectory_end > horizon:
        raise ValueError(
            f"{path}: 후보 시점 <= 궤적 종료 <= 일정 지평이어야 한다: "
            f"{max(candidates)} <= {trajectory_end} <= {horizon}"
        )
    if raw["state_kind"] != "ema":
        raise ValueError(f"{path}: training_state.state_kind는 'ema'만 지원한다.")
    if raw["selection_rule"] != "precommitted":
        raise ValueError(
            f"{path}: training_state.selection_rule은 'precommitted'여야 한다."
        )
    if model.kind not in _TRAINING_STATE_MODELS:
        raise ValueError(
            f"{path}: training_state는 {sorted(_TRAINING_STATE_MODELS)}만 지원한다: "
            f"{model.kind!r}"
        )
    if model.kind == "lookup_transformer":
        if model.params.get("epochs") != trajectory_end:
            raise ValueError(
                f"{path}: Lookup-Transformer model.params.epochs는 "
                "training_state.trajectory_end_epochs와 같아야 한다."
            )
        if model.params.get("validation_selection", "best") != "final":
            raise ValueError(
                f"{path}: 여러 학습 시점 Lookup-Transformer는 "
                "validation_selection='final'이어야 한다."
            )
        schedule = str(model.params.get("lr_schedule", "one_cycle")).lower()
        if schedule not in _TARGET_FREE_LOOKUP_SCHEDULES:
            raise ValueError(
                f"{path}: 여러 학습 시점 Lookup-Transformer 일정은 검증 목표값을 "
                f"참조할 수 없다: {schedule!r}"
            )
    else:
        if model.params.get("fixed_epochs") != trajectory_end:
            raise ValueError(
                f"{path}: RealMLP model.params.fixed_epochs는 "
                "training_state.trajectory_end_epochs와 같아야 한다."
            )
        if model.params.get("schedule_epochs") != horizon:
            raise ValueError(
                f"{path}: RealMLP model.params.schedule_epochs는 "
                "training_state.schedule_horizon_epochs와 같아야 한다."
            )
    return TrainingStateConfig(
        trajectory=trajectory.strip(),
        candidates=candidates,
        selected=selected,
        schedule_horizon_epochs=horizon,
        trajectory_end_epochs=trajectory_end,
        state_kind="ema",
        selection_rule="precommitted",
    )
