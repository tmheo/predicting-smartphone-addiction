"""재학습 계획 장부: 문법, 근거 계보 검증과 재학습 예산 재계산. (#373)

장부는 전체 자료 재학습에 무엇을 얼마나 돌릴지 고정한 기록이다.
이 module은 그 기록의 **문법**과 **관문**만 소유한다. 학습도 저장도 하지 않는다.

두 단계로 나뉜다.

- `RefitPlan.load()`는 문법만 본다. 파일 밖의 무엇도 읽지 않고, 알 수 없는 필드나
  빠진 필드, 자료형이 틀린 값을 그 자리에서 거부한다. 이 단계를 지난 계획은
  "읽히는 기록"일 뿐 실행 예산을 노출하지 않는다.
- `RefitPlan.validate_for_refit()`는 원시 근거와 현재 후보 풀을 본다. 실행 저장소에서
  근거 산출물을 읽어 계보를 맞춰 보고, 원시 값을 다시 변환하고, 재학습 예산을 다시
  계산해 저장값과 정확히 같을 때만 `ExecutableRefitPlan`을 돌려준다.

**저장된 예산은 믿지 않는다.** 실행 경로가 읽는 숫자는 언제나 `validate_for_refit()`가
원시 근거에서 다시 계산한 값이다. 사람이 장부의 예산을 고쳐도, 원시 값과 예산을 함께
고쳐도, 실행 저장소의 근거 산출물 해시와 계보 검증을 통과하지 못하면 실행이 열리지 않는다.
그래서 `budgets`를 노출하는 자료형은 `ExecutableRefitPlan` 하나뿐이다.

책임 경계(이슈 #329의 결정)는 이렇다.

- 모델 계열 연결부가 원시 의미 선언과 원시 근거 기록을 소유한다.
- `training_length`가 원시 의미 변환과 재학습 예산 산정을 소유한다.
- 이 module이 근거 계보 보존과 실행 관문을 소유한다.

실행 저장소는 `RunStore` 계약으로만 만진다. MLflow의 내부 경로를 직접 열지 않으므로
시험은 메모리 저장소로 같은 관문을 통과시킨다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import data
from .ensemble import COMBINER_REGISTRY
from .ledger import POOL_PATH, Pool
from .runs import MlflowRunStore, RunStore, RunStoreError, sha256_of
from .training_length import (
    FIXED_COUNT,
    HALF_UP_ROUNDING,
    MEDIAN_STATISTIC,
    ONE_BASED_COUNT,
    RAW_MEANINGS,
    ZERO_BASED_POSITION,
    ObservedTrainingLength,
    RefitBudgetDerivation,
    RefitBudgetPolicy,
    TrainingLengthError,
    derive_refit_budgets,
    observe_training_length,
    observed_length_from_raw,
)
from .training_state_manifest import (
    MANIFEST_NAME as TRAINING_STATE_MANIFEST,
    TrainingStateManifestError,
    validate_candidate_manifest,
    validate_candidate_parent_lineage,
)

SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = (2, SCHEMA_VERSION)

# 근거 상태. `unresolved`는 아직 원시 값을 확정하지 못한 칸이고,
# `not_applicable`은 반복 수라는 개념 자체가 없는 구성원이다.
STATUS_CONFIRMED = "confirmed"
STATUS_UNRESOLVED = "unresolved"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUSES = (STATUS_CONFIRMED, STATUS_UNRESOLVED, STATUS_NOT_APPLICABLE)

# 모델 계열이 자기 원시 선택값에 대해 선언할 수 있는 변환기. ADR 0002의 표와 같다.
# 연결부(#372)가 같은 식별자를 스스로 선언하고, 장부는 여기서 그 선언을 다시 맞춰 본다.
# 트리 세 계열은 조기 종료 선택값과 설정 고정 횟수를 모두 낼 수 있다(#413).
# 표에 없는 계열은 거부한다. 새 계열이 재학습에 들어오려면 자기 변환기를 먼저 등록해야 한다.
MODEL_FAMILY_CONVERTERS = {
    "lightgbm": (ONE_BASED_COUNT, FIXED_COUNT),
    "xgboost": (ZERO_BASED_POSITION, FIXED_COUNT),
    "catboost": (ZERO_BASED_POSITION, FIXED_COUNT),
    "lookup_transformer": (ZERO_BASED_POSITION,),
    "contextualized_spline_transformer": (ONE_BASED_COUNT,),
    "scalar_token_transformer": (ONE_BASED_COUNT,),
    "tab_cnn": (ONE_BASED_COUNT,),
    "tabm": (ONE_BASED_COUNT,),
    "realmlp": (FIXED_COUNT,),
}

# 바깥쪽 분할 수. 근거가 한 분할치 통째로 비어도 좌표 곱만으로는 드러나지 않으므로
# 재학습 규약이 고정한 5-fold를 관문에서 함께 확인한다. (ADR 0001, ADR 0002)
OUTER_FOLD_COUNT = 5

PLAN_FIELDS = ("schema_version", "source_pool_sha256", "protocol", "members")
PROTOCOL_FIELDS = (
    "iteration_multiplier",
    "budget_statistic",
    "budget_rounding",
    "cv_model_weight",
    "full_model_weight",
    "combiner",
)
MEMBER_FIELDS = (
    "config",
    "config_path",
    "lineage",
    "training_length_evidence",
    "refit_budget_derivation",
)
LINEAGE_FIELDS = (
    "source_run_id",
    "source_git_commit",
    "source_config_path",
    "source_config_sha256",
    "evidence_artifact_path",
    "evidence_artifact_sha256",
)
EVIDENCE_FIELDS = ("status", "model_family", "converter", "observations")
OBSERVATION_FIELDS = (
    "seed",
    "outer_fold",
    "inner_member",
    "raw_field",
    "raw_value",
    "raw_meaning",
    "observed_training_length",
)
DERIVATION_FIELDS = ("statistic", "multiplier", "rounding", "seeds")
SEED_BUDGET_FIELDS = ("seed", "observed_lengths", "median", "scaled", "budget")
TRAJECTORY_STATE_METHOD = "trajectory_state"
TRAJECTORY_DERIVATION_FIELDS = (
    "method",
    "completed_epochs",
    "schedule_horizon_epochs",
    "state_kind",
    "trajectory_identity_sha256",
    "seeds",
)
TRAJECTORY_SEED_BUDGET_FIELDS = ("seed", "observed_lengths", "budget")

# 손으로 예산을 덮어쓰는 통로. 문법에 두지 않으므로 알 수 없는 필드로 걸리지만,
# 왜 막혔는지 바로 알 수 있게 별도 문구로 거부한다.
HAND_EDIT_MARKERS = ("override", "manual")


class RefitPlanError(ValueError):
    """재학습 계획 장부의 문법이나 관문을 어겼다."""


@dataclass(frozen=True)
class MemberLineage:
    """원시 근거가 어느 실행에서 나왔는지 고정한 계보.

    경로는 저장소 기준 경로이고, 실행 저장소 안의 산출물 이름은 그 경로의 파일 이름이다.
    """

    source_run_id: str
    source_git_commit: str
    source_config_path: str
    source_config_sha256: str
    evidence_artifact_path: str
    evidence_artifact_sha256: str


@dataclass(frozen=True)
class RawObservation:
    """장부에 적힌 관측 하나. 아직 검증하지 않은 기록 원형이다."""

    seed: int
    outer_fold: int
    inner_member: int | None
    raw_field: str
    raw_value: int
    raw_meaning: str
    observed_training_length: int

    @property
    def coordinate(self) -> tuple[int, int, int | None]:
        return (self.seed, self.outer_fold, self.inner_member)


@dataclass(frozen=True)
class TrainingLengthEvidence:
    """구성원 하나의 관측 학습 길이 근거 묶음."""

    status: str
    model_family: str
    converter: str | None
    observations: tuple[RawObservation, ...]


@dataclass(frozen=True)
class SeedBudgetRecord:
    """장부가 주장하는 시드 하나의 계산 결과. 관문은 이 값을 믿지 않고 다시 계산한다."""

    seed: int
    observed_lengths: tuple[int, ...]
    median: float | None
    scaled: float | None
    budget: int | None


@dataclass(frozen=True)
class RefitBudgetRecord:
    """장부가 주장하는 계산 규약과 시드별 결과."""

    statistic: str
    multiplier: float
    rounding: str
    seeds: tuple[SeedBudgetRecord, ...]


@dataclass(frozen=True)
class TrajectoryStateBudgetRecord:
    """장부가 주장하는 동일 궤적 시점의 정확 종료 재학습 계약."""

    method: str
    completed_epochs: int
    schedule_horizon_epochs: int
    state_kind: str
    trajectory_identity_sha256: str
    seeds: tuple[SeedBudgetRecord, ...]


@dataclass(frozen=True)
class TrajectoryStateBudgetDerivation:
    """원시 근거와 후보 실행 계보에서 다시 확인한 정확 시점 예산."""

    completed_epochs: int
    schedule_horizon_epochs: int
    state_kind: str
    trajectory_identity_sha256: str
    seeds: tuple[SeedBudgetRecord, ...]

    def budgets(self) -> dict[int, int]:
        return {seed.seed: int(seed.budget) for seed in self.seeds}


@dataclass(frozen=True)
class RefitPlanMember:
    """검증 전 장부 구성원. 실행 예산을 노출하지 않는다.

    `entry_sha256`은 이 구성원의 장부 항목과 규약 블록만 덮는 내용 해시다(#69).
    """

    config: str
    config_path: Path
    lineage: MemberLineage
    evidence: TrainingLengthEvidence
    budget_derivation: RefitBudgetRecord | TrajectoryStateBudgetRecord
    entry_sha256: str


@dataclass(frozen=True)
class RefitProtocol:
    """계획 전체에 걸리는 계산과 조립 규약."""

    iteration_multiplier: float
    budget_statistic: str
    budget_rounding: str
    cv_model_weight: int
    full_model_weight: int
    combiner: str


@dataclass(frozen=True)
class ExecutableRefitMember:
    """관문을 통과한 구성원. `budgets`는 원시 근거에서 다시 계산한 값이다.

    실행 경로가 시드별 기록과 구성원 manifest에 원시 근거 계보와 파생 규약을 남기므로
    `lineage`와 `derivation`을 함께 들고 나온다. 실행 식별자는 계보에서 파생하며,
    같은 값을 두 자리에 두지 않는다.
    """

    config: str
    config_path: Path
    lineage: MemberLineage
    status: str
    budgets: dict[int, int | None]
    derivation: RefitBudgetDerivation | TrajectoryStateBudgetDerivation | None
    entry_sha256: str

    @property
    def run_id(self) -> str:
        return self.lineage.source_run_id

    @property
    def refit_method(self) -> str:
        if isinstance(self.derivation, TrajectoryStateBudgetDerivation):
            return TRAJECTORY_STATE_METHOD
        if self.derivation is None:
            return STATUS_NOT_APPLICABLE
        return "fold_scaled_median"


@dataclass(frozen=True)
class ExecutableRefitPlan:
    """관문을 통과한 계획. 파생 재학습 예산을 노출하는 유일한 자료형이다."""

    source_path: Path
    content_sha256: str
    source_pool_sha256: str
    protocol: RefitProtocol
    members: tuple[ExecutableRefitMember, ...]

    def member(self, name: str) -> ExecutableRefitMember:
        matches = [member for member in self.members if member.config == name]
        if len(matches) != 1:
            raise RefitPlanError(f"재학습 계획에 구성원 {name!r}이 정확히 하나 있지 않다.")
        return matches[0]

    def budgets(self) -> dict[str, dict[int, int | None]]:
        """구성원 -> 시드 -> 재학습 예산. 실행 경로가 읽는 최종 숫자다."""
        return {member.config: dict(member.budgets) for member in self.members}


@dataclass(frozen=True)
class RefitPlan:
    """문법만 통과한 재학습 계획 장부.

    실행 예산을 노출하지 않는다. 실행에 필요한 숫자는 `validate_for_refit()`가
    돌려주는 `ExecutableRefitPlan`에만 있다.
    """

    source_path: Path
    content_sha256: str
    schema_version: int
    source_pool_sha256: str
    protocol: RefitProtocol
    members: tuple[RefitPlanMember, ...]

    @classmethod
    def load(cls, path: Path) -> RefitPlan:
        """장부를 엄격한 문법으로 읽는다. 파일 밖의 무엇도 읽지 않는다.

        읽은 바이트의 SHA-256을 계획 내용 해시로 함께 들고 나온다. 실행 경로는 이 해시를
        시드별 기록과 manifest에 장부 맥락으로 남긴다.

        재개와 조립이 맞춰 보는 값은 구성원마다 따로 계산한 항목 해시다(`member_entry_sha256`).
        그 해시는 규약 블록과 그 구성원의 장부 항목(계보·원시 관측·예산 계산)만 덮으므로,
        그 구성원의 어느 칸이 바뀌어도 재개와 조립이 막히지만 다른 구성원이 장부에
        더해지거나 빠지는 것만으로는 이미 끝난 재학습이 무효가 되지 않는다(#69).
        """
        payload = path.read_bytes()
        raw = yaml.safe_load(payload.decode())
        document = _mapping(raw, "장부")
        _exact_fields(document, PLAN_FIELDS, "장부")

        version = _integer(document["schema_version"], "schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise RefitPlanError(
                f"장부 문법 판본은 {list(SUPPORTED_SCHEMA_VERSIONS)} 중 하나여야 한다: "
                f"{version}"
            )
        protocol_document = _mapping(document["protocol"], "protocol")
        members = tuple(
            _load_member(item, index, protocol_document, version)
            for index, item in enumerate(_sequence(document["members"], "members"))
        )
        if not members:
            raise RefitPlanError("장부에 구성원이 하나도 없다.")
        duplicates = _duplicates([member.config for member in members])
        if duplicates:
            raise RefitPlanError(f"장부에 같은 구성원이 두 번 있다: {duplicates}")
        return cls(
            source_path=path,
            content_sha256=sha256_of(payload),
            schema_version=version,
            source_pool_sha256=_text(document["source_pool_sha256"], "source_pool_sha256"),
            protocol=_load_protocol(document["protocol"]),
            members=members,
        )

    def validate_for_refit(
        self,
        *,
        store: RunStore | None = None,
        pool: Pool | None = None,
        pool_sha256: str | None = None,
    ) -> ExecutableRefitPlan:
        """원시 근거와 현재 후보 풀을 검증하고 실행 가능한 계획을 돌려준다.

        하나라도 어긋나면 `RefitPlanError`를 올리고 실행 가능한 계획을 만들지 않는다.
        기본값은 운영 경로(MLflow 실행 저장소와 커밋된 후보 풀 장부)이며,
        시험은 메모리 저장소와 시험용 풀을 넣어 같은 관문을 통과시킨다.
        """
        store = MlflowRunStore() if store is None else store
        pool = Pool.load() if pool is None else pool
        pool_sha256 = (
            data.file_sha256(POOL_PATH) if pool_sha256 is None else pool_sha256
        )

        self._validate_protocol()
        pool_members = self._validate_pool(pool, pool_sha256)
        members = tuple(
            _validate_member(
                member,
                pool_member,
                self.protocol,
                store,
                schema_version=self.schema_version,
            )
            for member, pool_member in zip(self.members, pool_members, strict=True)
        )
        return ExecutableRefitPlan(
            source_path=self.source_path,
            content_sha256=self.content_sha256,
            source_pool_sha256=self.source_pool_sha256,
            protocol=self.protocol,
            members=members,
        )

    def _validate_protocol(self) -> None:
        protocol = self.protocol
        if protocol.budget_statistic != MEDIAN_STATISTIC:
            raise RefitPlanError(
                f"재학습 예산 통계량은 {MEDIAN_STATISTIC!r}이어야 한다: "
                f"{protocol.budget_statistic!r}"
            )
        if protocol.iteration_multiplier != 1.25:
            raise RefitPlanError(
                f"전체 자료 학습 길이 배수는 1.25여야 한다: {protocol.iteration_multiplier}"
            )
        if protocol.budget_rounding != HALF_UP_ROUNDING:
            raise RefitPlanError(
                f"사사오입 방식은 {HALF_UP_ROUNDING!r}이어야 한다: "
                f"{protocol.budget_rounding!r}"
            )
        if (protocol.cv_model_weight, protocol.full_model_weight) != (5, 1):
            raise RefitPlanError("CV와 전체 자료 예측의 모델 개수 가중치는 5:1이어야 한다.")
        if protocol.combiner not in COMBINER_REGISTRY:
            raise RefitPlanError(
                f"등록되지 않은 결합 방식이다: {protocol.combiner} "
                f"(등록: {', '.join(COMBINER_REGISTRY)})"
            )

    def _validate_pool(self, pool: Pool, pool_sha256: str) -> list:
        if pool_sha256 != self.source_pool_sha256:
            raise RefitPlanError("재학습 계획의 후보 풀 SHA-256이 현재 장부와 다르다.")
        expected = [(member.config, member.run_id) for member in pool.members]
        actual = [
            (member.config, member.lineage.source_run_id) for member in self.members
        ]
        if actual != expected:
            raise RefitPlanError("재학습 계획의 구성원 순서나 실행 ID가 후보 풀과 다르다.")
        return list(pool.members)


def _validate_member(
    member: RefitPlanMember,
    pool_member,
    protocol: RefitProtocol,
    store: RunStore,
    *,
    schema_version: int,
) -> ExecutableRefitMember:
    if not member.config_path.is_file():
        raise RefitPlanError(f"{member.config}: 설정 파일이 없다: {member.config_path}")
    _validate_training_state_derivation_kind(member, schema_version)
    _validate_lineage(member, store)

    evidence = member.evidence
    allowed_seeds = _allowed_seeds(member, pool_member)
    if evidence.status == STATUS_UNRESOLVED:
        raise RefitPlanError(f"{member.config}: 관측 학습 길이 근거가 미확정이다.")
    if evidence.status == STATUS_NOT_APPLICABLE:
        _validate_not_applicable(member, allowed_seeds)
        return ExecutableRefitMember(
            config=member.config,
            config_path=member.config_path,
            lineage=member.lineage,
            status=evidence.status,
            budgets={seed: None for seed in allowed_seeds},
            derivation=None,
            entry_sha256=member.entry_sha256,
        )

    observations = _validate_observations(member, allowed_seeds)
    if isinstance(member.budget_derivation, TrajectoryStateBudgetRecord):
        derivation = _validate_trajectory_state_derivation(
            member, observations, allowed_seeds
        )
    else:
        derivation = _rederive_budgets(member, observations, protocol)
    return ExecutableRefitMember(
        config=member.config,
        config_path=member.config_path,
        lineage=member.lineage,
        status=evidence.status,
        budgets=dict(derivation.budgets()),
        derivation=derivation,
        entry_sha256=member.entry_sha256,
    )


def _validate_training_state_derivation_kind(
    member: RefitPlanMember, schema_version: int
) -> None:
    """판본 3 설정과 정확 시점 파생 방식이 반드시 함께 나타나는지 확인한다."""
    try:
        config = yaml.safe_load(member.config_path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise RefitPlanError(f"{member.config}: 설정 파일을 읽지 못했다.") from error
    state = config.get("training_state") if isinstance(config, dict) else None
    has_training_state = isinstance(state, dict)
    if schema_version < 3:
        if has_training_state:
            raise RefitPlanError(
                f"{member.config}: training_state 설정은 장부 문법 판본 3이 필요하다."
            )
        return
    has_trajectory_derivation = isinstance(
        member.budget_derivation, TrajectoryStateBudgetRecord
    )
    if has_training_state != has_trajectory_derivation:
        raise RefitPlanError(
            f"{member.config}: 문법 판본 3에서는 training_state 설정과 "
            "trajectory_state 재학습 방식이 함께 있어야 한다."
        )


def _validate_lineage(member: RefitPlanMember, store: RunStore) -> None:
    """계보의 실행이 실제로 있고, 소스 판본과 두 산출물 해시가 그 실행과 같은지 본다."""
    lineage = member.lineage
    try:
        facts = store.facts_of(lineage.source_run_id)
        config_sha256 = store.artifact_sha256_of(
            lineage.source_run_id, Path(lineage.source_config_path).name
        )
        evidence_sha256 = store.artifact_sha256_of(
            lineage.source_run_id, lineage.evidence_artifact_path
        )
    except RunStoreError as error:
        raise RefitPlanError(f"{member.config}: 근거 계보를 실행 저장소에서 확인하지 못했다: {error}") from error

    recorded_commit = facts.tags.get("git_commit")
    if recorded_commit != lineage.source_git_commit:
        raise RefitPlanError(
            f"{member.config}: 근거 계보의 소스 판본이 실행 기록과 다르다: "
            f"{lineage.source_git_commit} != {recorded_commit}"
        )
    if config_sha256 != lineage.source_config_sha256:
        raise RefitPlanError(
            f"{member.config}: 근거 계보의 설정 해시가 실행 기록과 다르다."
        )
    if evidence_sha256 != lineage.evidence_artifact_sha256:
        raise RefitPlanError(
            f"{member.config}: 원시 근거 산출물 해시가 실행 기록과 다르다."
        )
    if isinstance(member.budget_derivation, TrajectoryStateBudgetRecord):
        _validate_training_state_run(member, facts, store)


def _validate_training_state_run(member: RefitPlanMember, facts, store: RunStore) -> None:
    """정확 시점 재학습의 출처 실행과 후보 manifest가 같은 궤적을 가리키는지 본다."""
    record = member.budget_derivation
    assert isinstance(record, TrajectoryStateBudgetRecord)
    if facts.status != "FINISHED":
        raise RefitPlanError(f"{member.config}: 학습 시점 후보 실행이 완료 상태가 아니다.")
    run_id = member.lineage.source_run_id
    try:
        payload = store.artifact_bytes_of(run_id, TRAINING_STATE_MANIFEST)
        manifest = validate_candidate_manifest(
            manifest_bytes=payload,
            tags=facts.tags,
            params=facts.params,
            artifact_bytes_of=lambda name: store.artifact_bytes_of(run_id, name),
        )
        validate_candidate_parent_lineage(
            child_run_id=run_id,
            child_document=manifest,
            child_tags=facts.tags,
            facts_of=store.facts_of,
            artifact_bytes_of=store.artifact_bytes_of,
        )
    except (RunStoreError, TrainingStateManifestError) as error:
        raise RefitPlanError(
            f"{member.config}: 학습 시점 후보 manifest를 확인하지 못했다: {error}"
        ) from error
    expected_manifest = {
        "trajectory_identity_sha256": record.trajectory_identity_sha256,
        "completed_epochs": record.completed_epochs,
        "schedule_horizon_epochs": record.schedule_horizon_epochs,
        "state_kind": record.state_kind,
        "selection_rule": "precommitted",
        "validation_target_used_for_selection": False,
        "seeds": [seed.seed for seed in record.seeds],
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RefitPlanError(
                f"{member.config}: 학습 시점 후보 manifest의 {key}가 재학습 계약과 다르다."
            )
    try:
        config = yaml.safe_load(member.config_path.read_text())
        paths = config["data"]
        current_input_sha256 = {
            name: data.file_sha256(Path(paths[name]))
            for name in ("train", "test", "folds")
        }
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        raise RefitPlanError(
            f"{member.config}: 정확 시점 후보의 현재 입력 자료 해시를 계산하지 못했다."
        ) from error
    if manifest.get("input_sha256") != current_input_sha256:
        raise RefitPlanError(
            f"{member.config}: 현재 입력 자료가 채택된 학습 궤적의 입력과 다르다."
        )


def _allowed_seeds(member: RefitPlanMember, pool_member) -> tuple[int, ...]:
    """장부가 쓸 수 있는 시드. 후보 풀 시드 전부이거나 첫 시드 하나다.

    난수 시드가 결과를 바꾸지 않는 구성원만 첫 시드 하나로 재학습한다. (ADR 0002)
    """
    pool_seeds = tuple(pool_member.seeds)
    recorded = tuple(record.seed for record in member.budget_derivation.seeds)
    if isinstance(member.budget_derivation, TrajectoryStateBudgetRecord):
        if recorded != pool_seeds:
            raise RefitPlanError(
                f"{member.config}: 정확 시점 재학습은 후보 풀의 모든 시드를 써야 한다: "
                f"{list(recorded)} != {list(pool_seeds)}"
            )
        return recorded
    if recorded in (pool_seeds, pool_seeds[:1]):
        return recorded
    raise RefitPlanError(
        f"{member.config}: 계획 시드가 후보 풀 시드와 맞지 않는다: "
        f"{list(recorded)} vs {list(pool_seeds)}"
    )


def _validate_not_applicable(
    member: RefitPlanMember, allowed_seeds: tuple[int, ...]
) -> None:
    evidence = member.evidence
    if evidence.observations:
        raise RefitPlanError(
            f"{member.config}: 반복 수가 없는 구성원에 원시 관측이 있다."
        )
    if evidence.converter is not None:
        raise RefitPlanError(
            f"{member.config}: 반복 수가 없는 구성원에 변환기가 있다."
        )
    for record in member.budget_derivation.seeds:
        if record.observed_lengths or record.median is not None:
            raise RefitPlanError(
                f"{member.config}: 반복 수가 없는 구성원에 계산 중간값이 있다."
            )
        if record.scaled is not None or record.budget is not None:
            raise RefitPlanError(
                f"{member.config}: 반복 수가 없는 구성원에 재학습 예산이 있다."
            )
    if not allowed_seeds:
        raise RefitPlanError(f"{member.config}: 실행할 시드가 없다.")


def _validate_observations(
    member: RefitPlanMember, allowed_seeds: tuple[int, ...]
) -> list[ObservedTrainingLength]:
    """근거의 계열·변환기·좌표를 맞춰 보고 원시 값을 다시 변환한다."""
    evidence = member.evidence
    converters = MODEL_FAMILY_CONVERTERS.get(evidence.model_family)
    if converters is None:
        raise RefitPlanError(
            f"{member.config}: 변환기를 등록하지 않은 모델 계열이다: "
            f"{evidence.model_family!r} (등록: {', '.join(sorted(MODEL_FAMILY_CONVERTERS))})"
        )
    if evidence.converter not in converters:
        raise RefitPlanError(
            f"{member.config}: 모델 계열 {evidence.model_family!r}의 변환기는 "
            f"{list(converters)!r} 중 하나여야 한다: {evidence.converter!r}"
        )
    if not evidence.observations:
        raise RefitPlanError(f"{member.config}: 확정 근거에 원시 관측이 없다.")

    for observation in evidence.observations:
        if observation.raw_meaning != evidence.converter:
            raise RefitPlanError(
                f"{member.config}: 좌표 {observation.coordinate}의 원시 의미가 "
                "구성원이 선언한 변환기와 다르다: "
                f"{observation.raw_meaning!r} != {evidence.converter!r}"
            )

    _validate_coordinates(member, allowed_seeds)

    observed: list[ObservedTrainingLength] = []
    for observation in evidence.observations:
        try:
            recomputed = observed_length_from_raw(
                observation.raw_value, observation.raw_meaning
            )
            if recomputed != observation.observed_training_length:
                raise RefitPlanError(
                    f"{member.config}: 좌표 {observation.coordinate}의 원시 값을 다시 변환한 "
                    f"결과가 저장 관측 학습 길이와 다르다: {recomputed} != "
                    f"{observation.observed_training_length}"
                )
            observed.append(
                observe_training_length(
                    seed=observation.seed,
                    outer_fold=observation.outer_fold,
                    raw_field=observation.raw_field,
                    raw_value=observation.raw_value,
                    raw_meaning=observation.raw_meaning,
                    inner_member=observation.inner_member,
                )
            )
        except TrainingLengthError as error:
            raise RefitPlanError(
                f"{member.config}: 좌표 {observation.coordinate}의 원시 근거가 계약을 어겼다: {error}"
            ) from error
    return observed


def _validate_coordinates(
    member: RefitPlanMember, allowed_seeds: tuple[int, ...]
) -> None:
    """좌표가 빠졌거나 중복됐거나 더 들어왔는지 본다.

    기대 좌표는 계획 시드 × 바깥쪽 분할 × 내부 구성원의 곱 전부다.
    분할과 내부 구성원의 범위는 근거가 스스로 드러내되, 0부터 빈틈없이 이어져야 하고
    바깥쪽 분할 수는 재학습 규약이 고정한 값이어야 한다.
    """
    observations = member.evidence.observations
    coordinates = [observation.coordinate for observation in observations]
    duplicates = _duplicates(coordinates)
    if duplicates:
        raise RefitPlanError(f"{member.config}: 근거 좌표가 중복됐다: {duplicates}")

    folds = sorted({fold for _, fold, _ in coordinates})
    if folds != list(range(OUTER_FOLD_COUNT)):
        raise RefitPlanError(
            f"{member.config}: 바깥쪽 분할 좌표가 0..{OUTER_FOLD_COUNT - 1}과 다르다: {folds}"
        )
    inner_members = {inner for _, _, inner in coordinates}
    if inner_members == {None}:
        expected_inner: list[int | None] = [None]
    elif None in inner_members:
        raise RefitPlanError(
            f"{member.config}: 내부 구성원 좌표가 있는 근거와 없는 근거가 섞였다."
        )
    else:
        expected_inner = sorted(inner_members)
        if expected_inner != list(range(len(expected_inner))):
            raise RefitPlanError(
                f"{member.config}: 내부 구성원 좌표가 0부터 이어지지 않는다: {expected_inner}"
            )

    expected = {
        (seed, fold, inner)
        for seed in allowed_seeds
        for fold in range(OUTER_FOLD_COUNT)
        for inner in expected_inner
    }
    missing = sorted(expected - set(coordinates), key=_coordinate_key)
    if missing:
        raise RefitPlanError(f"{member.config}: 근거 좌표가 빠졌다: {missing}")
    extra = sorted(set(coordinates) - expected, key=_coordinate_key)
    if extra:
        raise RefitPlanError(f"{member.config}: 기대하지 않은 근거 좌표가 들어왔다: {extra}")


def _rederive_budgets(
    member: RefitPlanMember,
    observations: list[ObservedTrainingLength],
    protocol: RefitProtocol,
) -> RefitBudgetDerivation:
    """저장 계산 규약이 계획 규약과 같은지 보고, 원시 근거에서 예산을 다시 계산한다."""
    record = member.budget_derivation
    if (
        record.statistic != protocol.budget_statistic
        or record.multiplier != protocol.iteration_multiplier
        or record.rounding != protocol.budget_rounding
    ):
        raise RefitPlanError(
            f"{member.config}: 저장 계산 규약이 계획 규약과 다르다: "
            f"{record.statistic!r}/{record.multiplier}/{record.rounding!r}"
        )
    try:
        policy = RefitBudgetPolicy(
            statistic=record.statistic,
            multiplier=record.multiplier,
            rounding=record.rounding,
        )
        derivation = derive_refit_budgets(observations, policy)
    except TrainingLengthError as error:
        raise RefitPlanError(f"{member.config}: 재학습 예산을 계산하지 못했다: {error}") from error

    derived = {seed.seed: seed for seed in derivation.seeds}
    recorded = {seed.seed: seed for seed in record.seeds}
    if sorted(derived) != sorted(recorded):
        raise RefitPlanError(
            f"{member.config}: 계산한 시드와 저장 시드가 다르다: "
            f"{sorted(derived)} != {sorted(recorded)}"
        )
    for seed in sorted(derived):
        _compare_seed_budget(member.config, derived[seed], recorded[seed])
    return derivation


def _validate_trajectory_state_derivation(
    member: RefitPlanMember,
    observations: list[ObservedTrainingLength],
    allowed_seeds: tuple[int, ...],
) -> TrajectoryStateBudgetDerivation:
    """동일 궤적 후보는 배수를 쓰지 않고 선택 시점과 정확히 같은 예산만 허용한다."""
    record = member.budget_derivation
    assert isinstance(record, TrajectoryStateBudgetRecord)
    if record.method != TRAJECTORY_STATE_METHOD:
        raise RefitPlanError(
            f"{member.config}: 알 수 없는 정확 시점 재학습 방식이다: {record.method!r}"
        )
    if record.state_kind != "ema":
        raise RefitPlanError(
            f"{member.config}: 정확 시점 재학습 상태 종류는 'ema'여야 한다."
        )
    if not 1 <= record.completed_epochs <= record.schedule_horizon_epochs:
        raise RefitPlanError(
            f"{member.config}: 완료 시점은 1 이상 일정 지평 이하여야 한다."
        )
    try:
        config = yaml.safe_load(member.config_path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise RefitPlanError(
            f"{member.config}: 정확 시점 후보 설정을 읽지 못했다."
        ) from error
    if data.file_sha256(member.config_path) != member.lineage.source_config_sha256:
        raise RefitPlanError(
            f"{member.config}: 정확 시점 재학습 설정이 채택된 후보 실행의 설정과 다르다."
        )
    state = config.get("training_state") if isinstance(config, dict) else None
    expected_config = {
        "selected": record.completed_epochs,
        "schedule_horizon_epochs": record.schedule_horizon_epochs,
        "state_kind": record.state_kind,
        "selection_rule": "precommitted",
    }
    if not isinstance(state, dict) or any(
        state.get(key) != value for key, value in expected_config.items()
    ):
        raise RefitPlanError(
            f"{member.config}: 설정의 training_state가 정확 시점 재학습 기록과 다르다."
        )

    grouped: dict[int, list[int]] = {seed: [] for seed in allowed_seeds}
    for observation in observations:
        grouped[observation.seed].append(observation.value)
    recorded = {seed.seed: seed for seed in record.seeds}
    if tuple(recorded) != allowed_seeds:
        raise RefitPlanError(
            f"{member.config}: 정확 시점 기록 시드가 계획 시드와 다르다."
        )
    validated_seeds: list[SeedBudgetRecord] = []
    for seed in allowed_seeds:
        lengths = tuple(grouped[seed])
        if not lengths or set(lengths) != {record.completed_epochs}:
            raise RefitPlanError(
                f"{member.config}: 시드 {seed}의 모든 관측 학습 길이는 "
                f"정확히 {record.completed_epochs}여야 한다: {lengths}"
            )
        saved = recorded[seed]
        if saved.observed_lengths != lengths or saved.budget != record.completed_epochs:
            raise RefitPlanError(
                f"{member.config}: 시드 {seed}의 정확 시점 예산 기록이 원시 근거와 다르다."
            )
        if saved.median is not None or saved.scaled is not None:
            raise RefitPlanError(
                f"{member.config}: 정확 시점 예산에는 중앙값이나 배수 적용값을 기록할 수 없다."
            )
        validated_seeds.append(saved)
    return TrajectoryStateBudgetDerivation(
        completed_epochs=record.completed_epochs,
        schedule_horizon_epochs=record.schedule_horizon_epochs,
        state_kind=record.state_kind,
        trajectory_identity_sha256=record.trajectory_identity_sha256,
        seeds=tuple(validated_seeds),
    )


def _compare_seed_budget(config: str, derived, recorded: SeedBudgetRecord) -> None:
    """중간값까지 정확히 같아야 한다.

    중앙값은 정수이거나 `.5`이고 배수는 `1.25`이므로 두 값 모두 이진 부동소수로
    정확히 표현된다. 그래서 허용 오차를 두지 않고 그대로 비교해 손댄 값을 드러낸다.
    """
    if recorded.observed_lengths != derived.observed_lengths:
        raise RefitPlanError(
            f"{config}: 시드 {recorded.seed}의 저장 관측 학습 길이 목록이 근거와 다르다."
        )
    if recorded.median != derived.median:
        raise RefitPlanError(
            f"{config}: 시드 {recorded.seed}의 저장 중앙값이 다시 계산한 값과 다르다: "
            f"{recorded.median} != {derived.median}"
        )
    if recorded.scaled != derived.scaled:
        raise RefitPlanError(
            f"{config}: 시드 {recorded.seed}의 저장 배수 적용값이 다시 계산한 값과 다르다: "
            f"{recorded.scaled} != {derived.scaled}"
        )
    if recorded.budget != derived.budget:
        raise RefitPlanError(
            f"{config}: 시드 {recorded.seed}의 저장 재학습 예산이 다시 계산한 값과 다르다: "
            f"{recorded.budget} != {derived.budget}"
        )


def _load_protocol(value: object) -> RefitProtocol:
    protocol = _mapping(value, "protocol")
    _exact_fields(protocol, PROTOCOL_FIELDS, "protocol")
    return RefitProtocol(
        iteration_multiplier=_number(
            protocol["iteration_multiplier"], "protocol.iteration_multiplier"
        ),
        budget_statistic=_text(protocol["budget_statistic"], "protocol.budget_statistic"),
        budget_rounding=_text(protocol["budget_rounding"], "protocol.budget_rounding"),
        cv_model_weight=_integer(protocol["cv_model_weight"], "protocol.cv_model_weight"),
        full_model_weight=_integer(
            protocol["full_model_weight"], "protocol.full_model_weight"
        ),
        combiner=_text(protocol["combiner"], "protocol.combiner"),
    )


def _load_member(
    value: object, index: int, protocol: dict, schema_version: int
) -> RefitPlanMember:
    member = _mapping(value, f"members[{index}]")
    label = member.get("config") if isinstance(member.get("config"), str) else index
    where = f"구성원 {label}"
    _exact_fields(member, MEMBER_FIELDS, where)
    return RefitPlanMember(
        config=_text(member["config"], f"{where}.config"),
        config_path=Path(_text(member["config_path"], f"{where}.config_path")),
        lineage=_load_lineage(member["lineage"], where),
        evidence=_load_evidence(member["training_length_evidence"], where),
        budget_derivation=_load_derivation(
            member["refit_budget_derivation"], where, schema_version
        ),
        entry_sha256=member_entry_sha256(protocol, member),
    )


def member_entry_sha256(protocol: object, member: object) -> str:
    """규약 블록과 구성원 장부 항목 하나를 정규 JSON으로 직렬화한 SHA-256.

    키 순서와 공백을 정규화하므로 YAML의 표기 차이에는 흔들리지 않고, 값이 하나라도
    바뀌면 달라진다. 실행 경로와 조립 관문이 구성원 산출물의 정체성을 이 값으로 맞춘다.
    """
    payload = json.dumps(
        {"protocol": protocol, "member": member},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return sha256_of(payload.encode())


def member_entry_sha256_from_document(document: object, config: str) -> str:
    """장부 문서(파싱한 YAML)에서 이름이 `config`인 구성원의 항목 해시를 계산한다.

    문법 판본 2 기록처럼 항목 해시가 아직 없는 산출물을 검증할 때, 그 기록이 가리키는
    장부 내용에서 같은 계산을 다시 하는 데 쓴다. 장부 문법 전체를 다시 검사하지는 않는다.
    """
    mapping = _mapping(document, "장부")
    protocol = _mapping(mapping.get("protocol"), "protocol")
    matches = [
        item
        for item in _sequence(mapping.get("members"), "members")
        if isinstance(item, dict) and item.get("config") == config
    ]
    if len(matches) != 1:
        raise RefitPlanError(f"장부에 구성원 {config!r}이 정확히 하나 있지 않다.")
    return member_entry_sha256(protocol, matches[0])


def _load_lineage(value: object, where: str) -> MemberLineage:
    lineage = _mapping(value, f"{where}.lineage")
    _exact_fields(lineage, LINEAGE_FIELDS, f"{where}.lineage")
    return MemberLineage(
        **{
            field: _text(lineage[field], f"{where}.lineage.{field}")
            for field in LINEAGE_FIELDS
        }
    )


def _load_evidence(value: object, where: str) -> TrainingLengthEvidence:
    evidence = _mapping(value, f"{where}.training_length_evidence")
    _exact_fields(evidence, EVIDENCE_FIELDS, f"{where}.training_length_evidence")
    status = _text(evidence["status"], f"{where}.status")
    if status not in STATUSES:
        raise RefitPlanError(
            f"{where}: 알 수 없는 근거 상태다: {status!r} (가능한 값: {list(STATUSES)})"
        )
    converter = evidence["converter"]
    if converter is not None:
        converter = _text(converter, f"{where}.converter")
        if converter not in RAW_MEANINGS:
            raise RefitPlanError(
                f"{where}: 알 수 없는 변환기다: {converter!r} "
                f"(가능한 값: {list(RAW_MEANINGS)})"
            )
    observations = tuple(
        _load_observation(item, f"{where}.observations[{index}]")
        for index, item in enumerate(
            _sequence(evidence["observations"], f"{where}.observations")
        )
    )
    return TrainingLengthEvidence(
        status=status,
        model_family=_text(evidence["model_family"], f"{where}.model_family"),
        converter=converter,
        observations=observations,
    )


def _load_observation(value: object, where: str) -> RawObservation:
    observation = _mapping(value, where)
    _exact_fields(observation, OBSERVATION_FIELDS, where)
    inner_member = observation["inner_member"]
    return RawObservation(
        seed=_integer(observation["seed"], f"{where}.seed"),
        outer_fold=_integer(observation["outer_fold"], f"{where}.outer_fold"),
        inner_member=(
            None
            if inner_member is None
            else _integer(inner_member, f"{where}.inner_member")
        ),
        raw_field=_text(observation["raw_field"], f"{where}.raw_field"),
        raw_value=_integer(observation["raw_value"], f"{where}.raw_value"),
        raw_meaning=_text(observation["raw_meaning"], f"{where}.raw_meaning"),
        observed_training_length=_integer(
            observation["observed_training_length"],
            f"{where}.observed_training_length",
        ),
    )


def _load_derivation(
    value: object, where: str, schema_version: int
) -> RefitBudgetRecord | TrajectoryStateBudgetRecord:
    derivation = _mapping(value, f"{where}.refit_budget_derivation")
    if "method" in derivation:
        if schema_version < 3:
            raise RefitPlanError(
                f"{where}: trajectory_state 재학습 계약은 장부 문법 판본 3부터 지원한다."
            )
        _exact_fields(
            derivation,
            TRAJECTORY_DERIVATION_FIELDS,
            f"{where}.refit_budget_derivation",
        )
        method = _text(derivation["method"], f"{where}.method")
        if method != TRAJECTORY_STATE_METHOD:
            raise RefitPlanError(
                f"{where}: 알 수 없는 재학습 예산 방식이다: {method!r}"
            )
        seeds = tuple(
            _load_trajectory_seed_budget(
                item, f"{where}.refit_budget_derivation.seeds[{index}]"
            )
            for index, item in enumerate(
                _sequence(
                    derivation["seeds"], f"{where}.refit_budget_derivation.seeds"
                )
            )
        )
        _validate_seed_records(seeds, where)
        completed_epochs = _integer(
            derivation["completed_epochs"], f"{where}.completed_epochs"
        )
        schedule_horizon_epochs = _integer(
            derivation["schedule_horizon_epochs"],
            f"{where}.schedule_horizon_epochs",
        )
        trajectory_sha = _text(
            derivation["trajectory_identity_sha256"],
            f"{where}.trajectory_identity_sha256",
        )
        if len(trajectory_sha) != 64 or any(
            character not in "0123456789abcdef" for character in trajectory_sha.lower()
        ):
            raise RefitPlanError(f"{where}: 학습 궤적 정체성 SHA-256 형식이 잘못됐다.")
        return TrajectoryStateBudgetRecord(
            method=method,
            completed_epochs=completed_epochs,
            schedule_horizon_epochs=schedule_horizon_epochs,
            state_kind=_text(derivation["state_kind"], f"{where}.state_kind"),
            trajectory_identity_sha256=trajectory_sha,
            seeds=seeds,
        )
    _exact_fields(derivation, DERIVATION_FIELDS, f"{where}.refit_budget_derivation")
    seeds = tuple(
        _load_seed_budget(item, f"{where}.refit_budget_derivation.seeds[{index}]")
        for index, item in enumerate(
            _sequence(derivation["seeds"], f"{where}.refit_budget_derivation.seeds")
        )
    )
    _validate_seed_records(seeds, where)
    return RefitBudgetRecord(
        statistic=_text(derivation["statistic"], f"{where}.statistic"),
        multiplier=_number(derivation["multiplier"], f"{where}.multiplier"),
        rounding=_text(derivation["rounding"], f"{where}.rounding"),
        seeds=seeds,
    )


def _validate_seed_records(seeds: tuple[SeedBudgetRecord, ...], where: str) -> None:
    if not seeds:
        raise RefitPlanError(f"{where}: 재학습 예산 계산 결과에 시드가 없다.")
    duplicates = _duplicates([seed.seed for seed in seeds])
    if duplicates:
        raise RefitPlanError(f"{where}: 같은 시드가 두 번 있다: {duplicates}")


def _load_trajectory_seed_budget(value: object, where: str) -> SeedBudgetRecord:
    seed_budget = _mapping(value, where)
    _exact_fields(seed_budget, TRAJECTORY_SEED_BUDGET_FIELDS, where)
    lengths = tuple(
        _integer(item, f"{where}.observed_lengths[{index}]")
        for index, item in enumerate(
            _sequence(seed_budget["observed_lengths"], f"{where}.observed_lengths")
        )
    )
    return SeedBudgetRecord(
        seed=_integer(seed_budget["seed"], f"{where}.seed"),
        observed_lengths=lengths,
        median=None,
        scaled=None,
        budget=_integer(seed_budget["budget"], f"{where}.budget"),
    )


def _load_seed_budget(value: object, where: str) -> SeedBudgetRecord:
    seed_budget = _mapping(value, where)
    _exact_fields(seed_budget, SEED_BUDGET_FIELDS, where)
    lengths = tuple(
        _integer(item, f"{where}.observed_lengths[{index}]")
        for index, item in enumerate(
            _sequence(seed_budget["observed_lengths"], f"{where}.observed_lengths")
        )
    )
    return SeedBudgetRecord(
        seed=_integer(seed_budget["seed"], f"{where}.seed"),
        observed_lengths=lengths,
        median=_optional_number(seed_budget["median"], f"{where}.median"),
        scaled=_optional_number(seed_budget["scaled"], f"{where}.scaled"),
        budget=(
            None
            if seed_budget["budget"] is None
            else _integer(seed_budget["budget"], f"{where}.budget")
        ),
    )


def _exact_fields(mapping: dict, expected: tuple[str, ...], where: str) -> None:
    """필드 집합이 정확히 같아야 한다. 빠진 필드도 더 들어온 필드도 거부한다."""
    present = set(mapping)
    missing = sorted(set(expected) - present)
    if missing:
        raise RefitPlanError(f"{where}: 필수 필드가 없다: {missing}")
    unknown = sorted(present - set(expected))
    if unknown:
        hand_edited = [
            field
            for field in unknown
            if any(marker in field.lower() for marker in HAND_EDIT_MARKERS)
        ]
        if hand_edited:
            raise RefitPlanError(
                f"{where}: 손으로 바꾸는 예산이나 예외 필드는 장부 문법에 없다: {hand_edited}"
            )
        raise RefitPlanError(f"{where}: 알 수 없는 필드다: {unknown}")


def _mapping(value: object, where: str) -> dict:
    if not isinstance(value, dict):
        raise RefitPlanError(f"{where}: 이름 있는 항목의 묶음이어야 한다: {type(value).__name__}")
    unnamed = [key for key in value if not isinstance(key, str)]
    if unnamed:
        raise RefitPlanError(f"{where}: 필드 이름은 문자열이어야 한다: {unnamed}")
    return value


def _sequence(value: object, where: str) -> list:
    if not isinstance(value, list):
        raise RefitPlanError(f"{where}: 목록이어야 한다: {type(value).__name__}")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RefitPlanError(f"{where}: 비어 있지 않은 문자열이어야 한다: {value!r}")
    return value


def _integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RefitPlanError(f"{where}: 정수여야 한다: {value!r}")
    return value


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RefitPlanError(f"{where}: 수여야 한다: {value!r}")
    return float(value)


def _optional_number(value: object, where: str) -> float | None:
    return None if value is None else _number(value, where)


def _duplicates(values) -> list:
    seen = set()
    repeated = []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return repeated


def _coordinate_key(coordinate: tuple[int, int, int | None]) -> tuple[int, int, int]:
    seed, fold, inner = coordinate
    return (seed, fold, -1 if inner is None else inner)


def _describe(plan: ExecutableRefitPlan) -> str:
    lines = [
        f"{plan.source_path}: 구성원 {len(plan.members)}개, "
        f"전체 자료 재학습 {sum(len(member.budgets) for member in plan.members)}회",
        f"결합 방식 {plan.protocol.combiner}, 후보 풀 {plan.source_pool_sha256}",
    ]
    for member in plan.members:
        budgets = ", ".join(
            f"{seed}: {'해당 없음' if budget is None else budget}"
            for seed, budget in member.budgets.items()
        )
        lines.append(f"  {member.config:48} {budgets}")
    return "\n".join(lines)


def main() -> None:
    """장부를 관문에 통과시켜 본다. 학습하지 않고 아무것도 저장하지 않는다.

    실행 저장소와 현재 후보 풀을 실제로 읽으므로, 이 명령이 통과하면 전체 자료 재학습이
    읽을 숫자가 원시 근거에서 다시 계산한 값과 같다는 뜻이다.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="재학습 계획 장부를 검증한다.")
    parser.add_argument("path", type=Path, help="재학습 계획 장부 경로")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="검증만 하고 아무것도 실행하지 않는다(이 명령의 유일한 동작이라 기본값과 같다)",
    )
    parser.add_argument(
        "--syntax-only",
        action="store_true",
        help="실행 저장소를 읽지 않고 문법만 본다",
    )
    args = parser.parse_args()

    try:
        plan = RefitPlan.load(args.path)
        if args.syntax_only:
            print(f"{args.path}: 문법 판본 {plan.schema_version}, 구성원 {len(plan.members)}개")
            return
        print(_describe(plan.validate_for_refit()))
    except (RefitPlanError, RunStoreError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
