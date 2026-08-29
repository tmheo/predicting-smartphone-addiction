"""피처 계획. 실험 실행 하나가 학습할 컬럼 전체를 선언·검증·조율한다. (#71)

계획은 설정의 providers 목록을 적용 단계별로 정렬해 소유하고 두 불변식을 강제한다.

- 적용 순서: base -> dataset-wide -> row-wise -> placebo 자동 삽입 -> fold-fit.
  같은 단계 안에서는 providers 목록의 등장 순서가 곧 컬럼 순서다.
- placebo 상시 포함: 설정 항목이 아니라 계획의 내장 불변식이다(ADR 0001의 상시 카나리아 요구).

제공자 구현(인코더, 파생 함수)은 pipeline.features에 있고,
kind -> (적용 단계, 팩토리) 통합 레지스트리는 이 모듈이 소유한다.

누출 규율은 설정 적재 시점에 검증한다: 타깃 참조 제공자는 fold-fit 단계여야 하고,
placebo 파생 카나리아 컬럼을 하나 이상 선언해야 한다. 이 검증이 카나리아 0개 실행이
compare의 카나리아 게이트를 공허하게 통과하는 구멍을 적재 시점에 막는다.

선언 = 실제: apply_dataset_wide, build_matrix, add_fold_fit_columns가 제공자의 실제
산출 컬럼이 columns() 선언과 다르면 즉시 실패한다.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from . import features
from .denoising_autoencoder import DenoisingAutoencoderLatent
from .config import FeatureConfig
from .data import ID, TARGET
from .features import PLACEBO
from .fold_fit_reuse import (
    FoldFitReuseError,
    FoldFitReuseRequest,
    FoldFitReuseStore,
    canonical_json_bytes,
    provider_identity_document,
    validate_input_files,
    validate_runtime_identity,
)
from .fold_observability import recorded_operation, skipped_operation, timed_operation
from .training_rows import PARENT_ID

# 적용 단계. 원본 프록시 prior(#53)는 별도 단계 대신 row-wise로 들어갔다:
# 통계표가 해시 고정된 외부 파일에서 학습 전에 확정되므로 행 단위 결정적 매핑이고,
# 다른 행도 대회 타깃도 보지 않아 fold 분리가 필요 없다.
DATASET_WIDE = "dataset-wide"  # 훈련+테스트를 함께 보고 컬럼을 만든다.
ROW_WISE = "row-wise"  # 행 단위 결정적으로 컬럼을 만든다.
FOLD_FIT = "fold-fit"  # fold 루프 안에서 학습 fold로만 fit한다.
STAGE_ORDER = (DATASET_WIDE, ROW_WISE, FOLD_FIT)


class DatasetWideProvider(Protocol):
    """dataset-wide 컬럼 제공자. 훈련+테스트 전체를 보고 두 데이터의 새 컬럼을 계산한다.

    타깃을 참조할 수 없으므로(적재 시점 검증) fold 분리가 필요 없다.
    compute는 (train 새 컬럼 frame, test 새 컬럼 frame)을 각 입력과 같은 인덱스로 돌려준다.
    """

    uses_target: bool

    def columns(self) -> list[str]: ...

    def compute(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...


class RowWiseProvider(Protocol):
    """row-wise 컬럼 제공자. 다른 행을 보지 않고 행 단위로 결정적으로 컬럼을 계산한다.

    compute는 입력과 같은 인덱스의 새 컬럼 frame을 돌려준다.
    """

    uses_target: bool

    def columns(self) -> list[str]: ...

    def compute(self, df: pd.DataFrame) -> pd.DataFrame: ...


class FoldFitTransformer(Protocol):
    """fold 루프 안에서 학습하는 fold-fit 컬럼 제공자. (#32, #35, #71에서 선언 추가)

    fit은 타깃이 포함된 학습 fold의 DataFrame을 받아 상태를 새로 계산하고
    (fold마다 다시 불리므로 이전 fold의 상태를 남기면 안 된다),
    transform은 DataFrame을 받아 같은 인덱스의 새 컬럼 DataFrame을 돌려준다.
    두 입력 모두 원본 컬럼에 build_matrix 산출 컬럼(placebo 등)이 더해진 형태다. (#33 파급)

    transform은 fold마다 train 전체(학습 fold + 검증 fold 행)와 test로 두 번 불린다.
    학습 fold 행에 OOF 값을 줘야 하는 트랜스포머(타깃 인코딩의 내부 K-fold)는
    fit 때 학습 행의 id 집합을 저장해 행별로 OOF 값과 평균표 값을 구분해 돌려준다.
    위치 인덱스는 train과 test가 겹치므로 구분 기준은 id여야 한다.
    """

    uses_target: bool

    def columns(self) -> list[str]: ...

    def reuse_input_columns(self) -> list[str]: ...

    def reuse_settings(self) -> dict[str, object]: ...

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None: ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class ProviderKind:
    stage: str
    factory: Callable[..., Any]


class FeatureContractError(Exception):
    """컬럼 제공자의 선언 입력·출력 계약 위반."""


# kind -> (적용 단계, 팩토리). 새 컬럼 제공자는 features에 구현을 추가하고 여기 등록한다.
REGISTRY: dict[str, ProviderKind] = {
    "categorical_copies": ProviderKind(DATASET_WIDE, features.CategoricalCopies),
    "pair_ce": ProviderKind(DATASET_WIDE, features.PairCE),
    "derived": ProviderKind(ROW_WISE, features.DerivedColumns),
    "missing_indicators": ProviderKind(ROW_WISE, features.MissingIndicators),
    "original_prior": ProviderKind(ROW_WISE, features.OriginalPriorColumns),
    "original_knn": ProviderKind(ROW_WISE, features.OriginalKnnColumns),
    "original_cdf_diff": ProviderKind(ROW_WISE, features.OriginalClassCdfDiff),
    "original_kde_ratio": ProviderKind(ROW_WISE, features.OriginalKdeLogRatio),
    "target_encoding": ProviderKind(FOLD_FIT, features.ExactValueTargetEncoder),
    "lattice_pair_te": ProviderKind(FOLD_FIT, features.LatticePairTargetEncoder),
    "rich_lattice_encoding": ProviderKind(FOLD_FIT, features.RichLatticeEncoder),
    "frequency_encoding": ProviderKind(FOLD_FIT, features.FrequencyEncoder),
    "median_impute_aux": ProviderKind(FOLD_FIT, features.MedianImputeAux),
    "constrained_impute_aux": ProviderKind(FOLD_FIT, features.ConstrainedImputeAux),
    "xgb_impute_aux": ProviderKind(FOLD_FIT, features.XgbImputeAux),
    "dae_latent": ProviderKind(FOLD_FIT, DenoisingAutoencoderLatent),
}


def _base_columns_of(df: pd.DataFrame) -> list[str]:
    """base: raw의 결정적 정의. 원시 CSV 컬럼 전부에서 ID와 타깃만 뺀다. (#71)"""
    return [c for c in df.columns if c not in (ID, TARGET, "fold")]


class FeaturePlan:
    """실험이 학습할 컬럼 전체의 선언. run.py가 설정에서 한 번 만들어 cv에 주입한다."""

    def __init__(
        self,
        stages: dict[str, list[tuple[str, Any]]],
        exclude: list[str],
        fold_fit_specs: list[tuple[str, Callable[..., Any], dict[str, Any]]],
    ) -> None:
        self._stages = stages  # stage -> [(kind, provider), ...] providers 목록 순서 유지
        self._exclude = exclude  # base에서 뺄 raw 컬럼. 제공자 입력에는 남는다. (#79)
        self._fold_fit_specs = fold_fit_specs
        self._base_columns: list[str] | None = None
        self._raw_columns: list[str] | None = None
        self._fold_fit_reuse_store: FoldFitReuseStore | None = None
        self._fold_fit_runtime_identity: dict[str, object] | None = None
        self._fold_fit_input_files: dict[str, str] | None = None

    @classmethod
    def from_config(cls, cfg: FeatureConfig) -> FeaturePlan:
        """설정을 계획으로 만든다. 누출 규율과 선언 충돌은 여기서, 즉 적재 시점에 거부한다."""
        if cfg.base != "raw":
            raise ValueError(
                f"features.base는 'raw'만 지원한다(받은 값: {cfg.base!r}). "
                "명시적 컬럼 목록은 필요가 생길 때 추가한다. (#71)"
            )
        if PLACEBO in cfg.exclude:
            raise ValueError(
                f"features.exclude에 {PLACEBO}를 넣을 수 없다. "
                "placebo 상시 포함은 계획의 내장 불변식이다. (ADR 0001)"
            )
        overlap = set(cfg.exclude) & set(cfg.categorical)
        if overlap:
            raise ValueError(
                f"features.exclude와 categorical이 겹친다: {sorted(overlap)}. "
                "제외한 컬럼은 학습 행렬에 없으므로 categorical 선언도 같이 뺄 것."
            )
        stages: dict[str, list[tuple[str, Any]]] = {stage: [] for stage in STAGE_ORDER}
        fold_fit_specs: list[tuple[str, Callable[..., Any], dict[str, Any]]] = []
        for i, spec in enumerate(cfg.providers):
            params = dict(spec)
            kind = params.pop("kind", None)
            if kind not in REGISTRY:
                raise ValueError(
                    f"providers[{i}]: 알 수 없는 kind {kind!r}. "
                    f"등록된 kind: {', '.join(sorted(REGISTRY))}"
                )
            entry = REGISTRY[kind]
            try:
                provider = entry.factory(**params)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"providers[{i}] {kind}: {exc}") from exc
            stages[entry.stage].append((kind, provider))
            if entry.stage == FOLD_FIT:
                fold_fit_specs.append((kind, entry.factory, params))
        plan = cls(stages, list(cfg.exclude), fold_fit_specs)
        plan._validate_declarations()
        return plan

    def _validate_declarations(self) -> None:
        declared: dict[str, str] = {PLACEBO: "placebo(내장)"}
        for stage in STAGE_ORDER:
            for kind, provider in self._stages[stage]:
                if provider.uses_target and stage != FOLD_FIT:
                    raise ValueError(
                        f"{kind}는 타깃을 참조하므로 fold-fit 단계여야 한다(현재 {stage}). "
                        "타깃 참조 제공자를 fold 밖에서 실행하면 누출이다."
                    )
                if provider.uses_target and not any(PLACEBO in c for c in provider.columns()):
                    raise ValueError(
                        f"{kind}는 타깃을 참조하므로 {PLACEBO}를 포함한 spec을 하나 이상 "
                        "가져야 한다(단일이면 placebo_noise 자체, 결합이면 placebo_noise가 "
                        "든 목록). 카나리아 없는 타깃 참조는 누출 판정이 불가능하다. (#33, #71)"
                    )
                if stage == FOLD_FIT:
                    self._validate_fold_fit_reuse_declaration(kind, provider)
                for col in provider.columns():
                    if col in declared:
                        raise ValueError(
                            f"선언 컬럼 충돌: {col} ({declared[col]}와 {kind}가 같이 선언)"
                        )
                    declared[col] = kind
        provider_overlap = set(self._exclude) & set(declared)
        if provider_overlap:
            raise ValueError(
                f"features.exclude는 raw 컬럼 전용인데 제공자 컬럼이 섞였다: "
                f"{sorted(provider_overlap)}. 제공자 컬럼을 빼려면 provider 선언을 고칠 것."
            )

    @staticmethod
    def _validate_fold_fit_reuse_declaration(
        kind: str, provider: FoldFitTransformer
    ) -> None:
        try:
            inputs = provider.reuse_input_columns()
            settings = provider.reuse_settings()
        except AttributeError as exc:
            raise ValueError(
                f"{kind} fold-fit 제공자가 재사용 입력·설정 선언을 구현하지 않았다."
            ) from exc
        if not isinstance(inputs, list) or any(
            not isinstance(column, str) or not column for column in inputs
        ):
            raise ValueError(f"{kind} 재사용 입력 열 선언이 잘못됐다: {inputs!r}")
        if len(set(inputs)) != len(inputs):
            raise ValueError(f"{kind} 재사용 입력 열 선언에 중복이 있다: {inputs}")
        forbidden = {ID, TARGET} & set(inputs)
        if forbidden:
            raise ValueError(
                f"{kind} 재사용 입력 열에는 행 식별자와 타깃을 직접 선언하지 않는다: "
                f"{sorted(forbidden)}"
            )
        if not isinstance(settings, dict):
            raise ValueError(f"{kind} 재사용 설정 선언은 객체여야 한다.")
        try:
            canonical_json_bytes(settings)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{kind} 재사용 설정 선언을 정규 JSON으로 만들 수 없다.") from exc

    # ------------------------------------------------------------- 적용

    def apply_dataset_wide(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """dataset-wide 컬럼을 더한 새 frame 쌍을 돌려준다. 입력은 변형하지 않는다.

        base 컬럼(raw)이 여기서 확정된다. build_matrix보다 먼저 불려야 한다.
        """
        raw = _base_columns_of(train)
        assert _base_columns_of(test) == raw, "train/test의 raw 컬럼이 다르다."
        unknown = set(self._exclude) - set(raw)
        assert not unknown, f"features.exclude에 raw에 없는 컬럼이 있다: {sorted(unknown)}"
        provider_cols = {c for cols in self._declared_by_stage().values() for c in cols}
        overlap = provider_cols & set(raw)
        assert not overlap, f"제공자 선언 컬럼이 raw 컬럼과 충돌: {sorted(overlap)}"
        # 제외는 학습 행렬(base)에서만 적용한다. 원본 frame에는 남아 제공자 입력이 된다.
        self._base_columns = [c for c in raw if c not in self._exclude]
        self._raw_columns = list(raw)
        for kind, provider in self._stages[DATASET_WIDE]:
            new_train, new_test = provider.compute(train, test)
            for name, source, new in (("train", train, new_train), ("test", test, new_test)):
                assert new.index.equals(source.index), f"{kind}의 {name} 산출 인덱스가 다르다."
                assert list(new.columns) == provider.columns(), (
                    f"{kind}의 {name} 산출 컬럼이 선언과 다르다: "
                    f"{list(new.columns)} != {provider.columns()}"
                )
            train = pd.concat([train, new_train], axis=1)
            test = pd.concat([test, new_test], axis=1)
        return train, test

    def build_matrix(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        """모델 입력 행렬(base + dataset-wide + row-wise + placebo)을 만든다.

        fold-fit 컬럼은 fold 루프에서 add_fold_fit_columns로 더해진다.
        """
        assert self._base_columns is not None, "apply_dataset_wide 이전에 build_matrix를 불렀다."
        cols = list(self._base_columns)
        for kind, provider in self._stages[DATASET_WIDE]:
            cols += provider.columns()
        X = df[cols].copy()
        for kind, provider in self._stages[ROW_WISE]:
            new = provider.compute(df)
            assert new.index.equals(df.index), f"{kind}의 산출 인덱스가 원본과 다르다."
            assert list(new.columns) == provider.columns(), (
                f"{kind}의 산출 컬럼이 선언과 다르다: {list(new.columns)} != {provider.columns()}"
            )
            collision = set(new.columns) & set(X.columns)
            assert not collision, f"row-wise 컬럼 이름 충돌: {sorted(collision)}"
            for name in provider.columns():
                X[name] = new[name]
        # placebo 자동 삽입: 개선 판정 기준선. 설정으로 끌 수 없다. (#15, ADR 0001)
        assert PLACEBO not in X.columns, f"{PLACEBO} 컬럼이 이미 있다."
        X[PLACEBO] = features.placebo_series(df, seed)
        assert list(X.columns) == self.matrix_columns(), "행렬 컬럼이 선언과 다르다."
        return X

    def fold_fit_transformers(self) -> list[FoldFitTransformer]:
        """fold 루프가 fit할 fold-fit 제공자들. providers 목록 순서."""
        return [provider for _, provider in self._stages[FOLD_FIT]]

    def fold_fit_providers(self) -> list[tuple[str, FoldFitTransformer]]:
        """관측 가능한 제공자 이름과 구현을 설정의 선언 순서로 돌려준다."""
        return list(self._stages[FOLD_FIT])

    def new_fold_fit_providers(self) -> list[tuple[str, FoldFitTransformer]]:
        """한 폴드에만 사용할 새 fold-fit 제공자들을 선언 순서로 만든다."""
        return [
            (kind, factory(**copy.deepcopy(params)))
            for kind, factory, params in self._fold_fit_specs
        ]

    def configure_fold_fit_reuse(
        self,
        store: FoldFitReuseStore,
        *,
        runtime_identity: dict[str, object],
        input_files: dict[str, str],
    ) -> None:
        """실행 진입점이 검증한 공유 저장소와 판본 정체성을 내부 연산에 연결한다."""
        validate_runtime_identity(runtime_identity)
        validate_input_files(input_files)
        self._fold_fit_reuse_store = store
        self._fold_fit_runtime_identity = copy.deepcopy(runtime_identity)
        self._fold_fit_input_files = dict(input_files)

    def fold_fit_reuse_enabled(self) -> bool:
        return self._fold_fit_reuse_store is not None

    def unused_fold_fit_reuse_evidence(
        self, *, seed: int, fold: int, reason: str
    ) -> list[dict[str, object]]:
        return [
            {
                "seed": seed,
                "fold": fold,
                "provider": kind,
                "status": "unused",
                "reason": reason,
                "key": None,
                "manifest_sha256": None,
            }
            for kind, _ in self._stages[FOLD_FIT]
        ]

    def materialize_fold_fit_provider(
        self,
        *,
        kind: str,
        transformer: FoldFitTransformer,
        train_input: pd.DataFrame,
        test_input: pd.DataFrame,
        training_index: pd.Index,
        validation_index: pd.Index,
        seed: int,
        fold: int,
        recorder: object | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        """제공자 하나의 입력 제한, 재사용, 계산과 산출 선언 검증을 소유한다."""
        declared_inputs = transformer.reuse_input_columns()
        self._validate_fold_fit_reuse_declaration(kind, transformer)
        context_columns = list(
            getattr(transformer, "training_row_context_columns", lambda: [])()
        )
        unknown_context = sorted(set(context_columns) - {PARENT_ID})
        if unknown_context:
            raise FeatureContractError(
                f"{kind}가 알 수 없는 학습 행 변환 문맥을 선언했다: {unknown_context}"
            )
        active_context = [
            column for column in context_columns if column in train_input.columns
        ]
        required = [ID, *declared_inputs, *active_context]
        missing_train = [column for column in required if column not in train_input.columns]
        missing_test = [
            column
            for column in [ID, *declared_inputs]
            if column not in test_input.columns
        ]
        if missing_train or missing_test:
            raise FeatureContractError(
                f"{kind} 선언 입력이 없다: train={missing_train}, test={missing_test}"
            )
        provider_train = train_input[required].copy(deep=False)
        provider_test = test_input[[ID, *declared_inputs]].copy(deep=False)
        for column in active_context:
            provider_test[column] = pd.NA
        fit_train = provider_train
        if transformer.uses_target:
            if TARGET not in train_input.columns:
                raise FeatureContractError(f"{kind} 타깃 참조 제공자의 학습 타깃이 없다.")
            fit_train = pd.concat([provider_train, train_input[[TARGET]]], axis=1)

        output_columns = transformer.columns()

        def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
            try:
                with timed_operation(
                    recorder,
                    seed=seed,
                    fold=fold,
                    operation="fold_feature.provider_fit",
                    actor_kind="column_provider",
                    actor_name=kind,
                ):
                    transformer.fit(fit_train.loc[training_index], seed)
                with timed_operation(
                    recorder,
                    seed=seed,
                    fold=fold,
                    operation="fold_feature.provider_transform",
                    actor_kind="column_provider",
                    actor_name=kind,
                    dataset="train",
                ):
                    new_train = transformer.transform(provider_train)
                    self._validate_fold_fit_output(
                        kind, transformer, provider_train, new_train, "train"
                    )
                with timed_operation(
                    recorder,
                    seed=seed,
                    fold=fold,
                    operation="fold_feature.provider_transform",
                    actor_kind="column_provider",
                    actor_name=kind,
                    dataset="test",
                ):
                    new_test = transformer.transform(provider_test)
                    self._validate_fold_fit_output(
                        kind, transformer, provider_test, new_test, "test"
                    )
            except KeyError as exc:
                raise FeatureContractError(
                    f"{kind}가 선언하지 않은 입력 열을 읽으려 했다: {exc}"
                ) from exc
            return (
                pd.concat([provider_train[[ID]], new_train], axis=1),
                pd.concat([provider_test[[ID]], new_test], axis=1),
            )

        if self._fold_fit_reuse_store is None:
            train_values, test_values = compute()
            return (
                train_values.drop(columns=[ID]),
                test_values.drop(columns=[ID]),
                {
                    "seed": seed,
                    "fold": fold,
                    "provider": kind,
                    "status": "unused",
                    "reason": "disabled",
                    "key": None,
                    "manifest_sha256": None,
                    "fit_evidence": _fit_evidence_of(transformer),
                },
            )

        assert self._fold_fit_runtime_identity is not None
        assert self._fold_fit_input_files is not None
        external_hashes = getattr(
            transformer, "reuse_external_file_sha256", lambda: {}
        )()
        execution = getattr(transformer, "reuse_execution", lambda: {"mode": "cpu"})()
        provider_identity = provider_identity_document(
            kind=kind,
            provider=transformer,
            input_columns=declared_inputs,
            output_columns=output_columns,
            uses_target=transformer.uses_target,
            settings=transformer.reuse_settings(),
            external_file_sha256=external_hashes,
            execution=execution,
        )
        request = FoldFitReuseRequest(
            provider=provider_identity,
            runtime=self._fold_fit_runtime_identity,
            input_files=self._fold_fit_input_files,
            seed=seed,
            fold=fold,
            train_input=provider_train,
            test_input=provider_test,
            training_ids=provider_train.loc[training_index, ID],
            validation_ids=provider_train.loc[validation_index, ID],
            test_ids=provider_test[ID],
            training_target=(
                train_input.loc[training_index, TARGET]
                if transformer.uses_target
                else None
            ),
        )
        started_ns = time.monotonic_ns()
        computed = False

        def tracked_compute() -> tuple[pd.DataFrame, pd.DataFrame]:
            nonlocal computed
            computed = True
            return compute()

        try:
            result = self._fold_fit_reuse_store.resolve(request, tracked_compute)
        except FoldFitReuseError:
            if not computed:
                recorded_operation(
                    recorder,
                    seed=seed,
                    fold=fold,
                    operation="fold_feature.provider_fit",
                    actor_kind="column_provider",
                    actor_name=kind,
                    started_ns=started_ns,
                    duration_ns=time.monotonic_ns() - started_ns,
                    outcome="failed",
                    reason="FoldFitReuseError",
                )
            raise
        if not computed:
            recorded_operation(
                recorder,
                seed=seed,
                fold=fold,
                operation="fold_feature.provider_fit",
                actor_kind="column_provider",
                actor_name=kind,
                started_ns=started_ns,
                duration_ns=time.monotonic_ns() - started_ns,
                outcome="reused",
                reason="fold_fit_reuse_hit",
            )
            for dataset in ("train", "test"):
                skipped_operation(
                    recorder,
                    seed=seed,
                    fold=fold,
                    operation="fold_feature.provider_transform",
                    actor_kind="column_provider",
                    actor_name=kind,
                    dataset=dataset,
                    reason="fold_fit_reuse_hit",
                )
        if not result.train[ID].reset_index(drop=True).equals(
            provider_train[ID].reset_index(drop=True)
        ) or not result.test[ID].reset_index(drop=True).equals(
            provider_test[ID].reset_index(drop=True)
        ):
            raise FoldFitReuseError(f"{kind} 재사용 결과의 행 식별자나 순서가 다르다.")
        train_values = result.train.drop(columns=[ID])
        test_values = result.test.drop(columns=[ID])
        train_values.index = provider_train.index
        test_values.index = provider_test.index
        return (
            train_values,
            test_values,
            {
                "seed": seed,
                "fold": fold,
                "provider": kind,
                "status": result.status,
                "reason": None,
                "key": result.key,
                "manifest_sha256": result.manifest_sha256,
                # 이 실행이 직접 계산했을 때만 제공자의 적합 계보(학습 횟수 등)를 남긴다.
                "fit_evidence": _fit_evidence_of(transformer) if computed else None,
            },
        )

    @staticmethod
    def _validate_fold_fit_output(
        kind: str,
        transformer: FoldFitTransformer,
        source: pd.DataFrame,
        new: pd.DataFrame,
        dataset: str,
    ) -> None:
        if not new.index.equals(source.index):
            raise FeatureContractError(f"{kind}의 {dataset} 산출 인덱스가 다르다.")
        if list(new.columns) != transformer.columns():
            raise FeatureContractError(
                f"{kind}의 {dataset} 산출 컬럼이 선언과 다르다: "
                f"{list(new.columns)} != {transformer.columns()}"
            )

    @staticmethod
    def add_fold_fit_provider_columns(
        X: pd.DataFrame,
        df: pd.DataFrame,
        kind: str,
        transformer: FoldFitTransformer,
    ) -> pd.DataFrame:
        """fit이 끝난 제공자 하나의 선언 검증과 컬럼 추가를 수행한다."""
        new = transformer.transform(df)
        assert new.index.equals(df.index), f"{kind}의 transform 인덱스가 원본과 다르다."
        assert list(new.columns) == transformer.columns(), (
            f"{kind}의 산출 컬럼이 선언과 다르다: "
            f"{list(new.columns)} != {transformer.columns()}"
        )
        collision = set(new.columns) & set(X.columns)
        assert not collision, f"fold-fit 컬럼 이름 충돌: {sorted(collision)}"
        return pd.concat([X, new], axis=1)

    def add_fold_fit_columns(self, X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """fit된 fold-fit 제공자들의 새 컬럼을 X에 붙인 행렬을 돌려준다. 추가 전용."""
        out = X
        for kind, transformer in self._stages[FOLD_FIT]:
            out = self.add_fold_fit_provider_columns(out, df, kind, transformer)
        return out

    def build_full_matrices(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """전체 학습 자료로 fold-fit 제공자를 맞춘 최종 학습·시험 행렬을 만든다.

        타깃 인코더처럼 학습 행에 내부 OOF 값을 돌려주는 제공자의 계약은 그대로
        유지한다. 따라서 전체 자료 재학습에서도 학습 행 표현은 자기 타깃을 직접
        포함하지 않고, 시험 행은 전체 학습 자료로 맞춘 평균표를 쓴다.

        ``apply_dataset_wide``를 먼저 호출한 frame 쌍을 받아야 한다.
        """
        X_train = self.build_matrix(train, seed)
        X_test = self.build_matrix(test, seed)
        transformers = self.fold_fit_transformers()
        if transformers:
            train_ff = prepare_fold_fit_input(train, X_train)
            test_ff = prepare_fold_fit_input(test, X_test)
            for transformer in transformers:
                enter_full_data = getattr(transformer, "enter_full_data_fit", None)
                if enter_full_data is not None:
                    # 내부 분할 대신 사전 규칙 학습 횟수를 쓰는 제공자에게 경로를 알린다. (#483)
                    enter_full_data()
                transformer.fit(train_ff, seed)
            X_train = self.add_fold_fit_columns(X_train, train_ff)
            X_test = self.add_fold_fit_columns(X_test, test_ff)
        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("전체 자료 재학습의 train/test 피처 열이 다르다.")
        if list(X_train.columns) != self.all_columns():
            raise ValueError("전체 자료 재학습의 실제 피처 열이 계획 선언과 다르다.")
        return X_train, X_test

    # ------------------------------------------------------------- 선언 조회

    def matrix_columns(self) -> list[str]:
        """build_matrix가 돌려주는 컬럼 목록(fold-fit 제외). 적용 순서 불변식 그대로."""
        assert self._base_columns is not None, "apply_dataset_wide 이전에는 base가 미확정이다."
        cols = list(self._base_columns)
        for stage in (DATASET_WIDE, ROW_WISE):
            for _, provider in self._stages[stage]:
                cols += provider.columns()
        cols.append(PLACEBO)
        return cols

    def all_columns(self) -> list[str]:
        """실험이 학습하는 컬럼 전체의 선언. MLflow feature 목록 기록의 원천. (#71)"""
        cols = self.matrix_columns()
        for _, provider in self._stages[FOLD_FIT]:
            cols += provider.columns()
        return cols

    def raw_columns(self) -> list[str]:
        """원자료에서 id, 목표값과 fold만 뺀 열을 원래 순서로 돌려준다."""
        assert self._raw_columns is not None, "apply_dataset_wide 이전에는 raw 열이 미확정이다."
        return list(self._raw_columns)

    def validate_training_row_augmentation(self) -> None:
        """복제 행을 쓸 때 현재 피처 계획이 부모 내부 분할 상속을 지킬 수 있는지 확인한다."""
        unsupported_dataset_wide = [
            kind for kind, _ in self._stages[DATASET_WIDE]
            if kind != "categorical_copies"
        ]
        if unsupported_dataset_wide:
            raise ValueError(
                "복제 학습 행은 상태를 원본에만 맞출 수 없는 dataset-wide 제공자를 "
                f"지원하지 않는다: {unsupported_dataset_wide}"
            )
        unsupported = [
            kind
            for kind, provider in self._stages[FOLD_FIT]
            if provider.uses_target
            and PARENT_ID
            not in getattr(provider, "training_row_context_columns", lambda: [])()
        ]
        if unsupported:
            raise ValueError(
                "복제 학습 행의 부모 내부 분할 상속을 지원하지 않는 타깃 참조 제공자가 있다: "
                f"{unsupported}"
            )

    def recompute_training_row_dataset_wide(
        self, training_rows: pd.DataFrame, test: pd.DataFrame
    ) -> pd.DataFrame:
        """결측 가림 뒤 범주 복제 열을 실제 원자료 값에서 다시 만든다."""
        out = training_rows.copy()
        for kind, provider in self._stages[DATASET_WIDE]:
            if kind != "categorical_copies":
                raise ValueError(
                    f"복제 학습 행에서 다시 계산할 수 없는 dataset-wide 제공자다: {kind}"
                )
            new_training, _ = provider.compute(out, test)
            if list(new_training.columns) != provider.columns():
                raise AssertionError(f"{kind}의 복제 행 산출 컬럼이 선언과 다르다.")
            for column in provider.columns():
                reference = test[column]
                if isinstance(reference.dtype, pd.CategoricalDtype):
                    source_column = column.removesuffix("_cat")
                    out[column] = pd.Categorical(
                        out[source_column], categories=reference.cat.categories
                    )
                else:
                    out[column] = new_training[column]
        return out

    def _declared_by_stage(self) -> dict[str, list[str]]:
        return {
            stage: [c for _, p in self._stages[stage] for c in p.columns()]
            for stage in STAGE_ORDER
        }

    def describe(self, raw_columns: list[str]) -> list[tuple[str, str, list[str], bool]]:
        """--plan 출력용 (단계, kind, 산출 컬럼, 타깃 참조) 표. 데이터 적재 없이 쓸 수 있게
        raw 컬럼(CSV 헤더)을 밖에서 받는다."""
        base = [
            c for c in raw_columns if c not in (ID, TARGET, "fold") and c not in self._exclude
        ]
        rows: list[tuple[str, str, list[str], bool]] = [("base", "raw", base, False)]
        if self._exclude:
            rows.append(("base", "exclude(제외)", list(self._exclude), False))
        for stage in STAGE_ORDER:
            if stage == FOLD_FIT:
                rows.append(("placebo", "(내장)", [PLACEBO], False))
            for kind, provider in self._stages[stage]:
                rows.append((stage, kind, provider.columns(), provider.uses_target))
        return rows


def _fit_evidence_of(transformer: object) -> dict[str, object] | None:
    """제공자가 적합 계보를 내면 그대로, 없으면 None을 돌려준다."""
    evidence = getattr(transformer, "fit_evidence", None)
    if evidence is None:
        return None
    return evidence()


def prepare_fold_fit_input(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """원본 frame에 행렬 생성 단계의 추가 열을 붙여 fold-fit 입력을 만든다."""
    extra = [column for column in X.columns if column not in df.columns]
    return pd.concat([df, X[extra]], axis=1) if extra else df
