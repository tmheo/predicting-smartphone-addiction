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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from . import features
from .config import FeatureConfig
from .data import ID, TARGET
from .features import PLACEBO

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

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None: ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class ProviderKind:
    stage: str
    factory: Callable[..., Any]


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
}


def _base_columns_of(df: pd.DataFrame) -> list[str]:
    """base: raw의 결정적 정의. 원시 CSV 컬럼 전부에서 ID와 타깃만 뺀다. (#71)"""
    return [c for c in df.columns if c not in (ID, TARGET, "fold")]


class FeaturePlan:
    """실험이 학습할 컬럼 전체의 선언. run.py가 설정에서 한 번 만들어 cv에 주입한다."""

    def __init__(self, stages: dict[str, list[tuple[str, Any]]], exclude: list[str]) -> None:
        self._stages = stages  # stage -> [(kind, provider), ...] providers 목록 순서 유지
        self._exclude = exclude  # base에서 뺄 raw 컬럼. 제공자 입력에는 남는다. (#79)
        self._base_columns: list[str] | None = None

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
        plan = cls(stages, list(cfg.exclude))
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


def prepare_fold_fit_input(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """원본 frame에 행렬 생성 단계의 추가 열을 붙여 fold-fit 입력을 만든다."""
    extra = [column for column in X.columns if column not in df.columns]
    return pd.concat([df, X[extra]], axis=1) if extra else df
