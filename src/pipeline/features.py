"""피처 구성.

baseline은 원시 피처 그대로. (#16)
타깃 인코딩, other_screen 잔차 같은 후속 피처는 이 모듈에 함수를 추가하고
설정 파일의 [features] 섹션에서 켜는 식으로 확장한다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import FeatureConfig
from .data import ID, TARGET

PLACEBO = "placebo_noise"
# 플라시보에 입힐 결측 패턴의 원본 열. 결측 없는 잡음은 후보 피처(원본 열의 결측을
# 물려받음)와 다른 종류의 자가 되므로, 실제 열의 NaN 마스크를 복사한다. (#19 정정)
PLACEBO_MASK_SOURCE = "social_media_hours"

# 생성기의 화면 시간 합계 제약(daily = social + gaming + work + other)의 산술 표현. (#46)
SCREEN_TOTAL = "daily_screen_time_hours"
SCREEN_PARTS = ["social_media_hours", "gaming_hours", "work_study_hours"]


def _other_screen(df: pd.DataFrame) -> pd.Series:
    # 성분 넷 중 하나라도 결측이면 NaN. 전성분 관측 행에서만 말하는 엄격한 잔차.
    return df[SCREEN_TOTAL] - df[SCREEN_PARTS].sum(axis=1, skipna=False)


def _screen_slack(df: pd.DataFrame) -> pd.Series:
    # daily를 예산으로 보고 관측된(결측 아닌) 성분 합만 뺀 여유분. daily 결측이면 NaN.
    return df[SCREEN_TOTAL] - df[SCREEN_PARTS].sum(axis=1)


def _screen_slack_n_obs(df: pd.DataFrame) -> pd.Series:
    # slack에서 실제로 뺀 성분 개수. slack의 해석 짝이므로 slack이 정의되는 행(daily 관측)
    # 에서만 값을 준다. 일반 결측 개수 피처는 지도에서 배제 대상이라 범위를 이렇게 좁힌다.
    n = df[SCREEN_PARTS].notna().sum(axis=1).astype(float)
    return n.where(df[SCREEN_TOTAL].notna())


# 소수 첫째 자리가 실제로 변하는 격자 컬럼(정수 격자인 age 등은 제외). (#49)
DECIMAL_GRID_COLS = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "weekend_screen_time",
]


def _first_decimal(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    def f(df: pd.DataFrame) -> pd.Series:
        # 2.9*10=28.999… 같은 부동소수 오차가 자리를 깨지 않게 반올림 뒤 자리를 뗀다.
        return np.floor(np.round(df[col] * 10, 6) % 10)

    return f


# name -> 파생 컬럼 계산 함수. 타깃을 쓰지 않는 행 단위 결정적 파생만 등록한다.
DERIVED_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "other_screen": _other_screen,
    "screen_slack": _screen_slack,
    "screen_slack_n_obs": _screen_slack_n_obs,
    **{f"{c}_dec1": _first_decimal(c) for c in DECIMAL_GRID_COLS},
}


def build_features(df: pd.DataFrame, cfg: FeatureConfig, seed: int) -> pd.DataFrame:
    """모델 입력 행렬을 만든다. 반환 컬럼 목록이 곧 params로 기록되는 feature 목록."""
    if cfg.include == "raw":
        cols = [c for c in df.columns if c not in (ID, TARGET, "fold")]
    else:
        cols = list(cfg.include)
    X = df[cols].copy()
    for name in cfg.derived:
        assert name not in X.columns, f"파생 컬럼 이름 충돌: {name}"
        X[name] = DERIVED_REGISTRY[name](df)
    if cfg.placebo:
        # 개선 판정 기준선: 이 피처보다 importance가 낮으면 노이즈로 본다. (#15)
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=len(X))
        noise[df[PLACEBO_MASK_SOURCE].isna().to_numpy()] = np.nan
        X[PLACEBO] = noise
    return X


class FoldFitTransformer(Protocol):
    """fold 루프 안에서 학습하는 fold-fit 피처의 단위. (#32, #35)

    fit은 타깃이 포함된 학습 fold의 DataFrame을 받아 상태를 새로 계산하고
    (fold마다 다시 불리므로 이전 fold의 상태를 남기면 안 된다),
    transform은 DataFrame을 받아 같은 인덱스의 새 컬럼 DataFrame을 돌려준다.
    두 입력 모두 원본 컬럼에 build_features 산출 컬럼(placebo 등)이 더해진 형태다. (#33 파급)

    transform은 fold마다 train 전체(학습 fold + 검증 fold 행)와 test로 두 번 불린다.
    학습 fold 행에 OOF 값을 줘야 하는 트랜스포머(타깃 인코딩의 내부 K-fold)는
    fit 때 학습 행의 id 집합을 저장해 행별로 OOF 값과 평균표 값을 구분해 돌려준다.
    위치 인덱스는 train과 test가 겹치므로 구분 기준은 id여야 한다.
    컬럼 이름은 트랜스포머 책임이며, 기존 컬럼과 겹치면 파이프라인이 즉시 실패한다.
    """

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None: ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...


def _exact_keys(s: pd.Series, digits: int | None = None) -> pd.Series:
    """정확값 문자열 키. NaN은 별도 키로 명시 치환한다. (#33)

    digits를 주면 반올림해 해상도를 낮춘 키가 된다(반올림 표현, #49).
    """
    if digits is not None:
        s = s.round(digits)
    return s.astype(str).where(s.notna(), "__nan__")


# TE/CE 대상 지정: 단일 컬럼 이름 또는 컬럼 이름 목록(결합 키, #48/#51).
ColumnSpec = str | list[str]


def _spec_name(spec: ColumnSpec) -> str:
    """spec의 산출 컬럼 이름 어간. 결합 키는 구성 컬럼을 __로 잇는다."""
    return spec if isinstance(spec, str) else "__".join(spec)


def _spec_keys(df: pd.DataFrame, spec: ColumnSpec, digits: int | None = None) -> pd.Series:
    """spec의 정확값 키. 결합 키는 컬럼별 정확값 키를 |로 잇는다(#48 규약)."""
    if isinstance(spec, str):
        return _exact_keys(df[spec], digits)
    keys = _exact_keys(df[spec[0]], digits)
    for col in spec[1:]:
        keys = keys + "|" + _exact_keys(df[col], digits)
    return keys


def add_pair_ce(train: pd.DataFrame, test: pd.DataFrame, pairs: list[list[str]]) -> None:
    """결합 키의 훈련+테스트 합산 빈도 log1p를 <a>__<b>_ce 컬럼으로 두 데이터에 추가한다. (#48, #51)

    타깃을 참조하지 않으므로 fold-fit이 필요 없고, 합산 표라 미지 조합도 생기지 않는다.
    """
    for pair in pairs:
        name = f"{_spec_name(pair)}_ce"
        assert name not in train.columns, f"pair CE 컬럼 이름 충돌: {name}"
        train_keys = _spec_keys(train, pair)
        test_keys = _spec_keys(test, pair)
        counts = pd.concat([train_keys, test_keys], ignore_index=True).value_counts()
        train[name] = np.log1p(train_keys.map(counts)).astype("float64")
        test[name] = np.log1p(test_keys.map(counts)).astype("float64")


class ExactValueTargetEncoder:
    """정확값 키 타깃 인코딩. (#33 설계, #34)

    학습 fold 전체로 키별 타깃 평균표를 만들고,
    학습 fold 행에는 내부 층화 K-fold OOF 값을, 검증 fold와 test 행에는 전체 평균표 값을 준다.
    평활 없음, 미지 키는 fit 데이터의 전체 타깃 평균. 새 컬럼 이름은 <col><suffix>.
    key_digits로 키 해상도를 낮춘 반올림 표현을 만들 수 있다(suffix로 정확값 TE와 구분). (#49)
    cols 항목이 컬럼 목록이면 결합 키 TE가 되고, 이름은 컬럼들을 __로 이은 어간을 쓴다. (#48, #51)
    """

    def __init__(
        self,
        cols: list[ColumnSpec],
        inner_folds: int = 10,
        key_digits: int | None = None,
        suffix: str = "_te",
    ) -> None:
        self.cols = list(cols)
        self.inner_folds = inner_folds
        self.key_digits = key_digits
        self.suffix = suffix

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None:
        assert train_fold[ID].is_unique, "학습 fold의 id가 유일하지 않다."
        y = train_fold[TARGET]
        self.global_mean_ = float(y.mean())
        self.tables_: dict[str, pd.Series] = {}
        skf = StratifiedKFold(n_splits=self.inner_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(train_fold, y))
        oof: dict[str, np.ndarray] = {}
        for spec in self.cols:
            name = _spec_name(spec)
            keys = _spec_keys(train_fold, spec, self.key_digits)
            self.tables_[name] = y.groupby(keys).mean()
            vals = np.empty(len(train_fold))
            for tr_i, va_i in splits:
                inner_y = y.iloc[tr_i]
                inner_table = inner_y.groupby(keys.iloc[tr_i]).mean()
                mapped = keys.iloc[va_i].map(inner_table).fillna(inner_y.mean())
                vals[va_i] = mapped.to_numpy()
            oof[name] = vals
        self.oof_ = pd.DataFrame(oof, index=pd.Index(train_fold[ID], name=ID))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 학습 fold 행(fit 때 저장한 id)은 OOF 값, 그 외(검증 fold, test)는 전체 평균표 값.
        is_fit_row = df[ID].isin(self.oof_.index).to_numpy()
        out: dict[str, pd.Series] = {}
        for spec in self.cols:
            name = _spec_name(spec)
            mapped = _spec_keys(df, spec, self.key_digits).map(self.tables_[name])
            mapped = mapped.fillna(self.global_mean_)
            if is_fit_row.any():
                mapped = mapped.copy()
                mapped.iloc[is_fit_row] = self.oof_[name].loc[df.loc[is_fit_row, ID]].to_numpy()
            out[f"{name}{self.suffix}"] = mapped
        return pd.DataFrame(out, index=df.index)


class FrequencyEncoder:
    """정확값 키 빈도 인코딩(CE). (#49)

    타깃을 쓰지 않지만 데이터에서 표를 학습하므로 fold-fit으로 학습 fold에서만 센다.
    타깃 미사용이라 OOF 구분이 필요 없고 모든 행에 같은 표를 적용한다.
    미지 키는 0(한 번도 못 본 값). 새 컬럼 이름은 <col>_ce.
    """

    def __init__(self, cols: list[str]) -> None:
        self.cols = list(cols)

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None:
        self.tables_ = {
            col: _exact_keys(train_fold[col]).value_counts().astype(float) for col in self.cols
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {
            f"{col}_ce": _exact_keys(df[col]).map(self.tables_[col]).fillna(0.0)
            for col in self.cols
        }
        return pd.DataFrame(out, index=df.index)


class MedianImputeAux:
    """원시 NaN 열을 유지한 채 중앙값 대체본을 보조 열로 추가한다. (#49)

    원래 열을 덮는 대체는 지도의 배제 대상이고, 근거가 강한 것은 두 열 병행이다
    (docs/research/code-notebook-insights.md 전처리 절). 중앙값은 학습 fold에서만 계산한다.
    새 컬럼 이름은 <col>_fill.
    """

    def __init__(self, cols: list[str]) -> None:
        self.cols = list(cols)

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None:
        self.medians_ = {col: float(train_fold[col].median()) for col in self.cols}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {f"{col}_fill": df[col].fillna(self.medians_[col]) for col in self.cols}
        return pd.DataFrame(out, index=df.index)


# kind -> 트랜스포머 팩토리. 새 fold-fit 피처는 여기 등록만 하면 설정에서 켤 수 있다.
FOLD_FIT_REGISTRY: dict[str, Callable[..., FoldFitTransformer]] = {
    "target_encoding": ExactValueTargetEncoder,
    "frequency_encoding": FrequencyEncoder,
    "median_impute_aux": MedianImputeAux,
}


def make_fold_fit(cfg: FeatureConfig) -> list[FoldFitTransformer]:
    """설정의 fold_fit 목록을 트랜스포머 인스턴스로 만든다. kind 외 키는 생성자 인자."""
    transformers: list[FoldFitTransformer] = []
    for spec in cfg.fold_fit:
        params = dict(spec)
        kind = params.pop("kind")
        transformers.append(FOLD_FIT_REGISTRY[kind](**params))
    return transformers


def add_fold_fit_columns(
    transformers: list[FoldFitTransformer], X: pd.DataFrame, df: pd.DataFrame
) -> pd.DataFrame:
    """fit된 트랜스포머들의 새 컬럼을 X에 붙인 행렬을 돌려준다. 추가 전용."""
    out = X
    for t in transformers:
        new = t.transform(df)
        assert new.index.equals(df.index), f"{type(t).__name__}의 transform 인덱스가 원본과 다르다."
        collision = set(new.columns) & set(out.columns)
        assert not collision, f"fold-fit 컬럼 이름 충돌: {sorted(collision)}"
        out = pd.concat([out, new], axis=1)
    return out
