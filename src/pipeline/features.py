"""컬럼 제공자 구현. 어떤 컬럼을 어떻게 만드는가만 담는다.

어떤 제공자를 어떤 단계에 어떤 순서로 실행하는가는 pipeline.plan의 피처 계획이 소유한다. (#71)
새 컬럼 제공자는 여기 구현을 추가하고 plan.REGISTRY에 등록해 설정의 providers에서 켠다.
모든 제공자는 산출 컬럼 이름(columns)과 타깃 참조 여부(uses_target)를 정적으로 선언한다.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .data import ID, TARGET, file_sha256, union_categorical

PLACEBO = "placebo_noise"
# 플라시보에 입힐 결측 패턴의 원본 열. 결측 없는 잡음은 후보 피처(원본 열의 결측을
# 물려받음)와 다른 종류의 자가 되므로, 실제 열의 NaN 마스크를 복사한다. (#19 정정)
PLACEBO_MASK_SOURCE = "social_media_hours"

# 생성기의 화면 시간 합계 제약(daily = social + gaming + work + other)의 산술 표현. (#46)
SCREEN_TOTAL = "daily_screen_time_hours"
SCREEN_PARTS = ["social_media_hours", "gaming_hours", "work_study_hours"]


def placebo_series(df: pd.DataFrame, seed: int) -> pd.Series:
    """개선 판정 기준선 잡음. 피처 계획이 매 행렬에 자동 삽입한다. (#15, #71)"""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=len(df))
    noise[df[PLACEBO_MASK_SOURCE].isna().to_numpy()] = np.nan
    return pd.Series(noise, index=df.index)


def _other_screen(df: pd.DataFrame) -> pd.Series:
    # 성분 넷 중 하나라도 결측이면 NaN. 전성분 관측 행에서만 말하는 엄격한 잔차.
    return df[SCREEN_TOTAL] - df[SCREEN_PARTS].sum(axis=1, skipna=False)


def _screen_slack(df: pd.DataFrame) -> pd.Series:
    # daily를 예산으로 보고 관측된(결측 아닌) 성분 합만 뺀 여유분. daily 결측이면 NaN.
    return df[SCREEN_TOTAL] - df[SCREEN_PARTS].sum(axis=1)


def _sgw_sum(df: pd.DataFrame) -> pd.Series:
    # 관측된 성분(social+gaming+work)의 합. 전성분 결측 행은 0이다(#58 원문 규약 유지).
    return df[SCREEN_PARTS].sum(axis=1)


def _sgw_frac(df: pd.DataFrame) -> pd.Series:
    return _sgw_sum(df) / df[SCREEN_TOTAL].clip(lower=0.1)


def _slack_frac(df: pd.DataFrame) -> pd.Series:
    return _screen_slack(df) / df[SCREEN_TOTAL].clip(lower=0.1)


def _wk_minus_sgw(df: pd.DataFrame) -> pd.Series:
    return df["weekend_screen_time"] - _sgw_sum(df)


def _wk_other(df: pd.DataFrame) -> pd.Series:
    return df["weekend_screen_time"] - _screen_slack(df)


def _screen_slack_n_obs(df: pd.DataFrame) -> pd.Series:
    # slack에서 실제로 뺀 성분 개수. slack의 해석 짝이므로 slack이 정의되는 행(daily 관측)
    # 에서만 값을 준다. 일반 결측 개수 피처는 지도에서 배제 대상이라 범위를 이렇게 좁힌다.
    n = df[SCREEN_PARTS].notna().sum(axis=1).astype(float)
    return n.where(df[SCREEN_TOTAL].notna())


# 화면 관계 7특성 블록(#181). 공개 스택 계보 조사(public-stack-provenance 3순위)의
# beicicc 계약을 그대로 따른다: 차이 3개는 NaN을 자연 전파하고, 비율 4개는 분자·분모가
# 유한하고 분모가 양수일 때만 정의한다. epsilon, 대체, 클리핑, 정의 여부 플래그는 쓰지 않는다.
SCREEN_RELATION_DIFFS: dict[str, tuple[str, str]] = {
    "gaming_minus_work": ("gaming_hours", "work_study_hours"),
    "screen_minus_work": (SCREEN_TOTAL, "work_study_hours"),
    "weekend_minus_daily": ("weekend_screen_time", SCREEN_TOTAL),
}
SCREEN_RELATION_RATIOS: dict[str, tuple[str, str]] = {
    "social_share_screen": ("social_media_hours", SCREEN_TOTAL),
    "gaming_share_screen": ("gaming_hours", SCREEN_TOTAL),
    "work_share_screen": ("work_study_hours", SCREEN_TOTAL),
    "screen_to_sleep": (SCREEN_TOTAL, "sleep_hours"),
}


def _difference(left: str, right: str) -> Callable[[pd.DataFrame], pd.Series]:
    def f(df: pd.DataFrame) -> pd.Series:
        return df[left] - df[right]

    return f


def _guarded_ratio(numerator: str, denominator: str) -> Callable[[pd.DataFrame], pd.Series]:
    def f(df: pd.DataFrame) -> pd.Series:
        num = df[numerator]
        den = df[denominator]
        defined = np.isfinite(num) & np.isfinite(den) & (den > 0)
        return (num / den).where(defined)

    return f


# zhenrui 파생 수치 블록(#258). 공개 노트북(kernel 129907249, Apache-2.0, SHA-256
# ef16bb88782581ad4e880f295903db88278428464c1aa31643a0511d5197b116)의 상호작용·비율·
# 로그 16열 이식. 원본은 중앙값 대체 후 계산하고 비율 분모에 +1e-6을 더하지만, 이식은
# 저장소 규율을 따라 결측을 자연 전파하고 비율은 마스크 방식(분자·분모 유한, 분모 양수)
# 만 쓴다. 스트레스 상호작용의 원본 missing→1 대체도 결측 자연 전파로 바꿨다.
# 16열 중 4열(weekend_minus_daily, social_share_screen, gaming_share_screen,
# screen_to_sleep)은 정의가 같은 기존 등록을 그대로 쓴다.
STRESS_LEVEL_ORDER = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
ENGAGEMENT_PARTS = ["notifications_per_day", "app_opens_per_day"]


def _stress_numeric(df: pd.DataFrame) -> pd.Series:
    # stress_level은 파이프라인에서 범주형 dtype으로 들어오므로, map 결과가 범주형으로
    # 남아 산술이 막히지 않게 명시적으로 float64 수치 서열로 바꾼다.
    return df["stress_level"].astype("object").map(STRESS_LEVEL_ORDER).astype("float64")


def _total_screen(df: pd.DataFrame) -> pd.Series:
    return df[SCREEN_TOTAL] + df["weekend_screen_time"]


def _activity_total(df: pd.DataFrame) -> pd.Series:
    return df[SCREEN_PARTS].sum(axis=1, skipna=False)


def _engagement_total(df: pd.DataFrame) -> pd.Series:
    return df[ENGAGEMENT_PARTS].sum(axis=1, skipna=False)


def _masked_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    defined = np.isfinite(num) & np.isfinite(den) & (den > 0)
    return (num / den).where(defined)


def _activity_share_screen(df: pd.DataFrame) -> pd.Series:
    return _masked_ratio(_activity_total(df), df[SCREEN_TOTAL])


def _notif_per_app(df: pd.DataFrame) -> pd.Series:
    return _masked_ratio(df["notifications_per_day"], df["app_opens_per_day"])


def _screen_per_app(df: pd.DataFrame) -> pd.Series:
    # 원본 규약대로 시간을 분으로 환산해 app open 1회당 화면 시간을 만든다.
    return _masked_ratio(df[SCREEN_TOTAL] * 60.0, df["app_opens_per_day"])


def _sleep_deficit(df: pd.DataFrame) -> pd.Series:
    # 원본 규약의 기준 수면 9시간 대비 부족분. 음수는 0으로 자르고 결측은 유지한다.
    return (9.0 - df["sleep_hours"]).clip(lower=0.0)


def _log1p_of(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    def f(df: pd.DataFrame) -> pd.Series:
        return np.log1p(df[col])

    return f


def _times_stress(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    def f(df: pd.DataFrame) -> pd.Series:
        return df[col] * _stress_numeric(df)

    return f


ZHENRUI_DERIVED_NAMES = [
    "total_screen",
    "weekend_minus_daily",
    "social_share_screen",
    "gaming_share_screen",
    "activity_total",
    "activity_share_screen",
    "screen_to_sleep",
    "sleep_deficit",
    "engagement_total",
    "notif_per_app",
    "log_notifications",
    "log_app_opens",
    "screen_per_app",
    "screen_x_stress",
    "social_x_stress",
    "sleep_x_stress",
]


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


# 정체성·자리수 특성 블록(#184). 공개 스택 계보 조사(public-stack-provenance 5순위)의
# beicicc identity_digit_contract를 따른다: 수치 9열별로 반올림 3해상도(round0/1/2)와
# 그 절대 편차, 정수·소수1자리 여부 지시자, 소수 첫째·둘째 자리 값 10개를 만든다.
# 모두 타깃을 쓰지 않고 결측은 자연 전파한다(지시자·자리 값도 결측 행은 NaN).
DIGIT_IDENTITY_COLS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]
DIGIT_IDENTITY_SUFFIXES = [
    "round0",
    "absdiff_round0",
    "round1",
    "absdiff_round1",
    "round2",
    "absdiff_round2",
    "is_round0",
    "is_round1",
    "tenths",
    "hundredths",
]


def _rounded(col: str, decimals: int) -> Callable[[pd.DataFrame], pd.Series]:
    def f(df: pd.DataFrame) -> pd.Series:
        return np.round(df[col], decimals)

    return f


def _absdiff_rounded(col: str, decimals: int) -> Callable[[pd.DataFrame], pd.Series]:
    def f(df: pd.DataFrame) -> pd.Series:
        return (df[col] - np.round(df[col], decimals)).abs()

    return f


def _is_rounded(col: str, decimals: int) -> Callable[[pd.DataFrame], pd.Series]:
    def f(df: pd.DataFrame) -> pd.Series:
        # 2.9*10=28.999… 같은 부동소수 오차가 자리 판정을 깨지 않게 소수 6자리에서 끊는다.
        scaled = np.round(df[col] * 10**decimals, 6)
        return (scaled % 1 == 0).astype(float).where(df[col].notna())

    return f


def _decimal_digit(col: str, place: int) -> Callable[[pd.DataFrame], pd.Series]:
    def f(df: pd.DataFrame) -> pd.Series:
        return np.floor(np.round(df[col] * 10**place, 6) % 10)

    return f


def _digit_identity_registry() -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    registry: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}
    for col in DIGIT_IDENTITY_COLS:
        for decimals in (0, 1, 2):
            registry[f"{col}_round{decimals}"] = _rounded(col, decimals)
            registry[f"{col}_absdiff_round{decimals}"] = _absdiff_rounded(col, decimals)
        registry[f"{col}_is_round0"] = _is_rounded(col, 0)
        registry[f"{col}_is_round1"] = _is_rounded(col, 1)
        registry[f"{col}_tenths"] = _decimal_digit(col, 1)
        registry[f"{col}_hundredths"] = _decimal_digit(col, 2)
    return registry


DIGIT_IDENTITY_NAMES = [
    f"{col}_{suffix}" for col in DIGIT_IDENTITY_COLS for suffix in DIGIT_IDENTITY_SUFFIXES
]


# name -> 파생 컬럼 계산 함수. 타깃을 쓰지 않는 행 단위 결정적 파생만 등록한다.
DERIVED_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "other_screen": _other_screen,
    "screen_slack": _screen_slack,
    "screen_slack_n_obs": _screen_slack_n_obs,
    # #58 Lookup-Transformer 원문의 예산 토큰 재현: sgw 합·비율과 주말 대비 잔차.
    "sgw_sum": _sgw_sum,
    "sgw_frac": _sgw_frac,
    "slack_frac": _slack_frac,
    "wk_minus_sgw": _wk_minus_sgw,
    "wk_other": _wk_other,
    **{f"{c}_dec1": _first_decimal(c) for c in DECIMAL_GRID_COLS},
    # #181 화면 관계 7특성: 차이 3개 + 안전 비율 4개.
    **{name: _difference(a, b) for name, (a, b) in SCREEN_RELATION_DIFFS.items()},
    **{name: _guarded_ratio(n, d) for name, (n, d) in SCREEN_RELATION_RATIOS.items()},
    # #184 정체성·자리수 블록: 수치 9열 × 자리수 파생 10개.
    **_digit_identity_registry(),
    # #258 zhenrui 블록의 신규 12열(나머지 4열은 위 기존 등록을 그대로 쓴다).
    "total_screen": _total_screen,
    "activity_total": _activity_total,
    "activity_share_screen": _activity_share_screen,
    "sleep_deficit": _sleep_deficit,
    "engagement_total": _engagement_total,
    "notif_per_app": _notif_per_app,
    "log_notifications": _log1p_of("notifications_per_day"),
    "log_app_opens": _log1p_of("app_opens_per_day"),
    "screen_per_app": _screen_per_app,
    "screen_x_stress": _times_stress(SCREEN_TOTAL),
    "social_x_stress": _times_stress("social_media_hours"),
    "sleep_x_stress": _times_stress("sleep_hours"),
}


class DerivedColumns:
    """row-wise 제공자: DERIVED_REGISTRY의 파생 컬럼을 names 순서로 만든다. (#46, #71)"""

    uses_target = False

    def __init__(self, names: list[str]) -> None:
        unknown = [n for n in names if n not in DERIVED_REGISTRY]
        if unknown:
            raise ValueError(
                f"등록되지 않은 파생 이름 {unknown}. 등록: {', '.join(sorted(DERIVED_REGISTRY))}"
            )
        self.names = list(names)

    def columns(self) -> list[str]:
        return list(self.names)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: DERIVED_REGISTRY[name](df) for name in self.names})


class MissingIndicators:
    """row-wise 제공자: 지정한 열의 결측 여부를 0/1 수치 열로 만든다. (#145)

    Trompt는 원시 NaN을 직접 처리하지 않으므로 학습 fold 중앙값 대치와 짝을 이루는
    입력 계약으로만 사용한다.
    일반적인 결측 표시 특성 탐색을 여는 제공자가 아니다.
    """

    uses_target = False

    def __init__(self, cols: list[str]) -> None:
        if not cols:
            raise ValueError("cols는 하나 이상의 열을 가져야 한다.")
        if len(set(cols)) != len(cols):
            raise ValueError("cols에 중복 열이 있다.")
        self.cols = list(cols)

    def columns(self) -> list[str]:
        return [f"{col}_missing" for col in self.cols]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.cols if col not in df.columns]
        if missing:
            raise ValueError(f"결측 표시 입력 열이 없다: {missing}")
        return pd.DataFrame(
            {f"{col}_missing": df[col].isna().astype("float32") for col in self.cols},
            index=df.index,
        )


class CategoricalCopies:
    """dataset-wide 제공자: 수치 컬럼은 그대로 두고 <col>_cat 범주형 복제 컬럼을 만든다.

    train/test 값 합집합으로 카테고리를 고정해야 해서(코드 배정 어긋남 방지) dataset-wide다.
    (#31 변형 b)
    """

    uses_target = False

    def __init__(self, cols: list[str]) -> None:
        self.cols = list(cols)

    def columns(self) -> list[str]:
        return [f"{col}_cat" for col in self.cols]

    def compute(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        new_train: dict[str, pd.Categorical] = {}
        new_test: dict[str, pd.Categorical] = {}
        for col in self.cols:
            new_train[f"{col}_cat"], new_test[f"{col}_cat"] = union_categorical(
                train[col], test[col]
            )
        return (
            pd.DataFrame(new_train, index=train.index),
            pd.DataFrame(new_test, index=test.index),
        )


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


class PairCE:
    """dataset-wide 제공자: 결합 키의 훈련+테스트 합산 빈도 log1p를 <a>__<b>_ce로 만든다.

    타깃을 참조하지 않으므로 fold-fit이 필요 없고, 합산 표라 미지 조합도 생기지 않는다.
    (#48, #51)
    """

    uses_target = False

    def __init__(self, pairs: list[list[str]]) -> None:
        self.pairs = [list(pair) for pair in pairs]

    def columns(self) -> list[str]:
        return [f"{_spec_name(pair)}_ce" for pair in self.pairs]

    def compute(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        new_train: dict[str, pd.Series] = {}
        new_test: dict[str, pd.Series] = {}
        for pair in self.pairs:
            name = f"{_spec_name(pair)}_ce"
            train_keys = _spec_keys(train, pair)
            test_keys = _spec_keys(test, pair)
            counts = pd.concat([train_keys, test_keys], ignore_index=True).value_counts()
            new_train[name] = np.log1p(train_keys.map(counts)).astype("float64")
            new_test[name] = np.log1p(test_keys.map(counts)).astype("float64")
        return (
            pd.DataFrame(new_train, index=train.index),
            pd.DataFrame(new_test, index=test.index),
        )


class ExactValueTargetEncoder:
    """fold-fit 제공자: 정확값 키 타깃 인코딩. (#33 설계, #34)

    학습 fold 전체로 키별 타깃 평균표를 만들고,
    학습 fold 행에는 내부 층화 K-fold OOF 값을, 검증 fold와 test 행에는 전체 평균표 값을 준다.
    평활 없음, 미지 키는 fit 데이터의 전체 타깃 평균. 새 컬럼 이름은 <col><suffix>.
    key_digits로 키 해상도를 낮춘 반올림 표현을 만들 수 있다(suffix로 정확값 TE와 구분). (#49)
    cols 항목이 컬럼 목록이면 결합 키 TE가 되고, 이름은 컬럼들을 __로 이은 어간을 쓴다. (#48, #51)
    """

    uses_target = True

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

    def columns(self) -> list[str]:
        return [f"{_spec_name(spec)}{self.suffix}" for spec in self.cols]

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


# 격자 쌍 TE의 셀 해상도 -> 컬럼 이름 접미어 어간. (#75)
LATTICE_RESOLUTIONS = {"floor": "latf", "r1": "latr1"}


class LatticePairTargetEncoder:
    """fold-fit 제공자: 숫자 쌍 격자 셀 TE와 셀 개수 열의 일괄 블록. (#75)

    cols의 모든 2-조합을 개별 선별 없이 통째로 인코딩한다. 셀 키는 두 컬럼을 해상도로
    거칠게 만든 값의 결합이다: floor는 정수 내림, r1은 소수 첫째 자리 반올림. 셀별로
    평활 타깃 평균(<a>__<b>_<res>_te)과 학습 표본 개수(<a>__<b>_<res>_ct)를 함께 내
    모델이 얇은 셀의 평균을 스스로 불신하게 한다(개수 열이 블록 기제의 일부다).

    학습 fold 행은 내부 층화 K-fold OOF 값(자기 행이 자기 인코딩에 안 들어감),
    검증 fold와 test 행은 학습 fold 전체 표의 값을 받는다. 평활은 m-평활
    (n·p + m·g)/(n + m), 미지 셀은 TE가 전체 평균 g, 개수가 0이다.
    placebo 쌍 카나리아([placebo_noise, cols[0]])는 자동 포함한다(#48 규약).

    출처 구성(szymonkapiski full lattice, docs/research/code-notebook-insights-2.md)을
    따르되 키 표기는 이 저장소의 정확값 키 규약(__nan__ 결측 버킷, | 결합)을 쓴다.
    """

    uses_target = True

    def __init__(
        self,
        cols: list[str],
        resolutions: list[str] | None = None,
        inner_folds: int = 4,
        smoothing: float = 20.0,
    ) -> None:
        resolutions = ["floor"] if resolutions is None else resolutions
        if len(cols) < 2:
            raise ValueError(f"격자 쌍에는 컬럼이 2개 이상 필요하다(받은 값: {cols})")
        if len(set(cols)) != len(cols):
            raise ValueError(f"cols에 중복이 있다: {cols}")
        if PLACEBO in cols:
            raise ValueError(f"{PLACEBO}는 쌍 카나리아로 자동 포함되므로 cols에 넣지 않는다.")
        unknown = [r for r in resolutions if r not in LATTICE_RESOLUTIONS]
        if unknown:
            raise ValueError(
                f"알 수 없는 해상도 {unknown}. 지원: {', '.join(LATTICE_RESOLUTIONS)}"
            )
        if not resolutions or len(set(resolutions)) != len(resolutions):
            raise ValueError(f"resolutions는 중복 없는 비어 있지 않은 목록이어야 한다: {resolutions}")
        if inner_folds < 2:
            raise ValueError(f"inner_folds는 2 이상이어야 한다(받은 값: {inner_folds})")
        if smoothing <= 0:
            raise ValueError(f"smoothing은 0보다 커야 한다(받은 값: {smoothing})")
        self.cols = list(cols)
        self.resolutions = list(resolutions)
        self.inner_folds = inner_folds
        self.smoothing = float(smoothing)
        # 전 쌍 블록: cols의 모든 2-조합 + placebo 쌍 카나리아. 카나리아는 placebo가
        # 앞이어야 compare의 카나리아 인식(placebo_noise_ 접두어)에 걸린다.
        self.pairs: list[tuple[str, str]] = [
            (a, b) for i, a in enumerate(self.cols) for b in self.cols[i + 1 :]
        ]
        self.pairs.append((PLACEBO, self.cols[0]))

    def _stems(self) -> list[tuple[str, tuple[str, str], str]]:
        return [
            (f"{a}__{b}_{LATTICE_RESOLUTIONS[res]}", (a, b), res)
            for res in self.resolutions
            for a, b in self.pairs
        ]

    def columns(self) -> list[str]:
        return [f"{stem}_{kind}" for stem, _, _ in self._stems() for kind in ("te", "ct")]

    def _pair_keys(
        self,
        df: pd.DataFrame,
        pair: tuple[str, str],
        res: str,
        cache: dict[tuple[str, str], pd.Series],
    ) -> pd.Series:
        def col_keys(col: str) -> pd.Series:
            if (col, res) not in cache:
                cache[(col, res)] = (
                    _exact_keys(np.floor(df[col])) if res == "floor" else _exact_keys(df[col], 1)
                )
            return cache[(col, res)]

        return col_keys(pair[0]) + "|" + col_keys(pair[1])

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None:
        assert train_fold[ID].is_unique, "학습 fold의 id가 유일하지 않다."
        y = train_fold[TARGET]
        gm = self.global_mean_ = float(y.mean())
        m = self.smoothing
        skf = StratifiedKFold(n_splits=self.inner_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(train_fold, y))
        cache: dict[tuple[str, str], pd.Series] = {}
        self.te_tables_: dict[str, pd.Series] = {}
        self.ct_tables_: dict[str, pd.Series] = {}
        oof: dict[str, np.ndarray] = {}
        for stem, pair, res in self._stems():
            keys = self._pair_keys(train_fold, pair, res, cache)
            grp = y.groupby(keys).agg(["sum", "count"])
            self.te_tables_[stem] = (grp["sum"] + m * gm) / (grp["count"] + m)
            self.ct_tables_[stem] = grp["count"].astype("float64")
            te = np.empty(len(train_fold), dtype="float64")
            ct = np.empty(len(train_fold), dtype="float64")
            for tr_i, va_i in splits:
                g = y.iloc[tr_i].groupby(keys.iloc[tr_i]).agg(["sum", "count"])
                mp = (g["sum"] + m * gm) / (g["count"] + m)
                va_keys = keys.iloc[va_i]
                te[va_i] = va_keys.map(mp).fillna(gm).to_numpy()
                ct[va_i] = va_keys.map(g["count"]).fillna(0.0).to_numpy()
            oof[f"{stem}_te"] = te
            oof[f"{stem}_ct"] = ct
        self.oof_ = pd.DataFrame(oof, index=pd.Index(train_fold[ID], name=ID))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 학습 fold 행(fit 때 저장한 id)은 OOF 값, 그 외(검증 fold, test)는 전체 표 값.
        is_fit_row = df[ID].isin(self.oof_.index).to_numpy()
        fit_ids = df.loc[is_fit_row, ID]
        cache: dict[tuple[str, str], pd.Series] = {}
        out: dict[str, pd.Series] = {}
        for stem, pair, res in self._stems():
            keys = self._pair_keys(df, pair, res, cache)
            te = keys.map(self.te_tables_[stem]).fillna(self.global_mean_)
            ct = keys.map(self.ct_tables_[stem]).fillna(0.0)
            if is_fit_row.any():
                te, ct = te.copy(), ct.copy()
                te.iloc[is_fit_row] = self.oof_[f"{stem}_te"].loc[fit_ids].to_numpy()
                ct.iloc[is_fit_row] = self.oof_[f"{stem}_ct"].loc[fit_ids].to_numpy()
            out[f"{stem}_te"] = te.astype("float64")
            out[f"{stem}_ct"] = ct.astype("float64")
        return pd.DataFrame(out, index=df.index)


class FrequencyEncoder:
    """fold-fit 제공자: 정확값 키 빈도 인코딩(CE). (#49)

    타깃을 쓰지 않지만 데이터에서 표를 학습하므로 fold-fit으로 학습 fold에서만 센다.
    타깃 미사용이라 OOF 구분이 필요 없고 모든 행에 같은 표를 적용한다.
    미지 키는 0(한 번도 못 본 값). 새 컬럼 이름은 <col>_ce.
    """

    uses_target = False

    def __init__(self, cols: list[str]) -> None:
        self.cols = list(cols)

    def columns(self) -> list[str]:
        return [f"{col}_ce" for col in self.cols]

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


# 원본 프록시(#47에서 고정한 jayjoshi37 판본 1)의 CSV 해시. 이 해시와 다른 파일로는
# prior를 만들지 않는다(docs/research/original-proxy-data.md의 재현 경계). (#53)
ORIGINAL_PROXY_SHA256 = "2194ce1946e8559f26780049c8d972857d8378104f2c9ec25ed9ec35409f1074"
# 대회 설명변수로 쓰면 안 되는 프록시 전용 열(#47의 사용 경계).
_PROXY_FORBIDDEN_COLS = {"transaction_id", "user_id", "addiction_level", "addicted_label"}
_PROXY_LABEL = "addicted_label"


def _load_locked_proxy(path: str, sha256: str, used_cols: list[str]) -> pd.DataFrame:
    """해시 고정 원본 프록시를 읽고 사용 경계(#47)를 검증한다. 원본 계열 제공자 공용."""
    from pathlib import Path

    actual = file_sha256(Path(path))
    if actual != sha256:
        raise ValueError(
            f"원본 프록시 해시 불일치: {path}\n기대 {sha256}\n실제 {actual}\n"
            "docs/research/original-proxy-data.md의 재현 절차로 판본 1을 다시 받을 것."
        )
    proxy = pd.read_csv(path)
    forbidden = sorted(set(used_cols) & _PROXY_FORBIDDEN_COLS)
    if forbidden:
        raise ValueError(f"프록시 전용 열 {forbidden}은 대회 설명변수로 쓸 수 없다. (#47 경계)")
    missing = sorted(c for c in used_cols if c not in proxy.columns)
    if missing:
        raise ValueError(f"프록시에 없는 열 {missing}. 프록시 열: {sorted(proxy.columns)}")
    return proxy


def _aligned_keys(s: pd.Series) -> pd.Series:
    """데이터셋 경계를 넘는 정확값 키. 수치는 float64로 정규화한다.

    같은 frame 안에서만 쓰는 _exact_keys와 달리, 프록시와 대회 데이터처럼 서로 다른
    CSV에서 읽힌 열을 같은 키 공간으로 보내야 한다. 프록시의 age는 int64, 대회 train의
    age는 float64로 읽히므로 정규화 없이는 "23"과 "23.0"으로 어긋난다. (#53)
    """
    if pd.api.types.is_numeric_dtype(s):
        s = s.astype("float64")
    return s.astype(str).where(s.notna(), "__nan__")


def _aligned_spec_keys(df: pd.DataFrame, spec: ColumnSpec) -> pd.Series:
    if isinstance(spec, str):
        return _aligned_keys(df[spec])
    keys = _aligned_keys(df[spec[0]])
    for col in spec[1:]:
        keys = keys + "|" + _aligned_keys(df[col])
    return keys


class OriginalPriorColumns:
    """row-wise 제공자: 원본 프록시의 값별 라벨 통계를 새 열로 매핑한다. (#53)

    통계표는 해시가 고정된 프록시 파일만으로 만들고 대회 train 라벨은 읽지 않는다
    (uses_target=False). 표가 학습 전에 외부 파일로 고정되므로 매핑은 행 단위
    결정적이고 fold 분리가 필요 없다.

    값별 통계는 m-평활 라벨 평균 p = (n·p_raw + m·g) / (n + m) 기반이다(g는 프록시
    전체 평균). stats: mean(p), woe(logit(p) - logit(g)), entropy(p의 이진 엔트로피),
    count(log1p(n)). 프록시에 없는 키(대회 전용 값, 결측)는 unknown이 "nan"이면 NaN,
    "global"이면 n=0 대입값(g, 0, H(g), 0)을 준다. 새 컬럼 이름은 <spec>_orig_<stat>.
    """

    uses_target = False
    STATS = ("mean", "woe", "entropy", "count")

    def __init__(
        self,
        path: str,
        cols: list[ColumnSpec],
        stats: list[str] | None = None,
        smoothing: float = 20.0,
        unknown: str = "nan",
        sha256: str = ORIGINAL_PROXY_SHA256,
    ) -> None:
        stats = ["mean"] if stats is None else stats
        unknown_stats = [s for s in stats if s not in self.STATS]
        if unknown_stats:
            raise ValueError(f"알 수 없는 stat {unknown_stats}. 지원: {', '.join(self.STATS)}")
        if unknown not in ("nan", "global"):
            raise ValueError(f"unknown은 'nan' 또는 'global'이어야 한다(받은 값: {unknown!r})")
        if smoothing < 0:
            raise ValueError(f"smoothing은 0 이상이어야 한다(받은 값: {smoothing})")
        if smoothing == 0 and "woe" in stats:
            raise ValueError("woe는 평활 없는(p=0/1) 셀에서 발산하므로 smoothing > 0이 필요하다.")
        self.cols = list(cols)
        self.stats = list(stats)
        self.smoothing = float(smoothing)
        self.unknown = unknown

        used = sorted({c for spec in self.cols for c in ([spec] if isinstance(spec, str) else spec)})
        proxy = _load_locked_proxy(path, sha256, used)

        label = proxy[_PROXY_LABEL].astype("float64")
        g = float(label.mean())
        self._tables: dict[str, dict[str, pd.Series]] = {}
        self._fills: dict[str, float] = {
            "mean": g,
            "woe": 0.0,
            "entropy": float(_binary_entropy(np.array([g]))[0]),
            "count": 0.0,
        }
        m = self.smoothing
        for spec in self.cols:
            keys = _aligned_spec_keys(proxy, spec)
            n = label.groupby(keys).size().astype("float64")
            p_raw = label.groupby(keys).mean()
            p = (n * p_raw + m * g) / (n + m)
            per_stat = {
                "mean": p,
                "count": np.log1p(n),
                "entropy": pd.Series(_binary_entropy(p.to_numpy()), index=p.index),
            }
            if "woe" in self.stats:
                per_stat["woe"] = np.log(p / (1 - p)) - np.log(g / (1 - g))
            self._tables[_spec_name(spec)] = {
                stat: per_stat[stat].astype("float64") for stat in self.stats
            }

    def columns(self) -> list[str]:
        return [f"{_spec_name(spec)}_orig_{stat}" for spec in self.cols for stat in self.stats]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for spec in self.cols:
            name = _spec_name(spec)
            keys = _aligned_spec_keys(df, spec)
            for stat in self.stats:
                mapped = keys.map(self._tables[name][stat])
                if self.unknown == "global":
                    mapped = mapped.fillna(self._fills[stat])
                out[f"{name}_orig_{stat}"] = mapped.astype("float64")
        return pd.DataFrame(out, index=df.index)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """이진 엔트로피. p가 0/1인 셀(평활 0)은 극한값 0을 준다."""
    out = np.zeros_like(p, dtype="float64")
    interior = (p > 0) & (p < 1)
    q = p[interior]
    out[interior] = -(q * np.log(q) + (1 - q) * np.log(1 - q))
    return out


class OriginalKnnColumns:
    """row-wise 제공자: 원본 프록시 최근접 행들의 라벨 평균. (#50 진단이 연 #54의 잔여 범위)

    수치 컬럼을 프록시 표준편차로 표준화한 뒤, 관측된 컬럼만의 제곱차 합(NaN 인지
    거리, #50 진단과 같은 정의)으로 각 행의 최근접 프록시 행 k개를 찾아 라벨 평균을
    새 열로 준다. k=1이면 최근접 행 라벨 그 자체다. 거리와 존재 플래그 표현은
    진단(docs/research/original-overlap-diagnosis.md)이 닫았으므로 만들지 않는다.

    통계의 원천이 해시 고정 외부 파일뿐이고 대회 타깃을 읽지 않으므로 행 단위
    결정적이다(uses_target=False, original_prior와 같은 근거로 row-wise).
    수치 컬럼이 전부 결측인 행은 거리가 정의되지 않아 NaN을 준다.
    새 컬럼 이름은 orig_nn<k>_mean.
    """

    uses_target = False

    def __init__(
        self,
        path: str,
        cols: list[str],
        ks: list[int] | None = None,
        sha256: str = ORIGINAL_PROXY_SHA256,
    ) -> None:
        ks = [1] if ks is None else ks
        if not ks or any(not isinstance(k, int) or k < 1 for k in ks):
            raise ValueError(f"ks는 1 이상의 정수 목록이어야 한다(받은 값: {ks})")
        if len(set(ks)) != len(ks):
            raise ValueError(f"ks에 중복이 있다: {ks}")
        self.cols = list(cols)
        self.ks = list(ks)

        proxy = _load_locked_proxy(path, sha256, self.cols)
        non_numeric = [c for c in self.cols if not pd.api.types.is_numeric_dtype(proxy[c])]
        if non_numeric:
            raise ValueError(f"거리는 수치 열 전용이다. 수치가 아닌 열: {non_numeric}")
        if max(self.ks) > len(proxy):
            raise ValueError(f"ks 최대값 {max(self.ks)}이 프록시 행 수 {len(proxy)}를 넘는다.")

        scale = proxy[self.cols].std().to_numpy(dtype="float64")
        if (scale == 0).any() or np.isnan(scale).any():
            bad = [c for c, s in zip(self.cols, scale) if s == 0 or np.isnan(s)]
            raise ValueError(f"프록시 표준편차가 0이거나 정의되지 않는 열: {bad}")
        self._scale = scale
        self._proxy_points = proxy[self.cols].to_numpy(dtype="float64") / scale
        self._proxy_labels = proxy[_PROXY_LABEL].to_numpy(dtype="float64")

    def columns(self) -> list[str]:
        return [f"orig_nn{k}_mean" for k in self.ks]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.cols].to_numpy(dtype="float64") / self._scale
        observed = ~np.isnan(X)
        X0 = np.where(observed, X, 0.0)
        P = self._proxy_points
        P2 = (P**2).T
        kmax = max(self.ks)
        out = {name: np.full(len(df), np.nan) for name in self.columns()}
        chunk = 4000
        for s in range(0, len(df), chunk):
            e = min(s + chunk, len(df))
            x0, m = X0[s:e], observed[s:e].astype(float)
            # d2_ij = sum_k m_ik (x_ik - p_jk)^2. 관측 컬럼 수 나눗셈은 행 내 순위에
            # 영향이 없어 생략한다. (#50 진단과 같은 마스크 행렬곱 전개)
            d2 = (x0**2).sum(axis=1)[:, None] - 2.0 * (x0 @ P.T) + m @ P2
            if kmax == 1:
                idx = d2.argmin(axis=1)[:, None]
            else:
                part = np.argpartition(d2, kmax - 1, axis=1)[:, :kmax]
                # 선택된 kmax개를 거리 오름차순으로 고정해 k별 접두 평균이 결정적이게 한다.
                order = np.take_along_axis(d2, part, axis=1).argsort(axis=1, kind="stable")
                idx = np.take_along_axis(part, order, axis=1)
            labels = self._proxy_labels[idx]
            for k in self.ks:
                out[f"orig_nn{k}_mean"][s:e] = labels[:, :k].mean(axis=1)
        # 수치 전결측 행은 모든 프록시 행과의 거리가 0으로 붕괴하므로 정의하지 않는다.
        no_numeric = ~observed.any(axis=1)
        result = pd.DataFrame(out, index=df.index).astype("float64")
        result.loc[no_numeric] = np.nan
        return result[self.columns()]


def _class_values(proxy: pd.DataFrame, col: str) -> dict[int, np.ndarray]:
    """프록시 열의 클래스별 관측값(결측 제거). 클래스가 비면 참조 분포가 정의 불가라 거부."""
    label = proxy[_PROXY_LABEL].astype("int64")
    out: dict[int, np.ndarray] = {}
    for cls in (0, 1):
        vals = proxy.loc[label == cls, col].dropna().to_numpy(dtype="float64")
        if len(vals) == 0:
            raise ValueError(f"프록시 열 {col}의 클래스 {cls}에 관측값이 없어 참조 분포를 만들 수 없다.")
        out[cls] = vals
    return out


class OriginalClassCdfDiff:
    """row-wise 제공자: 원본 프록시의 클래스별 경험적 CDF 차 F0(x) - F1(x). (#84가 연 #87)

    kodaifukuda 레시피(docs/research/original-distribution-coordinate-recipe.md)의
    클래스별 누적분포 차 재현이다. 열마다 프록시의 클래스 0/1 관측값으로 우측 포함
    경험적 CDF를 만들고, 대회 값 x에 F0(x) - F1(x)를 준다. 입력 결측은 NaN 유지.

    레시피의 중복 제거(대회 train 해시 일치 행과 프록시 내부 중복 제거)는 고정
    프록시에서 제거 0행으로 검증됐으므로(#84) 다시 수행하지 않는다. 해시 고정이
    다른 파일을 거부하니 이 생략은 결과 동일성을 바꾸지 않고, 제공자는 대회
    train을 읽지 않는 행 단위 결정적 매핑으로 남는다(#53과 같은 row-wise 근거).
    새 컬럼 이름은 <col>_orig_cdf_diff.
    """

    uses_target = False

    def __init__(
        self,
        path: str,
        cols: list[str],
        sha256: str = ORIGINAL_PROXY_SHA256,
    ) -> None:
        self.cols = list(cols)
        proxy = _load_locked_proxy(path, sha256, self.cols)
        non_numeric = [c for c in self.cols if not pd.api.types.is_numeric_dtype(proxy[c])]
        if non_numeric:
            raise ValueError(f"경험적 CDF는 수치 열 전용이다. 수치가 아닌 열: {non_numeric}")
        self._refs: dict[str, dict[int, np.ndarray]] = {
            col: {cls: np.sort(vals) for cls, vals in _class_values(proxy, col).items()}
            for col in self.cols
        }

    def columns(self) -> list[str]:
        return [f"{col}_orig_cdf_diff" for col in self.cols]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for col in self.cols:
            x = df[col].to_numpy(dtype="float64")
            observed = ~np.isnan(x)
            vals = np.full(len(df), np.nan)
            refs = self._refs[col]
            # 우측 포함 CDF: F(x) = #{ref <= x} / n. (레시피의 side="right")
            f0 = np.searchsorted(refs[0], x[observed], side="right") / len(refs[0])
            f1 = np.searchsorted(refs[1], x[observed], side="right") / len(refs[1])
            vals[observed] = f0 - f1
            out[f"{col}_orig_cdf_diff"] = pd.Series(vals, index=df.index, dtype="float64")
        return pd.DataFrame(out, index=df.index)


# KDE 대역폭 규칙(레시피 그대로): Silverman 0.9·scale·n^(-1/5)를 [0.10, 1.00]으로
# 자르고, 관측값이 둘보다 적으면(또는 scale이 전부 0이면) 0.30을 쓴다.
_KDE_BW_CLIP = (0.10, 1.00)
_KDE_BW_FALLBACK = 0.30
_KDE_LLR_CLIP = 20.0


def _silverman_bandwidth(vals: np.ndarray) -> float:
    if len(vals) < 2:
        return _KDE_BW_FALLBACK
    # 골든 대역폭 검산(#84) 결과 클래스 내 scale은 표본 표준편차(ddof=1) 기준이다.
    std = float(vals.std(ddof=1))
    iqr = float(np.subtract(*np.percentile(vals, [75, 25])))
    candidates = [s for s in (std, iqr / 1.34) if np.isfinite(s) and s > 0]
    if not candidates:
        return _KDE_BW_FALLBACK
    return float(np.clip(0.9 * min(candidates) * len(vals) ** (-1 / 5), *_KDE_BW_CLIP))


class OriginalKdeLogRatio:
    """row-wise 제공자: 원본 프록시의 클래스별 1차원 가우시안 KDE 로그밀도비. (#84가 연 #87)

    kodaifukuda 레시피의 커널 밀도 로그우도비 재현이다. 열을 프록시 전체의 평균과
    모집단 표준편차(ddof=0)로 표준화하고, 클래스별 Silverman 대역폭의 가우시안
    KernelDensity를 맞춘 뒤 log p(x|1) - log p(x|0)를 [-20, 20]으로 잘라 준다.
    입력 결측은 NaN 유지. 중복 제거 생략 근거는 OriginalClassCdfDiff와 같다.
    새 컬럼 이름은 <col>_orig_kde_lr.
    """

    uses_target = False

    def __init__(
        self,
        path: str,
        cols: list[str],
        sha256: str = ORIGINAL_PROXY_SHA256,
    ) -> None:
        from sklearn.neighbors import KernelDensity

        self.cols = list(cols)
        proxy = _load_locked_proxy(path, sha256, self.cols)
        non_numeric = [c for c in self.cols if not pd.api.types.is_numeric_dtype(proxy[c])]
        if non_numeric:
            raise ValueError(f"KDE는 수치 열 전용이다. 수치가 아닌 열: {non_numeric}")
        self._standardize: dict[str, tuple[float, float]] = {}
        self._kdes: dict[str, dict[int, KernelDensity]] = {}
        for col in self.cols:
            full = proxy[col].dropna().to_numpy(dtype="float64")
            mu, sd = float(full.mean()), float(full.std(ddof=0))
            if sd == 0 or not np.isfinite(sd):
                raise ValueError(f"프록시 열 {col}의 표준편차가 0이거나 정의되지 않아 표준화할 수 없다.")
            self._standardize[col] = (mu, sd)
            self._kdes[col] = {}
            for cls, vals in _class_values(proxy, col).items():
                z = (vals - mu) / sd
                kde = KernelDensity(kernel="gaussian", bandwidth=_silverman_bandwidth(z))
                kde.fit(z.reshape(-1, 1))
                self._kdes[col][cls] = kde

    def columns(self) -> list[str]:
        return [f"{col}_orig_kde_lr" for col in self.cols]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for col in self.cols:
            mu, sd = self._standardize[col]
            x = df[col].to_numpy(dtype="float64")
            vals = np.full(len(df), np.nan)
            # 값 격자가 성기므로 고유값에서만 채점해 행으로 되돌린다(결과는 행별 채점과 동일).
            unique, inverse = np.unique(x[~np.isnan(x)], return_inverse=True)
            if len(unique):
                z = ((unique - mu) / sd).reshape(-1, 1)
                llr = self._kdes[col][1].score_samples(z) - self._kdes[col][0].score_samples(z)
                llr = np.clip(llr, -_KDE_LLR_CLIP, _KDE_LLR_CLIP)
                vals[~np.isnan(x)] = llr[inverse]
            out[f"{col}_orig_kde_lr"] = pd.Series(vals, index=df.index, dtype="float64")
        return pd.DataFrame(out, index=df.index)


class ConstrainedImputeAux:
    """fold-fit 제공자: 생성 규칙을 산술 경계로 쓰는 제약 결측 재구성 열을 병행 추가. (#74)

    학습 fold의 수치 열로 IterativeImputer를 fit하고, 화면 블록(daily와 세 성분)의
    결측 셀 추정치를 생성 규칙 `daily >= social + gaming + work`가 주는 실현 가능
    구간으로 잘라 <col>_recon 열로 준다. 관측 셀은 원시 값 그대로다(원시 열은 덮지 않음).

    - daily 결측: 관측 성분 합이 하한. 상한 없음.
    - 성분 결측: [0, daily - 관측 성분 합]. daily도 결측이면 하한 0만 적용.

    widths=True면 성분 열에 실현 가능 구간 폭(daily - 관측 성분 합)을 <col>_recon_width로
    병행하되, 재구성이 일어난(결측이었던) 셀에서만 값을 준다. daily의 구간은 위로 열려
    있어 폭이 0/NaN 결측 지표로 퇴화하므로 daily 폭 열은 만들지 않는다(지도의 배제 경계).
    폭 열은 seed 42 스크리닝에서 gain importance가 플라시보 미달이라 widths=False의
    recon 전용 변형을 함께 둔다.
    """

    uses_target = False

    def __init__(self, cols: list[str], max_iter: int = 20, widths: bool = True) -> None:
        screen = [SCREEN_TOTAL, *SCREEN_PARTS]
        missing = [c for c in screen if c not in cols]
        if missing:
            raise ValueError(f"cols에 화면 블록 열 {missing}이 없다. 제약 재구성의 대상 열이다.")
        self.cols = list(cols)
        self.max_iter = max_iter
        self.widths = widths

    def columns(self) -> list[str]:
        screen = [SCREEN_TOTAL, *SCREEN_PARTS]
        cols = [f"{c}_recon" for c in screen]
        if self.widths:
            cols += [f"{c}_recon_width" for c in SCREEN_PARTS]
        return cols

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        self.imputer_ = IterativeImputer(max_iter=self.max_iter, random_state=seed)
        self.imputer_.fit(train_fold[self.cols])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        est = pd.DataFrame(
            self.imputer_.transform(df[self.cols]), columns=self.cols, index=df.index
        )
        # 관측 성분 합(결측은 0 기여). 결측 성분의 상한이자 daily의 하한이다.
        obs_sum = df[SCREEN_PARTS].sum(axis=1)
        # 성분 결측 셀의 구간 폭. daily 결측이면 NaN(상한 없음). 규칙 위반 방어로 0 하한.
        slack = (df[SCREEN_TOTAL] - obs_sum).clip(lower=0.0)

        out: dict[str, pd.Series] = {}
        rec = df[SCREEN_TOTAL].copy()
        m = rec.isna()
        rec[m] = est.loc[m, SCREEN_TOTAL].clip(lower=obs_sum[m])
        out[f"{SCREEN_TOTAL}_recon"] = rec.astype("float64")
        for c in SCREEN_PARTS:
            rec = df[c].copy()
            m = rec.isna()
            clipped = est.loc[m, c].clip(lower=0.0)
            upper = slack[m]
            clipped = clipped.where(upper.isna(), np.minimum(clipped, upper))
            rec[m] = clipped
            out[f"{c}_recon"] = rec.astype("float64")
        if self.widths:
            for c in SCREEN_PARTS:
                out[f"{c}_recon_width"] = slack.where(df[c].isna()).astype("float64")
        return pd.DataFrame(out, index=df.index)[self.columns()]


# tomasa2 노트북이 공개한 조건부 복원기의 모형 설정. 레시피 재현이 목적이라 상수로 둔다.
# (docs/research/kaggle-synthetic-forensics-increment.md, enable_categorical은 fit에서 상시 켠다)
XGB_IMPUTE_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.08,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "tree_method": "hist",
}

_IMP_D = "daily_screen_time_hours"
_IMP_S = "social_media_hours"
_IMP_G = "gaming_hours"
_IMP_W = "work_study_hours"
_IMP_WK = "weekend_screen_time"
_IMP_SL = "sleep_hours"
_IMP_N = "notifications_per_day"
_IMP_O = "app_opens_per_day"

# tomasa2 노트북 fe()의 조성 피처. 복원 행렬(관측 원시 값 + 결측 복원값) 위에서 계산하며,
# 이름과 식은 공개 소스 재현 범위로 한정한다(#90 결정 경계). name -> (필요 열, 계산식).
XGB_IMPUTE_COMPOSITIONS: dict[
    str, tuple[list[str], Callable[[pd.DataFrame], pd.Series]]
] = {
    "resid": (
        [_IMP_D, _IMP_S, _IMP_G, _IMP_W],
        lambda r: r[_IMP_D] - (r[_IMP_S] + r[_IMP_G] + r[_IMP_W]),
    ),
    "leisure": ([_IMP_D, _IMP_W], lambda r: r[_IMP_D] - r[_IMP_W]),
    "social_frac": ([_IMP_S, _IMP_D], lambda r: r[_IMP_S] / r[_IMP_D]),
    "work_frac": ([_IMP_W, _IMP_D], lambda r: r[_IMP_W] / r[_IMP_D]),
    "leisure_frac": (
        [_IMP_D, _IMP_W],
        lambda r: (r[_IMP_D] - r[_IMP_W]) / r[_IMP_D],
    ),
    "resid_frac": (
        [_IMP_D, _IMP_S, _IMP_G, _IMP_W],
        lambda r: (r[_IMP_D] - (r[_IMP_S] + r[_IMP_G] + r[_IMP_W])) / r[_IMP_D],
    ),
    "wk_ratio": ([_IMP_WK, _IMP_D], lambda r: r[_IMP_WK] / r[_IMP_D]),
    "week_total": ([_IMP_D, _IMP_WK], lambda r: 5 * r[_IMP_D] + 2 * r[_IMP_WK]),
    "awake_screen_frac": ([_IMP_D, _IMP_SL], lambda r: r[_IMP_D] / (24 - r[_IMP_SL])),
    "free_time": (
        [_IMP_SL, _IMP_D, _IMP_W],
        lambda r: 24 - r[_IMP_SL] - r[_IMP_D] - r[_IMP_W],
    ),
    "notif_per_open": ([_IMP_N, _IMP_O], lambda r: r[_IMP_N] / r[_IMP_O]),
    "min_per_open": ([_IMP_D, _IMP_O], lambda r: r[_IMP_D] * 60 / r[_IMP_O]),
}


class XgbImputeAux:
    """fold-fit 제공자: 각 결측 수치 열을 나머지 열로 예측한 XGBoost 조건부 복원 열. (#86)

    tomasa2 레시피(docs/research/kaggle-synthetic-forensics-increment.md)를 이 저장소의
    fold 규율로 옮긴 것이다. 열마다 XGBRegressor를 학습 fold에서 그 열이 관측된 행으로만
    fit하고(예측 입력은 나머지 수치 열과 범주 열, 타깃 미사용), 결측 셀에만 예측값을 채운
    <col>_xgb_recon 열을 준다. 관측 셀은 원시 값 그대로다(원시 열 대체와 별도 결측 표시
    열은 지도의 배제 경계). 수치 예측 입력의 NaN은 XGBoost native 결측 처리에 맡긴다.

    원 레시피는 train+test 결합 피처로 복원기를 맞춘다(전이 학습). 훈련 부분 전용
    판정이 통과해(#86의 단계 조건) transductive_test_path로 그 변형을 연다: 해시 고정
    test CSV의 무목표값 피처 행을 복원기 학습 표본에 합쳐, 변환 유무의 OOF 차이로
    전이 학습의 한계 기여를 분리한다. test에는 타깃이 없고 검증 fold 행은 여전히
    학습 표본에 들어가지 않으므로 라벨 누출 경로는 없다. 제약 결측 재구성
    (constrained_impute_aux)과 별개 계열이므로 산술 경계 클리핑은 하지 않는다.

    emit은 실제로 복원 열을 내보낼 cols의 부분집합이다(기본은 전부). 일부 복원 열이
    게이트 미달로 빠져도 예측 입력은 cols 전체로 유지되므로, 남는 복원 열의 모델
    입력이 전체 구성과 완전히 같아 같은 seed에서 같은 복원값이 보존된다. 복원기는
    emit 열과 compositions가 요구하는 열에만 만든다.

    compositions는 복원 행렬 위에서 계산하는 tomasa2 조성 피처(XGB_IMPUTE_COMPOSITIONS)
    의 부분집합이다(#90). 열 이름은 imp_<name>. 조성이 요구하는 열의 복원기는 emit에
    없어도 추가로 만들지만, 열마다 독립으로 fit하므로 emit 복원 열의 값은 조성 유무와
    무관하게 같다(#86 채택 열의 보존 근거).
    """

    uses_target = False

    def __init__(
        self,
        cols: list[str],
        cat_cols: list[str] | None = None,
        emit: list[str] | None = None,
        compositions: list[str] | None = None,
        transductive_test_path: str | None = None,
        transductive_test_sha256: str | None = None,
    ) -> None:
        cat_cols = [] if cat_cols is None else cat_cols
        all_cols = [*cols, *cat_cols]
        forbidden = {ID, TARGET, PLACEBO} & set(all_cols)
        if forbidden:
            raise ValueError(f"복원 대상과 예측 입력에 쓸 수 없는 열: {sorted(forbidden)}")
        # 목록 사이 겹침도 목록 안 중복도 열 하나가 두 역할을 갖는 같은 오류다.
        duplicated = sorted({c for c in all_cols if all_cols.count(c) > 1})
        if duplicated:
            raise ValueError(f"cols/cat_cols에 겹치는 열이 있다: {duplicated}")
        if len(cols) - 1 + len(cat_cols) < 1:
            raise ValueError("복원 대상 열마다 예측 입력이 하나 이상 필요하다(cols 2개 이상 또는 cat_cols).")
        self.cols = list(cols)
        self.cat_cols = list(cat_cols)
        self._test: pd.DataFrame | None = None
        if transductive_test_path is not None:
            from pathlib import Path

            if transductive_test_sha256 is None:
                raise ValueError(
                    "전이 학습 test는 해시 고정이 필요하다: transductive_test_sha256을 함께 줄 것."
                )
            actual = file_sha256(Path(transductive_test_path))
            if actual != transductive_test_sha256:
                raise ValueError(
                    f"전이 학습 test 해시 불일치: {transductive_test_path}\n"
                    f"기대 {transductive_test_sha256}\n실제 {actual}"
                )
            test = pd.read_csv(transductive_test_path)
            missing = sorted(c for c in all_cols if c not in test.columns)
            if missing:
                raise ValueError(f"전이 학습 test에 없는 열 {missing}.")
            self._test = test[all_cols]
        elif transductive_test_sha256 is not None:
            raise ValueError("transductive_test_sha256은 transductive_test_path 없이 쓸 수 없다.")
        if emit is None:
            self.emit = list(cols)
        else:
            unknown = sorted(set(emit) - set(cols))
            if unknown:
                raise ValueError(f"emit은 cols의 부분집합이어야 한다. cols에 없는 열: {unknown}")
            if len(set(emit)) != len(emit) or not emit:
                raise ValueError(f"emit은 중복 없는 비어 있지 않은 목록이어야 한다: {emit}")
            self.emit = list(emit)
        if compositions is None:
            self.compositions: list[str] = []
        else:
            unknown = sorted(set(compositions) - set(XGB_IMPUTE_COMPOSITIONS))
            if unknown:
                raise ValueError(
                    f"알 수 없는 composition: {unknown}. "
                    f"등록된 이름: {', '.join(XGB_IMPUTE_COMPOSITIONS)}"
                )
            if len(set(compositions)) != len(compositions) or not compositions:
                raise ValueError(
                    f"compositions는 중복 없는 비어 있지 않은 목록이어야 한다: {compositions}"
                )
            need = {
                c for name in compositions for c in XGB_IMPUTE_COMPOSITIONS[name][0]
            }
            outside = sorted(need - set(cols))
            if outside:
                raise ValueError(
                    f"compositions가 요구하는 열 {outside}이 cols에 없다. "
                    "조성은 복원 행렬 위에서만 계산한다."
                )
            self.compositions = list(compositions)
        comp_need = {
            c for name in self.compositions for c in XGB_IMPUTE_COMPOSITIONS[name][0]
        }
        # 복원기를 만들 열. cols 순서를 유지해 fit 순서가 결정적이다.
        self._recon_cols = [c for c in self.cols if c in ({*self.emit} | comp_need)]

    def columns(self) -> list[str]:
        return [f"{c}_xgb_recon" for c in self.emit] + [
            f"imp_{name}" for name in self.compositions
        ]

    def _predictors(self, target_col: str) -> list[str]:
        return [c for c in self.cols if c != target_col] + self.cat_cols

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None:
        from xgboost import XGBRegressor

        non_numeric = [c for c in self.cols if not pd.api.types.is_numeric_dtype(train_fold[c])]
        if non_numeric:
            raise ValueError(f"조건부 복원은 수치 열 전용이다. 수치가 아닌 열: {non_numeric}")
        bad_cat = [
            c
            for c in self.cat_cols
            if not isinstance(train_fold[c].dtype, pd.CategoricalDtype)
        ]
        if bad_cat:
            raise ValueError(
                f"cat_cols는 category dtype이어야 한다(코드 정렬을 align_categories가 보장): {bad_cat}"
            )
        fit_df = train_fold
        if self._test is not None:
            # 전이 학습: test의 무목표값 피처 행을 복원기 학습 표본에 합친다.
            # CSV로 읽힌 test의 범주 열을 train_fold의 카테고리 체계로 정렬해
            # 코드 배정이 어긋나지 않게 한다(합집합 정렬이므로 미지 값은 없어야 한다).
            test = self._test.copy()
            for c in self.cat_cols:
                cats = train_fold[c].dtype.categories
                unseen = sorted(set(test[c].dropna()) - set(cats))
                if unseen:
                    raise ValueError(f"전이 학습 test의 {c}에 train에 없는 값 {unseen}이 있다.")
                test[c] = pd.Categorical(test[c], categories=cats)
            fit_df = pd.concat(
                [train_fold[[*self.cols, *self.cat_cols]], test], ignore_index=True
            )
        self.models_: dict[str, XGBRegressor] = {}
        for c in self._recon_cols:
            observed = fit_df[c].notna()
            if not observed.any():
                raise ValueError(f"학습 fold에서 {c}의 관측 행이 없어 복원기를 만들 수 없다.")
            model = XGBRegressor(
                **XGB_IMPUTE_PARAMS, enable_categorical=True, random_state=seed
            )
            model.fit(
                fit_df.loc[observed, self._predictors(c)], fit_df.loc[observed, c]
            )
            self.models_[c] = model

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        recon = pd.DataFrame(index=df.index)
        for c in self._recon_cols:
            rec = df[c].astype("float64").copy()
            m = rec.isna()
            if m.any():
                rec[m] = self.models_[c].predict(df.loc[m, self._predictors(c)]).astype("float64")
            recon[c] = rec
        out: dict[str, pd.Series] = {f"{c}_xgb_recon": recon[c] for c in self.emit}
        for name in self.compositions:
            out[f"imp_{name}"] = XGB_IMPUTE_COMPOSITIONS[name][1](recon).astype("float64")
        return pd.DataFrame(out, index=df.index)[self.columns()]


class MedianImputeAux:
    """fold-fit 제공자: 원시 NaN 열을 유지한 채 중앙값 대체본을 보조 열로 추가한다. (#49)

    원래 열을 덮는 대체는 지도의 배제 대상이고, 근거가 강한 것은 두 열 병행이다
    (docs/research/code-notebook-insights.md 전처리 절). 중앙값은 학습 fold에서만 계산한다.
    새 컬럼 이름은 <col>_fill.
    """

    uses_target = False

    def __init__(self, cols: list[str]) -> None:
        self.cols = list(cols)

    def columns(self) -> list[str]:
        return [f"{col}_fill" for col in self.cols]

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None:
        self.medians_ = {col: float(train_fold[col].median()) for col in self.cols}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {f"{col}_fill": df[col].fillna(self.medians_[col]) for col in self.cols}
        return pd.DataFrame(out, index=df.index)
