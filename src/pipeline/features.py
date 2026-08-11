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
        stats: list[str] = ["mean"],
        smoothing: float = 20.0,
        unknown: str = "nan",
        sha256: str = ORIGINAL_PROXY_SHA256,
    ) -> None:
        from pathlib import Path

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

        actual = file_sha256(Path(path))
        if actual != sha256:
            raise ValueError(
                f"원본 프록시 해시 불일치: {path}\n기대 {sha256}\n실제 {actual}\n"
                "docs/research/original-proxy-data.md의 재현 절차로 판본 1을 다시 받을 것."
            )
        proxy = pd.read_csv(path)
        used = {c for spec in self.cols for c in ([spec] if isinstance(spec, str) else spec)}
        forbidden = sorted(used & _PROXY_FORBIDDEN_COLS)
        if forbidden:
            raise ValueError(f"프록시 전용 열 {forbidden}은 대회 설명변수로 쓸 수 없다. (#47 경계)")
        missing = sorted(c for c in used if c not in proxy.columns)
        if missing:
            raise ValueError(f"프록시에 없는 열 {missing}. 프록시 열: {sorted(proxy.columns)}")

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
        ks: list[int] = [1],
        sha256: str = ORIGINAL_PROXY_SHA256,
    ) -> None:
        from pathlib import Path

        if not ks or any(not isinstance(k, int) or k < 1 for k in ks):
            raise ValueError(f"ks는 1 이상의 정수 목록이어야 한다(받은 값: {ks})")
        if len(set(ks)) != len(ks):
            raise ValueError(f"ks에 중복이 있다: {ks}")
        self.cols = list(cols)
        self.ks = list(ks)

        actual = file_sha256(Path(path))
        if actual != sha256:
            raise ValueError(
                f"원본 프록시 해시 불일치: {path}\n기대 {sha256}\n실제 {actual}\n"
                "docs/research/original-proxy-data.md의 재현 절차로 판본 1을 다시 받을 것."
            )
        proxy = pd.read_csv(path)
        forbidden = sorted(set(self.cols) & _PROXY_FORBIDDEN_COLS)
        if forbidden:
            raise ValueError(f"프록시 전용 열 {forbidden}은 대회 설명변수로 쓸 수 없다. (#47 경계)")
        missing = sorted(c for c in self.cols if c not in proxy.columns)
        if missing:
            raise ValueError(f"프록시에 없는 열 {missing}. 프록시 열: {sorted(proxy.columns)}")
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


class ConstrainedImputeAux:
    """fold-fit 제공자: 생성 규칙을 산술 경계로 쓰는 제약 결측 재구성 열을 병행 추가. (#74)

    학습 fold의 수치 열로 IterativeImputer를 fit하고, 화면 블록(daily와 세 성분)의
    결측 셀 추정치를 생성 규칙 `daily >= social + gaming + work`가 주는 실현 가능
    구간으로 잘라 <col>_recon 열로 준다. 관측 셀은 원시 값 그대로다(원시 열은 덮지 않음).

    - daily 결측: 관측 성분 합이 하한. 상한 없음.
    - 성분 결측: [0, daily - 관측 성분 합]. daily도 결측이면 하한 0만 적용.

    성분 열에는 실현 가능 구간 폭(daily - 관측 성분 합)을 <col>_recon_width로 병행하되,
    재구성이 일어난(결측이었던) 셀에서만 값을 준다. daily의 구간은 위로 열려 있어 폭이
    0/NaN 결측 지표로 퇴화하므로 daily 폭 열은 만들지 않는다(지도의 배제 경계).
    """

    uses_target = False

    def __init__(self, cols: list[str], max_iter: int = 20) -> None:
        screen = [SCREEN_TOTAL, *SCREEN_PARTS]
        missing = [c for c in screen if c not in cols]
        if missing:
            raise ValueError(f"cols에 화면 블록 열 {missing}이 없다. 제약 재구성의 대상 열이다.")
        self.cols = list(cols)
        self.max_iter = max_iter

    def columns(self) -> list[str]:
        screen = [SCREEN_TOTAL, *SCREEN_PARTS]
        return [f"{c}_recon" for c in screen] + [f"{c}_recon_width" for c in SCREEN_PARTS]

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
        for c in SCREEN_PARTS:
            out[f"{c}_recon_width"] = slack.where(df[c].isna()).astype("float64")
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
