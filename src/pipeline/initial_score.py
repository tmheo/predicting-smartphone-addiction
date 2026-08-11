"""잔차 부스팅의 행별 초기 로짓 생성기.

초기 점수 생성기는 합성 타깃을 입력으로 받지 않는다.
해시로 고정한 원본 프록시나 알려진 원본 규칙만으로 train/test의 초기 확률을 만들고,
안전하게 로짓으로 바꾼다.
LightGBM 모델은 이 로짓에서 시작해 합성 생성기 잔차만 학습한다. (#52)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import InitialScoreConfig
from .data import ID, TARGET, file_sha256
from .features import ORIGINAL_PROXY_SHA256


@dataclass(frozen=True)
class InitialScores:
    train: pd.Series
    test: pd.Series


class InitialScoreProvider(Protocol):
    def compute(self, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> InitialScores: ...

    def input_paths(self) -> dict[str, Path]: ...


def probabilities_to_logits(probabilities, clip: float = 1e-6) -> np.ndarray:
    """확률 극단값을 자른 뒤 유한한 float64 로짓으로 바꾼다."""
    if not 0 < clip < 0.5:
        raise ValueError(f"clip은 0과 0.5 사이여야 한다(받은 값: {clip})")
    p = np.asarray(probabilities, dtype="float64")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("초기 확률은 유한한 0 이상 1 이하 값이어야 한다.")
    p = np.clip(p, clip, 1.0 - clip)
    return np.log(p) - np.log1p(-p)


def _reject_synthetic_target(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if TARGET in train.columns or TARGET in test.columns:
        raise ValueError(
            f"초기 점수 생성기는 합성 타깃 {TARGET!r}을 입력으로 받을 수 없다. "
            "설명변수만 전달해야 한다."
        )


def _as_categorical_together(frames: list[pd.DataFrame], cols: list[str]) -> None:
    for col in cols:
        values = pd.concat([frame[col] for frame in frames], ignore_index=True)
        categories = sorted(values.dropna().astype(str).unique())
        for frame in frames:
            frame[col] = pd.Categorical(frame[col].astype("string"), categories=categories)


class OriginalProxyLightGBM:
    """원본 프록시만 학습한 LightGBM 시드 평균본의 예측 로짓."""

    def __init__(
        self,
        path: str,
        cols: list[str],
        categorical: list[str],
        model_params: dict,
        n_splits: int = 5,
        early_stopping_rounds: int = 100,
        clip: float = 1e-6,
        sha256: str = ORIGINAL_PROXY_SHA256,
    ) -> None:
        self.path = Path(path)
        self.cols = list(cols)
        self.categorical = list(categorical)
        self.model_params = dict(model_params)
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.clip = clip

        actual = file_sha256(self.path)
        if actual != sha256:
            raise ValueError(
                f"원본 프록시 해시 불일치: {path}\n기대 {sha256}\n실제 {actual}\n"
                "docs/research/original-proxy-data.md의 재현 절차로 판본 1을 다시 받을 것."
            )
        if n_splits < 2:
            raise ValueError(f"n_splits는 2 이상이어야 한다(받은 값: {n_splits})")
        probabilities_to_logits([0.5], clip)
        if not set(self.categorical) <= set(self.cols):
            raise ValueError("categorical은 cols의 부분집합이어야 한다.")

        self._proxy = pd.read_csv(self.path)
        forbidden = sorted(set(self.cols) & {"transaction_id", "user_id", "addiction_level", ID, TARGET})
        if forbidden:
            raise ValueError(f"프록시 전용 열 {forbidden}은 초기 점수 설명변수로 쓸 수 없다.")
        required = set(self.cols) | {TARGET}
        missing = sorted(required - set(self._proxy.columns))
        if missing:
            raise ValueError(f"원본 프록시에 필요한 열이 없다: {missing}")
        if self._proxy[TARGET].isna().any():
            raise ValueError("원본 프록시 타깃에 결측이 있다.")

    def compute(self, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> InitialScores:
        _reject_synthetic_target(train, test)
        for name, frame in (("train", train), ("test", test)):
            missing = sorted(set(self.cols) - set(frame.columns))
            if missing:
                raise ValueError(f"{name}에 초기 점수 설명변수가 없다: {missing}")

        import lightgbm as lgb

        proxy = self._proxy[self.cols + [TARGET]].copy()
        train_x = train[self.cols].copy()
        test_x = test[self.cols].copy()
        _as_categorical_together([proxy, train_x, test_x], self.categorical)
        X_proxy = proxy[self.cols]
        y_proxy = proxy[TARGET].astype(int)

        splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
        proxy_oof = np.zeros(len(proxy), dtype="float64")
        train_prob = np.zeros(len(train), dtype="float64")
        test_prob = np.zeros(len(test), dtype="float64")
        for tr_i, va_i in splitter.split(X_proxy, y_proxy):
            model = lgb.LGBMClassifier(**self.model_params, random_state=seed)
            model.fit(
                X_proxy.iloc[tr_i],
                y_proxy.iloc[tr_i],
                eval_X=X_proxy.iloc[va_i],
                eval_y=y_proxy.iloc[va_i],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
            proxy_oof[va_i] = model.predict_proba(X_proxy.iloc[va_i])[:, 1]
            train_prob += model.predict_proba(train_x)[:, 1] / self.n_splits
            test_prob += model.predict_proba(test_x)[:, 1] / self.n_splits

        print(
            "initial_score original_proxy_lightgbm "
            f"proxy_oof_auc={roc_auc_score(y_proxy, proxy_oof):.5f} "
            f"train_probability_range=[{train_prob.min():.6g}, {train_prob.max():.6g}]"
        )
        return InitialScores(
            train=pd.Series(
                probabilities_to_logits(train_prob, self.clip), index=train.index, dtype="float64"
            ),
            test=pd.Series(
                probabilities_to_logits(test_prob, self.clip), index=test.index, dtype="float64"
            ),
        )

    def input_paths(self) -> dict[str, Path]:
        return {"initial_score": self.path}


class KnownOriginalRule:
    """공개된 daily/social 임계값 원본 규칙의 로짓.

    daily > 8 또는 social > 4면 양성, daily <= 6이고 social <= 4면 음성,
    그 사이는 0.5다.
    원본의 0/1 출력을 clip한 뒤 로짓으로 바꾼다.
    """

    def __init__(self, clip: float = 1e-3) -> None:
        self.clip = clip
        probabilities_to_logits([0.5], clip)

    def compute(self, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> InitialScores:
        _reject_synthetic_target(train, test)

        def margins(df: pd.DataFrame) -> pd.Series:
            daily = df["daily_screen_time_hours"]
            social = df["social_media_hours"]
            p = pd.Series(0.5, index=df.index, dtype="float64")
            p[(daily > 8) | (social > 4)] = 1.0
            p[(daily <= 6) & (social <= 4)] = 0.0
            return pd.Series(probabilities_to_logits(p, self.clip), index=df.index)

        return InitialScores(train=margins(train), test=margins(test))

    def input_paths(self) -> dict[str, Path]:
        return {}


REGISTRY: dict[str, Callable[..., InitialScoreProvider]] = {
    "original_proxy_lightgbm": OriginalProxyLightGBM,
    "known_original_rule": KnownOriginalRule,
}


def create(cfg: InitialScoreConfig | None) -> InitialScoreProvider | None:
    if cfg is None:
        return None
    if cfg.kind not in REGISTRY:
        raise ValueError(
            f"알 수 없는 initial_score.kind {cfg.kind!r}. 등록된 kind: "
            f"{', '.join(sorted(REGISTRY))}"
        )
    try:
        return REGISTRY[cfg.kind](**cfg.params)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"initial_score {cfg.kind}: {exc}") from exc


def input_paths(cfg: InitialScoreConfig | None) -> dict[str, Path]:
    provider = create(cfg)
    return provider.input_paths() if provider is not None else {}
