"""잔차 부스팅의 행별 초기 로짓 생성기.

두 가지 계약이 있다.

- 자료 전체 계약(``InitialScoreProvider``): 합성 타깃을 입력으로 받지 않는다.
  해시로 고정한 원본 프록시나 알려진 원본 규칙만으로 train/test의 초기 확률을 만들고,
  안전하게 로짓으로 바꾼다. 시드마다 한 번 계산해 모든 바깥쪽 분할이 나눠 쓴다. (#52)
- 바깥쪽 분할 계약(``OuterFoldInitialScoreProvider``): 합성 타깃을 쓰되 바깥쪽 분할
  실행이 소유한다. 바깥쪽 학습 부분의 목표값만 받아 학습 부분 행은 내부 OOF로,
  바깥쪽 검증과 시험 행은 학습 부분 전체로 맞춘 1단 모형으로 예측한다.
  검증과 시험 frame에 목표값 열이 있으면 거부한다. (#505)

LightGBM 모델은 이 로짓에서 시작해 잔차만 학습한다.
"""

from __future__ import annotations

import hashlib
import time
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
from .features import ORIGINAL_PROXY_SHA256, PLACEBO

EVIDENCE_NAME = "initial_score_evidence.json"
EVIDENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class InitialScores:
    train: pd.Series
    test: pd.Series


class InitialScoreProvider(Protocol):
    def compute(self, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> InitialScores: ...

    def input_paths(self) -> dict[str, Path]: ...


@dataclass(frozen=True)
class OuterFoldInitialScores:
    """바깥쪽 분할 하나의 초기 로짓. 인덱스는 각각 학습 부분, 검증 부분, 시험 frame과 같다."""

    training: pd.Series
    validation: pd.Series
    test: pd.Series
    evidence: dict[str, object]


@dataclass(frozen=True)
class FullDataInitialScores:
    """전체 자료 재학습의 초기 로짓. 학습 행은 내부 OOF, 시험 행은 전체 학습 자료 적합이다."""

    train: pd.Series
    test: pd.Series
    evidence: dict[str, object]


class OuterFoldInitialScoreProvider(Protocol):
    """바깥쪽 분할 실행이 소유하는, 학습 부분 목표값을 쓰는 초기 점수 계약. (#505)"""

    def compute_outer_fold(
        self,
        training: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        seed: int,
        outer_fold: int,
    ) -> OuterFoldInitialScores: ...

    def compute_full(
        self, training: pd.DataFrame, test: pd.DataFrame, seed: int
    ) -> FullDataInitialScores: ...

    def input_paths(self) -> dict[str, Path]: ...


@dataclass(frozen=True)
class FoldInitialScores:
    """바깥쪽 분할 하나가 모델에 건네는 초기 로짓. 계약과 무관한 공통 형태."""

    training: pd.Series
    validation: pd.Series
    test: pd.Series
    evidence: dict[str, object] | None


def is_outer_fold_provider(provider: object) -> bool:
    return provider is not None and hasattr(provider, "compute_outer_fold")


def seed_level_scores(
    provider: object, train: pd.DataFrame, test: pd.DataFrame, seed: int
) -> InitialScores | None:
    """시드마다 한 번 계산하는 자료 전체 계약의 초기 점수.

    바깥쪽 분할 계약 생성기는 분할 안에서만 계산하므로 None을 돌려준다.
    ``train``은 목표값 열을 가진 학습 frame이다. 자료 전체 계약에는 목표값을 떼고 건넨다.
    """
    if provider is None or is_outer_fold_provider(provider):
        return None
    scores = provider.compute(train.drop(columns=[TARGET]), test, seed)
    if not scores.train.index.equals(train.index):
        raise AssertionError("train 초기 점수 인덱스가 다르다.")
    if not scores.test.index.equals(test.index):
        raise AssertionError("test 초기 점수 인덱스가 다르다.")
    return scores


def fold_scores(
    provider: object,
    seed_scores: InitialScores | None,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    fold: int,
    training_index: pd.Index,
    validation_index: pd.Index,
) -> FoldInitialScores | None:
    """바깥쪽 분할 하나의 초기 로짓을 계약에 맞게 만든다.

    자료 전체 계약은 시드 수준 점수를 분할 인덱스로 자르고, 바깥쪽 분할 계약은
    학습 부분(목표값 포함)과 검증 부분(목표값 제거)으로 새로 계산한다.
    """
    if provider is None:
        return None
    if is_outer_fold_provider(provider):
        result = provider.compute_outer_fold(
            train.loc[training_index],
            train.loc[validation_index].drop(columns=[TARGET]),
            test,
            seed,
            fold,
        )
        if not result.training.index.equals(pd.Index(training_index)):
            raise AssertionError(f"fold {fold} 학습 부분 초기 점수 인덱스가 다르다.")
        if not result.validation.index.equals(pd.Index(validation_index)):
            raise AssertionError(f"fold {fold} 검증 부분 초기 점수 인덱스가 다르다.")
        if not result.test.index.equals(test.index):
            raise AssertionError(f"fold {fold} 시험 초기 점수 인덱스가 다르다.")
        return FoldInitialScores(
            training=result.training,
            validation=result.validation,
            test=result.test,
            evidence=dict(result.evidence),
        )
    if seed_scores is None:
        raise AssertionError("자료 전체 계약의 시드 수준 초기 점수가 없다.")
    return FoldInitialScores(
        training=seed_scores.train.loc[training_index],
        validation=seed_scores.train.loc[validation_index],
        test=seed_scores.test,
        evidence=None,
    )


def full_data_scores(
    provider: object, train: pd.DataFrame, test: pd.DataFrame, seed: int
) -> InitialScores | None:
    """전체 자료 재학습의 초기 로짓. ``train``은 목표값 열을 가진 학습 frame이다."""
    if provider is None:
        return None
    if is_outer_fold_provider(provider):
        result = provider.compute_full(train, test, seed)
        if not result.train.index.equals(train.index):
            raise AssertionError("전체 자료 학습 초기 점수 인덱스가 다르다.")
        if not result.test.index.equals(test.index):
            raise AssertionError("전체 자료 시험 초기 점수 인덱스가 다르다.")
        print(f"initial_score full-data evidence: {result.evidence}")
        return InitialScores(train=result.train, test=result.test)
    scores = provider.compute(train.drop(columns=[TARGET]), test, seed)
    if not scores.train.index.equals(train.index):
        raise AssertionError("전체 자료 학습 초기 점수 인덱스가 다르다.")
    if not scores.test.index.equals(test.index):
        raise AssertionError("전체 자료 시험 초기 점수 인덱스가 다르다.")
    return scores


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


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype="float64").tobytes()
    ).hexdigest()


def _logit_range(values: np.ndarray) -> list[float]:
    return [float(values.min()), float(values.max())]


class NestedLogisticOnehot:
    """정확값 one-hot L2 로지스틱 회귀의 누출 없는 내부 OOF 로짓. (#505)

    바깥쪽 분할 계약이다. 바깥쪽 학습 부분의 목표값만 받아
    - 학습 부분 행의 초기 로짓은 학습 부분 안에서만 만든 ``inner_splits`` 층화 OOF 예측,
    - 바깥쪽 검증과 시험 행의 초기 로짓은 학습 부분 전체로 맞춘 회귀의 예측이다.
    로지스틱 회귀 예측은 초기 점수로만 쓰이고 일반 입력 열로 들어가지 않는다.
    인코딩과 적합은 ``logistic_onehot`` adapter를 그대로 재사용한다(exp058 계열).
    """

    def __init__(
        self,
        cols: list[str],
        categorical: list[str],
        C: float = 100.0,
        max_iter: int = 3000,
        onehot_max_card: int = 5000,
        inner_splits: int = 10,
        clip: float = 1e-6,
    ) -> None:
        self.cols = list(cols)
        self.categorical = list(categorical)
        self.inner_splits = int(inner_splits)
        self.clip = float(clip)
        self._model_params = {
            "C": float(C),
            "max_iter": int(max_iter),
            "onehot_max_card": int(onehot_max_card),
        }
        if not self.cols:
            raise ValueError("cols는 비어 있을 수 없다.")
        if len(set(self.cols)) != len(self.cols):
            raise ValueError("cols에 중복 열이 있다.")
        forbidden = sorted(set(self.cols) & {ID, TARGET, PLACEBO, "fold"})
        if forbidden:
            raise ValueError(f"{forbidden}은 초기 점수 설명변수로 쓸 수 없다.")
        if not set(self.categorical) <= set(self.cols):
            raise ValueError("categorical은 cols의 부분집합이어야 한다.")
        if self.inner_splits < 2:
            raise ValueError(f"inner_splits는 2 이상이어야 한다(받은 값: {inner_splits})")
        probabilities_to_logits([0.5], self.clip)
        # 파라미터 검증은 adapter 생성자가 소유한다. 잘못된 조합은 설정 적재 때 실패한다.
        self._adapter(0)

    def _adapter(self, seed: int):
        from .model import LogisticOnehotAdapter

        return LogisticOnehotAdapter(dict(self._model_params), {}, seed)

    def _matrix(self, frame: pd.DataFrame, name: str) -> pd.DataFrame:
        missing = sorted(set(self.cols) - set(frame.columns))
        if missing:
            raise ValueError(f"{name}에 초기 점수 설명변수가 없다: {missing}")
        return frame[self.cols].copy()

    def _training_target(self, training: pd.DataFrame) -> pd.Series:
        if TARGET not in training.columns:
            raise ValueError(f"학습 부분에 목표값 {TARGET!r} 열이 없다.")
        y = training[TARGET]
        if y.isna().any():
            raise ValueError("학습 부분 목표값에 결측이 있다.")
        return y.astype(int)

    def _inner_oof(
        self, X: pd.DataFrame, y: pd.Series, seed: int
    ) -> tuple[np.ndarray, list[int]]:
        splitter = StratifiedKFold(n_splits=self.inner_splits, shuffle=True, random_state=seed)
        oof = np.full(len(X), np.nan, dtype="float64")
        iterations: list[int] = []
        for tr_i, va_i in splitter.split(X, y):
            adapter = self._adapter(seed)
            oof[va_i] = np.asarray(
                adapter.fit(X.iloc[tr_i], y.iloc[tr_i], X.iloc[va_i], y.iloc[va_i]),
                dtype="float64",
            )
            iterations.append(adapter.training_iterations())
        if not np.isfinite(oof).all():
            raise AssertionError("내부 OOF 예측에 채워지지 않았거나 유한하지 않은 값이 있다.")
        return oof, iterations

    def _fit_and_predict(
        self, training: pd.DataFrame, others: dict[str, pd.DataFrame], seed: int
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
        started = time.perf_counter()
        X_tr = self._matrix(training, "training")
        y_tr = self._training_target(training)
        inner_oof, inner_iterations = self._inner_oof(X_tr, y_tr, seed)
        full = self._adapter(seed)
        full.fit_full(X_tr, y_tr, None)
        predictions = {
            name: np.asarray(full.predict(self._matrix(frame, name)), dtype="float64")
            for name, frame in others.items()
        }
        evidence: dict[str, object] = {
            "kind": "nested_logistic_onehot",
            "seed": int(seed),
            "inner_splits": self.inner_splits,
            "clip": self.clip,
            "model_params": dict(self._model_params),
            "cols": list(self.cols),
            "training_rows": int(len(training)),
            "inner_oof_auc": float(roc_auc_score(y_tr, inner_oof)),
            "inner_fit_iterations": inner_iterations,
            "full_fit_iterations": full.training_iterations(),
            "n_features": full.feature_count(),
            "seconds": time.perf_counter() - started,
        }
        return inner_oof, predictions, evidence

    def _logits(self, probabilities: np.ndarray, index: pd.Index) -> pd.Series:
        return pd.Series(
            probabilities_to_logits(probabilities, self.clip), index=index, dtype="float64"
        )

    def compute_outer_fold(
        self,
        training: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        seed: int,
        outer_fold: int,
    ) -> OuterFoldInitialScores:
        _reject_synthetic_target(validation, test)
        if not training.index.intersection(validation.index).empty:
            raise ValueError("바깥쪽 학습 부분과 검증 부분의 행이 겹친다.")
        inner_oof, predictions, evidence = self._fit_and_predict(
            training, {"validation": validation, "test": test}, seed
        )
        scores = OuterFoldInitialScores(
            training=self._logits(inner_oof, training.index),
            validation=self._logits(predictions["validation"], validation.index),
            test=self._logits(predictions["test"], test.index),
            evidence=evidence,
        )
        evidence.update(
            {
                "outer_fold": int(outer_fold),
                "validation_rows": int(len(validation)),
                "test_rows": int(len(test)),
                "logit_range": {
                    "training": _logit_range(scores.training.to_numpy()),
                    "validation": _logit_range(scores.validation.to_numpy()),
                    "test": _logit_range(scores.test.to_numpy()),
                },
                "sha256": {
                    "training": _array_sha256(scores.training.to_numpy()),
                    "validation": _array_sha256(scores.validation.to_numpy()),
                    "test": _array_sha256(scores.test.to_numpy()),
                },
            }
        )
        print(
            f"initial_score nested_logistic_onehot seed={seed} fold={outer_fold} "
            f"inner_oof_auc={evidence['inner_oof_auc']:.5f} "
            f"training_logit_range={evidence['logit_range']['training']} "
            f"seconds={evidence['seconds']:.1f}"
        )
        return scores

    def compute_full(
        self, training: pd.DataFrame, test: pd.DataFrame, seed: int
    ) -> FullDataInitialScores:
        if TARGET in test.columns:
            raise ValueError(f"시험 frame에 합성 타깃 {TARGET!r} 열이 있다.")
        inner_oof, predictions, evidence = self._fit_and_predict(training, {"test": test}, seed)
        scores = FullDataInitialScores(
            train=self._logits(inner_oof, training.index),
            test=self._logits(predictions["test"], test.index),
            evidence=evidence,
        )
        evidence.update(
            {
                "outer_fold": None,
                "test_rows": int(len(test)),
                "logit_range": {
                    "training": _logit_range(scores.train.to_numpy()),
                    "test": _logit_range(scores.test.to_numpy()),
                },
                "sha256": {
                    "training": _array_sha256(scores.train.to_numpy()),
                    "test": _array_sha256(scores.test.to_numpy()),
                },
            }
        )
        return scores

    def input_paths(self) -> dict[str, Path]:
        return {}


REGISTRY: dict[str, Callable[..., InitialScoreProvider | OuterFoldInitialScoreProvider]] = {
    "original_proxy_lightgbm": OriginalProxyLightGBM,
    "known_original_rule": KnownOriginalRule,
    "nested_logistic_onehot": NestedLogisticOnehot,
}


def create(
    cfg: InitialScoreConfig | None,
) -> InitialScoreProvider | OuterFoldInitialScoreProvider | None:
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
