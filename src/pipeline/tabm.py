"""TabM 학습기(pytabkit) fold 구현. (#61)

원문(szymonkapiski의 S6E8 TabM with constrained imputation, OOF 0.96867) 학습부의
재현: pytabkit TabM_D_Classifier(tabm-mini-normal, PWL 수치 embedding)를 fold 안에서
시드 평균(원문 N_SEEDS=3)해 잡음이 적은 예측 하나를 만든다. 앙상블에 모델 3개를
넣는 것이 아니라 fold를 떠나기 전에 평균하는 방식이라 후보 풀 규약과 충돌하지 않는다.

원문과 다른 점(티켓 #61의 누출 해소):
- 원문은 train+test 전체로 iterative imputer·중앙값 대체·격자 TE를 만들었다.
  여기서는 피처가 전부 파이프라인 provider(fold-fit)에서 오고, 이 모듈은 pytabkit이
  거부하는 수치 열의 남은 NaN 중앙값 대체만 학습 fold 통계로 수행한다.
- 원문의 자기 포함 평활 TE 대신 파이프라인의 내부 OOF TE를 쓴다(config 소관).
- 결측 지표·결측 개수 열(지도의 배제 경계)은 만들지 않는다.

pytabkit(torch)이 필요하므로 model.py의 adapter가 이 모듈을 lazy import한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 원문 노트북의 TabM 하이퍼파라미터 그대로가 기본값이다.
_PYTABKIT_DEFAULTS = {
    "arch_type": "tabm-mini-normal",
    "tabm_k": 16,
    "num_emb_type": "pwl",
    "d_embedding": 16,
    "batch_size": 128,
    "lr": 5e-4,
    "n_epochs": 150,
    "dropout": 0.02,
    "d_block": 160,
    "n_blocks": 10,
    "weight_decay": 1e-2,
    "patience": 5,
}


class TabMFold:
    """fold 하나의 전처리·시드 평균 학습·예측·중요도 상태. adapter가 소유한다."""

    def __init__(self, params: dict, seed: int) -> None:
        params = dict(params)
        self._n_seed_avg = int(params.pop("n_seed_avg", 3))
        self._perm_repeats = int(params.pop("perm_repeats", 3))
        self._perm_sample = int(params.pop("perm_sample", 50_000))
        self._pytabkit = {
            k: params.pop(k, default) for k, default in _PYTABKIT_DEFAULTS.items()
        }
        if params:
            raise ValueError(f"tabm이 모르는 params: {sorted(params)}")
        if self._n_seed_avg < 1:
            raise ValueError(f"n_seed_avg는 1 이상이어야 한다: {self._n_seed_avg}")
        self._seed = seed
        self._models: list = []
        self._medians: dict[str, float] = {}
        self._columns: list[str] | None = None
        self._val: tuple[pd.DataFrame, np.ndarray] | None = None
        self._val_auc: float | None = None

    # ---- 전처리: 중앙값은 학습 fold 통계만 쓴다(outer fold 규율) ----

    def _fit_medians(self, X_tr: pd.DataFrame) -> None:
        self._columns = list(X_tr.columns)
        self._medians = {}
        for col in self._columns:
            if isinstance(X_tr[col].dtype, pd.CategoricalDtype):
                continue  # 범주 결측은 pytabkit이 자체 카테고리로 다룬다.
            values = X_tr[col].replace([np.inf, -np.inf], np.nan)
            median = values.median()
            # 학습 fold가 전부 결측인 열은 원문(fillna(median) 후 fillna(0))처럼 0.
            self._medians[col] = float(median) if pd.notna(median) else 0.0

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert list(X.columns) == self._columns, "입력 컬럼이 학습 때와 다르다."
        out = X.copy()
        for col, median in self._medians.items():
            values = out[col].replace([np.inf, -np.inf], np.nan)
            out[col] = values.fillna(median).astype("float32")
        return out

    # ---- 학습: fold 안 시드 평균(원문의 SEED + 1000*s 파생 시드) ----

    def fit(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series
    ) -> np.ndarray:
        import torch
        from pytabkit import TabM_D_Classifier

        device = "cuda" if torch.cuda.is_available() else "cpu"
        extra: dict = {}
        if device == "cpu":
            # macOS에서 lightgbm 계열 libomp와 torch libomp가 같이 적재되면 CPU
            # 연산의 OpenMP 병렬이 충돌한다. pytabkit은 fit 안에서 스레드 수를
            # 물리 코어 수로 재설정하므로 n_threads=1을 명시해 단일 스레드로 돈다.
            extra["n_threads"] = 1
        self._fit_medians(X_tr)
        X_tr_t = self._transform(X_tr)
        X_va_t = self._transform(X_va)
        y_va_np = y_va.to_numpy(dtype="float64")

        self._models = []
        val_pred = np.zeros(len(X_va), dtype="float64")
        for s in range(self._n_seed_avg):
            model = TabM_D_Classifier(
                random_state=self._seed + 1000 * s,
                device=device,
                verbosity=0,
                **extra,
                **self._pytabkit,
            )
            model.fit(X_tr_t, y_tr, X_va_t, y_va)
            member_pred = model.predict_proba(X_va_t)[:, 1].astype("float64")
            val_pred += member_pred / self._n_seed_avg
            self._models.append(model)
            print(
                f"[tabm] member {s + 1}/{self._n_seed_avg} "
                f"valAUC={roc_auc_score(y_va_np, member_pred):.5f}",
                flush=True,
            )
        self._val = (X_va_t, y_va_np)
        self._val_auc = roc_auc_score(y_va_np, val_pred)
        print(f"[tabm] fold seed-avg valAUC={self._val_auc:.5f}", flush=True)
        return val_pred

    def _predict_transformed(self, X_t: pd.DataFrame) -> np.ndarray:
        pred = np.zeros(len(X_t), dtype="float64")
        for model in self._models:
            pred += model.predict_proba(X_t)[:, 1].astype("float64") / len(self._models)
        return pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_transformed(self._transform(X))

    # ---- 중요도: 검증 fold permutation(AUC 하락 폭)을 gain 축으로 (#97 규약) ----
    # 정밀도 파라미터(표본 크기·반복 수)는 계열 소유다. 시드 평균 예측 전체를 다시
    # 만들면 fold당 수백 회 추론이라, 검증 fold를 결정적으로 표본추출해 잰다.

    def importance(self) -> pd.DataFrame:
        X_va, y_va = self._val
        if len(X_va) > self._perm_sample:
            keep = np.random.default_rng(self._seed).choice(
                len(X_va), size=self._perm_sample, replace=False
            )
            keep.sort()
            X_va = X_va.iloc[keep].reset_index(drop=True)
            y_va = y_va[keep]
        base = roc_auc_score(y_va, self._predict_transformed(X_va))
        gains = []
        for j, col in enumerate(self._columns):
            drops = []
            for r in range(self._perm_repeats):
                rng = np.random.default_rng(self._seed * 10007 + j * 101 + r)
                X_p = X_va.copy()
                # take는 category dtype을 보존한다(numpy 왕복은 object로 퇴화).
                X_p[col] = X_va[col].take(rng.permutation(len(X_va))).set_axis(X_p.index)
                drops.append(base - roc_auc_score(y_va, self._predict_transformed(X_p)))
            gains.append(float(np.mean(drops)))
        return pd.DataFrame({"feature": self._columns, "gain": gains})
