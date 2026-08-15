"""lookup_transformer adapter 테스트. (#58)

- fit/predict/importance 계약(다른 adapter 스모크와 같은 축).
- 어휘·분위 fit이 학습 fold 전용이고 미관측 값·결측이 안전하게 처리되는지.
- permutation importance가 시드로 결정적인지(#97 계열 무관 중요도).
소형 데이터 + 소형 모델로 CPU에서 몇 초 안에 돈다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline import lookup_transformer
from pipeline import model as model_mod
from pipeline.config import ModelConfig

SEED = 7

SMALL_PARAMS = {
    "lookup_cols": ["v", "c"],
    "d_model": 16,
    "plr_k": 4,
    "layers": 1,
    "heads": 2,
    "epochs": 16,
    "batch_size": 64,
    "lr": 5e-3,
    # 소형 학습(수십 step)에서는 0.999 EMA가 초기 가중치에 머물러 감쇠를 낮춘다.
    "ema_decay": 0.7,
    "perm_repeats": 2,
}


def _data(n: int = 320) -> tuple[pd.DataFrame, pd.Series]:
    """정확값 격자 1열 + 결측 있는 범주 1열 + 연속(PLR 전용) 1열."""
    rng = np.random.default_rng(3)
    values = rng.choice([1.5, 2.5, 3.5, 4.5], size=n)
    cat = pd.Series(rng.choice(["Low", "High"], size=n)).where(rng.uniform(size=n) > 0.1)
    X = pd.DataFrame(
        {
            "v": values,
            "c": pd.Categorical(cat, categories=["High", "Low"]),
            "z": rng.normal(size=n),
        }
    )
    y = pd.Series((values > 2.5).astype(int) ^ (rng.uniform(size=n) < 0.1).astype(int))
    return X, y


def _adapter() -> model_mod.LookupTransformerAdapter:
    cfg = ModelConfig(kind="lookup_transformer", params=dict(SMALL_PARAMS), fit={})
    adapter = model_mod.create(cfg, seed=SEED)
    assert isinstance(adapter, model_mod.LookupTransformerAdapter)
    return adapter


def test_lookup_transformer_adapter_contract_and_learning():
    X, y = _data()
    adapter = _adapter()
    va_pred = adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])
    assert va_pred.shape == (80,)
    assert ((va_pred >= 0) & (va_pred <= 1)).all()
    # 정확값 lookup으로 값별 라벨 구조를 학습한다.
    assert roc_auc_score(y.iloc[240:], va_pred) > 0.8

    test_pred = adapter.predict(X.iloc[:10])
    assert test_pred.shape == (10,)
    assert np.isfinite(test_pred).all()

    imp = adapter.importance()
    assert list(imp.columns) == ["feature", "gain"]
    assert list(imp["feature"]) == list(X.columns)
    # 신호 컬럼의 permutation 하락 폭이 잡음 컬럼보다 크다.
    gain = imp.set_index("feature")["gain"]
    assert gain["v"] > gain["z"]

    # importance는 시드로 결정적이다(#97: 환산은 실행 시드로 결정적).
    imp2 = adapter.importance()
    assert np.allclose(imp["gain"], imp2["gain"])

    with pytest.raises(ValueError, match="초기 점수"):
        adapter.fit(
            X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:],
            pd.Series(np.zeros(240)), pd.Series(np.zeros(80)),
        )


def test_lookup_transformer_handles_unseen_values_and_missing():
    """어휘는 학습 fold 값 집합만 쓰고, 미관측 값(UNK)과 결측(NA)에도 유한 확률을 준다."""
    X, y = _data()
    adapter = _adapter()
    adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])
    unseen = pd.DataFrame(
        {
            "v": [9.9, np.nan],
            "c": pd.Categorical([None, "Low"], categories=["High", "Low"]),
            "z": [0.0, np.nan],
        }
    )
    pred = adapter.predict(unseen)
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()


def test_lookup_transformer_rejects_high_cardinality_lookup_col():
    """연속 컬럼을 lookup_cols에 넣으면 카디널리티 가드가 명확히 거부한다."""
    X, y = _data()
    params = dict(SMALL_PARAMS, lookup_cols=["v", "c", "z"], lookup_max_card=50)
    cfg = ModelConfig(kind="lookup_transformer", params=params, fit={})
    adapter = model_mod.create(cfg, seed=SEED)
    with pytest.raises(ValueError, match="lookup_max_card"):
        adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])


def test_lookup_transformer_rejects_unknown_params():
    cfg = ModelConfig(
        kind="lookup_transformer",
        params=dict(SMALL_PARAMS, no_such_param=1),
        fit={},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    X, y = _data(80)
    with pytest.raises(ValueError, match="no_such_param"):
        adapter.fit(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:])


def test_lookup_transformer_fold_initialization_average_derives_seeds(monkeypatch):
    """파이프라인 시드에 offset을 더한 구성원 예측을 fold 안에서 평균한다."""
    created_seeds = []

    class FakeMember:
        def __init__(self, params, seed, device=None, init_barrier=None):
            assert params == {"lookup_cols": ["v"], "perm_repeats": 1}
            self._seed = seed
            self._device = device or "cpu"
            created_seeds.append(seed)

        def fit(self, X_tr, y_tr, X_va, y_va):
            return np.full(len(X_va), self._seed / 10_000, dtype="float64")

        def predict(self, X):
            return np.full(len(X), self._seed / 10_000, dtype="float64")

    monkeypatch.setattr(lookup_transformer, "_LookupTransformerMember", FakeMember)
    fold = lookup_transformer.LookupTransformerFold(
        {
            "lookup_cols": ["v"],
            "perm_repeats": 1,
            "fold_seed_offsets": [0, 1000, 2000],
        },
        seed=SEED,
    )
    X = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 1, 0, 1])

    val_pred = fold.fit(X, y, X, y)

    assert created_seeds == [7, 1007, 2007]
    assert np.allclose(val_pred, 0.1007)
    assert np.allclose(fold.predict(X.iloc[:2]), 0.1007)
    importance = fold.importance()
    assert importance.to_dict("records") == [{"feature": "v", "gain": 0.0}]


@pytest.mark.parametrize("offsets", [[], [0, 0], [0, 1.5], "0,1000"])
def test_lookup_transformer_rejects_invalid_fold_seed_offsets(offsets):
    with pytest.raises(ValueError, match="fold_seed_offsets"):
        lookup_transformer.LookupTransformerFold(
            {"lookup_cols": ["v"], "fold_seed_offsets": offsets}, seed=SEED
        )


def test_lookup_transformer_fold_gpu_assignment(monkeypatch):
    monkeypatch.setenv("PIPELINE_FOLD_GPUS", "0,2,1")
    monkeypatch.setattr(lookup_transformer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(lookup_transformer.torch.cuda, "device_count", lambda: 3)

    assert lookup_transformer.LookupTransformerFold._parallel_devices(3) == [
        "cuda:0",
        "cuda:2",
        "cuda:1",
    ]


@pytest.mark.parametrize("gpu_ids", ["0,0,1", "0,1", "0,1,nope"])
def test_lookup_transformer_rejects_invalid_fold_gpu_assignment(monkeypatch, gpu_ids):
    monkeypatch.setenv("PIPELINE_FOLD_GPUS", gpu_ids)
    monkeypatch.setattr(lookup_transformer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(lookup_transformer.torch.cuda, "device_count", lambda: 3)

    with pytest.raises(ValueError, match="PIPELINE_FOLD_GPUS"):
        lookup_transformer.LookupTransformerFold._parallel_devices(3)
