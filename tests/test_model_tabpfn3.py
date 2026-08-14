"""TabPFN-3 adapter 테스트. (#102)

실제 gated 가중치를 읽지 않고 가짜 분류기로 전처리·청크·중요도 계약을 검증한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import model as model_mod
from pipeline import tabpfn3 as tabpfn3_mod
from pipeline.config import ModelConfig
from pipeline.tabpfn3 import TabPFN3Fold


class FakeTabPFN:
    def __init__(self) -> None:
        self.chunk_sizes: list[int] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FakeTabPFN:
        self.train = X.copy()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.chunk_sizes.append(len(X))
        logit = X["signal"].fillna(0).to_numpy(dtype="float64")
        prob = 1 / (1 + np.exp(-logit))
        return np.column_stack([1 - prob, prob])


def _data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(3)
    signal = rng.normal(size=180)
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=180),
            "kind": pd.Categorical(
                rng.choice(["a", "b", None], size=180), categories=["a", "b"]
            ),
        }
    )
    y = pd.Series((signal + rng.normal(scale=0.2, size=180) > 0).astype(int))
    return X, y


def test_tabpfn3_adapter_contract_chunking_and_importance(monkeypatch):
    made: list[FakeTabPFN] = []

    def fake_new(self: TabPFN3Fold) -> FakeTabPFN:
        model = FakeTabPFN()
        made.append(model)
        return model

    monkeypatch.setattr(TabPFN3Fold, "_new_classifier", fake_new)
    cfg = ModelConfig(
        kind="tabpfn3",
        params={"chunk_rows": 17, "perm_sample": 30, "perm_repeats": 1},
        fit={},
    )
    adapter = model_mod.create(cfg, seed=42)
    assert isinstance(adapter, model_mod.TabPFN3Adapter)

    X, y = _data()
    va_pred = adapter.fit(X.iloc[:120], y.iloc[:120], X.iloc[120:], y.iloc[120:])
    assert va_pred.shape == (60,)
    assert ((va_pred >= 0) & (va_pred <= 1)).all()
    assert made[0].train["kind"].isna().any()
    assert made[0].train["kind"].dropna().isin([0.0, 1.0]).all()

    test_pred = adapter.predict(X.iloc[:41])
    assert test_pred.shape == (41,)
    assert max(made[0].chunk_sizes) <= 17

    importance = adapter.importance()
    assert list(importance.columns) == ["feature", "gain"]
    assert list(importance["feature"]) == list(X.columns)
    signal_gain = importance.loc[importance["feature"] == "signal", "gain"].item()
    noise_gain = importance.loc[importance["feature"] == "noise", "gain"].item()
    assert signal_gain > noise_gain

    with pytest.raises(ValueError, match="초기 점수"):
        adapter.predict(X.iloc[:2], pd.Series([0.0, 0.0]))


def test_tabpfn3_retries_fit_once_with_memory_saving(monkeypatch):
    class FakeOOM(RuntimeError):
        pass

    class OOMTabPFN(FakeTabPFN):
        def fit(self, X: pd.DataFrame, y: pd.Series) -> FakeTabPFN:
            raise FakeOOM

    made: list[tuple[bool | str, FakeTabPFN]] = []

    def fake_new(self: TabPFN3Fold) -> FakeTabPFN:
        model = FakeTabPFN() if self._memory_saving_mode is True else OOMTabPFN()
        made.append((self._memory_saving_mode, model))
        return model

    monkeypatch.setattr(TabPFN3Fold, "_new_classifier", fake_new)
    monkeypatch.setattr(
        tabpfn3_mod, "_is_cuda_oom", lambda exc: isinstance(exc, FakeOOM)
    )
    monkeypatch.setattr(tabpfn3_mod, "_clear_cuda_cache", lambda: None)

    X, y = _data()
    fold = TabPFN3Fold({"chunk_rows": 20, "perm_sample": 20}, seed=42)
    pred = fold.fit(X.iloc[:120], y.iloc[:120], X.iloc[120:], y.iloc[120:])

    assert pred.shape == (60,)
    assert [mode for mode, _ in made] == ["auto", True]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"fit_mode": "low_memory"}, "fit_with_cache"),
        ({"chunk_rows": 0}, "chunk_rows"),
        ({"unknown": 1}, "모르는 params"),
    ],
)
def test_tabpfn3_rejects_invalid_params(params, message):
    with pytest.raises(ValueError, match=message):
        TabPFN3Fold(params, seed=42)
