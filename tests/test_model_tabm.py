"""tabm adapter 테스트. (#61)

- fit/predict/importance 계약(다른 adapter 스모크와 같은 축).
- 중앙값 대체가 학습 fold 통계만 쓰고 결측·미관측 입력이 안전하게 처리되는지.
- permutation importance가 시드로 결정적인지(#97 계열 무관 중요도).
소형 데이터 + 소형 모델로 CPU에서 수십 초 안에 돈다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline import model as model_mod
from pipeline.config import ModelConfig, load_config
from pipeline.features import ConstrainedImputeAux
from pipeline.plan import FeaturePlan

SEED = 7
REPO = Path(__file__).resolve().parents[1]

SMALL_PARAMS = {
    "tabm_k": 4,
    "d_embedding": 4,
    "batch_size": 64,
    "lr": 5e-3,
    "n_epochs": 12,
    "d_block": 32,
    "n_blocks": 2,
    "patience": 12,
    "n_seed_avg": 2,
    "perm_repeats": 2,
    "perm_sample": 200,
}


def _data(n: int = 320) -> tuple[pd.DataFrame, pd.Series]:
    """신호 수치 1열(결측 포함) + 결측 있는 범주 1열 + 잡음 수치 1열."""
    rng = np.random.default_rng(3)
    signal = rng.normal(size=n)
    cat = pd.Series(rng.choice(["Low", "High"], size=n)).where(rng.uniform(size=n) > 0.1)
    X = pd.DataFrame(
        {
            "v": pd.Series(signal).where(rng.uniform(size=n) > 0.1),
            "c": pd.Categorical(cat, categories=["High", "Low"]),
            "z": rng.normal(size=n),
        }
    )
    y = pd.Series((signal > 0).astype(int) ^ (rng.uniform(size=n) < 0.05).astype(int))
    return X, y


def _adapter(**overrides) -> model_mod.TabMAdapter:
    cfg = ModelConfig(kind="tabm", params=dict(SMALL_PARAMS, **overrides), fit={})
    adapter = model_mod.create(cfg, seed=SEED)
    assert isinstance(adapter, model_mod.TabMAdapter)
    return adapter


def test_tabm_adapter_contract_and_learning():
    X, y = _data()
    adapter = _adapter()
    va_pred = adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])
    assert va_pred.shape == (80,)
    assert ((va_pred >= 0) & (va_pred <= 1)).all()
    assert roc_auc_score(y.iloc[240:], va_pred) > 0.8

    test_pred = adapter.predict(X.iloc[:10])
    assert test_pred.shape == (10,)
    assert np.isfinite(test_pred).all()

    diagnostics = adapter.training_diagnostics()
    selected_epochs = [
        member["selected_epoch_count"] for member in diagnostics["members"]
    ]
    assert len(selected_epochs) == SMALL_PARAMS["n_seed_avg"]
    assert all(1 <= epoch <= SMALL_PARAMS["n_epochs"] for epoch in selected_epochs)

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


def test_tabm_handles_missing_and_inf_with_train_fold_medians():
    """수치 결측·inf는 학습 fold 중앙값으로만 채우고, 범주 결측도 유한 확률을 준다."""
    X, y = _data()
    adapter = _adapter()
    adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])
    unseen = pd.DataFrame(
        {
            "v": [np.nan, np.inf],
            "c": pd.Categorical([None, "Low"], categories=["High", "Low"]),
            "z": [0.0, np.nan],
        }
    )
    pred = adapter.predict(unseen)
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()


def test_tabm_rejects_unknown_params():
    X, y = _data(80)
    adapter = _adapter(no_such_param=1)
    with pytest.raises(ValueError, match="no_such_param"):
        adapter.fit(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:])


def test_tabm_full_fit_uses_all_rows_without_validation_split():
    X, y = _data(96)
    adapter = _adapter(n_seed_avg=1, n_epochs=2, patience=2)

    model_mod.fit_full(adapter, X, y, 2)

    prediction = adapter.predict(X.iloc[:8])
    assert prediction.shape == (8,)
    assert np.isfinite(prediction).all()
    diagnostics = adapter.training_diagnostics()
    assert diagnostics["full_fit"] is True
    assert diagnostics["epochs"] == 2


def test_tabm_muon_contract_and_learning():
    """optimizer=muon이 pytabkit 학습에서 작동하고 패치가 복원된다. (#196)"""
    import torch
    from pytabkit.models.alg_interfaces import tabm_interface

    original_adamw = torch.optim.AdamW
    original_groups = tabm_interface.make_parameter_groups

    X, y = _data()
    adapter = _adapter(optimizer="muon")
    va_pred = adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])
    assert va_pred.shape == (80,)
    assert roc_auc_score(y.iloc[240:], va_pred) > 0.8

    diagnostics = adapter.training_diagnostics()
    assert diagnostics["optimizer"] == "muon"
    # fit이 끝나면 pytabkit·torch 패치는 원본으로 돌아와야 한다.
    assert torch.optim.AdamW is original_adamw
    assert tabm_interface.make_parameter_groups is original_groups


def test_tabm_rejects_unknown_optimizer():
    X, y = _data(80)
    adapter = _adapter(optimizer="sgd")
    with pytest.raises(ValueError, match="optimizer"):
        adapter.fit(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:])


def test_tabm_muon_full_fit_uses_hybrid_optimizer():
    X, y = _data(96)
    adapter = _adapter(n_seed_avg=1, n_epochs=2, patience=2, optimizer="muon")

    model_mod.fit_full(adapter, X, y, 2)

    prediction = adapter.predict(X.iloc[:8])
    assert prediction.shape == (8,)
    assert np.isfinite(prediction).all()
    diagnostics = adapter.training_diagnostics()
    assert diagnostics["full_fit"] is True
    assert diagnostics["optimizer"] == "muon"


def test_tabm_recon_widths_config_is_exp065_widths_only_delta():
    baseline = load_config(REPO / "configs" / "exp065_tabm.yaml", "screen")
    challenger = load_config(
        REPO / "configs" / "exp137_tabm_recon_widths.yaml", "screen"
    )

    assert challenger.name == "exp137_tabm_recon_widths"
    assert challenger.data == baseline.data
    assert challenger.model == baseline.model
    assert challenger.initial_score == baseline.initial_score
    assert challenger.features.base == baseline.features.base
    assert challenger.features.categorical == baseline.features.categorical
    assert challenger.features.exclude == baseline.features.exclude
    assert len(challenger.features.providers) == len(baseline.features.providers)

    for baseline_provider, challenger_provider in zip(
        baseline.features.providers,
        challenger.features.providers,
        strict=True,
    ):
        if baseline_provider["kind"] == "constrained_impute_aux":
            assert baseline_provider["widths"] is False
            assert challenger_provider == {**baseline_provider, "widths": True}
        else:
            assert challenger_provider == baseline_provider

    baseline_aux = next(
        provider
        for provider in FeaturePlan.from_config(baseline.features).fold_fit_transformers()
        if isinstance(provider, ConstrainedImputeAux)
    )
    challenger_aux = next(
        provider
        for provider in FeaturePlan.from_config(challenger.features).fold_fit_transformers()
        if isinstance(provider, ConstrainedImputeAux)
    )
    baseline_columns = baseline_aux.columns()
    challenger_columns = challenger_aux.columns()
    assert [c for c in challenger_columns if c not in baseline_columns] == [
        "social_media_hours_recon_width",
        "gaming_hours_recon_width",
        "work_study_hours_recon_width",
    ]
    assert [c for c in baseline_columns if c not in challenger_columns] == []
