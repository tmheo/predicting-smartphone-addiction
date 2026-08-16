"""AMFormer 논문 기반 독립 구현의 adapter와 수치 경계 시험. (#144)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline import model as model_mod
from pipeline.config import ModelConfig


@pytest.fixture
def amformer_module():
    """시험 수집 중 torch를 불러와 XGBoost의 OpenMP와 충돌시키지 않는다."""
    from pipeline import amformer

    # XGBoost가 먼저 적재된 전체 시험에서도 PyTorch가 새 OpenMP 작업군을 만들지 않는다.
    amformer.torch.set_num_threads(1)
    return amformer


def _data(n: int = 240) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(144)
    signal = rng.normal(size=n)
    index = np.arange(3000, 3000 + n)
    category = pd.Series(np.where(signal > 0, "high", "low"), index=index)
    category.iloc[::19] = None
    X = pd.DataFrame(
        {
            "signal": pd.Series(signal, index=index).where(np.arange(n) % 17 != 0),
            "noise": pd.Series(rng.normal(size=n), index=index),
            "category": pd.Categorical(category, categories=["high", "low"]),
        },
        index=index,
    )
    y = pd.Series((signal > 0).astype(int), index=index)
    return X, y


def _adapter(**overrides) -> model_mod.AMFormerAdapter:
    params = {
        "missing_indicator_cols": ["signal"],
        "layers": 2,
        "d_model": 128,
        "heads": 8,
        "groups": [4, 4],
        "prod_num_per_group": [2, 2],
        "cluster": True,
        "target_mode": "mix",
        "token_descent": False,
        "use_prod": True,
        "use_cls_token": True,
        "attention_dropout": 0.0,
        "dropout": 0.0,
        "epochs": 10,
        "patience": 4,
        "batch_size": 32,
        "eval_batch_size": 128,
        "lr": 0.003,
        "weight_decay": 1e-5,
        "perm_sample": 32,
        "perm_repeats": 1,
        "device": "cpu",
    }
    params.update(overrides)
    adapter = model_mod.create(ModelConfig(kind="amformer", params=params, fit={}), seed=7)
    assert isinstance(adapter, model_mod.AMFormerAdapter)
    return adapter


def test_amformer_paper_equation_shapes_and_finite_backward(amformer_module):
    import torch

    encoder = amformer_module._FoldEncoder(["signal"])
    X, _ = _data(24)
    encoder.fit(X)
    numeric, categorical = encoder.transform(X)
    torch.manual_seed(3)
    network = amformer_module._AMFormer(
        token_count=4,
        numeric_positions=encoder.numeric_token_positions,
        categorical_positions=encoder.categorical_token_positions,
        categorical_cardinalities=encoder.categorical_cardinalities,
        d_model=16,
        heads=4,
        prod_top_ks=[2, 2],
        attention_dropout=0.0,
        dropout=0.0,
    )

    output = network(numeric, categorical)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output, torch.zeros(len(output))
    )
    loss.backward()

    assert output.shape == (24,)
    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )


def test_amformer_adapter_contract_importance_and_diagnostics():
    X, y = _data()
    adapter = _adapter()
    validation_pred = adapter.fit(
        X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:]
    )

    assert validation_pred.shape == (60,)
    assert np.isfinite(validation_pred).all()
    assert ((validation_pred >= 0) & (validation_pred <= 1)).all()
    assert roc_auc_score(y.iloc[180:], validation_pred) > 0.85

    prediction_input = X.iloc[180:190].copy()
    prediction_input.loc[prediction_input.index[0], "category"] = None
    prediction = adapter.predict(prediction_input)
    assert prediction.shape == (10,)
    assert np.isfinite(prediction).all()

    importance = adapter.importance()
    assert list(importance.columns) == ["feature", "gain"]
    assert importance["feature"].tolist() == list(X.columns)
    assert np.isfinite(importance["gain"]).all()

    diagnostics = adapter.entry_diagnostics()
    assert all(diagnostics.assertions.values())
    assert diagnostics.observations["token_count"] == 4
    assert diagnostics.observations["requested_d_model"] == 128
    assert diagnostics.observations["used_d_model"] == 128
    assert diagnostics.observations["probe"]["output_shape"] == [32]
    assert diagnostics.observations["importance_rows"] == 32


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"groups": [3, 3]}, "실제 token 수"),
        ({"target_mode": "prompt"}, "target_mode=mix"),
        ({"token_descent": True}, "token_descent=false"),
        ({"unknown": 1}, "모르는 params"),
    ],
)
def test_amformer_rejects_nonbaseline_or_inconsistent_params(override, message):
    X, y = _data()
    adapter = _adapter(**override)
    with pytest.raises((ValueError, RuntimeError), match=message):
        adapter.fit(X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:])
