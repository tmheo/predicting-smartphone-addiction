"""값 가리기 증강의 분포 형태 교체 계약. (#360)

- `value_dropout_sampler` 기본값은 기존 경로이고, 명시해도 수치가 같다.
- `row_mask`는 기대 가림 셀 수를 기존 경로와 같게 두고 열 사이 상관만 바꾼다.
- 풀에 결측이 없는 열은 기증받을 형태가 없으므로 기존 경로에 그대로 남는다.
- `mask_pool='test'`는 목표값을 읽지 않는 전처리 기준 집합의 test 행만 쓴다.
소형 데이터 + 소형 모델로 CPU에서 몇 초 안에 돈다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import model as model_mod
from pipeline.config import ModelConfig

SEED = 7

SMALL_PARAMS = {
    "lookup_cols": ["v", "c"],
    "d_model": 16,
    "plr_k": 4,
    "layers": 1,
    "heads": 2,
    "epochs": 8,
    "batch_size": 64,
    "lr": 5e-3,
    "ema_decay": 0.7,
    "perm_repeats": 2,
}


@pytest.fixture
def lookup_transformer_module():
    from pipeline import lookup_transformer

    return lookup_transformer


def _data(n: int = 320) -> tuple[pd.DataFrame, pd.Series]:
    """결측 있는 정확값 1열·범주 1열·연속 1열 + 결측 없는 연속 1열."""
    rng = np.random.default_rng(3)
    values = rng.choice([1.5, 2.5, 3.5, 4.5], size=n)
    cat = pd.Series(rng.choice(["Low", "High"], size=n)).where(rng.uniform(size=n) > 0.1)
    z = pd.Series(rng.normal(size=n)).where(rng.uniform(size=n) > 0.1)
    X = pd.DataFrame(
        {
            "v": pd.Series(values).where(rng.uniform(size=n) > 0.1),
            "c": pd.Categorical(cat, categories=["High", "Low"]),
            "z": z,
            "w": rng.normal(size=n),  # 결측이 없으므로 기증 열이 아니다.
        }
    )
    y = pd.Series((values > 2.5).astype(int) ^ (rng.uniform(size=n) < 0.1).astype(int))
    return X, y


def _adapter(**overrides) -> model_mod.LookupTransformerAdapter:
    params = dict(SMALL_PARAMS)
    params.update(overrides)
    cfg = ModelConfig(kind="lookup_transformer", params=params, fit={})
    adapter = model_mod.create(cfg, seed=SEED)
    assert isinstance(adapter, model_mod.LookupTransformerAdapter)
    return adapter


def _fit_predict(adapter, X, y) -> np.ndarray:
    return adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])


def _member_diagnostics(adapter) -> dict:
    return adapter.entry_diagnostics().observations["fold_initialization_members"][0]


def test_default_sampler_is_the_existing_independent_path():
    """기본값은 셀별 독립 균등 Bernoulli이고, 명시해도 예측이 완전히 같다."""
    X, y = _data()
    default_adapter = _adapter()
    default_pred = _fit_predict(default_adapter, X, y)
    explicit_pred = _fit_predict(_adapter(value_dropout_sampler="independent"), X, y)
    assert np.array_equal(default_pred, explicit_pred)

    diagnostics = _member_diagnostics(default_adapter)
    assert diagnostics["value_dropout_sampler"] == "independent"
    # 기증 마스크 풀을 만들지 않으므로 풀 계보는 비어 있다.
    assert diagnostics["mask_pool"] is None
    assert diagnostics["mask_pool_missing_rate"] is None
    assert diagnostics["value_dropout_alpha"] is None


def test_row_mask_sampler_records_pool_lineage_and_alpha():
    """alpha는 풀의 셀 단위 평균 결측률로 기대 가림 셀 수를 value_dropout에 맞춘다."""
    X, y = _data()
    adapter = _adapter(value_dropout_sampler="row_mask")
    row_mask_pred = _fit_predict(adapter, X, y)
    diagnostics = _member_diagnostics(adapter)

    assert diagnostics["value_dropout_sampler"] == "row_mask"
    assert diagnostics["mask_pool"] == "fold_train"
    assert diagnostics["mask_pool_rows"] == 240
    # v, c, z만 결측을 가진다. 결측 없는 w는 기증 열이 아니라 기존 경로에 남는다.
    assert diagnostics["mask_pool_donor_columns"] == 3
    rate = diagnostics["mask_pool_missing_rate"]
    alpha = diagnostics["value_dropout_alpha"]
    assert 0.0 < rate < 1.0
    assert rate * alpha == pytest.approx(diagnostics["value_dropout"])

    # 형태를 바꾸는 표본기이므로 기존 경로와 같은 예측이 나오면 안 된다.
    assert not np.array_equal(row_mask_pred, _fit_predict(_adapter(), X, y))


def test_row_mask_sampler_can_draw_the_pool_from_test_rows():
    """mask_pool='test'는 목표값 비참조 기준 집합의 test 행 마스크를 쓴다."""
    X, y = _data()
    X_test = _data(n=180)[0]
    adapter = _adapter(value_dropout_sampler="row_mask", mask_pool="test")
    model_mod.set_dataset_reference(adapter, X, X_test)
    _fit_predict(adapter, X, y)

    diagnostics = _member_diagnostics(adapter)
    assert diagnostics["mask_pool"] == "test"
    assert diagnostics["mask_pool_rows"] == 180


def test_row_mask_sampler_requires_a_dataset_reference_for_the_test_pool():
    X, y = _data()
    adapter = _adapter(value_dropout_sampler="row_mask", mask_pool="test")
    with pytest.raises(ValueError, match="전처리 기준 집합"):
        _fit_predict(adapter, X, y)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"value_dropout_sampler": "row"}, "value_dropout_sampler"),
        ({"value_dropout_sampler": "row_mask", "mask_pool": "train"}, "mask_pool"),
        ({"mask_pool": "test"}, "row_mask"),
    ],
)
def test_sampler_settings_are_validated(overrides, message):
    X, y = _data()
    with pytest.raises(ValueError, match=message):
        _fit_predict(_adapter(**overrides), X, y)


def test_sampler_keeps_the_expected_hidden_cell_count_and_adds_correlation(
    lookup_transformer_module,
):
    """같은 기대 가림 셀 수에서 독립 경로는 열 상관이 없고 row_mask는 상관을 옮긴다."""
    import torch

    torch.manual_seed(0)
    rate = 0.10
    rows, cols, batch = 4000, 5, 20000
    # 두 열이 함께 비고 두 열이 거의 안 비고 한 열은 아예 안 비는 풀:
    # 실측 결측의 상관과 두꺼운 꼬리, 그리고 기증 열이 아닌 열이 함께 있다.
    together = torch.rand(rows, 1) < 0.2
    pool = torch.cat(
        [
            together.expand(rows, 2),
            torch.rand(rows, 2) < 0.02,
            torch.zeros(rows, 1, dtype=torch.bool),
        ],
        dim=1,
    )
    donor_columns = pool.any(dim=0)
    assert donor_columns.tolist() == [True, True, True, True, False]
    pool_rate = float(pool[:, donor_columns].to(torch.float32).mean())
    alpha = min(1.0, rate / pool_rate)
    donor_columns = donor_columns.reshape(1, -1)

    independent = lookup_transformer_module._value_dropout_mask(
        (batch, cols),
        device="cpu",
        rate=rate,
        pool=None,
        alpha=None,
        donor_columns=None,
    ).to(torch.float32)
    row_mask = lookup_transformer_module._value_dropout_mask(
        (batch, cols),
        device="cpu",
        rate=rate,
        pool=pool,
        alpha=alpha,
        donor_columns=donor_columns,
    ).to(torch.float32)

    # 기대 가림 셀 수가 열마다 같다: 기증 열 전체도, 기증 열이 아닌 열도 rate다.
    assert float(independent.mean()) == pytest.approx(rate, abs=0.005)
    assert float(row_mask.mean()) == pytest.approx(rate, abs=0.005)
    assert float(row_mask[:, :4].mean()) == pytest.approx(rate, abs=0.005)
    assert float(row_mask[:, 4].mean()) == pytest.approx(rate, abs=0.01)

    def pair_correlation(drop: torch.Tensor) -> float:
        return float(np.corrcoef(drop[:, 0].numpy(), drop[:, 1].numpy())[0, 1])

    assert abs(pair_correlation(independent)) < 0.05
    assert pair_correlation(row_mask) > 0.3

    # 꼬리도 두꺼워진다: 한 행에서 3개 이상 가려지는 비율.
    def heavy_tail(drop: torch.Tensor) -> float:
        return float((drop.sum(dim=1) >= 3).to(torch.float32).mean())

    assert heavy_tail(row_mask) > heavy_tail(independent)
