"""lookup_transformer adapter 테스트. (#58)

- fit/predict/importance 계약(다른 adapter 스모크와 같은 축).
- 어휘·분위 fit이 학습 fold 전용이고 미관측 값·결측이 안전하게 처리되는지.
- permutation importance가 시드로 결정적인지(#97 계열 무관 중요도).
소형 데이터 + 소형 모델로 CPU에서 몇 초 안에 돈다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline import model as model_mod
from pipeline.config import ModelConfig, load_config
from pipeline.training_length import (
    ZERO_BASED_POSITION,
    observe_declaration,
)

SEED = 7
REPO = Path(__file__).resolve().parents[1]

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


def _cuda_device_count_in_subprocess() -> int:
    """시험 수집 중 torch를 불러와 XGBoost의 OpenMP와 충돌시키지 않는다."""
    completed = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return 0
    return int(completed.stdout.strip())


@pytest.fixture
def lookup_transformer_module():
    from pipeline import lookup_transformer

    return lookup_transformer


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

    diagnostics = adapter.entry_diagnostics().observations
    member = diagnostics["fold_initialization_members"][0]
    assert member["optimizer"] == "adamw"
    assert member["lr_schedule"] == "one_cycle"
    assert member["best_epoch"] is not None
    assert member["end_epoch"] >= member["best_epoch"]
    assert member["evaluations"]
    assert set(member["evaluations"][0]) == {
        "epoch",
        "learning_rate",
        "learning_rate_after_validation",
        "beta1",
        "training_loss",
        "validation_auc",
        "best_epoch",
        "best_validation_auc",
        "gradient_norm_mean",
        "gradient_clip_fraction",
    }

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


def test_lookup_transformer_train_test_reference_fits_vocab_and_quantiles(
    lookup_transformer_module,
):
    """train_test 범위는 검증과 test의 목표값 비참조 값까지 전처리에 포함한다."""
    member = lookup_transformer_module._LookupTransformerMember(
        {
            "lookup_cols": ["v"],
            "preprocessing_scope": "train_test",
            "validation_selection": "final",
            "perm_repeats": 1,
        },
        seed=SEED,
    )
    X_tr = pd.DataFrame({"v": [1.0, 2.0]})
    X_all_train = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    X_test = pd.DataFrame({"v": [4.0]})

    with pytest.raises(ValueError, match="전처리 기준 집합"):
        member._fit_specs(X_tr)

    member.set_dataset_reference(X_all_train, X_test)
    member._fit_specs(X_tr)

    _, vocabulary, quantiles = member._specs["v"]
    assert vocabulary == [1.0, 2.0, 3.0, 4.0]
    assert quantiles.n_quantiles_ == 4
    assert quantiles.quantiles_[-1, 0] == pytest.approx(4.0)
    encoded_ids, _, _ = member._encode(pd.DataFrame({"v": [3.0, 4.0, 9.0]}))
    assert encoded_ids[:, 0].tolist() == [3, 4, 5]


def test_lookup_transformer_final_selection_runs_fixed_epoch_schedule():
    X, y = _data(96)
    params = dict(
        SMALL_PARAMS,
        epochs=7,
        patience=1,
        perm_repeats=1,
        preprocessing_scope="train_test",
        validation_selection="final",
    )
    adapter = model_mod.create(
        ModelConfig(kind="lookup_transformer", params=params, fit={}), seed=SEED
    )
    model_mod.set_dataset_reference(adapter, X, X.iloc[:12])

    adapter.fit(X.iloc[:72], y.iloc[:72], X.iloc[72:], y.iloc[72:])

    member = adapter.training_diagnostics()["fold_initialization_members"][0]
    assert member["preprocessing_scope"] == "train_test"
    assert member["validation_selection"] == "final"
    assert member["end_epoch"] == 6
    assert member["best_epoch"] == 6


def test_lookup_transformer_full_fit_uses_fixed_epochs():
    X, y = _data(96)
    params = dict(SMALL_PARAMS, epochs=99, patience=2, perm_repeats=1)
    adapter = model_mod.create(
        ModelConfig(kind="lookup_transformer", params=params, fit={}), seed=SEED
    )

    model_mod.fit_full(adapter, X, y, 2)

    prediction = adapter.predict(X.iloc[:8])
    assert prediction.shape == (8,)
    assert np.isfinite(prediction).all()
    member = adapter.training_diagnostics()["fold_initialization_members"][0]
    assert member["full_fit"] is True
    assert member["best_epoch"] == 1
    assert member["end_epoch"] == 1


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


@pytest.mark.parametrize(
    ("param", "value", "message"),
    [
        ("optimizer", "sgd", "optimizer"),
        ("lr_schedule", "exponential", "lr_schedule"),
        ("lr", 0.0, "양수"),
        ("muon_lr_multiplier", 0.0, "muon_lr_multiplier는 양수"),
        ("muon_lr_multiplier", 2.0, "optimizer가 muon일 때만"),
    ],
)
def test_lookup_transformer_rejects_invalid_training_configuration(
    param, value, message
):
    params = dict(SMALL_PARAMS, **{param: value})
    cfg = ModelConfig(kind="lookup_transformer", params=params, fit={})
    adapter = model_mod.create(cfg, seed=SEED)
    X, y = _data(80)
    with pytest.raises(ValueError, match=message):
        adapter.fit(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:])


@pytest.mark.parametrize("name", ["adamw", "radam", "nadam"])
def test_lookup_transformer_optimizer_preserves_weight_decay_groups(
    name, lookup_transformer_module
):
    parameter = lookup_transformer_module.torch.nn.Parameter(
        lookup_transformer_module.torch.tensor(1.0)
    )
    optimizer = lookup_transformer_module._create_optimizer(
        name,
        [{"params": [parameter], "weight_decay": 3e-4}],
        2e-3,
    )

    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(3e-4)
    if name in {"radam", "nadam"}:
        assert optimizer.defaults["decoupled_weight_decay"] is True


@pytest.mark.parametrize(
    "name", ["warmup_cosine", "warmup_linear", "warmup_constant"]
)
def test_lookup_transformer_warmup_schedules_start_at_one_twenty_fifth(
    name, lookup_transformer_module
):
    parameter = lookup_transformer_module.torch.nn.Parameter(
        lookup_transformer_module.torch.tensor(1.0)
    )
    optimizer = lookup_transformer_module.torch.optim.AdamW([parameter], lr=2e-3)
    schedule = lookup_transformer_module._LearningRateController(
        optimizer,
        name=name,
        max_lr=2e-3,
        total_steps=20,
    )

    assert schedule.learning_rate == pytest.approx(2e-3 / 25)
    for _ in range(schedule.warmup_steps):
        optimizer.step()
        schedule.step_batch()
    assert schedule.learning_rate == pytest.approx(2e-3)

    for _ in range(20 - schedule.warmup_steps):
        optimizer.step()
        schedule.step_batch()
    expected = 2e-3 if name == "warmup_constant" else 2e-3 / 250_000
    assert schedule.learning_rate == pytest.approx(expected)


def test_lookup_transformer_plateau_reduces_after_two_bad_evaluations(
    lookup_transformer_module,
):
    parameter = lookup_transformer_module.torch.nn.Parameter(
        lookup_transformer_module.torch.tensor(1.0)
    )
    optimizer = lookup_transformer_module.torch.optim.AdamW([parameter], lr=2e-3)
    schedule = lookup_transformer_module._LearningRateController(
        optimizer,
        name="warmup_plateau",
        max_lr=2e-3,
        total_steps=20,
    )
    for _ in range(schedule.warmup_steps):
        optimizer.step()
        schedule.step_batch()

    schedule.step_validation(0.8)
    schedule.step_validation(0.7)
    assert schedule.learning_rate == pytest.approx(2e-3)
    schedule.step_validation(0.6)
    assert schedule.learning_rate == pytest.approx(2e-3 * 0.3)


def test_lookup_transformer_one_cycle_can_disable_momentum_cycle(
    lookup_transformer_module,
):
    betas = []
    for name in ("one_cycle", "one_cycle_fixed_momentum"):
        parameter = lookup_transformer_module.torch.nn.Parameter(
            lookup_transformer_module.torch.tensor(1.0)
        )
        optimizer = lookup_transformer_module.torch.optim.AdamW([parameter], lr=2e-3)
        schedule = lookup_transformer_module._LearningRateController(
            optimizer,
            name=name,
            max_lr=2e-3,
            total_steps=20,
        )
        before = schedule.beta1
        optimizer.step()
        schedule.step_batch()
        betas.append((before, schedule.beta1))

    assert betas[0][0] != betas[0][1]
    assert betas[1] == pytest.approx((0.9, 0.9))


def test_lookup_transformer_fold_initialization_average_derives_seeds(
    monkeypatch, lookup_transformer_module
):
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

    monkeypatch.setattr(lookup_transformer_module, "_LookupTransformerMember", FakeMember)
    fold = lookup_transformer_module.LookupTransformerFold(
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
def test_lookup_transformer_rejects_invalid_fold_seed_offsets(
    offsets, lookup_transformer_module
):
    with pytest.raises(ValueError, match="fold_seed_offsets"):
        lookup_transformer_module.LookupTransformerFold(
            {"lookup_cols": ["v"], "fold_seed_offsets": offsets}, seed=SEED
        )


def test_lookup_transformer_fold_gpu_assignment(monkeypatch, lookup_transformer_module):
    monkeypatch.setenv("PIPELINE_FOLD_GPUS", "0,2,1")
    monkeypatch.setattr(lookup_transformer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(lookup_transformer_module.torch.cuda, "device_count", lambda: 3)

    assert lookup_transformer_module.LookupTransformerFold._parallel_devices(3) == [
        "cuda:0",
        "cuda:2",
        "cuda:1",
    ]


@pytest.mark.parametrize("gpu_ids", ["0,0,1", "0,1", "0,1,nope"])
def test_lookup_transformer_rejects_invalid_fold_gpu_assignment(
    monkeypatch, gpu_ids, lookup_transformer_module
):
    monkeypatch.setenv("PIPELINE_FOLD_GPUS", gpu_ids)
    monkeypatch.setattr(lookup_transformer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(lookup_transformer_module.torch.cuda, "device_count", lambda: 3)

    with pytest.raises(ValueError, match="PIPELINE_FOLD_GPUS"):
        lookup_transformer_module.LookupTransformerFold._parallel_devices(3)


@pytest.mark.skipif(
    _cuda_device_count_in_subprocess() < 3,
    reason="실제 fold 구성원 병렬 검사는 CUDA GPU 3개가 필요하다.",
)
def test_lookup_transformer_trains_fold_members_on_three_gpus(monkeypatch):
    """원격 실행 전 검사: 초기화가 다른 세 구성원을 GPU별로 실제 학습한다."""
    monkeypatch.setenv("PIPELINE_FOLD_GPUS", "0,1,2")
    X, y = _data(96)
    params = dict(
        SMALL_PARAMS,
        epochs=2,
        patience=2,
        perm_repeats=1,
        fold_seed_offsets=[0, 1000, 2000],
    )
    cfg = ModelConfig(kind="lookup_transformer", params=params, fit={})
    adapter = model_mod.create(cfg, seed=SEED)

    val_pred = adapter.fit(X.iloc[:72], y.iloc[:72], X.iloc[72:], y.iloc[72:])
    test_pred = adapter.predict(X.iloc[:12])

    assert val_pred.shape == (24,)
    assert test_pred.shape == (12,)
    assert np.isfinite(val_pred).all()
    assert np.isfinite(test_pred).all()


def test_lookup_transformer_muon_parameter_names_select_hidden_matrices(
    lookup_transformer_module,
):
    """Muon 대상은 encoder 행렬과 head 은닉 Linear뿐이다. (#196)"""
    model = lookup_transformer_module._LookupTransformer(10, 3, 16, 4, 1, 2, 0.1)
    names = lookup_transformer_module._muon_parameter_names(model)

    parameters = dict(model.named_parameters())
    assert names, "encoder 행렬이 비어 있으면 안 된다."
    assert all(parameters[name].ndim == 2 for name in names)
    assert "head.1.weight" in names
    assert any(name.startswith("tr.") for name in names)
    # embedding·PLR·출력층은 AdamW로 남는다.
    assert "emb.weight" not in names
    assert "head.4.weight" not in names
    assert not any(name.startswith("plr.") for name in names)


def test_lookup_bivariate_recon_widths_config_is_exp131_widths_only_delta():
    from pipeline.features import ConstrainedImputeAux
    from pipeline.plan import FeaturePlan

    baseline = load_config(
        REPO / "configs" / "exp131_lookup_bivariate_plr5.yaml", "screen"
    )
    challenger = load_config(
        REPO / "configs" / "exp140_lookup_bivariate_plr5_recon_widths.yaml",
        "screen",
    )

    assert challenger.name == "exp140_lookup_bivariate_plr5_recon_widths"
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


def test_lookup_transformer_muon_optimizer_shares_groups_with_delegates(
    lookup_transformer_module,
):
    """혼성 optimizer의 자식이 부모의 그룹 dict를 공유해 lr 일정이 전달된다. (#196)"""
    from pipeline.muon import MuonWithAdamW

    model = lookup_transformer_module._LookupTransformer(10, 3, 16, 4, 1, 2, 0.1)
    names = lookup_transformer_module._muon_parameter_names(model)
    emb = [p for n, p in model.named_parameters() if n.startswith("emb")]
    rest = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("emb") and n not in names
    ]
    muon = [p for n, p in model.named_parameters() if n in names]

    optimizer = lookup_transformer_module._create_optimizer(
        "muon",
        [
            {"params": rest, "weight_decay": 1e-5, "algorithm": "adamw"},
            {"params": emb, "weight_decay": 3e-4, "algorithm": "adamw"},
            {"params": muon, "weight_decay": 1e-5, "algorithm": "muon"},
        ],
        2e-3,
    )

    assert isinstance(optimizer, MuonWithAdamW)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(3e-4)
    for group in optimizer.param_groups:
        group["lr"] = 1.23e-4
    for delegate in optimizer._delegates:
        assert all(g["lr"] == pytest.approx(1.23e-4) for g in delegate.param_groups)


def test_lookup_transformer_muon_learns_on_cpu():
    """optimizer=muon으로도 fit/predict 계약과 학습이 유지된다. (#196)"""
    cfg = ModelConfig(
        kind="lookup_transformer",
        params=dict(SMALL_PARAMS, optimizer="muon"),
        fit={},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    X, y = _data()
    va_pred = adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])
    assert va_pred.shape == (80,)
    assert roc_auc_score(y.iloc[240:], va_pred) > 0.8

    member = adapter.entry_diagnostics().observations["fold_initialization_members"][0]
    assert member["optimizer"] == "muon"


def _three_group_optimizer(lookup_transformer_module):
    """champion과 같은 그룹 3개(rest·emb·muon) 구조의 최소 optimizer."""
    torch = lookup_transformer_module.torch
    groups = [
        {"params": [torch.nn.Parameter(torch.tensor(1.0))]},
        {"params": [torch.nn.Parameter(torch.tensor(1.0))]},
        {"params": [torch.nn.Parameter(torch.tensor(1.0))]},
    ]
    return torch.optim.AdamW(groups, lr=2e-3)


@pytest.mark.parametrize("name", ["one_cycle", "warmup_cosine"])
def test_lookup_transformer_group_lr_scales_hold_through_the_schedule(
    name, lookup_transformer_module
):
    """그룹 배율은 일정 전 구간에서 그룹별 학습률 비로 유지된다. (#385)"""
    optimizer = _three_group_optimizer(lookup_transformer_module)
    schedule = lookup_transformer_module._LearningRateController(
        optimizer,
        name=name,
        max_lr=2e-3,
        total_steps=20,
        group_lr_scales=[1.0, 1.0, 4.0],
    )

    observed = []
    for _ in range(20):
        observed.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        schedule.step_batch()

    for first, second, third in observed:
        assert second == pytest.approx(first)
        assert third == pytest.approx(first * 4.0)
    # 배율이 붙은 그룹도 정점에서 자기 최고치에 도달한다.
    assert max(row[2] for row in observed) == pytest.approx(2e-3 * 4.0, rel=1e-3)
    # 진단이 읽는 축(param_groups[0])은 배율의 영향을 받지 않는다.
    assert max(row[0] for row in observed) == pytest.approx(2e-3, rel=1e-3)


def test_lookup_transformer_group_lr_scales_default_to_one(lookup_transformer_module):
    """배율을 주지 않으면 모든 그룹이 같은 학습률을 쓴다(#385 이전과 동일)."""
    optimizer = _three_group_optimizer(lookup_transformer_module)
    schedule = lookup_transformer_module._LearningRateController(
        optimizer,
        name="one_cycle",
        max_lr=2e-3,
        total_steps=20,
    )
    assert schedule.group_lr_scales == [1.0, 1.0, 1.0]
    for _ in range(5):
        optimizer.step()
        schedule.step_batch()
    lrs = [group["lr"] for group in optimizer.param_groups]
    assert lrs[1] == pytest.approx(lrs[0]) and lrs[2] == pytest.approx(lrs[0])


@pytest.mark.parametrize(
    ("scales", "message"),
    [([1.0, 1.0], "param_groups 수와 다르다"), ([1.0, 1.0, 0.0], "모두 양수")],
)
def test_lookup_transformer_group_lr_scales_reject_invalid_input(
    scales, message, lookup_transformer_module
):
    optimizer = _three_group_optimizer(lookup_transformer_module)
    with pytest.raises(ValueError, match=message):
        lookup_transformer_module._LearningRateController(
            optimizer,
            name="one_cycle",
            max_lr=2e-3,
            total_steps=20,
            group_lr_scales=scales,
        )


def test_lookup_transformer_muon_group_lr_scale_reaches_the_muon_delegate(
    lookup_transformer_module,
):
    """배율이 Muon 자식 optimizer가 실제로 읽는 그룹 학습률까지 도달한다. (#385)"""
    model = lookup_transformer_module._LookupTransformer(10, 3, 16, 4, 1, 2, 0.1)
    names = lookup_transformer_module._muon_parameter_names(model)
    emb = [p for n, p in model.named_parameters() if n.startswith("emb")]
    rest = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("emb") and n not in names
    ]
    muon = [p for n, p in model.named_parameters() if n in names]
    optimizer = lookup_transformer_module._create_optimizer(
        "muon",
        [
            {"params": rest, "weight_decay": 1e-5, "algorithm": "adamw"},
            {"params": emb, "weight_decay": 3e-4, "algorithm": "adamw"},
            {"params": muon, "weight_decay": 1e-5, "algorithm": "muon"},
        ],
        2e-3,
    )
    schedule = lookup_transformer_module._LearningRateController(
        optimizer,
        name="one_cycle",
        max_lr=2e-3,
        total_steps=20,
        group_lr_scales=[1.0, 1.0, 4.0],
    )
    for _ in range(5):
        optimizer.step()
        schedule.step_batch()

    from pipeline.muon import ALGORITHM_KEY

    delegate_muon_groups = [
        group
        for delegate in optimizer._delegates
        for group in delegate.param_groups
        if group.get(ALGORITHM_KEY) == "muon"
    ]
    assert delegate_muon_groups
    shared = optimizer.param_groups[0]["lr"]
    assert all(
        group["lr"] == pytest.approx(shared * 4.0) for group in delegate_muon_groups
    )


def test_lookup_transformer_muon_lr_multiplier_records_group_operating_point():
    """Muon 그룹 전용 학습률이 학습을 유지하고 진단에 그대로 남는다. (#385)"""
    cfg = ModelConfig(
        kind="lookup_transformer",
        params=dict(SMALL_PARAMS, optimizer="muon", muon_lr_multiplier=2.0),
        fit={},
    )
    adapter = model_mod.create(cfg, seed=SEED)
    X, y = _data()
    va_pred = adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])
    assert va_pred.shape == (80,)

    member = adapter.entry_diagnostics().observations["fold_initialization_members"][0]
    assert member["muon_lr_multiplier"] == pytest.approx(2.0)
    assert member["max_learning_rate"] == pytest.approx(SMALL_PARAMS["lr"])
    assert member["muon_max_learning_rate"] == pytest.approx(SMALL_PARAMS["lr"] * 2.0)


def test_lookup_transformer_records_no_muon_operating_point_without_muon():
    """muon이 아니면 Muon 전용 운전 지점은 비어 있다. (#385)"""
    cfg = ModelConfig(kind="lookup_transformer", params=dict(SMALL_PARAMS), fit={})
    adapter = model_mod.create(cfg, seed=SEED)
    X, y = _data()
    adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])

    member = adapter.entry_diagnostics().observations["fold_initialization_members"][0]
    assert member["muon_lr_multiplier"] == pytest.approx(1.0)
    assert member["muon_max_learning_rate"] is None


def test_lookup_orig_cdf_diff_config_is_exp131_feature_only_delta():
    proxy_columns = [
        "daily_screen_time_hours",
        "weekend_screen_time",
        "social_media_hours",
        "notifications_per_day",
        "app_opens_per_day",
    ]
    baseline = load_config(
        REPO / "configs" / "exp131_lookup_bivariate_plr5.yaml", "screen"
    )
    challenger = load_config(
        REPO / "configs" / "exp141_lookup_orig_cdf_diff.yaml", "screen"
    )

    assert challenger.name == "exp141_lookup_orig_cdf_diff"
    assert challenger.data == baseline.data
    assert challenger.model == baseline.model
    assert challenger.features.base == baseline.features.base
    assert challenger.features.categorical == baseline.features.categorical
    assert challenger.features.exclude == baseline.features.exclude
    assert challenger.features.providers == [
        *baseline.features.providers,
        {
            "kind": "original_cdf_diff",
            "path": "data/external/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv",
            "cols": proxy_columns,
        },
    ]
    assert all(
        f"{column}_orig_cdf_diff" not in challenger.model.params["lookup_cols"]
        for column in proxy_columns
    )


def test_lookup_transformer_declares_zero_based_epoch_evidence_per_member():
    """구성원마다 0부터 세는 위치를 선언하고 관측 학습 길이는 하나 크다. (#372)"""
    X, y = _data()
    adapter = _adapter()
    adapter.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])

    declaration = adapter.training_length_evidence()
    members = adapter.training_diagnostics()["fold_initialization_members"]
    assert declaration.model_family == "lookup_transformer"
    assert declaration.raw_field == "best_epoch"
    assert declaration.raw_meaning == ZERO_BASED_POSITION
    assert [item.inner_member for item in declaration.selections] == list(
        range(len(members))
    )
    assert [item.raw_value for item in declaration.selections] == [
        member["best_epoch"] for member in members
    ]

    evidence = observe_declaration(declaration, seed=SEED, outer_fold=1)
    assert [item.value for item in evidence.observations] == [
        member["best_epoch"] + 1 for member in members
    ]


def test_lookup_transformer_full_fit_declares_no_training_length_evidence():
    X, y = _data(96)
    adapter = _adapter()

    model_mod.fit_full(adapter, X, y, 2)

    with pytest.raises(RuntimeError, match="검증 분할"):
        adapter.training_length_evidence()
