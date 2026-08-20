"""Muon 혼성 optimizer 테스트. (#196)

- 그룹 표시 계약: 표시 없는 그룹과 muon 그룹 없는 구성을 거부한다.
- 위임 계약: 두 알고리즘 모두 실제로 매개변수를 갱신하고, 부모 그룹 dict를
  자식과 공유해 학습률 일정이 한 곳에서 전달된다.
- pytabkit TabM 그룹 분할: backbone 행렬만 muon 표시를 받고 weight decay
  그룹 구조가 보존된다.

torch가 필요하므로 test_model_* 격리 규약(pytest_openmp_guard)을 따른다.
"""

from __future__ import annotations

import pytest


def _hybrid_inputs():
    import torch

    matrix = torch.nn.Parameter(torch.zeros(8, 8))
    bias = torch.nn.Parameter(torch.zeros(8))
    return torch, matrix, bias


def test_muon_with_adamw_rejects_untagged_groups():
    torch, matrix, bias = _hybrid_inputs()
    from pipeline.muon import MuonWithAdamW

    with pytest.raises(ValueError, match="algorithm"):
        MuonWithAdamW([{"params": [matrix]}], lr=1e-3)


def test_muon_with_adamw_requires_a_muon_group():
    torch, matrix, bias = _hybrid_inputs()
    from pipeline.muon import MuonWithAdamW

    with pytest.raises(ValueError, match="muon"):
        MuonWithAdamW([{"params": [bias], "algorithm": "adamw"}], lr=1e-3)


def test_muon_with_adamw_updates_both_parameter_kinds():
    torch, matrix, bias = _hybrid_inputs()
    from pipeline.muon import MuonWithAdamW

    optimizer = MuonWithAdamW(
        [
            {"params": [bias], "weight_decay": 0.0, "algorithm": "adamw"},
            {"params": [matrix], "weight_decay": 0.0, "algorithm": "muon"},
        ],
        lr=1e-2,
    )
    torch.manual_seed(0)
    matrix.grad = torch.randn(8, 8)
    bias.grad = torch.randn(8)
    optimizer.step()

    assert float(matrix.abs().sum()) > 0.0
    assert float(bias.abs().sum()) > 0.0
    optimizer.zero_grad(set_to_none=True)
    assert matrix.grad is None and bias.grad is None


def test_muon_with_adamw_shares_group_dicts_with_delegates():
    torch, matrix, bias = _hybrid_inputs()
    from pipeline.muon import MuonWithAdamW

    optimizer = MuonWithAdamW(
        [
            {"params": [bias], "weight_decay": 0.0, "algorithm": "adamw"},
            {"params": [matrix], "weight_decay": 1e-2, "algorithm": "muon"},
        ],
        lr=1e-2,
    )
    delegate_groups = [
        group for delegate in optimizer._delegates for group in delegate.param_groups
    ]
    assert len(delegate_groups) == len(optimizer.param_groups)
    for group in optimizer.param_groups:
        assert any(group is shared for shared in delegate_groups)


def test_tabm_parameter_groups_tag_backbone_matrices_only():
    import torch  # noqa: F401  (pytabkit이 torch를 요구한다)
    from pytabkit.models.nn_models.tabm import Model

    from pipeline.muon import ALGORITHM_KEY, tabm_parameter_groups

    model = Model(
        n_num_features=6,
        cat_cardinalities=[3],
        n_classes=2,
        backbone={"type": "MLP", "n_blocks": 2, "d_block": 16, "dropout": 0.0},
        bins=None,
        num_embeddings=None,
        arch_type="tabm-mini-normal",
        k=4,
        share_training_batches=False,
    )
    groups = tabm_parameter_groups(model)

    tagged = {id(p): g[ALGORITHM_KEY] for g in groups for p in g["params"]}
    named = dict(model.named_parameters())
    assert len(tagged) == len(named)
    for name, parameter in named.items():
        expected = (
            "muon" if name.startswith("backbone.") and parameter.ndim == 2 else "adamw"
        )
        assert tagged[id(parameter)] == expected, name
    # weight decay 0 그룹 구조가 분할 뒤에도 남는다.
    zero_wd = [g for g in groups if g.get("weight_decay") == 0.0]
    assert zero_wd
