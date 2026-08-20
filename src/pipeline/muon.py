"""Muon 혼성 optimizer. (#196)

torch.optim.Muon은 은닉층의 2차원 행렬 가중치 전용이라, 나머지 매개변수
(embedding, 편향, LayerNorm, 원소별 adapter, 출력층)는 AdamW로 학습해야 한다.
이 모듈은 그룹별 algorithm 표시("muon"/"adamw")가 붙은 param group을 받아
단일 torch.optim.Optimizer 인터페이스 뒤에서 두 알고리즘에 위임하는 optimizer를
제공한다.

- 학습률 일정(OneCycleLR 등)과 GradScaler는 부모 optimizer 하나만 본다.
  자식 optimizer는 부모와 같은 그룹 dict 객체를 공유하므로, 일정이 부모
  param_groups의 lr·betas를 바꾸면 자식 step이 그대로 읽는다.
- Muon의 학습률은 adjust_lr_fn="match_rms_adamw"(Moonshot 판)로 보정한다.
  이 보정은 AdamW 갱신의 RMS에 맞추도록 설계돼, AdamW에 맞춘 기존 학습률과
  weight decay를 그대로 재사용하는 짝비교(#196의 "같은 설정") 계약을 지킨다.

torch가 필요하므로 사용하는 쪽(lookup_transformer, tabm)이 lazy import한다.
"""

from __future__ import annotations

import torch

# tabm의 pytabkit 패치가 torch.optim.AdamW 속성을 dispatcher로 바꾸는 동안에도
# 혼성 내부는 원본 클래스를 써야 하므로 import 시점에 직접 바인딩한다.
# (tabm의 패치 관리자는 이 모듈을 먼저 import한 뒤에 속성을 바꾼다.)
from torch.optim import AdamW as _AdamW
from torch.optim import Muon as _Muon

ALGORITHM_KEY = "algorithm"
_ALGORITHMS = {"adamw", "muon"}


class MuonWithAdamW(torch.optim.Optimizer):
    """algorithm 표시가 붙은 그룹을 Muon과 AdamW에 나눠 위임하는 optimizer."""

    def __init__(
        self,
        param_groups: list[dict[str, object]],
        *,
        lr: float,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
    ) -> None:
        for group in param_groups:
            if group.get(ALGORITHM_KEY) not in _ALGORITHMS:
                raise ValueError(
                    f"모든 그룹에 {ALGORITHM_KEY}가 {sorted(_ALGORITHMS)} 중 하나로 "
                    f"표시돼야 한다: {group.get(ALGORITHM_KEY)!r}"
                )
        # betas는 OneCycleLR(cycle_momentum=True)이 AdamW 그룹의 momentum을
        # 순환시킬 때 필요하다. Muon 그룹에도 주입되지만 Muon step은 읽지 않는다.
        super().__init__(
            param_groups,
            {"lr": lr, "betas": betas, "weight_decay": weight_decay},
        )
        muon_groups = [g for g in self.param_groups if g[ALGORITHM_KEY] == "muon"]
        adamw_groups = [g for g in self.param_groups if g[ALGORITHM_KEY] == "adamw"]
        if not muon_groups:
            raise ValueError("muon 그룹이 없다. 전부 AdamW라면 이 클래스를 쓰지 않는다.")
        # 자식 생성자에 부모의 그룹 dict를 그대로 넘겨 상태를 공유한다.
        # (torch.optim.Optimizer.add_param_group은 받은 dict 객체를 보관한다.)
        self._delegates: list[torch.optim.Optimizer] = []
        if adamw_groups:
            self._delegates.append(_AdamW(adamw_groups, lr=lr, betas=betas))
        self._delegates.append(
            _Muon(muon_groups, lr=lr, adjust_lr_fn="match_rms_adamw")
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for delegate in self._delegates:
            delegate.step()
        return loss


def hybrid_parameter_groups(
    param_groups: list[dict[str, object]],
    muon_parameters: list[torch.nn.Parameter],
) -> list[dict[str, object]]:
    """기존 그룹 설정을 보존하며 선택한 2차원 행렬만 Muon으로 분리한다."""
    if not muon_parameters:
        raise ValueError("Muon 대상 매개변수가 없다.")
    if any(parameter.ndim != 2 for parameter in muon_parameters):
        shapes = [tuple(parameter.shape) for parameter in muon_parameters]
        raise ValueError(f"Muon 대상은 모두 2차원 행렬이어야 한다: {shapes}")

    muon_ids = {id(parameter) for parameter in muon_parameters}
    seen_ids: set[int] = set()
    tagged: list[dict[str, object]] = []
    for group in param_groups:
        parameters = list(group["params"])
        adamw = [parameter for parameter in parameters if id(parameter) not in muon_ids]
        muon = [parameter for parameter in parameters if id(parameter) in muon_ids]
        seen_ids.update(id(parameter) for parameter in muon)
        if adamw:
            tagged.append({**group, "params": adamw, ALGORITHM_KEY: "adamw"})
        if muon:
            tagged.append({**group, "params": muon, ALGORITHM_KEY: "muon"})
    missing = muon_ids - seen_ids
    if missing:
        raise ValueError(f"기존 매개변수 그룹에 없는 Muon 대상이 있다: {len(missing)}개")
    return tagged


def tabm_parameter_groups(module: torch.nn.Module) -> list[dict[str, object]]:
    """pytabkit TabM의 weight decay 그룹을 유지한 채 algorithm 표시를 붙인다.

    Muon 대상은 backbone MLP의 2차원 행렬 가중치뿐이다. 수치·범주 embedding,
    원소별 tabm adapter(minimal_ensemble_adapter), 3차원 출력층(NLinear)과
    편향·정규화 매개변수는 AdamW로 남긴다.
    """
    from pytabkit.models.nn_models.tabm import make_parameter_groups

    muon_parameters = [
        parameter
        for name, parameter in module.named_parameters()
        if name.startswith("backbone.") and parameter.ndim == 2
    ]
    if not muon_parameters:
        raise ValueError("TabM backbone에서 Muon 대상 2차원 가중치를 찾지 못했다.")
    return hybrid_parameter_groups(make_parameter_groups(module), muon_parameters)
