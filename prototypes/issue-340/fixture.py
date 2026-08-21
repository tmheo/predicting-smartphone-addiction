"""#340 원형이 쓰는 작은 합성 예측 행렬과 합성 장부.

실제 후보 풀 산출물을 읽지 않는다.
목적은 절차의 누출 경계, 결정적 재현, 중간 저장과 산출물 모양을 싸게 확인하는 것이고
점수의 크기 자체에는 의미가 없다.

합성 풀은 실제 35개 풀이 가진 세 가지 성질을 일부러 흉내낸다.

- 같은 모델 계보 묶음 안의 이전판과 개선판.
- 같은 정보 관점을 공유하는 서로 다른 모델 계열.
- 정확 복제와 순수 잡음처럼 빼도 정보가 줄지 않는 구성원.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FIXTURE_SEED = 340340


@dataclass(frozen=True)
class MemberLedger:
    """구성원 하나의 구조 축 사실. OOF를 보지 않고 사전 고정할 수 있는 것만 담는다."""

    config: str
    lineage: str  # 모델 계보 묶음 이름.
    generation: int  # 계보 안 순서. 작을수록 이전판.
    perspective: str  # 정보 관점 이름.
    refits: int  # 전체 자료 재학습 횟수.


@dataclass(frozen=True)
class Fixture:
    """합성 예측 행렬 하나와 그 장부."""

    preds: pd.DataFrame
    y: pd.Series
    fold_of: pd.Series
    band_of: pd.Series
    ledger: dict[str, MemberLedger]

    @property
    def configs(self) -> list[str]:
        return list(self.preds.columns)


def _noisy_view(
    latent: np.ndarray, rng: np.random.Generator, noise: float, bias: np.ndarray | None
) -> np.ndarray:
    """잠재 점수에 관점 편향과 개별 잡음을 얹은 구성원 예측 하나."""
    score = latent + (0.0 if bias is None else bias) + rng.normal(0.0, noise, len(latent))
    return 1.0 / (1.0 + np.exp(-score))


def build_fixture(*, rows: int, wide: bool) -> Fixture:
    """결정적 합성 행렬. `wide`가 참이면 계보와 관점이 더 많은 12개 풀을 만든다."""
    rng = np.random.default_rng(FIXTURE_SEED)
    latent = rng.normal(0.0, 1.0, rows)
    y = pd.Series(
        (rng.uniform(0.0, 1.0, rows) < 1.0 / (1.0 + np.exp(-latent))).astype(np.int64),
        index=pd.RangeIndex(rows, name="id"),
        name="target",
    )

    # 정보 관점은 잠재 점수에 얹는 재현 가능한 편향으로 표현한다.
    perspectives = {
        "원시": rng.normal(0.0, 0.35, rows),
        "복원": rng.normal(0.0, 0.35, rows),
        "부호화": rng.normal(0.0, 0.35, rows),
        "근접": rng.normal(0.0, 0.35, rows),
    }

    plan: list[tuple[str, str, int, str, int, float]] = [
        # config, lineage, generation, perspective, refits, noise
        ("expA1_tree_v1", "tree", 1, "원시", 3, 0.90),
        ("expA2_tree_v2", "tree", 2, "원시", 3, 0.70),
        ("expB1_lookup_v1", "lookup", 1, "부호화", 3, 0.75),
        ("expB2_lookup_v2", "lookup", 2, "부호화", 3, 0.60),
        ("expC1_linear", "linear", 1, "복원", 1, 1.10),
        ("expD1_knn", "knn", 1, "근접", 1, 1.30),
    ]
    if wide:
        plan += [
            ("expA3_tree_v3", "tree", 3, "원시", 3, 0.65),
            ("expB3_lookup_v3", "lookup", 3, "부호화", 3, 0.58),
            ("expC2_linear_v2", "linear", 2, "복원", 1, 0.95),
            ("expE1_spline", "spline", 1, "복원", 3, 0.80),
            ("expF1_recon", "recon", 1, "복원", 1, 1.05),
            ("expG1_proxy", "proxy", 1, "근접", 1, 1.25),
        ]

    columns: dict[str, np.ndarray] = {}
    ledger: dict[str, MemberLedger] = {}
    for config, lineage, generation, perspective, refits, noise in plan:
        columns[config] = _noisy_view(latent, rng, noise, perspectives[perspective])
        ledger[config] = MemberLedger(config, lineage, generation, perspective, refits)

    # 빼도 정보가 줄지 않는 구성원 둘. 절차가 이것을 먼저 버리는지 보려고 넣는다.
    clone_source = "expA1_tree_v1"
    columns["expZ1_clone"] = columns[clone_source].copy()
    ledger["expZ1_clone"] = MemberLedger("expZ1_clone", "clone", 1, "원시", 1)
    columns["expZ2_noise"] = rng.uniform(0.0, 1.0, rows)
    ledger["expZ2_noise"] = MemberLedger("expZ2_noise", "noise", 1, "무관", 1)

    preds = pd.DataFrame(columns, index=y.index).astype(np.float64)

    # 실행 계약의 outer fold 배정과 결측 개수 구간을 합성으로 대신한다.
    fold_of = pd.Series(np.arange(rows) % 5, index=y.index, name="fold").astype(np.int64)
    band_of = pd.Series(
        rng.integers(0, 3, rows), index=y.index, name="band"
    ).astype(np.int8)
    return Fixture(preds=preds, y=y, fold_of=fold_of, band_of=band_of, ledger=ledger)
