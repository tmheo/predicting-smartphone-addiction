"""스크리닝 판정의 짝지은 기준선 테스트. (ADR 0001, #74 개정)

- 기준선은 champion 시드 평균본이 아니라 같은 시드(seed 42)의 OOF AUC다.
- 시드 평균본 기준으로는 미달인 같은 시드 실재 개선이 통과로 판정돼야 한다.
- champion.yaml에 seed_aucs가 없으면 백필을 요구하며 중단한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.compare import RunFacts, judge_screening
from pipeline.features import PLACEBO


def make_challenger(auc_oof: float) -> RunFacts:
    features = {"age", "age_te", PLACEBO, f"{PLACEBO}_te"}
    importance = pd.DataFrame(
        {
            "feature": ["age", "age_te", PLACEBO, f"{PLACEBO}_te"],
            "fold": [0, 0, 0, 0],
            "seed": [42, 42, 42, 42],
            "gain": [500.0, 400.0, 100.0, 10.0],
        }
    )
    return RunFacts(
        run_id="challenger",
        experiment="exp_test",
        auc_oof=auc_oof,
        features=features,
        seeds=[42],
        seed_aucs={42: auc_oof},
        fold_aucs={},
        git_commit="deadbeef",
        importance=importance,
    )


def make_champion() -> dict:
    return {
        "oof_auc": 0.96740,
        "seed_aucs": {42: 0.96705, 43: 0.96702, 44: 0.96709},
        "features": "age,age_te,placebo_noise,placebo_noise_te",
    }


def test_screening_pairs_same_seed_baseline():
    # 시드 평균본(0.96740) 기준이면 -0.00019 미달이지만, seed 42 짝지은 기준(0.96705)
    # 으로는 +0.00016 통과여야 한다. #74에서 확인된 실재 개선 사례.
    passed, lines = judge_screening(make_champion(), make_challenger(0.96721))
    assert passed
    assert any("0.96705" in line for line in lines)


def test_screening_fails_below_paired_baseline():
    passed, _ = judge_screening(make_champion(), make_challenger(0.96700))
    assert not passed


def test_screening_requires_seed_aucs_backfill():
    champion = make_champion()
    del champion["seed_aucs"]
    with pytest.raises(SystemExit):
        judge_screening(champion, make_challenger(0.96721))
