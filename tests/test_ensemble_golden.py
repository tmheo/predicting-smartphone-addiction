"""nested OOF 평가 리포트의 stdout golden. (#104)

InMemoryRunStore + 합성 풀(작은 N, 고정 시드 난수)로 run_report의 stdout을 박제한다.
시나리오는 reference adapter 2벌을 명시적으로 고정하므로, #64가 registry에 전략을
추가해도 이 golden들이 바이트 동일로 유지되는지가 회귀 검증이다.

갱신: UPDATE_GOLDEN=1 uv run pytest tests/test_ensemble_golden.py
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.data import ID
from pipeline.ensemble import COMBINER_REGISTRY, run_report
from pipeline.runs import InMemoryRunStore

GOLDEN_DIR = Path(__file__).parent / "golden" / "ensemble"
N = 100
MEMBERS = [("exp_alpha", "run-alpha"), ("exp_beta", "run-beta"), ("exp_gamma", "run-gamma")]


def _base(seed: int, y: pd.Series | None = None):
    rng = np.random.default_rng(seed)
    index = pd.Index(np.arange(N), name=ID)
    fold_of = pd.Series(np.arange(N) % 5, index=index, name="fold")
    if y is None:
        y = pd.Series(rng.integers(0, 2, N), index=index)
    return rng, index, fold_of, y


def scenario_adopt():
    """구성원이 유익해 두 전략 모두 채택 가능하고, 1위가 문턱 이상 앞선다."""
    rng, index, fold_of, y = _base(104)
    noise = {"exp_alpha": 0.15, "exp_beta": 0.20, "exp_gamma": 0.25}
    preds = {
        config: np.clip(0.35 + 0.3 * y + rng.normal(0, sigma, N), 0.01, 0.99)
        for config, sigma in noise.items()
    }
    return preds, index, fold_of, y, 0.90000


def scenario_tie_break():
    """fold별 클래스가 균형이라 두 전략 모두 nested AUC 1.0이다."""
    index = pd.Index(np.arange(N), name=ID)
    y = pd.Series(np.arange(N) % 2, index=index)
    rng, index, fold_of, y = _base(105, y)
    preds = {
        config: np.where(y == 1, 0.6 + 0.3 * rng.random(N), 0.1 + 0.3 * rng.random(N))
        for config, _ in MEMBERS
    }
    return preds, index, fold_of, y, 0.95000


def scenario_below_champion():
    """구성원 신호가 약해 어떤 전략도 champion에 못 미친다: 채택 없음."""
    rng, index, fold_of, y = _base(106)
    preds = {
        config: np.clip(0.5 + 0.05 * y + rng.normal(0, 0.2, N), 0.01, 0.99)
        for config, _ in MEMBERS
    }
    return preds, index, fold_of, y, 0.99000


SCENARIOS = {
    "adopt": scenario_adopt,
    "tie_break": scenario_tie_break,
    "below_champion": scenario_below_champion,
}


def transcript_of(scenario) -> str:
    preds, index, fold_of, y, champion_auc = scenario()
    store = InMemoryRunStore()
    for config, run_id in MEMBERS:
        store.add_run(run_id, oof=pd.DataFrame({ID: index, "pred": preds[config]}))
    # reference adapter 2벌을 명시적으로 고정한다: #64가 registry에 전략을 추가해도
    # 이 golden은 그대로여야 한다(module docstring).
    combiners = [COMBINER_REGISTRY["rank_mean"], COMBINER_REGISTRY["ridge_logit"]]
    out = io.StringIO()
    with redirect_stdout(out):
        run_report(combiners, MEMBERS, store, fold_of, y, champion_auc)
    return out.getvalue()


@pytest.mark.parametrize("name", SCENARIOS)
def test_report_matches_golden(name):
    transcript = transcript_of(SCENARIOS[name])
    golden_path = GOLDEN_DIR / f"{name}.txt"
    if os.environ.get("UPDATE_GOLDEN"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(transcript)
    assert golden_path.exists(), "golden 없음: UPDATE_GOLDEN=1로 먼저 박제할 것."
    assert transcript == golden_path.read_text()
