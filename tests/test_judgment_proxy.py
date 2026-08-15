"""느린 champion 모델 계열 앞단의 대리 스크리닝 판정. (#87)"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.features import PLACEBO
from pipeline.judgment import JudgmentError, RunFacts, judge_proxy_screening

MODEL_PARAMS = {
    "model.force_row_wise": "True",
    "model.learning_rate": "0.05",
    "model.metric": "auc",
}
INPUT_HASHES = {
    "sha256.folds": "folds",
    "sha256.test": "test",
    "sha256.train": "train",
}


def make_run(
    *,
    name: str,
    auc: float,
    features: set[str],
    new_gain: float = 200.0,
    model_params: dict[str, str] | None = None,
) -> RunFacts:
    rows = [
        {"feature": "age", "fold": 0, "seed": 42, "gain": 500.0},
        {"feature": PLACEBO, "fold": 0, "seed": 42, "gain": 100.0},
        {"feature": f"{PLACEBO}_te", "fold": 0, "seed": 42, "gain": 10.0},
    ]
    if "new_feature" in features:
        rows.append(
            {"feature": "new_feature", "fold": 0, "seed": 42, "gain": new_gain}
        )
    return RunFacts(
        run_id=name,
        experiment=name,
        auc_oof=auc,
        features=features,
        seeds=[42],
        seed_aucs={42: auc},
        fold_aucs={},
        git_commit="deadbeef",
        importance=pd.DataFrame(rows),
        model_params=model_params if model_params is not None else MODEL_PARAMS,
        input_hashes=INPUT_HASHES,
    )


def test_proxy_screening_requires_nonnegative_delta_and_new_feature_importance():
    baseline = make_run(
        name="baseline", auc=0.96700, features={"age", PLACEBO, f"{PLACEBO}_te"}
    )
    challenger = make_run(
        name="challenger",
        auc=0.96701,
        features={"age", "new_feature", PLACEBO, f"{PLACEBO}_te"},
    )

    verdict = judge_proxy_screening(baseline, challenger)

    assert verdict.passed
    assert verdict.delta == pytest.approx(0.00001)


@pytest.mark.parametrize(
    ("auc", "new_gain"),
    [(0.96699, 200.0), (0.96701, 50.0)],
)
def test_proxy_screening_rejects_auc_loss_or_below_placebo_feature(auc, new_gain):
    baseline = make_run(
        name="baseline", auc=0.96700, features={"age", PLACEBO, f"{PLACEBO}_te"}
    )
    challenger = make_run(
        name="challenger",
        auc=auc,
        features={"age", "new_feature", PLACEBO, f"{PLACEBO}_te"},
        new_gain=new_gain,
    )

    verdict = judge_proxy_screening(baseline, challenger)

    assert not verdict.passed


def test_proxy_screening_rejects_different_model_settings():
    baseline = make_run(
        name="baseline", auc=0.96700, features={"age", PLACEBO, f"{PLACEBO}_te"}
    )
    challenger = make_run(
        name="challenger",
        auc=0.96701,
        features={"age", "new_feature", PLACEBO, f"{PLACEBO}_te"},
        model_params={"model.learning_rate": "0.1"},
    )

    with pytest.raises(JudgmentError, match="모델 설정"):
        judge_proxy_screening(baseline, challenger)
