"""결합 전략 계약과 nested OOF 평가기의 단위 테스트. (#104)

- 균등 순위 평균 adapter가 수식을 소유하고, judgment의 rank_ensemble_auc가 같은
  수식을 쓴다(비트 동일).
- Fitted.predict는 outer fold 블록만 받고, 순위 변환의 모집단은 그 블록 자신이다.
- outer fold 루프는 fold k를 학습에서 제외하고 fold k만 예측한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline.data import ID
from pipeline.ensemble import (
    COMBINER_REGISTRY,
    RankMeanCombiner,
    RidgeLogitCombiner,
    evaluate_nested,
    member_matrix,
    member_stats,
    rank_mean,
)
from pipeline.judgment import rank_ensemble_auc
from pipeline.runs import InMemoryRunStore

N = 60


def make_index() -> pd.Index:
    return pd.Index(np.arange(N), name=ID)


def make_preds(seed: int = 0, members: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = make_index()
    return pd.DataFrame(
        {f"exp_{i}": rng.random(N) for i in range(members)}, index=index
    )


def make_labels(seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(0, 2, N), index=make_index())


def test_registry_holds_reference_adapters():
    assert list(COMBINER_REGISTRY) == ["rank_mean", "ridge_logit"]
    # 복잡도 서열: 무학습 < 학습. #64의 두 전략이 사이(2·3)에 선언된다.
    assert COMBINER_REGISTRY["rank_mean"].complexity < COMBINER_REGISTRY["ridge_logit"].complexity


def test_rank_ensemble_auc_uses_rank_mean_formula():
    # 기여 판정(계열 2)이 균등 순위 평균 adapter의 수식과 비트 동일해야 한다(#104 이관).
    preds = make_preds(members=11)
    y = make_labels()
    series = [preds[c].rename("pred") for c in preds.columns]
    legacy = float(
        roc_auc_score(
            y.to_numpy(), np.mean([p.rank(pct=True).to_numpy() for p in series], axis=0)
        )
    )
    assert rank_ensemble_auc(series, y) == legacy
    assert rank_ensemble_auc(series, y) == float(
        roc_auc_score(y.to_numpy(), rank_mean(preds))
    )


def test_rank_mean_population_is_the_given_block():
    # 순위의 모집단은 전달된 블록 자신이다: 부분 블록의 순위는 전체 블록과 다르다.
    preds = make_preds()
    block = preds.iloc[:10]
    expected = block.rank(pct=True).to_numpy().mean(axis=1)
    np.testing.assert_array_equal(rank_mean(block), expected)
    assert not np.array_equal(rank_mean(preds)[:10], rank_mean(block))


def test_rank_mean_combiner_fit_is_identity_and_summary_uniform():
    preds = make_preds()
    fitted = RankMeanCombiner().fit(preds, make_labels())
    assert fitted.summary() == {c: pytest.approx(1 / 3) for c in preds.columns}
    np.testing.assert_array_equal(fitted.predict(preds.iloc[:10]), rank_mean(preds.iloc[:10]))


def test_ridge_logit_learns_informative_member():
    rng = np.random.default_rng(2)
    index = make_index()
    y = make_labels()
    preds = pd.DataFrame(
        {
            "informative": np.clip(0.2 + 0.6 * y + rng.normal(0, 0.05, N), 0.01, 0.99),
            "noise": rng.random(N),
        },
        index=index,
    )
    fitted = RidgeLogitCombiner().fit(preds, y)
    summary = fitted.summary()
    assert set(summary) == {"informative", "noise"}
    assert summary["informative"] > abs(summary["noise"])
    auc = roc_auc_score(y.to_numpy(), fitted.predict(preds))
    assert auc > 0.95


def test_ridge_logit_clips_saturated_predictions():
    # 0/1 포화 예측도 epsilon 클리핑으로 logit이 발산하지 않아야 한다.
    y = make_labels()
    preds = pd.DataFrame({"hard": y.astype(float), "soft": 1.0 - y.astype(float)})
    preds.index = make_index()
    fitted = RidgeLogitCombiner().fit(preds, y)
    assert np.isfinite(fitted.predict(preds)).all()


def test_ridge_logit_alpha_is_constructor_argument():
    preds = make_preds()
    y = make_labels()
    loose = RidgeLogitCombiner(alpha=0.001).fit(preds, y).summary()
    tight = RidgeLogitCombiner(alpha=1000.0).fit(preds, y).summary()
    assert sum(abs(w) for w in tight.values()) < sum(abs(w) for w in loose.values())


class SpyCombiner:
    """outer fold 루프의 계약 검증용: fold k는 학습에서 제외되고 fold k만 예측한다."""

    name = "spy"
    complexity = 9

    def __init__(self) -> None:
        self.inner_ids: list[set[int]] = []

    def fit(self, inner_preds: pd.DataFrame, y: pd.Series):
        self.inner_ids.append(set(inner_preds.index))
        spy = self

        class Fitted:
            def predict(self, outer_preds: pd.DataFrame) -> np.ndarray:
                assert not (set(outer_preds.index) & spy.inner_ids[-1])
                return outer_preds.iloc[:, 0].to_numpy()

            def summary(self) -> dict[str, float]:
                return {"exp_0": 1.0, "exp_1": 0.0, "exp_2": 0.0}

        return Fitted()


def test_evaluate_nested_excludes_outer_fold_from_fit():
    preds = make_preds()
    y = make_labels()
    fold_of = pd.Series(np.arange(N) % 5, index=make_index())
    spy = SpyCombiner()
    evaluation = evaluate_nested(spy, preds, fold_of, y)

    assert [o.fold for o in evaluation.folds] == [0, 1, 2, 3, 4]
    assert len(spy.inner_ids) == 5
    for fold, inner in zip(range(5), spy.inner_ids):
        assert inner == set(fold_of[fold_of != fold].index)
    # 예측이 첫 구성원 그대로이므로 nested AUC는 그 구성원의 OOF AUC와 같다.
    assert evaluation.nested_auc == pytest.approx(
        roc_auc_score(y.to_numpy(), preds["exp_0"].to_numpy())
    )
    for outcome in evaluation.folds:
        mask = (fold_of == outcome.fold).to_numpy()
        assert outcome.auc == pytest.approx(
            roc_auc_score(y[mask].to_numpy(), preds["exp_0"][mask].to_numpy())
        )


def test_member_stats_aggregates_selection_and_mean_weight():
    preds = make_preds()
    y = make_labels()
    fold_of = pd.Series(np.arange(N) % 5, index=make_index())
    evaluation = evaluate_nested(SpyCombiner(), preds, fold_of, y)
    stats = {s.member: s for s in member_stats(evaluation)}
    assert stats["exp_0"].selected == 5
    assert stats["exp_0"].mean_weight == pytest.approx(1.0)
    assert stats["exp_1"].selected == 0
    assert stats["exp_1"].mean_weight == 0.0
    assert stats["exp_0"].fold_total == 5


def test_member_matrix_uses_config_names_and_float64():
    index = make_index()
    store = InMemoryRunStore()
    rng = np.random.default_rng(3)
    for run_id in ("run-a", "run-b"):
        store.add_run(
            run_id,
            oof=pd.DataFrame({ID: index, "pred": rng.random(N).astype(np.float32)}),
        )
    matrix = member_matrix([("exp_a", "run-a"), ("exp_b", "run-b")], store, index)
    assert list(matrix.columns) == ["exp_a", "exp_b"]
    assert (matrix.dtypes == np.float64).all()


def test_member_matrix_rejects_misaligned_ids():
    index = make_index()
    store = InMemoryRunStore()
    store.add_run(
        "run-a", oof=pd.DataFrame({ID: index[:-1], "pred": np.zeros(N - 1)})
    )
    with pytest.raises(AssertionError, match="일치하지"):
        member_matrix([("exp_a", "run-a")], store, index)
