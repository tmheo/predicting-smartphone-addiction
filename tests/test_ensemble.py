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

from pipeline.data import ID, TARGET
from pipeline.ensemble import (
    COMBINER_REGISTRY,
    BaggedGreedyRankMeanCombiner,
    CombinerConvergenceError,
    EmpiricalCDFTransformer,
    GreedyRankMeanCombiner,
    LogisticLinearCombiner,
    SHRINKAGE_LAMBDA_GRID,
    MissingnessInteractionLogisticCombiner,
    MissingnessSegmentedLogisticCombiner,
    NestedBaseline,
    NNLSCombiner,
    OptunaSubsetRankMeanCombiner,
    OptunaSubsetRidgeLogitCombiner,
    PerformanceWeightedRankMeanCombiner,
    RankMeanCombiner,
    RidgeLogitCombiner,
    ShrunkRankLogitCombiner,
    XGBoostRankLogitCombiner,
    evaluate_nested,
    full_fit_predictions,
    member_matrix,
    member_stats,
    member_test_matrix,
    missingness_bands,
    nested_baseline,
    rank_mean,
    record_nested_evaluation,
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


def test_registry_holds_reference_and_issue_64_adapters():
    assert list(COMBINER_REGISTRY) == [
        "rank_mean",
        "performance_weighted_rank_mean",
        "logit_logistic",
        "rank_logistic",
        "ridge_logit_alpha_0p01",
        "ridge_logit_alpha_0p1",
        "ridge_logit",
        "ridge_logit_alpha_10",
        "ridge_logit_alpha_100",
        "rank_gauss_logistic",
        "rank_logit_logistic",
        "greedy_rank_mean",
        "bagged_greedy_rank_mean",
        "optuna_subset_rank_mean",
        "optuna_subset_ridge_logit",
        "xgb_rank_logit",
        "missing_segmented_rank_logit",
        "missing_interaction_rank_logit",
        "missing_4plus_rank_logit",
        "nnls_logit",
        "nnls_rank",
        "shrunk_rank_logit_logistic",
    ]


def test_rank_ensemble_auc_uses_rank_mean_formula():
    # 기여 참고값(계열 2)이 균등 순위 평균 adapter의 수식과 비트 동일해야 한다(#104 이관).
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
    np.testing.assert_array_equal(
        fitted.predict(preds.iloc[:10]), rank_mean(preds.iloc[:10])
    )


def test_performance_weighted_rank_mean_uses_inner_auc_advantage():
    y = make_labels()
    preds = pd.DataFrame(
        {
            "strong": 0.1 + 0.8 * y,
            "weak": 0.4 + 0.2 * y,
            "inverse": 0.9 - 0.8 * y,
        },
        index=make_index(),
    )
    fitted = PerformanceWeightedRankMeanCombiner().fit(preds, y)
    assert fitted.summary() == {
        "strong": pytest.approx(0.5),
        "weak": pytest.approx(0.5),
        "inverse": 0.0,
    }
    outer = preds.iloc[:10].iloc[::-1]
    expected = outer.rank(pct=True).to_numpy() @ np.array([0.5, 0.5, 0.0])
    np.testing.assert_array_equal(fitted.predict(outer), expected)


def test_performance_weighted_rank_mean_falls_back_when_no_member_beats_random():
    y = make_labels()
    preds = pd.DataFrame(
        {"inverse": 1.0 - y, "constant": np.full(N, 0.5)}, index=make_index()
    )
    fitted = PerformanceWeightedRankMeanCombiner().fit(preds, y)
    assert fitted.summary() == {"inverse": 0.5, "constant": 0.5}


def selection_fixture() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(19)
    y = make_labels(seed=13)
    return (
        pd.DataFrame(
            {
                "strong_a": np.clip(0.15 + 0.65 * y + rng.normal(0, 0.08, N), 0, 1),
                "strong_b": np.clip(0.20 + 0.55 * y + rng.normal(0, 0.12, N), 0, 1),
                "noise": rng.random(N),
                "inverse": np.clip(0.85 - 0.65 * y + rng.normal(0, 0.08, N), 0, 1),
            },
            index=make_index(),
        ),
        y,
    )


def test_greedy_rank_mean_selects_sparse_subset_and_predicts_by_rank():
    preds, y = selection_fixture()
    fitted = GreedyRankMeanCombiner().fit(preds, y)
    weights = fitted.summary()
    assert sum(weight != 0.0 for weight in weights.values()) < len(preds.columns)
    assert sum(weights.values()) == pytest.approx(1.0)
    expected = preds.rank(pct=True).to_numpy() @ np.array(list(weights.values()))
    np.testing.assert_array_equal(fitted.predict(preds), expected)


def test_bagged_greedy_is_deterministic_and_averages_selection_frequencies():
    preds, y = selection_fixture()
    combiner = BaggedGreedyRankMeanCombiner(
        bags=4, sample_fraction=0.6, seed=7, workers=2
    )
    first = combiner.fit(preds, y).summary()
    second = combiner.fit(preds, y).summary()
    assert first == second
    assert sum(first.values()) == pytest.approx(1.0)
    assert any(weight == 0.0 for weight in first.values())


def test_optuna_subset_is_reproducible_and_keeps_full_member_summary():
    preds, y = selection_fixture()
    combiner = OptunaSubsetRankMeanCombiner(trials=8, seed=3)
    first = combiner.fit(preds, y)
    second = combiner.fit(preds, y)
    assert first.summary() == second.summary()
    assert set(first.summary()) == set(preds.columns)
    assert sum(first.summary().values()) == pytest.approx(1.0)
    assert np.isfinite(first.predict(preds)).all()


def test_optuna_subset_ridge_zero_fills_unselected_member_summary():
    preds, y = selection_fixture()
    fitted = OptunaSubsetRidgeLogitCombiner(trials=8, seed=3).fit(preds, y)
    assert set(fitted.summary()) == set(preds.columns)
    assert any(weight == 0.0 for weight in fitted.summary().values())
    assert np.isfinite(fitted.predict(preds)).all()


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


@pytest.mark.parametrize(
    "representation", ["rank", "logit", "rank_gauss", "rank_logit"]
)
def test_logistic_linear_representations_fit_and_predict_float64(representation):
    rng = np.random.default_rng(7)
    y = make_labels()
    preds = pd.DataFrame(
        {
            "informative": np.clip(0.2 + 0.6 * y + rng.normal(0, 0.05, N), 0.0, 1.0),
            "noise": rng.random(N),
        },
        index=make_index(),
    )
    fitted = LogisticLinearCombiner(f"test_{representation}", representation).fit(
        preds, y
    )
    prediction = fitted.predict(preds.iloc[:10])
    assert prediction.dtype == np.float64
    assert np.isfinite(prediction).all()
    assert set(fitted.summary()) == {"informative", "noise"}
    assert int(np.max(fitted.model.n_iter_)) < fitted.model.max_iter


def test_rank_transform_is_fitted_only_on_inner_rows():
    y = make_labels()
    inner = pd.DataFrame(
        {"a": np.linspace(0.2, 0.8, N), "b": np.linspace(0.1, 0.9, N)},
        index=make_index(),
    )
    fitted = LogisticLinearCombiner("rank_test", "rank").fit(inner, y)
    assert fitted.quantiles is not None
    assert fitted.quantiles.sorted_columns is not None
    np.testing.assert_array_equal(
        fitted.quantiles.sorted_columns[0][[0, -1]], [0.2, 0.8]
    )
    np.testing.assert_array_equal(
        fitted.quantiles.sorted_columns[1][[0, -1]], [0.1, 0.9]
    )
    outer = pd.DataFrame({"a": [-100.0, 100.0], "b": [-100.0, 100.0]})
    assert np.isfinite(fitted.predict(outer)).all()


def test_empirical_cdf_uses_midranks_and_clips_outer_extremes():
    inner = np.array([[0.0], [1.0], [1.0], [3.0]], dtype=np.float64)
    transformer = EmpiricalCDFTransformer("uniform").fit(inner)
    np.testing.assert_array_equal(
        transformer.transform(inner).ravel(), [0.125, 0.5, 0.5, 0.875]
    )
    outer = np.array([[-10.0], [10.0]], dtype=np.float64)
    np.testing.assert_array_equal(transformer.transform(outer).ravel(), [0.125, 0.875])

    normal = EmpiricalCDFTransformer("normal").fit(inner).transform(outer)
    assert np.isfinite(normal).all()
    assert normal[0, 0] < 0 < normal[1, 0]


def test_logistic_linear_rejects_non_convergence():
    preds = make_preds(members=8)
    y = make_labels()
    combiner = LogisticLinearCombiner("will_not_converge", "rank_logit", max_iter=1)
    with pytest.raises(CombinerConvergenceError, match=r"max\(n_iter_\)=1"):
        combiner.fit(preds, y)


def make_missingness_bands() -> pd.Series:
    return pd.Series(np.tile([0, 1, 2], N // 3), index=make_index(), dtype=np.int8)


def test_missingness_segmented_logistic_uses_every_band_and_predicts_float64():
    preds = make_preds(members=2)
    y = make_labels()
    fitted = MissingnessSegmentedLogisticCombiner(
        band_of=make_missingness_bands()
    ).fit(preds, y)
    assert set(fitted.models) == {0, 1, 2}
    assert fitted.global_model is None
    prediction = fitted.predict(preds)
    assert prediction.dtype == np.float64
    assert np.isfinite(prediction).all()
    assert set(fitted.summary()) == set(preds.columns)


def test_missingness_weak_band_combiner_keeps_global_fallback():
    preds = make_preds(members=2)
    y = make_labels()
    fitted = MissingnessSegmentedLogisticCombiner(
        band_of=make_missingness_bands(), specialized_bands=(2,)
    ).fit(preds, y)
    assert set(fitted.models) == {2}
    assert fitted.global_model is not None
    assert np.isfinite(fitted.predict(preds)).all()


def test_missingness_segmented_logistic_rejects_missing_row_context():
    preds = make_preds(members=2)
    y = make_labels()
    incomplete = make_missingness_bands().iloc[:-1]
    with pytest.raises(ValueError, match="요청한 id"):
        MissingnessSegmentedLogisticCombiner(band_of=incomplete).fit(preds, y)


def test_missingness_bands_excludes_id_and_target(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(
        {
            ID: [1, 2, 3],
            "a": [1.0, np.nan, np.nan],
            "b": [2.0, 2.0, np.nan],
            "c": [3.0, 3.0, np.nan],
            "d": [4.0, 4.0, np.nan],
            TARGET: [0, 1, 0],
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        {
            ID: [4],
            "a": [np.nan],
            "b": [np.nan],
            "c": [3.0],
            "d": [4.0],
        }
    ).to_csv(test_path, index=False)
    actual = missingness_bands(train_path, test_path)
    assert actual.to_dict() == {1: 0, 2: 0, 3: 2, 4: 1}


def test_missingness_interaction_logistic_predicts_float64():
    preds = make_preds(members=2)
    y = make_labels()
    fitted = MissingnessInteractionLogisticCombiner(
        band_of=make_missingness_bands()
    ).fit(preds, y)
    prediction = fitted.predict(preds)
    assert prediction.dtype == np.float64
    assert np.isfinite(prediction).all()
    assert set(fitted.summary()) == set(preds.columns)


def test_xgboost_rank_logit_is_deterministic_and_summarizes_members():
    preds, y = selection_fixture()
    combiner = XGBoostRankLogitCombiner(n_estimators=3, n_jobs=1)
    first = combiner.fit(preds, y)
    second = combiner.fit(preds, y)
    np.testing.assert_array_equal(first.predict(preds), second.predict(preds))
    assert set(first.summary()) == set(preds.columns)
    assert first.predict(preds).dtype == np.float64


def test_nnls_learns_nonnegative_normalized_weights():
    rng = np.random.default_rng(5)
    y = make_labels()
    preds = pd.DataFrame(
        {
            "informative": np.clip(0.2 + 0.6 * y + rng.normal(0, 0.05, N), 0.01, 0.99),
            "inverse": np.clip(0.8 - 0.6 * y + rng.normal(0, 0.05, N), 0.01, 0.99),
            "noise": rng.random(N),
        },
        index=make_index(),
    )
    for name in ("nnls_logit", "nnls_rank"):
        fitted = COMBINER_REGISTRY[name].fit(preds, y)
        weights = fitted.summary()
        assert set(weights) == set(preds.columns)
        assert all(weight >= 0.0 for weight in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["informative"] > weights["inverse"]
        prediction = fitted.predict(preds.iloc[:10])
        assert prediction.dtype == np.float64
        assert np.isfinite(prediction).all()


def test_nnls_falls_back_to_uniform_when_all_weights_are_zero():
    # 유일한 구성원이 역상관이면 NNLS 해가 0 벡터라 균등 가중치로 되돌린다.
    y = make_labels()
    preds = pd.DataFrame({"inverse": 0.9 - 0.8 * y}, index=make_index())
    fitted = NNLSCombiner("nnls_logit_test", "logit").fit(preds, y)
    assert fitted.summary() == {"inverse": 1.0}
    assert np.isfinite(fitted.predict(preds)).all()


def test_nnls_rejects_multi_feature_representations():
    with pytest.raises(ValueError, match="표현"):
        NNLSCombiner("nnls_bad", "rank_logit")


def test_shrunk_rank_logit_mixes_meta_and_rank_mean_in_rank_space():
    preds, y = selection_fixture()
    combiner = ShrunkRankLogitCombiner(fold_of=make_fold_of())
    fitted = combiner.fit(preds, y)
    assert fitted.shrinkage_lambda in SHRINKAGE_LAMBDA_GRID

    block = preds.iloc[:20]
    meta_ranks = (
        pd.Series(
            fitted.meta.predict(block[fitted.members]), index=block.index
        )
        .rank(pct=True)
        .to_numpy(dtype=np.float64)
    )
    expected = (
        fitted.shrinkage_lambda * meta_ranks
        + (1.0 - fitted.shrinkage_lambda)
        * block.rank(pct=True).to_numpy().mean(axis=1)
    )
    np.testing.assert_allclose(fitted.predict(block), expected)

    meta_summary = fitted.meta.summary()
    uniform = 1.0 / len(preds.columns)
    for member, weight in fitted.summary().items():
        assert weight == pytest.approx(
            fitted.shrinkage_lambda * meta_summary[member]
            + (1.0 - fitted.shrinkage_lambda) * uniform
        )


def test_shrunk_rank_logit_is_deterministic():
    preds, y = selection_fixture()
    combiner = ShrunkRankLogitCombiner(fold_of=make_fold_of())
    first = combiner.fit(preds, y)
    second = combiner.fit(preds, y)
    assert first.shrinkage_lambda == second.shrinkage_lambda
    assert first.summary() == second.summary()


def test_shrunk_rank_logit_requires_at_least_two_inner_folds():
    preds, y = selection_fixture()
    single = pd.Series(np.zeros(N, dtype=np.int64), index=make_index())
    with pytest.raises(ValueError, match="fold 2개"):
        ShrunkRankLogitCombiner(fold_of=single).fit(preds, y)


def test_shrunk_rank_logit_rejects_missing_fold_context():
    preds, y = selection_fixture()
    incomplete = make_fold_of().iloc[:-1]
    with pytest.raises(ValueError, match="요청한 id"):
        ShrunkRankLogitCombiner(fold_of=incomplete).fit(preds, y)


def test_shrunk_rank_logit_rejects_lambda_outside_unit_interval():
    with pytest.raises(ValueError, match="격자"):
        ShrunkRankLogitCombiner(lambda_grid=(0.5, 1.5))
    with pytest.raises(ValueError, match="격자"):
        ShrunkRankLogitCombiner(lambda_grid=())


def test_nested_evaluation_reports_non_convergent_outer_fold():
    preds = make_preds(members=8)
    y = make_labels()
    fold_of = pd.Series(np.arange(N) % 5, index=make_index())
    combiner = LogisticLinearCombiner("will_not_converge", "rank_logit", max_iter=1)
    with pytest.raises(CombinerConvergenceError, match="outer fold 0에서 미수렴"):
        evaluate_nested(combiner, preds, fold_of, y)


class SpyCombiner:
    """outer fold 루프의 계약 검증용: fold k는 학습에서 제외되고 fold k만 예측한다."""

    name = "spy"

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
    store.add_run("run-a", oof=pd.DataFrame({ID: index[:-1], "pred": np.zeros(N - 1)}))
    with pytest.raises(AssertionError, match="일치하지"):
        member_matrix([("exp_a", "run-a")], store, index)


def test_member_test_matrix_uses_submission_artifacts_and_reference_order(tmp_path):
    index = pd.Index([103, 101, 102], name=ID)
    store = InMemoryRunStore()
    for config, values in (("a", [0.1, 0.2, 0.3]), ("b", [0.4, 0.5, 0.6])):
        path = tmp_path / f"{config}.csv"
        pd.DataFrame({ID: [101, 102, 103], TARGET: values}).to_csv(path, index=False)
        store.add_run(f"run-{config}", submission_path=path)
    matrix = member_test_matrix([("exp_a", "run-a"), ("exp_b", "run-b")], store, index)
    assert list(matrix.columns) == ["exp_a", "exp_b"]
    assert matrix.index.equals(index)
    assert (matrix.dtypes == np.float64).all()
    np.testing.assert_array_equal(matrix["exp_a"], [0.3, 0.1, 0.2])


def test_full_fit_predictions_fits_on_oof_and_predicts_test_block():
    oof = make_preds()
    y = make_labels()
    test_preds = make_preds(seed=11).iloc[:10]
    actual = full_fit_predictions(RankMeanCombiner(), oof, y, test_preds)
    np.testing.assert_array_equal(actual, rank_mean(test_preds))


def test_full_fit_predictions_supports_missingness_context_on_test_ids():
    oof = make_preds(members=2)
    y = make_labels()
    test_index = pd.Index(np.arange(N, N + 6), name=ID)
    test_preds = pd.DataFrame(
        {"exp_0": np.linspace(0.1, 0.9, 6), "exp_1": np.linspace(0.2, 0.8, 6)},
        index=test_index,
    )
    band_of = pd.concat(
        [
            make_missingness_bands(),
            pd.Series([0, 1, 2, 0, 1, 2], index=test_index, dtype=np.int8),
        ]
    )
    actual = full_fit_predictions(
        MissingnessSegmentedLogisticCombiner(band_of=band_of), oof, y, test_preds
    )
    assert actual.shape == (6,)
    assert actual.dtype == np.float64
    assert np.isfinite(actual).all()


def make_fold_of() -> pd.Series:
    return pd.Series(np.arange(N) % 5, index=make_index())


def test_nested_baseline_measures_same_strategy_on_previous_subset():
    index = make_index()
    y = make_labels()
    fold_of = make_fold_of()
    preds = make_preds()
    store = InMemoryRunStore()
    for i in range(3):
        store.add_run(
            f"run-{i}",
            oof=pd.DataFrame({ID: index, "pred": preds[f"exp_{i}"].to_numpy()}),
        )
    store.add_run(
        "run-baseline",
        params={"ensemble.member_configs": "exp_0,exp_1"},
        metrics={"auc_oof": 0.9},
    )
    members = [(f"exp_{i}", f"run-{i}") for i in range(3)]
    best = evaluate_nested(RankMeanCombiner(), preds, fold_of, y)

    baseline = nested_baseline("run-baseline", best, members, store, fold_of, y)

    assert baseline.pool_size == 2
    assert baseline.previous_best_auc == 0.9
    assert baseline.new_member_configs == ["exp_2"]
    expected = evaluate_nested(
        RankMeanCombiner(), preds[["exp_0", "exp_1"]], fold_of, y
    ).nested_auc
    assert baseline.same_strategy_auc == pytest.approx(expected)


def test_nested_baseline_skips_same_strategy_when_member_left_the_pool():
    y = make_labels()
    fold_of = make_fold_of()
    preds = make_preds()
    store = InMemoryRunStore()
    store.add_run(
        "run-baseline",
        params={"ensemble.member_configs": "exp_0,exp_gone"},
        metrics={"auc_oof": 0.9},
    )
    members = [(f"exp_{i}", f"run-{i}") for i in range(3)]
    best = evaluate_nested(RankMeanCombiner(), preds, fold_of, y)

    baseline = nested_baseline("run-baseline", best, members, store, fold_of, y)

    assert baseline.same_strategy_auc is None
    assert baseline.new_member_configs == ["exp_1", "exp_2"]


def test_record_nested_evaluation_writes_derived_ensemble_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "pipeline.tracking.git_state",
        lambda: {"git_commit": "a" * 40, "git_dirty": "False"},
    )
    y = make_labels()
    evaluation = evaluate_nested(RankMeanCombiner(), make_preds(), make_fold_of(), y)
    members = [(f"exp_{i}", f"run-{i}") for i in range(3)]
    baseline = NestedBaseline(
        run_id="run-baseline",
        pool_size=2,
        previous_best_auc=0.9,
        same_strategy_auc=0.89,
        new_member_configs=["exp_2"],
    )
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    run_id = record_nested_evaluation(
        evaluation,
        members,
        issue=999,
        baseline=baseline,
        input_hashes={"train": "t1", "test": "t2", "folds": "t3"},
        tracking_uri=tracking_uri,
    )

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.info.run_name == "ensemble_rank_mean_issue999_pool3"
    assert run.info.status == "FINISHED"
    assert run.data.params["ensemble.strategy"] == "rank_mean"
    assert run.data.params["ensemble.member_configs"] == "exp_0,exp_1,exp_2"
    assert run.data.params["ensemble.member_run_ids"] == "run-0,run-1,run-2"
    assert run.data.params["ensemble.baseline_run_id"] == "run-baseline"
    assert run.data.params["ensemble.new_member_configs"] == "exp_2"
    assert run.data.params["git_dirty"] == "False"
    assert run.data.metrics["auc_oof"] == pytest.approx(evaluation.nested_auc)
    assert run.data.metrics["auc_fold_0"] == pytest.approx(evaluation.folds[0].auc)
    assert run.data.metrics["auc_pool2_previous_best"] == pytest.approx(0.9)
    assert run.data.metrics["delta_vs_pool2_previous_best"] == pytest.approx(
        evaluation.nested_auc - 0.9
    )
    assert run.data.metrics["auc_pool2_same_strategy"] == pytest.approx(0.89)
    assert run.data.metrics["delta_same_strategy"] == pytest.approx(
        evaluation.nested_auc - 0.89
    )
    assert run.data.tags["source.issue"] == "999"
    assert run.data.tags["source.kind"] == "derived_ensemble"
    assert run.data.tags["sha256.train"] == "t1"
    assert "recorded_at" in run.data.tags
    oof_path = client.download_artifacts(run_id, "oof.parquet")
    oof = pd.read_parquet(oof_path)
    assert list(oof.columns) == [ID, "prediction"]
    np.testing.assert_allclose(oof["prediction"], evaluation.prediction.to_numpy())
    import hashlib

    with open(oof_path, "rb") as handle:
        assert run.data.tags["sha256.oof_prediction"] == hashlib.sha256(
            handle.read()
        ).hexdigest()
    weights = pd.read_csv(client.download_artifacts(run_id, "member_weights.csv"))
    assert list(weights.columns) == ["member", "selected", "fold_total", "mean_weight"]
    assert list(weights["member"]) == ["exp_0", "exp_1", "exp_2"]
    assert (weights["selected"] == 5).all()
