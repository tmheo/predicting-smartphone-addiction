from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import refit
from pipeline.config import DataConfig, ExperimentConfig, FeatureConfig, ModelConfig
from pipeline.refit import RefitMember, RefitPlan, mix_member_predictions


def test_committed_refit_plan_matches_candidate_pool():
    plan = RefitPlan.load(Path("artifacts/full-refit-plan.yaml"))

    assert len(plan.members) == 30
    assert plan.cv_model_weight == 5
    assert plan.full_model_weight == 1
    ag25_gbm = plan.member("exp117_ag25_gbm_r21")
    assert ag25_gbm.budgets == {42: 23719, 43: 22945, 44: 22746}
    assert ag25_gbm.budget_source == "fold_median"
    tab_cnn = plan.member("exp113_tab_cnn_m0")
    assert tab_cnn.budgets == {42: 36, 43: 36, 44: 36}
    assert tab_cnn.budget_source == "fold_median"
    contextualized_spline = plan.member("exp085_contextual_spline_m0")
    assert contextualized_spline.budgets == {42: 14, 43: 13, 44: 11}
    assert contextualized_spline.budget_source == "fold_median"
    tabm_widths = plan.member("exp137_tabm_recon_widths")
    assert tabm_widths.budgets == {42: 18, 43: 16, 44: 16}
    assert tabm_widths.budget_source == "fold_median"
    realmlp = plan.member("exp124_realmlp_dtype_fix")
    assert realmlp.budgets == {42: 5, 43: 5, 44: 5}

    realmlp_muon = plan.member("exp134_realmlp_muon")
    assert realmlp_muon.budgets == {42: 5, 43: 5, 44: 5}
    assert realmlp_muon.budget_source == "fold_median"
    realmlp_muon_widths = plan.member("exp136_realmlp_muon_recon_widths")
    assert realmlp_muon_widths.budgets == {42: 5, 43: 5, 44: 5}
    assert realmlp_muon_widths.budget_source == "fold_median"
    assert realmlp.budget_source == "fold_median"
    xgb_hpo = plan.member("exp135_xgb_hpo_trial30")
    assert xgb_hpo.budgets == {42: 9758, 43: 10393, 44: 10368}
    assert xgb_hpo.budget_source == "fold_median"
    constrained_impute = plan.member("exp025_constrained_impute")
    assert constrained_impute.budgets == {42: 503, 43: 394, 44: 398}
    assert constrained_impute.budget_source == "fold_median"
    lookup_muon = plan.member("exp127_lookup_muon")
    assert lookup_muon.budgets == {42: 11, 43: 14, 44: 14}
    assert lookup_muon.budget_source == "fold_median"
    recon_ce = plan.member("exp027_recon_ce")
    assert recon_ce.budgets == {42: 514, 43: 474, 44: 464}
    assert recon_ce.budget_source == "fold_median"
    original_cdf_diff = plan.member("exp048_lgb_orig_cdf_diff")
    assert original_cdf_diff.budgets == {42: 578, 43: 365, 44: 388}
    assert original_cdf_diff.budget_source == "fold_median"


def test_mix_member_predictions_uses_model_count_weights():
    index = pd.Index([10, 11], name="id")
    cv = pd.DataFrame({"a": [0.2, 0.8], "b": [0.4, 0.6]}, index=index)
    full = pd.DataFrame({"a": [0.8, 0.2], "b": [1.0, 0.0]}, index=index)

    mixed = mix_member_predictions(cv, full, cv_weight=5, full_weight=1)

    assert mixed.to_numpy() == pytest.approx(
        np.array([[0.3, 0.5], [0.7, 0.5]], dtype=np.float64)
    )
    assert all(dtype == np.dtype("float64") for dtype in mixed.dtypes)


def test_mix_member_predictions_rejects_misaligned_inputs():
    cv = pd.DataFrame({"a": [0.2]}, index=pd.Index([10], name="id"))
    full = pd.DataFrame({"b": [0.8]}, index=pd.Index([10], name="id"))

    with pytest.raises(ValueError, match="구성원 순서"):
        mix_member_predictions(cv, full, cv_weight=5, full_weight=1)


def test_run_member_writes_lineage_checkpoint_and_resumes(monkeypatch, tmp_path):
    train = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "x": [0.0, 1.0, 2.0, 3.0],
            "social_media_hours": [0.5, 1.0, 1.5, 2.0],
            "addicted_label": [0, 0, 1, 1],
        }
    )
    test = pd.DataFrame(
        {"id": [5, 6], "x": [0.5, 2.5], "social_media_hours": [0.75, 1.75]}
    )
    config_path = tmp_path / "fake.yaml"
    config_path.write_text("name: fake\n")
    cfg = ExperimentConfig(
        name="fake",
        data=DataConfig(
            train=Path("train.csv"),
            test=Path("test.csv"),
            sample_submission=Path("sample.csv"),
            folds=Path("folds.parquet"),
        ),
        features=FeatureConfig(base="raw", categorical=[]),
        model=ModelConfig(kind="fake", params={}, fit={}),
        initial_score=None,
        seeds=[42, 43, 44],
        stage="confirm",
        source_path=config_path,
    )
    member = RefitMember(
        config="fake",
        config_path=config_path,
        run_id="run-1",
        budgets={42: 3},
        budget_source="fold_median",
    )
    plan = RefitPlan(
        source_path=tmp_path / "plan.yaml",
        source_pool_sha256="pool-hash",
        iteration_multiplier=1.25,
        budget_statistic="median",
        budget_rounding="half_up",
        cv_model_weight=5,
        full_model_weight=1,
        combiner="missing_interaction_rank_logit",
        members=(member,),
    )

    class FakeAdapter:
        fit_calls = 0

        def fit_full(self, X, y, training_budget, initial_score=None):
            FakeAdapter.fit_calls += 1
            assert training_budget == 3

        def predict(self, X, initial_score=None):
            return np.linspace(0.25, 0.75, len(X), dtype=np.float64)

    monkeypatch.setattr(refit, "load_config", lambda path, stage: cfg)
    monkeypatch.setattr(
        refit.data,
        "load_csv",
        lambda path: train.copy() if path == cfg.data.train else test.copy(),
    )
    monkeypatch.setattr(refit.data, "file_sha256", lambda path: f"hash:{path}")
    monkeypatch.setattr(
        refit.tracking,
        "git_state",
        lambda: {"git_commit": "commit-1", "git_dirty": "False"},
    )
    monkeypatch.setattr(refit.model, "create", lambda model_cfg, seed: FakeAdapter())

    first = refit.run_member(plan, member, tmp_path / "out")
    second = refit.run_member(plan, member, tmp_path / "out")

    assert first == second
    assert FakeAdapter.fit_calls == 1
    assert (first.parent / "test_pred_seed_42.json").is_file()
    assert (first.parent / "manifest.json").is_file()


def test_run_member_can_fit_seeds_independently_before_finalizing(monkeypatch, tmp_path):
    train = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "x": [0.0, 1.0, 2.0, 3.0],
            "social_media_hours": [0.5, 1.0, 1.5, 2.0],
            "addicted_label": [0, 0, 1, 1],
        }
    )
    test = pd.DataFrame(
        {"id": [5, 6], "x": [0.5, 2.5], "social_media_hours": [0.75, 1.75]}
    )
    config_path = tmp_path / "fake.yaml"
    config_path.write_text("name: fake\n")
    cfg = ExperimentConfig(
        name="fake",
        data=DataConfig(
            train=Path("train.csv"),
            test=Path("test.csv"),
            sample_submission=Path("sample.csv"),
            folds=Path("folds.parquet"),
        ),
        features=FeatureConfig(base="raw", categorical=[]),
        model=ModelConfig(kind="fake", params={}, fit={}),
        initial_score=None,
        seeds=[42, 43, 44],
        stage="confirm",
        source_path=config_path,
    )
    member = RefitMember(
        config="fake",
        config_path=config_path,
        run_id="run-1",
        budgets={42: 3, 43: 4, 44: 5},
        budget_source="fold_median",
    )
    plan = RefitPlan(
        source_path=tmp_path / "plan.yaml",
        source_pool_sha256="pool-hash",
        iteration_multiplier=1.25,
        budget_statistic="median",
        budget_rounding="half_up",
        cv_model_weight=5,
        full_model_weight=1,
        combiner="missing_interaction_rank_logit",
        members=(member,),
    )

    class FakeAdapter:
        fitted_seeds: list[int] = []

        def __init__(self, seed: int):
            self.seed = seed

        def fit_full(self, X, y, training_budget, initial_score=None):
            FakeAdapter.fitted_seeds.append(self.seed)
            assert training_budget == member.budgets[self.seed]

        def predict(self, X, initial_score=None):
            return np.full(len(X), self.seed / 100, dtype=np.float64)

    monkeypatch.setattr(refit, "load_config", lambda path, stage: cfg)
    monkeypatch.setattr(
        refit.data,
        "load_csv",
        lambda path: train.copy() if path == cfg.data.train else test.copy(),
    )
    monkeypatch.setattr(refit.data, "file_sha256", lambda path: f"hash:{path}")
    monkeypatch.setattr(
        refit.tracking,
        "git_state",
        lambda: {"git_commit": "commit-1", "git_dirty": "False"},
    )
    monkeypatch.setattr(refit.model, "create", lambda model_cfg, seed: FakeAdapter(seed))

    output = tmp_path / "out"
    for seed in member.budgets:
        path = refit.run_member(
            plan,
            member,
            output,
            seeds=(seed,),
            finalize=False,
        )
        assert path.name == f"test_pred_seed_{seed}.parquet"
        assert not (path.parent / "manifest.json").exists()

    final = refit.run_member(plan, member, output)

    assert FakeAdapter.fitted_seeds == [42, 43, 44]
    assert (final.parent / "manifest.json").is_file()
    averaged = pd.read_parquet(final)
    assert averaged["pred"].to_numpy() == pytest.approx([0.43, 0.43])
