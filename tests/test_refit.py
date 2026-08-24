from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pipeline import refit
from pipeline.config import DataConfig, ExperimentConfig, FeatureConfig, ModelConfig
from pipeline.refit import RefitMember, RefitPlan, mix_member_predictions


def version1_document() -> dict:
    """커밋된 문법 판본 2 장부를 판본 1 문법으로 되옮긴 시험용 장부.

    커밋된 `artifacts/full-refit-plan.yaml`은 이슈 #374에서 문법 판본 2로 옮겼고,
    그 장부의 내용 시험은 `tests/test_refit_plan.py`가 맡는다. 여기서는 아직 판본 1을
    읽는 `pipeline.refit`의 로더만 시험하므로, 같은 구성원과 같은 예산을 판본 1 문법으로
    되옮겨 쓴다. 두 문법이 같은 후보 풀을 가리키는지도 이 변환에서 함께 드러난다.
    """
    plan = yaml.safe_load(Path("artifacts/full-refit-plan.yaml").read_text())
    members = []
    for member in plan["members"]:
        status = member["training_length_evidence"]["status"]
        members.append(
            {
                "config": member["config"],
                "config_path": member["config_path"],
                "run_id": member["lineage"]["source_run_id"],
                "budget_source": (
                    "not_applicable" if status == "not_applicable" else "fold_median"
                ),
                "budgets": {
                    seed["seed"]: seed["budget"]
                    for seed in member["refit_budget_derivation"]["seeds"]
                },
            }
        )
    return {
        "schema_version": 1,
        "source_pool_sha256": plan["source_pool_sha256"],
        "protocol": plan["protocol"],
        "members": members,
    }


def test_version1_loader_still_reads_the_current_candidate_pool(tmp_path):
    path = tmp_path / "version1.yaml"
    path.write_text(yaml.safe_dump(version1_document(), allow_unicode=True, sort_keys=False))

    plan = RefitPlan.load(path)

    assert len(plan.members) == 32
    assert sum(len(member.budgets) for member in plan.members) == 94
    assert plan.member("exp127_lookup_muon").budgets == {42: 13, 43: 15, 44: 15}
    assert plan.member("exp059_lookup_transformer").budgets == {42: 15, 43: 15, 44: 18}


def test_refit_plan_rejects_unknown_combiner_before_execution(tmp_path):
    raw = version1_document()
    raw["protocol"]["combiner"] = "unregistered_combiner"
    path = tmp_path / "unknown-combiner.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))

    with pytest.raises(ValueError, match="등록되지 않은 결합 방식"):
        RefitPlan.load(path)


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


def test_assemble_uses_combiner_named_by_plan(monkeypatch, tmp_path):
    train_index = pd.Index([1, 2, 3, 4], name="id")
    test_index = pd.Index([10, 11], name="id")
    train = pd.DataFrame(
        {"id": train_index, "addicted_label": [0, 0, 1, 1]}
    )
    test = pd.DataFrame({"id": test_index})
    member = RefitMember(
        config="fake",
        config_path=tmp_path / "fake.yaml",
        run_id="run-1",
        budgets={42: None},
        budget_source="not_applicable",
    )
    plan = RefitPlan(
        source_path=tmp_path / "plan.yaml",
        source_pool_sha256="pool-hash",
        iteration_multiplier=1.25,
        budget_statistic="median",
        budget_rounding="half_up",
        cv_model_weight=5,
        full_model_weight=1,
        combiner="recording_combiner",
        members=(member,),
    )

    class RecordingCombiner:
        name = "recording_combiner"

    combiner = RecordingCombiner()
    calls = []
    monkeypatch.setitem(refit.COMBINER_REGISTRY, combiner.name, combiner)
    monkeypatch.setattr(
        refit.pd,
        "read_csv",
        lambda path: train.copy() if path == "data/train.csv" else test.copy(),
    )
    monkeypatch.setattr(
        refit,
        "member_matrix",
        lambda members, store, index: pd.DataFrame(
            {"fake": [0.1, 0.2, 0.8, 0.9]}, index=train_index
        ),
    )
    monkeypatch.setattr(
        refit,
        "member_test_matrix",
        lambda members, store, index: pd.DataFrame(
            {"fake": [0.25, 0.75]}, index=test_index
        ),
    )
    monkeypatch.setattr(
        refit,
        "_load_member_full_prediction",
        lambda plan, member, output, expected_ids: np.array([0.3, 0.7]),
    )

    def record_full_fit(selected, oof, y, test_predictions):
        calls.append(selected)
        return np.linspace(0.2, 0.8, len(test_predictions), dtype=np.float64)

    monkeypatch.setattr(refit, "full_fit_predictions", record_full_fit)

    refit.assemble(plan, tmp_path / "out")

    assert calls == [combiner, combiner, combiner]
