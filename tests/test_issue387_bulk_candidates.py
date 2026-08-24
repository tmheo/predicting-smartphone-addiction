from collections import Counter
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipeline.config import load_config
from pipeline import features as features_module


PLAN_PATH = Path("artifacts/issue387-bulk-candidate-plan.yaml")
PROXY_COLUMNS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]


@pytest.fixture(autouse=True)
def _locked_proxy_without_private_input(monkeypatch) -> None:
    proxy = pd.DataFrame(
        {
            **{column: range(10) for column in PROXY_COLUMNS},
            "addicted_label": [0, 1] * 5,
        }
    )
    monkeypatch.setattr(
        features_module,
        "_load_locked_proxy",
        lambda *_args, **_kwargs: proxy,
    )


def _plan() -> dict:
    return yaml.safe_load(PLAN_PATH.read_text())


def test_issue387_plan_has_the_precommitted_twelve_candidates() -> None:
    plan = _plan()
    candidates = plan["candidates"]

    assert plan["issue"] == 387
    assert plan["selection"]["target_count"] == 12
    assert len(candidates) == 12
    assert len({candidate["name"] for candidate in candidates}) == 12
    assert len({candidate["config"] for candidate in candidates}) == 12
    assert Counter(candidate["family"] for candidate in candidates) == {
        "xgboost": 5,
        "catboost": 4,
        "logistic_onehot": 3,
    }
    assert plan["downstream_registration"]["selection_procedure_required"] is True


def test_issue387_configs_match_the_plan_and_confirm_contract() -> None:
    for candidate in _plan()["candidates"]:
        config = load_config(candidate["config"], "confirm")
        assert config.name == candidate["name"]
        assert config.model.kind == candidate["family"]
        assert config.seeds == [42, 43, 44]


def test_issue387_tree_configs_partition_the_local_cpu() -> None:
    for candidate in _plan()["candidates"]:
        config = load_config(candidate["config"], "confirm")
        if config.model.kind == "xgboost":
            assert config.model.params["n_jobs"] == 4
        elif config.model.kind == "catboost":
            assert config.model.params["thread_count"] == 4
            assert "task_type" not in config.model.params
            assert "devices" not in config.model.params


def test_issue387_does_not_reproduce_already_promoted_xgboost_trials() -> None:
    plan = _plan()
    xgb_trials = {
        candidate["source"]["trial"]
        for candidate in plan["candidates"]
        if candidate["family"] == "xgboost"
    }
    assert xgb_trials.isdisjoint(plan["selection"]["excluded_xgb_trials"])


def test_issue387_tree_configs_match_their_search_trials() -> None:
    xgb_trials = {
        trial["number"]: trial
        for trial in json.loads(
            Path("artifacts/hpo/issue-288-xgb-search.json").read_text()
        )["trials"]
    }
    cat_trials = {
        trial["number"]: trial
        for trial in json.loads(
            Path("artifacts/judgments/issue-289-catboost-gpu-trials.json").read_text()
        )["trials"]
    }

    for candidate in _plan()["candidates"]:
        if candidate["family"] not in {"xgboost", "catboost"}:
            continue
        raw = yaml.safe_load(Path(candidate["config"]).read_text())
        actual = dict(raw["model"]["params"])
        trial_number = candidate["source"]["trial"]
        if candidate["family"] == "xgboost":
            actual.pop("n_jobs")
            source = xgb_trials[trial_number]
            expected = source["model_params"]
            assert candidate["screening_nearest_spearman"] == source[
                "nearest_pool_member"
            ]["spearman"]
        else:
            actual.pop("thread_count")
            source = cat_trials[trial_number]
            expected = dict(source["model_params"])
            expected.pop("task_type")
            expected.pop("devices")
            assert candidate["screening_nearest_spearman"] == source[
                "nearest_pool_spearman"
            ]
        assert actual == expected
        assert candidate["screening_fold0_auc"] == source["fold0_auc"]
