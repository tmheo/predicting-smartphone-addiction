import json
from pathlib import Path
from types import SimpleNamespace

import optuna
import pandas as pd
import yaml

from pipeline.xgb_hpo import (
    NearestPoolMember,
    PoolPrediction,
    PreparedSearch,
    TrialEvaluation,
    load_pool_predictions,
    main,
    model_config_for_trial,
    nearest_pool_member,
    prepare_search_data,
    run_search,
)
from pipeline.runs import InMemoryRunStore
from pipeline.config import DataConfig, ExperimentConfig, FeatureConfig, ModelConfig


def test_trial_model_configuration_matches_issue_288_contract():
    trial = optuna.trial.FixedTrial(
        {
            "learning_rate": 0.02,
            "max_depth": 7,
            "min_child_weight": 0.25,
            "subsample": 0.8,
            "colsample_bylevel": 0.7,
            "colsample_bynode": 0.9,
            "reg_alpha": 0.03,
            "reg_lambda": 0.4,
            "grow_policy": "lossguide",
            "max_cat_to_onehot": 32,
            "max_leaves": 128,
        }
    )

    config = model_config_for_trial(trial)

    assert config.kind == "xgboost"
    assert config.params == {
        "tree_method": "hist",
        "eval_metric": "auc",
        "n_estimators": 10000,
        "learning_rate": 0.02,
        "max_depth": 7,
        "min_child_weight": 0.25,
        "subsample": 0.8,
        "colsample_bylevel": 0.7,
        "colsample_bynode": 0.9,
        "reg_alpha": 0.03,
        "reg_lambda": 0.4,
        "grow_policy": "lossguide",
        "max_cat_to_onehot": 32,
        "max_leaves": 128,
    }
    assert config.fit == {"early_stopping_rounds": 200}


def test_nearest_pool_member_uses_spearman_rank_correlation():
    ids = pd.Index([10, 11, 12, 13], name="id")
    candidate = pd.Series([0.4, 0.1, 0.3, 0.2], index=ids)
    pool = [
        PoolPrediction(
            run_id="ascending-run",
            config="ascending",
            values=pd.Series([0.1, 0.2, 0.3, 0.4], index=ids),
        ),
        PoolPrediction(
            run_id="nearest-run",
            config="nearest",
            values=pd.Series([0.8, 0.2, 0.6, 0.4], index=ids),
        ),
    ]

    nearest = nearest_pool_member(candidate, pool)

    assert nearest.run_id == "nearest-run"
    assert nearest.config == "nearest"
    assert nearest.spearman == 1.0


def test_search_writes_complete_machine_readable_trial_records(tmp_path):
    calls = 0

    def evaluate(config):
        nonlocal calls
        calls += 1
        return TrialEvaluation(
            fold0_auc=0.96 + calls / 1000,
            nearest=NearestPoolMember(
                run_id=f"pool-{calls}",
                config=f"member-{calls}",
                spearman=0.99 + calls / 10000,
            ),
            training_seconds=10.0 + calls,
            best_iteration=100 + calls,
        )

    output = tmp_path / "search.json"
    run_search(evaluate, n_trials=2, output_path=output)

    artifact = json.loads(output.read_text())
    assert artifact["schema_version"] == 1
    assert artifact["issue"] == 288
    assert artifact["sampler"] == {
        "name": "TPESampler",
        "multivariate": True,
        "seed": 42,
    }
    assert artifact["trials_requested"] == 2
    assert [trial["number"] for trial in artifact["trials"]] == [0, 1]
    assert artifact["trials"][0]["fold0_auc"] == 0.961
    assert artifact["trials"][0]["nearest_pool_member"] == {
        "run_id": "pool-1",
        "config": "member-1",
        "spearman": 0.9901,
    }
    assert artifact["trials"][0]["training_seconds"] == 11.0
    assert artifact["trials"][0]["best_iteration"] == 101
    assert artifact["trials"][0]["model_params"]["tree_method"] == "hist"
    assert artifact["top_two_trial_numbers"] == [1, 0]


def test_pool_predictions_are_aligned_to_fold_zero_ids():
    store = InMemoryRunStore()
    store.add_run(
        "member-run",
        oof=pd.DataFrame(
            {"id": [1, 2, 3, 4], "fold": [1, 0, 1, 0], "pred": [0.1, 0.2, 0.3, 0.4]}
        ),
    )
    pool = SimpleNamespace(
        members=[SimpleNamespace(run_id="member-run", config="member-config")]
    )

    predictions = load_pool_predictions(pool, store, pd.Index([4, 2], name="id"))

    assert len(predictions) == 1
    assert predictions[0].run_id == "member-run"
    assert predictions[0].config == "member-config"
    pd.testing.assert_series_equal(
        predictions[0].values,
        pd.Series([0.4, 0.2], index=pd.Index([4, 2], name="id"), name="pred"),
    )


def test_cli_plan_reports_the_fixed_search_contract(capsys):
    main(["--plan"])

    output = capsys.readouterr().out
    assert "base config : configs/exp045_xgb_depth8.yaml" in output
    assert "validation  : fold 0" in output
    assert "trials      : 50" in output
    assert "sampler     : TPESampler(multivariate=True, seed=42)" in output
    assert "tree_method : hist (local CPU)" in output


def test_prepared_search_evaluates_auc_correlation_time_and_best_iteration():
    class FakeAdapter:
        def fit(self, X_train, y_train, X_valid, y_valid):
            return [0.1, 0.9, 0.2, 0.8]

        def training_diagnostics(self):
            return {"best_iteration": 17, "best_score": 1.0}

    ids = pd.Index([20, 21, 22, 23], name="id")
    prepared = PreparedSearch(
        X_train=pd.DataFrame({"x": [0.0, 1.0]}),
        y_train=pd.Series([0, 1]),
        X_valid=pd.DataFrame({"x": [0.0, 1.0, 0.2, 0.8]}),
        y_valid=pd.Series([0, 1, 0, 1]),
        validation_ids=ids,
        pool_predictions=[
            PoolPrediction(
                run_id="pool-run",
                config="pool-config",
                values=pd.Series([0.2, 0.8, 0.3, 0.7], index=ids),
            )
        ],
        adapter_factory=lambda config, seed: FakeAdapter(),
    )

    result = prepared.evaluate(ModelConfig("xgboost", {}, {}))

    assert result.fold0_auc == 1.0
    assert result.nearest.run_id == "pool-run"
    assert result.nearest.spearman == 1.0
    assert result.training_seconds >= 0.0
    assert result.best_iteration == 17


def test_prepare_search_data_builds_fold_zero_features_once(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    folds_path = tmp_path / "folds.parquet"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "social_media_hours": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "addicted_label": [0, 1, 0, 1, 0, 1],
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        {"id": [7, 8], "x": [0.7, 0.8], "social_media_hours": [7.0, 8.0]}
    ).to_csv(test_path, index=False)
    pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "fold": [0, 1, 2, 0, 1, 2]}).to_parquet(
        folds_path, index=False
    )
    config = ExperimentConfig(
        name="small",
        data=DataConfig(train_path, test_path, test_path, folds_path),
        features=FeatureConfig(base="raw", categorical=[], providers=[]),
        model=ModelConfig("xgboost", {}, {}),
        initial_score=None,
        seeds=[42],
        stage="screen",
        source_path=tmp_path / "small.yaml",
    )
    store = InMemoryRunStore()
    store.add_run(
        "pool-run",
        oof=pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6],
                "fold": [0, 1, 2, 0, 1, 2],
                "pred": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        ),
    )
    pool = SimpleNamespace(
        members=[SimpleNamespace(run_id="pool-run", config="pool-config")]
    )

    prepared = prepare_search_data(config, pool, store)

    assert prepared.validation_ids.tolist() == [1, 4]
    assert prepared.X_train.shape == (4, 3)
    assert prepared.X_valid.shape == (2, 3)
    assert list(prepared.X_valid.columns) == [
        "x",
        "social_media_hours",
        "placebo_noise",
    ]
    assert prepared.pool_predictions[0].values.tolist() == [0.1, 0.4]


def test_promoted_candidate_configs_match_top_two_search_trials():
    artifact = json.loads(
        Path("artifacts/hpo/issue-288-xgb-search.json").read_text()
    )
    base = yaml.safe_load(Path("configs/exp045_xgb_depth8.yaml").read_text())
    promoted = {
        15: yaml.safe_load(Path("configs/exp134_xgb_hpo_trial15.yaml").read_text()),
        30: yaml.safe_load(Path("configs/exp135_xgb_hpo_trial30.yaml").read_text()),
    }

    assert artifact["top_two_trial_numbers"] == [15, 30]
    trials = {trial["number"]: trial for trial in artifact["trials"]}
    for trial_number, config in promoted.items():
        assert config["features"] == base["features"]
        assert config["model"]["params"] == trials[trial_number]["model_params"]
        assert config["model"]["fit"] == {"early_stopping_rounds": 200}
