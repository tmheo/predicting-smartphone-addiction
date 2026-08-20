"""피처 계획 interface 테스트. (#71 완료 기준)

- 등록된 모든 kind의 선언 대 실제 산출(합성 소형 frame).
- 누출 검증 거부 사례.
- 컬럼 충돌과 적용 순서 불변식.
- exp011 config의 선언 컬럼 골든 테스트.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import plan as plan_mod
from pipeline.config import FeatureConfig, load_config
from pipeline.features import PLACEBO
from pipeline.plan import FeaturePlan

REPO = Path(__file__).resolve().parents[1]


def toy_frames(n: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """실제 컬럼 이름을 흉내 낸 합성 소형 frame 쌍. placebo 마스크 원본 열을 포함해야 한다."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "id": np.arange(n),
            "daily_screen_time_hours": rng.uniform(1, 10, n).round(1),
            "social_media_hours": rng.uniform(0, 5, n).round(1),
            "gaming_hours": rng.uniform(0, 5, n).round(1),
            "work_study_hours": rng.uniform(0, 5, n).round(1),
            "stress_level": rng.choice(["low", "mid", "high"], n),
            "addicted_label": np.tile([0, 1], n // 2),
        }
    )
    train.loc[::7, "social_media_hours"] = np.nan
    test = train.drop(columns=["addicted_label"]).copy()
    test["id"] = test["id"] + n
    return train, test


def make_config(providers: list[dict], exclude: list[str] | None = None) -> FeatureConfig:
    return FeatureConfig(
        base="raw",
        categorical=["stress_level"],
        providers=providers,
        exclude=exclude or [],
    )


def toy_proxy_file(tmp_path: Path) -> tuple[str, str]:
    """원본 프록시를 흉내 낸 소형 CSV와 그 SHA-256. original_prior 테스트 입력."""
    import hashlib

    rng = np.random.default_rng(1)
    proxy = pd.DataFrame(
        {
            "gaming_hours": rng.uniform(0, 5, 200).round(1),
            "stress_level": rng.choice(["low", "mid", "high"], 200),
            "addicted_label": rng.integers(0, 2, 200),
        }
    )
    path = tmp_path / "toy_proxy.csv"
    proxy.to_csv(path, index=False)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def all_kind_providers(proxy_path: str, proxy_sha: str) -> list[dict]:
    return [
        {"kind": "categorical_copies", "cols": ["gaming_hours"]},
        {"kind": "pair_ce", "pairs": [["gaming_hours", "stress_level"]]},
        {"kind": "derived", "names": ["other_screen", "screen_slack"]},
        {"kind": "missing_indicators", "cols": ["social_media_hours"]},
        {
            "kind": "original_prior",
            "path": proxy_path,
            "sha256": proxy_sha,
            "cols": ["gaming_hours", ["gaming_hours", "stress_level"]],
            "stats": ["mean", "count"],
        },
        {
            "kind": "original_knn",
            "path": proxy_path,
            "sha256": proxy_sha,
            "cols": ["gaming_hours"],
            "ks": [1, 3],
        },
        {
            "kind": "original_cdf_diff",
            "path": proxy_path,
            "sha256": proxy_sha,
            "cols": ["gaming_hours"],
        },
        {
            "kind": "original_kde_ratio",
            "path": proxy_path,
            "sha256": proxy_sha,
            "cols": ["gaming_hours"],
        },
        {"kind": "target_encoding", "inner_folds": 3, "cols": ["gaming_hours", PLACEBO]},
        {
            "kind": "lattice_pair_te",
            "cols": ["daily_screen_time_hours", "gaming_hours"],
            "resolutions": ["floor", "r1"],
            "inner_folds": 2,
        },
        {"kind": "frequency_encoding", "cols": ["work_study_hours"]},
        {"kind": "median_impute_aux", "cols": ["social_media_hours"]},
        {
            "kind": "constrained_impute_aux",
            "cols": [
                "daily_screen_time_hours",
                "social_media_hours",
                "gaming_hours",
                "work_study_hours",
            ],
            "max_iter": 5,
        },
        {
            "kind": "xgb_impute_aux",
            "cols": ["daily_screen_time_hours", "social_media_hours"],
        },
    ]


def test_declared_vs_actual_for_all_registered_kinds(tmp_path):
    """레지스트리의 모든 kind가 선언한 컬럼을 실제로 산출하고, 최종 행렬이 선언 전체와 같다."""
    providers = all_kind_providers(*toy_proxy_file(tmp_path))
    assert {spec["kind"] for spec in providers} == set(plan_mod.REGISTRY)
    plan = FeaturePlan.from_config(make_config(providers))
    train, test = toy_frames()
    train, test = plan.apply_dataset_wide(train, test)
    assert list(train.columns[-2:]) == ["gaming_hours_cat", "gaming_hours__stress_level_ce"]

    X = plan.build_matrix(train, seed=7)
    assert list(X.columns) == plan.matrix_columns()
    assert PLACEBO in X.columns  # placebo는 설정 없이도 상시 포함된다.

    # fold-fit: 앞 40행을 학습 fold로 fit하고 train 전체를 transform한다.
    df_ff = pd.concat([train, X[[c for c in X.columns if c not in train.columns]]], axis=1)
    for t in plan.fold_fit_transformers():
        t.fit(df_ff.iloc[:40], seed=7)
    X_full = plan.add_fold_fit_columns(X, df_ff)
    assert list(X_full.columns) == plan.all_columns()


def test_build_full_matrices_fits_fold_providers_on_all_training_rows():
    plan = FeaturePlan.from_config(
        make_config(
            [
                {
                    "kind": "target_encoding",
                    "inner_folds": 3,
                    "cols": ["gaming_hours", PLACEBO],
                }
            ]
        )
    )
    train, test = toy_frames()
    train, test = plan.apply_dataset_wide(train, test)

    X_train, X_test = plan.build_full_matrices(train, test, seed=7)

    assert list(X_train.columns) == plan.all_columns()
    assert list(X_test.columns) == plan.all_columns()
    assert X_train.index.equals(train.index)
    assert X_test.index.equals(test.index)
    assert X_train[["gaming_hours_te", "placebo_noise_te"]].notna().all().all()
    assert X_test[["gaming_hours_te", "placebo_noise_te"]].notna().all().all()


def test_target_encoding_canary_is_row_unique_despite_shared_missing_mask():
    plan = FeaturePlan.from_config(
        make_config(
            [
                {
                    "kind": "target_encoding",
                    "inner_folds": 3,
                    "smoothing": 10.0,
                    "cols": [PLACEBO],
                }
            ]
        )
    )
    train, test = toy_frames()
    train, test = plan.apply_dataset_wide(train, test)
    X = plan.build_matrix(train, seed=7)
    fit_rows = train.index[:40]
    validation_rows = train.index[40:]
    df_ff = pd.concat(
        [train, X[[column for column in X.columns if column not in train.columns]]], axis=1
    )
    transformer = plan.fold_fit_transformers()[0]
    transformer.fit(df_ff.loc[fit_rows], seed=7)
    X_full = plan.add_fold_fit_columns(X, df_ff)

    assert X_full.loc[validation_rows, PLACEBO].isna().any()
    assert X_full.loc[validation_rows, "placebo_noise_te"].nunique() == 1
    assert X_full.loc[validation_rows, "placebo_noise_te"].iloc[0] == pytest.approx(
        train.loc[fit_rows, "addicted_label"].mean()
    )


def test_stage_order_invariant_ignores_config_interleaving():
    """providers 목록이 단계를 섞어 나열해도 적용 순서는 base 뒤 dataset-wide,
    row-wise, placebo, fold-fit 순서이고 같은 단계 안에서는 목록 순서다."""
    plan = FeaturePlan.from_config(
        make_config(
            [
                {"kind": "frequency_encoding", "cols": ["work_study_hours"]},
                {"kind": "derived", "names": ["screen_slack"]},
                {"kind": "pair_ce", "pairs": [["gaming_hours", "stress_level"]]},
                {"kind": "target_encoding", "inner_folds": 3, "cols": [PLACEBO]},
                {"kind": "derived", "names": ["other_screen"]},
            ]
        )
    )
    train, test = toy_frames()
    plan.apply_dataset_wide(train, test)
    base = [c for c in train.columns if c not in ("id", "addicted_label")]
    assert plan.all_columns() == base + [
        "gaming_hours__stress_level_ce",  # dataset-wide
        "screen_slack",  # row-wise, 목록 등장 순서
        "other_screen",
        PLACEBO,  # placebo 자동 삽입
        "work_study_hours_ce",  # fold-fit, 목록 등장 순서
        "placebo_noise_te",
    ]


def test_exclude_removes_raw_from_matrix_but_provider_input_keeps_it():
    """제외한 raw 컬럼은 행렬에서 빠지되 제공자 입력에는 남는다: age 제외 + age_te 유지 시나리오. (#79)"""
    plan = FeaturePlan.from_config(
        make_config(
            [{"kind": "target_encoding", "inner_folds": 3, "cols": ["gaming_hours", PLACEBO]}],
            exclude=["gaming_hours"],
        )
    )
    train, test = toy_frames()
    train, test = plan.apply_dataset_wide(train, test)
    assert "gaming_hours" not in plan.matrix_columns()
    assert "gaming_hours_te" in plan.all_columns()

    X = plan.build_matrix(train, seed=7)
    assert "gaming_hours" not in X.columns
    df_ff = pd.concat([train, X[[c for c in X.columns if c not in train.columns]]], axis=1)
    for t in plan.fold_fit_transformers():
        t.fit(df_ff.iloc[:40], seed=7)
    X_full = plan.add_fold_fit_columns(X, df_ff)
    assert "gaming_hours_te" in X_full.columns
    assert list(X_full.columns) == plan.all_columns()


def test_exclude_overlapping_categorical_is_rejected():
    with pytest.raises(ValueError, match="categorical"):
        FeaturePlan.from_config(make_config([], exclude=["stress_level"]))


def test_exclude_placebo_is_rejected():
    with pytest.raises(ValueError, match=PLACEBO):
        FeaturePlan.from_config(make_config([], exclude=[PLACEBO]))


def test_exclude_provider_column_is_rejected():
    with pytest.raises(ValueError, match="raw 컬럼 전용"):
        FeaturePlan.from_config(
            make_config(
                [{"kind": "derived", "names": ["other_screen"]}], exclude=["other_screen"]
            )
        )


def test_exclude_unknown_raw_column_fails_at_apply():
    plan = FeaturePlan.from_config(make_config([], exclude=["no_such_column"]))
    train, test = toy_frames()
    with pytest.raises(AssertionError, match="raw에 없는 컬럼"):
        plan.apply_dataset_wide(train, test)


def test_target_provider_without_placebo_spec_is_rejected():
    with pytest.raises(ValueError, match=PLACEBO):
        FeaturePlan.from_config(
            make_config([{"kind": "target_encoding", "inner_folds": 3, "cols": ["gaming_hours"]}])
        )


def test_pair_spec_containing_placebo_satisfies_canary_rule():
    plan = FeaturePlan.from_config(
        make_config(
            [
                {
                    "kind": "target_encoding",
                    "inner_folds": 3,
                    "cols": [["gaming_hours", PLACEBO]],
                }
            ]
        )
    )
    assert plan.fold_fit_transformers()[0].columns() == [f"gaming_hours__{PLACEBO}_te"]


def test_target_provider_outside_fold_fit_stage_is_rejected(monkeypatch):
    class LeakyRowWise:
        uses_target = True

        def columns(self) -> list[str]:
            return [f"{PLACEBO}_leaky"]

        def compute(self, df: pd.DataFrame) -> pd.DataFrame:
            raise AssertionError("적재 시점에 거부되어야 한다.")

    monkeypatch.setitem(
        plan_mod.REGISTRY, "leaky", plan_mod.ProviderKind(plan_mod.ROW_WISE, LeakyRowWise)
    )
    with pytest.raises(ValueError, match="fold-fit"):
        FeaturePlan.from_config(make_config([{"kind": "leaky"}]))


def test_duplicate_declared_columns_are_rejected():
    with pytest.raises(ValueError, match="충돌"):
        FeaturePlan.from_config(
            make_config(
                [
                    {"kind": "derived", "names": ["other_screen"]},
                    {"kind": "derived", "names": ["other_screen"]},
                ]
            )
        )


def test_provider_column_colliding_with_raw_column_fails():
    """합성 frame에 이미 있는 컬럼을 제공자가 다시 선언하면 적용이 실패한다."""
    plan = FeaturePlan.from_config(make_config([{"kind": "derived", "names": ["other_screen"]}]))
    train, test = toy_frames()
    train["other_screen"] = 0.0
    test["other_screen"] = 0.0
    with pytest.raises(AssertionError, match="raw 컬럼과 충돌"):
        plan.apply_dataset_wide(train, test)


def test_build_matrix_requires_apply_dataset_wide_first():
    plan = FeaturePlan.from_config(make_config([]))
    train, _ = toy_frames()
    with pytest.raises(AssertionError, match="apply_dataset_wide"):
        plan.build_matrix(train, seed=7)


def test_unknown_kind_and_non_raw_base_are_rejected():
    with pytest.raises(ValueError, match="알 수 없는 kind"):
        FeaturePlan.from_config(make_config([{"kind": "no_such_kind"}]))
    with pytest.raises(ValueError, match="raw"):
        FeaturePlan.from_config(
            FeatureConfig(base="explicit", categorical=[], providers=[])
        )


# #71 이전 스키마로 종결된 실험 config. 역사 기록으로 보존하되 적재는 거부된다.
LEGACY_CONFIG_NUMBERS = set(range(1, 18)) - {11}


def test_legacy_configs_are_rejected_and_current_schema_loads():
    """종결 실험 config 16개는 옛 스키마라 명확한 오류로 거부되고, 나머지는 적재된다."""
    for path in sorted((REPO / "configs").glob("*.yaml")):
        experiment_number = path.name[3:6]
        if experiment_number.isdigit() and int(experiment_number) in LEGACY_CONFIG_NUMBERS:
            with pytest.raises(ValueError, match="#71 이전"):
                load_config(path, "screen")
        else:
            load_config(path, "screen")


def test_stage_fills_seeds_from_judgment_constants():
    """시드 정책의 유일 출처는 judgment다: stage가 시드를 채우고 config에는 없다. (#103)"""
    from pipeline.judgment import CONFIRM_SEEDS, SCREENING_SEEDS

    path = REPO / "configs" / "exp011_resid_pair.yaml"
    assert load_config(path, "screen").seeds == SCREENING_SEEDS
    assert load_config(path, "confirm").seeds == CONFIRM_SEEDS
    assert load_config(path, "confirm").stage == "confirm"
    with pytest.raises(ValueError, match="알 수 없는 stage"):
        load_config(path, "final")


def test_config_with_cv_block_is_rejected(tmp_path):
    """cv.seeds 잔재는 명확한 오류로 거부한다: 단계는 --stage로 지정한다. (#103)"""
    source = (REPO / "configs" / "exp011_resid_pair.yaml").read_text()
    path = tmp_path / "exp011_with_cv.yaml"
    path.write_text(source + "\ncv:\n  seeds: [42]\n")
    with pytest.raises(ValueError, match="--stage"):
        load_config(path, "screen")


def test_exp011_declared_columns_golden():
    """champion 계보 exp011의 선언 컬럼 골든 테스트. 순서까지 고정한다."""
    cfg = load_config(REPO / "configs" / "exp011_resid_pair.yaml", "confirm")
    plan = FeaturePlan.from_config(cfg.features)
    train_header = pd.read_csv(REPO / cfg.data.train, nrows=0)
    test_header = pd.read_csv(REPO / cfg.data.test, nrows=0)
    plan.apply_dataset_wide(train_header, test_header)
    assert plan.all_columns() == [
        "age",
        "daily_screen_time_hours",
        "social_media_hours",
        "gaming_hours",
        "work_study_hours",
        "sleep_hours",
        "notifications_per_day",
        "app_opens_per_day",
        "weekend_screen_time",
        "gender",
        "stress_level",
        "academic_work_impact",
        "other_screen",
        "screen_slack",
        "placebo_noise",
        "age_te",
        "daily_screen_time_hours_te",
        "social_media_hours_te",
        "work_study_hours_te",
        "sleep_hours_te",
        "notifications_per_day_te",
        "app_opens_per_day_te",
        "weekend_screen_time_te",
        "placebo_noise_te",
    ]
