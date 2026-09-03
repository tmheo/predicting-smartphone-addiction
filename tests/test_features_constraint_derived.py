"""제약 파생 4열 블록(#619, #622)의 값·결측·격자 규약과 범주 복제 검증.

계약(docs/research/constraint-derived-ladder-spec.md):
- 4열은 성분 하나라도 결측이면 결측이고 소수 둘째 자리에서 반올림한다.
- 비율 7열은 분자·분모가 유한하고 분모가 양수일 때만 정의한다.
- 자리수 32열은 정체성 블록 규약을 4열에 적용하되 round2 계열은 뺀다.
- categorical_copies의 derived 인자는 파생 열의 <col>_cat 복제를 train/test 합집합 범주로 만든다.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.config import FeatureConfig
from pipeline.features import (
    CONSTRAINT_DERIVED_COLS,
    CONSTRAINT_DERIVED_NAMES,
    CONSTRAINT_DERIVED_RATIOS,
    DERIVED_REGISTRY,
    CategoricalCopies,
    DerivedColumns,
)
from pipeline.plan import FeaturePlan


def make_df() -> pd.DataFrame:
    # 행: 0 전관측, 1 social 결측, 2 daily 결측, 3 gaming 결측, 4 부동소수 위험값, 5 daily 0
    return pd.DataFrame(
        {
            "daily_screen_time_hours": [8.0, 8.0, np.nan, 8.0, 5.3, 0.0],
            "social_media_hours": [2.0, np.nan, 2.0, 2.0, 1.1, 1.0],
            "gaming_hours": [1.5, 1.5, 1.5, np.nan, 2.2, 1.0],
            "work_study_hours": [4.0, 4.0, 4.0, 4.0, 1.9, 1.0],
        }
    )


def test_registry_has_43_constraint_derived_names() -> None:
    assert len(CONSTRAINT_DERIVED_NAMES) == 43
    assert len(set(CONSTRAINT_DERIVED_NAMES)) == 43
    assert set(CONSTRAINT_DERIVED_NAMES) <= set(DERIVED_REGISTRY)
    assert CONSTRAINT_DERIVED_NAMES[:4] == CONSTRAINT_DERIVED_COLS
    assert len(CONSTRAINT_DERIVED_RATIOS) == 7
    # 0.01 격자에서 항등이 되는 round2 계열은 등록하지 않는다.
    assert "fake_daily_round2" not in DERIVED_REGISTRY
    assert "fake_daily_absdiff_round2" not in DERIVED_REGISTRY


def test_four_columns_follow_strict_missing_convention() -> None:
    out = DerivedColumns(CONSTRAINT_DERIVED_COLS).compute(make_df())
    assert out.loc[0, "fake_daily"] == pytest.approx(7.5)
    assert out.loc[0, "fake_social"] == pytest.approx(8.0 - 1.5 - 4.0)
    assert out.loc[0, "fake_work"] == pytest.approx(8.0 - 2.0 - 1.5)
    assert out.loc[0, "fake_game"] == pytest.approx(8.0 - 2.0 - 4.0)
    # social 결측: fake_daily와 social을 성분으로 쓰는 두 열은 결측, fake_social만 정의된다.
    assert np.isnan(out.loc[1, "fake_daily"])
    assert np.isnan(out.loc[1, "fake_work"])
    assert np.isnan(out.loc[1, "fake_game"])
    assert out.loc[1, "fake_social"] == pytest.approx(2.5)
    # daily 결측: fake_daily만 정의되고 daily를 쓰는 세 열은 결측이다.
    assert out.loc[2, "fake_daily"] == pytest.approx(7.5)
    for col in ("fake_social", "fake_work", "fake_game"):
        assert np.isnan(out.loc[2, col])
    # gaming 결측: fake_game만 정의된다.
    assert out.loc[3, "fake_game"] == pytest.approx(2.0)
    assert np.isnan(out.loc[3, "fake_daily"])
    assert np.isnan(out.loc[3, "fake_social"])
    assert np.isnan(out.loc[3, "fake_work"])


def test_four_columns_land_on_hundredths_grid() -> None:
    out = DerivedColumns(CONSTRAINT_DERIVED_COLS).compute(make_df())
    # 5.3 - 2.2 - 1.9 = 1.2000000000000002 같은 부동소수 오차가 정확값 키를 깨지 않는다.
    assert out.loc[4, "fake_social"] == 1.2
    assert out.loc[4, "fake_daily"] == 5.2
    observed = out.to_numpy()[~np.isnan(out.to_numpy())]
    assert np.array_equal(observed, np.round(observed, 2))


def test_ratios_follow_guarded_ratio_convention() -> None:
    names = list(CONSTRAINT_DERIVED_RATIOS)
    out = DerivedColumns(names).compute(make_df())
    assert out.loc[0, "fake_daily_share_screen"] == pytest.approx(7.5 / 8.0)
    assert out.loc[0, "fake_social_share_screen"] == pytest.approx(2.5 / 8.0)
    assert out.loc[0, "social_share_fake_daily"] == pytest.approx(2.0 / 7.5)
    assert out.loc[0, "gaming_share_fake_daily"] == pytest.approx(1.5 / 7.5)
    assert out.loc[0, "work_share_fake_daily"] == pytest.approx(4.0 / 7.5)
    # 분모 0이면 미정의이고 무한대나 대체값을 만들지 않는다.
    assert np.isnan(out.loc[5, "fake_daily_share_screen"])
    # 분자(fake_daily)가 결측이면 미정의다.
    assert np.isnan(out.loc[1, "fake_daily_share_screen"])
    assert np.isnan(out.loc[1, "social_share_fake_daily"])
    values = out.to_numpy()
    assert np.isfinite(values[~np.isnan(values)]).all()


def test_digit_block_applies_identity_convention_to_rounded_values() -> None:
    digit_names = [
        name
        for name in CONSTRAINT_DERIVED_NAMES
        if name.startswith("fake_social_") and name not in CONSTRAINT_DERIVED_RATIOS
    ]
    assert len(digit_names) == 8
    out = DerivedColumns(digit_names).compute(make_df())
    # 행 0의 fake_social = 2.5: 정수 아님, 소수 1자리.
    assert out.loc[0, "fake_social_round0"] == pytest.approx(2.0)
    assert out.loc[0, "fake_social_absdiff_round0"] == pytest.approx(0.5)
    assert out.loc[0, "fake_social_is_round0"] == 0.0
    assert out.loc[0, "fake_social_round1"] == pytest.approx(2.5)
    assert out.loc[0, "fake_social_absdiff_round1"] == pytest.approx(0.0)
    assert out.loc[0, "fake_social_is_round1"] == 1.0
    assert out.loc[0, "fake_social_tenths"] == 5.0
    assert out.loc[0, "fake_social_hundredths"] == 0.0
    # 행 4의 fake_social = 1.2(격자 반올림 뒤): 자리 판정이 부동소수 오차에 흔들리지 않는다.
    assert out.loc[4, "fake_social_is_round1"] == 1.0
    assert out.loc[4, "fake_social_tenths"] == 2.0
    # 결측 행은 지시자와 자리 값도 결측이다.
    assert out.loc[2].isna().all()


def _copy_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = make_df()
    test = make_df().iloc[::-1].reset_index(drop=True)
    test.loc[0, "work_study_hours"] = 0.5  # test에만 있는 fake_daily 값을 만든다.
    return train, test


def test_categorical_copies_derived_uses_union_categories() -> None:
    provider = CategoricalCopies(cols=[], derived=["fake_daily", "fake_social"])
    assert provider.columns() == ["fake_daily_cat", "fake_social_cat"]
    train, test = _copy_frames()
    new_train, new_test = provider.compute(train, test)
    assert list(new_train.columns) == provider.columns()
    assert isinstance(new_train["fake_daily_cat"].dtype, pd.CategoricalDtype)
    expected = DerivedColumns(["fake_daily"]).compute(train)["fake_daily"]
    assert new_train["fake_daily_cat"].astype("float64").equals(expected)
    # 범주는 train/test 값 합집합이라 test에만 있는 값도 코드가 있다.
    assert list(new_train["fake_daily_cat"].cat.categories) == list(
        new_test["fake_daily_cat"].cat.categories
    )
    test_only = 0.5 + 1.0 + 1.0
    assert test_only in new_test["fake_daily_cat"].cat.categories
    # 결측 파생 값은 결측 범주(코드 -1)로 남는다.
    assert new_train["fake_daily_cat"].isna().sum() == 2


def test_categorical_copies_orders_raw_then_derived_and_rejects_bad_declarations() -> None:
    provider = CategoricalCopies(cols=["gaming_hours"], derived=["fake_game"])
    assert provider.columns() == ["gaming_hours_cat", "fake_game_cat"]
    with pytest.raises(ValueError, match="등록되지 않은 파생 이름"):
        CategoricalCopies(cols=[], derived=["fake_nothing"])
    with pytest.raises(ValueError, match="모두 비었다"):
        CategoricalCopies(cols=[])
    with pytest.raises(ValueError, match="중복"):
        CategoricalCopies(cols=["fake_game"], derived=["fake_game"])


def _plan_frames(n: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "id": np.arange(n),
            "daily_screen_time_hours": rng.uniform(1, 10, n).round(2),
            "social_media_hours": rng.uniform(0, 5, n).round(2),
            "gaming_hours": rng.uniform(0, 5, n).round(2),
            "work_study_hours": rng.uniform(0, 5, n).round(2),
            "stress_level": rng.choice(["low", "mid", "high"], n),
            "addicted_label": np.tile([0, 1], n // 2),
        }
    )
    train.loc[::7, "social_media_hours"] = np.nan
    test = train.drop(columns=["addicted_label"]).copy()
    test["id"] = test["id"] + n
    return train, test


def test_feature_plan_places_derived_copies_before_row_wise_derived() -> None:
    cfg = FeatureConfig(
        base="raw",
        categorical=["stress_level"],
        providers=[
            {"kind": "categorical_copies", "cols": [], "derived": CONSTRAINT_DERIVED_COLS},
            {"kind": "derived", "names": ["other_screen"] + CONSTRAINT_DERIVED_COLS},
        ],
        exclude=[],
    )
    plan = FeaturePlan.from_config(cfg)
    train, test = _plan_frames()
    train, test = plan.apply_dataset_wide(train, test)
    X = plan.build_matrix(train, seed=42)
    cat_cols = [f"{c}_cat" for c in CONSTRAINT_DERIVED_COLS]
    assert [c for c in X.columns if c.endswith("_cat")] == cat_cols
    # 복제 열의 값은 row-wise 파생 열의 값과 같고 범주 dtype만 다르다.
    for col in CONSTRAINT_DERIVED_COLS:
        assert X[f"{col}_cat"].astype("float64").equals(X[col].astype("float64"))
    # 복제 학습 행 경로도 파생 원천에서 범주를 다시 만들 수 있다.
    rebuilt = plan.recompute_training_row_dataset_wide(train.drop(columns=cat_cols), test)
    for col in cat_cols:
        assert rebuilt[col].astype("float64").equals(train[col].astype("float64"))
        assert list(rebuilt[col].cat.categories) == list(test[col].cat.categories)
