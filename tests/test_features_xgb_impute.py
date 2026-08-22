"""XGBoost 조건부 결측 복원 제공자 테스트. (#86)

- 적재 거부: 금지 열(id, 타깃, 플라시보), 중복, cols/cat_cols 겹침, 예측 입력이 비는 구성.
- fit 거부: 수치가 아닌 복원 대상, category dtype이 아닌 cat_cols, 관측 행 없는 열.
- 관측 셀은 원시 값 그대로, 결측 셀만 복원값(기록 대역으로 검증).
- 복원기는 대상 열이 관측된 행으로만 fit하고 예측 입력에서 대상 열 자신은 뺀다.
- 실데이터 성질: 같은 seed는 같은 산출(결정성), 산출은 전 셀 비결측 float64.
- compositions(#90): 이름·입력 열 검증, 복원 행렬 위 식 계산, emit 복원 열 값 불변.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from pipeline.data import ID, TARGET
from pipeline.cpu_budget import XGB_N_JOBS_ENV
from pipeline.features import PLACEBO, XgbImputeAux

COLS = ["daily_screen_time_hours", "sleep_hours"]
CATS = ["stress_level"]


def make_df(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "daily_screen_time_hours": rng.uniform(1, 10, n).round(1),
            "sleep_hours": rng.uniform(4, 9, n).round(1),
            "stress_level": pd.Categorical(rng.choice(["low", "mid", "high"], n)),
        }
    )
    df.loc[::5, "daily_screen_time_hours"] = np.nan
    df.loc[1::7, "sleep_hours"] = np.nan
    return df


class RecordingRegressor:
    """fit/predict 입력을 기록하고 상수를 돌려주는 대역."""

    instances: ClassVar[list[RecordingRegressor]] = []

    def __init__(self, **params) -> None:
        self.params = params
        self.fit_X: pd.DataFrame | None = None
        self.fit_y: pd.Series | None = None
        RecordingRegressor.instances.append(self)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.fit_X, self.fit_y = X, y

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 7.0)


@pytest.fixture
def recording(monkeypatch) -> type[RecordingRegressor]:
    RecordingRegressor.instances = []
    monkeypatch.setattr("xgboost.XGBRegressor", RecordingRegressor)
    return RecordingRegressor


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"cols": [ID, *COLS]}, "쓸 수 없는 열"),
        ({"cols": COLS, "cat_cols": [TARGET]}, "쓸 수 없는 열"),
        ({"cols": [PLACEBO, *COLS]}, "쓸 수 없는 열"),
        ({"cols": [*COLS, COLS[0]]}, "겹치는 열"),
        ({"cols": COLS, "cat_cols": [*CATS, *CATS]}, "겹치는 열"),
        ({"cols": COLS, "cat_cols": [COLS[0]]}, "겹치는 열"),
        ({"cols": [COLS[0]]}, "예측 입력이 하나 이상"),
        ({"cols": COLS, "emit": ["age"]}, "cols에 없는 열"),
        ({"cols": COLS, "emit": []}, "비어 있지 않은 목록"),
        ({"cols": COLS, "emit": [COLS[0], COLS[0]]}, "비어 있지 않은 목록"),
        ({"cols": COLS, "compositions": ["nope"]}, "알 수 없는 composition"),
        ({"cols": COLS, "compositions": []}, "비어 있지 않은 목록"),
        (
            {"cols": COLS, "compositions": ["awake_screen_frac", "awake_screen_frac"]},
            "비어 있지 않은 목록",
        ),
        ({"cols": COLS, "compositions": ["resid"]}, "cols에 없다"),
    ],
)
def test_init_rejections(kwargs, match):
    with pytest.raises(ValueError, match=match):
        XgbImputeAux(**kwargs)


def test_single_col_with_cat_predictor_is_allowed():
    provider = XgbImputeAux(cols=[COLS[0]], cat_cols=CATS)
    assert provider.columns() == [f"{COLS[0]}_xgb_recon"]


def test_fit_rejects_non_numeric_target_and_non_category_cat(recording):
    df = make_df()
    df["stress_str"] = df["stress_level"].astype(str)
    with pytest.raises(ValueError, match="수치 열 전용"):
        XgbImputeAux(cols=["stress_str", *COLS]).fit(df, seed=42)
    with pytest.raises(ValueError, match="category dtype"):
        XgbImputeAux(cols=COLS, cat_cols=["stress_str"]).fit(df, seed=42)


def test_fit_rejects_column_without_observed_rows(recording):
    df = make_df()
    df["daily_screen_time_hours"] = np.nan
    with pytest.raises(ValueError, match="관측 행이 없어"):
        XgbImputeAux(cols=COLS, cat_cols=CATS).fit(df, seed=42)


def test_fit_uses_only_observed_rows_and_excludes_target_column(recording):
    df = make_df()
    provider = XgbImputeAux(cols=COLS, cat_cols=CATS)
    provider.fit(df, seed=42)
    assert len(recording.instances) == len(COLS)
    for col, model in zip(COLS, recording.instances):
        assert model.params["random_state"] == 42
        assert model.params["enable_categorical"] is True
        assert not model.fit_y.isna().any()  # 대상 열 관측 행 전용
        assert len(model.fit_y) == df[col].notna().sum()
        expected = [c for c in COLS if c != col] + CATS
        assert list(model.fit_X.columns) == expected  # 대상 열 자신은 예측 입력에서 뺀다


def test_fit_limits_only_xgboost_when_parallel_worker_budget_is_present(
    recording, monkeypatch
):
    monkeypatch.setenv(XGB_N_JOBS_ENV, "30")
    provider = XgbImputeAux(cols=COLS, cat_cols=CATS)

    provider.fit(make_df(), seed=42)

    assert {model.params["n_jobs"] for model in recording.instances} == {30}


def test_emit_subset_keeps_full_predictor_inputs_and_skips_non_emitted_models(recording):
    df = make_df()
    df["age"] = np.arange(len(df), dtype="float64")
    provider = XgbImputeAux(cols=[*COLS, "age"], cat_cols=CATS, emit=COLS)
    provider.fit(df, seed=42)
    out = provider.transform(df)
    assert list(out.columns) == [f"{c}_xgb_recon" for c in COLS]  # age 복원 열은 없다
    assert len(recording.instances) == len(COLS)  # 내보내지 않는 열의 복원기는 만들지 않는다
    for col, model in zip(COLS, recording.instances):
        # 예측 입력은 emit이 아니라 cols 전체 기준이다(대상 열 자신만 뺀다).
        assert list(model.fit_X.columns) == [c for c in [*COLS, "age"] if c != col] + CATS


def test_emitted_recon_values_are_identical_to_full_emit_run():
    # 게이트 미달 열을 emit에서 빼도 예측 입력이 cols 전체로 같으므로,
    # 남는 복원 열의 값은 전체 구성과 동일해야 한다. (#86 축소 변형의 근거)
    df = make_df()
    df["age"] = np.arange(len(df), dtype="float64") % 23
    full = XgbImputeAux(cols=[*COLS, "age"], cat_cols=CATS)
    full.fit(df, seed=42)
    reduced = XgbImputeAux(cols=[*COLS, "age"], cat_cols=CATS, emit=COLS)
    reduced.fit(df, seed=42)
    pd.testing.assert_frame_equal(
        full.transform(df)[reduced.columns()], reduced.transform(df)
    )


def test_observed_cells_keep_raw_and_missing_cells_get_predictions(recording):
    df = make_df()
    provider = XgbImputeAux(cols=COLS, cat_cols=CATS)
    provider.fit(df, seed=42)
    out = provider.transform(df)
    assert list(out.columns) == provider.columns()
    for col in COLS:
        observed = df[col].notna()
        rec = out[f"{col}_xgb_recon"]
        assert (rec[observed] == df.loc[observed, col]).all()
        assert (rec[~observed] == 7.0).all()  # 결측 셀만 복원값
        assert rec.dtype == "float64"


def write_test_csv(tmp_path, df: pd.DataFrame) -> tuple[str, str]:
    import hashlib

    path = tmp_path / "toy_test.csv"
    df.to_csv(path, index=False)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def test_transductive_requires_matching_hash(tmp_path):
    test_df = make_df(30).drop(columns=["stress_level"]).assign(stress_level="low")
    path, sha = write_test_csv(tmp_path, test_df)
    with pytest.raises(ValueError, match="해시 고정"):
        XgbImputeAux(cols=COLS, cat_cols=CATS, transductive_test_path=path)
    with pytest.raises(ValueError, match="해시 불일치"):
        XgbImputeAux(
            cols=COLS, cat_cols=CATS, transductive_test_path=path,
            transductive_test_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="없이 쓸 수 없다"):
        XgbImputeAux(cols=COLS, cat_cols=CATS, transductive_test_sha256=sha)


def test_transductive_fit_adds_test_observed_rows_with_aligned_categories(
    recording, tmp_path
):
    df = make_df()
    test_df = make_df(40).drop(columns=["stress_level"]).assign(stress_level="low")
    test_df.loc[0, "stress_level"] = "high"
    path, sha = write_test_csv(tmp_path, test_df)
    provider = XgbImputeAux(
        cols=COLS, cat_cols=CATS,
        transductive_test_path=path, transductive_test_sha256=sha,
    )
    provider.fit(df, seed=42)
    for col, model in zip(COLS, recording.instances):
        # 학습 표본 = 훈련 부분 관측 행 + test 관측 행(검증 fold 행은 없음).
        assert len(model.fit_y) == df[col].notna().sum() + test_df[col].notna().sum()
        cat = model.fit_X["stress_level"]
        assert isinstance(cat.dtype, pd.CategoricalDtype)
        assert list(cat.dtype.categories) == list(df["stress_level"].dtype.categories)


def test_transductive_rejects_unseen_category_values(recording, tmp_path):
    df = make_df()
    test_df = make_df(20).drop(columns=["stress_level"]).assign(stress_level="unknown")
    path, sha = write_test_csv(tmp_path, test_df)
    provider = XgbImputeAux(
        cols=COLS, cat_cols=CATS,
        transductive_test_path=path, transductive_test_sha256=sha,
    )
    with pytest.raises(ValueError, match="train에 없는 값"):
        provider.fit(df, seed=42)


def test_composition_columns_computed_on_recon_matrix(recording):
    df = make_df()
    provider = XgbImputeAux(
        cols=COLS, cat_cols=CATS, emit=[COLS[0]], compositions=["awake_screen_frac"]
    )
    provider.fit(df, seed=42)
    out = provider.transform(df)
    # emit 복원 열 다음에 조성 열. sleep은 emit에 없어도 조성 입력이라 복원기를 만든다.
    assert list(out.columns) == [f"{COLS[0]}_xgb_recon", "imp_awake_screen_frac"]
    assert len(recording.instances) == 2
    recon = {c: df[c].astype("float64").fillna(7.0) for c in COLS}
    expected = recon["daily_screen_time_hours"] / (24 - recon["sleep_hours"])
    pd.testing.assert_series_equal(
        out["imp_awake_screen_frac"], expected, check_names=False
    )
    assert out["imp_awake_screen_frac"].dtype == "float64"


def test_bivariate_composition_formulas_follow_notebook_definitions(recording):
    # szymonkapiski build_continuous 이식 조성(#267). 복원 행렬 위 순수 산술이며
    # 원본의 epsilon 방어는 규율대로 뺀다.
    rng = np.random.default_rng(1)
    n = 90
    df = make_df(n)
    df["gaming_hours"] = rng.uniform(0, 5, n).round(1)
    df["weekend_screen_time"] = rng.uniform(1, 12, n).round(1)
    df.loc[2::9, "gaming_hours"] = np.nan
    df.loc[3::11, "weekend_screen_time"] = np.nan
    cols = [*COLS, "gaming_hours", "weekend_screen_time"]
    names = [
        "screen_over_sleep",
        "screen_minus_sleep",
        "gaming_over_daily",
        "weekend_minus_daily",
        "screen_mean_dw",
    ]
    provider = XgbImputeAux(cols=cols, cat_cols=CATS, emit=[cols[0]], compositions=names)
    provider.fit(df, seed=42)
    out = provider.transform(df)
    assert list(out.columns) == [
        f"{cols[0]}_xgb_recon",
        *(f"imp_{name}" for name in names),
    ]
    recon = {c: df[c].astype("float64").fillna(7.0) for c in cols}
    daily = recon["daily_screen_time_hours"]
    sleep = recon["sleep_hours"]
    gaming = recon["gaming_hours"]
    weekend = recon["weekend_screen_time"]
    expected = {
        "imp_screen_over_sleep": daily / sleep,
        "imp_screen_minus_sleep": daily - sleep,
        "imp_gaming_over_daily": gaming / daily,
        "imp_weekend_minus_daily": weekend - daily,
        "imp_screen_mean_dw": (daily + weekend) / 2,
    }
    for column, series in expected.items():
        pd.testing.assert_series_equal(out[column], series, check_names=False)


def test_emitted_recon_values_are_unchanged_by_compositions():
    # 조성이 sleep 복원기를 추가해도 열마다 독립 fit이므로 emit 복원 열 값은 같아야 한다.
    # (#86 채택 열의 보존 근거, #90)
    df = make_df()
    plain = XgbImputeAux(cols=COLS, cat_cols=CATS, emit=[COLS[0]])
    plain.fit(df, seed=42)
    with_comp = XgbImputeAux(
        cols=COLS, cat_cols=CATS, emit=[COLS[0]], compositions=["awake_screen_frac"]
    )
    with_comp.fit(df, seed=42)
    pd.testing.assert_frame_equal(
        plain.transform(df), with_comp.transform(df)[plain.columns()]
    )


def test_fit_transform_is_deterministic_per_seed():
    df = make_df()

    def run() -> pd.DataFrame:
        provider = XgbImputeAux(cols=COLS, cat_cols=CATS)
        provider.fit(df, seed=42)
        return provider.transform(df)

    first, second = run(), run()
    pd.testing.assert_frame_equal(first, second)
    assert first.notna().all().all()


def test_continuous_expansion_compositions_match_notebook_formulas(recording):
    # szymonkapiski build_continuous 이식 7열(#265): 복원 행렬 위 식이 원문 산술과
    # 같은지(epsilon 제거 제외), 포화 앵커의 클리핑 경계가 3.0·9.5인지 고정한다.
    rng = np.random.default_rng(1)
    n = 90
    df = pd.DataFrame(
        {
            "daily_screen_time_hours": rng.uniform(1, 12, n).round(1),
            "sleep_hours": rng.uniform(4, 9, n).round(1),
            "gaming_hours": rng.uniform(0, 5, n).round(1),
            "weekend_screen_time": rng.uniform(1, 14, n).round(1),
            "stress_level": pd.Categorical(rng.choice(["low", "mid", "high"], n)),
        }
    )
    df.loc[::5, "daily_screen_time_hours"] = np.nan
    df.loc[1::7, "sleep_hours"] = np.nan
    names = [
        "over_9h",
        "under_3h",
        "screen_over_sleep",
        "screen_minus_sleep",
        "gaming_over_daily",
        "weekend_minus_daily",
        "screen_mean_dw",
    ]
    cols = [
        "daily_screen_time_hours",
        "sleep_hours",
        "gaming_hours",
        "weekend_screen_time",
    ]
    provider = XgbImputeAux(
        cols=cols, cat_cols=CATS, emit=[cols[0]], compositions=names
    )
    provider.fit(df, seed=42)
    out = provider.transform(df)
    assert list(out.columns) == [f"{cols[0]}_xgb_recon", *[f"imp_{n}" for n in names]]
    recon = {c: df[c].astype("float64").fillna(7.0) for c in cols}
    d, sl = recon["daily_screen_time_hours"], recon["sleep_hours"]
    g, wk = recon["gaming_hours"], recon["weekend_screen_time"]
    expected = {
        "imp_over_9h": (d - 9.5).clip(lower=0.0),
        "imp_under_3h": (3.0 - d).clip(lower=0.0),
        "imp_screen_over_sleep": d / sl,
        "imp_screen_minus_sleep": d - sl,
        "imp_gaming_over_daily": g / d,
        "imp_weekend_minus_daily": wk - d,
        "imp_screen_mean_dw": (d + wk) / 2.0,
    }
    for name, series in expected.items():
        pd.testing.assert_series_equal(out[name], series, check_names=False)
        assert out[name].dtype == "float64"
    assert (out["imp_over_9h"] >= 0).all()
    assert (out["imp_under_3h"] >= 0).all()
