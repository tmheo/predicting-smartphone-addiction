"""쌍 결합 TE·CE 후보 66쌍의 약식 검증. (#48 규약, #51 실행)

사용법:
    uv run python scripts/screen_pairs.py

약식 검증은 선별 전용이다: 커밋된 folds.parquet의 fold 0 하나만 검증에 쓰고
나머지 4개 fold 전체 행으로 학습한다. 행 표본 축소는 하지 않는다.
가벼운 학습 설정(learning_rate 0.1, early stopping 100, num_leaves 255, 내부 TE 5-fold)
외에는 champion(exp011_resid_pair) 설정과 동일하다.

기준 실행 두 개를 만든다.
- base_plain: champion 피처 구성 그대로 (#48이 말하는 약식 기준 실행).
- base_canary: champion 피처 + 쌍 카나리아 TE. 후보 실행과 피처 차이가 후보 쌍뿐이라
  Δ가 후보 쌍의 기여만 잰다. 선별 게이트(Δ ≥ +0.0001, 상위 8)는 이 기준의 Δ로 판정하고
  base_plain 대비 Δ는 보조로 함께 기록한다.

후보 실행 하나 = champion 피처 + 해당 쌍 TE + 해당 쌍 CE + 쌍 카나리아 TE.
쌍 카나리아(placebo_noise × weekend_screen_time)의 gain이 플라시보를 넘으면 그 실행은 무효다.

결과는 run-logs/pair_screen.csv에 증분 기록한다. 이미 기록된 쌍은 건너뛴다(중단 재개용).
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import data, model as model_mod
from pipeline.config import FeatureConfig, ModelConfig
from pipeline.cv import _with_built_columns
from pipeline.features import PLACEBO
from pipeline.plan import FeaturePlan

SEED = 42
VALID_FOLD = 0
OUT = Path("run-logs/pair_screen.csv")

# 후보 컬럼: 정확값 수치 9 + 범주 3. (#48)
NUM_COLS = [
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
CAT_COLS = ["gender", "stress_level", "academic_work_impact"]
PAIRS = list(itertools.combinations(NUM_COLS + CAT_COLS, 2))

# 쌍 카나리아: 최고 카디널리티 컬럼과 플라시보의 쌍 TE. 모든 후보 실행에 상시 포함. (#48)
CANARY_PAIR = [PLACEBO, "weekend_screen_time"]

# champion exp011_resid_pair의 단일 TE 대상 (gaming_hours 제외 8종 + 플라시보 카나리아).
CHAMPION_TE_COLS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    PLACEBO,
]

QUICK_MODEL = ModelConfig(
    kind="lightgbm",
    params={
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.1,
        "num_leaves": 255,
        "n_estimators": 10000,
    },
    fit={"early_stopping_rounds": 100},
)


def make_plan(te_cols: list, pair: tuple | None = None) -> FeaturePlan:
    """champion 피처 구성의 약식 계획. pair를 주면 그 쌍의 CE 제공자를 함께 켠다."""
    providers: list[dict] = []
    if pair is not None:
        providers.append({"kind": "pair_ce", "pairs": [list(pair)]})
    providers.append({"kind": "derived", "names": ["other_screen", "screen_slack"]})
    providers.append({"kind": "target_encoding", "inner_folds": 5, "cols": te_cols})
    return FeaturePlan.from_config(
        FeatureConfig(base="raw", categorical=CAT_COLS, providers=providers)
    )


def quick_run(plan: FeaturePlan, train: pd.DataFrame) -> tuple[float, pd.Series]:
    """fold 0 검증의 약식 실행 하나. (fold0 AUC, 피처별 gain)을 돌려준다.

    plan은 apply_dataset_wide를 이미 거친 상태여야 한다.
    """
    X = plan.build_matrix(train, SEED)
    y = train[data.TARGET]
    transformers = plan.fold_fit_transformers()
    train_ff = _with_built_columns(train, X)
    va_idx = train.index[train["fold"] == VALID_FOLD]
    tr_idx = train.index[train["fold"] != VALID_FOLD]
    for t in transformers:
        t.fit(train_ff.loc[tr_idx], SEED)
    X_fold = plan.add_fold_fit_columns(X, train_ff)
    model, va_pred = model_mod.train_one_fold(
        QUICK_MODEL, X_fold.loc[tr_idx], y.loc[tr_idx], X_fold.loc[va_idx], y.loc[va_idx], SEED
    )
    auc = float(roc_auc_score(y.loc[va_idx], va_pred))
    gain = model_mod.gain_importance(model).set_index("feature")["gain"]
    return auc, gain


def main() -> None:
    train = data.load_csv(Path("data/train.csv"))
    test = data.load_csv(Path("data/test.csv"))
    data.align_categories(train, test, CAT_COLS)
    train = data.attach_folds(train, Path("artifacts/folds.parquet"))

    done: set[str] = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT)["name"])
    else:
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(
            "name,auc,delta_vs_canary_base,delta_vs_plain_base,"
            "te_gain,ce_gain,canary_gain,placebo_gain,valid,elapsed_sec\n"
        )

    def record(name: str, auc: float, row: dict) -> None:
        with OUT.open("a") as f:
            f.write(
                f"{name},{auc:.10f},{row.get('d_canary', '')},{row.get('d_plain', '')},"
                f"{row.get('te_gain', '')},{row.get('ce_gain', '')},"
                f"{row.get('canary_gain', '')},{row.get('placebo_gain', '')},"
                f"{row.get('valid', '')},{row.get('elapsed', '')}\n"
            )

    # 기준 실행 두 개. 이미 기록돼 있으면 CSV 값을 재사용한다.
    bases: dict[str, float] = {}
    for name, te_cols in [
        ("base_plain", CHAMPION_TE_COLS),
        ("base_canary", [*CHAMPION_TE_COLS, CANARY_PAIR]),
    ]:
        if name in done:
            df = pd.read_csv(OUT)
            bases[name] = float(df.loc[df["name"] == name, "auc"].iloc[0])
            print(f"{name}: 기록 재사용 auc={bases[name]:.6f}")
            continue
        t0 = time.time()
        plan = make_plan(te_cols)
        train_b, _ = plan.apply_dataset_wide(train, test)
        auc, gain = quick_run(plan, train_b)
        bases[name] = auc
        record(
            name,
            auc,
            {
                "placebo_gain": f"{gain.get(PLACEBO, 0.0):.1f}",
                "canary_gain": f"{gain.get('__'.join(CANARY_PAIR) + '_te', 0.0):.1f}"
                if name == "base_canary"
                else "",
                "valid": "",
                "elapsed": f"{time.time() - t0:.0f}",
            },
        )
        print(f"{name}: auc={auc:.6f} ({time.time() - t0:.0f}s)")

    canary_name = "__".join(CANARY_PAIR) + "_te"
    for i, pair in enumerate(PAIRS):
        name = "__".join(pair)
        if name in done:
            continue
        t0 = time.time()
        te_cols = [*CHAMPION_TE_COLS, list(pair), CANARY_PAIR]
        plan = make_plan(te_cols, pair)
        train_c, _ = plan.apply_dataset_wide(train, test)
        auc, gain = quick_run(plan, train_c)
        placebo_gain = float(gain.get(PLACEBO, 0.0))
        canary_gain = float(gain.get(canary_name, 0.0))
        valid = canary_gain < placebo_gain
        record(
            name,
            auc,
            {
                "d_canary": f"{auc - bases['base_canary']:+.10f}",
                "d_plain": f"{auc - bases['base_plain']:+.10f}",
                "te_gain": f"{gain.get(name + '_te', 0.0):.1f}",
                "ce_gain": f"{gain.get(name + '_ce', 0.0):.1f}",
                "canary_gain": f"{canary_gain:.1f}",
                "placebo_gain": f"{placebo_gain:.1f}",
                "valid": str(valid),
                "elapsed": f"{time.time() - t0:.0f}",
            },
        )
        print(
            f"[{i + 1}/{len(PAIRS)}] {name}: auc={auc:.6f} "
            f"d={auc - bases['base_canary']:+.6f} valid={valid} ({time.time() - t0:.0f}s)"
        )


if __name__ == "__main__":
    main()
