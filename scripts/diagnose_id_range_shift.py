"""ID 구간별 미세 분포 이동의 진입 진단. (#55)

전역 적대적 검증이 놓칠 수 있는 행 위치(id 순서)별 이동이 있는지 측정한다.
champion과 경쟁하는 실험이 아니라 측정만 하는 진입 진단이다.

측정 항목:
1. train을 id 순서 10구간으로 나눠 구간별 타깃 비율, 결측률, 주요 수치 평균.
2. 구간별 train-vs-test 적대적 AUC. 결측 영향 분리를 위해 raw(NaN 유지)와
   filled(중앙값·최빈값 대치, 결측 신호 제거) 두 변형으로 측정한다.
3. 전역 적대적 모델의 P(test)를 train 구간별로 평균해 꼬리 구간이 test와
   더 가까운지 본다.
4. train 내부 위치 적대적 검증(전반 vs 후반, 머리 20% vs 꼬리 20%) AUC.
5. 고정 Stratified 5-fold의 fold x id 구간 구성이 균일한지, fold별 타깃
   비율이 같은지 확인한다.

사용법:
    uv run python scripts/diagnose_id_range_shift.py
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
FOLDS_PATH = Path("artifacts/folds.parquet")
TARGET = "addicted_label"
SEED = 42
N_BINS = 10

NUMERIC = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day", "app_opens_per_day",
    "weekend_screen_time",
]
CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
FEATURES = NUMERIC + CATEGORICAL

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "verbosity": -1,
    "seed": SEED,
    "force_row_wise": True,
}


def prepare(df: pd.DataFrame, fill: bool, fill_stats: dict | None = None) -> pd.DataFrame:
    """적대적 검증용 특성 행렬. fill=True면 결측 신호를 제거해 값 분포 이동만 남긴다."""
    out = df[FEATURES].copy()
    if fill:
        assert fill_stats is not None
        for col in NUMERIC:
            out[col] = out[col].fillna(fill_stats[col])
        for col in CATEGORICAL:
            out[col] = out[col].fillna(fill_stats[col])
    for col in CATEGORICAL:
        out[col] = out[col].astype("category")
    return out


def fill_stats_from(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """대치 기준은 train+test 합본의 중앙값·최빈값. 진단 전용이라 누출 걱정이 없다."""
    both = pd.concat([train[FEATURES], test[FEATURES]], ignore_index=True)
    stats: dict = {}
    for col in NUMERIC:
        stats[col] = both[col].median()
    for col in CATEGORICAL:
        stats[col] = both[col].mode().iloc[0]
    return stats


def adversarial_auc(a: pd.DataFrame, b: pd.DataFrame, fill: bool, fill_stats: dict) -> float:
    """a(라벨 0) vs b(라벨 1)의 3-fold 적대적 AUC."""
    x = pd.concat(
        [prepare(a, fill, fill_stats), prepare(b, fill, fill_stats)], ignore_index=True
    )
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(x, y):
        model = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(x.iloc[tr_idx], label=y[tr_idx]),
            num_boost_round=100,
        )
        oof[va_idx] = model.predict(x.iloc[va_idx])
    return float(roc_auc_score(y, oof))


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    folds = pd.read_parquet(FOLDS_PATH)
    train = train.merge(folds, on="id", validate="one_to_one")
    train = train.sort_values("id").reset_index(drop=True)
    train["id_bin"] = pd.qcut(train["id"], N_BINS, labels=False)
    stats = fill_stats_from(train, test)
    rng = np.random.default_rng(SEED)

    # 1. 구간별 타깃 비율과 주요 분포 -------------------------------------------
    print(f"== 1. id {N_BINS}구간별 타깃 비율·결측률 (train {len(train)}행) ==")
    global_rate = train[TARGET].mean()
    rows = []
    for b, g in train.groupby("id_bin"):
        rate = g[TARGET].mean()
        se = np.sqrt(rate * (1 - rate) / len(g))
        rows.append({
            "bin": b,
            "n": len(g),
            "target_rate": rate,
            "ci95_half": 1.96 * se,
            "dev_from_global": rate - global_rate,
            "missing_rate": g[FEATURES].isna().to_numpy().mean(),
            "screen_mean": g["daily_screen_time_hours"].mean(),
            "sleep_mean": g["sleep_hours"].mean(),
        })
    per_bin = pd.DataFrame(rows)
    print(per_bin.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(train["id_bin"], train[TARGET]))
    print(f"global_rate={global_rate:.5f}  chi2={chi2:.2f}  p={pval:.4f}  "
          f"max|dev|={per_bin['dev_from_global'].abs().max():.5f}")

    # 2. 구간별 적대적 AUC ------------------------------------------------------
    print(f"\n== 2. 구간별 train-vs-test 적대적 AUC (test에서 구간 크기만큼 표본) ==")
    for fill in (False, True):
        label = "filled" if fill else "raw"
        aucs = []
        for b, g in train.groupby("id_bin"):
            t = test.iloc[rng.choice(len(test), size=len(g), replace=False)]
            aucs.append(adversarial_auc(g, t, fill, stats))
        line = "  ".join(f"b{b}={a:.4f}" for b, a in enumerate(aucs))
        print(f"[{label}] {line}")
        print(f"[{label}] mean={np.mean(aucs):.4f}  spread(max-min)={np.ptp(aucs):.4f}")

    # 전역 적대적 AUC (기준선)
    for fill in (False, True):
        label = "filled" if fill else "raw"
        sub = train.iloc[rng.choice(len(train), size=len(test), replace=False)]
        print(f"전역 train-vs-test [{label}]: {adversarial_auc(sub, test, fill, stats):.4f}")

    # 3. 전역 적대적 모델의 P(test)를 구간별 평균 -------------------------------
    print(f"\n== 3. 전역 적대적 모델 P(test)의 train 구간별 평균 ==")
    x = pd.concat([prepare(train, True, stats), prepare(test, True, stats)],
                  ignore_index=True)
    y = np.r_[np.zeros(len(train)), np.ones(len(test))]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(x, y):
        model = lgb.train(LGB_PARAMS, lgb.Dataset(x.iloc[tr_idx], label=y[tr_idx]),
                          num_boost_round=100)
        oof[va_idx] = model.predict(x.iloc[va_idx])
    train_p = pd.Series(oof[: len(train)], index=train.index)
    by_bin = train_p.groupby(train["id_bin"]).mean()
    print(by_bin.to_string(float_format=lambda v: f"{v:.5f}"))
    corr = np.corrcoef(train["id_bin"], train_p)[0, 1]
    print(f"corr(id_bin, P(test))={corr:.5f}  (양수면 꼬리 구간이 test에 더 가까움)")

    # 4. train 내부 위치 적대적 검증 -------------------------------------------
    print(f"\n== 4. train 내부 위치 적대적 AUC ==")
    half = len(train) // 2
    fifth = len(train) // 5
    for name, a, b in [
        ("전반 vs 후반", train.iloc[:half], train.iloc[half:]),
        ("머리20% vs 꼬리20%", train.iloc[:fifth], train.iloc[-fifth:]),
    ]:
        for fill in (False, True):
            label = "filled" if fill else "raw"
            print(f"{name} [{label}]: {adversarial_auc(a, b, fill, stats):.4f}")

    # 5. 고정 5-fold의 구간 구성 ------------------------------------------------
    print(f"\n== 5. 고정 Stratified 5-fold x id 구간 구성 ==")
    comp = pd.crosstab(train["fold"], train["id_bin"], normalize="index")
    print(comp.to_string(float_format=lambda v: f"{v:.4f}"))
    fold_rate = train.groupby("fold")[TARGET].mean()
    print("fold별 타깃 비율:")
    print(fold_rate.to_string(float_format=lambda v: f"{v:.5f}"))
    fold_pos = train.groupby("fold")["id"].mean()
    print("fold별 id 평균 (균일해야 정상):")
    print(fold_pos.to_string(float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main()
