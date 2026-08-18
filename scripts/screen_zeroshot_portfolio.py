"""AutoGluon zeroshot portfolio 2025 설정의 약식 검증. (#197)

포트폴리오 출처(하이퍼파라미터 dict를 그대로 옮겼다):
- 저장소: https://github.com/autogluon/autogluon (Apache License 2.0)
- 파일: tabular/src/autogluon/tabular/configs/zeroshot/zeroshot_portfolio_2025.py
- 마지막 변경 커밋: 2d7e6056b8b64dc44114faf652d4c99ec3c3770f (2026-01-08)
- 원본 파일 SHA-256: e2ffbe42850c6aa8cbd5c30df84c77a91f7bfe679af8a1806d54c30775241b27
- 원본 19설정 중 기존 adapter가 있는 GBM 3, CAT 5, XGB 2, TABM 6의 16설정만 이식했다.
  REALTABPFN-V2, TABICL, MITRA는 adapter가 없거나 기존 진입 진단에서 탈락한 계열이라 뺐다.
- 이식하며 바꾼 것: ag_args 제거, XGB enable_categorical 제거(adapter가 소유),
  TabM amp 제거(adapter 미지원), TabM batch_size "auto"는 AutoGluon의 표본 수 규칙으로
  1024를 대입(학습 55.3만 행 ≥ 108,000), TabM n_epochs 상한 200 부여(원본은 무제한 +
  patience 16), n_estimators/iterations 10000과 early stopping 200은 저장소 규약을 따랐다.
- AutoGluon 기본값 설정(CAT _default, TABM _default)은 AutoGluon 소스의 기본값을
  명시값으로 풀어 적었다(catboost/hyperparameters/parameters.py, tabm/_tabm_internal.py).

약식 검증은 선별 전용이다(#48 규약, scripts/screen_pairs.py 선례):
- 커밋된 folds.parquet의 fold 0 하나만 검증에 쓰고 나머지 4개 fold 전체 행으로 학습한다.
- 피처 계획은 champion 계열(exp065_tabm과 동일: derived + 내부 10-fold TE +
  constrained_impute_aux + xgb_impute_aux 조성 5열)로 전 설정 공통이다.
  fold-fit 변환기는 설정과 무관하므로 한 번만 학습해 전 설정이 공유한다.
- 같은 프로토콜의 기준 실행(base_lgb, base_xgb, base_cat, base_tabm)을 같은 CSV에
  기록해 계열별 Δ를 잰다. base_tabm은 exp065 모델 설정에서 n_seed_avg만 1로 줄인
  판이다(약식 선별은 초기화 평균 없이 설정 간 순위만 본다).
- MLflow 실행을 만들지 않고 champion/pool 장부를 바꾸지 않는다.

사용법:
    uv run python scripts/screen_zeroshot_portfolio.py                 # gbdt 계열(CPU)
    uv run python scripts/screen_zeroshot_portfolio.py --family tabm   # TabM 계열(GPU 필요)
    uv run python scripts/screen_zeroshot_portfolio.py --only ag25_gbm_r33

결과는 run-logs/zeroshot_portfolio_screen.csv에 증분 기록한다. 기록된 설정은 건너뛴다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import data
from pipeline import model as model_mod
from pipeline.config import FeatureConfig, ModelConfig
from pipeline.features import PLACEBO
from pipeline.judgment import check_canaries
from pipeline.plan import FeaturePlan, prepare_fold_fit_input

SEED = 42
VALID_FOLD = 0
OUT = Path("run-logs/zeroshot_portfolio_screen.csv")

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

# exp065_tabm의 TE 대상(champion 계열: gaming_hours 제외 8종 + 플라시보 카나리아).
TE_COLS = [
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

CANARY = f"{PLACEBO}_te"

# TabM 공통: 약식 선별은 fold 내 초기화 평균 없이 1판만 본다.
_TABM_SCREEN = {"n_seed_avg": 1, "perm_repeats": 1, "perm_sample": 1000}
# AutoGluon TabM 포트폴리오 공통 값. batch_size "auto"는 표본 수 규칙으로 1024,
# n_epochs 무제한은 상한 200으로 옮겼다(patience 16이 먼저 멈춘다).
_AG_TABM_COMMON = {
    "arch_type": "tabm-mini",
    "num_emb_type": "pwl",
    "batch_size": 1024,
    "n_epochs": 200,
    "patience": 16,
    "tabm_k": 32,
    "share_training_batches": False,
    "gradient_clipping_norm": 1.0,
    **_TABM_SCREEN,
}

_LGB_COMMON = {"objective": "binary", "metric": "auc", "n_estimators": 10000}
_XGB_COMMON = {"tree_method": "hist", "n_estimators": 10000, "eval_metric": "auc"}
_CAT_COMMON = {"iterations": 10000, "eval_metric": "AUC"}
_ES200 = {"early_stopping_rounds": 200}

# (이름, 계열, ModelConfig). ag25_* 이름의 접미사(_r33 등)는 원본 name_suffix다.
CANDIDATES: list[tuple[str, str, ModelConfig]] = [
    # 기준 실행: 같은 프로토콜에서의 저장소 기존 설정.
    (
        "base_lgb",
        "gbdt",
        ModelConfig(
            kind="lightgbm",
            params={**_LGB_COMMON, "learning_rate": 0.05, "num_leaves": 255},
            fit=_ES200,
        ),
    ),
    (
        "base_xgb",
        "gbdt",
        ModelConfig(
            kind="xgboost",
            params={
                **_XGB_COMMON,
                "grow_policy": "depthwise",
                "max_depth": 8,
                "learning_rate": 0.05,
            },
            fit=_ES200,
        ),
    ),
    (
        "base_cat",
        "gbdt",
        ModelConfig(
            kind="catboost",
            params={**_CAT_COMMON, "depth": 8, "learning_rate": 0.05},
            fit=_ES200,
        ),
    ),
    (
        "base_tabm",
        "tabm",
        ModelConfig(
            kind="tabm",
            params={
                "arch_type": "tabm-mini-normal",
                "tabm_k": 16,
                "num_emb_type": "pwl",
                "d_embedding": 16,
                "batch_size": 128,
                "lr": 5.0e-4,
                "n_epochs": 150,
                "dropout": 0.02,
                "d_block": 160,
                "n_blocks": 10,
                "weight_decay": 1.0e-2,
                "patience": 5,
                **_TABM_SCREEN,
            },
            fit={},
        ),
    ),
    # GBM(LightGBM) 3설정.
    (
        "ag25_gbm_r33",
        "gbdt",
        ModelConfig(
            kind="lightgbm",
            params={
                **_LGB_COMMON,
                "bagging_fraction": 0.9625293420216,
                "bagging_freq": 1,
                "cat_l2": 0.1236875455555,
                "cat_smooth": 68.8584757332856,
                "extra_trees": False,
                "feature_fraction": 0.6189215809382,
                "lambda_l1": 0.1641757352921,
                "lambda_l2": 0.6937755557881,
                "learning_rate": 0.0154031028561,
                "max_cat_to_onehot": 17,
                "min_data_in_leaf": 1,
                "min_data_per_group": 30,
                "num_leaves": 68,
            },
            fit=_ES200,
        ),
    ),
    (
        "ag25_gbm_r21",
        "gbdt",
        ModelConfig(
            kind="lightgbm",
            params={
                **_LGB_COMMON,
                "bagging_fraction": 0.7218730663234,
                "bagging_freq": 1,
                "cat_l2": 0.0296205152578,
                "cat_smooth": 0.0010255271303,
                "extra_trees": False,
                "feature_fraction": 0.4557131604374,
                "lambda_l1": 0.5219704038237,
                "lambda_l2": 0.1070959487853,
                "learning_rate": 0.0055891584996,
                "max_cat_to_onehot": 71,
                "min_data_in_leaf": 50,
                "min_data_per_group": 10,
                "num_leaves": 30,
            },
            fit=_ES200,
        ),
    ),
    (
        "ag25_gbm_r11",
        "gbdt",
        ModelConfig(
            kind="lightgbm",
            params={
                **_LGB_COMMON,
                "bagging_fraction": 0.775784726514,
                "bagging_freq": 1,
                "cat_l2": 0.3888471449178,
                "cat_smooth": 0.0057144748021,
                "extra_trees": True,
                "feature_fraction": 0.7732354787904,
                "lambda_l1": 0.2211002452568,
                "lambda_l2": 1.1318405980187,
                "learning_rate": 0.0090151778542,
                "max_cat_to_onehot": 15,
                "min_data_in_leaf": 4,
                "min_data_per_group": 15,
                "num_leaves": 2,
            },
            fit=_ES200,
        ),
    ),
    # CAT(CatBoost) 5설정.
    (
        "ag25_cat_default",
        "gbdt",
        ModelConfig(
            kind="catboost",
            params={**_CAT_COMMON, "learning_rate": 0.05},
            fit=_ES200,
        ),
    ),
    (
        "ag25_cat_r51",
        "gbdt",
        ModelConfig(
            kind="catboost",
            params={
                **_CAT_COMMON,
                "boosting_type": "Plain",
                "bootstrap_type": "Bernoulli",
                "colsample_bylevel": 0.8771035272558,
                "depth": 7,
                "grow_policy": "SymmetricTree",
                "l2_leaf_reg": 2.0107286863021,
                "leaf_estimation_iterations": 2,
                "learning_rate": 0.0058424016622,
                "max_bin": 254,
                "max_ctr_complexity": 4,
                "model_size_reg": 0.1307400355809,
                "one_hot_max_size": 23,
                "subsample": 0.809527841437,
            },
            fit=_ES200,
        ),
    ),
    (
        "ag25_cat_r10",
        "gbdt",
        ModelConfig(
            kind="catboost",
            params={
                **_CAT_COMMON,
                "boosting_type": "Plain",
                "bootstrap_type": "Bernoulli",
                "colsample_bylevel": 0.8994502668431,
                "depth": 6,
                "grow_policy": "Depthwise",
                "l2_leaf_reg": 1.8187025215896,
                "leaf_estimation_iterations": 7,
                "learning_rate": 0.005177304142,
                "max_bin": 254,
                "max_ctr_complexity": 4,
                "model_size_reg": 0.5247386875068,
                "one_hot_max_size": 53,
                "subsample": 0.8705228845742,
            },
            fit=_ES200,
        ),
    ),
    (
        "ag25_cat_r24",
        "gbdt",
        ModelConfig(
            kind="catboost",
            params={
                **_CAT_COMMON,
                "boosting_type": "Plain",
                "bootstrap_type": "Bernoulli",
                "colsample_bylevel": 0.8597809376276,
                "depth": 8,
                "grow_policy": "Depthwise",
                "l2_leaf_reg": 0.3628261923976,
                "leaf_estimation_iterations": 5,
                "learning_rate": 0.016851077771,
                "max_bin": 254,
                "max_ctr_complexity": 4,
                "model_size_reg": 0.1253820547902,
                "one_hot_max_size": 20,
                "subsample": 0.8120271122061,
            },
            fit=_ES200,
        ),
    ),
    (
        "ag25_cat_r91",
        "gbdt",
        ModelConfig(
            kind="catboost",
            params={
                **_CAT_COMMON,
                "boosting_type": "Plain",
                "bootstrap_type": "Bernoulli",
                "colsample_bylevel": 0.8959275863514,
                "depth": 4,
                "grow_policy": "SymmetricTree",
                "l2_leaf_reg": 0.0026915894253,
                "leaf_estimation_iterations": 12,
                "learning_rate": 0.0475233791203,
                "max_bin": 254,
                "max_ctr_complexity": 5,
                "model_size_reg": 0.1633175256924,
                "one_hot_max_size": 11,
                "subsample": 0.798554178926,
            },
            fit=_ES200,
        ),
    ),
    # XGB 2설정.
    (
        "ag25_xgb_r171",
        "gbdt",
        ModelConfig(
            kind="xgboost",
            params={
                **_XGB_COMMON,
                "colsample_bylevel": 0.9213705632288,
                "colsample_bynode": 0.6443385965381,
                "grow_policy": "lossguide",
                "learning_rate": 0.0068171645251,
                "max_cat_to_onehot": 8,
                "max_depth": 6,
                "max_leaves": 10,
                "min_child_weight": 0.0507304250576,
                "reg_alpha": 4.2446346389037,
                "reg_lambda": 1.4800570021253,
                "subsample": 0.9656290596647,
            },
            fit=_ES200,
        ),
    ),
    (
        "ag25_xgb_r40",
        "gbdt",
        ModelConfig(
            kind="xgboost",
            params={
                **_XGB_COMMON,
                "colsample_bylevel": 0.6377491713202,
                "colsample_bynode": 0.9237625621103,
                "grow_policy": "lossguide",
                "learning_rate": 0.0112462621131,
                "max_cat_to_onehot": 33,
                "max_depth": 10,
                "max_leaves": 35,
                "min_child_weight": 0.1403464856034,
                "reg_alpha": 3.4960653958503,
                "reg_lambda": 1.3062320805235,
                "subsample": 0.6948898835178,
            },
            fit=_ES200,
        ),
    ),
    # TABM 6설정.
    (
        "ag25_tabm_r184",
        "tabm",
        ModelConfig(
            kind="tabm",
            params={
                **_AG_TABM_COMMON,
                "d_block": 864,
                "d_embedding": 24,
                "dropout": 0.0,
                "lr": 0.0019256819924656217,
                "n_blocks": 3,
                "num_emb_n_bins": 3,
                "weight_decay": 0.0,
            },
            fit={},
        ),
    ),
    (
        "ag25_tabm_r69",
        "tabm",
        ModelConfig(
            kind="tabm",
            params={
                **_AG_TABM_COMMON,
                "d_block": 848,
                "d_embedding": 28,
                "dropout": 0.40215621636031007,
                "lr": 0.0010413640454559532,
                "n_blocks": 3,
                "num_emb_n_bins": 18,
                "weight_decay": 0.0,
            },
            fit={},
        ),
    ),
    (
        "ag25_tabm_r52",
        "tabm",
        ModelConfig(
            kind="tabm",
            params={
                **_AG_TABM_COMMON,
                "d_block": 1024,
                "d_embedding": 32,
                "dropout": 0.0,
                "lr": 0.0006297851297842611,
                "n_blocks": 4,
                "num_emb_n_bins": 22,
                "weight_decay": 0.06900108498839816,
            },
            fit={},
        ),
    ),
    (
        "ag25_tabm_default",
        "tabm",
        ModelConfig(
            kind="tabm",
            params={
                **_AG_TABM_COMMON,
                "d_block": 512,
                "d_embedding": 16,
                "dropout": 0.1,
                "lr": 2.0e-3,
                "n_blocks": 3,
                "num_emb_n_bins": 48,
                "weight_decay": 3.0e-4,
            },
            fit={},
        ),
    ),
    (
        "ag25_tabm_r191",
        "tabm",
        ModelConfig(
            kind="tabm",
            params={
                **_AG_TABM_COMMON,
                "d_block": 864,
                "d_embedding": 8,
                "dropout": 0.45321529282058803,
                "lr": 0.0003781238075322413,
                "n_blocks": 4,
                "num_emb_n_bins": 27,
                "weight_decay": 0.01766851962579851,
            },
            fit={},
        ),
    ),
    (
        "ag25_tabm_r49",
        "tabm",
        ModelConfig(
            kind="tabm",
            params={
                **_AG_TABM_COMMON,
                "d_block": 640,
                "d_embedding": 28,
                "dropout": 0.15296207419190627,
                "lr": 0.002277678490593717,
                "n_blocks": 3,
                "num_emb_n_bins": 48,
                "weight_decay": 0.0578159148243893,
            },
            fit={},
        ),
    ),
]


def make_plan() -> FeaturePlan:
    """exp065_tabm과 동일한 champion 계열 피처 계획."""
    providers: list[dict] = [
        {"kind": "derived", "names": ["other_screen", "screen_slack"]},
        {"kind": "target_encoding", "inner_folds": 10, "cols": TE_COLS},
        {"kind": "constrained_impute_aux", "widths": False, "cols": NUM_COLS},
        {
            "kind": "xgb_impute_aux",
            "cols": NUM_COLS,
            "emit": [
                "daily_screen_time_hours",
                "social_media_hours",
                "gaming_hours",
                "work_study_hours",
                "weekend_screen_time",
            ],
            "compositions": [
                "social_frac",
                "work_frac",
                "leisure_frac",
                "resid_frac",
                "week_total",
            ],
            "cat_cols": CAT_COLS,
        },
    ]
    return FeaturePlan.from_config(
        FeatureConfig(base="raw", categorical=CAT_COLS, providers=providers)
    )


def best_iteration_of(adapter) -> str:
    """계열별 early stopping 종착점. 없으면 빈 문자열."""
    model = getattr(adapter, "_model", None)
    if model is None:
        return ""
    for attr in ("best_iteration_", "best_iteration"):
        value = getattr(model, attr, None)
        if value is not None:
            return str(value)
    get_best = getattr(model, "get_best_iteration", None)
    if callable(get_best):
        return str(get_best())
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="zeroshot portfolio 2025 약식 검증")
    parser.add_argument(
        "--family",
        choices=["gbdt", "tabm", "all"],
        default="gbdt",
        help="실행할 계열. tabm은 CUDA GPU에서 실행할 것(CPU는 비현실적으로 느리다).",
    )
    parser.add_argument(
        "--only", help="쉼표로 구분한 이름 목록만 실행한다(분할 실행·재실행용)."
    )
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None
    selected = [
        (name, family, cfg)
        for name, family, cfg in CANDIDATES
        if (args.family == "all" or family == args.family)
        and (only is None or name in only)
    ]
    if not selected:
        raise SystemExit(f"실행 대상이 없다: family={args.family} only={args.only}")

    done: set[str] = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT)["name"])
    else:
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(
            "name,family,auc,best_iteration,placebo_gain,canary_gain,valid,elapsed_sec\n"
        )
    todo = [(n, f, c) for n, f, c in selected if n not in done]
    if not todo:
        print("전부 기록돼 있다. run-logs/zeroshot_portfolio_screen.csv 참고.")
        return

    print(f"공유 피처 행렬 준비를 시작한다(대상 {len(todo)}개).")
    t0 = time.time()
    train = data.load_csv(Path("data/train.csv"))
    test = data.load_csv(Path("data/test.csv"))
    data.align_categories(train, test, CAT_COLS)
    train = data.attach_folds(train, Path("artifacts/folds.parquet"))

    plan = make_plan()
    train_w, _ = plan.apply_dataset_wide(train, test)
    X = plan.build_matrix(train_w, SEED)
    y = train_w[data.TARGET]
    train_ff = prepare_fold_fit_input(train_w, X)
    va_idx = train_w.index[train_w["fold"] == VALID_FOLD]
    tr_idx = train_w.index[train_w["fold"] != VALID_FOLD]
    for transformer in plan.fold_fit_transformers():
        transformer.fit(train_ff.loc[tr_idx], SEED)
    X_fold = plan.add_fold_fit_columns(X, train_ff)
    print(f"피처 행렬 완료: {X_fold.shape} ({time.time() - t0:.0f}s)")

    for i, (name, family, model_cfg) in enumerate(todo):
        t1 = time.time()
        adapter = model_mod.create(model_cfg, SEED)
        va_pred = adapter.fit(
            X_fold.loc[tr_idx], y.loc[tr_idx], X_fold.loc[va_idx], y.loc[va_idx]
        )
        auc = float(roc_auc_score(y.loc[va_idx], va_pred))
        placebo_gain = canary_gain = ""
        valid = ""
        if family == "gbdt":
            # TE를 쓰므로 카나리아 유효성을 기록한다. TabM은 permutation importance가
            # 약식 예산을 넘어 생략한다(피처 계획은 정식 실행들이 이미 검증한 판).
            gain = adapter.importance().set_index("feature")["gain"]
            report = check_canaries({CANARY}, gain)
            placebo_gain = (
                f"{report.placebo_gain:.1f}" if report.placebo_gain is not None else ""
            )
            canary_gain = f"{report.checks[0].gain:.1f}" if report.checks else ""
            valid = str(report.ok)
        elapsed = time.time() - t1
        with OUT.open("a") as f:
            f.write(
                f"{name},{family},{auc:.10f},{best_iteration_of(adapter)},"
                f"{placebo_gain},{canary_gain},{valid},{elapsed:.0f}\n"
            )
        print(
            f"[{i + 1}/{len(todo)}] {name}: auc={auc:.6f} "
            f"best_iter={best_iteration_of(adapter)} valid={valid} ({elapsed:.0f}s)"
        )


if __name__ == "__main__":
    main()
