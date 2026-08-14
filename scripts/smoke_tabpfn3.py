"""TabPFN-3 fold 0 스모크 게이트 실측. (#102)

사용법:
    uv run python scripts/smoke_tabpfn3.py --variant raw --out artifacts/smoke_tabpfn3_raw.json
    uv run python scripts/smoke_tabpfn3.py --variant champion --out artifacts/smoke_tabpfn3_champion.json

ADR 0001 정식 경로에 앞선 게이트 실측이므로 MLflow 실행을 만들지 않는다.
fold 0 하나, seed 42 고정. 학습 fold(1~4, ~553k행)를 문맥으로 넣고 fold 0(~138k행)만 예측한다.

- raw 판: 원시 13열. placebo 없이 잰다(중요도 게이트가 없는 실측이므로).
- champion 판: 정확값 TE와 복원 열을 포함한 직전 GBDT champion(exp052)의 채택
  피처 계획을 fold 0 안에서 fold-fit해 그대로 쓴다(placebo 포함, 파이프라인과 동일).

가중치는 gated라 원격에서 받지 않는다. 로컬에서 내려받아 TabPFN 캐시 디렉터리
(리눅스 ~/.cache/tabpfn, 또는 TABPFN_MODEL_CACHE_DIR)에 넣어 두면 다운로드를 건너뛴다.

게이트 2(스크리닝 2시간 한도) 판정을 위해 fold-0 실측에서 5-fold 스크리닝
(fold별 검증 ~138k행 + 테스트 ~296k행 예측) 소요를 환산해 함께 기록한다.

라이선스: The TABPFN-3 Model is licensed by Prior Labs GmbH under the
TABPFN-3 Non-Commercial License.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline import data
from pipeline.config import load_config
from pipeline.data import ID, TARGET
from pipeline.plan import FeaturePlan

FOLD = 0
SEED = 42
CHAMPION_CONFIG = "configs/exp052_cat_xgb_impute_pass5.yaml"
# 정식 스크리닝의 fold당 예측 규모: 검증 ~138k행 + 테스트 ~296k행.
LICENSE_NOTICE = (
    "The TABPFN-3 Model is licensed by Prior Labs GmbH"
    " under the TABPFN-3 Non-Commercial License"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabPFN-3 fold 0 스모크 (#102)")
    parser.add_argument("--variant", required=True, choices=["raw", "champion"])
    parser.add_argument("--out", required=True, help="결과 JSON 경로")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--n-estimators", type=int, default=8, help="기본 8 = auto 해석값")
    parser.add_argument("--chunk-rows", type=int, default=1000)
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=90.0,
        help="예측 단계 시간 예산(분). 초과가 확실해지면 중단하고 부분 결과를 기록한다. 0이면 무제한.",
    )
    parser.add_argument(
        "--context-rows",
        type=int,
        default=0,
        help="0이면 학습 fold 전체를 문맥으로. 양수면 층화 부분표본(게이트 2 부분표본 재측정용).",
    )
    parser.add_argument(
        "--predict-rows", type=int, default=0, help="0이면 fold 0 전체. 양수면 앞에서 자른다(연결 검증용)."
    )
    parser.add_argument("--memory-saving", default="auto", choices=["auto", "on"])
    return parser.parse_args()


def build_features(
    variant: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[str, float]]:
    """변형별 학습 행렬을 만든다. 반환: (X, y, fold, id, 단계별 초).

    fold-fit 시간은 따로 잰다. 정식 스크리닝에서는 fold마다 다시 fit하므로
    환산 시 5배로 계산해야 한다.
    """
    t0 = time.monotonic()
    fold_fit_s = 0.0
    cfg = load_config(CHAMPION_CONFIG, "screen")
    train = data.load_csv(cfg.data.train)
    test = data.load_csv(cfg.data.test)
    data.align_categories(train, test, cfg.features.categorical)

    if variant == "raw":
        train = data.attach_folds(train, cfg.data.folds)
        raw_cols = [c for c in train.columns if c not in (ID, TARGET, "fold")]
        X = train[raw_cols].copy()
    else:
        # run.py와 같은 순서: dataset-wide 적용 후 fold를 붙인다.
        plan = FeaturePlan.from_config(cfg.features)
        train, test = plan.apply_dataset_wide(train, test)
        train = data.attach_folds(train, cfg.data.folds)
        X = plan.build_matrix(train, SEED)
        transformers = plan.fold_fit_transformers()
        if transformers:
            # cv.run_cv의 fold-fit 규율 그대로: 학습 fold로만 fit하고 train 전체에 적용한다.
            t_ff = time.monotonic()
            extra = [c for c in X.columns if c not in train.columns]
            train_ff = pd.concat([train, X[extra]], axis=1) if extra else train
            tr_idx = train.index[train["fold"] != FOLD]
            for t in transformers:
                t.fit(train_ff.loc[tr_idx], SEED)
            X = plan.add_fold_fit_columns(X, train_ff)
            fold_fit_s = time.monotonic() - t_ff
    times = {"feature_total": time.monotonic() - t0, "fold_fit": fold_fit_s}
    return X, train[TARGET], train["fold"], train[ID], times


def encode_categorical(X: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """category dtype 열을 결측 보존 정수 코드로 바꾸고 TabPFN 범주 인덱스를 돌려준다."""
    X = X.copy()
    cat_indices: list[int] = []
    for i, col in enumerate(X.columns):
        if isinstance(X[col].dtype, pd.CategoricalDtype) or X[col].dtype == object:
            codes = X[col].astype("category").cat.codes.astype("float64")
            X[col] = codes.where(codes >= 0, np.nan)
            cat_indices.append(i)
    return X, cat_indices


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # torch(및 tabpfn)는 피처 구축이 끝난 뒤에 불러온다. champion 판의 fold-fit이
    # XGBoost를 쓰는데, macOS에서 torch가 먼저 올라와 있으면 OpenMP 중복 적재로
    # segfault가 난다.
    X, y, fold, ids, feature_times = build_features(args.variant)
    feature_build_s = feature_times["feature_total"]
    X, cat_indices = encode_categorical(X)
    print(f"[{args.variant}] feature build {feature_build_s / 60:.1f}m"
          f" (fold_fit {feature_times['fold_fit'] / 60:.1f}m)", flush=True)

    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    tr_mask = (fold != FOLD).to_numpy()
    va_mask = ~tr_mask
    X_tr, y_tr = X.loc[tr_mask], y.loc[tr_mask]
    X_va, y_va, ids_va = X.loc[va_mask], y.loc[va_mask], ids.loc[va_mask]
    if args.context_rows > 0:
        rng = np.random.default_rng(SEED)
        # 층화 부분표본: 타깃 비율을 유지한다(리서치 권고의 SUBSAMPLE 경로 실측용).
        keep = []
        for cls, grp in y_tr.groupby(y_tr):
            n = round(args.context_rows * len(grp) / len(y_tr))
            keep.append(rng.choice(grp.index.to_numpy(), size=min(n, len(grp)), replace=False))
        keep_idx = np.sort(np.concatenate(keep))
        X_tr, y_tr = X_tr.loc[keep_idx], y_tr.loc[keep_idx]
    if args.predict_rows > 0:
        X_va, y_va = X_va.iloc[: args.predict_rows], y_va.iloc[: args.predict_rows]
        ids_va = ids_va.iloc[: args.predict_rows]

    use_cuda = torch.cuda.is_available() and args.device in ("auto", "cuda")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    def make_clf(memory_saving: bool | str) -> TabPFNClassifier:
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V3,
            n_estimators=args.n_estimators,
            device=args.device,
            random_state=SEED,
            fit_mode="fit_with_cache",
            categorical_features_indices=cat_indices or None,
            memory_saving_mode=memory_saving,
        )

    memory_saving: bool | str = True if args.memory_saving == "on" else "auto"
    clf = make_clf(memory_saving)

    t_fit = time.monotonic()
    try:
        clf.fit(X_tr, y_tr)
    except torch.cuda.OutOfMemoryError:
        # 티켓의 후퇴 경로: OOM이면 memory_saving_mode=True로 한 번 재시도한다.
        assert memory_saving != True, "memory_saving_mode=True에서도 fit OOM"  # noqa: E712
        memory_saving = True
        torch.cuda.empty_cache()
        clf = make_clf(memory_saving)
        clf.fit(X_tr, y_tr)
    fit_s = time.monotonic() - t_fit

    chunk = args.chunk_rows
    n_chunks = (len(X_va) + chunk - 1) // chunk
    preds = np.empty(len(X_va))
    chunk_times: list[float] = []
    aborted = False
    t_pred = time.monotonic()
    for i in range(n_chunks):
        t_c = time.monotonic()
        sl = slice(i * chunk, min((i + 1) * chunk, len(X_va)))
        try:
            preds[sl] = clf.predict_proba(X_va.iloc[sl])[:, 1]
        except torch.cuda.OutOfMemoryError:
            assert memory_saving != True, "memory_saving_mode=True에서도 predict OOM"  # noqa: E712
            memory_saving = True
            torch.cuda.empty_cache()
            clf = make_clf(memory_saving)
            t_refit = time.monotonic()
            clf.fit(X_tr, y_tr)
            fit_s += time.monotonic() - t_refit
            preds[sl] = clf.predict_proba(X_va.iloc[sl])[:, 1]
        chunk_times.append(time.monotonic() - t_c)
        elapsed = time.monotonic() - t_pred
        if i % 10 == 0 or i == n_chunks - 1:
            projected = elapsed / (i + 1) * n_chunks
            print(
                f"[{args.variant}] chunk {i + 1}/{n_chunks}"
                f" elapsed={elapsed / 60:.1f}m projected={projected / 60:.1f}m",
                flush=True,
            )
        if args.max_minutes > 0 and elapsed / (i + 1) * n_chunks > args.max_minutes * 60 and i >= 9:
            aborted = True
            n_done = i + 1
            break
    else:
        n_done = n_chunks
    predict_s = time.monotonic() - t_pred

    done_rows = min(n_done * chunk, len(X_va))
    auc = float(roc_auc_score(y_va.iloc[:done_rows], preds[:done_rows])) if done_rows else None

    # 정식 스크리닝 환산: fold마다 문맥 재인코딩(fit) 후 검증 fold 전체 + 테스트 전체를 예측한다.
    steady = float(np.median(chunk_times[1:])) if len(chunk_times) > 1 else float(chunk_times[0])
    n_test = sum(1 for _ in open(Path("data/test.csv"), "rb")) - 1
    va_full = int(va_mask.sum())
    per_fold_pred_s = steady * ((va_full + n_test) / chunk)
    # fold-fit은 정식 스크리닝에서 fold마다 반복된다. 나머지 피처 구축은 한 번.
    screening_projected_s = (
        (feature_build_s - feature_times["fold_fit"])
        + 5 * (feature_times["fold_fit"] + fit_s + per_fold_pred_s)
    )

    result = {
        "issue": 102,
        "variant": args.variant,
        "seed": SEED,
        "fold": FOLD,
        "n_features": int(X.shape[1]),
        "features": list(X.columns),
        "categorical_indices": cat_indices,
        "n_context_rows": int(len(X_tr)),
        "n_predict_rows_total": int(len(X_va)),
        "n_predict_rows_done": int(done_rows),
        "aborted_over_budget": aborted,
        "n_estimators": args.n_estimators,
        "fit_mode": "fit_with_cache",
        "memory_saving_mode_final": memory_saving,
        "device_cuda": bool(use_cuda),
        "gpu_max_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20) if use_cuda else None,
        "auc_fold0": auc,
        "time_s": {
            "feature_build": round(feature_build_s, 1),
            "fold_fit": round(feature_times["fold_fit"], 1),
            "fit": round(fit_s, 1),
            "predict": round(predict_s, 1),
            "chunk_first": round(chunk_times[0], 2),
            "chunk_steady_median": round(steady, 2),
        },
        "screening_projection_s": round(screening_projected_s),
        "screening_projection_note": (
            "(feature_build - fold_fit) + 5 x (fold_fit + fit"
            " + steady_chunk x (fold검증+테스트 행)/chunk)."
            f" 검증 {va_full}행, 테스트 {n_test}행 기준."
        ),
        "license": LICENSE_NOTICE,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    # 완주한 예측만 저장한다. 게이트 통과 시 champion OOF와의 잔차 상관 비교에 쓴다.
    pd.DataFrame(
        {
            "id": ids_va.iloc[:done_rows].to_numpy(),
            "fold": FOLD,
            "pred": preds[:done_rows],
        }
    ).to_parquet(out_path.with_suffix(".parquet"), index=False)
    print(json.dumps({k: v for k, v in result.items() if k != "features"}, ensure_ascii=False, indent=2))
    if aborted:
        sys.exit(3)


if __name__ == "__main__":
    main()
