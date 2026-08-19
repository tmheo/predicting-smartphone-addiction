"""축소 표본 fold 0 스모크로 TabR 학습 수렴과 epoch 비용을 실측한다. (#199)

게이트 1(축소 표본에서 정상 수렴) 판정과 fold 0 전체 실행의 max_epochs
산정 근거를 만든다.
후보 저장소는 fold 0 outer 학습 부분의 층화 표본으로 제한하고,
검증은 fold 0 검증 부분 전체를 사용한다(누수 규율 유지).

사용법:
    uv run python scripts/smoke_tabr.py configs/exp122_tabr.yaml \
        --out results/tabr-smoke.json --train-fraction 0.2 \
        --variant default --variant plr_lite
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pipeline import data
from pipeline.config import ModelConfig, load_config
from pipeline.model import TabRAdapter, create
from pipeline.plan import FeaturePlan

# 공식 higgs-small 2-plr-lite-evaluation 설정의 PLR(lite) 임베딩 값이다.
# 기본 TabR가 축소 표본에서 수렴 상한에 걸리는지 확인하는 구조 보강 탐침으로만 쓴다.
PLR_LITE_PROBE = {
    "n_frequencies": 75,
    "frequency_scale": 0.03482617399210428,
    "d_embedding": 45,
    "lite": True,
}
PLATEAU_TOLERANCE = 5e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabR 축소 표본 fold 0 스모크 (#199)")
    parser.add_argument("config", help="TabR 실험 설정 YAML")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.2)
    parser.add_argument("--max-epochs", type=int, default=48)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument(
        "--variant",
        action="append",
        choices=["default", "plr_lite"],
        default=None,
        help="반복 가능. 생략하면 default와 plr_lite를 모두 실행한다.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def prepare_fold_matrix(
    cfg, plan: FeaturePlan, fold: int, seed: int
):
    train = data.load_csv(cfg.data.train)
    test = data.load_csv(cfg.data.test)
    data.align_categories(train, test, cfg.features.categorical)
    train, test = plan.apply_dataset_wide(train, test)
    train = data.attach_folds(train, cfg.data.folds)
    if fold not in set(train["fold"].astype(int)):
        raise ValueError(f"자료에 fold {fold}가 없다.")

    X = plan.build_matrix(train, seed)
    va_idx = train.index[train["fold"] == fold]
    tr_idx = train.index[train["fold"] != fold]
    transformers = plan.fold_fit_transformers()
    if transformers:
        train_ff = train.copy()
        for column in X.columns:
            if column not in train_ff.columns:
                train_ff[column] = X[column]
        for transformer in transformers:
            transformer.fit(train_ff.loc[tr_idx], seed)
        X = plan.add_fold_fit_columns(X, train_ff)
    assert list(X.columns) == plan.all_columns(), "학습 컬럼이 피처 계획의 선언과 다르다."
    return train, X, tr_idx, va_idx


def stratified_subsample(y, tr_idx, fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    labels = y.loc[tr_idx]
    for value in sorted(labels.unique()):
        rows = tr_idx[labels == value].to_numpy()
        size = max(1, int(round(len(rows) * fraction)))
        keep.append(rng.choice(rows, size=size, replace=False))
    sampled = np.concatenate(keep)
    sampled.sort()
    return sampled


def run_variant(
    name: str,
    base_params: dict,
    X,
    y,
    sub_tr_idx,
    va_idx,
    *,
    max_epochs: int,
    patience: int,
    device: str | None,
    seed: int,
) -> dict:
    params = dict(base_params)
    params["max_epochs"] = max_epochs
    params["patience"] = patience
    if name == "plr_lite":
        params["num_embeddings"] = dict(PLR_LITE_PROBE)
    if device is not None:
        params["device"] = device
    adapter = create(ModelConfig(kind="tabr", params=params, fit={}), seed=seed)
    assert isinstance(adapter, TabRAdapter)

    started = time.monotonic()
    prediction = adapter.fit(
        X.loc[sub_tr_idx], y.loc[sub_tr_idx], X.loc[va_idx], y.loc[va_idx]
    )
    fit_seconds = time.monotonic() - started
    auc = float(roc_auc_score(y.loc[va_idx].to_numpy(dtype="float64"), prediction))

    eval_started = time.monotonic()
    adapter.predict(X.loc[va_idx])
    eval_seconds = time.monotonic() - eval_started

    observations = adapter.entry_diagnostics().observations
    assertions = adapter.entry_diagnostics().assertions
    aucs = [float(value) for value in observations["epoch_validation_aucs"]]
    seconds = [float(value) for value in observations["epoch_seconds"]]
    finite = bool(np.isfinite(aucs).all())
    learned = finite and len(aucs) >= 2 and max(aucs) > aucs[0]
    tail = aucs[-(patience + 1) :]
    plateaued = observations["stop_reason"] == "early_stopping" or (
        len(tail) >= 2 and (max(tail) - tail[0]) < PLATEAU_TOLERANCE
    )
    return {
        "variant": name,
        "params": {
            key: value for key, value in params.items() if key != "device"
        },
        "assertions": assertions,
        "best_validation_auc": auc,
        "epoch_validation_aucs": aucs,
        "epoch_seconds": seconds,
        "best_epoch": observations["best_epoch"],
        "stop_reason": observations["stop_reason"],
        "search_backend": observations["search_backend"],
        "faiss_version": observations["faiss_version"],
        "torch_version": observations["torch_version"],
        "fit_seconds": fit_seconds,
        "validation_eval_seconds": eval_seconds,
        "converged": bool(finite and learned and plateaued),
        "convergence_checks": {
            "finite": finite,
            "learned_beyond_first_epoch": learned,
            "plateaued_or_early_stopped": plateaued,
            "adapter_assertions": all(assertions.values()),
        },
    }


def main() -> None:
    args = parse_args()
    variants = args.variant or ["default", "plr_lite"]
    if not 0 < args.train_fraction <= 1:
        raise ValueError("train-fraction은 (0, 1] 범위여야 한다.")
    cfg = load_config(args.config, "screen")
    if cfg.model.kind != "tabr":
        raise ValueError(f"tabr 설정이 아니다: {cfg.model.kind}")
    plan = FeaturePlan.from_config(cfg.features)

    train, X, tr_idx, va_idx = prepare_fold_matrix(cfg, plan, args.fold, args.seed)
    y = train[data.TARGET]
    sub_tr_idx = stratified_subsample(y, tr_idx, args.train_fraction, args.seed)

    results = []
    for name in variants:
        print(f"[smoke_tabr] variant={name} 시작", flush=True)
        results.append(
            run_variant(
                name,
                dict(cfg.model.params),
                X,
                y,
                sub_tr_idx,
                va_idx,
                max_epochs=args.max_epochs,
                patience=args.patience,
                device=args.device,
                seed=args.seed,
            )
        )

    fraction = args.train_fraction
    for result in results:
        train_seconds = [
            max(epoch - result["validation_eval_seconds"], 0.0)
            for epoch in result["epoch_seconds"]
        ]
        mean_train = float(np.mean(train_seconds)) if train_seconds else None
        result["projection"] = {
            "train_fraction": fraction,
            "formula": "train_part * (1/f)^2 + eval_part * (1/f)",
            "projected_full_fold_epoch_seconds": (
                None
                if mean_train is None
                else mean_train / fraction**2
                + result["validation_eval_seconds"] / fraction
            ),
        }

    payload = {
        "smoke": "tabr_reduced_sample_fold0",
        "issue": 199,
        "config": str(cfg.source_path),
        "experiment": cfg.name,
        "fold": args.fold,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "rows": {
            "outer_training": int(len(tr_idx)),
            "subsampled_training": int(len(sub_tr_idx)),
            "validation": int(len(va_idx)),
        },
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "variants": results,
        "gate1_converged": {
            result["variant"]: result["converged"] for result in results
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(f"[smoke_tabr] 저장: {args.out}", flush=True)


if __name__ == "__main__":
    main()
