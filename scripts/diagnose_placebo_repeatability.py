"""플라시보 반복성 진단과 판독표 원자료 생성. (#261)

이슈 #256이 고정한 최소 실증의 원자료를 만든다. 두 하위 명령을 제공한다.

refit: 모델 seed와 fold를 고정한 채 플라시보 난수만 바꿔 fold 모델을 다시
맞추고, 셀 판독값(플라시보 상대 방향, 통과 수, 순위, 백분위, 최솟값·중앙값·
최댓값)을 기록한다. 플라시보 열은 행렬 구성 직후 지정 난수로 다시 만들므로
placebo_noise 파생 카나리아(TE)도 같은 난수를 따른다. 모델 학습과 fold-fit
변환의 난수는 스크리닝 모델 seed(42) 그대로다.

readout: 기존 실행의 feature_importance.parquet(feature, fold, seed, gain)을
같은 셀 판독 스키마로 변환한다. 새 실행과 기존 실행을 한 판독표에 합칠 때 쓴다.

새 피처 목록의 출처는 docs/research/placebo-gate-conflict-evidence.json이다(#255).

사용법:
    uv run python scripts/diagnose_placebo_repeatability.py refit \
        configs/exp027_recon_ce.yaml --placebo-seeds 101 202 303 404 \
        --fold 0 --out run-placebo-repeat/exp027_refit.json
    uv run python scripts/diagnose_placebo_repeatability.py readout \
        exp070_cat_exact_cats --importance <feature_importance.parquet> \
        --out run-placebo-repeat/exp070_readout.json

판독 정의(반복 4회를 경험적 P값이나 분위수 추정으로 해석하지 않는다):
- rank: 셀 안 전체 피처를 gain 내림차순으로 세운 경쟁 순위(동률은 최고 순위).
- percentile: 셀 안 다른 피처 중 gain이 엄격히 낮은 비율(%).
- above_placebo: 해당 셀에서 후보 gain > 플라시보 gain.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline import data, model as model_mod
from pipeline.config import load_config
from pipeline.features import PLACEBO, placebo_series
from pipeline.judgment import SCREENING_SEEDS
from pipeline.plan import FeaturePlan, prepare_fold_fit_input

EVIDENCE_PATH = Path("docs/research/placebo-gate-conflict-evidence.json")


def new_features_of(config_name: str) -> list[str]:
    """#255 충돌 장부 원자료에 기록된 감사 대상 새 피처 목록."""
    evidence = json.loads(EVIDENCE_PATH.read_text())
    for candidate in evidence["candidates"]:
        if candidate["config"] == config_name:
            return [f["feature"] for f in candidate["features"]]
    raise SystemExit(f"{EVIDENCE_PATH}에 {config_name} 항목이 없다.")


def cell_readout(
    importance: pd.DataFrame, new_features: list[str], cell: dict[str, int]
) -> dict:
    """셀(fold 모델 하나)의 gain 표를 판독값으로 바꾼다."""
    gain = importance.set_index("feature")["gain"].astype("float64")
    n = len(gain)
    placebo_gain = float(gain[PLACEBO])

    def rank_of(value: float) -> int:
        return 1 + int((gain > value).sum())

    def percentile_of(value: float) -> float:
        return 100.0 * float((gain < value).sum()) / (n - 1)

    features = []
    for feature in new_features:
        feature_gain = float(gain[feature])
        features.append(
            {
                "feature": feature,
                "gain": feature_gain,
                "ratio_to_placebo": (
                    feature_gain / placebo_gain if placebo_gain != 0 else None
                ),
                "above_placebo": bool(feature_gain > placebo_gain),
                "rank": rank_of(feature_gain),
                "percentile": percentile_of(feature_gain),
            }
        )
    return {
        "cell": cell,
        "n_features": n,
        "placebo": {
            "gain": placebo_gain,
            "rank": rank_of(placebo_gain),
            "percentile": percentile_of(placebo_gain),
        },
        "features": features,
        "above_count": sum(f["above_placebo"] for f in features),
        "all_above": all(f["above_placebo"] for f in features),
    }


def summarize(config_name: str, new_features: list[str], cells: list[dict]) -> dict:
    """셀 판독의 피처별 요약: 통과 수와 최솟값·중앙값·최댓값. 추정치가 아니라 관측 요약이다."""

    def spread(values: list[float]) -> dict:
        return {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    per_feature = []
    for feature in new_features:
        rows = [
            f for c in cells for f in c["features"] if f["feature"] == feature
        ]
        per_feature.append(
            {
                "feature": feature,
                "cells": len(rows),
                "above_count": sum(r["above_placebo"] for r in rows),
                "gain": spread([r["gain"] for r in rows]),
                "ratio_to_placebo": spread([r["ratio_to_placebo"] for r in rows]),
                "rank": spread([r["rank"] for r in rows]),
                "percentile": spread([r["percentile"] for r in rows]),
            }
        )
    return {
        "config": config_name,
        "new_features": new_features,
        "cells": cells,
        "placebo_gain": spread([c["placebo"]["gain"] for c in cells]),
        "placebo_rank": spread([c["placebo"]["rank"] for c in cells]),
        "per_feature": per_feature,
        "all_above_cells": sum(c["all_above"] for c in cells),
        "n_cells": len(cells),
    }


def run_refit(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config, "screen")
    model_seed = SCREENING_SEEDS[0]
    plan = FeaturePlan.from_config(cfg.features)
    new_features = new_features_of(cfg.name)

    train = data.load_csv(cfg.data.train)
    test = data.load_csv(cfg.data.test)
    data.align_categories(train, test, cfg.features.categorical)
    train, test = plan.apply_dataset_wide(train, test)
    train = data.attach_folds(train, cfg.data.folds)
    y = train[data.TARGET]
    va_idx = train.index[train["fold"] == args.fold]
    tr_idx = train.index[train["fold"] != args.fold]

    cells = []
    for placebo_seed in args.placebo_seeds:
        # 행렬은 모델 seed로 만들고 플라시보 열만 지정 난수로 덮는다.
        # 이후 fold-fit(placebo_noise TE 카나리아 포함)이 이 열을 그대로 본다.
        X = plan.build_matrix(train, model_seed)
        X[PLACEBO] = placebo_series(train, placebo_seed)
        X_fold = X
        transformers = plan.fold_fit_transformers()
        if transformers:
            train_ff = prepare_fold_fit_input(train, X)
            for t in transformers:
                t.fit(train_ff.loc[tr_idx], model_seed)
            X_fold = plan.add_fold_fit_columns(X, train_ff)

        adapter = model_mod.create(cfg.model, model_seed)
        va_pred = np.asarray(
            adapter.fit(
                X_fold.loc[tr_idx], y.loc[tr_idx], X_fold.loc[va_idx], y.loc[va_idx]
            ),
            dtype="float64",
        )
        auc = float(roc_auc_score(y.loc[va_idx], va_pred))
        cell = cell_readout(
            adapter.importance(),
            new_features,
            {"model_seed": model_seed, "fold": args.fold, "placebo_seed": placebo_seed},
        )
        cell["auc_fold"] = auc
        cells.append(cell)
        print(
            f"{cfg.name} placebo_seed={placebo_seed} auc_fold_{args.fold}={auc:.5f} "
            f"above={cell['above_count']}/{len(new_features)}"
        )

    result = summarize(cfg.name, new_features, cells)
    result["mode"] = "refit"
    result["model_seed"] = model_seed
    result["fold"] = args.fold
    result["placebo_seeds"] = list(args.placebo_seeds)
    return result


def run_readout(args: argparse.Namespace) -> dict:
    new_features = new_features_of(args.config_name)
    importance = pd.read_parquet(args.importance)
    cells = []
    for (seed, fold), group in sorted(importance.groupby(["seed", "fold"])):
        # 기존 실행은 플라시보 난수 = 모델 seed다(파이프라인 규약).
        cells.append(
            cell_readout(
                group,
                new_features,
                {"model_seed": int(seed), "fold": int(fold), "placebo_seed": int(seed)},
            )
        )
    result = summarize(args.config_name, new_features, cells)
    result["mode"] = "readout"
    result["importance_source"] = str(args.importance)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    refit = sub.add_parser("refit", help="플라시보 난수만 바꿔 fold 모델 재적합")
    refit.add_argument("config", help="실험 설정 YAML 경로")
    refit.add_argument("--placebo-seeds", type=int, nargs="+", required=True)
    refit.add_argument("--fold", type=int, default=0)
    refit.add_argument("--out", type=Path, required=True)

    readout = sub.add_parser("readout", help="기존 importance parquet의 셀 판독")
    readout.add_argument("config_name", help="충돌 장부 원자료의 config 이름")
    readout.add_argument("--importance", type=Path, required=True)
    readout.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    result = run_refit(args) if args.command == "refit" else run_readout(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
