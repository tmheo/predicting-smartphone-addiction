# PROTOTYPE (issue #17): 구조 확인용 뼈대.
"""실행 진입점.

사용법:
    uv run python -m pipeline.run configs/exp001_lgbm_baseline.yaml
    uv run python -m pipeline.run configs/exp001_lgbm_baseline.yaml --plan  # 실행 계획만 출력

실험 하나 = 설정 파일 하나 = MLflow run 하나.
"""

from __future__ import annotations

import argparse

import numpy as np

from . import cv, data, tracking
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="CV 파이프라인 실행")
    parser.add_argument("config", help="실험 설정 YAML 경로")
    parser.add_argument("--plan", action="store_true", help="학습 없이 실행 계획만 출력")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_hashes = {
        "train": data.file_sha256(cfg.data.train),
        "test": data.file_sha256(cfg.data.test),
        "folds": data.file_sha256(cfg.data.folds),
    }

    if args.plan:
        print(f"experiment : {cfg.name}")
        print(f"config     : {cfg.source_path}")
        print(f"seeds      : {cfg.seeds}")
        print(f"model      : {cfg.model.kind} {cfg.model.params}")
        for name, h in input_hashes.items():
            print(f"sha256.{name:<6}: {h[:16]}…")
        print("git        :", tracking.git_state())
        print("기록될 것   : params(feature 목록, 모델 파라미터), metrics(auc_fold_*, auc_oof),")
        print("             artifacts(설정 yaml, oof.parquet[id,fold,pred], test_pred.parquet[id,pred], submission.csv)")
        return

    train = data.load_train(cfg.data.train, cfg.features.categorical)
    test = data.load_test(cfg.data.test, cfg.features.categorical)
    train = data.attach_folds(train, cfg.data.folds)

    # 시드 반복: 예측은 평균, metric은 평균 예측 기준으로 다시 계산. (#15)
    results = [cv.run_cv(cfg, train, test, seed) for seed in cfg.seeds]
    final = results[0]
    if len(results) > 1:
        final.oof["pred"] = np.mean([r.oof["pred"] for r in results], axis=0)
        final.test_pred["pred"] = np.mean([r.test_pred["pred"] for r in results], axis=0)
        # TODO: 평균 예측으로 fold_aucs 재계산.

    run_id = tracking.log_run(cfg, final, input_hashes)
    print(f"run_id={run_id} auc_oof={final.fold_aucs['auc_oof']:.5f}")


if __name__ == "__main__":
    main()
