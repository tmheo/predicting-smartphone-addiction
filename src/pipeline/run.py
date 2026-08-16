"""실행 진입점.

사용법:
    uv run python -m pipeline.run configs/exp018_orig_mean.yaml --stage screen
    uv run python -m pipeline.run configs/exp018_orig_mean.yaml --stage confirm
    uv run python -m pipeline.run configs/exp018_orig_mean.yaml --stage screen --plan  # 실행 계획만 출력

실험 하나 = 설정 파일 하나 = MLflow run 하나. 단계(스크리닝·확정 재검증)는 config가
아니라 --stage가 정하므로, 같은 config 하나로 두 단계를 config diff 없이 실행한다(#103).
--stage는 필수이고 기본값이 없다: 단계 착오 하나가 GPU 몇 시간짜리 재실행이다.
관찰 규약(#43): 설정 적재 성공 직후, 데이터 적재보다 먼저 observe.RunObserver가
MLflow 실행을 만들고 수명주기를 소유한다. 진입점은 단계 전환과 결과만 통지한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv, data, initial_score, seed_parallel, tracking
from .config import STAGES, load_config
from .plan import FeaturePlan
from .recovery import FoldRecovery


def main() -> None:
    parser = argparse.ArgumentParser(description="CV 파이프라인 실행")
    parser.add_argument("config", help="실험 설정 YAML 경로")
    parser.add_argument(
        "--stage",
        required=True,
        choices=STAGES,
        help="실행 단계. 시드는 판정 계약(judgment)의 단계별 시드 상수로 정해진다. (#103)",
    )
    parser.add_argument("--plan", action="store_true", help="학습 없이 실행 계획만 출력")
    parser.add_argument(
        "--recovery-dir",
        type=Path,
        help="fold 복구 디렉터리. 기본값은 run-recovery/<실험>-<단계>다.",
    )
    args = parser.parse_args()

    # 검증된 설정 파일 = ExperimentConfig 생성 성공. 피처 계획의 누출 규율 검증 포함. (#43, #71)
    cfg = load_config(args.config, args.stage)
    plan = FeaturePlan.from_config(cfg.features)

    if args.plan:
        # 계획 출력은 MLflow 실행도 실행 로그 파일도 만들지 않는다. (#43 시나리오 5)
        print(f"experiment : {cfg.name}")
        print(f"config     : {cfg.source_path}")
        print(f"stage      : {cfg.stage}")
        print(f"seeds      : {cfg.seeds}")
        print(f"model      : {cfg.model.kind} {cfg.model.params}")
        if cfg.initial_score is not None:
            print(f"init score : {cfg.initial_score.kind} {cfg.initial_score.params}")
        for name, h in _input_hashes(cfg).items():
            print(f"sha256.{name:<6}: {h[:16]}…")
        print("git        :", tracking.git_state())
        print("feature plan (단계 / kind / 산출 컬럼 / 타깃 참조):")
        raw_columns = list(pd.read_csv(cfg.data.train, nrows=0).columns)
        for stage, kind, columns, uses_target in plan.describe(raw_columns):
            mark = "타깃 참조" if uses_target else "-"
            print(f"  [{stage}] {kind}: {', '.join(columns)} ({mark})")
        print("기록될 것   : params(feature 목록, 모델 파라미터), metrics(auc_fold_*, auc_oof, auc_oof_seed_*),")
        print("             progress.*/time.* 진행 기록, artifacts(설정 yaml, oof.parquet, oof_seed_*.parquet,")
        print("             test_pred.parquet, feature_importance.parquet, submission.csv,")
        print("             summary.html 등 결과 요약, logs/run.log)")
        print(f"fold 복구    : {args.recovery_dir or _default_recovery_dir(cfg)}")
        return

    from .observe import RunObserver

    observer = RunObserver.begin(cfg)
    try:
        observer.stage("setup")
        input_hashes = _input_hashes(cfg)
        observer.record_input_hashes(input_hashes)
        recovery = FoldRecovery.for_run(
            args.recovery_dir or _default_recovery_dir(cfg),
            cfg,
            input_hashes,
            git_commit=tracking.git_state()["git_commit"],
        )

        observer.stage("data_load")
        train = data.load_csv(cfg.data.train)
        test = data.load_csv(cfg.data.test)
        data.align_categories(train, test, cfg.features.categorical)
        # dataset-wide 컬럼은 새 frame으로 받는다. 제자리 변형 없음. (#71)
        train, test = plan.apply_dataset_wide(train, test)
        train = data.attach_folds(train, cfg.data.folds)
        n_folds = int(train["fold"].max()) + 1
        observer.data_loaded(seed_total=len(cfg.seeds), fold_total=n_folds)

        # 시드 반복: 예측은 평균, metric은 평균 예측 기준으로 다시 계산. (#15)
        # PIPELINE_SEED_GPUS가 있으면 시드 단위 프로세스 병렬로 GPU를 나눠 쓴다. (#99)
        results = seed_parallel.run_seeds(
            cfg, plan, train, test, recorder=observer, recovery=recovery
        )

        observer.stage("evaluation")
        # 시드별 OOF AUC는 평균 재채점으로 fold_aucs가 덮이기 전에 확보한다. (ADR 0001)
        seed_aucs = {seed: r.fold_aucs["auc_oof"] for seed, r in zip(cfg.seeds, results)}
        # 시드별 OOF도 평균 대입으로 results[0].oof가 덮이기 전에 확보한다. (#98 기록 규약)
        seed_oofs = {seed: r.oof.copy() for seed, r in zip(cfg.seeds, results)}
        final = results[0]
        # 선언 = 실제: 학습에 쓴 컬럼이 계획의 선언과 다르면 기록 전에 실패한다.
        # feature 목록 param은 이 검증을 거친 선언 기준 목록이 된다. (#71)
        assert final.feature_names == plan.all_columns(), (
            f"학습 컬럼이 피처 계획의 선언과 다르다: {final.feature_names} != {plan.all_columns()}"
        )
        if len(results) > 1:
            final.oof["pred"] = np.mean([r.oof["pred"] for r in results], axis=0)
            final.test_pred["pred"] = np.mean([r.test_pred["pred"] for r in results], axis=0)
            final.fold_aucs = cv.score_predictions(
                train[data.TARGET], train["fold"], final.oof["pred"].to_numpy()
            )
            final.importance = pd.concat([r.importance for r in results], ignore_index=True)
        final.recovery_evidence = [
            item for result in results for item in result.recovery_evidence
        ]
        # 확정 재검증의 시드별 비교를 위해 대표 metric과 함께 기록된다. (ADR 0001)
        for seed, auc in seed_aucs.items():
            final.fold_aucs[f"auc_oof_seed_{seed}"] = auc

        observer.stage("artifacts")
        observer.log_final(final, seed_oofs)
        tracking.warn_below_placebo(final.importance)
        print(f"run_id={observer.run_id} auc_oof={final.fold_aucs['auc_oof']:.5f}")
        observer.succeed()
    except BaseException as exc:  # noqa: BLE001 - 중단 신호까지 관찰 기록 후 종료한다.
        observer.fail(exc)
        sys.exit(130 if isinstance(exc, KeyboardInterrupt) else 1)


def _input_hashes(cfg) -> dict[str, str]:
    hashes = {
        "train": data.file_sha256(cfg.data.train),
        "test": data.file_sha256(cfg.data.test),
        "folds": data.file_sha256(cfg.data.folds),
    }
    hashes.update(
        {name: data.file_sha256(path) for name, path in initial_score.input_paths(cfg.initial_score).items()}
    )
    return hashes


def _default_recovery_dir(cfg) -> Path:
    return Path("run-recovery") / f"{cfg.name}-{cfg.stage}"


if __name__ == "__main__":
    main()
