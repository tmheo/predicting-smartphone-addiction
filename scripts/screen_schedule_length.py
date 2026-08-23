"""OneCycle 일정 길이 fold 0 짝비교. (#382)

champion `exp131_lookup_bivariate_plr5`에서 `epochs`만 바꾼 후보를 fold 0, seed 42로
겨루게 한다. 선별 전용 약식 검증이며 `scripts/screen_pairs.py`의 전례를 따른다:
커밋된 folds.parquet의 fold 0 하나만 검증에 쓰고 나머지 4개 fold 전체 행으로 학습한다.
행 표본 축소는 하지 않는다.

`pipeline.run`은 5개 fold를 모두 도는 정식 경로이고 여기에는 fold 하나만 도는 경로가 없다.
그래서 `cv_seed_execution.execute_seed`의 fold 본문 가운데 판정에 필요한 부분만 그대로 옮겨
적었다. 누출에 민감한 fold-fit 단계는 `plan.materialize_fold_fit_provider`를 같은 인수로
불러 정식 경로와 같은 상태를 만든다. 순열 중요도와 test 예측은 이 판정에 쓰지 않으므로
계산하지 않는다.

fold 0의 피처 행렬은 한 번만 만들고 모든 후보가 공유한다. 후보 사이의 유일한 차이가
`epochs`가 되므로 짝차이가 일정 길이만의 효과를 잰다.

사용법:
    uv run python scripts/screen_schedule_length.py
    uv run python scripts/screen_schedule_length.py --epochs 32 12 16 20 24

결과는 `--out` 경로에 JSON Lines로 증분 기록한다. 이미 기록된 `epochs`는 건너뛰므로
중단한 실행을 그대로 다시 시작하면 남은 후보부터 이어 달린다.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import data  # noqa: E402
from pipeline import model as model_mod  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.plan import FeaturePlan, prepare_fold_fit_input  # noqa: E402

CONFIG = "configs/exp131_lookup_bivariate_plr5.yaml"
SEED = 42
VALID_FOLD = 0
# champion run 54acd002의 저장된 oof_seed_42.parquet을 fold 0에서 다시 채점한 값.
# run-issue382/read_champion_fold0.py가 산출한다.
CHAMPION_FOLD0_AUC = 0.9686518
# 같은 실행의 fold 0 시드 3개(42/43/44) 값의 폭. 짝차이를 읽을 때의 잡음 눈금이다.
CHAMPION_FOLD0_SEED_SPREAD = 0.0000446


def build_fold0_matrices(cfg, plan: FeaturePlan, smoke_rows: int | None = None):
    """fold 0의 학습·검증 행렬을 정식 경로와 같은 순서로 만든다."""
    train = data.load_csv(cfg.data.train)
    test = data.load_csv(cfg.data.test)
    if smoke_rows:
        # 연결 점검 전용. 판정에 쓰는 실행에서는 절대 켜지 않는다.
        train = train.head(smoke_rows).copy()
        test = test.head(smoke_rows).copy()
    data.align_categories(train, test, cfg.features.categorical)
    train, test = plan.apply_dataset_wide(train, test)
    train = data.attach_folds(train, cfg.data.folds)

    y = train[data.TARGET]
    va_idx = train.index[train["fold"] == VALID_FOLD]
    tr_idx = train.index[train["fold"] != VALID_FOLD]

    X = plan.build_matrix(train, SEED)
    X_test = plan.build_matrix(test, SEED)

    providers = plan.new_fold_fit_providers()
    X_fold, X_test_fold = X, X_test
    if providers:
        train_ff = prepare_fold_fit_input(train, X)
        test_ff = prepare_fold_fit_input(test, X_test)
        for kind, transformer in providers:
            started = time.time()
            train_values, test_values, _ = plan.materialize_fold_fit_provider(
                kind=kind,
                transformer=transformer,
                train_input=train_ff,
                test_input=test_ff,
                training_index=tr_idx,
                validation_index=va_idx,
                seed=SEED,
                fold=VALID_FOLD,
                recorder=None,
            )
            collision = set(train_values.columns) & set(X_fold.columns)
            if collision:
                raise AssertionError(f"fold-fit 컬럼 이름 충돌: {sorted(collision)}")
            X_fold = pd.concat([X_fold, train_values], axis=1)
            X_test_fold = pd.concat([X_test_fold, test_values], axis=1)
            print(
                f"[fold-fit] {kind} {time.time() - started:.0f}s "
                f"컬럼 {len(train_values.columns)}개",
                flush=True,
            )

    feature_names = plan.all_columns()
    assert list(X_fold.columns) == feature_names, "fold 0 컬럼 집합이 피처 계획과 다르다."
    assert list(X_test_fold.columns) == feature_names, "test 컬럼 집합이 피처 계획과 다르다."
    return X_fold, X_test_fold, y, tr_idx, va_idx


def run_candidate(cfg, X_fold, X_test_fold, y, tr_idx, va_idx, epochs: int) -> dict:
    model_cfg = dataclasses.replace(
        cfg.model, params={**cfg.model.params, "epochs": epochs}
    )
    adapter = model_mod.create(model_cfg, SEED)
    model_mod.set_dataset_reference(adapter, X_fold, X_test_fold)
    started = time.time()
    va_pred = adapter.fit(
        X_fold.loc[tr_idx],
        y.loc[tr_idx],
        X_fold.loc[va_idx],
        y.loc[va_idx],
        None,
        None,
    )
    elapsed = time.time() - started
    auc = float(roc_auc_score(y.loc[va_idx], va_pred))
    diagnostics = model_mod.collect_training_diagnostics(adapter)
    return {
        "epochs": epochs,
        "seed": SEED,
        "fold": VALID_FOLD,
        "auc": auc,
        "diff_vs_champion": auc - CHAMPION_FOLD0_AUC,
        "fit_seconds": round(elapsed, 1),
        "model_training_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="일정 길이 fold 0 짝비교 (#382)")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[32, 12, 16, 20, 24],
        help="시험할 일정 길이. 첫 값 32는 champion 기준 재현이다.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("run-logs/schedule-length-fold0.jsonl"),
        help="결과 JSON Lines 경로. 기본값은 커밋하지 않는 run-logs/ 아래다.",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        help="연결 점검 전용. train/test 앞쪽 N행만 쓴다. 판정 실행에서는 쓰지 않는다.",
    )
    args = parser.parse_args()
    if args.smoke_rows:
        print("!! 연결 점검 모드: 이 결과는 판정에 쓰지 않는다.", flush=True)

    cfg = load_config(args.config, "screen")
    plan = FeaturePlan.from_config(cfg.features)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(int(json.loads(line)["epochs"]))
    todo = [e for e in args.epochs if e not in done]
    if done:
        print(f"이미 기록됨: {sorted(done)} | 남은 후보: {todo}", flush=True)
    if not todo:
        print("남은 후보 없음.", flush=True)
        return

    started = time.time()
    X_fold, X_test_fold, y, tr_idx, va_idx = build_fold0_matrices(cfg, plan, args.smoke_rows)
    print(
        f"[features] fold 0 행렬 완료 {time.time() - started:.0f}s "
        f"학습 {len(tr_idx)}행 검증 {len(va_idx)}행 컬럼 {X_fold.shape[1]}개",
        flush=True,
    )

    for epochs in todo:
        record = run_candidate(cfg, X_fold, X_test_fold, y, tr_idx, va_idx, epochs)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[result] epochs={epochs} auc={record['auc']:.7f} "
            f"짝차이={record['diff_vs_champion']:+.7f} "
            f"({record['fit_seconds']:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
