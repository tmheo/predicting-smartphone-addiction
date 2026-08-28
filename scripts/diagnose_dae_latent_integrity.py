"""이슈 #483 1단계: 분할 적합 8차원 잡음 제거 잠재 표현의 무결성 진단.

정식 실행과 같은 피처 계획으로 난수 42, 바깥쪽 분할 하나에서 자동부호화기를 두 번 맞춰
행 정렬과 분할 계보, 유한성, 잠재 열 8개의 비퇴화, 목표값·검증 행·시험 행 미사용,
반복 일치와 평균 입력 복원 대비 재구성 손실을 확인한다.
AUC는 계산하지 않는다(이 단계의 AUC는 성능 중단에 쓰지 않는다).

사용법:
    uv run python scripts/diagnose_dae_latent_integrity.py \
        configs/exp206_issue483_cat_exact_cats_dae8.yaml --out-dir run-issue483/stage1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import data
from pipeline.config import load_config
from pipeline.data import ID, TARGET, file_sha256
from pipeline.denoising_autoencoder import DenoisingAutoencoderLatent
from pipeline.plan import FeaturePlan, prepare_fold_fit_input

SCHEMA_VERSION = 1
DEFAULT_FOLD = 0
DEFAULT_SEED = 42
MIN_LATENT_STD = 1e-6
MAX_LATENT_ABS_CORRELATION = 0.999


class _SpyProvider(DenoisingAutoencoderLatent):
    """fit이 실제로 받은 열과 행 식별자를 기록해 금지 입력 미사용을 확인한다."""

    def fit(self, train_fold: pd.DataFrame, seed: int) -> None:
        self.seen_columns_ = list(train_fold.columns)
        self.seen_ids_ = train_fold[ID].to_numpy().copy() if ID in train_fold.columns else None
        super().fit(train_fold, seed)


def _sha256_frame(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(",".join(map(str, frame.columns)).encode())
    for column in frame.columns:
        digest.update(np.ascontiguousarray(frame[column].to_numpy()).tobytes())
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=float).encode()
    ).hexdigest()


def _materialize(
    plan: FeaturePlan,
    train_ff: pd.DataFrame,
    test_ff: pd.DataFrame,
    tr_idx: pd.Index,
    va_idx: pd.Index,
    seed: int,
    fold: int,
) -> tuple[_SpyProvider, pd.DataFrame, pd.DataFrame, float]:
    providers = plan.new_fold_fit_providers()
    kinds = [kind for kind, _ in providers]
    if kinds.count("dae_latent") != 1:
        raise SystemExit(f"피처 계획에 dae_latent 제공자가 정확히 하나 있어야 한다: {kinds}")
    (_, reference) = next(item for item in providers if item[0] == "dae_latent")
    spy = _SpyProvider(
        reference.numeric_cols, reference.categorical_cols, reference.full_data_epochs
    )
    started = time.perf_counter()
    train_values, test_values, _ = plan.materialize_fold_fit_provider(
        kind="dae_latent",
        transformer=spy,
        train_input=train_ff,
        test_input=test_ff,
        training_index=tr_idx,
        validation_index=va_idx,
        seed=seed,
        fold=fold,
        recorder=None,
    )
    return spy, train_values, test_values, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="이슈 #483 1단계 무결성 진단")
    parser.add_argument("config")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config, "screen")
    plan = FeaturePlan.from_config(cfg.features)
    input_hashes = {
        "train": file_sha256(cfg.data.train),
        "test": file_sha256(cfg.data.test),
        "folds": file_sha256(cfg.data.folds),
    }
    train = data.load_csv(cfg.data.train)
    test = data.load_csv(cfg.data.test)
    data.align_categories(train, test, cfg.features.categorical)
    train, test = plan.apply_dataset_wide(train, test)
    train = data.attach_folds(train, cfg.data.folds)
    X = plan.build_matrix(train, args.seed)
    X_test = plan.build_matrix(test, args.seed)
    train_ff = prepare_fold_fit_input(train, X)
    test_ff = prepare_fold_fit_input(test, X_test)
    va_idx = train.index[train["fold"] == args.fold]
    tr_idx = train.index[train["fold"] != args.fold]
    folds_file = pd.read_parquet(cfg.data.folds)
    expected_valid_ids = set(folds_file.loc[folds_file["fold"] == args.fold, ID].tolist())

    checks: dict[str, bool] = {}
    reasons: list[str] = []

    def check(name: str, passed: bool, reason: str) -> None:
        checks[name] = bool(passed)
        if not passed:
            reasons.append(reason)

    runs = []
    for repeat in range(2):
        spy, train_values, test_values, seconds = _materialize(
            plan, train_ff, test_ff, tr_idx, va_idx, args.seed, args.fold
        )
        runs.append((spy, train_values, test_values, seconds))
    (spy_a, train_a, test_a, seconds_a), (spy_b, train_b, test_b, seconds_b) = runs
    evidence_a = spy_a.fit_evidence_
    evidence_b = spy_b.fit_evidence_
    assert evidence_a is not None and evidence_b is not None

    # 1. 행 정렬과 분할 계보
    check(
        "train_rows_aligned",
        train_a.index.equals(train_ff.index) and len(train_a) == len(train),
        "학습 행 잠재값 인덱스가 원본과 다르다.",
    )
    check(
        "test_rows_aligned",
        test_a.index.equals(test_ff.index) and len(test_a) == len(test),
        "시험 행 잠재값 인덱스가 원본과 다르다.",
    )
    check(
        "fold_lineage",
        set(train.loc[va_idx, ID].tolist()) == expected_valid_ids
        and input_hashes["folds"] == "5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4",
        "바깥쪽 분할 검증 행이 커밋된 folds.parquet과 다르다.",
    )
    # 2. 유한성
    check(
        "finite_train",
        bool(np.isfinite(train_a.to_numpy()).all()),
        "학습 행 잠재값에 비유한 값이 있다.",
    )
    check(
        "finite_test",
        bool(np.isfinite(test_a.to_numpy()).all()),
        "시험 행 잠재값에 비유한 값이 있다.",
    )
    # 3. 잠재 열 8개 각각의 비퇴화
    stds = {
        "train": train_a.loc[tr_idx].std(ddof=0),
        "valid": train_a.loc[va_idx].std(ddof=0),
        "test": test_a.std(ddof=0),
    }
    corr = train_a.loc[tr_idx].corr().to_numpy()
    off_diagonal = np.abs(corr - np.eye(len(corr))).max()
    check(
        "latent_non_degenerate",
        all((s > MIN_LATENT_STD).all() for s in stds.values())
        and off_diagonal < MAX_LATENT_ABS_CORRELATION,
        "잠재 열이 상수이거나 서로 거의 같다.",
    )
    # 4. 목표값, 바깥쪽 검증 행, 시험 행과 원본 자료 미사용
    declared = spy_a.reuse_input_columns()
    training_ids = set(train.loc[tr_idx, ID].tolist())
    check(
        "fit_saw_only_declared_columns",
        spy_a.seen_columns_ == [ID, *declared] and TARGET not in spy_a.seen_columns_,
        f"fit이 선언 밖 열을 받았다: {spy_a.seen_columns_}",
    )
    check(
        "fit_saw_only_training_rows",
        spy_a.seen_ids_ is not None
        and len(spy_a.seen_ids_) == len(tr_idx)
        and set(spy_a.seen_ids_.tolist()) == training_ids,
        "fit이 바깥쪽 학습 행 밖의 행을 받았다.",
    )
    check(
        "uses_target_declared_false",
        spy_a.uses_target is False,
        "제공자가 타깃 참조를 선언했다.",
    )
    # transform은 행 단위다: 학습 행만 변환한 결과가 전체 변환의 학습 행과 같다.
    subset = spy_a.transform(train_ff.loc[tr_idx, [ID, *declared]])
    check(
        "transform_row_wise",
        np.array_equal(subset.to_numpy(), train_a.loc[tr_idx].to_numpy()),
        "transform 결과가 다른 행의 존재에 따라 달라진다.",
    )
    # 5. 반복 일치
    state = lambda spy: {  # noqa: E731
        "numeric_min": spy.numeric_min_,
        "numeric_max": spy.numeric_max_,
        "vocab": spy.vocab_,
    }
    check(
        "repeat_preprocessing_identical",
        _sha256_json(state(spy_a)) == _sha256_json(state(spy_b)),
        "같은 입력과 난수에서 전처리 상태가 다르다.",
    )
    check(
        "repeat_selected_epochs_identical",
        evidence_a["selected_epochs"] == evidence_b["selected_epochs"]
        and evidence_a["epochs_run"] == evidence_b["epochs_run"],
        "같은 입력과 난수에서 학습 종료 시점이 다르다.",
    )
    train_hash_a, train_hash_b = _sha256_frame(train_a), _sha256_frame(train_b)
    test_hash_a, test_hash_b = _sha256_frame(test_a), _sha256_frame(test_b)
    check(
        "repeat_latent_identical",
        train_hash_a == train_hash_b and test_hash_a == test_hash_b,
        "같은 입력과 난수에서 잠재값 해시가 다르다.",
    )
    # 6. 평균 입력 복원보다 나은 내부 재구성 손실
    check(
        "better_than_mean_input",
        evidence_a["best_valid_loss"] is not None
        and evidence_a["mean_input_baseline_valid_loss"] is not None
        and evidence_a["best_valid_loss"] < evidence_a["mean_input_baseline_valid_loss"],
        "내부 재구성 손실이 평균 입력 복원보다 낫지 않다.",
    )
    per_column_better = {
        c: evidence_a["best_valid_loss_per_column"][c]
        < evidence_a["mean_input_baseline_per_column"][c]
        for c in evidence_a["best_valid_loss_per_column"]
    }

    latent_train = pd.concat(
        [train[[ID, "fold"]].reset_index(drop=True), train_a.reset_index(drop=True)], axis=1
    )
    latent_test = pd.concat([test[[ID]].reset_index(drop=True), test_a.reset_index(drop=True)], axis=1)
    latent_train.to_parquet(out_dir / "latent_train_fold0_seed42.parquet", index=False)
    latent_test.to_parquet(out_dir / "latent_test_fold0_seed42.parquet", index=False)

    passed = all(checks.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "issue": 483,
        "stage": "integrity",
        "config": str(cfg.source_path),
        "experiment": cfg.name,
        "fold": args.fold,
        "seed": args.seed,
        "input_sha256": input_hashes,
        "provider_settings": spy_a.reuse_settings(),
        "execution": spy_a.reuse_execution(),
        "rows": {
            "train": int(len(train)),
            "outer_training": int(len(tr_idx)),
            "outer_validation": int(len(va_idx)),
            "test": int(len(test)),
        },
        "fit_seconds": {"first": seconds_a, "second": seconds_b},
        "fit_evidence": {
            key: value for key, value in evidence_a.items() if key != "history"
        },
        "history_first": evidence_a["history"],
        "latent_std": {name: series.round(10).to_dict() for name, series in stds.items()},
        "latent_max_abs_off_diagonal_correlation": float(off_diagonal),
        "latent_sha256": {
            "train_first": train_hash_a,
            "train_second": train_hash_b,
            "test_first": test_hash_a,
            "test_second": test_hash_b,
        },
        "per_column_better_than_mean_input": per_column_better,
        "checks": checks,
        "reasons": reasons,
        "passed": passed,
    }
    (out_dir / "dae_integrity.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n"
    )
    print(json.dumps({"passed": passed, "checks": checks, "reasons": reasons}, ensure_ascii=False))
    print(
        f"selected_epochs={evidence_a['selected_epochs']} epochs_run={evidence_a['epochs_run']} "
        f"valid_loss={evidence_a['best_valid_loss']:.6f} "
        f"baseline={evidence_a['mean_input_baseline_valid_loss']:.6f} "
        f"fit_seconds={seconds_a:.1f}/{seconds_b:.1f}"
    )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
