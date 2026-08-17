"""후보 풀 감사의 무결성, 중복, 구간과 영점 대조 계약. (#63)"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from pipeline.data import ID, TARGET
from pipeline.ledger import EntryEvidence, PoolMember
from pipeline.pool_audit import (
    AuditContext,
    CandidateArtifacts,
    audit_pool,
    prediction_array_sha256,
    verify_candidate,
)

N = 200
SEEDS = [42, 43, 44]


def make_context() -> AuditContext:
    rng = np.random.default_rng(1)
    ids = np.arange(1, N + 1)
    y = np.tile([0, 1], N // 2)
    train = pd.DataFrame(
        {
            ID: ids,
            TARGET: y,
            "age": rng.normal(30, 5, N),
            "screen": rng.normal(6, 2, N),
        }
    )
    train.loc[::4, "age"] = np.nan
    train.loc[::7, "screen"] = np.nan
    test = pd.DataFrame({ID: np.arange(1001, 1101), "age": 30.0, "screen": 6.0})
    folds = pd.DataFrame({ID: ids, "fold": np.arange(N) % 5})
    return AuditContext(
        train=train,
        test=test,
        folds=folds,
        input_hashes={"train": "train", "test": "test", "folds": "folds"},
    )


def make_member(config: str, auc: float) -> PoolMember:
    return PoolMember(
        run_id=f"run-{config}",
        config=config,
        oof_auc=auc,
        seeds=SEEDS,
        entered_at="2026-08-17",
        reason="test",
        evidence=EntryEvidence("champ", 0.9, 0.01, None, None, None, None, None),
    )


def make_candidate(
    context: AuditContext,
    config: str,
    pred: np.ndarray,
    test_pred: np.ndarray | None = None,
    *,
    seed_oofs: bool = True,
) -> CandidateArtifacts:
    test_values = test_pred if test_pred is not None else np.linspace(0.1, 0.9, len(context.test))
    oof = pd.DataFrame(
        {
            ID: context.train[ID],
            "fold": context.folds["fold"],
            "pred": np.asarray(pred, dtype=np.float64),
        }
    )
    test_frame = pd.DataFrame(
        {ID: context.test[ID], "pred": np.asarray(test_values, dtype=np.float64)}
    )
    config_bytes = yaml.safe_dump({"name": config}).encode()
    auc = float(roc_auc_score(context.train[TARGET], pred))
    seeds = {seed: oof.copy() for seed in SEEDS} if seed_oofs else {}
    return CandidateArtifacts(
        member=make_member(config, auc),
        status="FINISHED",
        params={"experiment": config, "seeds": "42,43,44"},
        metrics={"auc_oof": auc},
        tags={
            "git_dirty": "False",
            "sha256.train": "train",
            "sha256.test": "test",
            "sha256.folds": "folds",
        },
        config_bytes=config_bytes,
        committed_config_bytes=config_bytes,
        oof=oof,
        test_pred=test_frame,
        seed_oofs=seeds,
    )


def test_prediction_hash_is_little_endian_float64_array_bytes():
    values = pd.Series([0.1, 0.2, 0.3], dtype=np.float64)
    expected = hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()
    assert prediction_array_sha256(values) == expected


def test_verify_candidate_checks_seed_mean_and_marks_legacy_partial():
    context = make_context()
    pred = np.linspace(0.01, 0.99, N)
    complete = verify_candidate(make_candidate(context, "complete", pred), context)
    legacy = verify_candidate(
        make_candidate(context, "legacy", pred, seed_oofs=False), context
    )
    assert complete.valid
    assert complete.seed_mean_status == "완전 확인"
    assert legacy.valid
    assert legacy.seed_mean_status == "기존 기록 부분 확인"
    assert legacy.warnings


def test_verify_candidate_rejects_id_fold_precision_and_seed_mean_drift():
    context = make_context()
    pred = np.linspace(0.01, 0.99, N)
    candidate = make_candidate(context, "broken", pred)
    candidate.oof.loc[0, ID] = 999
    candidate.oof.loc[1, "fold"] = 9
    candidate.oof["pred"] = candidate.oof["pred"].astype(np.float32)
    candidate.seed_oofs[42].loc[2, "pred"] += 0.01
    check = verify_candidate(candidate, context)
    assert not check.valid
    assert any("id 순서" in failure for failure in check.failures)
    assert any("fold 배정" in failure for failure in check.failures)
    assert any("정밀도" in failure for failure in check.failures)
    assert any("시드 평균" in failure for failure in check.failures)


def test_audit_removes_exact_duplicate_before_quality_measurement():
    context = make_context()
    y = context.train[TARGET].to_numpy()
    rng = np.random.default_rng(4)
    strong = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.08, N), 0.001, 0.999)
    weaker_duplicate = strong.copy()
    independent = np.clip(0.25 + 0.5 * y + rng.normal(0, 0.16, N), 0.001, 0.999)
    test_values = np.linspace(0.1, 0.9, len(context.test))
    audit = audit_pool(
        [
            make_candidate(context, "strong", strong, test_values),
            make_candidate(context, "duplicate", weaker_duplicate, test_values),
            make_candidate(context, "independent", independent, test_values[::-1]),
        ],
        context,
        random_count=8,
    )
    assert "strong" in audit.retained_configs
    assert "duplicate" not in audit.retained_configs
    assert len(audit.duplicate_decisions) == 1
    assert audit.duplicate_decisions[0].reason == "배열 해시 정확 중복"


def test_null_controls_are_reproducible_and_classify_each_retained_member():
    context = make_context()
    y = context.train[TARGET].to_numpy()
    rng = np.random.default_rng(9)
    candidates = []
    for index, noise in enumerate((0.09, 0.13, 0.18)):
        pred = np.clip(0.2 + 0.6 * y + rng.normal(0, noise, N), 0.001, 0.999)
        test_pred = rng.random(len(context.test))
        candidates.append(make_candidate(context, f"m{index}", pred, test_pred))
    first = audit_pool(candidates, context, null_seed=123, random_count=16)
    second = audit_pool(candidates, context, null_seed=123, random_count=16)
    assert first.controls == second.controls
    assert first.controls.lower <= first.controls.upper
    assert first.controls.best_member == "m0"
    assert set(first.retained_configs) == {"m0", "m1", "m2"}
    assert {quality.action for quality in first.quality} == {"유지"}


def test_audit_keeps_negative_equal_rank_contributor_for_nested_evaluation():
    context = make_context()
    y = context.train[TARGET].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(21)
    strong_a = np.clip(0.1 + 0.8 * y + rng.normal(0, 0.04, N), 0.001, 0.999)
    strong_b = np.clip(0.1 + 0.8 * y + rng.normal(0, 0.06, N), 0.001, 0.999)
    harmful = np.clip(0.9 - 0.8 * y + rng.normal(0, 0.04, N), 0.001, 0.999)
    audit = audit_pool(
        [
            make_candidate(context, "strong_a", strong_a, rng.random(len(context.test))),
            make_candidate(context, "strong_b", strong_b, rng.random(len(context.test))),
            make_candidate(context, "harmful", harmful, rng.random(len(context.test))),
        ],
        context,
        random_count=8,
    )
    quality = {item.config: item for item in audit.quality}
    assert quality["harmful"].contribution < 0
    assert quality["harmful"].action == "유지"
    assert set(audit.retained_configs) == {"strong_a", "strong_b", "harmful"}


def test_audit_reports_fixed_missing_count_segments():
    context = make_context()
    y = context.train[TARGET].to_numpy()
    rng = np.random.default_rng(13)
    candidates = [
        make_candidate(
            context,
            "segment",
            np.clip(0.2 + 0.6 * y + rng.normal(0, 0.1, N), 0.001, 0.999),
            rng.random(len(context.test)),
        )
    ]
    audit = audit_pool(candidates, context, random_count=4)
    assert list(audit.quality[0].segment_aucs) == [
        "결측 0",
        "결측 1-2",
        "결측 3-5",
        "결측 6+",
    ]
