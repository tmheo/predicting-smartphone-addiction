"""동결된 초기 점수 후보의 짝비교와 후보 풀 경로를 기계 판독 기록으로 만든다.

사용법:
    uv run python scripts/record_initial_score_pair.py \
        --candidate-config exp210_issue520_cat_lr_onehot_init \
        --baseline-run <run_id> \
        --candidate-run <run_id> \
        --recorded-baseline-run <run_id> \
        --out artifacts/issue520-exp210-screen.json

3시드 후보의 candidate-pool-v2 판정이 끝난 뒤에는 ``--pool-judgment``를 더해
일반 추가 또는 원자 교체의 최종 양수 기여와 판정 기록 해시까지 같은 형식에 넣는다.
이 도구는 후보 풀 장부를 변경하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.data import ID, TARGET, file_sha256
from pipeline.judgment import (
    DUPLICATE_SPEARMAN,
    ENTRY_FLOOR_MARGIN,
    load_run_facts,
    spearman,
)
from pipeline.ledger import Champion, Pool
from pipeline.runs import TRACKING_URI, MlflowRunStore

DEFAULT_PRECOMMIT = Path("artifacts/issue520-initial-score-extension-precommit.json")
LINEAGE_GROUPS = {"issue505", "issue517-initial-score-extension"}
INPUT_HASH_KEYS = ("train", "test", "folds")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--recorded-baseline-run")
    parser.add_argument("--pool-judgment", type=Path)
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"경로는 저장소 안에 있어야 한다: {path}") from exc


def _load_precommit(path: Path) -> dict[str, Any]:
    absolute = path if path.is_absolute() else REPO_ROOT / path
    payload = json.loads(absolute.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "사전 기록 schema_version이 다르다.")
    _require(
        payload.get("contract_version") == "initial-score-paired-v1",
        "사전 기록 계약 판본이 다르다.",
    )
    without_hash = {
        key: value for key, value in payload.items() if key != "precommit_sha256"
    }
    _require(
        payload.get("precommit_sha256") == _canonical_sha256(without_hash),
        "사전 기록 자체 해시가 본문과 다르다.",
    )
    for pair in payload["pairs"]:
        for arm in ("baseline", "candidate"):
            record = pair[arm]
            current = REPO_ROOT / record["path"]
            _require(current.is_file(), f"동결 설정이 없다: {record['path']}")
            _require(
                file_sha256(current) == record["sha256"],
                f"동결 설정 해시가 바뀌었다: {record['path']}",
            )
            raw = yaml.safe_load(current.read_text(encoding="utf-8"))
            _require(
                _canonical_sha256(raw) == record["semantic_sha256"],
                f"동결 설정 의미 해시가 바뀌었다: {record['path']}",
            )
    folds = payload["inputs"]["folds"]
    folds_path = REPO_ROOT / folds["path"]
    _require(
        folds_path.is_file() and file_sha256(folds_path) == folds["sha256"],
        "동결 folds 내용 해시가 바뀌었다.",
    )
    coordinate = payload["initial_score_coordinate"]
    _require(
        coordinate["sha256"] == _canonical_sha256(coordinate["value"]),
        "초기 점수 좌표 해시가 본문과 다르다.",
    )
    return payload


def _pair_for_candidate(
    precommit: Mapping[str, Any], candidate_config: str
) -> dict[str, Any]:
    matches = [
        pair
        for pair in precommit["pairs"]
        if pair["candidate"]["name"] == candidate_config
        or pair["key"] == candidate_config
    ]
    _require(len(matches) == 1, f"동결 후보를 하나 찾지 못했다: {candidate_config}")
    return dict(matches[0])


def _artifact_frame(store: MlflowRunStore, run_id: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(store.artifact_bytes_of(run_id, name)))


def _seed_oof(store: MlflowRunStore, run_id: str, seed: int) -> pd.DataFrame:
    frame = _artifact_frame(store, run_id, f"oof_seed_{seed}.parquet")
    _require(list(frame.columns) == [ID, "fold", "pred"], f"{run_id}: 시드 OOF 열이 다르다.")
    _require(not frame[ID].duplicated().any(), f"{run_id}: 시드 OOF id가 중복됐다.")
    _require(np.isfinite(frame["pred"]).all(), f"{run_id}: 시드 OOF 예측이 유한하지 않다.")
    return frame.set_index(ID)[["fold", "pred"]]


def _fold_aucs(oof: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    return {
        str(int(fold)): float(roc_auc_score(y.reindex(part.index), part["pred"]))
        for fold, part in oof.groupby("fold")
    }


def _pairwise(
    baseline: pd.DataFrame, candidate: pd.DataFrame, y: pd.Series
) -> dict[str, Any]:
    if not baseline.index.equals(candidate.index):
        candidate = candidate.reindex(baseline.index)
    _require(candidate["pred"].notna().all(), "후보 OOF 행 집합이 기준과 다르다.")
    _require(
        np.array_equal(baseline["fold"].to_numpy(), candidate["fold"].to_numpy()),
        "기준과 후보의 분할 배정이 다르다.",
    )
    baseline_auc = float(roc_auc_score(y.reindex(baseline.index), baseline["pred"]))
    candidate_auc = float(roc_auc_score(y.reindex(candidate.index), candidate["pred"]))
    baseline_folds = _fold_aucs(baseline, y)
    candidate_folds = _fold_aucs(candidate, y)
    deltas = {
        fold: candidate_folds[fold] - baseline_folds[fold]
        for fold in sorted(baseline_folds, key=int)
    }
    return {
        "baseline_auc": baseline_auc,
        "candidate_auc": candidate_auc,
        "delta": candidate_auc - baseline_auc,
        "baseline_fold_aucs": baseline_folds,
        "candidate_fold_aucs": candidate_folds,
        "fold_deltas": deltas,
        "fold_wins": int(sum(delta > 0 for delta in deltas.values())),
        "fold_total": len(deltas),
        "spearman_baseline_candidate": float(
            spearman(baseline["pred"], candidate["pred"])
        ),
        "passed": candidate_auc > baseline_auc,
    }


def _run_identity(
    store: MlflowRunStore,
    run_id: str,
    frozen: Mapping[str, Any],
    *,
    initial_score_required: bool,
) -> tuple[Any, dict[str, Any]]:
    facts = load_run_facts(run_id, store)
    meta = store.facts_of(run_id)
    _require(meta.status == "FINISHED", f"실행 {run_id}이 완료 상태가 아니다.")
    _require(facts.experiment == frozen["name"], f"실행 {run_id}의 실험 이름이 다르다.")
    _require(not facts.git_dirty, f"실행 {run_id}은 깨끗한 코드 상태가 아니다.")
    config_path = REPO_ROOT / frozen["path"]
    config_document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(
        store.config_of(run_id) == config_document,
        f"실행 {run_id}의 설정 산출물 의미가 동결 설정과 다르다.",
    )
    config_name = Path(frozen["path"]).name
    _require(
        store.artifact_sha256_of(run_id, config_name) == frozen["sha256"],
        f"실행 {run_id}의 설정 파일 해시가 동결값과 다르다.",
    )
    if initial_score_required:
        _require(
            config_document.get("initial_score") is not None,
            f"후보 실행 {run_id}에 initial_score가 없다.",
        )
    else:
        _require(
            config_document.get("initial_score") is None,
            f"기준 실행 {run_id}에 initial_score가 있다.",
        )
    input_hashes = {
        key: meta.tags.get(f"sha256.{key}") for key in INPUT_HASH_KEYS
    }
    _require(all(input_hashes.values()), f"실행 {run_id}의 필수 입력 해시가 빠졌다.")
    provider = meta.tags.get("remote.provider")
    runtime_class = meta.tags.get("remote.runtime_class")
    _require(
        (provider is None) == (runtime_class is None),
        f"실행 {run_id}의 원격 공급자와 실행 환경 등급 기록이 비대칭이다.",
    )
    runtime_identity = {
        "provider": provider or "local",
        "runtime_class": runtime_class or "local",
    }
    return facts, {
        "run_id": run_id,
        "experiment": facts.experiment,
        "seeds": facts.seeds,
        "auc_oof": facts.auc_oof,
        "seed_aucs": facts.seed_aucs,
        "fold_aucs": facts.fold_aucs,
        "git_commit": facts.git_commit,
        "git_dirty": facts.git_dirty,
        "input_sha256": input_hashes,
        "runtime_identity": runtime_identity,
    }


def _artifact_hashes(
    store: MlflowRunStore,
    run_id: str,
    config_name: str,
    seeds: list[int],
    *,
    include_initial_score: bool,
) -> dict[str, str]:
    names = [
        config_name,
        "oof.parquet",
        "test_pred.parquet",
        "submission.csv",
        *(f"oof_seed_{seed}.parquet" for seed in seeds),
    ]
    if include_initial_score:
        names.append("initial_score_evidence.json")
    return {name: store.artifact_sha256_of(run_id, name) for name in names}


def _first_stage_evidence(
    store: MlflowRunStore,
    run_id: str,
    seeds: list[int],
    coordinate: Mapping[str, Any],
) -> dict[str, Any]:
    name = "initial_score_evidence.json"
    payload = json.loads(store.artifact_bytes_of(run_id, name))
    _require(payload.get("schema_version") == 1, "초기 점수 근거 schema_version이 다르다.")
    _require(payload.get("kind") == "nested_logistic_onehot", "초기 점수 종류가 다르다.")
    entries = payload.get("entries")
    _require(isinstance(entries, list), "초기 점수 근거 entries가 목록이 아니다.")
    expected_coordinates = {(seed, fold) for seed in seeds for fold in range(5)}
    actual_coordinates = {
        (int(entry["seed"]), int(entry["outer_fold"])) for entry in entries
    }
    _require(actual_coordinates == expected_coordinates, "초기 점수 시드와 분할 좌표가 완전하지 않다.")
    _require(len(entries) == len(expected_coordinates), "초기 점수 근거 좌표가 중복됐다.")

    fixed = coordinate["value"]["config"]["params"]
    for entry in entries:
        _require(entry["cols"] == fixed["cols"], "초기 점수 근거의 입력 열이 동결값과 다르다.")
        _require(entry["inner_splits"] == fixed["inner_splits"], "내부 분할 수가 다르다.")
        _require(math.isclose(float(entry["clip"]), fixed["clip"]), "clip이 다르다.")
        model_params = entry["model_params"]
        for key in ("C", "max_iter", "onehot_max_card"):
            _require(model_params[key] == fixed[key], f"초기 점수 {key}가 다르다.")
        _require(
            set(entry["logit_range"]) == {"training", "validation", "test"},
            "초기 로짓 범위 기록이 불완전하다.",
        )
        _require(
            set(entry["sha256"]) == {"training", "validation", "test"},
            "초기 로짓 해시 기록이 불완전하다.",
        )

    by_seed: dict[str, Any] = {}
    for seed in seeds:
        items = sorted(
            (entry for entry in entries if int(entry["seed"]) == seed),
            key=lambda entry: int(entry["outer_fold"]),
        )
        by_seed[str(seed)] = {
            "outer_folds": [int(item["outer_fold"]) for item in items],
            "inner_oof_auc": [float(item["inner_oof_auc"]) for item in items],
            "validation_first_stage_auc": [
                float(item["validation_first_stage_auc"]) for item in items
            ],
            "training_logit_range": [item["logit_range"]["training"] for item in items],
            "validation_logit_range": [
                item["logit_range"]["validation"] for item in items
            ],
            "test_logit_range": [item["logit_range"]["test"] for item in items],
            "full_fit_iterations": [int(item["full_fit_iterations"]) for item in items],
            "inner_fit_iterations_max": [
                max(int(value) for value in item["inner_fit_iterations"])
                for item in items
            ],
            "n_features": [int(item["n_features"]) for item in items],
            "coordinate_sha256": [
                _canonical_sha256(
                    {
                        "seed": int(item["seed"]),
                        "outer_fold": int(item["outer_fold"]),
                        "logit_sha256": item["sha256"],
                    }
                )
                for item in items
            ],
        }
    return {
        "artifact_sha256": store.artifact_sha256_of(run_id, name),
        "schema_version": payload["schema_version"],
        "kind": payload["kind"],
        "entry_count": len(entries),
        "by_seed": by_seed,
    }


def _baseline_reproduction(
    store: MlflowRunStore,
    recorded_run_id: str | None,
    baseline_frozen: Mapping[str, Any],
    local: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    if recorded_run_id is None:
        return {"status": "not_requested"}
    recorded_facts, recorded_identity = _run_identity(
        store,
        recorded_run_id,
        baseline_frozen,
        initial_score_required=False,
    )
    _require(42 in recorded_facts.seed_aucs, "기록된 기준 실행에 seed 42 OOF가 없다.")
    recorded = _seed_oof(store, recorded_run_id, 42).reindex(local.index)
    _require(recorded["pred"].notna().all(), "기록된 기준 OOF 행이 로컬 기준과 다르다.")
    recorded_auc = float(roc_auc_score(y.reindex(recorded.index), recorded["pred"]))
    local_auc = float(roc_auc_score(y.reindex(local.index), local["pred"]))
    return {
        "status": "recorded",
        "recorded": recorded_identity,
        "recorded_seed42_auc": recorded_auc,
        "local_seed42_auc": local_auc,
        "local_minus_recorded": local_auc - recorded_auc,
        "spearman_local_recorded": float(spearman(local["pred"], recorded["pred"])),
        "max_abs_pred_diff": float(
            np.abs(local["pred"].to_numpy() - recorded["pred"].to_numpy()).max()
        ),
        "recorded_seed42_oof_sha256": store.artifact_sha256_of(
            recorded_run_id, "oof_seed_42.parquet"
        ),
    }


def _lineage_group(member: Any) -> str | None:
    judgment = getattr(member, "judgment", None)
    if judgment is None:
        return None
    path = getattr(judgment, "path", None)
    if not isinstance(path, str):
        return None
    absolute = REPO_ROOT / path
    if not absolute.is_file():
        return None
    raw = yaml.safe_load(absolute.read_text(encoding="utf-8"))
    return raw.get("change", {}).get("candidate", {}).get("model_lineage_group")


def _entry_and_duplicate(
    store: MlflowRunStore,
    candidate_run_id: str,
    candidate_facts: Any,
    candidate_seed42: pd.DataFrame,
) -> dict[str, Any]:
    champion = Champion.load(REPO_ROOT / "artifacts/champion.yaml")
    pool = Pool.load(REPO_ROOT / "artifacts/pool.yaml")
    _require(
        all(member.run_id != candidate_run_id for member in pool.members),
        "후보 실행이 이미 현재 후보 풀에 있어 사전 진입 진단을 만들 수 없다.",
    )
    candidate_average = store.oof_of(candidate_run_id)
    correlations: list[dict[str, Any]] = []
    for member in pool.members:
        prediction = store.oof_of(member.run_id).reindex(candidate_average.index)
        _require(prediction.notna().all(), f"풀 구성원 {member.config} OOF 행이 다르다.")
        seed42_prediction = prediction.reindex(candidate_seed42.index)
        correlations.append(
            {
                "config": member.config,
                "run_id": member.run_id,
                "lineage_group": _lineage_group(member),
                "spearman_seed_average": float(spearman(candidate_average, prediction)),
                "spearman_seed42": float(
                    spearman(candidate_seed42["pred"], seed42_prediction)
                ),
            }
        )
    use_seed_average = candidate_facts.seeds == [42, 43, 44]
    correlation_key = "spearman_seed_average" if use_seed_average else "spearman_seed42"
    correlations.sort(key=lambda item: (-item[correlation_key], item["config"]))
    nearest = correlations[0]
    lineage_match = (
        nearest["config"] == "exp209_issue505_lgb_lr_onehot_init"
        or nearest["lineage_group"] in LINEAGE_GROUPS
    )
    duplicate = nearest[correlation_key] >= DUPLICATE_SPEARMAN
    entry_auc = (
        candidate_facts.auc_oof
        if use_seed_average
        else candidate_facts.seed_aucs[42]
    )
    floor = champion.oof_auc - ENTRY_FLOOR_MARGIN
    return {
        "candidate_prediction_basis": "seed_average" if use_seed_average else "seed42",
        "champion": {
            "config": champion.config,
            "run_id": champion.run_id,
            "auc_oof": champion.oof_auc,
        },
        "entry_floor_margin": ENTRY_FLOOR_MARGIN,
        "entry_floor": floor,
        "candidate_entry_auc": entry_auc,
        "entry_floor_passed": entry_auc >= floor,
        "duplicate_spearman_threshold": DUPLICATE_SPEARMAN,
        "nearest_pool_member": nearest,
        "nearest_is_duplicate": duplicate,
        "nearest_is_atomic_replacement_lineage": lineage_match,
        "top5": correlations[:5],
        "pool_member_count": len(pool.members),
    }


def _preliminary_route(
    pairwise: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    pair_passed = bool(pairwise["delta"] > 0)
    floor_passed = bool(entry["entry_floor_passed"])
    duplicate = bool(entry["nearest_is_duplicate"])
    lineage = bool(entry["nearest_is_atomic_replacement_lineage"])
    if not pair_passed or not floor_passed:
        return {
            "route": "stop",
            "eligible_for_pool_judgment": False,
            "reason": "seed 42 짝차이 또는 진입 하한이 미달이다.",
            "replacement_target_run_id": None,
        }
    if duplicate and lineage:
        return {
            "route": "atomic_replacement",
            "eligible_for_pool_judgment": True,
            "reason": "최근접 구성원이 중복 문턱 이상의 초기 점수 계보다.",
            "replacement_target_run_id": entry["nearest_pool_member"]["run_id"],
        }
    if duplicate:
        return {
            "route": "stop",
            "eligible_for_pool_judgment": False,
            "reason": "중복 문턱 이상이지만 허용한 원자 교체 계보가 아니다.",
            "replacement_target_run_id": None,
        }
    return {
        "route": "general_admission",
        "eligible_for_pool_judgment": True,
        "reason": "짝차이, 진입 하한과 일반 중복 관문을 통과했다.",
        "replacement_target_run_id": None,
    }


def _pool_judgment(
    path: Path | None,
    candidate_run_id: str,
    route: Mapping[str, Any],
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "pending" if route["eligible_for_pool_judgment"] else "not_applicable",
            "reference_contribution_positive": None,
            "record": None,
        }
    absolute = path if path.is_absolute() else REPO_ROOT / path
    relative = _relative(absolute)
    raw = yaml.safe_load(absolute.read_text(encoding="utf-8"))
    _require(raw.get("contract_version") == "candidate-pool-v2", "후보 풀 계약 판본이 다르다.")
    _require(
        raw.get("selection", {}).get("combiner_scope") == "core",
        "후보 풀 판정이 핵심 결합 방식 범위를 쓰지 않았다.",
    )
    change = raw.get("change", {})
    _require(
        change.get("candidate", {}).get("run_id") == candidate_run_id,
        "후보 풀 판정의 후보 실행이 다르다.",
    )
    expected_action = {
        "general_admission": "admission",
        "atomic_replacement": "replacement",
    }.get(route["route"])
    _require(expected_action is not None, "후보 풀 판정을 받을 수 없는 경로다.")
    _require(change.get("action") == expected_action, "후보 풀 판정 동작이 고정 경로와 다르다.")
    _require(
        change.get("replaces_run_id") == route["replacement_target_run_id"],
        "후보 풀 판정의 원자 교체 대상이 고정 경로와 다르다.",
    )
    comparison = raw.get("nested_oof_comparison", {})
    delta = float(comparison["delta"])
    result = raw.get("result", {})
    contribution_positive = delta > 0
    adopted = result.get("state") == "adopted" and contribution_positive
    evidence = raw.get("evidence", {})
    evidence_path = REPO_ROOT / evidence.get("path", "")
    _require(evidence_path.is_file(), "후보 풀 판정 근거 파일이 없다.")
    _require(
        file_sha256(evidence_path) == evidence.get("sha256"),
        "후보 풀 판정 근거 파일 해시가 기록과 다르다.",
    )
    return {
        "status": "adopted" if adopted else "rejected",
        "reference_contribution_positive": contribution_positive,
        "nested_oof_delta": delta,
        "outer_fold_delta": comparison.get("outer_fold_delta"),
        "outer_fold_wins": comparison.get("outer_fold_wins"),
        "decision": result.get("decision"),
        "record": {
            "path": relative,
            "sha256": file_sha256(absolute),
            "judgment_id": raw.get("judgment_id"),
            "evidence": evidence,
        },
    }


def _decision(
    route: Mapping[str, Any], pool_judgment: Mapping[str, Any], seeds: list[int]
) -> dict[str, Any]:
    if route["route"] == "stop":
        return {"state": "rejected", "next_action": "none", "reason": route["reason"]}
    if seeds == [42]:
        return {
            "state": "screen_passed",
            "next_action": "run_confirm_seeds_42_43_44",
            "reason": route["reason"],
        }
    _require(seeds == [42, 43, 44], "후보 시드는 42 또는 42,43,44여야 한다.")
    if pool_judgment["status"] == "pending":
        action = (
            "run_candidate_pool_v2_admission"
            if route["route"] == "general_admission"
            else "run_candidate_pool_v2_atomic_replacement"
        )
        return {"state": "confirmation_ready", "next_action": action, "reason": route["reason"]}
    if pool_judgment["status"] == "adopted":
        return {
            "state": "adopted",
            "next_action": "register_serially_then_refresh_pool",
            "reason": "고정 경로의 candidate-pool-v2 기여가 엄격히 양수다.",
        }
    return {
        "state": "rejected",
        "next_action": "none",
        "reason": "고정 경로의 candidate-pool-v2 기여가 양수가 아니다.",
    }


def main() -> None:
    args = _args()
    precommit = _load_precommit(args.precommit)
    pair = _pair_for_candidate(precommit, args.candidate_config)
    store = MlflowRunStore(args.tracking_uri)

    baseline_facts, baseline = _run_identity(
        store,
        args.baseline_run,
        pair["baseline"],
        initial_score_required=False,
    )
    candidate_facts, candidate = _run_identity(
        store,
        args.candidate_run,
        pair["candidate"],
        initial_score_required=True,
    )
    _require(
        baseline_facts.git_commit == candidate_facts.git_commit,
        "기준과 후보 실행 커밋이 다르다.",
    )
    _require(42 in baseline_facts.seed_aucs, "기준 실행에 seed 42 OOF가 없다.")
    _require(42 in candidate_facts.seed_aucs, "후보 실행에 seed 42 OOF가 없다.")
    _require(
        baseline["input_sha256"] == candidate["input_sha256"],
        "기준과 후보의 입력 해시가 다르다.",
    )
    _require(
        baseline["runtime_identity"] == candidate["runtime_identity"],
        "기준과 후보의 공급자 또는 실행 환경 등급이 다르다.",
    )
    _require(
        candidate_facts.seeds in ([42], [42, 43, 44]),
        "후보 실행 시드는 42 또는 42,43,44여야 한다.",
    )
    _require(
        candidate["input_sha256"]["folds"] == precommit["inputs"]["folds"]["sha256"],
        "실행 folds 해시가 사전 동결값과 다르다.",
    )

    baseline42 = _seed_oof(store, args.baseline_run, 42)
    candidate42 = _seed_oof(store, args.candidate_run, 42)
    train_path = REPO_ROOT / yaml.safe_load(
        (REPO_ROOT / pair["baseline"]["path"]).read_text(encoding="utf-8")
    )["data"]["train"]
    y = pd.read_csv(train_path, usecols=[ID, TARGET]).set_index(ID)[TARGET]
    pairwise = _pairwise(baseline42, candidate42, y)
    _require(
        math.isclose(
            pairwise["baseline_auc"],
            baseline_facts.seed_aucs[42],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "기준 seed 42 OOF 재채점값과 실행 지표가 다르다.",
    )
    _require(
        math.isclose(
            pairwise["candidate_auc"],
            candidate_facts.seed_aucs[42],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "후보 seed 42 OOF 재채점값과 실행 지표가 다르다.",
    )
    candidate_average = store.oof_of(args.candidate_run)
    candidate_average_auc = float(
        roc_auc_score(y.reindex(candidate_average.index), candidate_average)
    )
    _require(
        math.isclose(
            candidate_average_auc,
            candidate_facts.auc_oof,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "후보 시드 평균 OOF 재채점값과 실행 지표가 다르다.",
    )
    reproduction = _baseline_reproduction(
        store,
        args.recorded_baseline_run,
        pair["baseline"],
        baseline42,
        y,
    )
    first_stage = _first_stage_evidence(
        store,
        args.candidate_run,
        candidate_facts.seeds,
        precommit["initial_score_coordinate"],
    )
    entry = _entry_and_duplicate(
        store,
        args.candidate_run,
        candidate_facts,
        candidate42,
    )
    route = _preliminary_route(pairwise, entry)
    pool_judgment = _pool_judgment(args.pool_judgment, args.candidate_run, route)
    decision = _decision(route, pool_judgment, candidate_facts.seeds)

    baseline_hashes = _artifact_hashes(
        store,
        args.baseline_run,
        Path(pair["baseline"]["path"]).name,
        baseline_facts.seeds,
        include_initial_score=False,
    )
    candidate_hashes = _artifact_hashes(
        store,
        args.candidate_run,
        Path(pair["candidate"]["path"]).name,
        candidate_facts.seeds,
        include_initial_score=True,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract_version": "initial-score-paired-v1",
        "frozen_contract": {
            "path": _relative(
                args.precommit if args.precommit.is_absolute() else REPO_ROOT / args.precommit
            ),
            "precommit_sha256": precommit["precommit_sha256"],
            "pair_key": pair["key"],
            "pair_manifest_sha256": precommit["pair_manifest_sha256"],
            "initial_score_coordinate_sha256": precommit["initial_score_coordinate"]["sha256"],
            "folds_sha256": precommit["inputs"]["folds"]["sha256"],
        },
        "execution_identity": {
            "git_commit": candidate_facts.git_commit,
            "git_dirty": False,
            "same_commit_within_pair": True,
            "input_sha256": candidate["input_sha256"],
        },
        "baseline": baseline,
        "candidate": candidate,
        "seed42_pairwise": pairwise,
        "baseline_reproduction": reproduction,
        "first_stage": first_stage,
        "entry_and_duplicate": entry,
        "preliminary_route": route,
        "candidate_pool_v2": pool_judgment,
        "decision": decision,
        "artifact_hashes": {
            "baseline": baseline_hashes,
            "candidate": candidate_hashes,
        },
        "public_score_used": False,
        "final_rank_used": False,
    }
    result["record_sha256"] = _canonical_sha256(result)
    output = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    _relative(output)
    _require(not output.exists(), f"변경 불가 짝비교 기록이 이미 있다: {_relative(output)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate": candidate_facts.experiment,
                "seed42_delta": pairwise["delta"],
                "entry_floor_passed": entry["entry_floor_passed"],
                "nearest": entry["nearest_pool_member"]["config"],
                "nearest_spearman": entry["nearest_pool_member"][
                    "spearman_seed_average"
                    if entry["candidate_prediction_basis"] == "seed_average"
                    else "spearman_seed42"
                ],
                "route": route["route"],
                "state": decision["state"],
                "record_sha256": result["record_sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
