"""이슈 520의 초기 점수 확장 후보와 짝비교 판정 계약을 결과 전에 동결한다.

사용법:
    uv run python scripts/freeze_issue520_initial_score_candidates.py
    uv run python scripts/freeze_issue520_initial_score_candidates.py --verify-only

이 도구는 실행 저장소와 예측값을 읽지 않는다.
세 기준 설정, 세 후보 설정, 고정 folds와 판정 규칙만 읽어 변경 불가 기록을 만든다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.config import load_config
from pipeline.data import file_sha256
from pipeline.initial_score import create as create_initial_score
from pipeline.judgment import DUPLICATE_SPEARMAN, ENTRY_FLOOR_MARGIN

OUTPUT_PATH = REPO_ROOT / "artifacts/issue520-initial-score-extension-precommit.json"
FOLDS_PATH = REPO_ROOT / "artifacts/folds.parquet"
POOL_PATH = REPO_ROOT / "artifacts/pool.yaml"
CHAMPION_PATH = REPO_ROOT / "artifacts/champion.yaml"
PAIR_RECORDER_PATH = REPO_ROOT / "scripts/record_initial_score_pair.py"
RAW_COLUMNS = [
    "age",
    "gender",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "stress_level",
    "academic_work_impact",
]
CATEGORICAL_COLUMNS = ["gender", "stress_level", "academic_work_impact"]
INITIAL_SCORE = {
    "kind": "nested_logistic_onehot",
    "params": {
        "cols": RAW_COLUMNS,
        "categorical": CATEGORICAL_COLUMNS,
        "C": 100.0,
        "max_iter": 3000,
        "onehot_max_card": 5000,
        "inner_splits": 10,
        "clip": 1.0e-6,
    },
}
EFFECTIVE_LOGISTIC_DEFAULTS = {"penalty": "l2", "solver": "lbfgs"}
PAIRS = (
    (
        "catboost",
        "configs/exp071_cat_exact_no_te.yaml",
        "configs/exp210_issue520_cat_lr_onehot_init.yaml",
    ),
    (
        "xgboost",
        "configs/exp111_xgb_depth8_no_te.yaml",
        "configs/exp211_issue520_xgb_lr_onehot_init.yaml",
    ),
    (
        "lightgbm_fixed20",
        "configs/exp168_issue413_lgb_no_te_fixed20.yaml",
        "configs/exp212_issue520_lgb_fixed20_lr_onehot_init.yaml",
    ),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="기존 사전 기록과 현재 고정 입력의 내용 해시 및 구조만 검증한다.",
    )
    return parser.parse_args()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _self_hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["precommit_sha256"] = _canonical_sha256(result)
    return result


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"YAML 루트가 객체가 아니다: {_relative(path)}")
    return raw


def _input(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"동결 입력이 없다: {_relative(path)}")
    return {"path": _relative(path), "sha256": file_sha256(path)}


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _pair_record(
    ordinal: int,
    key: str,
    baseline_relative: str,
    candidate_relative: str,
) -> dict[str, Any]:
    baseline_path = REPO_ROOT / baseline_relative
    candidate_path = REPO_ROOT / candidate_relative
    baseline = _load_yaml(baseline_path)
    candidate = _load_yaml(candidate_path)

    baseline_loaded = load_config(baseline_path, "screen")
    candidate_loaded = load_config(candidate_path, "screen")
    provider = create_initial_score(candidate_loaded.initial_score)
    if provider is None:
        raise ValueError(f"{candidate_relative}: 초기 점수 생성기가 만들어지지 않았다.")
    if baseline_loaded.initial_score is not None:
        raise ValueError(f"{baseline_relative}: 기준 설정에 initial_score가 이미 있다.")
    if candidate.get("initial_score") != INITIAL_SCORE:
        raise ValueError(f"{candidate_relative}: 고정 초기 점수 좌표와 다르다.")
    if baseline_path.stem != baseline["name"]:
        raise ValueError(f"{baseline_relative}: 파일 이름과 실험 이름이 다르다.")
    if candidate_path.stem != candidate["name"]:
        raise ValueError(f"{candidate_relative}: 파일 이름과 실험 이름이 다르다.")

    normalized_baseline = dict(baseline)
    normalized_candidate = dict(candidate)
    normalized_baseline.pop("name")
    normalized_candidate.pop("name")
    initial_score = normalized_candidate.pop("initial_score", None)
    if normalized_candidate != normalized_baseline:
        raise ValueError(
            f"{candidate_relative}: name과 initial_score 외 구조가 기준과 다르다."
        )
    if initial_score != INITIAL_SCORE:
        raise ValueError(f"{candidate_relative}: initial_score 구조가 고정값과 다르다.")
    if baseline_loaded.model.kind != candidate_loaded.model.kind:
        raise ValueError(f"{candidate_relative}: 기준과 모형 계열이 다르다.")
    if key == "lightgbm_fixed20" and (
        candidate["model"]["params"].get("n_estimators") != 913
        or candidate["model"]["fit"] != {}
    ):
        raise ValueError("고정 LightGBM 후보의 913회 반복 또는 빈 fit 블록이 바뀌었다.")

    return {
        "ordinal": ordinal,
        "key": key,
        "model_kind": candidate_loaded.model.kind,
        "baseline": {
            "name": baseline["name"],
            **_input(baseline_path),
            "semantic_sha256": _canonical_sha256(baseline),
        },
        "candidate": {
            "name": candidate["name"],
            **_input(candidate_path),
            "semantic_sha256": _canonical_sha256(candidate),
        },
        "allowed_semantic_difference": [
            {"operation": "replace", "path": "/name"},
            {"operation": "add", "path": "/initial_score"},
        ],
        "common_config_semantic_sha256": _canonical_sha256(normalized_baseline),
    }


def _assert_candidate_names_are_new(records: list[dict[str, Any]]) -> None:
    candidate_paths = {
        (REPO_ROOT / record["candidate"]["path"]).resolve() for record in records
    }
    names: dict[str, list[str]] = {}
    for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        raw = _load_yaml(path)
        name = raw.get("name")
        if isinstance(name, str):
            names.setdefault(name, []).append(_relative(path))
    for record in records:
        candidate = record["candidate"]
        matches = names.get(candidate["name"], [])
        if matches != [candidate["path"]]:
            raise ValueError(
                f"새 실험 이름 {candidate['name']}이 충돌하거나 파일과 다르다: {matches}"
            )
        if (REPO_ROOT / candidate["path"]).resolve() not in candidate_paths:
            raise AssertionError("후보 설정 경로 집합이 일치하지 않는다.")


def build_precommit() -> dict[str, Any]:
    records = [
        _pair_record(ordinal, key, baseline, candidate)
        for ordinal, (key, baseline, candidate) in enumerate(PAIRS, start=1)
    ]
    _assert_candidate_names_are_new(records)
    coordinate = {
        "config": INITIAL_SCORE,
        "effective_logistic_defaults": EFFECTIVE_LOGISTIC_DEFAULTS,
        "input_scope": "raw 12 columns only",
        "excluded_inputs": [
            "placebo_noise",
            "categorical_copies outputs",
            "xgb_impute_aux outputs",
            "constrained_impute_aux outputs",
            "all other feature-provider outputs",
        ],
        "tree_input_includes_logistic_prediction": False,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract_version": "initial-score-paired-v1",
        "frozen_at_utc": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "issue": {
            "number": 520,
            "title": "세 후보 설정과 짝비교 판정 기록 형식을 결과 전에 동결한다",
            "url": "https://github.com/tmheo/predicting-smartphone-addiction/issues/520",
        },
        "map": {
            "number": 517,
            "title": "지도: 목표 평균 인코딩이 없는 트리 구성원에 정확값 선형 OOF 로짓 초기 점수를 확장한다",
            "url": "https://github.com/tmheo/predicting-smartphone-addiction/issues/517",
        },
        "source_commit_before_freeze": _git_head(),
        "prediction_values_read_while_freezing": False,
        "inputs": {
            "folds": {**_input(FOLDS_PATH), "role": "고정 바깥쪽 5분할과 행 순서"},
            "candidate_pool_at_freeze": _input(POOL_PATH),
            "champion_at_freeze": _input(CHAMPION_PATH),
        },
        "initial_score_coordinate": {
            "value": coordinate,
            "sha256": _canonical_sha256(coordinate),
        },
        "execution_order": [record["key"] for record in records],
        "pairs": records,
        "pair_manifest_sha256": _canonical_sha256(records),
        "execution_contract": {
            "screen_seed": 42,
            "confirmation_seeds": [42, 43, 44],
            "outer_folds": [0, 1, 2, 3, 4],
            "same_commit_within_pair": True,
            "same_provider_and_runtime_class_within_pair": True,
            "clean_git_state": True,
            "input_hashes_must_match_within_pair": [
                "train",
                "test",
                "folds",
            ],
            "candidate_config_artifact_must_match_frozen_sha256": True,
            "baseline_config_artifact_must_match_frozen_sha256": True,
            "catboost_local_cpu_timeout_seconds": 10800,
            "formal_candidate_training_in_this_ticket": False,
        },
        "judgment_contract": {
            "entry_floor_margin": ENTRY_FLOOR_MARGIN,
            "duplicate_spearman_threshold": DUPLICATE_SPEARMAN,
            "general_admission": {
                "required": [
                    "seed42_candidate_minus_baseline_auc > 0",
                    "candidate_auc >= champion_auc - entry_floor_margin",
                    "nearest_pool_member_spearman < duplicate_spearman_threshold",
                    "candidate-pool-v2 core-combiner admission delta > 0",
                ],
                "action": "candidate-pool-v2 admission",
            },
            "atomic_replacement": {
                "trigger": [
                    "nearest_pool_member_spearman >= duplicate_spearman_threshold",
                    "nearest member is exp209_issue505_lgb_lr_onehot_init or has lineage group issue505 or issue517-initial-score-extension",
                ],
                "required": [
                    "seed42_candidate_minus_baseline_auc > 0",
                    "candidate_auc >= champion_auc - entry_floor_margin",
                    "candidate-pool-v2 core-combiner replacement delta > 0",
                ],
                "action": "replace exactly the nearest lineage member",
            },
            "candidate_pool_contract_version": "candidate-pool-v2",
            "candidate_pool_combiner_scope": "core",
            "serial_registration": True,
            "public_score_used": False,
            "final_rank_used": False,
            "result_driven_coordinate_changes_allowed": False,
        },
        "result_record_contract": {
            "script": _relative(PAIR_RECORDER_PATH),
            "schema_version": 1,
            "required_sections": [
                "frozen_contract",
                "execution_identity",
                "baseline",
                "candidate",
                "seed42_pairwise",
                "baseline_reproduction",
                "first_stage",
                "entry_and_duplicate",
                "candidate_pool_v2",
                "decision",
                "artifact_hashes",
                "record_sha256",
            ],
        },
        "code": {
            "freeze_script": _input(Path(__file__).resolve()),
            "pair_recorder": _input(PAIR_RECORDER_PATH),
        },
    }
    return _self_hashed(payload)


def validate_precommit(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("사전 기록 schema_version이 1이 아니다.")
    if payload.get("contract_version") != "initial-score-paired-v1":
        raise ValueError("사전 기록 계약 판본이 다르다.")
    expected_hash = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "precommit_sha256"}
    )
    if payload.get("precommit_sha256") != expected_hash:
        raise ValueError("사전 기록 자체 해시가 본문과 다르다.")

    for section in ("inputs", "code"):
        for name, record in payload[section].items():
            path = REPO_ROOT / record["path"]
            if not path.is_file() or file_sha256(path) != record["sha256"]:
                raise ValueError(f"동결 입력이 바뀌었다: {section}.{name}")
    rebuilt_pairs = [
        _pair_record(ordinal, key, baseline, candidate)
        for ordinal, (key, baseline, candidate) in enumerate(PAIRS, start=1)
    ]
    _assert_candidate_names_are_new(rebuilt_pairs)
    if payload["pairs"] != rebuilt_pairs:
        raise ValueError("사전 기록의 설정 짝이 현재 구조 또는 내용과 다르다.")
    if payload["pair_manifest_sha256"] != _canonical_sha256(rebuilt_pairs):
        raise ValueError("설정 짝 manifest 해시가 다르다.")
    coordinate = payload["initial_score_coordinate"]
    if coordinate["value"]["config"] != INITIAL_SCORE:
        raise ValueError("사전 기록의 초기 점수 설정이 고정 좌표와 다르다.")
    if coordinate["sha256"] != _canonical_sha256(coordinate["value"]):
        raise ValueError("초기 점수 좌표 해시가 다르다.")
    if payload["inputs"]["folds"]["sha256"] != file_sha256(FOLDS_PATH):
        raise ValueError("folds 해시가 바뀌었다.")


def main() -> None:
    args = _args()
    if args.verify_only:
        payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        validate_precommit(payload)
        print(
            "검증 완료: "
            f"pairs={len(payload['pairs'])} "
            f"coordinate={payload['initial_score_coordinate']['sha256']} "
            f"precommit={payload['precommit_sha256']}"
        )
        return

    if OUTPUT_PATH.exists():
        raise ValueError(
            f"변경 불가 사전 기록이 이미 있다: {_relative(OUTPUT_PATH)}. "
            "검증은 --verify-only를 사용한다."
        )
    payload = build_precommit()
    validate_precommit(payload)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "동결 완료: "
        f"pairs={len(payload['pairs'])} "
        f"folds={payload['inputs']['folds']['sha256']} "
        f"coordinate={payload['initial_score_coordinate']['sha256']} "
        f"precommit={payload['precommit_sha256']}"
    )


if __name__ == "__main__":
    main()
