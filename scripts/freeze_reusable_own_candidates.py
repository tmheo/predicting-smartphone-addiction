"""재사용 적격 자체 후보의 변경 불가 동결 명세를 만든다. (#494)

현재 후보 풀 밖의 완료된 최상위 자체 실행을 실행 시작 시각과 실행 식별자 순서로
감사한다. 결과 지표, 근접 중복, 과거 기여와 모델 종류는 읽지 않고 다음만 검사한다.

- 시드 42, 43, 44의 평균본이고 깨끗한 코드 상태와 현행 입력 해시를 가진 실행인지.
- OOF와 시험 예측의 식별자 순서, 분할, 행 수, float64와 유한성이 맞는지.
- 현재 풀과 같은 구성 식별자, 같은 구성의 반복 실행과 정확 중복인지.
- 이슈 413과 419의 고정 학습 길이 트리 변형 41개가 사전 명단과 맞는지.

사용법:
    uv run python scripts/freeze_reusable_own_candidates.py \
        --run-cutoff 2026-08-28T13:01:08Z --verify-only
    uv run python scripts/freeze_reusable_own_candidates.py \
        --run-cutoff 2026-08-28T13:01:08Z

명세는 기본으로
`docs/research/reusable-own-candidate-freeze/<후보 집합 식별자>.json`에 쓴다.
같은 경로가 이미 있으면 덮어쓰지 않는다. 예측 배열은 읽기만 하며 복사하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.data import ID, file_sha256  # noqa: E402
from pipeline.judgment import CONFIRM_SEEDS  # noqa: E402
from pipeline.ledger import Pool  # noqa: E402
from pipeline.runs import TRACKING_URI  # noqa: E402
from pipeline.tracking import EXPERIMENT_NAME, mlflow_client  # noqa: E402

SCHEMA = "reusable-own-candidate-freeze/1"
GENERATOR_PATH = Path("scripts/freeze_reusable_own_candidates.py")
DEFAULT_OUT_DIR = Path("docs/research/reusable-own-candidate-freeze")
TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
FOLDS_PATH = Path("artifacts/folds.parquet")
POOL_PATH = Path("artifacts/pool.yaml")
MAP_ISSUE = {
    "number": 493,
    "title": "지도: 과거 자체 확정 실행으로 313개 확장 스택을 넓혀 교체 후보를 판정한다",
    "url": "https://github.com/tmheo/predicting-smartphone-addiction/issues/493",
}
TICKET_ISSUE = {
    "number": 494,
    "title": "재사용 적격 자체 후보 89개의 동결 명세를 만든다",
    "url": "https://github.com/tmheo/predicting-smartphone-addiction/issues/494",
}
EXPECTED_BASE_RUNS = 141
EXPECTED_DIRTY_RUNS = 8
EXPECTED_POOL_MEMBERS = 35
EXPECTED_AUDITED_RUNS = 98
EXPECTED_POOL_CONFIG_EXCLUSIONS = 7
EXPECTED_REPEATED_CONFIG_EXCLUSIONS = 1
EXPECTED_EXACT_DUPLICATE_EXCLUSIONS = 1
EXPECTED_CANDIDATES = 89
EXPECTED_FIXED_TREE_VARIANTS = 41
EXPECTED_RESTRAINED_CANDIDATES = 48

# 이슈 413의 18개 고정 학습 길이 트리 변형 가운데
# exp168_issue413_lgb_no_te_fixed20은 현재 풀 구성원이므로 후보 명단에서는 빠진다.
# 이슈 419의 24개와 합쳐 후보 안의 고정 학습 길이 트리 변형은 41개다.
FIXED_TRAINING_LENGTH_TREE_CONFIGS = (
    "exp159_issue413_xgb_hpo_fixed20",
    "exp160_issue413_xgb_hpo_fixed40",
    "exp161_issue413_xgb_hpo_fixed60",
    "exp162_issue413_xgb_no_te_fixed20",
    "exp163_issue413_xgb_no_te_fixed40",
    "exp164_issue413_xgb_no_te_fixed60",
    "exp165_issue413_lgb_ag25_fixed20",
    "exp166_issue413_lgb_ag25_fixed40",
    "exp167_issue413_lgb_ag25_fixed60",
    "exp169_issue413_lgb_no_te_fixed40",
    "exp170_issue413_lgb_no_te_fixed60",
    "exp171_issue413_cat_no_te_fixed20",
    "exp172_issue413_cat_no_te_fixed40",
    "exp173_issue413_cat_no_te_fixed60",
    "exp174_issue413_cat_exact_fixed20",
    "exp175_issue413_cat_exact_fixed40",
    "exp176_issue413_cat_exact_fixed60",
    "exp177_issue419_xgb_hpo_fixed10",
    "exp178_issue419_lgb_ag25_fixed10",
    "exp179_issue419_lgb_ag25_fixed05",
    "exp180_issue419_lgb_no_te_fixed10",
    "exp181_issue419_cat_no_te_fixed10",
    "exp182_issue419_cat_no_te_fixed05",
    "exp184_issue419_cat_exact_fixed05",
    "exp185_issue419_lgb_te_drop_gaming_fixed20",
    "exp186_issue419_lgb_te_drop_gaming_fixed10",
    "exp187_issue419_lgb_resid_pair_fixed20",
    "exp188_issue419_lgb_resid_pair_fixed10",
    "exp189_issue419_lgb_orig_knn_fixed20",
    "exp190_issue419_lgb_orig_knn_fixed10",
    "exp191_issue419_lgb_orig_proxy_residual_fixed20",
    "exp192_issue419_lgb_orig_proxy_residual_fixed10",
    "exp193_issue419_lgb_constrained_impute_fixed20",
    "exp194_issue419_lgb_constrained_impute_fixed10",
    "exp195_issue419_lgb_recon_orig_mean_top3_fixed20",
    "exp196_issue419_lgb_recon_orig_mean_top3_fixed10",
    "exp198_issue419_lgb_recon_ce_fixed10",
    "exp199_issue419_lgb_orig_cdf_diff_fixed20",
    "exp200_issue419_lgb_orig_cdf_diff_fixed10",
    "exp201_issue419_lgb_lattice_te_fixed20",
    "exp202_issue419_lgb_lattice_te_fixed10",
)

SELECTION_POLICY = (
    "완료된 최상위 자체 실행 가운데 seeds=42,43,44이고 git_dirty=False이며 현행 입력 해시가 맞는 실행만 감사한다.",
    "현재 후보 풀 실행 식별자는 감사 모집단에서 제외한다.",
    "감사 순서는 실행 시작 시각과 실행 식별자의 오름차순이며 결과 확인 뒤 바꾸지 않는다.",
    "현재 후보 풀과 구성 식별자가 같은 추가 실행은 제외한다.",
    "같은 구성 식별자의 반복 실행은 OOF와 시험 예측 배열 해시가 모두 같을 때만 첫 실행을 대표로 유지한다.",
    "현재 풀 또는 앞선 후보와 OOF와 시험 예측 배열 해시가 모두 같은 정확 중복은 제외한다.",
    "근접 중복, 단독 AUC, 과거 기여와 모델 종류는 읽거나 제외 근거로 사용하지 않는다.",
    "이슈 413과 419의 고정 학습 길이 트리 변형 명단은 결과와 무관하게 별도 절제 후보군으로 기록한다.",
    "후보 89개 또는 고정 학습 길이 트리 변형 41개를 뺀 절제 후보 48개와 맞지 않으면 동결하지 않는다.",
    "예측 배열은 저장소에 복사하거나 커밋하지 않는다.",
)


class FreezeError(RuntimeError):
    """재사용 적격 자체 후보 동결을 완결할 수 없는 불변식 위반."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeError(message)


def canonical_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def array_bytes(values: pd.Series | np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes(order="C")


def array_sha256(values: pd.Series | np.ndarray) -> str:
    return bytes_sha256(array_bytes(values))


def pair_sha256(oof: pd.Series | np.ndarray, test: pd.Series | np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(array_bytes(oof))
    digest.update(array_bytes(test))
    return digest.hexdigest()


def integer_array_sha256(values: pd.Series | np.ndarray) -> str:
    payload = np.ascontiguousarray(np.asarray(values, dtype="<i8")).tobytes(order="C")
    return bytes_sha256(payload)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_utc(value: str) -> tuple[datetime, str]:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "실행 기준 시각에 UTC 시간대가 없다")
    normalized = parsed.astimezone(UTC)
    return normalized, normalized.isoformat(timespec="seconds").replace("+00:00", "Z")


def git_bytes(commit: str, path: Path) -> bytes | None:
    shown = subprocess.run(
        ["git", "show", f"{commit}:{path.as_posix()}"],
        capture_output=True,
        check=False,
    )
    return shown.stdout if shown.returncode == 0 else None


def generator_state(*, require_clean: bool) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    current = GENERATOR_PATH.read_bytes()
    committed = git_bytes(commit, GENERATOR_PATH)
    if require_clean:
        _require(not dirty, "동결 명세는 깨끗한 코드 상태에서만 만들 수 있다")
        _require(committed is not None, f"생성 코드가 현재 커밋에 없다: {GENERATOR_PATH}")
        _require(committed == current, "현재 생성 코드가 기록된 커밋의 내용과 다르다")
    return {
        "path": str(GENERATOR_PATH),
        "sha256": bytes_sha256(current),
        "git_commit": commit,
        "git_dirty": dirty,
        "committed_content_matches": committed == current,
    }


def input_contract() -> tuple[dict[str, object], pd.Series, pd.Series, pd.Series]:
    for path in (TRAIN_PATH, TEST_PATH, FOLDS_PATH, POOL_PATH):
        _require(path.is_file(), f"필수 입력이 없다: {path}")
    train_ids = pd.read_csv(TRAIN_PATH, usecols=[ID])[ID]
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID])[ID]
    folds = pd.read_parquet(FOLDS_PATH, columns=[ID, "fold"])
    _require(not train_ids.duplicated().any(), "학습 식별자가 중복된다")
    _require(not test_ids.duplicated().any(), "시험 식별자가 중복된다")
    _require(not folds[ID].duplicated().any(), "분할 식별자가 중복된다")
    expected_folds = train_ids.map(folds.set_index(ID)["fold"])
    _require(not expected_folds.isna().any(), "학습 식별자 일부에 분할이 없다")
    _require(len(folds) == len(train_ids), "분할 행 수와 학습 행 수가 다르다")
    hashes = {
        "train": file_sha256(TRAIN_PATH),
        "test": file_sha256(TEST_PATH),
        "folds": file_sha256(FOLDS_PATH),
    }
    contract = {
        "train": {
            "path": str(TRAIN_PATH),
            "sha256": hashes["train"],
            "rows": len(train_ids),
            "id_sha256": integer_array_sha256(train_ids),
        },
        "test": {
            "path": str(TEST_PATH),
            "sha256": hashes["test"],
            "rows": len(test_ids),
            "id_sha256": integer_array_sha256(test_ids),
        },
        "folds": {
            "path": str(FOLDS_PATH),
            "sha256": hashes["folds"],
            "rows": len(folds),
            "assignment_sha256": integer_array_sha256(expected_folds),
        },
    }
    return contract, train_ids, test_ids, expected_folds


def run_sort_key(run: Any) -> tuple[int, str]:
    return int(run.info.start_time), str(run.info.run_id)


def run_summary(run: Any, reason: str) -> dict[str, object]:
    return {
        "config": run.data.params.get("experiment"),
        "run_id": run.info.run_id,
        "started_at": datetime.fromtimestamp(run.info.start_time / 1000, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "start_time_ms": run.info.start_time,
        "git_commit": run.data.tags.get("git_commit"),
        "git_dirty": run.data.tags.get("git_dirty"),
        "reason": reason,
    }


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_ids: pd.Series,
    expected_folds: pd.Series | None,
    label: str,
) -> None:
    required = [ID, "pred"] + (["fold"] if expected_folds is not None else [])
    missing = [column for column in required if column not in frame]
    _require(not missing, f"{label} 필수 열이 없다: {missing}")
    _require(len(frame) == len(expected_ids), f"{label} 행 수가 {len(frame)}이다")
    _require(not frame[ID].duplicated().any(), f"{label} 식별자가 중복된다")
    _require(
        np.array_equal(frame[ID].to_numpy(), expected_ids.to_numpy()),
        f"{label} 식별자 순서가 다르다",
    )
    if expected_folds is not None:
        _require(
            np.array_equal(frame["fold"].to_numpy(), expected_folds.to_numpy()),
            f"{label} 분할이 다르다",
        )
    _require(frame["pred"].dtype == np.dtype("float64"), f"{label} 예측 자료형이 {frame['pred'].dtype}이다")
    _require(
        bool(np.isfinite(frame["pred"].to_numpy()).all()),
        f"{label} 예측에 유한하지 않은 값이 있다",
    )


def artifact_path(client: Any, run_id: str, artifact_uri: str, name: str) -> Path:
    parsed = urlparse(artifact_uri)
    local_root = None
    if parsed.scheme == "file":
        local_root = Path(unquote(parsed.path))
    elif not parsed.scheme and Path(artifact_uri).is_absolute():
        local_root = Path(artifact_uri)
    if local_root is not None:
        path = local_root / name
        _require(path.is_file(), f"실행 {run_id}의 산출물이 없다: {name}")
        return path
    return Path(client.download_artifacts(run_id, name))


def root_artifacts(
    client: Any, run_id: str, artifact_uri: str
) -> tuple[str, dict[str, Path]]:
    names = {item.path for item in client.list_artifacts(run_id)}
    config_names = sorted(name for name in names if name.endswith((".yaml", ".yml")))
    _require(len(config_names) == 1, f"실행 {run_id}의 설정 YAML이 하나가 아니다: {config_names}")
    required = {"oof.parquet", "test_pred.parquet"}
    _require(required <= names, f"실행 {run_id}의 예측 산출물이 없다: {sorted(required - names)}")
    config_name = config_names[0]
    paths = {
        name: artifact_path(client, run_id, artifact_uri, name)
        for name in (config_name, "oof.parquet", "test_pred.parquet")
    }
    return config_name, paths


def audit_run(
    client: Any,
    run: Any,
    *,
    expected_config: str | None,
    input_hashes: dict[str, str],
    train_ids: pd.Series,
    test_ids: pd.Series,
    expected_folds: pd.Series,
) -> dict[str, object]:
    run_id = str(run.info.run_id)
    config = run.data.params.get("experiment")
    _require(run.info.status == "FINISHED", f"실행 {run_id}이 완료 상태가 아니다")
    _require(not run.data.tags.get("mlflow.parentRunId"), f"실행 {run_id}이 최상위 실행이 아니다")
    _require(run.data.params.get("seeds") == ",".join(map(str, CONFIRM_SEEDS)), f"실행 {run_id}의 시드가 다르다")
    _require(run.data.tags.get("git_dirty") == "False", f"실행 {run_id}의 코드 상태가 깨끗하지 않다")
    _require(bool(config), f"실행 {run_id}의 구성 식별자가 없다")
    if expected_config is not None:
        _require(config == expected_config, f"실행 {run_id}과 풀 장부의 구성 식별자가 다르다")
    for name, expected in input_hashes.items():
        _require(
            run.data.tags.get(f"sha256.{name}") == expected,
            f"실행 {run_id}의 {name} 입력 SHA-256이 현행 값과 다르다",
        )
    config_name, paths = root_artifacts(client, run_id, run.info.artifact_uri)
    config_bytes = paths[config_name].read_bytes()
    parsed = yaml.safe_load(config_bytes)
    _require(isinstance(parsed, dict), f"실행 {run_id}의 설정 YAML이 객체가 아니다")
    _require(parsed.get("name") == config, f"실행 {run_id}의 설정 name과 구성 식별자가 다르다")
    git_commit = run.data.tags.get("git_commit", "")
    _require(len(git_commit) == 40, f"실행 {run_id}의 git_commit이 올바르지 않다")
    committed = git_bytes(git_commit, Path("configs") / config_name)
    _require(committed is not None, f"실행 {run_id}의 커밋에서 설정을 찾지 못했다")
    _require(committed == config_bytes, f"실행 {run_id}의 설정 산출물과 실행 커밋이 다르다")
    oof = pd.read_parquet(paths["oof.parquet"])
    test = pd.read_parquet(paths["test_pred.parquet"])
    validate_prediction_frame(
        oof,
        expected_ids=train_ids,
        expected_folds=expected_folds,
        label=f"실행 {run_id} OOF",
    )
    validate_prediction_frame(
        test,
        expected_ids=test_ids,
        expected_folds=None,
        label=f"실행 {run_id} 시험 예측",
    )
    oof_values = oof["pred"]
    test_values = test["pred"]
    return {
        "config": config,
        "run_id": run_id,
        "started_at": datetime.fromtimestamp(run.info.start_time / 1000, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "start_time_ms": run.info.start_time,
        "seeds": list(CONFIRM_SEEDS),
        "git_commit": git_commit,
        "git_dirty": False,
        "config_artifact": {
            "name": config_name,
            "sha256": bytes_sha256(config_bytes),
            "committed_content_matches": True,
        },
        "input_sha256": dict(input_hashes),
        "oof": {
            "rows": len(oof),
            "array_sha256": array_sha256(oof_values),
        },
        "test": {
            "rows": len(test),
            "array_sha256": array_sha256(test_values),
        },
        "prediction_pair_sha256": pair_sha256(oof_values, test_values),
    }


def prediction_key(record: dict[str, object]) -> tuple[str, str]:
    oof = record["oof"]
    test = record["test"]
    assert isinstance(oof, dict) and isinstance(test, dict)
    return str(oof["array_sha256"]), str(test["array_sha256"])


def exclude_same_pool_configs(
    audited: list[dict[str, object]], pool_records: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pool_by_config = {str(record["config"]): record for record in pool_records}
    retained = []
    exclusions = []
    for record in audited:
        duplicate = pool_by_config.get(str(record["config"]))
        if duplicate is None:
            retained.append(record)
            continue
        exclusions.append(
            {
                "run": record,
                "reason": "현재 후보 풀과 구성 식별자가 같은 추가 실행",
                "pool_member": {
                    "config": duplicate["config"],
                    "run_id": duplicate["run_id"],
                },
            }
        )
    return retained, exclusions


def collapse_repeated_configs(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seen: dict[str, dict[str, object]] = {}
    retained = []
    exclusions = []
    for record in records:
        config = str(record["config"])
        representative = seen.get(config)
        if representative is None:
            seen[config] = record
            retained.append(record)
            continue
        _require(
            prediction_key(record) == prediction_key(representative),
            f"같은 구성 {config}의 반복 실행 예측 쌍이 다르다",
        )
        exclusions.append(
            {
                "run": record,
                "reason": "같은 구성의 예측 쌍이 같은 반복 실행",
                "representative": {
                    "config": representative["config"],
                    "run_id": representative["run_id"],
                },
            }
        )
    return retained, exclusions


def exclude_exact_duplicates(
    records: list[dict[str, object]], pool_records: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pool_by_pair: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in pool_records:
        pool_by_pair.setdefault(prediction_key(record), []).append(record)
    seen: dict[tuple[str, str], dict[str, object]] = {}
    retained = []
    exclusions = []
    for record in records:
        key = prediction_key(record)
        if key in pool_by_pair:
            exclusions.append(
                {
                    "run": record,
                    "reason": "현재 후보 풀 구성원과 OOF와 시험 예측 배열이 모두 같은 정확 중복",
                    "pool_members": [
                        {"config": item["config"], "run_id": item["run_id"]}
                        for item in pool_by_pair[key]
                    ],
                }
            )
            continue
        if key in seen:
            representative = seen[key]
            exclusions.append(
                {
                    "run": record,
                    "reason": "앞선 후보와 OOF와 시험 예측 배열이 모두 같은 정확 중복",
                    "representative": {
                        "config": representative["config"],
                        "run_id": representative["run_id"],
                    },
                }
            )
            continue
        seen[key] = record
        retained.append(record)
    return retained, exclusions


def verify_expected_counts(
    *,
    base_runs: list[Any],
    dirty_runs: list[Any],
    pool_records: list[dict[str, object]],
    audited: list[dict[str, object]],
    pool_config_exclusions: list[dict[str, object]],
    repeated_config_exclusions: list[dict[str, object]],
    exact_duplicate_exclusions: list[dict[str, object]],
    candidates: list[dict[str, object]],
    fixed_variants: list[dict[str, object]],
) -> None:
    expected = {
        "완료된 최상위 3시드 실행": (len(base_runs), EXPECTED_BASE_RUNS),
        "코드 상태가 깨끗하지 않은 실행": (len(dirty_runs), EXPECTED_DIRTY_RUNS),
        "현재 후보 풀 구성원": (len(pool_records), EXPECTED_POOL_MEMBERS),
        "풀 밖 감사 실행": (len(audited), EXPECTED_AUDITED_RUNS),
        "현재 풀과 같은 구성 식별자 제외": (
            len(pool_config_exclusions),
            EXPECTED_POOL_CONFIG_EXCLUSIONS,
        ),
        "같은 구성 반복 실행 제외": (
            len(repeated_config_exclusions),
            EXPECTED_REPEATED_CONFIG_EXCLUSIONS,
        ),
        "정확 중복 제외": (
            len(exact_duplicate_exclusions),
            EXPECTED_EXACT_DUPLICATE_EXCLUSIONS,
        ),
        "최종 후보": (len(candidates), EXPECTED_CANDIDATES),
        "고정 학습 길이 트리 변형": (
            len(fixed_variants),
            EXPECTED_FIXED_TREE_VARIANTS,
        ),
        "절제 뒤 후보": (
            len(candidates) - len(fixed_variants),
            EXPECTED_RESTRAINED_CANDIDATES,
        ),
    }
    failures = [f"{name} {actual} != {wanted}" for name, (actual, wanted) in expected.items() if actual != wanted]
    _require(not failures, "; ".join(failures))


def build_spec(
    *,
    run_cutoff: str,
    tracking_uri: str,
    require_clean_generator: bool,
) -> dict[str, object]:
    cutoff, normalized_cutoff = parse_utc(run_cutoff)
    generator = generator_state(require_clean=require_clean_generator)
    inputs, train_ids, test_ids, expected_folds = input_contract()
    input_hashes = {
        name: str(record["sha256"])
        for name, record in inputs.items()
        if isinstance(record, dict)
    }
    client, experiment_id = mlflow_client(tracking_uri)
    all_runs = client.search_runs([experiment_id], max_results=10_000)
    cutoff_ms = int(cutoff.timestamp() * 1000)
    expected_seed_text = ",".join(map(str, CONFIRM_SEEDS))
    base_runs = sorted(
        [
            run
            for run in all_runs
            if run.info.status == "FINISHED"
            and not run.data.tags.get("mlflow.parentRunId")
            and run.data.params.get("seeds") == expected_seed_text
            and run.info.start_time <= cutoff_ms
        ],
        key=run_sort_key,
    )
    dirty_runs = [run for run in base_runs if run.data.tags.get("git_dirty") != "False"]
    clean_runs = [run for run in base_runs if run.data.tags.get("git_dirty") == "False"]

    pool = Pool.load(POOL_PATH)
    pool_by_id = {member.run_id: member for member in pool.members}
    _require(len(pool_by_id) == len(pool.members), "현재 후보 풀 실행 식별자가 중복된다")
    clean_by_id = {run.info.run_id: run for run in clean_runs}
    missing_pool = sorted(set(pool_by_id) - set(clean_by_id))
    _require(not missing_pool, f"현재 후보 풀 실행이 자격 있는 모집단에 없다: {missing_pool}")
    pool_records = [
        audit_run(
            client,
            clean_by_id[member.run_id],
            expected_config=member.config,
            input_hashes=input_hashes,
            train_ids=train_ids,
            test_ids=test_ids,
            expected_folds=expected_folds,
        )
        for member in pool.members
    ]

    outside_runs = [run for run in clean_runs if run.info.run_id not in pool_by_id]
    audited = [
        audit_run(
            client,
            run,
            expected_config=None,
            input_hashes=input_hashes,
            train_ids=train_ids,
            test_ids=test_ids,
            expected_folds=expected_folds,
        )
        for run in outside_runs
    ]
    for order, record in enumerate(audited, start=1):
        record["audit_order"] = order

    after_pool_configs, pool_config_exclusions = exclude_same_pool_configs(
        audited, pool_records
    )
    after_repeats, repeated_config_exclusions = collapse_repeated_configs(
        after_pool_configs
    )
    candidates, exact_duplicate_exclusions = exclude_exact_duplicates(
        after_repeats, pool_records
    )
    for order, record in enumerate(candidates, start=1):
        record["order"] = order

    candidate_by_config = {str(record["config"]): record for record in candidates}
    _require(len(candidate_by_config) == len(candidates), "최종 후보 구성 식별자가 중복된다")
    unknown_fixed = sorted(set(FIXED_TRAINING_LENGTH_TREE_CONFIGS) - set(candidate_by_config))
    extra_fixed = sorted(
        config
        for config in candidate_by_config
        if "_issue413_" in config or "_issue419_" in config
        if config not in FIXED_TRAINING_LENGTH_TREE_CONFIGS
    )
    _require(not unknown_fixed, f"고정 학습 길이 트리 변형이 후보에 없다: {unknown_fixed}")
    _require(not extra_fixed, f"고정 학습 길이 이슈의 미분류 후보가 있다: {extra_fixed}")
    fixed_variants = [
        {
            "candidate_order": candidate_by_config[config]["order"],
            "config": config,
            "run_id": candidate_by_config[config]["run_id"],
        }
        for config in FIXED_TRAINING_LENGTH_TREE_CONFIGS
    ]
    fixed_variants.sort(key=lambda item: int(item["candidate_order"]))

    verify_expected_counts(
        base_runs=base_runs,
        dirty_runs=dirty_runs,
        pool_records=pool_records,
        audited=audited,
        pool_config_exclusions=pool_config_exclusions,
        repeated_config_exclusions=repeated_config_exclusions,
        exact_duplicate_exclusions=exact_duplicate_exclusions,
        candidates=candidates,
        fixed_variants=fixed_variants,
    )

    spec: dict[str, object] = {
        "schema": SCHEMA,
        "contract": {
            "map": MAP_ISSUE,
            "ticket": TICKET_ISSUE,
        },
        "run_cutoff": {
            "timestamp": normalized_cutoff,
            "basis": "Wayfinder 지도 이슈 493 생성 시점",
        },
        "selection_policy": list(SELECTION_POLICY),
        "generator": generator,
        "inputs": {
            **inputs,
            "pool": {
                "path": str(POOL_PATH),
                "sha256": file_sha256(POOL_PATH),
                "member_count": len(pool_records),
                "members": pool_records,
            },
            "run_store": {
                "tracking_uri": tracking_uri,
                "experiment_name": EXPERIMENT_NAME,
                "experiment_id": experiment_id,
                "database_file_sha256_omitted": (
                    "실행 저장소는 계속 추가되므로 파일 전체 해시 대신 기준 시각과 실행별 변경 불가 기록을 고정한다."
                ),
            },
        },
        "audit": {
            "query": {
                "status": "FINISHED",
                "top_level": True,
                "seeds": list(CONFIRM_SEEDS),
                "started_at_or_before": normalized_cutoff,
                "order": ["start_time_ms ASC", "run_id ASC"],
            },
            "counts": {
                "completed_top_level_three_seed_runs": len(base_runs),
                "dirty_runs_excluded": len(dirty_runs),
                "clean_runs": len(clean_runs),
                "current_pool_runs_excluded": len(pool_records),
                "outside_pool_runs_audited": len(audited),
                "same_pool_config_excluded": len(pool_config_exclusions),
                "repeated_config_excluded": len(repeated_config_exclusions),
                "exact_duplicate_excluded": len(exact_duplicate_exclusions),
            },
            "dirty_run_exclusions": [
                run_summary(run, "git_dirty가 False가 아님") for run in dirty_runs
            ],
            "same_pool_config_exclusions": pool_config_exclusions,
            "repeated_config_exclusions": repeated_config_exclusions,
            "exact_duplicate_exclusions": exact_duplicate_exclusions,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "fixed_training_length_tree_variants": {
            "definition": (
                "이슈 413의 학습 길이 20, 40, 60 고정 트리 변형 중 현재 풀 구성원 exp168을 뺀 17개와 "
                "이슈 419의 학습 길이 5, 10, 20 고정 트리 변형 24개다."
            ),
            "issue_urls": [
                "https://github.com/tmheo/predicting-smartphone-addiction/issues/413",
                "https://github.com/tmheo/predicting-smartphone-addiction/issues/419",
            ],
            "current_pool_omission": "exp168_issue413_lgb_no_te_fixed20",
            "count": len(fixed_variants),
            "members": fixed_variants,
        },
        "restrained_candidate_count": len(candidates) - len(fixed_variants),
    }
    spec["content_sha256"] = text_sha256(canonical_json(spec))
    spec["candidate_set_id"] = f"rocf-v1-{str(spec['content_sha256'])[:12]}"
    spec["frozen_at"] = now_iso()
    spec["spec_sha256"] = text_sha256(
        canonical_json({key: value for key, value in spec.items() if key != "spec_sha256"})
    )
    return spec


def verify_spec_file(path: Path) -> dict[str, object]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    _require(spec.get("schema") == SCHEMA, f"{path}: 명세 schema가 다르다")
    expected_spec = text_sha256(
        canonical_json({key: value for key, value in spec.items() if key != "spec_sha256"})
    )
    _require(spec.get("spec_sha256") == expected_spec, f"{path}: spec_sha256이 다르다")
    content = {
        key: value
        for key, value in spec.items()
        if key not in {"spec_sha256", "candidate_set_id", "frozen_at", "content_sha256"}
    }
    expected_content = text_sha256(canonical_json(content))
    _require(spec.get("content_sha256") == expected_content, f"{path}: content_sha256이 다르다")
    _require(
        spec.get("candidate_set_id") == f"rocf-v1-{expected_content[:12]}",
        f"{path}: 후보 집합 식별자가 내용 해시와 다르다",
    )
    _require(spec.get("candidate_count") == EXPECTED_CANDIDATES, f"{path}: 후보 수가 89가 아니다")
    _require(
        spec.get("restrained_candidate_count") == EXPECTED_RESTRAINED_CANDIDATES,
        f"{path}: 절제 후보 수가 48이 아니다",
    )
    fixed = spec.get("fixed_training_length_tree_variants")
    _require(isinstance(fixed, dict), f"{path}: 고정 학습 길이 변형 기록이 없다")
    _require(fixed.get("count") == EXPECTED_FIXED_TREE_VARIANTS, f"{path}: 고정 학습 길이 변형 수가 41이 아니다")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="재사용 적격 자체 후보 동결 명세 생성기 (#494)")
    parser.add_argument("--run-cutoff", help="후보 실행 시작 기준 시각(ISO 8601, UTC)")
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--out", type=Path, help="명세 파일 경로")
    parser.add_argument("--verify-only", action="store_true", help="감사만 하고 명세를 쓰지 않는다")
    parser.add_argument("--verify-spec", type=Path, help="기존 명세의 자체 해시와 개수를 검사한다")
    args = parser.parse_args()

    try:
        if args.verify_spec is not None:
            spec = verify_spec_file(args.verify_spec)
            print(
                f"명세 검사 통과: {args.verify_spec}, 후보 {spec['candidate_count']}개, "
                f"spec_sha256 {spec['spec_sha256']}"
            )
            return
        if not args.run_cutoff:
            parser.error("--run-cutoff가 필요하다")
        spec = build_spec(
            run_cutoff=args.run_cutoff,
            tracking_uri=args.tracking_uri,
            require_clean_generator=not args.verify_only,
        )
        print(
            f"감사 통과: 후보 {spec['candidate_count']}개, 절제 후보 "
            f"{spec['restrained_candidate_count']}개, 후보 집합 {spec['candidate_set_id']}"
        )
        if args.verify_only:
            return
        out = args.out or DEFAULT_OUT_DIR / f"{spec['candidate_set_id']}.json"
        if out.exists():
            raise FreezeError(f"동결 명세는 변경 불가다. 이미 있다: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        verify_spec_file(out)
        print(f"동결 명세 저장: {out}\n  spec_sha256 {spec['spec_sha256']}")
    except (FreezeError, OSError, KeyError, ValueError, yaml.YAMLError) as exc:
        sys.exit(f"동결 실패: {exc}")


if __name__ == "__main__":
    main()
