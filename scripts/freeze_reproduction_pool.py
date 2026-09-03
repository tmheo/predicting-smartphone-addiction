"""재현 전용 풀 동결 명세를 만든다. (#632, 지도 #619)

이슈 623에서 3시드로 학습해 main MLflow에 반입한 제약 파생 사다리 설정 12개를
재현 구성원으로, 계열별 기준 재실행 4개를 근거 기록으로, 자체 36개 기준값과
314 확장 기준값을 기준 팔 값으로 한 파일에 동결한다. 중첩 결합 판정 결과를 보기
전에 만들며, 판정 회차 스펙(scripts/round_issue624_*.py)이 이 파일만 봉인 입력으로
읽는다. 대회 기록(artifacts/pool.yaml, 314 확장 스택, 재학습 장부)은 읽기만 한다.

검사:
- 16개 실행이 완료 상태, 최상위, 시드 42·43·44, git_dirty=False, 실행 커밋이 기대 커밋인지.
- 실행의 입력 SHA-256(train, test, folds)이 현행 입력과 같은지.
- 설정 산출물이 실행 커밋의 설정 파일과 바이트 단위로 같은지.
- OOF·시험 예측의 식별자 순서, 분할 배정, 행 수, float64, 유한성.
- 대표 OOF가 세 시드 OOF의 평균이고 재채점 AUC가 기록 auc_oof와 같은지(1e-9).
- 자체 36개 기준 구성원의 OOF 해시가 이슈 513 precommit 기록과 같은지.
- 314 확장 기준값이 이슈 513 comparison·precommit·봉인 분할 기록에서 왔는지(파일 해시 동결).

사용법:
    uv run python scripts/freeze_reproduction_pool.py --verify-only
    uv run python scripts/freeze_reproduction_pool.py
    uv run python scripts/freeze_reproduction_pool.py --verify-spec <명세 경로>

명세는 기본으로 `docs/research/reproduction-pool-freeze/<구성원 집합 식별자>.json`에 쓴다.
같은 경로가 이미 있으면 덮어쓰지 않는다. 예측 배열은 읽기만 하며 복사하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.data import ID, TARGET, file_sha256  # noqa: E402
from pipeline.identity import array_identity, integer_identity, pair_identity  # noqa: E402
from pipeline.judgment import CONFIRM_SEEDS  # noqa: E402
from pipeline.ledger import Pool  # noqa: E402
from pipeline.runs import TRACKING_URI  # noqa: E402
from pipeline.sealed import canonical_json, canonical_sha256  # noqa: E402
from pipeline.tracking import EXPERIMENT_NAME, mlflow_client  # noqa: E402

SCHEMA = "reproduction-pool-freeze/1"
SET_ID_PREFIX = "rpf-v1"
GENERATOR_PATH = Path("scripts/freeze_reproduction_pool.py")
DEFAULT_OUT_DIR = Path("docs/research/reproduction-pool-freeze")
TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
FOLDS_PATH = Path("artifacts/folds.parquet")
POOL_PATH = Path("artifacts/pool.yaml")
MAP_ISSUE = {
    "number": 619,
    "title": "지도: 2위 제약 파생 4열 승격을 우리 파이프라인에서 재현해 중첩 결합 이득을 확정한다",
    "url": "https://github.com/tmheo/predicting-smartphone-addiction/issues/619",
}
TICKET_ISSUE = {
    "number": 632,
    "title": "재현 풀 동결 명세와 member source adapter를 구현해 두 기준 회차 스펙을 준비한다",
    "url": "https://github.com/tmheo/predicting-smartphone-addiction/issues/632",
}
TRAINING_ISSUE_URL = "https://github.com/tmheo/predicting-smartphone-addiction/issues/623"

# 이슈 623 실행 커밋. cdv2(862b12d) 위에 실행 스크립트만 더한 가지 issue623-run의 끝이다.
EXPECTED_RUN_COMMIT = "01d6cf316c7157f86c5ebf62656c08a7db1e727f"
FAMILY_ORDER = ("lightgbm", "xgboost", "catboost", "realmlp")
STAGE_ORDER = ("raw4", "cats_te", "ratio_round")
# 원격 실행 태그. Vast.ai 인스턴스 49726833(3회차)의 remote.job_id는 실행 스크립트가 붙인 작업 이름이다.
REMOTE_FAMILIES = {"realmlp": {"provider": "vast", "job_id": "issue623-realmlp-ladder-v1"}}

# (계열, 단계) -> (구성 식별자, main MLflow 반입 run). 이슈 623 명세 문서의 실행 기록 표와 같다.
MEMBER_RUNS: dict[tuple[str, str], tuple[str, str]] = {
    ("lightgbm", "raw4"): ("cdv2_lgb_raw4", "7386da78ae014813b95d552956c8e836"),
    ("lightgbm", "cats_te"): ("cdv2_lgb_cats_te", "800c976f19214f8ba7b3cb577a9f3b6b"),
    ("lightgbm", "ratio_round"): ("cdv2_lgb_ratio_round", "bf0d691326e74898a8996184cff591a4"),
    ("xgboost", "raw4"): ("cdv2_xgb_raw4", "4d16bada5d064ac0b1fd439d018b716e"),
    ("xgboost", "cats_te"): ("cdv2_xgb_cats_te", "02d896484e8341d681406ba76c049907"),
    ("xgboost", "ratio_round"): ("cdv2_xgb_ratio_round", "e80fbeed12fe4b7ea478d342c27d90c7"),
    ("catboost", "raw4"): ("cdv2_cat_raw4", "890860609db543fe8e2312cf40275ff5"),
    ("catboost", "cats_te"): ("cdv2_cat_cats_te", "ef690ee7832446b9a3a2195b20567f60"),
    ("catboost", "ratio_round"): ("cdv2_cat_ratio_round", "3383676700f44465a59d7b27547ebca3"),
    ("realmlp", "raw4"): ("cdv2_realmlp_raw4", "9f3af7ffb2de449ba2cc86d7250ab47c"),
    ("realmlp", "cats_te"): ("cdv2_realmlp_cats_te", "2f261e3949dd4d5c96586b48b125aa43"),
    ("realmlp", "ratio_round"): ("cdv2_realmlp_ratio_round", "c5dec94acb0d4e59b54ef056dda5a20f"),
}
# 계열 -> (기준 구성 식별자, 이슈 623 재실행 run). 기존 확정 설정의 재실행이라 재현 구성원이 아니다.
BASELINE_RERUNS: dict[str, tuple[str, str]] = {
    "lightgbm": ("exp117_ag25_gbm_r21", "fa1b60f193c54c02afb19c3cd8613833"),
    "xgboost": ("exp135_xgb_hpo_trial30", "398c6fcfee1e40a9a12209ea236d2719"),
    "catboost": ("exp070_cat_exact_cats", "cb326d8dcca14808adda3b765ad4a5c0"),
    "realmlp": ("exp139_realmlp_reference_qnormal_train_test", "fe6de111d0f749fe973b0080d8a15834"),
}

# 자체 36개 기준값: 이슈 514 최종 확정의 pool36_full 후보(nested OOF, shrunk_rank_logit_logistic).
OWN36_REFERENCE_RUN_ID = "223055f44dc9427da588a141bc3b1ca3"
OWN36_EVIDENCE_ARTIFACT = "pool36_full-oof-evidence.json"
OWN36_COMBINER = "shrunk_rank_logit_logistic"
OWN36_RECORD_PATH = Path("docs/research/extended-stack-final-assembly/issue514/submission-record.json")
# 314 확장 기준값: 이슈 513 재조립 판정 기록(파일럿 #553이 비트 단위로 재현했다).
EXT314_RECORD_DIR = Path("docs/research/extended-stack-pool-reassembly/issue513")
EXT314_MANIFEST_PATH = Path("docs/research/extended-stack-submission-2-manifest.json")
EXT314_COMBINER = "c_selected_shrunk_rank_logit_logistic"
EXT314_MEMBER_COUNT = 314
OWN_MEMBER_COUNT = 36
OUTER_FOLDS = (0, 1, 2, 3, 4)

SELECTION_POLICY = (
    "재현 구성원은 이슈 623에서 3시드(42, 43, 44)·고정 5분할 confirm 단계로 학습해 main MLflow에 반입한 제약 파생 사다리 설정 12개 전부다.",
    "단일 모형 짝비교 결과(RealMLP만 이득, 나무 3계열 무이득)는 읽거나 제외 근거로 쓰지 않는다(지도 619 확정: 단일 탈락 설정도 결합 구성원으로 넣는다).",
    "구성원 순서는 사다리 단계(raw4, cats_te, ratio_round) 오름차순, 단계 안에서는 계열(LightGBM, XGBoost, CatBoost, RealMLP) 순이며 결과 확인 뒤 바꾸지 않는다.",
    "누적 사다리 3단계(raw4 4개, cats_te 8개, ratio_round 12개)는 구성원 순서의 앞 부분집합이다.",
    "계열별 기준 재실행 4개는 현재 풀 구성원의 재실행이므로 재현 구성원에 넣지 않고 근거 기록으로만 남긴다.",
    "자체 36개 기준 팔은 artifacts/pool.yaml의 36개를 진입 순서 그대로 두고, 기준값은 이슈 514 최종 확정 실행(MLflow 223055f4)의 nested OOF 근거 산출물에서 가져온다.",
    "314 확장 기준 팔은 이슈 513 재조립 판정의 314 구성과 기준값을 그대로 가져온다.",
    "대회 기록(artifacts/pool.yaml, 314 확장 스택, 재학습 장부)은 바꾸지 않는다.",
    "예측 배열은 저장소에 복사하거나 커밋하지 않는다.",
)
JUDGMENT_RULES = (
    "평가 팔은 기준 팔 구성원 뒤에 재현 구성원을 사다리 단계 순서로 이은 누적 3구성이다.",
    "결합기는 기준 팔마다 기준값을 만든 결합기를 그대로 쓴다(자체 36개는 shrunk_rank_logit_logistic, 314 확장은 c_selected_shrunk_rank_logit_logistic).",
    "게이트는 현행 등록 문턱(nested AUC 차이 +0.00002 이상, 바깥 분할 5개 전부 양수)이다.",
    "통과 구성이 여럿이면 nested AUC가 가장 높은 구성을 제안하고, 같으면 구성원이 적은 쪽을 고른다.",
    "두 기준 팔의 판정이 갈리면 해석 규칙을 결과 확인 뒤 사용자와 정한다(지도 619 미지정 항목).",
    "결과를 본 뒤 문턱, 사다리 구성, 구성원 순서를 바꾸지 않는다.",
)


class FreezeError(RuntimeError):
    """재현 전용 풀 동결을 완결할 수 없는 불변식 위반."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeError(message)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def git_bytes(commit: str, path: str) -> bytes | None:
    shown = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=False)
    return shown.stdout if shown.returncode == 0 else None


def generator_state(*, require_clean: bool) -> dict[str, object]:
    commit = git_output("rev-parse", "HEAD").strip()
    dirty = bool(git_output("status", "--porcelain").strip())
    current = GENERATOR_PATH.read_bytes()
    committed = git_bytes(commit, GENERATOR_PATH.as_posix())
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


def file_record(path: Path) -> dict[str, object]:
    _require(path.is_file(), f"필수 입력이 없다: {path}")
    return {"path": str(path), "sha256": file_sha256(path)}


def input_contract() -> tuple[dict[str, object], pd.Series, pd.Series, pd.Series, np.ndarray]:
    for path in (TRAIN_PATH, TEST_PATH, FOLDS_PATH, POOL_PATH):
        _require(path.is_file(), f"필수 입력이 없다: {path}")
    train = pd.read_csv(TRAIN_PATH, usecols=[ID, TARGET])
    train_ids = train[ID]
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID])[ID]
    folds = pd.read_parquet(FOLDS_PATH, columns=[ID, "fold"])
    _require(not train_ids.duplicated().any(), "학습 식별자가 중복된다")
    _require(not test_ids.duplicated().any(), "시험 식별자가 중복된다")
    _require(not folds[ID].duplicated().any(), "분할 식별자가 중복된다")
    _require(len(folds) == len(train_ids), "분할 행 수와 학습 행 수가 다르다")
    # 구성원 행렬(pipeline.members)은 고정 분할의 id 순서로 OOF를 해시하므로, 학습 파일 순서와
    # 같아야 여기서 잰 배열 해시가 판정 회차의 hash-verified 대조와 같은 값이 된다.
    _require(
        np.array_equal(folds[ID].to_numpy(), train_ids.to_numpy()),
        "고정 분할의 id 순서가 학습 파일 순서와 다르다",
    )
    expected_folds = folds["fold"].reset_index(drop=True)
    contract = {
        "train": {
            "path": str(TRAIN_PATH),
            "sha256": file_sha256(TRAIN_PATH),
            "rows": int(len(train_ids)),
            "id_sha256": integer_identity(train_ids),
        },
        "test": {
            "path": str(TEST_PATH),
            "sha256": file_sha256(TEST_PATH),
            "rows": int(len(test_ids)),
            "id_sha256": integer_identity(test_ids),
        },
        "folds": {
            "path": str(FOLDS_PATH),
            "sha256": file_sha256(FOLDS_PATH),
            "rows": int(len(folds)),
            "outer_folds": [int(fold) for fold in sorted(folds["fold"].unique())],
            "assignment_sha256": integer_identity(expected_folds),
        },
    }
    labels = train[TARGET].to_numpy()
    return contract, train_ids, test_ids, expected_folds, labels


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_ids: pd.Series,
    expected_folds: pd.Series | None,
    label: str,
) -> np.ndarray:
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
            f"{label} 분할 배정이 고정 분할과 다르다",
        )
    _require(frame["pred"].dtype == np.dtype("float64"), f"{label} 예측 자료형이 {frame['pred'].dtype}이다")
    values = frame["pred"].to_numpy()
    _require(bool(np.isfinite(values).all()), f"{label} 예측에 유한하지 않은 값이 있다")
    return values


def artifact_names(client: Any, run_id: str) -> set[str]:
    return {item.path for item in client.list_artifacts(run_id)}


def artifact_path(client: Any, run_id: str, name: str) -> Path:
    return Path(client.download_artifacts(run_id, name))


def committed_config_path(commit: str, config_name: str) -> str:
    listed = git_output("ls-tree", "-r", "--name-only", commit, "configs").split()
    matches = [path for path in listed if Path(path).name == config_name]
    _require(len(matches) == 1, f"커밋 {commit[:8]}의 configs/에서 {config_name}이 하나가 아니다: {matches}")
    return matches[0]


def audit_run(
    client: Any,
    run_id: str,
    *,
    expected_config: str,
    family: str,
    input_hashes: dict[str, str],
    train_ids: pd.Series,
    test_ids: pd.Series,
    expected_folds: pd.Series,
    labels: np.ndarray,
) -> dict[str, object]:
    run = client.get_run(run_id)
    tags = dict(run.data.tags)
    params = dict(run.data.params)
    metrics = dict(run.data.metrics)
    config = params.get("experiment")
    label = f"실행 {run_id[:8]}({expected_config})"
    _require(config == expected_config, f"{label}의 구성 식별자가 {config!r}다")
    _require(run.info.status == "FINISHED", f"{label}이 완료 상태가 아니다: {run.info.status}")
    _require(not tags.get("mlflow.parentRunId"), f"{label}이 최상위 실행이 아니다")
    _require(params.get("stage") == "confirm", f"{label}의 단계가 confirm이 아니다")
    _require(params.get("seeds") == ",".join(map(str, CONFIRM_SEEDS)), f"{label}의 시드가 다르다")
    _require(tags.get("git_dirty") == "False", f"{label}의 코드 상태가 깨끗하지 않다")
    _require(tags.get("git_commit") == EXPECTED_RUN_COMMIT, f"{label}의 실행 커밋이 기대 커밋과 다르다")
    for name, expected in input_hashes.items():
        _require(tags.get(f"sha256.{name}") == expected, f"{label}의 {name} 입력 SHA-256이 현행 값과 다르다")
    remote = REMOTE_FAMILIES.get(family)
    provider = tags.get("remote.provider", "local")
    if remote is None:
        _require(provider == "local", f"{label}이 원격 실행으로 태그돼 있다: {provider}")
    else:
        _require(provider == remote["provider"], f"{label}의 원격 공급자가 {provider!r}다")
        _require(tags.get("remote.job_id") == remote["job_id"], f"{label}의 원격 작업 식별자가 다르다")

    names = artifact_names(client, run_id)
    config_names = sorted(name for name in names if name.endswith((".yaml", ".yml")))
    _require(len(config_names) == 1, f"{label}의 설정 YAML이 하나가 아니다: {config_names}")
    required = {"oof.parquet", "test_pred.parquet", *(f"oof_seed_{seed}.parquet" for seed in CONFIRM_SEEDS)}
    _require(required <= names, f"{label}의 예측 산출물이 없다: {sorted(required - names)}")
    config_name = config_names[0]
    config_bytes = artifact_path(client, run_id, config_name).read_bytes()
    parsed = yaml.safe_load(config_bytes)
    _require(isinstance(parsed, dict) and parsed.get("name") == config, f"{label}의 설정 name이 구성 식별자와 다르다")
    committed_path = committed_config_path(EXPECTED_RUN_COMMIT, config_name)
    _require(
        git_bytes(EXPECTED_RUN_COMMIT, committed_path) == config_bytes,
        f"{label}의 설정 산출물이 실행 커밋의 {committed_path}와 다르다",
    )

    oof = validate_prediction_frame(
        pd.read_parquet(artifact_path(client, run_id, "oof.parquet")),
        expected_ids=train_ids,
        expected_folds=expected_folds,
        label=f"{label} OOF",
    )
    test = validate_prediction_frame(
        pd.read_parquet(artifact_path(client, run_id, "test_pred.parquet")),
        expected_ids=test_ids,
        expected_folds=None,
        label=f"{label} 시험 예측",
    )
    seed_aucs: dict[str, float] = {}
    seed_predictions = []
    for seed in CONFIRM_SEEDS:
        prediction = validate_prediction_frame(
            pd.read_parquet(artifact_path(client, run_id, f"oof_seed_{seed}.parquet")),
            expected_ids=train_ids,
            expected_folds=expected_folds,
            label=f"{label} 시드 {seed} OOF",
        )
        seed_predictions.append(prediction)
        seed_aucs[str(seed)] = float(roc_auc_score(labels, prediction))
        stored = metrics.get(f"auc_oof_seed_{seed}")
        _require(
            stored is not None and math.isclose(seed_aucs[str(seed)], stored, rel_tol=0.0, abs_tol=1e-9),
            f"{label} 시드 {seed} OOF 재채점이 기록과 다르다",
        )
    _require(
        bool(np.allclose(np.mean(seed_predictions, axis=0), oof, rtol=0.0, atol=1e-12)),
        f"{label}의 대표 OOF가 세 시드 평균과 다르다",
    )
    auc = float(roc_auc_score(labels, oof))
    _require(
        math.isclose(auc, float(metrics["auc_oof"]), rel_tol=0.0, abs_tol=1e-9),
        f"{label} OOF 재채점이 기록 auc_oof와 다르다",
    )
    fold_values = expected_folds.to_numpy()
    fold_aucs = {
        str(fold): float(roc_auc_score(labels[fold_values == fold], oof[fold_values == fold]))
        for fold in sorted(np.unique(fold_values))
    }
    return {
        "config": config,
        "family": family,
        "run_id": run_id,
        "source_run_id": tags.get("source.run_id"),
        "import_bundle_sha256": tags.get("import.bundle_sha256"),
        "provider": provider,
        "remote_job_id": tags.get("remote.job_id"),
        "started_at": datetime.fromtimestamp(run.info.start_time / 1000, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "seeds": list(CONFIRM_SEEDS),
        "git_commit": EXPECTED_RUN_COMMIT,
        "git_dirty": False,
        "config_artifact": {
            "name": config_name,
            "sha256": bytes_sha256(config_bytes),
            "committed_path": committed_path,
            "committed_content_matches": True,
        },
        "input_sha256": dict(input_hashes),
        "oof": {
            "rows": int(len(oof)),
            "array_sha256": array_identity(oof),
            "auc": auc,
            "seed_aucs": seed_aucs,
            "fold_aucs": fold_aucs,
        },
        "test": {"rows": int(len(test)), "array_sha256": array_identity(test)},
        "prediction_pair_sha256": pair_identity(oof, test),
    }


MISSINGNESS_AUGMENTED_KEY = "missingness_augmented:"


def pool_config_of(candidate_key: str) -> str:
    """이슈 514 근거 산출물의 후보 키를 풀 등록 이름으로 옮긴다."""
    if candidate_key.startswith(MISSINGNESS_AUGMENTED_KEY):
        return f"mpv1_{candidate_key[len(MISSINGNESS_AUGMENTED_KEY):]}_missingness_augmented"
    return candidate_key


def own36_reference(
    client: Any,
    pool: Pool,
    *,
    train_ids: pd.Series,
    labels: np.ndarray,
) -> dict[str, object]:
    """자체 36개 기준 팔: 풀 순서의 구성원 해시와 이슈 514 pool36_full의 nested 기준값."""
    precommit_path = EXT314_RECORD_DIR / "precommit.json"
    recorded = json.loads(precommit_path.read_text(encoding="utf-8"))["reassembled"]["members"]
    recorded_own = [row for row in recorded if row["origin"] == "own"]
    _require(len(recorded_own) == OWN_MEMBER_COUNT, "이슈 513 precommit의 자체 구성원이 36개가 아니다")
    _require(len(pool.members) == OWN_MEMBER_COUNT, f"현재 풀이 36개가 아니다: {len(pool.members)}")

    members = []
    for order, (member, row) in enumerate(zip(pool.members, recorded_own, strict=True), start=1):
        _require(
            (member.config, member.run_id) == (row["column"], row["run_id"]),
            f"풀 {order}번째 {member.config}가 이슈 513 precommit의 자체 구성원 순서와 다르다",
        )
        frame = pd.read_parquet(artifact_path(client, member.run_id, "oof.parquet"))
        oof = validate_prediction_frame(
            frame, expected_ids=train_ids, expected_folds=None, label=f"풀 {member.config} OOF"
        )
        digest = array_identity(oof)
        _require(digest == row["oof_sha256"], f"풀 {member.config}의 OOF 해시가 이슈 513 precommit과 다르다")
        auc = float(roc_auc_score(labels, oof))
        _require(
            math.isclose(auc, member.oof_auc, rel_tol=0.0, abs_tol=1e-9),
            f"풀 {member.config}의 OOF 재채점이 장부 AUC와 다르다",
        )
        members.append(
            {"config": member.config, "run_id": member.run_id, "oof_sha256": digest, "auc": auc, "order": order}
        )

    run = client.get_run(OWN36_REFERENCE_RUN_ID)
    params = dict(run.data.params)
    metrics = dict(run.data.metrics)
    _require(run.info.status == "FINISHED", "자체 36개 기준값 실행이 완료 상태가 아니다")
    _require(params.get("ensemble.strategy") == OWN36_COMBINER, "자체 36개 기준값 실행의 결합기가 다르다")
    _require(params.get("ensemble.member_count") == str(OWN_MEMBER_COUNT), "자체 36개 기준값 실행의 구성원 수가 다르다")
    evidence_path = artifact_path(client, OWN36_REFERENCE_RUN_ID, OWN36_EVIDENCE_ARTIFACT)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    proposal = evidence["proposal"]
    _require(proposal["best_strategy"] == OWN36_COMBINER, "근거 산출물의 최선 결합기가 다르다")
    # 근거 산출물은 등록 전 후보 키(missingness_augmented:<원본>)로 5개를 적었고, 풀은 등록 이름
    # (mpv1_<원본>_missingness_augmented)으로 적는다. 순서와 구성원은 같다.
    _require(
        [pool_config_of(key) for key in proposal["members"]] == [member.config for member in pool.members],
        "근거 산출물의 구성원 순서가 현재 풀 순서와 다르다",
    )
    nested_auc = float(proposal["best_auc"])
    _require(nested_auc == float(metrics["auc_oof"]), "근거 산출물의 nested AUC가 실행 metric과 다르다")
    fold_aucs = {str(fold): float(proposal["best_fold_auc"][str(fold)]) for fold in OUTER_FOLDS}
    for fold, auc in fold_aucs.items():
        _require(auc == float(metrics[f"auc_fold_{fold}"]), f"근거 산출물의 분할 {fold} AUC가 실행 metric과 다르다")
    record = json.loads(OWN36_RECORD_PATH.read_text(encoding="utf-8"))
    candidate = record["candidates"]["pool36_full"]
    _require(
        candidate["mlflow_run_id"] == OWN36_REFERENCE_RUN_ID and candidate["nested_oof_auc"] == nested_auc,
        "이슈 514 제출 기록의 pool36_full 값이 기준값 실행과 다르다",
    )
    return {
        "name": "pool36-current",
        "combiner": OWN36_COMBINER,
        "member_count": OWN_MEMBER_COUNT,
        "members": members,
        "composition_sha256": canonical_sha256([[row["config"], row["oof_sha256"]] for row in members]),
        "nested_auc": nested_auc,
        "fold_aucs": fold_aucs,
        "values_source": {
            "mlflow_run_id": OWN36_REFERENCE_RUN_ID,
            "artifact": OWN36_EVIDENCE_ARTIFACT,
            "artifact_sha256": file_sha256(evidence_path),
            "git_commit": run.data.tags.get("git_commit"),
            "record": file_record(OWN36_RECORD_PATH),
            "note": (
                "nested 예측 해시는 기록에 없어 자기 검사는 분할별 AUC 동일성만 대조한다. "
                "근거 산출물의 후보 키 missingness_augmented:<원본> 5개는 풀 등록 이름 mpv1_<원본>_missingness_augmented와 같은 구성원이다."
            ),
        },
        "corroboration": {
            "issue513_precommit": file_record(precommit_path),
            "own_member_hashes_match": True,
        },
    }


def ext314_reference() -> dict[str, object]:
    """314 확장 기준 팔: 이슈 513 재조립 판정의 구성 해시, 기준값, 봉인 분할 기록."""
    precommit_path = EXT314_RECORD_DIR / "precommit.json"
    comparison_path = EXT314_RECORD_DIR / "comparison.json"
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))["reassembled"]
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))["reassembled"]
    _require(precommit["member_count"] == EXT314_MEMBER_COUNT, "이슈 513 precommit의 구성원 수가 314가 아니다")
    _require(comparison["member_count"] == EXT314_MEMBER_COUNT, "이슈 513 comparison의 구성원 수가 314가 아니다")
    _require(comparison["strategy"] == EXT314_COMBINER, "이슈 513 comparison의 결합기가 다르다")
    manifest = json.loads(EXT314_MANIFEST_PATH.read_text(encoding="utf-8"))
    oof_paths = {row["column"]: row.get("oof_path") for row in manifest["members"]}
    for row in precommit["members"]:
        if row["origin"] != "own":
            _require(bool(oof_paths.get(row["column"])), f"외부 구성원 {row['column']}의 OOF 경로가 manifest에 없다")
    sealed_folds = {}
    for fold in OUTER_FOLDS:
        path = EXT314_RECORD_DIR / "reassembled" / f"fold-{fold}" / "reassembled.json"
        body = json.loads(path.read_text(encoding="utf-8"))
        _require(body["sealed_fold"] == fold and body["member_count"] == EXT314_MEMBER_COUNT, f"봉인 분할 {fold} 기록이 어긋난다")
        _require(body["auc"] == comparison["fold_aucs"][str(fold)], f"봉인 분할 {fold}의 AUC가 comparison과 다르다")
        sealed_folds[str(fold)] = {
            **file_record(path),
            "auc": float(body["auc"]),
            "prediction_sha256": body["prediction_sha256"],
        }
    return {
        "name": "reassembled-314",
        "combiner": EXT314_COMBINER,
        "member_count": EXT314_MEMBER_COUNT,
        "own_member_count": int(precommit["own_member_count"]),
        "external_member_count": int(precommit["external_member_count"]),
        "composition_sha256": precommit["composition_sha256"],
        "nested_auc": float(comparison["nested_auc"]),
        "fold_aucs": {str(fold): float(comparison["fold_aucs"][str(fold)]) for fold in OUTER_FOLDS},
        "prediction_sha256": comparison["prediction_sha256"],
        "values_source": {
            "comparison": file_record(comparison_path),
            "precommit": file_record(precommit_path),
            "manifest": file_record(EXT314_MANIFEST_PATH),
            "pilot_replay": "docs/research/judgment-round-pilot/issue553 (#553, ADR-0009 실전 검증)",
        },
        "sealed_folds": sealed_folds,
    }


def build_spec(*, tracking_uri: str, require_clean_generator: bool) -> dict[str, object]:
    generator = generator_state(require_clean=require_clean_generator)
    inputs, train_ids, test_ids, expected_folds, labels = input_contract()
    input_hashes = {name: str(inputs[name]["sha256"]) for name in ("train", "test", "folds")}
    _require(list(inputs["folds"]["outer_folds"]) == list(OUTER_FOLDS), "고정 분할이 0~4가 아니다")
    client, experiment_id = mlflow_client(tracking_uri)
    pool = Pool.load(POOL_PATH)
    pool_run_ids = {member.run_id for member in pool.members}

    members = []
    for stage in STAGE_ORDER:
        for family in FAMILY_ORDER:
            config, run_id = MEMBER_RUNS[(family, stage)]
            _require(run_id not in pool_run_ids, f"재현 구성원 {config}의 실행이 현재 풀에 있다")
            record = audit_run(
                client,
                run_id,
                expected_config=config,
                family=family,
                input_hashes=input_hashes,
                train_ids=train_ids,
                test_ids=test_ids,
                expected_folds=expected_folds,
                labels=labels,
            )
            record["stage"] = stage
            record["order"] = len(members) + 1
            members.append(record)
            print(f"[감사] {config} {run_id[:8]} auc={record['oof']['auc']:.7f}", flush=True)
    pairs = {(row["oof"]["array_sha256"], row["test"]["array_sha256"]) for row in members}
    _require(len(pairs) == len(members), "재현 구성원 사이에 예측 쌍이 같은 정확 중복이 있다")

    pool_by_config = {member.config: member for member in pool.members}
    baseline_reruns = []
    for family in FAMILY_ORDER:
        config, run_id = BASELINE_RERUNS[family]
        record = audit_run(
            client,
            run_id,
            expected_config=config,
            family=family,
            input_hashes=input_hashes,
            train_ids=train_ids,
            test_ids=test_ids,
            expected_folds=expected_folds,
            labels=labels,
        )
        # 기준 4개는 계열별 원본 행 최고 설정이라 현재 풀에는 결측 증강판(mpv1_*) 등으로만 남은 것도 있다.
        pool_member = pool_by_config.get(config)
        record["current_pool_member"] = (
            None if pool_member is None else {"run_id": pool_member.run_id, "oof_auc": pool_member.oof_auc}
        )
        baseline_reruns.append(record)
        print(f"[감사] 기준 재실행 {config} {run_id[:8]} auc={record['oof']['auc']:.7f}", flush=True)

    ladder = []
    for stage in STAGE_ORDER:
        rung = [row["config"] for row in members if STAGE_ORDER.index(row["stage"]) <= STAGE_ORDER.index(stage)]
        ladder.append({"stage": stage, "member_count": len(rung), "members": rung})
    _require([r["member_count"] for r in ladder] == [4, 8, 12], "누적 사다리 크기가 4, 8, 12가 아니다")

    own36 = own36_reference(client, pool, train_ids=train_ids, labels=labels)
    ext314 = ext314_reference()

    spec: dict[str, object] = {
        "schema": SCHEMA,
        "contract": {"map": MAP_ISSUE, "ticket": TICKET_ISSUE, "training_issue": TRAINING_ISSUE_URL},
        "selection_policy": list(SELECTION_POLICY),
        "judgment_rules": list(JUDGMENT_RULES),
        "generator": generator,
        "inputs": {
            **inputs,
            "pool": {
                "path": str(POOL_PATH),
                "sha256": file_sha256(POOL_PATH),
                "member_count": len(pool.members),
            },
            "run_store": {
                "tracking_uri": tracking_uri,
                "experiment_name": EXPERIMENT_NAME,
                "experiment_id": experiment_id,
                "database_file_sha256_omitted": (
                    "실행 저장소는 계속 추가되므로 파일 전체 해시 대신 실행별 변경 불가 기록을 고정한다."
                ),
            },
        },
        "run_commit": EXPECTED_RUN_COMMIT,
        "member_count": len(members),
        "members": members,
        "ladder": ladder,
        "baseline_reruns": baseline_reruns,
        "reference_arms": {"own36": own36, "ext314": ext314},
    }
    spec["content_sha256"] = text_sha256(canonical_json(spec))
    spec["member_set_id"] = f"{SET_ID_PREFIX}-{str(spec['content_sha256'])[:12]}"
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
        if key not in {"spec_sha256", "member_set_id", "frozen_at", "content_sha256"}
    }
    expected_content = text_sha256(canonical_json(content))
    _require(spec.get("content_sha256") == expected_content, f"{path}: content_sha256이 다르다")
    _require(
        spec.get("member_set_id") == f"{SET_ID_PREFIX}-{expected_content[:12]}",
        f"{path}: 구성원 집합 식별자가 내용 해시와 다르다",
    )
    _require(spec.get("member_count") == len(MEMBER_RUNS) == len(spec.get("members", [])), f"{path}: 구성원 수가 12가 아니다")
    _require([rung["member_count"] for rung in spec.get("ladder", [])] == [4, 8, 12], f"{path}: 사다리 크기가 4, 8, 12가 아니다")
    _require(len(spec.get("baseline_reruns", [])) == len(BASELINE_RERUNS), f"{path}: 기준 재실행이 4개가 아니다")
    arms = spec.get("reference_arms", {})
    _require(arms.get("own36", {}).get("member_count") == OWN_MEMBER_COUNT, f"{path}: 자체 기준 팔이 36개가 아니다")
    _require(arms.get("ext314", {}).get("member_count") == EXT314_MEMBER_COUNT, f"{path}: 확장 기준 팔이 314개가 아니다")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="재현 전용 풀 동결 명세 생성기 (#632)")
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--out", type=Path, help="명세 파일 경로")
    parser.add_argument("--verify-only", action="store_true", help="감사만 하고 명세를 쓰지 않는다")
    parser.add_argument("--verify-spec", type=Path, help="기존 명세의 자체 해시와 개수를 검사한다")
    args = parser.parse_args()

    try:
        if args.verify_spec is not None:
            spec = verify_spec_file(args.verify_spec)
            print(
                f"명세 검사 통과: {args.verify_spec}, 구성원 {spec['member_count']}개, "
                f"spec_sha256 {spec['spec_sha256']}"
            )
            return
        spec = build_spec(
            tracking_uri=args.tracking_uri,
            require_clean_generator=not args.verify_only,
        )
        print(
            f"감사 통과: 재현 구성원 {spec['member_count']}개, 기준 재실행 "
            f"{len(spec['baseline_reruns'])}개, 구성원 집합 {spec['member_set_id']}"
        )
        if args.verify_only:
            return
        out = args.out or DEFAULT_OUT_DIR / f"{spec['member_set_id']}.json"
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
