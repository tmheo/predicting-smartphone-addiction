"""동결한 재사용 적격 자체 후보 확장 사다리를 중첩 OOF로 판정한다. (#495)

현재 313개 확장 구성을 기준으로 다음 세 구성을 고정 결합기
``shrunk_rank_logit_logistic``으로 다시 판정한다.

1. ``baseline313``: 현재 자체 35개와 외부 278개.
2. ``full402``: 기준 313개 뒤에 동결 자체 후보 89개를 모두 추가.
3. ``restrained361``: 기준 313개 뒤에 고정 학습 길이 트리 변형 41개를 뺀 후보 48개를 추가.

실행 순서:

    uv run python scripts/judge_reusable_own_extension.py precommit --workers 2 --threads 4
    uv run python scripts/judge_reusable_own_extension.py run
    uv run python scripts/judge_reusable_own_extension.py compare
    uv run python scripts/judge_reusable_own_extension.py report
    uv run python scripts/judge_reusable_own_extension.py verify
    uv run python scripts/judge_reusable_own_extension.py publish

``precommit``은 입력, 후보와 열 순서, 결합 절차, 문턱, 동률 규칙, 코드 상태와
실행 자원을 결과 확인 전에 고정한다. ``run``은 기준 313개를 먼저 실행해 이슈 489의
전체 및 분할별 AUC를 재현한 경우에만 두 후보 구성을 실행한다. 완료한 일부 결과는
판정에 쓰지 않으며 같은 precommit과 코드 상태에서만 재개한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import freeze_reusable_own_candidates as freeze

from pipeline import ensemble
from pipeline.data import ID, TARGET, file_sha256
from pipeline.judgment import DUPLICATE_SPEARMAN
from pipeline.pool_audit import prediction_array_sha256

ISSUE = 495
SCHEMA = "reusable-own-extension-judgment/1"
FREEZE_SPEC_PATH = Path(
    "docs/research/reusable-own-candidate-freeze/rocf-v1-b42e02ea2e2b.json"
)
ISSUE489_DIR = Path("docs/research/logistic-c-selection/issue489")
ISSUE489_PRECOMMIT_PATH = ISSUE489_DIR / "precommit.json"
ISSUE489_COMPARISON_PATH = ISSUE489_DIR / "comparison.json"
COMPARISON_MANIFEST_PATH = Path(
    "docs/research/extended-stack-submission-2-manifest.json"
)
ENSEMBLE_SOURCE = Path(ensemble.__file__)
UV_LOCK_PATH = Path("uv.lock")
OUT_DIR = Path("run-logs/reusable-own-extension/issue495")
PUBLISH_DIR = Path("docs/research/reusable-own-extension/issue495")

STRATEGY = "shrunk_rank_logit_logistic"
CONFIG_BASELINE = "baseline313"
CONFIG_FULL = "full402"
CONFIG_RESTRAINED = "restrained361"
CONFIG_ORDER = (CONFIG_BASELINE, CONFIG_FULL, CONFIG_RESTRAINED)
ALL_FOLDS = (0, 1, 2, 3, 4)
N_TRAIN = 691_369
N_TEST = 296_302
BASE_MEMBER_COUNT = 313
FULL_CANDIDATE_COUNT = 89
RESTRAINED_CANDIDATE_COUNT = 48
FULL_MEMBER_COUNT = 402
RESTRAINED_MEMBER_COUNT = 361

GATE_DELTA = 0.00002
FOLDS_REQUIRED_POSITIVE = 5
NOISE_FLOOR = 5.7e-06
REPRODUCTION_TOLERANCE = 1e-10
MAX_WORKERS = 3
DEFAULT_WORKERS = 2
DEFAULT_THREADS = 4
MEMORY_HEADROOM_MIN = 0.15
META_MAX_ITER = 1000
LOGIT_EPS = ensemble.LogisticLinearCombiner.LOGIT_EPS
LAMBDA_GRID = ensemble.SHRINKAGE_LAMBDA_GRID

CONTROL_REFERENCE = {
    "nested_auc": 0.970350946943525,
    "fold_aucs": {
        "0": 0.969754284537701,
        "1": 0.9705148747291037,
        "2": 0.9703990946138134,
        "3": 0.9709869070909464,
        "4": 0.9700995711834858,
    },
}


class JudgmentError(RuntimeError):
    """입력, 코드, 실행 또는 산출물 불변식이 깨져 판정할 수 없는 상태."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JudgmentError(message)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_sha256(data: object) -> str:
    return freeze.text_sha256(freeze.canonical_json(data))


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def default_source_root() -> Path:
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).resolve().parent


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value) if sys.platform == "darwin" else int(value) * 1024


def environment(*, threads: int | None = None) -> dict[str, object]:
    memory = psutil.virtual_memory()
    blas = getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {}).get("blas")
    thread_keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": int(memory.total),
        "memory_available_bytes": int(memory.available),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "blas": blas,
        "threads_requested": threads,
        "thread_environment": {key: os.environ.get(key) for key in thread_keys},
    }


def code_state() -> dict[str, object]:
    return {
        "git": git_state(),
        "script": {
            "path": str(Path(__file__).resolve().relative_to(Path.cwd())),
            "sha256": file_sha256(Path(__file__)),
        },
        "ensemble_module": {
            "path": str(ENSEMBLE_SOURCE),
            "sha256": file_sha256(ENSEMBLE_SOURCE),
        },
        "freeze_script": {
            "path": str(Path(freeze.__file__).resolve().relative_to(Path.cwd())),
            "sha256": file_sha256(Path(freeze.__file__)),
        },
        "uv_lock_sha256": file_sha256(UV_LOCK_PATH),
    }


def process_snapshot(repo_root: Path) -> list[dict[str, object]]:
    output = subprocess.run(
        ["ps", "-axo", "pid=,pcpu=,rss=,etime=,command="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) != 5:
            continue
        pid, cpu, rss, elapsed, command = parts
        if int(pid) == os.getpid():
            continue
        if repo_root.name not in command or "python" not in command.lower():
            continue
        rows.append(
            {
                "pid": int(pid),
                "cpu_percent": float(cpu),
                "rss_bytes": int(rss) * 1024,
                "elapsed": elapsed,
                "command": command,
            }
        )
    return rows


def source_paths(source_root: Path) -> dict[str, Path]:
    return {
        "train": source_root / "data/train.csv",
        "test": source_root / "data/test.csv",
        "folds": source_root / "artifacts/folds.parquet",
        "pool": source_root / "artifacts/pool.yaml",
        "base_cache": source_root
        / "run-logs/logistic-c-selection/issue489/cache/oof-313.parquet",
    }


def load_folds_and_labels(paths: dict[str, Path]) -> tuple[pd.Series, pd.Series, pd.Series]:
    train = pd.read_csv(paths["train"], usecols=[ID, TARGET])
    folds = pd.read_parquet(paths["folds"], columns=[ID, "fold"])
    _require(len(train) == N_TRAIN and len(folds) == N_TRAIN, "학습 또는 분할 행 수가 기대와 다르다.")
    _require(not train[ID].duplicated().any() and not folds[ID].duplicated().any(), "학습 또는 분할 식별자가 중복된다.")
    _require(np.array_equal(train[ID].to_numpy(), folds[ID].to_numpy()), "학습과 분할 식별자 순서가 다르다.")
    fold_of = folds.set_index(ID)["fold"].astype(np.int8)
    y = train.set_index(ID)[TARGET].astype(np.int8)
    _require(sorted(fold_of.unique().tolist()) == list(ALL_FOLDS), "분할이 0부터 4까지 다섯 개가 아니다.")
    return fold_of, y, train[ID]


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_ids: pd.Series,
    expected_folds: pd.Series | None,
    label: str,
) -> np.ndarray:
    required = [ID, "pred"] + (["fold"] if expected_folds is not None else [])
    _require(all(column in frame for column in required), f"{label}: 필수 열이 없다.")
    _require(len(frame) == len(expected_ids), f"{label}: 행 수가 기대와 다르다.")
    _require(not frame[ID].duplicated().any(), f"{label}: 식별자가 중복된다.")
    _require(np.array_equal(frame[ID].to_numpy(), expected_ids.to_numpy()), f"{label}: 식별자 순서가 다르다.")
    if expected_folds is not None:
        _require(np.array_equal(frame["fold"].to_numpy(), expected_folds.to_numpy()), f"{label}: 분할이 다르다.")
    _require(frame["pred"].dtype == np.dtype("float64"), f"{label}: 예측 자료형이 {frame['pred'].dtype}이다.")
    values = frame["pred"].to_numpy(np.float64)
    _require(bool(np.isfinite(values).all()), f"{label}: 유한하지 않은 예측이 있다.")
    return values


def candidate_artifact_dir(source_root: Path, spec: dict, run_id: str) -> Path:
    experiment_id = str(spec["inputs"]["run_store"]["experiment_id"])
    return source_root / "mlruns" / experiment_id / run_id / "artifacts"


def load_candidate_cache(
    source_root: Path,
    spec: dict,
    fold_of: pd.Series,
    train_ids: pd.Series,
    test_ids: pd.Series,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    columns: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    for expected_order, candidate in enumerate(spec["candidates"], start=1):
        _require(candidate["order"] == expected_order, "동결 후보 순서가 연속적이지 않다.")
        column = str(candidate["config"])
        _require(column not in columns, f"동결 후보 열이 중복된다: {column}")
        artifact_dir = candidate_artifact_dir(source_root, spec, candidate["run_id"])
        config_path = artifact_dir / candidate["config_artifact"]["name"]
        oof_path = artifact_dir / "oof.parquet"
        test_path = artifact_dir / "test_pred.parquet"
        for path in (config_path, oof_path, test_path):
            _require(path.is_file(), f"동결 후보 산출물이 없다: {path}")
        _require(file_sha256(config_path) == candidate["config_artifact"]["sha256"], f"{column}: 설정 산출물 해시가 동결 명세와 다르다.")
        oof_frame = pd.read_parquet(oof_path)
        test_frame = pd.read_parquet(test_path)
        oof = validate_prediction_frame(
            oof_frame,
            expected_ids=train_ids,
            expected_folds=fold_of.reset_index(drop=True),
            label=f"{column} OOF",
        )
        test = validate_prediction_frame(
            test_frame,
            expected_ids=test_ids,
            expected_folds=None,
            label=f"{column} 시험 예측",
        )
        _require(freeze.array_sha256(oof) == candidate["oof"]["array_sha256"], f"{column}: OOF 배열 해시가 동결 명세와 다르다.")
        _require(freeze.array_sha256(test) == candidate["test"]["array_sha256"], f"{column}: 시험 예측 배열 해시가 동결 명세와 다르다.")
        _require(freeze.pair_sha256(oof, test) == candidate["prediction_pair_sha256"], f"{column}: 예측 쌍 해시가 동결 명세와 다르다.")
        columns[column] = oof
        records.append(
            {
                "order": expected_order,
                "column": column,
                "run_id": candidate["run_id"],
                "started_at": candidate["started_at"],
                "config_sha256": candidate["config_artifact"]["sha256"],
                "oof_array_sha256": candidate["oof"]["array_sha256"],
                "test_array_sha256": candidate["test"]["array_sha256"],
                "prediction_pair_sha256": candidate["prediction_pair_sha256"],
                "artifacts": {
                    "config": {"path": str(config_path), "file_sha256": file_sha256(config_path)},
                    "oof": {"path": str(oof_path), "file_sha256": file_sha256(oof_path)},
                    "test": {"path": str(test_path), "file_sha256": file_sha256(test_path)},
                },
            }
        )
    matrix = pd.DataFrame(columns, index=fold_of.index).astype(np.float64)
    _require(matrix.shape == (N_TRAIN, FULL_CANDIDATE_COUNT), "동결 후보 OOF 행렬 형태가 기대와 다르다.")
    return matrix, records


def configuration_spec(
    base_columns: list[str], candidate_records: list[dict], fixed_columns: set[str]
) -> list[dict[str, object]]:
    candidate_columns = [row["column"] for row in candidate_records]
    restrained = [column for column in candidate_columns if column not in fixed_columns]
    _require(len(candidate_columns) == FULL_CANDIDATE_COUNT, "전체 후보 수가 89개가 아니다.")
    _require(len(restrained) == RESTRAINED_CANDIDATE_COUNT, "절제 후보 수가 48개가 아니다.")
    return [
        {
            "index": 0,
            "name": CONFIG_BASELINE,
            "description": "현재 자체 35개와 외부 278개, 합계 313개 기준 구성",
            "candidate_columns": [],
            "removed_candidate_columns": candidate_columns,
            "columns": base_columns,
            "member_count": BASE_MEMBER_COUNT,
        },
        {
            "index": 1,
            "name": CONFIG_FULL,
            "description": "기준 313개 뒤에 동결 자체 후보 89개를 모두 추가한 구성",
            "candidate_columns": candidate_columns,
            "removed_candidate_columns": [],
            "columns": base_columns + candidate_columns,
            "member_count": FULL_MEMBER_COUNT,
        },
        {
            "index": 2,
            "name": CONFIG_RESTRAINED,
            "description": "기준 313개 뒤에 고정 학습 길이 트리 변형 41개를 뺀 동결 자체 후보 48개를 추가한 구성",
            "candidate_columns": restrained,
            "removed_candidate_columns": [column for column in candidate_columns if column in fixed_columns],
            "columns": base_columns + restrained,
            "member_count": RESTRAINED_MEMBER_COUNT,
        },
    ]


def rules() -> dict[str, str]:
    return {
        "control": "현재 313개 구성원 신원과 순서는 extended-stack-submission-2-manifest.json을 그대로 쓴다.",
        "configurations": "사다리는 baseline313, full402, restrained361 세 구성으로 고정하며 결과를 본 뒤 구성원을 더하거나 빼지 않는다.",
        "combiner": "세 구성 모두 구성원별 순위와 logit 이중 표현, logit 절단 1e-6, StandardScaler, L2 LogisticRegression C=1.0, lbfgs, max_iter=1000, random_state=0을 사용한다.",
        "shrinkage": "각 바깥쪽 학습 부분 안에서 한 분할씩 빼 만든 예측을 이어 붙여 λ 후보 (0.25, 0.5, 0.75, 1.0) 가운데 AUC가 가장 큰 값을 고르며 동률이면 더 작은 λ를 고른다.",
        "nested": "바깥쪽 분할 하나를 봉인하고 열린 네 분할로 결합기를 맞춰 봉인 분할을 한 번 예측하며, 다섯 예측을 원래 행 순서로 이어 붙여 중첩 OOF AUC를 계산한다.",
        "reproduction": f"baseline313 전체 및 분할별 AUC가 이슈 489의 C=1.0 기준값과 절대 오차 {REPRODUCTION_TOLERANCE} 안에서 맞아야 후보 판정을 시작한다.",
        "gate": f"full402와 restrained361은 각각 baseline313보다 이어붙인 중첩 OOF AUC가 +{GATE_DELTA} 이상 높고 바깥쪽 검증 분할 {FOLDS_REQUIRED_POSITIVE}/5가 모두 엄격히 높을 때만 통과한다.",
        "pick": f"둘 다 통과하면 중첩 OOF가 높은 구성을 고르고, 두 값의 차이가 잡음 바닥 {NOISE_FLOOR} 이하면 구성원이 적은 restrained361을 고른다.",
        "none": "통과 구성이 없으면 baseline313 유지가 완결된 결론이다.",
        "diagnostics": f"후보별 단독 OOF AUC와 스피어만 순위 상관 {DUPLICATE_SPEARMAN} 이상 근접 중복은 진단값으로만 기록하며 구성 변경과 선택에 사용하지 않는다.",
        "failure": "계산 하나라도 실패하거나 끝나지 않으면 완료한 일부 결과를 쓰지 않고 전체를 판정 불가로 둔다.",
        "resume": "중단 뒤 재개는 입력, 캐시, 후보와 열 순서, 코드 커밋 및 도구 해시가 같은 precommit과 정확히 맞을 때만 허용한다.",
        "public": "Public 점수와 Kaggle 제출은 어느 단계에도 사용하지 않는다.",
    }


def precommit(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    _require(not (run_dir / "precommit.json").exists(), f"precommit.json이 이미 있다: {run_dir}")
    _require(1 <= args.workers <= MAX_WORKERS, f"동시 작업 수는 1부터 {MAX_WORKERS}까지다.")
    _require(args.threads == DEFAULT_THREADS, f"작업당 병렬 연산 수는 {DEFAULT_THREADS}개로 고정한다.")
    state = git_state()
    _require(not state["dirty"], "실제 판정은 커밋된 깨끗한 코드 상태에서만 시작한다.")
    source_root = Path(args.source_root).resolve()
    paths = source_paths(source_root)
    for name, path in paths.items():
        _require(path.is_file(), f"원본 입력이 없다({name}): {path}")

    spec = freeze.verify_spec_file(FREEZE_SPEC_PATH)
    _require(spec["candidate_set_id"] == "rocf-v1-b42e02ea2e2b", "동결 후보 집합 식별자가 다르다.")
    for key in ("train", "test", "folds", "pool"):
        actual = file_sha256(paths[key])
        expected = spec["inputs"][key]["sha256"]
        _require(actual == expected, f"{key} 해시가 동결 명세와 다르다: {actual}")

    fold_of, _, train_ids = load_folds_and_labels(paths)
    test_ids = pd.read_csv(paths["test"], usecols=[ID])[ID]
    _require(len(test_ids) == N_TEST and not test_ids.duplicated().any(), "시험 식별자 행 수 또는 중복 검사 실패")

    prior_precommit = read_json(ISSUE489_PRECOMMIT_PATH)
    prior_comparison = read_json(ISSUE489_COMPARISON_PATH)
    _require(prior_comparison["control"]["nested_auc"] == CONTROL_REFERENCE["nested_auc"], "이슈 489 대조군 전체 AUC가 고정 기준과 다르다.")
    _require(prior_comparison["control"]["fold_aucs"] == CONTROL_REFERENCE["fold_aucs"], "이슈 489 대조군 분할별 AUC가 고정 기준과 다르다.")
    base_cache_sha = file_sha256(paths["base_cache"])
    _require(base_cache_sha == prior_precommit["cache"]["oof-313.parquet"], "313 OOF 캐시가 이슈 489 precommit과 다르다.")
    base = pd.read_parquet(paths["base_cache"]).astype(np.float64)
    manifest = read_json(COMPARISON_MANIFEST_PATH)
    base_columns = [row["column"] for row in manifest["members"]]
    _require(len(base_columns) == BASE_MEMBER_COUNT and list(base.columns) == base_columns, "313 구성원 신원 또는 순서가 manifest와 다르다.")
    _require(base.index.equals(fold_of.index), "313 OOF 행 순서가 분할과 다르다.")
    _require(bool(np.isfinite(base.to_numpy()).all()), "313 OOF에 유한하지 않은 값이 있다.")
    prior_member_rows = prior_precommit["members"]["rows"]
    _require([row["column"] for row in prior_member_rows] == base_columns, "이슈 489 구성원 기록과 313 manifest 순서가 다르다.")

    candidates, candidate_records = load_candidate_cache(
        source_root, spec, fold_of, train_ids, test_ids
    )
    _require(not set(base_columns) & set(candidates.columns), "313 구성과 동결 후보의 열 이름이 겹친다.")
    fixed_columns = {
        row["config"] for row in spec["fixed_training_length_tree_variants"]["members"]
    }
    configs = configuration_spec(base_columns, candidate_records, fixed_columns)
    _require([config["name"] for config in configs] == list(CONFIG_ORDER), "구성 순서가 고정 사다리와 다르다.")

    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidate_cache_path = cache_dir / "candidates-89-oof.parquet"
    candidates.to_parquet(candidate_cache_path)
    candidate_cache_sha = file_sha256(candidate_cache_path)
    memory = psutil.virtual_memory()
    active = process_snapshot(source_root)
    deadline = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
    hours_remaining = (deadline - datetime.now(UTC)).total_seconds() / 3600
    _require(memory.total >= 40 * 2**30, "로컬 메모리가 40GiB보다 작아 고정 상한 안에서 안전하게 판정하기 어렵다.")
    _require(memory.available / memory.total >= MEMORY_HEADROOM_MIN, "사전 점검 메모리 여유율이 하한보다 낮다.")
    _require(not active, f"같은 저장소의 다른 Python 실행이 진행 중이다: {active}")
    _require(hours_remaining >= 8, "목적지 시각까지 남은 시간이 로컬 판정 여유 8시간보다 짧다.")

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "created_at": now_iso(),
        "question": "동결 자체 후보 전체 확장 402개와 고정 학습 길이 트리 변형을 뺀 절제 확장 361개 가운데 현재 313개 기준을 엄격 문턱으로 넘는 구성이 있는가.",
        "candidate_set_id": spec["candidate_set_id"],
        "freeze_spec": {
            "path": str(FREEZE_SPEC_PATH),
            "file_sha256": file_sha256(FREEZE_SPEC_PATH),
            "spec_sha256": spec["spec_sha256"],
            "content_sha256": spec["content_sha256"],
            "candidate_count": spec["candidate_count"],
            "restrained_candidate_count": spec["restrained_candidate_count"],
            "fixed_training_length_tree_variant_count": spec["fixed_training_length_tree_variants"]["count"],
        },
        "source_root": str(source_root),
        "inputs": {
            key: {"path": str(paths[key]), "sha256": file_sha256(paths[key])}
            for key in ("train", "test", "folds", "pool")
        },
        "folds": {
            "order": list(ALL_FOLDS),
            "vector_sha256": hashlib.sha256(fold_of.to_numpy(np.int8).tobytes()).hexdigest(),
            "rows": {str(fold): int((fold_of == fold).sum()) for fold in ALL_FOLDS},
        },
        "baseline": {
            "member_count": BASE_MEMBER_COUNT,
            "manifest": {"path": str(COMPARISON_MANIFEST_PATH), "sha256": file_sha256(COMPARISON_MANIFEST_PATH)},
            "members": prior_member_rows,
            "columns": base_columns,
            "column_order_sha256": canonical_sha256(base_columns),
            "cache": {"path": str(paths["base_cache"]), "sha256": base_cache_sha},
            "reference": {
                **CONTROL_REFERENCE,
                "source": "이슈 489의 C=1.0 대조군",
                "precommit_path": str(ISSUE489_PRECOMMIT_PATH),
                "precommit_file_sha256": file_sha256(ISSUE489_PRECOMMIT_PATH),
                "comparison_path": str(ISSUE489_COMPARISON_PATH),
                "comparison_file_sha256": file_sha256(ISSUE489_COMPARISON_PATH),
            },
            "reproduction_tolerance": REPRODUCTION_TOLERANCE,
        },
        "candidates": {
            "count": len(candidate_records),
            "rows": candidate_records,
            "columns": list(candidates.columns),
            "column_order_sha256": canonical_sha256(list(candidates.columns)),
            "cache": {"path": str(candidate_cache_path), "sha256": candidate_cache_sha},
        },
        "configurations": configs,
        "combiner": {
            "name": STRATEGY,
            "implementation": "pipeline.ensemble.COMBINER_REGISTRY 등록 결합기를 각 바깥쪽 분할에 독립 적합하고 봉인 분할을 예측한다.",
            "representation": "rank_logit",
            "logit_eps": LOGIT_EPS,
            "penalty": "l2",
            "c": 1.0,
            "solver": "lbfgs",
            "max_iter": META_MAX_ITER,
            "random_state": 0,
            "lambda_grid": list(LAMBDA_GRID),
            "lambda_tie": "AUC가 정확히 같으면 오름차순 격자에서 먼저 오는 더 작은 λ",
        },
        "gate": {
            "delta_required": GATE_DELTA,
            "folds_required_positive": FOLDS_REQUIRED_POSITIVE,
            "noise_floor": NOISE_FLOOR,
            "both_pass_tie": "두 후보의 중첩 OOF AUC 차이가 잡음 바닥 이하면 구성원이 적은 restrained361",
        },
        "rules": rules(),
        "resources": {
            "environment": "local CPU",
            "workers": args.workers,
            "workers_maximum": MAX_WORKERS,
            "threads_per_job": args.threads,
            "memory_headroom_minimum": MEMORY_HEADROOM_MIN,
            "runner_poll_seconds": 10,
            "preflight": {
                "memory_total_bytes": int(memory.total),
                "memory_available_bytes": int(memory.available),
                "memory_available_fraction": memory.available / memory.total,
                "active_repo_python_processes": active,
                "deadline": deadline.isoformat().replace("+00:00", "Z"),
                "hours_remaining": hours_remaining,
                "selection": "로컬 입력과 산출물이 원본 저장소에 있고 메모리, 경합과 남은 시간이 고정 상한 안에서 충분해 이 컴퓨터를 선택했다.",
            },
        },
        "environment": environment(threads=args.threads),
        "code_state": code_state(),
    }
    payload["precommit_sha256"] = canonical_sha256(payload)
    write_json(run_dir / "precommit.json", payload)
    print(f"precommit 저장: {run_dir / 'precommit.json'}")
    print(f"  후보 집합 {spec['candidate_set_id']}, 구성 313/402/361, 동시 {args.workers}, 작업당 병렬 연산 {args.threads}")
    print(f"  후보 캐시 {candidate_cache_sha}, precommit {payload['precommit_sha256']}")


def load_precommit(run_dir: Path) -> dict:
    path = run_dir / "precommit.json"
    _require(path.is_file(), f"precommit.json이 없다: {path}")
    payload = read_json(path)
    digest = canonical_sha256(
        {key: value for key, value in payload.items() if key != "precommit_sha256"}
    )
    _require(digest == payload["precommit_sha256"], "precommit.json이 제자리에서 바뀌었다.")
    _require(file_sha256(FREEZE_SPEC_PATH) == payload["freeze_spec"]["file_sha256"], "동결 명세 파일이 precommit과 다르다.")
    spec = freeze.verify_spec_file(FREEZE_SPEC_PATH)
    _require(spec["spec_sha256"] == payload["freeze_spec"]["spec_sha256"], "동결 명세 내용이 precommit과 다르다.")
    for entry in payload["inputs"].values():
        _require(file_sha256(Path(entry["path"])) == entry["sha256"], f"입력 해시가 precommit과 다르다: {entry['path']}")
    _require(file_sha256(Path(payload["baseline"]["cache"]["path"])) == payload["baseline"]["cache"]["sha256"], "313 OOF 캐시가 precommit과 다르다.")
    _require(file_sha256(Path(payload["candidates"]["cache"]["path"])) == payload["candidates"]["cache"]["sha256"], "89개 후보 OOF 캐시가 precommit과 다르다.")
    for key in ("manifest",):
        entry = payload["baseline"][key]
        _require(file_sha256(Path(entry["path"])) == entry["sha256"], f"313 {key}가 precommit과 다르다.")
    reference = payload["baseline"]["reference"]
    _require(file_sha256(Path(reference["precommit_path"])) == reference["precommit_file_sha256"], "이슈 489 precommit이 바뀌었다.")
    _require(file_sha256(Path(reference["comparison_path"])) == reference["comparison_file_sha256"], "이슈 489 판정 근거가 바뀌었다.")
    state = code_state()
    for label, actual, expected in (
        ("git commit", state["git"]["commit"], payload["code_state"]["git"]["commit"]),
        ("판정 도구", state["script"]["sha256"], payload["code_state"]["script"]["sha256"]),
        ("결합기 module", state["ensemble_module"]["sha256"], payload["code_state"]["ensemble_module"]["sha256"]),
        ("동결 도구", state["freeze_script"]["sha256"], payload["code_state"]["freeze_script"]["sha256"]),
        ("uv.lock", state["uv_lock_sha256"], payload["code_state"]["uv_lock_sha256"]),
    ):
        _require(actual == expected, f"코드 상태({label})가 precommit과 다르다. precommit부터 다시 해야 한다.")
    return payload


def config_by_name(payload: dict, name: str) -> dict:
    config = next((row for row in payload["configurations"] if row["name"] == name), None)
    _require(config is not None, f"precommit에 없는 구성이다: {name}")
    return config


def load_matrix(run_dir: Path, payload: dict, config: dict, fold_of: pd.Series) -> pd.DataFrame:
    base = pd.read_parquet(payload["baseline"]["cache"]["path"]).astype(np.float64)
    frames = [base]
    if config["candidate_columns"]:
        candidates = pd.read_parquet(
            payload["candidates"]["cache"]["path"],
            columns=config["candidate_columns"],
        ).astype(np.float64)
        frames.append(candidates)
    matrix = pd.concat(frames, axis=1)
    _require(list(matrix.columns) == config["columns"], f"{config['name']}: 열 순서가 precommit과 다르다.")
    _require(matrix.index.equals(fold_of.index), f"{config['name']}: 행 순서가 분할과 다르다.")
    _require(matrix.shape == (N_TRAIN, config["member_count"]), f"{config['name']}: 행렬 형태가 기대와 다르다.")
    _require(bool(np.isfinite(matrix.to_numpy()).all()), f"{config['name']}: 유한하지 않은 OOF가 있다.")
    return matrix


def fold_output(run_dir: Path, config: str, fold: int) -> Path:
    return run_dir / "configs" / config / f"fold-{fold}" / "fold.json"


def fold_job(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    fold = int(args.fold)
    _require(fold in payload["folds"]["order"], f"precommit에 없는 분할이다: {fold}")
    config = config_by_name(payload, args.config)
    out_dir = fold_output(run_dir, config["name"], fold).parent
    out_path = out_dir / "fold.json"
    _require(not out_path.exists(), f"완료 산출물이 이미 있다: {out_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    paths = source_paths(Path(payload["source_root"]))
    fold_of, y, _ = load_folds_and_labels(paths)
    matrix = load_matrix(run_dir, payload, config, fold_of)
    inner = (fold_of != fold).to_numpy()
    outer = (fold_of == fold).to_numpy()
    print(f"=== {config['name']} {matrix.shape[1]}구성원, 봉인 분할 {fold} ===", flush=True)
    try:
        fitted = ensemble.COMBINER_REGISTRY[STRATEGY].fit(matrix[inner], y[inner])
    except ensemble.CombinerConvergenceError as exc:
        raise JudgmentError(f"{config['name']} 분할 {fold} 미수렴: {exc}") from exc
    prediction = np.asarray(fitted.predict(matrix[outer]), dtype=np.float64)
    _require(prediction.shape == (int(outer.sum()),) and bool(np.isfinite(prediction).all()), "봉인 분할 예측의 형태 또는 유한값 검사 실패")
    prediction_path = out_dir / "predictions.parquet"
    pd.DataFrame(
        {ID: fold_of.index.to_numpy()[outer], "prediction": prediction}
    ).to_parquet(prediction_path, index=False)
    auc = float(roc_auc_score(y[outer].to_numpy(), prediction))
    record = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "precommit_sha256": payload["precommit_sha256"],
        "config": config["name"],
        "member_count": config["member_count"],
        "column_order_sha256": canonical_sha256(config["columns"]),
        "sealed_fold": fold,
        "rows": int(outer.sum()),
        "strategy": STRATEGY,
        "c": 1.0,
        "lambda": float(fitted.shrinkage_lambda),
        "auc": auc,
        "prediction_sha256": prediction_array_sha256(prediction),
        "prediction_file_sha256": file_sha256(prediction_path),
        "final_iterations": int(np.max(fitted.meta.model.n_iter_)),
        "final_coefficient_l2_norm": float(np.linalg.norm(fitted.meta.model.coef_[0])),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "environment": environment(threads=payload["resources"]["threads_per_job"]),
        "finished_at": now_iso(),
    }
    write_json(out_path, record)
    print(f"  AUC {auc:.15f}, λ={record['lambda']}, 반복 {record['final_iterations']}, {record['elapsed_seconds']:.0f}s", flush=True)


def load_config_outputs(
    run_dir: Path,
    payload: dict,
    config_name: str,
    fold_of: pd.Series,
    y: pd.Series,
) -> tuple[dict[str, dict], pd.Series]:
    records: dict[str, dict] = {}
    prediction = pd.Series(np.nan, index=fold_of.index, dtype=np.float64)
    for fold in payload["folds"]["order"]:
        record_path = fold_output(run_dir, config_name, fold)
        _require(record_path.is_file(), f"{config_name} 분할 {fold} 완료 산출물이 없다.")
        record = read_json(record_path)
        _require(record["precommit_sha256"] == payload["precommit_sha256"], f"{config_name} 분할 {fold}이 다른 precommit에서 나왔다.")
        part_path = record_path.parent / "predictions.parquet"
        _require(file_sha256(part_path) == record["prediction_file_sha256"], f"{config_name} 분할 {fold} 예측 파일 해시가 다르다.")
        part = pd.read_parquet(part_path).set_index(ID)["prediction"]
        ids = fold_of.index[(fold_of == fold).to_numpy()]
        _require(part.index.equals(pd.Index(ids)), f"{config_name} 분할 {fold} 식별자 순서가 다르다.")
        values = part.to_numpy(np.float64)
        _require(prediction_array_sha256(values) == record["prediction_sha256"], f"{config_name} 분할 {fold} 예측 배열 해시가 다르다.")
        recalculated = float(roc_auc_score(y.loc[ids].to_numpy(), values))
        _require(recalculated == record["auc"], f"{config_name} 분할 {fold} AUC 재계산이 기록과 다르다.")
        prediction.loc[ids] = values
        records[str(fold)] = record
    _require(prediction.notna().all(), f"{config_name} 이어붙인 예측에 빈 행이 있다.")
    return records, prediction


def reproduction_check(
    run_dir: Path, payload: dict, fold_of: pd.Series, y: pd.Series
) -> dict[str, object]:
    records, prediction = load_config_outputs(
        run_dir, payload, CONFIG_BASELINE, fold_of, y
    )
    nested_auc = float(roc_auc_score(y.to_numpy(), prediction.to_numpy()))
    reference = payload["baseline"]["reference"]
    delta = nested_auc - reference["nested_auc"]
    fold_deltas = {
        key: records[key]["auc"] - reference["fold_aucs"][key]
        for key in records
    }
    passes = abs(delta) <= REPRODUCTION_TOLERANCE and all(
        abs(value) <= REPRODUCTION_TOLERANCE for value in fold_deltas.values()
    )
    return {
        "precommit_sha256": payload["precommit_sha256"],
        "tolerance": REPRODUCTION_TOLERANCE,
        "reference_nested_auc": reference["nested_auc"],
        "reproduced_nested_auc": nested_auc,
        "delta": delta,
        "reference_fold_aucs": reference["fold_aucs"],
        "reproduced_fold_aucs": {key: row["auc"] for key, row in records.items()},
        "fold_deltas": fold_deltas,
        "fold_lambdas": {key: row["lambda"] for key, row in records.items()},
        "passes": bool(passes),
        "checked_at": now_iso(),
    }


def _memory_headroom() -> float:
    memory = psutil.virtual_memory()
    return memory.available / memory.total


def _running_jobs(run_dir: Path) -> set[tuple[str, int]]:
    output = subprocess.run(
        ["ps", "-axo", "command="], capture_output=True, text=True, check=False
    ).stdout
    pattern = re.compile(
        rf"judge_reusable_own_extension\.py fold --run-dir {re.escape(str(run_dir))} --config ([a-z0-9]+) --fold (\d+)"
    )
    return {(config, int(fold)) for config, fold in pattern.findall(output)}


def _run_phase(
    run_dir: Path,
    payload: dict,
    jobs: list[tuple[str, int]],
    phase: str,
) -> None:
    workers = int(payload["resources"]["workers"])
    threads = int(payload["resources"]["threads_per_job"])
    pending = [job for job in jobs if not fold_output(run_dir, *job).is_file()]
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = str(threads)
    env["OMP_DYNAMIC"] = "FALSE"
    print(f"[{phase}] 남은 작업 {len(pending)}/{len(jobs)}, 동시 {workers}, 작업당 병렬 연산 {threads}", flush=True)
    active: dict[tuple[str, int], tuple[subprocess.Popen, object]] = {}
    failures: list[tuple[str, int]] = []
    while pending or active:
        for job, (process, handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[job]
            if code != 0:
                failures.append(job)
            print(f"[{phase}] {'완료' if code == 0 else f'실패({code})'} {job[0]} 분할 {job[1]} {now_iso()}", flush=True)
        if failures:
            for process, handle in active.values():
                process.terminate()
                handle.close()
            raise JudgmentError(f"{phase} 실패 작업 {failures}. 완료한 일부 결과로 판정하지 않는다.")
        running = _running_jobs(run_dir) | set(active)
        while pending and len(running) < workers:
            headroom = _memory_headroom()
            if headroom < MEMORY_HEADROOM_MIN:
                print(f"[{phase}] 메모리 여유율 {headroom:.1%}가 하한 {MEMORY_HEADROOM_MIN:.0%}보다 낮아 새 작업을 보류한다.", flush=True)
                break
            job = pending.pop(0)
            if job in running or fold_output(run_dir, *job).is_file():
                continue
            config, fold = job
            handle = (log_dir / f"{config}-fold-{fold}.log").open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "fold",
                "--run-dir",
                str(run_dir),
                "--config",
                config,
                "--fold",
                str(fold),
            ]
            process = subprocess.Popen(
                command, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
            active[job] = (process, handle)
            running.add(job)
            print(f"[{phase}] 시작 {config} 분할 {fold} {now_iso()} (메모리 여유율 {headroom:.1%})", flush=True)
        time.sleep(10)


def run_jobs(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    baseline_jobs = [(CONFIG_BASELINE, fold) for fold in payload["folds"]["order"]]
    _run_phase(run_dir, payload, baseline_jobs, "기준 자기 검사")
    paths = source_paths(Path(payload["source_root"]))
    fold_of, y, _ = load_folds_and_labels(paths)
    reproduction = reproduction_check(run_dir, payload, fold_of, y)
    reproduction_path = run_dir / "configs" / CONFIG_BASELINE / "reproduction.json"
    if reproduction_path.exists():
        existing = read_json(reproduction_path)
        comparable_existing = {
            key: value for key, value in existing.items() if key != "checked_at"
        }
        comparable_current = {
            key: value for key, value in reproduction.items() if key != "checked_at"
        }
        _require(
            comparable_existing == comparable_current,
            "기존 기준 재현 기록과 재계산 결과가 다르다.",
        )
        reproduction = existing
    else:
        write_json(reproduction_path, reproduction)
    print(f"기준 재현 전체 차이 {reproduction['delta']:+.2e}, 분할 최대 차이 {max(abs(v) for v in reproduction['fold_deltas'].values()):.2e}: {'통과' if reproduction['passes'] else '실패'}", flush=True)
    _require(reproduction["passes"], "기준 313개 자기 검사가 실패해 402개와 361개 판정을 시작하지 않는다.")
    candidate_jobs = [
        (config, fold)
        for fold in payload["folds"]["order"]
        for config in (CONFIG_FULL, CONFIG_RESTRAINED)
    ]
    _run_phase(run_dir, payload, candidate_jobs, "후보 사다리")
    print("세 구성의 모든 분할 작업을 완료했다.", flush=True)


def near_duplicate_diagnostics(
    base: pd.DataFrame, candidates: pd.DataFrame
) -> dict[str, object]:
    started = time.monotonic()
    matrix = pd.concat([base, candidates], axis=1)
    ranks = matrix.rank(method="average").to_numpy(np.float64)
    correlation = np.corrcoef(ranks.T)
    columns = list(matrix.columns)
    pairs = []
    upper = np.triu_indices(len(columns), k=1)
    for i, j in zip(upper[0], upper[1], strict=True):
        if j < BASE_MEMBER_COUNT:
            continue
        value = float(correlation[i, j])
        if value >= DUPLICATE_SPEARMAN:
            pairs.append(
                {
                    "a": columns[int(i)],
                    "b": columns[int(j)],
                    "kind": "baseline-candidate" if i < BASE_MEMBER_COUNT else "candidate-candidate",
                    "spearman": value,
                }
            )
    maxima = {}
    for index in range(BASE_MEMBER_COUNT, len(columns)):
        values = correlation[index].copy()
        values[index] = -np.inf
        nearest = int(np.argmax(values))
        maxima[columns[index]] = {
            "nearest": columns[nearest],
            "spearman": float(values[nearest]),
        }
    return {
        "threshold": DUPLICATE_SPEARMAN,
        "scope": "전체 OOF 691,369행에서 후보가 포함된 쌍",
        "pair_count": len(pairs),
        "pairs": pairs,
        "candidate_nearest": maxima,
        "elapsed_seconds": time.monotonic() - started,
        "note": "진단값이며 구성 변경과 선택에 사용하지 않았다.",
    }


def compare(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    out_path = run_dir / "judgment.json"
    _require(not out_path.exists(), f"판정 기록이 이미 있다: {out_path}")
    payload = load_precommit(run_dir)
    paths = source_paths(Path(payload["source_root"]))
    fold_of, y, _ = load_folds_and_labels(paths)
    reproduction = reproduction_check(run_dir, payload, fold_of, y)
    _require(reproduction["passes"], "기준 자기 검사가 실패해 판정할 수 없다.")
    outputs: dict[str, dict[str, object]] = {}
    records_by_config: dict[str, dict[str, dict]] = {}
    predictions: dict[str, pd.Series] = {}
    for name in CONFIG_ORDER:
        records, prediction = load_config_outputs(run_dir, payload, name, fold_of, y)
        nested_auc = float(roc_auc_score(y.to_numpy(), prediction.to_numpy()))
        records_by_config[name] = records
        predictions[name] = prediction
        outputs[name] = {
            "member_count": config_by_name(payload, name)["member_count"],
            "nested_auc": nested_auc,
            "fold_aucs": {key: row["auc"] for key, row in records.items()},
            "fold_lambdas": {key: row["lambda"] for key, row in records.items()},
            "prediction_sha256": prediction_array_sha256(prediction.to_numpy()),
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in records.values()),
            "peak_rss_bytes_max": max(row["peak_rss_bytes"] for row in records.values()),
        }
    baseline_auc = outputs[CONFIG_BASELINE]["nested_auc"]
    gate_results: dict[str, dict[str, object]] = {}
    passing = []
    for name in (CONFIG_FULL, CONFIG_RESTRAINED):
        fold_deltas = {
            key: outputs[name]["fold_aucs"][key]
            - outputs[CONFIG_BASELINE]["fold_aucs"][key]
            for key in outputs[CONFIG_BASELINE]["fold_aucs"]
        }
        delta = outputs[name]["nested_auc"] - baseline_auc
        positive = sum(value > 0 for value in fold_deltas.values())
        passes = delta >= GATE_DELTA and positive == FOLDS_REQUIRED_POSITIVE
        gate_results[name] = {
            "delta_vs_baseline": delta,
            "fold_deltas": fold_deltas,
            "folds_positive": positive,
            "delta_gate_passes": bool(delta >= GATE_DELTA),
            "fold_gate_passes": bool(positive == FOLDS_REQUIRED_POSITIVE),
            "passes": bool(passes),
        }
        if passes:
            passing.append(name)

    selection_path = []
    if not passing:
        selected = CONFIG_BASELINE
        selection_path.append("full402와 restrained361 가운데 엄격 문턱을 모두 만족한 구성이 없어 baseline313을 유지했다.")
    elif len(passing) == 1:
        selected = passing[0]
        selection_path.append(f"엄격 문턱을 만족한 구성이 {selected} 하나여서 이를 선택했다.")
    else:
        difference = abs(outputs[CONFIG_FULL]["nested_auc"] - outputs[CONFIG_RESTRAINED]["nested_auc"])
        if difference <= NOISE_FLOOR:
            selected = CONFIG_RESTRAINED
            selection_path.append(f"두 후보가 모두 통과했고 중첩 OOF 차이 {difference:.12g}가 잡음 바닥 {NOISE_FLOOR} 이하여서 구성원이 적은 restrained361을 선택했다.")
        else:
            selected = max(passing, key=lambda name: outputs[name]["nested_auc"])
            selection_path.append(f"두 후보가 모두 통과했고 중첩 OOF 차이가 잡음 바닥보다 커서 값이 높은 {selected}을 선택했다.")

    diagnostics_started = time.monotonic()
    base = pd.read_parquet(payload["baseline"]["cache"]["path"]).astype(np.float64)
    candidates = pd.read_parquet(payload["candidates"]["cache"]["path"]).astype(np.float64)
    standalone = [
        {
            "order": index + 1,
            "column": column,
            "auc": float(roc_auc_score(y.to_numpy(), candidates[column].to_numpy())),
        }
        for index, column in enumerate(candidates.columns)
    ]
    near_duplicates = near_duplicate_diagnostics(base, candidates)
    diagnostics = {
        "candidate_standalone_auc": standalone,
        "near_duplicates": near_duplicates,
        "prediction_spearman": {
            "full402_vs_baseline313": float(spearmanr(predictions[CONFIG_FULL], predictions[CONFIG_BASELINE]).correlation),
            "restrained361_vs_baseline313": float(spearmanr(predictions[CONFIG_RESTRAINED], predictions[CONFIG_BASELINE]).correlation),
            "full402_vs_restrained361": float(spearmanr(predictions[CONFIG_FULL], predictions[CONFIG_RESTRAINED]).correlation),
        },
        "elapsed_seconds": time.monotonic() - diagnostics_started,
        "note": "모든 진단은 고정 사다리 실행 뒤 계산했으며 구성 변경과 선택에 사용하지 않았다.",
    }
    verdict = (
        "현재 313개 기준 구성 유지"
        if selected == CONFIG_BASELINE
        else f"{selected} 통과 및 선택"
    )
    record = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "candidate_set_id": payload["candidate_set_id"],
        "precommit_sha256": payload["precommit_sha256"],
        "reproduction": reproduction,
        "configurations": outputs,
        "gate": payload["gate"],
        "gate_results": gate_results,
        "selected_config": selected,
        "selected_member_count": outputs[selected]["member_count"],
        "selection_path": selection_path,
        "verdict": verdict,
        "diagnostics": diagnostics,
        "public_score_used": False,
        "kaggle_submission_created": False,
        "resources": {
            "fold_elapsed_seconds_total": sum(row["elapsed_seconds"] for row in outputs.values()),
            "fold_peak_rss_bytes_max": max(row["peak_rss_bytes_max"] for row in outputs.values()),
            "comparison_peak_rss_bytes": peak_rss_bytes(),
            "wall_elapsed_seconds_since_precommit": (
                datetime.now(UTC)
                - datetime.fromisoformat(payload["created_at"].replace("Z", "+00:00"))
            ).total_seconds(),
            "workers": payload["resources"]["workers"],
            "threads_per_job": payload["resources"]["threads_per_job"],
            "environment": "local CPU",
        },
        "compared_at": now_iso(),
    }
    write_json(out_path, record)
    print(f"기준 nested {baseline_auc:.12f}")
    for name in (CONFIG_FULL, CONFIG_RESTRAINED):
        gate = gate_results[name]
        print(f"{name} nested {outputs[name]['nested_auc']:.12f}, 차이 {gate['delta_vs_baseline']:+.9f}, 분할 양수 {gate['folds_positive']}/5, {'통과' if gate['passes'] else '미달'}")
    print(f"판정: {verdict}")


def _gb(value: int | float) -> str:
    return f"{value / 2**30:.1f}GiB"


def _manifest_files(run_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.name != "manifest.sha256"
        and path.relative_to(run_dir).parts[0] != "logs"
    ]


def report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.sha256"
    _require(not report_path.exists() and not manifest_path.exists(), "보고서 또는 해시 명세가 이미 있다.")
    payload = load_precommit(run_dir)
    judgment = read_json(run_dir / "judgment.json")
    _require(judgment["precommit_sha256"] == payload["precommit_sha256"], "판정 기록이 다른 precommit에서 나왔다.")
    configs = judgment["configurations"]
    lines = [f"# 재사용 적격 자체 후보 확장 사다리 판정 보고 (#{ISSUE})", ""]
    lines += ["## 판정", ""]
    lines.append(f"- 결과: **{judgment['verdict']}**.")
    lines.append(f"- 선택 구성은 `{judgment['selected_config']}` {judgment['selected_member_count']}개다.")
    lines.append(f"- {judgment['selection_path'][0]}")
    lines.append("- Public 점수와 Kaggle 제출은 사용하지 않았다.")
    lines.append("")
    lines += ["| 구성 | 구성원 | 이어붙인 중첩 OOF AUC | 기준 대비 차이 | 분할 양수 | 판정 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    lines.append(f"| `{CONFIG_BASELINE}` | {configs[CONFIG_BASELINE]['member_count']} | {configs[CONFIG_BASELINE]['nested_auc']:.12f} | 기준 | 기준 | 자기 검사 통과 |")
    for name in (CONFIG_FULL, CONFIG_RESTRAINED):
        gate = judgment["gate_results"][name]
        lines.append(f"| `{name}` | {configs[name]['member_count']} | {configs[name]['nested_auc']:.12f} | {gate['delta_vs_baseline']:+.9f} | {gate['folds_positive']}/5 | {'통과' if gate['passes'] else '미달'} |")
    lines += ["", "## 기준 자기 검사", ""]
    reproduction = judgment["reproduction"]
    lines.append(f"- 전체 AUC는 `{reproduction['reproduced_nested_auc']:.15f}`로 기준과 차이 `{reproduction['delta']:+.2e}`였다.")
    lines.append(f"- 분할별 최대 절대 차이는 `{max(abs(value) for value in reproduction['fold_deltas'].values()):.2e}`였고 허용 오차는 `{REPRODUCTION_TOLERANCE}`였다.")
    lines.append("- 자기 검사 통과 뒤에만 402개와 361개 분할 작업을 시작했다.")
    lines += ["", "## 분할별 결과", "", "| 분할 | 313 AUC | 402 AUC | 402 차이 | 361 AUC | 361 차이 | 313 λ | 402 λ | 361 λ |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for fold in map(str, ALL_FOLDS):
        base_auc = configs[CONFIG_BASELINE]["fold_aucs"][fold]
        full_auc = configs[CONFIG_FULL]["fold_aucs"][fold]
        restrained_auc = configs[CONFIG_RESTRAINED]["fold_aucs"][fold]
        lines.append(f"| {fold} | {base_auc:.12f} | {full_auc:.12f} | {full_auc - base_auc:+.9f} | {restrained_auc:.12f} | {restrained_auc - base_auc:+.9f} | {configs[CONFIG_BASELINE]['fold_lambdas'][fold]} | {configs[CONFIG_FULL]['fold_lambdas'][fold]} | {configs[CONFIG_RESTRAINED]['fold_lambdas'][fold]} |")
    lines += ["", "## 진단값", ""]
    standalone = judgment["diagnostics"]["candidate_standalone_auc"]
    near = judgment["diagnostics"]["near_duplicates"]
    lines.append(f"- 후보 89개의 단독 OOF AUC 범위는 `{min(row['auc'] for row in standalone):.7f}`부터 `{max(row['auc'] for row in standalone):.7f}`까지다.")
    lines.append(f"- 전체 OOF에서 후보가 포함된 스피어만 `{near['threshold']}` 이상 근접 중복 쌍은 {near['pair_count']}개다.")
    lines.append("- 단독 AUC와 근접 중복은 모두 판정 뒤 계산한 진단값이며 구성 변경과 선택에 사용하지 않았다.")
    lines += ["", "## 실행 환경과 자원", ""]
    env = payload["environment"]
    resources = judgment["resources"]
    lines.append(f"- `{env['platform']}` `{env['machine']}`, CPU {env['cpu_count']}개, 메모리 {_gb(env['memory_total_bytes'])}, Python {env['python']}, NumPy {env['numpy']}, pandas {env['pandas']}, scikit-learn {env['scikit_learn']}, SciPy {env['scipy']}를 사용했다.")
    lines.append(f"- BLAS 기록은 `{json.dumps(env['blas'], ensure_ascii=False, sort_keys=True)}`다.")
    lines.append(f"- 로컬 CPU에서 동시 작업 {resources['workers']}개와 작업당 병렬 연산 {resources['threads_per_job']}개를 사용했다.")
    lines.append(f"- 분할 작업 경과 시간 합계는 {resources['fold_elapsed_seconds_total'] / 60:.1f}분이고 작업 하나의 최대 메모리는 {_gb(resources['fold_peak_rss_bytes_max'])}다.")
    lines.append(f"- 사전 고정부터 판정까지 벽시계 경과 시간은 {resources['wall_elapsed_seconds_since_precommit'] / 60:.1f}분이고 비교 및 진단 프로세스 최대 메모리는 {_gb(resources['comparison_peak_rss_bytes'])}다.")
    lines += ["", "## 고정 입력과 산출물", ""]
    lines.append(f"- 후보 집합은 `{payload['candidate_set_id']}`, 동결 명세 `spec_sha256`은 `{payload['freeze_spec']['spec_sha256']}`다.")
    lines.append(f"- 313 열 순서 해시는 `{payload['baseline']['column_order_sha256']}`, 후보 89개 열 순서 해시는 `{payload['candidates']['column_order_sha256']}`다.")
    lines.append(f"- 판정 코드 커밋은 `{payload['code_state']['git']['commit']}`, 판정 도구 해시는 `{payload['code_state']['script']['sha256']}`다.")
    lines.append(f"- precommit SHA-256은 `{payload['precommit_sha256']}`다.")
    lines.append("- 분할별 예측 parquet와 기록 JSON, 판정 JSON, 이 보고서와 캐시는 `manifest.sha256`에 내용 해시로 열거한다.")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest_lines = [
        f"{file_sha256(path)}  {path.relative_to(run_dir)}"
        for path in _manifest_files(run_dir)
    ]
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"보고서 {report_path}, 해시 명세 {manifest_path} ({len(manifest_lines)}개 파일)")
    print(f"manifest.sha256 파일 해시 {file_sha256(manifest_path)}")


def verify(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    judgment = read_json(run_dir / "judgment.json")
    _require(judgment["precommit_sha256"] == payload["precommit_sha256"], "판정 JSON이 다른 precommit에서 나왔다.")
    manifest_path = run_dir / "manifest.sha256"
    _require(manifest_path.is_file(), "manifest.sha256이 없다.")
    checked = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = run_dir / relative
        _require(path.is_file(), f"해시 명세 파일이 없다: {relative}")
        _require(file_sha256(path) == digest, f"해시 명세가 파일과 다르다: {relative}")
        checked += 1
    _require(checked == len(_manifest_files(run_dir)), "해시 명세 파일 수가 실제 판정 묶음과 다르다.")
    print(f"판정 묶음 검사 통과: {checked}개 파일, manifest {file_sha256(manifest_path)}")


def publish(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    verify(argparse.Namespace(run_dir=run_dir))
    out_dir = Path(args.out_dir)
    _require(not out_dir.exists(), f"공개 기록 경로가 이미 있다: {out_dir}")
    files = [
        Path("precommit.json"),
        Path("judgment.json"),
        Path("report.md"),
        Path("manifest.sha256"),
        Path("configs") / CONFIG_BASELINE / "reproduction.json",
    ]
    for config in CONFIG_ORDER:
        for fold in ALL_FOLDS:
            files.append(Path("configs") / config / f"fold-{fold}" / "fold.json")
    for relative in files:
        source = run_dir / relative
        _require(source.is_file(), f"공개할 기록이 없다: {source}")
        destination = out_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(f"판정 기록 공개 경로: {out_dir} ({len(files)}개 파일, 예측 parquet와 캐시는 run-logs에만 유지)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("precommit")
    pre.add_argument("--run-dir", type=Path, default=OUT_DIR)
    pre.add_argument("--source-root", type=Path, default=default_source_root())
    pre.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    pre.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    pre.set_defaults(handler=precommit)

    run = sub.add_parser("run")
    run.add_argument("--run-dir", type=Path, default=OUT_DIR)
    run.set_defaults(handler=run_jobs)

    fold = sub.add_parser("fold")
    fold.add_argument("--run-dir", type=Path, default=OUT_DIR)
    fold.add_argument("--config", choices=CONFIG_ORDER, required=True)
    fold.add_argument("--fold", type=int, choices=ALL_FOLDS, required=True)
    fold.set_defaults(handler=fold_job)

    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--run-dir", type=Path, default=OUT_DIR)
    compare_parser.set_defaults(handler=compare)

    report_parser = sub.add_parser("report")
    report_parser.add_argument("--run-dir", type=Path, default=OUT_DIR)
    report_parser.set_defaults(handler=report)

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--run-dir", type=Path, default=OUT_DIR)
    verify_parser.set_defaults(handler=verify)

    publish_parser = sub.add_parser("publish")
    publish_parser.add_argument("--run-dir", type=Path, default=OUT_DIR)
    publish_parser.add_argument("--out-dir", type=Path, default=PUBLISH_DIR)
    publish_parser.set_defaults(handler=publish)

    args = parser.parse_args()
    try:
        args.handler(args)
    except (JudgmentError, freeze.FreezeError, OSError, KeyError, ValueError) as exc:
        sys.exit(f"판정 불가: {exc}")


if __name__ == "__main__":
    main()
