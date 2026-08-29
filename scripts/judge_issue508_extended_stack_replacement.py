"""현재 313개 결합에서 exp117 원자 교체의 nested 효과를 판정한다. (#508)

현재 두 번째 제출의 313개 구성원, 순서와 C 선택 결합 절차를 기준으로 고정한다.
후보 팔은 ``exp117_ag25_gbm_r21`` 열 하나를
``exp208_issue500_ag25_missingness_augmented``의 세 시드 평균 OOF로 바꾼다.
다섯 바깥쪽 분할마다 나머지 네 분할 안에서 C와 수축 계수를 다시 선택하므로,
구성원 교체가 선택 절차 전체에 주는 효과를 현재 제출과 직접 비교한다.

판정 문턱은 현재 두 번째 제출 대비 nested OOF ``+0.00002`` 이상과 바깥쪽 분할
``5/5`` 양수다.
중복 진단은 교체 전 구성원과 교체 후 구성원이 공통으로 남는 312개 구성원과 맺는
스피어만 순위 상관 관계를 비교한다.
나머지 구성원 사이의 관계는 두 팔에서 동일하므로 다시 계산하지 않는다.

산출물은 ``run-logs/extended-stack-atomic-replacement/issue508``에 먼저 만들고,
``publish``가 예측 배열과 캐시를 제외한 판정 근거를
``docs/research/extended-stack-atomic-replacement/issue508``에 복사한다.

실행 순서:

    uv run python scripts/judge_issue508_extended_stack_replacement.py precommit
    uv run python scripts/judge_issue508_extended_stack_replacement.py run --workers 3 --threads 4
    uv run python scripts/judge_issue508_extended_stack_replacement.py compare
    uv run python scripts/judge_issue508_extended_stack_replacement.py report
    uv run python scripts/judge_issue508_extended_stack_replacement.py publish
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge_extended_stack as ladder
import judge_logistic_c_selection as logistic

from pipeline import ensemble
from pipeline.data import ID, TARGET, file_sha256, labels
from pipeline.judgment import DUPLICATE_SPEARMAN
from pipeline.pool_audit import prediction_array_sha256
from pipeline.runs import MlflowRunStore


ISSUE = 508
SCHEMA = "extended-stack-atomic-replacement/1"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path("run-logs/extended-stack-atomic-replacement/issue508")
PUBLISH_DIR = Path("docs/research/extended-stack-atomic-replacement/issue508")
BASELINE_MANIFEST_PATH = Path("docs/research/extended-stack-submission-2-manifest.json")
BASELINE_PRECOMMIT_PATH = Path("docs/research/logistic-c-selection/issue489/precommit.json")
BASELINE_COMPARISON_PATH = Path("docs/research/logistic-c-selection/issue489/comparison.json")
BASELINE_ASSEMBLY_PATH = Path("docs/research/logistic-c-selection/issue489/assembly-manifest.json")
CANDIDATE_FREEZE_PATH = Path("artifacts/issue503-missingness-candidate-pool-freeze.json")
FOLDS_PATH = Path("artifacts/folds.parquet")
UV_LOCK_PATH = Path("uv.lock")
BASE_COLUMN = "exp117_ag25_gbm_r21"
BASE_RUN_ID = "d107ea874ebe4dbe8094694141a162b6"
CANDIDATE_COLUMN = "exp208_issue500_ag25_missingness_augmented"
CANDIDATE_RUN_ID = "e46d1ca38e0746209e049970d3dd2ab6"
CURRENT_SUBMISSION_RUN_ID = "30b6f97c30904995a79e476f02decf8f"
MEMBER_COUNT = 313
OWN_MEMBER_COUNT = 35
ALL_FOLDS = tuple(range(5))
GATE_DELTA = 0.00002
FOLDS_REQUIRED_POSITIVE = 5
MAX_WORKERS = 3
MEMORY_HEADROOM_MIN = 0.15
META_MAX_ITER = logistic.META_MAX_ITER
C_GRID = ensemble.C_SELECTION_GRID
LAMBDA_GRID = ensemble.SHRINKAGE_LAMBDA_GRID
CACHE_NAME = "replacement-oof-313.parquet"

read_json = logistic.read_json
write_json = logistic.write_json
canonical_sha256 = logistic.canonical_sha256
now_iso = logistic.now_iso
JudgmentError = logistic.JudgmentError
_require = logistic._require


def _assert_repo_root() -> None:
    _require(Path.cwd().resolve() == REPO_ROOT, f"저장소 루트에서 실행해야 한다: {REPO_ROOT}")


def _git_state() -> dict[str, object]:
    return logistic.strict.git_state()


def _code_state() -> dict[str, object]:
    return {
        "git": _git_state(),
        "script": {"path": str(Path(__file__).relative_to(REPO_ROOT)), "sha256": file_sha256(Path(__file__))},
        "ensemble_module": {"path": str(logistic.strict.ENSEMBLE_SOURCE), "sha256": file_sha256(logistic.strict.ENSEMBLE_SOURCE)},
        "uv_lock_sha256": file_sha256(UV_LOCK_PATH),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


def _environment() -> dict[str, object]:
    memory = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": int(memory.total),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    }


def _peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss) if sys.platform == "darwin" else int(rss) * 1024


def _source_root(freeze: dict, override: Path | None) -> Path:
    if override is not None:
        root = override.expanduser().resolve()
    else:
        artifact_uri = str(freeze["candidate"]["execution"]["artifact_uri"])
        _require(artifact_uri.startswith("/"), f"후보 artifact_uri가 절대 경로가 아니다: {artifact_uri}")
        artifact_dir = Path(artifact_uri).resolve()
        _require(artifact_dir.name == "artifacts" and artifact_dir.parent.name == CANDIDATE_RUN_ID, "후보 artifact_uri 모양이 예상과 다르다.")
        root = artifact_dir.parents[3]
    for path in (root / "data/train.csv", root / "data/test.csv", root / "mlflow.db"):
        _require(path.is_file(), f"실행 입력을 찾지 못했다: {path}")
    return root


def _load_fold_and_labels(source_root: Path) -> tuple[pd.Series, pd.Series]:
    fold_frame = pd.read_parquet(FOLDS_PATH)
    _require(list(fold_frame.columns) == [ID, "fold"], "고정 분할 열이 id, fold가 아니다.")
    fold_of = fold_frame.set_index(ID)["fold"]
    _require(sorted(fold_of.unique().tolist()) == list(ALL_FOLDS), "고정 분할이 0부터 4까지 다섯 개가 아니다.")
    y = labels(fold_of.index, train_path=source_root / "data/train.csv")
    return fold_of, y


def _load_baseline_matrix(source_root: Path, fold_of: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    manifest = read_json(BASELINE_MANIFEST_PATH)
    baseline_precommit = read_json(BASELINE_PRECOMMIT_PATH)
    members = manifest["members"]
    expected_rows = baseline_precommit["members"]["rows"]
    _require(manifest["issue"] == 457 and len(members) == MEMBER_COUNT, "현재 313개 manifest가 아니다.")
    _require(len(expected_rows) == MEMBER_COUNT, "이슈 489 precommit의 구성원이 313개가 아니다.")
    _require([row["column"] for row in expected_rows] == [row["column"] for row in members], "manifest와 이슈 489 precommit의 구성원 순서가 다르다.")
    store = MlflowRunStore(tracking_uri=f"sqlite:///{source_root / 'mlflow.db'}")
    expected_by_column = {row["column"]: row for row in expected_rows}
    columns: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for member in members:
        column = member["column"]
        if member["origin"] == "own":
            values = store.oof_of(member["run_id"]).reindex(fold_of.index)
            _require(values.notna().all(), f"{column}: 실행 OOF id가 고정 분할과 맞지 않는다.")
            array = values.to_numpy(np.float64)
        else:
            path, separator, selector = member["oof_path"].partition("[")
            source_spec = str(source_root / path)
            if separator:
                source_spec += f"[{selector}"
            array = ladder.load_ledger_array(source_spec)
            array = np.asarray(array, dtype=np.float64)
        _require(array.shape == (len(fold_of),) and bool(np.isfinite(array).all()), f"{column}: OOF 형태나 유한값 검사가 실패했다.")
        digest = prediction_array_sha256(array)
        _require(digest == expected_by_column[column]["oof_sha256"], f"{column}: OOF 해시가 이슈 489 precommit과 다르다.")
        columns[column] = array
        rows.append({"column": column, "origin": member["origin"], "run_id": member.get("run_id"), "oof_sha256": digest})
    matrix = pd.DataFrame(columns, index=fold_of.index).astype(np.float64)
    composition = canonical_sha256([(row["column"], row["oof_sha256"]) for row in rows])
    _require(composition == baseline_precommit["members"]["composition_sha256"], "현재 313개 구성 해시가 이슈 489 precommit과 다르다.")
    return matrix, rows


def _load_candidate(source_root: Path, freeze: dict, fold_of: pd.Series) -> tuple[np.ndarray, dict[str, object]]:
    candidate = freeze["candidate"]
    _require(candidate["run_id"] == CANDIDATE_RUN_ID and candidate["experiment"] == CANDIDATE_COLUMN, "후보 동결 기록의 실행 신원이 다르다.")
    artifact_dir = Path(candidate["execution"]["artifact_uri"])
    if not artifact_dir.is_absolute():
        artifact_dir = source_root / artifact_dir
    path = artifact_dir / candidate["prediction_artifacts"]["oof"]["artifact"]
    _require(path.is_file(), f"후보 OOF를 찾지 못했다: {path}")
    expected = candidate["prediction_artifacts"]["oof"]
    _require(file_sha256(path) == expected["file_sha256"], "후보 OOF 파일 해시가 동결 기록과 다르다.")
    frame = pd.read_parquet(path)
    _require(frame[ID].to_numpy().tolist() == fold_of.index.to_numpy().tolist(), "후보 OOF id 순서가 고정 분할과 다르다.")
    _require(frame["fold"].to_numpy().tolist() == fold_of.to_numpy().tolist(), "후보 OOF fold가 고정 분할과 다르다.")
    values = frame["pred"].to_numpy(np.float64)
    digest = prediction_array_sha256(values)
    _require(digest == expected["prediction_array_sha256"], "후보 OOF 예측 배열 해시가 동결 기록과 다르다.")
    return values, {"path": str(path), "file_sha256": expected["file_sha256"], "prediction_sha256": digest, "rows": len(values)}


def _tracked_input(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": file_sha256(path)}


def precommit(args: argparse.Namespace) -> None:
    _assert_repo_root()
    run_dir = Path(args.run_dir)
    _require(not (run_dir / "precommit.json").exists(), f"precommit.json이 이미 있다: {run_dir}")
    state = _code_state()
    _require(not state["git"]["dirty"], "판정은 커밋된 코드 상태에서만 시작한다.")
    baseline_precommit = read_json(BASELINE_PRECOMMIT_PATH)
    _require(state["ensemble_module"]["sha256"] == baseline_precommit["code_state"]["ensemble_module"]["sha256"], "현재 C 선택 결합기 구현이 이슈 489 판정 때와 다르다.")
    _require(state["uv_lock_sha256"] == baseline_precommit["code_state"]["uv_lock_sha256"], "현재 실행 환경 잠금 파일이 이슈 489 판정 때와 다르다.")
    freeze = read_json(CANDIDATE_FREEZE_PATH)
    source_root = _source_root(freeze, args.source_root)
    fold_of, _ = _load_fold_and_labels(source_root)
    baseline, baseline_rows = _load_baseline_matrix(source_root, fold_of)
    candidate, candidate_record = _load_candidate(source_root, freeze, fold_of)
    _require(BASE_COLUMN in baseline.columns and CANDIDATE_COLUMN not in baseline.columns, "교체 전후 열 신원이 예상과 다르다.")
    replacement = baseline.rename(columns={BASE_COLUMN: CANDIDATE_COLUMN}).copy()
    replacement[CANDIDATE_COLUMN] = candidate
    replacement_rows = []
    for row in baseline_rows:
        if row["column"] == BASE_COLUMN:
            replacement_rows.append({"column": CANDIDATE_COLUMN, "origin": "own", "run_id": CANDIDATE_RUN_ID, "oof_sha256": candidate_record["prediction_sha256"]})
        else:
            replacement_rows.append(row)
    _require(list(replacement.columns) == [row["column"] for row in replacement_rows], "교체 뒤 구성원 순서가 동결 명세와 다르다.")
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / CACHE_NAME
    replacement.to_parquet(cache_path)
    baseline_comparison = read_json(BASELINE_COMPARISON_PATH)
    baseline_assembly = read_json(BASELINE_ASSEMBLY_PATH)
    reference = baseline_comparison["candidate"]
    _require(reference["strategy"] == ensemble.CSelectedShrunkRankLogitCombiner.name, "현재 두 번째 제출 기준이 C 선택 결합 절차가 아니다.")
    _require(baseline_assembly["judged"]["candidate_nested_auc"] == reference["nested_auc"], "현재 두 번째 제출 조립과 판정 기준의 nested AUC가 다르다.")
    store = MlflowRunStore(tracking_uri=f"sqlite:///{source_root / 'mlflow.db'}")
    current_run = store.facts_of(CURRENT_SUBMISSION_RUN_ID)
    _require(current_run.status == "FINISHED", "현재 두 번째 제출 실행이 완료 상태가 아니다.")
    _require(current_run.metrics.get("auc_oof") == reference["nested_auc"], "현재 두 번째 제출 실행의 nested OOF가 판정 기록과 다르다.")
    _require(store.artifact_sha256_of(CURRENT_SUBMISSION_RUN_ID, "assembly-manifest.json") == file_sha256(BASELINE_ASSEMBLY_PATH), "현재 두 번째 제출 실행의 조립 기록이 저장소 기록과 다르다.")
    _require(store.artifact_sha256_of(CURRENT_SUBMISSION_RUN_ID, "comparison.json") == file_sha256(BASELINE_COMPARISON_PATH), "현재 두 번째 제출 실행의 판정 기록이 저장소 기록과 다르다.")
    inputs = {
        "baseline_manifest": _tracked_input(BASELINE_MANIFEST_PATH),
        "baseline_precommit": _tracked_input(BASELINE_PRECOMMIT_PATH),
        "baseline_comparison": _tracked_input(BASELINE_COMPARISON_PATH),
        "baseline_assembly": _tracked_input(BASELINE_ASSEMBLY_PATH),
        "candidate_freeze": _tracked_input(CANDIDATE_FREEZE_PATH),
        "folds": _tracked_input(FOLDS_PATH),
        "train": {"path": str(source_root / "data/train.csv"), "sha256": file_sha256(source_root / "data/train.csv")},
        "uv_lock": _tracked_input(UV_LOCK_PATH),
    }
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "created_at": now_iso(),
        "question": "현재 두 번째 제출의 313개 구성과 C 선택 결합 절차를 고정하고 exp117 한 열만 exp208로 바꿨을 때 nested OOF와 중복 관계가 최종 교체 문턱을 통과하는가.",
        "source_root": str(source_root),
        "outer_folds": list(ALL_FOLDS),
        "inputs": inputs,
        "baseline": {
            "submission_run_id": CURRENT_SUBMISSION_RUN_ID,
            "submission_run_name": current_run.run_name,
            "submission_source_run_id": current_run.tags.get("source.run_id"),
            "member_count": len(baseline_rows),
            "own_member_count": OWN_MEMBER_COUNT,
            "members": baseline_rows,
            "composition_sha256": canonical_sha256([(row["column"], row["oof_sha256"]) for row in baseline_rows]),
            "strategy": reference["strategy"],
            "nested_auc": reference["nested_auc"],
            "fold_aucs": reference["fold_aucs"],
            "fold_selected_c": reference["fold_selected_c"],
            "fold_selected_lambda": reference["fold_selected_lambda"],
            "prediction_sha256": reference["prediction_sha256"],
        },
        "replacement": {
            "member_count": len(replacement_rows),
            "members": replacement_rows,
            "composition_sha256": canonical_sha256([(row["column"], row["oof_sha256"]) for row in replacement_rows]),
            "replaced": {"column": BASE_COLUMN, "run_id": BASE_RUN_ID},
            "candidate": {"column": CANDIDATE_COLUMN, "run_id": CANDIDATE_RUN_ID, **candidate_record},
            "strategy": ensemble.CSelectedShrunkRankLogitCombiner.name,
            "c_grid": list(C_GRID),
            "lambda_grid": list(LAMBDA_GRID),
            "max_iter": META_MAX_ITER,
        },
        "cache": {CACHE_NAME: file_sha256(cache_path)},
        "gate": {"delta_required": GATE_DELTA, "folds_required_positive": FOLDS_REQUIRED_POSITIVE, "duplicate_spearman_threshold": DUPLICATE_SPEARMAN, "public_score_used": False},
        "rules": {
            "fixed": "나머지 312개 구성원, 전체 순서, 바깥쪽 5분할, C와 수축 계수 격자, 동률 규칙과 결합기 구현을 현재 두 번째 제출과 같게 둔다.",
            "replacement": "exp117 열 하나만 exp208 세 시드 평균 OOF로 바꾼다.",
            "gate": "교체 팔 nested OOF에서 현재 두 번째 제출 nested OOF를 뺀 차이가 +0.00002 이상이고 바깥쪽 분할 5개가 모두 엄격히 양수일 때만 통과한다.",
            "duplicates": "중복 관계 변화는 바뀐 구성원이 공통 312개 구성원과 맺는 0.998 이상 스피어만 순위 상관 관계만 비교한다.",
            "scope": "읽기 전용 판정이며 후보 풀 장부, 제출 파일과 Kaggle 제출 상태를 바꾸지 않는다.",
        },
        "environment": _environment(),
        "code_state": state,
    }
    payload["precommit_sha256"] = canonical_sha256(payload)
    write_json(run_dir / "precommit.json", payload)
    print(f"precommit 저장: {run_dir / 'precommit.json'}")
    print(f"  기준 구성 {payload['baseline']['composition_sha256']}")
    print(f"  교체 구성 {payload['replacement']['composition_sha256']}")
    print(f"  캐시 {cache_path} {payload['cache'][CACHE_NAME]}")


def load_precommit(run_dir: Path) -> dict:
    _assert_repo_root()
    path = run_dir / "precommit.json"
    _require(path.is_file(), f"precommit.json이 없다: {path}")
    payload = read_json(path)
    expected = canonical_sha256({key: value for key, value in payload.items() if key != "precommit_sha256"})
    _require(expected == payload["precommit_sha256"], "precommit.json이 제자리에서 바뀌었다.")
    for key, entry in payload["inputs"].items():
        _require(file_sha256(Path(entry["path"])) == entry["sha256"], f"입력 {key}의 해시가 precommit과 다르다.")
    cache_path = run_dir / "cache" / CACHE_NAME
    _require(file_sha256(cache_path) == payload["cache"][CACHE_NAME], "교체 OOF 캐시 해시가 precommit과 다르다.")
    state = _code_state()
    for label, actual, frozen in (
        ("git commit", state["git"]["commit"], payload["code_state"]["git"]["commit"]),
        ("판정 도구", state["script"]["sha256"], payload["code_state"]["script"]["sha256"]),
        ("결합기", state["ensemble_module"]["sha256"], payload["code_state"]["ensemble_module"]["sha256"]),
        ("실행 환경 잠금", state["uv_lock_sha256"], payload["code_state"]["uv_lock_sha256"]),
    ):
        _require(actual == frozen, f"코드 상태({label})가 precommit과 다르다.")
    _require(not state["git"]["dirty"], "판정 실행 중 작업 트리가 바뀌었다.")
    return payload


def _load_matrix(run_dir: Path, payload: dict, fold_of: pd.Series) -> pd.DataFrame:
    matrix = pd.read_parquet(run_dir / "cache" / CACHE_NAME).astype(np.float64)
    expected_columns = [row["column"] for row in payload["replacement"]["members"]]
    _require(list(matrix.columns) == expected_columns, "교체 OOF 캐시 열 순서가 precommit과 다르다.")
    _require(matrix.index.equals(fold_of.index), "교체 OOF 캐시 행 순서가 고정 분할과 다르다.")
    _require(matrix.shape == (len(fold_of), MEMBER_COUNT) and bool(np.isfinite(matrix.to_numpy()).all()), "교체 OOF 캐시 형태나 유한값 검사가 실패했다.")
    return matrix


def _save_fold(out_dir: Path, fold_of: pd.Series, outer: np.ndarray, prediction: np.ndarray) -> str:
    _require(prediction.shape == (int(outer.sum()),) and bool(np.isfinite(prediction).all()), "봉인 분할 예측 형태나 유한값 검사가 실패했다.")
    frame = pd.DataFrame({ID: fold_of.index.to_numpy()[outer], "prediction": prediction})
    path = out_dir / "predictions.parquet"
    frame.to_parquet(path, index=False)
    return prediction_array_sha256(prediction)


def fold_job(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    fold = int(args.fold)
    _require(fold in ALL_FOLDS, f"알 수 없는 분할 {fold}")
    out_dir = run_dir / "replacement" / f"fold-{fold}"
    _require(not (out_dir / "replacement.json").exists(), f"이미 완료된 분할이다: {fold}")
    out_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(payload["source_root"])
    fold_of, y = _load_fold_and_labels(source_root)
    matrix = _load_matrix(run_dir, payload, fold_of)
    inner = (fold_of != fold).to_numpy()
    outer = (fold_of == fold).to_numpy()
    started = time.monotonic()
    combiner = ensemble.CSelectedShrunkRankLogitCombiner(fold_of=fold_of, c_grid=C_GRID, lambda_grid=LAMBDA_GRID, max_iter=META_MAX_ITER)
    try:
        fitted = combiner.fit(matrix[inner], y[inner])
    except ensemble.CombinerConvergenceError as exc:
        raise JudgmentError(f"교체 팔 분할 {fold}이 수렴하지 않았다: {exc}") from exc
    prediction = np.asarray(fitted.predict(matrix[outer]), dtype=np.float64)
    digest = _save_fold(out_dir, fold_of, outer, prediction)
    auc = float(roc_auc_score(y[outer].to_numpy(), prediction))
    record = {
        "schema": SCHEMA,
        "precommit_sha256": payload["precommit_sha256"],
        "sealed_fold": fold,
        "strategy": ensemble.CSelectedShrunkRankLogitCombiner.name,
        "member_count": MEMBER_COUNT,
        "rows": int(outer.sum()),
        "auc": auc,
        "selected_c": fitted.c,
        "selected_lambda": fitted.shrinkage_lambda,
        "selected_inner_auc": fitted.selection_aucs[(fitted.c, fitted.shrinkage_lambda)],
        "selection_aucs": [{"c": c, "lambda": shrinkage, "auc": value} for (c, shrinkage), value in fitted.selection_aucs.items()],
        "inner_fits": [fit.__dict__ for fit in fitted.inner_fits],
        "final_iterations": fitted.final_iterations,
        "final_coefficient_l2_norm": fitted.final_coefficient_l2_norm,
        "prediction_sha256": digest,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "environment": _environment(),
        "finished_at": now_iso(),
    }
    write_json(out_dir / "replacement.json", record)
    print(f"분할 {fold}: AUC {auc:.15f}, C={fitted.c}, λ={fitted.shrinkage_lambda}, {record['elapsed_seconds']:.0f}초", flush=True)


def _job_done(run_dir: Path, fold: int) -> bool:
    return (run_dir / "replacement" / f"fold-{fold}" / "replacement.json").is_file()


def _running_folds(run_dir: Path) -> set[int]:
    listing = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True, check=False).stdout
    pattern = re.compile(rf"judge_issue508_extended_stack_replacement\.py fold --run-dir {re.escape(str(run_dir))} --fold (\d+)")
    return {int(fold) for fold in pattern.findall(listing)}


def _reference_check(run_dir: Path, payload: dict) -> None:
    baseline_precommit = read_json(BASELINE_PRECOMMIT_PATH)
    comparison = read_json(BASELINE_COMPARISON_PATH)
    assembly = read_json(BASELINE_ASSEMBLY_PATH)
    checks = {
        "ensemble_code_matches_issue489": payload["code_state"]["ensemble_module"]["sha256"] == baseline_precommit["code_state"]["ensemble_module"]["sha256"],
        "runtime_lock_matches_issue489": payload["code_state"]["uv_lock_sha256"] == baseline_precommit["code_state"]["uv_lock_sha256"],
        "composition_matches_issue489": payload["baseline"]["composition_sha256"] == baseline_precommit["members"]["composition_sha256"],
        "strategy_matches_current_submission": payload["baseline"]["strategy"] == comparison["candidate"]["strategy"],
        "nested_reference_matches_current_submission": payload["baseline"]["nested_auc"] == assembly["judged"]["candidate_nested_auc"],
        "current_submission_run_matches": payload["baseline"]["submission_run_id"] == CURRENT_SUBMISSION_RUN_ID,
    }
    _require(all(checks.values()), f"현재 두 번째 제출 기준 호환성 검사가 실패했다: {checks}")
    out_dir = run_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "reference.json", {"schema": SCHEMA, "precommit_sha256": payload["precommit_sha256"], "checks": checks, "passes": True, "reference": payload["baseline"], "checked_at": now_iso()})


def run_jobs(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    _require(1 <= args.workers <= MAX_WORKERS, f"동시 실행은 최대 {MAX_WORKERS}개다.")
    _reference_check(run_dir, payload)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = [fold for fold in ALL_FOLDS if not _job_done(run_dir, fold)]
    active: dict[int, tuple[subprocess.Popen, object]] = {}
    failed: dict[int, int] = {}
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = str(args.threads)
    print(f"남은 분할 {len(pending)}/5, 동시 상한 {args.workers}, 작업당 계산 줄기 {args.threads}", flush=True)
    while pending or active:
        for fold, (process, handle) in list(active.items()):
            code = process.poll()
            if code is not None:
                handle.close()
                del active[fold]
                if code != 0:
                    failed[fold] = code
                print(f"분할 {fold} {'완료' if code == 0 else f'실패({code})'} {now_iso()}", flush=True)
        running = _running_folds(run_dir) | set(active)
        while pending and len(running) < args.workers:
            headroom = psutil.virtual_memory().available / psutil.virtual_memory().total
            if headroom < MEMORY_HEADROOM_MIN:
                print(f"메모리 여유율 {headroom:.1%}가 하한 {MEMORY_HEADROOM_MIN:.0%}보다 낮아 새 작업을 기다린다.", flush=True)
                break
            fold = pending.pop(0)
            if fold in running or _job_done(run_dir, fold):
                continue
            handle = (log_dir / f"fold-{fold}.log").open("w")
            command = [sys.executable, __file__, "fold", "--run-dir", str(run_dir), "--fold", str(fold)]
            active[fold] = (subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT), handle)
            running.add(fold)
            print(f"분할 {fold} 시작 {now_iso()} (메모리 여유율 {headroom:.1%})", flush=True)
        if pending or active:
            time.sleep(10)
    _require(not failed, f"실패한 분할이 있다: {failed}. 로그는 {log_dir}에 있다.")
    print("다섯 분할 실행 완료", flush=True)


def _load_fold_results(run_dir: Path, payload: dict, fold_of: pd.Series, y: pd.Series) -> tuple[dict[str, dict], pd.Series]:
    records: dict[str, dict] = {}
    nested = pd.Series(np.nan, index=fold_of.index, dtype=np.float64)
    for fold in ALL_FOLDS:
        out_dir = run_dir / "replacement" / f"fold-{fold}"
        record = read_json(out_dir / "replacement.json")
        _require(record["precommit_sha256"] == payload["precommit_sha256"], f"분할 {fold}이 다른 precommit에서 나왔다.")
        part = pd.read_parquet(out_dir / "predictions.parquet").set_index(ID)["prediction"]
        ids = fold_of.index[(fold_of == fold).to_numpy()]
        _require(part.index.equals(pd.Index(ids)), f"분할 {fold} 예측 id가 고정 분할과 다르다.")
        _require(prediction_array_sha256(part.to_numpy()) == record["prediction_sha256"], f"분할 {fold} 예측 해시가 기록과 다르다.")
        _require(float(roc_auc_score(y.loc[ids].to_numpy(), part.to_numpy())) == record["auc"], f"분할 {fold} AUC 재계산이 기록과 다르다.")
        nested.loc[ids] = part.to_numpy()
        records[str(fold)] = record
    _require(nested.notna().all(), "이어붙인 교체 팔 예측에 빈 행이 있다.")
    return records, nested


def _spearman_against(values: np.ndarray, matrix: pd.DataFrame) -> list[dict[str, object]]:
    x = pd.Series(values).rank(method="average").to_numpy(np.float64, copy=True)
    x -= x.mean()
    x_norm = float(np.linalg.norm(x))
    rows: list[dict[str, object]] = []
    for column in matrix.columns:
        y = matrix[column].rank(method="average").to_numpy(np.float64, copy=True)
        y -= y.mean()
        correlation = float(np.dot(x, y) / (x_norm * np.linalg.norm(y)))
        rows.append({"column": column, "spearman": correlation})
    rows.sort(key=lambda row: (-row["spearman"], row["column"]))
    return rows


def _duplicate_diagnostics(run_dir: Path, payload: dict, fold_of: pd.Series) -> dict[str, object]:
    source_root = Path(payload["source_root"])
    matrix = _load_matrix(run_dir, payload, fold_of)
    candidate = matrix.pop(CANDIDATE_COLUMN).to_numpy(np.float64)
    store = MlflowRunStore(tracking_uri=f"sqlite:///{source_root / 'mlflow.db'}")
    baseline = store.oof_of(BASE_RUN_ID).reindex(fold_of.index)
    _require(baseline.notna().all(), "교체 전 exp117 OOF id가 고정 분할과 다르다.")
    baseline_values = baseline.to_numpy(np.float64)
    baseline_expected = next(row["oof_sha256"] for row in payload["baseline"]["members"] if row["column"] == BASE_COLUMN)
    _require(prediction_array_sha256(baseline_values) == baseline_expected, "교체 전 exp117 OOF 해시가 precommit과 다르다.")
    before = _spearman_against(baseline_values, matrix)
    after = _spearman_against(candidate, matrix)
    threshold = payload["gate"]["duplicate_spearman_threshold"]
    before_duplicates = [row for row in before if row["spearman"] >= threshold]
    after_duplicates = [row for row in after if row["spearman"] >= threshold]
    pair = float(pd.Series(baseline_values).corr(pd.Series(candidate), method="spearman"))
    before_names = {row["column"] for row in before_duplicates}
    after_names = {row["column"] for row in after_duplicates}
    return {
        "threshold": threshold,
        "changed_edges_only": True,
        "unchanged_member_pairs_inherited": int(len(matrix.columns) * (len(matrix.columns) - 1) // 2),
        "replacement_pair": {"before": BASE_COLUMN, "after": CANDIDATE_COLUMN, "spearman": pair, "removed_from_final_composition": True},
        "before": {"member": BASE_COLUMN, "nearest_remaining": before[0], "duplicates": before_duplicates, "all_correlations": before},
        "after": {"member": CANDIDATE_COLUMN, "nearest_remaining": after[0], "duplicates": after_duplicates, "all_correlations": after},
        "change": {
            "duplicate_count_delta": len(after_duplicates) - len(before_duplicates),
            "added_duplicate_columns": sorted(after_names - before_names),
            "removed_duplicate_columns": sorted(before_names - after_names),
            "retained_duplicate_columns": sorted(before_names & after_names),
            "nearest_remaining_delta": after[0]["spearman"] - before[0]["spearman"],
        },
    }


def compare(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    reference = read_json(run_dir / "baseline" / "reference.json")
    _require(reference["passes"] and reference["precommit_sha256"] == payload["precommit_sha256"], "현재 제출 기준 호환성 검사가 없다.")
    source_root = Path(payload["source_root"])
    fold_of, y = _load_fold_and_labels(source_root)
    records, nested = _load_fold_results(run_dir, payload, fold_of, y)
    replacement_auc = float(roc_auc_score(y.to_numpy(), nested.to_numpy()))
    baseline_auc = payload["baseline"]["nested_auc"]
    fold_deltas = {key: records[key]["auc"] - payload["baseline"]["fold_aucs"][key] for key in records}
    positive = sum(delta > 0.0 for delta in fold_deltas.values())
    delta = replacement_auc - baseline_auc
    gate_delta = delta >= payload["gate"]["delta_required"]
    gate_folds = positive >= payload["gate"]["folds_required_positive"]
    passes = bool(gate_delta and gate_folds)
    duplicates = _duplicate_diagnostics(run_dir, payload, fold_of)
    per_fold = []
    for key, record in records.items():
        per_fold.append({
            "fold": int(key),
            "baseline_auc": payload["baseline"]["fold_aucs"][key],
            "replacement_auc": record["auc"],
            "delta": fold_deltas[key],
            "baseline_selected_c": payload["baseline"]["fold_selected_c"][key],
            "replacement_selected_c": record["selected_c"],
            "baseline_selected_lambda": payload["baseline"]["fold_selected_lambda"][key],
            "replacement_selected_lambda": record["selected_lambda"],
            "replacement_selected_inner_auc": record["selected_inner_auc"],
            "replacement_final_iterations": record["final_iterations"],
            "replacement_final_coefficient_l2_norm": record["final_coefficient_l2_norm"],
            "elapsed_seconds": record["elapsed_seconds"],
            "peak_rss_bytes": record["peak_rss_bytes"],
        })
    verdict = "통과: exp208 원자 교체가 최종 두 번째 제출 교체 후보가 된다." if passes else "미달: 현재 두 번째 제출을 유지한다."
    record = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "precommit_sha256": payload["precommit_sha256"],
        "baseline_reference_check": reference["checks"],
        "baseline": {"submission_run_id": CURRENT_SUBMISSION_RUN_ID, "strategy": payload["baseline"]["strategy"], "nested_auc": baseline_auc, "fold_aucs": payload["baseline"]["fold_aucs"], "prediction_sha256": payload["baseline"]["prediction_sha256"]},
        "replacement": {"strategy": payload["replacement"]["strategy"], "nested_auc": replacement_auc, "fold_aucs": {key: record["auc"] for key, record in records.items()}, "fold_selected_c": {key: record["selected_c"] for key, record in records.items()}, "fold_selected_lambda": {key: record["selected_lambda"] for key, record in records.items()}, "prediction_sha256": prediction_array_sha256(nested.to_numpy())},
        "delta_vs_current_submission": delta,
        "delta_minus_gate": delta - payload["gate"]["delta_required"],
        "fold_deltas": fold_deltas,
        "folds_positive": positive,
        "gate_delta_passes": bool(gate_delta),
        "gate_folds_passes": bool(gate_folds),
        "passes_gate": passes,
        "verdict": verdict,
        "duplicates": duplicates,
        "per_fold": per_fold,
        "rows_scored": len(y),
        "elapsed_seconds_total": sum(row["elapsed_seconds"] for row in per_fold),
        "peak_rss_bytes_max": max(row["peak_rss_bytes"] for row in per_fold),
        "public_score_used": False,
        "submission_assembled": False,
        "compared_at": now_iso(),
    }
    write_json(run_dir / "comparison.json", record)
    print(f"현재 제출 nested {baseline_auc:.10f}, 교체 팔 {replacement_auc:.10f}, 차이 {delta:+.7f}, 문턱 여유 {record['delta_minus_gate']:+.7f}")
    print(f"바깥쪽 분할 양수 {positive}/5")
    print(f"중복 수 {len(duplicates['before']['duplicates'])} -> {len(duplicates['after']['duplicates'])}, 최근접 {duplicates['before']['nearest_remaining']['spearman']:.9f} -> {duplicates['after']['nearest_remaining']['spearman']:.9f}")
    print(f"판정: {verdict}")


def _manifest_files(run_dir: Path) -> list[Path]:
    paths = [run_dir / "precommit.json", run_dir / "baseline/reference.json", run_dir / "comparison.json", run_dir / "report.md"]
    paths.extend(run_dir / "replacement" / f"fold-{fold}" / "replacement.json" for fold in ALL_FOLDS)
    return [path for path in paths if path.is_file()]


def _gb(value: int) -> str:
    return f"{value / 2**30:.1f}GB"


def report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    comparison = read_json(run_dir / "comparison.json")
    _require(comparison["precommit_sha256"] == payload["precommit_sha256"], "comparison.json이 다른 precommit에서 나왔다.")
    duplicate = comparison["duplicates"]
    lines = [f"# 현재 313개 결합 exp117 원자 교체 판정 보고 (이슈 {ISSUE})", ""]
    lines += ["## 판정", ""]
    lines += [f"- 결과: **{comparison['verdict']}**"]
    lines += [f"- 교체 팔 nested OOF `{comparison['replacement']['nested_auc']:.10f}` - 현재 두 번째 제출 `{comparison['baseline']['nested_auc']:.10f}` = `{comparison['delta_vs_current_submission']:+.7f}`이며 문턱은 `+{GATE_DELTA:.5f}`다."]
    lines += [f"- 전체 차이 문턱은 {'충족' if comparison['gate_delta_passes'] else '미달'}했고 바깥쪽 분할 양수 조건은 `{comparison['folds_positive']}/5`로 {'충족' if comparison['gate_folds_passes'] else '미달'}했다."]
    lines += ["- Public 점수는 쓰지 않았고 후보 풀 장부, 제출 파일과 Kaggle 제출 상태를 바꾸지 않았다.", ""]
    lines += ["## 분할별 결과", "", "| 분할 | 현재 제출 AUC | 교체 팔 AUC | 차이 | 현재 C | 교체 C | 현재 수축 계수 | 교체 수축 계수 |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison["per_fold"]:
        lines.append(f"| {row['fold']} | {row['baseline_auc']:.10f} | {row['replacement_auc']:.10f} | {row['delta']:+.10f} | {row['baseline_selected_c']} | {row['replacement_selected_c']} | {row['baseline_selected_lambda']} | {row['replacement_selected_lambda']} |")
    lines += ["", "## 중복 관계 변화", ""]
    lines += [f"- 교체 대상 두 예측의 스피어만 순위 상관은 `{duplicate['replacement_pair']['spearman']:.12f}`이며 두 예측은 최종 구성에 함께 남지 않는다."]
    lines += [f"- 공통으로 남는 312개 구성원에 대한 0.998 이상 중복 관계는 `{len(duplicate['before']['duplicates'])}`건에서 `{len(duplicate['after']['duplicates'])}`건으로 바뀌었다."]
    lines += [f"- 교체 전 최근접 구성원은 `{duplicate['before']['nearest_remaining']['column']}` (`{duplicate['before']['nearest_remaining']['spearman']:.12f}`), 교체 후 최근접 구성원은 `{duplicate['after']['nearest_remaining']['column']}` (`{duplicate['after']['nearest_remaining']['spearman']:.12f}`)이다."]
    lines += [f"- 새 중복은 `{', '.join(duplicate['change']['added_duplicate_columns']) or '없음'}`, 사라진 중복은 `{', '.join(duplicate['change']['removed_duplicate_columns']) or '없음'}`, 유지된 중복은 `{', '.join(duplicate['change']['retained_duplicate_columns']) or '없음'}`이다."]
    lines += [f"- 바뀌지 않은 구성원끼리의 `{duplicate['unchanged_member_pairs_inherited']}`개 관계는 두 팔에서 입력이 같으므로 그대로 이어받았다.", ""]
    lines += ["## 동결과 재현성", ""]
    lines += [f"- 현재 제출 구성 해시 `{payload['baseline']['composition_sha256']}`, 교체 구성 해시 `{payload['replacement']['composition_sha256']}`다."]
    lines += [f"- 기준 판정 때와 C 선택 결합기 코드와 실행 환경 잠금 해시가 같고, 현재 313개 구성 해시와 현재 제출 실행 `{CURRENT_SUBMISSION_RUN_ID}`도 일치했다."]
    lines += [f"- precommit `{payload['precommit_sha256']}`, 교체 팔 nested 예측 `{comparison['replacement']['prediction_sha256']}`다."]
    env = payload["environment"]
    lines += [f"- 실행 환경은 {env['platform']} ({env['machine']}), CPU {env['cpu_count']}개, 메모리 {_gb(env['memory_total_bytes'])}, Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, scikit-learn {env['sklearn']}다."]
    lines += [f"- 분할 작업 경과 시간 합계는 {comparison['elapsed_seconds_total'] / 60:.1f}분이고 작업 최대 메모리는 {_gb(comparison['peak_rss_bytes_max'])}다.", ""]
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = [f"{file_sha256(path)}  {path.relative_to(run_dir)}" for path in _manifest_files(run_dir)]
    (run_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"보고 저장: {report_path}")


def publish(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    comparison = read_json(run_dir / "comparison.json")
    _require(comparison["precommit_sha256"] == payload["precommit_sha256"], "comparison.json이 다른 precommit에서 나왔다.")
    publish_dir = Path(args.publish_dir)
    _require(not publish_dir.exists(), f"게시 폴더가 이미 있다: {publish_dir}")
    files = _manifest_files(run_dir)
    _require(len(files) == 9, f"게시할 판정 파일 수가 9개가 아니다: {files}")
    for source in files:
        relative = source.relative_to(run_dir)
        destination = publish_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest = [f"{file_sha256(path)}  {path.relative_to(publish_dir)}" for path in sorted(publish_dir.rglob("*")) if path.is_file()]
    (publish_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"판정 근거 게시: {publish_dir} ({len(manifest)}개 파일)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("precommit")
    pre.add_argument("--run-dir", type=Path, default=OUT_DIR)
    pre.add_argument("--source-root", type=Path, default=None)
    pre.set_defaults(handler=precommit)
    run = sub.add_parser("run")
    run.add_argument("--run-dir", type=Path, default=OUT_DIR)
    run.add_argument("--workers", type=int, default=MAX_WORKERS)
    run.add_argument("--threads", type=int, default=4)
    run.set_defaults(handler=run_jobs)
    fold = sub.add_parser("fold")
    fold.add_argument("--run-dir", type=Path, default=OUT_DIR)
    fold.add_argument("--fold", type=int, required=True)
    fold.set_defaults(handler=fold_job)
    for name, handler in (("compare", compare), ("report", report), ("publish", publish)):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", type=Path, default=OUT_DIR)
        if name == "publish":
            command.add_argument("--publish-dir", type=Path, default=PUBLISH_DIR)
        command.set_defaults(handler=handler)
    args = parser.parse_args()
    try:
        args.handler(args)
    except JudgmentError as exc:
        sys.exit(f"판정 불가: {exc}")


if __name__ == "__main__":
    main()
