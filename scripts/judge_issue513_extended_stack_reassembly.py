"""공식 자체 풀로 현재 확장 스택을 재조립해 교체 문턱을 판정한다. (#513)

현재 두 번째 제출의 313개 구성과 C 선택 결합 절차를 기준 팔로 고정한다.
재조립 팔은 공식 자체 풀 36개를 장부 순서대로 두고, 기준 팔의 외부 구성원
278개를 기존 순서 그대로 붙인 314개 구성이다.

판정 문턱은 현재 두 번째 제출 대비 nested OOF ``+0.00002`` 이상과 바깥쪽
분할 ``5/5`` 양수다.
산출물은 ``run-logs/extended-stack-pool-reassembly/issue513``에 먼저 만들고,
``publish``가 예측 배열과 캐시를 제외한 판정 근거를
``docs/research/extended-stack-pool-reassembly/issue513``에 복사한다.

실행 순서:

    uv run python scripts/judge_issue513_extended_stack_reassembly.py precommit
    uv run python scripts/judge_issue513_extended_stack_reassembly.py run --workers 3 --threads 4
    uv run python scripts/judge_issue513_extended_stack_reassembly.py compare
    uv run python scripts/judge_issue513_extended_stack_reassembly.py report
    uv run python scripts/judge_issue513_extended_stack_reassembly.py publish
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import sklearn
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge_issue508_extended_stack_replacement as base
import judge_logistic_c_selection as logistic

from pipeline import ensemble
from pipeline.data import ID, file_sha256
from pipeline.pool_audit import prediction_array_sha256
from pipeline.runs import MlflowRunStore


ISSUE = 513
SCHEMA = "extended-stack-pool-reassembly/1"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path("run-logs/extended-stack-pool-reassembly/issue513")
PUBLISH_DIR = Path("docs/research/extended-stack-pool-reassembly/issue513")
POOL_PATH = Path("artifacts/pool.yaml")
ISSUE512_JUDGMENT_PATH = Path(
    "docs/research/missingness-propagation-batch/issue512/judgment.json"
)
BASELINE_MANIFEST_PATH = base.BASELINE_MANIFEST_PATH
BASELINE_PRECOMMIT_PATH = base.BASELINE_PRECOMMIT_PATH
BASELINE_COMPARISON_PATH = base.BASELINE_COMPARISON_PATH
BASELINE_ASSEMBLY_PATH = base.BASELINE_ASSEMBLY_PATH
FOLDS_PATH = base.FOLDS_PATH
UV_LOCK_PATH = base.UV_LOCK_PATH
CURRENT_SUBMISSION_RUN_ID = base.CURRENT_SUBMISSION_RUN_ID
BASELINE_MEMBER_COUNT = 313
BASELINE_OWN_MEMBER_COUNT = 35
EXTERNAL_MEMBER_COUNT = 278
OFFICIAL_OWN_MEMBER_COUNT = 36
REASSEMBLED_MEMBER_COUNT = 314
ALL_FOLDS = base.ALL_FOLDS
GATE_DELTA = 0.00002
FOLDS_REQUIRED_POSITIVE = 5
MAX_WORKERS = 3
MEMORY_HEADROOM_MIN = 0.15
META_MAX_ITER = base.META_MAX_ITER
C_GRID = base.C_GRID
LAMBDA_GRID = base.LAMBDA_GRID
CACHE_NAME = "reassembled-oof-314.parquet"
EXPECTED_POOL_FILE_SHA256 = (
    "40947563a00cab8212498c7e339517e387979b14c6477c6ce8e196036e02044c"
)
EXPECTED_REMOVED_OWN = {
    "exp035_lattice_te",
    "exp058_logreg_onehot",
    "exp070_cat_exact_cats",
    "exp110_lgb_kitopl_no_te",
    "exp117_ag25_gbm_r21",
    "exp131_lookup_bivariate_plr5",
}
EXPECTED_ADDED_OWN = {
    "mpv1_exp035_lattice_te_missingness_augmented",
    "mpv1_exp058_logreg_onehot_missingness_augmented",
    "mpv1_exp070_cat_exact_cats_missingness_augmented",
    "mpv1_exp110_lgb_kitopl_no_te_missingness_augmented",
    "mpv1_exp131_lookup_bivariate_plr5_missingness_augmented",
    "exp208_issue500_ag25_missingness_augmented",
    "exp209_issue505_lgb_lr_onehot_init",
}

read_json = logistic.read_json
write_json = logistic.write_json
canonical_sha256 = logistic.canonical_sha256
now_iso = logistic.now_iso
JudgmentError = logistic.JudgmentError
_require = logistic._require


def _assert_repo_root() -> None:
    _require(
        Path.cwd().resolve() == REPO_ROOT,
        f"저장소 루트에서 실행해야 한다: {REPO_ROOT}",
    )


def _code_state() -> dict[str, object]:
    return {
        "git": logistic.strict.git_state(),
        "script": {
            "path": str(Path(__file__).relative_to(REPO_ROOT)),
            "sha256": file_sha256(Path(__file__)),
        },
        "ensemble_module": {
            "path": str(logistic.strict.ENSEMBLE_SOURCE),
            "sha256": file_sha256(logistic.strict.ENSEMBLE_SOURCE),
        },
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
        "threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def _tracked_input(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _official_pool() -> tuple[list[dict[str, str]], dict]:
    document = yaml.safe_load(POOL_PATH.read_text(encoding="utf-8"))
    members = [
        {"column": str(row["config"]), "run_id": str(row["run_id"]), "origin": "own"}
        for row in document["members"]
    ]
    _require(
        len(members) == OFFICIAL_OWN_MEMBER_COUNT,
        "공식 자체 풀이 36개가 아니다.",
    )
    _require(
        len({row["column"] for row in members}) == len(members),
        "공식 자체 풀 구성원 이름이 고유하지 않다.",
    )
    _require(
        len({row["run_id"] for row in members}) == len(members),
        "공식 자체 풀 실행 식별자가 고유하지 않다.",
    )
    judgment = read_json(ISSUE512_JUDGMENT_PATH)
    _require(
        judgment["official_ledgers"]["candidate_pool"]["after_sha256"]
        == EXPECTED_POOL_FILE_SHA256
        == file_sha256(POOL_PATH),
        "공식 자체 풀 파일이 이슈 512 종결 기록과 다르다.",
    )
    _require(
        [row["column"] for row in members] == judgment["verdict"]["proposal_pool"],
        "공식 자체 풀 순서가 이슈 512 제안 풀과 다르다.",
    )
    return members, judgment


def _load_official_own_matrix(
    store: MlflowRunStore,
    fold_of: pd.Series,
    members: list[dict[str, str]],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    columns: dict[str, np.ndarray] = {}
    rows: list[dict[str, str]] = []
    for member in members:
        facts = store.facts_of(member["run_id"])
        _require(
            facts.status == "FINISHED",
            f"{member['column']}: 공식 자체 풀 실행이 완료 상태가 아니다.",
        )
        values = store.oof_of(member["run_id"]).reindex(fold_of.index)
        _require(
            values.notna().all(),
            f"{member['column']}: 실행 OOF id가 고정 분할과 맞지 않는다.",
        )
        array = values.to_numpy(np.float64)
        _require(
            array.shape == (len(fold_of),) and bool(np.isfinite(array).all()),
            f"{member['column']}: OOF 형태나 유한값 검사가 실패했다.",
        )
        digest = prediction_array_sha256(array)
        columns[member["column"]] = array
        rows.append({**member, "oof_sha256": digest})
    return pd.DataFrame(columns, index=fold_of.index).astype(np.float64), rows


def precommit(args: argparse.Namespace) -> None:
    _assert_repo_root()
    run_dir = Path(args.run_dir)
    _require(
        not (run_dir / "precommit.json").exists(),
        f"precommit.json이 이미 있다: {run_dir}",
    )
    state = _code_state()
    _require(not state["git"]["dirty"], "판정은 커밋된 코드 상태에서만 시작한다.")
    baseline_precommit = read_json(BASELINE_PRECOMMIT_PATH)
    _require(
        state["ensemble_module"]["sha256"]
        == baseline_precommit["code_state"]["ensemble_module"]["sha256"],
        "현재 C 선택 결합기 구현이 이슈 489 판정 때와 다르다.",
    )
    _require(
        state["uv_lock_sha256"]
        == baseline_precommit["code_state"]["uv_lock_sha256"],
        "현재 실행 환경 잠금 파일이 이슈 489 판정 때와 다르다.",
    )
    source_root = base._source_root(
        read_json(base.CANDIDATE_FREEZE_PATH), args.source_root
    )
    fold_of, _ = base._load_fold_and_labels(source_root)
    baseline, baseline_rows = base._load_baseline_matrix(source_root, fold_of)
    _require(
        len(baseline_rows) == BASELINE_MEMBER_COUNT,
        "현재 두 번째 제출 기준 팔이 313개가 아니다.",
    )
    baseline_own = [row for row in baseline_rows if row["origin"] == "own"]
    external_rows = [row for row in baseline_rows if row["origin"] == "external"]
    _require(
        len(baseline_own) == BASELINE_OWN_MEMBER_COUNT
        and len(external_rows) == EXTERNAL_MEMBER_COUNT,
        "현재 두 번째 제출의 자체·외부 구성원 수가 35·278이 아니다.",
    )
    official_members, issue512 = _official_pool()
    baseline_own_names = {row["column"] for row in baseline_own}
    official_names = {row["column"] for row in official_members}
    _require(
        baseline_own_names - official_names == EXPECTED_REMOVED_OWN,
        "기준 팔에서 빠지는 자체 구성원이 사전 계약과 다르다.",
    )
    _require(
        official_names - baseline_own_names == EXPECTED_ADDED_OWN,
        "재조립 팔에 들어오는 자체 구성원이 사전 계약과 다르다.",
    )
    store = MlflowRunStore(tracking_uri=f"sqlite:///{source_root / 'mlflow.db'}")
    own_matrix, own_rows = _load_official_own_matrix(
        store, fold_of, official_members
    )
    external_columns = [row["column"] for row in external_rows]
    external_matrix = baseline[external_columns]
    reassembled = pd.concat([own_matrix, external_matrix], axis=1)
    reassembled_rows = [*own_rows, *external_rows]
    _require(
        list(reassembled.columns) == [row["column"] for row in reassembled_rows],
        "재조립 팔 열 순서가 동결 명세와 다르다.",
    )
    _require(
        reassembled.shape == (len(fold_of), REASSEMBLED_MEMBER_COUNT),
        "재조립 팔이 공식 자체 36개와 외부 278개로 이루어지지 않았다.",
    )
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / CACHE_NAME
    reassembled.to_parquet(cache_path)
    baseline_comparison = read_json(BASELINE_COMPARISON_PATH)
    baseline_assembly = read_json(BASELINE_ASSEMBLY_PATH)
    reference = baseline_comparison["candidate"]
    _require(
        reference["strategy"] == ensemble.CSelectedShrunkRankLogitCombiner.name,
        "현재 두 번째 제출 기준이 C 선택 결합 절차가 아니다.",
    )
    _require(
        baseline_assembly["judged"]["candidate_nested_auc"]
        == reference["nested_auc"],
        "현재 두 번째 제출 조립과 판정 기준의 nested AUC가 다르다.",
    )
    current_run = store.facts_of(CURRENT_SUBMISSION_RUN_ID)
    _require(
        current_run.status == "FINISHED",
        "현재 두 번째 제출 실행이 완료 상태가 아니다.",
    )
    _require(
        current_run.metrics.get("auc_oof") == reference["nested_auc"],
        "현재 두 번째 제출 실행의 nested OOF가 판정 기록과 다르다.",
    )
    inputs = {
        "baseline_manifest": _tracked_input(BASELINE_MANIFEST_PATH),
        "baseline_precommit": _tracked_input(BASELINE_PRECOMMIT_PATH),
        "baseline_comparison": _tracked_input(BASELINE_COMPARISON_PATH),
        "baseline_assembly": _tracked_input(BASELINE_ASSEMBLY_PATH),
        "official_pool": _tracked_input(POOL_PATH),
        "issue512_judgment": _tracked_input(ISSUE512_JUDGMENT_PATH),
        "folds": _tracked_input(FOLDS_PATH),
        "train": {
            "path": str(source_root / "data/train.csv"),
            "sha256": file_sha256(source_root / "data/train.csv"),
        },
        "uv_lock": _tracked_input(UV_LOCK_PATH),
    }
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "created_at": now_iso(),
        "question": (
            "공식 자체 풀 36개와 현재 외부 구성원 278개를 재조립한 314개 팔이 "
            "현재 두 번째 제출의 313개 기준 팔보다 nested OOF +0.00002 이상 높고 "
            "바깥쪽 분할 5/5가 양수인가."
        ),
        "source_root": str(source_root),
        "outer_folds": list(ALL_FOLDS),
        "inputs": inputs,
        "baseline": {
            "submission_run_id": CURRENT_SUBMISSION_RUN_ID,
            "submission_run_name": current_run.run_name,
            "submission_source_run_id": current_run.tags.get("source.run_id"),
            "member_count": len(baseline_rows),
            "own_member_count": len(baseline_own),
            "external_member_count": len(external_rows),
            "members": baseline_rows,
            "composition_sha256": canonical_sha256(
                [(row["column"], row["oof_sha256"]) for row in baseline_rows]
            ),
            "strategy": reference["strategy"],
            "nested_auc": reference["nested_auc"],
            "fold_aucs": reference["fold_aucs"],
            "fold_selected_c": reference["fold_selected_c"],
            "fold_selected_lambda": reference["fold_selected_lambda"],
            "prediction_sha256": reference["prediction_sha256"],
        },
        "reassembled": {
            "member_count": len(reassembled_rows),
            "own_member_count": len(own_rows),
            "external_member_count": len(external_rows),
            "members": reassembled_rows,
            "composition_sha256": canonical_sha256(
                [(row["column"], row["oof_sha256"]) for row in reassembled_rows]
            ),
            "official_pool_file_sha256": file_sha256(POOL_PATH),
            "issue512_judgment_sha256": issue512["judgment_sha256"],
            "removed_own": sorted(EXPECTED_REMOVED_OWN),
            "added_own": sorted(EXPECTED_ADDED_OWN),
            "strategy": ensemble.CSelectedShrunkRankLogitCombiner.name,
            "c_grid": list(C_GRID),
            "lambda_grid": list(LAMBDA_GRID),
            "max_iter": META_MAX_ITER,
        },
        "cache": {CACHE_NAME: file_sha256(cache_path)},
        "gate": {
            "delta_required": GATE_DELTA,
            "folds_required_positive": FOLDS_REQUIRED_POSITIVE,
            "public_score_used": False,
        },
        "rules": {
            "baseline": (
                "313개는 현재 두 번째 제출의 기준 팔이다. 자체 35개와 외부 278개, "
                "구성원 순서와 봉인 예측을 이슈 489 기록으로 고정한다."
            ),
            "reassembly": (
                "재조립 팔은 이슈 512가 공식화한 자체 풀 36개 전부를 장부 순서로 두고 "
                "기준 팔의 외부 278개를 순서와 예측을 바꾸지 않고 붙인 314개 구성이다."
            ),
            "combiner": (
                "바깥쪽 5분할, C와 수축 계수 격자, 동률 규칙과 결합기 구현을 "
                "현재 두 번째 제출과 같게 둔다."
            ),
            "gate": (
                "재조립 팔 nested OOF에서 현재 두 번째 제출 nested OOF를 뺀 차이가 "
                "+0.00002 이상이고 바깥쪽 분할 5개가 모두 엄격히 양수일 때만 통과한다."
            ),
            "scope": (
                "읽기 전용 판정이며 제출 파일 조립, Kaggle 업로드와 최종 두 장 고정은 "
                "후속 이슈와 사용자 승인으로 남긴다."
            ),
        },
        "environment": _environment(),
        "code_state": state,
    }
    payload["precommit_sha256"] = canonical_sha256(payload)
    write_json(run_dir / "precommit.json", payload)
    print(f"precommit 저장: {run_dir / 'precommit.json'}")
    print(f"  기준 구성 {payload['baseline']['composition_sha256']}")
    print(f"  재조립 구성 {payload['reassembled']['composition_sha256']}")
    print(f"  캐시 {cache_path} {payload['cache'][CACHE_NAME]}")


def load_precommit(run_dir: Path) -> dict:
    _assert_repo_root()
    path = run_dir / "precommit.json"
    _require(path.is_file(), f"precommit.json이 없다: {path}")
    payload = read_json(path)
    expected = canonical_sha256(
        {key: value for key, value in payload.items() if key != "precommit_sha256"}
    )
    _require(
        expected == payload["precommit_sha256"],
        "precommit.json이 제자리에서 바뀌었다.",
    )
    for key, entry in payload["inputs"].items():
        _require(
            file_sha256(Path(entry["path"])) == entry["sha256"],
            f"입력 {key}의 해시가 precommit과 다르다.",
        )
    cache_path = run_dir / "cache" / CACHE_NAME
    _require(
        file_sha256(cache_path) == payload["cache"][CACHE_NAME],
        "재조립 OOF 캐시 해시가 precommit과 다르다.",
    )
    state = _code_state()
    for label, actual, frozen in (
        ("git commit", state["git"]["commit"], payload["code_state"]["git"]["commit"]),
        ("판정 도구", state["script"]["sha256"], payload["code_state"]["script"]["sha256"]),
        (
            "결합기",
            state["ensemble_module"]["sha256"],
            payload["code_state"]["ensemble_module"]["sha256"],
        ),
        (
            "실행 환경 잠금",
            state["uv_lock_sha256"],
            payload["code_state"]["uv_lock_sha256"],
        ),
    ):
        _require(actual == frozen, f"코드 상태({label})가 precommit과 다르다.")
    _require(not state["git"]["dirty"], "판정 실행 중 작업 트리가 바뀌었다.")
    return payload


def _load_matrix(run_dir: Path, payload: dict, fold_of: pd.Series) -> pd.DataFrame:
    matrix = pd.read_parquet(run_dir / "cache" / CACHE_NAME).astype(np.float64)
    expected_columns = [row["column"] for row in payload["reassembled"]["members"]]
    _require(
        list(matrix.columns) == expected_columns,
        "재조립 OOF 캐시 열 순서가 precommit과 다르다.",
    )
    _require(
        matrix.index.equals(fold_of.index),
        "재조립 OOF 캐시 행 순서가 고정 분할과 다르다.",
    )
    _require(
        matrix.shape == (len(fold_of), REASSEMBLED_MEMBER_COUNT)
        and bool(np.isfinite(matrix.to_numpy()).all()),
        "재조립 OOF 캐시 형태나 유한값 검사가 실패했다.",
    )
    return matrix


def fold_job(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    fold = int(args.fold)
    _require(fold in ALL_FOLDS, f"알 수 없는 분할 {fold}")
    out_dir = run_dir / "reassembled" / f"fold-{fold}"
    _require(
        not (out_dir / "reassembled.json").exists(),
        f"이미 완료된 분할이다: {fold}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(payload["source_root"])
    fold_of, y = base._load_fold_and_labels(source_root)
    matrix = _load_matrix(run_dir, payload, fold_of)
    inner = (fold_of != fold).to_numpy()
    outer = (fold_of == fold).to_numpy()
    started = time.monotonic()
    combiner = ensemble.CSelectedShrunkRankLogitCombiner(
        fold_of=fold_of,
        c_grid=C_GRID,
        lambda_grid=LAMBDA_GRID,
        max_iter=META_MAX_ITER,
    )
    try:
        fitted = combiner.fit(matrix[inner], y[inner])
    except ensemble.CombinerConvergenceError as exc:
        raise JudgmentError(f"재조립 팔 분할 {fold}이 수렴하지 않았다: {exc}") from exc
    prediction = np.asarray(fitted.predict(matrix[outer]), dtype=np.float64)
    digest = base._save_fold(out_dir, fold_of, outer, prediction)
    auc = float(roc_auc_score(y[outer].to_numpy(), prediction))
    record = {
        "schema": SCHEMA,
        "precommit_sha256": payload["precommit_sha256"],
        "sealed_fold": fold,
        "strategy": ensemble.CSelectedShrunkRankLogitCombiner.name,
        "member_count": REASSEMBLED_MEMBER_COUNT,
        "rows": int(outer.sum()),
        "auc": auc,
        "selected_c": fitted.c,
        "selected_lambda": fitted.shrinkage_lambda,
        "selected_inner_auc": fitted.selection_aucs[
            (fitted.c, fitted.shrinkage_lambda)
        ],
        "selection_aucs": [
            {"c": c, "lambda": shrinkage, "auc": value}
            for (c, shrinkage), value in fitted.selection_aucs.items()
        ],
        "inner_fits": [fit.__dict__ for fit in fitted.inner_fits],
        "final_iterations": fitted.final_iterations,
        "final_coefficient_l2_norm": fitted.final_coefficient_l2_norm,
        "prediction_sha256": digest,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": base._peak_rss_bytes(),
        "environment": _environment(),
        "finished_at": now_iso(),
    }
    write_json(out_dir / "reassembled.json", record)
    print(
        f"분할 {fold}: AUC {auc:.15f}, C={fitted.c}, "
        f"lambda={fitted.shrinkage_lambda}, {record['elapsed_seconds']:.0f}초",
        flush=True,
    )


def _job_done(run_dir: Path, fold: int) -> bool:
    return (run_dir / "reassembled" / f"fold-{fold}" / "reassembled.json").is_file()


def _running_folds(run_dir: Path) -> set[int]:
    listing = subprocess.run(
        ["ps", "-axo", "command"], capture_output=True, text=True, check=False
    ).stdout
    pattern = re.compile(
        rf"judge_issue513_extended_stack_reassembly\.py fold --run-dir "
        rf"{re.escape(str(run_dir))} --fold (\d+)"
    )
    return {int(fold) for fold in pattern.findall(listing)}


def _reference_check(run_dir: Path, payload: dict) -> None:
    baseline_precommit = read_json(BASELINE_PRECOMMIT_PATH)
    comparison = read_json(BASELINE_COMPARISON_PATH)
    assembly = read_json(BASELINE_ASSEMBLY_PATH)
    checks = {
        "ensemble_code_matches_issue489": (
            payload["code_state"]["ensemble_module"]["sha256"]
            == baseline_precommit["code_state"]["ensemble_module"]["sha256"]
        ),
        "runtime_lock_matches_issue489": (
            payload["code_state"]["uv_lock_sha256"]
            == baseline_precommit["code_state"]["uv_lock_sha256"]
        ),
        "composition_matches_issue489": (
            payload["baseline"]["composition_sha256"]
            == baseline_precommit["members"]["composition_sha256"]
        ),
        "strategy_matches_current_submission": (
            payload["baseline"]["strategy"] == comparison["candidate"]["strategy"]
        ),
        "nested_reference_matches_current_submission": (
            payload["baseline"]["nested_auc"]
            == assembly["judged"]["candidate_nested_auc"]
        ),
        "current_submission_run_matches": (
            payload["baseline"]["submission_run_id"] == CURRENT_SUBMISSION_RUN_ID
        ),
    }
    _require(
        all(checks.values()),
        f"현재 두 번째 제출 기준 호환성 검사가 실패했다: {checks}",
    )
    out_dir = run_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "reference.json",
        {
            "schema": SCHEMA,
            "precommit_sha256": payload["precommit_sha256"],
            "checks": checks,
            "passes": True,
            "reference": payload["baseline"],
            "checked_at": now_iso(),
        },
    )


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
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = str(args.threads)
    print(
        f"남은 분할 {len(pending)}/5, 동시 상한 {args.workers}, "
        f"작업당 계산 줄기 {args.threads}",
        flush=True,
    )
    while pending or active:
        for fold, (process, handle) in list(active.items()):
            code = process.poll()
            if code is not None:
                handle.close()
                del active[fold]
                if code != 0:
                    failed[fold] = code
                result = "완료" if code == 0 else f"실패({code})"
                print(f"분할 {fold} {result} {now_iso()}", flush=True)
        running = _running_folds(run_dir) | set(active)
        while pending and len(running) < args.workers:
            headroom = psutil.virtual_memory().available / psutil.virtual_memory().total
            if headroom < MEMORY_HEADROOM_MIN:
                print(
                    f"메모리 여유율 {headroom:.1%}가 하한 "
                    f"{MEMORY_HEADROOM_MIN:.0%}보다 낮아 새 작업을 기다린다.",
                    flush=True,
                )
                break
            fold = pending.pop(0)
            if fold in running or _job_done(run_dir, fold):
                continue
            handle = (log_dir / f"fold-{fold}.log").open("w")
            command = [
                sys.executable,
                __file__,
                "fold",
                "--run-dir",
                str(run_dir),
                "--fold",
                str(fold),
            ]
            active[fold] = (
                subprocess.Popen(
                    command, env=env, stdout=handle, stderr=subprocess.STDOUT
                ),
                handle,
            )
            running.add(fold)
            print(
                f"분할 {fold} 시작 {now_iso()} (메모리 여유율 {headroom:.1%})",
                flush=True,
            )
        if pending or active:
            time.sleep(10)
    _require(not failed, f"실패한 분할이 있다: {failed}. 로그는 {log_dir}에 있다.")
    print("다섯 분할 실행 완료", flush=True)


def _load_fold_results(
    run_dir: Path,
    payload: dict,
    fold_of: pd.Series,
    y: pd.Series,
) -> tuple[dict[str, dict], pd.Series]:
    records: dict[str, dict] = {}
    nested = pd.Series(np.nan, index=fold_of.index, dtype=np.float64)
    for fold in ALL_FOLDS:
        out_dir = run_dir / "reassembled" / f"fold-{fold}"
        record = read_json(out_dir / "reassembled.json")
        _require(
            record["precommit_sha256"] == payload["precommit_sha256"],
            f"분할 {fold}이 다른 precommit에서 나왔다.",
        )
        part = pd.read_parquet(out_dir / "predictions.parquet").set_index(ID)[
            "prediction"
        ]
        ids = fold_of.index[(fold_of == fold).to_numpy()]
        _require(
            part.index.equals(pd.Index(ids)),
            f"분할 {fold} 예측 id가 고정 분할과 다르다.",
        )
        _require(
            prediction_array_sha256(part.to_numpy()) == record["prediction_sha256"],
            f"분할 {fold} 예측 해시가 기록과 다르다.",
        )
        _require(
            float(roc_auc_score(y.loc[ids].to_numpy(), part.to_numpy()))
            == record["auc"],
            f"분할 {fold} AUC 재계산이 기록과 다르다.",
        )
        nested.loc[ids] = part.to_numpy()
        records[str(fold)] = record
    _require(nested.notna().all(), "이어붙인 재조립 팔 예측에 빈 행이 있다.")
    return records, nested


def compare(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    reference = read_json(run_dir / "baseline" / "reference.json")
    _require(
        reference["passes"]
        and reference["precommit_sha256"] == payload["precommit_sha256"],
        "현재 제출 기준 호환성 검사가 없다.",
    )
    source_root = Path(payload["source_root"])
    fold_of, y = base._load_fold_and_labels(source_root)
    records, nested = _load_fold_results(run_dir, payload, fold_of, y)
    candidate_auc = float(roc_auc_score(y.to_numpy(), nested.to_numpy()))
    baseline_auc = payload["baseline"]["nested_auc"]
    fold_deltas = {
        key: records[key]["auc"] - payload["baseline"]["fold_aucs"][key]
        for key in records
    }
    positive = sum(delta > 0.0 for delta in fold_deltas.values())
    delta = candidate_auc - baseline_auc
    gate_delta = delta >= payload["gate"]["delta_required"]
    gate_folds = positive >= payload["gate"]["folds_required_positive"]
    passes = bool(gate_delta and gate_folds)
    per_fold = []
    for key, record in records.items():
        per_fold.append(
            {
                "fold": int(key),
                "baseline_auc": payload["baseline"]["fold_aucs"][key],
                "reassembled_auc": record["auc"],
                "delta": fold_deltas[key],
                "baseline_selected_c": payload["baseline"]["fold_selected_c"][key],
                "reassembled_selected_c": record["selected_c"],
                "baseline_selected_lambda": payload["baseline"][
                    "fold_selected_lambda"
                ][key],
                "reassembled_selected_lambda": record["selected_lambda"],
                "reassembled_selected_inner_auc": record["selected_inner_auc"],
                "reassembled_final_iterations": record["final_iterations"],
                "reassembled_final_coefficient_l2_norm": record[
                    "final_coefficient_l2_norm"
                ],
                "elapsed_seconds": record["elapsed_seconds"],
                "peak_rss_bytes": record["peak_rss_bytes"],
            }
        )
    verdict = (
        "통과: 재조립 팔이 최종 두 번째 제출 교체 후보가 된다."
        if passes
        else "미달: 현재 두 번째 제출을 유지한다."
    )
    comparison = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "precommit_sha256": payload["precommit_sha256"],
        "baseline_reference_check": reference["checks"],
        "baseline": {
            "submission_run_id": CURRENT_SUBMISSION_RUN_ID,
            "member_count": payload["baseline"]["member_count"],
            "strategy": payload["baseline"]["strategy"],
            "nested_auc": baseline_auc,
            "fold_aucs": payload["baseline"]["fold_aucs"],
            "prediction_sha256": payload["baseline"]["prediction_sha256"],
        },
        "reassembled": {
            "member_count": payload["reassembled"]["member_count"],
            "strategy": payload["reassembled"]["strategy"],
            "nested_auc": candidate_auc,
            "fold_aucs": {key: record["auc"] for key, record in records.items()},
            "fold_selected_c": {
                key: record["selected_c"] for key, record in records.items()
            },
            "fold_selected_lambda": {
                key: record["selected_lambda"] for key, record in records.items()
            },
            "prediction_sha256": prediction_array_sha256(nested.to_numpy()),
        },
        "delta_vs_current_submission": delta,
        "delta_minus_gate": delta - payload["gate"]["delta_required"],
        "fold_deltas": fold_deltas,
        "folds_positive": positive,
        "gate_delta_passes": bool(gate_delta),
        "gate_folds_passes": bool(gate_folds),
        "passes_gate": passes,
        "verdict": verdict,
        "per_fold": per_fold,
        "rows_scored": len(y),
        "elapsed_seconds_total": sum(row["elapsed_seconds"] for row in per_fold),
        "peak_rss_bytes_max": max(row["peak_rss_bytes"] for row in per_fold),
        "public_score_used": False,
        "submission_assembled": False,
        "compared_at": now_iso(),
    }
    write_json(run_dir / "comparison.json", comparison)
    print(
        f"현재 제출 nested {baseline_auc:.10f}, 재조립 팔 {candidate_auc:.10f}, "
        f"차이 {delta:+.7f}, 문턱 여유 {comparison['delta_minus_gate']:+.7f}"
    )
    print(f"바깥쪽 분할 양수 {positive}/5")
    print(f"판정: {verdict}")


def _manifest_files(run_dir: Path) -> list[Path]:
    paths = [
        run_dir / "precommit.json",
        run_dir / "baseline/reference.json",
        run_dir / "comparison.json",
        run_dir / "report.md",
    ]
    paths.extend(
        run_dir / "reassembled" / f"fold-{fold}" / "reassembled.json"
        for fold in ALL_FOLDS
    )
    return [path for path in paths if path.is_file()]


def _gb(value: int) -> str:
    return f"{value / 2**30:.1f}GB"


def report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    comparison = read_json(run_dir / "comparison.json")
    _require(
        comparison["precommit_sha256"] == payload["precommit_sha256"],
        "comparison.json이 다른 precommit에서 나왔다.",
    )
    lines = [f"# 공식 자체 풀 확장 스택 재조립 판정 보고 (이슈 {ISSUE})", ""]
    lines += ["## 판정", ""]
    lines += [f"- 결과: **{comparison['verdict']}**"]
    lines += [
        f"- 재조립 팔 nested OOF `{comparison['reassembled']['nested_auc']:.10f}`에서 "
        f"현재 두 번째 제출 `{comparison['baseline']['nested_auc']:.10f}`을 뺀 차이는 "
        f"`{comparison['delta_vs_current_submission']:+.7f}`이고 문턱은 `+{GATE_DELTA:.5f}`다."
    ]
    lines += [
        f"- 전체 차이 문턱은 {'충족' if comparison['gate_delta_passes'] else '미달'}했고 "
        f"바깥쪽 분할 양수 조건은 `{comparison['folds_positive']}/5`로 "
        f"{'충족' if comparison['gate_folds_passes'] else '미달'}했다."
    ]
    lines += [
        "- Public 점수는 쓰지 않았고 제출 파일, Kaggle 제출 상태와 최종 두 장 고정은 바꾸지 않았다.",
        "",
    ]
    lines += ["## 구성", ""]
    lines += [
        "- 기준 팔은 현재 두 번째 제출의 자체 35개와 외부 278개, 합계 313개다."
    ]
    lines += [
        "- 재조립 팔은 이슈 512가 공식화한 자체 풀 36개와 기준 팔의 외부 278개, 합계 314개다."
    ]
    lines += [
        f"- 기준 구성 해시는 `{payload['baseline']['composition_sha256']}`이고 재조립 구성 해시는 `{payload['reassembled']['composition_sha256']}`다.",
        "",
    ]
    lines += [
        "## 분할별 결과",
        "",
        "| 분할 | 현재 제출 AUC | 재조립 팔 AUC | 차이 | 현재 C | 재조립 C | 현재 수축 계수 | 재조립 수축 계수 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["per_fold"]:
        lines.append(
            f"| {row['fold']} | {row['baseline_auc']:.10f} | "
            f"{row['reassembled_auc']:.10f} | {row['delta']:+.10f} | "
            f"{row['baseline_selected_c']} | {row['reassembled_selected_c']} | "
            f"{row['baseline_selected_lambda']} | "
            f"{row['reassembled_selected_lambda']} |"
        )
    lines += ["", "## 동결과 재현성", ""]
    lines += [
        "- 기준 판정 때와 C 선택 결합기 코드와 실행 환경 잠금 해시가 같고, 현재 313개 구성 해시와 현재 제출 실행도 일치했다."
    ]
    lines += [
        f"- 공식 후보 풀 파일 해시는 `{payload['reassembled']['official_pool_file_sha256']}`이고 이슈 512 판정 해시는 `{payload['reassembled']['issue512_judgment_sha256']}`다."
    ]
    lines += [
        f"- precommit은 `{payload['precommit_sha256']}`이고 재조립 팔 nested 예측 해시는 `{comparison['reassembled']['prediction_sha256']}`다."
    ]
    env = payload["environment"]
    lines += [
        f"- 실행 환경은 {env['platform']} ({env['machine']}), CPU {env['cpu_count']}개, 메모리 {_gb(env['memory_total_bytes'])}, Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, scikit-learn {env['sklearn']}다."
    ]
    lines += [
        f"- 분할 작업 경과 시간 합계는 {comparison['elapsed_seconds_total'] / 60:.1f}분이고 작업 최대 메모리는 {_gb(comparison['peak_rss_bytes_max'])}다.",
        "",
    ]
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = [
        f"{file_sha256(path)}  {path.relative_to(run_dir)}"
        for path in _manifest_files(run_dir)
    ]
    (run_dir / "manifest.sha256").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )
    print(f"보고 저장: {report_path}")


def publish(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    comparison = read_json(run_dir / "comparison.json")
    _require(
        comparison["precommit_sha256"] == payload["precommit_sha256"],
        "comparison.json이 다른 precommit에서 나왔다.",
    )
    publish_dir = Path(args.publish_dir)
    _require(not publish_dir.exists(), f"게시 폴더가 이미 있다: {publish_dir}")
    files = _manifest_files(run_dir)
    _require(len(files) == 9, f"게시할 판정 파일 수가 9개가 아니다: {files}")
    for source in files:
        relative = source.relative_to(run_dir)
        destination = publish_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest = [
        f"{file_sha256(path)}  {path.relative_to(publish_dir)}"
        for path in sorted(publish_dir.rglob("*"))
        if path.is_file()
    ]
    (publish_dir / "manifest.sha256").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )
    print(f"판정 근거 게시: {publish_dir} ({len(manifest)}개 파일)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
