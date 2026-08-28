"""현재 313개 확장 구성에서 로지스틱 규제 강도 선택 절차의 선택 절차 대조 판정. (#489)

대조군은 등록 결합기 `shrunk_rank_logit_logistic`(C=1.0 고정, λ는 안쪽 leave-one-fold-out)
그대로이고, 후보는 같은 안쪽 절차에서 규제 강도 C와 λ를 함께 고르는
`pipeline.ensemble.CSelectedShrunkRankLogitCombiner`다. 둘 다 바깥쪽 분할 5개를 하나씩
봉인하고 나머지 4분할 행으로 결합기를 맞춰 봉인 분할을 예측한 뒤 원래 행 순서로 이어붙여
채점한다. 후보값 각각을 독립 후보로 재평가하지 않고 선택 과정 전체를 판정한다.

판정은 읽기 전용이다. `artifacts/pool.yaml`, champion 판정, 안전판과 확장 스택 제출을 건드리지
않고 MLflow 실행을 만들지 않는다. 산출물은 `run-logs/logistic-c-selection/issue489/`(커밋 제외
경로)에 남긴다.

    precommit.json                       입력 해시, 313 구성원 순서, 후보값, 문턱, 대조군 기준값, 코드 상태(결과 확인 전에 고정)
    cache/oof-313.parquet                313 OOF 행렬(해시는 precommit에)
    control/fold-<k>/predictions.parquet 대조군 봉인 분할 예측, control.json에 AUC·λ·해시
    candidate/fold-<k>/predictions.parquet 후보 봉인 분할 예측, candidate.json에 AUC·선택 (C, λ)·모든 (C, λ) 내부 선택 AUC·설명 진단
    comparison.json                      대조군 재현 검사, 이어붙인 전체·분할별 AUC와 차이, 판정, 설명 진단
    full/proposal.json, 제출 CSV          통과한 경우에만 전체 OOF 제안 (C, λ)와 시험 예측
    report.md, manifest.sha256           사람이 읽는 판정 문서와 모든 산출물의 내용 해시

사용법(실행 순서):
    uv run python scripts/judge_logistic_c_selection.py precommit
    uv run python scripts/judge_logistic_c_selection.py run [--workers 3 --threads 4]
    uv run python scripts/judge_logistic_c_selection.py compare
    uv run python scripts/judge_logistic_c_selection.py full      # 통과한 경우에만
    uv run python scripts/judge_logistic_c_selection.py report

`run`은 대조군 작업 5개를 먼저 돌리고 전체·분할별 AUC가 #455 기준값과 절대 오차 1e-10 안에서
맞을 때만 후보 작업 5개를 시작한다. 로지스틱 적합 동시 실행은 최대 3개이고 메모리 여유율이
15% 아래면 새 작업을 시작하지 않는다. 모든 하위 명령은 시작할 때 입력 해시와 코드 상태를
precommit과 다시 대조하며 하나라도 어긋나면 판정 불가로 둔다.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import assemble_extended_stack as prior_assembly
import judge_extended_stack as ladder
import judge_strict_external_selection as strict

from pipeline import ensemble
from pipeline.data import ID, TARGET, TRAIN_PATH, file_sha256
from pipeline.judgment import FOLDS_PATH, missingness_reweighting, weighted_oof_auc
from pipeline.ledger import POOL_PATH
from pipeline.pool_audit import prediction_array_sha256

ISSUE = 489
SCHEMA = "logistic-c-selection/1"
OUT_DIR = Path("run-logs/logistic-c-selection/issue489")
REHEARSAL_DIR = Path("run-logs/logistic-c-selection/rehearsal")
STRATEGY = strict.STRATEGY
CANDIDATE_STRATEGY = ensemble.CSelectedShrunkRankLogitCombiner.name
C_GRID = ensemble.C_SELECTION_GRID
LAMBDA_GRID = ensemble.SHRINKAGE_LAMBDA_GRID
LOGIT_EPS = ensemble.LogisticLinearCombiner.LOGIT_EPS
META_MAX_ITER = strict.META_MAX_ITER
GATE_DELTA = 0.00002
FOLDS_REQUIRED_POSITIVE = 5
REPRODUCTION_TOLERANCE = 1e-10
MEMORY_HEADROOM_MIN = 0.15
MAX_WORKERS = 3
ALL_FOLDS = strict.ALL_FOLDS
N_TRAIN = strict.N_TRAIN
N_TEST = strict.N_TEST
COMPARISON_MANIFEST_PATH = strict.COMPARISON_MANIFEST_PATH
COMPARISON_EVIDENCE_PATH = strict.COMPARISON_EVIDENCE_PATH
COMPARISON_CONFIG = strict.COMPARISON_CONFIG
MEMBER_COUNT = strict.COMPARISON_MEMBER_COUNT
OWN_MEMBER_COUNT = strict.OWN_MEMBER_COUNT
OWN_TEST_PATH = strict.OWN_TEST_PATH
FULL_REFIT_MANIFEST_PATH = strict.FULL_REFIT_MANIFEST_PATH
TEST_PATH = strict.TEST_PATH
SUBMISSION_DIR = strict.SUBMISSION_DIR
UV_LOCK_PATH = Path("uv.lock")
CURRENT_RUN_ID = "443b3a71a2b045ba9052fbb3d821255d"
CURRENT_SUBMISSION_PATH = Path("artifacts/submissions/issue457-extended-stack-2.csv")
CURRENT_SUBMISSION_SHA256 = "a4d9c5dbcc90f4f63a972ddd885f64f10fcab23a99106c6118d4b1f6665456df"

# 이슈 #489 본문의 동결 입력 해시. precommit이 실제 파일과 대조한다.
EXPECTED_INPUT_SHA256 = {
    "train": (TRAIN_PATH, "f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c"),
    "test": (TEST_PATH, "8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e"),
    "folds": (FOLDS_PATH, "5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4"),
    "pool": (POOL_PATH, "caa1b90769720a4accbe07074dbc7efe0335ab6657fea80c6839b60121dc39d3"),
    "v2_ledger": (ladder.LEDGER_PATH, "e34d01f3f82f55d3255aa2de16c48466d5b1e90992b39a7f79a6c67b7196795e"),
    "ladder_2_evidence": (COMPARISON_EVIDENCE_PATH, "a2eedf4f4d5d92345463c6e6402a929a3268e31c9c674caa51090bd3108b7669"),
    "comparison_manifest": (COMPARISON_MANIFEST_PATH, "3d9a205c810150e7851731b687549fc2b94a2e98fb965b62883fbb4048538711"),
    "full_refit_manifest": (FULL_REFIT_MANIFEST_PATH, "d680982276e1d04b6f59dba167ed04d2bd0396860e6436e213328c85997187c6"),
}
# 이슈 #489 본문의 대조군 기준값(#455 `ablate_new_nhtquyn`, shrunk_rank_logit_logistic).
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

RULES = {
    "hypothesis": "서로 비슷하거나 약한 구성원이 많은 313개 확장 구성에서 더 강한 L2 규제가 계수 분산을 줄일 수 있다는 가설 하나만 검증한다. 1.0보다 큰 C는 넣지 않는다.",
    "control": "대조군은 등록 결합기 shrunk_rank_logit_logistic 그대로(rank_logit 이중 표현, logit 절단 1e-6, StandardScaler, L2 로지스틱 C=1.0 lbfgs max_iter=1000 random_state=0, λ 후보 (0.25, 0.5, 0.75, 1.0), 안쪽 leave-one-fold-out, 동률은 작은 λ).",
    "candidate": "후보는 각 바깥쪽 분할마다 열린 4분할 안에서 C마다 한 분할씩 빼 로지스틱을 맞추고 빠진 분할 예측을 만들어 모든 (C, λ) 조합의 이어붙인 AUC를 재고 최대 하나를 고른 뒤, 선택한 C로 열린 4분할 전체에 다시 맞춰 선택한 λ로 봉인 분할을 한 번 예측한다.",
    "tie": "AUC가 정확히 같으면 더 작은 C, 같은 C 안에서는 더 작은 λ.",
    "reproduction": f"대조군의 이어붙인 전체 AUC와 분할별 AUC가 #455 기준값과 절대 오차 {REPRODUCTION_TOLERANCE} 안에서 맞지 않으면 판정을 시작하지 않는다.",
    "gate": f"후보 이어붙인 nested AUC에서 대조군 {CONTROL_REFERENCE['nested_auc']}를 뺀 차이가 +{GATE_DELTA} 이상이고, 다섯 바깥쪽 분할의 후보 AUC가 각 대조군 분할 AUC보다 모두 엄격히 높을 때만 통과.",
    "fail": "하나라도 실패하면 후보를 기각하고 현재 C=1.0 확장 스택 제출을 유지한다.",
    "diagnostics": "가중 OOF AUC, 분할별 선택 C와 λ, 내부 선택 AUC, 계수 L2 크기, 반복 횟수, 현재 제출·대조군과의 스피어만 순위 상관은 설명 자료로만 기록한다.",
    "public_score": "Public 점수는 구현, 선택, 판정과 동률 해소 어디에도 쓰지 않는다.",
    "full_oof": "통과한 경우에만 전체 OOF에서 같은 규칙을 한 번 적용해 (C, λ)를 고르고 시험 예측과 제출 파일을 만든다. 전체 OOF 선택 점수는 통과 여부를 다시 판정하는 데 쓰지 않으며 Kaggle에 업로드하지 않는다.",
    "abort": "입력 해시, 313 구성원 신원·순서, 행 정렬, 분할 완전성, 유한값 검사, 대조군 재현, 수렴(max_iter=1000) 가운데 하나라도 실패하면 판정 불가. 완료한 일부 분할로 판정하지 않는다.",
    "resume": "중단 뒤 재개는 입력 해시, 후보값, 코드 커밋과 완료 산출물 해시가 정확히 같을 때만 허용한다.",
    "fixed": "결과를 본 뒤 C 후보값, λ 후보값, 표현, 구성원, 판정 문턱을 바꾸지 않는다.",
    "resources": f"모든 계산은 로컬 CPU. 로지스틱 적합 동시 실행 최대 {MAX_WORKERS}개, 메모리 여유율 {MEMORY_HEADROOM_MIN:.0%} 아래면 새 작업을 시작하지 않는다. Vast.ai, Runpod, Kaggle, Colab을 쓰지 않는다.",
}

now_iso = strict.now_iso
canonical_sha256 = strict.canonical_sha256
write_json = strict.write_json
read_json = strict.read_json
JudgmentError = strict.JudgmentError
_require = strict._require


# ---------------------------------------------------------------------------
# 상태 기록


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss) if sys.platform == "darwin" else int(rss) * 1024


def environment() -> dict[str, object]:
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


def code_state() -> dict[str, object]:
    return {
        "git": strict.git_state(),
        "script": {"path": str(Path(__file__).resolve().relative_to(Path.cwd())), "sha256": file_sha256(Path(__file__))},
        "ensemble_module": {"path": str(strict.ENSEMBLE_SOURCE), "sha256": file_sha256(strict.ENSEMBLE_SOURCE)},
        "strict_script_sha256": file_sha256(Path(strict.__file__)),
        "ladder_script_sha256": file_sha256(Path(ladder.__file__)),
        "uv_lock_sha256": file_sha256(UV_LOCK_PATH),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


# ---------------------------------------------------------------------------
# precommit


def precommit(args: argparse.Namespace) -> None:
    rehearsal = args.rehearsal_columns is not None
    run_dir = Path(args.run_dir) if args.run_dir is not None else (REHEARSAL_DIR if rehearsal else OUT_DIR)
    _require(not (run_dir / "precommit.json").exists(), f"precommit.json이 이미 있다(변경 불가): {run_dir}")
    _require(rehearsal or not strict.git_state()["dirty"], "판정은 커밋된 코드 상태에서만 시작한다(git dirty).")
    inputs: dict[str, dict] = {}
    for key, (path, expected) in EXPECTED_INPUT_SHA256.items():
        actual = file_sha256(path)
        _require(actual == expected, f"{key} 해시 {actual}가 이슈의 동결 입력 {expected}와 다르다.")
        inputs[key] = {"path": str(path), "sha256": actual}
    fold_of, y = strict.load_folds_and_labels()
    _require(sorted(fold_of.unique().tolist()) == list(ALL_FOLDS), "분할이 0~4 다섯 개가 아니다.")
    own, own_rows = strict.load_own(fold_of)
    matrix, member_rows = strict.load_comparison_arm(own, fold_of, y)
    _require(matrix.shape == (N_TRAIN, MEMBER_COUNT) and bool(np.isfinite(matrix.to_numpy()).all()), "313 OOF 행렬의 형태나 유한값 검사 실패")
    if rehearsal:
        _require(2 <= args.rehearsal_columns <= MEMBER_COUNT, "예행 열 수는 2 이상 313 이하다.")
        matrix = matrix.iloc[:, : args.rehearsal_columns]
        member_rows = member_rows[: args.rehearsal_columns]
    manifest = read_json(COMPARISON_MANIFEST_PATH)
    _require(manifest["judged"]["nested_auc"] == CONTROL_REFERENCE["nested_auc"], "manifest의 nested AUC가 이슈의 대조군 기준값과 다르다.")
    evidence = read_json(COMPARISON_EVIDENCE_PATH)
    reference = evidence["configs"][COMPARISON_CONFIG]["strategies"][STRATEGY]
    _require(reference["nested_auc"] == CONTROL_REFERENCE["nested_auc"], "#455 근거의 nested AUC가 이슈의 대조군 기준값과 다르다.")
    _require(reference["fold_aucs"] == CONTROL_REFERENCE["fold_aucs"], "#455 근거의 분할별 AUC가 이슈의 대조군 기준값과 다르다.")
    _require(reference["member_count"] == MEMBER_COUNT and evidence["selected_config"] == COMPARISON_CONFIG, "#455 근거의 선택 구성이 313구성원이 아니다.")
    own_test = pd.read_parquet(OWN_TEST_PATH)
    _require(len(own_test) == N_TEST, f"5:1 혼합판 시험 예측 행 수 {len(own_test)}")
    _require(file_sha256(CURRENT_SUBMISSION_PATH) == CURRENT_SUBMISSION_SHA256 == manifest["submission"]["file_sha256"], "현재 확장 스택 제출 CSV가 #457 manifest와 다르다.")

    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(cache_dir / "oof-313.parquet")
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "rehearsal": rehearsal,
        "created_at": now_iso(),
        "question": "313개 확장 구성에서 로지스틱 규제 강도 C를 바깥쪽 학습 부분 안에서 선택하는 절차가 C=1.0 고정 절차보다 nested OOF AUC를 +0.00002 이상 높이고 바깥쪽 분할 5개 모두에서 개선되는가.",
        "outer_folds": list(ALL_FOLDS),
        "inputs": {
            **inputs,
            "own_test": {"path": str(OWN_TEST_PATH), "sha256": file_sha256(OWN_TEST_PATH), "kind": "cv5_full1_mix"},
            "current_submission": {"path": str(CURRENT_SUBMISSION_PATH), "sha256": CURRENT_SUBMISSION_SHA256, "run_id": CURRENT_RUN_ID},
            "uv_lock": {"path": str(UV_LOCK_PATH), "sha256": file_sha256(UV_LOCK_PATH)},
            "fold_vector_sha256": hashlib.sha256(fold_of.to_numpy(np.int8).tobytes()).hexdigest(),
            "rows_per_fold": {str(k): int((fold_of == k).sum()) for k in ALL_FOLDS},
        },
        "members": {
            "count": len(member_rows),
            "own_count": len(own_rows),
            "external_count": len(member_rows) - len(own_rows),
            "config": COMPARISON_CONFIG,
            "rows": member_rows,
            "composition_sha256": canonical_sha256([(r["column"], r["oof_sha256"]) for r in member_rows]),
            "test_composition_sha256": canonical_sha256([(r["column"], r["test_prediction_sha256"]) for r in member_rows]),
        },
        "cache": {"oof-313.parquet": file_sha256(cache_dir / "oof-313.parquet")},
        "control": {
            "strategy": STRATEGY,
            "implementation": "pipeline.ensemble.COMBINER_REGISTRY 등록 결합기. 바깥 분할마다 fit(열린 4분할 행) → predict(봉인 분할).",
            "meta": {"representation": "rank_logit", "C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": META_MAX_ITER, "random_state": 0, "logit_eps": LOGIT_EPS},
            "lambda_grid": list(LAMBDA_GRID),
            "reference": {**CONTROL_REFERENCE, "source": f"#455 {COMPARISON_CONFIG} ({COMPARISON_EVIDENCE_PATH}), 현재 확장 제출 실행 {CURRENT_RUN_ID}"},
            "reproduction_tolerance": REPRODUCTION_TOLERANCE,
        },
        "candidate": {
            "strategy": CANDIDATE_STRATEGY,
            "implementation": "pipeline.ensemble.CSelectedShrunkRankLogitCombiner. 바깥 분할마다 열린 4분할 안 leave-one-fold-out으로 (C, λ)를 고르고 선택한 C로 다시 맞춰 봉인 분할을 예측.",
            "c_grid": list(C_GRID),
            "lambda_grid": list(LAMBDA_GRID),
            "meta": {"representation": "rank_logit", "penalty": "l2", "solver": "lbfgs", "max_iter": META_MAX_ITER, "random_state": 0, "logit_eps": LOGIT_EPS},
        },
        "gate": {"delta_required": GATE_DELTA, "folds_required_positive": FOLDS_REQUIRED_POSITIVE, "control_nested_auc": CONTROL_REFERENCE["nested_auc"], "control_fold_aucs": CONTROL_REFERENCE["fold_aucs"], "public_score_used": False},
        "resources": {"max_workers": MAX_WORKERS, "memory_headroom_min": MEMORY_HEADROOM_MIN, "compute": "local CPU only"},
        "rules": RULES,
        "environment": environment(),
        "code_state": code_state(),
    }
    payload["precommit_sha256"] = canonical_sha256(payload)
    write_json(run_dir / "precommit.json", payload)
    print(f"precommit 저장: {run_dir / 'precommit.json'}" + (" (예행: 도구 경로 확인용이며 판정이 아니다)" if rehearsal else ""))
    print(f"  구성원 {len(member_rows)}(자체 {len(own_rows)}), C 격자 {list(C_GRID)}, λ 격자 {list(LAMBDA_GRID)}, 대조군 기준 {CONTROL_REFERENCE['nested_auc']}")
    print(f"  precommit_sha256 {payload['precommit_sha256']}")


def load_precommit(run_dir: Path) -> dict:
    path = run_dir / "precommit.json"
    _require(path.is_file(), f"precommit.json이 없다: {run_dir}")
    payload = read_json(path)
    _require(canonical_sha256({k: v for k, v in payload.items() if k != "precommit_sha256"}) == payload["precommit_sha256"], "precommit.json이 제자리에서 바뀌었다.")
    for key, entry in payload["inputs"].items():
        if isinstance(entry, dict) and "path" in entry:
            _require(file_sha256(Path(entry["path"])) == entry["sha256"], f"{key} 해시가 precommit과 다르다.")
    for name, digest in payload["cache"].items():
        _require(file_sha256(run_dir / "cache" / name) == digest, f"캐시 {name}이 precommit과 다르다.")
    _require(payload["candidate"]["c_grid"] == list(C_GRID) and payload["candidate"]["lambda_grid"] == list(LAMBDA_GRID), "후보값이 precommit과 다르다.")
    state = code_state()
    for label, actual, expected in (
        ("판정 도구", state["script"]["sha256"], payload["code_state"]["script"]["sha256"]),
        ("결합기 module", state["ensemble_module"]["sha256"], payload["code_state"]["ensemble_module"]["sha256"]),
        ("uv.lock", state["uv_lock_sha256"], payload["code_state"]["uv_lock_sha256"]),
        ("git commit", state["git"]["commit"], payload["code_state"]["git"]["commit"]),
    ):
        _require(actual == expected, f"코드 상태({label})가 precommit과 다르다. precommit부터 다시 한다.")
    return payload


def load_matrix(run_dir: Path, payload: dict, fold_of: pd.Series) -> pd.DataFrame:
    matrix = pd.read_parquet(run_dir / "cache" / "oof-313.parquet").astype(np.float64)
    _require(list(matrix.columns) == [m["column"] for m in payload["members"]["rows"]], "313 열 순서가 precommit과 다르다.")
    _require(matrix.index.equals(fold_of.index), "313 행 순서가 folds와 다르다.")
    _require(bool(np.isfinite(matrix.to_numpy()).all()), "313 OOF 행렬에 비유한값이 있다.")
    return matrix


# ---------------------------------------------------------------------------
# 분할 작업


def _fold_masks(fold_of: pd.Series, sealed: int) -> tuple[np.ndarray, np.ndarray]:
    return (fold_of != sealed).to_numpy(), (fold_of == sealed).to_numpy()


def _save_fold(out_dir: Path, fold_of: pd.Series, outer: np.ndarray, prediction: np.ndarray) -> str:
    _require(prediction.shape == (int(outer.sum()),) and bool(np.isfinite(prediction).all()), "봉인 분할 예측이 유한하지 않다.")
    pd.DataFrame({ID: fold_of.index.to_numpy()[outer], "prediction": prediction}).to_parquet(out_dir / "predictions.parquet", index=False)
    return prediction_array_sha256(prediction)


def control_job(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    sealed = int(args.fold)
    out_dir = run_dir / "control" / f"fold-{sealed}"
    _require(not (out_dir / "control.json").exists(), f"이미 있다: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fold_of, y = strict.load_folds_and_labels()
    matrix = load_matrix(run_dir, payload, fold_of)
    inner, outer = _fold_masks(fold_of, sealed)
    print(f"=== 대조군 {matrix.shape[1]}구성원, 봉인 {sealed} ===", flush=True)
    try:
        fitted = ensemble.COMBINER_REGISTRY[STRATEGY].fit(matrix[inner], y[inner])
    except ensemble.CombinerConvergenceError as exc:
        raise JudgmentError(f"대조군 분할 {sealed} 미수렴({exc}). 전체 판정은 판정 불가다.") from exc
    prediction = np.asarray(fitted.predict(matrix[outer]), dtype=np.float64)
    digest = _save_fold(out_dir, fold_of, outer, prediction)
    auc = float(roc_auc_score(y[outer].to_numpy(), prediction))
    reference = payload["control"]["reference"]["fold_aucs"][str(sealed)]
    write_json(out_dir / "control.json", {
        "schema": SCHEMA,
        "precommit_sha256": payload["precommit_sha256"],
        "sealed_fold": sealed,
        "strategy": STRATEGY,
        "member_count": int(matrix.shape[1]),
        "rows": int(outer.sum()),
        "auc": auc,
        "reference_auc": reference,
        "delta_vs_reference": auc - reference,
        "reproduces": abs(auc - reference) <= REPRODUCTION_TOLERANCE,
        "lambda": float(fitted.shrinkage_lambda),
        "c": 1.0,
        "final_iterations": int(np.max(fitted.meta.model.n_iter_)),
        "final_coefficient_l2_norm": float(np.linalg.norm(fitted.meta.model.coef_[0])),
        "prediction_sha256": digest,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "environment": environment(),
        "finished_at": now_iso(),
    })
    print(f"  AUC {auc:.15f} (기준 {reference:.15f}, 차이 {auc - reference:+.2e}), λ={fitted.shrinkage_lambda} ({time.monotonic() - started:.0f}s)", flush=True)


def candidate_job(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    sealed = int(args.fold)
    out_dir = run_dir / "candidate" / f"fold-{sealed}"
    _require(not (out_dir / "candidate.json").exists(), f"이미 있다: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fold_of, y = strict.load_folds_and_labels()
    matrix = load_matrix(run_dir, payload, fold_of)
    inner, outer = _fold_masks(fold_of, sealed)
    print(f"=== 후보(C 선택) {matrix.shape[1]}구성원, 봉인 {sealed} ===", flush=True)
    combiner = ensemble.CSelectedShrunkRankLogitCombiner(fold_of=fold_of, c_grid=C_GRID, lambda_grid=LAMBDA_GRID, max_iter=META_MAX_ITER)
    try:
        fitted = combiner.fit(matrix[inner], y[inner])
    except ensemble.CombinerConvergenceError as exc:
        raise JudgmentError(f"후보 분할 {sealed} 미수렴({exc}). 전체 판정은 판정 불가다.") from exc
    prediction = np.asarray(fitted.predict(matrix[outer]), dtype=np.float64)
    digest = _save_fold(out_dir, fold_of, outer, prediction)
    auc = float(roc_auc_score(y[outer].to_numpy(), prediction))
    selection = [{"c": c, "lambda": lam, "auc": value} for (c, lam), value in fitted.selection_aucs.items()]
    write_json(out_dir / "candidate.json", {
        "schema": SCHEMA,
        "precommit_sha256": payload["precommit_sha256"],
        "sealed_fold": sealed,
        "strategy": CANDIDATE_STRATEGY,
        "member_count": int(matrix.shape[1]),
        "rows": int(outer.sum()),
        "auc": auc,
        "selected_c": fitted.c,
        "selected_lambda": fitted.shrinkage_lambda,
        "selected_inner_auc": fitted.selection_aucs[(fitted.c, fitted.shrinkage_lambda)],
        "selection_aucs": selection,
        "inner_fits": [fit.__dict__ for fit in fitted.inner_fits],
        "final_iterations": fitted.final_iterations,
        "final_coefficient_l2_norm": fitted.final_coefficient_l2_norm,
        "prediction_sha256": digest,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "environment": environment(),
        "finished_at": now_iso(),
    })
    print(f"  AUC {auc:.15f}, 선택 C={fitted.c} λ={fitted.shrinkage_lambda} (내부 {fitted.selection_aucs[(fitted.c, fitted.shrinkage_lambda)]:.7f}), 적합 {len(fitted.inner_fits) + 1}회 ({time.monotonic() - started:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# 대조군 재현 검사와 작업 실행기


def _load_fold_records(run_dir: Path, payload: dict, kind: str, fold_of: pd.Series, y: pd.Series) -> tuple[dict[str, dict], pd.Series]:
    """분할별 기록과 예측을 읽어 해시와 AUC를 다시 확인하고 원래 행 순서로 이어붙인다."""
    name = "control.json" if kind == "control" else "candidate.json"
    records: dict[str, dict] = {}
    concatenated = pd.Series(np.nan, index=fold_of.index)
    for fold in payload["outer_folds"]:
        fold_dir = run_dir / kind / f"fold-{fold}"
        path = fold_dir / name
        _require(path.is_file(), f"{kind} 분할 {fold}의 산출물이 없다. 전체 판정은 판정 불가다.")
        record = read_json(path)
        _require(record["precommit_sha256"] == payload["precommit_sha256"], f"{kind} 분할 {fold}이 다른 precommit에서 나왔다.")
        part = pd.read_parquet(fold_dir / "predictions.parquet").set_index(ID)["prediction"]
        ids = fold_of.index[(fold_of == fold).to_numpy()]
        _require(part.index.equals(pd.Index(ids)), f"{kind} 분할 {fold} 예측 id가 분할 배정과 다르다.")
        _require(prediction_array_sha256(part.to_numpy()) == record["prediction_sha256"], f"{kind} 분할 {fold} 예측 해시가 기록과 다르다.")
        _require(float(roc_auc_score(y.loc[ids].to_numpy(), part.to_numpy())) == record["auc"], f"{kind} 분할 {fold} AUC 재계산이 기록과 다르다.")
        concatenated.loc[ids] = part.to_numpy()
        records[str(fold)] = record
    _require(concatenated.notna().all(), f"{kind} 이어붙인 예측에 빈 행이 있다.")
    return records, concatenated


def reproduction_check(run_dir: Path, payload: dict, fold_of: pd.Series, y: pd.Series) -> dict:
    records, concatenated = _load_fold_records(run_dir, payload, "control", fold_of, y)
    reference = payload["control"]["reference"]
    nested_auc = float(roc_auc_score(y.to_numpy(), concatenated.to_numpy()))
    fold_deltas = {k: records[k]["auc"] - reference["fold_aucs"][k] for k in records}
    delta = nested_auc - reference["nested_auc"]
    passes = abs(delta) <= REPRODUCTION_TOLERANCE and all(abs(v) <= REPRODUCTION_TOLERANCE for v in fold_deltas.values())
    return {
        "tolerance": REPRODUCTION_TOLERANCE,
        "reference_nested_auc": reference["nested_auc"],
        "control_nested_auc": nested_auc,
        "delta": delta,
        "reference_fold_aucs": reference["fold_aucs"],
        "control_fold_aucs": {k: v["auc"] for k, v in records.items()},
        "fold_deltas": fold_deltas,
        "control_fold_lambdas": {k: v["lambda"] for k, v in records.items()},
        "passes": bool(passes),
    }


def _job_done(run_dir: Path, kind: str, fold: int) -> bool:
    return (run_dir / kind / f"fold-{fold}" / ("control.json" if kind == "control" else "candidate.json")).is_file()


def _running_jobs(run_dir: Path) -> set[tuple[str, int]]:
    listing = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True, check=False).stdout
    pattern = re.compile(rf"judge_logistic_c_selection\.py (control|candidate) --run-dir {re.escape(str(run_dir))} --fold (\d+)")
    return {(kind, int(fold)) for kind, fold in pattern.findall(listing)}


def _memory_headroom() -> float:
    memory = psutil.virtual_memory()
    return memory.available / memory.total


def _run_phase(run_dir: Path, kind: str, folds: list[int], workers: int, threads: int, log_dir: Path) -> None:
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = str(threads)
    pending = [fold for fold in folds if not _job_done(run_dir, kind, fold)]
    print(f"[{kind}] 남은 작업 {len(pending)}/{len(folds)}개, 동시 상한 {workers}, 스레드 {threads}", flush=True)
    active: dict[int, subprocess.Popen] = {}
    results: dict[int, int] = {}
    while pending or active:
        for fold, process in list(active.items()):
            code = process.poll()
            if code is not None:
                results[fold] = code
                del active[fold]
                print(f"[{kind}] {'완료' if code == 0 else f'실패({code})'} 분할 {fold} {now_iso()}", flush=True)
        running = {fold for job_kind, fold in _running_jobs(run_dir) if job_kind == kind} | set(active)
        while pending and len(running) < workers:
            headroom = _memory_headroom()
            if headroom < MEMORY_HEADROOM_MIN:
                print(f"[{kind}] 메모리 여유율 {headroom:.1%} < {MEMORY_HEADROOM_MIN:.0%}, 새 작업 보류", flush=True)
                break
            fold = pending.pop(0)
            if fold in running or _job_done(run_dir, kind, fold):
                continue
            handle = (log_dir / f"{kind}-fold-{fold}.log").open("w")
            active[fold] = subprocess.Popen([sys.executable, __file__, kind, "--run-dir", str(run_dir), "--fold", str(fold)], env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.add(fold)
            print(f"[{kind}] 시작 분할 {fold} {now_iso()} (메모리 여유율 {headroom:.1%})", flush=True)
        time.sleep(15)
    failed = [fold for fold, code in results.items() if code != 0]
    if failed:
        sys.exit(f"[{kind}] 실패한 분할 {failed}. 전체 판정은 판정 불가다. 로그: {log_dir}")


def run_jobs(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    _require(1 <= args.workers <= MAX_WORKERS, f"동시 실행은 최대 {MAX_WORKERS}개다.")
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(k) for k in payload["outer_folds"]]
    _run_phase(run_dir, "control", folds, args.workers, args.threads, log_dir)
    fold_of, y = strict.load_folds_and_labels()
    check = reproduction_check(run_dir, payload, fold_of, y)
    write_json(run_dir / "control" / "reproduction.json", {"schema": SCHEMA, "precommit_sha256": payload["precommit_sha256"], **check, "checked_at": now_iso()})
    print(f"대조군 재현: 전체 {check['control_nested_auc']:.15f} (기준 {check['reference_nested_auc']:.15f}, 차이 {check['delta']:+.2e}), 분할 최대 차이 {max(abs(v) for v in check['fold_deltas'].values()):.2e} → {'통과' if check['passes'] else '실패'}", flush=True)
    if not check["passes"]:
        if not payload["rehearsal"]:
            sys.exit("대조군 재현 실패. 판정을 시작하지 않는다(판정 불가).")
        print("예행이라 대조군 재현 실패를 무시하고 후보 작업을 계속한다.", flush=True)
    _run_phase(run_dir, "candidate", folds, args.workers, args.threads, log_dir)
    print("all jobs finished", flush=True)


# ---------------------------------------------------------------------------
# 비교


def compare(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    fold_of, y = strict.load_folds_and_labels()
    reproduction = reproduction_check(run_dir, payload, fold_of, y)
    control_records, control = _load_fold_records(run_dir, payload, "control", fold_of, y)
    candidate_records, candidate = _load_fold_records(run_dir, payload, "candidate", fold_of, y)
    label_values = y.to_numpy()
    control_auc = float(roc_auc_score(label_values, control.to_numpy()))
    candidate_auc = float(roc_auc_score(label_values, candidate.to_numpy()))
    reference_auc = payload["gate"]["control_nested_auc"]
    delta = candidate_auc - reference_auc
    fold_deltas = {k: candidate_records[k]["auc"] - control_records[k]["auc"] for k in control_records}
    positives = sum(v > 0.0 for v in fold_deltas.values())
    gate_delta = delta >= GATE_DELTA
    gate_folds = positives >= FOLDS_REQUIRED_POSITIVE
    passes = bool(reproduction["passes"] and gate_delta and gate_folds)

    reweighting = missingness_reweighting(TRAIN_PATH, strict.MISSINGNESS_TEST_PATH)
    weighted_control = weighted_oof_auc(control.rename("prediction"), y, reweighting)
    weighted_candidate = weighted_oof_auc(candidate.rename("prediction"), y, reweighting)
    rho = float(spearmanr(candidate.to_numpy(), control.to_numpy()).correlation)

    per_fold = []
    consistency = []
    for k in control_records:
        rec = candidate_records[k]
        c1_rows = [row for row in rec["selection_aucs"] if row["c"] == 1.0]
        best_c1 = max(c1_rows, key=lambda row: row["auc"])
        lambda_at_c1 = min(row["lambda"] for row in c1_rows if row["auc"] == best_c1["auc"])
        consistency.append(lambda_at_c1 == control_records[k]["lambda"])
        per_fold.append({
            "fold": int(k),
            "control_auc": control_records[k]["auc"],
            "candidate_auc": rec["auc"],
            "delta": fold_deltas[k],
            "control_lambda": control_records[k]["lambda"],
            "selected_c": rec["selected_c"],
            "selected_lambda": rec["selected_lambda"],
            "selected_inner_auc": rec["selected_inner_auc"],
            "inner_auc_at_c1_best": best_c1["auc"],
            "inner_lambda_at_c1": lambda_at_c1,
            "inner_gain_vs_c1": rec["selected_inner_auc"] - best_c1["auc"],
            "control_final_iterations": control_records[k]["final_iterations"],
            "candidate_final_iterations": rec["final_iterations"],
            "control_final_coefficient_l2_norm": control_records[k]["final_coefficient_l2_norm"],
            "candidate_final_coefficient_l2_norm": rec["final_coefficient_l2_norm"],
            "control_elapsed_seconds": control_records[k]["elapsed_seconds"],
            "candidate_elapsed_seconds": rec["elapsed_seconds"],
            "control_peak_rss_bytes": control_records[k]["peak_rss_bytes"],
            "candidate_peak_rss_bytes": rec["peak_rss_bytes"],
            "selection_aucs": rec["selection_aucs"],
            "inner_fits": rec["inner_fits"],
        })
    if payload["rehearsal"]:
        verdict = "예행(판정 아님)"
    elif not reproduction["passes"]:
        verdict = "판정 불가(대조군 재현 실패)"
    elif passes:
        verdict = "통과"
    else:
        verdict = "미달: 현재 C=1.0 확장 스택 제출 유지"
    record = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "precommit_sha256": payload["precommit_sha256"],
        "rehearsal": payload["rehearsal"],
        "outer_folds": payload["outer_folds"],
        "gate": payload["gate"],
        "reproduction": reproduction,
        "control": {"strategy": STRATEGY, "nested_auc": control_auc, "weighted_oof_auc": weighted_control.auc, "prediction_sha256": prediction_array_sha256(control.to_numpy()), "fold_aucs": {k: v["auc"] for k, v in control_records.items()}, "fold_lambdas": {k: v["lambda"] for k, v in control_records.items()}},
        "candidate": {"strategy": CANDIDATE_STRATEGY, "nested_auc": candidate_auc, "weighted_oof_auc": weighted_candidate.auc, "prediction_sha256": prediction_array_sha256(candidate.to_numpy()), "fold_aucs": {k: v["auc"] for k, v in candidate_records.items()}, "fold_selected_c": {k: v["selected_c"] for k, v in candidate_records.items()}, "fold_selected_lambda": {k: v["selected_lambda"] for k, v in candidate_records.items()}},
        "delta_vs_control_reference": delta,
        "delta_vs_control_reproduced": candidate_auc - control_auc,
        "delta_minus_gate": delta - GATE_DELTA,
        "fold_deltas": fold_deltas,
        "folds_positive": positives,
        "gate_delta_passes": bool(gate_delta),
        "gate_folds_pass": bool(gate_folds),
        "passes_gate": passes,
        "verdict": verdict,
        "diagnostics": {
            "weighted_delta": weighted_candidate.auc - weighted_control.auc,
            "weighted_effective_sample_fraction": weighted_control.effective_sample_fraction,
            "spearman_candidate_vs_control_nested": rho,
            "control_lambda_equals_candidate_argmax_at_c1": consistency,
            "note": "설명 자료. 판정에 쓰지 않는다.",
        },
        "per_fold": per_fold,
        "rows_scored": len(y),
        "elapsed_seconds_total": sum(v["elapsed_seconds"] for v in control_records.values()) + sum(v["elapsed_seconds"] for v in candidate_records.values()),
        "peak_rss_bytes_max": max([v["peak_rss_bytes"] for v in control_records.values()] + [v["peak_rss_bytes"] for v in candidate_records.values()]),
        "compared_at": now_iso(),
    }
    write_json(run_dir / "comparison.json", record)
    print(f"대조군 재현: {'통과' if reproduction['passes'] else '실패'} (차이 {reproduction['delta']:+.2e})")
    print(f"대조군 nested {control_auc:.10f}, 후보 nested {candidate_auc:.10f}, 차이 {delta:+.7f} (문턱 +{GATE_DELTA}), 분할 양수 {positives}/5")
    for row in per_fold:
        print(f"  분할 {row['fold']}: 대조 {row['control_auc']:.7f} 후보 {row['candidate_auc']:.7f} Δ {row['delta']:+.7f}  선택 C={row['selected_c']} λ={row['selected_lambda']} (내부 {row['selected_inner_auc']:.7f}, C=1 최선 {row['inner_auc_at_c1_best']:.7f})")
    print(f"가중 OOF: 대조 {weighted_control.auc:.7f} 후보 {weighted_candidate.auc:.7f}, 스피어만 {rho:.6f}")
    print(f"판정: {verdict}")


# ---------------------------------------------------------------------------
# 전체 OOF 제안(통과 뒤)


def full_proposal(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    comparison = read_json(run_dir / "comparison.json")
    _require(comparison["precommit_sha256"] == payload["precommit_sha256"], "comparison.json이 다른 precommit에서 나왔다.")
    _require(payload["rehearsal"] or (comparison["passes_gate"] and comparison["verdict"] == "통과"), "문턱을 통과하지 못했다. 전체 OOF 제안을 만들지 않는다.")
    started = time.monotonic()
    fold_of, y = strict.load_folds_and_labels()
    oof = load_matrix(run_dir, payload, fold_of)
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID])[ID]
    _require(len(test_ids) == N_TEST and not test_ids.duplicated().any(), "test.csv의 행 수나 id가 기대와 다르다.")
    own_test = pd.read_parquet(OWN_TEST_PATH)
    _require(own_test[ID].to_numpy().tolist() == test_ids.to_numpy().tolist(), "5:1 혼합판 시험 예측의 id 순서가 test.csv와 다르다.")
    own_test = own_test.set_index(ID)
    columns: dict[str, np.ndarray] = {}
    sources: dict[str, dict] = {}
    for entry in payload["members"]["rows"]:
        column = entry["column"]
        if entry["origin"] == "own":
            values = own_test[column].to_numpy(np.float64)
            sources[column] = {"kind": "own_cv5_full1_mix", "test_path": str(OWN_TEST_PATH)}
        else:
            values = ladder.load_ledger_array(entry["test_path"])
            sources[column] = {"kind": "external_cv_fold_average", "test_path": entry["test_path"]}
        _require(values.shape == (N_TEST,) and bool(np.isfinite(values).all()), f"{column}: 시험 배열 형태 {values.shape} 또는 비유한값")
        _require(prediction_array_sha256(values) == entry["test_prediction_sha256"], f"{column}: 시험 배열 해시가 #457 manifest와 다르다.")
        columns[column] = values
        sources[column]["prediction_sha256"] = entry["test_prediction_sha256"]
    test = pd.DataFrame(columns, index=test_ids.to_numpy()).astype(np.float64)
    _require(list(test.columns) == list(oof.columns), "시험 행렬의 열 순서가 OOF와 다르다.")
    combiner = ensemble.CSelectedShrunkRankLogitCombiner(fold_of=fold_of, c_grid=C_GRID, lambda_grid=LAMBDA_GRID, max_iter=META_MAX_ITER)
    fitted = combiner.fit(oof, y)
    prediction = np.asarray(fitted.predict(test), dtype=np.float64)
    _require(prediction.shape == (N_TEST,) and bool(np.isfinite(prediction).all()), "제안 시험 예측이 유한하지 않다.")
    current = pd.read_csv(CURRENT_SUBMISSION_PATH)
    _require(current[ID].to_numpy().tolist() == test_ids.to_numpy().tolist(), "현재 제출의 id 순서가 test.csv와 다르다.")
    rho = float(spearmanr(prediction, current[TARGET].to_numpy(np.float64)).correlation)
    submission_dir = run_dir / "full" if payload["rehearsal"] else SUBMISSION_DIR
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / f"issue{ISSUE}-c-selected-extended-stack{'-rehearsal' if payload['rehearsal'] else ''}.csv"
    pd.DataFrame({ID: test_ids.to_numpy(), TARGET: prediction}).to_csv(submission_path, index=False)
    out_dir = run_dir / "full"
    write_json(out_dir / "proposal.json", {
        "schema": SCHEMA,
        "issue": ISSUE,
        "rehearsal": payload["rehearsal"],
        "precommit_sha256": payload["precommit_sha256"],
        "strategy": CANDIDATE_STRATEGY,
        "selected_c": fitted.c,
        "selected_lambda": fitted.shrinkage_lambda,
        "selection_aucs": [{"c": c, "lambda": lam, "auc": value} for (c, lam), value in fitted.selection_aucs.items()],
        "selection_note": "전체 OOF 5분할 leave-one-fold-out 내부 예측의 AUC. 통과 여부를 다시 판정하는 데 쓰지 않는다.",
        "inner_fits": [fit.__dict__ for fit in fitted.inner_fits],
        "final_iterations": fitted.final_iterations,
        "final_coefficient_l2_norm": fitted.final_coefficient_l2_norm,
        "in_sample_oof_auc": float(roc_auc_score(y.to_numpy(), np.asarray(fitted.predict(oof), dtype=np.float64))),
        "members": [{"column": column, "weight": float(weight), "test": sources[column]} for column, weight in fitted.summary().items()],
        "submission": {"path": str(submission_path), "file_sha256": file_sha256(submission_path), "prediction_sha256": prediction_array_sha256(prediction), "checks": prior_assembly.rank_space_checks(prediction, test_ids), "spearman_vs_current_submission": {"path": str(CURRENT_SUBMISSION_PATH), "sha256": file_sha256(CURRENT_SUBMISSION_PATH), "run_id": CURRENT_RUN_ID, "spearman": rho}, "uploaded": False},
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "finished_at": now_iso(),
    })
    print(f"전체 OOF 제안 C={fitted.c} λ={fitted.shrinkage_lambda}, 제출 {submission_path} sha256 {file_sha256(submission_path)}, 현재 제출과 스피어만 {rho:.6f} (업로드하지 않음)")


# ---------------------------------------------------------------------------
# 보고


def _manifest_files(run_dir: Path) -> list[Path]:
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.relative_to(run_dir).parts[0] != "logs" and path.name != "manifest.sha256":
            files.append(path)
    return files


def _gb(value: int) -> str:
    return f"{value / 2**30:.1f}GB"


def report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    comparison = read_json(run_dir / "comparison.json")
    _require(comparison["precommit_sha256"] == payload["precommit_sha256"], "comparison.json이 다른 precommit에서 나왔다.")
    rep = comparison["reproduction"]
    lines: list[str] = []
    lines += [f"# 로지스틱 규제 강도 선택 절차 판정 보고 (#{ISSUE})", ""]
    if payload["rehearsal"]:
        lines += [f"**예행 실행이다.** 313 가운데 앞 {payload['members']['count']}열로 도구 경로만 확인했으며 판정이 아니다.", ""]
    lines += ["## 판정", ""]
    lines += [f"- 결과: **{comparison['verdict']}**."]
    lines += [f"- 후보 이어붙인 nested AUC `{comparison['candidate']['nested_auc']:.10f}` - 대조군 `{comparison['gate']['control_nested_auc']:.10f}` = `{comparison['delta_vs_control_reference']:+.7f}` (문턱 `+{GATE_DELTA}`, {'충족' if comparison['gate_delta_passes'] else '미달'})."]
    lines += [f"- 분할별 후보 > 대조군: {comparison['folds_positive']}/5 ({'충족' if comparison['gate_folds_pass'] else '미달'})."]
    lines += [f"- 대조군 재현: 전체 차이 `{rep['delta']:+.2e}`, 분할 최대 차이 `{max(abs(v) for v in rep['fold_deltas'].values()):.2e}`, 허용 `{REPRODUCTION_TOLERANCE}` → {'통과' if rep['passes'] else '실패'}."]
    lines += ["- 결과를 본 뒤 C 후보값, λ 후보값, 표현, 구성원, 문턱을 바꾸지 않았다.", ""]
    lines += ["## 분할별 결과", "", "| 분할 | 대조군 AUC | 후보 AUC | 차이 | 대조군 λ | 선택 C | 선택 λ | 내부 선택 AUC | C=1 최선 내부 AUC | 내부 이득 |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison["per_fold"]:
        lines.append(f"| {row['fold']} | {row['control_auc']:.7f} | {row['candidate_auc']:.7f} | {row['delta']:+.7f} | {row['control_lambda']} | {row['selected_c']} | {row['selected_lambda']} | {row['selected_inner_auc']:.7f} | {row['inner_auc_at_c1_best']:.7f} | {row['inner_gain_vs_c1']:+.7f} |")
    lines += ["", "## 설명 진단(판정에 쓰지 않음)", ""]
    diag = comparison["diagnostics"]
    lines += [f"- 가중 OOF AUC: 대조군 `{comparison['control']['weighted_oof_auc']:.7f}`, 후보 `{comparison['candidate']['weighted_oof_auc']:.7f}`, 차이 `{diag['weighted_delta']:+.7f}`."]
    lines += [f"- 후보와 대조군 nested 예측의 스피어만 순위 상관 `{diag['spearman_candidate_vs_control_nested']:.6f}`."]
    lines += [f"- 대조군 λ와 후보 표의 C=1.0 행 argmax λ 일치: {diag['control_lambda_equals_candidate_argmax_at_c1']}."]
    lines += ["", "| 분할 | 대조군 최종 반복 | 후보 최종 반복 | 대조군 계수 L2 | 후보 계수 L2 | 대조군 시간 | 후보 시간 | 대조군 최대 메모리 | 후보 최대 메모리 |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison["per_fold"]:
        lines.append(f"| {row['fold']} | {row['control_final_iterations']} | {row['candidate_final_iterations']} | {row['control_final_coefficient_l2_norm']:.4f} | {row['candidate_final_coefficient_l2_norm']:.4f} | {row['control_elapsed_seconds']:.0f}s | {row['candidate_elapsed_seconds']:.0f}s | {_gb(row['control_peak_rss_bytes'])} | {_gb(row['candidate_peak_rss_bytes'])} |")
    lines += ["", "분할별 모든 (C, λ) 내부 선택 AUC(열린 4분할 leave-one-fold-out 이어붙임):", ""]
    lines += ["| 분할 | C | " + " | ".join(f"λ={lam}" for lam in LAMBDA_GRID) + " | 반복 | 계수 L2 |", "| ---: | ---: | " + " | ".join("---:" for _ in LAMBDA_GRID) + " | --- | --- |"]
    for row in comparison["per_fold"]:
        by_c: dict[float, dict[float, float]] = {}
        for cell in row["selection_aucs"]:
            by_c.setdefault(cell["c"], {})[cell["lambda"]] = cell["auc"]
        fits_by_c: dict[float, list[dict]] = {}
        for fit in row["inner_fits"]:
            fits_by_c.setdefault(fit["c"], []).append(fit)
        for c in C_GRID:
            fits = fits_by_c.get(c, [])
            lines.append(f"| {row['fold']} | {c} | " + " | ".join(f"{by_c[c][lam]:.7f}" for lam in LAMBDA_GRID) + f" | {','.join(str(f['iterations']) for f in fits)} | {','.join(f'{f['coefficient_l2_norm']:.3f}' for f in fits)} |")
    proposal_path = run_dir / "full" / "proposal.json"
    lines += ["", "## 전체 OOF 제안", ""]
    if proposal_path.is_file():
        proposal = read_json(proposal_path)
        lines += [f"- 제안 (C, λ) = ({proposal['selected_c']}, {proposal['selected_lambda']}), 전체 OOF 적합 반복 {proposal['final_iterations']}, 계수 L2 `{proposal['final_coefficient_l2_norm']:.4f}`."]
        lines += [f"- 제출 파일 `{proposal['submission']['path']}` sha256 `{proposal['submission']['file_sha256']}`, 현재 제출과 스피어만 `{proposal['submission']['spearman_vs_current_submission']['spearman']:.6f}`, Kaggle 업로드 없음."]
    else:
        lines += ["- 문턱을 통과하지 못해 만들지 않았다. 현재 C=1.0 확장 스택 제출(`443b3a71`, `a4d9c5db…`)을 유지한다."]
    env = payload["environment"]
    lines += ["", "## 실행 환경과 자원", ""]
    lines += [f"- {env['platform']} ({env['machine']}), CPU {env['cpu_count']}개, 메모리 {_gb(env['memory_total_bytes'])}, Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, sklearn {env['sklearn']}."]
    lines += [f"- 로컬 CPU만 사용, 동시 실행 상한 {MAX_WORKERS}, 메모리 여유율 하한 {MEMORY_HEADROOM_MIN:.0%}."]
    lines += [f"- 분할 작업 경과 시간 합계 {comparison['elapsed_seconds_total'] / 60:.0f}분, 작업 최대 메모리 {_gb(comparison['peak_rss_bytes_max'])}."]
    lines += ["", "## 동결 입력과 코드 상태", ""]
    for key, entry in payload["inputs"].items():
        if isinstance(entry, dict) and "path" in entry:
            lines.append(f"- {key}: `{entry['path']}` `{entry['sha256']}`")
    lines += [f"- 313 구성 해시 `{payload['members']['composition_sha256']}`, 시험 구성 해시 `{payload['members']['test_composition_sha256']}`, 분할 벡터 `{payload['inputs']['fold_vector_sha256']}`."]
    code = payload["code_state"]
    lines += [f"- git `{code['git']['commit']}` (dirty {code['git']['dirty']}), 판정 도구 `{code['script']['sha256']}`, 결합기 module `{code['ensemble_module']['sha256']}`, uv.lock `{code['uv_lock_sha256']}`."]
    lines += [f"- precommit_sha256 `{payload['precommit_sha256']}`, 비교 {comparison['compared_at']}, 보고 작성 {now_iso()}.", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_lines = [f"{file_sha256(path)}  {path.relative_to(run_dir)}" for path in _manifest_files(run_dir)]
    (run_dir / "manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"보고 저장: {run_dir / 'report.md'}, manifest {len(manifest_lines)}개 파일")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("precommit", precommit), ("run", run_jobs), ("compare", compare), ("full", full_proposal), ("report", report), ("control", control_job), ("candidate", candidate_job)):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, default=None if name == "precommit" else OUT_DIR)
        if name == "precommit":
            p.add_argument("--rehearsal-columns", type=int, default=None, help="예행: 313 가운데 앞 K열만 쓴다(판정 아님)")
        if name in ("control", "candidate"):
            p.add_argument("--fold", type=int, required=True)
        if name == "run":
            p.add_argument("--workers", type=int, default=MAX_WORKERS)
            p.add_argument("--threads", type=int, default=4)
        p.set_defaults(handler=handler)
    args = parser.parse_args()
    try:
        args.handler(args)
    except JudgmentError as exc:
        sys.exit(f"판정 불가: {exc}")


if __name__ == "__main__":
    main()
