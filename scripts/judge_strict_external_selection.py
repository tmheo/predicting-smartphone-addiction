"""엄격 외부 후보의 중첩 선별 판정 도구. (#486, ADR-0005)

외부 후보 동결 명세(`scripts/freeze_external_candidates.py`)를 입력으로, 현재 313개
확장 구성을 비교 팔로 두고 자체 35개를 필수 시작 구성으로 삼는 후보 절차 팔을
바깥 분할 5곳에서 각각 선별해 교체 문턱(+0.00002, 5/5 양수)을 판정한다.

판정은 읽기 전용이다. `artifacts/pool.yaml`과 champion 판정을 건드리지 않고 MLflow
실행을 만들지 않는다. 산출물은 `run-logs/strict-external-selection/<후보 집합 식별자>/`
(커밋 제외 경로)에 ADR-0005의 변경 불가 산출물 이름으로 남긴다.

    precommit.json                  동결 입력·규칙·코드 상태(결과 확인 전에 고정)
    cache/*.parquet                 자체 35, 비교 팔 313, 후보 OOF 행렬(해시는 precommit에)
    fold-<k>/selection.json         바깥 분할 k의 선별 기록(모든 이동·승인·명단·중단 상태)
    fold-<k>/predictions.parquet    후보 절차 팔의 봉인 분할 예측
    fold-<k>/baseline-predictions.parquet  비교 팔 313의 같은 분할 예측
    nested-comparison.json          이어붙인 전체 AUC 차이·분할별 차이·문턱 판정
    selection-stability.json        후보별 분할 선택 수, 분할 명단과 전체 OOF 제안 명단의 차이
    full-selection.json             절차 통과 뒤에만: 전체 OOF 검색과 실제 제안 명단
    report.md, manifest.sha256      사람이 읽는 요약과 모든 산출물의 내용 해시

사용법(실행 회차 순서):
    uv run python scripts/judge_strict_external_selection.py precommit --spec <동결 명세>
    uv run python scripts/judge_strict_external_selection.py run --run-dir <출력 폴더> [--workers 5 --heavy-workers 2 --threads 2]
    uv run python scripts/judge_strict_external_selection.py compare --run-dir <출력 폴더>
    uv run python scripts/judge_strict_external_selection.py full --run-dir <출력 폴더>      # 통과했을 때만
    uv run python scripts/judge_strict_external_selection.py report --run-dir <출력 폴더>
    uv run python scripts/judge_strict_external_selection.py assemble --run-dir <출력 폴더> # 통과 뒤, 사용자 확인 뒤

`run`은 (baseline k, select k) 작업 10개를 하위 프로세스로 돌린다. 이미 산출물이 있는
작업은 건너뛰므로 중단 뒤 같은 명령으로 이어 달릴 수 있다. 모든 하위 명령은 시작할 때
precommit의 입력 해시와 코드 상태를 다시 계산해 정확히 같을 때만 진행한다(ADR-0005
재개 규칙). 하나라도 어긋나면 `판정 불가`로 두고 precommit부터 다시 한다.

후보 절차 팔의 결합기는 등록 전략 `shrunk_rank_logit_logistic`과 같은 계산(rank_logit
이중 표현 로지스틱, λ는 학습 부분 안의 leave-one-fold-out, 순위 공간 볼록 결합)을
열 부분집합에 대해 빠르게 반복하도록 순위 특성을 학습 분할 집합별로 캐시한 구현이다.
봉인 분할 예측에서는 등록 결합기를 그대로 한 번 더 적합해 두 구현의 예측 차이를
`selection.json`에 남긴다. 비교 팔 313은 등록 결합기로만 예측한다.

예행(자체 35개만, 합성 후보)은 `rehearsal-index`로 판본 3 모양의 합성 색인을 만들고
동결 생성기로 예행 명세를 만든 뒤 위 순서를 `--folds 0`으로 돌린다.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

import freeze_external_candidates as freeze
import judge_extended_stack as ladder

from pipeline import ensemble
from pipeline.data import ID, TARGET, TRAIN_PATH, file_sha256, labels
from pipeline.judgment import DUPLICATE_SPEARMAN, FOLDS_PATH
from pipeline.ledger import POOL_PATH, Pool
from pipeline.pool_audit import prediction_array_sha256
from pipeline.runs import MlflowRunStore

ISSUE = 486
SCHEMA = "strict-external-selection/1"
STRATEGY = "shrunk_rank_logit_logistic"
OUT_ROOT = Path("run-logs/strict-external-selection")
COMPARISON_MANIFEST_PATH = Path("docs/research/extended-stack-submission-2-manifest.json")
COMPARISON_EVIDENCE_PATH = Path("docs/research/extended-stack-ladder-2.json")
COMPARISON_CONFIG = "ablate_new_nhtquyn"
COMPARISON_MEMBER_COUNT = 313
OWN_MEMBER_COUNT = 35
TEST_PATH = Path("data/test.csv")
OWN_TEST_PATH = Path("artifacts/full-refit/member_test_cv_full.parquet")
FULL_REFIT_MANIFEST_PATH = Path("artifacts/full-refit/manifest.json")
SUBMISSION_DIR = Path("artifacts/submissions")
ENSEMBLE_SOURCE = Path(ensemble.__file__)

GATE_DELTA = 0.00002
FOLDS_REQUIRED_POSITIVE = 5
NOISE_FLOOR = 5.7e-06  # #386 결합 도구 잡음 바닥. 경계 보고에만 쓴다.
LAMBDA_GRID = ensemble.SHRINKAGE_LAMBDA_GRID
LOGIT_EPS = ensemble.LogisticLinearCombiner.LOGIT_EPS
META_MAX_ITER = 1000
N_TRAIN = 691369
N_TEST = 296302
ALL_FOLDS = (0, 1, 2, 3, 4)

SEARCH_RULES = {
    "start": "자체 35개만 담은 필수 시작 구성. 자체 구성원은 제거·교체하지 않는다.",
    "score": "열린 분할 안에서 한 분할씩 빼고 고정 결합기를 맞춰 예측을 이어붙인 AUC(검색 풀 점수).",
    "exclusion": f"외부 후보가 자체 구성원과 스피어만 {DUPLICATE_SPEARMAN} 이상이면 그 바깥 분할의 검색에서 제외.",
    "conflict": f"외부 후보끼리 스피어만 {DUPLICATE_SPEARMAN} 이상이면 상호 배타. 단독 성능으로 미리 빼지 않는다.",
    "invariant": f"외부 후보가 낀 모든 구성원 쌍이 스피어만 {DUPLICATE_SPEARMAN} 미만. 자체끼리의 쌍은 검색이 바꿀 수 없으므로 진단값으로만 기록한다(예행에서 exp131·exp157이 열린 4분할에서 0.9981).",
    "forward": "미선택 후보의 단일 추가와, 선택 후보 정확히 하나와 충돌하는 미선택 후보의 단일 원자 교체를 모두 평가해 AUC 차이가 가장 큰 엄격 양수 이동 하나를 받고 반복. 양수가 없으면 정지.",
    "backward": "선택 후보의 단일 제거를 모두 평가해 가장 큰 엄격 양수 이동 하나를 받고 반복.",
    "sequence": "순방향 수렴 → 후방 수렴 → 순방향 수렴 → 허용된 미선택 순서 없는 쌍 추가 1회 전수 평가 → (최선 쌍이 엄격 양수면 받고 순방향·후방·순방향 재수렴) → 종료.",
    "pair_allowed": "두 후보 모두 제외 대상이 아니고 선택 후보 누구와도, 서로도 충돌하지 않을 때만.",
    "tie": "최대 이동이 정확히 같으면 (들어오는 후보의 동결 순서, 나가는 후보의 동결 순서 없음<있음) 순으로 앞선 이동. 쌍은 동결 순서 사전식.",
    "lambda_tie": "λ 격자는 오름차순이라 AUC 동률이면 수축이 큰(작은 λ) 쪽.",
    "band": f"성능 동등 대역과 잡음 바닥 {NOISE_FLOOR}은 보고에만 쓰고 이동 승인 문턱을 바꾸지 않는다.",
}


# ---------------------------------------------------------------------------
# 공통 도우미


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
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def code_state() -> dict[str, object]:
    return {
        "git": git_state(),
        "script": {"path": str(Path(__file__).resolve().relative_to(Path.cwd())), "sha256": file_sha256(Path(__file__))},
        "freeze_script_sha256": file_sha256(Path(freeze.__file__)),
        "ensemble_module": {"path": str(ENSEMBLE_SOURCE), "sha256": file_sha256(ENSEMBLE_SOURCE)},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


class JudgmentError(RuntimeError):
    """판정을 계속할 수 없는 입력·상태 불일치. 전체 판정은 `판정 불가`로 둔다."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JudgmentError(message)


def run_dir_for(spec: dict) -> Path:
    return OUT_ROOT / spec["candidate_set_id"]


# ---------------------------------------------------------------------------
# 입력 적재


def load_folds_and_labels() -> tuple[pd.Series, pd.Series]:
    train = pd.read_csv(TRAIN_PATH)
    fold_of = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"]
    y = labels(fold_of.index)
    _require(len(train) == N_TRAIN, f"train 행 수 {len(train)}")
    _require(np.array_equal(fold_of.index.to_numpy(), train[ID].to_numpy()), "folds.parquet의 id 순서가 train.csv와 다르다.")
    return fold_of, y


def load_own(fold_of: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    pool = Pool.load()
    members = [(member.config, member.run_id) for member in pool.members]
    _require(len(members) == OWN_MEMBER_COUNT, f"후보 풀 {len(members)}구성원(기대 {OWN_MEMBER_COUNT})")
    matrix = ensemble.member_matrix(members, MlflowRunStore(), fold_of.index)
    rows = [
        {"column": config, "run_id": run_id, "oof_sha256": prediction_array_sha256(matrix[config])}
        for config, run_id in members
    ]
    return matrix, rows


def load_comparison_arm(own: pd.DataFrame, fold_of: pd.Series, y: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    """#457 manifest의 313구성원 OOF 행렬. 자체 35는 풀 순서, 외부는 판본 2 장부 경로."""
    manifest = read_json(COMPARISON_MANIFEST_PATH)
    _require(manifest["issue"] == 457 and len(manifest["members"]) == COMPARISON_MEMBER_COUNT, "비교 팔 manifest가 #457 313구성원이 아니다.")
    _, accepted = ladder.load_ledger()
    by_column = {f"ext_{row['member_id']}": row for row in accepted}
    label_values = y.to_numpy()
    columns: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for entry in manifest["members"]:
        column = entry["column"]
        if entry["origin"] == "own":
            _require(column in own.columns, f"비교 팔의 자체 구성원 {column}이 풀에 없다.")
            values = own[column].to_numpy(np.float64)
        else:
            ledger_row = by_column[column]
            _require(ledger_row["oof_path"] == entry["oof_path"], f"{column}: manifest와 장부의 OOF 경로가 다르다.")
            values = ladder.load_ledger_array(ledger_row["oof_path"])
            _require(len(values) == N_TRAIN and bool(np.isfinite(values).all()), f"{column}: 행 수 {len(values)} 또는 비유한값")
            delta = float(roc_auc_score(label_values, values)) - float(ledger_row["auc"])
            _require(abs(delta) < 1e-9, f"{column}: 장부 AUC와 {delta:+.2e} 차이")
        columns[column] = values
        rows.append({"column": column, "origin": entry["origin"], "oof_sha256": prediction_array_sha256(values), "test_path": entry["test"]["test_path"]})
    matrix = pd.DataFrame(columns, index=fold_of.index).astype(np.float64)
    _require(list(matrix.columns[:OWN_MEMBER_COUNT]) == list(own.columns), "비교 팔의 자체 35 순서가 풀과 다르다.")
    return matrix, rows


def candidate_column(member_id: str) -> str:
    return f"cand_{member_id}"


def load_candidates(spec: dict, fold_of: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    """동결 명세 순서의 후보 OOF 행렬. 배열 해시를 명세와 다시 대조한다."""
    columns: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for candidate in spec["candidates"]:
        oof = freeze.load_array(Path(candidate["oof_path"]), N_TRAIN, candidate["member_id"])
        test = freeze.load_array(Path(candidate["test_path"]), N_TEST, candidate["member_id"])
        _require(freeze.array_sha256(oof) == candidate["oof_sha256"], f"{candidate['member_id']}: OOF 해시가 명세와 다르다.")
        _require(freeze.array_sha256(test) == candidate["test_sha256"], f"{candidate['member_id']}: 시험 해시가 명세와 다르다.")
        _require(freeze.pair_sha256(oof, test) == candidate["pair_sha256"], f"{candidate['member_id']}: 쌍 해시가 명세와 다르다.")
        column = candidate_column(candidate["member_id"])
        columns[column] = oof
        rows.append({"order": candidate["order"], "member_id": candidate["member_id"], "column": column, "audit_record_id": candidate["audit_record_id"], "pair_sha256": candidate["pair_sha256"], "oof_sha256": candidate["oof_sha256"], "test_sha256": candidate["test_sha256"], "test_path": candidate["test_path"]})
    matrix = pd.DataFrame(columns, index=fold_of.index).astype(np.float64) if columns else pd.DataFrame(index=fold_of.index)
    return matrix, rows


# ---------------------------------------------------------------------------
# precommit


def precommit(args: argparse.Namespace) -> None:
    spec = freeze.verify_spec_file(args.spec)
    folds = tuple(args.folds) if args.folds else ALL_FOLDS
    _require(spec["rehearsal"] or folds == ALL_FOLDS, "실제 판정은 바깥 분할 5곳 전부를 써야 한다(--folds는 예행 전용).")
    run_dir = args.run_dir or run_dir_for(spec)
    _require(not (run_dir / "precommit.json").exists(), f"precommit.json이 이미 있다(변경 불가): {run_dir}")
    _require(spec["rehearsal"] or not git_state()["dirty"], "실제 판정은 커밋된 코드 상태에서만 시작한다(git dirty).")
    fold_of, y = load_folds_and_labels()
    fold_vector_sha = hashlib.sha256(fold_of.to_numpy(np.int8).tobytes()).hexdigest()
    _require(fold_vector_sha == spec["fold_spec"]["fold_vector_sha256"], "분할 벡터 해시가 동결 명세와 다르다.")

    own, own_rows = load_own(fold_of)
    comparison, comparison_rows = load_comparison_arm(own, fold_of, y)
    candidates, candidate_rows = load_candidates(spec, fold_of)
    evidence = read_json(COMPARISON_EVIDENCE_PATH)
    reference = evidence["configs"][COMPARISON_CONFIG]["strategies"][STRATEGY]
    _require(evidence["selected_config"] == COMPARISON_CONFIG and reference["member_count"] == COMPARISON_MEMBER_COUNT, "#455 근거의 선택 구성이 313구성원이 아니다.")

    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    own.to_parquet(cache_dir / "own35-oof.parquet")
    comparison.to_parquet(cache_dir / "comparison-arm-oof.parquet")
    candidates.to_parquet(cache_dir / "candidates-oof.parquet")
    caches = {name: file_sha256(cache_dir / name) for name in ("own35-oof.parquet", "comparison-arm-oof.parquet", "candidates-oof.parquet")}

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "candidate_set_id": spec["candidate_set_id"],
        "rehearsal": spec["rehearsal"],
        "created_at": now_iso(),
        "outer_folds": list(folds),
        "freeze_spec": {
            "path": str(args.spec),
            "sha256": spec["spec_sha256"],
            "content_sha256": spec["content_sha256"],
            "survey_cutoff": spec["survey_cutoff"],
            "contract_version": spec["contract_version"],
            "candidate_count": spec["candidate_count"],
            "candidates": [{"order": c["order"], "member_id": c["member_id"], "audit_record_id": c["audit_record_id"], "pair_sha256": c["pair_sha256"]} for c in spec["candidates"]],
            "user_exclusions": spec["user_exclusions"],
        },
        "inputs": {
            "train": {"path": str(TRAIN_PATH), "sha256": file_sha256(TRAIN_PATH)},
            "folds": {"path": str(FOLDS_PATH), "sha256": file_sha256(FOLDS_PATH), "fold_vector_sha256": fold_vector_sha, "rows_per_fold": {str(k): int((fold_of == k).sum()) for k in ALL_FOLDS}},
        },
        "own_start": {
            "pool_path": str(POOL_PATH),
            "pool_sha256": file_sha256(POOL_PATH),
            "member_count": len(own_rows),
            "members": own_rows,
            "composition_sha256": canonical_sha256([(r["column"], r["oof_sha256"]) for r in own_rows]),
        },
        "comparison_arm": {
            "manifest_path": str(COMPARISON_MANIFEST_PATH),
            "manifest_sha256": file_sha256(COMPARISON_MANIFEST_PATH),
            "v2_ledger_sha256": file_sha256(ladder.LEDGER_PATH),
            "member_count": len(comparison_rows),
            "members": comparison_rows,
            "composition_sha256": canonical_sha256([(r["column"], r["oof_sha256"]) for r in comparison_rows]),
            "reference": {"evidence_path": str(COMPARISON_EVIDENCE_PATH), "evidence_sha256": file_sha256(COMPARISON_EVIDENCE_PATH), "config": COMPARISON_CONFIG, "nested_auc": reference["nested_auc"], "fold_aucs": reference["fold_aucs"]},
        },
        "candidate_arm": {
            "columns": list(own.columns) + list(candidates.columns),
            "candidates": candidate_rows,
            "composition_sha256": canonical_sha256([(r["column"], r["oof_sha256"]) for r in candidate_rows]),
        },
        "caches": caches,
        "combiner": {
            "name": STRATEGY,
            "lambda_grid": list(LAMBDA_GRID),
            "meta": {"representation": "rank_logit", "C": 1.0, "solver": "lbfgs", "max_iter": META_MAX_ITER, "random_state": 0, "logit_eps": LOGIT_EPS},
            "candidate_arm_implementation": "학습 분할 집합별 순위 특성 캐시를 쓰는 같은 계산(봉인 분할 예측에서 등록 결합기와 대조)",
            "comparison_arm_implementation": "pipeline.ensemble.COMBINER_REGISTRY 등록 결합기",
        },
        "search_rules": SEARCH_RULES,
        "gate": {"delta_required": GATE_DELTA, "folds_required_positive": FOLDS_REQUIRED_POSITIVE, "comparison": "두 팔의 봉인 분할 예측을 원래 행 순서로 이어붙여 직접 채점한 AUC 차이", "public_score_used": False},
        "noise_floor": NOISE_FLOOR,
        "rules": {
            "failure": "계산 하나라도 실패하거나 끝나지 않으면 완료한 일부 결과를 쓰지 않고 전체를 판정 불가로 둔다.",
            "resume": "모든 입력 해시와 코드 상태가 precommit과 정확히 같을 때만 재개한다.",
            "full_selection": "절차 통과 뒤에만 같은 동결 검색을 전체 OOF에 한 번 적용한다. 전체 OOF 점수로 문턱을 다시 판정하지 않는다.",
            "upload": "문턱 통과 뒤 Kaggle 업로드와 최종 두 장 수동 고정은 사용자 승인 뒤에만 한다.",
        },
        "code_state": code_state(),
    }
    payload["precommit_sha256"] = canonical_sha256(payload)
    write_json(run_dir / "precommit.json", payload)
    print(f"precommit 저장: {run_dir / 'precommit.json'}\n  후보 {len(candidate_rows)}개, 비교 팔 {len(comparison_rows)}, 자체 {len(own_rows)}, 분할 {list(folds)}, precommit_sha256 {payload['precommit_sha256']}")


def load_precommit(run_dir: Path) -> dict:
    """precommit을 읽고 입력 해시와 코드 상태가 지금과 정확히 같은지 확인한다."""
    path = run_dir / "precommit.json"
    _require(path.is_file(), f"precommit.json이 없다: {run_dir}")
    payload = read_json(path)
    recorded = payload["precommit_sha256"]
    _require(canonical_sha256({k: v for k, v in payload.items() if k != "precommit_sha256"}) == recorded, "precommit.json이 제자리에서 바뀌었다.")
    spec = freeze.verify_spec_file(Path(payload["freeze_spec"]["path"]))
    _require(spec["spec_sha256"] == payload["freeze_spec"]["sha256"], "동결 명세가 precommit과 다르다.")
    for key in ("train", "folds"):
        _require(file_sha256(Path(payload["inputs"][key]["path"])) == payload["inputs"][key]["sha256"], f"{key} 해시가 precommit과 다르다.")
    _require(file_sha256(POOL_PATH) == payload["own_start"]["pool_sha256"], "pool.yaml이 precommit과 다르다.")
    _require(file_sha256(COMPARISON_MANIFEST_PATH) == payload["comparison_arm"]["manifest_sha256"], "비교 팔 manifest가 precommit과 다르다.")
    for name, digest in payload["caches"].items():
        _require(file_sha256(run_dir / "cache" / name) == digest, f"캐시 {name}이 precommit과 다르다.")
    state = code_state()
    for label, actual, expected in (
        ("판정 도구", state["script"]["sha256"], payload["code_state"]["script"]["sha256"]),
        ("동결 생성기", state["freeze_script_sha256"], payload["code_state"]["freeze_script_sha256"]),
        ("결합기 module", state["ensemble_module"]["sha256"], payload["code_state"]["ensemble_module"]["sha256"]),
        ("git commit", state["git"]["commit"], payload["code_state"]["git"]["commit"]),
    ):
        _require(actual == expected, f"코드 상태({label})가 precommit과 다르다. precommit부터 다시 한다.")
    return payload


def load_arm_matrices(run_dir: Path, payload: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = run_dir / "cache"
    own = pd.read_parquet(cache / "own35-oof.parquet")
    candidates = pd.read_parquet(cache / "candidates-oof.parquet")
    _require(list(own.columns) + list(candidates.columns) == payload["candidate_arm"]["columns"], "캐시 열 순서가 precommit과 다르다.")
    return own, candidates, cache / "comparison-arm-oof.parquet"


# ---------------------------------------------------------------------------
# 후보 절차 팔의 빠른 shrunk_rank_logit_logistic


class FastShrunk:
    """열 부분집합에 대해 등록 결합기와 같은 계산을 반복하는 구현.

    values: (n, m) float64 행렬(자체 35 뒤에 후보). fold: (n,) 분할. y: (n,).
    학습 분할 집합 T마다 모든 열의 순위 특성(EmpiricalCDF, uniform)을 전 행에 대해
    한 번 만들어 캐시한다. 로짓 특성은 학습 집합과 무관하다.
    """

    def __init__(self, values: np.ndarray, fold: np.ndarray, y: np.ndarray) -> None:
        self.values = np.ascontiguousarray(values, dtype=np.float64)
        self.fold = fold
        self.y = y
        self.logit = ensemble._logit(pd.DataFrame(self.values), LOGIT_EPS)
        self._ranks: dict[frozenset[int], np.ndarray] = {}
        self._fold_rows = {int(k): np.flatnonzero(fold == k) for k in np.unique(fold)}
        self.meta_fits = 0

    def rows_of(self, folds: frozenset[int]) -> np.ndarray:
        # 원래 행 순서를 지킨다. 등록 결합기는 `preds[mask]`로 학습하므로 행 순서까지 같아야
        # 로지스틱 회귀(lbfgs)가 같은 경로로 같은 계수에 닿는다.
        return np.sort(np.concatenate([self._fold_rows[k] for k in sorted(folds)]))

    def ranks(self, train_folds: frozenset[int]) -> np.ndarray:
        cached = self._ranks.get(train_folds)
        if cached is None:
            train_rows = self.rows_of(train_folds)
            transformer = ensemble.EmpiricalCDFTransformer("uniform").fit(self.values[train_rows])
            cached = transformer.transform(self.values)
            self._ranks[train_folds] = cached
        return cached

    def _features(self, train_folds: frozenset[int], rows: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
        ranks = self.ranks(train_folds)
        columns = list(subset)
        return np.column_stack((ranks[np.ix_(rows, columns)], self.logit[np.ix_(rows, columns)]))

    def fit_meta(self, train_folds: frozenset[int], subset: tuple[int, ...]) -> tuple[StandardScaler, LogisticRegression]:
        rows = self.rows_of(train_folds)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self._features(train_folds, rows, subset)).astype(np.float64, copy=False)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=META_MAX_ITER, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(scaled, self.y[rows])
        iterations = int(np.max(model.n_iter_))
        if iterations >= META_MAX_ITER:
            raise ensemble.CombinerConvergenceError(f"max(n_iter_)={iterations}, max_iter={META_MAX_ITER}")
        self.meta_fits += 1
        return scaler, model

    def predict_meta(self, fitted: tuple[StandardScaler, LogisticRegression], train_folds: frozenset[int], rows: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
        scaler, model = fitted
        scaled = scaler.transform(self._features(train_folds, rows, subset))
        return model.predict_proba(scaled)[:, 1].astype(np.float64, copy=False)

    def rank_mean(self, rows: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
        return ensemble.rank_mean(pd.DataFrame(self.values[np.ix_(rows, list(subset))]))

    @staticmethod
    def shrunk(meta_prediction: np.ndarray, block_rank_mean: np.ndarray, shrinkage_lambda: float) -> np.ndarray:
        meta_ranks = pd.Series(meta_prediction).rank(pct=True).to_numpy(dtype=np.float64)
        return shrinkage_lambda * meta_ranks + (1.0 - shrinkage_lambda) * block_rank_mean

    def fit_predict(self, train_folds: frozenset[int], predict_rows: np.ndarray, subset: tuple[int, ...], block_rank_mean: np.ndarray) -> tuple[np.ndarray, float, list[float]]:
        """등록 결합기의 fit(train_folds 행) → predict(predict_rows)와 같은 계산.

        λ는 train_folds 안의 leave-one-fold-out으로 고르고, 최종 메타는 train_folds
        전체로 맞춘다. (예측, λ, λ별 LOFO AUC)를 돌려준다.
        """
        train_rows = self.rows_of(train_folds)
        position = np.full(len(self.values), -1, dtype=np.int64)
        position[train_rows] = np.arange(len(train_rows))
        combined = {shrinkage_lambda: np.full(len(train_rows), np.nan) for shrinkage_lambda in LAMBDA_GRID}
        for held in sorted(train_folds):
            meta_folds = train_folds - {held}
            fitted = self.fit_meta(meta_folds, subset)
            block = self._fold_rows[held]
            meta_prediction = self.predict_meta(fitted, meta_folds, block, subset)
            block_mean = self.rank_mean(block, subset)
            slots = position[block]
            for shrinkage_lambda in LAMBDA_GRID:
                combined[shrinkage_lambda][slots] = self.shrunk(meta_prediction, block_mean, shrinkage_lambda)
        train_y = self.y[train_rows]
        aucs = [float(roc_auc_score(train_y, combined[shrinkage_lambda])) for shrinkage_lambda in LAMBDA_GRID]
        best_lambda = float(LAMBDA_GRID[int(np.argmax(aucs))])
        final = self.fit_meta(train_folds, subset)
        meta_prediction = self.predict_meta(final, train_folds, predict_rows, subset)
        return self.shrunk(meta_prediction, block_rank_mean, best_lambda), best_lambda, aucs

    def pool_score(self, open_folds: frozenset[int], subset: tuple[int, ...]) -> tuple[float, dict[str, float]]:
        """검색 풀 점수: 열린 분할 안에서 한 분할씩 빼고 예측을 이어붙인 AUC."""
        open_rows = self.rows_of(open_folds)
        prediction = np.full(len(self.values), np.nan)
        lambdas: dict[str, float] = {}
        for held in sorted(open_folds):
            block = self._fold_rows[held]
            block_mean = self.rank_mean(block, subset)
            prediction[block], best_lambda, _ = self.fit_predict(open_folds - {held}, block, subset, block_mean)
            lambdas[str(held)] = best_lambda
        return float(roc_auc_score(self.y[open_rows], prediction[open_rows])), lambdas


# ---------------------------------------------------------------------------
# 결정적 검색


class Search:
    """ADR-0005의 허용 이동과 정지 규칙을 그대로 실행하고 모든 평가를 기록한다."""

    def __init__(self, engine: FastShrunk, open_folds: frozenset[int], own_count: int, candidate_order: list[int], excluded: set[int], conflicts: dict[int, set[int]], log_path: Path | None) -> None:
        self.engine = engine
        self.open_folds = open_folds
        self.own = tuple(range(own_count))
        self.order = candidate_order  # 후보 열 인덱스, 동결 순서
        self.rank_of = {column: i for i, column in enumerate(candidate_order)}
        self.excluded = excluded
        self.conflicts = conflicts
        self.log_path = log_path
        self.selected: list[int] = []
        self.scores: dict[tuple[int, ...], float] = {}
        self.stages: list[dict] = []
        self.evaluations = 0
        self.started = time.monotonic()

    # 상태 ------------------------------------------------------------------

    def subset(self, selected: list[int]) -> tuple[int, ...]:
        return self.own + tuple(sorted(selected, key=self.rank_of.__getitem__))

    def score(self, selected: list[int]) -> float:
        key = self.subset(selected)
        if key not in self.scores:
            self.scores[key], _ = self.engine.pool_score(self.open_folds, key)
            self.evaluations += 1
        return self.scores[key]

    def _log(self, record: dict) -> None:
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _unselected(self) -> list[int]:
        chosen = set(self.selected)
        return [c for c in self.order if c not in chosen and c not in self.excluded]

    def _conflicting_selected(self, candidate: int) -> list[int]:
        return [s for s in self.selected if s in self.conflicts.get(candidate, set())]

    # 단계 ------------------------------------------------------------------

    def _step(self, stage: str, moves: list[dict]) -> dict | None:
        """이동 목록을 모두 평가하고 가장 큰 엄격 양수 이동을 받는다."""
        current = self.score(self.selected)
        evaluated = []
        for move in moves:
            started = time.monotonic()
            new_score = self.score(move["selected"])
            entry = {**{k: v for k, v in move.items() if k != "selected"}, "score": new_score, "delta": new_score - current, "seconds": round(time.monotonic() - started, 1)}
            evaluated.append(entry)
            self._log({"stage": stage, "current": current, **entry, "elapsed": round(time.monotonic() - self.started)})
        accepted = None
        best = None
        for move, entry in zip(moves, evaluated, strict=True):
            if entry["delta"] <= 0.0:
                continue
            key = (entry["delta"], -move["tie"][0], -move["tie"][1])
            if best is None or key > best[0]:
                best = (key, move, entry)
        if best is not None:
            _, move, entry = best
            self.selected = list(move["selected"])
            accepted = entry
        self.stages.append({"stage": stage, "start_score": current, "evaluated": evaluated, "accepted": accepted, "selected_after": [c for c in self.selected]})
        return accepted

    def forward(self, label: str) -> None:
        step = 0
        while True:
            step += 1
            moves = []
            for candidate in self._unselected():
                conflicting = self._conflicting_selected(candidate)
                if not conflicting:
                    moves.append({"move": "add", "incoming": candidate, "outgoing": None, "selected": self.selected + [candidate], "tie": (self.rank_of[candidate], -1)})
                elif len(conflicting) == 1:
                    outgoing = conflicting[0]
                    moves.append({"move": "swap", "incoming": candidate, "outgoing": outgoing, "selected": [s for s in self.selected if s != outgoing] + [candidate], "tie": (self.rank_of[candidate], self.rank_of[outgoing])})
            if not moves or self._step(f"{label}:forward:{step}", moves) is None:
                return

    def backward(self, label: str) -> None:
        step = 0
        while True:
            step += 1
            moves = [{"move": "remove", "incoming": None, "outgoing": s, "selected": [x for x in self.selected if x != s], "tie": (self.rank_of[s], -1)} for s in self.selected]
            if not moves or self._step(f"{label}:backward:{step}", moves) is None:
                return

    def pair_sweep(self) -> bool:
        allowed = [c for c in self._unselected() if not self._conflicting_selected(c)]
        moves = []
        for first, second in itertools.combinations(allowed, 2):
            if second in self.conflicts.get(first, set()):
                continue
            moves.append({"move": "pair_add", "incoming": [first, second], "outgoing": None, "selected": self.selected + [first, second], "tie": (self.rank_of[first], self.rank_of[second])})
        if not moves:
            self.stages.append({"stage": "pair_sweep", "start_score": self.score(self.selected), "evaluated": [], "accepted": None, "selected_after": list(self.selected)})
            return False
        return self._step("pair_sweep", moves) is not None

    def run(self) -> str:
        self.score(self.selected)
        self.forward("phase1")
        self.backward("phase1")
        self.forward("phase1b")
        if self.pair_sweep():
            self.forward("phase2")
            self.backward("phase2")
            self.forward("phase2b")
            return "pair_accepted_then_converged"
        return "no_positive_pair"


# ---------------------------------------------------------------------------
# 분할별 작업


def spearman_matrix(values: np.ndarray) -> np.ndarray:
    ranks = pd.DataFrame(values).rank().to_numpy(dtype=np.float64)
    return np.corrcoef(ranks.T)


def select_fold(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    sealed = int(args.fold)
    _require(sealed in payload["outer_folds"], f"분할 {sealed}은 precommit의 바깥 분할이 아니다.")
    fold_dir = run_dir / f"fold-{sealed}"
    out_path = fold_dir / "selection.json"
    _require(args.force or not out_path.exists(), f"이미 있다: {out_path}")
    fold_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    fold_of, y = load_folds_and_labels()
    own, candidates, _ = load_arm_matrices(run_dir, payload)
    columns = list(own.columns) + list(candidates.columns)
    values = np.ascontiguousarray(pd.concat([own, candidates], axis=1).to_numpy(np.float64))
    fold = fold_of.to_numpy(np.int64)
    label_values = y.to_numpy()
    open_folds = frozenset(ALL_FOLDS) - {sealed}
    engine = FastShrunk(values, fold, label_values)
    open_rows = engine.rows_of(open_folds)
    sealed_rows = engine.rows_of(frozenset({sealed}))
    own_count = own.shape[1]
    candidate_indices = list(range(own_count, len(columns)))
    print(f"=== 봉인 {sealed}: 열린 {sorted(open_folds)} {len(open_rows)}행, 후보 {len(candidate_indices)}개 ===", flush=True)

    # 진단값: 단독 AUC, 스피어만, 제외, 충돌 (열린 행만)
    diagnostics = diagnose(values, label_values, open_rows, columns, own_count, candidate_indices)
    excluded = {c for c, entry in diagnostics["candidates"].items() if entry["excluded_for_own_conflict"]}
    conflicts = {c: set(entry["conflicts"]) for c, entry in diagnostics["candidates"].items()}
    log_path = fold_dir / "progress.jsonl"
    if log_path.exists():
        log_path.unlink()
    search = Search(engine, open_folds, own_count, candidate_indices, excluded, conflicts, log_path)
    stop_reason = search.run()
    final_subset = search.subset(search.selected)
    invariant = check_invariant(diagnostics["spearman"], final_subset, own_count)
    print(f"  검색 종료({stop_reason}): 평가 {search.evaluations}회, 선택 {[columns[c] for c in search.selected]}, 점수 {search.scores[final_subset]:.7f}", flush=True)

    # 잠근 명단으로 봉인 분할 예측(빠른 구현)과 등록 결합기 대조
    block_mean = engine.rank_mean(sealed_rows, final_subset)
    prediction, best_lambda, lambda_aucs = engine.fit_predict(open_folds, sealed_rows, final_subset, block_mean)
    registered_seconds = None
    registered_max_abs_diff = None
    registered_lambda = None
    if not args.skip_registered_check:
        started_registered = time.monotonic()
        frame = pd.DataFrame(values[:, list(final_subset)], index=fold_of.index, columns=[columns[c] for c in final_subset])
        open_mask = np.isin(fold, sorted(open_folds))
        fitted = ensemble.COMBINER_REGISTRY[STRATEGY].fit(frame[open_mask], y[open_mask])
        registered = np.asarray(fitted.predict(frame[~open_mask]), dtype=np.float64)
        registered_lambda = float(fitted.shrinkage_lambda)
        registered_max_abs_diff = float(np.max(np.abs(registered - prediction)))
        registered_seconds = time.monotonic() - started_registered
        _require(registered_lambda == best_lambda and registered_max_abs_diff <= 1e-9, f"빠른 구현과 등록 결합기의 봉인 예측이 다르다: λ {best_lambda} vs {registered_lambda}, 최대 차이 {registered_max_abs_diff:.2e}")
    pd.DataFrame({ID: fold_of.index.to_numpy()[sealed_rows], "prediction": prediction}).to_parquet(fold_dir / "predictions.parquet", index=False)

    record = {
        "schema": SCHEMA,
        "candidate_set_id": payload["candidate_set_id"],
        "precommit_sha256": payload["precommit_sha256"],
        "sealed_fold": sealed,
        "open_folds": sorted(open_folds),
        "open_rows": int(len(open_rows)),
        "sealed_rows": int(len(sealed_rows)),
        "columns": columns,
        "own_count": own_count,
        "diagnostics": {k: v for k, v in diagnostics.items() if k != "spearman"},
        "start": {"members": list(own.columns), "score": search.scores[search.subset([])]},
        "stages": [named_stage(stage, columns) for stage in search.stages],
        "evaluations": search.evaluations,
        "meta_fits": engine.meta_fits,
        "stop_reason": stop_reason,
        "final_selected": [columns[c] for c in sorted(search.selected, key=search.rank_of.__getitem__)],
        "final_selected_member_ids": [columns[c][len("cand_"):] for c in sorted(search.selected, key=search.rank_of.__getitem__)],
        "final_members": [columns[c] for c in final_subset],
        "final_score": search.scores[final_subset],
        "invariant": invariant,
        "sealed_prediction": {
            "lambda": best_lambda,
            "lofo_lambda_aucs": dict(zip((str(v) for v in LAMBDA_GRID), lambda_aucs, strict=True)),
            "registered_lambda": registered_lambda,
            "registered_max_abs_diff": registered_max_abs_diff,
            "registered_seconds": registered_seconds,
            "prediction_sha256": prediction_array_sha256(prediction),
        },
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": now_iso(),
    }
    _require(invariant["ok"], f"후보 풀 중복 불변식 위반: {invariant['violations']}")
    write_json(out_path, record)
    print(f"  저장: {out_path} ({record['elapsed_seconds']:.0f}s, 등록 결합기 최대 차이 {registered_max_abs_diff})", flush=True)


def diagnose(values: np.ndarray, label_values: np.ndarray, rows: np.ndarray, columns: list[str], own_count: int, candidate_indices: list[int]) -> dict:
    block = values[rows]
    y = label_values[rows]
    spearman = spearman_matrix(block)
    own_pairs = [(columns[i], columns[j], float(spearman[i, j])) for i in range(own_count) for j in range(i + 1, own_count) if spearman[i, j] >= DUPLICATE_SPEARMAN]
    entries: dict[int, dict] = {}
    for c in candidate_indices:
        own_rho = spearman[c, :own_count]
        closest = int(np.argmax(own_rho))
        conflicts = [d for d in candidate_indices if d != c and spearman[c, d] >= DUPLICATE_SPEARMAN]
        entries[c] = {
            "column": columns[c],
            "standalone_auc": float(roc_auc_score(y, block[:, c])),
            "spearman_vs_own_max": float(own_rho[closest]),
            "spearman_vs_own_closest": columns[closest],
            "excluded_for_own_conflict": bool(own_rho[closest] >= DUPLICATE_SPEARMAN),
            "conflicts": conflicts,
            "conflict_columns": [columns[d] for d in conflicts],
            "spearman_vs_candidates_max": float(max((spearman[c, d] for d in candidate_indices if d != c), default=float("nan"))),
        }
    return {
        "rows": int(len(rows)),
        "duplicate_spearman": DUPLICATE_SPEARMAN,
        "own_standalone_auc": {columns[i]: float(roc_auc_score(y, block[:, i])) for i in range(own_count)},
        "own_pairs_max_spearman": float(max(spearman[i, j] for i in range(own_count) for j in range(i + 1, own_count))) if own_count > 1 else None,
        "own_pairs_at_or_above_threshold": own_pairs,
        "candidates": entries,
        "spearman": spearman,
    }


def check_invariant(spearman: np.ndarray, subset: tuple[int, ...], own_count: int) -> dict:
    """후보 풀 중복 불변식: 외부 후보가 낀 쌍만 본다. 자체끼리의 쌍은 diagnostics에 남긴다."""
    pairs = [(i, j) for i, j in itertools.combinations(subset, 2) if j >= own_count]
    violations = [(int(i), int(j), float(spearman[i, j])) for i, j in pairs if spearman[i, j] >= DUPLICATE_SPEARMAN]
    return {"ok": not violations, "violations": violations, "max_pair_spearman_with_candidate": float(max((spearman[i, j] for i, j in pairs), default=float("nan")))}


def named_stage(stage: dict, columns: list[str]) -> dict:
    def name(value: object) -> object:
        if value is None:
            return None
        if isinstance(value, list):
            return [columns[v] for v in value]
        return columns[int(value)]

    def entry(move: dict) -> dict:
        return {**move, "incoming": name(move.get("incoming")), "outgoing": name(move.get("outgoing"))}

    return {
        "stage": stage["stage"],
        "start_score": stage["start_score"],
        "evaluated": [entry(move) for move in stage["evaluated"]],
        "accepted": None if stage["accepted"] is None else entry(stage["accepted"]),
        "selected_after": [columns[c] for c in stage["selected_after"]],
    }


def baseline_fold(args: argparse.Namespace) -> None:
    """비교 팔 313의 봉인 분할 예측. 등록 결합기를 evaluate_nested의 분할 하나와 같게 쓴다."""
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    sealed = int(args.fold)
    _require(sealed in payload["outer_folds"], f"분할 {sealed}은 precommit의 바깥 분할이 아니다.")
    fold_dir = run_dir / f"fold-{sealed}"
    out_path = fold_dir / "baseline-predictions.parquet"
    _require(args.force or not out_path.exists(), f"이미 있다: {out_path}")
    fold_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fold_of, y = load_folds_and_labels()
    _, _, comparison_path = load_arm_matrices(run_dir, payload)
    preds = pd.read_parquet(comparison_path).astype(np.float64)
    _require(list(preds.columns) == [m["column"] for m in payload["comparison_arm"]["members"]], "비교 팔 열 순서가 precommit과 다르다.")
    inner = (fold_of != sealed).to_numpy()
    outer = (fold_of == sealed).to_numpy()
    print(f"=== 비교 팔 {preds.shape[1]}구성원, 봉인 {sealed} ===", flush=True)
    fitted = ensemble.COMBINER_REGISTRY[STRATEGY].fit(preds[inner], y[inner])
    prediction = np.asarray(fitted.predict(preds[outer]), dtype=np.float64)
    _require(prediction.shape == (int(outer.sum()),) and bool(np.isfinite(prediction).all()), "비교 팔 예측이 유한하지 않다.")
    pd.DataFrame({ID: fold_of.index.to_numpy()[outer], "prediction": prediction}).to_parquet(out_path, index=False)
    write_json(fold_dir / "baseline.json", {
        "schema": SCHEMA,
        "candidate_set_id": payload["candidate_set_id"],
        "precommit_sha256": payload["precommit_sha256"],
        "sealed_fold": sealed,
        "member_count": int(preds.shape[1]),
        "composition_sha256": payload["comparison_arm"]["composition_sha256"],
        "lambda": float(fitted.shrinkage_lambda),
        "prediction_sha256": prediction_array_sha256(prediction),
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": now_iso(),
    })
    print(f"  저장: {out_path} ({time.monotonic() - started:.0f}s, λ={fitted.shrinkage_lambda})", flush=True)


# ---------------------------------------------------------------------------
# 작업 실행기


def jobs_for(payload: dict) -> list[tuple[str, int]]:
    return [("baseline", k) for k in payload["outer_folds"]] + [("select", k) for k in payload["outer_folds"]]


def job_output(run_dir: Path, kind: str, fold: int) -> Path:
    return run_dir / f"fold-{fold}" / ("baseline-predictions.parquet" if kind == "baseline" else "selection.json")


def _running_jobs(run_dir: Path) -> set[tuple[str, int]]:
    listing = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True, check=False).stdout
    pattern = re.compile(rf"judge_strict_external_selection\.py (baseline|select) --run-dir {re.escape(str(run_dir))} --fold (\d+)")
    return {(kind, int(fold)) for kind, fold in pattern.findall(listing)}


def run_jobs(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    pending = [(kind, fold) for kind, fold in jobs_for(payload) if not job_output(run_dir, kind, fold).exists()]
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = str(args.threads)
    print(f"남은 작업 {len(pending)}개, 동시 상한 {args.workers}(무거운 baseline {args.heavy_workers}), 스레드 {args.threads}", flush=True)
    active: dict[tuple[str, int], subprocess.Popen] = {}
    results: dict[tuple[str, int], int] = {}
    while pending or active:
        for job, process in list(active.items()):
            code = process.poll()
            if code is not None:
                results[job] = code
                del active[job]
                print(f"{'완료' if code == 0 else f'실패({code})'} {job[0]} fold {job[1]}", flush=True)
        running = _running_jobs(run_dir) | set(active)
        heavy = sum(1 for kind, _ in running if kind == "baseline")
        while pending and len(running) < args.workers:
            kind, fold = pending[0]
            if kind == "baseline" and heavy >= args.heavy_workers:
                # 무거운 작업이 상한이면 가벼운 작업을 먼저 띄운다.
                light = next((job for job in pending if job[0] == "select"), None)
                if light is None:
                    break
                pending.remove(light)
                kind, fold = light
            else:
                pending.pop(0)
            if (kind, fold) in running:
                continue
            command = [sys.executable, __file__, kind, "--run-dir", str(run_dir), "--fold", str(fold)]
            handle = (log_dir / f"{kind}-fold-{fold}.log").open("w")
            active[(kind, fold)] = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.add((kind, fold))
            heavy += kind == "baseline"
            print(f"시작 {kind} fold {fold}", flush=True)
        time.sleep(15)
    failed = [job for job, code in results.items() if code != 0]
    if failed:
        sys.exit(f"실패한 작업 {failed}. 전체 판정은 판정 불가다. 로그: {log_dir}")
    print("all jobs finished", flush=True)


# ---------------------------------------------------------------------------
# 비교·안정성·전체 OOF·보고


def _load_fold_outputs(run_dir: Path, payload: dict) -> tuple[dict[int, dict], dict[int, dict]]:
    selections: dict[int, dict] = {}
    baselines: dict[int, dict] = {}
    for fold in payload["outer_folds"]:
        fold_dir = run_dir / f"fold-{fold}"
        for kind, store, name in (("selection", selections, "selection.json"), ("baseline", baselines, "baseline.json")):
            path = fold_dir / name
            _require(path.is_file(), f"분할 {fold}의 {kind} 산출물이 없다. 전체 판정은 판정 불가다.")
            record = read_json(path)
            _require(record["precommit_sha256"] == payload["precommit_sha256"], f"분할 {fold}의 {kind} 산출물이 다른 precommit에서 나왔다.")
            store[fold] = record
    return selections, baselines


def compare(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    fold_of, y = load_folds_and_labels()
    selections, baselines = _load_fold_outputs(run_dir, payload)
    folds = payload["outer_folds"]
    candidate = pd.Series(np.nan, index=fold_of.index)
    baseline = pd.Series(np.nan, index=fold_of.index)
    per_fold: dict[str, dict] = {}
    for fold in folds:
        fold_dir = run_dir / f"fold-{fold}"
        cand = pd.read_parquet(fold_dir / "predictions.parquet").set_index(ID)["prediction"]
        base = pd.read_parquet(fold_dir / "baseline-predictions.parquet").set_index(ID)["prediction"]
        ids = fold_of.index[(fold_of == fold).to_numpy()]
        _require(cand.index.equals(pd.Index(ids)) and base.index.equals(pd.Index(ids)), f"분할 {fold}의 예측 id가 분할 배정과 다르다.")
        _require(prediction_array_sha256(cand.to_numpy()) == selections[fold]["sealed_prediction"]["prediction_sha256"], f"분할 {fold}의 후보 팔 예측 해시가 selection.json과 다르다.")
        _require(prediction_array_sha256(base.to_numpy()) == baselines[fold]["prediction_sha256"], f"분할 {fold}의 비교 팔 예측 해시가 baseline.json과 다르다.")
        candidate.loc[ids] = cand.to_numpy()
        baseline.loc[ids] = base.to_numpy()
        fold_y = y.loc[ids].to_numpy()
        cand_auc = float(roc_auc_score(fold_y, cand.to_numpy()))
        base_auc = float(roc_auc_score(fold_y, base.to_numpy()))
        per_fold[str(fold)] = {"candidate_auc": cand_auc, "baseline_auc": base_auc, "delta": cand_auc - base_auc, "candidate_lambda": selections[fold]["sealed_prediction"]["lambda"], "baseline_lambda": baselines[fold]["lambda"], "selected": selections[fold]["final_selected_member_ids"]}
    complete = list(folds) == list(ALL_FOLDS)
    mask = candidate.notna().to_numpy()
    _require(bool(baseline.notna().to_numpy()[mask].all()), "두 팔의 예측 행이 다르다.")
    yy = y.to_numpy()[mask]
    cand_auc = float(roc_auc_score(yy, candidate.to_numpy()[mask]))
    base_auc = float(roc_auc_score(yy, baseline.to_numpy()[mask]))
    delta = cand_auc - base_auc
    positives = sum(entry["delta"] > 0.0 for entry in per_fold.values())
    reference = payload["comparison_arm"]["reference"]
    reproduction = {
        "reference_nested_auc": reference["nested_auc"],
        "baseline_concatenated_auc": base_auc,
        "delta": base_auc - reference["nested_auc"] if complete else None,
        "within_noise_floor": (abs(base_auc - reference["nested_auc"]) <= NOISE_FLOOR) if complete else None,
        "fold_deltas_vs_reference": {k: per_fold[k]["baseline_auc"] - reference["fold_aucs"][k] for k in per_fold},
    }
    if complete:
        passes = bool(delta >= GATE_DELTA and positives >= FOLDS_REQUIRED_POSITIVE)
        verdict = "통과" if passes else "미달"
    else:
        passes = False
        verdict = "판정 불가(부분 분할, 예행)"
    record = {
        "schema": SCHEMA,
        "candidate_set_id": payload["candidate_set_id"],
        "precommit_sha256": payload["precommit_sha256"],
        "rehearsal": payload["rehearsal"],
        "outer_folds": folds,
        "complete": complete,
        "gate": payload["gate"],
        "candidate_arm_auc": cand_auc,
        "comparison_arm_auc": base_auc,
        "delta": delta,
        "folds_positive": positives,
        "fold_results": per_fold,
        "passes": passes,
        "verdict": verdict,
        "boundary_report": {"noise_floor": NOISE_FLOOR, "delta_within_noise_floor": abs(delta) <= NOISE_FLOOR, "delta_minus_gate": delta - GATE_DELTA},
        "comparison_arm_reproduction": reproduction,
        "rows_scored": int(mask.sum()),
        "compared_at": now_iso(),
    }
    write_json(run_dir / "nested-comparison.json", record)
    stability(run_dir, payload, selections)
    print(f"후보 팔 {cand_auc:.7f} vs 비교 팔 {base_auc:.7f}: 차이 {delta:+.7f}, 분할 양수 {positives}/{len(folds)} → {verdict}")
    print(f"비교 팔 재현: {base_auc:.7f} (#455 {reference['nested_auc']:.7f}, 차이 {reproduction['delta']})")


def stability(run_dir: Path, payload: dict, selections: dict[int, dict]) -> None:
    candidates = [c["member_id"] for c in payload["freeze_spec"]["candidates"]]
    per_fold = {str(fold): record["final_selected_member_ids"] for fold, record in selections.items()}
    full_path = run_dir / "full-selection.json"
    full = read_json(full_path)["proposal"]["external_member_ids"] if full_path.is_file() else None
    counts = {member: sum(member in chosen for chosen in per_fold.values()) for member in candidates}
    excluded = {str(fold): [e["column"][len("cand_"):] for e in record["diagnostics"]["candidates"].values() if e["excluded_for_own_conflict"]] for fold, record in selections.items()}
    record = {
        "schema": SCHEMA,
        "candidate_set_id": payload["candidate_set_id"],
        "precommit_sha256": payload["precommit_sha256"],
        "fold_selections": per_fold,
        "fold_exclusions_for_own_conflict": excluded,
        "selection_count_by_candidate": counts,
        "full_oof_proposal": full,
        "fold_vs_full": None if full is None else {
            fold: {"only_in_fold": sorted(set(chosen) - set(full)), "only_in_full": sorted(set(full) - set(chosen))}
            for fold, chosen in per_fold.items()
        },
        "note": "분할별 명단은 감시값이다. 투표·교집합·합집합으로 제안 명단을 만들지 않는다.",
    }
    write_json(run_dir / "selection-stability.json", record)


def full_selection(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    comparison = read_json(run_dir / "nested-comparison.json")
    _require(comparison["precommit_sha256"] == payload["precommit_sha256"], "nested-comparison.json이 다른 precommit에서 나왔다.")
    _require(comparison["complete"] and comparison["passes"], "절차가 통과하지 않았으므로 전체 OOF 제안 명단을 만들지 않는다.")
    out_path = run_dir / "full-selection.json"
    _require(args.force or not out_path.exists(), f"이미 있다: {out_path}")
    started = time.monotonic()
    fold_of, y = load_folds_and_labels()
    own, candidates, _ = load_arm_matrices(run_dir, payload)
    columns = list(own.columns) + list(candidates.columns)
    values = np.ascontiguousarray(pd.concat([own, candidates], axis=1).to_numpy(np.float64))
    fold = fold_of.to_numpy(np.int64)
    label_values = y.to_numpy()
    engine = FastShrunk(values, fold, label_values)
    all_rows = np.arange(len(values))
    own_count = own.shape[1]
    candidate_indices = list(range(own_count, len(columns)))
    diagnostics = diagnose(values, label_values, all_rows, columns, own_count, candidate_indices)
    excluded = {c for c, entry in diagnostics["candidates"].items() if entry["excluded_for_own_conflict"]}
    conflicts = {c: set(entry["conflicts"]) for c, entry in diagnostics["candidates"].items()}
    log_path = run_dir / "full-progress.jsonl"
    if log_path.exists():
        log_path.unlink()
    search = Search(engine, frozenset(ALL_FOLDS), own_count, candidate_indices, excluded, conflicts, log_path)
    stop_reason = search.run()
    final_subset = search.subset(search.selected)
    invariant = check_invariant(diagnostics["spearman"], final_subset, own_count)
    _require(invariant["ok"], f"후보 풀 중복 불변식 위반: {invariant['violations']}")
    selected_ids = [columns[c][len("cand_"):] for c in sorted(search.selected, key=search.rank_of.__getitem__)]
    by_member = {c["member_id"]: c for c in payload["candidate_arm"]["candidates"]}
    record = {
        "schema": SCHEMA,
        "candidate_set_id": payload["candidate_set_id"],
        "precommit_sha256": payload["precommit_sha256"],
        "nested_comparison_passes": True,
        "open_folds": list(ALL_FOLDS),
        "rows": int(len(all_rows)),
        "diagnostics": {k: v for k, v in diagnostics.items() if k != "spearman"},
        "start": {"members": list(own.columns), "score": search.scores[search.subset([])]},
        "stages": [named_stage(stage, columns) for stage in search.stages],
        "evaluations": search.evaluations,
        "stop_reason": stop_reason,
        "full_oof_search_score": search.scores[final_subset],
        "full_oof_score_note": "전체 OOF 검색 점수는 문턱을 다시 판정하지 않는다.",
        "invariant": invariant,
        "proposal": {
            "member_count": len(final_subset),
            "own_member_count": own_count,
            "external_member_count": len(selected_ids),
            "external_member_ids": selected_ids,
            "oof_columns": [columns[c] for c in final_subset],
            "test_columns": [{"column": columns[c], "origin": "own", "test_path": str(OWN_TEST_PATH)} if c < own_count else {"column": columns[c], "origin": "external", "member_id": columns[c][len("cand_"):], "test_path": by_member[columns[c][len("cand_"):]]["test_path"], "test_sha256": by_member[columns[c][len("cand_"):]]["test_sha256"]} for c in final_subset],
        },
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": now_iso(),
    }
    write_json(out_path, record)
    selections, _ = _load_fold_outputs(run_dir, payload)
    stability(run_dir, payload, selections)
    print(f"전체 OOF 제안 명단: 자체 {own_count} + 외부 {len(selected_ids)} {selected_ids} (평가 {search.evaluations}회, {record['elapsed_seconds']:.0f}s)")


def _manifest_files(run_dir: Path) -> list[Path]:
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if relative.parts[0] == "logs" or relative.name in ("manifest.sha256",) or relative.suffix == ".jsonl":
            continue
        files.append(path)
    return files


def report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    comparison_path = run_dir / "nested-comparison.json"
    _require(comparison_path.is_file(), "nested-comparison.json이 없다. compare를 먼저 한다.")
    comparison = read_json(comparison_path)
    stability_record = read_json(run_dir / "selection-stability.json")
    selections, baselines = _load_fold_outputs(run_dir, payload)
    full_path = run_dir / "full-selection.json"
    full = read_json(full_path) if full_path.is_file() else None
    lines: list[str] = []
    lines += [f"# 엄격 외부 후보 중첩 선별 판정 보고 (`{payload['candidate_set_id']}`)", ""]
    if payload["rehearsal"]:
        lines += ["**예행 실행이다.** 합성 후보로 도구 경로만 확인했으며 실제 후보 판정이 아니다.", ""]
    lines += ["## 판정", ""]
    lines += [
        f"- 결과: **{comparison['verdict']}** (후보 절차 팔 {comparison['candidate_arm_auc']:.7f} vs 비교 팔 313 {comparison['comparison_arm_auc']:.7f}, 차이 `{comparison['delta']:+.7f}`, 분할 양수 {comparison['folds_positive']}/{len(comparison['outer_folds'])})",
        f"- 문턱: 차이 `+{GATE_DELTA}` 이상, 바깥 분할 {FOLDS_REQUIRED_POSITIVE}/5 엄격 양수. 결과 확인 뒤 바꾸지 않는다.",
        f"- 경계 보고: 문턱과의 차이 `{comparison['boundary_report']['delta_minus_gate']:+.7f}`, 잡음 바닥 `{NOISE_FLOOR}` 안 여부 {comparison['boundary_report']['delta_within_noise_floor']}",
        f"- 비교 팔 재현: 이어붙인 AUC {comparison['comparison_arm_reproduction']['baseline_concatenated_auc']:.7f}, #455 기준 {comparison['comparison_arm_reproduction']['reference_nested_auc']:.7f}, 차이 {comparison['comparison_arm_reproduction']['delta']}",
        "",
    ]
    lines += ["## 분할별", "", "| 봉인 분할 | 후보 팔 AUC | 비교 팔 AUC | 차이 | 후보 팔 λ | 비교 팔 λ | 선택 외부 후보 | 평가 횟수 | 등록 결합기 최대 차이 |", "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for fold, entry in comparison["fold_results"].items():
        selection = selections[int(fold)]
        lines.append(f"| {fold} | {entry['candidate_auc']:.7f} | {entry['baseline_auc']:.7f} | {entry['delta']:+.7f} | {entry['candidate_lambda']} | {entry['baseline_lambda']} | {', '.join(entry['selected']) or '(없음)'} | {selection['evaluations']} | {selection['sealed_prediction']['registered_max_abs_diff']} |")
    lines += ["", "## 선택 안정성(감시값)", "", "| 후보 | 선택된 분할 수 | 전체 OOF 제안 |", "| --- | ---: | --- |"]
    full_ids = set(stability_record["full_oof_proposal"] or [])
    for member, count in stability_record["selection_count_by_candidate"].items():
        lines.append(f"| `{member}` | {count} | {'포함' if member in full_ids else ('-' if stability_record['full_oof_proposal'] is None else '제외')} |")
    lines += ["", "## 후보 진단(열린 분할 단독 AUC, 자격·순서에 쓰지 않음)", "", "| 후보 | " + " | ".join(f"분할 {fold}" for fold in comparison["outer_folds"]) + " | 자체와 최대 스피어만 | 제외 분할 |", "| --- | " + " | ".join("---:" for _ in comparison["outer_folds"]) + " | ---: | --- |"]
    for candidate in payload["freeze_spec"]["candidates"]:
        column = candidate_column(candidate["member_id"])
        cells = []
        max_rho = 0.0
        excluded_folds = []
        for fold in comparison["outer_folds"]:
            entry = next(e for e in selections[fold]["diagnostics"]["candidates"].values() if e["column"] == column)
            cells.append(f"{entry['standalone_auc']:.6f}")
            max_rho = max(max_rho, entry["spearman_vs_own_max"])
            if entry["excluded_for_own_conflict"]:
                excluded_folds.append(str(fold))
        lines.append(f"| `{candidate['member_id']}` | " + " | ".join(cells) + f" | {max_rho:.6f} | {', '.join(excluded_folds) or '-'} |")
    if full is not None:
        proposal = full["proposal"]
        lines += ["", "## 엄격 외부 제안 구성(전체 OOF 1회 검색)", "", f"- 자체 {proposal['own_member_count']} + 외부 {proposal['external_member_count']} = {proposal['member_count']}구성원: {', '.join(f'`{m}`' for m in proposal['external_member_ids']) or '(외부 없음)'}", f"- 전체 OOF 검색 점수 {full['full_oof_search_score']:.7f} (문턱 재판정에 쓰지 않음), 평가 {full['evaluations']}회"]
    lines += ["", "## 실행 인계 완결 조건(#481) 대조", ""]
    lines += [
        f"- 변경 불가 감사 기록·자격 판정: 동결 명세 `{payload['freeze_spec']['path']}` (spec_sha256 `{payload['freeze_spec']['sha256']}`)의 후보 {payload['freeze_spec']['candidate_count']}개, 사용자 제외 {len(payload['freeze_spec']['user_exclusions'])}개.",
        f"- 외부 후보 동결 명세: 후보 집합 식별자 `{payload['candidate_set_id']}`, 계약 판본 {payload['freeze_spec']['contract_version']}, 조사 기준 시각 `{payload['freeze_spec']['survey_cutoff']}`.",
        f"- 두 입력 명세와 내용 해시: 비교 팔 313 `{payload['comparison_arm']['composition_sha256']}`, 자체 35 `{payload['own_start']['composition_sha256']}`, 후보 `{payload['candidate_arm']['composition_sha256']}`.",
        f"- 고정 결합기와 검색 규칙: `precommit.json`의 `combiner`·`search_rules` (λ 격자 {payload['combiner']['lambda_grid']}).",
        f"- 교체 문턱: `precommit.json`의 `gate` (+{GATE_DELTA}, {FOLDS_REQUIRED_POSITIVE}/5).",
        "- 산출물: `precommit.json`, `fold-<k>/selection.json`, `fold-<k>/predictions.parquet`, `fold-<k>/baseline-predictions.parquet`, `nested-comparison.json`, `selection-stability.json`, `full-selection.json`(통과 시), `report.md`, `manifest.sha256`.",
        "- 실패·재개 규칙: `precommit.json`의 `rules` (모든 하위 명령이 입력 해시와 코드 상태를 다시 확인).",
        "- 실행 경계: 업로드와 최종 두 장 수동 고정은 사용자 승인 뒤에만.",
        "",
        "## 코드 상태",
        "",
        f"- git `{payload['code_state']['git']['commit']}` (dirty {payload['code_state']['git']['dirty']}), 판정 도구 sha256 `{payload['code_state']['script']['sha256']}`, 결합기 module sha256 `{payload['code_state']['ensemble_module']['sha256']}`, sklearn {payload['code_state']['sklearn']}, numpy {payload['code_state']['numpy']}.",
        f"- precommit_sha256 `{payload['precommit_sha256']}`, 작성 {now_iso()}.",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_lines = [f"{file_sha256(path)}  {path.relative_to(run_dir)}" for path in _manifest_files(run_dir)]
    (run_dir / "manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"보고 저장: {run_dir / 'report.md'}, manifest {len(manifest_lines)}개 파일")


# ---------------------------------------------------------------------------
# 조립(통과 뒤, 사용자 확인 뒤)


def assemble(args: argparse.Namespace) -> None:
    import assemble_extended_stack as prior_assembly

    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    _require(not payload["rehearsal"], "예행 실행은 조립하지 않는다.")
    comparison = read_json(run_dir / "nested-comparison.json")
    full = read_json(run_dir / "full-selection.json")
    _require(comparison["passes"] and full["precommit_sha256"] == payload["precommit_sha256"] and comparison["precommit_sha256"] == payload["precommit_sha256"], "통과한 전체 OOF 제안 명단이 없다.")
    started = time.monotonic()
    fold_of, y = load_folds_and_labels()
    own, candidates, _ = load_arm_matrices(run_dir, payload)
    proposal = full["proposal"]
    oof = pd.concat([own, candidates], axis=1)[proposal["oof_columns"]].astype(np.float64)
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID])[ID]
    _require(len(test_ids) == N_TEST and not test_ids.duplicated().any(), "test.csv의 행 수나 id가 기대와 다르다.")
    own_test = pd.read_parquet(OWN_TEST_PATH)
    _require(own_test[ID].to_numpy().tolist() == test_ids.to_numpy().tolist(), "5:1 혼합판 시험 예측의 id 순서가 test.csv와 다르다.")
    own_test = own_test.set_index(ID)
    columns: dict[str, np.ndarray] = {}
    sources: dict[str, dict] = {}
    for entry in proposal["test_columns"]:
        if entry["origin"] == "own":
            values = own_test[entry["column"]].to_numpy(np.float64)
            sources[entry["column"]] = {"kind": "own_cv5_full1_mix", "test_path": str(OWN_TEST_PATH)}
        else:
            values = freeze.load_array(Path(entry["test_path"]), N_TEST, entry["member_id"])
            _require(freeze.array_sha256(values) == entry["test_sha256"], f"{entry['member_id']}: 시험 배열 해시가 제안 명단과 다르다.")
            sources[entry["column"]] = {"kind": "external_cv_fold_average", "test_path": entry["test_path"], "member_id": entry["member_id"]}
        columns[entry["column"]] = values
        sources[entry["column"]]["prediction_sha256"] = prediction_array_sha256(values)
    test = pd.DataFrame(columns, index=test_ids.to_numpy()).astype(np.float64)
    combiner = ensemble.COMBINER_REGISTRY[STRATEGY]
    fitted = combiner.fit(oof, y)
    prediction = ensemble.full_fit_predictions(combiner, oof, y, test)
    in_sample_auc = float(roc_auc_score(y.to_numpy(), np.asarray(fitted.predict(oof), dtype=np.float64)))
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / f"strict-external-{payload['candidate_set_id']}.csv"
    pd.DataFrame({ID: test_ids.to_numpy(), TARGET: prediction}).to_csv(submission_path, index=False)
    manifest = {
        "schema_version": 1,
        "issue": ISSUE,
        "candidate_set_id": payload["candidate_set_id"],
        "precommit_sha256": payload["precommit_sha256"],
        "git": git_state(),
        "strategy": STRATEGY,
        "judged": {"nested_comparison": comparison["delta"], "folds_positive": comparison["folds_positive"], "candidate_arm_auc": comparison["candidate_arm_auc"], "comparison_arm_auc": comparison["comparison_arm_auc"], "verdict": comparison["verdict"]},
        "assembled": {"member_count": proposal["member_count"], "own_member_count": proposal["own_member_count"], "external_member_count": proposal["external_member_count"], "external_member_ids": proposal["external_member_ids"]},
        "combiner": {"shrinkage_lambda": float(fitted.shrinkage_lambda), "lambda_grid": list(combiner.lambda_grid), "fit_protocol": "전체 OOF 1회 적합(ensemble.full_fit_predictions), λ는 5분할 leave-one-fold-out", "in_sample_oof_auc": in_sample_auc},
        "inputs": {"train_sha256": file_sha256(TRAIN_PATH), "test_sha256": file_sha256(TEST_PATH), "folds_sha256": file_sha256(FOLDS_PATH), "pool_sha256": file_sha256(POOL_PATH), "freeze_spec": payload["freeze_spec"], "own_test": {"kind": "cv5_full1_mix", "path": str(OWN_TEST_PATH), "full_refit_manifest_sha256": file_sha256(FULL_REFIT_MANIFEST_PATH)}},
        "members": [{"column": column, "weight": float(weight), "test": sources[column]} for column, weight in fitted.summary().items()],
        "submission": {"path": str(submission_path), "file_sha256": file_sha256(submission_path), "prediction_sha256": prediction_array_sha256(prediction), "checks": prior_assembly.rank_space_checks(prediction, test_ids)},
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path = run_dir / "assembly-manifest.json"
    write_json(manifest_path, manifest)
    print(f"제출 파일 {submission_path} sha256 {manifest['submission']['file_sha256']}\nmanifest {manifest_path} (커밋할 때는 docs/research/로 복사)")


# ---------------------------------------------------------------------------
# 예행용 합성 색인


def rehearsal_index(args: argparse.Namespace) -> None:
    """자체 35개에서 파생한 합성 후보로 판본 3 모양의 색인을 만든다(예행 전용).

    후보 k는 자체 구성원의 로짓에 정규 잡음을 더한 것이다. 마지막 두 후보는 각각
    자체 구성원의 근접 복제(자체 충돌 제외 경로)와 첫 후보의 근접 복제(후보끼리
    충돌·교체 경로)다. 실제 외부 후보의 배열은 전혀 읽지 않는다.
    """
    out_dir = Path(args.out)
    _require(not out_dir.exists(), f"이미 있다: {out_dir}")
    fold_of, y = load_folds_and_labels()
    own, _ = load_own(fold_of)
    own_test = pd.read_parquet(OWN_TEST_PATH).set_index(ID)
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(own.shape[1], size=args.count, replace=False)
    records_dir = out_dir / "records"
    arrays_dir = out_dir / "normalized"
    records_dir.mkdir(parents=True)
    arrays_dir.mkdir()
    label_values = y.to_numpy()

    def perturb(values: np.ndarray, sigma: float, shift: np.ndarray | None = None) -> np.ndarray:
        clipped = np.clip(values, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped))
        if shift is not None:
            logit = logit + shift
        return 1.0 / (1.0 + np.exp(-(logit + rng.normal(0.0, sigma, size=len(values)))))

    rows = []
    specs = [(int(p), args.sigma, f"syn{i:02d}", i < args.informative) for i, p in enumerate(picks)]
    specs.append((int(picks[0]), 0.005, "syn_own_dup", False))  # 자체 구성원 근접 복제 → 제외 경로
    first_oof = None
    for source, sigma, name, informative in specs:
        config = own.columns[source]
        # 예행 전용: 앞 후보 몇 개는 목표값 쪽으로 로짓을 밀어 추가가 실제로 승인되는 경로를 연다.
        shift = args.shift * (2.0 * label_values - 1.0) if informative else None
        oof = perturb(own[config].to_numpy(np.float64), sigma, shift)
        test = perturb(own_test[config].to_numpy(np.float64), sigma)
        if name == "syn00":
            first_oof, first_test = oof, test
        rows.append((name, config, sigma, oof, test))
    dup_oof = perturb(first_oof, 0.005)
    dup_test = perturb(first_test, 0.005)
    rows.append(("syn_cand_dup", own.columns[int(picks[0])], 0.005, dup_oof, dup_test))  # 후보끼리 충돌·교체 경로

    current_records = []
    for order, (name, config, sigma, oof, test) in enumerate(rows, start=1):
        member_id = f"rehearsal:{name}"
        oof_path = arrays_dir / f"oof_{name}.npy"
        test_path = arrays_dir / f"test_{name}.npy"
        np.save(oof_path, oof)
        np.save(test_path, test)
        record = {
            "contract_version": "3.0",
            "ledger_version": 3,
            "rehearsal": True,
            "audit_record_id": None,
            "audit_revision": 1,
            "supersedes_audit_record_id": None,
            "identity": {"member_id": member_id, "display_name": f"합성 {name} (자체 {config}, σ={sigma})"},
            "fixed_source": {"kernel_ref": "rehearsal/synthetic", "script_version_id": 0},
            "predictions": {"normalized": {"oof_path": str(oof_path), "test_path": str(test_path), "oof_sha256": freeze.array_sha256(oof), "test_sha256": freeze.array_sha256(test), "pair_sha256": freeze.pair_sha256(oof, test)}, "rescored_auc": float(roc_auc_score(label_values, oof))},
            "audit": {"audit_state": "감사 완료", "eligibility": "자격 있음", "exclusion_reason_codes": [], "insufficiency_reasons": [], "caveat_codes": ["rehearsal_synthetic"], "evidence_manifest_sha256": None},
            "record_sha256": None,
        }
        record["audit_record_id"] = "emar3-rehearsal-" + freeze.text_sha256(member_id)[:12]
        record["record_sha256"] = freeze.text_sha256(freeze.canonical_json({k: v for k, v in record.items() if k != "record_sha256"}))
        (records_dir / f"{record['audit_record_id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        normalized = record["predictions"]["normalized"]
        current_records.append({
            "member_id": member_id, "display_name": record["identity"]["display_name"], "audit_record_id": record["audit_record_id"], "audit_revision": 1, "supersedes_audit_record_id": None,
            "audit_state": "감사 완료", "eligibility": "자격 있음", "exclusion_reason_codes": [], "insufficiency_reasons": [], "caveat_codes": ["rehearsal_synthetic"],
            "kernel_ref": "rehearsal/synthetic", "script_version_id": 0, "input_population": "rehearsal", "rescored_auc": record["predictions"]["rescored_auc"],
            **{k: normalized[k] for k in ("oof_path", "test_path", "oof_sha256", "test_sha256", "pair_sha256")},
            "record_sha256": record["record_sha256"], "evidence_manifest_sha256": None,
        })
    fold_vector_sha = hashlib.sha256(fold_of.to_numpy(np.int8).tobytes()).hexdigest()
    index = {
        "issue": ISSUE, "rehearsal": True, "ledger_version": 3, "contract_version": "3.0", "contract_ref": freeze.AUDIT_CONTRACT_REF,
        "generated_at": now_iso(), "tool": {"script": "scripts/judge_strict_external_selection.py rehearsal-index", "seed": args.seed},
        "fold_spec": {"id": "community-skf5-shuffle-seed42-train-csv-order", "fold_vector_sha256": fold_vector_sha, "folds_path": str(FOLDS_PATH)},
        "row_contract": {"train_rows": N_TRAIN, "test_rows": N_TEST},
        "summary": {"record_count": len(current_records), "eligible": len(current_records)},
        "eligible_current_records_in_order": [{k: r[k] for k in ("member_id", "audit_record_id", "pair_sha256", "oof_path", "test_path")} for r in current_records],
        "current_records": current_records,
        "superseded_record_ids": [],
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"예행 색인 저장: {out_dir / 'index.json'} (합성 후보 {len(current_records)}개)")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="엄격 외부 후보 중첩 선별 판정 (#486, ADR-0005)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("precommit", help="동결 입력·규칙·코드 상태를 고정하고 캐시를 만든다.")
    p.add_argument("--spec", type=Path, required=True, help="외부 후보 동결 명세")
    p.add_argument("--run-dir", type=Path, help="출력 폴더(기본 run-logs/strict-external-selection/<후보 집합 식별자>)")
    p.add_argument("--folds", type=int, nargs="+", help="예행 전용: 봉인할 바깥 분할 부분집합")
    p.set_defaults(func=precommit)

    for name, func, help_text in (("select", select_fold, "바깥 분할 하나의 후보 선별과 봉인 예측"), ("baseline", baseline_fold, "비교 팔 313의 봉인 분할 예측")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--fold", type=int, required=True)
        p.add_argument("--force", action="store_true")
        if name == "select":
            p.add_argument("--skip-registered-check", action="store_true", help="등록 결합기 대조를 건너뛴다(권장하지 않음)")
        p.set_defaults(func=func)

    p = sub.add_parser("run", help="남은 (baseline, select) 작업을 하위 프로세스로 실행한다.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--heavy-workers", type=int, default=2, help="313열 baseline 동시 상한(작업당 10GB대)")
    p.add_argument("--threads", type=int, default=2)
    p.set_defaults(func=run_jobs)

    for name, func in (("compare", compare), ("report", report), ("assemble", assemble)):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
        p.set_defaults(func=func)

    p = sub.add_parser("full", help="절차 통과 뒤 전체 OOF에 동결 검색을 한 번 적용한다.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=full_selection)

    p = sub.add_parser("rehearsal-index", help="예행용 합성 색인을 만든다(자체 35개 파생, 실제 후보 미사용).")
    p.add_argument("--out", type=Path, default=OUT_ROOT / "rehearsal-index")
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.6)
    p.add_argument("--informative", type=int, default=2, help="목표값 쪽으로 밀어 승인 경로를 여는 앞 후보 수(예행 전용)")
    p.add_argument("--shift", type=float, default=0.3, help="목표값 방향 로짓 이동 폭(예행 전용)")
    p.add_argument("--seed", type=int, default=486)
    p.set_defaults(func=rehearsal_index)

    args = parser.parse_args()
    try:
        args.func(args)
    except (JudgmentError, freeze.FreezeError) as exc:
        sys.exit(f"판정 불가: {exc}")


if __name__ == "__main__":
    main()
