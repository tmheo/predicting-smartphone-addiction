"""엄격 외부 후보의 사전 고정 사다리 판정 도구. (#486 → #491, ADR-0006)

외부 후보 동결 명세(`scripts/freeze_external_candidates.py`)를 입력으로, 현재 313개
확장 구성(비교 팔) 위에 정확 중복을 뺀 후보를 더한 사다리 구성을 nested OOF 5분할로
재어 교체 문턱(+0.00002, 바깥 분할 5/5 양수)을 판정한다.

판정은 읽기 전용이다. `artifacts/pool.yaml`과 champion 판정을 건드리지 않고 MLflow
실행을 만들지 않는다. 산출물은 `run-logs/strict-external-selection/<후보 집합 식별자>/`
(커밋 제외 경로)에 ADR-0006의 변경 불가 산출물 이름으로 남긴다.

    precommit.json                          동결 입력·정확 중복·사다리 구성·선택 규칙·코드 상태(결과 확인 전에 고정)
    cache/*.parquet                         자체 35, 비교 팔 313, 후보 OOF 행렬(해시는 precommit에)
    fold-<k>/baseline-predictions.parquet   비교 팔 313의 봉인 분할 예측(자기 검사·분할별 비교 기준)
    fold-<k>/baseline.json                  그 AUC와 λ
    ladder/<구성>/fold-<k>/predictions.parquet  구성의 봉인 분할 예측
    ladder/<구성>/nested.json               이어붙인 nested AUC, 가중 OOF AUC(진단), 분할별 AUC·λ
    ladder-comparison.json                  자기 검사, 구성별 313 대비 차이와 분할 부호, 문턱 판정, 선택 규칙 적용과 제안 구성
    report.md, manifest.sha256              사람이 읽는 요약과 모든 산출물의 내용 해시

사용법(실행 회차 순서):
    uv run python scripts/judge_strict_external_selection.py precommit --spec <동결 명세>
    uv run python scripts/judge_strict_external_selection.py run --run-dir <출력 폴더> [--workers 3 --threads 4]
    uv run python scripts/judge_strict_external_selection.py compare --run-dir <출력 폴더>
    uv run python scripts/judge_strict_external_selection.py report --run-dir <출력 폴더>
    uv run python scripts/judge_strict_external_selection.py assemble --run-dir <출력 폴더> # 통과 뒤, 사용자 확인 뒤

`run`은 `baseline k`(비교 팔 313의 봉인 분할 예측 5개)와 `ladder <구성>`(구성 하나의
nested 5분할) 작업을 하위 프로세스로 돌린다. 모두 300열대 shrunk 적합이라 동시 상한은
3이다(#455: 400열대는 5개에서 커널 패닉). 이미 산출물이 있는 작업은 건너뛰므로 중단 뒤
같은 명령으로 이어 달릴 수 있다. 모든 하위 명령은 시작할 때 precommit의 입력 해시와
코드 상태를 다시 계산해 정확히 같을 때만 진행한다(재개 규칙). 하나라도 어긋나면
`판정 불가`로 두고 precommit부터 다시 한다.

비교 팔과 사다리 구성은 모두 고정 결합기 `CSelectedShrunkRankLogitCombiner`(#489, 규제 강도
C와 수축 계수 λ를 안쪽 leave-one-fold-out으로 함께 고르는 shrunk)를
`pipeline.ensemble.evaluate_nested`와 같은 분할 순서·행 순서로 적합한다. 현재 두 번째 장
`30b6f97c`(#489)가 이 결합기로 만든 313개 판이므로 자기 검사 기준값도 #489 `comparison.json`의
후보 nested·분할별 AUC다(#498에서 #455 C=1.0 기준에서 바꿈). 결합기는 학습 행 순서에
민감하므로(#486 예행, 행 순서만으로 7.96e-04 차이) 원래 행 순서를 지킨다. 분할 하나의
적합은 로지스틱 29회라 C=1.0판보다 약 1.6배 걸린다(313열 분할당 약 14분).

ADR-0005의 정확 검색(`adr0005-select`, `adr0005-compare`, `adr0005-full`)과 빠른 결합기
`FastShrunk`, `Search`는 계약이 되살아날 때를 위해 보존하며 사다리 판정에서는 쓰지 않는다.
예행(자체 35개만, 합성 후보)은 `rehearsal-index`로 판본 3 모양의 합성 색인을 만든다.
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
from pipeline.judgment import (
    DUPLICATE_SPEARMAN,
    FOLDS_PATH,
    missingness_reweighting,
    weighted_oof_auc,
)
from pipeline.ledger import POOL_PATH, Pool
from pipeline.pool_audit import prediction_array_sha256
from pipeline.runs import MlflowRunStore

ISSUE = 491
SCHEMA = "strict-external-selection/2"
CONTRACT_ADR_PATH = Path("docs/adr/0006-strict-external-candidate-ladder.md")
STRATEGY = "shrunk_rank_logit_logistic"  # ADR-0005 보존 경로와 #455 계보 확인에만 쓴다.
LADDER_STRATEGY = ensemble.CSelectedShrunkRankLogitCombiner.name  # 비교 팔·사다리·조립의 고정 결합기(#489, #498)
C_GRID = ensemble.C_SELECTION_GRID
COMPARISON_REFERENCE_PATH = Path("docs/research/logistic-c-selection/issue489/comparison.json")
COMPARISON_RUN_ID = "30b6f97c30904995a79e476f02decf8f"
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
MISSINGNESS_TEST_PATH = ensemble.MISSINGNESS_TEST_PATH

FULL_CONFIG = "ext313_strict_all"
LADDER_DIR = "ladder"
CONFIG_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.:-]+")
CONCAT_AUC_TOLERANCE = 1e-12  # nested.json의 이어붙인 AUC와 parquet 재계산의 허용 차이

LADDER_RULES = {
    "comparison_arm": "비교 팔은 현재 313개 확장 구성이며 바꾸지 않는다.",
    "exact_duplicate": "예측 쌍 SHA-256(float64 정규화 OOF·시험 배열)이 313 구성원 하나와 같은 후보는 정확 중복이며 사다리 후보에서 자동으로 빠진다.",
    "full": f"`{FULL_CONFIG}`: 313 + 정확 중복을 뺀 사다리 후보 전부.",
    "ablate_source": "`ablate_source_<출처>`: 전체 구성에서 출처(고정 공개 판본의 작성자) 하나의 후보를 모두 뺀 구성을 출처마다 하나(동결 순서).",
    "ablate_caveat": "`ablate_caveat_<부류>`: 전체 구성에서 주의 사항 부류 하나를 가진 후보를 모두 뺀 구성을 부류마다 하나(동결 순서). 사다리 후보 전원이 공통으로 가진 부류는 절제하면 313과 같아지므로 두지 않는다.",
    "dedupe": "구성원 집합이 앞선 구성과 같은 구성은 하나만 남긴다.",
    "no_single_add": "후보 하나씩 더한 구성은 두지 않는다.",
    "column_order": "구성원 열 순서는 313 manifest 순서 뒤 동결 명세 순서다.",
    "near_duplicate": f"스피어만 {DUPLICATE_SPEARMAN} 이상 쌍(자체끼리, 313끼리, 후보와 313 사이, 후보끼리)은 열린 4분할의 진단값으로만 기록하고 제외하지 않는다.",
    "empty": "정확 중복을 뺀 사다리 후보가 없으면 사다리를 실행하지 않고 현재 두 장 유지를 완결된 결론으로 기록한다.",
}
SELECTION_RULES = {
    "self_check": f"시작 조건(비교 팔 자기 검사): 313의 봉인 분할 예측 5개를 고정 결합기로 다시 만들어 이어붙인 AUC와 분할별 AUC가 #489 규제 강도 선택판(30b6f97c) 기준값과 잡음 바닥 {NOISE_FLOOR} 안에서 맞아야 하며, 실패하면 전체가 판정 불가다. 이 봉인 예측이 분할별 비교의 기준이다.",
    "gate": f"313 대비 이어붙인 nested AUC 차이 +{GATE_DELTA} 이상이고 바깥 분할 5곳의 차이가 모두 엄격히 양수인 구성만 통과.",
    "pick": f"통과 구성이 여럿이면 nested 최고. 최고와의 차이가 잡음 바닥 {NOISE_FLOOR} 안이면 그 가운데 구성원 수가 가장 적은 구성, 구성원 수까지 같으면 사다리 순서가 앞선 구성.",
    "none": "통과 구성이 없으면 현재 두 장(e88f706e + 30b6f97c) 유지가 완결된 결론.",
    "weighted": "가중 OOF AUC는 진단값이며 판정에 쓰지 않는다.",
    "fixed": "사다리 구성, 선택 규칙, 문턱은 결과 확인 전에 고정하고 결과를 본 뒤 더하거나 빼지 않는다.",
    "public_score": "공개 점수는 어느 단계에도 쓰지 않는다.",
}

# ADR-0005 정확 검색 규칙(보존). 사다리 판정에서는 쓰지 않는다.
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
    """#457 manifest의 313구성원 OOF 행렬. 자체 35는 풀 순서, 외부는 판본 2 장부 경로.

    정확 중복 판정을 위해 시험 배열도 읽어 manifest의 시험 예측 해시와 대조하고
    후보와 같은 규칙(float64 OOF·시험 배열)의 예측 쌍 SHA-256을 구성원마다 남긴다.
    """
    manifest = read_json(COMPARISON_MANIFEST_PATH)
    _require(manifest["issue"] == 457 and len(manifest["members"]) == COMPARISON_MEMBER_COUNT, "비교 팔 manifest가 #457 313구성원이 아니다.")
    _, accepted = ladder.load_ledger()
    by_column = {f"ext_{row['member_id']}": row for row in accepted}
    own_test = pd.read_parquet(OWN_TEST_PATH).set_index(ID)
    _require(len(own_test) == N_TEST, f"5:1 혼합판 시험 예측 행 수 {len(own_test)}")
    label_values = y.to_numpy()
    columns: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for entry in manifest["members"]:
        column = entry["column"]
        if entry["origin"] == "own":
            _require(column in own.columns, f"비교 팔의 자체 구성원 {column}이 풀에 없다.")
            values = own[column].to_numpy(np.float64)
            test = own_test[column].to_numpy(np.float64)
        else:
            ledger_row = by_column[column]
            _require(ledger_row["oof_path"] == entry["oof_path"], f"{column}: manifest와 장부의 OOF 경로가 다르다.")
            values = ladder.load_ledger_array(ledger_row["oof_path"])
            _require(len(values) == N_TRAIN and bool(np.isfinite(values).all()), f"{column}: 행 수 {len(values)} 또는 비유한값")
            delta = float(roc_auc_score(label_values, values)) - float(ledger_row["auc"])
            _require(abs(delta) < 1e-9, f"{column}: 장부 AUC와 {delta:+.2e} 차이")
            _require(ledger_row["test_path"] == entry["test"]["test_path"], f"{column}: manifest와 장부의 시험 경로가 다르다.")
            test = ladder.load_ledger_array(entry["test"]["test_path"])
        _require(test.shape == (N_TEST,) and bool(np.isfinite(test).all()), f"{column}: 시험 배열 형태 {test.shape} 또는 비유한값")
        _require(prediction_array_sha256(test) == entry["test"]["prediction_sha256"], f"{column}: 시험 배열 해시가 #457 manifest와 다르다.")
        columns[column] = values
        rows.append({
            "column": column,
            "origin": entry["origin"],
            "oof_sha256": prediction_array_sha256(values),
            "test_path": entry["test"]["test_path"],
            "test_sha256": freeze.array_sha256(test),
            "test_prediction_sha256": entry["test"]["prediction_sha256"],
            "pair_sha256": freeze.pair_sha256(values, test),
        })
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
        rows.append({"order": candidate["order"], "member_id": candidate["member_id"], "column": column, "audit_record_id": candidate["audit_record_id"], "kernel_ref": candidate["kernel_ref"], "source": source_of(candidate), "caveat_codes": list(candidate["caveat_codes"]), "rescored_auc": candidate["rescored_auc"], "pair_sha256": candidate["pair_sha256"], "oof_sha256": candidate["oof_sha256"], "test_sha256": candidate["test_sha256"], "test_path": candidate["test_path"]})
    matrix = pd.DataFrame(columns, index=fold_of.index).astype(np.float64) if columns else pd.DataFrame(index=fold_of.index)
    return matrix, rows


# ---------------------------------------------------------------------------
# 사다리 구성(ADR-0006)


def source_of(candidate: dict) -> str:
    """출처는 고정 공개 판본의 작성자(kernel_ref의 첫 조각)다."""
    return str(candidate["kernel_ref"]).split("/")[0]


def _unique_in_order(values) -> list:
    seen: list = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def find_exact_duplicates(candidate_rows: list[dict], comparison_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """예측 쌍 SHA-256이 313 구성원과 같은 후보(정확 중복)와, OOF만 같은 후보(진단)."""
    pair_to_column = {row["pair_sha256"]: row["column"] for row in comparison_rows}
    oof_to_column = {row["oof_sha256"]: row["column"] for row in comparison_rows}
    exact: list[dict] = []
    oof_only: list[dict] = []
    for row in candidate_rows:
        base = {"order": row["order"], "member_id": row["member_id"], "pair_sha256": row["pair_sha256"]}
        if row["pair_sha256"] in pair_to_column:
            exact.append({**base, "matches_column": pair_to_column[row["pair_sha256"]]})
        elif row["oof_sha256"] in oof_to_column:
            oof_only.append({**base, "oof_sha256": row["oof_sha256"], "matches_column": oof_to_column[row["oof_sha256"]], "note": "OOF만 같고 시험 배열이 다르다. 정확 중복이 아니므로 사다리 후보에 남는다."})
    return exact, oof_only


def build_ladder(candidate_rows: list[dict], duplicate_ids: set[str], comparison_columns: list[str]) -> dict:
    """동결 명세 순서에서 결정적으로 사다리 구성을 만든다(ADR-0006 사다리 구성 규칙)."""
    pool = [row for row in candidate_rows if row["member_id"] not in duplicate_ids]
    configs: list[dict] = []
    omitted: list[dict] = []
    seen: dict[frozenset[str], str] = {}

    def add(name: str, kind: str, description: str, keep) -> None:
        _require(CONFIG_NAME_PATTERN.fullmatch(name) is not None, f"구성 이름에 쓸 수 없는 문자: {name}")
        kept = [row for row in pool if keep(row)]
        removed = [row["member_id"] for row in pool if not keep(row)]
        key = frozenset(row["member_id"] for row in kept)
        if not kept:
            omitted.append({"name": name, "kind": kind, "reason": "후보가 남지 않아 313과 같다", "removed_member_ids": removed})
            return
        if key in seen:
            omitted.append({"name": name, "kind": kind, "reason": f"구성원 집합이 `{seen[key]}`와 같다", "same_members_as": seen[key], "removed_member_ids": removed})
            return
        seen[key] = name
        candidate_columns = [row["column"] for row in kept]
        configs.append({
            "index": len(configs),
            "name": name,
            "kind": kind,
            "description": description,
            "candidate_member_ids": [row["member_id"] for row in kept],
            "candidate_columns": candidate_columns,
            "removed_member_ids": removed,
            "columns": list(comparison_columns) + candidate_columns,
            "member_count": len(comparison_columns) + len(candidate_columns),
        })

    if pool:
        add(FULL_CONFIG, "full", f"313 + 정확 중복을 뺀 사다리 후보 {len(pool)}개 전부", lambda row: True)
        for source in _unique_in_order(row["source"] for row in pool):
            add(f"ablate_source_{source}", "ablate_source", f"전체 구성에서 출처 `{source}`의 후보 제외", lambda row, s=source: row["source"] != s)
        codes = _unique_in_order(code for row in pool for code in row["caveat_codes"])
        common = [code for code in codes if all(code in row["caveat_codes"] for row in pool)]
        for code in codes:
            if code in common:
                omitted.append({"name": f"ablate_caveat_{code}", "kind": "ablate_caveat", "reason": "사다리 후보 전원이 공통으로 가진 부류라 절제하면 313과 같다"})
                continue
            add(f"ablate_caveat_{code}", "ablate_caveat", f"전체 구성에서 주의 사항 부류 `{code}`의 후보 제외", lambda row, c=code: c not in row["caveat_codes"])
    else:
        common = []
    return {
        "candidate_count": len(pool),
        "candidate_member_ids": [row["member_id"] for row in pool],
        "candidate_columns": [row["column"] for row in pool],
        "sources_in_order": _unique_in_order(row["source"] for row in pool),
        "caveat_codes_in_order": _unique_in_order(code for row in pool for code in row["caveat_codes"]),
        "common_caveat_codes": common,
        "config_count": len(configs),
        "configs": configs,
        "omitted": omitted,
        "rules": LADDER_RULES,
    }


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
    _require(set(reference["fold_aucs"]) == {str(k) for k in ALL_FOLDS}, "#455 근거에 분할별 AUC 5개가 없다.")
    c1_reference = reference
    current = read_json(COMPARISON_REFERENCE_PATH)
    _require(current["issue"] == 489 and current["reproduction"]["passes"] and current["control"]["nested_auc"] == c1_reference["nested_auc"], "#489 근거가 #455 대조군 위의 판정이 아니다.")
    _require(current["candidate"]["strategy"] == LADDER_STRATEGY and set(current["candidate"]["fold_aucs"]) == {str(k) for k in ALL_FOLDS}, "#489 근거에 규제 강도 선택판의 분할별 AUC 5개가 없다.")
    reference = {key: current["candidate"][key] for key in ("nested_auc", "weighted_oof_auc", "fold_aucs", "fold_selected_c", "fold_selected_lambda", "prediction_sha256")}

    exact_duplicates, oof_only_matches = find_exact_duplicates(candidate_rows, comparison_rows)
    ladder_spec = build_ladder(candidate_rows, {row["member_id"] for row in exact_duplicates}, list(comparison.columns))

    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    own.to_parquet(cache_dir / "own35-oof.parquet")
    comparison.to_parquet(cache_dir / "comparison-arm-oof.parquet")
    candidates.to_parquet(cache_dir / "candidates-oof.parquet")
    caches = {name: file_sha256(cache_dir / name) for name in ("own35-oof.parquet", "comparison-arm-oof.parquet", "candidates-oof.parquet")}

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "contract": {"adr": str(CONTRACT_ADR_PATH), "adr_sha256": file_sha256(CONTRACT_ADR_PATH) if CONTRACT_ADR_PATH.is_file() else None},
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
            "test": {"path": str(TEST_PATH), "sha256": file_sha256(TEST_PATH)},
            "folds": {"path": str(FOLDS_PATH), "sha256": file_sha256(FOLDS_PATH), "fold_vector_sha256": fold_vector_sha, "rows_per_fold": {str(k): int((fold_of == k).sum()) for k in ALL_FOLDS}},
            "own_test": {"path": str(OWN_TEST_PATH), "sha256": file_sha256(OWN_TEST_PATH), "kind": "cv5_full1_mix"},
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
            "pair_composition_sha256": canonical_sha256([(r["column"], r["pair_sha256"]) for r in comparison_rows]),
            "reference": {"run_id": COMPARISON_RUN_ID, "issue": 489, "evidence_path": str(COMPARISON_REFERENCE_PATH), "evidence_sha256": file_sha256(COMPARISON_REFERENCE_PATH), "config": COMPARISON_CONFIG, "strategy": LADDER_STRATEGY, **reference, "c1_lineage": {"evidence_path": str(COMPARISON_EVIDENCE_PATH), "evidence_sha256": file_sha256(COMPARISON_EVIDENCE_PATH), "nested_auc": c1_reference["nested_auc"], "fold_aucs": c1_reference["fold_aucs"], "note": "#455 C=1.0판(443b3a71)의 값. #489에서 규제 강도 선택판으로 교체돼 자기 검사 기준이 아니다."}},
        },
        "candidate_arm": {
            "columns": list(own.columns) + list(candidates.columns),
            "candidates": candidate_rows,
            "composition_sha256": canonical_sha256([(r["column"], r["oof_sha256"]) for r in candidate_rows]),
        },
        "exact_duplicates": exact_duplicates,
        "oof_only_matches": oof_only_matches,
        "ladder": ladder_spec,
        "caches": caches,
        "combiner": {
            "name": LADDER_STRATEGY,
            "c_grid": list(C_GRID),
            "lambda_grid": list(LAMBDA_GRID),
            "meta": {"representation": "rank_logit", "penalty": "l2", "solver": "lbfgs", "max_iter": META_MAX_ITER, "random_state": 0, "logit_eps": LOGIT_EPS},
            "implementation": "pipeline.ensemble.CSelectedShrunkRankLogitCombiner(#489). 비교 팔과 사다리 구성 모두 evaluate_nested와 같은 분할 순서·행 순서로 분할마다 fit → predict.",
            "nested": "바깥 분할 하나를 봉인하고 나머지 4분할 행으로 결합기를 맞춰 봉인 분할을 예측한 뒤 5개 예측을 원래 행 순서로 이어붙여 채점. (C, λ)는 학습 부분 안의 leave-one-fold-out에서 함께 고르고 동률이면 작은 C, 같은 C 안에서 작은 λ.",
            "weighted_oof": {"test_path": str(MISSINGNESS_TEST_PATH), "note": "test 결측 패턴 구성비 재가중(#383). 진단값."},
        },
        "gate": {"delta_required": GATE_DELTA, "folds_required_positive": FOLDS_REQUIRED_POSITIVE, "comparison": "구성의 봉인 분할 예측 5개를 원래 행 순서로 이어붙인 AUC에서 비교 팔 313의 같은 값을 뺀 차이, 분할별 차이는 같은 봉인 분할 예측끼리", "public_score_used": False},
        "noise_floor": NOISE_FLOOR,
        "self_check": {"reference_nested_auc": reference["nested_auc"], "reference_fold_aucs": reference["fold_aucs"], "tolerance": NOISE_FLOOR, "rule": SELECTION_RULES["self_check"]},
        "selection_rules": SELECTION_RULES,
        "rules": {
            "failure": "계산 하나라도 실패하거나 끝나지 않으면 완료한 일부 결과를 쓰지 않고 전체를 판정 불가로 둔다.",
            "resume": "모든 입력 해시와 코드 상태가 precommit과 정확히 같을 때만 재개한다.",
            "assembly": "선택한 구성이 엄격 외부 제안 구성이다. 조립은 #444 규칙대로 전체 OOF에 결합기를 한 번 맞추며, 자체 35의 시험 예측은 5:1 혼합판, 313의 외부는 장부 시험 배열, 후보는 동결 명세의 정규화 시험 배열이다.",
            "upload": "문턱 통과 뒤 조립, Kaggle 업로드와 최종 두 장 수동 고정은 사용자 승인 뒤에만 한다.",
        },
        "preserved_adr0005": {"search_rules": SEARCH_RULES, "note": "ADR-0005 정확 검색 규칙. 사다리 판정에서 쓰지 않으며 adr0005-* 명령이 되살아날 때를 위해 남긴다."},
        "code_state": code_state(),
    }
    payload["precommit_sha256"] = canonical_sha256(payload)
    write_json(run_dir / "precommit.json", payload)
    print(f"precommit 저장: {run_dir / 'precommit.json'}")
    print(f"  동결 후보 {len(candidate_rows)}개, 정확 중복 {len(exact_duplicates)}개 → 사다리 후보 {ladder_spec['candidate_count']}개, 구성 {ladder_spec['config_count']}개(생략 {len(ladder_spec['omitted'])}), 비교 팔 {len(comparison_rows)}, 자체 {len(own_rows)}, 분할 {list(folds)}")
    for row in exact_duplicates:
        print(f"  정확 중복 {row['order']:>2} {row['member_id']} = {row['matches_column']}")
    for config in ladder_spec["configs"]:
        print(f"  구성 {config['index']} {config['name']:<44} {config['member_count']}구성원 (후보 {len(config['candidate_member_ids'])}, 뺌 {len(config['removed_member_ids'])})")
    for entry in ladder_spec["omitted"]:
        print(f"  생략 {entry['name']}: {entry['reason']}")
    print(f"  비교 팔 기준(#489, {COMPARISON_RUN_ID[:8]}): nested {reference['nested_auc']:.7f}, 결합기 {LADDER_STRATEGY}, C 격자 {list(C_GRID)}")
    print(f"  precommit_sha256 {payload['precommit_sha256']}")


def load_precommit(run_dir: Path) -> dict:
    """precommit을 읽고 입력 해시와 코드 상태가 지금과 정확히 같은지 확인한다."""
    path = run_dir / "precommit.json"
    _require(path.is_file(), f"precommit.json이 없다: {run_dir}")
    payload = read_json(path)
    recorded = payload["precommit_sha256"]
    _require(canonical_sha256({k: v for k, v in payload.items() if k != "precommit_sha256"}) == recorded, "precommit.json이 제자리에서 바뀌었다.")
    spec = freeze.verify_spec_file(Path(payload["freeze_spec"]["path"]))
    _require(spec["spec_sha256"] == payload["freeze_spec"]["sha256"], "동결 명세가 precommit과 다르다.")
    for key in ("train", "test", "folds", "own_test"):
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
        "open_rows": len(open_rows),
        "sealed_rows": len(sealed_rows),
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
        "rows": len(rows),
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


def ladder_combiner(fold_of: pd.Series) -> ensemble.CSelectedShrunkRankLogitCombiner:
    """고정 결합기: #489 규제 강도 선택판과 같은 C·λ 안쪽 선택 shrunk."""
    return ensemble.CSelectedShrunkRankLogitCombiner(fold_of=fold_of, c_grid=C_GRID, lambda_grid=LAMBDA_GRID, max_iter=META_MAX_ITER)


def baseline_fold(args: argparse.Namespace) -> None:
    """비교 팔 313의 봉인 분할 예측. 고정 결합기를 evaluate_nested의 분할 하나와 같게 쓴다."""
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
    try:
        fitted = ladder_combiner(fold_of).fit(preds[inner], y[inner])
    except ensemble.CombinerConvergenceError as exc:
        raise JudgmentError(f"비교 팔 분할 {sealed} 미수렴({exc}). 전체 판정은 판정 불가다.") from exc
    prediction = np.asarray(fitted.predict(preds[outer]), dtype=np.float64)
    _require(prediction.shape == (int(outer.sum()),) and bool(np.isfinite(prediction).all()), "비교 팔 예측이 유한하지 않다.")
    reference_auc = payload["comparison_arm"]["reference"]["fold_aucs"][str(sealed)]
    auc = float(roc_auc_score(y[outer].to_numpy(), prediction))
    pd.DataFrame({ID: fold_of.index.to_numpy()[outer], "prediction": prediction}).to_parquet(out_path, index=False)
    write_json(fold_dir / "baseline.json", {
        "schema": SCHEMA,
        "candidate_set_id": payload["candidate_set_id"],
        "precommit_sha256": payload["precommit_sha256"],
        "sealed_fold": sealed,
        "member_count": int(preds.shape[1]),
        "composition_sha256": payload["comparison_arm"]["composition_sha256"],
        "strategy": LADDER_STRATEGY,
        "lambda": float(fitted.shrinkage_lambda),
        "c": float(fitted.c),
        "auc": auc,
        "reference_auc": reference_auc,
        "delta_vs_reference": auc - reference_auc,
        "prediction_sha256": prediction_array_sha256(prediction),
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": now_iso(),
    })
    print(f"  저장: {out_path} ({time.monotonic() - started:.0f}s, C={fitted.c} λ={fitted.shrinkage_lambda}, AUC {auc:.7f}, #489 기준 차이 {auc - reference_auc:+.2e})", flush=True)


# ---------------------------------------------------------------------------
# 사다리 구성 작업(ADR-0006)


def load_ladder_matrix(run_dir: Path, payload: dict, config: dict, fold_of: pd.Series) -> pd.DataFrame:
    """구성의 OOF 행렬: 비교 팔 313(manifest 순서) 뒤에 그 구성의 후보 열(동결 순서)."""
    cache = run_dir / "cache"
    comparison = pd.read_parquet(cache / "comparison-arm-oof.parquet").astype(np.float64)
    _require(list(comparison.columns) == [m["column"] for m in payload["comparison_arm"]["members"]], "비교 팔 열 순서가 precommit과 다르다.")
    frames = [comparison]
    if config["candidate_columns"]:
        candidates = pd.read_parquet(cache / "candidates-oof.parquet", columns=config["candidate_columns"]).astype(np.float64)
        frames.append(candidates)
    matrix = pd.concat(frames, axis=1)
    _require(list(matrix.columns) == config["columns"], f"{config['name']}: 열 순서가 precommit과 다르다.")
    _require(matrix.index.equals(fold_of.index), f"{config['name']}: 행 순서가 folds와 다르다.")
    return matrix


def near_duplicate_diagnostics(matrix: pd.DataFrame, fold_of: pd.Series, own_count: int, comparison_count: int) -> dict:
    """열린 4분할마다 스피어만 0.998 이상 쌍을 종류별로 기록한다(진단값, 제외 없음)."""
    columns = list(matrix.columns)
    values = matrix.to_numpy(np.float64)
    fold = fold_of.to_numpy(np.int64)

    def kind_of(i: int, j: int) -> str:
        def group(c: int) -> str:
            return "own" if c < own_count else ("ext313" if c < comparison_count else "candidate")
        return f"{group(i)}-{group(j)}"

    per_fold: dict[str, dict] = {}
    for sealed in ALL_FOLDS:
        rows = np.flatnonzero(fold != sealed)
        spearman = spearman_matrix(values[rows])
        upper = np.triu_indices(len(columns), k=1)
        hits = np.flatnonzero(spearman[upper] >= DUPLICATE_SPEARMAN)
        pairs = [{"a": columns[upper[0][h]], "b": columns[upper[1][h]], "kind": kind_of(int(upper[0][h]), int(upper[1][h])), "spearman": float(spearman[upper[0][h], upper[1][h]])} for h in hits]
        counts: dict[str, int] = {}
        for pair in pairs:
            counts[pair["kind"]] = counts.get(pair["kind"], 0) + 1
        candidate_max = {columns[c]: float(np.max(np.delete(spearman[c], c))) for c in range(comparison_count, len(columns))}
        per_fold[str(sealed)] = {"open_rows": len(rows), "pair_count": len(pairs), "counts_by_kind": counts, "pairs": pairs, "candidate_max_spearman_vs_any": candidate_max}
        del spearman
    return {"threshold": DUPLICATE_SPEARMAN, "note": "열린 4분할의 진단값. 제외·선택에 쓰지 않는다.", "by_sealed_fold": per_fold}


def ladder_job(args: argparse.Namespace) -> None:
    """사다리 구성 하나의 nested 5분할. evaluate_nested와 같은 분할 순서·행 순서로 등록 결합기를 적합한다."""
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    config = next((c for c in payload["ladder"]["configs"] if c["name"] == args.config), None)
    _require(config is not None, f"precommit의 사다리 구성이 아니다: {args.config}")
    out_dir = run_dir / LADDER_DIR / config["name"]
    out_path = out_dir / "nested.json"
    _require(args.force or not out_path.exists(), f"이미 있다: {out_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fold_of, y = load_folds_and_labels()
    matrix = load_ladder_matrix(run_dir, payload, config, fold_of)
    print(f"=== 사다리 {config['name']} ({matrix.shape[1]}구성원: 313 + 후보 {len(config['candidate_columns'])}) ===", flush=True)
    diagnostics = None
    if config["name"] == FULL_CONFIG:
        diagnostics_started = time.monotonic()
        diagnostics = near_duplicate_diagnostics(matrix, fold_of, payload["own_start"]["member_count"], payload["comparison_arm"]["member_count"])
        diagnostics["elapsed_seconds"] = time.monotonic() - diagnostics_started
        print(f"  근접 중복 진단: 분할별 쌍 수 {[v['pair_count'] for v in diagnostics['by_sealed_fold'].values()]} ({diagnostics['elapsed_seconds']:.0f}s)", flush=True)
    combiner = ladder_combiner(fold_of)
    nested = np.full(len(matrix), np.nan)
    folds_out: dict[str, dict] = {}
    for fold in payload["outer_folds"]:
        fold_started = time.monotonic()
        inner = (fold_of != fold).to_numpy()
        outer = (fold_of == fold).to_numpy()
        try:
            fitted = combiner.fit(matrix[inner], y[inner])
        except ensemble.CombinerConvergenceError as exc:
            raise JudgmentError(f"{config['name']}: 바깥 분할 {fold}에서 미수렴({exc}). 전체 판정은 판정 불가다.") from exc
        prediction = np.asarray(fitted.predict(matrix[outer]), dtype=np.float64)
        _require(prediction.shape == (int(outer.sum()),) and bool(np.isfinite(prediction).all()), f"{config['name']}: 분할 {fold} 예측이 유한하지 않다.")
        nested[outer] = prediction
        fold_dir = out_dir / f"fold-{fold}"
        fold_dir.mkdir(exist_ok=True)
        pd.DataFrame({ID: fold_of.index.to_numpy()[outer], "prediction": prediction}).to_parquet(fold_dir / "predictions.parquet", index=False)
        folds_out[str(fold)] = {"rows": int(outer.sum()), "auc": float(roc_auc_score(y[outer].to_numpy(), prediction)), "lambda": float(fitted.shrinkage_lambda), "c": float(fitted.c), "prediction_sha256": prediction_array_sha256(prediction), "elapsed_seconds": time.monotonic() - fold_started}
        print(f"  분할 {fold}: AUC {folds_out[str(fold)]['auc']:.7f}, C={fitted.c} λ={fitted.shrinkage_lambda} ({folds_out[str(fold)]['elapsed_seconds']:.0f}s)", flush=True)
    complete = list(payload["outer_folds"]) == list(ALL_FOLDS)
    scored = ~np.isnan(nested)
    series = pd.Series(nested, index=matrix.index, name="prediction")
    nested_auc = float(roc_auc_score(y.to_numpy()[scored], nested[scored]))
    weighted = weighted_oof_auc(series[scored], y[scored], missingness_reweighting(TRAIN_PATH, MISSINGNESS_TEST_PATH))
    record = {
        "schema": SCHEMA,
        "candidate_set_id": payload["candidate_set_id"],
        "precommit_sha256": payload["precommit_sha256"],
        "config": {k: config[k] for k in ("index", "name", "kind", "description", "candidate_member_ids", "removed_member_ids", "member_count")},
        "strategy": LADDER_STRATEGY,
        "member_count": int(matrix.shape[1]),
        "columns": list(matrix.columns),
        "outer_folds": list(payload["outer_folds"]),
        "complete": complete,
        "nested_auc": nested_auc,
        "weighted_oof_auc": weighted.auc,
        "weighted": {"effective_sample_size": weighted.effective_sample_size, "effective_sample_fraction": weighted.effective_sample_fraction, "zero_weight_rows": weighted.zero_weight_rows, "test_only_pattern_count": weighted.test_only_pattern_count},
        "fold_aucs": {k: v["auc"] for k, v in folds_out.items()},
        "folds": folds_out,
        "near_duplicate_diagnostics": diagnostics,
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": now_iso(),
    }
    write_json(out_path, record)
    print(f"  nested {nested_auc:.7f}, 가중 {weighted.auc:.7f} → {out_path} ({record['elapsed_seconds']:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# 작업 실행기


def jobs_for(payload: dict) -> list[tuple[str, str]]:
    """(종류, 키) 작업 목록. 자기 검사의 baseline을 먼저, 사다리 구성을 사다리 순서로."""
    if not payload["ladder"]["configs"]:
        return []
    return [("baseline", str(k)) for k in payload["outer_folds"]] + [("ladder", c["name"]) for c in payload["ladder"]["configs"]]


def job_output(run_dir: Path, kind: str, key: str) -> Path:
    if kind == "baseline":
        return run_dir / f"fold-{key}" / "baseline-predictions.parquet"
    return run_dir / LADDER_DIR / key / "nested.json"


def job_command(run_dir: Path, kind: str, key: str) -> list[str]:
    option = "--fold" if kind == "baseline" else "--config"
    return [sys.executable, __file__, kind, "--run-dir", str(run_dir), option, key]


def _running_jobs(run_dir: Path) -> set[tuple[str, str]]:
    listing = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True, check=False).stdout
    pattern = re.compile(rf"judge_strict_external_selection\.py (baseline|ladder) --run-dir {re.escape(str(run_dir))} --(?:fold|config) (\S+)")
    return {(kind, key) for kind, key in pattern.findall(listing)}


def run_jobs(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    jobs = jobs_for(payload)
    if not jobs:
        print("사다리 후보가 없다. 실행할 작업이 없고 compare가 현재 두 장 유지를 기록한다.")
        return
    pending = [job for job in jobs if not job_output(run_dir, *job).exists()]
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = str(args.threads)
    print(f"남은 작업 {len(pending)}/{len(jobs)}개, 동시 상한 {args.workers}(300열대 shrunk는 작업당 10GB대, 상한 3), 스레드 {args.threads}", flush=True)
    active: dict[tuple[str, str], subprocess.Popen] = {}
    results: dict[tuple[str, str], int] = {}
    while pending or active:
        for job, process in list(active.items()):
            code = process.poll()
            if code is not None:
                results[job] = code
                del active[job]
                print(f"{'완료' if code == 0 else f'실패({code})'} {job[0]} {job[1]}", flush=True)
        running = _running_jobs(run_dir) | set(active)
        while pending and len(running) < args.workers:
            job = pending.pop(0)
            if job in running or job_output(run_dir, *job).exists():
                continue
            handle = (log_dir / f"{job[0]}-{job[1]}.log").open("w")
            active[job] = subprocess.Popen(job_command(run_dir, *job), env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.add(job)
            print(f"시작 {job[0]} {job[1]}", flush=True)
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


def adr0005_compare(args: argparse.Namespace) -> None:
    """ADR-0005 보존: 후보 절차 팔(selection.json)과 비교 팔의 이어붙인 비교. 사다리 판정에서 쓰지 않는다."""
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
        "rows": len(all_rows),
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


def _load_baseline(run_dir: Path, payload: dict, fold_of: pd.Series, y: pd.Series) -> tuple[pd.Series, dict[str, dict]]:
    """비교 팔 313의 봉인 분할 예측 5개를 원래 행 순서로 모으고 분할별 AUC·λ를 돌려준다."""
    baseline = pd.Series(np.nan, index=fold_of.index)
    per_fold: dict[str, dict] = {}
    for fold in payload["outer_folds"]:
        fold_dir = run_dir / f"fold-{fold}"
        path = fold_dir / "baseline.json"
        _require(path.is_file(), f"분할 {fold}의 baseline 산출물이 없다. 전체 판정은 판정 불가다.")
        record = read_json(path)
        _require(record["precommit_sha256"] == payload["precommit_sha256"], f"분할 {fold}의 baseline이 다른 precommit에서 나왔다.")
        base = pd.read_parquet(fold_dir / "baseline-predictions.parquet").set_index(ID)["prediction"]
        ids = fold_of.index[(fold_of == fold).to_numpy()]
        _require(base.index.equals(pd.Index(ids)), f"분할 {fold}의 baseline 예측 id가 분할 배정과 다르다.")
        _require(prediction_array_sha256(base.to_numpy()) == record["prediction_sha256"], f"분할 {fold}의 baseline 예측 해시가 baseline.json과 다르다.")
        baseline.loc[ids] = base.to_numpy()
        per_fold[str(fold)] = {"auc": float(roc_auc_score(y.loc[ids].to_numpy(), base.to_numpy())), "lambda": record["lambda"], "c": record.get("c"), "prediction_sha256": record["prediction_sha256"]}
    return baseline, per_fold


def _load_ladder_result(run_dir: Path, payload: dict, config: dict, fold_of: pd.Series, y: pd.Series) -> tuple[dict, pd.Series]:
    """구성의 nested.json과 분할 예측을 읽어 해시와 이어붙인 AUC를 다시 확인한다."""
    out_dir = run_dir / LADDER_DIR / config["name"]
    path = out_dir / "nested.json"
    _require(path.is_file(), f"사다리 구성 {config['name']}의 산출물이 없다. 전체 판정은 판정 불가다.")
    record = read_json(path)
    _require(record["precommit_sha256"] == payload["precommit_sha256"], f"{config['name']}: 다른 precommit에서 나왔다.")
    _require(record["columns"] == config["columns"], f"{config['name']}: 산출물의 열 순서가 precommit과 다르다.")
    prediction = pd.Series(np.nan, index=fold_of.index)
    for fold in payload["outer_folds"]:
        part = pd.read_parquet(out_dir / f"fold-{fold}" / "predictions.parquet").set_index(ID)["prediction"]
        ids = fold_of.index[(fold_of == fold).to_numpy()]
        _require(part.index.equals(pd.Index(ids)), f"{config['name']}: 분할 {fold} 예측 id가 분할 배정과 다르다.")
        _require(prediction_array_sha256(part.to_numpy()) == record["folds"][str(fold)]["prediction_sha256"], f"{config['name']}: 분할 {fold} 예측 해시가 nested.json과 다르다.")
        _require(abs(float(roc_auc_score(y.loc[ids].to_numpy(), part.to_numpy())) - record["fold_aucs"][str(fold)]) <= CONCAT_AUC_TOLERANCE, f"{config['name']}: 분할 {fold} AUC 재계산이 nested.json과 다르다.")
        prediction.loc[ids] = part.to_numpy()
    mask = prediction.notna().to_numpy()
    recomputed = float(roc_auc_score(y.to_numpy()[mask], prediction.to_numpy()[mask]))
    _require(abs(recomputed - record["nested_auc"]) <= CONCAT_AUC_TOLERANCE, f"{config['name']}: 이어붙인 AUC 재계산 {recomputed:.9f}이 nested.json {record['nested_auc']:.9f}과 다르다.")
    return record, prediction


def compare(args: argparse.Namespace) -> None:
    """자기 검사 → 구성별 313 대비 차이·분할 부호 → 문턱 → 선택 규칙. `ladder-comparison.json`."""
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    fold_of, y = load_folds_and_labels()
    folds = payload["outer_folds"]
    complete = list(folds) == list(ALL_FOLDS)
    configs = payload["ladder"]["configs"]
    record: dict[str, object] = {
        "schema": SCHEMA,
        "candidate_set_id": payload["candidate_set_id"],
        "precommit_sha256": payload["precommit_sha256"],
        "rehearsal": payload["rehearsal"],
        "outer_folds": folds,
        "complete": complete,
        "gate": payload["gate"],
        "noise_floor": NOISE_FLOOR,
        "selection_rules": payload["selection_rules"],
        "exact_duplicates": payload["exact_duplicates"],
        "ladder_candidate_count": payload["ladder"]["candidate_count"],
    }
    if not configs:
        record.update(self_check=None, configs=[], passing_configs=[], selected_config=None, verdict="사다리 후보 없음: 현재 두 장(e88f706e + 30b6f97c) 유지", compared_at=now_iso())
        write_json(run_dir / "ladder-comparison.json", record)
        print(record["verdict"])
        return

    baseline, base_folds = _load_baseline(run_dir, payload, fold_of, y)
    mask = baseline.notna().to_numpy()
    base_auc = float(roc_auc_score(y.to_numpy()[mask], baseline.to_numpy()[mask]))
    reference = payload["self_check"]
    fold_deltas_vs_reference = {k: base_folds[k]["auc"] - reference["reference_fold_aucs"][k] for k in base_folds}
    self_check = {
        "reference_nested_auc": reference["reference_nested_auc"],
        "baseline_concatenated_auc": base_auc,
        "delta": base_auc - reference["reference_nested_auc"],
        "baseline_fold_aucs": {k: v["auc"] for k, v in base_folds.items()},
        "baseline_fold_lambdas": {k: v["lambda"] for k, v in base_folds.items()},
        "baseline_fold_cs": {k: v["c"] for k, v in base_folds.items()},
        "reference_run_id": payload["comparison_arm"]["reference"].get("run_id"),
        "fold_deltas_vs_reference": fold_deltas_vs_reference,
        "tolerance": NOISE_FLOOR,
        "passes": bool(complete and abs(base_auc - reference["reference_nested_auc"]) <= NOISE_FLOOR and all(abs(v) <= NOISE_FLOOR for v in fold_deltas_vs_reference.values())),
    }

    results: list[dict] = []
    for config in configs:
        nested_record, _ = _load_ladder_result(run_dir, payload, config, fold_of, y)
        fold_deltas = {k: nested_record["fold_aucs"][k] - base_folds[k]["auc"] for k in base_folds}
        delta = nested_record["nested_auc"] - base_auc
        positives = sum(v > 0.0 for v in fold_deltas.values())
        results.append({
            "index": config["index"],
            "name": config["name"],
            "kind": config["kind"],
            "description": config["description"],
            "member_count": config["member_count"],
            "candidate_member_ids": config["candidate_member_ids"],
            "removed_member_ids": config["removed_member_ids"],
            "nested_auc": nested_record["nested_auc"],
            "weighted_oof_auc": nested_record["weighted_oof_auc"],
            "fold_aucs": nested_record["fold_aucs"],
            "fold_lambdas": {k: v["lambda"] for k, v in nested_record["folds"].items()},
            "fold_cs": {k: v.get("c") for k, v in nested_record["folds"].items()},
            "delta_vs_313": delta,
            "delta_minus_gate": delta - GATE_DELTA,
            "delta_within_noise_floor": abs(delta) <= NOISE_FLOOR,
            "weighted_delta_vs_313_reference": nested_record["weighted_oof_auc"] - payload["comparison_arm"]["reference"]["weighted_oof_auc"],
            "fold_deltas_vs_313": fold_deltas,
            "folds_positive": positives,
            "passes_gate": bool(self_check["passes"] and delta >= GATE_DELTA and positives >= FOLDS_REQUIRED_POSITIVE),
            "elapsed_seconds": nested_record["elapsed_seconds"],
        })
    passing = [r for r in results if r["passes_gate"]]
    selected = None
    selection_trace = None
    if passing:
        top = max(passing, key=lambda r: r["nested_auc"])
        tied = [r for r in passing if top["nested_auc"] - r["nested_auc"] <= NOISE_FLOOR]
        chosen = min(tied, key=lambda r: (r["member_count"], r["index"]))
        selected = chosen["name"]
        selection_trace = {"highest_nested": top["name"], "within_noise_floor_of_highest": [r["name"] for r in tied], "fewest_members_then_ladder_order": chosen["name"]}
    if not complete:
        verdict = "판정 불가(부분 분할, 예행)"
    elif not self_check["passes"]:
        verdict = "판정 불가(비교 팔 자기 검사 실패)"
    elif selected is not None:
        verdict = "통과"
    else:
        verdict = "미달: 현재 두 장(e88f706e + 30b6f97c) 유지"
    record.update(self_check=self_check, comparison_arm_auc=base_auc, configs=results, passing_configs=[r["name"] for r in passing], selection=selection_trace, selected_config=selected, verdict=verdict, rows_scored=int(mask.sum()), compared_at=now_iso())
    write_json(run_dir / "ladder-comparison.json", record)
    print(f"자기 검사: 313 이어붙인 {base_auc:.7f} (#489 {reference['reference_nested_auc']:.7f}, 차이 {self_check['delta']:+.2e}, 분할 최대 {max(abs(v) for v in fold_deltas_vs_reference.values()):.2e}) → {'통과' if self_check['passes'] else '실패'}")
    for r in results:
        print(f"  {r['name']:<44} {r['member_count']:>4} nested {r['nested_auc']:.7f} 가중 {r['weighted_oof_auc']:.7f} Δ {r['delta_vs_313']:+.7f} 분할 {r['folds_positive']}/5{' 통과' if r['passes_gate'] else ''}")
    print(f"판정: {verdict}" + (f", 선택 {selected}" if selected else ""))


def report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = load_precommit(run_dir)
    comparison_path = run_dir / "ladder-comparison.json"
    _require(comparison_path.is_file(), "ladder-comparison.json이 없다. compare를 먼저 한다.")
    comparison = read_json(comparison_path)
    _require(comparison["precommit_sha256"] == payload["precommit_sha256"], "ladder-comparison.json이 다른 precommit에서 나왔다.")
    by_member = {c["member_id"]: c for c in payload["candidate_arm"]["candidates"]}
    lines: list[str] = []
    lines += [f"# 엄격 외부 후보 사다리 판정 보고 (`{payload['candidate_set_id']}`)", ""]
    if payload["rehearsal"]:
        lines += ["**예행 실행이다.** 합성 후보로 도구 경로만 확인했으며 실제 후보 판정이 아니다.", ""]
    lines += ["## 판정", ""]
    lines += [f"- 결과: **{comparison['verdict']}**" + (f", 제안 구성 `{comparison['selected_config']}`" if comparison.get("selected_config") else "")]
    lines += [f"- 문턱: 313 대비 이어붙인 nested AUC 차이 `+{GATE_DELTA}` 이상, 바깥 분할 {FOLDS_REQUIRED_POSITIVE}/5 엄격 양수. 결과 확인 뒤 바꾸지 않는다."]
    lines += [f"- 선택 규칙: {payload['selection_rules']['pick']}"]
    lines += [f"- 동결 후보 {payload['freeze_spec']['candidate_count']}개 가운데 정확 중복 {len(payload['exact_duplicates'])}개를 뺀 사다리 후보 {payload['ladder']['candidate_count']}개, 구성 {payload['ladder']['config_count']}개."]
    self_check = comparison.get("self_check")
    if self_check is not None:
        lines += [f"- 비교 팔 자기 검사: 313 이어붙인 AUC {self_check['baseline_concatenated_auc']:.7f}, #489 규제 강도 선택판 기준 {self_check['reference_nested_auc']:.7f}, 차이 `{self_check['delta']:+.2e}`, 분할 최대 차이 `{max(abs(v) for v in self_check['fold_deltas_vs_reference'].values()):.2e}`, 잡음 바닥 `{NOISE_FLOOR}` → **{'통과' if self_check['passes'] else '실패(판정 불가)'}**"]
    if comparison.get("selection"):
        trace = comparison["selection"]
        lines += [f"- 선택 경로: nested 최고 `{trace['highest_nested']}`, 잡음 바닥 안 {', '.join(f'`{n}`' for n in trace['within_noise_floor_of_highest'])} → 구성원 적은 쪽·사다리 순서 `{trace['fewest_members_then_ladder_order']}`"]
    lines += [""]
    if comparison.get("configs"):
        folds = comparison["outer_folds"]
        lines += ["## 사다리", "", "| 순서 | 구성 | 구성원 | nested AUC | 가중 OOF(진단) | 313 대비 | " + " | ".join(f"분할 {k}" for k in folds) + " | 양수 | 통과 |", "| ---: | --- | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in folds) + " | ---: | --- |"]
        for r in comparison["configs"]:
            lines.append(f"| {r['index']} | `{r['name']}` | {r['member_count']} | {r['nested_auc']:.7f} | {r['weighted_oof_auc']:.7f} | {r['delta_vs_313']:+.7f} | " + " | ".join(f"{r['fold_deltas_vs_313'][str(k)]:+.7f}" for k in folds) + f" | {r['folds_positive']}/5 | {'통과' if r['passes_gate'] else '-'} |")
        lines += ["", "구성별 후보:", ""]
        for r in comparison["configs"]:
            lines.append(f"- `{r['name']}`: {r['description']}. 후보 {len(r['candidate_member_ids'])}개" + (f", 뺀 후보 {', '.join(f'`{m}`' for m in r['removed_member_ids'])}" if r["removed_member_ids"] else "") + f", C {r['fold_cs']}, λ {r['fold_lambdas']}")
        if self_check is not None:
            lines += ["", "비교 팔 313 분할별 AUC(자기 검사 기준): " + ", ".join(f"분할 {k} {v:.7f}" for k, v in self_check["baseline_fold_aucs"].items()) + f", C {self_check['baseline_fold_cs']}, λ {self_check['baseline_fold_lambdas']}"]
    lines += ["", "## 정확 중복(자동 제외)", ""]
    if payload["exact_duplicates"]:
        lines += ["| 동결 순서 | 후보 | 313 구성원 |", "| ---: | --- | --- |"]
        lines += [f"| {d['order']} | `{d['member_id']}` | `{d['matches_column']}` |" for d in payload["exact_duplicates"]]
    else:
        lines += ["- 없음"]
    if payload["oof_only_matches"]:
        lines += ["", "OOF만 같은 후보(정확 중복 아님, 진단): " + ", ".join(f"`{d['member_id']}`=`{d['matches_column']}`" for d in payload["oof_only_matches"])]
    lines += ["", "## 사다리 후보(동결 순서, 단독 AUC는 진단값)", "", "| 동결 순서 | 후보 | 출처 | 주의 사항 | 전체 OOF 단독 AUC |", "| ---: | --- | --- | --- | ---: |"]
    for member_id in payload["ladder"]["candidate_member_ids"]:
        c = by_member[member_id]
        lines.append(f"| {c['order']} | `{member_id}` | {c['source']} | {', '.join(c['caveat_codes'])} | {c['rescored_auc']:.6f} |")
    if payload["ladder"]["omitted"]:
        lines += ["", "생략한 구성: " + "; ".join(f"`{o['name']}` ({o['reason']})" for o in payload["ladder"]["omitted"])]
    full_path = run_dir / LADDER_DIR / FULL_CONFIG / "nested.json"
    if full_path.is_file():
        diagnostics = read_json(full_path).get("near_duplicate_diagnostics")
        if diagnostics:
            lines += ["", f"## 근접 중복 진단(스피어만 {DUPLICATE_SPEARMAN} 이상, 열린 4분할, 제외 없음)", "", "| 봉인 분할 | 쌍 수 | 종류별 | 후보가 낀 쌍 |", "| ---: | ---: | --- | --- |"]
            for fold, entry in diagnostics["by_sealed_fold"].items():
                involving = [p for p in entry["pairs"] if "candidate" in p["kind"]]
                lines.append(f"| {fold} | {entry['pair_count']} | {entry['counts_by_kind']} | {', '.join(f'`{p['a']}`·`{p['b']}` {p['spearman']:.6f}' for p in involving) or '-'} |")
    lines += ["", "## 실행 인계 완결 조건(#481, ADR-0006 개정 항목) 대조", ""]
    lines += [
        f"- 변경 불가 감사 기록·자격 판정: 동결 명세 `{payload['freeze_spec']['path']}` (spec_sha256 `{payload['freeze_spec']['sha256']}`)의 후보 {payload['freeze_spec']['candidate_count']}개, 사용자 제외 {len(payload['freeze_spec']['user_exclusions'])}개, 조사 기준 시각 `{payload['freeze_spec']['survey_cutoff']}`.",
        f"- 313개와 사다리 후보의 입력 명세: 비교 팔 313 구성 해시 `{payload['comparison_arm']['composition_sha256']}`, 후보 구성 해시 `{payload['candidate_arm']['composition_sha256']}`, 자체 35 `{payload['own_start']['composition_sha256']}`.",
        "- 사다리 구성 목록·선택 규칙: `precommit.json`의 `ladder`·`selection_rules`.",
        f"- 교체 문턱: `precommit.json`의 `gate` (+{GATE_DELTA}, {FOLDS_REQUIRED_POSITIVE}/5), 잡음 바닥 {NOISE_FLOOR}.",
        "- 비교 팔 자기 검사·구성별 봉인 예측·사다리 비교: `fold-<k>/baseline-predictions.parquet`·`baseline.json`, `ladder/<구성>/fold-<k>/predictions.parquet`·`nested.json`, `ladder-comparison.json`.",
        "- 실패·재개 규칙: `precommit.json`의 `rules` (모든 하위 명령이 입력 해시와 코드 상태를 다시 확인).",
        "- 실행 경계: 조립·업로드와 최종 두 장 수동 고정은 사용자 승인 뒤에만(#488).",
        "",
        "## 코드 상태",
        "",
        f"- git `{payload['code_state']['git']['commit']}` (dirty {payload['code_state']['git']['dirty']}), 판정 도구 sha256 `{payload['code_state']['script']['sha256']}`, 동결 생성기 sha256 `{payload['code_state']['freeze_script_sha256']}`, 결합기 module sha256 `{payload['code_state']['ensemble_module']['sha256']}`, sklearn {payload['code_state']['sklearn']}, numpy {payload['code_state']['numpy']}.",
        f"- precommit_sha256 `{payload['precommit_sha256']}`, 비교 {comparison['compared_at']}, 보고 작성 {now_iso()}.",
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
    comparison = read_json(run_dir / "ladder-comparison.json")
    _require(comparison["precommit_sha256"] == payload["precommit_sha256"], "ladder-comparison.json이 다른 precommit에서 나왔다.")
    _require(comparison["complete"] and comparison["self_check"] and comparison["self_check"]["passes"], "자기 검사를 통과한 완전한 판정이 없다.")
    _require(comparison["selected_config"] is not None, "문턱을 통과한 구성이 없다. 현재 두 장을 유지한다.")
    config = next(c for c in payload["ladder"]["configs"] if c["name"] == comparison["selected_config"])
    result = next(r for r in comparison["configs"] if r["name"] == config["name"])
    _require(result["passes_gate"], f"{config['name']}이 문턱을 넘지 못했다.")
    started = time.monotonic()
    fold_of, y = load_folds_and_labels()
    oof = load_ladder_matrix(run_dir, payload, config, fold_of)
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID])[ID]
    _require(len(test_ids) == N_TEST and not test_ids.duplicated().any(), "test.csv의 행 수나 id가 기대와 다르다.")
    own_test = pd.read_parquet(OWN_TEST_PATH)
    _require(own_test[ID].to_numpy().tolist() == test_ids.to_numpy().tolist(), "5:1 혼합판 시험 예측의 id 순서가 test.csv와 다르다.")
    own_test = own_test.set_index(ID)
    by_comparison = {row["column"]: row for row in payload["comparison_arm"]["members"]}
    by_candidate = {row["column"]: row for row in payload["candidate_arm"]["candidates"]}
    columns: dict[str, np.ndarray] = {}
    sources: dict[str, dict] = {}
    for column in config["columns"]:
        if column in by_candidate:
            entry = by_candidate[column]
            values = freeze.load_array(Path(entry["test_path"]), N_TEST, entry["member_id"])
            _require(freeze.array_sha256(values) == entry["test_sha256"], f"{entry['member_id']}: 시험 배열 해시가 동결 명세와 다르다.")
            sources[column] = {"kind": "external_frozen_candidate", "test_path": entry["test_path"], "member_id": entry["member_id"], "audit_record_id": entry["audit_record_id"]}
        else:
            entry = by_comparison[column]
            if entry["origin"] == "own":
                values = own_test[column].to_numpy(np.float64)
                sources[column] = {"kind": "own_cv5_full1_mix", "test_path": str(OWN_TEST_PATH)}
            else:
                values = ladder.load_ledger_array(entry["test_path"])
                sources[column] = {"kind": "external_cv_fold_average", "test_path": entry["test_path"]}
            _require(values.shape == (N_TEST,) and bool(np.isfinite(values).all()), f"{column}: 시험 배열 형태 {values.shape} 또는 비유한값")
            _require(prediction_array_sha256(values) == entry["test_prediction_sha256"], f"{column}: 시험 배열 해시가 #457 manifest와 다르다.")
        columns[column] = values
        sources[column]["prediction_sha256"] = prediction_array_sha256(values)
    test = pd.DataFrame(columns, index=test_ids.to_numpy()).astype(np.float64)
    _require(list(test.columns) == list(oof.columns), "시험 행렬의 열 순서가 OOF와 다르다.")
    combiner = ladder_combiner(fold_of)
    fitted = combiner.fit(oof, y)
    prediction = ensemble.full_fit_predictions(combiner, oof, y, test)
    in_sample_auc = float(roc_auc_score(y.to_numpy(), np.asarray(fitted.predict(oof), dtype=np.float64)))
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / f"strict-external-{payload['candidate_set_id']}-{config['name']}.csv"
    pd.DataFrame({ID: test_ids.to_numpy(), TARGET: prediction}).to_csv(submission_path, index=False)
    manifest = {
        "schema_version": 2,
        "issue": ISSUE,
        "candidate_set_id": payload["candidate_set_id"],
        "precommit_sha256": payload["precommit_sha256"],
        "git": git_state(),
        "strategy": LADDER_STRATEGY,
        "judged": {"config": config["name"], "nested_auc": result["nested_auc"], "weighted_oof_auc": result["weighted_oof_auc"], "delta_vs_313": result["delta_vs_313"], "folds_positive": result["folds_positive"], "comparison_arm_auc": comparison["comparison_arm_auc"], "verdict": comparison["verdict"], "self_check_passes": comparison["self_check"]["passes"]},
        "assembled": {"member_count": config["member_count"], "own_member_count": payload["own_start"]["member_count"], "comparison_external_count": payload["comparison_arm"]["member_count"] - payload["own_start"]["member_count"], "candidate_count": len(config["candidate_member_ids"]), "candidate_member_ids": config["candidate_member_ids"]},
        "combiner": {"c": float(fitted.c), "shrinkage_lambda": float(fitted.shrinkage_lambda), "c_grid": list(combiner.c_grid), "lambda_grid": list(combiner.lambda_grid), "selection_aucs": [{"c": c, "lambda": lam, "auc": value} for (c, lam), value in fitted.selection_aucs.items()], "final_iterations": fitted.final_iterations, "final_coefficient_l2_norm": fitted.final_coefficient_l2_norm, "fit_protocol": "전체 OOF 1회 적합(ensemble.full_fit_predictions), (C, λ)는 5분할 leave-one-fold-out", "in_sample_oof_auc": in_sample_auc},
        "inputs": {"train_sha256": file_sha256(TRAIN_PATH), "test_sha256": file_sha256(TEST_PATH), "folds_sha256": file_sha256(FOLDS_PATH), "pool_sha256": file_sha256(POOL_PATH), "comparison_manifest_sha256": payload["comparison_arm"]["manifest_sha256"], "freeze_spec": payload["freeze_spec"], "own_test": {"kind": "cv5_full1_mix", "path": str(OWN_TEST_PATH), "full_refit_manifest_sha256": file_sha256(FULL_REFIT_MANIFEST_PATH)}},
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
    parser = argparse.ArgumentParser(description="엄격 외부 후보 사다리 판정 (#491, ADR-0006)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("precommit", help="동결 입력·정확 중복·사다리 구성·선택 규칙·코드 상태를 고정하고 캐시를 만든다.")
    p.add_argument("--spec", type=Path, required=True, help="외부 후보 동결 명세")
    p.add_argument("--run-dir", type=Path, help="출력 폴더(기본 run-logs/strict-external-selection/<후보 집합 식별자>)")
    p.add_argument("--folds", type=int, nargs="+", help="예행 전용: 봉인할 바깥 분할 부분집합")
    p.set_defaults(func=precommit)

    p = sub.add_parser("baseline", help="비교 팔 313의 봉인 분할 예측(자기 검사·분할별 비교 기준)")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=baseline_fold)

    p = sub.add_parser("ladder", help="사다리 구성 하나의 nested 5분할")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=ladder_job)

    p = sub.add_parser("run", help="남은 (baseline, ladder) 작업을 하위 프로세스로 실행한다.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=3, help="동시 작업 상한(300열대 shrunk 작업당 10GB대, 48GB 기계에서 3)")
    p.add_argument("--threads", type=int, default=4)
    p.set_defaults(func=run_jobs)

    for name, func in (("compare", compare), ("report", report), ("assemble", assemble)):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
        p.set_defaults(func=func)

    # ADR-0005 보존 명령. 사다리 판정에서 쓰지 않는다.
    p = sub.add_parser("adr0005-select", help="(보존) 바깥 분할 하나의 후보 정확 검색과 봉인 예측")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-registered-check", action="store_true", help="등록 결합기 대조를 건너뛴다(권장하지 않음)")
    p.set_defaults(func=select_fold)
    p = sub.add_parser("adr0005-compare", help="(보존) 후보 절차 팔과 비교 팔의 이어붙인 비교")
    p.add_argument("--run-dir", type=Path, required=True)
    p.set_defaults(func=adr0005_compare)
    p = sub.add_parser("adr0005-full", help="(보존) 절차 통과 뒤 전체 OOF에 동결 검색을 한 번 적용한다.")
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
