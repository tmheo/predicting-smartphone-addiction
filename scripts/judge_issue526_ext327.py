"""엄격 후보 13개를 314 확장 팔에 더한 327 구성원을 판정하고 조립한다. (#526 확장 회차)

비교 팔은 현재 두 번째 제출인 #514의 314 확장 결합(자체 36 + 외부 278,
nested OOF 0.9703843058098193, MLflow 3279e114)이며, #513 판정 기록을
기준값으로 고정한다. 후보 팔은 314 열 뒤에 #526 동결 명세
`ecf-v3-b18bc301d500`의 사다리 후보 13개를 사다리 열 순서로 붙인 327 구성이다.

- 자체 36 OOF는 #513 precommit의 (column, run_id)로 적재하고 열마다
  기록된 OOF 해시와 대조한다.
- 외부 278 OOF는 #526 실행 캐시(313 비교 팔)에서 열을 골라 #513 기록과
  열 이름·해시를 대조한다.
- 후보 13 OOF는 #526 실행 캐시에서 골라 동결 명세의 배열 해시와 대조한다.
- 자기 검사는 314 열만으로 봉인 분할 0을 다시 만들어 #513의 분할 0
  예측 해시와 일치해야 한다.

판정 문턱은 314 팔 대비 nested OOF `+0.00002` 이상과 바깥쪽 분할 `5/5`
양수다. Public 점수는 판정에 쓰지 않는다.

실행 순서:

    uv run python scripts/judge_issue526_ext327.py precommit
    uv run python scripts/judge_issue526_ext327.py run --workers 3 --threads 4
    uv run python scripts/judge_issue526_ext327.py compare
    uv run python scripts/judge_issue526_ext327.py report
    uv run python scripts/judge_issue526_ext327.py publish
    uv run python scripts/judge_issue526_ext327.py assemble --full-refit-dir <#514 전체 재학습 폴더>
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge_issue513_extended_stack_reassembly as m513
import judge_strict_external_selection as strict
import freeze_external_candidates as freeze

base = m513.base
logistic = m513.logistic

from pipeline import ensemble, refit
from pipeline.data import ID, TARGET, TRAIN_PATH, file_sha256, labels
from pipeline.judgment import FOLDS_PATH
from pipeline.pool_audit import prediction_array_sha256
from pipeline.runs import MlflowRunStore

read_json = logistic.read_json
write_json = logistic.write_json
canonical_sha256 = logistic.canonical_sha256
now_iso = logistic.now_iso
JudgmentError = logistic.JudgmentError
_require = logistic._require

ISSUE = 526
SCHEMA = "extended-stack-ext327/1"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = Path("run-logs/issue526-ext327")
PUBLISH_DIR = Path("docs/research/extended-stack-ext327/issue526")
CACHE_NAME = "ext327-oof.parquet"

ISSUE513_PRECOMMIT_PATH = Path("docs/research/extended-stack-pool-reassembly/issue513/precommit.json")
ISSUE513_COMPARISON_PATH = Path("docs/research/extended-stack-pool-reassembly/issue513/comparison.json")
ISSUE513_FOLD0_PATH = Path("docs/research/extended-stack-pool-reassembly/issue513/reassembled/fold-0/reassembled.json")
ISSUE514_RECORD_PATH = Path("docs/research/extended-stack-final-assembly/issue514/submission-record.json")
ISSUE489_PRECOMMIT_PATH = base.BASELINE_PRECOMMIT_PATH
BASELINE_MANIFEST_PATH = base.BASELINE_MANIFEST_PATH
STRICT_RUN_DIR = Path("run-logs/strict-external-selection/ecf-v3-b18bc301d500")
FREEZE_SPEC_PATH = Path("docs/research/external-candidate-freeze/ecf-v3-b18bc301d500.json")
PLAN_PATH = Path("artifacts/full-refit-plan.yaml")

COMPARISON_MEMBER_COUNT = 314
OWN_MEMBER_COUNT = 36
EXTERNAL_MEMBER_COUNT = 278
CANDIDATE_COUNT = 13
EXT327_MEMBER_COUNT = 327
LADDER_CONFIG_NAME = "ext313_strict_all"
ALL_FOLDS = (0, 1, 2, 3, 4)
GATE_DELTA = 0.00002
FOLDS_REQUIRED_POSITIVE = 5
MAX_WORKERS = 3
C_GRID = m513.C_GRID
LAMBDA_GRID = m513.LAMBDA_GRID
META_MAX_ITER = m513.META_MAX_ITER


def _assert_repo_root() -> None:
    _require(Path.cwd().resolve() == REPO_ROOT, f"저장소 루트에서 실행해야 한다: {REPO_ROOT}")


def _code_state() -> dict[str, object]:
    return {
        "git": logistic.strict.git_state(),
        "script": {"path": "scripts/judge_issue526_ext327.py", "sha256": file_sha256(Path(__file__))},
        "ensemble_module": {"sha256": file_sha256(Path(ensemble.__file__))},
        "uv_lock_sha256": file_sha256(Path("uv.lock")),
    }


def _tracked(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _verify_frozen_json(path: Path) -> dict:
    payload = read_json(path)
    digest = canonical_sha256({k: v for k, v in payload.items() if k != "precommit_sha256"})
    _require(digest == payload["precommit_sha256"], f"{path}이 제자리에서 바뀌었다.")
    return payload


def _load_folds_and_labels() -> tuple[pd.Series, pd.Series]:
    fold_of = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"]
    y = labels(fold_of.index)
    return fold_of, y


def _build_matrix(fold_of: pd.Series) -> tuple[pd.DataFrame, dict]:
    """327 OOF 행렬을 만들고 열마다 동결 기록과 해시를 대조한다."""
    i513 = _verify_frozen_json(ISSUE513_PRECOMMIT_PATH)
    members_314 = i513["reassembled"]["members"]
    _require(len(members_314) == COMPARISON_MEMBER_COUNT, "이슈 513 재조립 팔이 314개가 아니다.")
    own_members = members_314[:OWN_MEMBER_COUNT]
    ext_members = members_314[OWN_MEMBER_COUNT:]
    _require(all(row["origin"] == "own" for row in own_members), "이슈 513 앞 36개가 자체 구성원이 아니다.")
    _require(all(row["origin"] == "external" for row in ext_members), "이슈 513 뒤 278개가 외부 구성원이 아니다.")

    store = MlflowRunStore()
    own = ensemble.member_matrix(
        [(row["column"], row["run_id"]) for row in own_members], store, fold_of.index
    ).astype(np.float64)
    for row in own_members:
        digest = prediction_array_sha256(own[row["column"]].to_numpy(np.float64))
        _require(digest == row["oof_sha256"], f"{row['column']}: 자체 OOF 해시가 이슈 513과 다르다.")

    strict_pre = _verify_frozen_json(STRICT_RUN_DIR / "precommit.json")
    for name, digest in strict_pre["caches"].items():
        _require(file_sha256(STRICT_RUN_DIR / "cache" / name) == digest, f"#526 캐시 {name}이 precommit과 다르다.")
    arm313 = pd.read_parquet(STRICT_RUN_DIR / "cache" / "comparison-arm-oof.parquet").astype(np.float64)
    ext_columns = [row["column"] for row in ext_members]
    _require(set(ext_columns) <= set(arm313.columns), "이슈 513 외부 열이 #526 비교 팔 캐시에 없다.")
    ext = arm313[ext_columns]
    for row in ext_members:
        digest = prediction_array_sha256(ext[row["column"]].to_numpy(np.float64))
        _require(digest == row["oof_sha256"], f"{row['column']}: 외부 OOF 해시가 이슈 513과 다르다.")

    spec = freeze.verify_spec_file(FREEZE_SPEC_PATH)
    ladder_config = next(
        config for config in strict_pre["ladder"]["configs"] if config["name"] == LADDER_CONFIG_NAME
    )
    candidate_columns = list(ladder_config["candidate_columns"])
    _require(len(candidate_columns) == CANDIDATE_COUNT, f"사다리 후보 열이 {CANDIDATE_COUNT}개가 아니다.")
    candidates_all = pd.read_parquet(STRICT_RUN_DIR / "cache" / "candidates-oof.parquet").astype(np.float64)
    spec_by_member = {row["member_id"]: row for row in spec["candidates"]}
    candidates = candidates_all[candidate_columns]
    candidate_rows = []
    for column in candidate_columns:
        member_id = column.removeprefix("cand_")
        frozen = spec_by_member[member_id]
        digest = freeze.array_sha256(candidates[column].to_numpy(np.float64))
        _require(digest == frozen["oof_sha256"], f"{column}: 후보 OOF 해시가 동결 명세와 다르다.")
        candidate_rows.append(
            {
                "column": column,
                "member_id": member_id,
                "origin": "candidate",
                "oof_sha256": prediction_array_sha256(candidates[column].to_numpy(np.float64)),
                "frozen_oof_sha256": frozen["oof_sha256"],
                "test_path": frozen["test_path"],
                "test_sha256": frozen["test_sha256"],
            }
        )

    matrix = pd.concat([own, ext, candidates], axis=1)
    _require(
        matrix.shape == (len(fold_of), EXT327_MEMBER_COUNT)
        and bool(np.isfinite(matrix.to_numpy()).all()),
        "327 행렬 형태나 유한값 검사가 실패했다.",
    )
    meta = {
        "issue513_precommit": i513,
        "strict_precommit": strict_pre,
        "members": [*own_members, *ext_members, *candidate_rows],
        "candidate_rows": candidate_rows,
    }
    return matrix, meta


def precommit(args: argparse.Namespace) -> None:
    _assert_repo_root()
    _require(not (RUN_DIR / "precommit.json").exists(), f"precommit.json이 이미 있다: {RUN_DIR}")
    state = _code_state()
    _require(not state["git"]["dirty"], "판정은 커밋된 코드 상태에서만 시작한다.")
    fold_of, _ = _load_folds_and_labels()
    matrix, meta = _build_matrix(fold_of)
    i513 = meta["issue513_precommit"]
    _require(
        state["ensemble_module"]["sha256"] == i513["code_state"]["ensemble_module"]["sha256"],
        "결합기 구현이 이슈 513 판정 때와 다르다.",
    )
    _require(
        state["uv_lock_sha256"] == i513["inputs"]["uv_lock"]["sha256"],
        "실행 환경 잠금 파일이 이슈 513 판정 때와 다르다.",
    )
    comparison = read_json(ISSUE513_COMPARISON_PATH)
    _require(
        comparison["precommit_sha256"] == i513["precommit_sha256"],
        "이슈 513 comparison이 precommit과 다른 판정이다.",
    )
    record514 = read_json(ISSUE514_RECORD_PATH)
    _require(
        record514["candidates"]["extended314_own_full"]["nested_oof_auc"]
        == comparison["reassembled"]["nested_auc"],
        "현재 두 번째 제출의 nested OOF가 이슈 513 판정과 다르다.",
    )
    fold0 = read_json(ISSUE513_FOLD0_PATH)
    _require(fold0["sealed_fold"] == 0, "이슈 513 분할 0 기록이 아니다.")
    cache_dir = RUN_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(cache_dir / CACHE_NAME)
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "created_at": now_iso(),
        "question": (
            "엄격 후보 13개를 현재 두 번째 제출의 314 확장 팔에 더한 327 구성원이 "
            "314 팔보다 nested OOF +0.00002 이상 높고 바깥쪽 분할 5/5가 양수인가."
        ),
        "outer_folds": list(ALL_FOLDS),
        "inputs": {
            "issue513_precommit": _tracked(ISSUE513_PRECOMMIT_PATH),
            "issue513_comparison": _tracked(ISSUE513_COMPARISON_PATH),
            "issue513_fold0": _tracked(ISSUE513_FOLD0_PATH),
            "issue514_record": _tracked(ISSUE514_RECORD_PATH),
            "strict_precommit": _tracked(STRICT_RUN_DIR / "precommit.json"),
            "freeze_spec": _tracked(FREEZE_SPEC_PATH),
            "train": _tracked(TRAIN_PATH),
            "folds": _tracked(FOLDS_PATH),
            "uv_lock": _tracked(Path("uv.lock")),
        },
        "comparison_arm": {
            "description": "#514 현재 두 번째 제출(자체 36 + 외부 278)",
            "submission_run_id": record514["candidates"]["extended314_own_full"]["mlflow_run_id"],
            "member_count": COMPARISON_MEMBER_COUNT,
            "strategy": comparison["reassembled"]["strategy"],
            "nested_auc": comparison["reassembled"]["nested_auc"],
            "fold_aucs": comparison["reassembled"]["fold_aucs"],
            "prediction_sha256": comparison["reassembled"]["prediction_sha256"],
            "fold0_prediction_sha256": fold0["prediction_sha256"],
        },
        "candidate_arm": {
            "member_count": EXT327_MEMBER_COUNT,
            "own_member_count": OWN_MEMBER_COUNT,
            "external_member_count": EXTERNAL_MEMBER_COUNT,
            "candidate_count": CANDIDATE_COUNT,
            "candidate_set_id": meta["strict_precommit"]["freeze_spec"].get("candidate_set_id")
            or read_json(FREEZE_SPEC_PATH)["candidate_set_id"],
            "ladder_config": LADDER_CONFIG_NAME,
            "members": meta["members"],
            "composition_sha256": canonical_sha256(
                [(row["column"], row["oof_sha256"]) for row in meta["members"]]
            ),
            "strategy": ensemble.CSelectedShrunkRankLogitCombiner.name,
            "c_grid": list(C_GRID),
            "lambda_grid": list(LAMBDA_GRID),
            "max_iter": META_MAX_ITER,
        },
        "cache": {CACHE_NAME: file_sha256(cache_dir / CACHE_NAME)},
        "gate": {
            "delta_required": GATE_DELTA,
            "folds_required_positive": FOLDS_REQUIRED_POSITIVE,
            "public_score_used": False,
        },
        "rules": {
            "selfcheck": (
                "314 열만으로 봉인 분할 0을 다시 만들어 이슈 513 분할 0 예측 해시와 "
                "일치해야 판정을 신뢰한다."
            ),
            "gate": (
                "327 nested OOF에서 314 nested OOF를 뺀 차이가 +0.00002 이상이고 "
                "바깥쪽 분할 5개가 모두 엄격히 양수일 때만 통과한다. "
                "결과를 본 뒤 문턱을 바꾸지 않는다."
            ),
            "scope": (
                "조립, Kaggle 업로드와 최종 두 장 고정은 사용자 승인 뒤에만 한다. "
                "Public 점수는 판정에 쓰지 않는다."
            ),
        },
        "environment": m513._environment(),
        "code_state": state,
    }
    payload["precommit_sha256"] = canonical_sha256(payload)
    write_json(RUN_DIR / "precommit.json", payload)
    print(f"precommit 저장: {RUN_DIR / 'precommit.json'}")
    print(f"  후보 팔 구성 {payload['candidate_arm']['composition_sha256']}")
    print(f"  캐시 {payload['cache'][CACHE_NAME]}")
    print(f"  비교 팔 nested {payload['comparison_arm']['nested_auc']}")


def load_precommit() -> dict:
    _assert_repo_root()
    payload = _verify_frozen_json(RUN_DIR / "precommit.json")
    for key, entry in payload["inputs"].items():
        _require(file_sha256(Path(entry["path"])) == entry["sha256"], f"입력 {key}의 해시가 precommit과 다르다.")
    _require(
        file_sha256(RUN_DIR / "cache" / CACHE_NAME) == payload["cache"][CACHE_NAME],
        "327 OOF 캐시 해시가 precommit과 다르다.",
    )
    state = _code_state()
    for label, actual, frozen in (
        ("git commit", state["git"]["commit"], payload["code_state"]["git"]["commit"]),
        ("판정 도구", state["script"]["sha256"], payload["code_state"]["script"]["sha256"]),
        ("결합기 module", state["ensemble_module"]["sha256"], payload["code_state"]["ensemble_module"]["sha256"]),
        ("실행 환경 잠금", state["uv_lock_sha256"], payload["code_state"]["uv_lock_sha256"]),
    ):
        _require(actual == frozen, f"코드 상태({label})가 precommit과 다르다. precommit부터 다시 한다.")
    return payload


def _load_matrix(payload: dict, fold_of: pd.Series) -> pd.DataFrame:
    matrix = pd.read_parquet(RUN_DIR / "cache" / CACHE_NAME).astype(np.float64)
    expected = [row["column"] for row in payload["candidate_arm"]["members"]]
    _require(list(matrix.columns) == expected, "327 캐시 열 순서가 precommit과 다르다.")
    _require(matrix.index.equals(fold_of.index), "327 캐시 행 순서가 고정 분할과 다르다.")
    return matrix


def _fit_and_seal(matrix: pd.DataFrame, fold_of: pd.Series, y: pd.Series, fold: int) -> tuple[np.ndarray, object]:
    inner = (fold_of != fold).to_numpy()
    outer = (fold_of == fold).to_numpy()
    combiner = ensemble.CSelectedShrunkRankLogitCombiner(
        fold_of=fold_of, c_grid=C_GRID, lambda_grid=LAMBDA_GRID, max_iter=META_MAX_ITER
    )
    fitted = combiner.fit(matrix[inner], y[inner])
    prediction = np.asarray(fitted.predict(matrix[outer]), dtype=np.float64)
    return prediction, fitted


def selfcheck_job(args: argparse.Namespace) -> None:
    payload = load_precommit()
    out_dir = RUN_DIR / "selfcheck"
    _require(not (out_dir / "selfcheck.json").exists(), "이미 완료된 자기 검사다.")
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_of, y = _load_folds_and_labels()
    matrix = _load_matrix(payload, fold_of)
    arm314 = matrix.iloc[:, :COMPARISON_MEMBER_COUNT]
    started = time.monotonic()
    prediction, fitted = _fit_and_seal(arm314, fold_of, y, 0)
    digest = prediction_array_sha256(prediction)
    expected = payload["comparison_arm"]["fold0_prediction_sha256"]
    outer = (fold_of == 0).to_numpy()
    auc = float(roc_auc_score(y[outer].to_numpy(), prediction))
    record = {
        "schema": SCHEMA,
        "precommit_sha256": payload["precommit_sha256"],
        "sealed_fold": 0,
        "member_count": COMPARISON_MEMBER_COUNT,
        "auc": auc,
        "expected_auc": payload["comparison_arm"]["fold_aucs"]["0"]
        if isinstance(payload["comparison_arm"]["fold_aucs"], dict)
        else payload["comparison_arm"]["fold_aucs"][0],
        "prediction_sha256": digest,
        "expected_prediction_sha256": expected,
        "matches": digest == expected,
        "selected_c": fitted.c,
        "selected_lambda": fitted.shrinkage_lambda,
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": now_iso(),
    }
    write_json(out_dir / "selfcheck.json", record)
    _require(record["matches"], f"자기 검사 실패: 분할 0 예측 해시 {digest} != {expected}")
    print(f"자기 검사 통과: 분할 0 AUC {auc:.15f}, 해시 일치", flush=True)


def fold_job(args: argparse.Namespace) -> None:
    payload = load_precommit()
    fold = int(args.fold)
    _require(fold in ALL_FOLDS, f"알 수 없는 분할 {fold}")
    out_dir = RUN_DIR / "ext327" / f"fold-{fold}"
    _require(not (out_dir / "nested.json").exists(), f"이미 완료된 분할이다: {fold}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_of, y = _load_folds_and_labels()
    matrix = _load_matrix(payload, fold_of)
    started = time.monotonic()
    prediction, fitted = _fit_and_seal(matrix, fold_of, y, fold)
    outer = (fold_of == fold).to_numpy()
    ids = fold_of.index[outer]
    pd.DataFrame({ID: ids, "prediction": prediction}).to_parquet(out_dir / "predictions.parquet")
    auc = float(roc_auc_score(y[outer].to_numpy(), prediction))
    record = {
        "schema": SCHEMA,
        "precommit_sha256": payload["precommit_sha256"],
        "sealed_fold": fold,
        "member_count": EXT327_MEMBER_COUNT,
        "rows": int(outer.sum()),
        "auc": auc,
        "selected_c": fitted.c,
        "selected_lambda": fitted.shrinkage_lambda,
        "final_iterations": fitted.final_iterations,
        "final_coefficient_l2_norm": fitted.final_coefficient_l2_norm,
        "prediction_sha256": prediction_array_sha256(prediction),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": base._peak_rss_bytes(),
        "finished_at": now_iso(),
    }
    write_json(out_dir / "nested.json", record)
    print(
        f"분할 {fold}: AUC {auc:.15f}, C={fitted.c}, lambda={fitted.shrinkage_lambda}, "
        f"{record['elapsed_seconds']:.0f}초",
        flush=True,
    )


JOBS = [("selfcheck", None), *(("fold", fold) for fold in ALL_FOLDS)]


def _job_done(kind: str, fold: int | None) -> bool:
    if kind == "selfcheck":
        return (RUN_DIR / "selfcheck" / "selfcheck.json").is_file()
    return (RUN_DIR / "ext327" / f"fold-{fold}" / "nested.json").is_file()


def _running_jobs() -> set[str]:
    listing = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True, check=False).stdout
    tags = set()
    if re.search(r"judge_issue526_ext327\.py selfcheck-job", listing):
        tags.add("selfcheck")
    for match in re.finditer(r"judge_issue526_ext327\.py fold --fold (\d+)", listing):
        tags.add(f"fold-{match.group(1)}")
    return tags


def run_jobs(args: argparse.Namespace) -> None:
    load_precommit()
    _require(1 <= args.workers <= MAX_WORKERS, f"동시 실행은 최대 {MAX_WORKERS}개다.")
    log_dir = RUN_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = str(args.threads)
    pending = [(kind, fold) for kind, fold in JOBS if not _job_done(kind, fold)]
    active: dict[str, tuple[subprocess.Popen, object]] = {}
    failed: dict[str, int] = {}
    print(f"남은 작업 {len(pending)}/{len(JOBS)}, 동시 상한 {args.workers}, 스레드 {args.threads}", flush=True)
    while pending or active:
        for tag, (process, handle) in list(active.items()):
            code = process.poll()
            if code is not None:
                handle.close()
                del active[tag]
                if code != 0:
                    failed[tag] = code
                print(f"{tag} {'완료' if code == 0 else f'실패({code})'} {now_iso()}", flush=True)
        running = _running_jobs() | set(active)
        while pending and len(running) < args.workers:
            kind, fold = pending.pop(0)
            tag = "selfcheck" if kind == "selfcheck" else f"fold-{fold}"
            if tag in running or _job_done(kind, fold):
                continue
            handle = (log_dir / f"{tag}.log").open("w")
            command = [sys.executable, __file__]
            command += ["selfcheck-job"] if kind == "selfcheck" else ["fold", "--fold", str(fold)]
            active[tag] = (
                subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT),
                handle,
            )
            running.add(tag)
            print(f"{tag} 시작 {now_iso()}", flush=True)
        if pending or active:
            time.sleep(10)
    _require(not failed, f"실패한 작업이 있다: {failed}. 로그는 {log_dir}에 있다.")
    print("자기 검사와 다섯 분할 실행 완료", flush=True)


def compare(args: argparse.Namespace) -> None:
    payload = load_precommit()
    selfcheck = read_json(RUN_DIR / "selfcheck" / "selfcheck.json")
    _require(selfcheck["precommit_sha256"] == payload["precommit_sha256"], "자기 검사가 다른 precommit에서 나왔다.")
    _require(selfcheck["matches"], "자기 검사가 실패 상태다. 판정 불가.")
    fold_of, y = _load_folds_and_labels()
    nested = pd.Series(np.nan, index=fold_of.index, dtype=np.float64)
    records: dict[str, dict] = {}
    for fold in ALL_FOLDS:
        out_dir = RUN_DIR / "ext327" / f"fold-{fold}"
        record = read_json(out_dir / "nested.json")
        _require(record["precommit_sha256"] == payload["precommit_sha256"], f"분할 {fold}이 다른 precommit에서 나왔다.")
        part = pd.read_parquet(out_dir / "predictions.parquet").set_index(ID)["prediction"]
        ids = fold_of.index[(fold_of == fold).to_numpy()]
        _require(part.index.equals(pd.Index(ids)), f"분할 {fold} 예측 id가 고정 분할과 다르다.")
        _require(
            prediction_array_sha256(part.to_numpy()) == record["prediction_sha256"],
            f"분할 {fold} 예측 해시가 기록과 다르다.",
        )
        nested.loc[ids] = part.to_numpy()
        records[str(fold)] = record
    _require(bool(nested.notna().all()), "이어붙인 327 예측에 빈 행이 있다.")
    nested_auc = float(roc_auc_score(y.to_numpy(), nested.to_numpy()))
    reference = payload["comparison_arm"]
    ref_fold_aucs = reference["fold_aucs"]
    fold_deltas = {
        str(fold): records[str(fold)]["auc"]
        - (ref_fold_aucs[str(fold)] if isinstance(ref_fold_aucs, dict) else ref_fold_aucs[fold])
        for fold in ALL_FOLDS
    }
    delta = nested_auc - reference["nested_auc"]
    folds_positive = sum(1 for value in fold_deltas.values() if value > 0)
    passes = delta >= GATE_DELTA and folds_positive == FOLDS_REQUIRED_POSITIVE
    result = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "precommit_sha256": payload["precommit_sha256"],
        "selfcheck": {
            "matches": selfcheck["matches"],
            "auc": selfcheck["auc"],
            "prediction_sha256": selfcheck["prediction_sha256"],
        },
        "comparison_arm": {
            "member_count": reference["member_count"],
            "nested_auc": reference["nested_auc"],
            "fold_aucs": ref_fold_aucs,
        },
        "ext327": {
            "member_count": EXT327_MEMBER_COUNT,
            "nested_auc": nested_auc,
            "fold_aucs": {str(fold): records[str(fold)]["auc"] for fold in ALL_FOLDS},
            "fold_selected_c": {str(fold): records[str(fold)]["selected_c"] for fold in ALL_FOLDS},
            "fold_selected_lambda": {str(fold): records[str(fold)]["selected_lambda"] for fold in ALL_FOLDS},
            "prediction_sha256": prediction_array_sha256(nested.to_numpy()),
        },
        "delta_vs_comparison_arm": delta,
        "fold_deltas": fold_deltas,
        "folds_positive": folds_positive,
        "gate": payload["gate"],
        "passes_gate": passes,
        "verdict": "통과" if passes else "미달",
        "finished_at": now_iso(),
    }
    write_json(RUN_DIR / "comparison.json", result)
    print(f"327 nested {nested_auc:.15f}, 314 대비 {delta:+.10f}, 분할 양수 {folds_positive}/5, 판정 {result['verdict']}")


def report(args: argparse.Namespace) -> None:
    payload = load_precommit()
    comparison = read_json(RUN_DIR / "comparison.json")
    lines = [
        "# 이슈 526 확장 회차: 엄격 후보 13개 + 314 확장 팔 = 327 판정",
        "",
        f"- precommit `{payload['precommit_sha256']}`",
        f"- 비교 팔: #514 현재 두 번째 제출 314구성원, nested `{payload['comparison_arm']['nested_auc']}`",
        f"- 자기 검사: 314 봉인 분할 0 재현 해시 일치 `{comparison['selfcheck']['matches']}`",
        f"- 327 nested `{comparison['ext327']['nested_auc']}`",
        f"- 314 대비 차이 `{comparison['delta_vs_comparison_arm']:+.10f}`",
        f"- 분할별 차이 {comparison['fold_deltas']}",
        f"- 분할 양수 {comparison['folds_positive']}/5, 문턱 +0.00002 & 5/5 → 판정 **{comparison['verdict']}**",
        "",
        "Public 점수는 판정에 쓰지 않았다.",
    ]
    (RUN_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report 저장: {RUN_DIR / 'report.md'}")


def publish(args: argparse.Namespace) -> None:
    _assert_repo_root()
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    names = ["precommit.json", "comparison.json", "report.md", "selfcheck/selfcheck.json"]
    names += [f"ext327/fold-{fold}/nested.json" for fold in ALL_FOLDS]
    for name in names:
        source = RUN_DIR / name
        target = PUBLISH_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    print(f"publish 완료: {PUBLISH_DIR}")


def assemble(args: argparse.Namespace) -> None:
    payload = load_precommit()
    comparison = read_json(RUN_DIR / "comparison.json")
    _require(comparison["precommit_sha256"] == payload["precommit_sha256"], "comparison이 다른 precommit에서 나왔다.")
    full_refit_dir = Path(args.full_refit_dir).expanduser().resolve()
    record514 = read_json(ISSUE514_RECORD_PATH)
    _require(
        file_sha256(full_refit_dir / "manifest.json") == record514["full_refit"]["manifest_sha256"],
        "전체 재학습 manifest가 이슈 514 제출 기록과 다르다.",
    )
    fold_of, y = _load_folds_and_labels()
    matrix = _load_matrix(payload, fold_of)
    test = pd.read_csv(Path("data/test.csv"))
    test_index = pd.Index(test[ID], name=ID)

    store = MlflowRunStore(tracking_uri="sqlite:///mlflow.db")
    plan = refit.load_executable_plan(PLAN_PATH, store=store)
    _require(len(plan.members) == OWN_MEMBER_COUNT, "재학습 계획이 36개가 아니다.")
    own_columns = [row["column"] for row in payload["candidate_arm"]["members"][:OWN_MEMBER_COUNT]]
    _require([m.config for m in plan.members] == own_columns, "재학습 계획 순서가 327 행렬의 자체 열과 다르다.")
    own_full = {}
    for member in plan.members:
        values = refit._load_member_full_prediction(plan, member, full_refit_dir, test[ID])
        own_full[member.config] = pd.Series(np.asarray(values, dtype=np.float64), index=test_index)
    own_full_frame = pd.DataFrame(own_full, index=test_index, dtype=np.float64)

    issue489_precommit = read_json(ISSUE489_PRECOMMIT_PATH)
    frozen_by_column = {row["column"]: row for row in issue489_precommit["members"]["rows"]}
    ext_test = {}
    ladder = strict.ladder
    for row in payload["candidate_arm"]["members"][OWN_MEMBER_COUNT : OWN_MEMBER_COUNT + EXTERNAL_MEMBER_COUNT]:
        column = row["column"]
        frozen = frozen_by_column[column]
        values = np.asarray(ladder.load_ledger_array(frozen["test_path"]), dtype=np.float64)
        _require(
            prediction_array_sha256(values) == frozen["test_prediction_sha256"],
            f"{column}: 외부 시험 예측 해시가 이슈 489 기록과 다르다.",
        )
        ext_test[column] = pd.Series(values, index=test_index)
    ext_test_frame = pd.DataFrame(ext_test, index=test_index, dtype=np.float64)

    cand_test = {}
    for row in payload["candidate_arm"]["members"][OWN_MEMBER_COUNT + EXTERNAL_MEMBER_COUNT :]:
        values = freeze.load_array(Path(row["test_path"]), len(test), row["member_id"])
        _require(
            freeze.array_sha256(values) == row["test_sha256"],
            f"{row['column']}: 후보 시험 예측 해시가 동결 명세와 다르다.",
        )
        cand_test[row["column"]] = pd.Series(np.asarray(values, dtype=np.float64), index=test_index)
    cand_test_frame = pd.DataFrame(cand_test, index=test_index, dtype=np.float64)

    test_matrix = pd.concat([own_full_frame, ext_test_frame, cand_test_frame], axis=1)
    _require(list(test_matrix.columns) == list(matrix.columns), "시험 예측 열 순서가 OOF와 다르다.")

    combiner = ensemble.CSelectedShrunkRankLogitCombiner(
        fold_of=fold_of, c_grid=C_GRID, lambda_grid=LAMBDA_GRID, max_iter=META_MAX_ITER
    )
    fitted = combiner.fit(matrix, y)
    prediction = np.asarray(fitted.predict(test_matrix), dtype=np.float64)
    _require(
        prediction.shape == (len(test),) and bool(np.isfinite(prediction).all()),
        "327 제출 예측이 유효하지 않다.",
    )
    out_dir = Path("artifacts/submissions")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "issue526-ext327.csv"
    pd.DataFrame({ID: test[ID], TARGET: prediction}).to_csv(csv_path, index=False)
    manifest = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "precommit_sha256": payload["precommit_sha256"],
        "comparison_sha256": canonical_sha256(comparison),
        "member_count": EXT327_MEMBER_COUNT,
        "strategy": ensemble.CSelectedShrunkRankLogitCombiner.name,
        "selected_c": fitted.c,
        "selected_lambda": fitted.shrinkage_lambda,
        "nested_oof_auc": comparison["ext327"]["nested_auc"],
        "delta_vs_comparison_arm": comparison["delta_vs_comparison_arm"],
        "folds_positive": comparison["folds_positive"],
        "passes_gate": comparison["passes_gate"],
        "full_refit_dir": str(full_refit_dir),
        "full_refit_manifest_sha256": file_sha256(full_refit_dir / "manifest.json"),
        "test_composition_sha256": canonical_sha256(
            [
                (column, prediction_array_sha256(test_matrix[column].to_numpy(np.float64)))
                for column in test_matrix.columns
            ]
        ),
        "in_sample_oof_auc": float(roc_auc_score(y.to_numpy(), fitted.predict(matrix))),
        "submission": {"path": str(csv_path), "sha256": file_sha256(csv_path)},
        "assembled_at": now_iso(),
    }
    write_json(RUN_DIR / "assembly-manifest.json", manifest)
    print(
        f"조립 완료: {csv_path}\n  sha256 {manifest['submission']['sha256']}\n"
        f"  선택 C={fitted.c} lambda={fitted.shrinkage_lambda}, in-sample OOF {manifest['in_sample_oof_auc']:.7f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("precommit").set_defaults(handler=precommit)
    run = sub.add_parser("run")
    run.add_argument("--workers", type=int, default=MAX_WORKERS)
    run.add_argument("--threads", type=int, default=4)
    run.set_defaults(handler=run_jobs)
    fold = sub.add_parser("fold")
    fold.add_argument("--fold", type=int, required=True)
    fold.set_defaults(handler=fold_job)
    sub.add_parser("selfcheck-job").set_defaults(handler=selfcheck_job)
    sub.add_parser("compare").set_defaults(handler=compare)
    sub.add_parser("report").set_defaults(handler=report)
    sub.add_parser("publish").set_defaults(handler=publish)
    asm = sub.add_parser("assemble")
    asm.add_argument("--full-refit-dir", type=Path, required=True)
    asm.set_defaults(handler=assemble)
    args = parser.parse_args()
    try:
        args.handler(args)
    except JudgmentError as exc:
        sys.exit(f"판정 불가: {exc}")


if __name__ == "__main__":
    main()
