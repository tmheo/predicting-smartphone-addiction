"""통과한 확장 스택 구성으로 두 번째 최종 제출물을 조립한다. (#444)

#443이 판정한 own35 + ext209(`shrunk_rank_logit_logistic`, nested 0.9703167)를 바탕으로,
#444에서 사용자와 정한 조립 규칙을 기계 적용한다.

- 구성: 자체 35(`artifacts/pool.yaml`) + #442 장부 통과 209 가운데 전체 자료 TE 누출이
  코드 수준에서 확인된 2개(szymon74 pub_rmlp, pub_tabm)를 뺀 242구성원.
  OOF 행렬은 #443 절제 구성 `ablate_te_leak`과 열 순서까지 같다(nested 0.9702876).
- 자체 35의 시험 예측은 #69 첫 후보와 같은 5:1 혼합판(`artifacts/full-refit/`)이고,
  외부 207의 시험 예측은 장부 `test_path`의 CV 분할 평균이다.
- 결합기는 기존 두 장(b24e5ba7, e88f706e)과 같은 계약으로 전체 OOF에 한 번 적합해
  시험 행렬에 적용한다. λ는 5분할 leave-one-fold-out으로 고른다.
- 산출물은 제출 CSV(artifacts/submissions/, 커밋 제외)와 manifest JSON(docs/research/, 커밋)뿐이다.
  MLflow 실행은 만들지 않으며, 기록은 #445가
  업로드 뒤 `pipeline.submit --record-existing`으로 남긴다.

이 조립은 외부 예측을 `artifacts/pool.yaml`에 넣지 않고 champion 판정에도 쓰지 않는다.

사용법:
    uv run python scripts/judge_extended_stack.py --prepare   # 캐시가 없을 때만
    uv run python scripts/assemble_extended_stack.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagnose_external94_width as ext94
import judge_extended_stack as judge

from pipeline import ensemble
from pipeline.data import ID, TARGET, TRAIN_PATH, file_sha256, labels
from pipeline.judgment import FOLDS_PATH
from pipeline.ledger import Pool
from pipeline.pool_audit import prediction_array_sha256

ISSUE = 444
STRATEGY = "shrunk_rank_logit_logistic"
JUDGED_CONFIG = "own35_ext209"
ASSEMBLED_CONFIG = "ablate_te_leak"
TEST_PATH = Path("data/test.csv")
FULL_REFIT_DIR = Path("artifacts/full-refit")
OWN_TEST_PATH = FULL_REFIT_DIR / "member_test_cv_full.parquet"
FULL_REFIT_MANIFEST_PATH = FULL_REFIT_DIR / "manifest.json"
LADDER_EVIDENCE_PATH = judge.EVIDENCE_PATH
OUT_DIR = Path("artifacts/submissions")
SUBMISSION_PATH = OUT_DIR / "issue444-extended-stack.csv"
MANIFEST_PATH = Path("docs/research/extended-stack-submission-manifest.json")
REFERENCE_SUBMISSIONS = {
    "e88f706e_candidate_1_cv_full_mix": OUT_DIR / "issue69-candidate-1.csv",
    "b24e5ba7_candidate_2_cv_only": OUT_DIR / "issue69-candidate-2.csv",
}
N_TEST = 296302


def _git_head() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _part(config: str) -> dict:
    return json.loads(judge.part_path(config, STRATEGY).read_text())


def build_test_matrix(
    oof_columns: list[str], test_ids: pd.Series
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """OOF와 같은 열 순서의 시험 예측 행렬. 자체 35는 5:1 혼합판, 외부는 장부 test_path."""
    own = pd.read_parquet(OWN_TEST_PATH)
    if not own[ID].to_numpy().tolist() == test_ids.to_numpy().tolist():
        raise ValueError("5:1 혼합판 시험 예측의 id 순서가 test.csv와 다르다.")
    own = own.set_index(ID)
    _, accepted = judge.load_ledger()
    by_key = {f"ext_{row['member_id']}": row for row in accepted}
    columns: dict[str, np.ndarray] = {}
    sources: dict[str, dict] = {}
    for name in oof_columns:
        if name.startswith("ext_"):
            row = by_key[name]
            values = judge.load_ledger_array(row["test_path"])
            sources[name] = {
                "kind": "external_cv_fold_average",
                "test_path": row["test_path"],
            }
        else:
            values = own[name].to_numpy(np.float64)
            sources[name] = {"kind": "own_cv5_full1_mix", "test_path": str(OWN_TEST_PATH)}
        if values.shape != (N_TEST,) or not np.isfinite(values).all():
            raise ValueError(f"{name}: 시험 예측 행 수가 {values.shape}이거나 유한하지 않다.")
        columns[name] = values
        sources[name]["prediction_sha256"] = prediction_array_sha256(values)
    matrix = pd.DataFrame(columns, index=test_ids.to_numpy()).astype(np.float64)
    if list(matrix.columns) != oof_columns:
        raise ValueError("시험 행렬의 열 순서가 OOF와 다르다.")
    return matrix, sources


def rank_space_checks(
    prediction: np.ndarray, test_ids: pd.Series
) -> dict[str, object]:
    """제출 눈금 확인: 값 범위, 동률, 기존 두 장과의 순위 상관."""
    checks: dict[str, object] = {
        "rows": int(len(prediction)),
        "finite": bool(np.isfinite(prediction).all()),
        "min": float(prediction.min()),
        "max": float(prediction.max()),
        "within_unit_interval": bool(prediction.min() >= 0.0 and prediction.max() <= 1.0),
        "distinct_values": int(np.unique(prediction).size),
        "spearman_vs": {},
    }
    for label, path in REFERENCE_SUBMISSIONS.items():
        if not path.is_file():
            checks["spearman_vs"][label] = None
            continue
        reference = pd.read_csv(path)
        if reference[ID].to_numpy().tolist() != test_ids.to_numpy().tolist():
            raise ValueError(f"{path}: id 순서가 test.csv와 다르다.")
        rho = spearmanr(prediction, reference[TARGET].to_numpy(np.float64)).correlation
        checks["spearman_vs"][label] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "spearman": float(rho),
        }
    return checks


def assemble() -> None:
    started = time.monotonic()
    git = _git_head()
    train = pd.read_csv(TRAIN_PATH)
    fold_of = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"]
    y = labels(fold_of.index)
    ext94.verify_row_order(train, fold_of)
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID])[ID]
    if len(test_ids) != N_TEST or test_ids.duplicated().any():
        raise ValueError("test.csv의 행 수나 id가 기대와 다르다.")

    judged = _part(JUDGED_CONFIG)
    assembled = _part(ASSEMBLED_CONFIG)
    oof = judge.build_matrix(ASSEMBLED_CONFIG, fold_of)
    if list(oof.columns) != assembled["members"]:
        raise ValueError("OOF 행렬의 열 순서가 #443 절제 구성 산출물과 다르다.")
    excluded = sorted(set(judged["members"]) - set(assembled["members"]))
    print(
        f"OOF {oof.shape}, 판정 구성 {judged['member_count']} → 조립 구성 "
        f"{assembled['member_count']} (제외 {excluded})",
        flush=True,
    )

    test, test_sources = build_test_matrix(list(oof.columns), test_ids)
    print(f"시험 행렬 {test.shape}", flush=True)

    combiner = ensemble.COMBINER_REGISTRY[STRATEGY]
    fit_started = time.monotonic()
    fitted = combiner.fit(oof.astype(np.float64), y)
    fit_seconds = time.monotonic() - fit_started
    prediction = np.asarray(fitted.predict(test), dtype=np.float64)
    if prediction.shape != (N_TEST,) or not np.isfinite(prediction).all():
        raise ValueError("결합 시험 예측의 길이가 다르거나 유한하지 않다.")
    in_sample = np.asarray(fitted.predict(oof), dtype=np.float64)
    in_sample_auc = float(roc_auc_score(y.to_numpy(), in_sample))
    print(
        f"적합 {fit_seconds:.0f}s, λ={fitted.shrinkage_lambda}, "
        f"in-sample OOF AUC {in_sample_auc:.7f} (참고치)",
        flush=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({ID: test_ids.to_numpy(), TARGET: prediction})
    frame.to_csv(SUBMISSION_PATH, index=False)
    checks = rank_space_checks(prediction, test_ids)

    _, accepted = judge.load_ledger()
    ledger_rows = {f"ext_{row['member_id']}": row for row in accepted}
    pool = Pool.load()
    pool_run_ids = {member.config: member.run_id for member in pool.members}
    full_refit_manifest = json.loads(FULL_REFIT_MANIFEST_PATH.read_text())
    weights = fitted.summary()
    members = []
    for name in oof.columns:
        entry: dict[str, object] = {
            "column": name,
            "weight": float(weights[name]),
            "test": test_sources[name],
        }
        if name.startswith("ext_"):
            row = ledger_rows[name]
            entry.update(
                origin="external",
                member_id=row["member_id"],
                source=row["source"],
                dataset=row["dataset"],
                license=row["license"],
                oof_path=row["oof_path"],
                ledger_sha256=row["sha256"],
                fold_evidence=row["fold_evidence"],
                caveats=row["caveats"],
            )
        else:
            entry.update(origin="own", run_id=pool_run_ids[name])
        members.append(entry)
    licenses: dict[str, int] = {}
    datasets: dict[str, dict] = {}
    for entry in members:
        if entry["origin"] != "external":
            continue
        licenses[entry["license"]] = licenses.get(entry["license"], 0) + 1
        bucket = datasets.setdefault(
            entry["dataset"], {"license": entry["license"], "member_count": 0}
        )
        bucket["member_count"] += 1

    manifest = {
        "schema_version": 1,
        "issue": ISSUE,
        "git": git,
        "strategy": STRATEGY,
        "judged": {
            "issue": 443,
            "config": JUDGED_CONFIG,
            "member_count": judged["member_count"],
            "nested_auc": judged["nested_auc"],
            "evidence_path": str(LADDER_EVIDENCE_PATH),
            "evidence_sha256": file_sha256(LADDER_EVIDENCE_PATH),
        },
        "assembled": {
            "config": ASSEMBLED_CONFIG,
            "member_count": assembled["member_count"],
            "nested_auc": assembled["nested_auc"],
            "fold_aucs": assembled["fold_aucs"],
            "excluded_members": [
                {
                    "column": name,
                    "reason": "원 노트북이 전체 자료 TE를 쓴 판을 그대로 실행(#174, "
                    "docs/research/code-notebook-insights.md 22·24번). 분할 간 목표 "
                    "누출이 코드 수준에서 확인돼 조립에서 뺐다.",
                    "caveats": ledger_rows[name]["caveats"],
                }
                for name in excluded
            ],
        },
        "combiner": {
            "shrinkage_lambda": float(fitted.shrinkage_lambda),
            "lambda_grid": list(combiner.lambda_grid),
            "fit_protocol": "전체 OOF 1회 적합(ensemble.full_fit_predictions와 같은 계약), "
            "λ는 5분할 leave-one-fold-out",
            "fit_seconds": fit_seconds,
            "in_sample_oof_auc": in_sample_auc,
        },
        "inputs": {
            "train_sha256": file_sha256(TRAIN_PATH),
            "test_sha256": file_sha256(TEST_PATH),
            "folds_sha256": file_sha256(FOLDS_PATH),
            "pool_sha256": file_sha256(Path("artifacts/pool.yaml")),
            "ledger_path": str(judge.LEDGER_PATH),
            "ledger_sha256": file_sha256(judge.LEDGER_PATH),
            "own_test": {
                "kind": "cv5_full1_mix",
                "path": str(OWN_TEST_PATH),
                "full_refit_manifest_sha256": file_sha256(FULL_REFIT_MANIFEST_PATH),
                "plan_sha256": full_refit_manifest["plan_sha256"],
                "source_pool_sha256": full_refit_manifest["source_pool_sha256"],
                "cv_model_weight": full_refit_manifest["cv_model_weight"],
                "full_model_weight": full_refit_manifest["full_model_weight"],
            },
        },
        "external_summary": {
            "member_count": sum(1 for e in members if e["origin"] == "external"),
            "licenses": licenses,
            "datasets": datasets,
        },
        "submission": {
            "path": str(SUBMISSION_PATH),
            "file_sha256": file_sha256(SUBMISSION_PATH),
            "prediction_sha256": prediction_array_sha256(prediction),
            "checks": checks,
        },
        "members": members,
        "elapsed_seconds": time.monotonic() - started,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"제출 파일 {SUBMISSION_PATH} sha256 {manifest['submission']['file_sha256']}")
    print(f"manifest {MANIFEST_PATH}")
    print(json.dumps(checks, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="확장 스택 두 번째 최종 제출물 조립 (#444)")
    parser.parse_args()
    if not (judge.CACHE_DIR / "ext209.parquet").is_file():
        sys.exit("외부 행렬 캐시가 없다. 먼저 judge_extended_stack.py --prepare를 실행할 것.")
    for config in (JUDGED_CONFIG, ASSEMBLED_CONFIG):
        if not judge.part_path(config, STRATEGY).is_file():
            sys.exit(f"#443 산출물이 없다: {judge.part_path(config, STRATEGY)}")
    assemble()


if __name__ == "__main__":
    main()
