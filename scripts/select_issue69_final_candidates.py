"""이슈 #69: #225 선정 규칙을 조립 산출물에 기계 적용해 최종 제출 후보 두 개를 고른다.

사용법: uv run python scripts/select_issue69_final_candidates.py (pipeline.refit --assemble 뒤에 실행)

입력: artifacts/full-refit/{submission_cv.csv, submission_cv_full.csv, member_test_cv_full.parquet, manifest.json}
      run-logs/issue337/ensemble-evaluation.json (동결 풀 35개의 전략별 nested OOF, #337)
출력: artifacts/submissions/issue69-candidate-1.csv, issue69-candidate-2.csv, artifacts/judgments/issue69-final-candidates.yaml
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

from pipeline.data import ID, TARGET
from pipeline.ensemble import rank_mean
from pipeline.pool_audit import prediction_array_sha256
from pipeline import tracking

REFIT = Path("artifacts/full-refit")
EVAL = Path("run-logs/issue337/ensemble-evaluation.json")
TOLERANCE = -0.0005


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((REFIT / "manifest.json").read_text())
    assert manifest["combiner"] == "shrunk_rank_logit_logistic"
    cv = pd.read_csv(REFIT / "submission_cv.csv")
    mixed = pd.read_csv(REFIT / "submission_cv_full.csv")
    members = pd.read_parquet(REFIT / "member_test_cv_full.parquet").set_index(ID)
    assert (cv[ID].to_numpy() == mixed[ID].to_numpy()).all() and members.index.equals(pd.Index(cv[ID], name=ID))
    evaluation = json.loads(EVAL.read_text())
    nested = {s["name"]: s["nested_oof_auc"] for s in evaluation["strategies"]}
    assert evaluation["member_count"] == members.shape[1] == 35

    first = mixed[TARGET].to_numpy()
    axes = {
        "shrunk_rank_logit_logistic_cv_only": (cv[TARGET].to_numpy(), nested["shrunk_rank_logit_logistic"]),
        "rank_mean_cv_full_mix": (rank_mean(members), nested["rank_mean"]),
    }
    first_nested = nested["shrunk_rank_logit_logistic"]
    rows = []
    for name, (pred, axis_nested) in axes.items():
        rho = float(pd.Series(first).corr(pd.Series(pred), method="spearman"))
        delta = axis_nested - first_nested
        rows.append({"axis": name, "nested_oof_auc": axis_nested, "delta_vs_first": delta,
                     "eligible": bool(delta >= TOLERANCE), "spearman_vs_first": rho,
                     "prediction_sha256": prediction_array_sha256(pred)})
    eligible = [r for r in rows if r["eligible"]]
    second = min(eligible, key=lambda r: r["spearman_vs_first"])
    second_pred = axes[second["axis"]][0]

    out = Path("artifacts/submissions"); out.mkdir(exist_ok=True)
    c1 = out / "issue69-candidate-1.csv"; c2 = out / "issue69-candidate-2.csv"
    shutil.copyfile(REFIT / "submission_cv_full.csv", c1)
    if second["axis"] == "shrunk_rank_logit_logistic_cv_only":
        shutil.copyfile(REFIT / "submission_cv.csv", c2)
    else:
        pd.DataFrame({ID: cv[ID], TARGET: second_pred}).to_csv(c2, index=False)
    state = tracking.git_state()
    record = {
        "judgment_id": "issue69-final-candidates",
        "decision_issue": 69, "rule_issue": 225,
        "git_commit": state["git_commit"], "git_dirty": state["git_dirty"],
        "plan_sha256": manifest["plan_sha256"], "source_pool_sha256": manifest["source_pool_sha256"],
        "combiner": manifest["combiner"],
        "mix_weights": {"cv": manifest["cv_model_weight"], "full": manifest["full_model_weight"]},
        "nested_source": {"path": str(EVAL), "sha256": sha(EVAL), "member_count": evaluation["member_count"]},
        "first_candidate": {"axis": "shrunk_rank_logit_logistic_cv_full_mix", "nested_oof_auc": first_nested,
                             "file": str(c1), "file_sha256": sha(c1),
                             "prediction_sha256": prediction_array_sha256(first)},
        "tolerance_delta": TOLERANCE, "axes": rows,
        "second_candidate": {**second, "file": str(c2), "file_sha256": sha(c2)},
        "mix_exception": {"triggered": False,
                          "evidence": "#226 짝 계측: CV 전용 0.97082 vs 5:1 혼합 0.97087 (+0.00005, 악화 아님)"},
        "member_spearman_cv_vs_full": manifest["member_spearman_cv_vs_full"],
        "assembly_prediction_sha256": manifest["prediction_sha256"],
    }
    Path("artifacts/judgments/issue69-final-candidates.yaml").write_text(
        yaml.safe_dump(record, allow_unicode=True, sort_keys=False))
    print(yaml.safe_dump({k: record[k] for k in ("first_candidate", "axes", "second_candidate")}, allow_unicode=True, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
