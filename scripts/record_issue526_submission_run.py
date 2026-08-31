"""이슈 526 ext327 조립본을 제출 가능한 MLflow 실행으로 기록한다."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from pipeline.data import file_sha256
from pipeline.tracking import git_state, mlflow_client

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = "sqlite:///mlflow.db"
RUN_NAME = "ensemble_issue526_ext327_strict_candidates"
RUN_DIR = Path("run-logs/issue526-ext327")
SUBMISSION_PATH = Path("artifacts/submissions/issue526-ext327.csv")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle-ref", type=int, required=True)
    parser.add_argument("--public-auc", type=float, required=True)
    parser.parse_args()
    args = parser.parse_args()

    require(Path.cwd().resolve() == REPO_ROOT, "저장소 루트에서 실행해야 한다.")
    manifest_path = RUN_DIR / "assembly-manifest.json"
    manifest = load_json(manifest_path)
    comparison = load_json(RUN_DIR / "comparison.json")
    require(manifest["precommit_sha256"] == comparison["precommit_sha256"], "조립과 판정의 precommit이 다르다.")
    submission_sha256 = file_sha256(SUBMISSION_PATH)
    require(submission_sha256 == manifest["submission"]["sha256"], "제출 CSV 해시가 manifest와 다르다.")

    state = git_state()
    client, experiment_id = mlflow_client(TRACKING_URI)
    existing = [
        run
        for run in client.search_runs(
            [experiment_id],
            filter_string="tags.`source.issue` = '526' AND tags.`candidate.key` = 'ext327'",
            max_results=100,
        )
        if run.info.status != "FAILED"
    ]
    if existing:
        require(len(existing) == 1, "실패하지 않은 MLflow 실행이 중복됐다.")
        require(
            existing[0].data.tags.get("sha256.submission") == submission_sha256,
            "기존 MLflow 실행의 제출 해시가 다르다.",
        )
        print(json.dumps({"run_id": existing[0].info.run_id, "created": False}))
        return

    run = client.create_run(experiment_id, run_name=RUN_NAME)
    run_id = run.info.run_id
    try:
        params = {
            "experiment": RUN_NAME,
            "stage": "final_submission_candidate",
            "model.kind": "ensemble",
            "ensemble.strategy": manifest["strategy"],
            "ensemble.member_count": manifest["member_count"],
            "ensemble.own_member_count": 36,
            "ensemble.external_member_count": 278,
            "ensemble.strict_candidate_count": 13,
            "ensemble.selected_c": manifest["selected_c"],
            "ensemble.selected_lambda": manifest["selected_lambda"],
            "full_refit.scope": "entire_training_dataset",
            "candidate.key": "ext327",
        }
        for key, value in params.items():
            client.log_param(run_id, key, value)
        tags = {
            "git_commit": state["git_commit"],
            "git_dirty": state["git_dirty"],
            "source.kind": "final_ensemble_assembly",
            "source.issue": "526",
            "candidate.key": "ext327",
            "sha256.submission": submission_sha256,
            "sha256.assembly_manifest": file_sha256(manifest_path),
            "kaggle.submission_ref": str(args.kaggle_ref),
            "gate.passed": str(manifest["passes_gate"]),
            "gate.note": "nested +0.0000047 vs 314, folds 3/5, user override for upload",
        }
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
        client.log_metric(run_id, "auc_oof", float(manifest["nested_oof_auc"]))
        client.log_metric(run_id, "auc_oof_insample", float(manifest["in_sample_oof_auc"]))
        client.log_metric(run_id, "auc_public", float(args.public_auc))
        for fold, auc in comparison["ext327"]["fold_aucs"].items():
            client.log_metric(run_id, f"auc_fold_{fold}", float(auc))
        with tempfile.TemporaryDirectory() as staging:
            root = Path(staging)
            shutil.copyfile(SUBMISSION_PATH, root / "submission.csv")
            shutil.copyfile(manifest_path, root / "assembly-manifest.json")
            shutil.copyfile(RUN_DIR / "comparison.json", root / "ext327-oof-evidence.json")
            for artifact in sorted(root.iterdir()):
                client.log_artifact(run_id, str(artifact))
        client.set_terminated(run_id, status="FINISHED")
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    print(json.dumps({"run_id": run_id, "created": True}))


if __name__ == "__main__":
    main()
