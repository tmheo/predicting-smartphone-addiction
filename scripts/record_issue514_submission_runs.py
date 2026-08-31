"""이슈 514의 두 조립본을 제출 가능한 MLflow 실행으로 기록한다."""

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
ASSEMBLY_SCHEMA = "final-full-refit-candidates/1"
CANDIDATES = {
    "pool36_full": {
        "run_name": "ensemble_issue514_pool36_full_refit",
        "oof_evidence": Path(
            "docs/research/missingness-propagation-batch/issue512/direct-nested-gate.json"
        ),
    },
    "extended314_own_full": {
        "run_name": "ensemble_issue514_extended314_own_full_refit",
        "oof_evidence": Path(
            "docs/research/extended-stack-pool-reassembly/issue513/comparison.json"
        ),
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def current_git_state() -> dict[str, str]:
    state = git_state()
    require(state["git_dirty"] == "False", "깨끗한 git 작업 폴더에서만 기록할 수 있다.")
    return state


def existing_run(client, experiment_id: str, candidate: str, submission_sha256: str):
    matches = [
        run
        for run in client.search_runs(
            [experiment_id],
            filter_string=f"tags.`source.issue` = '514' AND tags.`candidate.key` = '{candidate}'",
            max_results=100,
        )
        if run.info.status != "FAILED"
    ]
    if not matches:
        return None
    require(len(matches) == 1, f"{candidate}: 실패하지 않은 MLflow 실행이 중복됐다.")
    require(
        matches[0].data.tags.get("sha256.submission") == submission_sha256,
        f"{candidate}: 기존 MLflow 실행의 제출 해시가 다르다.",
    )
    return matches[0]


def record_candidate(
    *,
    client,
    experiment_id: str,
    assembly_dir: Path,
    manifest_path: Path,
    manifest: dict,
    candidate_key: str,
    commit: str,
) -> tuple[str, bool]:
    specification = CANDIDATES[candidate_key]
    candidate = manifest["candidates"][candidate_key]
    submission_path = assembly_dir / candidate["submission"]["name"]
    submission_sha256 = file_sha256(submission_path)
    require(
        submission_sha256 == candidate["submission"]["sha256"],
        f"{candidate_key}: 제출 CSV 해시가 manifest와 다르다.",
    )
    prior = existing_run(client, experiment_id, candidate_key, submission_sha256)
    if prior is not None:
        return prior.info.run_id, False

    run = client.create_run(experiment_id, run_name=specification["run_name"])
    run_id = run.info.run_id
    try:
        params = {
            "experiment": specification["run_name"],
            "stage": "final_submission_candidate",
            "model.kind": "ensemble",
            "ensemble.strategy": candidate["strategy"],
            "ensemble.member_count": candidate["member_count"],
            "full_refit.scope": "entire_training_dataset",
            "candidate.key": candidate_key,
        }
        for optional in ("own_member_count", "external_member_count", "selected_c", "selected_lambda"):
            if optional in candidate:
                params[f"ensemble.{optional}"] = candidate[optional]
        for key, value in params.items():
            client.log_param(run_id, key, value)

        tags = {
            "git_commit": commit,
            "git_dirty": "False",
            "source.kind": "final_ensemble_assembly",
            "source.issue": "514",
            "candidate.key": candidate_key,
            "sha256.submission": submission_sha256,
            "sha256.assembly_manifest": file_sha256(manifest_path),
            "full_refit.scope": "entire_training_dataset",
        }
        for key, value in tags.items():
            client.set_tag(run_id, key, value)

        client.log_metric(run_id, "auc_oof", float(candidate["nested_oof_auc"]))
        client.log_metric(
            run_id,
            "auc_oof_insample",
            float(candidate["in_sample_oof_auc"]),
        )
        for fold, auc in candidate["nested_fold_aucs"].items():
            client.log_metric(run_id, f"auc_fold_{fold}", float(auc))

        with tempfile.TemporaryDirectory() as staging:
            root = Path(staging)
            shutil.copyfile(submission_path, root / "submission.csv")
            shutil.copyfile(manifest_path, root / "assembly-manifest.json")
            shutil.copyfile(
                specification["oof_evidence"],
                root / f"{candidate_key}-oof-evidence.json",
            )
            for artifact in sorted(root.iterdir()):
                client.log_artifact(run_id, str(artifact))
        client.set_terminated(run_id, status="FINISHED")
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    return run_id, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", type=Path, required=True)
    args = parser.parse_args()

    require(Path.cwd().resolve() == REPO_ROOT, "저장소 루트에서 실행해야 한다.")
    assembly_dir = args.assembly_dir.expanduser().resolve()
    manifest_path = assembly_dir / "manifest.json"
    require(manifest_path.is_file(), f"조립 manifest가 없다: {manifest_path}")
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == ASSEMBLY_SCHEMA, "조립 manifest 스키마가 다르다.")
    require(manifest.get("issue") == 514, "이슈 514 조립 manifest가 아니다.")

    state = current_git_state()
    require(manifest["git_commit"] == state["git_commit"], "조립 커밋이 현재 HEAD와 다르다.")
    client, experiment_id = mlflow_client(TRACKING_URI)
    results = {}
    for candidate_key in CANDIDATES:
        run_id, created = record_candidate(
            client=client,
            experiment_id=experiment_id,
            assembly_dir=assembly_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            candidate_key=candidate_key,
            commit=state["git_commit"],
        )
        results[candidate_key] = {"run_id": run_id, "created": created}
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
