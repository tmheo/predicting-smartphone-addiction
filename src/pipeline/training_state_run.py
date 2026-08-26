"""사전 고정한 여러 학습 시점을 한 궤적에서 독립 후보 실행으로 게시한다.

사용법:
    uv run python -m pipeline.training_state_run configs/a.yaml configs/b.yaml --stage confirm
    uv run python -m pipeline.training_state_run configs/a.yaml configs/b.yaml --stage confirm --plan
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv, data, summary, tracking
from .config import STAGES, ExperimentConfig, load_config
from .judgment import missingness_reweighting, weighted_oof_auc
from .plan import FeaturePlan
from .training_state_contract import (
    CANDIDATE_RUNS_NAME,
    MANIFEST_NAME,
    PARENT_MANIFEST_NAME,
    CandidateIdentity,
    TrainingStateRunContract,
    build_run_contract,
    content_sha256,
    frame_content_sha256,
)
from .training_state_cv import run_training_state_seed
from .training_state_manifest import (
    CANDIDATE_MANIFEST_SCHEMA_VERSION,
    validate_candidate_manifest,
)
from .training_state_recovery import (
    TrainingStateCandidate,
    TrainingStateRecovery,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="한 학습 궤적의 여러 고정 시점을 독립 후보 실행으로 게시"
    )
    parser.add_argument("configs", nargs="+", help="후보 시점별 고유 설정 YAML 경로")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--plan", action="store_true", help="학습과 MLflow 기록 없이 계약만 확인")
    parser.add_argument(
        "--recovery-dir",
        type=Path,
        help="전체 후보 집합을 원자적으로 저장할 복구 디렉터리",
    )
    args = parser.parse_args()

    configs = [load_config(path, args.stage) for path in args.configs]
    first = configs[0]
    plan = FeaturePlan.from_config(first.features)
    input_hashes = _input_hashes(first)
    git_state = tracking.git_state()
    contract = build_run_contract(
        configs,
        git_commit=git_state["git_commit"],
        input_sha256=input_hashes,
    )
    _validate_supported_feature_path(plan)

    recovery_root = args.recovery_dir or Path("run-recovery") / (
        f"training-state-{contract.trajectory}-{contract.stage}"
    )
    if args.plan:
        _print_plan(contract, recovery_root, git_state)
        return
    if git_state["git_dirty"] != "False":
        raise SystemExit(
            "여러 학습 시점 확정 실행은 깨끗한 커밋에서만 허용한다. 변경을 커밋한 뒤 다시 실행할 것."
        )

    from .observe import RunObserver

    observer = RunObserver.begin(first)
    child_run_ids: list[str] = []
    client, _ = tracking.mlflow_client()
    try:
        _record_parent_start(client, observer.run_id, contract, configs)
        observer.stage("setup")
        observer.record_input_hashes(input_hashes)
        recovery = TrainingStateRecovery.for_run(
            recovery_root,
            [
                TrainingStateCandidate(
                    config_name=candidate.config.name,
                    config_path=candidate.config_path,
                    config_sha256=candidate.config_sha256,
                    completed_epochs=candidate.completed_epochs,
                    schedule_horizon_epochs=contract.state.schedule_horizon_epochs,
                )
                for candidate in contract.candidates
            ],
            input_hashes,
            git_commit=git_state["git_commit"],
            trajectory_identity_sha256=contract.trajectory_identity_sha256,
            declared_candidate_set_sha256=contract.candidate_set_sha256,
            stage=contract.stage,
            seeds=list(contract.seeds),
            model_kind=contract.model_kind,
            trajectory_end_epochs=contract.state.trajectory_end_epochs,
        )
        if recovery.candidate_set_sha256 != contract.candidate_set_sha256:
            raise ValueError("실행 계약과 복구 경계의 후보 집합 SHA-256이 다르다.")
        observer.record_execution_identity(recovery.execution_identity)

        observer.stage("data_load")
        train = data.load_csv(first.data.train)
        test = data.load_csv(first.data.test)
        data.align_categories(train, test, first.features.categorical)
        train, test = plan.apply_dataset_wide(train, test)
        train = data.attach_folds(train, first.data.folds)
        n_folds = int(train["fold"].max()) + 1
        observer.data_loaded(seed_total=len(first.seeds), fold_total=n_folds)

        seed_results = [
            run_training_state_seed(
                first,
                plan,
                train,
                test,
                seed,
                recorder=observer,
                recovery=recovery,
            )
            for seed in first.seeds
        ]

        observer.stage("evaluation")
        aggregated = {
            candidate.completed_epochs: _aggregate_candidate(
                candidate.config,
                candidate.completed_epochs,
                seed_results,
                train,
            )
            for candidate in contract.candidates
        }

        observer.stage("artifacts")
        child_run_ids = _publish_candidates(
            client,
            observer.run_id,
            contract,
            aggregated,
            observer.stage_durations(),
        )
        _record_candidate_runs(client, observer.run_id, contract, child_run_ids)
        observer.succeed()
        parent = client.get_run(observer.run_id)
        if parent.info.status != "FINISHED":
            raise RuntimeError(
                "학습 궤적 부모 실행이 FINISHED로 확정되지 않아 child 게시를 열 수 없다: "
                f"{parent.info.status}"
            )
        for run_id in child_run_ids:
            client.set_tag(run_id, "training_state.ready", "true")
        for candidate, run_id in zip(contract.candidates, child_run_ids):
            result, _ = aggregated[candidate.completed_epochs]
            print(
                f"candidate={candidate.config.name} completed_epochs={candidate.completed_epochs} "
                f"run_id={run_id} auc_oof={result.fold_aucs['auc_oof']:.5f}"
            )
    except BaseException as exc:  # noqa: BLE001 - 모든 생성 실행의 상태를 확정한다.
        for run_id in child_run_ids:
            try:
                run = client.get_run(run_id)
                if run.info.status == "RUNNING":
                    client.set_terminated(run_id, status="FAILED")
            except Exception:
                pass
        observer.fail(exc)
        sys.exit(130 if isinstance(exc, KeyboardInterrupt) else 1)


def _validate_supported_feature_path(plan: FeaturePlan) -> None:
    providers = plan.fold_fit_providers()
    if providers:
        raise ValueError(
            "여러 학습 시점 경로는 fold-fit 특성 제공자를 지원하지 않는다: "
            + ", ".join(kind for kind, _ in providers)
        )


def _input_hashes(cfg: ExperimentConfig) -> dict[str, str]:
    return {
        "train": data.file_sha256(cfg.data.train),
        "test": data.file_sha256(cfg.data.test),
        "folds": data.file_sha256(cfg.data.folds),
    }


def _print_plan(
    contract: TrainingStateRunContract,
    recovery_root: Path,
    git_state: dict[str, str],
) -> None:
    print(f"trajectory  : {contract.trajectory}")
    print(f"model       : {contract.model_kind}")
    print(f"stage       : {contract.stage}")
    print(f"seeds       : {list(contract.seeds)}")
    print(f"candidates  : {list(contract.state.candidates)}")
    print(f"trajectory end: {contract.state.trajectory_end_epochs}")
    print(f"schedule horizon: {contract.state.schedule_horizon_epochs}")
    print(f"state kind  : {contract.state.state_kind}")
    print(f"selection   : {contract.state.selection_rule}")
    print(f"trajectory sha256: {contract.trajectory_identity_sha256}")
    print(f"candidate set sha256: {contract.candidate_set_sha256}")
    for candidate in contract.candidates:
        print(
            f"  {candidate.completed_epochs}: {candidate.config.name} "
            f"({candidate.config.source_path}, {candidate.config_sha256[:16]})"
        )
    for name, digest in contract.input_sha256.items():
        print(f"sha256.{name:<6}: {digest}")
    print(f"git         : {git_state}")
    print(f"recovery    : {recovery_root}")
    print("publication : ineligible parent + one FINISHED ready child per candidate")


def _aggregate_candidate(
    cfg: ExperimentConfig,
    completed_epochs: int,
    seed_results: list[dict[int, cv.CVResult]],
    train: pd.DataFrame,
) -> tuple[cv.CVResult, dict[int, pd.DataFrame]]:
    results = [result[completed_epochs] for result in seed_results]
    seed_aucs = {
        seed: result.fold_aucs["auc_oof"]
        for seed, result in zip(cfg.seeds, results)
    }
    seed_oofs = {
        seed: result.oof.copy() for seed, result in zip(cfg.seeds, results)
    }
    final = results[0]
    if final.feature_names != FeaturePlan.from_config(cfg.features).all_columns():
        raise AssertionError("학습 시점 후보의 실제 특성 목록이 설정 선언과 다르다.")
    if len(results) > 1:
        final.oof["pred"] = np.mean(
            [result.oof["pred"] for result in results], axis=0
        )
        final.test_pred["pred"] = np.mean(
            [result.test_pred["pred"] for result in results], axis=0
        )
        final.fold_aucs = cv.score_predictions(
            train[data.TARGET], train["fold"], final.oof["pred"].to_numpy()
        )
        final.importance = pd.concat(
            [result.importance for result in results], ignore_index=True
        )
        final.model_training_diagnostics = [
            item
            for result in results
            for item in result.model_training_diagnostics
        ]
    final.recovery_evidence = [
        item for result in results for item in result.recovery_evidence
    ]
    final.fold_feature_reuse_evidence = [
        item
        for result in results
        for item in result.fold_feature_reuse_evidence
    ]
    for seed, auc in seed_aucs.items():
        final.fold_aucs[f"auc_oof_seed_{seed}"] = auc
    final.fold_aucs.update(_weighted_oof_metrics(cfg, final.oof, train[data.TARGET]))
    return final, seed_oofs


def _weighted_oof_metrics(
    cfg: ExperimentConfig, oof: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    index = pd.Index(oof[data.ID], name=data.ID)
    prediction = pd.Series(oof["pred"].to_numpy(), index=index)
    y = pd.Series(target.to_numpy(), index=index)
    reweighting = missingness_reweighting(cfg.data.train, cfg.data.test)
    return weighted_oof_auc(prediction, y, reweighting).metrics()


def _record_parent_start(
    client,
    parent_run_id: str,
    contract: TrainingStateRunContract,
    configs: list[ExperimentConfig],
) -> None:
    tags = {
        "run.kind": "training_state_trajectory",
        "judgment.eligible": "false",
        "training_state.trajectory": contract.trajectory,
        "training_state.trajectory_identity_sha256": (
            contract.trajectory_identity_sha256
        ),
        "training_state.candidate_set_sha256": contract.candidate_set_sha256,
        "training_state.ready": "false",
    }
    for key, value in tags.items():
        client.set_tag(parent_run_id, key, value)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = contract.parent_manifest()
        path = root / PARENT_MANIFEST_NAME
        path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        client.log_artifact(parent_run_id, str(path))
        client.set_tag(
            parent_run_id,
            "sha256.training_state_trajectory",
            data.file_sha256(path),
        )
        for cfg in configs:
            client.log_artifact(
                parent_run_id,
                str(cfg.source_path),
                artifact_path="training_state/configs",
            )


def _publish_candidates(
    client,
    parent_run_id: str,
    contract: TrainingStateRunContract,
    aggregated: dict[int, tuple[cv.CVResult, dict[int, pd.DataFrame]]],
    stage_durations: list[tuple[str, int, float]],
) -> list[str]:
    _, experiment_id = tracking.mlflow_client()
    created: list[str] = []
    try:
        for candidate in contract.candidates:
            cfg = candidate.config
            result, seed_oofs = aggregated[candidate.completed_epochs]
            run = client.create_run(
                experiment_id,
                run_name=cfg.name,
                tags={"mlflow.parentRunId": parent_run_id},
            )
            run_id = run.info.run_id
            created.append(run_id)
            tracking.log_start_records(
                client,
                run_id,
                cfg,
                fixed_git_state={
                    "git_commit": contract.git_commit,
                    "git_dirty": "False",
                },
                fixed_config_artifact=(
                    Path(candidate.config_path).name,
                    candidate.config_bytes,
                ),
            )
            tracking.log_input_hashes(client, run_id, contract.input_sha256)
            for key, value in _candidate_tags(
                parent_run_id, contract, candidate
            ).items():
                client.set_tag(run_id, key, value)
            tracking.log_final_records(client, run_id, cfg, result, seed_oofs)
            summary.generate_and_log(
                client,
                run_id,
                cfg,
                result,
                stage_durations,
            )
            _record_candidate_manifest(
                client,
                run_id,
                parent_run_id,
                contract,
                candidate,
                result,
                seed_oofs,
            )
            client.set_terminated(run_id, status="FINISHED")
    except BaseException:
        for run_id in created:
            try:
                if client.get_run(run_id).info.status == "RUNNING":
                    client.set_terminated(run_id, status="FAILED")
            except Exception:
                pass
        raise
    return created


def _candidate_tags(
    parent_run_id: str,
    contract: TrainingStateRunContract,
    candidate: CandidateIdentity,
) -> dict[str, str]:
    return {
        "run.kind": "training_state_snapshot",
        "training_state.ready": "false",
        "training_state.trajectory": contract.trajectory,
        "training_state.trajectory_run_id": parent_run_id,
        "training_state.trajectory_identity_sha256": (
            contract.trajectory_identity_sha256
        ),
        "training_state.candidate_set_sha256": contract.candidate_set_sha256,
        "training_state.snapshot_identity_sha256": (
            candidate.snapshot_identity_sha256
        ),
        "training_state.completed_epochs": str(candidate.completed_epochs),
        "training_state.schedule_horizon_epochs": str(
            contract.state.schedule_horizon_epochs
        ),
        "training_state.state_kind": contract.state.state_kind,
        "training_state.selection_rule": contract.state.selection_rule,
        "observability.source_run_id": parent_run_id,
    }


def _published_artifact_payloads(client, run_id: str) -> dict[str, bytes]:
    """manifest 기록 직전 child 루트에 실제 게시된 모든 파일을 고정한다."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(client.download_artifacts(run_id, "", tmp))
        payloads = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
    if MANIFEST_NAME in payloads:
        raise ValueError(f"새 child run에 기존 {MANIFEST_NAME}이 이미 있다.")
    return payloads


def _record_candidate_manifest(
    client,
    run_id: str,
    parent_run_id: str,
    contract: TrainingStateRunContract,
    candidate: CandidateIdentity,
    result: cv.CVResult,
    seed_oofs: dict[int, pd.DataFrame],
) -> None:
    artifact_payloads = _published_artifact_payloads(client, run_id)
    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "run_kind": "training_state_snapshot",
        "trajectory_run_id": parent_run_id,
        "trajectory": contract.trajectory,
        "trajectory_identity_sha256": contract.trajectory_identity_sha256,
        "candidate_set_sha256": contract.candidate_set_sha256,
        "precommitted_candidates": list(contract.state.candidates),
        "selection_rule": contract.state.selection_rule,
        "validation_target_used_for_selection": False,
        "state_kind": contract.state.state_kind,
        "completed_epochs": candidate.completed_epochs,
        "schedule_horizon_epochs": contract.state.schedule_horizon_epochs,
        "trajectory_end_epochs": contract.state.trajectory_end_epochs,
        "candidate": candidate.to_json(),
        "git_commit": contract.git_commit,
        "input_sha256": contract.input_sha256,
        "stage": contract.stage,
        "seeds": list(contract.seeds),
        "model_kind": contract.model_kind,
        "prediction_content_sha256": {
            "oof": frame_content_sha256(result.oof),
            "test": frame_content_sha256(result.test_pred),
            **{
                f"oof_seed_{seed}": frame_content_sha256(oof)
                for seed, oof in seed_oofs.items()
            },
        },
        "importance_content_sha256": frame_content_sha256(result.importance),
        "artifact_file_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(artifact_payloads.items())
        },
    }
    manifest["manifest_content_sha256"] = content_sha256(manifest)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / MANIFEST_NAME
        path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        manifest_bytes = path.read_bytes()
        client.log_artifact(run_id, str(path))
        client.set_tag(run_id, "sha256.training_state_manifest", data.file_sha256(path))
        run = client.get_run(run_id)
        prospective_tags = dict(run.data.tags)
        prospective_tags["training_state.ready"] = "true"
        validate_candidate_manifest(
            manifest_bytes=manifest_bytes,
            tags=prospective_tags,
            params=dict(run.data.params),
            artifact_bytes_of=artifact_payloads.__getitem__,
        )


def _record_candidate_runs(
    client,
    parent_run_id: str,
    contract: TrainingStateRunContract,
    run_ids: list[str],
) -> None:
    document = {
        "schema_version": 1,
        "trajectory_identity_sha256": contract.trajectory_identity_sha256,
        "runs": [
            {
                **candidate.to_json(),
                "run_id": run_id,
                "status": "FINISHED",
            }
            for candidate, run_id in zip(contract.candidates, run_ids)
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / CANDIDATE_RUNS_NAME
        path.write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        client.log_artifact(parent_run_id, str(path))
        client.set_tag(
            parent_run_id,
            "sha256.training_state_candidate_runs",
            data.file_sha256(path),
        )


if __name__ == "__main__":
    main()
