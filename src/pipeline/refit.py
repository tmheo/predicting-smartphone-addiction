"""후보 풀 전체 자료 재학습과 최종 결합 예측 조립. (#66, #375)

재학습 계획 장부는 후보 실행, 시드, CV에서 확정한 학습 길이를 고정한다.
구성원 실행은 시드별 시험 예측을 먼저 저장한 뒤 평균본과 계보 manifest를 만든다.
조립은 기존 5-fold 시험 예측과 전체 자료 예측을 모델 개수 기준 5:1로 섞고,
계획이 지정한 등록 결합 방식을 전체 OOF에서 한 번 맞춘다.

**이 module은 장부의 저장 숫자를 읽지 않는다.** 장부의 문법과 관문은 `refit_plan`이
소유하고, 여기서 쓰는 재학습 예산은 언제나 그 관문이 원시 근거에서 다시 계산한 값이다.
그 값을 노출하는 자료형은 `ExecutableRefitPlan` 하나뿐이므로, 이 module의 실행 함수는
검증을 통과한 계획 말고는 애초에 받을 수 없다. 손으로 예산을 적어 넣는 통로는 없다.

관문은 계획 전체에 걸린다. `--member`로 구성원 하나만 요청해도 후보 풀 정체성과
모든 반복형 구성원의 근거·예산을 먼저 검증하며, 그 검증은 자료 적재보다,
피처 계획 생성보다, 모델 연결부 생성보다, 출력 폴더 생성보다 먼저 끝난다.

끝난 산출물의 정체성은 구성원 단위다(#69). 시드별 기록과 manifest는 규약 블록과 그
구성원 장부 항목의 내용 해시를 남기고, 재개와 조립은 그 값을 맞춰 본다. 장부 전체 해시는
맥락으로만 남기므로 다른 구성원이 장부에 더해져도 이미 끝난 재학습은 유효하다.

사용법:
    uv run python -m pipeline.refit artifacts/full-refit-plan.yaml --member exp006_te_drop_gaming
    uv run python -m pipeline.refit artifacts/full-refit-plan.yaml --all
    uv run python -m pipeline.refit artifacts/full-refit-plan.yaml --assemble
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
import pandas as pd

from . import data, initial_score, model, tracking
from .config import load_config
from .data import ID, TARGET
from .ensemble import (
    COMBINER_REGISTRY,
    full_fit_predictions,
    member_matrix,
    member_test_matrix,
)
from .ledger import Pool
from .plan import FeaturePlan
from .pool_audit import prediction_array_sha256
from .refit_plan import (
    ExecutableRefitMember,
    ExecutableRefitPlan,
    RefitPlan,
    RefitPlanError,
    TrajectoryStateBudgetDerivation,
    member_entry_sha256_from_document,
)
from .runs import MlflowRunStore, RunStore, RunStoreError, sha256_of

DEFAULT_OUTPUT = Path("artifacts/full-refit")

# 시드별 기록과 구성원 manifest의 문법 판본.
# 판본 3은 구성원 항목 해시(`member_entry_sha256`)를 남기고, 재개와 조립은 그 값을 맞춰 본다(#69).
# 판본 2 기록은 장부 전체 해시(`plan_sha256`)만 남겼으므로, 기록의 git 커밋에 있던 장부에서
# 같은 항목 해시를 다시 계산해 받아들인다.
RECORD_SCHEMA_VERSION = 3
LEGACY_RECORD_SCHEMA_VERSION = 2


def load_executable_plan(
    path: Path,
    *,
    store: RunStore | None = None,
    pool: Pool | None = None,
    pool_sha256: str | None = None,
) -> ExecutableRefitPlan:
    """장부를 읽어 관문에 통과시킨다. 전체 자료 재학습이 계획을 얻는 유일한 통로다.

    기본값은 운영 경로(MLflow 실행 저장소와 커밋된 후보 풀 장부)이고,
    시험은 메모리 저장소와 시험용 풀을 넣어 같은 관문을 통과시킨다.
    """
    return RefitPlan.load(path).validate_for_refit(
        store=store, pool=pool, pool_sha256=pool_sha256
    )


def run_member(
    plan: ExecutableRefitPlan,
    member: ExecutableRefitMember,
    output: Path,
    *,
    seeds: tuple[int, ...] | None = None,
    finalize: bool = True,
) -> Path:
    """구성원 하나를 전체 자료로 시드별 재학습하고 평균 시험 예측을 저장한다.

    `member`가 `ExecutableRefitMember`라는 사실이 곧 계획 전체가 관문을 통과했다는 뜻이고,
    `member.budgets`는 원시 근거에서 다시 계산한 재학습 예산이다.
    """
    cfg = load_config(member.config_path, "confirm")
    if cfg.training_rows is not None and cfg.training_rows.replica_count:
        raise ValueError(
            f"{member.config}: 바깥쪽 분할 좌표가 없는 전체 자료 재학습에는 "
            "결측 증강 마스크 계약을 적용할 수 없다."
        )
    if cfg.name != member.config:
        raise ValueError(f"계획 구성원과 설정 name이 다르다: {member.config} != {cfg.name}")
    if list(member.budgets) != cfg.seeds and list(member.budgets) != [cfg.seeds[0]]:
        raise ValueError(f"{member.config}: 설정 단계와 계획 시드가 다르다.")
    selected_seeds = tuple(member.budgets) if seeds is None else seeds
    unknown_seeds = sorted(set(selected_seeds) - set(member.budgets))
    if unknown_seeds:
        raise ValueError(f"{member.config}: 계획에 없는 시드다: {unknown_seeds}")
    if not selected_seeds:
        raise ValueError(f"{member.config}: 실행할 시드가 없다.")
    if finalize and selected_seeds != tuple(member.budgets):
        raise ValueError("일부 시드 실행에서는 최종 평균을 만들 수 없다.")

    state = tracking.git_state()
    git_dirty = state["git_dirty"] == "True"
    if git_dirty:
        raise ValueError("전체 자료 재학습은 깨끗한 git 작업 폴더에서만 실행한다.")

    member_dir = output / member.config
    member_dir.mkdir(parents=True, exist_ok=True)
    train = data.load_csv(cfg.data.train)
    test = data.load_csv(cfg.data.test)
    data.align_categories(train, test, cfg.features.categorical)
    feature_plan = FeaturePlan.from_config(cfg.features)
    train, test = feature_plan.apply_dataset_wide(train, test)
    y = train[TARGET]
    input_hashes = _input_hashes(cfg)
    config_sha256 = data.file_sha256(member.config_path)
    provenance = {**_ledger_context(plan), **_member_provenance(member)}

    seed_predictions: list[np.ndarray] = []
    seed_records = []
    for seed in selected_seeds:
        budget = member.budgets[seed]
        prediction_path = member_dir / f"test_pred_seed_{seed}.parquet"
        record_path = member_dir / f"test_pred_seed_{seed}.json"
        seed_identity = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "config": member.config,
            "config_sha256": config_sha256,
            **provenance,
            "git_commit": state["git_commit"],
            "input_sha256": input_hashes,
            "seed": seed,
            "training_budget": budget,
            **_training_state_seed_identity(member),
        }
        if prediction_path.exists():
            if not record_path.exists():
                raise ValueError(f"{prediction_path}: 재개 검증 기록이 없다.")
            prediction = _load_prediction(prediction_path, test[ID])
            record = json.loads(record_path.read_text())
            expected_record = {
                **seed_identity,
                "prediction_sha256": prediction_array_sha256(prediction),
            }
            if not _matches_specification(record, expected_record, plan, member):
                raise ValueError(f"{prediction_path}: 현재 재학습 명세와 계보가 다르다.")
        else:
            if record_path.exists():
                raise ValueError(f"{record_path}: 대응하는 예측 파일이 없다.")
            X_train, X_test = feature_plan.build_full_matrices(train, test, seed)
            provider = initial_score.create(cfg.initial_score)
            # 바깥쪽 분할 계약은 전체 학습 행의 내부 OOF와 전체 학습 자료 적합의 시험
            # 초기 점수를 같은 계약으로 만든다. (#505)
            scores = initial_score.full_data_scores(provider, train, test, seed)
            adapter = model.create(cfg.model, seed)
            model.set_dataset_reference(adapter, X_train, X_test)
            if isinstance(member.derivation, TrajectoryStateBudgetDerivation):
                if cfg.training_state is None:
                    raise ValueError(
                        f"{member.config}: 정확 시점 재학습 계획인데 설정에 training_state가 없다."
                    )
                actual_training_state = model.fit_full_training_state(
                    adapter,
                    X_train,
                    y,
                    cfg.training_state,
                    scores.train if scores is not None else None,
                )
                actual_training_state = {
                    key: actual_training_state[key]
                    for key in (
                        "completed_epochs",
                        "schedule_horizon_epochs",
                        "state_kind",
                    )
                }
                if actual_training_state != seed_identity["training_state_actual"]:
                    raise ValueError(
                        f"{member.config}: 모델이 보고한 정확 시점 재학습 계약이 계획과 다르다."
                    )
            else:
                model.fit_full(
                    adapter,
                    X_train,
                    y,
                    budget,
                    scores.train if scores is not None else None,
                )
                actual_training_state = None
            prediction = np.asarray(
                adapter.predict(X_test, scores.test if scores is not None else None),
                dtype=np.float64,
            )
            _validate_prediction(prediction, len(test), f"{member.config} seed {seed}")
            _atomic_parquet(
                pd.DataFrame({ID: test[ID].to_numpy(), "pred": prediction}),
                prediction_path,
            )
            _atomic_json(
                {
                    **seed_identity,
                    "prediction_sha256": prediction_array_sha256(prediction),
                },
                record_path,
            )
        seed_predictions.append(prediction)
        seed_records.append(
            {
                "seed": seed,
                "training_budget": budget,
                **_training_state_seed_identity(member),
                "prediction_sha256": prediction_array_sha256(prediction),
            }
        )

    if not finalize:
        return member_dir / f"test_pred_seed_{selected_seeds[-1]}.parquet"

    averaged = np.mean(seed_predictions, axis=0, dtype=np.float64)
    averaged_path = member_dir / "test_pred_full.parquet"
    _atomic_parquet(
        pd.DataFrame({ID: test[ID].to_numpy(), "pred": averaged}), averaged_path
    )
    manifest = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "config": member.config,
        "config_path": str(member.config_path),
        "config_sha256": config_sha256,
        **provenance,
        "git_commit": state["git_commit"],
        "git_dirty": git_dirty,
        "input_sha256": input_hashes,
        "feature_columns": feature_plan.all_columns(),
        "seeds": seed_records,
        "prediction_sha256": prediction_array_sha256(averaged),
    }
    _atomic_json(manifest, member_dir / "manifest.json")
    return averaged_path


def _ledger_context(plan: ExecutableRefitPlan) -> dict:
    """산출물이 어느 장부에서 나왔는지 적는 맥락. 기록만 하고 맞춰 보지는 않는다.

    장부 전체 해시는 다른 구성원이 더해지기만 해도 바뀌므로, 이 값을 관문에 쓰면
    이미 끝난 재학습이 그 구성원과 무관한 변경으로 무효가 된다(#69에서 걷어낸 규칙).
    """
    return {
        "source_pool_sha256": plan.source_pool_sha256,
        "plan_sha256": plan.content_sha256,
    }


def _member_provenance(member: ExecutableRefitMember) -> dict:
    """시드별 기록과 manifest가 함께 남기고, 재개와 조립이 다시 맞춰 보는 계보·규약 묶음.

    원시 근거가 어느 실행의 어느 산출물에서 왔는지, 그 원시 값을 어떤 규약으로 예산에
    옮겼는지, 그리고 그 계산이 나온 장부 항목(규약 블록 + 이 구성원 항목)의 내용 해시를
    한자리에 적는다. 이 구성원의 어느 칸이 바뀌어도 걸리지만 다른 구성원의 변경에는 걸리지 않는다.
    """
    lineage = member.lineage
    derivation = member.derivation
    if isinstance(derivation, TrajectoryStateBudgetDerivation):
        budget_derivation = {
            "status": member.status,
            "method": member.refit_method,
            "completed_epochs": derivation.completed_epochs,
            "schedule_horizon_epochs": derivation.schedule_horizon_epochs,
            "state_kind": derivation.state_kind,
            "trajectory_identity_sha256": derivation.trajectory_identity_sha256,
        }
    else:
        budget_derivation = {
            "status": member.status,
            "statistic": None if derivation is None else derivation.policy.statistic,
            "multiplier": None if derivation is None else derivation.policy.multiplier,
            "rounding": None if derivation is None else derivation.policy.rounding,
        }
    return {
        "member_entry_sha256": member.entry_sha256,
        "evidence_lineage": {
            "source_run_id": lineage.source_run_id,
            "source_git_commit": lineage.source_git_commit,
            "source_config_path": lineage.source_config_path,
            "source_config_sha256": lineage.source_config_sha256,
            "evidence_artifact_path": lineage.evidence_artifact_path,
            "evidence_artifact_sha256": lineage.evidence_artifact_sha256,
        },
        "refit_budget_derivation": budget_derivation,
    }


def _training_state_seed_identity(member: ExecutableRefitMember) -> dict[str, object]:
    derivation = member.derivation
    if not isinstance(derivation, TrajectoryStateBudgetDerivation):
        return {}
    contract = {
        "completed_epochs": derivation.completed_epochs,
        "schedule_horizon_epochs": derivation.schedule_horizon_epochs,
        "state_kind": derivation.state_kind,
        "trajectory_identity_sha256": derivation.trajectory_identity_sha256,
    }
    return {
        "refit_method": member.refit_method,
        "training_state_contract": contract,
        "training_state_actual": {
            key: contract[key]
            for key in (
                "completed_epochs",
                "schedule_horizon_epochs",
                "state_kind",
            )
        },
    }


def assemble(plan: ExecutableRefitPlan, output: Path) -> dict[str, Path]:
    """CV 전용, 전체 자료 전용, 모델 수 5:1 혼합의 최종 결합 예측을 만든다."""
    protocol = plan.protocol
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    members = [(member.config, member.run_id) for member in plan.members]
    store = MlflowRunStore()
    train_index = pd.Index(train[ID], name=ID)
    test_index = pd.Index(test[ID], name=ID)
    oof = member_matrix(members, store, train_index)
    cv_test = member_test_matrix(members, store, test_index)
    full_columns = {
        member.config: pd.Series(
            _load_member_full_prediction(plan, member, output, test[ID]),
            index=test_index,
        )
        for member in plan.members
    }
    full_test = pd.DataFrame(full_columns, index=test_index, dtype=np.float64)
    mixed_test = mix_member_predictions(
        cv_test,
        full_test,
        cv_weight=protocol.cv_model_weight,
        full_weight=protocol.full_model_weight,
    )

    combiner = COMBINER_REGISTRY[protocol.combiner]
    y = train.set_index(ID).loc[train_index, TARGET]
    predictions = {
        "cv": full_fit_predictions(combiner, oof, y, cv_test),
        "full": full_fit_predictions(combiner, oof, y, full_test),
        "cv_full": full_fit_predictions(combiner, oof, y, mixed_test),
    }
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, prediction in predictions.items():
        path = output / f"submission_{name}.csv"
        _atomic_csv(pd.DataFrame({ID: test[ID], TARGET: prediction}), path)
        paths[name] = path
    _atomic_parquet(full_test.reset_index(), output / "member_test_full.parquet")
    _atomic_parquet(mixed_test.reset_index(), output / "member_test_cv_full.parquet")
    correlations = {
        member.config: float(
            pd.Series(cv_test[member.config]).corr(
                pd.Series(full_test[member.config]), method="spearman"
            )
        )
        for member in plan.members
    }
    manifest = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "source_pool_sha256": plan.source_pool_sha256,
        "plan_sha256": plan.content_sha256,
        "combiner": protocol.combiner,
        "cv_model_weight": protocol.cv_model_weight,
        "full_model_weight": protocol.full_model_weight,
        "member_spearman_cv_vs_full": correlations,
        "prediction_sha256": {
            name: prediction_array_sha256(prediction)
            for name, prediction in predictions.items()
        },
    }
    _atomic_json(manifest, output / "manifest.json")
    return paths


def mix_member_predictions(
    cv_test: pd.DataFrame,
    full_test: pd.DataFrame,
    *,
    cv_weight: int,
    full_weight: int,
) -> pd.DataFrame:
    """같은 구성원 순서의 CV와 전체 자료 예측을 학습 모델 수로 가중 평균한다."""
    if not cv_test.index.equals(full_test.index) or list(cv_test) != list(full_test):
        raise ValueError("CV와 전체 자료 예측 행렬의 id 또는 구성원 순서가 다르다.")
    if cv_weight < 1 or full_weight < 1:
        raise ValueError("예측 혼합 가중치는 양의 정수여야 한다.")
    mixed = (cv_weight * cv_test + full_weight * full_test) / (
        cv_weight + full_weight
    )
    values = mixed.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("혼합 예측에 유한하지 않은 값이 있다.")
    return mixed.astype(np.float64)


def _input_hashes(cfg) -> dict[str, str]:
    paths = {
        "train": cfg.data.train,
        "test": cfg.data.test,
        "folds": cfg.data.folds,
        **initial_score.input_paths(cfg.initial_score),
    }
    return {name: data.file_sha256(path) for name, path in paths.items()}


def _resolve_member_entry_sha256(
    record: dict, plan: ExecutableRefitPlan, member: ExecutableRefitMember
) -> str | None:
    """기록이 가리키는 구성원 항목 해시. 없으면 `None`.

    문법 판본 3 기록은 값을 직접 갖는다. 문법 판본 2 기록은 장부 전체 해시와 git 커밋만
    남겼으므로, 그 커밋의 장부를 git 이력에서 읽어 전체 해시가 기록과 같은지 확인한 뒤
    그 장부의 이 구성원 항목에서 같은 계산을 한다. 그 장부를 찾지 못하거나 해시가 다르면
    기록이 가리키는 항목을 알 수 없으므로 `None`이다.
    """
    if isinstance(record.get("member_entry_sha256"), str):
        return record["member_entry_sha256"]
    if record.get("schema_version") != LEGACY_RECORD_SCHEMA_VERSION:
        return None
    commit, plan_sha256 = record.get("git_commit"), record.get("plan_sha256")
    if not isinstance(commit, str) or not isinstance(plan_sha256, str):
        return None
    payload = _plan_bytes_at_commit(commit, plan.source_path)
    if payload is None or sha256_of(payload) != plan_sha256:
        return None
    try:
        return member_entry_sha256_from_document(
            yaml.safe_load(payload.decode()), member.config
        )
    except (RefitPlanError, UnicodeDecodeError, yaml.YAMLError):
        return None


def _plan_bytes_at_commit(commit: str, source_path: Path) -> bytes | None:
    """git 이력의 `commit`에 있던 장부 파일 바이트. 없으면 `None`."""
    toplevel = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if toplevel.returncode != 0:
        return None
    root = Path(toplevel.stdout.strip()).resolve()
    try:
        relative = source_path.resolve().relative_to(root)
    except ValueError:
        return None
    shown = subprocess.run(
        ["git", "show", f"{commit}:{relative.as_posix()}"],
        capture_output=True,
        check=False,
    )
    return shown.stdout if shown.returncode == 0 else None


def _matches_specification(
    record: dict,
    expected: dict,
    plan: ExecutableRefitPlan,
    member: ExecutableRefitMember,
) -> bool:
    """저장된 기록이 지금의 구성원 명세와 같은 산출물을 가리키는지 본다.

    `expected`는 지금 실행이 남길 기록이다. 장부 맥락(`plan_sha256`, `source_pool_sha256`),
    산출물을 만든 git 커밋과 문법 판본은 정보로만 남기고 맞춰 보지 않는다. 나머지
    (설정 내용, 입력 자료, 계보, 파생 규약, 시드, 예산, 예측 해시)는 전부 같아야 하고,
    구성원 항목 해시는 기록이 가리키는 값을 풀어서 맞춘다.
    """
    informational = {"schema_version", "git_commit", "git_dirty", *_ledger_context(plan)}
    for key, value in expected.items():
        if key in informational or key == "member_entry_sha256":
            continue
        if record.get(key) != value:
            return False
    return _resolve_member_entry_sha256(record, plan, member) == member.entry_sha256


def _load_member_full_prediction(
    plan: ExecutableRefitPlan,
    member: ExecutableRefitMember,
    output: Path,
    expected_ids: pd.Series,
) -> np.ndarray:
    """저장해 둔 구성원 평균 예측을 읽고, 그 manifest가 지금의 구성원 명세와 같은지 본다.

    설정 파일 내용, 입력 자료 해시, 계보, 파생 규약, 구성원 항목 해시, 예산, 예측 해시를
    맞춰 본다. 장부 전체 해시는 보지 않으므로 다른 구성원이 장부에 더해져도 이 manifest는
    유효하다(#69).
    """
    member_dir = output / member.config
    prediction = _load_prediction(member_dir / "test_pred_full.parquet", expected_ids)
    manifest_path = member_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{member.config}: 구성원 manifest가 없다.")
    manifest = json.loads(manifest_path.read_text())
    cfg = load_config(member.config_path, "confirm")
    expected = {
        "config": member.config,
        "config_sha256": data.file_sha256(member.config_path),
        "input_sha256": _input_hashes(cfg),
        **_member_provenance(member),
        "prediction_sha256": prediction_array_sha256(prediction),
    }
    for key, value in expected.items():
        if key == "member_entry_sha256":
            continue
        if manifest.get(key) != value:
            raise ValueError(f"{member.config}: manifest의 {key} 계보가 다르다.")
    if _resolve_member_entry_sha256(manifest, plan, member) != member.entry_sha256:
        raise ValueError(
            f"{member.config}: manifest의 member_entry_sha256 계보가 다르다."
        )
    if isinstance(member.derivation, TrajectoryStateBudgetDerivation):
        _validate_training_state_manifest_seeds(
            member,
            manifest,
            member_dir,
            expected_ids,
            prediction,
        )
        return prediction
    recorded_budgets = {
        record.get("seed"): record.get("training_budget")
        for record in manifest.get("seeds", [])
    }
    if recorded_budgets != member.budgets:
        raise ValueError(f"{member.config}: manifest의 재학습 예산이 검증된 예산과 다르다.")
    return prediction


def _validate_training_state_manifest_seeds(
    member: ExecutableRefitMember,
    manifest: dict,
    member_dir: Path,
    expected_ids: pd.Series,
    averaged_prediction: np.ndarray,
) -> None:
    """정확 시점 구성원의 시드별 실제 계약과 평균 예측을 모두 다시 맞춘다."""
    records = manifest.get("seeds")
    if not isinstance(records, list) or len(records) != len(member.budgets):
        raise ValueError(f"{member.config}: manifest의 정확 시점 시드 기록 수가 다르다.")
    expected_identity = _training_state_seed_identity(member)
    expected_fields = {
        "seed",
        "training_budget",
        "refit_method",
        "training_state_contract",
        "training_state_actual",
        "prediction_sha256",
    }
    seed_predictions = []
    for (seed, budget), record in zip(member.budgets.items(), records, strict=True):
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError(
                f"{member.config}: manifest의 시드 {seed} 정확 시점 필드가 다르다."
            )
        expected_record = {
            "seed": seed,
            "training_budget": budget,
            **expected_identity,
        }
        if any(record.get(key) != value for key, value in expected_record.items()):
            raise ValueError(
                f"{member.config}: manifest의 시드 {seed} 정확 시점 계약이 다르다."
            )
        seed_prediction = _load_prediction(
            member_dir / f"test_pred_seed_{seed}.parquet", expected_ids
        )
        if record["prediction_sha256"] != prediction_array_sha256(seed_prediction):
            raise ValueError(
                f"{member.config}: manifest의 시드 {seed} 예측 해시가 다르다."
            )
        seed_predictions.append(seed_prediction)
    recomputed = np.mean(seed_predictions, axis=0, dtype=np.float64)
    if not np.array_equal(recomputed, averaged_prediction):
        raise ValueError(
            f"{member.config}: 정확 시점 시드 예측 평균이 구성원 전체 예측과 다르다."
        )


def _validate_prediction(prediction: np.ndarray, size: int, name: str) -> None:
    if prediction.shape != (size,) or not np.isfinite(prediction).all():
        raise ValueError(f"{name}: 예측 길이가 다르거나 유한하지 않다.")


def _load_prediction(path: Path, expected_ids: pd.Series) -> np.ndarray:
    frame = pd.read_parquet(path)
    if list(frame.columns) != [ID, "pred"]:
        raise ValueError(f"{path}: 예측 열이 다르다.")
    if frame["pred"].dtype != np.dtype("float64"):
        raise ValueError(f"{path}: 예측 정밀도가 float64가 아니다.")
    if not frame[ID].reset_index(drop=True).equals(expected_ids.reset_index(drop=True)):
        raise ValueError(f"{path}: 시험 id 순서가 다르다.")
    prediction = frame["pred"].to_numpy(dtype=np.float64)
    _validate_prediction(prediction, len(expected_ids), str(path))
    return prediction


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="후보 풀 전체 자료 재학습")
    parser.add_argument("plan", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--member")
    action.add_argument("--all", action="store_true")
    action.add_argument("--assemble", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.seed is not None and args.member is None:
        parser.error("--seed는 --member와 함께 사용해야 한다.")

    # 관문이 먼저다. 여기를 통과하지 못하면 자료도 읽지 않고 폴더도 만들지 않는다.
    try:
        plan = load_executable_plan(args.plan)
        members = (
            ()
            if args.assemble
            else (plan.members if args.all else (plan.member(args.member),))
        )
    except (RefitPlanError, RunStoreError) as error:
        sys.exit(str(error))

    if args.assemble:
        for name, path in assemble(plan, args.out_dir).items():
            print(f"{name}: {path}")
        return
    for member in members:
        print(f"[{member.config}] 전체 자료 재학습 시작", flush=True)
        path = run_member(
            plan,
            member,
            args.out_dir,
            seeds=None if args.seed is None else (args.seed,),
            finalize=args.seed is None,
        )
        print(f"[{member.config}] 완료: {path}", flush=True)


if __name__ == "__main__":
    main()
