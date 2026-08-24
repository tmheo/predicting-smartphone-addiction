"""후보 풀 전체 자료 재학습과 최종 결합 예측 조립. (#66)

재학습 계획은 후보 실행, 시드, CV에서 확정한 학습 길이를 고정한다.
구성원 실행은 시드별 시험 예측을 먼저 저장한 뒤 평균본과 계보 manifest를 만든다.
조립은 기존 5-fold 시험 예측과 전체 자료 예측을 모델 개수 기준 5:1로 섞고,
계획이 지정한 등록 결합 방식을 전체 OOF에서 한 번 맞춘다.

이 module의 `RefitPlan`은 현행 `artifacts/full-refit-plan.yaml`(문법 판본 1)을 읽는
실행 경로다. 원시 근거 계보와 예산 재계산을 소유하는 새 장부는 `refit_plan` module에
있고(문법 판본 2, #373), 32개 구성원의 근거 자료를 그 문법으로 옮기는 후속 이슈에서
이 자료형과 검증을 대체한다.

사용법:
    uv run python -m pipeline.refit artifacts/full-refit-plan.yaml --member exp006_te_drop_gaming
    uv run python -m pipeline.refit artifacts/full-refit-plan.yaml --all
    uv run python -m pipeline.refit artifacts/full-refit-plan.yaml --assemble
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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
from .runs import MlflowRunStore

DEFAULT_OUTPUT = Path("artifacts/full-refit")


@dataclass(frozen=True)
class RefitMember:
    config: str
    config_path: Path
    run_id: str
    budgets: dict[int, int | None]
    budget_source: str


@dataclass(frozen=True)
class RefitPlan:
    source_path: Path
    source_pool_sha256: str
    iteration_multiplier: float
    budget_statistic: str
    budget_rounding: str
    cv_model_weight: int
    full_model_weight: int
    combiner: str
    members: tuple[RefitMember, ...]

    @classmethod
    def load(cls, path: Path) -> RefitPlan:
        with path.open() as stream:
            raw = yaml.safe_load(stream)
        protocol = raw["protocol"]
        members = tuple(
            RefitMember(
                config=item["config"],
                config_path=Path(item["config_path"]),
                run_id=item["run_id"],
                budgets={
                    int(seed): (None if budget is None else int(budget))
                    for seed, budget in item["budgets"].items()
                },
                budget_source=item["budget_source"],
            )
            for item in raw["members"]
        )
        plan = cls(
            source_path=path,
            source_pool_sha256=raw["source_pool_sha256"],
            iteration_multiplier=float(protocol["iteration_multiplier"]),
            budget_statistic=protocol["budget_statistic"],
            budget_rounding=protocol["budget_rounding"],
            cv_model_weight=int(protocol["cv_model_weight"]),
            full_model_weight=int(protocol["full_model_weight"]),
            combiner=protocol["combiner"],
            members=members,
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if data.file_sha256(Path("artifacts/pool.yaml")) != self.source_pool_sha256:
            raise ValueError("재학습 계획의 후보 풀 SHA-256이 현재 장부와 다르다.")
        pool = Pool.load()
        expected = [(member.config, member.run_id) for member in pool.members]
        actual = [(member.config, member.run_id) for member in self.members]
        if actual != expected:
            raise ValueError("재학습 계획의 구성원 순서나 실행 ID가 후보 풀과 다르다.")
        if self.iteration_multiplier != 1.25:
            raise ValueError("전체 자료 학습 길이 배수는 1.25여야 한다.")
        if self.budget_statistic != "median" or self.budget_rounding != "half_up":
            raise ValueError("학습 길이 규약은 fold 중앙값과 사사오입이어야 한다.")
        if (self.cv_model_weight, self.full_model_weight) != (5, 1):
            raise ValueError("CV와 전체 자료 예측의 모델 개수 가중치는 5:1이어야 한다.")
        if self.combiner not in COMBINER_REGISTRY:
            raise ValueError(
                f"등록되지 않은 결합 방식이다: {self.combiner} "
                f"(등록: {', '.join(COMBINER_REGISTRY)})"
            )
        for member in self.members:
            if list(member.budgets) not in ([42], [42, 43, 44]):
                raise ValueError(f"{member.config}: 허용되지 않은 시드 목록이다.")
            for budget in member.budgets.values():
                if budget is not None and budget < 1:
                    raise ValueError(f"{member.config}: 학습 길이는 양수여야 한다.")
            allowed_sources = {"fold_median", "configured_cap", "not_applicable"}
            if member.budget_source not in allowed_sources:
                raise ValueError(f"{member.config}: 학습 길이 출처가 올바르지 않다.")
            budgets = list(member.budgets.values())
            if member.budget_source == "not_applicable" and any(
                budget is not None for budget in budgets
            ):
                raise ValueError(f"{member.config}: 반복 수가 없는 모델에 학습 길이가 있다.")
            if member.budget_source != "not_applicable" and any(
                budget is None for budget in budgets
            ):
                raise ValueError(f"{member.config}: 반복형 모델의 학습 길이가 비어 있다.")
            if not member.config_path.is_file():
                raise ValueError(f"{member.config}: 설정 파일이 없다: {member.config_path}")

    def member(self, name: str) -> RefitMember:
        matches = [member for member in self.members if member.config == name]
        if len(matches) != 1:
            raise ValueError(f"재학습 계획에 구성원 {name!r}이 정확히 하나 있지 않다.")
        return matches[0]


def run_member(
    plan: RefitPlan,
    member: RefitMember,
    output: Path,
    *,
    seeds: tuple[int, ...] | None = None,
    finalize: bool = True,
) -> Path:
    """구성원 하나를 전체 자료로 시드별 재학습하고 평균 시험 예측을 저장한다."""
    cfg = load_config(member.config_path, "confirm")
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

    seed_predictions: list[np.ndarray] = []
    seed_records = []
    for seed in selected_seeds:
        budget = member.budgets[seed]
        prediction_path = member_dir / f"test_pred_seed_{seed}.parquet"
        record_path = member_dir / f"test_pred_seed_{seed}.json"
        seed_identity = {
            "schema_version": 1,
            "config": member.config,
            "config_sha256": config_sha256,
            "source_run_id": member.run_id,
            "source_pool_sha256": plan.source_pool_sha256,
            "git_commit": state["git_commit"],
            "input_sha256": input_hashes,
            "seed": seed,
            "training_budget": budget,
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
            if record != expected_record:
                raise ValueError(f"{prediction_path}: 현재 재학습 명세와 계보가 다르다.")
        else:
            if record_path.exists():
                raise ValueError(f"{record_path}: 대응하는 예측 파일이 없다.")
            X_train, X_test = feature_plan.build_full_matrices(train, test, seed)
            provider = initial_score.create(cfg.initial_score)
            scores = (
                provider.compute(train.drop(columns=[TARGET]), test, seed)
                if provider is not None
                else None
            )
            adapter = model.create(cfg.model, seed)
            model.set_dataset_reference(adapter, X_train, X_test)
            model.fit_full(
                adapter,
                X_train,
                y,
                budget,
                scores.train if scores is not None else None,
            )
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
        "schema_version": 1,
        "config": member.config,
        "config_path": str(member.config_path),
        "config_sha256": config_sha256,
        "source_run_id": member.run_id,
        "source_pool_sha256": plan.source_pool_sha256,
        "git_commit": state["git_commit"],
        "git_dirty": git_dirty,
        "input_sha256": input_hashes,
        "feature_columns": feature_plan.all_columns(),
        "seeds": seed_records,
        "budget_source": member.budget_source,
        "prediction_sha256": prediction_array_sha256(averaged),
    }
    _atomic_json(manifest, member_dir / "manifest.json")
    return averaged_path


def assemble(plan: RefitPlan, output: Path) -> dict[str, Path]:
    """CV 전용, 전체 자료 전용, 모델 수 5:1 혼합의 최종 결합 예측을 만든다."""
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
        cv_weight=plan.cv_model_weight,
        full_weight=plan.full_model_weight,
    )

    combiner = COMBINER_REGISTRY[plan.combiner]
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
        "schema_version": 1,
        "source_pool_sha256": plan.source_pool_sha256,
        "combiner": plan.combiner,
        "cv_model_weight": plan.cv_model_weight,
        "full_model_weight": plan.full_model_weight,
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


def _load_member_full_prediction(
    plan: RefitPlan,
    member: RefitMember,
    output: Path,
    expected_ids: pd.Series,
) -> np.ndarray:
    member_dir = output / member.config
    prediction = _load_prediction(member_dir / "test_pred_full.parquet", expected_ids)
    manifest_path = member_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{member.config}: 구성원 manifest가 없다.")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "config": member.config,
        "source_run_id": member.run_id,
        "source_pool_sha256": plan.source_pool_sha256,
        "prediction_sha256": prediction_array_sha256(prediction),
        "budget_source": member.budget_source,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{member.config}: manifest의 {key} 계보가 다르다.")
    return prediction


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

    plan = RefitPlan.load(args.plan)
    if args.assemble:
        for name, path in assemble(plan, args.out_dir).items():
            print(f"{name}: {path}")
        return
    members = plan.members if args.all else (plan.member(args.member),)
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
