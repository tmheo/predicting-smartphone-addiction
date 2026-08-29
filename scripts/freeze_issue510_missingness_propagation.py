"""이슈 510의 34개 결측 증강 짝비교 설정과 출처 학습 길이를 동결한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.data import file_sha256  # noqa: E402
from pipeline.paired_training_length import CONTRACT, SCHEMA_VERSION  # noqa: E402
from pipeline.runs import MlflowRunStore  # noqa: E402


POOL_PATH = Path("artifacts/pool.yaml")
REFIT_PLAN_PATH = Path("artifacts/full-refit-plan.yaml")
CAPACITY_PATH = Path("artifacts/issue509-parallel-capacity-freeze.json")
EVIDENCE_PATH = Path("artifacts/issue510-paired-training-lengths.json")
FREEZE_PATH = Path("artifacts/issue510-missingness-propagation-precommit.json")
CONFIG_DIR = Path("configs/missingness-propagation")
EXPECTED_POOL_SHA256 = "c513443b6d1cc8af348dc06f8c547ed2728a659261cf7d78dc4e17a27ca668d9"
EXCLUDED = {
    "exp067_tabpfn3",
    "exp208_issue500_ag25_missingness_augmented",
}
SEEDS = [42, 43, 44]
OUTER_FOLDS = [0, 1, 2, 3, 4]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="원천 실행 저장소 mlflow.db가 있는 저장소 루트",
    )
    return parser.parse_args()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def _yaml_bytes(value: Any) -> bytes:
    return yaml.safe_dump(
        value,
        allow_unicode=True,
        sort_keys=False,
        width=100,
    ).encode()


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"YAML 루트가 객체가 아니다: {path}")
    return raw


def _pool_members() -> list[dict[str, Any]]:
    if file_sha256(POOL_PATH) != EXPECTED_POOL_SHA256:
        raise ValueError("후보 풀 해시가 이슈 507에서 고정한 값과 다르다.")
    members = _load_yaml(POOL_PATH)["members"]
    selected = [member for member in members if member["config"] not in EXCLUDED]
    if len(members) != 36 or len(selected) != 34:
        raise ValueError("후보 풀 36개에서 지정한 2개를 뺀 34개가 되지 않는다.")
    return selected


def _runtime_classes() -> dict[str, str]:
    raw = json.loads(CAPACITY_PATH.read_text())
    result: dict[str, str] = {}
    for runtime_class, section in raw["runtime_classes"].items():
        for member in section["members"]:
            if member in result:
                raise ValueError(f"자원 등급에 후보가 중복됐다: {member}")
            result[member] = runtime_class
    return result


def _observations_from_diagnostics(store: MlflowRunStore, run_id: str) -> list[dict]:
    diagnostics = json.loads(
        store.artifact_bytes_of(run_id, "model_training_diagnostics.json")
    )
    observations: list[dict] = []
    for fold_record in diagnostics:
        evidence = fold_record.get("training_length_evidence")
        if not isinstance(evidence, dict):
            continue
        observations.extend(evidence.get("observations", []))
    return observations


def _normalize_observations(raw: list[dict], member: str) -> list[dict]:
    normalized = []
    for observation in raw:
        item = dict(observation)
        item["seed"] = int(item["seed"])
        item["outer_fold"] = int(item["outer_fold"])
        item["inner_member"] = (
            0 if item.get("inner_member") is None else int(item["inner_member"])
        )
        item["observed_training_length"] = int(item["observed_training_length"])
        normalized.append(item)
    normalized.sort(
        key=lambda item: (item["seed"], item["outer_fold"], item["inner_member"])
    )
    expected_coordinates = {
        (seed, fold) for seed in SEEDS for fold in OUTER_FOLDS
    }
    actual_coordinates = {
        (item["seed"], item["outer_fold"]) for item in normalized
    }
    if actual_coordinates != expected_coordinates:
        raise ValueError(f"{member}: 출처 학습 길이 좌표 15개가 완전하지 않다.")
    counts = {}
    for item in normalized:
        key = (item["seed"], item["outer_fold"])
        counts.setdefault(key, []).append(item["inner_member"])
    member_counts = {len(indices) for indices in counts.values()}
    if len(member_counts) != 1 or any(
        sorted(indices) != list(range(len(indices))) for indices in counts.values()
    ):
        raise ValueError(f"{member}: 내부 구성원 좌표가 0부터 연속되지 않는다.")
    return normalized


def _base_configs_and_evidence(
    selected: list[dict[str, Any]], store: MlflowRunStore
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    plan = _load_yaml(REFIT_PLAN_PATH)
    plan_by_name = {member["config"]: member for member in plan["members"]}
    evidence_members = []
    base_configs: dict[str, dict[str, Any]] = {}
    for pool_member in selected:
        member = pool_member["config"]
        run_id = pool_member["run_id"]
        facts = store.facts_of(run_id)
        if facts.status != "FINISHED" or facts.tags.get("git_dirty") != "False":
            raise ValueError(f"{member}: 출처 실행이 완료된 깨끗한 실행이 아니다.")
        if facts.params.get("seeds") != "42,43,44":
            raise ValueError(f"{member}: 출처 실행의 시드가 42,43,44가 아니다.")
        config_artifact = f"{member}.yaml"
        if member == "exp209_issue505_lgb_lr_onehot_init":
            config_path = Path("configs") / config_artifact
            raw_observations = _observations_from_diagnostics(store, run_id)
            status = "confirmed"
            model_kind = facts.params["model.kind"]
            evidence_artifact = "model_training_diagnostics.json"
        else:
            plan_member = plan_by_name.get(member)
            if plan_member is None:
                raise ValueError(f"{member}: 전체 자료 재학습 장부에 정규화 설정이 없다.")
            config_path = Path(plan_member["config_path"])
            training_evidence = plan_member["training_length_evidence"]
            raw_observations = training_evidence["observations"]
            status = training_evidence["status"]
            model_kind = training_evidence["model_family"]
            evidence_artifact = plan_member["lineage"]["evidence_artifact_path"]
        base = _load_yaml(config_path)
        if base["name"] != member or base["model"]["kind"] != model_kind:
            raise ValueError(f"{member}: 정규화 설정의 이름 또는 모형 계열이 다르다.")
        for forbidden in ("training_state", "training_rows", "paired_training_length"):
            if forbidden in base:
                raise ValueError(f"{member}: 정규화 설정에 {forbidden}가 이미 있다.")
        base_payload = _yaml_bytes(base)
        base_configs[member] = base
        observations = (
            []
            if status == "not_applicable"
            else _normalize_observations(raw_observations, member)
        )
        source_identity = {
            "run_id": run_id,
            "git_commit": facts.tags["git_commit"],
            "config_artifact": config_artifact,
            "config_artifact_sha256": store.artifact_sha256_of(
                run_id, config_artifact
            ),
            "oof_artifact": "oof.parquet",
            "oof_artifact_sha256": store.artifact_sha256_of(run_id, "oof.parquet"),
            "training_length_artifact": evidence_artifact,
            "training_length_artifact_sha256": store.artifact_sha256_of(
                run_id, evidence_artifact
            ),
            "normalized_config_path": config_path.as_posix(),
            "normalized_config_sha256": _sha256(base_payload),
            "input_sha256": {
                "train": facts.tags["sha256.train"],
                "test": facts.tags["sha256.test"],
                "folds": facts.tags["sha256.folds"],
            },
        }
        evidence_members.append(
            {
                "member": member,
                "model_kind": model_kind,
                "status": status,
                "source_identity": source_identity,
                "observations": observations,
            }
        )
    return evidence_members, base_configs


def _arm_config(
    base: dict[str, Any],
    member: str,
    arm: str,
    probability: float,
    evidence_sha256: str,
) -> dict[str, Any]:
    config = dict(base)
    config["name"] = f"mpv1_{member}_{arm}"
    config["training_rows"] = {
        "arm": arm,
        "replica_count": 2,
        "observed_cell_mask_probability": probability,
    }
    config["paired_training_length"] = {
        "source": EVIDENCE_PATH.as_posix(),
        "sha256": evidence_sha256,
        "member": member,
    }
    return config


def main() -> None:
    args = _args()
    source_root = args.source_root.resolve()
    database = source_root / "mlflow.db"
    if not database.is_file():
        raise FileNotFoundError(f"원천 실행 저장소가 없다: {database}")
    selected = _pool_members()
    runtime_classes = _runtime_classes()
    selected_names = [member["config"] for member in selected]
    if set(runtime_classes) != set(selected_names):
        raise ValueError("이슈 509의 자원 등급 후보와 이번 34개 후보가 다르다.")
    store = MlflowRunStore(tracking_uri=f"sqlite:///{database}")
    evidence_members, base_configs = _base_configs_and_evidence(selected, store)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "issue": {
            "number": 510,
            "title": "모든 모델 계열에서 결측 증강 짝비교 실행 경계를 구현하고 진단한다",
            "url": "https://github.com/tmheo/predicting-smartphone-addiction/issues/510",
        },
        "candidate_pool_sha256": EXPECTED_POOL_SHA256,
        "seeds": SEEDS,
        "outer_folds": OUTER_FOLDS,
        "members": evidence_members,
    }
    evidence_payload = _json_bytes(evidence)
    EVIDENCE_PATH.write_bytes(evidence_payload)
    evidence_sha256 = _sha256(evidence_payload)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_records = []
    pair_records = []
    for ordinal, member in enumerate(selected_names, start=1):
        arms = []
        for arm, probability in (
            ("tripled", 0.0),
            ("missingness_augmented", 0.25),
        ):
            config = _arm_config(
                base_configs[member], member, arm, probability, evidence_sha256
            )
            path = CONFIG_DIR / f"{ordinal:02d}_{member}_{arm}.yaml"
            payload = _yaml_bytes(config)
            path.write_bytes(payload)
            record = {
                "arm": arm,
                "path": path.as_posix(),
                "sha256": _sha256(payload),
                "name": config["name"],
            }
            config_records.append(record | {"member": member})
            arms.append(record)
        common = dict(base_configs[member])
        common_sha256 = _canonical_sha256(common)
        pair_records.append(
            {
                "ordinal": ordinal,
                "member": member,
                "runtime_class": runtime_classes[member],
                "common_config_semantic_sha256": common_sha256,
                "arms": arms,
            }
        )
    freeze = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "issue": evidence["issue"],
        "map": {
            "number": 506,
            "title": "지도: 결측 증강을 후보 풀 전 계열에 전파해 마지막 제출 개선을 판정한다",
            "url": "https://github.com/tmheo/predicting-smartphone-addiction/issues/506",
        },
        "inputs": {
            "candidate_pool": {
                "path": POOL_PATH.as_posix(),
                "sha256": file_sha256(POOL_PATH),
                "member_count": 36,
            },
            "full_refit_plan": {
                "path": REFIT_PLAN_PATH.as_posix(),
                "sha256": file_sha256(REFIT_PLAN_PATH),
            },
            "capacity_freeze": {
                "path": CAPACITY_PATH.as_posix(),
                "sha256": file_sha256(CAPACITY_PATH),
            },
        },
        "scope": {
            "selected_member_count": 34,
            "pair_count": 34,
            "arm_count": 68,
            "excluded": sorted(EXCLUDED),
            "member_order": selected_names,
        },
        "paired_training_length_evidence": {
            "path": EVIDENCE_PATH.as_posix(),
            "sha256": evidence_sha256,
        },
        "pair_contract": {
            "config_differences": ["name", "training_rows"],
            "tripled": {
                "replica_count": 2,
                "observed_cell_mask_probability": 0.0,
            },
            "missingness_augmented": {
                "replica_count": 2,
                "observed_cell_mask_probability": 0.25,
            },
            "same_source_training_lengths": True,
            "seeds": SEEDS,
            "outer_folds": OUTER_FOLDS,
        },
        "pairs": pair_records,
        "config_manifest_sha256": _canonical_sha256(config_records),
    }
    FREEZE_PATH.write_bytes(_json_bytes(freeze))
    print(
        f"동결 완료: pairs={len(pair_records)} configs={len(config_records)} "
        f"evidence_sha256={evidence_sha256}"
    )


if __name__ == "__main__":
    main()
