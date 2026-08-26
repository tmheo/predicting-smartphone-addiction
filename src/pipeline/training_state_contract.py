"""한 학습 궤적에서 미리 고정한 여러 후보 시점을 보존하는 실행 계약."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .config import ExperimentConfig, TrainingStateConfig
from .data import file_sha256

SCHEMA_VERSION = 1
MANIFEST_NAME = "training_state_manifest.json"
PARENT_MANIFEST_NAME = "training_state_trajectory.json"
CANDIDATE_RUNS_NAME = "training_state_candidate_runs.json"


class TrainingStateContractError(ValueError):
    """후보 설정 집합이 하나의 고정 학습 궤적을 정의하지 못한다."""


def _repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise TrainingStateContractError("후보 config의 Git 저장소 루트를 찾을 수 없다.")
    return Path(result.stdout.strip()).resolve()


def _canonical_committed_config(
    cfg: ExperimentConfig,
    *,
    repository_root: Path,
    git_commit: str,
) -> tuple[str, str, bytes]:
    resolved = cfg.source_path.resolve()
    config_root = (repository_root / "configs").resolve()
    if (
        resolved.parent != config_root
        or resolved.suffix not in {".yaml", ".yml"}
        or resolved.stem != cfg.name
    ):
        raise TrainingStateContractError(
            f"{cfg.source_path}: 학습 시점 후보 config는 저장소의 configs/ 바로 아래 YAML이어야 한다."
        )
    relative = resolved.relative_to(repository_root).as_posix()
    shown = subprocess.run(
        ["git", "show", f"{git_commit}:{relative}"],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    if shown.returncode != 0:
        raise TrainingStateContractError(
            f"{relative}: claimed commit {git_commit}에 후보 config가 없다."
        )
    committed_sha256 = hashlib.sha256(shown.stdout).hexdigest()
    current_sha256 = file_sha256(resolved)
    if current_sha256 != committed_sha256:
        raise TrainingStateContractError(
            f"{relative}: 현재 후보 config가 claimed commit {git_commit}의 내용과 다르다."
        )
    return relative, committed_sha256, shown.stdout


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def frame_content_sha256(frame: pd.DataFrame) -> str:
    """행과 열 순서를 포함한 자료틀 내용 해시."""
    payload = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CandidateIdentity:
    config: ExperimentConfig
    config_path: str
    config_sha256: str
    config_bytes: bytes
    snapshot_identity_sha256: str

    @property
    def completed_epochs(self) -> int:
        assert self.config.training_state is not None
        return self.config.training_state.selected

    def to_json(self) -> dict[str, object]:
        return {
            "config_name": self.config.name,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "completed_epochs": self.completed_epochs,
            "snapshot_identity_sha256": self.snapshot_identity_sha256,
        }


@dataclass(frozen=True)
class TrainingStateRunContract:
    trajectory: str
    state: TrainingStateConfig
    stage: str
    seeds: tuple[int, ...]
    model_kind: str
    git_commit: str
    input_sha256: dict[str, str]
    shared_config_sha256: str
    candidate_set_sha256: str
    trajectory_identity_sha256: str
    candidates: tuple[CandidateIdentity, ...]

    def parent_manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_kind": "training_state_trajectory",
            "trajectory": self.trajectory,
            "trajectory_identity_sha256": self.trajectory_identity_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "shared_config_sha256": self.shared_config_sha256,
            "git_commit": self.git_commit,
            "input_sha256": dict(sorted(self.input_sha256.items())),
            "stage": self.stage,
            "seeds": list(self.seeds),
            "model_kind": self.model_kind,
            "state_kind": self.state.state_kind,
            "selection_rule": self.state.selection_rule,
            "precommitted_candidates": list(self.state.candidates),
            "schedule_horizon_epochs": self.state.schedule_horizon_epochs,
            "trajectory_end_epochs": self.state.trajectory_end_epochs,
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }


def build_run_contract(
    configs: list[ExperimentConfig],
    *,
    git_commit: str,
    input_sha256: dict[str, str],
) -> TrainingStateRunContract:
    """고유 후보 설정 전체가 같은 물리 궤적을 가리키는지 검증한다."""
    if not configs:
        raise TrainingStateContractError("학습 시점 후보 설정이 하나도 없다.")
    if not isinstance(git_commit, str) or not git_commit:
        raise TrainingStateContractError("학습 시점 실행의 claimed Git commit이 비어 있다.")
    repository_root = _repository_root()
    if len({cfg.name for cfg in configs}) != len(configs):
        raise TrainingStateContractError("후보 설정 이름에 중복이 있다.")
    if len({cfg.source_path.resolve() for cfg in configs}) != len(configs):
        raise TrainingStateContractError("후보 설정 경로에 중복이 있다.")
    for cfg in configs:
        if cfg.training_state is None:
            raise TrainingStateContractError(
                f"{cfg.source_path}: training_state 계약이 없는 일반 실행 설정이다."
            )
        if cfg.source_path.stem != cfg.name:
            raise TrainingStateContractError(
                f"{cfg.source_path}: 파일 이름과 experiment 이름이 다르다: "
                f"{cfg.source_path.stem!r} != {cfg.name!r}"
            )
        if cfg.initial_score is not None:
            raise TrainingStateContractError(
                f"{cfg.name}: 여러 학습 시점 경로는 initial_score를 지원하지 않는다."
            )

    first = configs[0]
    assert first.training_state is not None
    expected_state = first.training_state
    expected_common = (
        expected_state.trajectory,
        expected_state.candidates,
        expected_state.schedule_horizon_epochs,
        expected_state.trajectory_end_epochs,
        expected_state.state_kind,
        expected_state.selection_rule,
        first.stage,
        tuple(first.seeds),
        first.model.kind,
    )
    for cfg in configs[1:]:
        assert cfg.training_state is not None
        actual_common = (
            cfg.training_state.trajectory,
            cfg.training_state.candidates,
            cfg.training_state.schedule_horizon_epochs,
            cfg.training_state.trajectory_end_epochs,
            cfg.training_state.state_kind,
            cfg.training_state.selection_rule,
            cfg.stage,
            tuple(cfg.seeds),
            cfg.model.kind,
        )
        if actual_common != expected_common:
            raise TrainingStateContractError(
                f"{cfg.name}: 후보 설정의 궤적, 시드, 단계 또는 모델 계약이 다르다."
            )

    selected = tuple(sorted(cfg.training_state.selected for cfg in configs))
    if selected != expected_state.candidates:
        raise TrainingStateContractError(
            "전달한 후보 설정의 selected 집합이 사전 고정 candidates와 다르다: "
            f"{selected} != {expected_state.candidates}"
        )

    committed_configs = {
        cfg.name: _canonical_committed_config(
            cfg,
            repository_root=repository_root,
            git_commit=git_commit,
        )
        for cfg in configs
    }
    normalized = [
        _normalized_config(cfg, source_bytes=committed_configs[cfg.name][2])
        for cfg in configs
    ]
    if any(value != normalized[0] for value in normalized[1:]):
        raise TrainingStateContractError(
            "후보 설정은 name과 training_state.selected 외의 내용이 모두 같아야 한다."
        )
    shared_config_sha256 = content_sha256(normalized[0])

    ordered_cfgs = sorted(configs, key=lambda cfg: cfg.training_state.selected)
    candidate_records = [
        {
            "config_name": cfg.name,
            "config_path": committed_configs[cfg.name][0],
            "config_sha256": committed_configs[cfg.name][1],
            "completed_epochs": cfg.training_state.selected,
            "schedule_horizon_epochs": cfg.training_state.schedule_horizon_epochs,
        }
        for cfg in ordered_cfgs
    ]
    candidate_set_sha256 = content_sha256(candidate_records)
    trajectory_document = {
        "schema_version": SCHEMA_VERSION,
        "trajectory": expected_state.trajectory,
        "git_commit": git_commit,
        "input_sha256": dict(sorted(input_sha256.items())),
        "stage": first.stage,
        "seeds": list(first.seeds),
        "model_kind": first.model.kind,
        "shared_config_sha256": shared_config_sha256,
        "candidate_set_sha256": candidate_set_sha256,
        "precommitted_candidates": list(expected_state.candidates),
        "schedule_horizon_epochs": expected_state.schedule_horizon_epochs,
        "trajectory_end_epochs": expected_state.trajectory_end_epochs,
        "state_kind": expected_state.state_kind,
        "selection_rule": expected_state.selection_rule,
    }
    trajectory_identity_sha256 = content_sha256(trajectory_document)
    identities = tuple(
        CandidateIdentity(
            config=cfg,
            config_path=record["config_path"],
            config_sha256=record["config_sha256"],
            config_bytes=committed_configs[cfg.name][2],
            snapshot_identity_sha256=content_sha256(
                {
                    "trajectory_identity_sha256": trajectory_identity_sha256,
                    "config_sha256": record["config_sha256"],
                    "completed_epochs": record["completed_epochs"],
                    "schedule_horizon_epochs": record["schedule_horizon_epochs"],
                }
            ),
        )
        for cfg, record in zip(ordered_cfgs, candidate_records)
    )
    return TrainingStateRunContract(
        trajectory=expected_state.trajectory,
        state=expected_state,
        stage=first.stage,
        seeds=tuple(first.seeds),
        model_kind=first.model.kind,
        git_commit=git_commit,
        input_sha256=dict(sorted(input_sha256.items())),
        shared_config_sha256=shared_config_sha256,
        candidate_set_sha256=candidate_set_sha256,
        trajectory_identity_sha256=trajectory_identity_sha256,
        candidates=identities,
    )


def _normalized_config(
    cfg: ExperimentConfig, *, source_bytes: bytes | None = None
) -> dict[str, object]:
    try:
        raw = yaml.safe_load(
            source_bytes if source_bytes is not None else cfg.source_path.read_bytes()
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TrainingStateContractError(
            f"후보 설정을 읽을 수 없다: {cfg.source_path}"
        ) from exc
    if not isinstance(raw, dict):
        raise TrainingStateContractError(f"후보 설정 루트는 객체여야 한다: {cfg.source_path}")
    normalized = json.loads(json.dumps(raw))
    normalized.pop("name", None)
    state = normalized.get("training_state")
    if not isinstance(state, dict):
        raise TrainingStateContractError(
            f"후보 설정 training_state는 객체여야 한다: {cfg.source_path}"
        )
    state.pop("selected", None)
    return normalized
