"""판정 회차(JudgmentRound, CONTEXT.md 용어)의 정본 구현. (#552, 지도 #550)

스택 교체 판정 스크립트 5곳이 재구현하던 공통 골격(봉인 → 실행 → 비교 → 보고 →
게시, 재개 검사, compare 불변식, 보고서 골격, 게시 manifest)을 이 module이
소유한다. 이슈별 스펙 스크립트는 RoundSpec을 조립해 main(spec)을 부르는 얇은
층만 쓰고, 회차 고유 선택 규칙(잡음 바닥 동률, 사다리 선택)과 진단 절만 소유한다.

회차 id "<주제 슬러그>/<이슈 번호>"가 작업 산출물(run-logs/<id>/)과 커밋 기록
(docs/research/<id>/)의 두 뿌리 경로를 결정한다. 수동 RUN_DIR·PUBLISH_DIR
상수는 스펙에 없다.

동사 5개:
- precommit: 구성원 행렬을 hash-verified로 적재해 캐시 parquet을 만들고 입력
  해시·코드 상태·기준값·게이트를 봉인 기록(<계약판>/precommit/1)으로 남긴다.
  dirty git이면 거부한다.
- run: 자기 검사·분할 작업을 스펙 스크립트 재호출 subprocess로 pipeline.jobs에
  맡긴다. 완료된 작업은 건너뛴다.
- compare: 분할 기록을 취합해 judgment.judge_stack_replacement로 판정하고
  comparison 기록을 봉인한다.
- report: 공통 골격(## 판정 → ## 분할별 결과 → 회차 고유 절 → ## 동결과 재현성)의
  보고서를 쓴다.
- publish: 화이트리스트(precommit·selfcheck·fold·comparison·report)를
  docs/research/<id>/로 복사하고 게시 폴더 기준으로 manifest.sha256을 재생성해
  자기 대조한다. 게시 위치에 없는 파일은 목록에 없다(지도 #550 사실 조사의
  manifest 결함 교정).

재개 검사: 모든 동사가 precommit을 SealedRecord.open으로 열고 입력 파일 해시와
코드 상태(git commit·dirty·uv.lock·관련 소스 해시)를 재계산 대조한다. 불일치는
판정 불가다. fold·selfcheck·comparison 기록은 precommit의 derive로 계보를 잇는다.

자기 검사는 3등급 열거값(해시 동일성 / 봉인 분할 1개 재현 / 전 분할 재현)이고
기본값이 없다: 회차가 등급과 기준값 출처를 명시한다(지도 #550 결정 6).

CLI 번역: JudgmentError 계열은 "판정 불가: ..." 종료로 번역한다(compare.py 관례).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

from . import data as data_module
from . import ensemble
from . import identity as identity_module
from . import jobs as jobs_module
from . import judgment as judgment_module
from . import member_sources as member_sources_module
from . import members as members_module
from . import sealed as sealed_module
from .data import ID, labels as load_labels
from .ensemble import CombinerConvergenceError, evaluate_outer_fold
from .identity import array_identity, file_identity
from .judgment import (
    FoldScores,
    JudgmentError,
    StackGate,
    StackVerdict,
    judge_stack_replacement,
)
from .jobs import Job, JobsError, JobsFailed, run_jobs
from .members import HASH_VERIFIED, MemberSource, MembersError, load_members
from .runs import RunStore, RunStoreError
from .sealed import SealedRecord, SealedRecordError, canonical_sha256

RUN_LOGS_ROOT = Path("run-logs")
RESEARCH_ROOT = Path("docs/research")

SELF_CHECK_HASH_IDENTITY = "hash-identity"
SELF_CHECK_SEALED_FOLD_REPLAY = "sealed-fold-replay"
SELF_CHECK_ALL_FOLDS_REPLAY = "all-folds-replay"
SELF_CHECK_GRADES = (
    SELF_CHECK_HASH_IDENTITY,
    SELF_CHECK_SEALED_FOLD_REPLAY,
    SELF_CHECK_ALL_FOLDS_REPLAY,
)

# 코드 상태에 해시로 동결하는 관련 소스. 스펙 스크립트 자신은 code_state가 따로 담는다.
_SOURCE_MODULES = (
    sys.modules[__name__],
    data_module,
    ensemble,
    identity_module,
    jobs_module,
    judgment_module,
    member_sources_module,
    members_module,
    sealed_module,
)

_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JudgmentError(message)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _require_name(value: str, what: str) -> None:
    if not _NAME_PATTERN.fullmatch(value) or value in (".", ".."):
        raise JudgmentError(f"{what}은(는) 파일 이름으로 쓸 수 있어야 한다: {value!r}")


@dataclass(frozen=True)
class ReferenceValues:
    """기준 팔의 동결된 점수와 그 출처. compare가 이 값과만 비교한다."""

    source: str  # 기준값 출처(파일 경로나 기록 이름). 지도 #550 결정 6.
    nested_auc: float
    fold_aucs: Mapping[int, float]

    def __post_init__(self) -> None:
        _require(bool(self.source), "기준값 출처(source)가 비어 있다.")
        _require(bool(self.fold_aucs), "기준 fold_aucs가 비어 있다.")
        object.__setattr__(
            self, "fold_aucs", {int(k): float(v) for k, v in dict(self.fold_aucs).items()}
        )

    def fold_scores(self) -> FoldScores:
        return FoldScores(nested_auc=self.nested_auc, fold_aucs=dict(self.fold_aucs))


@dataclass(frozen=True)
class ReferenceArm:
    """비교 팔: 동결된 기준값과, 자기 검사 재현 등급에서 다시 돌릴 구성원 출처."""

    name: str
    values: ReferenceValues
    source: MemberSource | None = None  # 재현 등급 자기 검사에만 필요하다.

    def __post_init__(self) -> None:
        _require_name(self.name, "기준 팔 이름")


@dataclass(frozen=True)
class CandidateArm:
    """평가 팔 하나. 사다리 회차는 여러 개를 나열한다."""

    name: str
    source: MemberSource

    def __post_init__(self) -> None:
        _require_name(self.name, "평가 팔 이름")


@dataclass(frozen=True)
class ExpectedFold:
    """자기 검사 재현이 대조할 분할 하나의 기대값."""

    auc: float
    prediction_sha256: str | None = None


@dataclass(frozen=True)
class SelfCheckSpec:
    """자기 검사 등급과 기준값 출처. 기본값 없음(지도 #550 결정 6)."""

    grade: str
    source: str  # 기대값이 어디서 왔는가.
    expected: Mapping[int, ExpectedFold] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(
            self.grade in SELF_CHECK_GRADES,
            f"자기 검사 등급은 {SELF_CHECK_GRADES} 중 하나여야 한다: {self.grade!r}",
        )
        _require(bool(self.source), "자기 검사 기준값 출처(source)가 비어 있다.")
        object.__setattr__(self, "expected", {int(k): v for k, v in dict(self.expected).items()})
        if self.grade == SELF_CHECK_HASH_IDENTITY:
            _require(not self.expected, "해시 동일성 등급은 재현 기대값을 갖지 않는다.")
        if self.grade == SELF_CHECK_SEALED_FOLD_REPLAY:
            _require(
                len(self.expected) == 1,
                "봉인 분할 1개 재현 등급은 정확히 분할 1개의 기대값이 필요하다.",
            )
        if self.grade == SELF_CHECK_ALL_FOLDS_REPLAY:
            _require(bool(self.expected), "전 분할 재현 등급은 분할별 기대값이 필요하다.")

    @property
    def replays(self) -> bool:
        return self.grade != SELF_CHECK_HASH_IDENTITY


@dataclass(frozen=True)
class RoundRecords:
    """보고서 회차 고유 절(render hook)에 건네는 읽기 전용 문맥."""

    spec: RoundSpec
    precommit: SealedRecord
    comparison: SealedRecord
    folds: Mapping[str, Mapping[int, Mapping[str, object]]]  # 팔 → 분할 → 기록 본문.


@dataclass(frozen=True)
class ReportSection:
    """회차 고유 진단 절. render는 '## {title}' 아래에 들어갈 markdown 줄 목록을 돌려준다."""

    title: str
    render: Callable[[RoundRecords], list[str]]


@dataclass(frozen=True)
class RoundSpec:
    """판정 회차 하나의 선언. 회차 id가 산출물 두 뿌리의 경로를 결정한다."""

    round_id: str  # "<주제 슬러그>/<이슈 번호>", 예: "extended-stack-pool-reassembly/issue513".
    title: str
    question: str
    contract: str  # 봉인 기록 schema의 계약판(첫 분절).
    reference: ReferenceArm
    candidates: tuple[CandidateArm, ...]
    combiner: str  # 결합기 등록 이름. ensemble.combiner_for_context가 해석한다.
    gate: StackGate
    selfcheck: SelfCheckSpec
    folds_path: Path = Path("artifacts/folds.parquet")
    train_path: Path = Path("data/train.csv")
    sealed_inputs: tuple[Path, ...] = ()  # 추가로 해시 동결할 입력 파일(구성원 명세 등).
    publish_extra: tuple[str, ...] = ()  # run-dir 상대 경로의 추가 게시 파일.
    report_sections: tuple[ReportSection, ...] = ()

    def __post_init__(self) -> None:
        segments = self.round_id.split("/")
        _require(
            len(segments) == 2 and all(segments),
            f"회차 id는 '<주제 슬러그>/<이슈 번호>' 두 분절이어야 한다: {self.round_id!r}",
        )
        for segment in segments:
            _require_name(segment, "회차 id 분절")
        _require(bool(self.title), "회차 title이 비어 있다.")
        _require(bool(self.question), "회차 question이 비어 있다.")
        _require(
            "/" not in self.contract and bool(self.contract),
            f"contract는 '/' 없는 계약판 한 분절이어야 한다: {self.contract!r}",
        )
        _require(bool(self.candidates), "평가 팔이 없다.")
        names = [self.reference.name, *(arm.name for arm in self.candidates)]
        _require(
            len(set(names)) == len(names),
            f"팔 이름이 중복된다: {sorted({n for n in names if names.count(n) > 1})}",
        )
        known = set(ensemble.COMBINER_REGISTRY) | {
            ensemble.CSelectedShrunkRankLogitCombiner.name
        }
        _require(
            self.combiner in known,
            f"결합기 이름 {self.combiner!r}을 해석할 수 없다.",
        )
        _require(
            self.gate.folds_required_positive >= 1,
            "게이트의 양수 fold 요구 수는 1 이상이어야 한다.",
        )
        if self.selfcheck.replays:
            _require(
                self.reference.source is not None,
                f"자기 검사 등급 {self.selfcheck.grade}에는 기준 팔 구성원 출처가 필요하다.",
            )
        for extra in self.publish_extra:
            _require(
                not Path(extra).is_absolute() and ".." not in Path(extra).parts,
                f"게시 추가 목록은 run-dir 상대 경로여야 한다: {extra!r}",
            )

    @property
    def precommit_schema(self) -> str:
        return f"{self.contract}/precommit/1"


def default_code_state(root: Path, script: Path | None) -> dict[str, object]:
    """git 상태와 관련 소스·환경 잠금 해시. 재개 검사가 dict 전체 동일성으로 대조한다."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    state: dict[str, object] = {
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        },
        "uv_lock_sha256": file_identity(root / "uv.lock"),
        "sources": {
            module.__name__.rsplit(".", 1)[-1]: file_identity(Path(module.__file__))
            for module in _SOURCE_MODULES
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    if script is not None:
        state["script"] = {"path": str(script), "sha256": file_identity(script)}
    return state


def _environment() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


def _write_document(path: Path, record: SealedRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record.write(path)


@dataclass
class JudgmentRound:
    """RoundSpec 하나를 동사 5개로 실행하는 오케스트레이터.

    store는 precommit의 구성원 적재에만 쓴다(분할 작업은 캐시 parquet만 읽는다).
    script는 run이 subprocess로 재호출할 스펙 스크립트 경로다.
    code_state는 시험이 결정적 상태를 주입하는 자리이며 기본은 default_code_state다.
    """

    spec: RoundSpec
    store: RunStore | None = None
    root: Path = Path(".")
    script: Path | None = None
    code_state: Callable[[], dict[str, object]] | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.code_state is None:
            self.code_state = lambda: default_code_state(self.root, self.script)

    # ------------------------------------------------------------------ 경로
    @property
    def run_dir(self) -> Path:
        return self.root / RUN_LOGS_ROOT / self.spec.round_id

    @property
    def publish_dir(self) -> Path:
        return self.root / RESEARCH_ROOT / self.spec.round_id

    @property
    def precommit_path(self) -> Path:
        return self.run_dir / "precommit.json"

    def _cache_path(self, arm: str) -> Path:
        return self.run_dir / "cache" / f"{arm}.parquet"

    def _fold_dir(self, arm: str, fold: int) -> Path:
        return self.run_dir / "arms" / arm / f"fold-{fold}"

    def _fold_record_path(self, arm: str, fold: int) -> Path:
        return self._fold_dir(arm, fold) / "fold.json"

    def _selfcheck_paths(self) -> list[Path]:
        if self.spec.selfcheck.replays:
            return [
                self.run_dir / "selfcheck" / f"fold-{fold}.json"
                for fold in sorted(self.spec.selfcheck.expected)
            ]
        return [self.run_dir / "selfcheck" / "selfcheck.json"]

    # ------------------------------------------------------------- 공통 적재
    def _load_fold_and_labels(self) -> tuple[pd.Series, pd.Series]:
        folds_path = self.root / self.spec.folds_path
        _require(folds_path.is_file(), f"고정 분할 파일이 없다: {folds_path}")
        frame = pd.read_parquet(folds_path)
        _require(
            list(frame.columns) == [ID, "fold"],
            f"고정 분할 열이 [{ID}, fold]가 아니다: {list(frame.columns)}",
        )
        fold_of = frame.set_index(ID)["fold"]
        y = load_labels(fold_of.index, train_path=self.root / self.spec.train_path)
        return fold_of, y

    def _outer_folds(self, fold_of: pd.Series) -> list[int]:
        return [int(fold) for fold in sorted(fold_of.unique())]

    def _combiner(self, fold_of: pd.Series) -> ensemble.Combiner:
        return ensemble.combiner_for_context(
            self.spec.combiner, fold_of=fold_of, band_of=None
        )

    def _matrix_arms(self) -> list[tuple[str, MemberSource]]:
        arms = [(arm.name, arm.source) for arm in self.spec.candidates]
        if self.spec.selfcheck.replays:
            arms.insert(0, (self.spec.reference.name, self.spec.reference.source))
        return arms

    def _load_cached_matrix(self, payload: Mapping, arm: str, fold_of: pd.Series) -> pd.DataFrame:
        matrix = pd.read_parquet(self._cache_path(arm)).astype(np.float64)
        expected_columns = [row["member_id"] for row in payload["arms"][arm]["members"]]
        _require(
            list(matrix.columns) == expected_columns,
            f"{arm}: 캐시 열 순서가 precommit과 다르다.",
        )
        _require(
            matrix.index.equals(fold_of.index),
            f"{arm}: 캐시 행 순서가 고정 분할과 다르다.",
        )
        _require(
            bool(np.isfinite(matrix.to_numpy()).all()),
            f"{arm}: 캐시에 비유한값이 있다.",
        )
        return matrix

    # ---------------------------------------------------------- 동사 1: 봉인
    def precommit(self) -> SealedRecord:
        _require(self.store is not None, "precommit에는 실행 저장소(store)가 필요하다.")
        _require(
            not self.precommit_path.exists(),
            f"precommit.json이 이미 있다: {self.precommit_path}",
        )
        state = self.code_state()
        _require(not state["git"]["dirty"], "판정은 커밋된 코드 상태에서만 시작한다.")
        fold_of, y = self._load_fold_and_labels()
        outer_folds = self._outer_folds(fold_of)
        reference = self.spec.reference
        _require(
            sorted(reference.values.fold_aucs) == outer_folds,
            "기준값의 분할 구성이 고정 분할과 다르다: "
            f"{sorted(reference.values.fold_aucs)} != {outer_folds}",
        )
        selfcheck = self.spec.selfcheck
        if selfcheck.replays:
            _require(
                set(selfcheck.expected) <= set(outer_folds),
                "자기 검사 기대값에 고정 분할 밖의 분할이 있다.",
            )
            if selfcheck.grade == SELF_CHECK_ALL_FOLDS_REPLAY:
                _require(
                    sorted(selfcheck.expected) == outer_folds,
                    "전 분할 재현 등급의 기대값이 전 분할을 덮지 않는다.",
                )
        inputs: dict[str, dict[str, str]] = {}
        for label, path in (
            ("folds", self.spec.folds_path),
            ("train", self.spec.train_path),
            *((str(extra), extra) for extra in self.spec.sealed_inputs),
        ):
            resolved = self.root / path
            _require(resolved.is_file(), f"동결할 입력 파일이 없다: {resolved}")
            inputs[label] = {"path": str(path), "sha256": file_identity(resolved)}
        arms_payload: dict[str, dict[str, object]] = {}
        for name, source in self._matrix_arms():
            matrix = load_members(source, fold_of.index, self.store, labels=y)
            matrix.require(HASH_VERIFIED)
            rows = [
                {
                    "member_id": str(row.member_id),
                    "origin": str(row.origin),
                    "verification": str(row.verification),
                    "oof_sha256": str(row.oof_sha256),
                    "rescored_auc": float(row.rescored_auc),
                }
                for row in matrix.members.itertuples()
            ]
            cache_path = self._cache_path(name)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            matrix.oof_frame().to_parquet(cache_path)
            arms_payload[name] = {
                "role": "reference" if name == reference.name else "candidate",
                "member_count": len(rows),
                "members": rows,
                "composition_sha256": canonical_sha256(
                    [[row["member_id"], row["oof_sha256"]] for row in rows]
                ),
                "cache_sha256": file_identity(cache_path),
            }
        payload: dict[str, object] = {
            "round_id": self.spec.round_id,
            "title": self.spec.title,
            "question": self.spec.question,
            "created_at": _now_iso(),
            "outer_folds": outer_folds,
            "combiner": self.spec.combiner,
            "gate": {
                "delta_required": self.spec.gate.delta_required,
                "folds_required_positive": self.spec.gate.folds_required_positive,
            },
            "reference": {
                "name": reference.name,
                "values_source": reference.values.source,
                "nested_auc": reference.values.nested_auc,
                "fold_aucs": {
                    str(fold): auc for fold, auc in sorted(reference.values.fold_aucs.items())
                },
            },
            "candidates": [arm.name for arm in self.spec.candidates],
            "selfcheck": {
                "grade": selfcheck.grade,
                "source": selfcheck.source,
                "expected": {
                    str(fold): {
                        "auc": expected.auc,
                        "prediction_sha256": expected.prediction_sha256,
                    }
                    for fold, expected in sorted(selfcheck.expected.items())
                },
            },
            "arms": arms_payload,
            "inputs": inputs,
            "environment": _environment(),
            "code_state": state,
        }
        record = SealedRecord.seal(self.spec.precommit_schema, payload)
        _write_document(self.precommit_path, record)
        print(f"precommit 봉인: {self.precommit_path} ({record.sealed_sha256})")
        return record

    # ------------------------------------------------------------- 재개 검사
    def open_precommit(self) -> SealedRecord:
        """봉인을 열고 입력 해시와 코드 상태를 재계산 대조한다. 모든 동사의 시작이다."""
        record = SealedRecord.open(self.precommit_path, schema=self.spec.precommit_schema)
        payload = record.payload
        for label, entry in payload["inputs"].items():
            path = self.root / entry["path"]
            _require(path.is_file(), f"동결 입력 {label} 파일이 사라졌다: {path}")
            _require(
                file_identity(path) == entry["sha256"],
                f"동결 입력 {label}의 해시가 precommit과 다르다: {path}",
            )
        for name, arm in payload["arms"].items():
            cache_path = self._cache_path(name)
            _require(cache_path.is_file(), f"{name}: 캐시 parquet이 사라졌다.")
            _require(
                file_identity(cache_path) == arm["cache_sha256"],
                f"{name}: 캐시 parquet 해시가 precommit과 다르다.",
            )
        state = self.code_state()
        frozen = payload["code_state"]
        _require(not state["git"]["dirty"], "판정 실행 중 작업 트리가 바뀌었다(dirty).")
        for label, actual, expected in (
            ("git commit", state["git"]["commit"], frozen["git"]["commit"]),
            ("uv.lock", state.get("uv_lock_sha256"), frozen.get("uv_lock_sha256")),
            ("script", state.get("script"), frozen.get("script")),
            ("sources", state.get("sources"), frozen.get("sources")),
        ):
            _require(
                actual == expected,
                f"코드 상태({label})가 precommit과 다르다: {actual!r} != {expected!r}",
            )
        return record

    # ---------------------------------------------------------- 동사 2: 실행
    def run(self, *, workers: int, threads: int, poll_seconds: float = 10.0) -> None:
        precommit = self.open_precommit()
        payload = precommit.payload
        jobs: list[Job] = []
        if self.spec.selfcheck.replays:
            _require(
                self.script is not None,
                "run은 subprocess 재호출용 스펙 스크립트 경로가 필요하다.",
            )
            for fold in sorted(self.spec.selfcheck.expected):
                jobs.append(
                    Job(
                        f"selfcheck-fold-{fold}",
                        (sys.executable, str(self.script), "selfcheck", "--fold", str(fold)),
                        self.run_dir / "selfcheck" / f"fold-{fold}.json",
                    )
                )
        else:
            self._write_hash_identity_selfcheck(precommit)
        fold_jobs = [
            (arm, fold)
            for arm in payload["candidates"]
            for fold in payload["outer_folds"]
            if not self._fold_record_path(arm, fold).is_file()
        ]
        if fold_jobs:
            _require(
                self.script is not None,
                "run은 subprocess 재호출용 스펙 스크립트 경로가 필요하다.",
            )
        for arm, fold in fold_jobs:
            jobs.append(
                Job(
                    f"{arm}-fold-{fold}",
                    (
                        sys.executable,
                        str(self.script),
                        "fold",
                        "--arm",
                        arm,
                        "--fold",
                        str(fold),
                    ),
                    self._fold_record_path(arm, fold),
                )
            )
        try:
            run_jobs(
                jobs,
                workers=workers,
                threads=threads,
                log_dir=self.run_dir / "logs",
                poll_seconds=poll_seconds,
            )
        except JobsFailed as exc:
            raise JudgmentError(
                f"분할 작업이 실패했다: {', '.join(exc.tags)}. 로그는 {self.run_dir / 'logs'}에 있다."
            ) from exc

    def _write_hash_identity_selfcheck(self, precommit: SealedRecord) -> None:
        path = self.run_dir / "selfcheck" / "selfcheck.json"
        if path.is_file():
            existing = SealedRecord.open(path, schema=f"{self.spec.contract}/selfcheck/1")
            _require(
                existing.payload["parent_sealed_sha256"] == precommit.sealed_sha256,
                "자기 검사 기록이 다른 precommit에서 나왔다.",
            )
            return
        payload = precommit.payload
        record = precommit.derive(
            "selfcheck",
            {
                "grade": SELF_CHECK_HASH_IDENTITY,
                "source": self.spec.selfcheck.source,
                "verified_inputs": {
                    label: entry["sha256"] for label, entry in payload["inputs"].items()
                },
                "verified_caches": {
                    name: arm["cache_sha256"] for name, arm in payload["arms"].items()
                },
                "matches": True,
                "finished_at": _now_iso(),
            },
        )
        _write_document(path, record)
        print("자기 검사(해시 동일성) 기록: 입력·캐시 해시가 precommit과 같다.", flush=True)

    # -------------------------------------------------- 분할·자기 검사 작업자
    def _evaluate_fold(
        self, payload: Mapping, arm: str, fold: int
    ) -> tuple[ensemble.FoldOutcome, int, float]:
        fold_of, y = self._load_fold_and_labels()
        _require(fold in payload["outer_folds"], f"알 수 없는 분할이다: {fold}")
        matrix = self._load_cached_matrix(payload, arm, fold_of)
        started = time.monotonic()
        try:
            outcome = evaluate_outer_fold(self._combiner(fold_of), matrix, fold_of, y, fold)
        except CombinerConvergenceError as exc:
            raise JudgmentError(f"{arm} 분할 {fold}이 수렴하지 않았다: {exc}") from exc
        return outcome, int((fold_of == fold).sum()), time.monotonic() - started

    def fold_job(self, arm: str, fold: int) -> None:
        precommit = self.open_precommit()
        payload = precommit.payload
        _require(arm in payload["candidates"], f"알 수 없는 평가 팔이다: {arm}")
        record_path = self._fold_record_path(arm, fold)
        _require(not record_path.exists(), f"이미 완료된 분할이다: {arm} 분할 {fold}")
        outcome, rows, elapsed = self._evaluate_fold(payload, arm, fold)
        out_dir = self._fold_dir(arm, fold)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {ID: outcome.prediction.index, "prediction": outcome.prediction.to_numpy()}
        ).to_parquet(out_dir / "predictions.parquet")
        record = precommit.derive(
            "fold",
            {
                "arm": arm,
                "fold": fold,
                "rows": rows,
                "auc": outcome.auc,
                "prediction_sha256": outcome.prediction_identity,
                "summary": {str(k): float(v) for k, v in outcome.summary.items()},
                "diagnostics": {str(k): float(v) for k, v in outcome.diagnostics.items()},
                "elapsed_seconds": elapsed,
                "finished_at": _now_iso(),
            },
        )
        _write_document(record_path, record)
        print(f"{arm} 분할 {fold}: AUC {outcome.auc:.15f}, {elapsed:.0f}초", flush=True)

    def selfcheck_job(self, fold: int) -> None:
        precommit = self.open_precommit()
        payload = precommit.payload
        selfcheck = self.spec.selfcheck
        _require(selfcheck.replays, "해시 동일성 등급에는 재현 작업이 없다.")
        _require(
            fold in selfcheck.expected,
            f"자기 검사 기대값에 없는 분할이다: {fold}",
        )
        path = self.run_dir / "selfcheck" / f"fold-{fold}.json"
        _require(not path.exists(), f"이미 완료된 자기 검사다: 분할 {fold}")
        reference = payload["reference"]["name"]
        outcome, rows, elapsed = self._evaluate_fold(payload, reference, fold)
        expected = selfcheck.expected[fold]
        auc_matches = outcome.auc == expected.auc
        hash_matches = (
            expected.prediction_sha256 is None
            or outcome.prediction_identity == expected.prediction_sha256
        )
        matches = bool(auc_matches and hash_matches)
        record = precommit.derive(
            "selfcheck",
            {
                "grade": selfcheck.grade,
                "source": selfcheck.source,
                "fold": fold,
                "rows": rows,
                "expected": {
                    "auc": expected.auc,
                    "prediction_sha256": expected.prediction_sha256,
                },
                "actual": {
                    "auc": outcome.auc,
                    "prediction_sha256": outcome.prediction_identity,
                },
                "matches": matches,
                "elapsed_seconds": elapsed,
                "finished_at": _now_iso(),
            },
        )
        _write_document(path, record)
        _require(
            matches,
            f"자기 검사 실패: 분할 {fold}의 재현이 기준값({selfcheck.source})과 다르다.",
        )
        print(f"자기 검사 통과: 분할 {fold} AUC {outcome.auc:.15f}", flush=True)

    # ---------------------------------------------------------- 동사 3: 비교
    def _verify_selfcheck(self, precommit: SealedRecord) -> dict[str, object]:
        frozen = precommit.payload["selfcheck"]
        _require(
            frozen["grade"] == self.spec.selfcheck.grade
            and frozen["source"] == self.spec.selfcheck.source,
            "스펙의 자기 검사 선언이 precommit에 동결된 것과 다르다.",
        )
        schema = f"{self.spec.contract}/selfcheck/1"
        results = []
        for path in self._selfcheck_paths():
            _require(path.is_file(), f"자기 검사 기록이 없다: {path}. run을 먼저 완료한다.")
            record = SealedRecord.open(path, schema=schema)
            _require(
                record.payload["parent_sealed_sha256"] == precommit.sealed_sha256,
                f"자기 검사 기록이 다른 precommit에서 나왔다: {path}",
            )
            _require(record.payload["matches"] is True, f"자기 검사가 실패 상태다: {path}")
            results.append(dict(record.payload))
        return {
            "grade": self.spec.selfcheck.grade,
            "source": self.spec.selfcheck.source,
            "passed": True,
            "records": len(results),
        }

    def _collect_arm(
        self,
        precommit: SealedRecord,
        arm: str,
        fold_of: pd.Series,
        y: pd.Series,
    ) -> tuple[FoldScores, dict[int, Mapping[str, object]], str]:
        payload = precommit.payload
        schema = f"{self.spec.contract}/fold/1"
        nested = pd.Series(np.nan, index=fold_of.index, dtype=np.float64)
        records: dict[int, Mapping[str, object]] = {}
        for fold in payload["outer_folds"]:
            record_path = self._fold_record_path(arm, fold)
            _require(record_path.is_file(), f"{arm} 분할 {fold} 기록이 없다. run을 먼저 완료한다.")
            record = SealedRecord.open(record_path, schema=schema)
            body = record.payload
            _require(
                body["parent_sealed_sha256"] == precommit.sealed_sha256,
                f"{arm} 분할 {fold} 기록이 다른 precommit에서 나왔다.",
            )
            _require(
                body["arm"] == arm and int(body["fold"]) == fold,
                f"{arm} 분할 {fold} 기록의 팔·분할 표기가 어긋난다.",
            )
            part = (
                pd.read_parquet(self._fold_dir(arm, fold) / "predictions.parquet")
                .set_index(ID)["prediction"]
            )
            ids = fold_of.index[(fold_of == fold).to_numpy()]
            _require(
                part.index.equals(pd.Index(ids)),
                f"{arm} 분할 {fold} 예측 id가 고정 분할과 다르다.",
            )
            _require(
                array_identity(part.to_numpy()) == body["prediction_sha256"],
                f"{arm} 분할 {fold} 예측 해시가 기록과 다르다.",
            )
            _require(
                float(roc_auc_score(y.loc[ids].to_numpy(), part.to_numpy())) == body["auc"],
                f"{arm} 분할 {fold} AUC 재계산이 기록과 다르다.",
            )
            nested.loc[ids] = part.to_numpy()
            records[fold] = body
        _require(bool(nested.notna().all()), f"{arm}: 이어붙인 예측에 빈 행이 있다.")
        scores = FoldScores(
            nested_auc=float(roc_auc_score(y.to_numpy(), nested.to_numpy())),
            fold_aucs={fold: float(body["auc"]) for fold, body in records.items()},
        )
        return scores, records, array_identity(nested.to_numpy())

    def compare(self) -> SealedRecord:
        precommit = self.open_precommit()
        payload = precommit.payload
        selfcheck_summary = self._verify_selfcheck(precommit)
        fold_of, y = self._load_fold_and_labels()
        gate = StackGate(
            delta_required=float(payload["gate"]["delta_required"]),
            folds_required_positive=int(payload["gate"]["folds_required_positive"]),
        )
        _require(
            gate == self.spec.gate,
            "스펙의 게이트가 precommit에 동결된 게이트와 다르다.",
        )
        reference_scores = FoldScores(
            nested_auc=float(payload["reference"]["nested_auc"]),
            fold_aucs={
                int(fold): float(auc)
                for fold, auc in payload["reference"]["fold_aucs"].items()
            },
        )
        verdicts: dict[str, StackVerdict] = {}
        arms_payload: dict[str, dict[str, object]] = {}
        elapsed_total = 0.0
        for arm in payload["candidates"]:
            scores, records, nested_sha = self._collect_arm(precommit, arm, fold_of, y)
            verdict = judge_stack_replacement(scores, reference_scores, gate)
            verdicts[arm] = verdict
            elapsed_total += sum(float(body["elapsed_seconds"]) for body in records.values())
            arms_payload[arm] = {
                "member_count": payload["arms"][arm]["member_count"],
                "composition_sha256": payload["arms"][arm]["composition_sha256"],
                "nested_auc": scores.nested_auc,
                "fold_aucs": {str(fold): auc for fold, auc in sorted(scores.fold_aucs.items())},
                "prediction_sha256": nested_sha,
                "verdict": verdict.to_record(),
            }
        record = precommit.derive(
            "comparison",
            {
                "round_id": self.spec.round_id,
                "reference": dict(payload["reference"]),
                "selfcheck": selfcheck_summary,
                "arms": arms_payload,
                "rows_scored": int(len(y)),
                "elapsed_seconds_total": elapsed_total,
                "compared_at": _now_iso(),
            },
        )
        _write_document(self.run_dir / "comparison.json", record)
        for arm, verdict in verdicts.items():
            verdict.require_decidable()
            print(
                f"{arm}: nested {arms_payload[arm]['nested_auc']:.10f}, "
                f"기준 {reference_scores.nested_auc:.10f}, 차이 {verdict.delta:+.7f}, "
                f"분할 양수 {verdict.folds_positive}/{len(verdict.fold_deltas)} "
                f"→ {'통과' if verdict.passes_gate else '미달'}"
            )
        return record

    # ---------------------------------------------------------- 동사 4: 보고
    def _open_comparison(self, precommit: SealedRecord) -> SealedRecord:
        record = SealedRecord.open(
            self.run_dir / "comparison.json", schema=f"{self.spec.contract}/comparison/1"
        )
        _require(
            record.payload["parent_sealed_sha256"] == precommit.sealed_sha256,
            "comparison 기록이 다른 precommit에서 나왔다.",
        )
        return record

    def report(self) -> Path:
        precommit = self.open_precommit()
        comparison = self._open_comparison(precommit)
        payload = precommit.payload
        body = comparison.payload
        folds: dict[str, dict[int, Mapping[str, object]]] = {}
        fold_schema = f"{self.spec.contract}/fold/1"
        for arm in payload["candidates"]:
            folds[arm] = {
                fold: SealedRecord.open(
                    self._fold_record_path(arm, fold), schema=fold_schema
                ).payload
                for fold in payload["outer_folds"]
            }
        reference = payload["reference"]
        gate = payload["gate"]
        lines = [f"# {self.spec.title}", "", self.spec.question, ""]
        lines += ["## 판정", ""]
        for arm in payload["candidates"]:
            verdict = body["arms"][arm]["verdict"]
            _require(
                verdict["delta"] is not None,
                f"{arm}: 판정 불가 상태의 comparison 기록으로는 보고서를 만들 수 없다.",
            )
            lines.append(
                f"- {arm}: **{'통과' if verdict['passes_gate'] else '미달'}** — "
                f"nested `{body['arms'][arm]['nested_auc']:.10f}`에서 기준 "
                f"`{reference['nested_auc']:.10f}`을 뺀 차이 `{verdict['delta']:+.7f}` "
                f"(문턱 `+{gate['delta_required']:.5f}`, 여유 `{verdict['delta_minus_gate']:+.7f}`), "
                f"분할 양수 `{verdict['folds_positive']}/{len(payload['outer_folds'])}` "
                f"(요구 `{gate['folds_required_positive']}`)."
            )
        lines += [
            f"- 기준 팔 {reference['name']}의 기준값 출처: {reference['values_source']}.",
            "",
        ]
        lines += ["## 분할별 결과", ""]
        for arm in payload["candidates"]:
            arm_folds = folds[arm]
            diagnostic_keys = sorted(
                {key for record in arm_folds.values() for key in record["diagnostics"]}
            )
            header = ["분할", "기준 AUC", "후보 AUC", "차이", *diagnostic_keys]
            lines += [f"### {arm}", ""]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + " ---: |" * len(header))
            for fold in payload["outer_folds"]:
                record = arm_folds[fold]
                base_auc = float(reference["fold_aucs"][str(fold)])
                diagnostics = record["diagnostics"]
                cells = [
                    str(fold),
                    f"{base_auc:.10f}",
                    f"{record['auc']:.10f}",
                    f"{record['auc'] - base_auc:+.10f}",
                    *(
                        "" if key not in diagnostics else f"{diagnostics[key]:.6g}"
                        for key in diagnostic_keys
                    ),
                ]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        records = RoundRecords(
            spec=self.spec, precommit=precommit, comparison=comparison, folds=folds
        )
        for section in self.spec.report_sections:
            lines += [f"## {section.title}", "", *section.render(records), ""]
        env = payload["environment"]
        lines += ["## 동결과 재현성", ""]
        lines.append(f"- precommit 봉인 해시는 `{precommit.sealed_sha256}`다.")
        for name, arm in payload["arms"].items():
            lines.append(
                f"- {name}({arm['role']}) 구성원 {arm['member_count']}개, "
                f"구성 해시 `{arm['composition_sha256']}`."
            )
        lines.append(
            f"- 자기 검사 등급은 {body['selfcheck']['grade']}이고 기준값 출처는 "
            f"{body['selfcheck']['source']}이며 통과했다."
        )
        lines.append(
            f"- 실행 환경은 {env['platform']} ({env['machine']}), CPU {env['cpu_count']}개, "
            f"Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, "
            f"scikit-learn {env['sklearn']}다."
        )
        lines.append(
            f"- 분할 작업 경과 시간 합계는 {body['elapsed_seconds_total'] / 60:.1f}분이다."
        )
        lines.append("")
        report_path = self.run_dir / "report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"보고 저장: {report_path}")
        return report_path

    # ---------------------------------------------------------- 동사 5: 게시
    def _publish_files(self, payload: Mapping) -> list[Path]:
        relative = [Path("precommit.json")]
        relative += [path.relative_to(self.run_dir) for path in self._selfcheck_paths()]
        relative += [
            self._fold_record_path(arm, fold).relative_to(self.run_dir)
            for arm in payload["candidates"]
            for fold in payload["outer_folds"]
        ]
        relative += [Path("comparison.json"), Path("report.md")]
        relative += [Path(extra) for extra in self.spec.publish_extra]
        for path in relative:
            _require((self.run_dir / path).is_file(), f"게시할 파일이 없다: {self.run_dir / path}")
        return relative

    def publish(self) -> Path:
        precommit = self.open_precommit()
        self._open_comparison(precommit)
        files = self._publish_files(precommit.payload)
        _require(
            not self.publish_dir.exists(),
            f"게시 폴더가 이미 있다: {self.publish_dir}",
        )
        for relative in files:
            destination = self.publish_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.run_dir / relative, destination)
        published = [
            path for path in sorted(self.publish_dir.rglob("*")) if path.is_file()
        ]
        manifest_path = self.publish_dir / "manifest.sha256"
        manifest_path.write_text(
            "\n".join(
                f"{file_identity(path)}  {path.relative_to(self.publish_dir)}"
                for path in published
            )
            + "\n",
            encoding="utf-8",
        )
        listed = {
            line.split("  ", 1)[1]
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
        }
        actual = {
            str(path.relative_to(self.publish_dir))
            for path in self.publish_dir.rglob("*")
            if path.is_file() and path != manifest_path
        }
        _require(
            listed == actual,
            f"manifest 자기 대조 실패: 목록 {sorted(listed)} != 실제 {sorted(actual)}",
        )
        print(f"판정 근거 게시: {self.publish_dir} ({len(published)}개 파일 + manifest)")
        return self.publish_dir


def _translate(exc: Exception) -> str:
    message = str(exc)
    return message if message.startswith("판정 불가") else f"판정 불가: {message}"


def main(
    spec: RoundSpec,
    *,
    store_factory: Callable[[], RunStore] | None = None,
    argv: list[str] | None = None,
    root: Path = Path("."),
    code_state: Callable[[], dict[str, object]] | None = None,
) -> None:
    """스펙 스크립트의 진입점. 동사 5개와 작업자 2개(fold·selfcheck)를 노출한다."""
    parser = argparse.ArgumentParser(
        description=f"판정 회차 {spec.round_id}: {spec.title}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("precommit")
    run = sub.add_parser("run")
    run.add_argument("--workers", type=int, default=3)
    run.add_argument("--threads", type=int, default=4)
    fold = sub.add_parser("fold")
    fold.add_argument("--arm", required=True)
    fold.add_argument("--fold", type=int, required=True)
    selfcheck = sub.add_parser("selfcheck")
    selfcheck.add_argument("--fold", type=int, required=True)
    for name in ("compare", "report", "publish"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    script = Path(sys.argv[0])
    store: RunStore | None = None
    if args.command == "precommit":
        if store_factory is None:
            from .runs import MlflowRunStore

            store_factory = MlflowRunStore
        store = store_factory()
    round_ = JudgmentRound(
        spec, store=store, root=root, script=script, code_state=code_state
    )
    try:
        if args.command == "precommit":
            round_.precommit()
        elif args.command == "run":
            round_.run(workers=args.workers, threads=args.threads)
        elif args.command == "fold":
            round_.fold_job(args.arm, args.fold)
        elif args.command == "selfcheck":
            round_.selfcheck_job(args.fold)
        elif args.command == "compare":
            round_.compare()
        elif args.command == "report":
            round_.report()
        elif args.command == "publish":
            round_.publish()
    except (JudgmentError, SealedRecordError, MembersError, JobsError, RunStoreError) as exc:
        sys.exit(_translate(exc))
    except json.JSONDecodeError as exc:
        sys.exit(_translate(exc))
