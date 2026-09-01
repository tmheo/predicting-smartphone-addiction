"""구성원 행렬(CONTEXT.md 용어)의 적재와 무결성 검증 정본. (#531)

"(config, run_id)나 파일 경로로 OOF·시험 예측을 해시 대조하며 읽는다"가
judge 스크립트 5곳에 재구현돼 있었고 최근 fix 커밋이 전부 그 복사본에서 터졌다.
이 module이 적재와 검증을 한 곳으로 모은다. 파일 형식 해석은 member_sources의
얇은 adapter 소관이고, 여기서는 일반 스키마(MemberSpec/MemberSource)와 검증만 소유한다.

불변식:
- 열 순서 = 출처가 준 동결 순서. 등록 결합기가 행·열 순서에 민감하다는 교훈(#486)을
  이 module이 소유한다.
- 유한성 관문: 비유한값이 있으면 적재를 거부한다(identity는 순수 바이트 해시라 여기서 막는다).
- 부분 성공 없음: 무결성 위반은 예외로 끝난다. "통과한 구성원만으로 계속"을
  interface가 열어주지 않는다.

검증 수준 3단계(출처가 최소 수준을 선언하고, 실제 달성 수준을 구성원별로 기록한다):
- hash-verified: 예측 신원(oof_sha256·test_sha256·pair_sha256, #529) 대조까지 통과.
- auc-verified: 해시 없이 장부 AUC 재채점 대조(허용 오차 1e-9)만 통과.
- identity-only: (config, run_id) 신원만.
판정 caller는 require(HASH_VERIFIED)를 쓴다. 조립·진단 같은 비판정 용도는 낮은 수준을 허용한다.

오류 모형: 무결성 위반은 MemberIntegrityError(member, check, expected, actual)로,
출처 선언 자체의 계약 위반은 MemberSourceInvalid로 던진다. 둘 다 MembersError의
하위이므로 CLI는 하나만 잡아 판정 불가로 번역한다(runs.py의 RunNotFound 관례).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .identity import array_identity, pair_identity
from .runs import RunStore

HASH_VERIFIED = "hash-verified"
AUC_VERIFIED = "auc-verified"
IDENTITY_ONLY = "identity-only"
_LEVEL_RANK = {IDENTITY_ONLY: 0, AUC_VERIFIED: 1, HASH_VERIFIED: 2}

# 장부 AUC 재채점 허용 오차. judge_strict_external_selection 이래의 관례다.
AUC_TOLERANCE = 1e-9


class MembersError(Exception):
    """구성원 행렬 오류의 공통 뿌리. CLI는 이것 하나만 잡아 판정 불가로 번역한다."""


class MemberSourceInvalid(MembersError):
    """출처 선언이나 호출 계약 자체가 어긋나 적재를 시작할 수 없다."""


class MemberIntegrityError(MembersError):
    """구성원 하나의 무결성 검사 실패."""

    def __init__(self, member: str, check: str, expected: object, actual: object) -> None:
        self.member = member
        self.check = check
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"구성원 {member}: {check} 검사 실패 (기대 {expected!r}, 실제 {actual!r})"
        )


@dataclass(frozen=True)
class MemberSpec:
    """출처가 선언하는 구성원 한 명의 신원과 검증 근거.

    OOF의 출처는 run_id(RunStore 경유, mlruns 직접 접근 금지)와 oof_path 중 정확히 하나다.
    경로는 장부 표기(`경로`, `경로[열]`은 parquet 열, `경로[:, i]`는 npy 행렬 열)를 따른다.
    """

    member_id: str
    origin: str
    verification: str
    run_id: str | None = None
    oof_path: str | None = None
    test_path: str | None = None
    oof_sha256: str | None = None
    test_sha256: str | None = None
    pair_sha256: str | None = None
    expected_auc: float | None = None

    @property
    def declared_hashes(self) -> bool:
        return any((self.oof_sha256, self.test_sha256, self.pair_sha256))


@dataclass(frozen=True)
class MemberSource:
    """동결 순서의 구성원 선언 목록. adapter가 파일 형식 해석을 끝낸 뒤의 일반 형태다."""

    name: str
    members: tuple[MemberSpec, ...]
    train_rows: int | None = None
    test_rows: int | None = None

    def __post_init__(self) -> None:
        if not self.members:
            raise MemberSourceInvalid(f"{self.name}: 구성원이 없다.")
        ids = [spec.member_id for spec in self.members]
        if len(set(ids)) != len(ids):
            duplicated = sorted({i for i in ids if ids.count(i) > 1})
            raise MemberSourceInvalid(f"{self.name}: member_id가 중복된다: {duplicated}")
        with_test = {spec.member_id for spec in self.members if spec.test_path}
        if with_test and len(with_test) != len(self.members):
            raise MemberSourceInvalid(
                f"{self.name}: 시험 예측은 전원이 갖거나 전원이 없어야 한다."
            )
        for spec in self.members:
            if (spec.run_id is None) == (spec.oof_path is None):
                raise MemberSourceInvalid(
                    f"{self.name}/{spec.member_id}: OOF 출처는 run_id와 oof_path 중 하나여야 한다."
                )
            if spec.verification not in _LEVEL_RANK:
                raise MemberSourceInvalid(
                    f"{self.name}/{spec.member_id}: 검증 수준 {spec.verification!r}은 없다."
                )
            if spec.verification == HASH_VERIFIED and not spec.declared_hashes:
                raise MemberSourceInvalid(
                    f"{self.name}/{spec.member_id}: hash-verified 선언에 해시 근거가 없다."
                )
            if spec.verification == AUC_VERIFIED and spec.expected_auc is None:
                raise MemberSourceInvalid(
                    f"{self.name}/{spec.member_id}: auc-verified 선언에 expected_auc가 없다."
                )
            if (spec.test_sha256 or spec.pair_sha256) and spec.test_path is None:
                raise MemberSourceInvalid(
                    f"{self.name}/{spec.member_id}: 시험 해시가 있는데 test_path가 없다."
                )


@dataclass(frozen=True)
class MemberMatrix:
    """열 순서가 출처 동결 순서인 구성원 행렬.

    oof·test는 float64 2차원 배열이고 members가 구성원 메타 표
    (member_id, origin, verification=달성 수준, 예측 신원 3종, rescored_auc)다.
    """

    oof: np.ndarray
    test: np.ndarray | None
    index: pd.Index
    members: pd.DataFrame

    @property
    def member_ids(self) -> list[str]:
        return list(self.members["member_id"])

    def oof_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.oof, index=self.index, columns=self.member_ids)

    def test_frame(self) -> pd.DataFrame:
        if self.test is None:
            raise MembersError("이 출처는 시험 예측을 선언하지 않았다.")
        return pd.DataFrame(self.test, columns=self.member_ids)

    def require(self, level: str) -> None:
        """모든 구성원의 달성 검증 수준이 level 이상임을 강제한다. 판정 caller는 hash-verified."""
        if level not in _LEVEL_RANK:
            raise MembersError(f"검증 수준 {level!r}은 없다.")
        achieved = self.members["verification"]
        below = self.members[achieved.map(_LEVEL_RANK) < _LEVEL_RANK[level]]
        if len(below):
            first = below.iloc[0]
            raise MemberIntegrityError(
                f"{first['member_id']} 외 {len(below) - 1}명" if len(below) > 1 else first["member_id"],
                "verification_level",
                level,
                first["verification"],
            )


_PATH_SPEC = re.compile(r"(.+?)\[(.+)\]")
_NPY_COLUMN = re.compile(r":\s*,\s*(\d+)")


def _load_path_array(spec: str, member: str, kind: str) -> np.ndarray:
    """장부 경로 표기의 예측 배열을 1차원 float64로 읽는다."""
    match = _PATH_SPEC.fullmatch(spec)
    path = Path(match.group(1) if match else spec)
    if not path.is_file():
        raise MemberIntegrityError(member, f"{kind}_file", str(path), "없음")
    if match is None:
        values = np.load(path)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.reshape(-1)
        if values.ndim != 1:
            raise MemberIntegrityError(member, f"{kind}_shape", "1차원", values.shape)
        return values.astype(np.float64)
    selector = match.group(2)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=[selector])
        return frame[selector].to_numpy(np.float64)
    column = _NPY_COLUMN.fullmatch(selector)
    if column is None:
        raise MemberIntegrityError(member, f"{kind}_path", "`경로[열]` 또는 `경로[:, i]`", spec)
    return np.load(path)[:, int(column.group(1))].astype(np.float64)


def _resolve_oof(spec: MemberSpec, index: pd.Index, store: RunStore) -> np.ndarray:
    if spec.run_id is not None:
        pred = store.oof_of(spec.run_id)
        if not pred.index.is_unique:
            raise MemberIntegrityError(
                spec.member_id, "oof_id_unique", 0, int(pred.index.duplicated().sum())
            )
        missing = index.difference(pred.index)
        if len(missing):
            raise MemberIntegrityError(spec.member_id, "oof_ids", 0, len(missing))
        return pred.reindex(index).to_numpy(np.float64)
    values = _load_path_array(spec.oof_path, spec.member_id, "oof")
    if len(values) != len(index):
        raise MemberIntegrityError(spec.member_id, "oof_rows", len(index), len(values))
    return values


def _require_finite(values: np.ndarray, member: str, check: str) -> None:
    nonfinite = int((~np.isfinite(values)).sum())
    if nonfinite:
        raise MemberIntegrityError(member, check, 0, nonfinite)


def _check_hash(expected: str | None, actual: str, member: str, check: str) -> bool:
    if expected is None:
        return False
    if actual != expected:
        raise MemberIntegrityError(member, check, expected, actual)
    return True


def load_members(
    source: MemberSource,
    index: pd.Index,
    store: RunStore,
    *,
    labels: pd.Series | None = None,
) -> MemberMatrix:
    """출처 동결 순서대로 구성원 예측을 적재하고 무결성을 검증한다.

    labels(id 순서가 index와 같은 라벨)를 주면 구성원마다 AUC를 재채점하고
    expected_auc가 선언된 구성원은 허용 오차 1e-9로 대조한다.
    auc-verified를 선언한 출처는 labels 없이 적재할 수 없다.
    """
    if not index.is_unique:
        raise MemberSourceInvalid("기준 index의 id가 중복된다.")
    if source.train_rows is not None and len(index) != source.train_rows:
        raise MemberSourceInvalid(
            f"{source.name}: 행 계약 train_rows={source.train_rows}와 index 길이 {len(index)}가 다르다."
        )
    if labels is not None and not labels.index.equals(index):
        raise MemberSourceInvalid("labels의 id 순서가 기준 index와 다르다.")
    label_values = None if labels is None else labels.to_numpy()

    has_test = source.members[0].test_path is not None
    oof_columns: list[np.ndarray] = []
    test_columns: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    test_rows = source.test_rows
    for spec in source.members:
        oof = _resolve_oof(spec, index, store)
        _require_finite(oof, spec.member_id, "oof_finite")
        oof_digest = array_identity(oof)
        test = test_digest = pair_digest = None
        if has_test:
            test = _load_path_array(spec.test_path, spec.member_id, "test")
            if test_rows is None:
                test_rows = len(test)
            if len(test) != test_rows:
                raise MemberIntegrityError(spec.member_id, "test_rows", test_rows, len(test))
            _require_finite(test, spec.member_id, "test_finite")
            test_digest = array_identity(test)
            pair_digest = pair_identity(oof, test)

        hash_checked = _check_hash(spec.oof_sha256, oof_digest, spec.member_id, "oof_sha256")
        if has_test:
            hash_checked |= _check_hash(
                spec.test_sha256, test_digest, spec.member_id, "test_sha256"
            )
            hash_checked |= _check_hash(
                spec.pair_sha256, pair_digest, spec.member_id, "pair_sha256"
            )

        rescored = None
        if label_values is not None:
            rescored = float(roc_auc_score(label_values, oof))
            if spec.expected_auc is not None and abs(rescored - spec.expected_auc) > AUC_TOLERANCE:
                raise MemberIntegrityError(
                    spec.member_id, "ledger_auc", spec.expected_auc, rescored
                )

        if hash_checked:
            achieved = HASH_VERIFIED
        elif spec.expected_auc is not None and rescored is not None:
            achieved = AUC_VERIFIED
        else:
            achieved = IDENTITY_ONLY
        if _LEVEL_RANK[achieved] < _LEVEL_RANK[spec.verification]:
            raise MemberSourceInvalid(
                f"{source.name}/{spec.member_id}: 선언 수준 {spec.verification}에 필요한 "
                f"검증을 수행하지 못했다(달성 {achieved}). auc-verified는 labels가 필요하다."
            )

        oof_columns.append(oof)
        if has_test:
            test_columns.append(test)
        rows.append(
            {
                "member_id": spec.member_id,
                "origin": spec.origin,
                "verification": achieved,
                "oof_sha256": oof_digest,
                "test_sha256": test_digest,
                "pair_sha256": pair_digest,
                "rescored_auc": rescored,
            }
        )

    return MemberMatrix(
        oof=np.column_stack(oof_columns),
        test=np.column_stack(test_columns) if has_test else None,
        index=index,
        members=pd.DataFrame(rows),
    )
