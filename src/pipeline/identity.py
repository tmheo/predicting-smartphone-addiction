"""예측 신원(CONTEXT.md 용어)의 정본 구현. (#529)

예측 배열의 값 내용에서 계산해, 같은 값이면 저장 형식과 무관하게 같아지는 의미 해시를 만든다.
규약은 명시적 little-endian(`<f8`, `<i8`)과 C 순서 contiguous 바이트의 SHA-256이며,
기존 4개 재구현(pool_audit, freeze_external_candidates, freeze_reusable_own_candidates,
build_external_member_ledger_v3)이 little-endian 장비에서 남긴 모든 기록과 digest가 같다.

정책 없음: NaN·유한성 검사는 하지 않는다(유한성은 구성원 행렬의 관문, #531 소관).
canonical JSON 해시는 봉인 기록(#530) 소관이다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _float_bytes(values: pd.Series | np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes(order="C")


def array_identity(values: pd.Series | np.ndarray) -> str:
    """float 배열의 의미 해시."""
    return hashlib.sha256(_float_bytes(values)).hexdigest()


def integer_identity(values: pd.Series | np.ndarray) -> str:
    """정수 배열(`<i8`)의 의미 해시. id·fold 배정 식별용."""
    payload = np.ascontiguousarray(np.asarray(values, dtype="<i8")).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def pair_identity(oof: pd.Series | np.ndarray, test: pd.Series | np.ndarray) -> str:
    """OOF·시험 예측 쌍의 신원. 기존 기록 호환을 위해 oof, test 순서로 구분자 없이 잇는다."""
    digest = hashlib.sha256()
    digest.update(_float_bytes(oof))
    digest.update(_float_bytes(test))
    return digest.hexdigest()


def file_identity(path: Path) -> str:
    """파일 바이트 해시. 예측 신원과 구분되는 계보 기록용."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
