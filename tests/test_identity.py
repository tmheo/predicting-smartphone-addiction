"""pipeline.identity: 예측 신원의 계약 시험과 golden 시험. (#529)

golden 시험은 커밋된 실제 동결 명세(ecf-v3-b18bc301d500)의 후보 하나로
기록된 oof_sha256·test_sha256·pair_sha256을 배열 파일에서 재현한다.
배열 파일(data/external/)은 gitignore 대상이라, 없는 체크아웃에서는 skip한다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.identity import array_identity, file_identity, integer_identity, pair_identity

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_SPEC = REPO_ROOT / "docs/research/external-candidate-freeze/ecf-v3-b18bc301d500.json"


def test_array_identity_fixed_digest():
    values = np.array([0.1, 0.2, 0.3])
    expected = hashlib.sha256(values.astype("<f8").tobytes(order="C")).hexdigest()
    assert array_identity(values) == expected


def test_array_identity_series_ndarray_equal():
    values = np.array([0.5, np.nan, -1.25e-7])
    assert array_identity(pd.Series(values)) == array_identity(values)


def test_array_identity_is_value_hash_not_layout_hash():
    values = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert array_identity(values) == array_identity(np.asfortranarray(values))


def test_integer_identity_fixed_digest():
    values = np.array([1, 2, 3])
    expected = hashlib.sha256(values.astype("<i8").tobytes(order="C")).hexdigest()
    assert integer_identity(values) == expected
    assert integer_identity(pd.Series(values)) == expected


def test_pair_identity_matches_concatenation_and_is_order_sensitive():
    oof = np.array([0.1, 0.2])
    test = np.array([0.3])
    joined = hashlib.sha256(
        oof.astype("<f8").tobytes() + test.astype("<f8").tobytes()
    ).hexdigest()
    assert pair_identity(oof, test) == joined
    assert pair_identity(test, oof) != joined
    # 무구분자 연결이므로 경계만 옮긴 쌍은 같은 신원이다(기존 기록 규약).
    assert pair_identity(np.array([0.1]), np.array([0.2, 0.3])) == pair_identity(
        np.array([0.1, 0.2]), np.array([0.3])
    )


def test_file_identity_matches_bytes(tmp_path):
    payload = b"identity" * 300_000  # 1MiB 청크 경계를 넘긴다.
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert file_identity(path) == hashlib.sha256(payload).hexdigest()


def test_golden_freeze_spec_candidate_hashes_reproduced():
    candidate = json.loads(FREEZE_SPEC.read_text())["candidates"][0]
    oof_path = REPO_ROOT / candidate["oof_path"]
    test_path = REPO_ROOT / candidate["test_path"]
    if not (oof_path.is_file() and test_path.is_file()):
        pytest.skip("동결 명세의 배열 파일이 이 체크아웃에 없다(data/external은 gitignore).")
    oof = np.load(oof_path)
    test = np.load(test_path)
    assert array_identity(oof) == candidate["oof_sha256"]
    assert array_identity(test) == candidate["test_sha256"]
    assert pair_identity(oof, test) == candidate["pair_sha256"]
