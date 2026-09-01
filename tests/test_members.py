"""pipeline.members: 구성원 행렬의 계약 시험과 golden 시험. (#531)

계약 시험은 InMemoryRunStore와 합성 출처로 열 순서 불변식, 유한성 관문,
각 검증 실패의 예외, 검증 수준 기록을 확인한다.
golden 시험은 커밋된 실제 동결 명세(ecf-v3-b18bc301d500)로 후보를 적재해
기록된 해시와 AUC를 재현한다. 배열 파일(data/)이 없는 체크아웃에서는 skip한다.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline.data import ID, TARGET
from pipeline.identity import array_identity, pair_identity
from pipeline.member_sources import freeze_spec_members, manifest_members, pool_members
from pipeline.members import (
    AUC_VERIFIED,
    HASH_VERIFIED,
    IDENTITY_ONLY,
    MemberIntegrityError,
    MemberSource,
    MemberSourceInvalid,
    MemberSpec,
    load_members,
)
from pipeline.ledger import EntryEvidence, Pool, PoolMember
from pipeline.runs import InMemoryRunStore

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_SPEC = REPO_ROOT / "docs/research/external-candidate-freeze/ecf-v3-b18bc301d500.json"

INDEX = pd.Index([10, 11, 12, 13], name=ID)
LABELS = pd.Series([0, 1, 0, 1], index=INDEX)


def oof_frame(index: pd.Index, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({ID: index.to_numpy(), "pred": np.asarray(values, dtype=np.float64)})


def store_with(**runs: list[float]) -> InMemoryRunStore:
    store = InMemoryRunStore()
    for run_id, values in runs.items():
        store.add_run(run_id, oof=oof_frame(INDEX, values))
    return store


def identity_spec(member_id: str, run_id: str, **overrides) -> MemberSpec:
    return MemberSpec(
        member_id=member_id,
        origin="pool",
        verification=IDENTITY_ONLY,
        run_id=run_id,
        **overrides,
    )


def source_of(*specs: MemberSpec, **overrides) -> MemberSource:
    return MemberSource(name="합성", members=tuple(specs), **overrides)


def test_column_order_follows_source_frozen_order():
    store = store_with(r1=[0.1, 0.2, 0.3, 0.4], r2=[0.5, 0.6, 0.7, 0.8])
    source = source_of(identity_spec("b", "r2"), identity_spec("a", "r1"))
    matrix = load_members(source, INDEX, store)
    assert matrix.member_ids == ["b", "a"]
    assert matrix.oof.dtype == np.float64 and matrix.oof.shape == (4, 2)
    np.testing.assert_array_equal(matrix.oof[:, 0], [0.5, 0.6, 0.7, 0.8])
    assert matrix.oof_frame().columns.tolist() == ["b", "a"]
    assert matrix.test is None


def test_store_oof_is_reindexed_by_id():
    store = InMemoryRunStore()
    shuffled = pd.DataFrame({ID: [13, 10, 12, 11], "pred": [0.4, 0.1, 0.3, 0.2]})
    store.add_run("r1", oof=shuffled)
    matrix = load_members(source_of(identity_spec("a", "r1")), INDEX, store)
    np.testing.assert_array_equal(matrix.oof[:, 0], [0.1, 0.2, 0.3, 0.4])


def test_missing_id_in_store_oof_is_refused():
    store = InMemoryRunStore()
    store.add_run("r1", oof=pd.DataFrame({ID: [10, 11, 12], "pred": [0.1, 0.2, 0.3]}))
    with pytest.raises(MemberIntegrityError) as caught:
        load_members(source_of(identity_spec("a", "r1")), INDEX, store)
    assert caught.value.check == "oof_ids"


def test_nonfinite_oof_is_refused():
    store = store_with(r1=[0.1, np.nan, 0.3, 0.4])
    with pytest.raises(MemberIntegrityError) as caught:
        load_members(source_of(identity_spec("a", "r1")), INDEX, store)
    assert caught.value.check == "oof_finite" and caught.value.actual == 1


def test_oof_hash_mismatch_is_structured(tmp_path):
    values = np.array([0.1, 0.2, 0.3, 0.4])
    path = tmp_path / "oof.npy"
    np.save(path, values)
    spec = MemberSpec(
        member_id="ext",
        origin="external",
        verification=HASH_VERIFIED,
        oof_path=str(path),
        oof_sha256="0" * 64,
    )
    with pytest.raises(MemberIntegrityError) as caught:
        load_members(source_of(spec), INDEX, InMemoryRunStore())
    assert caught.value.member == "ext"
    assert caught.value.check == "oof_sha256"
    assert caught.value.expected == "0" * 64
    assert caught.value.actual == array_identity(values)


def test_hash_verified_records_level_and_checks_pair(tmp_path):
    oof = np.array([0.1, 0.2, 0.3, 0.4])
    test = np.array([0.9, 0.8])
    np.save(tmp_path / "oof.npy", oof)
    np.save(tmp_path / "test.npy", test)
    spec = MemberSpec(
        member_id="ext",
        origin="external",
        verification=HASH_VERIFIED,
        oof_path=str(tmp_path / "oof.npy"),
        test_path=str(tmp_path / "test.npy"),
        oof_sha256=array_identity(oof),
        test_sha256=array_identity(test),
        pair_sha256=pair_identity(oof, test),
    )
    matrix = load_members(source_of(spec), INDEX, InMemoryRunStore())
    row = matrix.members.iloc[0]
    assert row["verification"] == HASH_VERIFIED
    assert row["pair_sha256"] == pair_identity(oof, test)
    np.testing.assert_array_equal(matrix.test_frame()["ext"], test)
    matrix.require(HASH_VERIFIED)


def test_pair_hash_mismatch_is_refused(tmp_path):
    oof = np.array([0.1, 0.2, 0.3, 0.4])
    test = np.array([0.9, 0.8])
    np.save(tmp_path / "oof.npy", oof)
    np.save(tmp_path / "test.npy", test)
    spec = MemberSpec(
        member_id="ext",
        origin="external",
        verification=HASH_VERIFIED,
        oof_path=str(tmp_path / "oof.npy"),
        test_path=str(tmp_path / "test.npy"),
        pair_sha256=pair_identity(test, oof),  # 순서가 뒤집힌 잘못된 기대값
    )
    with pytest.raises(MemberIntegrityError) as caught:
        load_members(source_of(spec), INDEX, InMemoryRunStore())
    assert caught.value.check == "pair_sha256"


def test_ledger_auc_rescoring_and_level():
    values = [0.1, 0.9, 0.2, 0.8]
    store = store_with(r1=values)
    auc = float(roc_auc_score(LABELS.to_numpy(), np.array(values)))
    good = identity_spec("a", "r1", expected_auc=auc)
    matrix = load_members(source_of(good), INDEX, store, labels=LABELS)
    assert matrix.members.iloc[0]["verification"] == AUC_VERIFIED
    assert matrix.members.iloc[0]["rescored_auc"] == pytest.approx(auc, abs=0)

    bad = identity_spec("a", "r1", expected_auc=auc + 1e-6)
    with pytest.raises(MemberIntegrityError) as caught:
        load_members(source_of(bad), INDEX, store, labels=LABELS)
    assert caught.value.check == "ledger_auc"


def test_identity_only_without_labels_stays_identity_only():
    store = store_with(r1=[0.1, 0.9, 0.2, 0.8])
    source = source_of(identity_spec("a", "r1", expected_auc=0.5))
    matrix = load_members(source, INDEX, store)
    assert matrix.members.iloc[0]["verification"] == IDENTITY_ONLY
    with pytest.raises(MemberIntegrityError) as caught:
        matrix.require(HASH_VERIFIED)
    assert caught.value.check == "verification_level"


def test_auc_verified_declaration_requires_labels():
    store = store_with(r1=[0.1, 0.9, 0.2, 0.8])
    spec = dataclasses.replace(identity_spec("a", "r1", expected_auc=0.75), verification=AUC_VERIFIED)
    with pytest.raises(MemberSourceInvalid):
        load_members(source_of(spec), INDEX, store)


def test_source_declaration_contract_is_validated():
    with pytest.raises(MemberSourceInvalid):  # 해시 근거 없는 hash-verified 선언
        source_of(
            MemberSpec(member_id="a", origin="own", verification=HASH_VERIFIED, run_id="r1")
        )
    with pytest.raises(MemberSourceInvalid):  # member_id 중복
        source_of(identity_spec("a", "r1"), identity_spec("a", "r2"))
    with pytest.raises(MemberSourceInvalid):  # OOF 출처 이중 선언
        source_of(identity_spec("a", "r1", oof_path="x.npy"))
    with pytest.raises(MemberSourceInvalid):  # 시험 예측 혼재
        source_of(identity_spec("a", "r1"), identity_spec("b", "r2", test_path="t.npy"))


def test_row_count_contracts(tmp_path):
    short = np.array([0.1, 0.2, 0.3])
    np.save(tmp_path / "oof.npy", short)
    spec = MemberSpec(
        member_id="ext", origin="external", verification=IDENTITY_ONLY,
        oof_path=str(tmp_path / "oof.npy"),
    )
    with pytest.raises(MemberIntegrityError) as caught:
        load_members(source_of(spec), INDEX, InMemoryRunStore())
    assert caught.value.check == "oof_rows"

    store = store_with(r1=[0.1, 0.2, 0.3, 0.4])
    with pytest.raises(MemberSourceInvalid):
        load_members(source_of(identity_spec("a", "r1"), train_rows=5), INDEX, store)


def test_path_notations_parquet_column_and_npy_matrix(tmp_path):
    frame = pd.DataFrame({"m1": [0.1, 0.2, 0.3, 0.4], "m2": [0.5, 0.6, 0.7, 0.8]})
    frame.to_parquet(tmp_path / "oof.parquet")
    matrix2d = np.array([[0.11, 0.21], [0.12, 0.22], [0.13, 0.23], [0.14, 0.24]])
    np.save(tmp_path / "grid.npy", matrix2d)
    source = source_of(
        MemberSpec(
            member_id="m2", origin="external", verification=IDENTITY_ONLY,
            oof_path=f"{tmp_path / 'oof.parquet'}[m2]",
        ),
        MemberSpec(
            member_id="g1", origin="external", verification=IDENTITY_ONLY,
            oof_path=f"{tmp_path / 'grid.npy'}[:, 1]",
        ),
    )
    matrix = load_members(source, INDEX, InMemoryRunStore())
    np.testing.assert_array_equal(matrix.oof[:, 0], [0.5, 0.6, 0.7, 0.8])
    np.testing.assert_array_equal(matrix.oof[:, 1], [0.21, 0.22, 0.23, 0.24])


# ---------------------------------------------------------------------------
# adapter 3종


def test_manifest_adapter_maps_own_and_external(tmp_path):
    manifest = {
        "members": [
            {
                "column": "exp001",
                "origin": "own",
                "run_id": "r1",
                "test": {"kind": "own_cv5_full1_mix", "test_path": "artifacts/t.parquet",
                         "prediction_sha256": "a" * 64},
            },
            {
                "column": "ext_x:y",
                "origin": "external",
                "member_id": "x:y",
                "oof_path": "data/external/oof.npy",
                "test": {"kind": "external_cv_fold_average", "test_path": "data/external/test.npy",
                         "prediction_sha256": "b" * 64},
            },
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    source = manifest_members(path)
    own, ext = source.members
    assert own.run_id == "r1" and own.oof_path is None
    assert own.test_path == "artifacts/t.parquet[exp001]"
    assert own.test_sha256 == "a" * 64 and own.verification == HASH_VERIFIED
    assert ext.member_id == "ext_x:y" and ext.oof_path == "data/external/oof.npy"
    assert ext.test_path == "data/external/test.npy" and ext.verification == HASH_VERIFIED


def test_freeze_spec_adapter_carries_quadruple_hashes(tmp_path):
    spec = {
        "row_contract": {"train_rows": 4, "test_rows": 2},
        "candidates": [
            {
                "member_id": "nb_a:m", "order": 1,
                "oof_path": "oof.npy", "test_path": "test.npy",
                "oof_sha256": "a" * 64, "test_sha256": "b" * 64, "pair_sha256": "c" * 64,
                "rescored_auc": 0.9,
            }
        ],
    }
    path = tmp_path / "ecf.json"
    path.write_text(json.dumps(spec))
    source = freeze_spec_members(path)
    assert source.train_rows == 4 and source.test_rows == 2
    candidate = source.members[0]
    assert candidate.verification == HASH_VERIFIED
    assert (candidate.oof_sha256, candidate.test_sha256, candidate.pair_sha256) == (
        "a" * 64, "b" * 64, "c" * 64,
    )
    assert candidate.expected_auc == 0.9

    spec["candidates"][0]["order"] = 2
    path.write_text(json.dumps(spec))
    with pytest.raises(MemberSourceInvalid):
        freeze_spec_members(path)


def test_pool_adapter_is_identity_only():
    evidence = EntryEvidence(
        champion_run_id="r1", champion_oof_auc=0.9, floor_margin=0.01,
        nearest_run_id=None, nearest_spearman=None,
        ensemble_auc_with=None, ensemble_auc_without=None, contribution=None,
    )
    pool = Pool(members=[
        PoolMember(run_id="r1", config="exp001", oof_auc=0.9, seeds=[42],
                   entered_at="2026-08-11", reason="시험", evidence=evidence),
    ])
    source = pool_members(pool)
    member = source.members[0]
    assert member.member_id == "exp001" and member.run_id == "r1"
    assert member.verification == IDENTITY_ONLY and member.expected_auc == 0.9


# ---------------------------------------------------------------------------
# golden: 커밋된 실제 동결 명세로 기록 해시·AUC 재현


def test_golden_freeze_spec_members_reproduce_recorded_identity():
    train_path = REPO_ROOT / "data/train.csv"
    if not train_path.is_file():
        pytest.skip("data/train.csv가 이 체크아웃에 없다.")
    source = freeze_spec_members(FREEZE_SPEC)
    trimmed = dataclasses.replace(source, members=source.members[:3])
    if not all(
        (REPO_ROOT / spec.oof_path).is_file() and (REPO_ROOT / spec.test_path).is_file()
        for spec in trimmed.members
    ):
        pytest.skip("동결 명세의 배열 파일이 이 체크아웃에 없다(data/external은 gitignore).")
    train = pd.read_csv(train_path, usecols=[ID, TARGET])
    index = pd.Index(train[ID], name=ID)
    labels = pd.Series(train[TARGET].to_numpy(), index=index)
    matrix = load_members(trimmed, index, InMemoryRunStore(), labels=labels)
    matrix.require(HASH_VERIFIED)
    for spec, (_, row) in zip(trimmed.members, matrix.members.iterrows()):
        assert row["oof_sha256"] == spec.oof_sha256
        assert row["test_sha256"] == spec.test_sha256
        assert row["pair_sha256"] == spec.pair_sha256
        assert abs(row["rescored_auc"] - spec.expected_auc) <= 1e-9
