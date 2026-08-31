"""봉인 기록 module의 계약 시험."""

from __future__ import annotations

import json

import pytest

from pipeline.sealed import (
    LINEAGE_KEY,
    SchemaMismatch,
    SealBroken,
    SealedRecord,
    SealedRecordNotFound,
    SealRefused,
    canonical_sha256,
)

SCHEMA = "sealed-contract-v1/precommit/1"


def test_seal_open_round_trip(tmp_path):
    record = SealedRecord.seal(SCHEMA, {"b": 2, "a": "한글", "nested": {"x": [1, 2]}})
    path = tmp_path / "record.json"
    record.write(path)

    reopened = SealedRecord.open(path, schema=SCHEMA)

    assert reopened == record
    assert reopened.sealed_sha256 == canonical_sha256(
        {"a": "한글", "b": 2, "nested": {"x": [1, 2]}, "schema": SCHEMA}
    )


def test_open_detects_single_byte_tampering(tmp_path):
    record = SealedRecord.seal(SCHEMA, {"value": "abc"})
    path = tmp_path / "record.json"
    document = record.document
    document["value"] = "abd"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SealBroken):
        SealedRecord.open(path, schema=SCHEMA)


def test_open_rejects_schema_mismatch(tmp_path):
    record = SealedRecord.seal(SCHEMA, {"value": 1})
    path = tmp_path / "record.json"
    record.write(path)

    with pytest.raises(SchemaMismatch):
        SealedRecord.open(path, schema="sealed-contract-v1/precommit/2")


def test_open_rejects_missing_file(tmp_path):
    with pytest.raises(SealedRecordNotFound):
        SealedRecord.open(tmp_path / "absent.json", schema=SCHEMA)


def test_seal_rejects_preexisting_hash_and_schema_keys():
    with pytest.raises(SealRefused):
        SealedRecord.seal(SCHEMA, {"sealed_sha256": "stale", "value": 1})
    with pytest.raises(SealRefused):
        SealedRecord.seal(SCHEMA, {"schema": "other", "value": 1})


def test_seal_rejects_malformed_schema():
    for bad in ("no-slashes", "a/b", "a/b/c/d", "a//1", "a/b/0", "a/b/one"):
        with pytest.raises(SealRefused):
            SealedRecord.seal(bad, {"value": 1})


def test_derive_propagates_lineage(tmp_path):
    parent = SealedRecord.seal(SCHEMA, {"value": 1})
    derived = parent.derive("judgment", {"verdict": "pass"})

    assert derived.schema == "sealed-contract-v1/judgment/1"
    assert derived.payload[LINEAGE_KEY] == parent.sealed_sha256

    path = tmp_path / "derived.json"
    derived.write(path)
    reopened = SealedRecord.open(path, schema=derived.schema)
    assert reopened.payload[LINEAGE_KEY] == parent.sealed_sha256


def test_derive_rejects_bad_kind_and_preexisting_lineage():
    parent = SealedRecord.seal(SCHEMA, {"value": 1})
    with pytest.raises(SealRefused):
        parent.derive("with/slash", {})
    with pytest.raises(SealRefused):
        parent.derive("judgment", {LINEAGE_KEY: "stale"})
