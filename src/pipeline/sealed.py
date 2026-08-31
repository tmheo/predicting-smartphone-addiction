"""봉인 기록(sealed record)의 정본 구현.

봉인 기록은 canonical JSON(sort_keys, 최소 구분자, ensure_ascii=False)의
SHA-256을 sealed_sha256 키로 본문에 담아 제자리 수정을 탐지하는 기록이다.
schema 문자열은 기존 관례 "<계약판>/<종류>/<판>"을 따르고 별도 kind 필드는 두지 않는다.

과거 precommit_sha256·spec_sha256 등 다른 해시 키를 쓰는 파일의 읽기 호환은 하지 않는다.
그런 파일은 각 동결 시점의 verify 코드 소관으로 남는다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

SEALED_HASH_KEY = "sealed_sha256"
SCHEMA_KEY = "schema"
LINEAGE_KEY = "parent_sealed_sha256"


class SealedRecordError(RuntimeError):
    """봉인 기록 계약 위반."""


class SealRefused(SealedRecordError):
    """봉인 시점의 입력이 계약을 위반해 봉인을 거부했다."""


class SealedRecordNotFound(SealedRecordError):
    """봉인 기록 파일이 없다."""


class SealBroken(SealedRecordError):
    """읽은 기록의 sealed_sha256이 본문 재계산과 다르거나 기록 형태가 아니다."""


class SchemaMismatch(SealedRecordError):
    """읽은 기록의 schema가 기대한 schema와 다르다."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_schema(schema: str) -> None:
    segments = schema.split("/")
    if len(segments) != 3 or not all(segments):
        raise SealRefused(f"schema는 '<계약판>/<종류>/<판>' 형식이어야 한다: {schema!r}")
    if not segments[2].isdigit() or int(segments[2]) < 1:
        raise SealRefused(f"schema의 판은 1 이상의 정수여야 한다: {schema!r}")


@dataclass(frozen=True)
class SealedRecord:
    """schema와 본문을 sealed_sha256으로 봉인한 불변 기록."""

    schema: str
    payload: Mapping[str, Any]
    sealed_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    @classmethod
    def seal(cls, schema: str, payload: Mapping[str, Any]) -> SealedRecord:
        """canonical JSON의 SHA-256으로 본문을 봉인한다.

        payload에 schema나 sealed_sha256 키가 이미 있으면 거부한다.
        선존재하는 낡은 해시 값이 digest에 섞이면 verify가 나중에 실패하기 때문이다.
        """
        _require_schema(schema)
        for reserved in (SCHEMA_KEY, SEALED_HASH_KEY):
            if reserved in payload:
                raise SealRefused(f"payload에 예약 키 {reserved}가 이미 있다.")
        body = {**payload, SCHEMA_KEY: schema}
        return cls(schema=schema, payload=payload, sealed_sha256=canonical_sha256(body))

    @classmethod
    def open(cls, path: Path | str, *, schema: str) -> SealedRecord:
        """파일을 읽고 재계산 대조로 제자리 수정을 탐지한다."""
        _require_schema(schema)
        path = Path(path)
        if not path.exists():
            raise SealedRecordNotFound(f"봉인 기록 파일이 없다: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise SealBroken(f"봉인 기록이 JSON 객체가 아니다: {path}")
        actual_schema = raw.get(SCHEMA_KEY)
        if actual_schema != schema:
            raise SchemaMismatch(
                f"schema가 다르다: 기대 {schema!r}, 기록 {actual_schema!r} ({path})"
            )
        recorded = raw.get(SEALED_HASH_KEY)
        body = {key: value for key, value in raw.items() if key != SEALED_HASH_KEY}
        if recorded != canonical_sha256(body):
            raise SealBroken(f"{SEALED_HASH_KEY}가 본문 내용과 다르다: {path}")
        payload = {
            key: value
            for key, value in raw.items()
            if key not in (SCHEMA_KEY, SEALED_HASH_KEY)
        }
        return cls(schema=actual_schema, payload=payload, sealed_sha256=recorded)

    @property
    def document(self) -> dict[str, Any]:
        """schema와 sealed_sha256을 포함한 저장 형태 전체."""
        return {**self.payload, SCHEMA_KEY: self.schema, SEALED_HASH_KEY: self.sealed_sha256}

    def write(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def derive(self, kind: str, payload: Mapping[str, Any], *, revision: int = 1) -> SealedRecord:
        """이 기록의 자기 해시를 계보 필드로 물려받는 파생 기록을 봉인한다.

        파생 schema는 같은 계약판에 kind와 revision을 붙인 "<계약판>/<kind>/<revision>"이다.
        """
        if "/" in kind or not kind:
            raise SealRefused(f"kind는 '/' 없는 비어 있지 않은 문자열이어야 한다: {kind!r}")
        if LINEAGE_KEY in payload:
            raise SealRefused(f"payload에 계보 키 {LINEAGE_KEY}가 이미 있다.")
        contract = self.schema.split("/")[0]
        derived_schema = f"{contract}/{kind}/{revision}"
        return SealedRecord.seal(derived_schema, {**payload, LINEAGE_KEY: self.sealed_sha256})
