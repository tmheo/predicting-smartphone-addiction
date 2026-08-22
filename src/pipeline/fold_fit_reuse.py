"""내용 기반 fold-fit 재사용 결과 저장소.

저장소의 항목은 컬럼 제공자 하나, 시드 하나, 바깥쪽 fold 하나의 학습 및 시험 행
피처 값이다. 항목은 내용 키 아래 불변으로 공개하며, 예상 키의 손상은 미적중으로
바꾸지 않는다.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import shutil
import struct
import sys
import uuid
import zipfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import ID, file_sha256

SCHEMA_VERSION = 1
DEFAULT_ROOT = Path("run-cache/fold-fit/v1")
MANIFEST_NAME = "manifest.json"
TRAIN_NAME = "train.parquet"
TEST_NAME = "test.parquet"
EVIDENCE_NAME = "fold_feature_reuse.json"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_NAME = "fold_fit_reuse_bundle.json"
_ITEM_NAMES = (MANIFEST_NAME, TRAIN_NAME, TEST_NAME)


class FoldFitReuseError(Exception):
    """재사용 저장소의 손상 또는 사용 불가."""


@dataclass(frozen=True)
class FoldFitReuseRequest:
    """한 제공자와 fold의 내용 정체성 계산 입력."""

    provider: dict[str, object]
    runtime: dict[str, object]
    input_files: dict[str, str]
    seed: int
    fold: int
    train_input: pd.DataFrame
    test_input: pd.DataFrame
    training_ids: pd.Series
    validation_ids: pd.Series
    test_ids: pd.Series
    training_target: pd.Series | None

    def identity_document(self) -> dict[str, object]:
        validate_runtime_identity(self.runtime)
        validate_input_files(self.input_files)
        _validate_provider_identity(self.provider)
        document: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "provider": self.provider,
            "input_file_sha256": dict(sorted(self.input_files.items())),
            "input_value_sha256": {
                "train": dataframe_value_sha256(self.train_input),
                "test": dataframe_value_sha256(self.test_input),
            },
            "seed": int(self.seed),
            "fold": int(self.fold),
            "row_ids": {
                "training": series_value_document(self.training_ids),
                "validation": series_value_document(self.validation_ids),
                "test": series_value_document(self.test_ids),
            },
            "output_row_ids": {
                "train": series_value_document(self.train_input[ID]),
                "test": series_value_document(self.test_input[ID]),
            },
            "runtime": self.runtime,
        }
        uses_target = bool(self.provider.get("uses_target"))
        if uses_target:
            if self.training_target is None:
                raise FoldFitReuseError("타깃 참조 제공자의 학습 fold 타깃이 없다.")
            if len(self.training_target) != len(self.training_ids):
                raise FoldFitReuseError("학습 fold 행 식별자와 타깃 길이가 다르다.")
            target_frame = pd.DataFrame(
                {
                    ID: self.training_ids.to_numpy(),
                    "target": self.training_target.to_numpy(),
                }
            )
            document["training_target_value_sha256"] = dataframe_value_sha256(
                target_frame
            )
        elif self.training_target is not None:
            raise FoldFitReuseError("타깃 비참조 제공자 정체성에 타깃이 전달됐다.")
        # Python 튜플처럼 JSON 배열로 기록되는 값도 manifest 재판독 뒤 같은
        # 정체성이 되도록 키 계산 전에 정규 JSON 자료형으로 고정한다.
        return json.loads(canonical_json_bytes(document))


@dataclass(frozen=True)
class FoldFitReuseResult:
    train: pd.DataFrame
    test: pd.DataFrame
    status: str
    key: str
    manifest_sha256: str


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_scalar(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return {"type": "missing"}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "missing"}
        if math.isinf(value):
            return {"type": "float", "value": "+inf" if value > 0 else "-inf"}
        return {
            "type": "float64",
            "bits": struct.pack("<d", value).hex(),
        }
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, pd.Timestamp):
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {"type": "timedelta_ns", "value": str(value.value)}
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": str(value),
    }


def dtype_document(series: pd.Series) -> dict[str, object]:
    dtype = series.dtype
    document: dict[str, object] = {"dtype": str(dtype)}
    if isinstance(dtype, pd.CategoricalDtype):
        document.update(
            {
                "kind": "category",
                "ordered": bool(dtype.ordered),
                "categories": [_json_scalar(value) for value in dtype.categories.tolist()],
            }
        )
    else:
        document["kind"] = "plain"
    return document


def dataframe_schema(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {"name": str(name), **dtype_document(frame[name])}
        for name in frame.columns
    ]


def _update_length_prefixed(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _canonical_numeric_bytes(series: pd.Series) -> bytes | None:
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_extension_array_dtype(dtype):
        return None
    if pd.api.types.is_float_dtype(dtype):
        values = series.to_numpy(copy=True)
        values[np.isnan(values)] = np.nan
    elif (
        pd.api.types.is_integer_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_timedelta64_dtype(dtype)
    ):
        values = series.to_numpy(copy=False)
    else:
        return None
    values = np.asarray(values)
    if values.dtype.byteorder == ">" or (
        values.dtype.byteorder == "=" and sys.byteorder == "big"
    ):
        values = values.byteswap().view(values.dtype.newbyteorder("<"))
    return np.ascontiguousarray(values).tobytes()


def dataframe_value_sha256(frame: pd.DataFrame) -> str:
    """열·자료형·범주·결측·행 순서와 부동소수점 비트를 보존해 순차 해시한다."""
    if len(set(map(str, frame.columns))) != len(frame.columns):
        raise FoldFitReuseError("내용 해시에 중복 컬럼을 사용할 수 없다.")
    digest = hashlib.sha256()
    digest.update(b"fold-fit-dataframe-v1\0")
    digest.update(struct.pack("<Q", len(frame)))
    digest.update(struct.pack("<Q", len(frame.columns)))
    for name in frame.columns:
        series = frame[name]
        _update_length_prefixed(digest, str(name).encode("utf-8"))
        _update_length_prefixed(digest, canonical_json_bytes(dtype_document(series)))
        if isinstance(series.dtype, pd.CategoricalDtype):
            digest.update(b"category\0")
            codes = series.cat.codes.to_numpy(dtype="<i8", copy=False)
            digest.update(np.ascontiguousarray(codes).tobytes())
            continue
        numeric = _canonical_numeric_bytes(series)
        if numeric is not None:
            digest.update(b"numeric\0")
            digest.update(numeric)
            continue
        digest.update(b"scalar\0")
        for value in series.tolist():
            _update_length_prefixed(digest, canonical_json_bytes(_json_scalar(value)))
    return digest.hexdigest()


def series_value_document(series: pd.Series) -> dict[str, object]:
    """행 식별자 값과 순서를 큰 JSON 배열 없이 내용 해시로 고정한다."""
    normalized = series.reset_index(drop=True)
    frame = normalized.to_frame(name="value")
    return {
        "row_count": len(normalized),
        "dtype": dtype_document(normalized),
        "value_sha256": dataframe_value_sha256(frame),
    }


def build_runtime_identity(
    *,
    git_commit: str,
    git_dirty: bool,
    lock_path: Path,
) -> dict[str, object]:
    """한 프로세스에서 공유하는 Python·의존성·운영체제 정체성을 확정한다."""
    if git_dirty:
        raise FoldFitReuseError(
            "비커밋 변경이 있는 실행은 Git 정체성을 확정할 수 없어 fold-fit 결과를 공유할 수 없다."
        )
    if not lock_path.is_file():
        raise FoldFitReuseError(f"의존성 잠금 파일이 없다: {lock_path}")
    packages = sorted(
        {
            (
                (dist.metadata.get("Name") or dist.name).lower().replace("_", "-"),
                dist.version,
            )
            for dist in importlib.metadata.distributions()
        }
    )
    return {
        "git_commit": git_commit,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "dependency_lock_sha256": file_sha256(lock_path),
        "installed_packages": [
            {"name": name, "version": version} for name, version in packages
        ],
        "platform": {
            "operating_system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }


def validate_runtime_identity(identity: dict[str, object]) -> None:
    required = {
        "git_commit",
        "python",
        "dependency_lock_sha256",
        "installed_packages",
        "platform",
    }
    if not isinstance(identity, dict) or not required.issubset(identity):
        missing = required - set(identity) if isinstance(identity, dict) else required
        raise FoldFitReuseError(
            f"fold-fit 재사용 실행 환경 정체성이 부족하다: {sorted(missing)}"
        )
    commit = identity["git_commit"]
    if not isinstance(commit, str) or len(commit) not in {40, 64} or not _is_hex(commit):
        raise FoldFitReuseError("fold-fit 재사용 Git 커밋 정체성이 잘못됐다.")
    _require_sha256(identity["dependency_lock_sha256"], "의존성 잠금 파일")
    python = identity["python"]
    if not isinstance(python, dict) or not all(
        isinstance(python.get(name), str) and python.get(name)
        for name in ("implementation", "version")
    ):
        raise FoldFitReuseError("fold-fit 재사용 Python 정체성이 잘못됐다.")
    packages = identity["installed_packages"]
    if not isinstance(packages, list) or not packages:
        raise FoldFitReuseError("fold-fit 재사용 설치 패키지 판본이 없다.")
    for package in packages:
        if not isinstance(package, dict) or not all(
            isinstance(package.get(name), str) and package.get(name)
            for name in ("name", "version")
        ):
            raise FoldFitReuseError("fold-fit 재사용 설치 패키지 판본이 잘못됐다.")
    system = identity["platform"]
    if not isinstance(system, dict) or not all(
        isinstance(system.get(name), str) and system.get(name)
        for name in ("operating_system", "release", "machine")
    ):
        raise FoldFitReuseError("fold-fit 재사용 운영체제 정체성이 잘못됐다.")
    canonical_json_bytes(identity)


def validate_input_files(input_files: dict[str, str]) -> None:
    required = {"train", "test", "folds"}
    if not isinstance(input_files, dict) or not required.issubset(input_files):
        missing = required - set(input_files) if isinstance(input_files, dict) else required
        raise FoldFitReuseError(
            f"fold-fit 재사용 입력 파일 해시가 부족하다: {sorted(missing)}"
        )
    for name, digest in input_files.items():
        _require_sha256(digest, f"입력 파일 {name}")


def _validate_provider_identity(provider: dict[str, object]) -> None:
    required = {
        "kind",
        "implementation",
        "implementation_sha256",
        "settings",
        "input_columns",
        "output_columns",
        "uses_target",
        "external_file_sha256",
        "execution",
    }
    if not isinstance(provider, dict) or not required.issubset(provider):
        raise FoldFitReuseError("fold-fit 재사용 제공자 정체성이 부족하다.")
    if not all(
        isinstance(provider[name], str) and provider[name]
        for name in ("kind", "implementation")
    ):
        raise FoldFitReuseError("fold-fit 재사용 제공자 종류나 구현 정체성이 잘못됐다.")
    _require_sha256(provider["implementation_sha256"], "제공자 구현")
    if not isinstance(provider["settings"], dict):
        raise FoldFitReuseError("fold-fit 재사용 제공자의 의미 있는 설정이 잘못됐다.")
    for name in ("input_columns", "output_columns"):
        columns = provider[name]
        if (
            not isinstance(columns, list)
            or any(not isinstance(column, str) or not column for column in columns)
            or len(set(columns)) != len(columns)
        ):
            raise FoldFitReuseError(f"fold-fit 재사용 제공자 {name} 선언이 잘못됐다.")
    if not isinstance(provider["uses_target"], bool):
        raise FoldFitReuseError("fold-fit 재사용 제공자의 타깃 참조 선언이 잘못됐다.")
    external = provider["external_file_sha256"]
    if not isinstance(external, dict):
        raise FoldFitReuseError("fold-fit 재사용 제공자의 외부 파일 해시가 잘못됐다.")
    for name, digest in external.items():
        if not isinstance(name, str) or not name:
            raise FoldFitReuseError("fold-fit 재사용 제공자의 외부 파일 이름이 잘못됐다.")
        _require_sha256(digest, f"제공자 외부 파일 {name}")
    execution = provider["execution"]
    if not isinstance(execution, dict) or execution.get("mode") not in {"cpu", "cuda"}:
        raise FoldFitReuseError("fold-fit 재사용 제공자의 실제 실행 방식이 잘못됐다.")
    if execution["mode"] == "cuda":
        cuda_fields = {"gpu_model", "compute_capability", "cuda_version", "driver_version"}
        if not all(
            isinstance(execution.get(name), str) and execution.get(name)
            for name in cuda_fields
        ):
            raise FoldFitReuseError("CUDA 재사용 정체성에 GPU와 CUDA 환경 판본이 부족하다.")
    canonical_json_bytes(provider)


def provider_identity_document(
    *,
    kind: str,
    provider: object,
    input_columns: list[str],
    output_columns: list[str],
    uses_target: bool,
    settings: dict[str, object],
    external_file_sha256: dict[str, str],
    execution: dict[str, object],
) -> dict[str, object]:
    try:
        source = inspect.getsource(type(provider)).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise FoldFitReuseError(f"{kind} 제공자 구현 내용을 확정할 수 없다.") from exc
    return {
        "kind": kind,
        "implementation": f"{type(provider).__module__}.{type(provider).__qualname__}",
        "implementation_sha256": hashlib.sha256(source).hexdigest(),
        "settings": settings,
        "input_columns": list(input_columns),
        "output_columns": list(output_columns),
        "uses_target": bool(uses_target),
        "external_file_sha256": dict(sorted(external_file_sha256.items())),
        "execution": execution,
    }


class FoldFitReuseStore:
    """파일 잠금과 같은 파일시스템의 원자적 이름 변경을 쓰는 불변 저장소."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._locks = self.root / ".locks"
        self._prepare()

    def _prepare(self) -> None:
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not self.root.is_dir():
                raise NotADirectoryError(self.root)
            self._locks.mkdir(mode=0o700, exist_ok=True)
            self._probe_filesystem()
        except FoldFitReuseError:
            raise
        except OSError as exc:
            raise FoldFitReuseError(
                f"fold-fit 재사용 저장소를 사용할 수 없다: {self.root}"
            ) from exc

    def _probe_filesystem(self) -> None:
        """실제 잠금 배제, 파일 동기화와 같은 장치의 원자적 이름 변경을 확인한다."""
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - 지원 환경은 macOS/Linux다.
            raise FoldFitReuseError("운영체제 파일 잠금을 지원하지 않는다.") from exc

        probe = f".probe-{os.getpid()}-{uuid.uuid4().hex}"
        lock_path = self._locks / f"{probe}.lock"
        source = self.root / f"{probe}.source"
        target = self.root / f"{probe}.target"
        fd1 = fd2 = None
        try:
            fd1 = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            fd2 = os.open(lock_path, os.O_RDWR)
            fcntl.flock(fd1, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise FoldFitReuseError("파일 잠금이 동시 접근을 배제하지 못한다.")
            with source.open("xb") as stream:
                stream.write(b"fold-fit-reuse-probe")
                stream.flush()
                os.fsync(stream.fileno())
            if source.stat().st_dev != self.root.stat().st_dev:
                raise FoldFitReuseError("임시 파일과 재사용 저장소가 같은 파일시스템이 아니다.")
            os.rename(source, target)
            _fsync_directory(self.root)
            if target.read_bytes() != b"fold-fit-reuse-probe":
                raise FoldFitReuseError("원자적 이름 변경 검사 결과가 다르다.")
        except FoldFitReuseError:
            raise
        except OSError as exc:
            raise FoldFitReuseError(
                "파일 잠금, 동기화 또는 원자적 이름 변경을 보장할 수 없다."
            ) from exc
        finally:
            if fd2 is not None:
                os.close(fd2)
            if fd1 is not None:
                os.close(fd1)
            source.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            lock_path.unlink(missing_ok=True)

    @staticmethod
    def key_of(identity: dict[str, object]) -> str:
        return content_sha256(identity)

    def item_path(self, key: str) -> Path:
        _validate_key(key)
        return self.root / key

    @contextmanager
    def _key_lock(self, key: str) -> Iterator[None]:
        try:
            import fcntl

            lock_path = self._locks / f"{key}.lock"
            with lock_path.open("a+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                yield
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except FoldFitReuseError:
            raise
        except OSError as exc:
            raise FoldFitReuseError(f"내용 키 {key}의 파일 잠금을 사용할 수 없다.") from exc

    def resolve(
        self,
        request: FoldFitReuseRequest,
        compute: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
    ) -> FoldFitReuseResult:
        identity = request.identity_document()
        key = self.key_of(identity)
        final = self.item_path(key)
        if final.exists():
            train, test, manifest_sha = self._read_item(
                final, expected_key=key, expected_identity=identity
            )
            return FoldFitReuseResult(train, test, "hit", key, manifest_sha)

        with self._key_lock(key):
            if final.exists():
                train, test, manifest_sha = self._read_item(
                    final, expected_key=key, expected_identity=identity
                )
                return FoldFitReuseResult(train, test, "hit", key, manifest_sha)
            self._remove_leftover_temporary_directories(key)
            train, test = compute()
            self._validate_computed_output(request, train, test)
            manifest_sha = self._publish(key, identity, train, test)
            loaded_train, loaded_test, loaded_manifest_sha = self._read_item(
                final, expected_key=key, expected_identity=identity
            )
            if loaded_manifest_sha != manifest_sha:
                raise FoldFitReuseError("공개 직후 명세 기록 내용 해시가 바뀌었다.")
            return FoldFitReuseResult(
                loaded_train,
                loaded_test,
                "generated",
                key,
                loaded_manifest_sha,
            )

    def _remove_leftover_temporary_directories(self, key: str) -> None:
        for path in self.root.glob(f".tmp-{key}-*"):
            if path.is_dir():
                shutil.rmtree(path)

    @staticmethod
    def _validate_computed_output(
        request: FoldFitReuseRequest,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        output_columns = list(request.provider["output_columns"])
        expected_columns = [ID, *output_columns]
        for name, frame, ids in (
            ("train", train, request.train_input[ID]),
            ("test", test, request.test_input[ID]),
        ):
            if list(frame.columns) != expected_columns:
                raise FoldFitReuseError(
                    f"{name} 재사용 결과 컬럼이 선언과 다르다: "
                    f"{list(frame.columns)} != {expected_columns}"
                )
            if not frame[ID].reset_index(drop=True).equals(ids.reset_index(drop=True)):
                raise FoldFitReuseError(f"{name} 재사용 결과 행 식별자나 순서가 다르다.")

    def _publish(
        self,
        key: str,
        identity: dict[str, object],
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> str:
        temporary = self.root / f".tmp-{key}-{os.getpid()}-{uuid.uuid4().hex}"
        final = self.item_path(key)
        try:
            temporary.mkdir(mode=0o700)
            train_path = temporary / TRAIN_NAME
            test_path = temporary / TEST_NAME
            train.to_parquet(train_path, index=False)
            test.to_parquet(test_path, index=False)
            _fsync_file(train_path)
            _fsync_file(test_path)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "key": key,
                "identity": identity,
                "identity_sha256": content_sha256(identity),
                "tables": {
                    "train": _table_manifest(train_path, train),
                    "test": _table_manifest(test_path, test),
                },
            }
            manifest_path = temporary / MANIFEST_NAME
            manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
            _fsync_file(manifest_path)
            _make_item_immutable(temporary)
            _fsync_directory(temporary)
            os.rename(temporary, final)
            final.chmod(0o555)
            _fsync_directory(final)
            _fsync_directory(self.root)
            return file_sha256(final / MANIFEST_NAME)
        except FoldFitReuseError:
            raise
        except OSError as exc:
            raise FoldFitReuseError(f"내용 키 {key}를 원자적으로 공개할 수 없다.") from exc
        finally:
            if temporary.exists():
                _remove_temporary_item(temporary)

    def _read_item(
        self,
        path: Path,
        *,
        expected_key: str | None = None,
        expected_identity: dict[str, object] | None = None,
        require_path_name: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        try:
            if not path.is_dir():
                raise FoldFitReuseError(f"재사용 항목이 디렉터리가 아니다: {path}")
            names = sorted(item.name for item in path.iterdir())
            if names != sorted(_ITEM_NAMES):
                raise FoldFitReuseError(f"재사용 항목 파일 구성이 잘못됐다: {names}")
            manifest_path = path / MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text())
            key = str(manifest.get("key"))
            _validate_key(key)
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise FoldFitReuseError(
                    f"지원하지 않는 재사용 형식 판본이다: {manifest.get('schema_version')}"
                )
            if require_path_name and path.name != key:
                raise FoldFitReuseError("재사용 항목 디렉터리와 명세 내용 키가 다르다.")
            identity = manifest.get("identity")
            if not isinstance(identity, dict) or content_sha256(identity) != manifest.get(
                "identity_sha256"
            ):
                raise FoldFitReuseError("재사용 항목의 정체성 문서가 손상됐다.")
            if self.key_of(identity) != key:
                raise FoldFitReuseError("재사용 항목의 내용 키가 정체성 문서와 다르다.")
            if expected_key is not None and key != expected_key:
                raise FoldFitReuseError("예상 내용 키와 재사용 항목의 키가 다르다.")
            if expected_identity is not None and identity != expected_identity:
                raise FoldFitReuseError("예상 정체성과 재사용 항목의 정체성이 다르다.")
            tables = manifest.get("tables")
            if not isinstance(tables, dict):
                raise FoldFitReuseError("재사용 항목의 표 명세가 없다.")
            provider = identity.get("provider")
            output_row_ids = identity.get("output_row_ids")
            if not isinstance(provider, dict) or not isinstance(output_row_ids, dict):
                raise FoldFitReuseError("재사용 항목의 제공자 또는 행 정체성이 없다.")
            output_columns = provider.get("output_columns")
            if not isinstance(output_columns, list):
                raise FoldFitReuseError("재사용 항목의 제공자 출력 선언이 없다.")
            frames: dict[str, pd.DataFrame] = {}
            for table, file_name in (("train", TRAIN_NAME), ("test", TEST_NAME)):
                table_path = path / file_name
                specification = tables.get(table)
                if not isinstance(specification, dict):
                    raise FoldFitReuseError(f"{table} 표 명세가 없다.")
                if file_sha256(table_path) != specification.get("file_sha256"):
                    raise FoldFitReuseError(f"{table} Parquet 파일 내용 해시가 다르다.")
                frame = pd.read_parquet(table_path)
                if list(frame.columns) != [ID, *output_columns]:
                    raise FoldFitReuseError(f"{table} 재사용 결과 컬럼이 제공자 선언과 다르다.")
                if len(frame) != specification.get("row_count"):
                    raise FoldFitReuseError(f"{table} 재사용 결과 행 수가 다르다.")
                if dataframe_schema(frame) != specification.get("schema"):
                    raise FoldFitReuseError(f"{table} 재사용 결과 열이나 자료형이 다르다.")
                if dataframe_value_sha256(frame) != specification.get("value_sha256"):
                    raise FoldFitReuseError(f"{table} 재사용 결과 값 내용 해시가 다르다.")
                if not frame[ID].is_unique:
                    raise FoldFitReuseError(f"{table} 재사용 결과 행 식별자가 고유하지 않다.")
                if series_value_document(frame[ID]) != output_row_ids.get(table):
                    raise FoldFitReuseError(f"{table} 재사용 결과 행 식별자나 순서가 다르다.")
                frames[table] = frame
            return frames["train"], frames["test"], file_sha256(manifest_path)
        except FoldFitReuseError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
            raise FoldFitReuseError(f"재사용 항목이 손상돼 읽을 수 없다: {path}") from exc

    def validate_item(self, key: str) -> str:
        _, _, manifest_sha = self._read_item(self.item_path(key), expected_key=key)
        return manifest_sha

    def export_bundle(self, keys: Iterable[str], out_path: Path) -> Path:
        unique_keys = list(dict.fromkeys(keys))
        if not unique_keys:
            raise FoldFitReuseError("내보낼 fold-fit 재사용 내용 키가 없다.")
        items: list[dict[str, object]] = []
        for key in unique_keys:
            manifest_sha = self.validate_item(key)
            item_path = self.item_path(key)
            items.append(
                {
                    "key": key,
                    "manifest_sha256": manifest_sha,
                    "files": {
                        name: file_sha256(item_path / name) for name in _ITEM_NAMES
                    },
                }
            )
        bundle_manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "items": items,
        }
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                BUNDLE_MANIFEST_NAME,
                canonical_json_bytes(bundle_manifest) + b"\n",
            )
            for item in items:
                key = str(item["key"])
                for name in _ITEM_NAMES:
                    archive.write(self.item_path(key) / name, f"items/{key}/{name}")
        return out_path

    def import_bundle(self, bundle_path: Path) -> list[str]:
        try:
            with zipfile.ZipFile(bundle_path) as archive:
                manifest = json.loads(archive.read(BUNDLE_MANIFEST_NAME))
                if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
                    raise FoldFitReuseError(
                        f"지원하지 않는 재사용 묶음 판본이다: {manifest.get('schema_version')}"
                    )
                items = manifest.get("items")
                if not isinstance(items, list) or not items:
                    raise FoldFitReuseError("재사용 묶음에 항목이 없다.")
                expected_members = {BUNDLE_MANIFEST_NAME}
                keys: list[str] = []
                for item in items:
                    key = str(item.get("key"))
                    _validate_key(key)
                    keys.append(key)
                    expected_members.update(f"items/{key}/{name}" for name in _ITEM_NAMES)
                if len(set(keys)) != len(keys):
                    raise FoldFitReuseError("재사용 묶음에 중복 내용 키가 있다.")
                members = archive.namelist()
                if len(set(members)) != len(members) or set(members) != expected_members:
                    raise FoldFitReuseError("재사용 묶음 파일 구성이 명세와 다르다.")
                staged: dict[str, Path] = {}
                try:
                    # 저장소를 바꾸기 전에 묶음의 모든 파일과 항목 내용을 먼저 검증한다.
                    for item in items:
                        key = str(item["key"])
                        temporary = self.root / (
                            f".bundle-{key}-{os.getpid()}-{uuid.uuid4().hex}"
                        )
                        temporary.mkdir(mode=0o700)
                        staged[key] = temporary
                        file_hashes = item.get("files")
                        if not isinstance(file_hashes, dict):
                            raise FoldFitReuseError("재사용 묶음 파일 해시가 없다.")
                        for name in _ITEM_NAMES:
                            payload = archive.read(f"items/{key}/{name}")
                            digest = hashlib.sha256(payload).hexdigest()
                            if digest != file_hashes.get(name):
                                raise FoldFitReuseError(
                                    f"재사용 묶음의 {key}/{name} 내용 해시가 다르다."
                                )
                            path = temporary / name
                            path.write_bytes(payload)
                            _fsync_file(path)
                        _, _, manifest_sha = self._read_item(
                            temporary,
                            expected_key=key,
                            require_path_name=False,
                        )
                        if manifest_sha != item.get("manifest_sha256"):
                            raise FoldFitReuseError(
                                f"재사용 묶음의 {key} 명세 내용 해시가 다르다."
                            )
                        _make_item_immutable(temporary)
                        _fsync_directory(temporary)

                    imported: list[str] = []
                    for item in items:
                        key = str(item["key"])
                        with self._key_lock(key):
                            final = self.item_path(key)
                            if final.exists():
                                existing_sha = self.validate_item(key)
                                if existing_sha != item.get("manifest_sha256"):
                                    raise FoldFitReuseError(
                                        f"내용 키 {key}의 기존 불변 항목이 묶음과 다르다."
                                    )
                            else:
                                self._remove_leftover_temporary_directories(key)
                                os.rename(staged[key], final)
                                final.chmod(0o555)
                                _fsync_directory(final)
                                _fsync_directory(self.root)
                            imported.append(key)
                    return imported
                finally:
                    for temporary in staged.values():
                        if temporary.exists():
                            _remove_temporary_item(temporary)
        except FoldFitReuseError:
            raise
        except (OSError, KeyError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            raise FoldFitReuseError(f"fold-fit 재사용 묶음을 반입할 수 없다: {bundle_path}") from exc


def _table_manifest(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "row_count": len(frame),
        "schema": dataframe_schema(frame),
        "file_sha256": file_sha256(path),
        "value_sha256": dataframe_value_sha256(frame),
    }


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_item_immutable(path: Path) -> None:
    for name in _ITEM_NAMES:
        item = path / name
        item.chmod(0o444)
        _fsync_file(item)


def _remove_temporary_item(path: Path) -> None:
    path.chmod(0o700)
    for name in _ITEM_NAMES:
        item = path / name
        if item.exists():
            item.chmod(0o600)
    shutil.rmtree(path)


def _validate_key(key: str) -> None:
    if len(key) != 64 or not _is_hex(key):
        raise FoldFitReuseError(f"잘못된 fold-fit 재사용 내용 키다: {key!r}")


def _is_hex(value: str) -> bool:
    return all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or not _is_hex(value):
        raise FoldFitReuseError(f"{label} SHA-256이 잘못됐다.")


def keys_from_evidence(path: Path) -> list[str]:
    try:
        document = json.loads(path.read_text())
        if document.get("schema_version") != SCHEMA_VERSION:
            raise FoldFitReuseError("지원하지 않는 fold_feature_reuse.json 판본이다.")
        entries = document.get("entries")
        if not isinstance(entries, list):
            raise FoldFitReuseError("fold_feature_reuse.json entries가 목록이 아니다.")
        return list(
            dict.fromkeys(
                str(entry["key"])
                for entry in entries
                if entry.get("status") in {"hit", "generated"} and entry.get("key")
            )
        )
    except FoldFitReuseError:
        raise
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FoldFitReuseError(f"재사용 근거 파일을 읽을 수 없다: {path}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="fold-fit 재사용 결과 전송 묶음")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="근거 파일의 항목을 묶음으로 내보낸다")
    export_parser.add_argument("--store", type=Path, default=DEFAULT_ROOT)
    export_parser.add_argument("--evidence", type=Path, required=True)
    export_parser.add_argument("--out", type=Path, required=True)
    import_parser = subparsers.add_parser("import", help="검증한 묶음을 불변 저장소에 반입한다")
    import_parser.add_argument("bundle", type=Path)
    import_parser.add_argument("--store", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    try:
        store = FoldFitReuseStore(args.store)
        if args.command == "export":
            out = store.export_bundle(keys_from_evidence(args.evidence), args.out)
            print(f"fold-fit 재사용 묶음 생성: {out} sha256={file_sha256(out)}")
        else:
            keys = store.import_bundle(args.bundle)
            print(f"fold-fit 재사용 묶음 반입: {len(keys)}개")
    except FoldFitReuseError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
