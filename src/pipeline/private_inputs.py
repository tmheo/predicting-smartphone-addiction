"""새 작업 폴더의 비커밋 입력을 준비하고 검증하는 실행 진입점."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import stat
import string
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeclaredInput:
    sha256: str
    path: Path


class ManifestError(ValueError):
    """검증 목록 자체가 준비 계약을 위반했다."""


class PreparationError(RuntimeError):
    """입력 준비가 대상을 바꾸기 전에 중단되어야 한다."""


def _load_manifest(path: Path) -> list[DeclaredInput]:
    declared: list[DeclaredInput] = []
    seen: set[Path] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            continue
        try:
            sha256, relative = line.split(maxsplit=1)
        except ValueError as exc:
            raise ManifestError(f"{path}:{line_number}: 'SHA-256  경로' 형식이 아님") from exc
        if len(sha256) != 64 or any(character not in string.hexdigits for character in sha256):
            raise ManifestError(f"{path}:{line_number}: SHA-256 형식이 아님")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or len(relative_path.parts) < 2
            or relative_path.parts[0] != "data"
            or ".." in relative_path.parts
        ):
            raise ManifestError(
                f"{path}:{line_number}: data/ 아래 상대 경로만 허용: {relative}"
            )
        if relative_path in seen:
            raise ManifestError(f"{path}:{line_number}: 중복 경로: {relative}")
        seen.add(relative_path)
        declared.append(DeclaredInput(sha256=sha256.lower(), path=relative_path))
    if not declared:
        raise ManifestError(f"{path}: 입력 선언이 비어 있음")
    return declared


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _symlink_component(root: Path, relative: Path) -> Path | None:
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return current.relative_to(root)
    return None


def _check(
    root: Path,
    manifest: Path,
    *,
    require_read_only: bool = True,
    reject_undeclared: bool = True,
) -> list[str]:
    declared = _load_manifest(manifest)
    problems: list[str] = []
    for item in declared:
        path = root / item.path
        linked = _symlink_component(root, item.path)
        if linked is not None:
            problems.append(f"심볼릭 링크: {linked.as_posix()}")
        elif not path.exists():
            problems.append(f"누락: {item.path.as_posix()}")
        elif not path.is_file():
            problems.append(f"일반 파일 아님: {item.path.as_posix()}")
        elif _sha256(path) != item.sha256:
            problems.append(f"해시 불일치: {item.path.as_posix()}")
        elif require_read_only and path.stat().st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            problems.append(f"쓰기 가능: {item.path.as_posix()}")
    if reject_undeclared:
        declared_paths = {item.path for item in declared}
        managed_roots = {Path(item.path.parts[0]) for item in declared}
        for managed_relative in sorted(managed_roots):
            managed = root / managed_relative
            if not managed.is_dir():
                continue
            for path in sorted(managed.rglob("*")):
                relative = path.relative_to(root)
                if (path.is_file() or path.is_symlink()) and relative not in declared_paths:
                    problems.append(f"목록 밖 입력: {relative.as_posix()}")
    return problems


def _prepare(source_root: Path, root: Path, manifest: Path) -> None:
    with tempfile.TemporaryDirectory(prefix=".private-inputs-", dir=root) as staging_name:
        staging = Path(staging_name)
        for item in _load_manifest(manifest):
            destination = staging / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_root / item.path, destination)
            destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        staging_problems = _check(staging, manifest)
        if staging_problems:
            raise PreparationError("준비 중 사본 검증 실패: " + "; ".join(staging_problems))
        (staging / "data").rename(root / "data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="새 작업 폴더의 비커밋 입력 준비 관문")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="현재 입력 상태를 검증한다")
    check.add_argument("--root", type=Path, default=Path.cwd(), help="검증할 작업 폴더")
    check.add_argument("--manifest", type=Path, default=Path("private-inputs.sha256"))
    prepare = subparsers.add_parser("prepare", help="선언된 비커밋 입력을 새 작업 폴더에 준비한다")
    prepare.add_argument("--source-root", type=Path, required=True, help="입력을 읽을 작업 폴더")
    prepare.add_argument("--root", type=Path, default=Path.cwd(), help="준비할 새 작업 폴더")
    prepare.add_argument("--manifest", type=Path, default=Path("private-inputs.sha256"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "prepare":
            if args.root.is_symlink() or not args.root.is_dir():
                raise PreparationError(f"대상 작업 폴더가 일반 디렉터리가 아님: {args.root}")
            target_data = args.root / "data"
            if target_data.exists() or target_data.is_symlink():
                raise PreparationError(f"대상 data/가 이미 존재: {target_data}")
            source_problems = _check(
                args.source_root,
                args.manifest,
                require_read_only=False,
                reject_undeclared=False,
            )
            if source_problems:
                print("원본 비커밋 입력 검증 실패:", file=sys.stderr)
                for problem in source_problems:
                    print(f"- {problem}", file=sys.stderr)
                raise SystemExit(1)
            _prepare(args.source_root, args.root, args.manifest)
        problems = _check(args.root, args.manifest)
    except (ManifestError, PreparationError, OSError) as exc:
        print(f"비커밋 입력 준비 실패: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    if problems:
        print("비커밋 입력 검증 실패:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        raise SystemExit(1)
    print("비커밋 입력 검증 통과")


if __name__ == "__main__":
    main()
