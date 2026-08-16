from __future__ import annotations

import hashlib
import stat
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[1]


def _manifest(tmp_path: Path, files: dict[str, bytes]) -> Path:
    manifest = tmp_path / "private-inputs.sha256"
    lines = [f"{hashlib.sha256(content).hexdigest()}  {name}" for name, content in files.items()]
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pipeline.private_inputs", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_check_reports_every_missing_declared_input(tmp_path: Path) -> None:
    files = {
        "data/train.csv": b"train\n",
        "data/test.csv": b"test\n",
    }
    manifest = _manifest(tmp_path, files)
    target = tmp_path / "fresh-worktree"
    target.mkdir()

    result = _run("check", "--root", str(target), "--manifest", str(manifest))

    assert result.returncode != 0
    assert "data/train.csv" in result.stderr
    assert "data/test.csv" in result.stderr


def test_prepare_copies_exactly_the_declared_inputs(tmp_path: Path) -> None:
    files = {
        "data/train.csv": b"train\n",
        "data/external/proxy.csv": b"proxy\n",
    }
    manifest = _manifest(tmp_path, files)
    source = tmp_path / "source"
    target = tmp_path / "fresh-worktree"
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    extra = source / "data/external/stale.csv"
    extra.write_bytes(b"stale\n")
    target.mkdir()

    result = _run(
        "prepare",
        "--source-root",
        str(source),
        "--root",
        str(target),
        "--manifest",
        str(manifest),
    )

    prepared = {
        path.relative_to(target).as_posix(): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file()
    }
    assert result.returncode == 0, result.stderr
    assert prepared == files


def test_check_rejects_hash_mismatch(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, {"data/train.csv": b"expected\n"})
    target = tmp_path / "worktree"
    path = target / "data/train.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"stale\n")

    result = _run("check", "--root", str(target), "--manifest", str(manifest))

    assert result.returncode != 0
    assert "해시 불일치: data/train.csv" in result.stderr


def test_prepare_rejects_source_hash_mismatch_before_copying(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, {"data/train.csv": b"expected\n"})
    source = tmp_path / "source"
    source_path = source / "data/train.csv"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"stale\n")
    target = tmp_path / "fresh-worktree"
    target.mkdir()

    result = _run(
        "prepare",
        "--source-root",
        str(source),
        "--root",
        str(target),
        "--manifest",
        str(manifest),
    )

    assert result.returncode != 0
    assert "해시 불일치: data/train.csv" in result.stderr
    assert not (target / "data").exists()


@pytest.mark.parametrize("linked_component", ["data", "data/train.csv"])
def test_check_rejects_symlinks_in_declared_input_path(
    tmp_path: Path, linked_component: str
) -> None:
    content = b"train\n"
    manifest = _manifest(tmp_path, {"data/train.csv": content})
    target = tmp_path / "worktree"
    external = tmp_path / "external"
    external.mkdir()
    if linked_component == "data":
        (external / "train.csv").write_bytes(content)
        target.mkdir()
        (target / "data").symlink_to(external, target_is_directory=True)
    else:
        external_file = external / "train.csv"
        external_file.write_bytes(content)
        (target / "data").mkdir(parents=True)
        (target / "data/train.csv").symlink_to(external_file)

    result = _run("check", "--root", str(target), "--manifest", str(manifest))

    assert result.returncode != 0
    assert f"심볼릭 링크: {linked_component}" in result.stderr


def test_check_rejects_non_regular_declared_input(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, {"data/train.csv": b"train\n"})
    target = tmp_path / "worktree"
    (target / "data/train.csv").mkdir(parents=True)

    result = _run("check", "--root", str(target), "--manifest", str(manifest))

    assert result.returncode != 0
    assert "일반 파일 아님: data/train.csv" in result.stderr


def test_check_rejects_writable_declared_input(tmp_path: Path) -> None:
    content = b"train\n"
    manifest = _manifest(tmp_path, {"data/train.csv": content})
    target = tmp_path / "worktree"
    path = target / "data/train.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    path.chmod(path.stat().st_mode | stat.S_IWUSR)

    result = _run("check", "--root", str(target), "--manifest", str(manifest))

    assert result.returncode != 0
    assert "쓰기 가능: data/train.csv" in result.stderr


def test_check_rejects_undeclared_input(tmp_path: Path) -> None:
    content = b"train\n"
    manifest = _manifest(tmp_path, {"data/train.csv": content})
    target = tmp_path / "worktree"
    declared = target / "data/train.csv"
    declared.parent.mkdir(parents=True)
    declared.write_bytes(content)
    declared.chmod(0o444)
    (target / "data/stale.csv").write_bytes(b"stale\n")

    result = _run("check", "--root", str(target), "--manifest", str(manifest))

    assert result.returncode != 0
    assert "목록 밖 입력: data/stale.csv" in result.stderr


def test_manifest_cannot_declare_path_outside_data(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, {"../secret.csv": b"secret\n"})
    target = tmp_path / "worktree"
    target.mkdir()

    result = _run("check", "--root", str(target), "--manifest", str(manifest))

    assert result.returncode != 0
    assert "data/ 아래 상대 경로만 허용" in result.stderr
    assert "Traceback" not in result.stderr


def test_prepare_refuses_existing_data_without_modifying_it(tmp_path: Path) -> None:
    expected = b"expected\n"
    manifest = _manifest(tmp_path, {"data/train.csv": expected})
    source = tmp_path / "source"
    source_path = source / "data/train.csv"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(expected)
    target = tmp_path / "existing-worktree"
    target_path = target / "data/train.csv"
    target_path.parent.mkdir(parents=True)
    target_path.write_bytes(b"keep\n")

    result = _run(
        "prepare",
        "--source-root",
        str(source),
        "--root",
        str(target),
        "--manifest",
        str(manifest),
    )

    assert result.returncode != 0
    assert "대상 data/가 이미 존재" in result.stderr
    assert target_path.read_bytes() == b"keep\n"


def test_repository_manifest_declares_the_verified_private_inputs() -> None:
    expected = {
        "data/train.csv": "f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c",
        "data/test.csv": "8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e",
        "data/sample_submission.csv": "206763fe5786fb9c80d4e9289a3b812030d3dbb36450c6eb63348098154ce63e",
        "data/external/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv": (
            "2194ce1946e8559f26780049c8d972857d8378104f2c9ec25ed9ec35409f1074"
        ),
    }

    actual = {}
    for line in (REPO_ROOT / "private-inputs.sha256").read_text().splitlines():
        sha256, path = line.split(maxsplit=1)
        actual[path] = sha256

    assert actual == expected
