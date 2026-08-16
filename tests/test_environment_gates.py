from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts" / "verify_environment_gates.sh"


def _fake_commands(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    trace = tmp_path / "commands.trace"
    uv = bin_dir / "uv"
    uv.write_text(
        """#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$ENVIRONMENT_GATE_TRACE"
case "$*" in
  *"pipeline.private_inputs prepare"*) touch "$ENVIRONMENT_GATE_STATE/prepared" ;;
  *"pipeline.private_inputs check"*)
    test -f "$ENVIRONMENT_GATE_STATE/prepared"
    touch "$ENVIRONMENT_GATE_STATE/checked"
    ;;
  *"pytest --collect-only"*)
    test -f "$ENVIRONMENT_GATE_STATE/checked"
    touch "$ENVIRONMENT_GATE_STATE/collected"
    ;;
  *"pytest tests/test_remote_python_contract.py"*)
    test -f "$ENVIRONMENT_GATE_STATE/collected"
    touch "$ENVIRONMENT_GATE_STATE/remote"
    ;;
  *"pytest"*)
    test -f "$ENVIRONMENT_GATE_STATE/remote"
    touch "$ENVIRONMENT_GATE_STATE/full-suite"
    ;;
  *) exit 64 ;;
esac
""",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/sh\nset -eu\n[ \"$1\" = info ]\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    return bin_dir, trace


def test_environment_gate_command_checks_every_boundary_before_the_full_suite(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "fresh-worktree"
    (worktree / "scripts").mkdir(parents=True)
    shutil.copy2(VERIFY, worktree / "scripts" / VERIFY.name)
    (worktree / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (worktree / "private-inputs.sha256").write_text(
        "0" * 64 + "  data/train.csv\n",
        encoding="utf-8",
    )
    source = tmp_path / "verified-source"
    source.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    bin_dir, trace = _fake_commands(tmp_path)
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join((str(bin_dir), env["PATH"]))
    env["ENVIRONMENT_GATE_TRACE"] = str(trace)
    env["ENVIRONMENT_GATE_STATE"] = str(state)

    completed = subprocess.run(
        [str(worktree / "scripts" / VERIFY.name), "--source-root", str(source)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (state / "full-suite").exists()
    assert trace.read_text().splitlines() == [
        "run --frozen python -m pipeline.private_inputs prepare "
        f"--source-root {source}",
        "run --frozen python -m pipeline.private_inputs check",
        "run --frozen pytest --collect-only",
        "run --frozen pytest tests/test_remote_python_contract.py",
        "run --frozen pytest",
    ]
