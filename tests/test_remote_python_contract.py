from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "remote_python_contract"
RUNNER = ROOT / "scripts" / "run_remote_python.sh"
PEP668_IMAGE = "node@sha256:f32b81066cde10a75dbac96646099533316d94bac4150c55da1636e1f0ffdc46"


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker is required")
def test_remote_runner_uses_a_locked_virtual_environment_under_pep668(tmp_path: Path) -> None:
    evidence = tmp_path / "environment.json"
    trace = tmp_path / "system-python.trace"
    entry_marker = tmp_path / "entry-diagnostic-started"
    container_script = """
set -eu
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    python3 python3-pip python3-venv >/dev/null
if python3 -m pip install wheel >/tmp/system-pip.out 2>&1; then
    echo "system pip unexpectedly accepted an install" >&2
    exit 1
fi
grep -q externally-managed-environment /tmp/system-pip.out
PATH=/contract-fixture:$PATH \
SYSTEM_PYTHON_TRACE=/result/system-python.trace \
/repo/scripts/run_remote_python.sh \
    --system-python system-python \
    --project /contract-fixture \
    --venv /tmp/contract-venv \
    --evidence /result/environment.json \
    -- \
    -m pipeline.entry_diagnostic \
    --marker /result/entry-diagnostic-started \
    --expected-python-prefix /tmp/contract-venv
"""

    container_name = f"remote-python-contract-{uuid.uuid4().hex}"
    try:
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "--volume",
                f"{ROOT}:/repo:ro",
                "--volume",
                f"{FIXTURE}:/contract-fixture:ro",
                "--volume",
                f"{tmp_path}:/result",
                PEP668_IMAGE,
                "sh",
                "-c",
                container_script,
            ],
            text=True,
            capture_output=True,
            timeout=180,
        )
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert entry_marker.exists()
    assert trace.read_text().splitlines() == ["-m venv /tmp/contract-venv"]
    recorded = json.loads(evidence.read_text())
    assert recorded["python_executable"] == "/tmp/contract-venv/bin/python"
    assert recorded["installer"] == {"name": "uv", "version": "0.11.7"}
    assert recorded["packages"]["idna"] == "3.10"
    assert "--break-system-packages" not in RUNNER.read_text()


def test_remote_runner_rejects_a_stale_lock_before_model_entry(tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            '    "idna==3.10",\n',
            '    "idna==3.10",\n    "certifi==2025.8.3",\n',
        )
    )
    venv = tmp_path / "venv"
    evidence = tmp_path / "environment.json"
    model_marker = tmp_path / "model-entered"

    completed = subprocess.run(
        [
            str(RUNNER),
            "--system-python",
            sys.executable,
            "--project",
            str(project),
            "--venv",
            str(venv),
            "--evidence",
            str(evidence),
            "--",
            "-c",
            f"from pathlib import Path; Path({str(model_marker)!r}).touch()",
        ],
        text=True,
        capture_output=True,
        timeout=90,
    )

    assert completed.returncode != 0
    assert "dependency lock is not current" in completed.stderr
    assert not evidence.exists()
    assert not model_marker.exists()
