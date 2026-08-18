from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "remote_python_contract"
RUNNER = ROOT / "scripts" / "run_remote_python.sh"
IMAGE_CHECKER = ROOT / "scripts" / "verify_remote_image_python.sh"
PEP668_IMAGE = "node@sha256:f32b81066cde10a75dbac96646099533316d94bac4150c55da1636e1f0ffdc46"


def _export_macos_keychain_certificates(output: Path) -> None:
    """회사 TLS 검사 인증서를 Docker 안의 공개 CA 묶음에 보탤 수 있게 내보낸다."""
    if sys.platform != "darwin" or shutil.which("security") is None:
        return
    keychains = [
        Path.home() / "Library/Keychains/login.keychain-db",
        Path("/Library/Keychains/System.keychain"),
        Path("/System/Library/Keychains/SystemRootCertificates.keychain"),
    ]
    with output.open("wb") as stream:
        completed = subprocess.run(
            [
                "security",
                "find-certificate",
                "-a",
                "-p",
                *(str(path) for path in keychains if path.exists()),
            ],
            stdout=stream,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0 or b"-----BEGIN CERTIFICATE-----" not in output.read_bytes():
        pytest.fail(
            "macOS 키체인 공개 인증서를 내보내지 못했다: "
            + completed.stderr.decode(errors="replace")
        )


def _fake_docker(tmp_path: Path) -> Path:
    executable = tmp_path / "docker"
    executable.write_text(
        """#!/bin/sh
set -eu
if [ "$1" = info ]; then
    exit 0
fi
printf '%s\\n' "$@" > "$DOCKER_TRACE"
exit "${DOCKER_RUN_STATUS:-0}"
"""
    )
    executable.chmod(0o755)
    return executable


def test_target_image_python_gate_checks_the_exact_image(tmp_path: Path) -> None:
    _fake_docker(tmp_path)
    trace = tmp_path / "docker.trace"
    environment = {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "DOCKER_TRACE": str(trace),
    }

    completed = subprocess.run(
        [
            str(IMAGE_CHECKER),
            "--platform",
            "linux/amd64",
            "registry.example/gpu@sha256:fixed",
        ],
        text=True,
        capture_output=True,
        env=environment,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    arguments = trace.read_text().splitlines()
    assert arguments[:7] == [
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--entrypoint",
        "sh",
        "registry.example/gpu@sha256:fixed",
    ]
    assert arguments[7] == "-c"
    probe = "\n".join(arguments[8:])
    assert 'python3 -m venv "$probe_root"' in probe
    assert '"$probe_root/bin/python" -m pip --version' in probe


def test_target_image_python_gate_propagates_an_incompatible_image(tmp_path: Path) -> None:
    _fake_docker(tmp_path)
    environment = {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "DOCKER_TRACE": str(tmp_path / "docker.trace"),
        "DOCKER_RUN_STATUS": "41",
    }

    completed = subprocess.run(
        [str(IMAGE_CHECKER), "registry.example/gpu:missing-venv"],
        text=True,
        capture_output=True,
        env=environment,
        timeout=10,
    )

    assert completed.returncode == 41


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker is required")
def test_remote_runner_uses_a_locked_virtual_environment_under_pep668(tmp_path: Path) -> None:
    evidence = tmp_path / "environment.json"
    trace = tmp_path / "system-python.trace"
    entry_marker = tmp_path / "entry-diagnostic-started"
    _export_macos_keychain_certificates(tmp_path / "host-keychain-ca.pem")
    container_script = """
set -eu
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    python3 python3-pip python3-venv >/dev/null
if [ -s /result/host-keychain-ca.pem ]; then
    cat /etc/ssl/certs/ca-certificates.crt /result/host-keychain-ca.pem \
        > /tmp/python-ca-certificates.pem
    export PIP_CERT=/tmp/python-ca-certificates.pem
    export SSL_CERT_FILE=/tmp/python-ca-certificates.pem
fi
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
