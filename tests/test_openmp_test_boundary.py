"""macOS OpenMP 충돌을 막는 시험 프로세스 경계의 종단 시험."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ISOLATED_FILE_ENV = "PIPELINE_PYTEST_ISOLATED_TORCH_FILE"


def _run_probe(test_file: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPOSITORY_ROOT)
        if not pythonpath
        else os.pathsep.join((str(REPOSITORY_ROOT), pythonpath))
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-vv",
            "-c",
            os.devnull,
            "-p",
            "pipeline.pytest_openmp_guard",
            str(test_file),
        ],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_model_test_file_runs_in_an_isolated_python_process(tmp_path: Path) -> None:
    test_file = tmp_path / "test_model_isolated_probe.py"
    test_file.write_text(
        textwrap.dedent(
            f"""
            import os
            from pathlib import Path

            def test_isolated_file_environment_names_this_file():
                assert os.environ[{ISOLATED_FILE_ENV!r}] == str(Path(__file__).resolve())
            """
        ),
        encoding="utf-8",
    )

    completed = _run_probe(test_file)

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_model_test_file_cannot_import_torch_during_collection(tmp_path: Path) -> None:
    test_file = tmp_path / "test_model_imports_torch.py"
    test_file.write_text(
        "import torch\n\ndef test_unreachable():\n    pass\n",
        encoding="utf-8",
    )

    completed = _run_probe(test_file)

    assert completed.returncode != 0
    assert "시험 수집 단계에서 PyTorch를 적재했다" in completed.stdout + completed.stderr


@pytest.mark.parametrize(
    ("package_name", "training_code"),
    [
        (
            "xgboost",
            """
            import xgboost as tree_library
            tree_library.XGBClassifier(n_estimators=2, n_jobs=2).fit(X, y)
            """,
        ),
        (
            "lightgbm",
            """
            import lightgbm as tree_library
            tree_library.LGBMClassifier(
                n_estimators=2, n_jobs=2, verbosity=-1
            ).fit(X, y)
            """,
        ),
    ],
)
def test_dangerous_openmp_order_cannot_terminate_parent_pytest(
    tmp_path: Path, package_name: str, training_code: str
) -> None:
    test_file = tmp_path / f"test_model_openmp_{package_name}.py"
    test_body = textwrap.dedent(
        """
        import numpy as np
        import torch

        torch.set_num_threads(4)
        left = torch.randn(512, 512)
        right = torch.randn(512, 512)
        _ = left @ right
        X = np.arange(256, dtype=np.float32).reshape(64, 4)
        y = np.array([0, 1] * 32)
        """
    ) + textwrap.dedent(training_code)
    test_file.write_text(
        "def test_torch_cpu_work_before_tree_training():\n"
        + textwrap.indent(test_body, "    "),
        encoding="utf-8",
    )

    completed = _run_probe(test_file)
    output = completed.stdout + completed.stderr

    assert completed.returncode >= 0, output
    if completed.returncode != 0:
        child_signal = re.search(r"종료 코드: (-\d+)", output)
        assert child_signal is not None, output
        assert int(child_signal.group(1)) < 0
