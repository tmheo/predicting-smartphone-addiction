"""PyTorch 모형 시험을 macOS OpenMP 충돌에서 격리하는 pytest 플러그인."""

from __future__ import annotations

from collections.abc import Generator, Iterable
import os
from pathlib import Path
import subprocess
import sys

import pytest


ISOLATED_FILE_ENV = "PIPELINE_PYTEST_ISOLATED_TORCH_FILE"
TORCH_TEST_FILE_PATTERN = "test_model_*.py"


def _is_torch_test_file(path: Path) -> bool:
    return path.match(TORCH_TEST_FILE_PATTERN)


def _is_selected_isolated_file(path: Path) -> bool:
    selected = os.environ.get(ISOLATED_FILE_ENV)
    return selected is not None and Path(selected).resolve() == path.resolve()


class IsolatedTorchTestFile(pytest.File):
    """모형 시험 파일 하나를 나타내는 부모 프로세스용 수집 항목."""

    def collect(self) -> Iterable[pytest.Item]:
        yield IsolatedTorchTestItem.from_parent(self, name="isolated-process")


class IsolatedTorchTestItem(pytest.Item):
    """실제 모형 시험 파일을 새 Python 프로세스에서 실행한다."""

    def runtest(self) -> None:
        test_file = Path(self.path).resolve()
        env = os.environ.copy()
        env[ISOLATED_FILE_ENV] = str(test_file)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "pipeline.pytest_openmp_guard",
                str(test_file),
            ],
            cwd=self.config.invocation_params.dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            pytest.fail(
                "격리된 PyTorch 모형 시험 파일이 실패했다.\n"
                f"종료 코드: {completed.returncode}\n"
                f"표준 출력:\n{completed.stdout}\n"
                f"표준 오류:\n{completed.stderr}",
                pytrace=False,
            )

    def reportinfo(self) -> tuple[Path, int | None, str]:
        return Path(self.path), None, "격리된 PyTorch 모형 시험"


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collect_file(
    file_path: Path, parent: pytest.Collector
) -> Generator[None, list[pytest.Collector], list[pytest.Collector]]:
    collectors = yield
    if _is_torch_test_file(file_path) and not _is_selected_isolated_file(file_path):
        return [IsolatedTorchTestFile.from_parent(parent, path=file_path)]
    return collectors


def pytest_collection_finish(session: pytest.Session) -> None:
    imported = sorted(
        name for name in sys.modules if name == "torch" or name.startswith("torch.")
    )
    if imported:
        raise pytest.UsageError(
            "시험 수집 단계에서 PyTorch를 적재했다. "
            "PyTorch 모형 시험은 test_model_*.py에 두고 필요한 시점에만 적재해야 한다. "
            f"최초 확인 모듈: {imported[0]}"
        )
