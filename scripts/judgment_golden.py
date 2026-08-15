"""판정 golden 박제: compare·pool CLI stdout의 특성화 기준선. (#92, 지도 #91)

판정 리팩토링(T2~T5)의 "출력 바이트 동일" 검증 도구다. 실제 artifacts와 MLflow의
기존 run들에 compare·pool CLI를 돌려 stdout·stderr·종료 코드를 시나리오별
golden 파일(tests/golden/judgment/*.txt)로 남기고, 리팩토링 후 같은 호출을
재실행해 diff한다.

사용법:
    uv run python scripts/judgment_golden.py            # 재실행 후 golden과 diff
    uv run python scripts/judgment_golden.py --update   # golden 갱신(박제)
    uv run python scripts/judgment_golden.py --only compare_screening

각 시나리오는 저장소 상태를 건드리지 않도록 임시 sandbox에서 돈다:
mlflow.db와 artifacts/(champion.yaml, pool.yaml, folds.parquet)는 사본,
data/는 읽기 전용이라 심볼릭 링크다. --adopt·--admit이 쓰는 파일과
pool의 탈락 태그 주석이 모두 사본에 떨어진다. MLflow run 산출물 경로는
DB에 절대 경로로 박혀 있어 sandbox에서도 원본 mlruns/를 읽는다.

시나리오 선정 원칙:
- 통과 경로는 실제 판정 이력의 재현이다: 스크리닝·확정 채택은 이슈 #90(exp057),
  풀 등록은 이슈 #56(exp058), 대리 스크리닝은 exp051→exp056 짝.
- 장부를 과거로 되감는 시나리오는 tests/golden/judgment/fixtures/의 고정본을 쓴다.
- 값싼 오류 경로(인자 위반, 자기 자신 비교, 부트스트랩 시드 위반 등)도 박제한다.
- 탈락 교체(_drop_member)를 일으키는 --admit 시나리오는 넣지 않는다.
  구성원 탈락은 실행 저장소에 태그 주석을 남기는데, 그 기록까지 sandbox 사본에
  가두는 것만 확인하면 충분하고 여기서는 굳이 일으킬 이유가 없다.

golden 파일에는 표준 출력과 표준 오류를 모두 남긴다. 판정 규칙의 특성화 대상은
stdout이고, sys.exit 메시지가 stderr로 나가므로 오류 경로의 특성화에는 stderr도
필요하다.
"""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO / "tests" / "golden" / "judgment"
FIXTURES = GOLDEN_DIR / "fixtures"

# 시나리오에 쓰는 실제 run들. mlflow.db에 있는 완료된 실행이다.
RUN = {
    "exp057_confirm": "3d5239b07c6e475baa7dde03fb0b99bc",  # 현 champion. 3시드.
    "exp057_screen": "dd1369ef67ba4f768a0819e97f8fdf0a",  # champion의 스크리닝 run. seed 42.
    "exp052_confirm": "2c6150364304443c813302d6fa19528d",  # 직전 champion. 3시드.
    "exp058_confirm": "e2b76edd9d204290810e39a7457c4c48",  # 풀 구성원(#56). 3시드.
    "exp051_proxy_base": "f139eeec4b2c495a9aeab56aba3ec665",  # 대리 스크리닝 기준 실행. seed 42.
    "exp056_proxy_chal": "32d5caccd76045d8b624f636c548a075",  # 대리 스크리닝 challenger. seed 42.
    "exp037_3seed": "007cab1f0a49414ea0e629d326789aae",  # 풀 비구성원 3시드. 기여 음수.
    "exp043_3seed": "737f4dae278c4615ba294372352ac5bb",  # 풀 비구성원 3시드. 중복 탈락.
    "exp022_screen": "aa9010fe9afe4f6aaa4913146fb6bb97",  # 풀 비구성원 단일 시드.
}


def _use_champion_exp052(sandbox: Path) -> None:
    """champion을 이슈 #90 채택 직전(exp052)으로 되감는다."""
    shutil.copy2(FIXTURES / "champion_exp052.yaml", sandbox / "artifacts" / "champion.yaml")


def _use_champion_exp057(sandbox: Path) -> None:
    """champion을 박제 당시(#92, exp057)로 고정한다.

    champion을 고정하지 않은 시나리오는 이후 채택(#58 exp059)마다 출력이 드리프트해
    특성화 기준선이 무너진다. 판정 경로가 champion 값에 의존하는 시나리오는 전부
    이 고정본 위에서 돈다. (#103에서 드리프트 발견)
    """
    shutil.copy2(FIXTURES / "champion_exp057.yaml", sandbox / "artifacts" / "champion.yaml")


def _use_champion_exp057_and_pool_before_exp058(sandbox: Path) -> None:
    _use_champion_exp057(sandbox)
    _use_pool_before_exp058(sandbox)


def _use_champion_exp057_and_pool_before_exp059(sandbox: Path) -> None:
    """champion과 풀 장부를 모두 박제 당시(#95)로 고정한다. 기여 판정처럼 풀 구성
    전체가 출력에 들어가는 시나리오는 풀 장부 드리프트(#58 exp059 등록)도 막아야 한다."""
    _use_champion_exp057(sandbox)
    shutil.copy2(FIXTURES / "pool_before_exp059.yaml", sandbox / "artifacts" / "pool.yaml")


def _use_pool_before_exp058(sandbox: Path) -> None:
    """풀 장부를 이슈 #56 등록 직전(exp058 제외 10명)으로 되감는다."""
    shutil.copy2(FIXTURES / "pool_before_exp058.yaml", sandbox / "artifacts" / "pool.yaml")


def _remove_champion(sandbox: Path) -> None:
    (sandbox / "artifacts" / "champion.yaml").unlink()


def _adopt_dirty_run(sandbox: Path) -> None:
    """채택 자격 검사(#95)용: champion을 되감고 challenger를 git_dirty=True로 바꾼다."""
    _use_champion_exp052(sandbox)
    with sqlite3.connect(sandbox / "mlflow.db") as conn:
        conn.execute(
            "UPDATE tags SET value = 'True' WHERE key = 'git_dirty' AND run_uuid = ?",
            (RUN["exp057_confirm"],),
        )


def _adopt_stale_folds(sandbox: Path) -> None:
    """채택 자격 검사(#95)용: champion을 되감고 folds 사본을 한 바이트 늘려 sha256을 어긋낸다."""
    _use_champion_exp052(sandbox)
    folds = sandbox / "artifacts" / "folds.parquet"
    folds.write_bytes(folds.read_bytes() + b"\0")


@dataclass(frozen=True)
class Scenario:
    name: str
    argv: list[str]  # python -m 뒤에 붙는 module과 인자.
    setup: Callable[[Path], None] | None = None


SCENARIOS = [
    # --- compare: 판정 리포트 경로 ---
    Scenario(
        "compare_screening",
        ["pipeline.compare", RUN["exp057_screen"]],
        setup=_use_champion_exp057,
    ),
    Scenario(
        "compare_confirmation_not_improved",
        ["pipeline.compare", RUN["exp052_confirm"]],
        setup=_use_champion_exp057,
    ),
    Scenario(
        "compare_confirmation_adopt",
        ["pipeline.compare", RUN["exp057_confirm"], "--adopt", "--reason", "골든 박제: 이슈 #90 채택 재현"],
        setup=_use_champion_exp052,
    ),
    Scenario(
        "compare_proxy_screening",
        ["pipeline.compare", RUN["exp056_proxy_chal"], "--proxy-baseline", RUN["exp051_proxy_base"]],
    ),
    # --- compare: 오류·거부 경로 ---
    Scenario(
        "compare_screening_adopt_refused",
        ["pipeline.compare", RUN["exp057_screen"], "--adopt", "--reason", "골든 박제: 거부 경로"],
        setup=_use_champion_exp057,
    ),
    Scenario(
        "compare_adopt_without_reason",
        ["pipeline.compare", RUN["exp057_screen"], "--adopt"],
    ),
    Scenario(
        "compare_proxy_with_adopt_refused",
        ["pipeline.compare", RUN["exp056_proxy_chal"], "--proxy-baseline", RUN["exp051_proxy_base"], "--adopt", "--reason", "골든 박제: 거부 경로"],
    ),
    Scenario(
        "compare_self_champion",
        ["pipeline.compare", RUN["exp057_confirm"]],
        setup=_use_champion_exp057,
    ),
    Scenario(
        "compare_bootstrap_hint",
        ["pipeline.compare", RUN["exp052_confirm"]],
        setup=_remove_champion,
    ),
    Scenario(
        "compare_bootstrap_seed_violation",
        ["pipeline.compare", RUN["exp057_screen"]],
        setup=_remove_champion,
    ),
    Scenario(
        "compare_adopt_dirty_refused",
        ["pipeline.compare", RUN["exp057_confirm"], "--adopt", "--reason", "골든 박제: 채택 자격 거부 경로"],
        setup=_adopt_dirty_run,
    ),
    Scenario(
        "compare_adopt_stale_folds_refused",
        ["pipeline.compare", RUN["exp057_confirm"], "--adopt", "--reason", "골든 박제: 채택 자격 거부 경로"],
        setup=_adopt_stale_folds,
    ),
    # --- pool: 판정 리포트 경로 ---
    Scenario(
        "pool_report_reject_contribution",
        ["pipeline.pool", RUN["exp037_3seed"]],
        setup=_use_champion_exp057_and_pool_before_exp059,
    ),
    Scenario(
        "pool_report_reject_duplicate",
        ["pipeline.pool", RUN["exp043_3seed"]],
        setup=_use_champion_exp057,
    ),
    Scenario(
        "pool_report_single_seed",
        ["pipeline.pool", RUN["exp022_screen"]],
        setup=_use_champion_exp057,
    ),
    Scenario(
        "pool_admit_replay_exp058",
        ["pipeline.pool", RUN["exp058_confirm"], "--admit", "--reason", "골든 박제: 이슈 #56 등록 재현"],
        setup=_use_champion_exp057_and_pool_before_exp058,
    ),
    # --- pool: 오류·거부 경로 ---
    Scenario(
        "pool_already_member",
        ["pipeline.pool", RUN["exp057_confirm"]],
    ),
    Scenario(
        "pool_admit_without_reason",
        ["pipeline.pool", RUN["exp058_confirm"], "--admit"],
    ),
    Scenario(
        "pool_no_champion",
        ["pipeline.pool", RUN["exp058_confirm"]],
        setup=_remove_champion,
    ),
]


def make_sandbox(root: Path) -> Path:
    """저장소 상태를 격리한 실행 디렉터리. 장부와 DB는 사본, 데이터는 링크."""
    sandbox = root / "sandbox"
    (sandbox / "artifacts").mkdir(parents=True)
    shutil.copy2(REPO / "mlflow.db", sandbox / "mlflow.db")
    for name in ("champion.yaml", "pool.yaml", "folds.parquet"):
        shutil.copy2(REPO / "artifacts" / name, sandbox / "artifacts" / name)
    (sandbox / "data").symlink_to(REPO / "data")
    return sandbox


def run_scenario(scenario: Scenario) -> str:
    with tempfile.TemporaryDirectory(prefix="judgment-golden-") as tmp:
        sandbox = make_sandbox(Path(tmp))
        if scenario.setup is not None:
            scenario.setup(sandbox)
        env = os.environ | {
            "PYTHONPATH": str(REPO / "src"),
            # 진행률 막대는 속도 표시가 비결정적이라 끈다.
            "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR": "false",
            "COLUMNS": "80",
        }
        result = subprocess.run(
            [sys.executable, "-m", *scenario.argv],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    return (
        f"$ python -m {' '.join(scenario.argv)}\n"
        f"exit: {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}"
        f"--- stderr ---\n{result.stderr}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="compare·pool CLI stdout 특성화 golden (#92)")
    parser.add_argument("--update", action="store_true", help="현재 출력을 golden으로 박제")
    parser.add_argument("--only", help="이 이름의 시나리오만 실행")
    args = parser.parse_args()

    scenarios = [s for s in SCENARIOS if args.only is None or s.name == args.only]
    if not scenarios:
        sys.exit(f"시나리오 없음: {args.only}")

    failed: list[str] = []
    for scenario in scenarios:
        golden_path = GOLDEN_DIR / f"{scenario.name}.txt"
        transcript = run_scenario(scenario)
        if args.update:
            golden_path.write_text(transcript)
            print(f"박제: {golden_path.relative_to(REPO)}")
            continue
        if not golden_path.exists():
            failed.append(scenario.name)
            print(f"[없음] {scenario.name}: golden 파일이 없다. --update로 먼저 박제할 것.")
            continue
        golden = golden_path.read_text()
        if transcript == golden:
            print(f"[동일] {scenario.name}")
        else:
            failed.append(scenario.name)
            diff = difflib.unified_diff(
                golden.splitlines(keepends=True),
                transcript.splitlines(keepends=True),
                fromfile=f"golden/{scenario.name}",
                tofile=f"현재/{scenario.name}",
            )
            print(f"[다름] {scenario.name}")
            sys.stdout.writelines(diff)

    if failed:
        sys.exit(f"golden과 다른 시나리오 {len(failed)}개: {', '.join(failed)}")
    if not args.update:
        print(f"시나리오 {len(scenarios)}개 모두 golden과 바이트 동일.")


if __name__ == "__main__":
    main()
