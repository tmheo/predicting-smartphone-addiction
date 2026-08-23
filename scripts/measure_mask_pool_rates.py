"""값 가리기 증강의 기증 마스크 풀 계보 측정. (#360)

champion `exp131_lookup_bivariate_plr5`의 피처 계획으로 fold 0의 학습 행렬과 test
행렬을 만들고, 인코딩 열 집합에서 열별·셀 단위 결측률과 `alpha` 후보를 인쇄한다.
표본기 계약의 `alpha` 정규화가 실제 자료에서 어떤 값이 되는지 실행 전에 확인하는
진단이며, 판정에는 쓰지 않는다. 구현은 풀에 결측이 있는 열(기증 열)에서만 alpha를
재므로 "기증 열 기준" 값이 실제로 쓰이는 값이다.

사용법:
    uv run python scripts/measure_mask_pool_rates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pipeline.config import load_config  # noqa: E402
from pipeline.plan import FeaturePlan  # noqa: E402
from screen_schedule_length import build_fold0_matrices  # noqa: E402

CONFIG = "configs/exp131_lookup_bivariate_plr5.yaml"


def report(name: str, frame: pd.DataFrame, value_dropout: float) -> None:
    rate = frame.isna().mean()
    missing_cols = rate[rate > 0].sort_values(ascending=False)
    cell_rate_all = float(frame.isna().to_numpy().mean())
    cell_rate_missing_cols = (
        float(frame[missing_cols.index].isna().to_numpy().mean())
        if len(missing_cols)
        else 0.0
    )
    print(f"\n== {name} ==")
    print(f"행 {len(frame)}, 열 {frame.shape[1]}, 기증 열 {len(missing_cols)}")
    print(f"전체 열 셀 단위 결측률 : {cell_rate_all:.6f}")
    print(f"기증 열 셀 단위 결측률 : {cell_rate_missing_cols:.6f}")
    print(f"alpha(전체 열 기준)    : {min(1.0, value_dropout / cell_rate_all):.6f}")
    if cell_rate_missing_cols > 0:
        print(
            "alpha(기증 열 기준)    : "
            f"{min(1.0, value_dropout / cell_rate_missing_cols):.6f}"
        )
    print("열별 결측률:")
    for column, value in missing_cols.items():
        print(f"  {column:40s} {value:.6f}")


def main() -> None:
    cfg = load_config(CONFIG, "screen")
    plan = FeaturePlan.from_config(cfg.features)
    X_fold, X_test_fold, _, tr_idx, _ = build_fold0_matrices(cfg, plan)
    value_dropout = float(cfg.model.params.get("value_dropout", 0.10))
    report("fold 0 학습 행(360-a 풀)", X_fold.loc[tr_idx], value_dropout)
    report("test 행(360-b 풀)", X_test_fold, value_dropout)


if __name__ == "__main__":
    main()
