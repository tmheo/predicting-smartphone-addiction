"""결측 개수 구간별 OOF AUC 분해. (#360)

값 가리기 증강의 분포 형태를 바꾼 후보가 이득을 낸다면 결측이 많은 행에 몰릴
것이라는 예상을 확인하는 진단이다. 판정 자체는 `pipeline.compare`(nested OOF)와
`judgment.weighted_oof_auc`(가중 OOF)가 하고, 이 스크립트는 그 근거 옆에 남길
분해표만 만든다.

행별 결측 개수는 12개 설명변수(`id`, 목표값 제외) 기준이다.

사용법:
    uv run python scripts/decompose_missing_count_auc.py \\
        --train data/train.csv \\
        --oof champion=/path/to/oof.parquet \\
        --oof row_mask_fold=/path/to/oof.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.data import ID, TARGET  # noqa: E402

BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 12)]


def _bucket_label(low: int, high: int) -> str:
    return f"{low}" if low == high else f"{low}+"


def main() -> None:
    parser = argparse.ArgumentParser(description="결측 개수 구간별 OOF AUC 분해 (#360)")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--oof",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="이름과 oof parquet 경로. 여러 번 줄 수 있다.",
    )
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    explanatory = [column for column in train.columns if column not in {ID, TARGET}]
    missing_count = train[explanatory].isna().sum(axis=1)
    frame = pd.DataFrame(
        {ID: train[ID], TARGET: train[TARGET], "missing_count": missing_count}
    ).set_index(ID)

    predictions: dict[str, pd.Series] = {}
    for spec in args.oof:
        name, _, raw_path = spec.partition("=")
        if not name or not raw_path:
            sys.exit(f"--oof는 NAME=PATH 형식이어야 한다: {spec!r}")
        oof = pd.read_parquet(raw_path)
        series = pd.Series(oof["pred"].to_numpy(), index=pd.Index(oof[ID], name=ID))
        predictions[name] = series.reindex(frame.index)
        if predictions[name].isna().any():
            sys.exit(f"{name}의 OOF 예측에 train 행이 빠져 있다.")

    names = list(predictions)
    header = ["구간", "행 수", "구성비"] + names
    if len(names) > 1:
        header += [f"{name} - {names[0]}" for name in names[1:]]
    rows = []
    for low, high in BUCKETS:
        selected = frame["missing_count"].between(low, high)
        subset = frame[selected]
        if subset.empty or subset[TARGET].nunique() < 2:
            continue
        aucs = {
            name: roc_auc_score(subset[TARGET], predictions[name][selected])
            for name in names
        }
        row = [
            _bucket_label(low, high),
            f"{len(subset)}",
            f"{len(subset) / len(frame):.4f}",
        ] + [f"{aucs[name]:.7f}" for name in names]
        if len(names) > 1:
            row += [f"{aucs[name] - aucs[names[0]]:+.7f}" for name in names[1:]]
        rows.append(row)

    overall = {
        name: roc_auc_score(frame[TARGET], predictions[name]) for name in names
    }
    row = ["전체", f"{len(frame)}", "1.0000"] + [
        f"{overall[name]:.7f}" for name in names
    ]
    if len(names) > 1:
        row += [f"{overall[name] - overall[names[0]]:+.7f}" for name in names[1:]]
    rows.append(row)

    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))
    ]
    print(" | ".join(h.rjust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(c.rjust(w) for c, w in zip(r, widths)))

    print()
    print("행별 결측 개수 기준 열:", ", ".join(explanatory))
    print("전체 평균 결측 개수:", f"{float(np.mean(missing_count)):.4f}")


if __name__ == "__main__":
    main()
