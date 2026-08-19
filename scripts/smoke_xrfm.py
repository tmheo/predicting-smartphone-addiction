"""xRFM 재귀 특성 커널 머신을 작은 공개 이진 자료에서 검증한다. (#198)"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from pipeline.config import ModelConfig
from pipeline.model import XRFMAdapter, create


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="xRFM 공개 이진 자료 건전성 검사")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.astype("float64")
    X.iloc[::17, 0] = np.nan
    X["grade"] = pd.Categorical(
        np.where(bunch.data["mean area"] > bunch.data["mean area"].median(), "big", "small")
    )
    y = bunch.target.astype("int64")
    train_index, validation_index = train_test_split(
        np.arange(len(X)), test_size=0.25, stratify=y, random_state=42
    )
    params = {
        "max_leaf_size": 2000,
        "inner_val_frac": 0.25,
        "eval_batch_size": 256,
        "perm_sample": 64,
        "verbose": False,
    }
    if args.device is not None:
        params["device"] = args.device
    adapter = create(ModelConfig(kind="xrfm", params=params, fit={}), seed=42)
    assert isinstance(adapter, XRFMAdapter)
    prediction = adapter.fit(
        X.iloc[train_index],
        y.iloc[train_index],
        X.iloc[validation_index],
        y.iloc[validation_index],
    )
    diagnostics = adapter.entry_diagnostics().observations
    importance = adapter.importance()
    auc = float(roc_auc_score(y.iloc[validation_index], prediction))
    passed = bool(
        auc > 0.95
        and np.isfinite(prediction).all()
        and np.isfinite(importance["gain"]).all()
    )
    result = {
        "dataset": "sklearn.datasets.load_breast_cancer",
        "rows": len(X),
        "train_rows": len(train_index),
        "validation_rows": len(validation_index),
        "validation_auc": auc,
        "passed": passed,
        "diagnostics": diagnostics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"passed": passed, "validation_auc": auc}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
