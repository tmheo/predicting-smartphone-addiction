"""TALENT 기본 Trompt 구조를 작은 공개 이진 자료에서 검증한다. (#145)"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from pipeline.config import ModelConfig
from pipeline.model import TromptAdapter, create


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trompt 공개 이진 자료 건전성 검사")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.astype("float64")
    X.iloc[::17, 0] = np.nan
    X["mean radius_missing"] = X["mean radius"].isna().astype("float32")
    y = bunch.target.astype("int64")
    train_index, validation_index = train_test_split(
        np.arange(len(X)), test_size=0.25, stratify=y, random_state=42
    )
    params = {
        "prompts": 128,
        "width": 128,
        "cells": 6,
        "epochs": 15,
        "batch_size": 64,
        "eval_batch_size": 256,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "patience": 5,
        "perm_sample": 64,
        "perm_repeats": 1,
    }
    if args.device is not None:
        params["device"] = args.device
    adapter = create(ModelConfig(kind="trompt", params=params, fit={}), seed=42)
    assert isinstance(adapter, TromptAdapter)
    prediction = adapter.fit(
        X.iloc[train_index],
        y.iloc[train_index],
        X.iloc[validation_index],
        y.iloc[validation_index],
    )
    diagnostics = adapter.entry_diagnostics().observations
    losses = diagnostics["training_losses"]
    auc = float(roc_auc_score(y.iloc[validation_index], prediction))
    passed = bool(losses[-1] < losses[0] and auc > 0.5)
    result = {
        "dataset": "sklearn.datasets.load_breast_cancer",
        "rows": len(X),
        "train_rows": len(train_index),
        "validation_rows": len(validation_index),
        "model": {key: params[key] for key in ("prompts", "width", "cells")},
        "learning_rate": params["lr"],
        "weight_decay": params["weight_decay"],
        "loss_first": losses[0],
        "loss_last": losses[-1],
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
