"""Read-only RepLeafGBM entry diagnostic against the frozen 313-member OOF.

The public OOF comes from masayakawamata/s6e8-repleafgbm-cv-0-968187.
This script does not treat the external array as a champion candidate. Folds 0
and 1 select one predeclared blend arm. Folds 2, 3, and 4 audit that frozen arm.
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


ID = "id"
TARGET = "addicted_label"
SCREEN_FOLDS = (0, 1)
HOLDOUT_FOLDS = (2, 3, 4)
WEIGHTS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
BLEND_KINDS = ("raw", "logit", "rank")
EPS = 1e-6
BASELINE_AUC = 0.970350946943525
HOLDOUT_MEAN_DELTA_MIN = 0.00001
SOURCE_URL = "https://www.kaggle.com/code/masayakawamata/s6e8-repleafgbm-cv-0-968187"


def find_input(name: str) -> Path:
    hits = [Path(path) for path in glob.glob(f"/kaggle/input/**/{name}", recursive=True)]
    if len(hits) != 1:
        raise SystemExit(f"Expected one {name}, found {len(hits)}: {hits}")
    return hits[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_rank(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") - 0.5) / len(values)


def blend_prediction(
    baseline: np.ndarray,
    candidate: np.ndarray,
    kind: str,
    weight: float,
) -> np.ndarray:
    if kind == "raw":
        return (1.0 - weight) * baseline + weight * candidate
    if kind == "logit":
        return expit(
            (1.0 - weight) * logit(np.clip(baseline, EPS, 1 - EPS))
            + weight * logit(np.clip(candidate, EPS, 1 - EPS))
        )
    if kind == "rank":
        return (
            (1.0 - weight) * normalized_rank(baseline)
            + weight * normalized_rank(candidate)
        )
    raise AssertionError(f"Unknown blend kind: {kind}")


def score_arm(
    y: np.ndarray,
    folds: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    valid_fold: int,
    kind: str,
    weight: float,
) -> dict[str, object]:
    mask = folds == valid_fold
    baseline_auc = float(roc_auc_score(y[mask], baseline[mask]))
    candidate_auc = float(roc_auc_score(y[mask], candidate[mask]))
    prediction = blend_prediction(
        baseline[mask], candidate[mask], kind=kind, weight=weight
    )
    auc = float(roc_auc_score(y[mask], prediction))
    return {
        "fold": valid_fold,
        "kind": kind,
        "weight": weight,
        "baseline_auc": baseline_auc,
        "candidate_auc": candidate_auc,
        "blend_auc": auc,
        "delta": auc - baseline_auc,
    }


def main() -> None:
    train_path = find_input("train.csv")
    baseline_path = find_input("current313_oof.parquet")
    candidate_path = find_input("oof_repleaf.csv")

    train = pd.read_csv(train_path, usecols=[ID, TARGET])
    baseline_frame = pd.read_parquet(baseline_path)
    candidate_frame = pd.read_csv(candidate_path)
    if list(baseline_frame.columns) != [ID, "fold", "pred"]:
        raise SystemExit(f"Unexpected baseline columns: {list(baseline_frame.columns)}")
    if [str(column) for column in candidate_frame.columns] != [ID, "0", "1"]:
        raise SystemExit(f"Unexpected candidate columns: {list(candidate_frame.columns)}")
    if not (
        len(train) == len(baseline_frame) == len(candidate_frame)
        and np.array_equal(train[ID].to_numpy(), baseline_frame[ID].to_numpy())
        and np.array_equal(train[ID].to_numpy(), candidate_frame[ID].to_numpy())
    ):
        raise SystemExit("Train, baseline, and RepLeafGBM row identities do not match")

    y = train[TARGET].to_numpy(dtype=np.int8)
    folds = baseline_frame["fold"].to_numpy(dtype=np.int8)
    baseline = baseline_frame["pred"].to_numpy(dtype=np.float64)
    candidate = candidate_frame["1"].to_numpy(dtype=np.float64)
    baseline_auc = float(roc_auc_score(y, baseline))
    candidate_auc = float(roc_auc_score(y, candidate))
    if abs(baseline_auc - BASELINE_AUC) > 1e-12:
        raise SystemExit(f"Baseline AUC mismatch: {baseline_auc}")

    screen_rows = [
        score_arm(y, folds, baseline, candidate, fold, kind, weight)
        for fold in SCREEN_FOLDS
        for kind in BLEND_KINDS
        for weight in WEIGHTS
    ]
    screen = pd.DataFrame(screen_rows)
    wide = screen.pivot(index=["kind", "weight"], columns="fold", values="delta")
    wide.columns = [f"delta_fold{int(column)}" for column in wide.columns]
    screen_delta_columns = list(wide.columns)
    wide["selection_score"] = wide[screen_delta_columns].min(axis=1)
    wide["positive_screen_folds"] = (wide[screen_delta_columns] > 0).sum(axis=1)
    wide = wide.sort_values(
        ["positive_screen_folds", "selection_score", "weight"],
        ascending=[False, False, True],
    )
    chosen_kind, chosen_weight = wide.index[0]
    had_joint_positive = bool(wide.iloc[0]["positive_screen_folds"] == len(SCREEN_FOLDS))

    holdout_rows = [
        score_arm(
            y,
            folds,
            baseline,
            candidate,
            fold,
            str(chosen_kind),
            float(chosen_weight),
        )
        for fold in HOLDOUT_FOLDS
    ]
    holdout = pd.DataFrame(holdout_rows)
    holdout_positive_folds = int((holdout["delta"] > 0).sum())
    holdout_mean_delta = float(holdout["delta"].mean())
    passed = bool(
        had_joint_positive
        and holdout_positive_folds == len(HOLDOUT_FOLDS)
        and holdout_mean_delta >= HOLDOUT_MEAN_DELTA_MIN
    )

    screen.to_csv("/kaggle/working/screen_fold_scores.csv", index=False)
    wide.to_csv("/kaggle/working/screen_arm_summary.csv")
    holdout.to_csv("/kaggle/working/holdout_scores.csv", index=False)
    payload = {
        "schema": "repleaf-313-entry-screen/1",
        "purpose": "read_only_diagnostic_not_improvement_judgment",
        "source_url": SOURCE_URL,
        "train_sha256": file_sha256(train_path),
        "baseline_sha256": file_sha256(baseline_path),
        "candidate_sha256": file_sha256(candidate_path),
        "rows": len(train),
        "baseline_auc": baseline_auc,
        "candidate_auc": candidate_auc,
        "spearman_with_baseline": float(spearmanr(baseline, candidate).statistic),
        "screen_folds": SCREEN_FOLDS,
        "holdout_folds": HOLDOUT_FOLDS,
        "blend_kinds": BLEND_KINDS,
        "weights": WEIGHTS,
        "chosen_arm": {"kind": chosen_kind, "weight": float(chosen_weight)},
        "had_joint_positive_on_screen": had_joint_positive,
        "screen_deltas": {
            column: float(wide.iloc[0][column]) for column in screen_delta_columns
        },
        "holdout_deltas": {
            f"fold{int(row.fold)}": float(row.delta)
            for row in holdout.itertuples(index=False)
        },
        "holdout_positive_folds": holdout_positive_folds,
        "holdout_mean_delta": holdout_mean_delta,
        "holdout_gate": {
            "positive_folds_required": len(HOLDOUT_FOLDS),
            "mean_delta_min": HOLDOUT_MEAN_DELTA_MIN,
        },
        "passed": passed,
    }
    Path("/kaggle/working/report.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
