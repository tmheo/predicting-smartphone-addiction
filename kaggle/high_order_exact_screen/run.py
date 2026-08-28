"""CPU-only diagnostic for high-order exact-value target statistics.

This is a candidate screen, not an improvement judgment. Candidate selection uses
folds 0 and 1. Folds 2, 3, and 4 remain untouched until the final diagnostic.
The current 313-member nested prediction is a fixed baseline.
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET = "addicted_label"
ID = "id"
FOLD_SEED = 42
SMOOTHING = 20.0
SCREEN_FOLD = 0
SELECTION_FOLD = 1
HOLDOUT_FOLDS = (2, 3, 4)
TOP_AFTER_SCREEN = 40
TOP_AFTER_SELECTION = 16
CORRECTION_WEIGHTS = (0.25, 0.5, 1.0)
EPS = 1e-6

RAW_COLUMNS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "gender",
    "stress_level",
    "academic_work_impact",
]
DERIVED_COLUMNS = ["other_screen", "screen_slack"]
ALL_COLUMNS = RAW_COLUMNS + DERIVED_COLUMNS
ANCHORS = {
    "daily_screen_time_hours",
    "weekend_screen_time",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "other_screen",
    "screen_slack",
}


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


def attach_features(train: pd.DataFrame) -> pd.DataFrame:
    parts = ["social_media_hours", "gaming_hours", "work_study_hours"]
    train = train.copy()
    train["other_screen"] = train["daily_screen_time_hours"] - train[parts].sum(
        axis=1, skipna=False
    )
    train["screen_slack"] = train["daily_screen_time_hours"] - train[parts].sum(axis=1)
    return train


def make_folds(y: np.ndarray) -> np.ndarray:
    folds = np.empty(len(y), dtype=np.int8)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    for fold, (_, valid_index) in enumerate(splitter.split(np.zeros(len(y)), y)):
        folds[valid_index] = fold
    return folds


def candidate_combinations() -> list[tuple[str, ...]]:
    candidates: list[tuple[str, ...]] = []
    for combo in itertools.combinations(ALL_COLUMNS, 3):
        if len(ANCHORS.intersection(combo)) >= 2:
            candidates.append(combo)
    for combo in itertools.combinations(ALL_COLUMNS, 4):
        if len(ANCHORS.intersection(combo)) >= 3:
            candidates.append(combo)
    return candidates


def column_hashes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        column: pd.util.hash_pandas_object(
            frame[column], index=False, categorize=True
        ).to_numpy(dtype=np.uint64)
        for column in ALL_COLUMNS
    }


def combination_hash(
    combo: tuple[str, ...], hashes: dict[str, np.ndarray]
) -> np.ndarray:
    result = np.full(len(next(iter(hashes.values()))), np.uint64(1469598103934665603))
    prime = np.uint64(1099511628211)
    for column in combo:
        result ^= hashes[column]
        result *= prime
    return result


def mapped_group_correction(
    keys: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
    folds: np.ndarray,
    valid_fold: int,
) -> dict[str, object]:
    fit_mask = folds != valid_fold
    valid_mask = ~fit_mask
    fit_keys = keys[fit_mask]
    valid_keys = keys[valid_mask]
    fit_y = y[fit_mask]
    fit_baseline = baseline[fit_mask]

    unique, inverse, counts = np.unique(
        fit_keys, return_inverse=True, return_counts=True
    )
    target_sums = np.bincount(inverse, weights=fit_y)
    baseline_sums = np.bincount(inverse, weights=fit_baseline)
    positions = np.searchsorted(unique, valid_keys)
    safe_positions = np.minimum(positions, len(unique) - 1)
    seen = (positions < len(unique)) & (unique[safe_positions] == valid_keys)

    mapped_count = np.zeros(len(valid_keys), dtype=np.int64)
    mapped_count[seen] = counts[safe_positions[seen]]
    global_mean = float(fit_y.mean())
    target_rate = np.full(len(valid_keys), global_mean, dtype=np.float64)
    baseline_rate = np.full(len(valid_keys), global_mean, dtype=np.float64)
    target_rate[seen] = (
        target_sums[safe_positions[seen]] + SMOOTHING * global_mean
    ) / (mapped_count[seen] + SMOOTHING)
    baseline_rate[seen] = (
        baseline_sums[safe_positions[seen]] + SMOOTHING * global_mean
    ) / (mapped_count[seen] + SMOOTHING)

    correction = logit(np.clip(target_rate, EPS, 1 - EPS)) - logit(
        np.clip(baseline_rate, EPS, 1 - EPS)
    )
    valid_y = y[valid_mask]
    valid_baseline = np.clip(baseline[valid_mask], EPS, 1 - EPS)
    baseline_auc = float(roc_auc_score(valid_y, valid_baseline))
    standalone_auc = float(roc_auc_score(valid_y, target_rate))

    weight_aucs: dict[str, float] = {}
    for weight in CORRECTION_WEIGHTS:
        prediction = expit(logit(valid_baseline) + weight * correction)
        weight_aucs[str(weight)] = float(roc_auc_score(valid_y, prediction))
    best_weight, best_auc = max(
        weight_aucs.items(), key=lambda item: (item[1], -float(item[0]))
    )
    return {
        "valid_fold": valid_fold,
        "fit_groups": int(len(unique)),
        "coverage_seen": float(seen.mean()),
        "coverage_count_ge_5": float((mapped_count >= 5).mean()),
        "coverage_count_ge_20": float((mapped_count >= 20).mean()),
        "standalone_auc": standalone_auc,
        "baseline_auc": baseline_auc,
        "best_weight": float(best_weight),
        "best_auc": best_auc,
        "delta": best_auc - baseline_auc,
        "weight_aucs": weight_aucs,
    }


def evaluate_candidates(
    candidates: list[tuple[str, ...]],
    hashes: dict[str, np.ndarray],
    y: np.ndarray,
    baseline: np.ndarray,
    folds: np.ndarray,
    valid_fold: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    started = time.time()
    for index, combo in enumerate(candidates, start=1):
        metrics = mapped_group_correction(
            combination_hash(combo, hashes), y, baseline, folds, valid_fold
        )
        rows.append(
            {
                "candidate": "__".join(combo),
                "order": len(combo),
                "columns": json.dumps(combo),
                **{key: value for key, value in metrics.items() if key != "weight_aucs"},
                **{
                    f"auc_weight_{weight}": auc
                    for weight, auc in metrics["weight_aucs"].items()
                },
            }
        )
        if index % 20 == 0 or index == len(candidates):
            elapsed = time.time() - started
            print(
                f"fold {valid_fold}: {index}/{len(candidates)} candidates, "
                f"elapsed {elapsed:.1f}s",
                flush=True,
            )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["delta", "coverage_count_ge_5", "candidate"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def selected_candidates(
    screen: pd.DataFrame, selection: pd.DataFrame
) -> list[tuple[str, ...]]:
    merged = screen[["candidate", "delta"]].merge(
        selection[["candidate", "delta"]], on="candidate", suffixes=("_fold0", "_fold1")
    )
    merged["selection_score"] = merged[["delta_fold0", "delta_fold1"]].min(axis=1)
    merged = merged.sort_values(
        ["selection_score", "candidate"], ascending=[False, True]
    )
    positive = merged[(merged["delta_fold0"] > 0) & (merged["delta_fold1"] > 0)]
    source = positive if len(positive) else merged
    names = source.head(TOP_AFTER_SELECTION)["candidate"].tolist()
    lookup = {
        "__".join(combo): combo for combo in candidate_combinations()
    }
    return [lookup[name] for name in names]


def main() -> None:
    train_path = find_input("train.csv")
    baseline_path = find_input("current313_oof.parquet")
    train = attach_features(pd.read_csv(train_path))
    baseline_frame = pd.read_parquet(baseline_path)
    if list(baseline_frame.columns) != [ID, "fold", "pred"]:
        raise SystemExit(f"Unexpected baseline columns: {list(baseline_frame.columns)}")
    if len(train) != len(baseline_frame) or not np.array_equal(
        train[ID].to_numpy(), baseline_frame[ID].to_numpy()
    ):
        raise SystemExit("Baseline row order does not match competition train.csv")

    y = train[TARGET].to_numpy(dtype=np.int8)
    folds = make_folds(y)
    if not np.array_equal(folds, baseline_frame["fold"].to_numpy(dtype=np.int8)):
        raise SystemExit("Baseline fold vector does not match frozen StratifiedKFold")
    baseline = baseline_frame["pred"].to_numpy(dtype=np.float64)
    baseline_auc = float(roc_auc_score(y, baseline))
    if abs(baseline_auc - 0.970350946943525) > 1e-12:
        raise SystemExit(f"Baseline AUC mismatch: {baseline_auc}")

    candidates = candidate_combinations()
    hashes = column_hashes(train)
    print(
        f"rows={len(train)} candidates={len(candidates)} baseline_auc={baseline_auc:.12f}",
        flush=True,
    )

    screen = evaluate_candidates(candidates, hashes, y, baseline, folds, SCREEN_FOLD)
    screen.to_csv("/kaggle/working/fold0_screen.csv", index=False)
    fold1_candidates = [
        tuple(json.loads(value))
        for value in screen.head(TOP_AFTER_SCREEN)["columns"].tolist()
    ]
    selection = evaluate_candidates(
        fold1_candidates, hashes, y, baseline, folds, SELECTION_FOLD
    )
    selection.to_csv("/kaggle/working/fold1_selection.csv", index=False)

    promoted = selected_candidates(screen, selection)
    holdout_frames: list[pd.DataFrame] = []
    for fold in HOLDOUT_FOLDS:
        result = evaluate_candidates(promoted, hashes, y, baseline, folds, fold)
        result.to_csv(f"/kaggle/working/fold{fold}_holdout.csv", index=False)
        holdout_frames.append(result.assign(holdout_fold=fold))

    holdout = pd.concat(holdout_frames, ignore_index=True)
    wide = holdout.pivot(index="candidate", columns="holdout_fold", values="delta")
    wide.columns = [f"delta_fold{int(column)}" for column in wide.columns]
    delta_columns = list(wide.columns)
    wide["holdout_min_delta"] = wide[delta_columns].min(axis=1)
    wide["holdout_mean_delta"] = wide[delta_columns].mean(axis=1)
    wide["holdout_positive_folds"] = (wide[delta_columns] > 0).sum(axis=1)
    wide = wide.sort_values(
        ["holdout_positive_folds", "holdout_min_delta", "holdout_mean_delta"],
        ascending=[False, False, False],
    )
    wide.to_csv("/kaggle/working/holdout_summary.csv")

    passed = wide[
        (wide["holdout_positive_folds"] == len(HOLDOUT_FOLDS))
        & (wide["holdout_mean_delta"] >= 0.00001)
    ]
    promoted_payload = {
        "schema": "high-order-exact-key-screen/1",
        "purpose": "diagnostic_only_not_improvement_judgment",
        "train_sha256": file_sha256(train_path),
        "baseline_sha256": file_sha256(baseline_path),
        "baseline_auc": baseline_auc,
        "fold_seed": FOLD_SEED,
        "smoothing": SMOOTHING,
        "correction_weights": CORRECTION_WEIGHTS,
        "candidate_count": len(candidates),
        "screen_fold": SCREEN_FOLD,
        "selection_fold": SELECTION_FOLD,
        "holdout_folds": HOLDOUT_FOLDS,
        "selected_before_holdout": [list(combo) for combo in promoted],
        "holdout_gate": {
            "positive_folds_required": len(HOLDOUT_FOLDS),
            "mean_delta_min": 0.00001,
        },
        "passed": [name.split("__") for name in passed.index.tolist()],
    }
    Path("/kaggle/working/promoted_candidates.json").write_text(
        json.dumps(promoted_payload, indent=2) + "\n"
    )
    print(json.dumps(promoted_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
