"""Lookup-Transformer 조회 어휘 미등록값 진입 진단. (#128)

각 fold의 학습 부분에서 정확값 조회 어휘를 만들고, 결측과 구분해 검증 및
테스트의 조회 어휘 미등록값 비율을 측정한다.
완료된 OOF 실행을 함께 지정하면 미등록값 포함 행의 행 단위 손실과 AUC 순위
오류 기여도도 계산한다.

사용법:
    uv run python scripts/diagnose_lookup_unk.py \
        --config configs/exp067_lookup_xgb_impute_comps5.yaml \
        --run-id 2bd55026ae63430aa774bce20a359b4a
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

from pipeline import data
from pipeline.config import load_config
from pipeline.runs import MlflowRunStore


@dataclass(frozen=True)
class AucErrorContribution:
    """미등록값 포함 행과 관련된 양성-음성 쌍의 AUC 오류 기여."""

    related_pair_share: float
    related_auc_loss: float
    related_error_share: float
    related_pair_error_rate: float


def unknown_mask(reference: pd.DataFrame, evaluated: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """reference 어휘에 없고 결측도 아닌 evaluated 값을 True로 표시한다."""
    out = pd.DataFrame(False, index=evaluated.index, columns=columns)
    for col in columns:
        vocab = set(reference[col].dropna().unique())
        out[col] = evaluated[col].notna() & ~evaluated[col].isin(vocab)
    return out


def profile_unknowns(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """fold별 열 상세, fold별 행 요약, 검증 행별 미등록값 개수를 돌려준다."""
    details: list[dict] = []
    summaries: list[dict] = []
    validation_rows: list[pd.DataFrame] = []

    for fold in sorted(train["fold"].unique()):
        train_fold = train[train["fold"] != fold]
        validation = train[train["fold"] == fold]
        validation_unknown = unknown_mask(train_fold, validation, columns)
        test_unknown = unknown_mask(train_fold, test, columns)

        for split, frame, mask in (
            ("validation", validation, validation_unknown),
            ("test", test, test_unknown),
        ):
            for col in columns:
                selected = mask[col]
                details.append(
                    {
                        "fold": int(fold),
                        "split": split,
                        "column": col,
                        "rows": len(frame),
                        "missing": int(frame[col].isna().sum()),
                        "unknown": int(selected.sum()),
                        "unknown_unique": int(frame.loc[selected, col].nunique()),
                        "unknown_rate": float(selected.mean()),
                    }
                )

        summaries.append(
            {
                "fold": int(fold),
                "validation_rows": len(validation),
                "validation_any_unknown": int(validation_unknown.any(axis=1).sum()),
                "validation_any_unknown_rate": float(validation_unknown.any(axis=1).mean()),
                "test_rows": len(test),
                "test_any_unknown": int(test_unknown.any(axis=1).sum()),
                "test_any_unknown_rate": float(test_unknown.any(axis=1).mean()),
            }
        )
        validation_rows.append(
            pd.DataFrame(
                {
                    data.ID: validation[data.ID],
                    "fold": int(fold),
                    "unknown_count": validation_unknown.sum(axis=1).to_numpy(),
                }
            )
        )

    return (
        pd.DataFrame(details),
        pd.DataFrame(summaries),
        pd.concat(validation_rows, ignore_index=True),
    )


def auc_error_contribution(
    labels: pd.Series, predictions: pd.Series, any_unknown: pd.Series
) -> AucErrorContribution:
    """미등록값 포함 행과 관련된 순위 쌍이 전체 AUC 오류에서 차지하는 몫을 잰다.

    전체 양성-음성 쌍에서 알려진 값 행끼리의 쌍을 빼면 미등록값 포함 행이 적어도
    하나 들어간 쌍이다.
    그 쌍의 순위 오류를 모두 고친다는 낙관적 상한이 related_auc_loss다.
    """
    y = labels.to_numpy(dtype="int8")
    pred = predictions.to_numpy(dtype="float64")
    unknown = any_unknown.to_numpy(dtype=bool)
    known = ~unknown

    positives = int(y.sum())
    negatives = len(y) - positives
    known_positives = int(y[known].sum())
    known_negatives = int(known.sum()) - known_positives
    total_pairs = positives * negatives
    known_pairs = known_positives * known_negatives
    related_pairs = total_pairs - known_pairs
    if related_pairs == 0:
        return AucErrorContribution(0.0, 0.0, 0.0, 0.0)

    total_auc = roc_auc_score(y, pred)
    known_auc = roc_auc_score(y[known], pred[known])
    total_errors = (1.0 - total_auc) * total_pairs
    known_errors = (1.0 - known_auc) * known_pairs
    related_errors = max(0.0, total_errors - known_errors)

    return AucErrorContribution(
        related_pair_share=related_pairs / total_pairs,
        related_auc_loss=related_errors / total_pairs,
        related_error_share=related_errors / total_errors if total_errors else 0.0,
        related_pair_error_rate=related_errors / related_pairs,
    )


def print_oof_report(
    run_id: str, validation_rows: pd.DataFrame, train: pd.DataFrame, config: dict
) -> None:
    store = MlflowRunStore()
    meta = store.facts_of(run_id)
    run_config = store.config_of(run_id)
    expected_lookup = config["model"]["params"]["lookup_cols"]
    actual_lookup = run_config["model"]["params"]["lookup_cols"]
    if actual_lookup != expected_lookup:
        raise ValueError(
            f"run {run_id}의 lookup_cols가 진단 설정과 다르다: {actual_lookup} != {expected_lookup}"
        )

    indexed_rows = validation_rows.set_index(data.ID).sort_index()
    labels = train.set_index(data.ID)[data.TARGET].reindex(indexed_rows.index)
    predictions = store.oof_of(run_id).reindex(indexed_rows.index)
    if labels.isna().any() or predictions.isna().any():
        raise ValueError("OOF, 라벨과 진단 행의 id가 일치하지 않는다.")
    any_unknown = indexed_rows["unknown_count"] > 0

    print("\n== OOF 오류 기여 ==")
    print(f"run={run_id} name={meta.run_name}")
    rows = []
    for group, selected in (
        ("all", pd.Series(True, index=labels.index)),
        ("known", ~any_unknown),
        ("any_unknown", any_unknown),
    ):
        group_y = labels[selected]
        group_pred = predictions[selected]
        group_auc = float(roc_auc_score(group_y, group_pred)) if group_y.nunique() == 2 else np.nan
        rows.append(
            {
                "group": group,
                "rows": len(group_y),
                "row_share": len(group_y) / len(labels),
                "positive_rate": float(group_y.mean()),
                "auc": group_auc,
                "log_loss": float(log_loss(group_y, group_pred, labels=[0, 1])),
                "brier": float(mean_squared_error(group_y, group_pred)),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.9f}"))

    clipped = predictions.clip(np.finfo("float64").eps, 1.0 - np.finfo("float64").eps)
    per_row_loss = -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    loss_share = float(per_row_loss[any_unknown].sum() / per_row_loss.sum())
    contribution = auc_error_contribution(labels, predictions, any_unknown)
    print(f"unknown_log_loss_share={loss_share:.9f}")
    print(f"related_pair_share={contribution.related_pair_share:.9f}")
    print(f"related_auc_loss={contribution.related_auc_loss:.9f}")
    print(f"related_error_share={contribution.related_error_share:.9f}")
    print(f"related_pair_error_rate={contribution.related_pair_error_rate:.9f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lookup-Transformer 조회 어휘 미등록값 진단")
    parser.add_argument("--config", required=True, help="Lookup-Transformer 설정 YAML")
    parser.add_argument("--run-id", help="오류 기여를 측정할 완료된 OOF 실행")
    args = parser.parse_args()

    cfg = load_config(args.config, "confirm")
    train = data.load_csv(cfg.data.train)
    test = data.load_csv(cfg.data.test)
    train = data.attach_folds(train, cfg.data.folds)
    lookup_cols = list(cfg.model.params["lookup_cols"])
    details, summaries, validation_rows = profile_unknowns(train, test, lookup_cols)

    print("== fold별 행 단위 조회 어휘 미등록값 ==")
    print(summaries.to_string(index=False, float_format=lambda value: f"{value:.9f}"))
    print("\n== fold별 열 단위 조회 어휘 미등록값 ==")
    print(details.to_string(index=False, float_format=lambda value: f"{value:.9f}"))

    total_unknown_rows = int((validation_rows["unknown_count"] > 0).sum())
    print("\n== 검증 전체 요약 ==")
    print(
        f"rows={len(validation_rows)} any_unknown={total_unknown_rows} "
        f"any_unknown_rate={total_unknown_rows / len(validation_rows):.9f} "
        f"max_unknown_count={int(validation_rows['unknown_count'].max())}"
    )

    if args.run_id:
        with cfg.source_path.open() as file:
            import yaml

            raw_config = yaml.safe_load(file)
        print_oof_report(args.run_id, validation_rows, train, raw_config)


if __name__ == "__main__":
    main()
