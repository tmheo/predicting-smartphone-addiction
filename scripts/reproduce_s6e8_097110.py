"""S6E8 0.97110 노트북의 핵심 계산을 고정 입력에서 독립 재현한다.

검토 대상은 공개 판본 22, 실행 판본 343453035다.
원문 주소:
https://www.kaggle.com/code/raykkretzschmar/why-every-s6e8-notebook-above-0-97110-overfits

이 구현은 원문 코드를 복사하지 않고 문서화된 계산 질문을 별도로 구현한다.
Naji 입력은 Kaggle 표시상 사용 조건이 Unknown이므로 입력과 이 스크립트가 만든
관련 산출물을 저장소에 배포하지 않는다.

사용 예:
    uv run python scripts/reproduce_s6e8_097110.py \
        --train data/train.csv \
        --test data/test.csv \
        --teacher-oof /path/to/19_blend_oof_predictions.csv \
        --teacher-test /path/to/19_blend_submission.csv.csv \
        --signals /path/to/transductive_signals.npz \
        --leaderboards /path/to/s6_leaderboards.csv \
        --stored-submission /path/to/submission.csv \
        --output-json /tmp/s6e8-097110-reproduction.json \
        --figure-dir /tmp/s6e8-097110-figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

TARGET = "addicted_label"
PUBLIC_ROWS = 59_260
PRIVATE_ROWS = 237_042
SEPARATED_TRIALS = 50
SEPARATED_SEED = 940_813
AUC_EPISODES = frozenset({"S6E2", "S6E3", "S6E5"})

EXPECTED_SHA256 = {
    "train": "f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c",
    "test": "8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e",
    "teacher_oof": "8d4caad066e599b0afbb1a84a48af0af063a1c6ccafbdba92a91278f57ed0429",
    "teacher_test": "476db09857e1e7b452a0b0987ab3ed4d16ac9aed1efec6188c82739892d907be",
    "signals": "16016e8a2d0941a7107f3dc12ecbbc03f53213bc3824043f48a44d7f408e8bbd",
    "leaderboards": "02937c8f5d0bf8266874bb9b42c03de1cda75ef364f018ca457afbecc26835a6",
    "stored_submission": "66781f9298c0b695f68669e18403fc230cfea2a7d9f0d93a3d91bacdd2c6d06d",
}

# 이 네 공개 점수는 독립 계산값이 아니라 지정 노트북 작성자의 기록이다.
AUTHOR_RECORDED_PUBLIC_SCORES = {
    -0.12: {"public_score": 0.97114, "submission_reference": "55584411"},
    -0.08: {"public_score": 0.97115, "submission_reference": "55584395"},
    0.00: {
        "public_score": 0.97113,
        "submission_reference": "published Naji v19 artifact",
    },
    0.12: {"public_score": 0.97103, "submission_reference": "55584375"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes(paths: dict[str, Path]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, path in paths.items():
        value = sha256(path)
        expected = EXPECTED_SHA256[name]
        if value != expected:
            raise ValueError(f"{name} SHA-256 불일치: {value} != {expected}")
        actual[name] = value
    return actual


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return (rankdata(values, method="average") - 0.5) / len(values)


def deterministic_rank(values: np.ndarray, tie_breaker: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    tie_breaker = np.asarray(tie_breaker)
    if values.shape != tie_breaker.shape:
        raise ValueError("순위 값과 동률 해소 값의 모양이 다르다")
    order = np.lexsort((tie_breaker, values))
    result = np.empty(len(order), dtype=np.float64)
    result[order] = (np.arange(len(order), dtype=np.float64) + 0.5) / len(order)
    return result


def candidate_predictions(
    teacher: np.ndarray, student: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    teacher = np.asarray(teacher)
    student = np.asarray(student)
    weights = np.asarray(weights)
    if teacher.shape != student.shape:
        raise ValueError("교사와 학생 예측의 모양이 다르다")
    return (1 - weights[:, None]) * teacher[None, :] + weights[:, None] * student[None, :]


def aucs(y: np.ndarray, predictions: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    if rows is None:
        rows = np.arange(len(y))
    return np.asarray([roc_auc_score(y[rows], prediction[rows]) for prediction in predictions])


def assert_unique_ids(frame: pd.DataFrame, name: str) -> None:
    if frame["id"].isna().any() or not frame["id"].is_unique:
        raise ValueError(f"{name} id가 비어 있거나 중복됐다")


def align_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    teacher_oof: pd.DataFrame,
    teacher_test: pd.DataFrame,
    signals: Any,
) -> dict[str, Any]:
    for name, frame in {
        "train": train,
        "test": test,
        "teacher_oof": teacher_oof,
        "teacher_test": teacher_test,
    }.items():
        assert_unique_ids(frame, name)

    train_ids = train["id"].to_numpy()
    test_ids = test["id"].to_numpy()
    oof_by_id = teacher_oof.set_index("id").reindex(train_ids)
    if oof_by_id[TARGET].isna().any() or set(teacher_oof["id"]) != set(train_ids):
        raise ValueError("교사 OOF id 집합이 공식 훈련 자료와 다르다")
    teacher_test_ids = teacher_test["id"].to_numpy()
    if not np.array_equal(teacher_test_ids, test_ids):
        raise ValueError("교사 시험 예측 id 순서가 공식 시험 자료와 다르다")

    required_signals = {"oof_soft_student", "test_soft_student"}
    missing = required_signals.difference(signals.files)
    if missing:
        raise ValueError(f"학생 신호가 없다: {sorted(missing)}")
    student_oof_raw = np.asarray(signals["oof_soft_student"])
    student_test_raw = np.asarray(signals["test_soft_student"])
    if len(student_oof_raw) != len(train) or len(student_test_raw) != len(test):
        raise ValueError("학생 예측 길이가 공식 자료와 다르다")

    return {
        "y": train[TARGET].to_numpy(),
        "teacher_oof": percentile_rank(oof_by_id[TARGET].to_numpy()),
        "student_oof": percentile_rank(student_oof_raw),
        "teacher_test": percentile_rank(teacher_test[TARGET].to_numpy()),
        "student_test": percentile_rank(student_test_raw),
        "test_ids": test_ids,
        "student_row_binding": "길이만 확인 가능하며 학생 배열에 id가 없다",
    }


def oof_weight_audit(y: np.ndarray, teacher: np.ndarray, student: np.ndarray) -> dict[str, Any]:
    weights = np.round(np.arange(-0.20, 0.301, 0.01), 2)
    predictions = candidate_predictions(teacher, student, weights)
    scores = aucs(y, predictions)
    teacher_index = int(np.flatnonzero(weights == 0)[0])
    teacher_auc = float(scores[teacher_index])
    best_index = int(scores.argmax())

    author_points = []
    for weight, recorded in AUTHOR_RECORDED_PUBLIC_SCORES.items():
        index = int(np.flatnonzero(weights == weight)[0])
        author_points.append(
            {
                "student_weight": weight,
                "oof_auc": float(scores[index]),
                "delta_vs_teacher": float(scores[index] - teacher_auc),
                **recorded,
                "public_score_provenance": "작성자가 코드에 직접 적은 기록",
            }
        )

    return {
        "weights": weights,
        "scores": scores,
        "deltas": scores - teacher_auc,
        "teacher_auc": teacher_auc,
        "best_weight": float(weights[best_index]),
        "best_auc": float(scores[best_index]),
        "best_delta": float(scores[best_index] - teacher_auc),
        "author_recorded_points": author_points,
    }


def notebook_split_reproduction(
    y: np.ndarray, teacher: np.ndarray, student: np.ndarray
) -> dict[str, Any]:
    weights = np.round(np.arange(-0.20, 0.301, 0.05), 2)
    predictions = candidate_predictions(teacher, student, weights)
    full_scores = aucs(y, predictions)
    reference_index = int(full_scores.argmax())
    splitter = StratifiedShuffleSplit(
        n_splits=10,
        train_size=PUBLIC_ROWS,
        random_state=SEPARATED_SEED,
    )
    trials = []
    for trial, (public_rows, other_rows) in enumerate(
        splitter.split(np.zeros(len(y)), y), start=1
    ):
        public_scores = aucs(y, predictions, public_rows)
        selected_index = int(public_scores.argmax())
        other_scores = aucs(y, predictions, other_rows)
        trials.append(
            {
                "trial": trial,
                "selected_weight": float(weights[selected_index]),
                "apparent_public_gain": float(
                    public_scores[selected_index] - public_scores[reference_index]
                ),
                "other_rows_delta": float(
                    other_scores[selected_index] - other_scores[reference_index]
                ),
            }
        )

    return {
        "public_rows": PUBLIC_ROWS,
        "other_rows": len(y) - PUBLIC_ROWS,
        "full_oof_reference_weight": float(weights[reference_index]),
        "comparison_leakage": (
            "비교 기준 가중치를 public과 other를 합친 전체 OOF에서 골랐다"
        ),
        "mean_apparent_public_gain": float(
            np.mean([trial["apparent_public_gain"] for trial in trials])
        ),
        "mean_other_rows_delta": float(
            np.mean([trial["other_rows_delta"] for trial in trials])
        ),
        "different_weight_trials": int(
            sum(
                trial["selected_weight"] != float(weights[reference_index])
                for trial in trials
            )
        ),
        "trials": trials,
    }


def split_three_ways(
    y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    held_out_rows = PUBLIC_ROWS + PRIVATE_ROWS
    development_rows = len(y) - held_out_rows
    if development_rows <= 0:
        raise ValueError("개발, 모의 공개, 모의 비공개 자료로 나눌 행이 부족하다")

    outer = StratifiedShuffleSplit(
        n_splits=1,
        train_size=development_rows,
        test_size=held_out_rows,
        random_state=seed,
    )
    development, held_out = next(outer.split(np.zeros(len(y)), y))
    inner = StratifiedShuffleSplit(
        n_splits=1,
        train_size=PUBLIC_ROWS,
        test_size=PRIVATE_ROWS,
        random_state=seed + 1_000_003,
    )
    public_relative, private_relative = next(
        inner.split(np.zeros(len(held_out)), y[held_out])
    )
    return development, held_out[public_relative], held_out[private_relative]


def quantile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std(ddof=1)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "negative_trials": int((array < 0).sum()),
        "positive_trials": int((array > 0).sum()),
        "zero_trials": int((array == 0).sum()),
    }


def separated_selection_control(
    y: np.ndarray, teacher: np.ndarray, student: np.ndarray
) -> dict[str, Any]:
    weights = np.round(np.arange(-0.20, 0.301, 0.05), 2)
    predictions = candidate_predictions(teacher, student, weights)
    teacher_index = int(np.flatnonzero(weights == 0)[0])
    trials = []

    for offset in range(SEPARATED_TRIALS):
        seed = SEPARATED_SEED + offset
        development, public, private = split_three_ways(y, seed)
        development_scores = aucs(y, predictions, development)
        public_scores = aucs(y, predictions, public)
        private_scores = aucs(y, predictions, private)
        development_index = int(development_scores.argmax())
        public_index = int(public_scores.argmax())
        trials.append(
            {
                "seed": seed,
                "development_weight": float(weights[development_index]),
                "public_weight": float(weights[public_index]),
                "public_gain_vs_development_choice": float(
                    public_scores[public_index] - public_scores[development_index]
                ),
                "private_delta_vs_development_choice": float(
                    private_scores[public_index] - private_scores[development_index]
                ),
                "private_delta_vs_teacher": float(
                    private_scores[public_index] - private_scores[teacher_index]
                ),
                "development_choice_private_delta_vs_teacher": float(
                    private_scores[development_index] - private_scores[teacher_index]
                ),
            }
        )

    private_deltas = [
        trial["private_delta_vs_development_choice"] for trial in trials
    ]
    return {
        "development_rows": len(y) - PUBLIC_ROWS - PRIVATE_ROWS,
        "public_rows": PUBLIC_ROWS,
        "private_rows": PRIVATE_ROWS,
        "trial_count": SEPARATED_TRIALS,
        "seed_range": [SEPARATED_SEED, SEPARATED_SEED + SEPARATED_TRIALS - 1],
        "different_weight_trials": int(
            sum(
                trial["development_weight"] != trial["public_weight"]
                for trial in trials
            )
        ),
        "public_gain_vs_development_choice": quantile_summary(
            [trial["public_gain_vs_development_choice"] for trial in trials]
        ),
        "private_delta_vs_development_choice": quantile_summary(private_deltas),
        "public_choice_private_delta_vs_teacher": quantile_summary(
            [trial["private_delta_vs_teacher"] for trial in trials]
        ),
        "development_choice_private_delta_vs_teacher": quantile_summary(
            [
                trial["development_choice_private_delta_vs_teacher"]
                for trial in trials
            ]
        ),
        "trials": trials,
    }


def reproduce_submission(
    test_ids: np.ndarray,
    teacher: np.ndarray,
    student: np.ndarray,
    stored_submission: pd.DataFrame,
) -> dict[str, Any]:
    raw = 1.08 * teacher - 0.08 * student
    prediction = deterministic_rank(raw, teacher)
    reproduced = pd.DataFrame({"id": test_ids, TARGET: prediction})
    csv_bytes = reproduced.to_csv(index=False).encode()
    reproduced_hash = hashlib.sha256(csv_bytes).hexdigest()
    stored_bytes_hash = EXPECTED_SHA256["stored_submission"]

    id_match = np.array_equal(reproduced["id"].to_numpy(), stored_submission["id"].to_numpy())
    stored_values = stored_submission[TARGET].to_numpy()
    value_match = np.array_equal(reproduced[TARGET].to_numpy(), stored_values)
    if reproduced_hash != stored_bytes_hash or not id_match:
        raise ValueError("독립 재계산 제출이 저장 제출과 일치하지 않는다")

    rearranged_raw = teacher - 0.08 * (student - teacher)
    rearranged_prediction = deterministic_rank(rearranged_raw, teacher)
    rank_step = 1 / len(reproduced)
    rearranged_difference = rearranged_prediction - prediction
    return {
        "rows": len(reproduced),
        "reproduced_sha256": reproduced_hash,
        "stored_sha256": stored_bytes_hash,
        "byte_hash_match": True,
        "parsed_value_exact_match": value_match,
        "parsed_value_max_abs_difference": float(
            np.max(np.abs(reproduced[TARGET].to_numpy() - stored_values))
        ),
        "unique_predictions": int(reproduced[TARGET].nunique()),
        "algebraic_rearrangement_sensitivity": {
            "formula": "teacher - 0.08 * (student - teacher)",
            "different_rows": int(np.count_nonzero(rearranged_difference)),
            "maximum_rank_steps": int(
                np.rint(np.max(np.abs(rearranged_difference)) / rank_step)
            ),
            "maximum_prediction_difference": float(
                np.max(np.abs(rearranged_difference))
            ),
        },
    }


def leaderboard_history(leaderboards: pd.DataFrame) -> dict[str, Any]:
    required = {
        "episode",
        "is_host_baseline",
        "public_rank",
        "private_rank",
    }
    missing = required.difference(leaderboards.columns)
    if missing:
        raise ValueError(f"순위 자료 열이 없다: {sorted(missing)}")

    rows = []
    for episode, group in leaderboards.groupby("episode", sort=True):
        group = group.loc[~group["is_host_baseline"]].dropna(
            subset=["public_rank", "private_rank"]
        )
        public_top_ten = group.nsmallest(10, "public_rank")
        rows.append(
            {
                "episode": episode,
                "teams": len(group),
                "rank_correlation": float(
                    spearmanr(group["public_rank"], group["private_rank"]).statistic
                ),
                "public_top_ten_survivors": int(
                    (public_top_ten["private_rank"] <= 10).sum()
                ),
                "public_winner_private_rank": int(
                    public_top_ten.nsmallest(1, "public_rank")["private_rank"].iloc[0]
                ),
                "worst_private_rank_in_public_top_ten": int(
                    public_top_ten["private_rank"].max()
                ),
            }
        )

    public_submission_columns = {
        "public_submission_id",
        "public_submission_reference",
    }
    private_submission_columns = {
        "private_submission_id",
        "private_submission_reference",
    }
    has_public_submission = bool(public_submission_columns.intersection(leaderboards.columns))
    has_private_submission = bool(private_submission_columns.intersection(leaderboards.columns))
    same_submission_available = has_public_submission and has_private_submission
    auc_rows = [row for row in rows if row["episode"] in AUC_EPISODES]

    return {
        "episodes": rows,
        "all_episode_survivors": int(sum(row["public_top_ten_survivors"] for row in rows)),
        "all_episode_public_top_ten_slots": 10 * len(rows),
        "auc_episode_survivors": int(
            sum(row["public_top_ten_survivors"] for row in auc_rows)
        ),
        "auc_episode_public_top_ten_slots": 10 * len(auc_rows),
        "same_submission_control_available": same_submission_available,
        "same_submission_control_reason": (
            "입력에는 공개 최고 제출과 비공개 최종 선택 제출의 식별자가 없다"
            if not same_submission_available
            else "입력에 양쪽 제출 식별자가 있다"
        ),
        "available_columns": list(leaderboards.columns),
    }


def plot_results(
    figure_dir: Path,
    oof: dict[str, Any],
    separated: dict[str, Any],
    history: dict[str, Any],
) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(oof["weights"], oof["deltas"] * 1e5, linewidth=2)
    ax.axhline(0, color="0.6", linewidth=1)
    ax.axvline(0, color="0.6", linewidth=1)
    for point in oof["author_recorded_points"]:
        ax.scatter(point["student_weight"], point["delta_vs_teacher"] * 1e5, s=55)
        ax.annotate(
            f"recorded public {point['public_score']:.5f}",
            (point["student_weight"], point["delta_vs_teacher"] * 1e5),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(
        xlabel="Student weight",
        ylabel="OOF AUC delta (x1e-5)",
        title="Independent OOF sweep with author-recorded public scores",
    )
    ax.grid(alpha=0.2)
    path = figure_dir / "oof-weight-sweep.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    private_deltas = [
        trial["private_delta_vs_development_choice"]
        for trial in separated["trials"]
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(np.asarray(private_deltas) * 1e5, bins=14, color="#3977b5")
    ax.axvline(0, color="0.4", linewidth=1)
    ax.set(
        xlabel="Private AUC delta: public choice minus development choice (x1e-5)",
        ylabel="Trials",
        title="Separated 395,067 / 59,260 / 237,042 row control",
    )
    ax.grid(axis="y", alpha=0.2)
    path = figure_dir / "separated-selection-control.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    episode_rows = history["episodes"]
    values = [row["public_top_ten_survivors"] for row in episode_rows]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    colors = ["#cf3f3f" if value == 0 else "#3977b5" for value in values]
    ax.bar([row["episode"] for row in episode_rows], values, color=colors)
    ax.axhline(10, color="0.5", linestyle="--", linewidth=1)
    ax.set(
        ylim=(0, 10.5),
        ylabel="Public top-10 teams still in private top 10",
        title="Season 6 team-level public top-10 survival",
    )
    ax.grid(axis="y", alpha=0.2)
    path = figure_dir / "season6-team-survival.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--teacher-oof", type=Path, required=True)
    parser.add_argument("--teacher-test", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--leaderboards", type=Path, required=True)
    parser.add_argument("--stored-submission", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "train": args.train,
        "test": args.test,
        "teacher_oof": args.teacher_oof,
        "teacher_test": args.teacher_test,
        "signals": args.signals,
        "leaderboards": args.leaderboards,
        "stored_submission": args.stored_submission,
    }
    hashes = verify_hashes(paths)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    teacher_oof = pd.read_csv(args.teacher_oof)
    teacher_test = pd.read_csv(args.teacher_test)
    signals = np.load(args.signals)
    stored_submission = pd.read_csv(args.stored_submission, float_precision="round_trip")
    aligned = align_inputs(train, test, teacher_oof, teacher_test, signals)

    oof = oof_weight_audit(
        aligned["y"], aligned["teacher_oof"], aligned["student_oof"]
    )
    notebook_split = notebook_split_reproduction(
        aligned["y"], aligned["teacher_oof"], aligned["student_oof"]
    )
    separated = separated_selection_control(
        aligned["y"], aligned["teacher_oof"], aligned["student_oof"]
    )
    submission = reproduce_submission(
        aligned["test_ids"],
        aligned["teacher_test"],
        aligned["student_test"],
        stored_submission,
    )
    history = leaderboard_history(pd.read_csv(args.leaderboards))
    result = {
        "source": {
            "public_version": 22,
            "execution_version": 343_453_035,
            "implementation": "원문 코드를 복사하지 않은 독립 구현",
        },
        "input_sha256": hashes,
        "rows": {"train": len(train), "test": len(test)},
        "input_binding": {
            "teacher_oof_ids_match_train": True,
            "teacher_test_ids_and_order_match_test": True,
            "student_row_binding": aligned["student_row_binding"],
        },
        "oof_weight_audit": oof,
        "notebook_split_reproduction": notebook_split,
        "separated_selection_control": separated,
        "submission_reproduction": submission,
        "leaderboard_history": history,
    }
    if args.figure_dir is not None:
        result["local_figures"] = plot_results(args.figure_dir, oof, separated, history)

    rendered = json.dumps(json_ready(result), ensure_ascii=False, indent=2)
    if args.output_json is None:
        print(rendered)
    else:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
        print(args.output_json)


if __name__ == "__main__":
    main()
