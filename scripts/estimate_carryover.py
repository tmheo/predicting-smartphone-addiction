"""S6E8 public 보드 carry-over 사전 추정. (#204)

디스커션 734005 파생 노트북(georgymamarin, "Three of seven S6 boards erased
the public top ten")이 남긴 미해결 과제를 수행한다: 사후 수치인 carry-over를
마감 전에 분산 분해로 근사한다.

    carry-over ≈ 1 - (노이즈 분산 / 관측 산포 분산)

1. 노이즈: champion(exp081) OOF 예측을 public 채점 표본 크기(59,260행)로
   재표집해 AUC 표준편차를 잰다. 부트스트랩(복원)과 부분표집(비복원)을 모두
   계산하고 Hanley-McNeil 근사로 교차 확인한다.
2. 산포: `kaggle competitions leaderboard` public LB CSV에서 상위 1%~60%
   컷을 스윕하며 밴드 내 public 점수 표준편차를 잰다.
3. 컷별 carry-over 표와 스윕 그림을 산출한다. 노이즈 상관 민감도로
   스레드의 paired sigma(0.00009~0.00011)에서 유도한 팀별 고유 노이즈
   하한(≈0.00007)을 병기한다.

사용법:
    uv run python scripts/estimate_carryover.py --lb-csv <publicleaderboard.csv>

OOF가 없는 보드(다른 에피소드)는 Hanley-McNeil 근사만으로 돌 수 있다. (#206)
AUC는 컷별 상위 밴드 public 점수 중앙값을 대입한다:

    uv run python scripts/estimate_carryover.py --lb-csv <csv> \
        --hm-only --n-public 54000 --prevalence 0.4483 --label S6E2

산출: 표준 출력 리포트, run-logs/carryover_cut_sweep.csv,
docs/research/assets/carryover-preclose-cut-sweep.png
(--csv-out / --fig-out으로 경로를 바꿀 수 있다)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

TRAIN_PATH = Path("data/train.csv")
CHAMPION_PATH = Path("artifacts/champion.yaml")
MLRUNS = Path("mlruns")
TARGET = "addicted_label"
SEED = 42

N_PUBLIC = 59_260  # test 296,302행의 20%
N_BOOT = 500
CUTS_PCT = [1, 2, 3, 5, 7.5, 10, 15, 17, 20, 25, 30, 40, 50, 60]

# 노트북의 사후 기준(7개 완료 에피소드): Theil-Sen carry-over
ERASED_BAND = (0.07, 0.32)
KEPT_BAND = (0.73, 0.91)

# 스레드 734005: 유사한 상위 모델 두 개의 점수 차 표준편차(paired sigma)는
# 0.00009~0.00011. 두 팀의 고유 노이즈가 같다면 팀별 고유 노이즈는
# paired sigma / sqrt(2) ≈ 0.00007이 하한이다.
IDIOSYNCRATIC_NOISE_SD = 0.0001 / np.sqrt(2)

# S6E8에서 고유 노이즈 하한이 전체 노이즈(부트스트랩 SD 0.000573)에서 차지하는
# 비율. paired sigma가 없는 다른 보드에 고유 노이즈 하한을 이식할 때는 이
# 비율이 보드 간에 유지된다고 가정하고 해당 보드의 HM 노이즈에 곱한다. (#206)
IDIO_NOISE_FRACTION = IDIOSYNCRATIC_NOISE_SD / 0.000573

CSV_OUT = Path("run-logs/carryover_cut_sweep.csv")
FIG_OUT = Path("docs/research/assets/carryover-preclose-cut-sweep.png")


def load_champion_oof() -> tuple[pd.Series, pd.Series, float]:
    champion = yaml.safe_load(CHAMPION_PATH.read_text())
    run_id = champion["run_id"]
    matches = list(MLRUNS.glob(f"*/{run_id}/artifacts/oof.parquet"))
    if len(matches) != 1:
        raise SystemExit(f"champion OOF를 찾지 못했다: run_id={run_id}, matches={matches}")
    oof = pd.read_parquet(matches[0])
    train = pd.read_csv(TRAIN_PATH, usecols=["id", TARGET])
    merged = oof.merge(train, on="id", validate="one_to_one")
    auc = roc_auc_score(merged[TARGET], merged["pred"])
    recorded = float(champion["oof_auc"])
    if abs(auc - recorded) > 1e-9:
        raise SystemExit(f"OOF 재채점 불일치: {auc} != champion.yaml {recorded}")
    return merged["pred"].to_numpy(), merged[TARGET].to_numpy(), auc


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney 순위 기반 AUC. roc_auc_score보다 반복 계산에 빠르다."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    # 동률 보정: 같은 예측값에는 평균 순위를 부여
    sorted_p = p[order]
    ties_idx = np.flatnonzero(np.diff(sorted_p) == 0)
    if len(ties_idx):
        s = pd.Series(ranks[order]).groupby(pd.Series(sorted_p)).transform("mean")
        ranks[order] = s.to_numpy()
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def noise_sd(pred: np.ndarray, y: np.ndarray, replace: bool) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(pred)
    aucs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.choice(n, size=N_PUBLIC, replace=replace)
        aucs[b] = fast_auc(y[idx], pred[idx])
    return float(aucs.std(ddof=1)), float(aucs.mean())


def hanley_mcneil_sd(auc: float, prevalence: float, n_public: int = N_PUBLIC) -> float:
    n_pos = round(n_public * prevalence)
    n_neg = n_public - n_pos
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc**2)
        + (n_neg - 1) * (q2 - auc**2)
    ) / (n_pos * n_neg)
    return float(np.sqrt(var))


def cut_sweep(scores: np.ndarray, noise_sd_of_band, idio_sd_of_band) -> pd.DataFrame:
    """컷별 carry-over 표. 노이즈 SD는 밴드 점수를 받는 함수로 준다.

    상수 노이즈(S6E8 부트스트랩)는 `lambda band: sd`로, 컷마다 달라지는
    HM-only 노이즈는 밴드 중앙값 AUC를 대입하는 함수로 넘긴다.
    """
    rows = []
    for pct in CUTS_PCT:
        k = max(int(round(len(scores) * pct / 100)), 5)
        band = scores[:k]
        obs_sd = float(band.std(ddof=1))
        noise = float(noise_sd_of_band(band))
        idio = float(idio_sd_of_band(band))
        rows.append(
            {
                "cut_pct": pct,
                "n_teams": k,
                "band_score_min": float(band.min()),
                "obs_sd": obs_sd,
                "noise_sd": noise,
                "idio_sd": idio,
                "carryover_full_noise": max(0.0, 1 - (noise / obs_sd) ** 2),
                "carryover_idio_noise": max(0.0, 1 - (idio / obs_sd) ** 2),
            }
        )
    return pd.DataFrame(rows)


def plot_sweep(sweep: pd.DataFrame, fig_out: Path, label: str) -> None:
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(sweep["cut_pct"], sweep["obs_sd"], "o-", color="#1f77b4", label="Observed band SD")
    ax1.plot(
        sweep["cut_pct"],
        sweep["noise_sd"],
        "--",
        color="#d62728",
        label="Single-score noise SD",
    )
    ax1.plot(
        sweep["cut_pct"],
        sweep["idio_sd"],
        ":",
        color="#ff7f0e",
        label="Idiosyncratic noise floor",
    )
    ax1.set_xlabel("Top band cut (% of teams)")
    ax1.set_ylabel("Public score SD (AUC)")
    ax1.set_title(f"Dispersion vs. noise, {label} public LB")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.axhspan(*KEPT_BAND, color="#2ca02c", alpha=0.15, label="Kept boards (post-hoc 0.73-0.91)")
    ax2.axhspan(*ERASED_BAND, color="#d62728", alpha=0.15, label="Erased boards (post-hoc 0.07-0.32)")
    ax2.plot(
        sweep["cut_pct"],
        sweep["carryover_full_noise"],
        "o-",
        color="#1f77b4",
        label="Carry-over (full noise SD)",
    )
    ax2.plot(
        sweep["cut_pct"],
        sweep["carryover_idio_noise"],
        "s--",
        color="#ff7f0e",
        label="Carry-over (idiosyncratic noise)",
    )
    ax2.set_xlabel("Top band cut (% of teams)")
    ax2.set_ylabel("Estimated carry-over = 1 - noise var / observed var")
    ax2.set_title("Pre-close carry-over estimate by cut")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8, loc="center right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_out, dpi=150)
    print(f"그림 저장: {fig_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lb-csv", required=True, help="kaggle leaderboard CSV 경로")
    parser.add_argument(
        "--hm-only",
        action="store_true",
        help="OOF 없이 Hanley-McNeil 노이즈만 사용 (#206). 컷별 밴드 중앙값 AUC를 대입한다",
    )
    parser.add_argument("--n-public", type=int, default=N_PUBLIC, help="public 채점 표본 행 수")
    parser.add_argument("--prevalence", type=float, help="train 양성 비율 (--hm-only 필수)")
    parser.add_argument("--label", default="S6E8", help="그림 제목에 쓸 보드 이름")
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--fig-out", type=Path, default=FIG_OUT)
    args = parser.parse_args()

    if args.hm_only:
        if args.prevalence is None:
            raise SystemExit("--hm-only에는 --prevalence가 필요하다")
        prevalence = args.prevalence

        def noise_sd_of_band(band: np.ndarray) -> float:
            return hanley_mcneil_sd(float(np.median(band)), prevalence, args.n_public)

        def idio_sd_of_band(band: np.ndarray) -> float:
            return IDIO_NOISE_FRACTION * noise_sd_of_band(band)

        print(f"HM-only 모드: n_public={args.n_public}, 양성 비율 {prevalence:.4f}")
    else:
        pred, y, auc = load_champion_oof()
        prevalence = float(y.mean())
        print(f"champion OOF AUC 재확인: {auc:.7f} (n={len(y)}, 양성 비율 {prevalence:.4f})")

        sd_boot, mean_boot = noise_sd(pred, y, replace=True)
        sd_sub, mean_sub = noise_sd(pred, y, replace=False)
        sd_hm = hanley_mcneil_sd(auc, prevalence, args.n_public)
        print(f"노이즈 SD (부트스트랩, 복원, B={N_BOOT}): {sd_boot:.6f} (평균 AUC {mean_boot:.6f})")
        print(f"노이즈 SD (부분표집, 비복원, B={N_BOOT}): {sd_sub:.6f} (평균 AUC {mean_sub:.6f})")
        print(f"노이즈 SD (Hanley-McNeil 근사):          {sd_hm:.6f}")

        # 주 방법(이슈 명세): 부트스트랩 재표집
        def noise_sd_of_band(band: np.ndarray) -> float:
            return sd_boot

        def idio_sd_of_band(band: np.ndarray) -> float:
            return IDIOSYNCRATIC_NOISE_SD

    lb = pd.read_csv(args.lb_csv)
    scores = lb["Score"].sort_values(ascending=False).to_numpy()
    print(f"public LB: {len(scores)}팀, 1위 {scores[0]}, 중앙값 {np.median(scores):.5f}")

    sweep = cut_sweep(scores, noise_sd_of_band, idio_sd_of_band)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.csv_out, index=False)
    print(f"컷 스윕 저장: {args.csv_out}")
    print(sweep.to_string(index=False))

    plot_sweep(sweep, args.fig_out, args.label)


if __name__ == "__main__":
    main()
