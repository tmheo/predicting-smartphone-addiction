"""carry-over 사전 추정기를 완료 에피소드로 보정한다. (#206)

#204의 분산 분해 추정기(`scripts/estimate_carryover.py`)를 정답지가 있는
S6 완료 AUC 에피소드 3개(S6E2 전멸, S6E3 유지, S6E5 유지)에 적용해,
사전 추정치가 사후 체제를 분리하는지와 어느 노이즈 가정이 현실에
가까운지를 확인한다.

에피소드별 계산:

1. 마감 후 public LB(kaggle leaderboard download 엔드포인트)로 컷 스윕.
   OOF가 없으므로 노이즈는 Hanley-McNeil 근사(컷별 밴드 중앙값 AUC 대입).
2. 사후 기준: georgymamarin의 S6 리더보드 데이터셋(public+private 결합,
   Apache-2.0)에서 컷별 private~public Theil-Sen 기울기를 재계산.
3. 역산: 사후 기울기를 재현하는 유효 노이즈 SD와 그 HM 대비 비율.
   이 비율이 에피소드 간에 일관되면 S6E8에 소급 적용할 수 있다.

사용법:
    uv run python scripts/calibrate_carryover.py

산출: 표준 출력 리포트, run-logs/carryover_calibration_sweep.csv,
docs/research/assets/carryover-calibration-completed.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from estimate_carryover import (
    CUTS_PCT,
    ERASED_BAND,
    IDIO_NOISE_FRACTION,
    KEPT_BAND,
    cut_sweep,
    hanley_mcneil_sd,
)

PUBLIC_RATIO = 0.2  # Meta Kaggle Competitions.csv LeaderboardPercentage, 세 에피소드 공통

# n_test와 양성 비율의 출처는 docs/research/carryover-calibration-completed-episodes.md 참조
EPISODES = {
    "S6E2": {"n_test": 270_000, "prevalence": 0.44834, "regime": "erased"},
    "S6E3": {"n_test": 254_655, "prevalence": 0.2252, "regime": "kept"},
    "S6E5": {"n_test": 188_165, "prevalence": 0.19898, "regime": "kept"},
}

# S6E8 소급 적용용: #204 스윕(docs/research/carryover-preclose-estimate.md 표)의
# 관측 SD와 부트스트랩 노이즈 SD
S6E8_NOISE_SD = 0.000573
S6E8_OBS_SD = {17: 0.000171, 20: 0.000306, 25: 0.000555, 30: 0.000877}

LB_DIR = Path("run-logs/kaggle-out/issue206")
JOINED_CSV = LB_DIR / "s6-leaderboards" / "s6_leaderboards.csv"
CSV_OUT = Path("run-logs/carryover_calibration_sweep.csv")
FIG_OUT = Path("docs/research/assets/carryover-calibration-completed.png")


def load_fresh_lb(episode: str) -> pd.DataFrame:
    slug = f"playground-series-{episode.lower()}"
    matches = sorted((LB_DIR / episode.lower()).glob(f"{slug}-publicleaderboard-*.csv"))
    if not matches:
        raise SystemExit(f"{episode} public LB CSV가 없다: {LB_DIR / episode.lower()}")
    return pd.read_csv(matches[-1])


def cross_check(episode: str, fresh: pd.DataFrame, joined: pd.DataFrame) -> None:
    """마감 후 재채점·팀 삭제 여부 점검: 두 독립 스냅샷의 public 점수를 대조한다."""
    ep = joined[(joined["episode"] == episode) & (~joined["is_host_baseline"])]
    merged = fresh.merge(
        ep[["team_id", "public_score"]], left_on="TeamId", right_on="team_id", how="inner"
    )
    diff = (merged["Score"] - merged["public_score"]).abs()
    print(
        f"  교차 검증: 최신 다운로드 {len(fresh)}팀 vs 데이터셋 {len(ep)}팀, "
        f"team_id 일치 {len(merged)}팀, 점수 차 최대 {diff.max():.6f}, "
        f"불일치(>1e-9) {int((diff > 1e-9).sum())}팀"
    )


def posthoc_slope(episode: str, joined: pd.DataFrame) -> pd.DataFrame:
    """컷별 private~public Theil-Sen 기울기(노트북의 사후 carry-over 정의)."""
    ep = joined[(joined["episode"] == episode) & (~joined["is_host_baseline"])]
    ep = ep.sort_values("public_rank")
    rows = []
    for pct in CUTS_PCT:
        k = max(int(round(len(ep) * pct / 100)), 5)
        band = ep.head(k)
        slope = theilslopes(band["private_score"], band["public_score"]).slope
        rows.append({"cut_pct": pct, "posthoc_slope": float(slope)})
    return pd.DataFrame(rows)


def calibrate_episode(episode: str, joined: pd.DataFrame) -> pd.DataFrame:
    cfg = EPISODES[episode]
    n_public = round(cfg["n_test"] * PUBLIC_RATIO)
    prevalence = cfg["prevalence"]
    print(f"{episode} ({cfg['regime']}): n_public={n_public:,}, 양성 비율 {prevalence:.4f}")

    fresh = load_fresh_lb(episode)
    cross_check(episode, fresh, joined)
    scores = fresh["Score"].sort_values(ascending=False).to_numpy()

    def noise_sd_of_band(band: np.ndarray) -> float:
        return hanley_mcneil_sd(float(np.median(band)), prevalence, n_public)

    def idio_sd_of_band(band: np.ndarray) -> float:
        return IDIO_NOISE_FRACTION * noise_sd_of_band(band)

    sweep = cut_sweep(scores, noise_sd_of_band, idio_sd_of_band)
    sweep = sweep.merge(posthoc_slope(episode, joined), on="cut_pct")

    # 역산: 사후 기울기를 분산 비율 정의로 재현하는 유효 노이즈 SD와 HM 대비 비율.
    # 기울기가 [0, 1] 밖이면 분산 비율로 해석할 수 없어 NaN으로 둔다.
    valid = (sweep["posthoc_slope"] >= 0) & (sweep["posthoc_slope"] <= 1)
    sweep["effective_noise_sd"] = np.where(
        valid, sweep["obs_sd"] * np.sqrt((1 - sweep["posthoc_slope"]).clip(lower=0)), np.nan
    )
    sweep["effective_noise_frac"] = sweep["effective_noise_sd"] / sweep["noise_sd"]
    sweep.insert(0, "episode", episode)
    sweep.insert(1, "regime", cfg["regime"])
    return sweep


def plot_calibration(all_sweeps: pd.DataFrame) -> None:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    episodes = list(EPISODES)
    fig, axes = plt.subplots(2, len(episodes), figsize=(15, 8), sharex=True)

    for i, ep in enumerate(episodes):
        s = all_sweeps[all_sweeps["episode"] == ep]
        regime = EPISODES[ep]["regime"]

        ax = axes[0][i]
        ax.axhspan(*KEPT_BAND, color="#2ca02c", alpha=0.15, label="Kept boards (post-hoc)")
        ax.axhspan(*ERASED_BAND, color="#d62728", alpha=0.15, label="Erased boards (post-hoc)")
        ax.plot(
            s["cut_pct"],
            s["carryover_full_noise"],
            "o-",
            color="#1f77b4",
            label="Estimate (independent HM noise)",
        )
        ax.plot(
            s["cut_pct"],
            s["carryover_idio_noise"],
            "s--",
            color="#ff7f0e",
            label="Estimate (idiosyncratic floor)",
        )
        ax.plot(
            s["cut_pct"],
            s["posthoc_slope"],
            "k^-",
            markersize=5,
            label="Post-hoc Theil-Sen slope",
        )
        ax.set_title(f"{ep} ({regime})")
        ax.set_ylim(-0.05, 1.25)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("Carry-over")
            ax.legend(fontsize=7, loc="upper left")

        ax = axes[1][i]
        ax.plot(s["cut_pct"], s["obs_sd"], "o-", color="#1f77b4", label="Observed band SD")
        ax.plot(s["cut_pct"], s["noise_sd"], "--", color="#d62728", label="HM noise SD")
        ax.plot(
            s["cut_pct"],
            s["effective_noise_sd"],
            "k^-",
            markersize=5,
            label="Effective noise SD (from post-hoc)",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Top band cut (% of teams)")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("Public score SD (AUC)")
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Calibrating the pre-close carry-over estimator on finished S6 AUC boards")
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    print(f"그림 저장: {FIG_OUT}")


def s6e8_retro(all_sweeps: pd.DataFrame) -> None:
    """보정된 유효 노이즈 비율을 S6E8의 #204 관측 SD에 소급 적용한다."""
    print("\nS6E8 소급 적용 (carry-over = 1 - (frac x 0.000573 / obs_sd)^2):")
    at25 = all_sweeps[all_sweeps["cut_pct"] == 25]
    fracs = at25.dropna(subset=["effective_noise_frac"])
    for _, row in fracs.iterrows():
        frac = row["effective_noise_frac"]
        line = [f"  frac={frac:.3f} ({row['episode']}, {row['regime']}):"]
        for cut, obs in sorted(S6E8_OBS_SD.items()):
            co = max(0.0, 1 - (frac * S6E8_NOISE_SD / obs) ** 2)
            line.append(f"{cut}% cut -> {co:.2f}")
        print(" ".join(line))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    joined = pd.read_csv(JOINED_CSV)
    sweeps = [calibrate_episode(ep, joined) for ep in EPISODES]
    all_sweeps = pd.concat(sweeps, ignore_index=True)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    all_sweeps.to_csv(CSV_OUT, index=False)
    print(f"\n보정 스윕 저장: {CSV_OUT}")
    cols = [
        "episode",
        "regime",
        "cut_pct",
        "n_teams",
        "obs_sd",
        "noise_sd",
        "carryover_full_noise",
        "carryover_idio_noise",
        "posthoc_slope",
        "effective_noise_frac",
    ]
    print(all_sweeps[cols].to_string(index=False))

    plot_calibration(all_sweeps)
    s6e8_retro(all_sweeps)


if __name__ == "__main__":
    main()
