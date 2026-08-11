"""원본 겹침 구조의 진입 진단. (#50)

원본 프록시(jayjoshi37 판본 1, docs/research/original-proxy-data.md)와 합성 train의
값 구조를 측정해 거리·일치 기반 후속 트랙(#52, #53, #54)을 열 가치가 있는지 판단한다.
champion과 경쟁하는 실험이 아니라 측정만 하는 진입 진단이다.

사용법:
    uv run python scripts/diagnose_original_overlap.py

프록시 CSV는 커밋하지 않는 조건(#47)이라 kagglehub 캐시 경로를 읽는다.
없으면 docs/research/original-proxy-data.md의 다운로드 절차를 따를 것.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROXY_PATH = Path.home() / (
    ".cache/kagglehub/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction/"
    "versions/1/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
)
TRAIN_PATH = Path("data/train.csv")
TARGET = "addicted_label"

NUMERIC = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day", "app_opens_per_day",
    "weekend_screen_time",
]
CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
SHARED = NUMERIC + CATEGORICAL


def grid_membership(train: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    """컬럼별: 관측값 중 프록시 정확값 눈금에 포함되는 비율과, 눈금 밖 값의 최근접 거리."""
    rows = []
    for col in SHARED:
        obs = train[col].dropna()
        grid = np.sort(proxy[col].dropna().unique())
        in_grid = obs.isin(grid)
        row = {
            "column": col,
            "n_grid_values": len(grid),
            "obs_rows": len(obs),
            "in_grid_share": float(in_grid.mean()),
        }
        if col in NUMERIC and (~in_grid).any():
            out_vals = obs[~in_grid].to_numpy(dtype=float)
            pos = np.searchsorted(grid, out_vals).clip(1, len(grid) - 1)
            nearest = np.minimum(
                np.abs(out_vals - grid[pos - 1]), np.abs(out_vals - grid[pos])
            )
            row.update(
                out_dist_p50=float(np.quantile(nearest, 0.5)),
                out_dist_p90=float(np.quantile(nearest, 0.9)),
                out_dist_max=float(nearest.max()),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def combo_existence(train: pd.DataFrame, proxy: pd.DataFrame, max_k: int = 3) -> pd.DataFrame:
    """1~3개 컬럼 조합 값이 프록시에 존재하는 비율과, 존재/부재 그룹의 타깃 차이.

    조합 컬럼이 전부 관측된 행만 대상으로 한다(프록시 설명변수에는 결측이 없어,
    결측 포함 행은 정의상 전부 부재가 되어 비율만 흐린다).
    """
    y = train[TARGET]
    rows = []
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(SHARED, k):
            mask = train[list(combo)].notna().all(axis=1)
            sub = train.loc[mask, list(combo)]
            # 문자열 키 결합은 정확값 TE와 같은 등가 기준. 프록시 쪽 키 집합과 대조한다.
            key = sub[combo[0]].astype(str)
            pkey = proxy[combo[0]].astype(str)
            for c in combo[1:]:
                key = key + "|" + sub[c].astype(str)
                pkey = pkey + "|" + proxy[c].astype(str)
            exists = key.isin(set(pkey))
            y_sub = y[mask]
            rate_in = float(y_sub[exists].mean()) if exists.any() else np.nan
            rate_out = float(y_sub[~exists].mean()) if (~exists).any() else np.nan
            rows.append({
                "k": k,
                "combo": "+".join(combo),
                "coverage": float(mask.mean()),
                "exists_share": float(exists.mean()),
                "target_in": rate_in,
                "target_out": rate_out,
                "target_gap": rate_in - rate_out if exists.any() and (~exists).any() else np.nan,
                "auc_flag": float(roc_auc_score(y_sub, exists.astype(int)))
                if 0 < exists.mean() < 1 else np.nan,
            })
    return pd.DataFrame(rows)


def nearest_row_signal(
    train: pd.DataFrame, proxy: pd.DataFrame, return_series: bool = False
) -> dict:
    """수치 9개 컬럼의 NaN 인지 정규화 거리로 최근접 프록시 행을 찾고 단독 신호를 잰다.

    행별 거리는 관측된 컬럼만의 표준화 제곱차 평균이다. 프록시 설명변수는 완전하므로
    마스크는 train 쪽에만 있다. 691k x 7.5k라 행 청크 행렬곱으로 계산한다.
    """
    scale = proxy[NUMERIC].std().to_numpy()
    P = (proxy[NUMERIC].to_numpy(dtype=float) / scale)  # (m, d) 완전
    X = (train[NUMERIC].to_numpy(dtype=float) / scale)  # (n, d) 결측 포함
    M = ~np.isnan(X)
    X0 = np.where(M, X, 0.0)
    n_obs = M.sum(axis=1)

    P2 = (P ** 2).T  # (d, m)
    n = len(X0)
    nearest_idx = np.empty(n, dtype=int)
    nearest_d2 = np.full(n, np.nan)
    chunk = 4000
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        x0, m = X0[s:e], M[s:e].astype(float)
        # d2_ij = sum_k m_ik (x_ik - p_jk)^2 = sum m x^2 - 2 (m*x) P^T + m P^2
        d2 = (
            ((x0 ** 2).sum(axis=1))[:, None]
            - 2.0 * x0 @ P.T
            + m @ P2
        )
        idx = d2.argmin(axis=1)
        nearest_idx[s:e] = idx
        nearest_d2[s:e] = d2[np.arange(e - s), idx]

    with np.errstate(invalid="ignore"):
        mean_d2 = nearest_d2 / n_obs  # 관측 컬럼 수로 정규화
    y = train[TARGET].to_numpy()
    nn_label = proxy[TARGET].to_numpy()[nearest_idx]

    ok = n_obs > 0
    if return_series:
        return {"nn_label": nn_label, "ok": ok, "mean_d2": mean_d2}
    exact = ok & (nearest_d2 < 1e-12)
    out = {
        "rows_no_numeric": int((~ok).sum()),
        "exact_row_match_share": float(exact.mean()),
        "dist_p50": float(np.nanquantile(mean_d2[ok], 0.5)),
        "dist_p90": float(np.nanquantile(mean_d2[ok], 0.9)),
        "auc_nn_label": float(roc_auc_score(y[ok], nn_label[ok])),
        "auc_neg_dist": float(roc_auc_score(y[ok], -mean_d2[ok])),
        "auc_dist": float(roc_auc_score(y[ok], mean_d2[ok])),
    }
    if exact.any():
        out["target_exact"] = float(y[exact].mean())
        out["target_nonexact"] = float(y[ok & ~exact].mean())
        out["nn_label_agree_on_exact"] = float((nn_label[exact] == y[exact]).mean())
    return out


def te_overlap_probe(train: pd.DataFrame, proxy: pd.DataFrame) -> dict:
    """최근접 프록시 행 라벨이 정확값 TE가 이미 아는 신호와 얼마나 겹치는지 잰다.

    진단 전용 근사: 컬럼별 값 키 타깃 평균(train 전체로 계산한 조잡한 TE)을 행마다
    평균한 점수를 TE 앙상블의 대리로 쓰고, nn_label과의 순위 상관 및
    소량 혼합 시 AUC 변화로 신규 정보량을 가늠한다. 폴드 규율이 없는 낙관치이므로
    절대값이 아니라 '겹침 정도'만 읽는다.
    """
    y = train[TARGET]
    te_cols = []
    for col in SHARED:
        key = train[col].astype(str).where(train[col].notna(), "__nan__")
        te_cols.append(y.groupby(key).transform("mean"))
    crude_te = pd.concat(te_cols, axis=1).mean(axis=1)

    nn = nearest_row_signal(train, proxy, return_series=True)
    nn_label = nn["nn_label"]
    ok = nn["ok"]
    blend = 0.9 * crude_te[ok].rank(pct=True) + 0.1 * pd.Series(nn_label[ok]).rank(pct=True).to_numpy()
    return {
        "auc_crude_te": float(roc_auc_score(y[ok], crude_te[ok])),
        "spearman_nn_vs_te": float(
            np.corrcoef(crude_te[ok].rank().to_numpy(), pd.Series(nn_label[ok]).rank().to_numpy())[0, 1]
        ),
        "auc_blend_90_10": float(roc_auc_score(y[ok], blend)),
    }


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    proxy = pd.read_csv(PROXY_PATH)
    # int64 컬럼의 문자열 키가 train의 float 표기("34.0")와 어긋나지 않게 통일한다.
    proxy[NUMERIC] = proxy[NUMERIC].astype(float)
    pd.set_option("display.width", 200)

    print("=== 1. 컬럼별 눈금 포함과 눈금 밖 거리 ===")
    print(grid_membership(train, proxy).to_string(index=False))

    print("\n=== 2. 조합 존재율과 타깃 차이 (전 조합, k<=3) ===")
    combos = combo_existence(train, proxy)
    for k in (1, 2, 3):
        sub = combos[combos.k == k].sort_values("auc_flag", ascending=False)
        head = sub.head(8) if k > 1 else sub
        print(f"-- k={k} ({len(sub)}개, auc_flag 상위) --")
        print(head.drop(columns="k").to_string(index=False))

    print("\n=== 3. 최근접 프록시 행 신호 (수치 9컬럼, NaN 인지 표준화 거리) ===")
    for k, v in nearest_row_signal(train, proxy).items():
        print(f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== 4. 정확값 TE와의 겹침 탐침 ===")
    for k, v in te_overlap_probe(train, proxy).items():
        print(f"{k}: {v:.5f}")


if __name__ == "__main__":
    main()
