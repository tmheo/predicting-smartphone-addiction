"""후보 생성 과정의 지문 재현과 반증 진단. (#89)

이슈 #83의 규약(docs/research/generator-fingerprint-protocol.md)을 따라, 원본
프록시(7,500행)에 후보 생성 과정을 버전·설정 고정으로 적합하고, 값 표현·공동분포·
제약·목표값 지문을 대회 train(재표본 신뢰구간)과 비교해 묶음별로 반증한다.

후보(첫 후보군, 정수 수치형 표현):
- nc_row            원본 행 복원추출 음성 대조군
- nc_col            열별 독립 복원추출 음성 대조군
- gc_beta_r1/r0     SDV 1.38.0 GaussianCopula(beta), 반올림 켬/끔
- gc_kde_r1/r0      SDV 1.38.0 GaussianCopula(수치 열 gaussian_kde), 반올림 켬/끔
- ctgan_r1/r0       SDV 1.38.0 CTGANSynthesizer 기본 설정, 반올림 켬/끔
- tvae_r1/r0        SDV 1.38.0 TVAESynthesizer 기본 설정, 반올림 켬/끔
- tabddpm_r0/r1     공식 커밋 b476257 코드, quantile 변환, y조건 MLP(256,256),
                    timesteps 1000. r1은 프록시 소수 자릿수 반올림 후처리.

#88이 전달한 재현 제약 조건: (a) 예산 부등식은 프록시에 없던 신규 구조여야 하고
자르기·투영·재계산은 반증됐으므로 거부 표본 추출 변형(rej__)을 함께 잰다.
(b) 결측 묶음은 후보 생성기가 결측을 만들지 않으므로 #88의 판정(행별 잠재
결측률)을 그대로 인용하고 여기서 다시 재지 않는다.

실행 환경: 프로젝트 기본 env가 아니라 고정 env(sdv==1.38.0, ctgan 0.12.1,
torch 2.13)를 쓴다. scripts/generator_env/requirements-lock.txt 참고.

사용법:
    python scripts/diagnose_generator_fingerprints.py competition --workdir WD
    python scripts/diagnose_generator_fingerprints.py fit --candidate CAND --workdir WD
    python scripts/diagnose_generator_fingerprints.py measure --workdir WD
    python scripts/diagnose_generator_fingerprints.py report --workdir WD
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROXY_PATH = Path("data/external/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
TARGET = "addicted_label"

NUMERIC = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day", "app_opens_per_day",
    "weekend_screen_time",
]
INT_COLS = ["age", "notifications_per_day", "app_opens_per_day"]
TIME_COLS = [c for c in NUMERIC if c not in INT_COLS]
CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
SHARED = NUMERIC + CATEGORICAL
BUDGET = ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
          "work_study_hours"]

# --- 사전 등록 상수(대회 지문과 후보 결과를 보기 전에 고정) --------------------
FIT_SEEDS = [101, 102, 103]
SAMPLE_SEEDS = [201, 202, 203]
SCREEN_N = 100_000          # 조합당 첫 선별 표본 행 수
COMP_REPS = 80              # 대회 재표본 반복 수
REP_N = 100_000             # 재표본 행 수(후보 표본과 같은 규모로 맞춤)
NN_SUB_N = 20_000           # 최근접 원본 행 거리의 부분표본
MIN_BIN_ROWS = 50           # 목표값 곡선 구간 최소 행 수
Z_PASS = 3.0                # 지표 통과 기준 표준화 거리
BUNDLE_PASS_SHARE = 0.9     # 묶음 적합: 핵심 지표의 90% 이상 z<=3
BUNDLE_MAX_Z = 6.0          # 묶음 적합: 최대 z<=6
MOCKID_MIN_ACC = 4 / 6      # 모의 식별: 6계열 중 4계열 이상 재식별해야 지표 유지
TABDDPM_STEPS = 20_000      # 규약이 비워둔 학습 단계 수. 실행 전 고정.
TABDDPM_TIMESTEPS = 1000
TABDDPM_BATCH = 4096
TABDDPM_LR = 1e-3
TABDDPM_WD = 1e-5

# 예산 잔차 경계 질량 구간(#88의 0~0.1 격자 포함)
BUDGET_EDGES = [-np.inf, -8, -6, -4, -3, -2, -1.5, -1, -0.5, -0.25, -1e-9,
                0.02, 0.04, 0.06, 0.08, 0.10, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8,
                np.inf]
# 목표값 곡선의 잔차 축 구간(프록시·대회 양쪽을 덮는 고정 격자)
TCURVE_EDGES = [-np.inf, -6, -4, -2, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.25,
                1.5, 2, 2.5, 3, 4, 6, np.inf]
# 공동분포 충돌 곡선의 사전 고정 열 부분집합
SUBSET2 = ["daily_screen_time_hours", "social_media_hours"]
SUBSET3 = ["daily_screen_time_hours", "social_media_hours", "weekend_screen_time"]
SUBSET6 = ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
           "work_study_hours", "sleep_hours", "weekend_screen_time"]

CANDIDATES = ["nc_row", "nc_col",
              "gc_beta_r1", "gc_beta_r0", "gc_kde_r1", "gc_kde_r0",
              "ctgan_r1", "ctgan_r0", "tvae_r1", "tvae_r0",
              "tabddpm_r0", "tabddpm_r1"]
FAMILY = {c: c.split("_r")[0] if "_r" in c else c for c in CANDIDATES}
FAMILY["nc_row"] = "nc_row"
FAMILY["nc_col"] = "nc_col"
FAMILY["gc_beta_r1"] = FAMILY["gc_beta_r0"] = "gc_beta"
FAMILY["gc_kde_r1"] = FAMILY["gc_kde_r0"] = "gc_kde"


# ---------------------------------------------------------------------------
# 자료 적재와 참조 객체
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_proxy() -> pd.DataFrame:
    df = pd.read_csv(PROXY_PATH)
    df = df[SHARED + [TARGET]].copy()
    for c in TIME_COLS:
        df[c] = df[c].astype(float)
    for c in INT_COLS:
        df[c] = df[c].astype("int64")
    df[TARGET] = df[TARGET].astype("int64")
    return df


def load_competition() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train = train[SHARED + [TARGET]].copy()
    test = test[SHARED].copy()
    return train, test


def build_ref(proxy: pd.DataFrame) -> dict:
    """프록시에서 파생하는 사전 등록 참조 객체. 대회 자료를 보지 않는다."""
    ref = {"grids": {}, "gaps": {}, "singletons": {}, "counts": {},
           "col_min": {}, "col_max": {}, "num_std": {}, "num_mean": {}}
    for c in NUMERIC:
        vals = proxy[c].astype(float).to_numpy()
        grid = np.sort(np.unique(vals))
        ref["grids"][c] = grid
        cnt = pd.Series(vals).value_counts()
        ref["counts"][c] = cnt
        ref["singletons"][c] = cnt[cnt == 1].index.to_numpy(dtype=float)
        ref["col_min"][c] = float(vals.min())
        ref["col_max"][c] = float(vals.max())
        ref["num_std"][c] = float(vals.std())
        ref["num_mean"][c] = float(vals.mean())
    ratio = proxy["weekend_screen_time"] / proxy["daily_screen_time_hours"]
    ref["ratio_min"] = float(ratio.min())
    ref["ratio_max"] = float(ratio.max())
    ref["daily_quartiles"] = [float(q) for q in
                              proxy["daily_screen_time_hours"].quantile([0.25, 0.5, 0.75])]
    ref["cat_values"] = {c: sorted(proxy[c].astype(str).unique()) for c in CATEGORICAL}
    ref["proxy_rows"] = proxy
    return ref


# ---------------------------------------------------------------------------
# 지문 측정
# ---------------------------------------------------------------------------

def _offgrid_metrics(vals: np.ndarray, grid: np.ndarray) -> dict:
    on = np.isin(vals, grid)
    out = {"offgrid_rate": float(1.0 - on.mean())}
    off = vals[~on]
    if off.size == 0:
        out["offgrid_reldist_p50"] = 0.0
        out["offgrid_reldist_p90"] = 0.0
        return out
    idx = np.searchsorted(grid, off)
    idx_lo = np.clip(idx - 1, 0, grid.size - 1)
    idx_hi = np.clip(idx, 0, grid.size - 1)
    lo, hi = grid[idx_lo], grid[idx_hi]
    dist = np.minimum(np.abs(off - lo), np.abs(off - hi))
    # 프록시 범위 밖 값은 국소 간격이 0이 되므로 중앙 눈금 간격으로 정규화한다.
    med_gap = float(np.median(np.diff(grid))) if grid.size > 1 else 1.0
    gap = np.where(hi > lo, hi - lo, med_gap)
    rel = dist / np.maximum(gap, 1e-12)
    out["offgrid_reldist_p50"] = float(np.quantile(rel, 0.5))
    out["offgrid_reldist_p90"] = float(np.quantile(rel, 0.9))
    return out


def _freq_metrics(vals: np.ndarray, ref: dict, col: str) -> dict:
    grid = ref["grids"][col]
    proxy_cnt = ref["counts"][col]
    scale = 7500.0 / max(vals.size, 1)
    cnt = pd.Series(vals).value_counts()
    cnt = cnt.reindex(proxy_cnt.index, fill_value=0)
    rho = float(pd.Series(cnt.values).corr(pd.Series(proxy_cnt.values),
                                           method="spearman"))
    support = float((cnt.values > 0).mean())
    singles = ref["singletons"][col]
    if singles.size:
        s_cnt = pd.Series(vals).value_counts().reindex(singles, fill_value=0)
        amp = float(s_cnt.mean() * scale)
    else:
        amp = 0.0
    return {"freq_spearman": rho, "support_cover": support, "singleton_amp": amp}


def _dup_metrics(df: pd.DataFrame, ref: dict, rng: np.random.Generator) -> dict:
    comp = df.dropna(subset=SHARED)
    n = min(len(comp), SCREEN_N)
    sub = comp.sample(n=n, random_state=int(rng.integers(2**31)))
    out = {}
    key = sub[SHARED].astype(str).agg("|".join, axis=1)
    out["dup_full_rate"] = float(1.0 - key.nunique() / n)
    for name, cols in [("collide2", SUBSET2), ("collide3", SUBSET3),
                       ("collide6", SUBSET6)]:
        k = sub[cols].astype(str).agg("|".join, axis=1)
        out[f"{name}_rate"] = float(1.0 - k.nunique() / n)
    proxy = ref["proxy_rows"].copy()
    for c in NUMERIC:
        proxy[c] = proxy[c].astype(float)
    pkey = set(proxy[SHARED].astype(str).agg("|".join, axis=1))
    out["proxy_match_rate"] = float(key.isin(pkey).mean())

    from sklearn.neighbors import KDTree
    m = min(len(sub), NN_SUB_N)
    q = sub[NUMERIC].astype(float).sample(n=m, random_state=0).to_numpy()
    stds = np.array([ref["num_std"][c] for c in NUMERIC])
    tree = KDTree(proxy[NUMERIC].astype(float).to_numpy() / stds)
    d, _ = tree.query(q / stds, k=1)
    out["nn_proxy_p50"] = float(np.quantile(d, 0.5))
    out["nn_proxy_p01"] = float(np.quantile(d, 0.01))
    return out


def _cond_metrics(df: pd.DataFrame, ref: dict) -> dict:
    out = {}
    slices: list[tuple[str, str, pd.Series]] = []
    for g in CATEGORICAL:
        for v in ref["cat_values"][g]:
            slices.append((g, str(v), df[g].astype(str) == str(v)))
    q1, q2, q3 = ref["daily_quartiles"]
    d = df["daily_screen_time_hours"]
    slices += [("dailyq", "q1", d <= q1), ("dailyq", "q2", (d > q1) & (d <= q2)),
               ("dailyq", "q3", (d > q2) & (d <= q3)), ("dailyq", "q4", d > q3)]
    for g, v, mask in slices:
        sub = df[mask]
        for c in NUMERIC:
            if c == g:
                continue
            vals = sub[c].dropna().astype(float)
            tag = f"cond__{g}_{v}__{c}"
            if len(vals) < 200:
                continue
            out[f"{tag}__mean"] = float(vals.mean())
            out[f"{tag}__q10"] = float(vals.quantile(0.1))
            out[f"{tag}__q50"] = float(vals.quantile(0.5))
            out[f"{tag}__q90"] = float(vals.quantile(0.9))
    return out


def _budget_metrics(df: pd.DataFrame, ref: dict, prefix: str = "") -> dict:
    sub = df.dropna(subset=BUDGET)
    resid = (sub["daily_screen_time_hours"] - sub["social_media_hours"]
             - sub["gaming_hours"] - sub["work_study_hours"]).to_numpy()
    out = {}
    out[f"{prefix}budget_viol_rate"] = float((resid < -1e-9).mean())
    out[f"{prefix}budget_zero_mass"] = float((np.abs(resid) <= 1e-9).mean())
    for q in (1, 10, 50):
        out[f"{prefix}budget_resid_q{q:02d}"] = float(np.quantile(resid, q / 100))
    hist, _ = np.histogram(resid, bins=BUDGET_EDGES)
    for i, h in enumerate(hist):
        out[f"{prefix}budget_bin{i:02d}"] = float(h / max(resid.size, 1))
    return out


def _constraint_metrics(df: pd.DataFrame, ref: dict) -> dict:
    out = _budget_metrics(df, ref)
    sub = df.dropna(subset=["weekend_screen_time", "daily_screen_time_hours"])
    ratio = sub["weekend_screen_time"] / sub["daily_screen_time_hours"]
    out["ratio_oob_rate"] = float(((ratio < ref["ratio_min"] - 1e-9)
                                   | (ratio > ref["ratio_max"] + 1e-9)).mean())
    oob = 0.0
    tot = 0
    for c in NUMERIC:
        vals = df[c].dropna().astype(float).to_numpy()
        oob += ((vals < ref["col_min"][c] - 1e-9)
                | (vals > ref["col_max"][c] + 1e-9)).sum()
        tot += vals.size
    out["minmax_oob_rate"] = float(oob / max(tot, 1))
    # 거부 표본 추출 변형: 예산 위반 행을 버린 뒤의 경계 질량(#88의 잔존 후보)
    keep = df.dropna(subset=BUDGET)
    resid = (keep["daily_screen_time_hours"] - keep["social_media_hours"]
             - keep["gaming_hours"] - keep["work_study_hours"])
    rej = keep[resid >= -1e-9]
    out.update(_budget_metrics(rej, ref, prefix="rej__"))
    return out


def _target_metrics(df: pd.DataFrame, ref: dict, rng: np.random.Generator) -> dict:
    out = {"target_rate": float(df[TARGET].mean())}
    sub = df.dropna(subset=BUDGET)
    resid = (sub["daily_screen_time_hours"] - sub["social_media_hours"]
             - sub["gaming_hours"] - sub["work_study_hours"]).to_numpy()
    y = sub[TARGET].to_numpy()
    idx = np.digitize(resid, TCURVE_EDGES[1:-1])
    widths = np.diff([-8 if not np.isfinite(e) and e < 0 else
                      10 if not np.isfinite(e) and e > 0 else e
                      for e in TCURVE_EDGES])
    mid_mass = 0.0
    mid08 = 0.0
    ent_w = 0.0
    ent_n = 0.0
    for b in range(len(TCURVE_EDGES) - 1):
        m = idx == b
        nb = int(m.sum())
        if nb < MIN_BIN_ROWS:
            out[f"tcurve_p_bin{b:02d}"] = float("nan")
            continue
        p = float(y[m].mean())
        out[f"tcurve_p_bin{b:02d}"] = p
        share = nb / y.size
        if 0.1 < p < 0.9:
            mid_mass += share
            mid08 += float(widths[b]) if np.isfinite(widths[b]) else 0.0
        if 0 < p < 1:
            ent = -(p * math.log(p) + (1 - p) * math.log(1 - p))
            ent_w += ent * nb
            ent_n += nb
    out["tcurve_mid_mass"] = float(mid_mass)
    out["tcurve_mid_width"] = float(mid08)
    out["tcurve_entropy"] = float(ent_w / max(ent_n, 1))
    # 같은 정확값 안의 목표값 불일치율(사전 고정 3열 키, 100k 부분표본)
    comp = df.dropna(subset=SUBSET3 + [TARGET])
    n = min(len(comp), SCREEN_N)
    s = comp.sample(n=n, random_state=int(rng.integers(2**31)))
    key = s[SUBSET3].astype(str).agg("|".join, axis=1)
    grp = s.groupby(key)[TARGET]
    sizes = grp.size()
    multi = sizes[sizes >= 2].index
    if len(multi):
        gm = grp.mean().loc[multi]
        gs = sizes.loc[multi]
        mixed = (gm > 0) & (gm < 1)
        out["samekey_disagree"] = float((gs[mixed].sum()) / gs.sum())
        out["samekey_multi_share"] = float(gs.sum() / n)
    else:
        out["samekey_disagree"] = 0.0
        out["samekey_multi_share"] = 0.0
    return out


def compute_metrics(df: pd.DataFrame, ref: dict, with_target: bool,
                    seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # 수치 열을 float로 통일해 문자열 키('21' 대 '21.0') 불일치를 막는다.
    df = df.copy()
    for c in NUMERIC:
        df[c] = df[c].astype(float)
    out = {}
    for c in NUMERIC:
        vals = df[c].dropna().astype(float).to_numpy()
        for k, v in _offgrid_metrics(vals, ref["grids"][c]).items():
            out[f"{k}__{c}"] = v
        for k, v in _freq_metrics(vals, ref, c).items():
            out[f"{k}__{c}"] = v
    out.update(_dup_metrics(df, ref, rng))
    out.update(_cond_metrics(df, ref))
    out.update(_constraint_metrics(df, ref))
    if with_target and TARGET in df.columns:
        out.update(_target_metrics(df, ref, rng))
    return out


BUNDLE_OF_PREFIX = [
    ("offgrid_", "V"), ("freq_", "V"), ("support_", "V"), ("singleton_", "V"),
    ("dup_", "J"), ("collide", "J"), ("proxy_match", "J"), ("nn_", "J"),
    ("cond__", "J"),
    ("budget_", "C"), ("rej__", "C"), ("ratio_", "C"), ("minmax_", "C"),
    ("target_rate", "T"), ("tcurve_", "T"), ("samekey_", "T"),
]


def bundle_of(metric: str) -> str | None:
    for pre, b in BUNDLE_OF_PREFIX:
        if metric.startswith(pre):
            return b
    return None


# 묶음별 핵심 지표(판정에 쓰는 부분집합). cond__와 tcurve_p는 이름 앞부분으로 지정.
CORE_PATTERNS = {
    "V": ["offgrid_rate__", "offgrid_reldist_p50__", "offgrid_reldist_p90__",
          "freq_spearman__", "support_cover__", "singleton_amp__"],
    "J": ["dup_full_rate", "collide2_rate", "collide3_rate", "collide6_rate",
          "proxy_match_rate", "nn_proxy_p50", "nn_proxy_p01", "cond__"],
    "C": ["budget_viol_rate", "budget_zero_mass", "budget_resid_q",
          "budget_bin", "ratio_oob_rate", "minmax_oob_rate", "rej__budget_bin"],
    "T": ["target_rate", "tcurve_p_bin", "tcurve_mid_mass", "tcurve_entropy",
          "samekey_disagree"],
}


def is_core(metric: str) -> bool:
    b = bundle_of(metric)
    if b is None:
        return False
    return any(metric.startswith(p) for p in CORE_PATTERNS[b])


# ---------------------------------------------------------------------------
# 후보 적합과 표본 생성
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def sdv_metadata(proxy: pd.DataFrame):
    from sdv.metadata import SingleTableMetadata
    md = SingleTableMetadata()
    md.detect_from_dataframe(proxy)
    for c in CATEGORICAL + [TARGET]:
        md.update_column(c, sdtype="categorical")
    for c in NUMERIC:
        md.update_column(c, sdtype="numerical")
    return md


def make_synthesizer(cand: str, md, fit_seed: int):
    from sdv.single_table import (CTGANSynthesizer, GaussianCopulaSynthesizer,
                                  TVAESynthesizer)
    rounding = cand.endswith("_r1")
    if cand.startswith("gc_beta"):
        return GaussianCopulaSynthesizer(
            md, enforce_min_max_values=True, enforce_rounding=rounding,
            default_distribution="beta")
    if cand.startswith("gc_kde"):
        return GaussianCopulaSynthesizer(
            md, enforce_min_max_values=True, enforce_rounding=rounding,
            default_distribution="beta",
            numerical_distributions={c: "gaussian_kde" for c in NUMERIC})
    if cand.startswith("ctgan"):
        return CTGANSynthesizer(
            md, enforce_min_max_values=True, enforce_rounding=rounding,
            epochs=300, batch_size=500, embedding_dim=128,
            generator_dim=(256, 256), discriminator_dim=(256, 256),
            log_frequency=True, pac=10, verbose=False, cuda=False)
    if cand.startswith("tvae"):
        return TVAESynthesizer(
            md, enforce_min_max_values=True, enforce_rounding=rounding,
            epochs=300, batch_size=500, embedding_dim=128,
            compress_dims=(128, 128), decompress_dims=(128, 128),
            l2scale=1e-5, loss_factor=2, cuda=False)
    raise ValueError(cand)


def fit_negative_control(cand: str, proxy: pd.DataFrame, out_dir: Path) -> None:
    for fs in FIT_SEEDS:
        for ss in SAMPLE_SEEDS:
            path = out_dir / f"f{fs}_s{ss}.parquet"
            if path.exists():
                continue
            rng = np.random.default_rng(fs * 10_000 + ss)
            if cand == "nc_row":
                idx = rng.integers(0, len(proxy), SCREEN_N)
                sample = proxy.iloc[idx].reset_index(drop=True)
            else:
                sample = pd.DataFrame({
                    c: proxy[c].to_numpy()[rng.integers(0, len(proxy), SCREEN_N)]
                    for c in SHARED + [TARGET]})
            sample.to_parquet(path)
            print(f"[{cand}] wrote {path.name}")


def patch_fast_kde_ppf() -> None:
    """copulas GaussianKDE.percent_point를 고해상도 격자 보간으로 가속한다.

    기본 구현(chandrupatla 이분법)은 100k행 표본 추출에 후보당 약 8시간이
    걸려 실행 불가능하다. KDE 누적분포는 매끄러우므로 20만 점 격자에서
    정확한 CDF를 계산한 뒤 역방향 선형 보간해도 수치적으로 등가다
    (실행 시 정확 해와 최대 절대 오차를 검증해 기록한다). 분포 자체와
    적합은 SDV 1.38.0 그대로다.
    """
    from copulas.univariate import gaussian_kde as gk

    grid_n = 200_000
    orig = gk.GaussianKDE.percent_point

    def fast_percent_point(self, U, method="chandrupatla"):
        self.check_fit()
        cache = getattr(self, "_fast_ppf_cache", None)
        if cache is None:
            lower, upper = self._get_bounds()
            grid = np.linspace(lower, upper, grid_n)
            cdf = np.empty(grid_n)
            for i in range(0, grid_n, 10_000):
                cdf[i:i + 10_000] = self.cumulative_distribution(
                    grid[i:i + 10_000])
            cdf = np.maximum.accumulate(cdf)
            cache = (grid, cdf)
            self._fast_ppf_cache = cache
        grid, cdf = cache
        U = np.asarray(U, dtype=float)
        return np.interp(U, cdf, grid)

    fast_percent_point._orig = orig
    gk.GaussianKDE.percent_point = fast_percent_point


def fit_sdv(cand: str, proxy: pd.DataFrame, out_dir: Path) -> None:
    if cand.startswith("gc_kde"):
        patch_fast_kde_ppf()
    md = sdv_metadata(proxy)
    for fs in FIT_SEEDS:
        targets = [out_dir / f"f{fs}_s{ss}.parquet" for ss in SAMPLE_SEEDS]
        if all(t.exists() for t in targets):
            continue
        set_seeds(fs)
        synth = make_synthesizer(cand, md, fs)
        synth.fit(proxy)
        for ss, path in zip(SAMPLE_SEEDS, targets):
            if path.exists():
                continue
            set_seeds(ss)
            sample = synth.sample(num_rows=SCREEN_N)
            sample.to_parquet(path)
            print(f"[{cand}] wrote {path.name}", flush=True)


def fit_tabddpm(proxy: pd.DataFrame, out_dir_r0: Path, out_dir_r1: Path,
                repo: Path) -> None:
    import torch
    from sklearn.preprocessing import QuantileTransformer
    sys.path.insert(0, str(repo))
    from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
    from tab_ddpm.modules import MLPDiffusion

    device = torch.device("cpu")  # float64 버퍼 때문에 MPS 불가
    cat_maps = {c: {v: i for i, v in enumerate(sorted(proxy[c].unique()))}
                for c in CATEGORICAL}
    K = np.array([len(cat_maps[c]) for c in CATEGORICAL])
    y_np = proxy[TARGET].to_numpy()
    y_dist = torch.tensor(np.bincount(y_np), dtype=torch.float)

    for fs in FIT_SEEDS:
        targets = [(out_dir_r0 / f"f{fs}_s{ss}.parquet",
                    out_dir_r1 / f"f{fs}_s{ss}.parquet") for ss in SAMPLE_SEEDS]
        if all(a.exists() and b.exists() for a, b in targets):
            continue
        set_seeds(fs)
        qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(len(proxy) // 30, 1000), 10),
            subsample=int(1e9), random_state=fs)
        X_num = qt.fit_transform(proxy[NUMERIC].astype(float).to_numpy())
        X_cat = np.stack([proxy[c].map(cat_maps[c]).to_numpy()
                          for c in CATEGORICAL], axis=1)
        X = np.concatenate([X_num, X_cat], axis=1).astype(np.float32)

        model = MLPDiffusion(
            d_in=int(len(NUMERIC) + K.sum()), num_classes=2, is_y_cond=True,
            rtdl_params={"d_layers": [256, 256], "dropout": 0.0})
        diffusion = GaussianMultinomialDiffusion(
            num_classes=K, num_numerical_features=len(NUMERIC),
            denoise_fn=model, num_timesteps=TABDDPM_TIMESTEPS,
            gaussian_loss_type="mse", scheduler="cosine", device=device)
        diffusion.to(device).train()
        opt = torch.optim.AdamW(diffusion.parameters(), lr=TABDDPM_LR,
                                weight_decay=TABDDPM_WD)
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y_np.astype(np.int64))
        n = len(proxy)
        rng = np.random.default_rng(fs)
        for step in range(TABDDPM_STEPS):
            idx = rng.integers(0, n, TABDDPM_BATCH)
            xb = Xt[idx].to(device)
            yb = {"y": yt[idx].to(device)}
            opt.zero_grad()
            loss_multi, loss_gauss = diffusion.mixed_loss(xb, yb)
            (loss_multi + loss_gauss).backward()
            opt.step()
            lr = TABDDPM_LR * (1 - step / TABDDPM_STEPS)
            for g in opt.param_groups:
                g["lr"] = lr
            if (step + 1) % 2000 == 0:
                print(f"[tabddpm f{fs}] step {step+1}/{TABDDPM_STEPS} "
                      f"mloss={loss_multi.item():.4f} gloss={loss_gauss.item():.4f}",
                      flush=True)
        diffusion.eval()
        for ss, (p0, p1) in zip(SAMPLE_SEEDS, targets):
            if p0.exists() and p1.exists():
                continue
            set_seeds(ss)
            xs, ys = [], []
            remaining = SCREEN_N
            while remaining > 0:
                b = min(20_000, remaining)
                x_gen, y_gen = diffusion.sample_all(b, b, y_dist.clone(),
                                                    ddim=False)
                xs.append(x_gen.cpu().numpy())
                ys.append(y_gen.cpu().numpy())
                remaining -= b
            Xg = np.concatenate(xs)[:SCREEN_N]
            yg = np.concatenate(ys)[:SCREEN_N]
            num = qt.inverse_transform(Xg[:, :len(NUMERIC)])
            sample = pd.DataFrame(num, columns=NUMERIC)
            for j, c in enumerate(CATEGORICAL):
                inv = {i: v for v, i in cat_maps[c].items()}
                sample[c] = pd.Series(
                    Xg[:, len(NUMERIC) + j].astype(int)).map(inv).to_numpy()
            sample[TARGET] = yg.astype("int64")
            sample.to_parquet(p0)
            r1 = sample.copy()
            for c in INT_COLS:
                r1[c] = r1[c].round(0)
            for c in TIME_COLS:
                r1[c] = r1[c].round(2)
            r1.to_parquet(p1)
            print(f"[tabddpm] wrote f{fs}_s{ss} (r0, r1)", flush=True)


# ---------------------------------------------------------------------------
# 단계 실행
# ---------------------------------------------------------------------------

def phase_competition(workdir: Path) -> None:
    proxy = load_proxy()
    ref = build_ref(proxy)
    train, test = load_competition()
    for name, df, with_t in [("train", train, True), ("test", test, False)]:
        rows = []
        reps = COMP_REPS if name == "train" else COMP_REPS // 2
        for r in range(reps):
            sub = df.sample(n=min(REP_N, len(df)), random_state=r)
            rows.append(compute_metrics(sub, ref, with_target=with_t, seed=r))
            if (r + 1) % 10 == 0:
                print(f"[comp {name}] rep {r+1}/{reps}", flush=True)
        tab = pd.DataFrame(rows)
        stats = {c: {"mean": float(tab[c].mean()), "sd": float(tab[c].std())}
                 for c in tab.columns}
        (workdir / f"comp_{name}.json").write_text(json.dumps(stats))
    meta = {"proxy_sha256": sha256(PROXY_PATH), "train_sha256": sha256(TRAIN_PATH),
            "test_sha256": sha256(TEST_PATH), "comp_reps": COMP_REPS,
            "rep_n": REP_N}
    (workdir / "comp_meta.json").write_text(json.dumps(meta, indent=2))
    # 프록시 자신의 지문(참조용)
    m = compute_metrics(proxy, ref, with_target=True, seed=0)
    (workdir / "proxy_metrics.json").write_text(json.dumps(m))
    print("competition targets registered")


def phase_fit(cand: str, workdir: Path, tabddpm_repo: Path | None) -> None:
    proxy = load_proxy()
    out_dir = workdir / "samples" / cand
    out_dir.mkdir(parents=True, exist_ok=True)
    if cand in ("nc_row", "nc_col"):
        fit_negative_control(cand, proxy, out_dir)
    elif cand.startswith(("gc_", "ctgan", "tvae")):
        fit_sdv(cand, proxy, out_dir)
    elif cand == "tabddpm":
        r0 = workdir / "samples" / "tabddpm_r0"
        r1 = workdir / "samples" / "tabddpm_r1"
        r0.mkdir(parents=True, exist_ok=True)
        r1.mkdir(parents=True, exist_ok=True)
        assert tabddpm_repo is not None
        fit_tabddpm(proxy, r0, r1, tabddpm_repo)
    else:
        raise ValueError(cand)


def phase_measure(workdir: Path) -> None:
    proxy = load_proxy()
    ref = build_ref(proxy)
    for cand in CANDIDATES:
        sdir = workdir / "samples" / cand
        mdir = workdir / "metrics" / cand
        mdir.mkdir(parents=True, exist_ok=True)
        if not sdir.exists():
            continue
        for p in sorted(sdir.glob("f*_s*.parquet")):
            out = mdir / (p.stem + ".json")
            if out.exists():
                continue
            df = pd.read_parquet(p)
            for c in NUMERIC:
                df[c] = df[c].astype(float)
            m = compute_metrics(df, ref, with_target=True,
                                seed=hash(p.stem) % (2**31))
            out.write_text(json.dumps(m))
            print(f"[measure] {cand}/{p.stem}", flush=True)


def load_candidate_metrics(workdir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for cand in CANDIDATES:
        mdir = workdir / "metrics" / cand
        if not mdir.exists():
            continue
        rows = {}
        for p in sorted(mdir.glob("f*_s*.json")):
            rows[p.stem] = json.loads(p.read_text())
        if rows:
            out[cand] = pd.DataFrame(rows).T
    return out


def phase_report(workdir: Path) -> None:
    comp = json.loads((workdir / "comp_train.json").read_text())
    cands = load_candidate_metrics(workdir)
    metrics = sorted(set().union(*[set(t.columns) for t in cands.values()]))
    metrics = [m for m in metrics if bundle_of(m) is not None]

    # 모의 식별: f103_s203을 보류하고 나머지 평균·표준편차로 계열을 재식별한다.
    hold_key = f"f{FIT_SEEDS[-1]}_s{SAMPLE_SEEDS[-1]}"
    excluded = []
    kept = []
    for m in metrics:
        ok, total = 0, 0
        groups = {}
        for cand, tab in cands.items():
            if m not in tab.columns or hold_key not in tab.index:
                continue
            rest = tab.drop(index=hold_key)[m].astype(float)
            if rest.isna().any() or np.isnan(tab.loc[hold_key, m]):
                continue
            groups[cand] = (float(rest.mean()), float(rest.std()) + 1e-12,
                            float(tab.loc[hold_key, m]))
        fam_seen = set()
        for cand, (mu, sd, hold) in groups.items():
            fam = FAMILY[cand]
            if fam in fam_seen:
                continue
            fam_seen.add(fam)
            best_fam, best_d = None, np.inf
            for c2, (mu2, sd2, _) in groups.items():
                d = abs(hold - mu2) / sd2
                if d < best_d:
                    best_fam, best_d = FAMILY[c2], d
            total += 1
            if best_fam == fam:
                ok += 1
        if total >= 4 and ok / total >= MOCKID_MIN_ACC:
            kept.append(m)
        else:
            excluded.append(m)

    core_kept = [m for m in kept if is_core(m) and m in comp
                 and not np.isnan(comp[m]["mean"])]
    lines = ["# 지문 판정 리포트", "",
             f"- 전체 지표 {len(metrics)}, 모의 식별 통과 {len(kept)}, "
             f"판정용 핵심 지표 {len(core_kept)}", ""]
    verdicts = {}
    for cand, tab in cands.items():
        row = {}
        for b in ["V", "J", "C", "T"]:
            zs = []
            worst = []
            for m in core_kept:
                if bundle_of(m) != b or m not in tab.columns:
                    continue
                vals = tab[m].astype(float).dropna()
                if not len(vals):
                    continue
                c_mean, c_sd = comp[m]["mean"], comp[m]["sd"]
                z = abs(vals.mean() - c_mean) / math.sqrt(
                    c_sd**2 + float(vals.std() or 0)**2 + 1e-12)
                zs.append(z)
                worst.append((z, m))
            if not zs:
                row[b] = None
                continue
            zs_arr = np.array(zs)
            worst.sort(reverse=True)
            row[b] = {
                "n": len(zs), "median_z": float(np.median(zs_arr)),
                "max_z": float(zs_arr.max()),
                "pass_share": float((zs_arr <= Z_PASS).mean()),
                "verdict": ("적합" if (zs_arr <= Z_PASS).mean() >= BUNDLE_PASS_SHARE
                            and zs_arr.max() <= BUNDLE_MAX_Z else "반증"),
                "worst": [(round(z, 1), m) for z, m in worst[:5]],
            }
        verdicts[cand] = row
    (workdir / "verdicts.json").write_text(json.dumps(verdicts, indent=2,
                                                      ensure_ascii=False))
    (workdir / "mockid_excluded.json").write_text(json.dumps(excluded))

    lines.append("| 후보 | V 값표현 | J 공동분포 | C 제약 | T 목표값 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for cand in CANDIDATES:
        if cand not in verdicts:
            continue
        cells = []
        for b in ["V", "J", "C", "T"]:
            r = verdicts[cand][b]
            cells.append("-" if r is None else
                         f"{r['verdict']} (중앙 z {r['median_z']:.1f}, "
                         f"최대 {r['max_z']:.0f}, 통과 {r['pass_share']:.0%})")
        lines.append(f"| {cand} | " + " | ".join(cells) + " |")
    report = "\n".join(lines)
    (workdir / "report.md").write_text(report)
    print(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["competition", "fit", "measure", "report"])
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--candidate")
    ap.add_argument("--tabddpm-repo", type=Path)
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.phase == "competition":
        phase_competition(args.workdir)
    elif args.phase == "fit":
        phase_fit(args.candidate, args.workdir, args.tabddpm_repo)
    elif args.phase == "measure":
        phase_measure(args.workdir)
    else:
        phase_report(args.workdir)


if __name__ == "__main__":
    main()
