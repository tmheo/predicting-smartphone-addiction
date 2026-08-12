"""숨은 제약과 결측 주입 구조의 진입 진단. (#88)

이슈 #85의 판별 규약(docs/research/constraint-and-generation-order-diagnostics.md)을
따라, 원본 프록시와 대회 train을 각각 발견 50% / 확인 50%로 층화 분할하고 test를
독립 복제 자료로 써서 다음을 측정한다.

- 스키마, 단변량 범위, 값 눈금(소수 자릿수), 경계 정확값 질량
- 함수 종속성과 조건부 함수 종속성
- 작은 산술 문법(합·차·비율·최소·최대, 작은 유리수 계수, 깊이 2)의 완전 열거
- 조건부 범위(범주 및 age 구간 조건)
- 결측 마스크 의존: 열별 비율, 마스크 상관, 행별 결측 개수의 Poisson-binomial 적합,
  관측 X와 Y에 대한 교차적합 예측 가능성

식, 절편, 허용오차, 최소 지지는 발견 자료에서만 정하고 확인 자료에서 바꾸지 않는다.
champion과 경쟁하는 실험이 아니라 측정만 하는 진입 진단이다.

사용법:
    uv run python scripts/diagnose_hidden_constraints.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROXY_PATH = Path("data/external/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
TARGET = "addicted_label"
SEED = 88  # 이 진단 전용 분할 시드. 발견/확인 분할에만 쓴다.

NUMERIC = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day", "app_opens_per_day",
    "weekend_screen_time",
]
CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
SHARED = NUMERIC + CATEGORICAL

# 사전 등록 허용오차와 지지 기준(발견 자료를 보기 전에 고정).
MACHINE_TOL = 1e-9          # 저장값의 기계 정밀도 수준 정확 일치
DISPLAY_UNIT = {            # 프록시 표시 단위. 반올림 설명 가능 일치의 오차 전파에 쓴다.
    "age": 1.0, "notifications_per_day": 1.0, "app_opens_per_day": 1.0,
    "daily_screen_time_hours": 0.01, "social_media_hours": 0.01, "gaming_hours": 0.01,
    "work_study_hours": 0.01, "sleep_hours": 0.01, "weekend_screen_time": 0.01,
}
COEFS = [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]
MIN_SUPPORT_FRAC = 0.5      # 산술식의 유효(유한 잔차) 행 비율 최소값
EQ_SHARE = 0.999            # 발견 자료 등식 후보 기준: 표시 허용오차 내 비율
INEQ_VIOL = 0.001           # 발견 자료 부등식 후보 기준: 위반 비율 상한
INEQ_BOUNDARY = 0.005       # 부등식이 자명하지 않으려면 경계 근방 질량이 이만큼은 있어야 함
FD_MINSUP_PROXY = 3         # 함수 종속성 키 최소 행 수(프록시 발견)
FD_MINSUP_TRAIN = 5         # 함수 종속성 키 최소 행 수(train 발견)
FD_MIN_KEYS = 20            # 지지 키가 이보다 적으면 우연 성립으로 보고 보고하지 않음


# ---------------------------------------------------------------------------
# 자료 적재와 분할
# ---------------------------------------------------------------------------

def load_datasets() -> dict[str, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    proxy = pd.read_csv(PROXY_PATH)
    proxy[NUMERIC] = proxy[NUMERIC].astype(float)

    p_disc, p_conf = train_test_split(
        proxy, test_size=0.5, random_state=SEED, stratify=proxy[TARGET]
    )
    t_disc, t_conf = train_test_split(
        train, test_size=0.5, random_state=SEED, stratify=train[TARGET]
    )
    return {
        "proxy_disc": p_disc, "proxy_conf": p_conf,
        "train_disc": t_disc, "train_conf": t_conf,
        "test": test, "train": train,
    }


# ---------------------------------------------------------------------------
# 1. 스키마, 값 눈금, 경계 질량
# ---------------------------------------------------------------------------

def decimal_share(s: pd.Series, max_d: int = 6) -> dict[int, float]:
    """관측값이 소수 d자리로 정확히 표현되는 최소 d의 분포."""
    v = s.dropna().to_numpy(dtype=float)
    out: dict[int, float] = {}
    remaining = np.ones(len(v), dtype=bool)
    for d in range(max_d + 1):
        scaled = v * 10**d
        ok = remaining & np.isclose(scaled, np.round(scaled), atol=1e-6)
        if ok.any():
            out[d] = float(ok.mean())
            remaining &= ~ok
    if remaining.any():
        out[max_d + 1] = float(remaining.mean())
    return out


def schema_table(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name in ["proxy_disc", "proxy_conf", "train_disc", "train_conf", "test"]:
        df = sets[name]
        for col in NUMERIC:
            v = df[col].dropna()
            dec = decimal_share(v)
            rows.append({
                "set": name, "column": col, "n": len(v),
                "min": float(v.min()), "max": float(v.max()),
                "n_unique": int(v.nunique()),
                "mass_at_min": float((v == v.min()).mean()),
                "mass_at_max": float((v == v.max()).mean()),
                "share_dec_le2": float(sum(s for d, s in dec.items() if d <= 2)),
                "max_decimals": max(dec),
            })
    return pd.DataFrame(rows)


def boundary_lineage(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """프록시 발견 자료의 전역 경계를 기준으로 대회 자료의 보존/완화/신규를 판정."""
    rows = []
    disc = sets["proxy_disc"]
    for col in NUMERIC:
        lo, hi = float(disc[col].min()), float(disc[col].max())
        row: dict[str, object] = {"column": col, "proxy_min": lo, "proxy_max": hi}
        for name in ["proxy_conf", "train_conf", "test"]:
            v = sets[name][col].dropna()
            row[f"{name}_below"] = float((v < lo - MACHINE_TOL).mean())
            row[f"{name}_above"] = float((v > hi + MACHINE_TOL).mean())
            row[f"{name}_at_lo"] = float((v == lo).mean())
            row[f"{name}_at_hi"] = float((v == hi).mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. 함수 종속성과 조건부 함수 종속성
# ---------------------------------------------------------------------------

def fd_scan(disc: pd.DataFrame, minsup: int, cols: list[str]) -> pd.DataFrame:
    """발견 자료에서 A -> B 정확 성립 후보(지지 키에서 위반 0)를 찾는다."""
    rows = []
    for a, b in itertools.permutations(cols, 2):
        sub = disc[[a, b]].dropna()
        if sub.empty:
            continue
        g = sub.groupby(a, observed=True)[b]
        cnt = g.count()
        keys = cnt[cnt >= minsup].index
        if len(keys) < FD_MIN_KEYS:
            continue
        nun = g.nunique()
        viol_keys = int((nun.loc[keys] > 1).sum())
        if viol_keys == 0:
            rows.append({
                "A": a, "B": b, "n_keys": len(keys),
                "coverage": float(cnt.loc[keys].sum() / len(sub)),
            })
    return pd.DataFrame(rows)


def fd_confirm(cand: pd.DataFrame, conf: pd.DataFrame, minsup: int) -> pd.DataFrame:
    """발견 후보를 확인 자료에서 재평가한다(지지 키 중 위반 키 비율)."""
    rows = []
    for _, r in cand.iterrows():
        sub = conf[[r.A, r.B]].dropna()
        g = sub.groupby(r.A, observed=True)[r.B]
        cnt, nun = g.count(), g.nunique()
        keys = cnt[cnt >= minsup].index
        viol = int((nun.loc[keys] > 1).sum()) if len(keys) else 0
        rows.append({**r.to_dict(), "conf_keys": len(keys), "conf_viol_keys": viol})
    return pd.DataFrame(rows)


def cfd_scan(disc: pd.DataFrame, minsup: int, cols: list[str]) -> pd.DataFrame:
    """범주 조건 C=c 안에서만 성립하는 조건부 함수 종속성 후보."""
    frames = []
    for c in CATEGORICAL:
        for level in disc[c].dropna().unique():
            sub = disc[disc[c] == level]
            found = fd_scan(sub, minsup, [x for x in cols if x != c])
            if not found.empty:
                found.insert(0, "condition", f"{c}={level}")
                frames.append(found)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. 작은 산술 문법
# ---------------------------------------------------------------------------

def _expr_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df[NUMERIC].to_numpy(dtype=float)
    return x, np.isfinite(x).all(axis=1)


def enumerate_grammar(x: np.ndarray) -> list[dict]:
    """합, 차, 비율, 최소, 최대의 이항식을 열거한다. 반환: 식 메타와 값 배열."""
    exprs = []
    idx = range(len(NUMERIC))
    for i, j in itertools.combinations(idx, 2):
        a, b = NUMERIC[i], NUMERIC[j]
        ua, ub = DISPLAY_UNIT[a], DISPLAY_UNIT[b]
        exprs.append({"name": f"{a}+{b}", "cols": {i, j}, "val": x[:, i] + x[:, j],
                      "tol": 0.5 * ua + 0.5 * ub})
        exprs.append({"name": f"min({a},{b})", "cols": {i, j},
                      "val": np.minimum(x[:, i], x[:, j]), "tol": 0.5 * max(ua, ub)})
        exprs.append({"name": f"max({a},{b})", "cols": {i, j},
                      "val": np.maximum(x[:, i], x[:, j]), "tol": 0.5 * max(ua, ub)})
    for i, j in itertools.permutations(idx, 2):
        a, b = NUMERIC[i], NUMERIC[j]
        ua, ub = DISPLAY_UNIT[a], DISPLAY_UNIT[b]
        exprs.append({"name": f"{a}-{b}", "cols": {i, j}, "val": x[:, i] - x[:, j],
                      "tol": 0.5 * ua + 0.5 * ub})
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = x[:, i] / x[:, j]
            rtol = (0.5 * ua / np.abs(x[:, j])
                    + np.abs(x[:, i]) * 0.5 * ub / np.square(x[:, j]))
        exprs.append({"name": f"{a}/{b}", "cols": {i, j}, "val": ratio, "tol": rtol})
    return exprs


def grammar_metrics(
    xc: np.ndarray, uc: float, expr: dict, k: float
) -> dict[str, float] | None:
    """잔차 r = c - k*expr 의 정확 일치, 표시 일치, 부호 위반, 경계 질량."""
    r = xc - k * expr["val"]
    ok = np.isfinite(r)
    n = int(ok.sum())
    if n < MIN_SUPPORT_FRAC * len(r):
        return None
    r = r[ok]
    tol = expr["tol"]
    tol = tol[ok] if isinstance(tol, np.ndarray) else tol
    disp_tol = 0.5 * uc + abs(k) * tol
    return {
        "n": n,
        "exact": float((np.abs(r) <= MACHINE_TOL).mean()),
        "disp": float((np.abs(r) <= disp_tol).mean()),
        "neg": float((r < -disp_tol).mean()),      # c < k*expr 방향 위반
        "pos": float((r > disp_tol).mean()),        # c > k*expr 방향 위반
        "near0": float((np.abs(r) <= 2 * np.maximum(disp_tol, 0)).mean()),
        "med_abs": float(np.median(np.abs(r))),
    }


def grammar_scan(disc: pd.DataFrame) -> pd.DataFrame:
    """발견 자료(전 수치 관측 행)에서 등식·부등식 후보를 찾는다."""
    x, complete = _expr_arrays(disc)
    x = x[complete]
    exprs = enumerate_grammar(x)
    rows = []
    for ci, c in enumerate(NUMERIC):
        xc = x[:, ci]
        uc = DISPLAY_UNIT[c]
        # 단항: c ~ k*a
        for ai, a in enumerate(NUMERIC):
            if ai == ci:
                continue
            e = {"name": a, "cols": {ai}, "val": x[:, ai], "tol": 0.5 * DISPLAY_UNIT[a]}
            for k in COEFS:
                m = grammar_metrics(xc, uc, e, k)
                if m is None:
                    continue
                if m["disp"] >= EQ_SHARE or (
                    min(m["neg"], m["pos"]) <= INEQ_VIOL and m["near0"] >= INEQ_BOUNDARY
                ):
                    rows.append({"c": c, "expr": f"{k:g}*{e['name']}", **m})
        # 이항
        for e in exprs:
            if ci in e["cols"]:
                continue
            for k in COEFS:
                m = grammar_metrics(xc, uc, e, k)
                if m is None:
                    continue
                if m["disp"] >= EQ_SHARE or (
                    min(m["neg"], m["pos"]) <= INEQ_VIOL and m["near0"] >= INEQ_BOUNDARY
                ):
                    rows.append({"c": c, "expr": f"{k:g}*({e['name']})", **m})
    return pd.DataFrame(rows)


def _parse_expr(expr: str) -> tuple[float, str, list[str]]:
    """후보 문자열 'k*(a op b)' 또는 'k*a'를 (k, op, 열들)로 되돌린다."""
    k_str, body = expr.split("*", 1)
    body = body[1:-1] if body.startswith("(") else body
    for fn in ("min", "max"):
        if body.startswith(fn + "("):
            a, b = body[len(fn) + 1:-1].split(",")
            return float(k_str), fn, [a, b]
    for op in ("+", "-", "/"):
        # 열 이름에 없는 연산자 문자로만 나눈다.
        parts = [p for p in body.split(op) if p in NUMERIC]
        if op in body and len(parts) == 2:
            return float(k_str), op, parts
    return float(k_str), "id", [body]


def pairwise_recheck(surv: pd.DataFrame, sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """생존 부등식을 식에 쓰인 열만 관측된 행 전체에서 재평가한다.

    전 수치 관측 행 제한은 두 열짜리 관계의 위반을 놓칠 수 있으므로,
    비함의 생존 후보는 이 쌍별 재확인까지 통과해야 제약으로 인정한다.
    """
    ops = {"+": np.add, "-": np.subtract, "/": np.divide,
           "min": np.minimum, "max": np.maximum}
    rows = []
    for _, r in surv.iterrows():
        k, op, cols = _parse_expr(r["expr"])
        row = {"c": r["c"], "expr": r["expr"]}
        for name in ["train_conf", "test"]:
            df = sets[name][[r["c"], *cols]].dropna()
            e = (df[cols[0]].to_numpy() if op == "id"
                 else ops[op](df[cols[0]].to_numpy(), df[cols[1]].to_numpy()))
            resid = df[r["c"]].to_numpy() - k * e
            if op == "/":
                a, b = df[cols[0]].to_numpy(), df[cols[1]].to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    tol = 0.5 * DISPLAY_UNIT[r["c"]] + abs(k) * (
                        0.5 * DISPLAY_UNIT[cols[0]] / np.abs(b)
                        + np.abs(a) * 0.5 * DISPLAY_UNIT[cols[1]] / np.square(b)
                    )
                keep = np.isfinite(resid)
                resid, tol = resid[keep], tol[keep]
            elif op in ("min", "max"):
                # 발견 스캔의 max(u_a, u_b) 허용오차는 min/max가 실제로 고른 열과
                # 무관하게 굵은 쪽 표시 단위(age=1.0)를 전파해 부등식을 물로 만든다.
                # 재확인에서는 행마다 min/max가 고른 열의 표시 단위만 전파한다.
                a, b = df[cols[0]].to_numpy(), df[cols[1]].to_numpy()
                picks_a = (a <= b) if op == "min" else (a >= b)
                u_e = np.where(
                    picks_a, DISPLAY_UNIT[cols[0]], DISPLAY_UNIT[cols[1]]
                )
                tol = 0.5 * DISPLAY_UNIT[r["c"]] + abs(k) * 0.5 * u_e
            else:
                tol = 0.5 * DISPLAY_UNIT[r["c"]] + abs(k) * sum(
                    0.5 * DISPLAY_UNIT[c] for c in cols
                )
            row[f"{name}_n"] = len(df)
            row[f"{name}_viol"] = int(
                min((resid < -tol).sum(), (resid > tol).sum())
            )
        rows.append(row)
    return pd.DataFrame(rows)


def budget_residual(df: pd.DataFrame) -> pd.Series:
    """사전 등록 시간 예산식: daily - (social + gaming + work)."""
    return (
        df["daily_screen_time_hours"]
        - df[["social_media_hours", "gaming_hours", "work_study_hours"]].sum(
            axis=1, skipna=False
        )
    )


def budget_table(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    disp_tol = 0.5 * 0.01 * 4  # 네 열의 표시 반올림 오차 전파
    rows = []
    for name in ["proxy_disc", "proxy_conf", "train_disc", "train_conf", "test"]:
        r = budget_residual(sets[name]).dropna()
        rows.append({
            "set": name, "n": len(r),
            "viol_share": float((r < -disp_tol).mean()),
            "exact0": float((r.abs() <= MACHINE_TOL).mean()),
            "disp0": float((r.abs() <= disp_tol).mean()),
            "q00": float(r.min()), "q01": float(r.quantile(0.01)),
            "q50": float(r.quantile(0.5)), "q99": float(r.quantile(0.99)),
            "max": float(r.max()),
        })
    return pd.DataFrame(rows)


def budget_grid_mass(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """경계(잔차 0) 근방 0.01 격자점별 질량. 자르기/투영이면 0에 질량이 쌓인다."""
    rows = []
    for name in ["train_conf", "test"]:
        r = np.round(budget_residual(sets[name]).dropna(), 2)
        rows.append({
            "set": name,
            **{f"{k / 100:.2f}": float((r == round(k * 0.01, 2)).mean())
               for k in range(11)},
        })
    return pd.DataFrame(rows)


def eval_candidates(
    cand: pd.DataFrame, sets: dict[str, pd.DataFrame], disc_name: str
) -> pd.DataFrame:
    """발견 후보 식을 확인·복제 자료에서 같은 허용오차로 재평가한다.

    box_implied: 발견 자료의 단변량 [min,max] 상자만으로 이미 함의되는 부등식.
    두 열의 공동 구조가 아니라 각 열의 범위가 만든 자명한 관계라는 뜻이다.
    """
    if cand.empty:
        return cand
    evals = {}
    for name in ["proxy_conf", "train_conf", "test"]:
        x, complete = _expr_arrays(sets[name])
        x = x[complete]
        exprs = {e["name"]: e for e in enumerate_grammar(x)}
        for a_i, a in enumerate(NUMERIC):
            exprs[a] = {"name": a, "cols": {a_i}, "val": x[:, a_i],
                        "tol": 0.5 * DISPLAY_UNIT[a]}
        evals[name] = (x, exprs)
    x_disc, complete_disc = _expr_arrays(sets[disc_name])
    x_disc = x_disc[complete_disc]
    disc_exprs = {e["name"]: e for e in enumerate_grammar(x_disc)}
    for a_i, a in enumerate(NUMERIC):
        disc_exprs[a] = {"name": a, "val": x_disc[:, a_i]}
    out = []
    for _, r in cand.iterrows():
        k_str, e_name = r["expr"].split("*", 1)
        k = float(k_str)
        e_name = e_name[1:-1] if e_name.startswith("(") else e_name
        ci = NUMERIC.index(r["c"])
        row = r.to_dict()
        ev = k * disc_exprs[e_name]["val"]
        ev = ev[np.isfinite(ev)]
        c_lo, c_hi = float(x_disc[:, ci].min()), float(x_disc[:, ci].max())
        margin = 2 * DISPLAY_UNIT[r["c"]]
        if r["neg"] <= r["pos"]:  # c >= k*expr 방향
            row["box_implied"] = bool(float(ev.max()) <= c_lo + margin)
        else:                      # c <= k*expr 방향
            row["box_implied"] = bool(c_hi <= float(ev.min()) + margin)
        for name, (x, exprs) in evals.items():
            m = grammar_metrics(x[:, ci], DISPLAY_UNIT[r["c"]], exprs[e_name], k)
            for key in ["disp", "neg", "pos"]:
                row[f"{name}_{key}"] = m[key] if m else np.nan
        row["conf_exact"] = bool(
            min(row["train_conf_neg"], row["train_conf_pos"]) == 0
            and min(row["test_neg"], row["test_pos"]) == 0
        )
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 4. 조건부 범위
# ---------------------------------------------------------------------------

def _conditions(disc: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    conds = []
    for c in CATEGORICAL:
        for level in sorted(disc[c].dropna().unique()):
            conds.append((f"{c}={level}", lambda df, c=c, level=level: df[c] == level))
    age_edges = disc["age"].quantile([0.25, 0.5, 0.75]).to_numpy()
    lo = -np.inf
    for i, hi in enumerate([*age_edges, np.inf]):
        conds.append((
            f"age_q{i}({lo:g},{hi:g}]",
            lambda df, lo=lo, hi=hi: (df["age"] > lo) & (df["age"] <= hi),
        ))
        lo = hi
    return conds


def conditional_ranges(disc: pd.DataFrame, sets: dict[str, pd.DataFrame],
                       eval_names: list[str]) -> pd.DataFrame:
    """발견 자료 조건별 [min,max]가 전역보다 유의미하게 좁은 경우만 확인 자료로 평가."""
    rows = []
    for cond_name, cond_fn in _conditions(disc):
        sub = disc[cond_fn(disc)]
        for col in NUMERIC:
            if cond_name.startswith("age_q") and col == "age":
                continue
            v = sub[col].dropna()
            if len(v) < 100:
                continue
            g = disc[col].dropna()
            u = DISPLAY_UNIT[col]
            lo, hi = float(v.min()), float(v.max())
            tighter_lo = lo > float(g.min()) + 2 * u
            tighter_hi = hi < float(g.max()) - 2 * u
            if not (tighter_lo or tighter_hi):
                continue
            row = {"condition": cond_name, "column": col, "disc_n": len(v),
                   "lo": lo, "hi": hi, "tight": ("lo" if tighter_lo else "")
                   + ("hi" if tighter_hi else "")}
            for name in eval_names:
                df = sets[name]
                ev = df.loc[cond_fn(df), col].dropna()
                row[f"{name}_out"] = float(
                    ((ev < lo - MACHINE_TOL) | (ev > hi + MACHINE_TOL)).mean()
                ) if len(ev) else np.nan
                row[f"{name}_n"] = len(ev)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. 결측 마스크
# ---------------------------------------------------------------------------

def poisson_binomial_pmf(p: np.ndarray) -> np.ndarray:
    pmf = np.array([1.0])
    for pi in p:
        pmf = np.convolve(pmf, [1 - pi, pi])
    return pmf


def mask_marginals(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name in ["train", "test"]:
        m = sets[name][SHARED].isna()
        for col in SHARED:
            rows.append({"set": name, "column": col, "rate": float(m[col].mean())})
    return pd.DataFrame(rows).pivot(index="column", columns="set", values="rate")


def mask_dependence(df: pd.DataFrame, name: str, with_target: bool) -> dict:
    m = df[SHARED].isna().to_numpy()
    n = len(df)
    # 쌍별 파이 상관
    corr = np.corrcoef(m.T)
    iu = np.triu_indices(len(SHARED), k=1)
    phi = corr[iu]
    z = phi * np.sqrt(n)
    top = np.argsort(-np.abs(phi))[:3]
    pairs = [
        f"{SHARED[iu[0][t]]}~{SHARED[iu[1][t]]}: phi={phi[t]:+.4f} (z={z[t]:+.1f})"
        for t in top
    ]
    # 행별 결측 개수 대 Poisson-binomial
    counts = m.sum(axis=1)
    obs = np.bincount(counts, minlength=len(SHARED) + 1) / n
    pmf = poisson_binomial_pmf(m.mean(axis=0))
    tv = 0.5 * float(np.abs(obs - pmf).sum())
    var_ratio = float(counts.var() / (m.mean(axis=0) * (1 - m.mean(axis=0))).sum())
    out = {
        "set": name, "n": n, "max_abs_phi": float(np.abs(phi).max()),
        "mean_phi": float(phi.mean()), "share_phi_pos": float((phi > 0).mean()),
        "top_phi": pairs, "rowcount_tv": tv, "var_ratio": var_ratio,
        "rowcount_obs": obs[:9].round(5).tolist(),
        "rowcount_pb": pmf[:9].round(5).tolist(),
    }
    # 목표값과의 의존
    if with_target:
        y = df[TARGET].to_numpy()
        diffs = {}
        for j, col in enumerate(SHARED):
            r1, r0 = y[m[:, j]].mean(), y[~m[:, j]].mean()
            se = np.sqrt(
                r1 * (1 - r1) / m[:, j].sum() + r0 * (1 - r0) / (~m[:, j]).sum()
            )
            diffs[col] = (float(r1 - r0), float((r1 - r0) / se))
        out["target_rate_diff"] = diffs
    return out


def mask_mean_shift(df: pd.DataFrame) -> pd.DataFrame:
    """M_j별 다른 수치 열 관측값의 표준화 평균 차이(최대 |d| 상위)."""
    m = df[SHARED].isna()
    rows = []
    for j in SHARED:
        for k in NUMERIC:
            if j == k:
                continue
            v = df[k]
            v1, v0 = v[m[j]], v[~m[j]]
            if v1.notna().sum() < 100:
                continue
            sd = v.std()
            d = float((v1.mean() - v0.mean()) / sd)
            rows.append({"mask": j, "column": k, "cohen_d": d})
    out = pd.DataFrame(rows)
    return out.reindex(out.cohen_d.abs().sort_values(ascending=False).index)


def mask_age_dependence(df: pd.DataFrame) -> tuple[list[float], float]:
    """모든 M_j에 대한 age의 표준화 평균 차이와 corr(age, 행별 결측 수)."""
    m = df[SHARED].isna()
    v = df["age"]
    ds = [
        float((v[m[j]].mean() - v[~m[j]].mean()) / v.std())
        for j in SHARED if j != "age"
    ]
    cnt = m.sum(axis=1)
    ok = v.notna()
    corr = float(np.corrcoef(v[ok], cnt[ok])[0, 1])
    return ds, corr


def high_decimal_values(sets: dict[str, pd.DataFrame]) -> None:
    """프록시 격자(소수 2자리)를 벗어나는 3자리 이상 소수 값의 열거."""
    for name in ["train", "test"]:
        for col in NUMERIC:
            v = sets[name][col].dropna()
            s = v * 100
            odd = v[~np.isclose(s, np.round(s), atol=1e-6)]
            if len(odd):
                print(f"   {name}.{col}: {len(odd)}개 {sorted(odd.tolist())[:8]}")


def mask_only_predictability(disc: pd.DataFrame, conf: pd.DataFrame) -> pd.DataFrame:
    """다른 열의 결측 마스크만으로 M_j를 예측하는 AUC. 값·목표값 신호를 배제한다."""
    rows = []
    for j in SHARED:
        cols = [c for c in SHARED if c != j]
        clf = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.1, num_leaves=31,
            min_child_samples=100, random_state=SEED, n_jobs=-1, verbose=-1,
        )
        clf.fit(disc[cols].isna().astype(int), disc[j].isna())
        auc = roc_auc_score(
            conf[j].isna(), clf.predict_proba(conf[cols].isna().astype(int))[:, 1]
        )
        rows.append({"mask": j, "auc_mask_only": float(auc)})
    return pd.DataFrame(rows)


def mask_predictability(
    disc: pd.DataFrame, conf: pd.DataFrame, with_target: bool, n_perm: int = 1
) -> pd.DataFrame:
    """발견에서 학습해 확인에서 채점하는 M_j 예측 AUC와 순열 대조."""
    feats = [c for c in SHARED]
    rows = []
    rng = np.random.default_rng(SEED)
    for j in SHARED:
        cols = [c for c in feats if c != j]
        parts = []
        for df in (disc, conf):
            xx = df[cols].copy()
            for c in CATEGORICAL:
                if c in xx:
                    xx[c] = xx[c].astype("category")
            if with_target:
                xx[TARGET] = df[TARGET]
            parts.append(xx)
        x_disc, x_conf = parts
        y_disc = disc[j].isna().to_numpy()
        y_conf = conf[j].isna().to_numpy()
        clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.08, num_leaves=63,
            min_child_samples=100, random_state=SEED, n_jobs=-1, verbose=-1,
        )
        clf.fit(x_disc, y_disc)
        auc = roc_auc_score(y_conf, clf.predict_proba(x_conf)[:, 1])
        null_aucs = []
        for _ in range(n_perm):
            clf_p = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.08, num_leaves=63,
                min_child_samples=100, random_state=SEED, n_jobs=-1, verbose=-1,
            )
            clf_p.fit(x_disc, rng.permutation(y_disc))
            null_aucs.append(
                roc_auc_score(y_conf, clf_p.predict_proba(x_conf)[:, 1])
            )
        n1, n0 = int(y_conf.sum()), int((~y_conf).sum())
        se_null = np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
        rows.append({
            "mask": j, "auc": float(auc), "null_auc": float(np.mean(null_aucs)),
            "null_se_analytic": float(se_null),
            "z_vs_null": float((auc - 0.5) / se_null),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def main() -> None:
    pd.set_option("display.width", 240)
    sets = load_datasets()

    print("=== 1. 스키마, 값 눈금, 경계 질량 ===")
    print(schema_table(sets).to_string(index=False))
    print("\n-- 프록시 발견 경계 기준의 계보(below/above=경계 밖 비율, at=경계 질량) --")
    print(boundary_lineage(sets).to_string(index=False))

    print("\n=== 2. 함수 종속성(A -> B, 발견 자료 정확 성립 후보) ===")
    cols13 = SHARED + [TARGET]
    for src, minsup in [("proxy", FD_MINSUP_PROXY), ("train", FD_MINSUP_TRAIN)]:
        cand = fd_scan(sets[f"{src}_disc"], minsup, cols13)
        print(f"-- {src} 발견: 후보 {len(cand)}개 --")
        if not cand.empty:
            print(fd_confirm(cand, sets[f"{src}_conf"], minsup).to_string(index=False))
    print("-- 조건부 함수 종속성(범주 조건, 발견 자료) --")
    for src, minsup in [("proxy", FD_MINSUP_PROXY), ("train", FD_MINSUP_TRAIN)]:
        cfd = cfd_scan(sets[f"{src}_disc"], minsup, cols13)
        print(f"{src}: 후보 {len(cfd)}개")
        if not cfd.empty:
            print(cfd.to_string(index=False))
    # 프록시 전용 열의 참고 확인: addiction_level -> addicted_label
    lvl = sets["proxy_disc"].groupby("addiction_level")[TARGET].agg(["nunique", "count"])
    print("-- 참고: 프록시 전용 addiction_level -> addicted_label --")
    print(lvl.to_string())

    print("\n=== 3. 작은 산술 문법 ===")
    print("-- 사전 등록 시간 예산식 daily - (social+gaming+work) --")
    print(budget_table(sets).to_string(index=False))
    print("-- 예산 잔차의 경계 근방 격자 질량(자르기/투영이면 0.00에 쌓임) --")
    print(budget_grid_mass(sets).round(5).to_string(index=False))
    for src in ["proxy", "train"]:
        cand = grammar_scan(sets[f"{src}_disc"])
        print(f"-- {src} 발견 후보 {len(cand)}개 (등식 disp>={EQ_SHARE}, "
              f"부등식 viol<={INEQ_VIOL} & 경계질량>={INEQ_BOUNDARY}) --")
        if not cand.empty:
            ev = eval_candidates(cand, sets, f"{src}_disc")
            n_eq = int((ev["disp"] >= EQ_SHARE).sum())
            n_box = int(ev["box_implied"].sum())
            survivors = ev[~ev["box_implied"] & ev["conf_exact"]]
            print(f"   등식 후보 {n_eq}개, 단변량 상자 함의 {n_box}개, "
                  f"비함의·확인 정확 생존 {len(survivors)}개")
            if not survivors.empty:
                print("   생존 후보의 쌍별 관측 행 재확인(전 수치 관측 제한 해제):")
                print(pairwise_recheck(survivors, sets).to_string(index=False))
            print("   전체 후보(위반률 요약):")
            print(ev.round(5).to_string(index=False))

    print("\n=== 4. 조건부 범위(전역보다 좁은 발견 조건 범위의 확인) ===")
    for src in ["proxy", "train"]:
        eval_names = (["proxy_conf", "train_conf", "test"] if src == "proxy"
                      else ["train_conf", "test", "proxy_conf"])
        cr = conditional_ranges(sets[f"{src}_disc"], sets, eval_names)
        print(f"-- {src} 발견: {len(cr)}개 --")
        if not cr.empty:
            print(cr.round(5).to_string(index=False))

    print("\n=== 5. 결측 마스크 ===")
    print("-- 열별 결측률 --")
    print(mask_marginals(sets).round(4).to_string())
    for name, with_t in [("train", True), ("test", False)]:
        dep = mask_dependence(sets[name], name, with_t)
        print(f"-- {name}: max|phi|={dep['max_abs_phi']:.4f}, "
              f"mean phi={dep['mean_phi']:.4f}, 양의 쌍 비율={dep['share_phi_pos']:.2f}, "
              f"행별 개수 TV={dep['rowcount_tv']:.5f}, 분산비={dep['var_ratio']:.4f}")
        for p in dep["top_phi"]:
            print(f"   {p}")
        print(f"   행별 결측 수 분포 관측: {dep['rowcount_obs']}")
        print(f"   Poisson-binomial 기대: {dep['rowcount_pb']}")
        if with_t:
            print("   목표값 비율 차이(결측-관측, z):")
            for col, (d, z) in dep["target_rate_diff"].items():
                print(f"     {col}: {d:+.4f} (z={z:+.1f})")
    print("-- M_j별 다른 열 관측값의 표준화 평균 차이 상위 10 (train) --")
    print(mask_mean_shift(sets["train"]).head(10).round(4).to_string(index=False))
    for name in ["train", "test"]:
        ds, corr = mask_age_dependence(sets[name])
        print(f"-- {name}: age의 마스크별 표준화 차이 범위 "
              f"[{min(ds):+.4f}, {max(ds):+.4f}], corr(age, 결측 수)={corr:+.4f}")
    print("-- 프록시 격자 밖 3자리 이상 소수 값 --")
    high_decimal_values(sets)
    print("-- 교차적합 M_j 예측 AUC (train: 발견 학습 -> 확인 채점, Y 포함) --")
    print(mask_predictability(sets["train_disc"], sets["train_conf"], True)
          .round(4).to_string(index=False))
    print("-- 교차적합 M_j 예측 AUC (train, Y 제외) --")
    print(mask_predictability(sets["train_disc"], sets["train_conf"], False)
          .round(4).to_string(index=False))
    t_disc, t_conf = train_test_split(sets["test"], test_size=0.5, random_state=SEED)
    print("-- 교차적합 M_j 예측 AUC (test, Y 없음) --")
    print(mask_predictability(t_disc, t_conf, False).round(4).to_string(index=False))
    print("-- 마스크만으로 M_j 예측 AUC (값·목표값 배제) --")
    both = mask_only_predictability(sets["train_disc"], sets["train_conf"]).merge(
        mask_only_predictability(t_disc, t_conf), on="mask",
        suffixes=("_train", "_test"),
    )
    print(both.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
