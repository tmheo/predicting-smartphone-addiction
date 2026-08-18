# PROTOTYPE (issue #212) - throwaway storyboard, not the final notebook.
# 3시드 OOF/Public 분석 노트북의 이야기 흐름과 그림 구성을 실제 mlflow.db 자료로
# 미리 그려 보는 1회용 스크립트다. 검토가 끝나면 폐기 브랜치에만 보존한다.
# 실행: uv run python notebooks/prototype_issue212_storyboard.py
import base64
import io
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "mlflow.db"
OUT_PATH = Path(__file__).resolve().parent / "prototype_issue212_storyboard.html"

if not DB_PATH.exists():
    sys.exit(f"ABORT: mlflow db not found at {DB_PATH} (never create a new one)")

con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
runs = pd.read_sql("select run_uuid, status, start_time from runs where experiment_id=1", con)
params = pd.read_sql("select run_uuid, key, value from params", con)
metrics = pd.read_sql("select run_uuid, key, value from latest_metrics", con)
tags = pd.read_sql("select run_uuid, key, value from tags", con)
con.close()

pw = params.pivot(index="run_uuid", columns="key", values="value")
mw = metrics.pivot(index="run_uuid", columns="key", values="value")
tw = tags.pivot(index="run_uuid", columns="key", values="value")
df = runs.set_index("run_uuid").join(pw, how="left", rsuffix="_p").join(mw, rsuffix="_m").join(tw, rsuffix="_t")
df["start"] = df["start_time"].map(lambda ms: datetime.fromtimestamp(ms / 1000))
df["dirty"] = df.get("git_dirty", pd.Series(index=df.index)).eq("True")
ens_keys = [k for k in pw.columns if str(k).startswith("ensemble.")]
df["is_ensemble"] = df[ens_keys].notna().any(axis=1) if ens_keys else False

# --- 모집단 재도출 (#209 판별 규칙) ---
p0 = df
p1 = df[(df["status"] == "FINISHED") & (df["seeds"] == "42,43,44")].copy()
seed_cols = ["auc_oof_seed_42", "auc_oof_seed_43", "auc_oof_seed_44"]
p1["strict"] = p1[seed_cols].notna().all(axis=1)
p2 = p1[p1["strict"]]
p1["config"] = p1["experiment"].replace({"exp033_lattice_te": "exp035_lattice_te"})
reps = []
for cfg, g in p1.groupby("config"):
    clean = g[~g["dirty"]]
    pick = (clean if len(clean) else g).sort_values("start").iloc[-1]
    reps.append(pick.name)
p3 = p1.loc[reps]
trans = p1[p1["public_auc"].notna()].copy()
ensemble = df[df["is_ensemble"] & (df["status"] == "FINISHED")]

counts = dict(P0=len(p0), P1=len(p1), P2=len(p2), P3=len(p3), TRANS=len(trans), ENS=len(ensemble))
expected = dict(P0=153, P1=42, P2=39, P3=32, TRANS=10, ENS=7)
assert counts == expected, f"population mismatch: {counts} != {expected}"

p1["short"] = p1["experiment"].str.extract(r"^(exp\d+)")[0].fillna(p1["experiment"])
for c in ["auc_oof", "public_auc"] + seed_cols:
    p1[c] = p1[c].astype(float)
p3 = p1.loc[p3.index]
trans = p1.loc[trans.index]


def b64fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


figs = {}

# F1-A: timeline with cumulative best (all 42, reps emphasized)
t = p1.sort_values("start")
cummax = t["auc_oof"].cummax()
fig, ax = plt.subplots(figsize=(9.5, 4.2))
dup = ~t.index.isin(p3.index)
ax.scatter(t.loc[dup, "start"], t.loc[dup, "auc_oof"], s=18, c="#bbb", label="non-representative rerun")
ax.scatter(t.loc[~dup, "start"], t.loc[~dup, "auc_oof"], s=28, c="#1f77b4", label="representative run")
ax.step(t["start"], cummax, where="post", c="#d62728", lw=1.5, label="running best (story device)")
new_best = t[t["auc_oof"] >= cummax - 1e-12]
seen = set()
for _, r in new_best.iterrows():
    if r["short"] in seen:
        continue
    seen.add(r["short"])
    ax.annotate(r["short"], (r["start"], r["auc_oof"]), textcoords="offset points",
                xytext=(0, 7), fontsize=7, ha="center", color="#d62728")
ax.set_ylim(0.9595, 0.9698)
ax.set_xlabel("run start time")
ax.set_ylabel("seed-averaged OOF AUC")
ax.set_title("A. Chronological view: all 42 runs + running best line")
ax.legend(fontsize=7, loc="lower right")
figs["f1a"] = b64fig(fig)

# F1-B: ranked dot plot of representatives (32)
r = p3.sort_values("auc_oof")
fig, ax = plt.subplots(figsize=(7.5, 7.2))
colors = ["#d62728" if d else ("#ff7f0e" if not s else "#1f77b4") for d, s in zip(r["dirty"], r["strict"])]
ax.scatter(r["auc_oof"], range(len(r)), c=colors, s=30)
ax.set_yticks(range(len(r)))
ax.set_yticklabels(r["experiment"], fontsize=6.5)
ax.set_xlabel("seed-averaged OOF AUC")
ax.set_title("B. Ranked view: 32 representative configs\n(blue=strict, orange=legacy metrics, red=dirty-only exception)")
ax.grid(axis="x", lw=0.3, alpha=0.5)
figs["f1b"] = b64fig(fig)

# F2: seed stability - seed range strip + averaging gain strip
s = p2.copy()
for c in seed_cols:
    s[c] = s[c].astype(float)
s["auc_oof"] = s["auc_oof"].astype(float)
s["seed_range"] = s[seed_cols].max(axis=1) - s[seed_cols].min(axis=1)
s["avg_gain"] = s["auc_oof"] - s[seed_cols].mean(axis=1)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
axes[0].scatter(s["seed_range"], [0.5] * len(s), s=25, alpha=0.6, c="#1f77b4")
axes[0].axvline(0.00002, c="#2ca02c", ls="--", lw=1, label="tie threshold 0.00002")
axes[0].axvline(0.0002, c="#d62728", ls="--", lw=1, label="material threshold 0.0002")
axes[0].set_xscale("log")
axes[0].set_yticks([])
axes[0].set_xlabel("seed range = max-min of per-seed OOF AUC (log scale)")
axes[0].set_title("Seed range of 39 strict runs vs ADR 0001 thresholds")
axes[0].legend(fontsize=7)
axes[1].scatter(s["avg_gain"], [0.5] * len(s), s=25, alpha=0.6, c="#1f77b4")
axes[1].axvline(0.0003, c="#7f7f7f", ls=":", lw=1, label="assumed +0.0003 (ADR #74)")
p1s = p1.loc[s.index]
for _, row in s.assign(short=p1s["short"], exp=p1s["experiment"]).iterrows():
    if row["avg_gain"] > 0.0008 or row["avg_gain"] < 0.00005:
        axes[1].annotate(row["short"], (row["avg_gain"], 0.5), textcoords="offset points",
                         xytext=(0, 8), fontsize=6.5, ha="center", rotation=45)
axes[1].set_yticks([])
axes[1].set_xlabel("averaging gain = blended OOF - mean of per-seed OOF")
axes[1].set_title("Averaging gain: descriptive, outliers labeled")
axes[1].legend(fontsize=7)
figs["f2"] = b64fig(fig)

# F3-A: transition scatter
tr = trans.sort_values("auc_oof")
gap = tr["public_auc"] - tr["auc_oof"]
fig, ax = plt.subplots(figsize=(5.4, 5.0))
ax.scatter(tr["auc_oof"], tr["public_auc"], s=35, c="#1f77b4")
lims = [0.9664, 0.9705]
ax.plot(lims, [v + gap.median() for v in lims], c="#7f7f7f", ls="--", lw=1,
        label=f"y = x + median gap ({gap.median():+.5f})")
for _, row in tr.iterrows():
    ax.annotate(row["short"], (row["auc_oof"], row["public_auc"]), textcoords="offset points",
                xytext=(5, -3), fontsize=7)
ax.set_xlabel("seed-averaged OOF AUC")
ax.set_ylabel("Public AUC")
ax.set_title("A. Scatter: 10 submitted runs, 0 rank inversions")
ax.legend(fontsize=7)
figs["f3a"] = b64fig(fig)

# F3-B: slopegraph
fig, ax = plt.subplots(figsize=(5.4, 5.0))
for _, row in tr.iterrows():
    ax.plot([0, 1], [row["auc_oof"], row["public_auc"]], c="#1f77b4", lw=1, marker="o", ms=3)
    ax.annotate(row["short"], (0, row["auc_oof"]), textcoords="offset points", xytext=(-8, -2),
                fontsize=7, ha="right")
ax.set_xticks([0, 1])
ax.set_xticklabels(["OOF AUC", "Public AUC"])
ax.set_title("B. Slopegraph: parallel lines = order preserved")
ax.set_xlim(-0.35, 1.15)
figs["f3b"] = b64fig(fig)

# F4: selection bias strips
fig, ax = plt.subplots(figsize=(8.5, 3.0))
sub = p1.index.isin(trans.index)
ax.scatter(p1.loc[~sub, "auc_oof"], [0] * (~sub).sum(), s=25, alpha=0.6, c="#bbb", label="not submitted (32)")
ax.scatter(p1.loc[sub, "auc_oof"], [1] * sub.sum(), s=30, c="#1f77b4", label="submitted (10)")
ax.set_yticks([0, 1])
ax.set_yticklabels(["no Public", "has Public"])
ax.set_xlabel("seed-averaged OOF AUC (42 runs)")
ax.set_title("Selection: submitted runs cluster at the top of the OOF distribution")
ax.legend(fontsize=7, loc="center left")
figs["f4"] = b64fig(fig)

# F5: ensemble section dot plot
e = ensemble.copy()
e["auc_oof"] = e["auc_oof"].astype(float)
e["public_auc"] = e["public_auc"].astype(float)
fig, ax = plt.subplots(figsize=(8.5, 3.2))
eo = e.dropna(subset=["auc_oof"]).sort_values("auc_oof")
labels = [t[:38] for t in (tw.loc[eo.index, "mlflow.runName"] if "mlflow.runName" in tw else eo.index)]
ax.scatter(eo["auc_oof"], range(len(eo)), c="#9467bd", s=35, label="ensemble nested OOF")
best_single = p3["auc_oof"].max()
ax.axvline(best_single, c="#1f77b4", ls="--", lw=1, label=f"best single model OOF ({best_single:.5f})")
ax.set_yticks(range(len(eo)))
ax.set_yticklabels(labels, fontsize=6.5)
ax.set_xlabel("nested OOF AUC")
ax.set_title("Ensemble section: 6 runs with OOF (full-refit run has none)")
ax.legend(fontsize=7)
figs["f5"] = b64fig(fig)

# transition stats
inv = sum(
    1
    for i in range(len(tr))
    for j in range(i + 1, len(tr))
    if (tr["auc_oof"].iloc[i] - tr["auc_oof"].iloc[j]) * (tr["public_auc"].iloc[i] - tr["public_auc"].iloc[j]) < 0
)
tau = tr[["auc_oof", "public_auc"]].corr(method="kendall").iloc[0, 1]

css = """
body{font-family:'Apple SD Gothic Neo',sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}
h1{font-size:1.4em} h2{font-size:1.15em;border-bottom:2px solid #1f77b4;padding-top:14px;padding-bottom:4px}
img{max-width:100%} .note{background:#eef5fc;border-left:4px solid #1f77b4;padding:8px 12px;margin:8px 0;font-size:.92em}
.warn{background:#fdf0ef;border-left:4px solid #d62728;padding:8px 12px;margin:8px 0;font-size:.92em}
.variant{border:1px dashed #ff7f0e;padding:10px;margin:10px 0;background:#fffaf3}
.variant h4{margin:2px 0;color:#c65b00}
table{border-collapse:collapse;font-size:.88em;margin:8px 0} td,th{border:1px solid #ccc;padding:3px 8px}
.q{display:inline-block;background:#ff7f0e;color:#fff;border-radius:10px;padding:1px 9px;font-size:.8em;margin-right:6px}
.banner{background:#fff3cd;border:2px solid #c65b00;padding:8px 12px;font-weight:bold}
"""

funnel = "".join(
    f"<tr><td>{k}</td><td>{d}</td><td>{v}</td></tr>"
    for k, d, v in [
        ("P0 원자료", "실험의 모든 실행", counts["P0"]),
        ("P1 전체 OOF", "완료 + seeds=42,43,44", counts["P1"]),
        ("P2 엄격 시드", "P1 중 시드별 metric 3종 보유", counts["P2"]),
        ("P3 대표", "동일 구성마다 대표 1건", counts["P3"]),
        ("전이 분석", "P1 중 public_auc 보유", counts["TRANS"]),
        ("앙상블 구획", "ensemble.* param 보유 완료 실행", counts["ENS"]),
    ]
)

rec_rows = "".join(
    f"<tr><td>{r['experiment']}</td><td>{r['auc_oof']:.5f}</td><td>{'-' if not r['strict'] else '(시드 범위)'}</td>"
    f"<td>{'전이 일치' if r.name in set(trans.index) else '미제출'}</td><td>...</td><td><b>→ #213 규칙으로 채움</b></td></tr>"
    for _, r in p3.sort_values("auc_oof", ascending=False).head(5).iterrows()
)

html = f"""<!doctype html><meta charset="utf-8"><title>PROTOTYPE issue212 notebook storyboard</title>
<style>{css}</style>
<p class="banner">PROTOTYPE (이슈 #212) - 노트북 이야기 흐름·그림 구성 검토용 1회용 초안. 실제 mlflow.db({counts['P0']}건)에서 그렸다. 최종 노트북이 아니다.</p>
<h1>3시드 OOF·Public 회고 노트북 - 이야기 흐름 초안</h1>
<p>아래 §0~§8이 제안하는 노트북 셀 순서다. 각 구획의 그림·표는 실제 자료로 렌더링했다.
주황 점선 상자는 <b>변형 A/B 중 하나를 골라야 하는 지점</b>이다.</p>

<h2>§0 머리말과 재현성 계약</h2>
<div class="note">DB 경로를 명시적으로 입력받고, 파일이 없으면 즉시 중단(생성 금지).
판별 규칙으로 모집단을 재도출한 뒤 감사 문서의 기준 건수와 대조해 표로 출력한다. 이 프로토타입도 같은 검증을 통과했다:
P0={counts['P0']}, P1={counts['P1']}, P2={counts['P2']}, P3={counts['P3']}, 전이={counts['TRANS']}, 앙상블={counts['ENS']} (모두 일치).</div>

<h2>§1 모집단 감사 요약</h2>
<table><tr><th>모집단</th><th>정의</th><th>건수</th></tr>{funnel}</table>
<div class="note">제외·표시 사유 코드 7종(NOT_FINISHED, SINGLE_SEED, LEGACY_FORMAT, DUPLICATE, DIRTY, ENSEMBLE_DERIVED, NO_OOF)의 건수 표를 함께 싣는다. 상세는 감사 문서 링크로 대신한다.</div>

<h2>§2 전체 OOF 회고 (단일 모델)</h2>
<div class="variant"><span class="q">선택 1</span><h4>주 그림을 무엇으로 여나</h4>
<p><b>변형 A - 시간순.</b> 42건 전부와 시간순 최고선(이야기 재료). 실험 프로그램의 진행 서사가 먼저 보인다.</p>
<img src="data:image/png;base64,{figs['f1a']}">
<p><b>변형 B - 순위순.</b> 대표 32건의 순위 점 그림. 현재 지형(어떤 구성이 어디쯤인가)이 먼저 보인다.</p>
<img src="data:image/png;base64,{figs['f1b']}">
<p>권고: <b>A를 §2 주 그림으로, B를 바로 뒤 보조 그림으로 둘 다 싣는다</b>(서사 → 지형 순).</p></div>

<h2>§3 시드 안정성 (엄격 39건)</h2>
<img src="data:image/png;base64,{figs['f2']}">
<div class="note">왼쪽: 시드 범위 분포로 ADR 눈금(0.00002/0.0002)의 타당성을 실측 검증(#210 결정). 오른쪽: 평균화 이득은 기술 통계로만, 이상치(exp106, logreg 계열)는 이름표. LEGACY_FORMAT 3건은 표에서 표시 처리.</div>

<h2>§4 OOF와 Public의 전이 분석 (10건)</h2>
<div class="warn"><b>고정 경고문(#211 결정, 구획 앞머리 필수):</b> 이 표본은 유망 실행만 선택적으로 제출된 10건이며, 모든 결론은 이 표본 안의 관찰 서술이다. 미제출 실행이나 미래 실행으로 일반화하지 않는다.</div>
<p>요약 통계: 역전 쌍 {inv}/45, Kendall τ={tau:.2f} (p값 없음), 격차 중앙값 {gap.median():+.5f}, 범위 {gap.min():+.5f}~{gap.max():+.5f}.</p>
<div class="variant"><span class="q">선택 2</span><h4>전이 그림의 형태</h4>
<table><tr><td><b>변형 A - 산점도</b><br><img src="data:image/png;base64,{figs['f3a']}"></td>
<td><b>변형 B - 기울기 그림</b><br><img src="data:image/png;base64,{figs['f3b']}"></td></tr></table>
<p>권고: <b>A 산점도 하나만.</b> B는 순위 보존이 직관적이지만 축 두 개의 눈금이 달라 격차 크기를 오독하기 쉽다.</p></div>
<p>선택 편향은 같은 구획 안에서 바로 이어 보여준다(제출 10건이 OOF 상위에 쏠림 + 분위수 요약 한 줄):</p>
<img src="data:image/png;base64,{figs['f4']}">

<h2>§5 앙상블 구획 (7건)</h2>
<img src="data:image/png;base64,{figs['f5']}">
<div class="note">nested OOF 축으로만 비교, 시드 안정성 분석은 적용 불가 명시(#210). full refit 제출(c2171fa9)은 Public 존재 확인 한 줄. 단일 모델 최고선과의 대비는 참고선 하나로만.</div>

<h2>§6 민감도 분석 (P3 32 vs P1 42)</h2>
<div class="note">새 그림 없이 <b>결론 차이만</b> 표로: 순위 역전 여부, 시간순 최고선 모양 변화, 동률·경계 판정 변화. 차이가 없으면 "대표 선정이 결론을 바꾸지 않음" 한 줄(#210 결정).</div>

<h2>§7 다음 실험 권고 표 (형태만 - 규칙은 #213)</h2>
<table><tr><th>구성</th><th>OOF</th><th>시드 안정성</th><th>전이 표시</th><th>근거 요약</th><th>권고</th></tr>{rec_rows}</table>
<div class="note">이 티켓에서는 <b>열 구성과 위치(맨 끝, 모든 증거 뒤)</b>만 확정한다. 권고 값을 채우는 규칙은 #213의 결정 사항.</div>

<h2>§8 재현성 부록</h2>
<div class="note">사용한 판별 규칙 요약, 모집단 건수 검증 결과 재출력, 실행 환경(패키지 버전), 감사 문서·지도 이슈 링크.</div>

<h2>남은 검토 질문</h2>
<ol>
<li><span class="q">선택 1</span> §2 주 그림: A 시간순 먼저 + B 보조로 둘 다? 하나만?</li>
<li><span class="q">선택 2</span> §4 전이 그림: A 산점도만? B 기울기 그림만? 둘 다?</li>
<li><span class="q">선택 3</span> 선택 편향 그림을 §4 안에 두는 것(현안) vs 별도 §로 분리?</li>
<li><span class="q">선택 4</span> 민감도 분석을 §6 하나로 모으는 것(현안) vs 각 구획 끝에 흩어 두기?</li>
<li><span class="q">선택 5</span> 전체 순서 §0→§8 자체에 대한 이견?</li>
</ol>
"""
OUT_PATH.write_text(html)
print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.0f} KB)")
print("counts ok:", counts)
