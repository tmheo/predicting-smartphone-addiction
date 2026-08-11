"""PROTOTYPE (이슈 #41, 버리는 코드) - MLflow 결과 요약의 표와 그림 형태 후보를 실데이터로 만들어 본다.

챔피언 실행 264f7e6f의 feature_importance.parquet을 읽어,
실행이 MLflow artifacts/summary/ 아래에 남길 후보 파일들을 생성한다.

    uv run python scripts/prototype41_summary.py <출력 디렉터리>

생성물:
- feature_importance_summary.csv : 전체 특성 순위표 (평균 gain, 표준편차, 순위, 점유율, 플라시보 비교)
- top30_gain.png                 : 그림 A안 - 평균 gain 절대값(선형 축) 가로 막대 + 표준편차 + 플라시보 기준선
- top30_gain_log.png             : 그림 B안 - 같은 값을 로그 축으로, 플라시보 기준선과의 간격이 전 구간에서 보이는 안
- stage_durations.csv            : 단계별 소요 시간 (이슈 #40 규약의 time.* 지표를 표로 편 것, 표본값)
- summary.html                   : 위 전부를 한 파일에 모은 자체 완결 HTML (MLflow UI에서 클릭 한 번에 보는 안)
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_ID = "264f7e6f8e45402e8322e522c426e8eb"
IMPORTANCE = Path(f"mlruns/1/{RUN_ID}/artifacts/feature_importance.parquet")
PLACEBO_FEATURES = ["placebo_noise", "placebo_noise_te"]
TOP_N = 30

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def build_summary_table(importance: pd.DataFrame) -> pd.DataFrame:
    """(feature, fold, seed, gain) → 특성별 요약표. 순위는 평균 gain 내림차순."""
    g = importance.groupby("feature")["gain"]
    table = pd.DataFrame({"gain_mean": g.mean(), "gain_std": g.std()})
    table["gain_share_pct"] = table["gain_mean"] / table["gain_mean"].sum() * 100
    placebo_ref = table.loc[
        [f for f in PLACEBO_FEATURES if f in table.index], "gain_mean"
    ].max()
    table["vs_placebo"] = table["gain_mean"] / placebo_ref
    table["below_placebo"] = table["gain_mean"] < placebo_ref
    table = table.sort_values("gain_mean", ascending=False).reset_index()
    table.insert(0, "rank", range(1, len(table) + 1))
    return table.round(
        {"gain_mean": 1, "gain_std": 1, "gain_share_pct": 2, "vs_placebo": 2}
    )


def sample_stage_durations() -> pd.DataFrame:
    """이슈 #40 규약(time.<stage>_seconds, 반복 단계는 step=시드 순번)을 표로 편 표본값.

    챔피언 실행에는 아직 time.* 지표가 없으므로 그럴듯한 값을 손으로 넣는다.
    """
    rows = [
        ("setup", 0, 1.8),
        ("data_load", 0, 3.2),
        ("feature_build", 0, 0.9),
        ("feature_build", 1, 0.8),
        ("feature_build", 2, 0.8),
        ("training", 0, 41.5),
        ("training", 1, 40.2),
        ("training", 2, 42.7),
        ("evaluation", 0, 0.6),
        ("artifacts", 0, 4.9),
    ]
    return pd.DataFrame(rows, columns=["stage", "step", "seconds"])


def plot_top(table: pd.DataFrame, title: str, placebo_line: float, out: Path,
             log_scale: bool) -> None:
    top = table.head(TOP_N).iloc[::-1]
    colors = [
        "#d62728" if f in PLACEBO_FEATURES else "#4c72b0" for f in top["feature"]
    ]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(top))))
    ax.barh(
        top["feature"], top["gain_mean"], xerr=top["gain_std"],
        color=colors, error_kw={"ecolor": "#555", "capsize": 2},
    )
    if log_scale:
        ax.set_xscale("log")
    ax.axvline(placebo_line, color="#d62728", linestyle="--", linewidth=1)
    ax.annotate(
        f"플라시보 기준 {placebo_line:,.1f}", xy=(placebo_line, -0.5),
        xytext=(6, 0), textcoords="offset points", color="#d62728", fontsize=9, va="bottom",
    )
    ax.set_title(title)
    ax.set_xlabel("평균 gain" + (" (로그 축)" if log_scale else ""))
    ax.margins(y=0.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def img_tag(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'


def build_html(table: pd.DataFrame, durations: pd.DataFrame, out_dir: Path) -> str:
    def table_rows() -> str:
        rows = []
        for _, r in table.iterrows():
            cls = ' class="placebo"' if r["feature"] in PLACEBO_FEATURES else (
                ' class="below"' if r["below_placebo"] else ""
            )
            rows.append(
                f"<tr{cls}><td>{r['rank']}</td><td>{r['feature']}</td>"
                f"<td>{r['gain_mean']:,.1f}</td><td>{r['gain_std']:,.1f}</td>"
                f"<td>{r['gain_share_pct']:.2f}%</td><td>{r['vs_placebo']:.2f}×</td></tr>"
            )
        return "\n".join(rows)

    total = durations["seconds"].sum()
    max_sec = durations["seconds"].max()

    def duration_rows() -> str:
        rows = []
        for _, r in durations.iterrows():
            width = r["seconds"] / max_sec * 100
            rows.append(
                f"<tr><td>{r['stage']}</td><td>{r['step']}</td>"
                f"<td>{r['seconds']:.1f}s</td>"
                f'<td><div class="bar" style="width:{width:.0f}%"></div></td></tr>'
            )
        return "\n".join(rows)

    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>실행 요약 - exp006_te_drop_gaming</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem auto; max-width: 960px; color: #222; }}
h1 {{ font-size: 1.3rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; font-size: 0.85rem; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
th {{ background: #f5f5f5; }} td:nth-child(2), td:first-child {{ text-align: left; }}
tr.placebo td {{ background: #fdecea; color: #b02a25; }}
tr.below td {{ background: #fff8e1; }}
.bar {{ background: #4c72b0; height: 10px; border-radius: 2px; }}
.meta {{ color: #666; font-size: 0.85rem; }}
.note {{ background: #eef4fb; padding: 8px 12px; font-size: 0.85rem; border-radius: 4px; }}
</style></head><body>
<h1>실행 요약: exp006_te_drop_gaming</h1>
<p class="meta">run_id {RUN_ID} · auc_oof 0.96659 · 시드 42,43,44 · 5 fold</p>
<p class="note">프로토타입 표본입니다. 단계별 소요 시간은 이슈 #40 규약을 표로 편 가짜 값입니다.</p>

<h2>단계별 소요 시간 (합계 {total:.1f}s)</h2>
<table><tr><th>stage</th><th>step(시드 순번)</th><th>seconds</th><th style="width:40%"></th></tr>
{duration_rows()}</table>

<h2>그림 A안: 상위 {TOP_N} 평균 gain (선형 축 + 표준편차)</h2>
{img_tag(out_dir / "top30_gain.png")}

<h2>그림 B안: 같은 값을 로그 축으로 (플라시보와의 간격이 전 구간에서 보임)</h2>
{img_tag(out_dir / "top30_gain_log.png")}

<h2>특성별 요약표 (전체 {len(table)}개)</h2>
<p class="meta">빨강 = 플라시보 특성, 노랑 = 플라시보보다 평균 gain이 낮은 특성. vs_placebo는 플라시보 최대 평균 gain 대비 배수.</p>
<table><tr><th>rank</th><th>feature</th><th>gain_mean</th><th>gain_std</th><th>share</th><th>vs_placebo</th></tr>
{table_rows()}</table>
</body></html>"""


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    importance = pd.read_parquet(IMPORTANCE)
    table = build_summary_table(importance)
    table.to_csv(out_dir / "feature_importance_summary.csv", index=False)

    placebo_mean = table.loc[
        table["feature"].isin(PLACEBO_FEATURES), "gain_mean"
    ].max()
    plot_top(
        table, f"상위 {TOP_N} 특성 평균 gain (fold×seed 표준편차)",
        placebo_mean, out_dir / "top30_gain.png", log_scale=False,
    )
    plot_top(
        table, f"상위 {TOP_N} 특성 평균 gain (로그 축)",
        placebo_mean, out_dir / "top30_gain_log.png", log_scale=True,
    )

    durations = sample_stage_durations()
    durations.to_csv(out_dir / "stage_durations.csv", index=False)

    (out_dir / "summary.html").write_text(build_html(table, durations, out_dir))
    print(f"생성 완료: {out_dir}")


if __name__ == "__main__":
    main()
