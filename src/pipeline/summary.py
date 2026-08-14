"""요약 생성기. artifacts 단계에서 결과 요약(CSV, PNG, HTML)을 만든다. (#41, #43)

산출물(기존 아티팩트와 같은 루트):
- feature_importance_summary.csv : 전체 특성 순위표(평균 gain, 표준편차, 점유율, 플라시보 비교)
- top30_gain.png                 : 상위 30개 평균 gain 가로 막대, 로그 축 + 플라시보 기준선 (#41 채택안)
- stage_durations.csv            : 단계별 소요 시간(stage, step=시드 순번, seconds)
- summary.html                   : 전부를 한 파일에 모은 자체 완결 HTML(그림 base64 내장)

여기서 던진 예외는 그대로 전파되어 실행이 FAILED가 된다. (#42, #43)
원본 산출물은 이미 저장된 뒤이므로 필요하면 사후 재생성으로 복구한다.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .config import ExperimentConfig
from .cv import CVResult
from .features import PLACEBO
from .judgment import fold_aucs_of, mean_gain_of

TOP_N = 30

# 글리프 단위 폴백(matplotlib 3.6+): 숫자·기호는 DejaVu, 한글은 설치된 한글 폰트가 맡는다.
# 한글 폰트는 macOS의 AppleGothic, Kaggle/리눅스의 나눔·Noto 계열 중 설치된 것만 넣는다
# (미설치 폰트를 목록에 남기면 findfont 경고가 그림마다 쏟아진다).
_KOREAN_FONTS = ("AppleGothic", "NanumGothic", "Noto Sans CJK KR", "NanumBarunGothic")
_available = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [
    "DejaVu Sans", *(f for f in _KOREAN_FONTS if f in _available), "sans-serif"
]
plt.rcParams["axes.unicode_minus"] = False


def _placebo_features(features) -> list[str]:
    """플라시보 원본과 그 파생(카나리아 placebo_noise_te 등)을 함께 잡는다."""
    return [f for f in features if f == PLACEBO or f.startswith(f"{PLACEBO}_")]


def build_summary_table(importance: pd.DataFrame) -> pd.DataFrame:
    """(feature, fold, seed, gain) -> 특성별 요약표. 순위는 평균 gain 내림차순. (#41)"""
    table = pd.DataFrame(
        {
            "gain_mean": mean_gain_of(importance),
            "gain_std": importance.groupby("feature")["gain"].std(),
        }
    )
    table["gain_share_pct"] = table["gain_mean"] / table["gain_mean"].sum() * 100
    # 플라시보 기준값은 플라시보 특성들의 평균 gain 최댓값. (#41)
    placebo_ref = table.loc[_placebo_features(table.index), "gain_mean"].max()
    table["vs_placebo"] = table["gain_mean"] / placebo_ref
    table["below_placebo"] = table["gain_mean"] < placebo_ref
    table = table.sort_values("gain_mean", ascending=False).reset_index()
    table.insert(0, "rank", range(1, len(table) + 1))
    # gain은 계열마다 축척이 다르다(트리 gain은 수백, permutation은 1e-4 안팎, #97).
    # 반올림하면 작은 축척이 0으로 뭉개지므로 원값을 유지하고 표시 단계에서 유효숫자로 줄인다.
    return table.round({"gain_share_pct": 2, "vs_placebo": 2})


def plot_top_gain(table: pd.DataFrame, out: Path) -> None:
    """상위 30개 평균 gain 그림. 로그 축이라 플라시보 기준선과의 간격이 전 구간에서 보인다. (#41)"""
    placebo = _placebo_features(table["feature"])
    placebo_line = table.loc[table["feature"].isin(placebo), "gain_mean"].max()
    top = table.head(TOP_N).iloc[::-1]
    colors = ["#d62728" if f in placebo else "#4c72b0" for f in top["feature"]]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(top))))
    ax.barh(
        top["feature"],
        top["gain_mean"],
        xerr=top["gain_std"],
        color=colors,
        error_kw={"ecolor": "#555", "capsize": 2},
    )
    ax.set_xscale("log")
    ax.axvline(placebo_line, color="#d62728", linestyle="--", linewidth=1)
    ax.annotate(
        f"플라시보 기준 {placebo_line:.3g}",
        xy=(placebo_line, -0.5),
        xytext=(6, 0),
        textcoords="offset points",
        color="#d62728",
        fontsize=9,
        va="bottom",
    )
    ax.set_title(f"상위 {TOP_N} 특성 평균 gain (fold×seed 표준편차, 로그 축)")
    ax.set_xlabel("평균 gain (로그 축)")
    ax.margins(y=0.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _img_tag(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'


def build_html(
    cfg: ExperimentConfig,
    run_id: str,
    result: CVResult,
    table: pd.DataFrame,
    durations: pd.DataFrame,
    png_path: Path,
) -> str:
    """배치: 실행 메타 -> 단계별 소요 시간 -> 로그 축 gain 그림 -> 전체 순위표. (#41)"""
    placebo = set(_placebo_features(table["feature"]))
    fold_total = len(fold_aucs_of(result.fold_aucs))

    def table_rows() -> str:
        rows = []
        for _, r in table.iterrows():
            cls = (
                ' class="placebo"'
                if r["feature"] in placebo
                else (' class="below"' if r["below_placebo"] else "")
            )
            rows.append(
                f"<tr{cls}><td>{r['rank']}</td><td>{r['feature']}</td>"
                f"<td>{r['gain_mean']:.4g}</td><td>{r['gain_std']:.4g}</td>"
                f"<td>{r['gain_share_pct']:.2f}%</td><td>{r['vs_placebo']:.2f}×</td></tr>"
            )
        return "\n".join(rows)

    total = durations["seconds"].sum()
    max_sec = durations["seconds"].max() if len(durations) else 1.0

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
<title>실행 요약 - {cfg.name}</title>
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
</style></head><body>
<h1>실행 요약: {cfg.name}</h1>
<p class="meta">run_id {run_id} · auc_oof {result.fold_aucs["auc_oof"]:.5f} ·
시드 {",".join(map(str, cfg.seeds))} · {fold_total} fold</p>

<h2>단계별 소요 시간 (합계 {total:.1f}s, artifacts 단계는 진행 중이라 제외)</h2>
<table><tr><th>stage</th><th>step(시드 순번)</th><th>seconds</th><th style="width:40%"></th></tr>
{duration_rows()}</table>

<h2>상위 {TOP_N} 특성 평균 gain (로그 축)</h2>
{_img_tag(png_path)}

<h2>특성별 요약표 (전체 {len(table)}개)</h2>
<p class="meta">빨강 = 플라시보 특성, 노랑 = 플라시보보다 평균 gain이 낮은 특성.
vs_placebo는 플라시보 최대 평균 gain 대비 배수.</p>
<table><tr><th>rank</th><th>feature</th><th>gain_mean</th><th>gain_std</th><th>share</th><th>vs_placebo</th></tr>
{table_rows()}</table>
</body></html>"""


def generate_and_log(
    client,
    run_id: str,
    cfg: ExperimentConfig,
    result: CVResult,
    stage_durations: list[tuple[str, int, float]],
) -> None:
    """결과 요약 산출물 네 개를 만들어 기존 아티팩트 루트에 올린다. (#41)"""
    durations = pd.DataFrame(stage_durations, columns=["stage", "step", "seconds"])
    table = build_summary_table(result.importance)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        table.to_csv(tmp_dir / "feature_importance_summary.csv", index=False, float_format="%.6g")
        durations.to_csv(tmp_dir / "stage_durations.csv", index=False)
        plot_top_gain(table, tmp_dir / "top30_gain.png")
        html = build_html(cfg, run_id, result, table, durations, tmp_dir / "top30_gain.png")
        (tmp_dir / "summary.html").write_text(html)
        for name in (
            "feature_importance_summary.csv",
            "stage_durations.csv",
            "top30_gain.png",
            "summary.html",
        ):
            client.log_artifact(run_id, str(tmp_dir / name))
