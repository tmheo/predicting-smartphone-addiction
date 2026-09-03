"""이슈 #623 계열별 짝비교 표. 반입 감사 JSON(import_and_audit.py 출력)만 읽는다.

계열마다 기준 실행과 사다리 3단계를 짝지어 시드별 OOF, 3시드 평균 차이, 시드 평균 OOF의
분할별 부호(5개), 시드x분할 부호(15개)를 마크다운 표로 낸다.

    uv run --frozen python scripts/issue623/report.py \\
        --audit run-logs/issue623/import-audit-local.json \\
        --audit run-logs/issue623/vast/results-1/import-audit.json \\
        --out run-logs/issue623/pair-report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SEEDS = ("42", "43", "44")
FAMILIES = {
    "LightGBM": ("exp117_ag25_gbm_r21", ["cdv2_lgb_raw4", "cdv2_lgb_cats_te", "cdv2_lgb_ratio_round"]),
    "XGBoost": ("exp135_xgb_hpo_trial30", ["cdv2_xgb_raw4", "cdv2_xgb_cats_te", "cdv2_xgb_ratio_round"]),
    "CatBoost": ("exp070_cat_exact_cats", ["cdv2_cat_raw4", "cdv2_cat_cats_te", "cdv2_cat_ratio_round"]),
    "RealMLP": (
        "exp139_realmlp_reference_qnormal_train_test",
        ["cdv2_realmlp_raw4", "cdv2_realmlp_cats_te", "cdv2_realmlp_ratio_round"],
    ),
}


def load(audits: list[Path]) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for path in audits:
        for run in json.loads(path.read_text())["runs"]:
            if run["experiment"] in runs:
                raise RuntimeError(f"실험 {run['experiment']}이 감사 파일 여러 곳에 있다.")
            runs[run["experiment"]] = run
    return runs


def signs(deltas: list[float]) -> str:
    plus = sum(1 for d in deltas if d > 0)
    return f"{plus}/{len(deltas)} " + "".join("+" if d > 0 else ("-" if d < 0 else "0") for d in deltas)


def fold_keys(run: dict) -> list[str]:
    return sorted(k for k in run["fold_aucs"] if k.startswith("auc_fold_"))


def render(runs: dict[str, dict]) -> str:
    lines: list[str] = []
    for family, (baseline, ladder) in FAMILIES.items():
        lines.append(f"### {family}")
        lines.append("")
        base = runs.get(baseline)
        if base is None:
            lines.append(f"기준 `{baseline}` 실행 기록이 아직 없다.")
            lines.append("")
            continue
        lines.append(
            "| 실험 | run | seed 42 | seed 43 | seed 44 | 3시드 평균 OOF | 기준 대비 | 분할 부호(시드 평균, 5) | 시드x분할 부호(15) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        base_seed = {s: base["auc_oof_seed"][s] for s in SEEDS}
        base_mean = sum(base_seed.values()) / len(SEEDS)
        lines.append(
            f"| `{baseline}` (기준) | `{base['main_run_id'][:8]}` | "
            + " | ".join(f"{base_seed[s]:.7f}" for s in SEEDS)
            + f" | {base_mean:.7f} | - | - | - |"
        )
        for experiment in ladder:
            run = runs.get(experiment)
            if run is None:
                lines.append(f"| `{experiment}` | (없음) | | | | | | | |")
                continue
            seed = {s: run["auc_oof_seed"][s] for s in SEEDS}
            mean = sum(seed.values()) / len(SEEDS)
            fold_deltas = [run["fold_aucs"][k] - base["fold_aucs"][k] for k in fold_keys(base)]
            seed_fold_deltas = [
                run["seed_fold_aucs"][s][f] - base["seed_fold_aucs"][s][f]
                for s in SEEDS
                for f in sorted(base["seed_fold_aucs"][s], key=int)
            ]
            lines.append(
                f"| `{experiment}` | `{run['main_run_id'][:8]}` | "
                + " | ".join(f"{seed[s]:.7f}" for s in SEEDS)
                + f" | {mean:.7f} | {mean - base_mean:+.7f} | {signs(fold_deltas)} | {signs(seed_fold_deltas)} |"
            )
        lines.append("")
        lines.append("시드별 기준 대비 차이: " + "; ".join(
            f"`{experiment}` "
            + ", ".join(
                f"s{s} {runs[experiment]['auc_oof_seed'][s] - base_seed[s]:+.7f}" for s in SEEDS
            )
            for experiment in ladder
            if experiment in runs
        ))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    text = render(load(args.audit))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
