"""이슈 624 중첩 결합 판정의 보조 진단: 같은 크기 대조군과 짝지은 행 부트스트랩. (#624, 원칙 A5·A6)

판정 회차(`round_issue624_own36.py`, `round_issue624_ext314.py`)가 끝난 뒤에 돈다.
판정 자체는 바꾸지 않으며, 두 가지 보조값만 기록한다.

- 대조군: 기준 팔에 동결 명세의 계열별 기준 재실행 4개(`baseline_reruns`, 풀 밖 검증
  구성원, 새 특성 없음)를 raw4 단과 같은 크기로 더한 구성의 nested AUC. raw4 단의 증분이
  4열을 더한 일반 효과인지 제약 파생 열의 정보인지 가른다.
- 부트스트랩: 기준 팔 nested 예측과 각 평가 팔·대조군 nested 예측을 같은 행 재표집으로
  채점한 AUC 차이의 백분위 구간(95%)과 차이가 0 이하인 비율.

기준 팔 nested 예측은 판정 회차가 저장하지 않으므로(자기 검사는 AUC·해시만 기록)
같은 캐시 행렬과 결합기로 다시 만든다.

    # 1) 기준 팔·대조군 분할 재현(회차마다 5+5개, 동시 3개 상한)
    uv run python scripts/aux_diagnostics_issue624.py replay --round own36 --arm reference --fold 0
    uv run python scripts/aux_diagnostics_issue624.py replay --round own36 --arm control --fold 0
    # 2) 부트스트랩과 기록
    uv run python scripts/aux_diagnostics_issue624.py bootstrap --round own36
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from pipeline import ensemble
from pipeline.data import ID
from pipeline.members import HASH_VERIFIED, MemberSource, MemberSpec, load_members
from pipeline.round import JudgmentRound
from pipeline.runs import MlflowRunStore
from pipeline.sealed import canonical_sha256

ROOT = Path.cwd()
SCRIPTS = {"own36": Path("scripts/round_issue624_own36.py"), "ext314": Path("scripts/round_issue624_ext314.py")}
FREEZE_SPEC = Path("docs/research/reproduction-pool-freeze/rpf-v1-6fa08f3da327.json")
OUT_ROOT = Path("docs/research/reproduction-pool-aux-diagnostics/issue624")
CONTROL_ARM = "control-baseline-reruns"


def _load_spec(name: str):
    script = SCRIPTS[name]
    module_spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[script.stem] = module
    module_spec.loader.exec_module(module)
    return module.SPEC


def _round(name: str, *, store: bool) -> JudgmentRound:
    spec = _load_spec(name)
    return JudgmentRound(spec, store=MlflowRunStore() if store else None, root=ROOT, script=SCRIPTS[name])


def _precommit_payload(round_: JudgmentRound) -> dict:
    return json.loads(round_.precommit_path.read_text(encoding="utf-8"))


def _aux_dir(round_: JudgmentRound) -> Path:
    return round_.run_dir / "aux"


def _control_source(reference: MemberSource) -> MemberSource:
    freeze = json.loads(FREEZE_SPEC.read_text(encoding="utf-8"))
    reruns = tuple(
        MemberSpec(
            member_id=f"rerun:{row['config']}",
            origin="reproduction",
            verification=HASH_VERIFIED,
            run_id=row["run_id"],
            oof_sha256=row["oof"]["array_sha256"],
            expected_auc=row["oof"]["auc"],
        )
        for row in freeze["baseline_reruns"]
    )
    return MemberSource(name=f"{reference.name}+{FREEZE_SPEC}#baseline_reruns", members=reruns, train_rows=reference.train_rows)


def replay(name: str, arm: str, fold: int) -> None:
    round_ = _round(name, store=arm == "control")
    payload = _precommit_payload(round_)
    fold_of, y = round_._load_fold_and_labels()
    reference_name = round_.spec.reference.name
    matrix = round_._load_cached_matrix(payload, reference_name, fold_of)
    out_dir = _aux_dir(round_) / ("reference" if arm == "reference" else CONTROL_ARM) / f"fold-{fold}"
    if (out_dir / "fold.json").exists():
        print(f"건너뜀(완료): {name} {arm} 분할 {fold}")
        return
    if arm == "control":
        control = load_members(_control_source(round_.spec.reference.source), fold_of.index, round_.store, labels=y)
        extra = control.oof_frame().astype(np.float64)
        matrix = pd.concat([matrix, extra], axis=1)
    started = time.monotonic()
    outcome = ensemble.evaluate_outer_fold(round_._combiner(fold_of), matrix, fold_of, y, fold)
    elapsed = time.monotonic() - started
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: outcome.prediction.index, "prediction": outcome.prediction.to_numpy()}).to_parquet(out_dir / "predictions.parquet")
    record = {
        "round_id": round_.spec.round_id,
        "arm": arm,
        "members": list(matrix.columns),
        "member_count": int(matrix.shape[1]),
        "composition_sha256": canonical_sha256(list(matrix.columns)),
        "fold": fold,
        "auc": outcome.auc,
        "prediction_sha256": outcome.prediction_identity,
        "elapsed_seconds": elapsed,
    }
    (out_dir / "fold.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{name} {arm} 분할 {fold}: AUC {outcome.auc:.15f}, {elapsed:.0f}초", flush=True)


def _nested(dir_of_arm: Path, folds: list[int]) -> pd.Series:
    parts = [pd.read_parquet(dir_of_arm / f"fold-{fold}" / "predictions.parquet").set_index(ID)["prediction"] for fold in folds]
    return pd.concat(parts)


def _fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    ranks = rankdata(score, method="average")
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap(name: str, resamples: int, seed: int) -> None:
    round_ = _round(name, store=False)
    payload = _precommit_payload(round_)
    fold_of, y = round_._load_fold_and_labels()
    folds = [int(f) for f in payload["outer_folds"]]
    aux = _aux_dir(round_)
    reference = _nested(aux / "reference", folds).reindex(fold_of.index)
    arms: dict[str, pd.Series] = {}
    for arm in round_.spec.candidates:
        arms[arm.name] = _nested(round_.run_dir / "arms" / arm.name, folds).reindex(fold_of.index)
    arms[CONTROL_ARM] = _nested(aux / CONTROL_ARM, folds).reindex(fold_of.index)
    for label, series in [("reference", reference), *arms.items()]:
        if series.isna().any():
            raise SystemExit(f"{label}: nested 예측에 빈 행이 있다.")
    y_arr = y.to_numpy().astype(np.int8)
    ref_arr = reference.to_numpy()
    full_ref_auc = _fast_auc(y_arr, ref_arr)
    expected_ref = float(payload["reference"]["nested_auc"])
    ref_replayed = abs(full_ref_auc - expected_ref) < 1e-9
    rng = np.random.default_rng(seed)
    n = y_arr.size
    deltas = {arm: np.empty(resamples) for arm in arms}
    arm_arrays = {arm: series.to_numpy() for arm, series in arms.items()}
    started = time.monotonic()
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        y_b = y_arr[idx]
        ref_b = _fast_auc(y_b, ref_arr[idx])
        for arm, arr in arm_arrays.items():
            deltas[arm][i] = _fast_auc(y_b, arr[idx]) - ref_b
    elapsed = time.monotonic() - started
    control_fold_aucs = {
        str(fold): json.loads((aux / CONTROL_ARM / f"fold-{fold}" / "fold.json").read_text(encoding="utf-8"))["auc"] for fold in folds
    }
    reference_fold_aucs = {str(fold): float(v) for fold, v in payload["reference"]["fold_aucs"].items()}
    result = {
        "schema": "reproduction-pool-aux-diagnostics/1",
        "round_id": round_.spec.round_id,
        "precommit_sha256": payload["sealed_sha256"],
        "reference": {
            "name": round_.spec.reference.name,
            "nested_auc_expected": expected_ref,
            "nested_auc_replayed": full_ref_auc,
            "replay_matches": ref_replayed,
        },
        "control": {
            "arm": CONTROL_ARM,
            "definition": "기준 팔 + 동결 명세 baseline_reruns 4개(exp117·exp135·exp070·exp139 재실행, 3시드, 새 특성 없음). raw4 단과 같은 크기.",
            "member_count": int(payload["arms"][round_.spec.reference.name]["member_count"]) + 4,
            "nested_auc": _fast_auc(y_arr, arm_arrays[CONTROL_ARM]),
            "delta": _fast_auc(y_arr, arm_arrays[CONTROL_ARM]) - full_ref_auc,
            "fold_aucs": control_fold_aucs,
            "fold_deltas": {k: control_fold_aucs[k] - reference_fold_aucs[k] for k in control_fold_aucs},
            "folds_positive": sum(control_fold_aucs[k] > reference_fold_aucs[k] for k in control_fold_aucs),
        },
        "bootstrap": {
            "method": "짝지은 행 재표집(복원 추출, 전체 행 수), 같은 표본으로 기준·평가 팔 AUC를 채점한 차이의 백분위 구간",
            "resamples": resamples,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "arms": {
                arm: {
                    "point_delta": float(_fast_auc(y_arr, arm_arrays[arm]) - full_ref_auc),
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "ci90": [float(np.percentile(d, 5)), float(np.percentile(d, 95))],
                    "bootstrap_mean": float(d.mean()),
                    "bootstrap_sd": float(d.std(ddof=1)),
                    "fraction_nonpositive": float((d <= 0).mean()),
                    "fraction_at_or_above_gate": float((d >= 0.00002).mean()),
                }
                for arm, d in deltas.items()
            },
        },
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / f"{name}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"기록: {out} (기준 재현 {'일치' if ref_replayed else '불일치'}, 부트스트랩 {elapsed:.0f}초)")
    for arm, body in result["bootstrap"]["arms"].items():
        print(f"  {arm}: delta {body['point_delta']:+.7f} 95% [{body['ci95'][0]:+.7f}, {body['ci95'][1]:+.7f}] P(<=0) {body['fraction_nonpositive']:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("replay")
    rep.add_argument("--round", required=True, choices=sorted(SCRIPTS))
    rep.add_argument("--arm", required=True, choices=["reference", "control"])
    rep.add_argument("--fold", type=int, required=True)
    boot = sub.add_parser("bootstrap")
    boot.add_argument("--round", required=True, choices=sorted(SCRIPTS))
    boot.add_argument("--resamples", type=int, default=1000)
    boot.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args(argv)
    if args.command == "replay":
        replay(args.round, args.arm, args.fold)
    else:
        bootstrap(args.round, args.resamples, args.seed)


if __name__ == "__main__":
    main()
