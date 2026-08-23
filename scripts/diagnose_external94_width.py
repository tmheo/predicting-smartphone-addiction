"""외부 94 verified OOF로 "폭" 가설의 상한을 읽기 전용으로 진단한다. (#386)

dariushafshar의 94 Verified OOFs 스택(public 0.97097)의 구성원 94개를 우리 고정
5분할과 등록된 결합 전략 집합으로 다시 평가한다. 세 구성을 잰다.

1. own32   - 우리 후보 풀 구성원만
2. ext94   - 외부 94구성원만
3. union   - 32 + 94 합본

이 진단은 **읽기 전용**이다. 외부 예측을 후보 풀 장부에 넣지 않고 champion 판정에도
쓰지 않으며 MLflow 실행도 만들지 않는다(지도 172의 범위 밖 규칙 유지).
najiama 계열 5개는 라이선스가 불명이라 재배포하지 않고 로컬 진단에만 쓴다.

사용법:
    uv run python scripts/diagnose_external94_width.py --verify-only
    uv run python scripts/diagnose_external94_width.py --config own32 --config ext94 --config union

산출: run-logs/issue386/<config>.json 과 표준 출력 리포트.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline import ensemble
from pipeline.data import ID, TRAIN_PATH, labels
from pipeline.judgment import (
    FOLDS_PATH,
    MISSINGNESS_TEST_PATH,
    missingness_reweighting,
)
from pipeline.ledger import Pool
from pipeline.runs import MlflowRunStore

EXT_ROOT = Path("data/external/ext94")
SZYMON = EXT_ROOT / "s6e8-oof-library-47-models" / "oof"
FM = EXT_ROOT / "s6e8-fm-lattice-blend-members"
GOLEM = EXT_ROOT / "s6e8-golem-oof-library"
BEICICC = EXT_ROOT / "beicicc"
OUT_DIR = Path("run-logs/issue386")

N_TRAIN = 691369

# 94 Verified OOFs 노트북(Apache 2.0)의 SCORED_82 이름 목록을 그대로 옮긴다.
# 구성원 집합을 개수가 아니라 이름으로 못박는 것이 원문의 계약이다.
SCORED_82 = [
    "a", "altview", "cat", "cat_tuned", "d", "digit_cat", "digit_lgbm", "digit_xgb",
    "e", "et", "f", "fmdeep", "fmnum", "fmplr", "fmpure", "fmwide", "hgb", "imp_cat",
    "imp_lgbm", "imp_lgbm_tuned", "imp_xgb", "imp_xgb_tuned", "knn", "lat_cat",
    "lat_lgbm", "lat_lgbm_s5", "lat_xgb", "latmax_lgbm", "latr1_lgbm", "latr1_xgb",
    "lattri_lgbm", "lattri_xgb", "latwide_cat", "latwide_lgbm", "latwide_xgb", "lgbm",
    "lgbm_tuned", "logreg", "lookup", "mlp", "naji01", "naji02", "naji03", "naji04",
    "naji05", "nn2", "pub_cat", "pub_donlgbm", "pub_evg", "pub_resnet", "pub_rmlp",
    "pub_ryota", "pub_tabm", "pub_tabnet", "pubfe_cat", "pubfe_lgb", "pubfe_xgb",
    "pubmk_cat", "pubmk_nn", "realmlp", "realmlp15", "rf", "rmlp_lat", "rmlp_lat3",
    "tabm_bounds", "tabm_deep", "tabm_deeper", "tabm_div", "tabm_imp", "tabm_seed3",
    "tabm_wide", "tabm_x12", "view_bounds_cat", "view_bounds_lgbm",
    "view_nolattice_lgbm", "view_rank_cat", "view_rank_lgbm", "view_resid_cat",
    "view_resid_lgbm", "view_resid_xgb", "xgb", "xgb_tuned",
]
FM_KEYS = ["fmplr", "fmnum", "fmdeep", "fmwide", "fmpure"]
# 원문이 열 순서를 보존하는 이유(lbfgs 수치 경로)를 그대로 따른다.
BASE_ORDER = sorted(k for k in SCORED_82 if k not in FM_KEYS) + FM_KEYS

REJECT_NEW = {"sixmember_meta", "sixmember_equal_rank", "sixmember_meta_perp"}
DROP_DUP_NEW = {"exact_value_catboost_fixed4000", "xgb_identity_digit_enhanced103"}

CONFIG_NAMES = ("own32", "ext94", "union", "ext85", "union85")

# 계보 조사(#174)가 재현 불가 5개(naji)와 부분 재현 4개(golem)로 판정한 구성원.
# golem a와 f는 저자가 검증 fold 조기 종료 낙관을 공표했고, naji 5개는 생성 코드가
# 없어 상류가 검증 라벨을 보지 않았다는 보장이 없다. 이 9개를 뺀 85구성원이 우리가
# 실제로 재현 경로를 가진 폭이다. 폭 가설의 실행 가능한 상한은 이쪽으로 읽는다.
UNVERIFIABLE = ("naji01", "naji02", "naji03", "naji04", "naji05", "a", "d", "e", "f")


@dataclass(frozen=True)
class MemberCheck:
    """외부 구성원 하나의 정합 검증 결과."""

    member: str
    source: str
    rows: int
    finite: bool
    auc: float
    manifest_auc: float | None
    auc_delta: float | None
    fold_check: str


def _base_paths() -> dict[str, Path]:
    """SCORED_82의 82개 구성원 파일 경로. 원문의 oof_*.npy 수집 규칙을 따른다."""
    paths: dict[str, Path] = {}
    for root in (SZYMON, FM, GOLEM):
        for path in sorted(root.glob("oof_*.npy")):
            key = path.name[4:-4]
            if key == "pub_ravi" or "band" in key:
                continue
            if key in SCORED_82 and key not in paths:
                paths[key] = path
    missing = [k for k in SCORED_82 if k not in paths]
    assert not missing, f"SCORED_82 구성원 누락 {len(missing)}개: {missing}"
    return paths


def _new_paths() -> list[tuple[str, Path, Path]]:
    """beicicc 신규 12구성원. 원문과 같은 거부·중복 제거와 해시 검사를 건다."""
    candidates: list[tuple[str, Path, Path]] = []
    seen: set[str] = set()
    for path in sorted(BEICICC.glob("**/*_oof.npy")):
        key = path.name[:-8]
        if key in REJECT_NEW or key in DROP_DUP_NEW or path.name.startswith("oof_"):
            continue
        test_path = path.with_name(key + "_test.npy")
        if not test_path.exists():
            continue
        digest = hashlib.sha256(path.read_bytes() + test_path.read_bytes()).hexdigest()
        assert digest not in seen, f"예상 밖 바이트 중복: {key}"
        seen.add(digest)
        candidates.append((key, path, test_path))
    assert len(candidates) == 12, f"신규 구성원 12개를 기대했으나 {len(candidates)}개"
    return candidates


def _szymon_manifest() -> dict[str, float]:
    frame = pd.read_csv(SZYMON.parent / "manifest.csv")
    return dict(zip(frame["model"], frame["oof_auc"], strict=True))


def _golem_manifest() -> dict[str, float]:
    frame = pd.read_csv(GOLEM / "manifest.csv")
    return dict(zip(frame["member"], frame["oof_auc"], strict=True))


def _beicicc_fold_check(oof_path: Path, fold_of: pd.Series) -> str:
    """beicicc 계약의 fold_id.npy(1-based)가 우리 고정 분할과 같은지 본다."""
    key = oof_path.name[:-8]
    fold_path = oof_path.with_name(f"{key}_fold_id.npy")
    if not fold_path.exists():
        # 데이터셋에 따라 구성원 접두사 없는 공용 fold_id.npy를 둔다.
        candidates = sorted(oof_path.parent.glob("*fold_id.npy"))
        if not candidates:
            return "fold 파일 없음"
        fold_path = candidates[0]
    external = np.load(fold_path)
    if len(external) != len(fold_of):
        return f"행 수 불일치 {len(external)}"
    ours = fold_of.to_numpy()
    if np.array_equal(external, ours):
        return "일치(0-based)"
    if np.array_equal(external - 1, ours):
        return "일치(1-based 보정)"
    return "불일치"


def verify_row_order(train: pd.DataFrame, fold_of: pd.Series) -> None:
    """외부 라이브러리가 전제하는 위치 정렬이 우리 기준 순서와 같은지 확인한다."""
    assert len(train) == N_TRAIN, f"train 행 수 {len(train)}"
    assert np.array_equal(fold_of.index.to_numpy(), train[ID].to_numpy()), (
        "artifacts/folds.parquet의 id 순서가 train.csv 파일 순서와 다르다."
    )
    keys = pd.read_parquet(SZYMON.parent / "train_keys.parquet")
    assert np.array_equal(keys[ID].to_numpy(), train[ID].to_numpy()), (
        "szymonkapiski train_keys.parquet의 id 순서가 train.csv와 다르다."
    )


def load_external(fold_of: pd.Series, y: pd.Series) -> tuple[pd.DataFrame, list[MemberCheck]]:
    """외부 94구성원 OOF 행렬과 구성원별 정합 검증 결과."""
    base_paths = _base_paths()
    new_paths = _new_paths()
    szymon_auc = _szymon_manifest()
    golem_auc = _golem_manifest()
    labels_array = y.to_numpy()

    columns: dict[str, np.ndarray] = {}
    checks: list[MemberCheck] = []

    for key in BASE_ORDER:
        path = base_paths[key]
        values = np.load(path).astype(np.float64)
        assert len(values) == N_TRAIN, f"{key}: 행 수 {len(values)}"
        finite = bool(np.isfinite(values).all())
        auc = float(roc_auc_score(labels_array, values))
        source = path.parent.parent.name if path.parent.name == "oof" else path.parent.name
        manifest = szymon_auc.get(key, golem_auc.get(key))
        columns[f"ext_{key}"] = values
        checks.append(
            MemberCheck(
                member=key,
                source=source,
                rows=len(values),
                finite=finite,
                auc=auc,
                manifest_auc=None if manifest is None else float(manifest),
                auc_delta=None if manifest is None else float(auc - manifest),
                fold_check="위치 정렬(계약상 id·fold 열 없음)",
            )
        )

    for key, oof_path, _ in new_paths:
        values = np.load(oof_path).astype(np.float64)
        assert len(values) == N_TRAIN, f"{key}: 행 수 {len(values)}"
        columns[f"ext_{key}"] = values
        checks.append(
            MemberCheck(
                member=key,
                source=oof_path.parent.name,
                rows=len(values),
                finite=bool(np.isfinite(values).all()),
                auc=float(roc_auc_score(labels_array, values)),
                manifest_auc=None,
                auc_delta=None,
                fold_check=_beicicc_fold_check(oof_path, fold_of),
            )
        )

    matrix = pd.DataFrame(columns, index=fold_of.index).astype(np.float64)
    assert matrix.shape[1] == 94, f"외부 구성원 {matrix.shape[1]}개"
    return matrix, checks


def report_verification(checks: list[MemberCheck]) -> dict[str, object]:
    """정합 검증 요약을 출력하고 산출물에 남길 형태로 돌려준다."""
    frame = pd.DataFrame([asdict(check) for check in checks])
    rescored = frame.dropna(subset=["auc_delta"])
    print(f"외부 구성원 {len(frame)}개 정합 검증")
    print(f"  전 구성원 행 수 {N_TRAIN} 일치, 유한값 {bool(frame['finite'].all())}")
    print(
        f"  manifest 대조 가능 {len(rescored)}개: "
        f"|AUC 차| 최대 {rescored['auc_delta'].abs().max():.2e}"
    )
    print(f"  단독 AUC 범위 {frame['auc'].min():.5f} ~ {frame['auc'].max():.5f}")
    for verdict, count in frame["fold_check"].value_counts().items():
        print(f"  fold 검사 '{verdict}': {count}개")
    worst = frame.reindex(frame["auc_delta"].abs().sort_values(ascending=False).index)
    print("  manifest 대비 오차 큰 5개:")
    for row in worst.head(5).itertuples():
        print(f"    {row.member}: 재채점 {row.auc:.6f}, manifest {row.manifest_auc}")
    return {
        "member_count": int(len(frame)),
        "all_finite": bool(frame["finite"].all()),
        "max_abs_auc_delta": float(rescored["auc_delta"].abs().max()),
        "solo_auc_min": float(frame["auc"].min()),
        "solo_auc_max": float(frame["auc"].max()),
        "fold_checks": {str(k): int(v) for k, v in frame["fold_check"].value_counts().items()},
        "members": frame.to_dict(orient="records"),
    }


def evaluate_config(
    name: str,
    matrix: pd.DataFrame,
    fold_of: pd.Series,
    y: pd.Series,
    reweighting,
    only: list[str] | None,
) -> dict[str, object]:
    """한 구성의 등록 결합 전략 전수 nested 평가."""
    combiner_names = list(only) if only else list(ensemble.DEFAULT_COMBINER_NAMES)
    print(f"\n=== 구성 {name}: 구성원 {matrix.shape[1]}명, 전략 {len(combiner_names)}개 ===")
    rows: list[dict[str, object]] = []
    for combiner_name in combiner_names:
        combiner = ensemble.COMBINER_REGISTRY[combiner_name]
        started = time.monotonic()
        try:
            evaluation = ensemble.evaluate_nested(combiner, matrix, fold_of, y, reweighting)
        except ensemble.CombinerConvergenceError as exc:
            print(f"  {combiner_name}: 제외 ({exc})")
            rows.append(
                {
                    "strategy": combiner_name,
                    "failed": True,
                    "reason": str(exc),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            continue
        weighted = evaluation.weighted
        print(
            f"  {combiner_name}: nested {evaluation.nested_auc:.7f}, "
            f"가중 {weighted.auc:.7f} ({weighted.auc - evaluation.nested_auc:+.7f}), "
            f"{evaluation.elapsed_seconds:.0f}s",
            flush=True,
        )
        rows.append(
            {
                "strategy": combiner_name,
                "failed": False,
                "nested_auc": evaluation.nested_auc,
                "weighted_oof_auc": weighted.auc,
                "weighted_delta": weighted.auc - evaluation.nested_auc,
                "weighted_effective_sample_size": weighted.effective_sample_size,
                "weighted_effective_sample_fraction": weighted.effective_sample_fraction,
                "fold_aucs": {str(o.fold): o.auc for o in evaluation.folds},
                "elapsed_seconds": evaluation.elapsed_seconds,
            }
        )
    succeeded = [row for row in rows if not row["failed"]]
    best = max(succeeded, key=lambda row: row["nested_auc"]) if succeeded else None
    if best is not None:
        print(
            f"  최선 전략 {best['strategy']}: nested {best['nested_auc']:.7f}, "
            f"가중 {best['weighted_oof_auc']:.7f}"
        )
    return {
        "config": name,
        "member_count": int(matrix.shape[1]),
        "members": list(matrix.columns),
        "strategies": rows,
        "best": best,
    }


WIDTH_CURVE_SIZES = (5, 10, 20, 40, 60, 85)
WIDTH_CURVE_DRAWS = 3
WIDTH_CURVE_STRATEGY = "rank_logit_logistic"


def width_curve(
    own: pd.DataFrame,
    external: pd.DataFrame,
    fold_of: pd.Series,
    y: pd.Series,
    reweighting,
    strategy: str,
) -> dict[str, object]:
    """외부 구성원을 k개씩 더할 때 중첩 OOF가 얼마나 오르는지의 곡선.

    폭 가설의 참거짓이 아니라 기울기를 잰다. 구성원 생산 규모(#387)를 정하려면
    "94개를 다 더하면 얼마"보다 "k개를 더하면 얼마"가 필요하다. 무작위 뽑기라
    표본마다 흔들리므로 크기마다 서로 다른 씨앗으로 여러 번 뽑아 폭을 함께 남긴다.
    """
    combiner = ensemble.COMBINER_REGISTRY[strategy]
    base = ensemble.evaluate_nested(combiner, own, fold_of, y, reweighting)
    print(f"\n=== 폭 곡선 ({strategy}) ===")
    print(f"  자체 {own.shape[1]}구성원: nested {base.nested_auc:.7f}")
    points: list[dict[str, object]] = []
    for size in WIDTH_CURVE_SIZES:
        if size > external.shape[1]:
            continue
        draws = []
        for draw in range(WIDTH_CURVE_DRAWS):
            rng = np.random.default_rng(1000 * size + draw)
            picked = list(
                external.columns[rng.choice(external.shape[1], size, replace=False)]
            )
            matrix = pd.concat([own, external[picked]], axis=1)
            evaluation = ensemble.evaluate_nested(
                combiner, matrix, fold_of, y, reweighting
            )
            draws.append(
                {
                    "draw": draw,
                    "members": picked,
                    "nested_auc": evaluation.nested_auc,
                    "weighted_oof_auc": evaluation.weighted.auc,
                    "delta": evaluation.nested_auc - base.nested_auc,
                }
            )
        deltas = [row["delta"] for row in draws]
        print(
            f"  +{size:3d}구성원: 평균 nested "
            f"{np.mean([row['nested_auc'] for row in draws]):.7f}, "
            f"증분 평균 {np.mean(deltas):+.7f} "
            f"(최소 {min(deltas):+.7f}, 최대 {max(deltas):+.7f})",
            flush=True,
        )
        points.append(
            {
                "added": size,
                "mean_nested_auc": float(np.mean([r["nested_auc"] for r in draws])),
                "mean_delta": float(np.mean(deltas)),
                "min_delta": float(min(deltas)),
                "max_delta": float(max(deltas)),
                "draws": draws,
            }
        )
    return {
        "strategy": strategy,
        "base_member_count": int(own.shape[1]),
        "base_nested_auc": base.nested_auc,
        "base_weighted_oof_auc": base.weighted.auc,
        "external_pool_size": int(external.shape[1]),
        "draws_per_size": WIDTH_CURVE_DRAWS,
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="외부 94 verified OOF 폭 진단 (#386)")
    parser.add_argument(
        "--config",
        action="append",
        choices=CONFIG_NAMES,
        help="평가할 구성. 여러 번 지정 가능. 기본은 세 구성 전부.",
    )
    parser.add_argument(
        "--only",
        action="append",
        help="이 이름의 등록 결합 전략만 평가(기본은 DEFAULT_COMBINER_NAMES 전부).",
    )
    parser.add_argument("--verify-only", action="store_true", help="정합 검증만 하고 끝낸다.")
    parser.add_argument(
        "--width-curve",
        action="store_true",
        help="구성을 평가하는 대신 외부 구성원 k개 추가의 폭 곡선을 잰다.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    train = pd.read_csv(TRAIN_PATH)
    fold_of = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"]
    y = labels(fold_of.index)
    verify_row_order(train, fold_of)

    external, checks = load_external(fold_of, y)
    verification = report_verification(checks)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2)
    )
    print(f"정합 검증 저장: {args.out_dir / 'verification.json'}")
    if args.verify_only:
        return

    pool = Pool.load()
    members = [(member.config, member.run_id) for member in pool.members]
    own = ensemble.member_matrix(members, MlflowRunStore(), fold_of.index)
    print(f"자체 후보 풀 {own.shape[1]}구성원 적재 완료")

    reweighting = missingness_reweighting(TRAIN_PATH, MISSINGNESS_TEST_PATH)
    verifiable = [f"ext_{name}" for name in UNVERIFIABLE]
    external85 = external.drop(columns=verifiable)
    assert external85.shape[1] == 85, f"재현 가능 구성원 {external85.shape[1]}개"
    matrices = {
        "own32": own,
        "ext94": external,
        "union": pd.concat([own, external], axis=1),
        "ext85": external85,
        "union85": pd.concat([own, external85], axis=1),
    }
    if args.width_curve:
        # 곡선은 재현 경로가 있는 85구성원에서 뽑는다. naji·golem은 우리가 만들 수
        # 없는 구성원이라 생산 규모의 근거가 되지 못한다.
        curve = width_curve(
            own,
            external85,
            fold_of,
            y,
            reweighting,
            (args.only or [WIDTH_CURVE_STRATEGY])[0],
        )
        path = args.out_dir / "width-curve.json"
        path.write_text(json.dumps(curve, ensure_ascii=False, indent=2))
        print(f"폭 곡선 저장: {path}")
        return

    for name in args.config or CONFIG_NAMES:
        result = evaluate_config(name, matrices[name], fold_of, y, reweighting, args.only)
        path = args.out_dir / f"{name}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"구성 {name} 저장: {path}")


if __name__ == "__main__":
    main()
