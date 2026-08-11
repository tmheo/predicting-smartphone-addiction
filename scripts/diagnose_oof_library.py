"""외부 OOF 라이브러리 대비 후보 풀 다양성의 진입 진단. (#77)

szymonkapiski 25구성원 OOF 라이브러리(CC0)를 읽기 전용 진단으로 사용해,
우리 후보 풀에 빠진 오류 계열이 무엇인지 측정한다. 라이브러리 예측은
후보 풀과 최종 제출에 쓰지 않는다(#76 결정).

측정 항목:
1. 선행 검증: 행 수·id 순서·라벨·fold 스펙 일치 확인, 라이브러리 OOF를
   우리 라벨로 재채점해 manifest 점수와 대조. 불일치 구성원은 제외.
2. 우리 후보 풀 구성원 x 라이브러리 구성원의 OOF 스피어만 상관 행렬.
3. 라이브러리 blend(fold 재적합 로지스틱 스택) 위에 우리 구성원을 얹었을 때의
   한계 기여. 영정보 대조 대역(난수 열·최강 구성원 복제)과 비교한다.
4. 우리 풀 blend 위에 라이브러리 구성원을 하나씩 얹었을 때의 한계 기여로
   빠진 오류 계열의 우선순위를 잰다.

사용법:
    uv run python scripts/diagnose_oof_library.py

산출: 표준 출력 리포트와 run-logs/oof_library_corr.csv(전체 상관 행렬).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pipeline.pool import POOL_PATH, _member_pred

LIB = Path("data/external/s6e8-oof-library")
TRAIN_PATH = Path("data/train.csv")
FOLDS_PATH = Path("artifacts/folds.parquet")
CORR_OUT = Path("run-logs/oof_library_corr.csv")
TARGET = "addicted_label"
SEED = 42
# manifest는 소수 5자리라 반올림 오차 5e-6에 float32 저장 오차 여유를 더한다.
RESCORE_TOLERANCE = 1e-4


def load_library() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    manifest = pd.read_csv(LIB / "manifest.csv")
    keys = pd.read_parquet(LIB / "train_keys.parquet")
    oofs = {m: np.load(LIB / "oof" / f"oof_{m}.npy") for m in manifest["model"]}
    return manifest, keys, oofs


def check_alignment(keys: pd.DataFrame, train: pd.DataFrame, folds: pd.DataFrame) -> None:
    """행 수, id 순서, 라벨, fold 스펙이 우리 것과 일치하는지 검증한다."""
    assert len(keys) == len(train), f"행 수 불일치: 라이브러리 {len(keys)} vs train {len(train)}"
    assert np.array_equal(keys["id"].to_numpy(), train["id"].to_numpy()), "id 순서 불일치"
    assert np.array_equal(
        keys[TARGET].to_numpy(), train[TARGET].to_numpy()
    ), "라벨 불일치"
    assert np.array_equal(folds["id"].to_numpy(), train["id"].to_numpy()), (
        "folds.parquet id 순서가 train과 다르다"
    )
    print(f"선행 검증 통과: {len(train)}행, id 순서·라벨 일치, "
          f"fold 스펙은 라이브러리 README와 동일한 StratifiedKFold(5, shuffle, seed 42)")


def rescore(manifest: pd.DataFrame, oofs: dict[str, np.ndarray], y: np.ndarray) -> list[str]:
    """라이브러리 OOF를 우리 라벨로 재채점해 manifest와 대조하고 일치 구성원만 남긴다."""
    kept: list[str] = []
    print("\n== 재채점 대조 (우리 라벨 기준 AUC vs manifest) ==")
    for _, row in manifest.iterrows():
        name = row["model"]
        auc = roc_auc_score(y, oofs[name])
        diff = auc - row["oof_auc"]
        ok = abs(diff) <= RESCORE_TOLERANCE
        flag = "" if ok else "  ← 불일치, 진단 제외"
        print(f"  {name:12s} manifest {row['oof_auc']:.5f}  재채점 {auc:.5f}  "
              f"차이 {diff:+.6f}{flag}")
        if ok:
            kept.append(name)
    return kept


def spearman_matrix(ours: dict[str, np.ndarray], lib: dict[str, np.ndarray]) -> pd.DataFrame:
    """우리 구성원 x 라이브러리 구성원의 스피어만 상관 행렬."""
    all_preds = {**ours, **lib}
    ranks = np.column_stack(
        [pd.Series(p).rank().to_numpy() for p in all_preds.values()]
    )
    corr = np.corrcoef(ranks, rowvar=False)
    full = pd.DataFrame(corr, index=list(all_preds), columns=list(all_preds))
    return full.loc[list(ours), list(lib)]


def blend_auc(cols: list[np.ndarray], y: np.ndarray, fold_ids: np.ndarray) -> float:
    """fold 재적합 로지스틱 스택의 OOF AUC. 라이브러리 README의 정직한 blend 절차."""
    X = logit(np.clip(np.column_stack(cols), 1e-6, 1 - 1e-6))
    oof = np.zeros(len(y))
    for f in np.unique(fold_ids):
        tr, va = fold_ids != f, fold_ids == f
        lr = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        oof[va] = lr.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, oof))


def marginal_report(
    title: str,
    base_cols: list[np.ndarray],
    additions: dict[str, np.ndarray],
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> dict[str, float]:
    base = blend_auc(base_cols, y, fold_ids)
    print(f"\n== {title} (기준 blend OOF AUC {base:.5f}) ==")
    deltas = {}
    for name, col in additions.items():
        with_auc = blend_auc(base_cols + [col], y, fold_ids)
        deltas[name] = with_auc - base
        print(f"  + {name:24s} → {with_auc:.5f} (한계 기여 {deltas[name]:+.6f})")
    return deltas


def main() -> None:
    train = pd.read_csv(TRAIN_PATH, usecols=["id", TARGET])
    folds = pd.read_parquet(FOLDS_PATH)
    manifest, keys, lib_oofs = load_library()
    check_alignment(keys, train, folds)

    y = train[TARGET].to_numpy()
    fold_ids = folds["fold"].to_numpy()
    kept = rescore(manifest, lib_oofs, y)
    lib_oofs = {m: lib_oofs[m] for m in kept}

    # 우리 풀 구성원의 OOF를 train 행 순서로 정렬해 로드한다.
    with POOL_PATH.open() as f:
        pool = yaml.safe_load(f)
    ours: dict[str, np.ndarray] = {}
    for m in pool["members"]:
        pred = _member_pred(m["run_id"]).reindex(train["id"])
        assert pred.notna().all(), f"구성원 {m['config']}의 OOF id가 train과 어긋난다"
        ours[m["config"]] = pred.to_numpy()

    corr = spearman_matrix(ours, lib_oofs)
    CORR_OUT.parent.mkdir(exist_ok=True)
    corr.round(5).to_csv(CORR_OUT)
    print(f"\n== 스피어만 상관 (전체 행렬: {CORR_OUT}) ==")
    print("라이브러리 구성원별 우리 풀과의 최대 상관 (낮을수록 우리에게 없는 오류 계열):")
    max_corr = corr.max(axis=0).sort_values()
    for name, rho in max_corr.items():
        fam = manifest.set_index("model").loc[name, "family"]
        print(f"  {name:12s} {rho:.5f}  ({fam})")

    # 영정보 대조: 난수 열과 라이브러리 최강 구성원의 정확한 복제.
    rng = np.random.default_rng(SEED)
    best_lib = manifest.set_index("model").loc[kept, "oof_auc"].idxmax()
    placebos = {
        "placebo_random": rng.uniform(1e-6, 1 - 1e-6, len(y)),
        f"placebo_dup_{best_lib}": lib_oofs[best_lib].astype(np.float64),
    }

    lib_cols = [lib_oofs[m] for m in kept]
    additions = {**placebos, **{f"ours_{k}": v for k, v in ours.items()}}
    marginal_report(
        f"라이브러리 {len(kept)}구성원 blend 위 한계 기여", lib_cols, additions, y, fold_ids
    )
    all_with = blend_auc(lib_cols + list(ours.values()), y, fold_ids)
    base = blend_auc(lib_cols, y, fold_ids)
    print(f"  + 우리 4구성원 동시     → {all_with:.5f} (한계 기여 {all_with - base:+.6f})")

    # 역방향: 우리 풀 blend 위에 라이브러리 구성원을 하나씩 얹어 우선순위를 잰다.
    our_cols = list(ours.values())
    rev_placebos = {
        "placebo_random": placebos["placebo_random"],
        "placebo_dup_ours_best": ours[max(ours, key=lambda k: roc_auc_score(y, ours[k]))],
    }
    rev_additions = {**rev_placebos, **lib_oofs}
    rev_deltas = marginal_report(
        "우리 풀 4구성원 blend 위 한계 기여 (재구현 우선순위)",
        our_cols, rev_additions, y, fold_ids,
    )
    print("\n한계 기여 내림차순 (라이브러리 구성원만):")
    for name in sorted(kept, key=lambda n: rev_deltas[n], reverse=True):
        fam = manifest.set_index("model").loc[name, "family"]
        print(f"  {name:12s} {rev_deltas[name]:+.6f}  (최대 상관 {max_corr[name]:.5f}, {fam})")


if __name__ == "__main__":
    main()
