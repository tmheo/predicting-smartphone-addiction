"""트리 모델 계열 challenger의 champion 대비 다양성 진단. (#59)

계열별 OOF 실행(run_id)을 받아 champion과 비교한 다양성 지표를 출력한다.
채택 판정은 compare(계열 1)·pool(계열 2)이 소유하고, 이 스크립트는
티켓 #59가 요구하는 관찰 지표만 계산한다.

측정 항목(run_id마다):
1. 단독 OOF AUC와 fold별 AUC, champion 대비 fold 승리 수.
2. champion OOF와의 스피어만 순위 상관.
3. champion과의 잔차 상관: corr(pred - y, champ_pred - y) 피어슨.
4. champion과의 단순 결합 기여: 순위 평균 2원 blend OOF AUC - champion OOF AUC.

사용법:
    uv run python scripts/diagnose_family_diversity.py <run_id> [<run_id> ...]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yaml

from pipeline.compare import CHAMPION_PATH, TRACKING_URI
from pipeline.cv import score_predictions
from pipeline.data import ID, TARGET
from pipeline.pool import _load_labels, _member_pred, rank_ensemble_auc, spearman

FOLDS_PATH = "artifacts/folds.parquet"


def load_run(run_id: str) -> tuple[str, pd.Series]:
    """run의 실험 이름과 OOF 예측(id 인덱스)을 돌려준다."""
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    run = client.get_run(run_id)
    return run.data.params["experiment"], _member_pred(run_id)


def main() -> None:
    run_ids = sys.argv[1:]
    if not run_ids:
        sys.exit("사용법: diagnose_family_diversity.py <run_id> [<run_id> ...]")

    with CHAMPION_PATH.open() as f:
        champion = yaml.safe_load(f)
    champ_pred = _member_pred(champion["run_id"])
    y = _load_labels(champ_pred.index)
    folds = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"].reindex(champ_pred.index)
    assert folds.notna().all(), "folds.parquet의 id가 champion OOF와 일치하지 않는다."
    champ_fold_aucs = score_predictions(y, folds, champ_pred.to_numpy())
    champ_resid = champ_pred.to_numpy() - y.to_numpy()

    print(
        f"champion: {champion['config']} run {champion['run_id']} "
        f"(auc_oof {champ_fold_aucs['auc_oof']:.5f})"
    )
    for run_id in run_ids:
        name, pred = load_run(run_id)
        pred = pred.reindex(champ_pred.index)
        assert pred.notna().all(), f"run {run_id}의 OOF id가 champion과 일치하지 않는다."
        fold_aucs = score_predictions(y, folds, pred.to_numpy())
        wins = sum(
            fold_aucs[f"auc_fold_{f}"] > champ_fold_aucs[f"auc_fold_{f}"]
            for f in range(5)
        )
        rho = spearman(pred, champ_pred)
        resid_corr = float(np.corrcoef(pred.to_numpy() - y.to_numpy(), champ_resid)[0, 1])
        blend_auc = rank_ensemble_auc([champ_pred, pred], y)
        blend_delta = blend_auc - champ_fold_aucs["auc_oof"]

        print(f"\n{name} run {run_id}")
        print(f"  단독 OOF AUC : {fold_aucs['auc_oof']:.5f} "
              f"(champion 대비 {fold_aucs['auc_oof'] - champ_fold_aucs['auc_oof']:+.5f})")
        per_fold = ", ".join(f"{fold_aucs[f'auc_fold_{f}']:.5f}" for f in range(5))
        print(f"  fold별 AUC   : {per_fold} (champion 대비 승리 {wins}/5)")
        print(f"  순위 상관    : 스피어만 {rho:.5f}")
        print(f"  잔차 상관    : 피어슨 {resid_corr:.5f}")
        print(f"  단순 결합    : champion+이 run 순위 평균 blend OOF AUC {blend_auc:.5f} "
              f"(champion 대비 {blend_delta:+.5f})")


if __name__ == "__main__":
    main()
