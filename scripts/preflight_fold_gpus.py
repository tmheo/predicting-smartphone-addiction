"""원격 다중 GPU 사전 검사. (#360)

`docs/agents/vast-resource-control.md`는 다중 GPU 작업이 본 실행 전에 "실제 모델을
각 장치에 배정하는 짧은 학습 검사"를 통과할 것을 요구한다. 저장소의 같은 검사는
`tests/test_model_lookup_transformer.py`에 있지만 pytest는 개발 의존성이라 원격
실행 환경(`uv sync --no-dev`)에 없다. 이 스크립트는 그 검사만 떼어 낸 것이다.

`PIPELINE_FOLD_GPUS`가 가리키는 장치마다 초기화 시드가 다른 구성원 하나를 올려
소형 자료로 짧게 학습하고, 구성원이 실제로 서로 다른 장치에 배정됐는지 확인한다.
초기화 시드 오프셋은 요청 장치 수에 맞춰 0부터 1000 간격으로 만든다.

사용법:
    PIPELINE_FOLD_GPUS=0,1,2 python -m scripts.preflight_fold_gpus
    python scripts/preflight_fold_gpus.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import model as model_mod  # noqa: E402
from pipeline.config import ModelConfig  # noqa: E402

SEED = 7
PARAMS = {
    "lookup_cols": ["v", "c"],
    "d_model": 16,
    "plr_k": 4,
    "layers": 1,
    "heads": 2,
    "epochs": 2,
    "batch_size": 64,
    "lr": 5e-3,
    "ema_decay": 0.7,
    "patience": 2,
    "perm_repeats": 1,
}


def _data(n: int = 96) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(3)
    values = rng.choice([1.5, 2.5, 3.5, 4.5], size=n)
    cat = pd.Series(rng.choice(["Low", "High"], size=n)).where(rng.uniform(size=n) > 0.1)
    X = pd.DataFrame(
        {
            "v": pd.Series(values).where(rng.uniform(size=n) > 0.1),
            "c": pd.Categorical(cat, categories=["High", "Low"]),
            "z": rng.normal(size=n),
        }
    )
    y = pd.Series((values > 2.5).astype(int))
    return X, y


def main() -> None:
    gpus = os.environ.get("PIPELINE_FOLD_GPUS", "").strip()
    if not gpus:
        sys.exit("PIPELINE_FOLD_GPUS가 없다. 다중 GPU 사전 검사를 건너뛰지 않는다.")

    import torch

    device_count = torch.cuda.device_count()
    requested = [int(part.strip()) for part in gpus.split(",")]
    if device_count < len(requested):
        sys.exit(f"장치 {device_count}개로는 PIPELINE_FOLD_GPUS={gpus}를 만족할 수 없다.")

    X, y = _data()
    offsets = [index * 1000 for index in range(len(requested))]
    params = dict(PARAMS, fold_seed_offsets=offsets)
    cfg = ModelConfig(kind="lookup_transformer", params=params, fit={})
    adapter = model_mod.create(cfg, seed=SEED)
    val_pred = adapter.fit(X.iloc[:72], y.iloc[:72], X.iloc[72:], y.iloc[72:])
    test_pred = adapter.predict(X.iloc[:12])

    if val_pred.shape != (24,) or not np.isfinite(val_pred).all():
        sys.exit("검증 예측이 온전하지 않다.")
    if test_pred.shape != (12,) or not np.isfinite(test_pred).all():
        sys.exit("시험 예측이 온전하지 않다.")

    members = adapter.entry_diagnostics().observations["fold_initialization_members"]
    devices = [member["device"] for member in members]
    if len(set(devices)) != len(requested):
        sys.exit(f"구성원이 서로 다른 장치에 배정되지 않았다: {devices}")

    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device_count": device_count,
                "requested": requested,
                "fold_seed_offsets": offsets,
                "member_devices": devices,
            }
        )
    )


if __name__ == "__main__":
    main()
