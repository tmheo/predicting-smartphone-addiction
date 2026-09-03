"""판정 회차 스펙의 구성원 출처 스모크. (#632)

스펙 스크립트를 불러 RoundSpec 조립(선언 계약 검증)을 통과시키고, 기준 팔(재현
등급일 때)과 평가 팔 전부를 hash-verified로 적재해 팔별 구성원 수와 구성 해시를
찍는다. run-logs/에 아무것도 쓰지 않으므로 precommit 전에 몇 번이든 돌릴 수 있다.
--replay-fold를 주면 기준 팔을 그 분할 하나로 다시 결합해 자기 검사 기대값(AUC,
예측 해시)과 대조한다. 기준값이 이 코드 상태에서 재현되는지 precommit 전에 확인하는 용도다.

    uv run python scripts/round_members_smoke.py scripts/round_issue624_own36.py --replay-fold 0
    uv run python scripts/round_members_smoke.py scripts/round_issue624_ext314.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ensemble
from pipeline.data import ID, labels as load_labels
from pipeline.members import HASH_VERIFIED, MembersError, load_members
from pipeline.round import RoundSpec
from pipeline.runs import MlflowRunStore
from pipeline.sealed import canonical_sha256


def load_spec(script: Path) -> RoundSpec:
    module_spec = importlib.util.spec_from_file_location(script.stem, script)
    if module_spec is None or module_spec.loader is None:
        raise SystemExit(f"스펙 스크립트를 불러올 수 없다: {script}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = getattr(module, "SPEC", None)
    if not isinstance(spec, RoundSpec):
        raise SystemExit(f"{script}에 RoundSpec SPEC이 없다.")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="판정 회차 스펙 구성원 출처 스모크 (#632)")
    parser.add_argument("script", type=Path, help="RoundSpec SPEC을 가진 스펙 스크립트")
    parser.add_argument("--replay-fold", type=int, default=None, help="기준 팔을 이 분할로 재현해 자기 검사 기대값과 대조한다")
    args = parser.parse_args()

    spec = load_spec(args.script)
    print(f"회차 {spec.round_id}: 결합기 {spec.combiner}, 자기 검사 {spec.selfcheck.grade}")
    for path in (spec.folds_path, spec.train_path, *spec.sealed_inputs):
        if not Path(path).is_file():
            raise SystemExit(f"봉인할 입력 파일이 없다: {path}")
    frame = pd.read_parquet(spec.folds_path)
    fold_of = frame.set_index(ID)["fold"]
    y = load_labels(fold_of.index, train_path=spec.train_path)
    store = MlflowRunStore()

    arms = [(arm.name, arm.source) for arm in spec.candidates]
    if spec.reference.source is not None:
        arms.insert(0, (spec.reference.name, spec.reference.source))
    reference_matrix = None
    for name, source in arms:
        started = time.monotonic()
        try:
            matrix = load_members(source, fold_of.index, store, labels=y)
            matrix.require(HASH_VERIFIED)
        except MembersError as exc:
            raise SystemExit(f"{name}: 구성원 출처 검증 실패: {exc}") from exc
        rows = matrix.members
        composition = canonical_sha256([[m, h] for m, h in zip(rows["member_id"], rows["oof_sha256"], strict=True)])
        levels = sorted(set(rows["verification"]))
        aucs = rows["rescored_auc"].to_numpy(np.float64)
        print(
            f"- {name}: 구성원 {len(rows)}개 {levels} 구성 해시 {composition[:16]}… "
            f"단독 AUC {aucs.min():.6f}~{aucs.max():.6f}, {time.monotonic() - started:.0f}초"
        )
        if name == spec.reference.name:
            reference_matrix = matrix
        else:
            del matrix

    if args.replay_fold is None:
        return
    if reference_matrix is None:
        raise SystemExit("기준 팔 구성원 출처가 없어 재현할 수 없다(해시 동일성 등급).")
    expected = spec.selfcheck.expected.get(args.replay_fold)
    if expected is None:
        raise SystemExit(f"자기 검사 기대값에 분할 {args.replay_fold}이 없다.")
    combiner = ensemble.combiner_for_context(spec.combiner, fold_of=fold_of, band_of=None)
    started = time.monotonic()
    outcome = ensemble.evaluate_outer_fold(combiner, reference_matrix.oof_frame(), fold_of, y, args.replay_fold)
    auc_ok = outcome.auc == expected.auc
    hash_ok = expected.prediction_sha256 is None or outcome.prediction_identity == expected.prediction_sha256
    print(
        f"- 기준 팔 분할 {args.replay_fold} 재현: AUC {outcome.auc!r} (기대 {expected.auc!r}) "
        f"{'일치' if auc_ok else '불일치'}, 예측 해시 {outcome.prediction_identity[:16]}… "
        f"{'대조 생략' if expected.prediction_sha256 is None else ('일치' if hash_ok else '불일치')}, "
        f"{time.monotonic() - started:.0f}초"
    )
    if not (auc_ok and hash_ok):
        sys.exit("기준 팔 재현이 자기 검사 기대값과 다르다. precommit 전에 기준값 출처를 다시 본다.")


if __name__ == "__main__":
    main()
