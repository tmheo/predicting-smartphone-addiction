"""champion 용량 하향과 학습률 하향 fold 0 짝비교. (#397)

champion `exp131_lookup_bivariate_plr5`에서 한 번에 한 축만 아래로 내린 후보를
fold 0, seed 42로 겨루게 한다.

#108이 champion 계열의 용량을 잰 것은 `exp075` d_model 128 → 192와 `exp076`
layers 4 → 6으로 **증가 방향 두 점뿐**이다. 규제만 양방향이었고 `d_model` 96과
`layers` 3은 이 계열에서 실행된 적이 없다. #385는 학습률 격자의 위쪽만 봤으므로
`2e-3`이 봉우리인지 오른쪽 어깨인지도 측정되지 않았다. 이 스크립트가 그 두 빈칸을
한 배치로 메운다.

이 모델에는 임베딩 차원이라는 별도 눈금이 없어 `d_model` 하나가 lookup 임베딩,
PLR 출력, 토큰, Transformer 폭을 동시에 지정한다. 파라미터의 절반 가까이가
임베딩 표라서 `layers` 하향(임베딩을 안 건드리는 순수 용량 하향)과 `d_model`
하향(정확값 기억 용량까지 깎는 하향)은 성격이 다른 실험이다. 그래서 후보마다
임베딩 표와 본체의 파라미터 수를 갈라 기록해 축별로 읽을 수 있게 한다.

선별 전용 약식 검증이며 `scripts/screen_muon_lr.py`(#385)의 구조를 그대로 따른다.
커밋된 folds.parquet의 fold 0 하나만 검증에 쓰고 나머지 4개 fold 전체 행으로
학습한다. 행 표본 축소는 하지 않는다. fold 0의 피처 행렬은 한 번만 만들고 모든
후보가 공유하므로 후보 사이의 유일한 차이가 바꾼 축 하나가 된다.

사용법:
    uv run python scripts/screen_capacity_lr_down.py \\
        --candidates base layers=3 d_model=96 lr=1e-3 lr=1.5e-3

후보는 `축=값` 꼴이고 `base`는 아무 축도 바꾸지 않은 champion 재현이다. 한 후보가
바꿀 수 있는 축은 하나뿐이다. 결과는 `--out` 경로에 JSON Lines로 증분 기록한다.
이미 기록된 후보는 건너뛰므로 중단한 실행을 그대로 다시 시작하면 남은 후보부터
이어 달린다.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import data  # noqa: E402
from pipeline import model as model_mod  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.plan import FeaturePlan, prepare_fold_fit_input  # noqa: E402

CONFIG = "configs/exp131_lookup_bivariate_plr5.yaml"
SEED = 42
VALID_FOLD = 0
# champion run 54acd002의 저장된 oof_seed_42.parquet을 fold 0에서 다시 채점한 값.
CHAMPION_FOLD0_AUC = 0.9686518
# 같은 실행의 fold 0 시드 3개(42/43/44) 값의 폭. 사전 등록 관문의 눈금이다.
CHAMPION_FOLD0_SEED_SPREAD = 0.0000446
# champion 원본 학습 일정. #382가 12·16·20·24를 전부 잡음 안으로 판정했으므로
# 이 축의 기준값을 그대로 쓴다.
CHAMPION_EPOCHS = 32
# 이 배치가 바꿀 수 있는 축과 그 값을 읽는 방법. 다른 이름은 받지 않는다.
AXES: dict[str, type] = {"layers": int, "d_model": int, "lr": float}


@dataclasses.dataclass(frozen=True)
class Candidate:
    """후보 하나. 기준 후보는 아무 축도 바꾸지 않은 champion 재현이다."""

    axis: str | None = None
    value: object = None

    @property
    def key(self) -> str:
        if self.axis is None:
            return "base"
        return f"{self.axis}={self.value:g}"

    @property
    def overrides(self) -> dict[str, object]:
        return {} if self.axis is None else {self.axis: self.value}


def parse_candidate(token: str) -> Candidate:
    if token == "base":
        return Candidate()
    axis, separator, value_text = token.partition("=")
    if not separator or axis not in AXES:
        raise argparse.ArgumentTypeError(
            f"후보는 base 또는 {sorted(AXES)} 중 한 축의 `축=값`이어야 한다: {token!r}"
        )
    try:
        value = AXES[axis](value_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{axis} 값을 읽을 수 없다: {token!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{axis} 값은 양수여야 한다: {token!r}")
    return Candidate(axis=axis, value=value)


def build_fold0_matrices(cfg, plan: FeaturePlan, smoke_rows: int | None = None):
    """fold 0의 학습·검증 행렬을 정식 경로와 같은 순서로 만든다."""
    train = data.load_csv(cfg.data.train)
    test = data.load_csv(cfg.data.test)
    if smoke_rows:
        # 연결 점검 전용. 판정에 쓰는 실행에서는 절대 켜지 않는다.
        train = train.head(smoke_rows).copy()
        test = test.head(smoke_rows).copy()
    data.align_categories(train, test, cfg.features.categorical)
    train, test = plan.apply_dataset_wide(train, test)
    train = data.attach_folds(train, cfg.data.folds)

    y = train[data.TARGET]
    va_idx = train.index[train["fold"] == VALID_FOLD]
    tr_idx = train.index[train["fold"] != VALID_FOLD]

    X = plan.build_matrix(train, SEED)
    X_test = plan.build_matrix(test, SEED)

    providers = plan.new_fold_fit_providers()
    X_fold, X_test_fold = X, X_test
    if providers:
        train_ff = prepare_fold_fit_input(train, X)
        test_ff = prepare_fold_fit_input(test, X_test)
        for kind, transformer in providers:
            started = time.time()
            train_values, test_values, _ = plan.materialize_fold_fit_provider(
                kind=kind,
                transformer=transformer,
                train_input=train_ff,
                test_input=test_ff,
                training_index=tr_idx,
                validation_index=va_idx,
                seed=SEED,
                fold=VALID_FOLD,
                recorder=None,
            )
            collision = set(train_values.columns) & set(X_fold.columns)
            if collision:
                raise AssertionError(f"fold-fit 컬럼 이름 충돌: {sorted(collision)}")
            X_fold = pd.concat([X_fold, train_values], axis=1)
            X_test_fold = pd.concat([X_test_fold, test_values], axis=1)
            print(
                f"[fold-fit] {kind} {time.time() - started:.0f}s "
                f"컬럼 {len(train_values.columns)}개",
                flush=True,
            )

    feature_names = plan.all_columns()
    assert list(X_fold.columns) == feature_names, "fold 0 컬럼 집합이 피처 계획과 다르다."
    assert list(X_test_fold.columns) == feature_names, "test 컬럼 집합이 피처 계획과 다르다."
    return X_fold, X_test_fold, y, tr_idx, va_idx


def summarize_members(diagnostics: dict | None) -> dict:
    """구성원별 정점 시점과 기울기 관측을 판독용으로 요약한다.

    #382는 다섯 일정에서, #385는 다섯 학습률에서 정점 epoch가 9에 고정된다고
    관측했다. 용량을 내렸을 때도 그 자리에 머무는지가 이 티켓의 판독 항목이다.
    기울기 제한 적용 비율은 #161이 고학습률 불안정을 판별한 축과 같다.
    """
    if not diagnostics:
        return {}
    members = diagnostics.get("fold_initialization_members") or []
    best_epochs: list[object] = []
    end_epochs: list[object] = []
    best_aucs: list[object] = []
    peak_lr_fractions: list[float] = []
    gradient_norm_means: list[float] = []
    clip_fractions: list[float] = []
    for member in members:
        if not member:
            continue
        best_epochs.append(member.get("best_epoch"))
        end_epochs.append(member.get("end_epoch"))
        best_aucs.append(member.get("best_validation_auc"))
        evaluations = member.get("evaluations") or []
        max_lr = member.get("max_learning_rate")
        best_epoch = member.get("best_epoch")
        for observation in evaluations:
            if observation.get("epoch") == best_epoch and max_lr:
                peak_lr_fractions.append(
                    float(observation["learning_rate"]) / float(max_lr)
                )
        if evaluations:
            gradient_norm_means.append(
                sum(float(o["gradient_norm_mean"]) for o in evaluations)
                / len(evaluations)
            )
            clip_fractions.append(
                max(float(o["gradient_clip_fraction"]) for o in evaluations)
            )
    return {
        "member_best_epochs": best_epochs,
        "member_end_epochs": end_epochs,
        "member_best_validation_aucs": best_aucs,
        "member_peak_lr_fraction_of_max": peak_lr_fractions,
        "member_gradient_norm_mean": gradient_norm_means,
        "member_gradient_clip_fraction_max": clip_fractions,
    }


def count_parameters(adapter) -> dict:
    """임베딩 표와 Transformer 본체의 파라미터 수를 갈라 센다.

    `d_model`은 임베딩 표와 본체를 동시에 움직이고 `layers`는 본체만 움직인다.
    두 축의 성격 차이를 점수와 나란히 읽으려면 이 구분이 필요하다. fold의 세
    구성원은 모양이 같으므로 첫 구성원 하나만 센다.
    """
    members = getattr(adapter, "_members", None) or []
    model = next((getattr(m, "_model", None) for m in members), None)
    if model is None:
        return {}
    embedding = 0
    encoder = 0
    total = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        root = name.split(".", 1)[0]
        if root == "emb":
            embedding += count
        elif root == "tr":
            encoder += count
    return {
        "embedding_parameters": embedding,
        "encoder_parameters": encoder,
        "total_parameters": total,
    }


def run_candidate(
    cfg,
    X_fold,
    X_test_fold,
    y,
    tr_idx,
    va_idx,
    candidate: Candidate,
    epochs: int,
) -> dict:
    params = {**cfg.model.params, "epochs": epochs, **candidate.overrides}
    model_cfg = dataclasses.replace(cfg.model, params=params)
    adapter = model_mod.create(model_cfg, SEED)
    model_mod.set_dataset_reference(adapter, X_fold, X_test_fold)
    started = time.time()
    va_pred = adapter.fit(
        X_fold.loc[tr_idx],
        y.loc[tr_idx],
        X_fold.loc[va_idx],
        y.loc[va_idx],
        None,
        None,
    )
    elapsed = time.time() - started
    auc = float(roc_auc_score(y.loc[va_idx], va_pred))
    diagnostics = model_mod.collect_training_diagnostics(adapter)
    return {
        "candidate": candidate.key,
        "axis": candidate.axis or "none",
        "layers": int(params.get("layers", 4)),
        "d_model": int(params.get("d_model", 128)),
        "lr": float(params.get("lr", 2e-3)),
        "epochs": epochs,
        "seed": SEED,
        "fold": VALID_FOLD,
        "auc": auc,
        "diff_vs_champion": auc - CHAMPION_FOLD0_AUC,
        "fit_seconds": round(elapsed, 1),
        **count_parameters(adapter),
        **summarize_members(diagnostics),
        "model_training_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="용량·학습률 하향 fold 0 짝비교 (#397)")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument(
        "--candidates",
        type=parse_candidate,
        nargs="+",
        default=[
            parse_candidate(t)
            for t in ("base", "layers=3", "d_model=96", "lr=1e-3", "lr=1.5e-3")
        ],
        help="후보 목록. `축=값` 꼴이고 `base`는 champion 재현이다.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CHAMPION_EPOCHS,
        help="학습 일정 길이. 기본값은 champion 원본이다.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("run-logs/capacity-lr-down-fold0.jsonl"),
        help="결과 JSON Lines 경로. 기본값은 커밋하지 않는 run-logs/ 아래다.",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        help="연결 점검 전용. train/test 앞쪽 N행만 쓴다. 판정 실행에서는 쓰지 않는다.",
    )
    args = parser.parse_args()
    if args.smoke_rows:
        print("!! 연결 점검 모드: 이 결과는 판정에 쓰지 않는다.", flush=True)

    cfg = load_config(args.config, "screen")
    plan = FeaturePlan.from_config(cfg.features)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done.add((record["candidate"], int(record["epochs"])))
    todo = [c for c in args.candidates if (c.key, args.epochs) not in done]
    if done:
        print(
            f"이미 기록됨: {sorted(done)} | 남은 후보: {[c.key for c in todo]}",
            flush=True,
        )
    if not todo:
        print("남은 후보 없음.", flush=True)
        return

    started = time.time()
    X_fold, X_test_fold, y, tr_idx, va_idx = build_fold0_matrices(
        cfg, plan, args.smoke_rows
    )
    print(
        f"[features] fold 0 행렬 완료 {time.time() - started:.0f}s "
        f"학습 {len(tr_idx)}행 검증 {len(va_idx)}행 컬럼 {X_fold.shape[1]}개",
        flush=True,
    )

    for candidate in todo:
        record = run_candidate(
            cfg, X_fold, X_test_fold, y, tr_idx, va_idx, candidate, args.epochs
        )
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[result] {record['candidate']} auc={record['auc']:.7f} "
            f"짝차이={record['diff_vs_champion']:+.7f} "
            f"정점epoch={record.get('member_best_epochs')} "
            f"파라미터={record.get('total_parameters')} "
            f"(임베딩 {record.get('embedding_parameters')}, "
            f"본체 {record.get('encoder_parameters')}) "
            f"제한비율max={record.get('member_gradient_clip_fraction_max')} "
            f"({record['fit_seconds']:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
