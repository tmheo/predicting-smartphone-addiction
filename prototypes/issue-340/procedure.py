"""#340 원형: 선택 편향 없는 후보 풀 재심사 절차.

절차의 뼈대는 앞선 결정 티켓이 이미 고정했다.

- #344: 유지 증거 규칙, 하향 4단계 조사 순서, 5단 동률 해소, 짝 양쪽의 전략 재선택.
- #347: 동등 대역 하한 하나, 점추정 Δ 판정, 35개 기준 풀 앵커, 궤적 위 전 지점 후보.

이 원형이 확인하려는 것은 그 규칙을 코드로 옮겼을 때의 네 가지다.

1. 누출 경계: 바깥쪽 채점 분할의 어떤 값도 그 분할의 풀·전략·순서 결정에 닿지 않는가.
2. 결정적 재현: 같은 입력이 언제나 같은 산출물을 주는가.
3. 중간 저장: 중간에 끊고 이어받아도 같은 결과가 나오는가.
4. 산출물 모양: #347 11절이 요구한 항목이 대조마다 실제로 남는가.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline import ensemble as ensemble_module  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture import Fixture, MemberLedger  # noqa: E402

# #347이 확정한 단일 전역 하한. 원형은 이 값을 바꾸지 않는다.
EQUIVALENCE_LOWER = -0.000027669802

# 합성 행렬에서도 실행 계약의 기본 평가 범위를 그대로 쓴다.
FULL_STRATEGIES = ensemble_module.DEFAULT_COMBINER_NAMES
# 누출·재현·재개 시험은 절차의 모양만 보면 되므로 싼 부분집합으로 돌린다.
TINY_STRATEGIES = (
    "rank_mean",
    "logit_logistic",
    "ridge_logit",
    "rank_logit_logistic",
    "shrunk_rank_logit_logistic",
)


class ProcedureError(RuntimeError):
    """절차를 시작하기 전에 멈춰야 하는 계약 위반."""


def build_registry(
    fixture: Fixture, names: tuple[str, ...]
) -> dict[str, ensemble_module.Combiner]:
    """등록 전략을 합성 fold·결측 구간으로 다시 만든다.

    실행 계약의 전략 구현을 그대로 쓰되, 실제 `artifacts/folds.parquet`와
    `data/*.csv`를 읽는 두 자리에만 합성 장부를 주입한다.
    주입하지 않으면 합성 id가 실제 장부에 없어 전략이 멈춘다.
    """
    registry: dict[str, ensemble_module.Combiner] = {}
    for name in names:
        if name == "shrunk_rank_logit_logistic":
            registry[name] = ensemble_module.ShrunkRankLogitCombiner(
                fold_of=fixture.fold_of
            )
        elif name == "missing_segmented_rank_logit":
            registry[name] = ensemble_module.MissingnessSegmentedLogisticCombiner(
                band_of=fixture.band_of
            )
        elif name == "missing_interaction_rank_logit":
            registry[name] = ensemble_module.MissingnessInteractionLogisticCombiner(
                band_of=fixture.band_of
            )
        elif name == "missing_4plus_rank_logit":
            registry[name] = ensemble_module.MissingnessSegmentedLogisticCombiner(
                band_of=fixture.band_of,
                specialized_bands=(2,),
                name="missing_4plus_rank_logit",
            )
        else:
            registry[name] = ensemble_module.COMBINER_REGISTRY[name]
    return registry


@dataclass
class FitCounter:
    """결합 전략 적합 횟수와 소요 시간. 실제 규모 비용을 외삽하는 근거다."""

    fits: int = 0
    rows_fitted: int = 0

    def record(self, rows: int) -> None:
        self.fits += 1
        self.rows_fitted += rows


@dataclass(frozen=True)
class StrategyChoice:
    """학습 분할 하나에서 고른 최선 전략과 그 근거."""

    name: str
    auc: float
    strategy_auc: dict[str, float]
    failures: dict[str, str]


@dataclass(frozen=True)
class RemovalRecord:
    """소급 제거 대조 하나. #347 11절이 요구한 항목을 그대로 담는다."""

    step: int
    stage: int
    removed: tuple[str, ...]
    working_members_before: int
    working_members_after: int
    delta_vs_working: float
    delta_vs_anchor: float
    verdict: str  # "제거" 또는 "유지".
    before_strategy: str
    after_strategy: str
    strategy_changed: bool
    fold_delta: dict[str, float]
    negative_folds: int
    fold_total: int


@dataclass
class SplitOutcome:
    """학습 분할 하나에서 절차가 낸 축소안."""

    label: str
    anchor_auc: float
    anchor_strategy: str
    final_pool: tuple[str, ...]
    final_strategy: str
    final_auc: float
    trajectory: list[RemovalRecord] = field(default_factory=list)
    fits: int = 0


def _lofo_prediction(
    combiner: ensemble_module.Combiner,
    preds: pd.DataFrame,
    fold_of: pd.Series,
    y: pd.Series,
    counter: FitCounter,
) -> pd.Series:
    """주어진 행 집합 안에서만 도는 leave-one-fold-out 예측.

    바깥쪽 채점 분할의 행은 `preds`에 아예 들어오지 않는다.
    이 함수가 누출 경계다. 여기서 보이는 행이 결정에 쓰이는 행의 전부다.
    """
    folds = sorted(fold_of.unique())
    if len(folds) < 2:
        raise ProcedureError("학습 분할 안 leave-one-fold-out에는 fold 2개 이상이 필요하다.")
    out = np.full(len(preds), np.nan)
    positions = {fold: (fold_of == fold).to_numpy() for fold in folds}
    for fold in folds:
        validate = positions[fold]
        train = ~validate
        fitted = combiner.fit(preds[train], y[train])
        counter.record(int(train.sum()))
        out[validate] = np.asarray(
            fitted.predict(preds[validate]), dtype=np.float64
        )
    assert not np.isnan(out).any(), "학습 분할의 fold 배정이 전 행을 덮지 않는다."
    return pd.Series(out, index=preds.index, name="prediction")


def score_pool(
    pool: tuple[str, ...],
    preds: pd.DataFrame,
    fold_of: pd.Series,
    y: pd.Series,
    registry: dict[str, ensemble_module.Combiner],
    counter: FitCounter,
) -> tuple[StrategyChoice, dict[int, float]]:
    """학습 분할 안에서 등록 전략 전부를 다시 평가하고 최선을 고른다.

    #344 2절: 짝의 양쪽 모두 학습 분할에서 재선택한다.
    전체 풀에 맞춰 고른 전략을 축소 풀에 그대로 씌우면 축소 풀이 부당하게 불리해진다.
    동률은 등록 순서로 가른다. 결과를 보고 흔들리지 않는 결정적 최후 수단이다.
    """
    block = preds[list(pool)]
    strategy_auc: dict[str, float] = {}
    failures: dict[str, str] = {}
    best_name: str | None = None
    best_auc = -np.inf
    best_fold_auc: dict[int, float] = {}
    for name, combiner in registry.items():
        try:
            prediction = _lofo_prediction(combiner, block, fold_of, y, counter)
        except (ensemble_module.CombinerConvergenceError, ValueError) as exc:
            failures[name] = str(exc)
            continue
        auc = float(roc_auc_score(y.to_numpy(), prediction.to_numpy()))
        strategy_auc[name] = auc
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_fold_auc = {
                int(fold): float(
                    roc_auc_score(
                        y[(fold_of == fold).to_numpy()].to_numpy(),
                        prediction[(fold_of == fold).to_numpy()].to_numpy(),
                    )
                )
                for fold in sorted(fold_of.unique())
            }
    if best_name is None:
        raise ProcedureError("수렴한 등록 전략이 하나도 없다.")
    return (
        StrategyChoice(
            name=best_name,
            auc=best_auc,
            strategy_auc=strategy_auc,
            failures=failures,
        ),
        best_fold_auc,
    )


def exclusion_contributions(
    pool: tuple[str, ...],
    strategy: str,
    preds: pd.DataFrame,
    fold_of: pd.Series,
    y: pd.Series,
    registry: dict[str, ensemble_module.Combiner],
    counter: FitCounter,
) -> dict[str, float]:
    """학습 분할에서 다시 계산한 제외 기여. 조사 순서의 2차 정렬 축이다.

    전략 재선택까지 구성원마다 다시 하면 순서 하나를 정하려고 절차를 통째로 한 번 더
    도는 셈이라, 작업 풀이 고른 전략 하나로 고정해 잰다.
    이 값은 순서에만 쓰고 판정에는 쓰지 않으므로 전략 고정이 판정을 흔들지 않는다.
    """
    combiner = registry[strategy]
    block = preds[list(pool)]
    base = float(
        roc_auc_score(
            y.to_numpy(),
            _lofo_prediction(combiner, block, fold_of, y, counter).to_numpy(),
        )
    )
    contributions: dict[str, float] = {}
    for member in pool:
        rest = [name for name in pool if name != member]
        if not rest:
            contributions[member] = 0.0
            continue
        reduced = float(
            roc_auc_score(
                y.to_numpy(),
                _lofo_prediction(
                    combiner, block[rest], fold_of, y, counter
                ).to_numpy(),
            )
        )
        contributions[member] = base - reduced
    return contributions


def removal_stages(
    pool: tuple[str, ...],
    ledger: dict[str, MemberLedger],
    contributions: dict[str, float],
) -> list[tuple[int, tuple[str, ...]]]:
    """#344 4절의 하향 4단계 조사 순서를 정렬 규칙으로만 만든다.

    이름 목록을 못 박지 않는다.
    구조 축(계보 묶음 크기, 계보 역할, 관점 크기)은 장부의 사실이라 사전 고정할 수
    있고, 성능 축(제외 기여)은 학습 분할에서 다시 계산한 값만 쓴다.
    2단계는 1단계 결과에 달려 있어 궤적을 도는 중에 다시 만든다.
    """
    members = set(pool)
    lineages: dict[str, list[str]] = {}
    perspectives: dict[str, list[str]] = {}
    for config in pool:
        entry = ledger[config]
        lineages.setdefault(entry.lineage, []).append(config)
        perspectives.setdefault(entry.perspective, []).append(config)

    stages: list[tuple[int, tuple[str, ...]]] = []

    # 1단계: 다구성원 모델 계보 묶음 통째 제거, 크기 내림차순.
    multi = [(name, group) for name, group in lineages.items() if len(group) > 1]
    multi.sort(key=lambda item: (-len(item[1]), item[0]))
    for _, group in multi:
        stages.append((1, tuple(sorted(group))))

    # 3단계: 정보 관점 묶음 통째 제거, 구성원 수 오름차순.
    #   큰 관점도 지우지 않고 순서 뒤에 등재해 앞 단계가 많이 통과하면 도달하게 남긴다.
    views = sorted(perspectives.items(), key=lambda item: (len(item[1]), item[0]))
    for _, group in views:
        if len(group) == len(members):
            continue  # 풀 전체를 비우는 대조는 실질 후보가 아니다.
        stages.append((3, tuple(sorted(group))))

    # 4단계: 남은 전체 개별 훑기, 학습 분할 제외 기여 오름차순.
    singles = sorted(pool, key=lambda config: (contributions[config], config))
    for config in singles:
        stages.append((4, (config,)))
    return stages


def lineage_members_in_order(
    group: tuple[str, ...], ledger: dict[str, MemberLedger]
) -> list[str]:
    """모델 계보 묶음 안에서는 이전판부터 조사한다."""
    return sorted(group, key=lambda config: (ledger[config].generation, config))


def _fold_deltas(
    before: dict[int, float], after: dict[int, float]
) -> tuple[dict[str, float], int]:
    """fold별 Δ와 음수 fold 수. 기록만 하고 판정을 뒤집지 않는다. (#347 5절)

    키를 문자열로 두는 이유는 중간 저장을 거친 재개 결과와 처음부터 돈 결과의 산출물이
    글자 단위로 같아야 재현 검사가 성립하기 때문이다.
    """
    deltas = {str(fold): after[fold] - before[fold] for fold in sorted(before)}
    return deltas, sum(1 for value in deltas.values() if value < 0.0)


def run_split(
    label: str,
    pool: tuple[str, ...],
    preds: pd.DataFrame,
    fold_of: pd.Series,
    y: pd.Series,
    registry: dict[str, ensemble_module.Combiner],
    ledger: dict[str, MemberLedger],
    counter: FitCounter,
    progress: Callable[[str], None] | None = None,
) -> SplitOutcome:
    """학습 분할 하나에서 순차 제거 궤적을 끝까지 돌린다.

    앵커는 동결한 기준 풀이고 단계 판정은 직전 작업 풀 대비다. (#344 2절, #347 7절)
    궤적 위 모든 지점을 후보로 보되, 유지 판정은 작업 풀을 그대로 두고 다음 대조로 간다.
    """
    started_fits = counter.fits
    anchor_choice, anchor_fold_auc = score_pool(
        pool, preds, fold_of, y, registry, counter
    )
    working = tuple(pool)
    working_choice = anchor_choice
    working_fold_auc = anchor_fold_auc

    contributions = exclusion_contributions(
        working, working_choice.name, preds, fold_of, y, registry, counter
    )
    stages = removal_stages(working, ledger, contributions)

    trajectory: list[RemovalRecord] = []
    step = 0

    def attempt(stage: int, removal: tuple[str, ...]) -> str:
        """대조 하나를 재고 판정한다. 제거면 작업 풀을 옮기고, 유지면 그대로 둔다."""
        nonlocal step, working, working_choice, working_fold_auc
        removal = tuple(name for name in removal if name in working)
        if not removal or len(removal) >= len(working):
            return "생략"
        candidate = tuple(name for name in working if name not in removal)
        choice, fold_auc = score_pool(candidate, preds, fold_of, y, registry, counter)
        delta_vs_working = choice.auc - working_choice.auc
        verdict = "유지" if delta_vs_working < EQUIVALENCE_LOWER else "제거"
        fold_delta, negative = _fold_deltas(working_fold_auc, fold_auc)
        step += 1
        trajectory.append(
            RemovalRecord(
                step=step,
                stage=stage,
                removed=removal,
                working_members_before=len(working),
                working_members_after=len(candidate),
                delta_vs_working=delta_vs_working,
                delta_vs_anchor=choice.auc - anchor_choice.auc,
                verdict=verdict,
                before_strategy=working_choice.name,
                after_strategy=choice.name,
                strategy_changed=choice.name != working_choice.name,
                fold_delta=fold_delta,
                negative_folds=negative,
                fold_total=len(fold_delta),
            )
        )
        if progress is not None:
            progress(
                f"{label} step {step} stage {stage} "
                f"제거 {'+'.join(removal)} → {verdict} Δ={delta_vs_working:+.12f}"
            )
        if verdict == "제거":
            working = candidate
            working_choice = choice
            working_fold_auc = fold_auc
        return verdict

    # 단계는 하향 계층이다. 큰 단위를 먼저 시도하고 유지가 나면 그때 쪼갠다. (#344 4절)
    kept_lineages: list[tuple[str, ...]] = []
    for stage, removal in stages:
        if stage != 1:
            continue
        if attempt(1, removal) == "유지":
            kept_lineages.append(removal)
    for group in kept_lineages:
        for member in lineage_members_in_order(group, ledger):
            attempt(2, (member,))
    for stage, removal in stages:
        if stage == 3:
            attempt(3, removal)
    for stage, removal in stages:
        if stage == 4:
            attempt(4, removal)

    return SplitOutcome(
        label=label,
        anchor_auc=anchor_choice.auc,
        anchor_strategy=anchor_choice.name,
        final_pool=working,
        final_strategy=working_choice.name,
        final_auc=working_choice.auc,
        trajectory=trajectory,
        fits=counter.fits - started_fits,
    )


@dataclass
class ProcedureResult:
    """절차 한 번의 전체 산출물."""

    outer: dict[int, SplitOutcome]
    nested_auc: float
    anchor_nested_auc: float
    final_run: SplitOutcome
    fits: int
    rows_fitted: int

    def decision_payload(self) -> dict[str, Any]:
        """재개해도 반드시 같아야 하는 결정 산출물.

        실행 계측(적합 횟수, 소요 시간)은 여기 들어가지 않는다.
        이어받은 실행은 이미 끝난 단위를 다시 적합하지 않으므로 계측은 당연히 다르고,
        그것을 재현 검사에 섞으면 정상 재개가 실패로 보인다.
        """
        payload = self.payload()
        payload.pop("cost")
        for block in [payload["final_run"], *payload["outer"].values()]:
            block.pop("fits")
        return payload

    def payload(self) -> dict[str, Any]:
        return {
            "equivalence_lower": EQUIVALENCE_LOWER,
            "nested_auc": self.nested_auc,
            "anchor_nested_auc": self.anchor_nested_auc,
            "nested_delta_vs_anchor": self.nested_auc - self.anchor_nested_auc,
            "outer": {
                str(fold): _split_payload(outcome)
                for fold, outcome in sorted(self.outer.items())
            },
            "final_run": _split_payload(self.final_run),
            "cost": {"fits": self.fits, "rows_fitted": self.rows_fitted},
        }


def _split_payload(outcome: SplitOutcome) -> dict[str, Any]:
    return {
        "label": outcome.label,
        "anchor_auc": outcome.anchor_auc,
        "anchor_strategy": outcome.anchor_strategy,
        "final_pool": list(outcome.final_pool),
        "final_members": len(outcome.final_pool),
        "final_strategy": outcome.final_strategy,
        "final_auc": outcome.final_auc,
        "fits": outcome.fits,
        "trajectory": [asdict(record) for record in outcome.trajectory],
    }


def run_procedure(
    fixture: Fixture,
    registry: dict[str, ensemble_module.Combiner],
    *,
    checkpoint: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> ProcedureResult:
    """바깥쪽 검증 5분할 + 전체 OOF 최종 실행.

    바깥쪽 분할 k의 결정은 fold k 행을 한 줄도 보지 않은 채 끝난다.
    fold k가 등장하는 자리는 고른 풀과 전략을 학습 분할 전체로 적합해 예측할 때뿐이다.
    전체 OOF 최종 실행은 채점이 아니라 인수할 고정 풀을 만드는 자리다.
    """
    counter = FitCounter()
    pool = tuple(fixture.configs)
    done = _load_checkpoint(checkpoint)

    outer: dict[int, SplitOutcome] = {}
    nested = pd.Series(np.nan, index=fixture.preds.index, dtype=np.float64)
    anchor_nested = pd.Series(np.nan, index=fixture.preds.index, dtype=np.float64)

    for fold in sorted(fixture.fold_of.unique()):
        key = f"outer-{int(fold)}"
        train = (fixture.fold_of != fold).to_numpy()
        score = (fixture.fold_of == fold).to_numpy()
        if key in done:
            outcome = _restore_split(done[key]["split"])
            nested_block = pd.Series(
                done[key]["prediction"], index=fixture.preds.index[score]
            )
            anchor_block = pd.Series(
                done[key]["anchor_prediction"], index=fixture.preds.index[score]
            )
        else:
            outcome = run_split(
                key,
                pool,
                fixture.preds[train],
                fixture.fold_of[train],
                fixture.y[train],
                registry,
                fixture.ledger,
                counter,
                progress,
            )
            nested_block = _fit_and_predict(
                registry[outcome.final_strategy],
                fixture.preds.loc[train, list(outcome.final_pool)],
                fixture.y[train],
                fixture.preds.loc[score, list(outcome.final_pool)],
                counter,
            )
            anchor_block = _fit_and_predict(
                registry[outcome.anchor_strategy],
                fixture.preds.loc[train, list(pool)],
                fixture.y[train],
                fixture.preds.loc[score, list(pool)],
                counter,
            )
            _append_checkpoint(
                checkpoint,
                key,
                {
                    "split": _split_payload(outcome),
                    "prediction": [float(value) for value in nested_block],
                    "anchor_prediction": [float(value) for value in anchor_block],
                },
            )
        outer[int(fold)] = outcome
        nested.loc[nested_block.index] = nested_block.to_numpy()
        anchor_nested.loc[anchor_block.index] = anchor_block.to_numpy()

    assert not nested.isna().any(), "바깥쪽 분할이 전 행을 덮지 않는다."

    final_key = "final"
    if final_key in done:
        final_run = _restore_split(done[final_key]["split"])
    else:
        final_run = run_split(
            final_key,
            pool,
            fixture.preds,
            fixture.fold_of,
            fixture.y,
            registry,
            fixture.ledger,
            counter,
            progress,
        )
        _append_checkpoint(
            checkpoint, final_key, {"split": _split_payload(final_run)}
        )

    return ProcedureResult(
        outer=outer,
        nested_auc=float(roc_auc_score(fixture.y.to_numpy(), nested.to_numpy())),
        anchor_nested_auc=float(
            roc_auc_score(fixture.y.to_numpy(), anchor_nested.to_numpy())
        ),
        final_run=final_run,
        fits=counter.fits,
        rows_fitted=counter.rows_fitted,
    )


def _fit_and_predict(
    combiner: ensemble_module.Combiner,
    train_preds: pd.DataFrame,
    y: pd.Series,
    score_preds: pd.DataFrame,
    counter: FitCounter,
) -> pd.Series:
    fitted = combiner.fit(train_preds.astype(np.float64), y)
    counter.record(len(train_preds))
    return pd.Series(
        np.asarray(fitted.predict(score_preds.astype(np.float64)), dtype=np.float64),
        index=score_preds.index,
    )


def _load_checkpoint(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        done[record["key"]] = record["value"]
    return done


def _append_checkpoint(path: Path | None, key: str, value: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"key": key, "value": value}, ensure_ascii=False, sort_keys=True)
            + "\n"
        )


def _restore_split(payload: dict[str, Any]) -> SplitOutcome:
    return SplitOutcome(
        label=payload["label"],
        anchor_auc=payload["anchor_auc"],
        anchor_strategy=payload["anchor_strategy"],
        final_pool=tuple(payload["final_pool"]),
        final_strategy=payload["final_strategy"],
        final_auc=payload["final_auc"],
        trajectory=[RemovalRecord(**record) for record in payload["trajectory"]],
        fits=payload["fits"],
    )
