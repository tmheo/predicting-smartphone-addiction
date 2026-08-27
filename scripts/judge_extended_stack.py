"""자체 35개 풀과 검증 외부 구성원의 확장 스택을 nested OOF 사다리로 판정한다. (#443)

지도 #441의 두 번째 최종 제출 후보를 고르는 판정이다. 사다리는 결과 확인 전에 고정한다.

1. own35        - 자체 후보 풀 35구성원만
2. own35_ext85  - 자체 35 + #386 재현가능 85 (기준 재현 계열)
3. own35_ext209 - 자체 35 + #442 장부 통과 209
4. ext209       - 장부 통과 209만 (자체 풀의 한계 기여 확인)

기준 재현 검사 `repro_own32_ext85`는 #386의 합본 117(자체 32 + 외부 85)을 그대로 다시
재서 `0.9700014`와 잡음 바닥 `5.7e-06` 안에서 맞는지 본다. 절제 구성(`ablate_*`)은
own35_ext209에서 장부 `caveats` 부류 하나씩을 뺀 해석용 구성이다.

등록 문턱은 35개 풀 nested `0.9698106`(run-logs/issue337) 대비 `+0.0001` 이상이고,
바깥쪽 검증 분할 5개 전부에서 own35 최선 전략보다 높아야 한다.

이 판정은 **읽기 전용**이다. 외부 예측을 `artifacts/pool.yaml`에 넣지 않고 champion
판정에도 쓰지 않으며 MLflow 실행도 만들지 않는다.

사용법:
    uv run python scripts/judge_extended_stack.py --prepare
    uv run python scripts/judge_extended_stack.py --list-jobs
    uv run python scripts/judge_extended_stack.py --config own35_ext209 --only shrunk_rank_logit_logistic
    uv run python scripts/judge_extended_stack.py --report

작업 하나는 (구성, 전략) 한 쌍이고 산출물은 run-logs/issue443/parts/<구성>__<전략>.json이다.
이미 있는 산출물은 건너뛰므로 중단 뒤 같은 명령으로 이어 달릴 수 있다.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagnose_external94_width as ext94

from pipeline import ensemble
from pipeline.data import ID, TRAIN_PATH, labels
from pipeline.judgment import (
    FOLDS_PATH,
    MISSINGNESS_TEST_PATH,
    missingness_reweighting,
)
from pipeline.ledger import Pool
from pipeline.runs import MlflowRunStore

ISSUE = 443
LEDGER_PATH = Path("docs/research/external-member-ledger.json")
BASELINE_EVALUATION_PATH = Path("run-logs/issue337/ensemble-evaluation.json")
EXT94_EVIDENCE_PATH = Path("docs/research/external94-width-evidence.json")
OUT_DIR = Path("run-logs/issue443")
PARTS_DIR = OUT_DIR / "parts"
CACHE_DIR = OUT_DIR / "cache"
EVIDENCE_PATH = Path("docs/research/extended-stack-ladder-evidence.json")

N_TRAIN = 691369
# 지도 #441의 등록 문턱. 35개 풀 nested는 run-logs/issue337에서 읽어 이 값과 대조한다.
POOL35_NESTED = 0.9698105828357245
GATE_DELTA = 0.0001
# #386 합본 117의 기준값과 결합 도구 잡음 바닥.
EXT85_UNION_REFERENCE = 0.9700014
NOISE_FLOOR = 5.7e-06
# #386 자체 32구성원 가운데 현재 풀에 없는 구성원. 기준 재현에만 쓴다.
OWN32_ONLY_MEMBER = ("exp127_lookup_muon", "7124425b5b51421dbbeba597229554da")

DEFAULT_STRATEGIES = ("shrunk_rank_logit_logistic", "rank_logit_logistic")
LADDER = ("own35", "own35_ext85", "own35_ext209", "ext209")
REPRO = "repro_own32_ext85"
ABLATIONS = {
    "ablate_float32": "float32 저장",
    "ablate_weak50": "레시피·모델 종류 비공개",
    "ablate_te_leak": "원 노트북이 전체 자료 TE를 쓴 판을 그대로 실행",
    "ablate_author_statement": None,  # fold_evidence == author_statement
}
CONFIG_NAMES = LADDER + (REPRO,) + tuple(ABLATIONS)

# 오래 걸리는 전략을 먼저 배치해 병렬 작업자의 꼬리를 줄인다.
STRATEGY_PRIORITY = (
    "greedy_rank_mean",
    "shrunk_rank_logit_logistic",
    "missing_interaction_rank_logit",
    "missing_segmented_rank_logit",
    "missing_4plus_rank_logit",
    "rank_logit_logistic",
    "rank_gauss_logistic",
    "rank_logistic",
    "xgb_rank_logit",
    "nnls_rank",
    "performance_weighted_rank_mean",
    "logit_logistic",
    "nnls_logit",
    "rank_mean",
    "ridge_logit_alpha_0p01",
    "ridge_logit_alpha_0p1",
    "ridge_logit",
    "ridge_logit_alpha_10",
    "ridge_logit_alpha_100",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.lru_cache(maxsize=8)
def _load_npy(path: str) -> np.ndarray:
    return np.load(path)


def load_ledger_array(spec: str) -> np.ndarray:
    """장부의 `경로`, `경로[열]`(parquet), `경로[:, i]`(npy 행렬) 표기를 읽는다."""
    match = re.fullmatch(r"(.+?)\[(.+)\]", spec)
    if match is None:
        values = np.load(spec).astype(np.float64)
        assert values.ndim == 1 or (values.ndim == 2 and values.shape[1] == 1), (
            values.shape
        )
        return values.reshape(-1)
    path, selector = match.groups()
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=[selector])[selector].to_numpy(np.float64)
    column = int(selector.split(",")[1])
    return _load_npy(path)[:, column].astype(np.float64)


def load_ledger() -> tuple[dict, list[dict]]:
    ledger = json.loads(LEDGER_PATH.read_text())
    accepted = [row for row in ledger["members"] if row["status"] == "accepted"]
    assert len(accepted) == ledger["summary"]["accepted_count"] == 209, len(accepted)
    return ledger, accepted


def build_ext209(fold_of: pd.Series, y: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    """장부 통과 209구성원의 OOF 행렬. 장부 순서를 열 순서로 쓴다."""
    _, accepted = load_ledger()
    label_values = y.to_numpy()
    columns: dict[str, np.ndarray] = {}
    checks: list[dict] = []
    for row in accepted:
        values = load_ledger_array(row["oof_path"])
        assert len(values) == N_TRAIN, f"{row['member_id']}: 행 수 {len(values)}"
        assert np.isfinite(values).all(), f"{row['member_id']}: 비유한값"
        auc = float(roc_auc_score(label_values, values))
        delta = auc - float(row["auc"])
        assert abs(delta) < 1e-9, f"{row['member_id']}: 장부 AUC와 {delta:+.2e} 차이"
        key = f"ext_{row['member_id']}"
        assert key not in columns, key
        columns[key] = values
        checks.append(
            {"member_id": row["member_id"], "auc": auc, "ledger_auc_delta": delta}
        )
    matrix = pd.DataFrame(columns, index=fold_of.index).astype(np.float64)
    assert matrix.shape[1] == 209
    return matrix, checks


def build_ext85(fold_of: pd.Series, y: pd.Series) -> tuple[pd.DataFrame, dict]:
    """#386 재현가능 85구성원. 열 이름과 순서까지 #386 도구를 그대로 쓴다."""
    external, checks = ext94.load_external(fold_of, y)
    verification = ext94.report_verification(checks)
    drop = [f"ext_{name}" for name in ext94.UNVERIFIABLE]
    external85 = external.drop(columns=drop)
    assert external85.shape[1] == 85
    return external85, verification


def prepare(fold_of: pd.Series, y: pd.Series) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ext85, verification85 = build_ext85(fold_of, y)
    ext85.to_parquet(CACHE_DIR / "ext85.parquet")
    ext209, checks209 = build_ext209(fold_of, y)
    ext209.to_parquet(CACHE_DIR / "ext209.parquet")
    overlap = [
        c
        for c in ext209.columns
        if c.split(":")[-1] in {c85[4:] for c85 in ext85.columns}
    ]
    payload = {
        "ledger_sha256": _sha256(LEDGER_PATH),
        "ext85": {
            "member_count": int(ext85.shape[1]),
            "members": list(ext85.columns),
            "verification": {k: v for k, v in verification85.items() if k != "members"},
        },
        "ext209": {
            "member_count": int(ext209.shape[1]),
            "members": list(ext209.columns),
            "max_abs_ledger_auc_delta": max(
                abs(c["ledger_auc_delta"]) for c in checks209
            ),
            "name_overlap_with_ext85": len(overlap),
        },
    }
    (CACHE_DIR / "verification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )
    print(
        f"외부 행렬 캐시 저장: {CACHE_DIR} (ext85 {ext85.shape}, ext209 {ext209.shape})"
    )


def load_own(fold_of: pd.Series, own32: bool = False) -> pd.DataFrame:
    pool = Pool.load()
    members = [(member.config, member.run_id) for member in pool.members]
    assert len(members) == 35, f"후보 풀 {len(members)}구성원"
    if own32:
        reference = json.loads(EXT94_EVIDENCE_PATH.read_text())["configs"]["own32"][
            "members"
        ]
        by_config = dict(members)
        by_config[OWN32_ONLY_MEMBER[0]] = OWN32_ONLY_MEMBER[1]
        missing = [name for name in reference if name not in by_config]
        assert not missing, f"own32 구성원 누락: {missing}"
        members = [(name, by_config[name]) for name in reference]
    return ensemble.member_matrix(members, MlflowRunStore(), fold_of.index)


def _ablation_members(name: str) -> set[str]:
    """절제 구성에서 뺄 장부 구성원 열 이름."""
    _, accepted = load_ledger()
    if name == "ablate_author_statement":
        picked = [r for r in accepted if r["fold_evidence"] == "author_statement"]
    else:
        needle = ABLATIONS[name]
        picked = [r for r in accepted if any(needle in c for c in r["caveats"])]
    return {f"ext_{r['member_id']}" for r in picked}


def build_matrix(name: str, fold_of: pd.Series) -> pd.DataFrame:
    if name == REPRO:
        own = load_own(fold_of, own32=True)
        ext = pd.read_parquet(CACHE_DIR / "ext85.parquet")
        return pd.concat([own, ext], axis=1)
    if name == "ext209":
        return pd.read_parquet(CACHE_DIR / "ext209.parquet")
    own = load_own(fold_of)
    if name == "own35":
        return own
    if name == "own35_ext85":
        return pd.concat([own, pd.read_parquet(CACHE_DIR / "ext85.parquet")], axis=1)
    ext = pd.read_parquet(CACHE_DIR / "ext209.parquet")
    if name == "own35_ext209":
        return pd.concat([own, ext], axis=1)
    drop = _ablation_members(name)
    assert drop, name
    return pd.concat([own, ext.drop(columns=sorted(drop))], axis=1)


def jobs() -> list[tuple[str, str]]:
    """(구성, 전략) 작업 목록. 사다리는 등록 전략 전체, 재현·절제는 기본 두 전략."""
    default = [
        name for name in STRATEGY_PRIORITY if name in ensemble.DEFAULT_COMBINER_NAMES
    ]
    assert set(default) == set(ensemble.DEFAULT_COMBINER_NAMES), (
        "전략 우선순위 목록이 등록과 다르다"
    )
    ordered: list[tuple[str, str]] = []
    for strategy in default:
        if strategy in DEFAULT_STRATEGIES:
            ordered.append((REPRO, strategy))
        for config in LADDER:
            ordered.append((config, strategy))
    for config in ABLATIONS:
        for strategy in DEFAULT_STRATEGIES:
            ordered.append((config, strategy))
    return ordered


def part_path(config: str, strategy: str) -> Path:
    return PARTS_DIR / f"{config}__{strategy}.json"


def run_job(
    config: str, strategy: str, fold_of: pd.Series, y: pd.Series, force: bool
) -> None:
    path = part_path(config, strategy)
    if path.exists() and not force:
        print(f"건너뜀(있음): {path}")
        return
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix(config, fold_of)
    reweighting = missingness_reweighting(TRAIN_PATH, MISSINGNESS_TEST_PATH)
    combiner = ensemble.COMBINER_REGISTRY[strategy]
    print(f"=== {config} ({matrix.shape[1]}구성원) × {strategy} ===", flush=True)
    started = time.monotonic()
    record: dict[str, object] = {
        "config": config,
        "strategy": strategy,
        "member_count": int(matrix.shape[1]),
        "members": list(matrix.columns),
    }
    try:
        evaluation = ensemble.evaluate_nested(combiner, matrix, fold_of, y, reweighting)
    except ensemble.CombinerConvergenceError as exc:
        record.update(
            failed=True, reason=str(exc), elapsed_seconds=time.monotonic() - started
        )
        print(f"  제외 ({exc})")
    else:
        weighted = evaluation.weighted
        record.update(
            failed=False,
            nested_auc=evaluation.nested_auc,
            weighted_oof_auc=weighted.auc,
            weighted_delta=weighted.auc - evaluation.nested_auc,
            fold_aucs={str(o.fold): o.auc for o in evaluation.folds},
            elapsed_seconds=evaluation.elapsed_seconds,
        )
        print(
            f"  nested {evaluation.nested_auc:.7f}, 가중 {weighted.auc:.7f}, "
            f"{evaluation.elapsed_seconds:.0f}s",
            flush=True,
        )
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2))


def _load_parts() -> dict[str, dict[str, dict]]:
    parts: dict[str, dict[str, dict]] = {}
    for path in sorted(PARTS_DIR.glob("*.json")):
        record = json.loads(path.read_text())
        parts.setdefault(record["config"], {})[record["strategy"]] = record
    return parts


def _best(strategies: dict[str, dict]) -> dict | None:
    succeeded = [r for r in strategies.values() if not r["failed"]]
    return max(succeeded, key=lambda r: r["nested_auc"]) if succeeded else None


def report() -> None:
    parts = _load_parts()
    baseline = json.loads(BASELINE_EVALUATION_PATH.read_text())
    assert baseline["best_nested_oof_auc"] == POOL35_NESTED, baseline[
        "best_nested_oof_auc"
    ]
    assert baseline["member_count"] == 35
    gate = POOL35_NESTED + GATE_DELTA
    expected = jobs()
    done = [(c, s) for c, s in expected if s in parts.get(c, {})]
    print(f"작업 {len(done)}/{len(expected)} 완료")

    own35_best = _best(parts.get("own35", {}))
    configs: dict[str, dict] = {}
    for config in CONFIG_NAMES:
        strategies = parts.get(config)
        if not strategies:
            continue
        best = _best(strategies)
        entry: dict[str, object] = {
            "member_count": next(iter(strategies.values()))["member_count"],
            "strategies": {
                name: {k: v for k, v in rec.items() if k != "members"}
                for name, rec in sorted(strategies.items())
            },
            "best_strategy": None if best is None else best["strategy"],
            "best_nested_auc": None if best is None else best["nested_auc"],
            "best_weighted_oof_auc": None if best is None else best["weighted_oof_auc"],
        }
        if best is not None and own35_best is not None and config != "own35":
            fold_deltas = {
                fold: best["fold_aucs"][fold] - own35_best["fold_aucs"][fold]
                for fold in sorted(best["fold_aucs"])
            }
            same = strategies.get(own35_best["strategy"])
            entry.update(
                delta_vs_pool35_ledger=best["nested_auc"] - POOL35_NESTED,
                delta_vs_own35_rerun=best["nested_auc"] - own35_best["nested_auc"],
                fold_deltas_vs_own35_best=fold_deltas,
                folds_positive=sum(v > 0 for v in fold_deltas.values()),
                same_strategy_delta_vs_own35=(
                    None
                    if same is None or same["failed"]
                    else same["nested_auc"] - own35_best["nested_auc"]
                ),
                passes_gate=bool(
                    best["nested_auc"] >= gate
                    and all(v > 0 for v in fold_deltas.values())
                ),
            )
        configs[config] = entry

    repro = (
        configs.get(REPRO, {}).get("strategies", {}).get("shrunk_rank_logit_logistic")
    )
    repro_check = None
    if repro is not None and not repro["failed"]:
        repro_check = {
            "nested_auc": repro["nested_auc"],
            "reference": EXT85_UNION_REFERENCE,
            "delta": repro["nested_auc"] - EXT85_UNION_REFERENCE,
            "within_noise_floor": abs(repro["nested_auc"] - EXT85_UNION_REFERENCE)
            <= NOISE_FLOOR,
        }

    passing = [
        (name, configs[name])
        for name in LADDER
        if name in configs and configs[name].get("passes_gate")
    ]
    selected = None
    if passing:
        top = max(passing, key=lambda item: item[1]["best_nested_auc"])
        # 잡음 바닥 안이면 구성원이 적은 쪽을 고른다.
        tied = [
            item
            for item in passing
            if top[1]["best_nested_auc"] - item[1]["best_nested_auc"] <= NOISE_FLOOR
        ]
        selected = min(tied, key=lambda item: item[1]["member_count"])[0]

    payload = {
        "issue": ISSUE,
        "schema_version": 1,
        "gate": {
            "pool35_nested": POOL35_NESTED,
            "pool35_source": str(BASELINE_EVALUATION_PATH),
            "delta_required": GATE_DELTA,
            "threshold": gate,
            "folds_required_positive": 5,
            "noise_floor": NOISE_FLOOR,
        },
        "ladder": list(LADDER),
        "default_strategies": list(DEFAULT_STRATEGIES),
        "jobs_done": len(done),
        "jobs_expected": len(expected),
        "reproduction_check": repro_check,
        "configs": configs,
        "passing_configs": [name for name, _ in passing],
        "selected_config": selected,
        "cache_verification": json.loads((CACHE_DIR / "verification.json").read_text()),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(
        f"\n문턱: nested >= {gate:.7f} (35개 풀 {POOL35_NESTED:.7f} + {GATE_DELTA}), 분할 5/5 양수"
    )
    if repro_check:
        print(
            f"기준 재현 {REPRO}: {repro_check['nested_auc']:.7f} "
            f"(참조 {EXT85_UNION_REFERENCE}, 차이 {repro_check['delta']:+.2e}, "
            f"{'잡음 바닥 안' if repro_check['within_noise_floor'] else '잡음 바닥 밖'})"
        )
    print(
        f"\n{'구성':<26}{'구성원':>6} {'최선 전략':<32}{'nested':>11}{'가중':>11}{'문턱대비':>12} 분할"
    )
    for name, entry in configs.items():
        if entry["best_strategy"] is None:
            continue
        delta = (
            ""
            if "delta_vs_pool35_ledger" not in entry
            else f"{entry['delta_vs_pool35_ledger']:+.7f}"
        )
        folds = "" if "folds_positive" not in entry else f"{entry['folds_positive']}/5"
        flag = "" if not entry.get("passes_gate") else " 통과"
        print(
            f"{name:<26}{entry['member_count']:>6} {entry['best_strategy']:<32}"
            f"{entry['best_nested_auc']:>11.7f}{entry['best_weighted_oof_auc']:>11.7f}"
            f"{delta:>12} {folds}{flag}"
        )
    print(f"\n선택: {selected}")
    print(f"근거 저장: {EVIDENCE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="확장 스택 nested OOF 사다리 판정 (#443)"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="외부 행렬을 검증하고 캐시한다."
    )
    parser.add_argument(
        "--list-jobs", action="store_true", help="(구성, 전략) 작업 목록을 출력한다."
    )
    parser.add_argument("--config", choices=CONFIG_NAMES)
    parser.add_argument("--only", help="이 이름의 등록 결합 전략만 평가한다.")
    parser.add_argument(
        "--force", action="store_true", help="이미 있는 산출물도 다시 만든다."
    )
    parser.add_argument(
        "--report", action="store_true", help="산출물을 모아 판정 근거를 쓴다."
    )
    args = parser.parse_args()

    if args.list_jobs:
        for config, strategy in jobs():
            if not part_path(config, strategy).exists():
                print(config, strategy)
        return
    if args.report:
        report()
        return

    train = pd.read_csv(TRAIN_PATH)
    fold_of = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"]
    y = labels(fold_of.index)
    ext94.verify_row_order(train, fold_of)
    if args.prepare:
        prepare(fold_of, y)
        return
    if args.config is None or args.only is None:
        parser.error(
            "--config와 --only를 함께 주거나 --prepare/--list-jobs/--report 중 하나를 쓴다."
        )
    if args.only not in ensemble.COMBINER_REGISTRY:
        parser.error(f"결합 전략 없음: {args.only}")
    run_job(args.config, args.only, fold_of, y, args.force)


if __name__ == "__main__":
    main()
