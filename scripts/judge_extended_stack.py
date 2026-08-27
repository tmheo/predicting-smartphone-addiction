"""넓힌 확장 스택을 nested OOF 사다리로 재판정한다. (#455)

지도 #451의 두 번째 최종 제출 교체 판정이다. 판본 2 장부(#454)의 통과 400 = 판본 1의 209
(`added_in == 442`) + 신규 191(`added_in == 454`)을 자체 35개 풀에 더해 잰다. 사다리는 결과
확인 전에 고정한다.

1. own35_ext207        - 현재 판 재현. 자체 35 + 판본 1 통과 209에서 TE 누출 2개를 뺀 207.
                         #443 `ablate_te_leak`(0.9702876)과 잡음 바닥 5.7e-06 안에서 맞아야 한다.
2. own35_ext207_new191 - 현재 판 + 신규 191 전체.
3. ablate_new_*        - 2에서 신규를 공급원별로 뺀 절제.
4. ablate_license_unknown, ablate_fold_evidence_none, ablate_author_statement
                       - 장부 부류 절제. 저자 서술은 판본 1·2 양쪽에서 뺀다.
5. ablate_nhtquyn_duplicates   - nhtquyn 120을 0.998 군집 대표 62로 줄인 구성(티켓 (4)).
   ablate_near_duplicate_cluster - `near_duplicate_cluster` 부류 67 전체를 뺀 부류 절제판.

교체 문턱은 현재 판 nested `0.9702876` 대비 `+0.00002` 이상, 바깥쪽 검증 분할 5/5 양수다.
분할별 양수 여부는 이번 세션에서 다시 잰 own35_ext207 최선 전략과 비교한다.
문턱을 넘는 구성이 여럿이면 nested가 가장 높은 구성을 고르되, 차이가 잡음 바닥 안이면
구성원이 적은 쪽을 고른다. 결과를 본 뒤 문턱과 사다리를 바꾸지 않는다.

이 판정은 **읽기 전용**이다. 외부 예측을 `artifacts/pool.yaml`에 넣지 않고 champion
판정에도 쓰지 않으며 MLflow 실행도 만들지 않는다.

사용법:
    uv run python scripts/judge_extended_stack.py --prepare
    uv run python scripts/judge_extended_stack.py --list-jobs
    uv run python scripts/judge_extended_stack.py --run-jobs --workers 3 --threads 4
    uv run python scripts/judge_extended_stack.py --config own35_ext207_new191 --only shrunk_rank_logit_logistic
    uv run python scripts/judge_extended_stack.py --report

작업 하나는 (구성, 전략) 한 쌍이고 산출물은 run-logs/issue455/parts/<구성>__<전략>.json이다.
이미 있는 산출물은 건너뛰므로 중단 뒤 같은 명령으로 이어 달릴 수 있다.

#443 사다리(자체 35 + 판본 1 장부 209, 등록 전략 19개 전수)의 정의는 PR #447 시점의 이
파일이고 결과는 `docs/research/extended-stack-ladder-evidence.json`이다.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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

ISSUE = 455
LEDGER_PATH = Path("docs/research/external-member-ledger.json")
PRIOR_EVIDENCE_PATH = Path("docs/research/extended-stack-ladder-evidence.json")
OUT_DIR = Path("run-logs/issue455")
PARTS_DIR = OUT_DIR / "parts"
CACHE_DIR = OUT_DIR / "cache"
LOG_DIR = OUT_DIR / "logs"
EVIDENCE_PATH = Path("docs/research/extended-stack-ladder-2.json")

N_TRAIN = 691369
LEDGER_VERSION = 2
BASE_ISSUE = 442
NEW_ISSUE = 454
ACCEPTED_COUNTS = {BASE_ISSUE: 209, NEW_ISSUE: 191}
# #444가 코드 수준 TE 누출로 조립에서 뺀 판본 1 구성원. 현재 판(242구성원)의 정의다.
TE_LEAK_MEMBERS = ("szymon74:pub_rmlp", "szymon74:pub_tabm")
# 현재 판 nested는 #443 절제 구성 `ablate_te_leak`의 값이다. 정확한 값은 #443 근거에서 읽는다.
PRIOR_PLATE_CONFIG = "ablate_te_leak"
CURRENT_PLATE_NESTED = 0.9702876
GATE_DELTA = 0.00002
NOISE_FLOOR = 5.7e-06
FOLDS_REQUIRED_POSITIVE = 5

DEFAULT_STRATEGIES = ("shrunk_rank_logit_logistic", "rank_logit_logistic")
BASE = "own35_ext207"
FULL = "own35_ext207_new191"


def _caveat_classes(row: dict) -> set[str]:
    """장부 `caveats`의 부류(첫 낱말, 콜론 앞)."""
    return {caveat.split(":")[0].split()[0] for caveat in row["caveats"]}


def _is_new(row: dict) -> bool:
    return row["added_in"] == NEW_ISSUE


@dataclass(frozen=True)
class LadderConfig:
    name: str
    description: str
    include_new: bool
    expected_external: int
    # True를 돌려주는 통과 구성원을 뺀다. None이면 빼지 않는다.
    drop: Callable[[dict], bool] | None = None


CONFIGS: tuple[LadderConfig, ...] = (
    LadderConfig(
        BASE, "현재 판 재현: 판본 1 통과 209에서 TE 누출 2개 제외", False, 207
    ),
    LadderConfig(FULL, "현재 판 + 신규 191 전체", True, 398),
    LadderConfig(
        "ablate_new_paiky6",
        "신규 가운데 paiky1995 5분할 신경망 6개 제외",
        True,
        392,
        lambda r: _is_new(r) and r["source"] == "paiky6",
    ),
    LadderConfig(
        "ablate_new_nhtquyn",
        "신규 가운데 nhtquyn 고전 확률 모델 120개 제외",
        True,
        278,
        lambda r: _is_new(r) and r["source"] == "nhtquyn",
    ),
    LadderConfig(
        "ablate_new_hboyang150",
        "신규 가운데 hboyang 150-fusion 단일 11개 제외",
        True,
        387,
        lambda r: _is_new(r) and r["source"] == "hboyang150",
    ),
    LadderConfig(
        "ablate_new_masaya",
        "신규 가운데 masayakawamata 1개 제외",
        True,
        397,
        lambda r: _is_new(r) and r["source"] == "masaya",
    ),
    LadderConfig(
        "ablate_new_beicicc_other",
        "신규 가운데 beicicc other 라이선스 3개 제외",
        True,
        395,
        lambda r: _is_new(r) and r["source"].startswith("beicicc:"),
    ),
    LadderConfig(
        "ablate_new_szymon74_naji",
        "신규 가운데 szymonkapiski의 najiama 재게시 5개 제외",
        True,
        393,
        lambda r: _is_new(r) and r["source"] == "szymon74",
    ),
    LadderConfig(
        "ablate_new_notebooks",
        "신규 가운데 공개 노트북 출력물 45개 제외",
        True,
        353,
        lambda r: _is_new(r) and r["source"].startswith("nb_"),
    ),
    LadderConfig(
        "ablate_license_unknown",
        "`license_unknown` 부류 64개 제외",
        True,
        334,
        lambda r: "license_unknown" in _caveat_classes(r),
    ),
    LadderConfig(
        "ablate_fold_evidence_none",
        "`fold_evidence_none` 부류 3개 제외",
        True,
        395,
        lambda r: "fold_evidence_none" in _caveat_classes(r),
    ),
    LadderConfig(
        "ablate_author_statement",
        "분할 근거가 저자 서술뿐인 구성원 제외(154개 가운데 TE 누출 2개는 이미 빠져 152개)",
        True,
        246,
        lambda r: r["fold_evidence"] == "author_statement",
    ),
    LadderConfig(
        "ablate_nhtquyn_duplicates",
        "nhtquyn 120개를 0.998 군집 대표 62개로 줄임(nhtquyn의 near_duplicate_cluster 58개 제외)",
        True,
        340,
        lambda r: (
            r["source"] == "nhtquyn" and "near_duplicate_cluster" in _caveat_classes(r)
        ),
    ),
    LadderConfig(
        "ablate_near_duplicate_cluster",
        "`near_duplicate_cluster` 부류 67개 전체 제외",
        True,
        331,
        lambda r: "near_duplicate_cluster" in _caveat_classes(r),
    ),
)
CONFIG_BY_NAME = {config.name: config for config in CONFIGS}
CONFIG_NAMES = tuple(config.name for config in CONFIGS)


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


@functools.lru_cache(maxsize=1)
def _ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text())


def load_ledger() -> tuple[dict, list[dict]]:
    """판본 2 장부와 통과 400구성원(장부 순서). `added_in`별 수를 단언한다."""
    ledger = _ledger()
    assert ledger["version"] == LEDGER_VERSION, ledger["version"]
    accepted = [row for row in ledger["members"] if row["status"] == "accepted"]
    counts = {
        issue: sum(row["added_in"] == issue for row in accepted)
        for issue in ACCEPTED_COUNTS
    }
    assert counts == ACCEPTED_COUNTS, counts
    assert (
        len(accepted)
        == ledger["summary"]["accepted_count"]
        == sum(ACCEPTED_COUNTS.values())
    ), len(accepted)
    return ledger, accepted


def _column(row: dict) -> str:
    return f"ext_{row['member_id']}"


def config_members(name: str) -> tuple[list[str], list[str]]:
    """구성의 외부 열 이름(장부 순서)과 뺀 구성원 열 이름."""
    config = CONFIG_BY_NAME[name]
    _, accepted = load_ledger()
    kept: list[str] = []
    removed: list[str] = []
    for row in accepted:
        column = _column(row)
        if row["member_id"] in TE_LEAK_MEMBERS:
            continue
        if _is_new(row) and not config.include_new:
            continue
        if config.drop is not None and config.drop(row):
            removed.append(column)
            continue
        kept.append(column)
    assert len(kept) == config.expected_external, (
        f"{name}: 외부 {len(kept)}구성원, 사다리 정의는 {config.expected_external}"
    )
    return kept, removed


def build_ext_all(fold_of: pd.Series, y: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    """장부 통과 400구성원의 OOF 행렬. 장부 순서를 열 순서로 쓴다."""
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
        key = _column(row)
        assert key not in columns, key
        columns[key] = values
        checks.append(
            {
                "member_id": row["member_id"],
                "added_in": row["added_in"],
                "auc": auc,
                "ledger_auc_delta": delta,
            }
        )
    matrix = pd.DataFrame(columns, index=fold_of.index).astype(np.float64)
    assert matrix.shape[1] == sum(ACCEPTED_COUNTS.values())
    return matrix, checks


def prior_evidence() -> dict:
    """#443 근거. 현재 판(`ablate_te_leak`)의 nested·분할 AUC·열 순서를 준다."""
    evidence = json.loads(PRIOR_EVIDENCE_PATH.read_text())
    assert evidence["issue"] == 443, evidence["issue"]
    plate = evidence["configs"][PRIOR_PLATE_CONFIG]
    assert plate["member_count"] == 242, plate["member_count"]
    shrunk = plate["strategies"]["shrunk_rank_logit_logistic"]
    assert round(shrunk["nested_auc"], 7) == CURRENT_PLATE_NESTED, shrunk["nested_auc"]
    return evidence


def prepare(fold_of: pd.Series, y: pd.Series) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ext_all, checks = build_ext_all(fold_of, y)
    ext_all.to_parquet(CACHE_DIR / "ext400.parquet")
    prior = prior_evidence()
    base_members, _ = config_members(BASE)
    prior_members = [
        name
        for name in prior["cache_verification"]["ext209"]["members"]
        if name[4:] not in TE_LEAK_MEMBERS
    ]
    assert base_members == prior_members, "현재 판 재현의 외부 열 순서가 #443과 다르다."
    configs = {}
    for config in CONFIGS:
        kept, removed = config_members(config.name)
        configs[config.name] = {
            "description": config.description,
            "external_count": len(kept),
            "removed_count": len(removed),
            "removed_members": removed,
        }
    payload = {
        "ledger_sha256": _sha256(LEDGER_PATH),
        "ledger_version": LEDGER_VERSION,
        "prior_evidence_sha256": _sha256(PRIOR_EVIDENCE_PATH),
        "ext400": {
            "member_count": int(ext_all.shape[1]),
            "members": list(ext_all.columns),
            "max_abs_ledger_auc_delta": max(abs(c["ledger_auc_delta"]) for c in checks),
            "accepted_by_added_in": {
                str(issue): sum(c["added_in"] == issue for c in checks)
                for issue in ACCEPTED_COUNTS
            },
        },
        "base_order_matches_443": True,
        "configs": configs,
    }
    (CACHE_DIR / "verification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )
    print(f"외부 행렬 캐시 저장: {CACHE_DIR} (ext400 {ext_all.shape})")
    for name, entry in configs.items():
        print(
            f"  {name:<32} 외부 {entry['external_count']:>3} (뺌 {entry['removed_count']})"
        )


def load_own(fold_of: pd.Series) -> pd.DataFrame:
    pool = Pool.load()
    members = [(member.config, member.run_id) for member in pool.members]
    assert len(members) == 35, f"후보 풀 {len(members)}구성원"
    return ensemble.member_matrix(members, MlflowRunStore(), fold_of.index)


def build_matrix(name: str, fold_of: pd.Series) -> pd.DataFrame:
    own = load_own(fold_of)
    kept, _ = config_members(name)
    ext = pd.read_parquet(CACHE_DIR / "ext400.parquet", columns=kept)
    assert list(ext.columns) == kept
    return pd.concat([own, ext], axis=1)


def jobs() -> list[tuple[str, str]]:
    """(구성, 전략) 작업 목록. 판정을 가르는 두 shrunk 작업을 먼저, 빠른 전략을 그다음에 둔다."""
    shrunk, fast = DEFAULT_STRATEGIES
    assert all(name in ensemble.COMBINER_REGISTRY for name in DEFAULT_STRATEGIES)
    ordered = [(BASE, shrunk), (FULL, shrunk)]
    ordered += [(name, fast) for name in CONFIG_NAMES]
    ordered += [(name, shrunk) for name in CONFIG_NAMES if name not in (BASE, FULL)]
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


def _running_jobs() -> set[tuple[str, str]]:
    """이 기계에서 지금 돌고 있는 (구성, 전략) 작업. 드라이버가 죽어도 남은 작업자를 센다."""
    listing = subprocess.run(
        ["ps", "-axo", "command"], capture_output=True, text=True, check=False
    ).stdout
    pattern = re.compile(r"judge_extended_stack\.py --config (\S+) --only (\S+)")
    return {match.groups() for match in pattern.finditer(listing)}


def run_jobs(workers: int, threads: int, force: bool) -> None:
    """남은 작업을 병렬 실행한다.

    workers는 이 기계에서 동시에 도는 작업 프로세스 수의 상한이다. 이전 드라이버가 남긴
    작업자도 세므로, 드라이버를 죽이고 다른 상한으로 다시 띄워도 같은 작업을 겹쳐 돌리지
    않는다. 작업당 BLAS 스레드 수는 threads다.
    """
    running = _running_jobs()
    pending = [
        (config, strategy)
        for config, strategy in jobs()
        if (force or not part_path(config, strategy).exists())
        and (config, strategy) not in running
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = str(threads)
    print(
        f"남은 작업 {len(pending)}개(실행 중 {len(running)}개 제외), "
        f"동시 상한 {workers}, 스레드 {threads}",
        flush=True,
    )
    slot = threading.Lock()

    def run(job: tuple[str, str]) -> tuple[tuple[str, str], int, float]:
        config, strategy = job
        command = [sys.executable, __file__, "--config", config, "--only", strategy]
        if force:
            command.append("--force")
        while True:
            with slot:
                if len(_running_jobs()) < workers:
                    started = time.monotonic()
                    handle = (LOG_DIR / f"{config}__{strategy}.log").open("w")
                    process = subprocess.Popen(
                        command, env=env, stdout=handle, stderr=handle
                    )
                    break
            time.sleep(30)
        with handle:
            code = process.wait()
        return job, code, time.monotonic() - started

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (config, strategy), code, elapsed in pool.map(run, pending):
            status = "완료" if code == 0 else f"실패({code})"
            print(f"{status} {config} × {strategy} {elapsed:.0f}s", flush=True)
    print("all jobs finished", flush=True)


def _load_parts() -> dict[str, dict[str, dict]]:
    parts: dict[str, dict[str, dict]] = {}
    for path in sorted(PARTS_DIR.glob("*.json")):
        record = json.loads(path.read_text())
        parts.setdefault(record["config"], {})[record["strategy"]] = record
    return parts


def _best(strategies: dict[str, dict]) -> dict | None:
    succeeded = [r for r in strategies.values() if not r["failed"]]
    return max(succeeded, key=lambda r: r["nested_auc"]) if succeeded else None


def _fold_deltas(record: dict, reference: dict) -> dict[str, float]:
    return {
        fold: record["fold_aucs"][fold] - reference["fold_aucs"][fold]
        for fold in sorted(record["fold_aucs"])
    }


def _strategy_entry(
    record: dict, base: dict | None, full: dict | None
) -> dict[str, object]:
    entry = {k: v for k, v in record.items() if k != "members"}
    if record["failed"]:
        return entry
    if base is not None and not base["failed"]:
        deltas = _fold_deltas(record, base)
        entry.update(
            delta_vs_base_same_strategy=record["nested_auc"] - base["nested_auc"],
            weighted_delta_vs_base_same_strategy=(
                record["weighted_oof_auc"] - base["weighted_oof_auc"]
            ),
            fold_deltas_vs_base_same_strategy=deltas,
            folds_positive_vs_base_same_strategy=sum(v > 0 for v in deltas.values()),
        )
    if full is not None and not full["failed"]:
        entry["delta_vs_full_same_strategy"] = record["nested_auc"] - full["nested_auc"]
    return entry


def report() -> None:
    parts = _load_parts()
    prior = prior_evidence()
    plate = prior["configs"][PRIOR_PLATE_CONFIG]["strategies"]
    plate_nested = plate["shrunk_rank_logit_logistic"]["nested_auc"]
    gate = plate_nested + GATE_DELTA
    expected = jobs()
    done = [(c, s) for c, s in expected if s in parts.get(c, {})]
    print(f"작업 {len(done)}/{len(expected)} 완료")
    verification = json.loads((CACHE_DIR / "verification.json").read_text())

    base_parts = parts.get(BASE, {})
    full_parts = parts.get(FULL, {})
    base_best = _best(base_parts)

    reproduction: dict[str, object] = {}
    for strategy in DEFAULT_STRATEGIES:
        record = base_parts.get(strategy)
        if record is None or record["failed"]:
            continue
        reference = plate[strategy]
        deltas = _fold_deltas(record, reference)
        reproduction[strategy] = {
            "nested_auc": record["nested_auc"],
            "reference": reference["nested_auc"],
            "delta": record["nested_auc"] - reference["nested_auc"],
            "within_noise_floor": abs(record["nested_auc"] - reference["nested_auc"])
            <= NOISE_FLOOR,
            "weighted_delta": record["weighted_oof_auc"]
            - reference["weighted_oof_auc"],
            "max_abs_fold_delta": max(abs(v) for v in deltas.values()),
        }
    shrunk_repro = reproduction.get("shrunk_rank_logit_logistic")

    configs: dict[str, dict] = {}
    for name in CONFIG_NAMES:
        strategies = parts.get(name)
        if not strategies:
            continue
        best = _best(strategies)
        spec = verification["configs"][name]
        entry: dict[str, object] = {
            "description": spec["description"],
            "member_count": next(iter(strategies.values()))["member_count"],
            "external_count": spec["external_count"],
            "removed_count": spec["removed_count"],
            "removed_members": spec["removed_members"],
            "strategies": {
                strategy: _strategy_entry(
                    record, base_parts.get(strategy), full_parts.get(strategy)
                )
                for strategy, record in sorted(strategies.items())
            },
            "best_strategy": None if best is None else best["strategy"],
            "best_nested_auc": None if best is None else best["nested_auc"],
            "best_weighted_oof_auc": None if best is None else best["weighted_oof_auc"],
        }
        if best is not None and base_best is not None and name != BASE:
            fold_deltas = _fold_deltas(best, base_best)
            entry.update(
                delta_vs_current_plate=best["nested_auc"] - plate_nested,
                delta_vs_base_rerun=best["nested_auc"] - base_best["nested_auc"],
                fold_deltas_vs_base_best=fold_deltas,
                folds_positive=sum(v > 0 for v in fold_deltas.values()),
                passes_gate=bool(
                    best["nested_auc"] >= gate
                    and sum(v > 0 for v in fold_deltas.values())
                    >= FOLDS_REQUIRED_POSITIVE
                ),
            )
        configs[name] = entry

    passing = [
        (name, entry) for name, entry in configs.items() if entry.get("passes_gate")
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
            "current_plate_config_443": PRIOR_PLATE_CONFIG,
            "current_plate_nested": plate_nested,
            "current_plate_source": str(PRIOR_EVIDENCE_PATH),
            "delta_required": GATE_DELTA,
            "threshold": gate,
            "folds_required_positive": FOLDS_REQUIRED_POSITIVE,
            "noise_floor": NOISE_FLOOR,
            "fold_reference": f"{BASE} 최선 전략(이번 세션 재측정)",
        },
        "ladder": list(CONFIG_NAMES),
        "base_config": BASE,
        "full_config": FULL,
        "default_strategies": list(DEFAULT_STRATEGIES),
        "jobs_done": len(done),
        "jobs_expected": len(expected),
        "reproduction_check": reproduction,
        "reproduction_passes": bool(
            shrunk_repro and shrunk_repro["within_noise_floor"]
        ),
        "configs": configs,
        "passing_configs": [name for name, _ in passing],
        "selected_config": selected,
        "cache_verification": {k: v for k, v in verification.items() if k != "configs"},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(
        f"\n문턱: nested >= {gate:.7f} (현재 판 {plate_nested:.7f} + {GATE_DELTA}), "
        f"분할 {FOLDS_REQUIRED_POSITIVE}/5 양수"
    )
    for strategy, check in reproduction.items():
        print(
            f"현재 판 재현 {BASE} × {strategy}: {check['nested_auc']:.7f} "
            f"(#443 {check['reference']:.7f}, 차이 {check['delta']:+.2e}, "
            f"{'잡음 바닥 안' if check['within_noise_floor'] else '잡음 바닥 밖'}, "
            f"분할 최대 차이 {check['max_abs_fold_delta']:.2e})"
        )
    print(
        f"\n{'구성':<32}{'구성원':>6} {'최선 전략':<28}{'nested':>11}{'가중':>11}"
        f"{'현재판대비':>12} 분할"
    )
    for name, entry in configs.items():
        if entry["best_strategy"] is None:
            continue
        delta = (
            ""
            if "delta_vs_current_plate" not in entry
            else f"{entry['delta_vs_current_plate']:+.7f}"
        )
        folds = "" if "folds_positive" not in entry else f"{entry['folds_positive']}/5"
        flag = "" if not entry.get("passes_gate") else " 통과"
        print(
            f"{name:<32}{entry['member_count']:>6} {entry['best_strategy']:<28}"
            f"{entry['best_nested_auc']:>11.7f}{entry['best_weighted_oof_auc']:>11.7f}"
            f"{delta:>12} {folds}{flag}"
        )
    print(f"\n선택: {selected}")
    print(f"근거 저장: {EVIDENCE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="넓힌 확장 스택 nested OOF 사다리 재판정 (#455)"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="외부 행렬을 검증하고 캐시한다."
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="남은 (구성, 전략) 작업 목록을 출력한다.",
    )
    parser.add_argument(
        "--run-jobs", action="store_true", help="남은 작업을 병렬 작업자로 실행한다."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="--run-jobs 동시 작업 프로세스 상한(이전 드라이버가 남긴 작업자 포함)",
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="--run-jobs 작업자당 BLAS 스레드 수"
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
    if args.run_jobs:
        run_jobs(args.workers, args.threads, args.force)
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
            "--config와 --only를 함께 주거나 --prepare/--list-jobs/--run-jobs/--report 중 하나를 쓴다."
        )
    if args.only not in ensemble.COMBINER_REGISTRY:
        parser.error(f"결합 전략 없음: {args.only}")
    run_job(args.config, args.only, fold_of, y, args.force)


if __name__ == "__main__":
    main()
