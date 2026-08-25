"""PROTOTYPE (이슈 #410) - 버리는 코드. 공식 경로가 아니다.

`candidate-pool-rebuild-v1` 계약(ADR 0003)의 정확 검색을 현재 33개 OOF에 적용해
실행 시간, 결과 풀, 동결 OOF 조건부 절차 점수를 재현 가능하게 남길 수 있는지 잰다.

허용한 가속은 결과 동일성이 성립하는 것뿐이다.
- 경험적 누적분포 변환(rank_logit 표현의 rank 열)을 학습 행 집합(제외 fold 집합 E)마다
  미리 계산해 저장한다. 변환은 원소별 함수이므로 참조 구현과 같은 값이다.
- 같은 학습 행 집합에 대한 메타 적합은 한 번만 수행한다. 전체 5분할에서 바깥 f의
  LOFO g와 바깥 g의 LOFO f는 같은 행 집합·같은 순서를 학습하므로 같은 적합이다.
- 블록 안 백분위 순위(rank_mean)는 열마다 독립이라 구성원별로 미리 계산한다.

가지치기, 상위 일부 선별, 대리 점수 판정은 없다. 모든 이동을 정확 평가한다.

실행 순서:
    uv run python scripts/prototype_pool_rebuild_search.py prepare
    uv run python scripts/prototype_pool_rebuild_search.py search --scopes full,0,1,2,3,4
    uv run python scripts/prototype_pool_rebuild_search.py finish
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import ensemble as E  # noqa: E402
from pipeline.data import ID, TARGET  # noqa: E402
from pipeline.pool_audit import prediction_array_sha256  # noqa: E402

OUT = ROOT / "run-logs/pool-rebuild-prototype"
STRATEGY = "shrunk_rank_logit_logistic"
LAMBDA_GRID = E.SHRINKAGE_LAMBDA_GRID
DUPLICATE_THRESHOLD = 0.998
EXCLUSIVE_PAIR = ("exp131_lookup_bivariate_plr5", "exp157_lookup_muon_initavg8")
BAND_ABS = 2.2073889871859763e-05  # docs/research/pool-reduction-equivalence-band.md 전체 하한 절대값
FOLDS = (0, 1, 2, 3, 4)
LOGIT_EPS = E.LogisticLinearCombiner.LOGIT_EPS


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1).encode()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(_json(payload))
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _maxrss_mb() -> dict[str, float]:
    scale = 1.0 / (1 << 20)  # macOS ru_maxrss는 바이트
    return {
        "driver": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale,
        "children_max": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale,
    }


# ---------------------------------------------------------------- prepare


def _stage_predictions(path: Path) -> list[dict[str, str]]:
    from pipeline.runs import MlflowRunStore

    pool = yaml.safe_load((ROOT / "artifacts/pool.yaml").read_text())
    champion = yaml.safe_load((ROOT / "artifacts/champion.yaml").read_text())
    members = [(champion["config"], champion["run_id"])] + [
        (m["config"], m["run_id"]) for m in pool["members"]
    ]
    folds = pd.read_parquet(ROOT / "artifacts/folds.parquet")
    ids = pd.Index(folds[ID], name=ID)
    store = MlflowRunStore()
    frame = pd.DataFrame({ID: ids.to_numpy()})
    records = []
    for index, (config, run_id) in enumerate(members):
        prediction = store.oof_of(run_id).reindex(ids)
        assert not prediction.isna().any(), config
        values = prediction.to_numpy(dtype=np.float64)
        frame[config] = values
        records.append(
            {
                "index": index,
                "config": config,
                "run_id": run_id,
                "oof_sha256": prediction_array_sha256(values),
                "role": "champion" if index == 0 else "candidate",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return records


def _all_excluded_sets() -> list[tuple[int, ...]]:
    sets: list[tuple[int, ...]] = []
    for size in (1, 2, 3):
        sets.extend(itertools.combinations(FOLDS, size))
    return sets


def _rank_file(excluded: tuple[int, ...]) -> Path:
    return OUT / "features" / f"rank_excl_{'-'.join(map(str, excluded))}.npy"


def _build_rank_file(args: tuple[tuple[int, ...], str]) -> str:
    excluded, _ = args
    with threadpool_limits(1):
        pred = np.load(OUT / "features/pred.npy", mmap_mode="r")
        folds = np.load(OUT / "features/folds.npy")
        train = ~np.isin(folds, excluded)
        out = np.lib.format.open_memmap(
            _rank_file(excluded), mode="w+", dtype=np.float64, shape=pred.shape
        )
        for column in range(pred.shape[1]):
            values = np.asarray(pred[:, column])
            q = E.EmpiricalCDFTransformer("uniform").fit(values[train, None])
            out[:, column] = q.transform(values[:, None])[:, 0]
        out.flush()
        del out
    return str(excluded)


def prepare(jobs: int) -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUT / "predictions.parquet"
    records = _stage_predictions(prediction_path)
    staged = pd.read_parquet(prediction_path)
    folds_frame = pd.read_parquet(ROOT / "artifacts/folds.parquet")
    train = pd.read_csv(ROOT / "data/train.csv", usecols=[ID, TARGET])
    assert staged[ID].equals(folds_frame[ID]) and staged[ID].equals(train[ID])
    names = [r["config"] for r in records]
    pred = staged[names].to_numpy(dtype=np.float64)
    folds = folds_frame["fold"].to_numpy(dtype=np.int64)
    labels = train[TARGET].to_numpy(dtype=np.int64)
    assert np.isfinite(pred).all()

    features = OUT / "features"
    features.mkdir(exist_ok=True)
    np.save(features / "pred.npy", pred)
    np.save(features / "folds.npy", folds)
    np.save(features / "labels.npy", labels)
    np.save(features / "logit.npy", np.log(pred.clip(LOGIT_EPS, 1 - LOGIT_EPS) / (1 - pred.clip(LOGIT_EPS, 1 - LOGIT_EPS))))
    block = np.empty_like(pred)
    for fold in FOLDS:
        rows = folds == fold
        block[rows] = pd.DataFrame(pred[rows]).rank(pct=True).to_numpy()
    np.save(features / "blockrank.npy", block)

    t_rank = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(_build_rank_file, [(excluded, "") for excluded in _all_excluded_sets()]))
    rank_seconds = time.time() - t_rank

    spearman = pd.DataFrame(pred, columns=names).rank().corr().to_numpy()
    pairs_over = [
        (names[i], names[j], float(spearman[i, j]))
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if spearman[i, j] >= DUPLICATE_THRESHOLD
    ]
    np.save(features / "spearman.npy", spearman)

    feature_hashes = {p.name: _sha256_file(p) for p in sorted(features.glob("*.npy"))}
    precommit = {
        "ticket": 410,
        "contract": "candidate-pool-rebuild-v1",
        "prototype": True,
        "members": records,
        "champion_index": 0,
        "frozen_order_note": "champion 1번 뒤 artifacts/pool.yaml 리스트 순서",
        "strategy": STRATEGY,
        "lambda_grid": list(LAMBDA_GRID),
        "exclusive_pair": [names.index(EXCLUSIVE_PAIR[0]), names.index(EXCLUSIVE_PAIR[1])],
        "duplicate_threshold": DUPLICATE_THRESHOLD,
        "pairs_at_or_over_threshold": pairs_over,
        "band_abs_warning": BAND_ABS,
        "procedure": [
            "forward1: 전진 추가·원자 교체 최대 상승 수렴",
            "backward1: 단일 제거 최대 상승 수렴(champion 제외)",
            "forward2: 전진 재수렴",
            "pair: 남은 후보 순서 없는 2개 묶음 전수 평가 1회, 최선 묶음 양수면 원자 추가",
            "forward3, backward2: 묶음 채택 시에만 재수렴 뒤 종료",
            "모든 이동은 AUC 차이 > 0 일 때만 수락, 동률은 동결 순서로 해소",
        ],
        "sources": {
            "predictions_parquet_sha256": _sha256_file(prediction_path),
            "folds_parquet_sha256": _sha256_file(ROOT / "artifacts/folds.parquet"),
            "labels_sha256": prediction_array_sha256(labels.astype(np.float64)),
            "pool_yaml_sha256": _sha256_file(ROOT / "artifacts/pool.yaml"),
            "champion_yaml_sha256": _sha256_file(ROOT / "artifacts/champion.yaml"),
        },
        "features_sha256": feature_hashes,
        "code_sha256": _sha256_file(Path(__file__)),
        "git_commit": _git_commit(),
        "prepare_seconds": time.time() - started,
        "rank_feature_seconds": rank_seconds,
        "rows": int(len(labels)),
    }
    precommit["identity_sha256"] = hashlib.sha256(
        _json({k: v for k, v in precommit.items() if k not in {"prepare_seconds", "rank_feature_seconds"}})
    ).hexdigest()
    _write_json(OUT / "precommit.json", precommit)
    print(f"prepare done in {time.time() - started:.1f}s; rank files {rank_seconds:.1f}s")
    print("pairs >= threshold:", pairs_over)


# ---------------------------------------------------------------- worker


class _Worker:
    def __init__(self) -> None:
        f = OUT / "features"
        self.pred = np.load(f / "pred.npy", mmap_mode="r")
        self.logit = np.load(f / "logit.npy", mmap_mode="r")
        self.block = np.load(f / "blockrank.npy", mmap_mode="r")
        self.labels = np.load(f / "labels.npy")
        self.folds = np.load(f / "folds.npy")
        self.rows = {fold: np.flatnonzero(self.folds == fold) for fold in FOLDS}
        self.n = len(self.labels)
        self._rank: dict[tuple[int, ...], np.ndarray] = {}

    def rank(self, excluded: tuple[int, ...]) -> np.ndarray:
        if excluded not in self._rank:
            self._rank[excluded] = np.load(_rank_file(excluded), mmap_mode="r")
        return self._rank[excluded]

    def design(self, excluded: tuple[int, ...], rows: np.ndarray, pool: tuple[int, ...]) -> np.ndarray:
        columns = list(pool)
        ranked = self.rank(excluded)[np.ix_(rows, columns)]
        logit = self.logit[np.ix_(rows, columns)]
        return np.column_stack((ranked, logit))

    def fit_context(
        self, excluded: tuple[int, ...], pool: tuple[int, ...], predict_folds: list[int]
    ) -> tuple[dict[int, np.ndarray], int]:
        train = np.flatnonzero(~np.isin(self.folds, excluded))
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.design(excluded, train, pool)).astype(np.float64, copy=False)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(scaled, self.labels[train])
        iterations = int(np.max(model.n_iter_))
        if iterations >= 1000:
            raise RuntimeError(f"미수렴 excluded={excluded} pool={pool}")
        probs = {}
        for fold in predict_folds:
            block = scaler.transform(self.design(excluded, self.rows[fold], pool))
            probs[fold] = model.predict_proba(block)[:, 1].astype(np.float64, copy=False)
        return probs, iterations

    def outer(
        self,
        pool: tuple[int, ...],
        scope: int | None,
        outer_fold: int,
        cache: dict[tuple[tuple[int, ...], int], np.ndarray],
        fits: list[int],
    ) -> tuple[np.ndarray, float, list[float]]:
        """바깥 fold 하나의 nested 예측: LOFO로 λ를 고르고 나머지 전체로 적합해 outer_fold를 예측."""
        inner = [f for f in FOLDS if f != scope and f != outer_fold]
        base = tuple(sorted(f for f in (scope, outer_fold) if f is not None))

        def probs_of(excluded: tuple[int, ...], fold: int) -> np.ndarray:
            key = (excluded, fold)
            if key not in cache:
                needed = [f for f in FOLDS if f in excluded and f != scope]
                probs, iterations = self.fit_context(excluded, pool, needed)
                fits.append(iterations)
                for f, value in probs.items():
                    cache[(excluded, f)] = value
            return cache[key]

        combined = {lam: np.full(self.n, np.nan) for lam in LAMBDA_GRID}
        for g in inner:
            excluded = tuple(sorted(base + (g,)))
            meta_ranks = pd.Series(probs_of(excluded, g)).rank(pct=True).to_numpy(dtype=np.float64)
            rank_mean = self.block[np.ix_(self.rows[g], list(pool))].mean(axis=1)
            for lam in LAMBDA_GRID:
                combined[lam][self.rows[g]] = lam * meta_ranks + (1.0 - lam) * rank_mean
        inner_rows = np.flatnonzero(np.isin(self.folds, inner))
        y_inner = self.labels[inner_rows]
        aucs = [float(roc_auc_score(y_inner, combined[lam][inner_rows])) for lam in LAMBDA_GRID]
        best = LAMBDA_GRID[int(np.argmax(aucs))]
        meta_ranks = pd.Series(probs_of(base, outer_fold)).rank(pct=True).to_numpy(dtype=np.float64)
        rank_mean = self.block[np.ix_(self.rows[outer_fold], list(pool))].mean(axis=1)
        prediction = best * meta_ranks + (1.0 - best) * rank_mean
        return prediction, float(best), aucs

    def evaluate(self, pool: tuple[int, ...], scope: int | None, surrogate: bool) -> dict[str, Any]:
        started = time.time()
        cache: dict[tuple[tuple[int, ...], int], np.ndarray] = {}
        fits: list[int] = []
        nested = np.full(self.n, np.nan)
        fold_auc, lambdas, lambda_aucs = {}, {}, {}
        for f in FOLDS:
            if f == scope:
                continue
            prediction, lam, aucs = self.outer(pool, scope, f, cache, fits)
            nested[self.rows[f]] = prediction
            fold_auc[str(f)] = float(roc_auc_score(self.labels[self.rows[f]], prediction))
            lambdas[str(f)] = lam
            lambda_aucs[str(f)] = aucs
        scope_rows = np.flatnonzero(self.folds != scope) if scope is not None else np.arange(self.n)
        auc = float(roc_auc_score(self.labels[scope_rows], nested[scope_rows]))
        result = {
            "pool": list(pool),
            "scope": scope,
            "auc": auc,
            "fold_auc": fold_auc,
            "lambda": lambdas,
            "lambda_auc": lambda_aucs,
            "fits": len(fits),
            "lbfgs_iterations": fits,
            "seconds": time.time() - started,
        }
        if surrogate:
            ranks = pd.DataFrame(np.asarray(self.pred[np.ix_(scope_rows, list(pool))])).rank(pct=True)
            result["surrogate_rank_mean_auc"] = float(
                roc_auc_score(self.labels[scope_rows], ranks.to_numpy().mean(axis=1))
            )
        return result

    def held_out(self, pool: tuple[int, ...], fold: int) -> dict[str, Any]:
        cache: dict[tuple[tuple[int, ...], int], np.ndarray] = {}
        fits: list[int] = []
        prediction, lam, aucs = self.outer(pool, None, fold, cache, fits)
        return {
            "pool": list(pool),
            "fold": fold,
            "lambda": lam,
            "lambda_auc": aucs,
            "prediction": prediction.tolist(),
            "auc": float(roc_auc_score(self.labels[self.rows[fold]], prediction)),
            "fits": len(fits),
        }


_WORKER: _Worker | None = None


def _worker() -> _Worker:
    global _WORKER
    if _WORKER is None:
        _WORKER = _Worker()
    return _WORKER


def _task_evaluate(args: tuple[tuple[int, ...], int | None, bool]) -> dict[str, Any]:
    pool, scope, surrogate = args
    with threadpool_limits(1):
        return _worker().evaluate(pool, scope, surrogate)


def _task_held_out(args: tuple[tuple[int, ...], int]) -> dict[str, Any]:
    pool, fold = args
    with threadpool_limits(1):
        return _worker().held_out(pool, fold)


# ---------------------------------------------------------------- search driver


class Search:
    def __init__(self, scope: int | None, executor: ProcessPoolExecutor, precommit: dict[str, Any]) -> None:
        self.scope = scope
        self.executor = executor
        self.precommit = precommit
        self.identity = precommit["identity_sha256"]
        self.names = [m["config"] for m in precommit["members"]]
        self.exclusive = tuple(precommit["exclusive_pair"])
        self.all_candidates = list(range(1, len(self.names)))
        self.dir = OUT / ("scope-full" if scope is None else f"scope-{scope}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.dir / "evaluations.jsonl"
        self.cache: dict[str, dict[str, Any]] = {}
        self.cache_hits = 0
        self.computed = 0
        self.spearman = np.load(OUT / "features/spearman.npy")
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                record = json.loads(line)
                if record["identity"] != self.identity:
                    raise RuntimeError("평가 캐시의 동결 신원이 다르다. 이어서 실행할 수 없다.")
                self.cache[record["key"]] = record["result"]
        self.state_path = self.dir / "state.json"
        if self.state_path.exists():
            self.state = _read_json(self.state_path)
            if self.state["identity"] != self.identity:
                raise RuntimeError("검색 상태의 동결 신원이 다르다. 이어서 실행할 수 없다.")
            self.state["resumed"] = self.state.get("resumed", 0) + 1
        else:
            self.state = {
                "identity": self.identity,
                "scope": scope,
                "stage": "init",
                "pool": [0],
                "auc": None,
                "steps": 0,
                "pair_adopted": None,
                "done": False,
                "resumed": 0,
                "wall_seconds": 0.0,
            }

    def key(self, pool: tuple[int, ...]) -> str:
        return ",".join(map(str, pool))

    def evaluate_many(self, pools: list[tuple[int, ...]], surrogate: bool) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        pending = []
        for pool in pools:
            k = self.key(pool)
            cached = self.cache.get(k)
            if cached is not None and (not surrogate or "surrogate_rank_mean_auc" in cached):
                results[k] = cached
                self.cache_hits += 1
            else:
                pending.append(pool)
        futures = {self.executor.submit(_task_evaluate, (pool, self.scope, surrogate)): pool for pool in pending}
        with self.cache_path.open("a") as f:
            for future in as_completed(futures):
                result = future.result()
                k = self.key(tuple(result["pool"]))
                self.cache[k] = result
                results[k] = result
                self.computed += 1
                f.write(json.dumps({"identity": self.identity, "key": k, "result": result}) + "\n")
                f.flush()
        return results

    def canonical(self, members: set[int]) -> tuple[int, ...]:
        return tuple(sorted(members))

    def moves(self, stage: str, pool: tuple[int, ...]) -> list[dict[str, Any]]:
        members = set(pool)
        out: list[dict[str, Any]] = []
        if stage.startswith("forward"):
            for c in self.all_candidates:
                if c in members:
                    continue
                partner = self.partner(c)
                if partner is not None and partner in members:
                    out.append({"kind": "swap", "out": partner, "in": c, "order": (c,), "pool": self.canonical(members - {partner} | {c})})
                else:
                    out.append({"kind": "add", "in": c, "order": (c,), "pool": self.canonical(members | {c})})
        elif stage.startswith("backward"):
            for m in pool:
                if m == 0:
                    continue
                out.append({"kind": "remove", "out": m, "order": (m,), "pool": self.canonical(members - {m})})
        elif stage == "pair":
            eligible = [
                c for c in self.all_candidates
                if c not in members and not (self.partner(c) is not None and self.partner(c) in members)
            ]
            for a, b in itertools.combinations(eligible, 2):
                if {a, b} == set(self.exclusive):
                    continue
                out.append({"kind": "pair", "in": [a, b], "order": (a, b), "pool": self.canonical(members | {a, b})})
        return out

    def partner(self, c: int) -> int | None:
        if c == self.exclusive[0]:
            return self.exclusive[1]
        if c == self.exclusive[1]:
            return self.exclusive[0]
        return None

    def check_duplicates(self, pool: tuple[int, ...]) -> float:
        best = 0.0
        for i, j in itertools.combinations(pool, 2):
            best = max(best, float(self.spearman[i, j]))
        if best >= DUPLICATE_THRESHOLD:
            raise RuntimeError(f"중복 불변식 위반: {pool} max spearman {best}")
        return best

    def save(self) -> None:
        _write_json(self.state_path, self.state)

    def run(self, max_steps: int | None) -> dict[str, Any]:
        state = self.state
        if state["stage"] == "init":
            initial = self.evaluate_many([(0,)], surrogate=False)["0"]
            state["auc"] = initial["auc"]
            state["stage"] = "forward1"
            state["initial"] = {"auc": initial["auc"], "fold_auc": initial["fold_auc"], "seconds": initial["seconds"]}
            self.save()
        order = ["forward1", "backward1", "forward2", "pair", "forward3", "backward2", "done"]
        steps_this_run = 0
        while not state["done"]:
            if max_steps is not None and steps_this_run >= max_steps:
                self.save()
                return {"interrupted_after_steps": steps_this_run, "state": state}
            stage = state["stage"]
            pool = tuple(state["pool"])
            moves = self.moves(stage, pool)
            step_started = time.time()
            results = self.evaluate_many([m["pool"] for m in moves], surrogate=stage.startswith("backward"))
            current_auc = state["auc"]
            current = self.cache[self.key(pool)]
            evaluated = []
            for m in moves:
                r = results[self.key(m["pool"])]
                delta = r["auc"] - current_auc
                wins = sum(1 for f, v in r["fold_auc"].items() if v > current["fold_auc"][f])
                evaluated.append({**{k: v for k, v in m.items() if k != "order"}, "order": list(m["order"]), "auc": r["auc"], "delta": delta, "fold_wins": wins, "fold_total": len(r["fold_auc"]), "lambda": r["lambda"], "seconds": r["seconds"], "fits": r["fits"], "surrogate_rank_mean_auc": r.get("surrogate_rank_mean_auc")})
            positive = [e for e in evaluated if e["delta"] > 0]
            chosen = None
            if positive:
                chosen = max(positive, key=lambda e: (e["delta"], [-x for x in e["order"]]))
            step = {
                "step": state["steps"],
                "stage": stage,
                "pool_before": list(pool),
                "auc_before": current_auc,
                "moves": sorted(evaluated, key=lambda e: (-e["delta"], e["order"])),
                "chosen": chosen,
                "seconds": time.time() - step_started,
                "warnings": [],
            }
            if stage.startswith("backward"):
                by_exact = sorted(evaluated, key=lambda e: (-e["auc"], e["order"]))
                by_surrogate = sorted(evaluated, key=lambda e: (-e["surrogate_rank_mean_auc"], e["order"]))
                if by_exact and by_surrogate:
                    surrogate_top = by_surrogate[0]
                    step["surrogate_comparison"] = {
                        "surrogate_top_move": surrogate_top["order"],
                        "exact_top_move": by_exact[0]["order"],
                        "surrogate_top_exact_rank": 1 + [e["order"] for e in by_exact].index(surrogate_top["order"]),
                        "surrogate_top_is_exact_top": surrogate_top["order"] == by_exact[0]["order"],
                        "surrogate_top_delta": surrogate_top["delta"],
                        "exact_top_delta": by_exact[0]["delta"],
                    }
            if chosen is not None:
                if abs(chosen["delta"]) < BAND_ABS:
                    step["warnings"].append(f"성능 동등 대역 안의 이동: delta {chosen['delta']:.3e} < {BAND_ABS:.3e}")
                if chosen["fold_wins"] * 2 <= chosen["fold_total"]:
                    step["warnings"].append(f"분할 승수 {chosen['fold_wins']}/{chosen['fold_total']}")
                step["max_spearman_after"] = self.check_duplicates(tuple(chosen["pool"]))
                state["pool"] = list(chosen["pool"])
                state["auc"] = chosen["auc"]
                if stage == "pair":
                    state["pair_adopted"] = chosen["in"]
                    state["stage"] = "forward3"
            else:
                if stage == "pair":
                    state["pair_adopted"] = False
                    state["stage"] = "done"
                elif stage == "backward2":
                    state["stage"] = "done"
                else:
                    state["stage"] = order[order.index(stage) + 1]
            state["done"] = state["stage"] == "done"
            step["pool_after"] = state["pool"]
            step["auc_after"] = state["auc"]
            step["stage_after"] = state["stage"]
            _write_json(self.dir / "steps" / f"{state['steps']:03d}.json", step)
            state["steps"] += 1
            steps_this_run += 1
            state["wall_seconds"] += time.time() - step_started
            self.save()
            print(f"[{self.dir.name}] step {state['steps']-1} {stage} k={len(pool)} moves={len(moves)} chosen={chosen and chosen['order']} delta={chosen and chosen['delta']} auc={state['auc']:.10f} pool={len(state['pool'])} {step['seconds']:.0f}s cache_hits={self.cache_hits} computed={self.computed}", flush=True)
        state["final_pool_names"] = [self.names[i] for i in state["pool"]]
        state["max_spearman_final"] = self.check_duplicates(tuple(state["pool"]))
        state["cache_hits_last_run"] = self.cache_hits
        state["computed_last_run"] = self.computed
        state["evaluations_total"] = len(self.cache)
        state["maxrss_mb"] = _maxrss_mb()
        self.save()
        return {"state": state}


def search(scopes: list[str], jobs: int, max_steps: int | None) -> None:
    precommit = _read_json(OUT / "precommit.json")
    if precommit["code_sha256"] != _sha256_file(Path(__file__)):
        raise RuntimeError("코드 해시가 동결 장부와 다르다. prepare를 다시 실행하거나 코드를 되돌려라.")
    for name, digest in precommit["features_sha256"].items():
        if _sha256_file(OUT / "features" / name) != digest:
            raise RuntimeError(f"특성 파일 해시 불일치: {name}")
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for scope_name in scopes:
            scope = None if scope_name == "full" else int(scope_name)
            outcome = Search(scope, executor, precommit).run(max_steps)
            print(json.dumps({k: v for k, v in outcome["state"].items() if k not in {"initial"}}, ensure_ascii=False))
            if "interrupted_after_steps" in outcome:
                return


# ---------------------------------------------------------------- finish


def _reference_nested(names: list[str], strategy: str, pred: pd.DataFrame, folds: pd.Series, labels: pd.Series, bands: pd.Series) -> E.NestedEvaluation:
    combiner = E.combiner_for_context(strategy, fold_of=folds, band_of=bands)
    return E.evaluate_nested(combiner, pred[names], folds, labels)


def equivalence(jobs: int) -> dict[str, Any]:
    """참조 구현(ensemble.evaluate_nested)과 시제품 채점의 동일성 확인."""
    precommit = _read_json(OUT / "precommit.json")
    names = [m["config"] for m in precommit["members"]]
    pred = pd.read_parquet(OUT / "predictions.parquet").set_index(ID)
    folds = pd.Series(np.load(OUT / "features/folds.npy"), index=pred.index)
    labels = pd.Series(np.load(OUT / "features/labels.npy"), index=pred.index)
    checks = []
    cases = [((0, 24), None), ((0, 9, 12, 16, 24), None), (tuple(range(0, 33, 3)), None), ((0, 9, 12, 16, 24), 2), ((0, 9, 12, 16, 24, 27, 30), 4)]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(_task_evaluate, (pool, scope, False)): (pool, scope) for pool, scope in cases}
        for future in as_completed(futures):
            pool, scope = futures[future]
            mine = future.result()
            mask = (folds != scope).to_numpy() if scope is not None else np.ones(len(folds), dtype=bool)
            with threadpool_limits(1):
                ref = _reference_nested([names[i] for i in pool], STRATEGY, pred[mask], folds[mask], labels[mask], folds[mask])
            checks.append({
                "pool": list(pool), "scope": scope,
                "reference_auc": ref.nested_auc, "prototype_auc": mine["auc"],
                "abs_diff": abs(ref.nested_auc - mine["auc"]),
                "reference_fold_auc": {str(o.fold): o.auc for o in ref.folds},
                "prototype_fold_auc": mine["fold_auc"],
                "reference_seconds": ref.elapsed_seconds, "prototype_seconds": mine["seconds"],
                "prototype_fits": mine["fits"],
            })
    return {"checks": checks, "max_abs_diff": max(c["abs_diff"] for c in checks)}


def finish(jobs: int) -> None:
    started = time.time()
    precommit = _read_json(OUT / "precommit.json")
    names = [m["config"] for m in precommit["members"]]
    current32 = tuple(range(1, 33))
    states = {s: _read_json(OUT / f"scope-{s}/state.json") for s in ["full", *map(str, FOLDS)]}
    for s, st in states.items():
        if not st["done"]:
            raise RuntimeError(f"scope {s} 검색이 끝나지 않았다.")
    proposal = tuple(states["full"]["pool"])
    scope_pools = {f: tuple(states[str(f)]["pool"]) for f in FOLDS}

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        tasks = {}
        for f in FOLDS:
            tasks[executor.submit(_task_held_out, (scope_pools[f], f))] = ("proposal", f)
            tasks[executor.submit(_task_held_out, (current32, f))] = ("current32", f)
        full_eval = {
            executor.submit(_task_evaluate, (current32, None, False)): "current32",
        }
        held = {"proposal": {}, "current32": {}}
        for future in as_completed(tasks):
            kind, f = tasks[future]
            held[kind][f] = future.result()
        current32_full = next(iter(full_eval)).result()
        equiv = equivalence(jobs)

    labels = np.load(OUT / "features/labels.npy")
    folds = np.load(OUT / "features/folds.npy")
    procedure = {}
    for kind in ("proposal", "current32"):
        nested = np.full(len(labels), np.nan)
        for f in FOLDS:
            nested[folds == f] = np.asarray(held[kind][f]["prediction"])
        assert np.isfinite(nested).all()
        procedure[kind] = {
            "auc": float(roc_auc_score(labels, nested)),
            "fold_auc": {str(f): held[kind][f]["auc"] for f in FOLDS},
            "lambda": {str(f): held[kind][f]["lambda"] for f in FOLDS},
            "fold_pool_size": {str(f): len(held[kind][f]["pool"]) for f in FOLDS},
        }
    procedure["delta"] = procedure["proposal"]["auc"] - procedure["current32"]["auc"]
    procedure["fold_wins"] = sum(1 for f in FOLDS if procedure["proposal"]["fold_auc"][str(f)] > procedure["current32"]["fold_auc"][str(f)])

    stability = {
        "member_retention": {names[m]: sum(1 for f in FOLDS if m in scope_pools[f]) for m in proposal},
        "fold_only_members": {str(f): [names[m] for m in scope_pools[f] if m not in proposal] for f in FOLDS},
        "fold_pool_size": {str(f): len(scope_pools[f]) for f in FOLDS},
        "fold_pools": {str(f): [names[m] for m in scope_pools[f]] for f in FOLDS},
    }

    pred = pd.read_parquet(OUT / "predictions.parquet").set_index(ID)
    folds_s = pd.Series(folds, index=pred.index)
    labels_s = pd.Series(labels, index=pred.index)
    bands = E.missingness_bands(ROOT / "data/train.csv", ROOT / "data/test.csv").reindex(pred.index).astype(np.int8)
    direct = {}
    t_direct = time.time()
    for kind, pool in (("proposal", proposal), ("current32", current32)):
        direct[kind] = {}
        for strategy in E.CANDIDATE_POOL_CORE_COMBINER_NAMES:
            ev = _reference_nested([names[i] for i in pool], strategy, pred, folds_s, labels_s, bands)
            direct[kind][strategy] = {"auc": ev.nested_auc, "fold_auc": {str(o.fold): o.auc for o in ev.folds}, "seconds": ev.elapsed_seconds}
        direct[kind]["best_strategy"] = max(direct[kind], key=lambda s: direct[kind][s]["auc"] if isinstance(direct[kind][s], dict) else -1)
        direct[kind]["best_auc"] = direct[kind][direct[kind]["best_strategy"]]["auc"]
    direct["best_delta"] = direct["proposal"]["best_auc"] - direct["current32"]["best_auc"]
    direct["per_strategy_delta"] = {s: direct["proposal"][s]["auc"] - direct["current32"][s]["auc"] for s in E.CANDIDATE_POOL_CORE_COMBINER_NAMES}
    direct["seconds"] = time.time() - t_direct
    direct["shrunk_reference_vs_prototype_full"] = {
        "proposal": abs(direct["proposal"][STRATEGY]["auc"] - states["full"]["auc"]),
        "current32": abs(direct["current32"][STRATEGY]["auc"] - current32_full["auc"]),
    }

    spearman = np.load(OUT / "features/spearman.npy")
    invariants = {}
    for label, pool in [("proposal", proposal), *[(f"scope-{f}", scope_pools[f]) for f in FOLDS], ("current32", current32)]:
        pairs = list(itertools.combinations(pool, 2))
        worst = max(pairs, key=lambda p: spearman[p[0], p[1]]) if pairs else None
        invariants[label] = {
            "size": len(pool),
            "max_spearman": float(spearman[worst[0], worst[1]]) if worst else None,
            "max_pair": [names[worst[0]], names[worst[1]]] if worst else None,
            "passes": bool(worst is None or spearman[worst[0], worst[1]] < DUPLICATE_THRESHOLD),
            "all_pairs": {f"{names[a]}|{names[b]}": float(spearman[a, b]) for a, b in pairs},
        }

    timings = {}
    for s, st in states.items():
        steps = [_read_json(p) for p in sorted((OUT / f"scope-{s}/steps").glob("*.json"))]
        evals = [json.loads(line)["result"] for line in (OUT / f"scope-{s}/evaluations.jsonl").read_text().splitlines()]
        by_stage: dict[str, dict[str, float]] = {}
        for step in steps:
            b = by_stage.setdefault(step["stage"], {"steps": 0, "moves": 0, "seconds": 0.0})
            b["steps"] += 1
            b["moves"] += len(step["moves"])
            b["seconds"] += step["seconds"]
        timings[s] = {
            "wall_seconds": st["wall_seconds"],
            "steps": st["steps"],
            "evaluations": len(evals),
            "fits": sum(e["fits"] for e in evals),
            "eval_seconds_sum": sum(e["seconds"] for e in evals),
            "eval_seconds_mean": float(np.mean([e["seconds"] for e in evals])),
            "eval_seconds_max": max(e["seconds"] for e in evals),
            "by_stage": by_stage,
            "resumed": st["resumed"],
            "maxrss_mb": st.get("maxrss_mb"),
            "final_auc": st["auc"],
            "final_pool_size": len(st["pool"]),
            "pair_adopted": st["pair_adopted"],
            "surrogate_comparisons": [
                {"step": step["step"], "stage": step["stage"], **step["surrogate_comparison"]}
                for step in steps if "surrogate_comparison" in step
            ],
            "warnings": [{"step": step["step"], "stage": step["stage"], "warnings": step["warnings"]} for step in steps if step["warnings"]],
            "accepted_moves": [
                {"step": step["step"], "stage": step["stage"], "kind": step["chosen"]["kind"], "move": step["chosen"]["order"], "names": [names[i] for i in step["chosen"]["order"]], "delta": step["chosen"]["delta"], "auc": step["chosen"]["auc"], "fold_wins": f"{step['chosen']['fold_wins']}/{step['chosen']['fold_total']}"}
                for step in steps if step["chosen"] is not None
            ],
        }

    report = {
        "identity_sha256": precommit["identity_sha256"],
        "code_sha256": precommit["code_sha256"],
        "git_commit": _git_commit(),
        "proposal_pool": [names[i] for i in proposal],
        "proposal_pool_size": len(proposal),
        "proposal_full_auc": states["full"]["auc"],
        "current32_full_auc": current32_full["auc"],
        "full_delta": states["full"]["auc"] - current32_full["auc"],
        "removed_from_current32": [names[i] for i in current32 if i not in proposal],
        "procedure": procedure,
        "stability": stability,
        "direct_core3": direct,
        "duplicate_invariants": invariants,
        "timings": timings,
        "equivalence": equiv,
        "finish_seconds": time.time() - started,
        "maxrss_mb": _maxrss_mb(),
        "gate": {
            "procedure_strictly_higher": procedure["delta"] > 0,
            "direct_best_strictly_higher": direct["best_delta"] > 0,
            "all_invariants_pass": all(v["passes"] for v in invariants.values()),
        },
    }
    _write_json(OUT / "report.json", report)
    print(json.dumps({k: report[k] for k in ("proposal_pool", "proposal_full_auc", "current32_full_auc", "full_delta", "gate")}, ensure_ascii=False, indent=1))
    print("procedure", json.dumps(procedure, indent=1))
    print("direct", json.dumps({k: v for k, v in direct.items() if k in ("best_delta", "per_strategy_delta")}, indent=1))
    print("equivalence max_abs_diff", equiv["max_abs_diff"])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--jobs", type=int, default=8)
    p = sub.add_parser("equivalence")
    p.add_argument("--jobs", type=int, default=6)
    p = sub.add_parser("search")
    p.add_argument("--scopes", default="full,0,1,2,3,4")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=None)
    p = sub.add_parser("finish")
    p.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.jobs)
    elif args.command == "equivalence":
        result = equivalence(args.jobs)
        _write_json(OUT / "equivalence.json", result)
        print(json.dumps(result, indent=1))
    elif args.command == "search":
        search(args.scopes.split(","), args.jobs, args.max_steps)
    elif args.command == "finish":
        finish(args.jobs)


if __name__ == "__main__":
    main()
