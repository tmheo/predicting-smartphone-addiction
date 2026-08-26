"""candidate-pool-v2 판정 기록 생성 경로의 사용자 계약."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from pipeline.ensemble import (
    CANDIDATE_POOL_CORE_COMBINER_NAMES,
    DEFAULT_COMBINER_NAMES,
)
from pipeline.judgment import JudgmentError, load_pool_admission_authorization
from pipeline.ledger import EntryEvidence, Pool, PoolMember
from pipeline.pool_judgment import (
    GenerationRequest,
    PoolJudgmentError,
    generate_pool_judgment,
)
from pipeline.pool_rereview import PoolScore
from pipeline.runs import InMemoryRunStore


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _DeterministicEvaluator:
    def __init__(self, context, *, mutation=None, nonpositive: bool = False) -> None:
        self.context = context
        self.mutation = mutation
        self.nonpositive = nonpositive
        self.seen_arms: list[tuple[tuple[str, ...], ...]] = []
        self._mutated = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def _auc(self, members: tuple[str, ...], excluded_fold: int | None) -> float:
        candidates = [name for name in members if name.startswith("candidate:")]
        if not candidates or self.nonpositive:
            return 0.5
        preferred = 0.9 if candidates[0].endswith("cand-b") else 0.8
        if excluded_fold is not None and excluded_fold % 2 == 0:
            preferred += 0.2 if candidates[0].endswith("cand-a") else 0.0
        return preferred

    def evaluate_many(self, arms, *, excluded_fold, capture_prediction=False):
        normalized = tuple(tuple(members) for members, _clone in arms)
        names = tuple(self.context.ledger["strategies"]["included"])
        self.seen_arms.append(normalized)
        if self.mutation is not None and not self._mutated:
            self.mutation()
            self._mutated = True
        scores = []
        mask = np.ones(len(self.context.folds), dtype=bool)
        if excluded_fold is not None:
            mask = self.context.folds.to_numpy() != excluded_fold
        labels = self.context.labels.to_numpy()[mask]
        folds = self.context.folds.to_numpy()[mask]
        for members in normalized:
            best_auc = self._auc(members, excluded_fold)
            fold_auc = {
                str(fold): best_auc for fold in sorted(np.unique(folds).astype(int))
            }
            strategy_auc = {
                name: best_auc - index * 1e-7
                for index, name in enumerate(names)
            }
            prediction = None
            if capture_prediction:
                has_candidate = any(
                    name.startswith("candidate:") for name in members
                )
                if has_candidate and not self.nonpositive:
                    prediction = labels.astype(np.float64) * 0.8 + 0.1
                else:
                    prediction = np.full(len(labels), 0.5, dtype=np.float64)
                best_auc = float(roc_auc_score(labels, prediction))
                fold_auc = {
                    str(fold): float(
                        roc_auc_score(labels[folds == fold], prediction[folds == fold])
                    )
                    for fold in sorted(np.unique(folds).astype(int))
                }
                strategy_auc = {
                    name: best_auc - index * 1e-7
                    for index, name in enumerate(names)
                }
            scores.append(
                PoolScore(
                    members=members,
                    strategy_auc=strategy_auc,
                    strategy_fold_auc={
                        name: fold_auc for name in names
                    },
                    best_strategy=names[0],
                    best_auc=best_auc,
                    best_fold_auc=fold_auc,
                    prediction=prediction,
                )
            )
        return scores

    def predict_outer(self, strategy, members, held_out_fold):
        del strategy
        mask = self.context.folds.to_numpy() == held_out_fold
        labels = self.context.labels.to_numpy()[mask]
        has_candidate = any(name.startswith("candidate:") for name in members)
        if has_candidate and not self.nonpositive:
            return labels.astype(np.float64) * 0.8 + 0.1
        return np.full(len(labels), 0.5, dtype=np.float64)


class _MixedScaleEvaluator:
    """분할별 순서는 같지만 결합 전략에 따라 예측 눈금만 다르게 만든다."""

    def __init__(self, context) -> None:
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def evaluate_many(self, arms, *, excluded_fold, capture_prediction=False):
        del capture_prediction
        names = tuple(self.context.ledger["strategies"]["included"])
        mask = np.ones(len(self.context.folds), dtype=bool)
        if excluded_fold is not None:
            mask = self.context.folds.to_numpy() != excluded_fold
        folds = self.context.folds.to_numpy()[mask]
        fold_auc = {str(fold): 1.0 for fold in sorted(np.unique(folds).astype(int))}
        scores = []
        for members, _clone in arms:
            has_candidate = any(name.startswith("candidate:") for name in members)
            best_strategy = (
                names[1]
                if not has_candidate and excluded_fold == 3
                else names[0]
            )
            best_auc = 0.9 if has_candidate else 0.8
            strategy_auc = {
                name: best_auc if name == best_strategy else best_auc - 0.1
                for name in names
            }
            scores.append(
                PoolScore(
                    members=tuple(members),
                    strategy_auc=strategy_auc,
                    strategy_fold_auc={name: fold_auc for name in names},
                    best_strategy=best_strategy,
                    best_auc=best_auc,
                    best_fold_auc=fold_auc,
                    prediction=None,
                )
            )
        return scores

    def predict_outer(self, strategy, members, held_out_fold):
        del members
        mask = self.context.folds.to_numpy() == held_out_fold
        labels = self.context.labels.to_numpy()[mask]
        if strategy == tuple(self.context.ledger["strategies"]["included"])[1]:
            return labels.astype(np.float64) * 0.001 + 0.001
        return labels.astype(np.float64) * 0.5 + 0.25


class _ReplayOnlyMixedScaleEvaluator(_MixedScaleEvaluator):
    def evaluate_many(self, *_args, **_kwargs):
        raise AssertionError("재생 경로가 전체 후보 평가를 다시 실행했다")


class _FailingEvaluator:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def evaluate_many(self, *_args, **_kwargs):
        raise RuntimeError("의도한 결합 평가 중단")

    def predict_outer(self, *_args, **_kwargs):
        raise AssertionError("도달하면 안 된다")


def _write_fixture(tmp_path: Path, *, candidates=("cand-a", "cand-b")):
    artifacts = tmp_path / "artifacts"
    data = tmp_path / "data"
    artifacts.mkdir()
    data.mkdir()
    ids = np.arange(100, 200)
    labels = np.tile([0, 1], 50)
    folds = np.arange(100) % 5
    train = pd.DataFrame(
        {"id": ids, "feature": np.linspace(0.0, 1.0, 100), "addicted_label": labels}
    )
    test = pd.DataFrame(
        {"id": np.arange(300, 320), "feature": np.linspace(0.0, 1.0, 20)}
    )
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": ids, "fold": folds}).to_parquet(
        artifacts / "folds.parquet", index=False
    )

    pool = Pool(
        members=[
            PoolMember(
                run_id="pool-run",
                config="pool-model",
                oof_auc=0.5,
                seeds=[42, 43, 44],
                entered_at="2026-08-23",
                reason="시험 기준 구성원",
                evidence=EntryEvidence(
                    champion_run_id="pool-run",
                    champion_oof_auc=0.5,
                    floor_margin=0.01,
                    nearest_run_id=None,
                    nearest_spearman=None,
                    ensemble_auc_with=None,
                    ensemble_auc_without=None,
                    contribution=None,
                ),
            )
        ]
    )
    pool.save(artifacts / "pool.yaml")

    fold_sha = _sha256(artifacts / "folds.parquet")
    store = InMemoryRunStore()
    pool_prediction = pd.DataFrame(
        {"id": ids, "pred": np.full(len(ids), 0.5, dtype=np.float64)}
    )
    store.add_run("pool-run", oof=pool_prediction)
    for index, run_id in enumerate(candidates):
        prediction = labels.astype(np.float64) * (0.7 + index * 0.1) + 0.1
        store.add_run(
            run_id,
            params={
                "experiment": run_id,
                "seeds": "42,43,44",
                "features": "placebo_noise",
            },
            metrics={"auc_oof": float(roc_auc_score(labels, prediction))},
            tags={"git_dirty": "False", "sha256.folds": fold_sha},
            oof=pd.DataFrame({"id": ids, "pred": prediction}),
            importance=pd.DataFrame(
                {"feature": ["placebo_noise"], "fold": [0], "seed": [42], "gain": [0.0]}
            ),
        )
    return store


def _request(*, candidates=("cand-a",), action="admission", replaces=None):
    return GenerationRequest(
        judgment_id="test-pool-judgment",
        action=action,
        candidate_run_ids=tuple(candidates),
        model_lineage_group="test-lineage",
        selection_description="결과를 보기 전에 고정한 시험 후보",
        output_path=Path("artifacts/judgments/test-pool-judgment.yaml"),
        evidence_path=Path("run-logs/test-pool-judgment/evidence.json"),
        replaces_run_id=replaces,
    )


def _generate(
    tmp_path,
    request,
    store,
    *,
    nonpositive=False,
    mutation=None,
    names_provider=None,
    evaluator_factory=None,
):
    return generate_pool_judgment(
        request,
        store=store,
        repo_root=tmp_path,
        pool_path=Path("artifacts/pool.yaml"),
        folds_path=Path("artifacts/folds.parquet"),
        train_path=Path("data/train.csv"),
        test_path=Path("data/test.csv"),
        evaluator_factory=(
            evaluator_factory
            or (
                lambda context, _jobs: _DeterministicEvaluator(
                    context, mutation=mutation, nonpositive=nonpositive
                )
            )
        ),
        registered_combiner_names_provider=names_provider,
    )


def test_single_candidate_record_is_consumed_without_conversion(monkeypatch, tmp_path):
    store = _write_fixture(tmp_path, candidates=("cand-a",))

    result = _generate(tmp_path, _request(), store)

    assert result.state == "adopted"
    monkeypatch.chdir(tmp_path)
    authorization = load_pool_admission_authorization(
        Path("artifacts/judgments/test-pool-judgment.yaml"),
        candidate_run_id="cand-a",
        candidate_config="cand-a",
        pool_path=Path("artifacts/pool.yaml"),
        folds_path=Path("artifacts/folds.parquet"),
    )
    assert authorization.action == "admission"
    assert authorization.contract_version == "candidate-pool-v2"
    assert authorization.nested_oof_delta > 0.0
    record = yaml.safe_load(result.record_path.read_text())
    assert record["contract_version"] == "candidate-pool-v2"
    assert record["selection"]["kind"] == "precommitted_single"
    assert record["selection"]["combiner_scope"] == "core"
    assert record["frozen_input"]["registered_combiners"]["scope"] == "core"
    assert record["frozen_input"]["registered_combiners"]["names"] == list(
        CANDIDATE_POOL_CORE_COMBINER_NAMES
    )
    assert result.input_snapshot_path.is_file()


def test_full_combiner_scope_keeps_all_optional_strategies(monkeypatch, tmp_path):
    store = _write_fixture(tmp_path, candidates=("cand-a",))
    request = replace(_request(), combiner_scope="full")

    result = _generate(tmp_path, request, store)

    record = yaml.safe_load(result.record_path.read_text())
    assert record["selection"]["combiner_scope"] == "full"
    assert record["frozen_input"]["registered_combiners"]["scope"] == "full"
    assert record["frozen_input"]["registered_combiners"]["names"] == list(
        DEFAULT_COMBINER_NAMES
    )
    monkeypatch.chdir(tmp_path)
    authorization = load_pool_admission_authorization(
        Path("artifacts/judgments/test-pool-judgment.yaml"),
        candidate_run_id="cand-a",
        candidate_config="cand-a",
        pool_path=Path("artifacts/pool.yaml"),
        folds_path=Path("artifacts/folds.parquet"),
    )
    assert authorization.contract_version == "candidate-pool-v2"


def test_multiple_variants_repeat_selection_inside_each_outer_fold(tmp_path):
    store = _write_fixture(tmp_path)

    result = _generate(tmp_path, _request(candidates=("cand-a", "cand-b")), store)

    record = yaml.safe_load(result.record_path.read_text())
    evidence = yaml.safe_load(result.evidence_path.read_text())
    assert result.state == "adopted"
    assert record["selection"]["kind"] == "nested_selection"
    assert record["change"]["candidate"]["run_id"] == "cand-b"
    assert [candidate["run_id"] for candidate in record["selection"]["candidates"]] == [
        "cand-a",
        "cand-b",
    ]
    assert (
        record["nested_oof_comparison"]["after"]["strategy_role"]
        == "full_oof_final_selection_reference"
    )
    choices = evidence["evaluation"]["outer_fold_choices"]
    assert set(choices) == {"0", "1", "2", "3", "4"}
    assert {choice["candidate_run_id"] for choice in choices.values()} == {
        "cand-a",
        "cand-b",
    }
    assert all(choice["strategy_count"] == 3 for choice in choices.values())


def test_multiple_variants_normalize_mixed_outer_strategy_scales(tmp_path):
    store = _write_fixture(tmp_path)

    result = _generate(
        tmp_path,
        _request(candidates=("cand-a", "cand-b")),
        store,
        evaluator_factory=lambda context, _jobs: _MixedScaleEvaluator(context),
    )

    record = yaml.safe_load(result.record_path.read_text())
    evidence = yaml.safe_load(result.evidence_path.read_text())
    comparison = record["nested_oof_comparison"]
    assert result.state == "rejected"
    assert comparison["delta"] == 0.0
    assert comparison["outer_fold_delta"] == {str(fold): 0.0 for fold in range(5)}
    assert comparison["before"]["scale_normalized"] is True
    assert comparison["after"]["scale_normalized"] is False
    assert evidence["evaluation"]["scale_normalization"] == {
        "method": "within_outer_fold_percentile_rank",
        "before": True,
        "after": False,
    }


def test_selection_replay_verifies_source_and_only_recomputes_outer_predictions(
    tmp_path,
):
    store = _write_fixture(tmp_path)
    source = _generate(
        tmp_path,
        _request(candidates=("cand-a", "cand-b")),
        store,
        evaluator_factory=lambda context, _jobs: _MixedScaleEvaluator(context),
    )
    replay_request = replace(
        _request(candidates=("cand-a", "cand-b")),
        judgment_id="test-pool-judgment-replay",
        output_path=Path("artifacts/judgments/test-pool-judgment-replay.yaml"),
        evidence_path=Path("run-logs/test-pool-judgment-replay/evidence.json"),
        replay_selection_path=source.record_path.relative_to(tmp_path),
    )

    replay = _generate(
        tmp_path,
        replay_request,
        store,
        evaluator_factory=lambda context, _jobs: _ReplayOnlyMixedScaleEvaluator(
            context
        ),
    )

    source_record = yaml.safe_load(source.record_path.read_text())
    replay_record = yaml.safe_load(replay.record_path.read_text())
    replay_evidence = yaml.safe_load(replay.evidence_path.read_text())
    assert replay.state == source.state == "rejected"
    assert replay_record["nested_oof_comparison"] == source_record[
        "nested_oof_comparison"
    ]
    assert replay_evidence["evaluation"]["mode"] == "nested_selection_replay"
    assert replay_evidence["evaluation"]["replay_source"]["record"]["path"] == str(
        source.record_path.relative_to(tmp_path)
    )


def test_replacement_compares_current_pool_with_atomic_replacement(tmp_path):
    store = _write_fixture(tmp_path, candidates=("cand-a",))
    request = _request(action="replacement", replaces="pool-run")

    result = _generate(tmp_path, request, store)

    record = yaml.safe_load(result.record_path.read_text())
    evidence = yaml.safe_load(result.evidence_path.read_text())
    assert record["change"]["action"] == "replacement"
    assert record["change"]["replaces_run_id"] == "pool-run"
    assert evidence["evaluation"]["after"]["members"] == ["cand-a"]
    assert result.state == "adopted"


def test_restoration_requires_a_changed_pool_or_combiner_baseline(monkeypatch, tmp_path):
    store = _write_fixture(tmp_path, candidates=("cand-a",))
    original_path = tmp_path / "artifacts/judgments/original-removal.yaml"
    original_path.parent.mkdir()
    original_path.write_text(
        yaml.safe_dump(
            {
                "contract_version": "candidate-pool-v2",
                "status": "adopted",
                "frozen_input": {
                    "candidate_pool": {
                        "sha256": _sha256(tmp_path / "artifacts/pool.yaml")
                    },
                    "registered_combiners": {
                        "names_sha256": hashlib.sha256(
                            (
                                json.dumps(
                                    list(CANDIDATE_POOL_CORE_COMBINER_NAMES),
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            ).encode()
                        ).hexdigest()
                    },
                },
                "removals": [{"member": {"run_id": "cand-a"}}],
            },
            sort_keys=False,
        )
    )
    request = replace(
        _request(action="restoration"),
        restores_judgment_path=Path("artifacts/judgments/original-removal.yaml"),
    )

    unchanged = _generate(tmp_path, request, store)

    assert unchanged.state == "indeterminate"
    unchanged.record_path.unlink()
    unchanged.evidence_path.unlink()
    original = yaml.safe_load(original_path.read_text())
    original["frozen_input"]["candidate_pool"]["sha256"] = "0" * 64
    original_path.write_text(yaml.safe_dump(original, sort_keys=False))

    restored = _generate(tmp_path, request, store)

    assert restored.state == "adopted"
    monkeypatch.chdir(tmp_path)
    authorization = load_pool_admission_authorization(
        Path("artifacts/judgments/test-pool-judgment.yaml"),
        candidate_run_id="cand-a",
        candidate_config="cand-a",
        pool_path=Path("artifacts/pool.yaml"),
        folds_path=Path("artifacts/folds.parquet"),
    )
    assert authorization.action == "restoration"


def test_missing_original_removal_record_cannot_restore(tmp_path):
    store = _write_fixture(tmp_path, candidates=("cand-a",))
    request = replace(
        _request(action="restoration"),
        restores_judgment_path=Path("artifacts/judgments/missing-removal.yaml"),
    )

    result = _generate(tmp_path, request, store)

    record = yaml.safe_load(result.record_path.read_text())
    assert result.state == "indeterminate"
    assert record["result"]["decision"] == "no_change"
    assert record["change"]["restores_judgment"]["sha256"] is None


def test_nonpositive_delta_never_creates_adopted_result(monkeypatch, tmp_path):
    store = _write_fixture(tmp_path, candidates=("cand-a",))

    result = _generate(tmp_path, _request(), store, nonpositive=True)

    record = yaml.safe_load(result.record_path.read_text())
    assert result.state == "rejected"
    assert record["result"]["state"] == "rejected"
    monkeypatch.chdir(tmp_path)
    with pytest.raises(JudgmentError):
        load_pool_admission_authorization(
            Path("artifacts/judgments/test-pool-judgment.yaml"),
            candidate_run_id="cand-a",
            candidate_config="cand-a",
            pool_path=Path("artifacts/pool.yaml"),
            folds_path=Path("artifacts/folds.parquet"),
        )


@pytest.mark.parametrize("changed", ["pool", "folds", "combiners"])
def test_frozen_input_change_refuses_every_output(tmp_path, changed):
    store = _write_fixture(tmp_path, candidates=("cand-a",))
    names = [CANDIDATE_POOL_CORE_COMBINER_NAMES]

    def mutate():
        if changed == "pool":
            (tmp_path / "artifacts/pool.yaml").write_text("members: []\n")
        elif changed == "folds":
            (tmp_path / "artifacts/folds.parquet").write_bytes(b"changed")
        else:
            names[0] = CANDIDATE_POOL_CORE_COMBINER_NAMES[:-1]

    with pytest.raises(PoolJudgmentError, match="동결 입력"):
        _generate(
            tmp_path,
            _request(),
            store,
            mutation=mutate,
            names_provider=lambda: names[0],
        )

    assert not (tmp_path / "artifacts/judgments/test-pool-judgment.yaml").exists()
    assert not (tmp_path / "run-logs/test-pool-judgment").exists()


def test_missing_candidate_artifact_remains_indeterminate(tmp_path):
    store = _write_fixture(tmp_path, candidates=())
    store.add_run(
        "cand-a",
        params={
            "experiment": "cand-a",
            "seeds": "42,43,44",
            "features": "placebo_noise",
        },
        metrics={"auc_oof": 0.8},
        tags={
            "git_dirty": "False",
            "sha256.folds": _sha256(tmp_path / "artifacts/folds.parquet"),
        },
    )
    pool_before = (tmp_path / "artifacts/pool.yaml").read_bytes()

    result = _generate(tmp_path, _request(), store)

    record = yaml.safe_load(result.record_path.read_text())
    assert result.state == "indeterminate"
    assert record["result"]["state"] == "indeterminate"
    assert "산출물" in record["result"]["reason"]
    assert (tmp_path / "artifacts/pool.yaml").read_bytes() == pool_before
    assert result.input_snapshot_path is None


def test_combiner_execution_failure_remains_indeterminate(tmp_path):
    store = _write_fixture(tmp_path, candidates=("cand-a",))
    pool_before = (tmp_path / "artifacts/pool.yaml").read_bytes()

    result = _generate(
        tmp_path,
        _request(),
        store,
        evaluator_factory=lambda _context, _jobs: _FailingEvaluator(),
    )

    record = yaml.safe_load(result.record_path.read_text())
    assert result.state == "indeterminate"
    assert record["result"]["decision"] == "no_change"
    assert "의도한 결합 평가 중단" in record["result"]["reason"]
    assert (tmp_path / "artifacts/pool.yaml").read_bytes() == pool_before


def test_existing_immutable_record_is_not_overwritten(tmp_path):
    store = _write_fixture(tmp_path, candidates=("cand-a",))
    request = _request()
    first = _generate(tmp_path, request, store)
    before = first.record_path.read_bytes()

    with pytest.raises(PoolJudgmentError, match="이미 있다"):
        _generate(tmp_path, replace(request, selection_description="바뀐 설명"), store)

    assert first.record_path.read_bytes() == before
