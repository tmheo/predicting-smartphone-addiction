"""풀 진입 판정·채택 자격 검사 테스트. (ADR 0001 계열 2, 지도 #91의 #95)

후보는 InMemoryRunStore에 심고 load_candidate로 읽어, 태그 규약(git_dirty,
sha256.folds)의 해석까지 함께 검증한다. 라벨은 판정 함수에 직접 넘기므로
대회 데이터 파일 없이 돈다. 진입 하한·중복 게이트(교체와 탈락)·기여 참고값을
각각 겨냥하고, 중복 게이트의 탈락이 조기 확정(최종 논리곱이 아님)이라는
의미를 교체 경로로 고정한다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipeline.ensemble import DEFAULT_COMBINER_NAMES
from pipeline.judgment import (
    JudgmentError,
    canonical_name_list_sha256,
    check_adoption_eligibility,
    judge_entry,
    load_candidate,
    load_pool_admission_authorization,
)
from pipeline.ledger import Champion, EntryEvidence, Pool, PoolMember
from pipeline.runs import InMemoryRunStore

IDS = list(range(1, 9))
Y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=pd.Index(IDS, name="id"))

# 완벽한 순서의 예측(AUC 1.0)과, 인접 순위 하나가 뒤집힌 예측(AUC < 1.0, 스피어만 < 0.998).
PERFECT = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
IMPERFECT = [0.1, 0.2, 0.3, 0.5, 0.4, 0.6, 0.7, 0.8]


def oof_frame(preds: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"id": IDS, "pred": preds})


def make_store_with_member(member_preds: list[float]) -> InMemoryRunStore:
    store = InMemoryRunStore()
    store.add_run("m1", oof=oof_frame(member_preds))
    return store


def make_candidate(
    store: InMemoryRunStore, *, auc_oof: float, preds: list[float], seeds: str = "42,43,44"
):
    store.add_run(
        "cand",
        params={"experiment": "exp_test", "seeds": seeds},
        metrics={"auc_oof": auc_oof},
        tags={"git_dirty": "False", "sha256.folds": "folds-sha"},
        oof=oof_frame(preds),
    )
    return load_candidate("cand", store)


def make_champion() -> Champion:
    return Champion(
        run_id="champ",
        oof_auc=0.97000,
        seed_aucs={},
        fold_aucs={},
        config="exp_champ",
        features=set(),
        git_commit="cafebabe",
        adopted_at="2026-08-13",
        reason="테스트 champion",
    )


def member(run_id: str, oof_auc: float) -> PoolMember:
    """진입 판정이 읽는 건 run_id와 oof_auc뿐이고, 나머지는 장부 기록 형식 채우기다."""
    return PoolMember(
        run_id=run_id,
        config=f"exp_{run_id}",
        oof_auc=oof_auc,
        seeds=[42, 43, 44],
        entered_at="2026-08-11",
        reason="테스트 구성원",
        evidence=EntryEvidence(
            champion_run_id="champ",
            champion_oof_auc=0.97000,
            floor_margin=0.0,
            nearest_run_id=None,
            nearest_spearman=None,
            ensemble_auc_with=None,
            ensemble_auc_without=None,
            contribution=None,
        ),
    )


def pool_with(members: list[PoolMember]) -> Pool:
    return Pool(members=members)


@pytest.mark.parametrize(("auc_oof", "admit"), [(0.96000, True), (0.95999, False)])
def test_entry_floor_gate_on_empty_pool(auc_oof, admit):
    # champion 0.97 − 0.01 = 0.96이 진입 하한. 풀이 비면 다른 판정은 묻지 않는다.
    store = InMemoryRunStore()
    candidate = make_candidate(store, auc_oof=auc_oof, preds=PERFECT)
    verdict = judge_entry(pool_with([]), candidate, make_champion(), store, Y)
    assert verdict.floor == pytest.approx(0.96)
    assert verdict.duplicate is None and verdict.contribution is None
    assert verdict.admit is admit


def test_entry_positive_contribution_admits():
    # 완벽한 후보가 뒤집힌 구성원의 앙상블을 끌어올린다 → 기여 양수, 진입.
    store = make_store_with_member(IMPERFECT)
    candidate = make_candidate(store, auc_oof=0.96900, preds=PERFECT)
    verdict = judge_entry(
        pool_with([member("m1", 0.96500)]), candidate, make_champion(), store, Y
    )
    assert not verdict.duplicate.duplicate
    assert verdict.contribution.contribution > 0
    assert verdict.drop_run_id is None
    assert verdict.admit


def test_entry_negative_contribution_is_diagnostic_and_admits():
    # 균등 순위 평균을 망쳐도 다른 결합 전략에는 도움이 될 수 있어 기여값은 진입을 막지 않는다.
    store = make_store_with_member(IMPERFECT)
    candidate = make_candidate(store, auc_oof=0.96900, preds=PERFECT[::-1])
    verdict = judge_entry(
        pool_with([member("m1", 0.96500)]), candidate, make_champion(), store, Y
    )
    assert not verdict.duplicate.duplicate
    assert verdict.contribution.contribution < 0
    assert verdict.admit


def test_entry_duplicate_with_lower_auc_is_early_rejection():
    # 예측 순위가 같으면(스피어만 1.0) 중복이고, 기존 구성원이 더 높으면 그 자리에서 탈락.
    store = make_store_with_member(IMPERFECT)
    candidate = make_candidate(store, auc_oof=0.96400, preds=IMPERFECT)
    verdict = judge_entry(
        pool_with([member("m1", 0.96500)]), candidate, make_champion(), store, Y
    )
    assert verdict.duplicate.duplicate and not verdict.duplicate.replace
    assert verdict.drop_run_id is None
    assert verdict.contribution is None  # 기여 참고값까지 계산하지 않는다.
    assert not verdict.admit


def test_entry_duplicate_with_higher_auc_replaces_member():
    # 후보가 더 높으면 기존 구성원을 교체 대상으로 정하고, 남은 풀이 없으면 기여는 묻지
    # 않는다. 중복 게이트는 최종 논리곱이 아니므로 진입은 하한만으로 정해진다.
    store = make_store_with_member(IMPERFECT)
    candidate = make_candidate(store, auc_oof=0.96600, preds=IMPERFECT)
    verdict = judge_entry(
        pool_with([member("m1", 0.96500)]), candidate, make_champion(), store, Y
    )
    assert verdict.duplicate.replace
    assert verdict.drop_run_id == "m1"
    assert verdict.contribution is None
    assert verdict.admit


def test_entry_replacement_measures_contribution_against_remaining_pool():
    # 두 구성원 중 하나를 교체하면 기여는 남은 구성원 기준으로 잰다.
    store = make_store_with_member(IMPERFECT)
    store.add_run("m2", oof=oof_frame([0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7]))
    candidate = make_candidate(store, auc_oof=0.96600, preds=IMPERFECT)
    members = [member("m1", 0.96500), member("m2", 0.96300)]
    verdict = judge_entry(pool_with(members), candidate, make_champion(), store, Y)
    assert verdict.drop_run_id == "m1"
    assert verdict.contribution is not None
    assert verdict.admit is verdict.floor_ok


def test_adoption_eligibility_passes_clean_three_seed_run():
    eligibility = check_adoption_eligibility(
        seeds=[42, 43, 44], git_dirty=False, folds_sha256="abc", committed_folds_sha256="abc"
    )
    assert eligibility.ok


@pytest.mark.parametrize(
    ("seeds", "git_dirty", "folds_sha256"),
    [
        ([42], False, "abc"),  # 단일 시드는 장부에 못 오른다.
        ([42, 43, 44], True, "abc"),  # git_dirty 실행 거부. (#14)
        ([42, 43, 44], False, "stale"),  # 커밋된 folds와 sha256 불일치 거부.
    ],
)
def test_adoption_eligibility_rejects_each_condition(seeds, git_dirty, folds_sha256):
    eligibility = check_adoption_eligibility(
        seeds=seeds, git_dirty=git_dirty, folds_sha256=folds_sha256,
        committed_folds_sha256="abc",
    )
    assert not eligibility.ok


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pool_judgment(tmp_path: Path, *, delta: float = 0.00001) -> Path:
    pool_path = tmp_path / "pool.yaml"
    folds_path = tmp_path / "folds.parquet"
    evidence_path = tmp_path / "evidence.json"
    pool_path.write_text("members: []\n")
    folds_path.write_bytes(b"folds")
    evidence_path.write_text('{"nested_oof": "complete"}\n')
    record = {
        "schema_version": 1,
        "judgment_id": "test-candidate-pool-admission",
        "contract_version": "candidate-pool-v1",
        "change": {
            "action": "admission",
            "candidate": {
                "run_id": "cand",
                "config": "exp_test",
                "model_lineage_group": "exp_test",
            },
            "replaces_run_id": None,
        },
        "selection": {
            "kind": "precommitted_single",
            "description": "결과를 보기 전에 고정한 후보 하나",
        },
        "frozen_input": {
            "candidate_pool": {"sha256": _sha256(pool_path)},
            "folds": {"sha256": _sha256(folds_path)},
            "registered_combiners": {
                "names": list(DEFAULT_COMBINER_NAMES),
                "names_sha256": canonical_name_list_sha256(DEFAULT_COMBINER_NAMES),
            },
        },
        "nested_oof_comparison": {
            "before": {"strategy": "rank_mean", "auc": 0.96},
            "after": {"strategy": "rank_mean", "auc": 0.96 + delta},
            "delta": delta,
            "outer_fold_delta": {
                0: delta,
                1: -delta,
                2: delta,
                3: -delta,
                4: delta,
            },
            "outer_fold_wins": 3,
            "boundary_contribution": 0.0 < delta <= 0.000027669802,
        },
        "evidence": {
            "path": "evidence.json",
            "sha256": _sha256(evidence_path),
        },
        "result": {"state": "adopted", "decision": "admit"},
    }
    path = tmp_path / "judgment.yaml"
    path.write_text(yaml.safe_dump(record, allow_unicode=True, sort_keys=False))
    return path


def test_pool_admission_requires_current_positive_nested_oof_judgment(
    monkeypatch, tmp_path
):
    path = _write_pool_judgment(tmp_path)
    monkeypatch.chdir(tmp_path)

    authorization = load_pool_admission_authorization(
        Path("judgment.yaml"),
        candidate_run_id="cand",
        candidate_config="exp_test",
        pool_path=Path("pool.yaml"),
        folds_path=Path("folds.parquet"),
    )

    assert authorization.judgment_id == "test-candidate-pool-admission"
    assert authorization.nested_oof_delta == pytest.approx(0.00001)
    assert authorization.boundary_contribution
    assert authorization.replaced_run_id is None


def test_pool_admission_rejects_stale_pool_judgment(monkeypatch, tmp_path):
    _write_pool_judgment(tmp_path)
    (tmp_path / "pool.yaml").write_text("members:\n- changed\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(JudgmentError, match="동결 후보 풀 해시"):
        load_pool_admission_authorization(
            Path("judgment.yaml"),
            candidate_run_id="cand",
            candidate_config="exp_test",
            pool_path=Path("pool.yaml"),
            folds_path=Path("folds.parquet"),
        )


def test_pool_admission_rejects_nonpositive_nested_oof(monkeypatch, tmp_path):
    _write_pool_judgment(tmp_path, delta=-0.00001)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(JudgmentError, match="전체 차이가 양수가 아니다"):
        load_pool_admission_authorization(
            Path("judgment.yaml"),
            candidate_run_id="cand",
            candidate_config="exp_test",
            pool_path=Path("pool.yaml"),
            folds_path=Path("folds.parquet"),
        )
