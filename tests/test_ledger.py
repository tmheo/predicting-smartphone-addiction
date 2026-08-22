"""장부 타입 round-trip 테스트. (지도 #91의 #96)

커밋된 실제 장부(artifacts/champion.yaml·pool.yaml)를 load 후 save 했을 때 바이트가
동일해야 한다: YAML 모양(키 순서·int 키·시드 문자열·따옴표·줄바꿈)의 소유가 ledger
한곳임을 고정하는 특성화 테스트다. 합성 왕복은 YAML을 거친 값 보존(seed·fold의
int 키 정규화, features 집합, 진입 근거의 None 필드)을 검증한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pipeline.ledger import (
    Champion,
    EntryEvidence,
    Pool,
    PoolJudgmentPointer,
    PoolMember,
)

REPO = Path(__file__).resolve().parent.parent


def test_champion_committed_ledger_roundtrips_bytes(tmp_path):
    committed = REPO / "artifacts" / "champion.yaml"
    saved = tmp_path / "champion.yaml"
    Champion.load(committed).save(saved)
    assert saved.read_bytes() == committed.read_bytes()


def test_pool_committed_ledger_roundtrips_bytes(tmp_path):
    committed = REPO / "artifacts" / "pool.yaml"
    saved = tmp_path / "pool.yaml"
    Pool.load(committed).save(saved)
    assert saved.read_bytes() == committed.read_bytes()


def test_pool_is_frozen_35_member_order_with_only_decided_removals():
    baseline = yaml.safe_load(
        (REPO / "artifacts" / "pool-baseline-2026-08-21.yaml").read_text()
    )
    removed = {
        "exp033_recon_orig_mean_top3_raw",
        "exp107_logreg_onehot_nn10",
        "exp108_logreg_onehot_nn10_l1",
    }
    expected = [
        (member["config"], member["run_id"])
        for member in baseline["members"]
        if member["config"] not in removed
    ]
    actual = [
        (member.config, member.run_id)
        for member in Pool.load(REPO / "artifacts" / "pool.yaml").members
    ]

    assert len(actual) == 32
    assert actual == expected


def test_pool_reduction_judgment_matches_committed_32_member_result():
    judgment = yaml.safe_load(
        (
            REPO / "artifacts" / "judgments" / "issue346-pool-reduction.yaml"
        ).read_text()
    )
    pool_configs = [
        member.config
        for member in Pool.load(REPO / "artifacts" / "pool.yaml").members
    ]

    assert judgment["contract_version"] == "candidate-pool-v1"
    assert judgment["frozen_input"]["candidate_pool"]["member_count"] == 35
    assert judgment["final_result"]["members"] == pool_configs
    assert judgment["final_result"]["member_count"] == 32
    assert judgment["final_result"]["full_refit_count_after"] == 94
    assert sum(
        removal["full_refit_reduction"] for removal in judgment["removals"]
    ) == 5
    for removal in judgment["removals"]:
        comparison = removal["comparison"]
        assert comparison["after_auc"] - comparison["before_auc"] == pytest.approx(
            comparison["delta"], abs=1e-15
        )
        for fold in range(5):
            assert (
                comparison["outer_fold_auc_after"][fold]
                - comparison["outer_fold_auc_before"][fold]
            ) == pytest.approx(comparison["outer_fold_delta"][fold], abs=1e-15)


def test_champion_load_normalizes_string_keys(tmp_path):
    # 손으로 고친 장부가 seed·fold 키를 문자열로 남겨도 load가 int로 정규화한다.
    path = tmp_path / "champion.yaml"
    path.write_text(
        "run_id: r1\n"
        "oof_auc: 0.9\n"
        "seed_aucs:\n  '42': 0.91\n"
        "fold_aucs:\n  '0': 0.92\n"
        "config: exp_test\n"
        "features: age,placebo_noise\n"
        "git_commit: cafebabe\n"
        "adopted_at: '2026-08-13'\n"
        "reason: 정규화 테스트\n"
    )
    champion = Champion.load(path)
    assert champion.seed_aucs == {42: pytest.approx(0.91)}
    assert champion.fold_aucs == {0: pytest.approx(0.92)}
    assert champion.features == {"age", "placebo_noise"}


def test_pool_roundtrips_values(tmp_path):
    pool = Pool(
        members=[
            PoolMember(
                run_id="r1",
                config="exp_test",
                oof_auc=0.965,
                seeds=[42, 43, 44],
                entered_at="2026-08-13",
                reason="합성 왕복 테스트",
                evidence=EntryEvidence(
                    champion_run_id="champ",
                    champion_oof_auc=0.97,
                    floor_margin=0.005,
                    nearest_run_id=None,
                    nearest_spearman=None,
                    ensemble_auc_with=None,
                    ensemble_auc_without=None,
                    contribution=None,
                ),
                judgment=PoolJudgmentPointer(
                    judgment_id="test-admission",
                    contract_version="candidate-pool-v1",
                    path="artifacts/judgments/test-admission.yaml",
                    sha256="a" * 64,
                ),
            )
        ]
    )
    path = tmp_path / "pool.yaml"
    pool.save(path)
    assert Pool.load(path) == pool


def test_pool_load_missing_file_is_empty(tmp_path):
    assert Pool.load(tmp_path / "없는_풀.yaml") == Pool(members=[])
