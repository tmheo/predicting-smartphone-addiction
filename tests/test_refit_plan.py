"""재학습 계획 장부의 문법·근거 계보·예산 재계산 관문 테스트. (#373)

전용 시험 장부 하나를 만들어 두고, 각 시험이 그 사본을 한 곳만 망가뜨려
관문이 정확히 그 이유로 막는지 본다. 현행 `artifacts/full-refit-plan.yaml`은
아직 이 문법이 아니므로 여기서 읽지 않는다(자료 전환은 후속 이슈).

실행 저장소는 메모리 adapter를 넣어 MLflow 없이 같은 계보 검증을 통과시킨다.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from pipeline.ledger import EntryEvidence, Pool, PoolMember
from pipeline.refit_plan import (
    ExecutableRefitPlan,
    RefitPlan,
    RefitPlanError,
)
from pipeline.runs import InMemoryRunStore, sha256_of
from pipeline.training_length import (
    FIXED_COUNT,
    ONE_BASED_COUNT,
    ZERO_BASED_POSITION,
    derive_refit_budgets,
    observe_training_length,
)

POOL_SHA256 = "pool-sha256-of-the-test-ledger"
COMBINER = "shrunk_rank_logit_logistic"
EVIDENCE_ARTIFACT = "model_training_diagnostics.json"
SEEDS = (42, 43, 44)

# exp135_xgb_hpo_trial30의 확정 원시 위치(#327). 0부터 세는 위치라 +1해야 9759/10394/10369이 된다.
XGB_RAW = {
    42: [[8491], [7806], [7382], [7488], [8320]],
    43: [[8314], [7608], [7032], [8450], [8404]],
    44: [[8057], [8484], [7157], [8294], [8442]],
}
# exp127_lookup_muon의 확정 원시 위치(#327). 분할마다 내부 구성원 3개를 기록한다.
LOOKUP_RAW = {
    42: [[11, 11, 9], [11, 9, 9], [11, 11, 9], [9, 9, 9], [9, 9, 11]],
    43: [[11, 9, 11], [11, 11, 11], [11, 11, 11], [9, 9, 9], [11, 9, 11]],
    44: [[11, 11, 11], [11, 9, 11], [11, 11, 11], [11, 9, 11], [11, 11, 9]],
}
# 고정 일정 계열. 설정이 고정한 4 epoch도 근거로 적어 같은 계산을 거친다.
REALMLP_RAW = {seed: [[4]] * 5 for seed in SEEDS}


def flatten(raw_table: dict[int, list[list[int]]]):
    """시드 -> 바깥쪽 분할 -> 내부 구성원 원시 값 표를 좌표 순서대로 편다."""
    for seed, folds in raw_table.items():
        for outer_fold, inner_values in enumerate(folds):
            for inner_member, raw_value in enumerate(inner_values):
                yield (
                    seed,
                    outer_fold,
                    None if len(inner_values) == 1 else inner_member,
                    raw_value,
                )


def evidence_block(
    model_family: str, converter: str, raw_field: str, raw_table
) -> dict:
    from pipeline.training_length import observed_length_from_raw

    return {
        "status": "confirmed",
        "model_family": model_family,
        "converter": converter,
        "observations": [
            {
                "seed": seed,
                "outer_fold": outer_fold,
                "inner_member": inner_member,
                "raw_field": raw_field,
                "raw_value": raw_value,
                "raw_meaning": converter,
                "observed_training_length": observed_length_from_raw(
                    raw_value, converter
                ),
            }
            for seed, outer_fold, inner_member, raw_value in flatten(raw_table)
        ],
    }


def derivation_block(converter: str, raw_field: str, raw_table) -> dict:
    """공통 계산부가 내는 값을 그대로 적는다. 관문은 이 값을 믿지 않고 다시 계산한다."""
    derivation = derive_refit_budgets(
        [
            observe_training_length(
                seed=seed,
                outer_fold=outer_fold,
                raw_field=raw_field,
                raw_value=raw_value,
                raw_meaning=converter,
                inner_member=inner_member,
            )
            for seed, outer_fold, inner_member, raw_value in flatten(raw_table)
        ]
    )
    return {
        "statistic": "median",
        "multiplier": 1.25,
        "rounding": "half_up",
        "seeds": [
            {
                "seed": seed.seed,
                "observed_lengths": list(seed.observed_lengths),
                "median": seed.median,
                "scaled": seed.scaled,
                "budget": seed.budget,
            }
            for seed in derivation.seeds
        ],
    }


def lineage_block(run_id: str, config_path: Path, config_bytes: bytes) -> dict:
    return {
        "source_run_id": run_id,
        "source_git_commit": f"commit-{run_id}",
        "source_config_path": str(config_path),
        "source_config_sha256": sha256_of(config_bytes),
        "evidence_artifact_path": EVIDENCE_ARTIFACT,
        "evidence_artifact_sha256": sha256_of(evidence_bytes(run_id)),
    }


def evidence_bytes(run_id: str) -> bytes:
    """원시 근거 산출물의 자리. 관문은 형식을 해석하지 않고 내용 해시만 맞춰 본다."""
    return f'[{{"run_id": "{run_id}"}}]\n'.encode()


@dataclass
class Ledger:
    """전용 시험 장부와 그 관문 입력(실행 저장소, 후보 풀)."""

    path: Path
    document: dict
    store: InMemoryRunStore
    pool: Pool
    pool_sha256: str = POOL_SHA256

    def write(self, document: dict | None = None) -> Path:
        target = self.path
        target.write_text(
            yaml.safe_dump(
                self.document if document is None else document,
                allow_unicode=True,
                sort_keys=False,
            )
        )
        return target

    def load(self, document: dict | None = None) -> RefitPlan:
        return RefitPlan.load(self.write(document))

    def validate(self, document: dict | None = None) -> ExecutableRefitPlan:
        return self.load(document).validate_for_refit(
            store=self.store, pool=self.pool, pool_sha256=self.pool_sha256
        )

    def edited(self) -> dict:
        return copy.deepcopy(self.document)

    def member(self, document: dict, config: str) -> dict:
        (found,) = [item for item in document["members"] if item["config"] == config]
        return found


@dataclass
class MemberSpec:
    config: str
    run_id: str
    seeds: tuple[int, ...]
    model_family: str | None = None
    converter: str | None = None
    raw_field: str = ""
    raw_table: dict = field(default_factory=dict)


SPECS = (
    MemberSpec(
        config="exp135_xgb_hpo_trial30",
        run_id="run-xgb",
        seeds=SEEDS,
        model_family="xgboost",
        converter=ZERO_BASED_POSITION,
        raw_field="best_iteration",
        raw_table=XGB_RAW,
    ),
    MemberSpec(
        config="exp127_lookup_muon",
        run_id="run-lookup",
        seeds=SEEDS,
        model_family="lookup_transformer",
        converter=ZERO_BASED_POSITION,
        raw_field="best_epoch",
        raw_table=LOOKUP_RAW,
    ),
    MemberSpec(
        config="exp140_realmlp_orig_cdf_diff",
        run_id="run-realmlp",
        seeds=SEEDS,
        model_family="realmlp",
        converter=FIXED_COUNT,
        raw_field="fixed_epochs",
        raw_table=REALMLP_RAW,
    ),
    # 반복 수가 없는 구성원. lbfgs라 시드 하나만 재학습한다. (ADR 0002)
    MemberSpec(config="exp058_logreg_onehot", run_id="run-logreg", seeds=(42,)),
)


def pool_member(spec: MemberSpec) -> PoolMember:
    return PoolMember(
        run_id=spec.run_id,
        config=spec.config,
        oof_auc=0.969,
        seeds=list(SEEDS),
        entered_at="2026-08-24",
        reason="시험용 장부",
        evidence=EntryEvidence(
            champion_run_id="run-champion",
            champion_oof_auc=0.969,
            floor_margin=0.0001,
            nearest_run_id=None,
            nearest_spearman=None,
            ensemble_auc_with=None,
            ensemble_auc_without=None,
            contribution=None,
        ),
    )


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    store = InMemoryRunStore()
    members = []
    for spec in SPECS:
        config_path = tmp_path / f"{spec.config}.yaml"
        config_path.write_text(f"name: {spec.config}\n")
        config_bytes = config_path.read_bytes()
        store.add_run(
            spec.run_id,
            run_name=spec.config,
            tags={"git_commit": f"commit-{spec.run_id}", "git_dirty": "False"},
            artifacts={
                config_path.name: config_bytes,
                EVIDENCE_ARTIFACT: evidence_bytes(spec.run_id),
            },
        )
        if spec.model_family is None:
            evidence = {
                "status": "not_applicable",
                "model_family": "logistic_onehot",
                "converter": None,
                "observations": [],
            }
            derivation = {
                "statistic": "median",
                "multiplier": 1.25,
                "rounding": "half_up",
                "seeds": [
                    {
                        "seed": seed,
                        "observed_lengths": [],
                        "median": None,
                        "scaled": None,
                        "budget": None,
                    }
                    for seed in spec.seeds
                ],
            }
        else:
            evidence = evidence_block(
                spec.model_family, spec.converter, spec.raw_field, spec.raw_table
            )
            derivation = derivation_block(
                spec.converter, spec.raw_field, spec.raw_table
            )
        members.append(
            {
                "config": spec.config,
                "config_path": str(config_path),
                "lineage": lineage_block(spec.run_id, config_path, config_bytes),
                "training_length_evidence": evidence,
                "refit_budget_derivation": derivation,
            }
        )
    document = {
        "schema_version": 2,
        "source_pool_sha256": POOL_SHA256,
        "protocol": {
            "iteration_multiplier": 1.25,
            "budget_statistic": "median",
            "budget_rounding": "half_up",
            "cv_model_weight": 5,
            "full_model_weight": 1,
            "combiner": COMBINER,
        },
        "members": members,
    }
    return Ledger(
        path=tmp_path / "refit-plan.yaml",
        document=document,
        store=store,
        pool=Pool(members=[pool_member(spec) for spec in SPECS]),
    )


# 통과 경로


def test_validated_plan_derives_confirmed_budgets(ledger: Ledger):
    """확정 사례의 예산이 원시 근거에서 다시 나온다."""
    plan = ledger.validate()

    assert plan.member("exp135_xgb_hpo_trial30").budgets == {
        42: 9759,
        43: 10394,
        44: 10369,
    }
    assert plan.member("exp127_lookup_muon").budgets == {42: 13, 43: 15, 44: 15}
    assert plan.member("exp140_realmlp_orig_cdf_diff").budgets == {42: 5, 43: 5, 44: 5}
    assert plan.member("exp058_logreg_onehot").budgets == {42: None}


def test_validated_plan_exposes_every_intermediate_value(ledger: Ledger):
    (seed_derivation, *_) = ledger.validate().member("exp135_xgb_hpo_trial30").derivation.seeds

    assert seed_derivation.seed == 42
    assert seed_derivation.observed_lengths == (8492, 7807, 7383, 7489, 8321)
    assert seed_derivation.median == 7807.0
    assert seed_derivation.scaled == 9758.75
    assert seed_derivation.budget == 9759


def test_only_validated_plan_exposes_refit_budgets(ledger: Ledger):
    """검증 전 계획은 실행 예산을 제공하지 않는다."""
    plan = ledger.load()

    assert not hasattr(plan, "budgets")
    assert all(not hasattr(member, "budgets") for member in plan.members)
    assert ledger.validate().budgets()["exp135_xgb_hpo_trial30"][42] == 9759


# 문법


@pytest.mark.parametrize(
    "field",
    ["override_budget", "manual_budget", "budget_override", "manual_note"],
)
def test_load_rejects_hand_edited_budget_fields(ledger: Ledger, field: str):
    document = ledger.edited()
    ledger.member(document, "exp135_xgb_hpo_trial30")[field] = 9758

    with pytest.raises(RefitPlanError, match="손으로 바꾸는"):
        ledger.load(document)


def test_load_rejects_unknown_field(ledger: Ledger):
    document = ledger.edited()
    ledger.member(document, "exp135_xgb_hpo_trial30")["budget_source"] = "fold_median"

    with pytest.raises(RefitPlanError, match="알 수 없는 필드"):
        ledger.load(document)


def test_load_rejects_missing_field(ledger: Ledger):
    document = ledger.edited()
    del ledger.member(document, "exp135_xgb_hpo_trial30")["lineage"]

    with pytest.raises(RefitPlanError, match="필수 필드가 없다"):
        ledger.load(document)


def test_load_rejects_other_schema_version(ledger: Ledger):
    document = ledger.edited()
    document["schema_version"] = 1

    with pytest.raises(RefitPlanError, match="장부 문법 판본"):
        ledger.load(document)


def test_load_rejects_non_integer_raw_value(ledger: Ledger):
    document = ledger.edited()
    observations = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "training_length_evidence"
    ]["observations"]
    observations[0]["raw_value"] = 8491.0

    with pytest.raises(RefitPlanError, match="정수여야 한다"):
        ledger.load(document)


def test_load_rejects_unknown_status(ledger: Ledger):
    document = ledger.edited()
    ledger.member(document, "exp135_xgb_hpo_trial30")["training_length_evidence"][
        "status"
    ] = "probably_fine"

    with pytest.raises(RefitPlanError, match="알 수 없는 근거 상태"):
        ledger.load(document)


# 근거 상태와 좌표


def test_unresolved_evidence_blocks_execution(ledger: Ledger):
    document = ledger.edited()
    ledger.member(document, "exp127_lookup_muon")["training_length_evidence"][
        "status"
    ] = "unresolved"

    with pytest.raises(RefitPlanError, match="미확정"):
        ledger.validate(document)


def test_missing_coordinate_is_rejected(ledger: Ledger):
    document = ledger.edited()
    evidence = ledger.member(document, "exp127_lookup_muon")["training_length_evidence"]
    del evidence["observations"][4]

    with pytest.raises(RefitPlanError, match="근거 좌표가 빠졌다"):
        ledger.validate(document)


def test_duplicate_coordinate_is_rejected(ledger: Ledger):
    document = ledger.edited()
    evidence = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "training_length_evidence"
    ]
    evidence["observations"].append(copy.deepcopy(evidence["observations"][0]))

    with pytest.raises(RefitPlanError, match="근거 좌표가 중복됐다"):
        ledger.validate(document)


def test_extra_coordinate_is_rejected(ledger: Ledger):
    document = ledger.edited()
    evidence = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "training_length_evidence"
    ]
    extra = copy.deepcopy(evidence["observations"][0])
    extra["outer_fold"] = 5
    evidence["observations"].append(extra)

    with pytest.raises(RefitPlanError, match="바깥쪽 분할 좌표"):
        ledger.validate(document)


def test_unplanned_seed_coordinate_is_rejected(ledger: Ledger):
    document = ledger.edited()
    evidence = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "training_length_evidence"
    ]
    extra = copy.deepcopy(evidence["observations"][0])
    extra["seed"] = 45
    evidence["observations"].append(extra)

    with pytest.raises(RefitPlanError, match="기대하지 않은 근거 좌표"):
        ledger.validate(document)


def test_non_positive_observation_is_rejected(ledger: Ledger):
    document = ledger.edited()
    observation = ledger.member(document, "exp140_realmlp_orig_cdf_diff")[
        "training_length_evidence"
    ]["observations"][0]
    observation["raw_value"] = 0
    observation["observed_training_length"] = 0

    with pytest.raises(RefitPlanError, match="1 이상"):
        ledger.validate(document)


def test_reconverted_raw_value_must_match_recorded_length(ledger: Ledger):
    document = ledger.edited()
    observation = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "training_length_evidence"
    ]["observations"][0]
    observation["observed_training_length"] = observation["raw_value"]

    with pytest.raises(RefitPlanError, match="다시 변환한"):
        ledger.validate(document)


def test_model_family_and_converter_must_agree(ledger: Ledger):
    document = ledger.edited()
    evidence = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "training_length_evidence"
    ]
    evidence["converter"] = ONE_BASED_COUNT

    with pytest.raises(RefitPlanError, match="변환기는"):
        ledger.validate(document)


def test_observation_raw_meaning_must_match_declared_converter(ledger: Ledger):
    document = ledger.edited()
    evidence = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "training_length_evidence"
    ]
    evidence["observations"][0]["raw_meaning"] = ONE_BASED_COUNT

    with pytest.raises(RefitPlanError, match="계열이 선언한 변환기와 다르다"):
        ledger.validate(document)


def test_unregistered_model_family_is_rejected(ledger: Ledger):
    document = ledger.edited()
    ledger.member(document, "exp135_xgb_hpo_trial30")["training_length_evidence"][
        "model_family"
    ] = "brand_new_family"

    with pytest.raises(RefitPlanError, match="변환기를 등록하지 않은"):
        ledger.validate(document)


# 계보


def test_unknown_source_run_is_rejected(ledger: Ledger):
    document = ledger.edited()
    ledger.member(document, "exp135_xgb_hpo_trial30")["lineage"][
        "source_run_id"
    ] = "run-does-not-exist"

    with pytest.raises(RefitPlanError, match="실행 ID가 후보 풀과 다르다"):
        ledger.validate(document)


def test_source_run_missing_from_store_is_rejected(ledger: Ledger):
    document = ledger.edited()
    pool = Pool(members=copy.deepcopy(ledger.pool.members))
    pool.members[0] = PoolMember(**{**pool.members[0].__dict__, "run_id": "run-gone"})
    ledger.pool = pool
    ledger.member(document, "exp135_xgb_hpo_trial30")["lineage"][
        "source_run_id"
    ] = "run-gone"

    with pytest.raises(RefitPlanError, match="실행 저장소에서 확인하지 못했다"):
        ledger.validate(document)


def test_source_revision_mismatch_is_rejected(ledger: Ledger):
    document = ledger.edited()
    ledger.member(document, "exp135_xgb_hpo_trial30")["lineage"][
        "source_git_commit"
    ] = "commit-somewhere-else"

    with pytest.raises(RefitPlanError, match="소스 판본"):
        ledger.validate(document)


def test_config_hash_mismatch_is_rejected(ledger: Ledger):
    document = ledger.edited()
    ledger.member(document, "exp135_xgb_hpo_trial30")["lineage"][
        "source_config_sha256"
    ] = "0" * 64

    with pytest.raises(RefitPlanError, match="설정 해시"):
        ledger.validate(document)


def test_evidence_artifact_hash_mismatch_is_rejected(ledger: Ledger):
    document = ledger.edited()
    ledger.member(document, "exp135_xgb_hpo_trial30")["lineage"][
        "evidence_artifact_sha256"
    ] = "0" * 64

    with pytest.raises(RefitPlanError, match="원시 근거 산출물 해시"):
        ledger.validate(document)


def test_missing_evidence_artifact_is_rejected(ledger: Ledger):
    document = ledger.edited()
    ledger.member(document, "exp135_xgb_hpo_trial30")["lineage"][
        "evidence_artifact_path"
    ] = "no_such_artifact.json"

    with pytest.raises(RefitPlanError, match="실행 저장소에서 확인하지 못했다"):
        ledger.validate(document)


def test_missing_config_file_is_rejected(ledger: Ledger, tmp_path: Path):
    document = ledger.edited()
    ledger.member(document, "exp135_xgb_hpo_trial30")["config_path"] = str(
        tmp_path / "gone.yaml"
    )

    with pytest.raises(RefitPlanError, match="설정 파일이 없다"):
        ledger.validate(document)


# 현재 후보 풀


def test_pool_content_hash_mismatch_is_rejected(ledger: Ledger):
    ledger.pool_sha256 = "another-pool-sha256"

    with pytest.raises(RefitPlanError, match="후보 풀 SHA-256"):
        ledger.validate()


def test_member_order_mismatch_is_rejected(ledger: Ledger):
    document = ledger.edited()
    document["members"][0], document["members"][1] = (
        document["members"][1],
        document["members"][0],
    )

    with pytest.raises(RefitPlanError, match="구성원 순서나 실행 ID"):
        ledger.validate(document)


def test_member_config_mismatch_is_rejected(ledger: Ledger):
    document = ledger.edited()
    ledger.member(document, "exp135_xgb_hpo_trial30")["config"] = "exp999_unknown"

    with pytest.raises(RefitPlanError, match="구성원 순서나 실행 ID"):
        ledger.validate(document)


def test_seeds_must_match_the_candidate_pool(ledger: Ledger):
    document = ledger.edited()
    seeds = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "refit_budget_derivation"
    ]["seeds"]
    del seeds[-1]

    with pytest.raises(RefitPlanError, match="계획 시드가 후보 풀 시드와 맞지 않는다"):
        ledger.validate(document)


# 계산 규약과 저장 예산


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("budget_statistic", "mean", "통계량"),
        ("iteration_multiplier", 1.3, "배수는 1.25"),
        ("budget_rounding", "half_even", "사사오입 방식"),
    ],
)
def test_protocol_must_follow_the_recorded_convention(
    ledger: Ledger, key: str, value, message: str
):
    document = ledger.edited()
    document["protocol"][key] = value

    with pytest.raises(RefitPlanError, match=message):
        ledger.validate(document)


def test_member_derivation_convention_must_match_the_protocol(ledger: Ledger):
    document = ledger.edited()
    ledger.member(document, "exp135_xgb_hpo_trial30")["refit_budget_derivation"][
        "multiplier"
    ] = 1.5

    with pytest.raises(RefitPlanError, match="저장 계산 규약"):
        ledger.validate(document)


def test_unregistered_combiner_is_rejected(ledger: Ledger):
    document = ledger.edited()
    document["protocol"]["combiner"] = "unregistered_combiner"

    with pytest.raises(RefitPlanError, match="등록되지 않은 결합 방식"):
        ledger.validate(document)


def test_hand_edited_budget_is_rejected(ledger: Ledger):
    """원시 근거를 그대로 두고 예산만 고치면 다시 계산한 값과 어긋난다."""
    document = ledger.edited()
    seeds = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "refit_budget_derivation"
    ]["seeds"]
    seeds[0]["budget"] = 9758

    with pytest.raises(RefitPlanError, match="저장 재학습 예산"):
        ledger.validate(document)


def test_hand_edited_median_is_rejected(ledger: Ledger):
    document = ledger.edited()
    seeds = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "refit_budget_derivation"
    ]["seeds"]
    seeds[0]["median"] = 7806.0

    with pytest.raises(RefitPlanError, match="저장 중앙값"):
        ledger.validate(document)


def test_hand_edited_scaled_value_is_rejected(ledger: Ledger):
    document = ledger.edited()
    seeds = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "refit_budget_derivation"
    ]["seeds"]
    seeds[0]["scaled"] = 9757.5

    with pytest.raises(RefitPlanError, match="저장 배수 적용값"):
        ledger.validate(document)


def test_recorded_observed_lengths_must_match_the_evidence(ledger: Ledger):
    document = ledger.edited()
    seeds = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "refit_budget_derivation"
    ]["seeds"]
    seeds[0]["observed_lengths"][0] = 8491

    with pytest.raises(RefitPlanError, match="저장 관측 학습 길이 목록"):
        ledger.validate(document)


# 반복 수가 없는 구성원


def test_non_iterative_member_may_not_carry_a_budget(ledger: Ledger):
    document = ledger.edited()
    seeds = ledger.member(document, "exp058_logreg_onehot")["refit_budget_derivation"][
        "seeds"
    ]
    seeds[0]["budget"] = 30

    with pytest.raises(RefitPlanError, match="재학습 예산이 있다"):
        ledger.validate(document)


def test_non_iterative_member_may_not_carry_observations(ledger: Ledger):
    document = ledger.edited()
    member = ledger.member(document, "exp058_logreg_onehot")
    member["training_length_evidence"]["observations"] = copy.deepcopy(
        ledger.member(document, "exp140_realmlp_orig_cdf_diff")[
            "training_length_evidence"
        ]["observations"]
    )

    with pytest.raises(RefitPlanError, match="원시 관측이 있다"):
        ledger.validate(document)


def test_iterative_member_may_not_carry_an_empty_budget(ledger: Ledger):
    document = ledger.edited()
    seeds = ledger.member(document, "exp140_realmlp_orig_cdf_diff")[
        "refit_budget_derivation"
    ]["seeds"]
    seeds[0]["budget"] = None

    with pytest.raises(RefitPlanError, match="저장 재학습 예산"):
        ledger.validate(document)


def test_confirmed_member_without_observations_is_rejected(ledger: Ledger):
    document = ledger.edited()
    ledger.member(document, "exp140_realmlp_orig_cdf_diff")[
        "training_length_evidence"
    ]["observations"] = []

    with pytest.raises(RefitPlanError, match="원시 관측이 없다"):
        ledger.validate(document)
