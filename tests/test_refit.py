"""전체 자료 재학습 실행 경로가 검증된 재학습 예산만 읽는지 본다. (#375)

장부의 문법과 관문 자체는 `tests/test_refit_plan.py`가 시험한다. 여기서는 그 관문이
실행 명령에 어떻게 걸려 있는지를 본다.

- 잘못된 장부로는 자료 적재도, 피처 계획도, 모델 연결부 생성도, 학습도 일어나지 않는다.
- 정상 장부에서는 원시 근거에서 다시 계산한 예산이 그대로 모델 연결부에 닿는다.
- 시드별 기록과 구성원 manifest에 원시 근거 계보와 파생 규약이 남고,
  재개는 그 묶음과 계획 내용 해시까지 다시 맞춰 본다.

커밋된 `artifacts/full-refit-plan.yaml`의 근거와 계산은 그대로 두고 계보만 메모리
실행 저장소로 옮겨 시험한다. 그래서 여기서 확인하는 숫자는 실제 재학습이 읽을 숫자와
같은 원시 관측에서 나온다. 후보 풀과 그 내용 해시는 커밋된 장부를 그대로 쓴다.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pipeline import data, refit, refit_plan
from pipeline.config import DataConfig, ExperimentConfig, FeatureConfig, ModelConfig
from pipeline.ledger import POOL_PATH, Pool
from pipeline.refit import mix_member_predictions
from pipeline.refit_plan import (
    ExecutableRefitMember,
    ExecutableRefitPlan,
    MemberLineage,
    RefitProtocol,
)
from pipeline.runs import InMemoryRunStore, sha256_of

COMMITTED_PLAN = Path("artifacts/full-refit-plan.yaml")

# 이슈 #375의 완료 조건이 이름을 짚은 예산. 원시 근거에서 다시 계산해 여기에 닿아야 한다.
CRITERION_BUDGETS = {
    "exp135_xgb_hpo_trial30": {42: 9759, 43: 10394, 44: 10369},
    "exp157_lookup_muon_initavg8": {42: 14, 43: 15, 44: 15},
    "exp081_lookup_fold_initialization_avg3": {42: 15, 43: 15, 44: 15},
    "exp059_lookup_transformer": {42: 15, 43: 15, 44: 18},
    "exp133_scalar_token_transformer_oof_te": {42: 20, 43: 18, 44: 15},
    # 반복 수가 없는 구성원. 시드 하나를 `None` 예산으로 재학습한다.
    "exp058_logreg_onehot": {42: None},
}


def evidence_bytes(config: str) -> bytes:
    """원시 근거 산출물의 자리. 관문은 형식을 해석하지 않고 내용 해시만 맞춰 본다."""
    return f'{{"config": "{config}"}}\n'.encode()


@dataclass
class MemoryLedger:
    """커밋된 장부와, 그 관문을 통과시킬 메모리 실행 저장소·후보 풀."""

    path: Path
    document: dict
    store: InMemoryRunStore
    pool: Pool
    pool_sha256: str

    def write(self, document: dict | None = None) -> Path:
        self.path.write_text(
            yaml.safe_dump(
                self.document if document is None else document,
                allow_unicode=True,
                sort_keys=False,
            )
        )
        return self.path

    def executable(self, document: dict | None = None) -> ExecutableRefitPlan:
        return refit.load_executable_plan(
            self.write(document),
            store=self.store,
            pool=self.pool,
            pool_sha256=self.pool_sha256,
        )

    def edited(self) -> dict:
        return copy.deepcopy(self.document)

    def member(self, document: dict, config: str) -> dict:
        (found,) = [item for item in document["members"] if item["config"] == config]
        return found

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """운영 기본값 자리에 이 장부의 실행 저장소와 후보 풀을 끼운다.

        명령행 진입점은 저장소도 풀도 주입받지 않으므로, 끝에서 끝까지 시험은
        기본값 자리를 바꿔 같은 관문을 메모리에서 통과시킨다.
        """
        real_sha256 = refit_plan.data.file_sha256
        monkeypatch.setattr(refit_plan, "MlflowRunStore", lambda: self.store)
        monkeypatch.setattr(refit_plan.Pool, "load", lambda: self.pool)
        monkeypatch.setattr(
            refit_plan.data,
            "file_sha256",
            lambda path: (
                self.pool_sha256 if Path(path) == POOL_PATH else real_sha256(path)
            ),
        )


@pytest.fixture
def ledger(tmp_path: Path) -> MemoryLedger:
    """커밋된 장부의 근거와 계산은 그대로 두고 계보만 메모리 저장소로 옮긴다.

    당시 실행 이후 설정 파일이 바뀐 구성원이 있어 계보의 설정 해시는 지금 파일 내용으로
    다시 적는다. 관측 학습 길이와 예산 계산 부분은 커밋된 장부 그대로이므로, 관문이
    돌려주는 숫자는 실제 재학습이 읽을 숫자와 같다.
    """
    document = yaml.safe_load(COMMITTED_PLAN.read_text())
    store = InMemoryRunStore()
    for member in document["members"]:
        lineage = member["lineage"]
        config_bytes = Path(lineage["source_config_path"]).read_bytes()
        evidence = evidence_bytes(member["config"])
        lineage["source_config_sha256"] = sha256_of(config_bytes)
        lineage["evidence_artifact_sha256"] = sha256_of(evidence)
        store.add_run(
            lineage["source_run_id"],
            run_name=member["config"],
            tags={"git_commit": lineage["source_git_commit"], "git_dirty": "False"},
            artifacts={
                Path(lineage["source_config_path"]).name: config_bytes,
                lineage["evidence_artifact_path"]: evidence,
            },
        )
    return MemoryLedger(
        path=tmp_path / "full-refit-plan.yaml",
        document=document,
        store=store,
        pool=Pool.load(),
        pool_sha256=data.file_sha256(POOL_PATH),
    )


TOY_TRAIN = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "x": [0.0, 1.0, 2.0, 3.0],
        "social_media_hours": [0.5, 1.0, 1.5, 2.0],
        "addicted_label": [0, 0, 1, 1],
    }
)
TOY_TEST = pd.DataFrame(
    {"id": [5, 6], "x": [0.5, 2.5], "social_media_hours": [0.75, 1.75]}
)


def toy_config(config_path: Path, name: str, seeds: list[int]) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        data=DataConfig(
            train=Path("train.csv"),
            test=Path("test.csv"),
            sample_submission=Path("sample.csv"),
            folds=Path("folds.parquet"),
        ),
        features=FeatureConfig(base="raw", categorical=[]),
        model=ModelConfig(kind="fake", params={}, fit={}),
        initial_score=None,
        seeds=seeds,
        stage="confirm",
        source_path=config_path,
    )


class FakeAdapter:
    """모델 연결부 대역. 자기가 받은 재학습 예산을 불린 순서대로 적는다."""

    def __init__(self, seed: int, fits: list[tuple[int, int | None]]) -> None:
        self.seed = seed
        self._fits = fits

    def fit_full(self, X, y, training_budget, initial_score=None) -> None:
        self._fits.append((self.seed, training_budget))

    def predict(self, X, initial_score=None):
        return np.full(len(X), self.seed / 100, dtype=np.float64)


def stub_training(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int | None]]:
    """학습 바깥의 무거운 자리를 대역으로 바꾸고, 연결부가 받은 예산 기록을 돌려준다.

    설정 적재, 자료, 입력 해시, git 상태를 대역으로 바꾼다. 재학습 예산이 어디서 오는지가
    이 시험의 관심사이므로, 그 예산을 실제로 소비하는 자리만 진짜로 남긴다.
    """
    fits: list[tuple[int, int | None]] = []
    monkeypatch.setattr(
        refit,
        "load_config",
        lambda path, stage: toy_config(path, Path(path).stem, [42, 43, 44]),
    )
    monkeypatch.setattr(
        refit.data,
        "load_csv",
        lambda path: TOY_TRAIN.copy() if str(path) == "train.csv" else TOY_TEST.copy(),
    )
    monkeypatch.setattr(refit.data, "file_sha256", lambda path: f"hash:{path}")
    monkeypatch.setattr(
        refit.tracking,
        "git_state",
        lambda: {"git_commit": "commit-1", "git_dirty": "False"},
    )
    monkeypatch.setattr(
        refit.model, "create", lambda model_cfg, seed: FakeAdapter(seed, fits)
    )
    return fits


def forbid_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """관문 뒤의 모든 부작용 자리를 막는다. 하나라도 불리면 시험이 깨진다."""

    def refuse(*args, **kwargs):
        raise AssertionError("검증을 통과하지 못한 장부로 실행 경로가 움직였다.")

    monkeypatch.setattr(refit.data, "load_csv", refuse)
    monkeypatch.setattr(refit, "FeaturePlan", refuse)
    monkeypatch.setattr(refit.model, "create", refuse)
    monkeypatch.setattr(refit.model, "fit_full", refuse)


def run_cli(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["pipeline.refit", *arguments])
    refit.main()


# 검증된 예산이 모델 연결부에 닿는다


@pytest.mark.parametrize("config", sorted(CRITERION_BUDGETS))
def test_validated_budgets_reach_the_model(
    ledger: MemoryLedger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: str
):
    plan = ledger.executable()
    member = plan.member(config)
    fits = stub_training(monkeypatch)

    refit.run_member(plan, member, tmp_path / "out")

    assert dict(fits) == CRITERION_BUDGETS[config]
    assert member.budgets == CRITERION_BUDGETS[config]


def test_the_whole_ledger_recomputes_to_ninety_four_refits(ledger: MemoryLedger):
    plan = ledger.executable()

    assert len(plan.members) == 32
    assert sum(len(member.budgets) for member in plan.members) == 94
    assert plan.content_sha256 == sha256_of(ledger.path.read_bytes())


# 잘못된 장부는 실행 경로를 열지 못한다


def test_invalid_plan_with_a_hand_edited_budget_stops_before_data_loading(
    ledger: MemoryLedger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """저장 재학습 예산만 한 단위 바꾼 XGBoost 사례."""
    document = ledger.edited()
    seeds = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "refit_budget_derivation"
    ]["seeds"]
    seeds[0]["budget"] -= 1
    path = ledger.write(document)
    ledger.install(monkeypatch)
    forbid_execution(monkeypatch)
    output = tmp_path / "out"

    with pytest.raises(SystemExit) as failure:
        run_cli(monkeypatch, str(path), "--member", "exp135_xgb_hpo_trial30", "--out-dir", str(output))

    assert "저장 재학습 예산이 다시 계산한 값과 다르다" in str(failure.value)
    assert not output.exists()


def test_invalid_plan_with_a_changed_source_run_id_stops_before_data_loading(
    ledger: MemoryLedger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """원시 실행 식별자만 바꾼 사례. 근거 계보가 후보 풀과 어긋난다."""
    document = ledger.edited()
    ledger.member(document, "exp135_xgb_hpo_trial30")["lineage"]["source_run_id"] = (
        "0" * 32
    )
    path = ledger.write(document)
    ledger.install(monkeypatch)
    forbid_execution(monkeypatch)
    output = tmp_path / "out"

    with pytest.raises(SystemExit) as failure:
        run_cli(monkeypatch, str(path), "--all", "--out-dir", str(output))

    assert "실행 ID가 후보 풀과 다르다" in str(failure.value)
    assert not output.exists()


def test_invalid_plan_in_another_member_stops_before_data_loading(
    ledger: MemoryLedger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """요청하지 않은 구성원의 근거가 미확정이어도 실행이 열리지 않는다."""
    document = ledger.edited()
    ledger.member(document, "exp157_lookup_muon_initavg8")["training_length_evidence"][
        "status"
    ] = "unresolved"
    path = ledger.write(document)
    ledger.install(monkeypatch)
    forbid_execution(monkeypatch)
    output = tmp_path / "out"

    with pytest.raises(SystemExit) as failure:
        run_cli(monkeypatch, str(path), "--member", "exp135_xgb_hpo_trial30", "--out-dir", str(output))

    assert "exp157_lookup_muon_initavg8: 관측 학습 길이 근거가 미확정이다." in str(
        failure.value
    )
    assert not output.exists()


def test_invalid_plan_with_legacy_budget_fields_stops_before_data_loading(
    ledger: MemoryLedger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """이전 문법의 손입력 통로(`budget_source`, `budgets`)는 남아 있지 않다."""
    document = ledger.edited()
    member = ledger.member(document, "exp135_xgb_hpo_trial30")
    member["budget_source"] = "fold_median"
    member["budgets"] = {42: 1, 43: 1, 44: 1}
    path = ledger.write(document)
    ledger.install(monkeypatch)
    forbid_execution(monkeypatch)
    output = tmp_path / "out"

    with pytest.raises(SystemExit) as failure:
        run_cli(monkeypatch, str(path), "--all", "--out-dir", str(output))

    assert "알 수 없는 필드" in str(failure.value)
    assert not output.exists()


def test_invalid_plan_with_the_previous_schema_version_stops_before_data_loading(
    ledger: MemoryLedger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    document = ledger.edited()
    document["schema_version"] = 1
    path = ledger.write(document)
    ledger.install(monkeypatch)
    forbid_execution(monkeypatch)
    output = tmp_path / "out"

    with pytest.raises(SystemExit) as failure:
        run_cli(monkeypatch, str(path), "--all", "--out-dir", str(output))

    assert "장부 문법 판본" in str(failure.value)
    assert not output.exists()


# 시드별 기록과 재개


def test_run_member_records_evidence_lineage_and_resumes(
    ledger: MemoryLedger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    plan = ledger.executable()
    member = plan.member("exp157_lookup_muon_initavg8")
    fits = stub_training(monkeypatch)
    output = tmp_path / "out"

    first = refit.run_member(plan, member, output)
    second = refit.run_member(plan, member, output)

    assert first == second
    assert fits == [(42, 14), (43, 15), (44, 15)]

    record = json.loads((first.parent / "test_pred_seed_42.json").read_text())
    assert record["schema_version"] == 2
    assert record["training_budget"] == 14
    assert record["plan_sha256"] == plan.content_sha256
    assert record["source_pool_sha256"] == plan.source_pool_sha256
    assert record["evidence_lineage"] == {
        "source_run_id": member.lineage.source_run_id,
        "source_git_commit": member.lineage.source_git_commit,
        "source_config_path": member.lineage.source_config_path,
        "source_config_sha256": member.lineage.source_config_sha256,
        "evidence_artifact_path": member.lineage.evidence_artifact_path,
        "evidence_artifact_sha256": member.lineage.evidence_artifact_sha256,
    }
    assert record["refit_budget_derivation"] == {
        "status": "confirmed",
        "statistic": "median",
        "multiplier": 1.25,
        "rounding": "half_up",
    }

    manifest = json.loads((first.parent / "manifest.json").read_text())
    assert manifest["plan_sha256"] == plan.content_sha256
    assert manifest["evidence_lineage"] == record["evidence_lineage"]
    assert manifest["refit_budget_derivation"] == record["refit_budget_derivation"]
    assert [entry["training_budget"] for entry in manifest["seeds"]] == [14, 15, 15]
    assert "budget_source" not in manifest


def test_resume_rejects_a_different_plan_content(
    ledger: MemoryLedger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """예산이 같아도 장부 내용이 달라지면 이어서 쓰지 않는다."""
    stub_training(monkeypatch)
    output = tmp_path / "out"
    plan = ledger.executable()
    refit.run_member(plan, plan.member("exp157_lookup_muon_initavg8"), output)

    document = ledger.edited()
    observations = ledger.member(document, "exp135_xgb_hpo_trial30")[
        "training_length_evidence"
    ]["observations"]
    observations[0]["raw_field"] = "best_iteration_renamed"
    other = ledger.executable(document)

    assert other.member("exp157_lookup_muon_initavg8").budgets == {
        42: 14,
        43: 15,
        44: 15,
    }
    assert other.content_sha256 != plan.content_sha256
    with pytest.raises(ValueError, match="현재 재학습 명세와 계보가 다르다"):
        refit.run_member(other, other.member("exp157_lookup_muon_initavg8"), output)


def test_resume_rejects_a_different_refit_budget(
    ledger: MemoryLedger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """원시 근거가 바뀌어 예산이 달라지면 저장된 시드 예측을 재사용하지 않는다."""
    stub_training(monkeypatch)
    output = tmp_path / "out"
    plan = ledger.executable()
    refit.run_member(plan, plan.member("exp157_lookup_muon_initavg8"), output)

    document = ledger.edited()
    for observation in ledger.member(document, "exp157_lookup_muon_initavg8")[
        "training_length_evidence"
    ]["observations"]:
        if observation["seed"] == 42:
            observation["raw_value"] = 15
            observation["observed_training_length"] = 16
    seeds = ledger.member(document, "exp157_lookup_muon_initavg8")[
        "refit_budget_derivation"
    ]["seeds"]
    seeds[0].update(observed_lengths=[16] * 40, median=16.0, scaled=20.0, budget=20)
    other = ledger.executable(document)

    assert other.member("exp157_lookup_muon_initavg8").budgets[42] == 20
    with pytest.raises(ValueError, match="현재 재학습 명세와 계보가 다르다"):
        refit.run_member(other, other.member("exp157_lookup_muon_initavg8"), output)


def test_run_member_can_fit_seeds_independently_before_finalizing(
    ledger: MemoryLedger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    plan = ledger.executable()
    member = plan.member("exp157_lookup_muon_initavg8")
    fits = stub_training(monkeypatch)
    output = tmp_path / "out"

    for seed in member.budgets:
        path = refit.run_member(plan, member, output, seeds=(seed,), finalize=False)
        assert path.name == f"test_pred_seed_{seed}.parquet"
        assert not (path.parent / "manifest.json").exists()

    final = refit.run_member(plan, member, output)

    assert fits == [(42, 14), (43, 15), (44, 15)]
    assert (final.parent / "manifest.json").is_file()
    averaged = pd.read_parquet(final)
    assert averaged["pred"].to_numpy() == pytest.approx([0.43, 0.43])


# 조립


def executable_plan_of_one(tmp_path: Path, combiner: str) -> ExecutableRefitPlan:
    member = ExecutableRefitMember(
        config="fake",
        config_path=tmp_path / "fake.yaml",
        lineage=MemberLineage(
            source_run_id="run-1",
            source_git_commit="commit-1",
            source_config_path="configs/fake.yaml",
            source_config_sha256="config-sha256",
            evidence_artifact_path="logs/run.log",
            evidence_artifact_sha256="evidence-sha256",
        ),
        status="not_applicable",
        budgets={42: None},
        derivation=None,
    )
    return ExecutableRefitPlan(
        source_path=tmp_path / "plan.yaml",
        content_sha256="plan-sha256",
        source_pool_sha256="pool-hash",
        protocol=RefitProtocol(
            iteration_multiplier=1.25,
            budget_statistic="median",
            budget_rounding="half_up",
            cv_model_weight=5,
            full_model_weight=1,
            combiner=combiner,
        ),
        members=(member,),
    )


def test_assemble_uses_combiner_named_by_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    train_index = pd.Index([1, 2, 3, 4], name="id")
    test_index = pd.Index([10, 11], name="id")
    train = pd.DataFrame({"id": train_index, "addicted_label": [0, 0, 1, 1]})
    test = pd.DataFrame({"id": test_index})
    plan = executable_plan_of_one(tmp_path, "recording_combiner")

    class RecordingCombiner:
        name = "recording_combiner"

    combiner = RecordingCombiner()
    calls = []
    monkeypatch.setitem(refit.COMBINER_REGISTRY, combiner.name, combiner)
    monkeypatch.setattr(
        refit.pd,
        "read_csv",
        lambda path: train.copy() if path == "data/train.csv" else test.copy(),
    )
    monkeypatch.setattr(
        refit,
        "member_matrix",
        lambda members, store, index: pd.DataFrame(
            {"fake": [0.1, 0.2, 0.8, 0.9]}, index=train_index
        ),
    )
    monkeypatch.setattr(
        refit,
        "member_test_matrix",
        lambda members, store, index: pd.DataFrame(
            {"fake": [0.25, 0.75]}, index=test_index
        ),
    )
    monkeypatch.setattr(
        refit,
        "_load_member_full_prediction",
        lambda plan, member, output, expected_ids: np.array([0.3, 0.7]),
    )
    monkeypatch.setattr(refit, "MlflowRunStore", lambda: None)

    def record_full_fit(selected, oof, y, test_predictions):
        calls.append(selected)
        return np.linspace(0.2, 0.8, len(test_predictions), dtype=np.float64)

    monkeypatch.setattr(refit, "full_fit_predictions", record_full_fit)

    refit.assemble(plan, tmp_path / "out")

    assert calls == [combiner, combiner, combiner]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["plan_sha256"] == "plan-sha256"


def test_assemble_reads_a_manifest_written_by_the_validated_run(
    ledger: MemoryLedger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """구성원 실행이 남긴 manifest를 조립이 그대로 받아들이고, 예산이 바뀌면 거부한다."""
    plan = ledger.executable()
    member = plan.member("exp157_lookup_muon_initavg8")
    stub_training(monkeypatch)
    output = tmp_path / "out"
    refit.run_member(plan, member, output)

    prediction = refit._load_member_full_prediction(
        plan, member, output, TOY_TEST["id"]
    )

    assert prediction.shape == (2,)

    manifest_path = output / member.config / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["seeds"][0]["training_budget"] = 13
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))

    with pytest.raises(ValueError, match="재학습 예산이 검증된 예산과 다르다"):
        refit._load_member_full_prediction(plan, member, output, TOY_TEST["id"])


def test_mix_member_predictions_uses_model_count_weights():
    index = pd.Index([10, 11], name="id")
    cv = pd.DataFrame({"a": [0.2, 0.8], "b": [0.4, 0.6]}, index=index)
    full = pd.DataFrame({"a": [0.8, 0.2], "b": [1.0, 0.0]}, index=index)

    mixed = mix_member_predictions(cv, full, cv_weight=5, full_weight=1)

    assert mixed.to_numpy() == pytest.approx(
        np.array([[0.3, 0.5], [0.7, 0.5]], dtype=np.float64)
    )
    assert all(dtype == np.dtype("float64") for dtype in mixed.dtypes)


def test_mix_member_predictions_rejects_misaligned_inputs():
    cv = pd.DataFrame({"a": [0.2]}, index=pd.Index([10], name="id"))
    full = pd.DataFrame({"b": [0.8]}, index=pd.Index([10], name="id"))

    with pytest.raises(ValueError, match="구성원 순서"):
        mix_member_predictions(cv, full, cv_weight=5, full_weight=1)
