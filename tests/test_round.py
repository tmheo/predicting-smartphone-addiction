"""pipeline.round: 판정 회차 module의 계약 시험. (#552, 지도 #550)

합성 구성원 행렬 + InMemoryRunStore로 동사 5개 왕복(run은 스펙 스크립트 재호출
subprocess까지 실제로), 재개 검사의 불일치 거부, 자기 검사 등급별 동작, derive
계보 검증, 게시 manifest 자기 완결성을 확인한다. 실측 재현은 파일럿 회차 소관이다.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.data import ID, TARGET, labels as load_labels
from pipeline.ensemble import COMBINER_REGISTRY, evaluate_nested
from pipeline.judgment import JudgmentError, StackGate
from pipeline.members import HASH_VERIFIED, MemberSource, MemberSpec
from pipeline.round import (
    SELF_CHECK_HASH_IDENTITY,
    SELF_CHECK_SEALED_FOLD_REPLAY,
    CandidateArm,
    ExpectedFold,
    JudgmentRound,
    ReferenceArm,
    ReferenceValues,
    ReportSection,
    RoundSpec,
    SelfCheckSpec,
)
from pipeline.runs import InMemoryRunStore
from pipeline.sealed import SealedRecord, SealedRecordError

ROWS = 60
FOLD_COUNT = 3
SRC_ROOT = str(Path(__file__).parents[1] / "src")


def fake_code_state() -> dict[str, object]:
    return {
        "git": {"commit": "test-commit", "dirty": False},
        "uv_lock_sha256": "test-lock",
        "sources": {"round": "test-source"},
    }


def member_spec(member_id: str, run_id: str, values: np.ndarray) -> MemberSpec:
    from pipeline.identity import array_identity

    return MemberSpec(
        member_id=member_id,
        origin="own",
        verification=HASH_VERIFIED,
        run_id=run_id,
        oof_sha256=array_identity(values),
    )


class Environment:
    """합성 자료 한 벌: 고정 분할, 라벨, 기준 2명 + 후보 3명의 구성원."""

    def __init__(self, root: Path) -> None:
        self.root = root
        rng = np.random.default_rng(0)
        ids = np.arange(1, ROWS + 1)
        fold = ids % FOLD_COUNT
        y = ids % 2
        (root / "data").mkdir(parents=True)
        pd.DataFrame({ID: ids, TARGET: y}).to_csv(root / "data/train.csv", index=False)
        (root / "artifacts").mkdir(parents=True)
        pd.DataFrame({ID: ids, "fold": fold}).to_parquet(root / "artifacts/folds.parquet")
        self.fold_of = pd.DataFrame({ID: ids, "fold": fold}).set_index(ID)["fold"]
        self.y = load_labels(self.fold_of.index, train_path=root / "data/train.csv")
        self.members = {
            "m1": rng.random(ROWS),
            "m2": rng.random(ROWS),
            "perfect": y.astype(np.float64),
        }
        self.store = InMemoryRunStore()
        for name, values in self.members.items():
            self.store.add_run(
                f"run-{name}", oof=pd.DataFrame({ID: ids, "pred": values})
            )

    def source(self, name: str, member_ids: list[str]) -> MemberSource:
        return MemberSource(
            name=name,
            members=tuple(
                member_spec(m, f"run-{m}", self.members[m]) for m in member_ids
            ),
        )

    def reference_evaluation(self):
        frame = pd.DataFrame(
            {m: self.members[m] for m in ("m1", "m2")}, index=self.fold_of.index
        )
        return evaluate_nested(
            COMBINER_REGISTRY["rank_mean"], frame, self.fold_of, self.y
        )


def make_spec(env: Environment, *, selfcheck: SelfCheckSpec | None = None, **overrides) -> RoundSpec:
    evaluation = env.reference_evaluation()
    if selfcheck is None:
        fold0 = evaluation.folds[0]
        selfcheck = SelfCheckSpec(
            grade=SELF_CHECK_SEALED_FOLD_REPLAY,
            source="시험 안 evaluate_nested 재계산",
            expected={
                fold0.fold: ExpectedFold(
                    auc=fold0.auc, prediction_sha256=fold0.prediction_identity
                )
            },
        )
    fields = {
        "round_id": "synthetic-stack/issue552",
        "title": "합성 스택 교체 판정",
        "question": "후보 팔이 기준 팔보다 nested AUC 문턱 이상 높은가.",
        "contract": "round-contract-test",
        "reference": ReferenceArm(
            name="reference",
            values=ReferenceValues(
                source="시험 안 evaluate_nested 재계산",
                nested_auc=evaluation.nested_auc,
                fold_aucs={o.fold: o.auc for o in evaluation.folds},
            ),
            source=env.source("reference-source", ["m1", "m2"]),
        ),
        "candidates": (
            CandidateArm(name="candidate", source=env.source("candidate-source", ["m1", "m2", "perfect"])),
        ),
        "combiner": "rank_mean",
        "gate": StackGate(folds_required_positive=FOLD_COUNT),
        "selfcheck": selfcheck,
        **overrides,
    }
    return RoundSpec(**fields)


def make_round(env: Environment, spec: RoundSpec, **overrides) -> JudgmentRound:
    kwargs = {
        "store": env.store,
        "root": env.root,
        "code_state": fake_code_state,
        **overrides,
    }
    return JudgmentRound(spec, **kwargs)


def write_spec_script(env: Environment, spec: RoundSpec, path: Path) -> Path:
    """run의 subprocess 재호출이 쓸 실제 스펙 스크립트를 만든다."""
    payload_path = path.with_suffix(".pickle")
    payload_path.write_bytes(
        pickle.dumps(
            {"spec": spec, "root": str(env.root), "code_state": fake_code_state()}
        )
    )
    path.write_text(
        "\n".join(
            [
                "import pickle, sys",
                "from pathlib import Path",
                f"sys.path.insert(0, {SRC_ROOT!r})",
                "from pipeline import round as round_module",
                f"payload = pickle.loads(Path({str(payload_path)!r}).read_bytes())",
                "round_module.main(",
                "    payload['spec'],",
                "    root=Path(payload['root']),",
                "    code_state=lambda: payload['code_state'],",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    return path


def complete_folds(round_: JudgmentRound) -> None:
    for fold in range(FOLD_COUNT):
        round_.fold_job("candidate", fold)


# ---------------------------------------------------------------- 왕복


def test_five_verb_roundtrip_with_subprocess_run(tmp_path):
    env = Environment(tmp_path / "root")
    spec = make_spec(env)
    script = write_spec_script(env, spec, tmp_path / "spec_script.py")
    round_ = make_round(env, spec, script=script)

    precommit = round_.precommit()
    round_.run(workers=2, threads=1, poll_seconds=0.1)
    comparison = round_.compare()
    round_.report()
    publish_dir = round_.publish()

    verdict = comparison.payload["arms"]["candidate"]["verdict"]
    assert verdict["passes_gate"] is True
    assert verdict["delta"] > 0
    assert verdict["folds_positive"] == FOLD_COUNT
    assert comparison.payload["parent_sealed_sha256"] == precommit.sealed_sha256

    selfcheck = SealedRecord.open(
        round_.run_dir / "selfcheck" / "fold-0.json",
        schema="round-contract-test/selfcheck/1",
    )
    assert selfcheck.payload["matches"] is True
    assert selfcheck.payload["parent_sealed_sha256"] == precommit.sealed_sha256

    for fold in range(FOLD_COUNT):
        record = SealedRecord.open(
            round_.run_dir / "arms" / "candidate" / f"fold-{fold}" / "fold.json",
            schema="round-contract-test/fold/1",
        )
        assert record.payload["parent_sealed_sha256"] == precommit.sealed_sha256

    report = (round_.run_dir / "report.md").read_text(encoding="utf-8")
    for heading in ("## 판정", "## 분할별 결과", "## 동결과 재현성"):
        assert heading in report

    manifest = (publish_dir / "manifest.sha256").read_text(encoding="utf-8")
    listed = {line.split("  ", 1)[1] for line in manifest.splitlines()}
    actual = {
        str(p.relative_to(publish_dir))
        for p in publish_dir.rglob("*")
        if p.is_file() and p.name != "manifest.sha256"
    }
    assert listed == actual
    assert "precommit.json" in listed and "comparison.json" in listed and "report.md" in listed
    assert not any("cache" in name or "predictions" in name or "logs" in name for name in listed)


def test_report_hook_section_is_rendered(tmp_path):
    env = Environment(tmp_path / "root")
    section = ReportSection(
        title="회차 고유 진단",
        render=lambda records: [
            f"- 팔 수: {len(records.folds)}",
            f"- precommit: {records.precommit.sealed_sha256[:8]}",
        ],
    )
    evaluation = env.reference_evaluation()
    spec = make_spec(
        env,
        selfcheck=SelfCheckSpec(
            grade=SELF_CHECK_HASH_IDENTITY, source="입력 해시 재검증"
        ),
        reference=ReferenceArm(
            name="reference",
            values=ReferenceValues(
                source="시험 안 evaluate_nested 재계산",
                nested_auc=evaluation.nested_auc,
                fold_aucs={o.fold: o.auc for o in evaluation.folds},
            ),
            source=None,
        ),
        report_sections=(section,),
    )
    round_ = make_round(env, spec)
    round_.precommit()
    complete_folds(round_)
    round_.run(workers=1, threads=1, poll_seconds=0.05)
    round_.compare()
    round_.report()
    report = (round_.run_dir / "report.md").read_text(encoding="utf-8")
    assert "## 회차 고유 진단" in report
    assert "- 팔 수: 1" in report


# ------------------------------------------------------ 자기 검사 등급


def test_hash_identity_selfcheck_written_by_run(tmp_path):
    env = Environment(tmp_path / "root")
    spec = make_spec(
        env,
        selfcheck=SelfCheckSpec(grade=SELF_CHECK_HASH_IDENTITY, source="입력 해시 재검증"),
    )
    round_ = make_round(env, spec)
    precommit = round_.precommit()
    complete_folds(round_)
    round_.run(workers=1, threads=1, poll_seconds=0.05)
    record = SealedRecord.open(
        round_.run_dir / "selfcheck" / "selfcheck.json",
        schema="round-contract-test/selfcheck/1",
    )
    assert record.payload["grade"] == SELF_CHECK_HASH_IDENTITY
    assert record.payload["matches"] is True
    assert record.payload["parent_sealed_sha256"] == precommit.sealed_sha256
    assert record.payload["verified_inputs"].keys() == precommit.payload["inputs"].keys()


def test_sealed_fold_replay_selfcheck_passes_in_process(tmp_path):
    env = Environment(tmp_path / "root")
    round_ = make_round(env, make_spec(env))
    round_.precommit()
    round_.selfcheck_job(0)
    record = SealedRecord.open(
        round_.run_dir / "selfcheck" / "fold-0.json",
        schema="round-contract-test/selfcheck/1",
    )
    assert record.payload["matches"] is True
    assert record.payload["expected"]["auc"] == record.payload["actual"]["auc"]


def test_sealed_fold_replay_mismatch_is_undecidable(tmp_path):
    env = Environment(tmp_path / "root")
    spec = make_spec(
        env,
        selfcheck=SelfCheckSpec(
            grade=SELF_CHECK_SEALED_FOLD_REPLAY,
            source="틀린 기준값",
            expected={0: ExpectedFold(auc=0.5, prediction_sha256=None)},
        ),
    )
    round_ = make_round(env, spec)
    round_.precommit()
    with pytest.raises(JudgmentError, match="자기 검사 실패"):
        round_.selfcheck_job(0)
    record = SealedRecord.open(
        round_.run_dir / "selfcheck" / "fold-0.json",
        schema="round-contract-test/selfcheck/1",
    )
    assert record.payload["matches"] is False
    complete_folds(round_)
    with pytest.raises(JudgmentError, match="자기 검사가 실패 상태"):
        round_.compare()


# ------------------------------------------------------ 재개 검사 거부


def test_input_change_after_precommit_is_rejected(tmp_path):
    env = Environment(tmp_path / "root")
    round_ = make_round(env, make_spec(env))
    round_.precommit()
    train = env.root / "data/train.csv"
    train.write_text(train.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(JudgmentError, match="동결 입력 train"):
        round_.fold_job("candidate", 0)


def test_code_state_drift_is_rejected(tmp_path):
    env = Environment(tmp_path / "root")
    spec = make_spec(env)
    make_round(env, spec).precommit()
    drifted = fake_code_state()
    drifted["git"] = {"commit": "other-commit", "dirty": False}
    round_ = make_round(env, spec, code_state=lambda: drifted)
    with pytest.raises(JudgmentError, match="git commit"):
        round_.compare()


def test_dirty_tree_is_rejected(tmp_path):
    env = Environment(tmp_path / "root")
    spec = make_spec(env)
    dirty = fake_code_state()
    dirty["git"] = {"commit": "test-commit", "dirty": True}
    round_ = make_round(env, spec, code_state=lambda: dirty)
    with pytest.raises(JudgmentError, match="커밋된 코드 상태"):
        round_.precommit()


def test_precommit_refuses_overwrite(tmp_path):
    env = Environment(tmp_path / "root")
    spec = make_spec(env)
    make_round(env, spec).precommit()
    with pytest.raises(JudgmentError, match="이미 있다"):
        make_round(env, spec).precommit()


# ------------------------------------------------------------ 계보


def test_tampered_fold_record_is_rejected(tmp_path):
    env = Environment(tmp_path / "root")
    round_ = make_round(env, make_spec(env))
    round_.precommit()
    round_.selfcheck_job(0)
    complete_folds(round_)
    path = round_.run_dir / "arms" / "candidate" / "fold-0" / "fold.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"auc":', '"auc_":'), encoding="utf-8")
    with pytest.raises(SealedRecordError):
        round_.compare()


def test_fold_record_from_other_precommit_is_rejected(tmp_path):
    env = Environment(tmp_path / "root")
    round_ = make_round(env, make_spec(env))
    precommit = round_.precommit()
    round_.selfcheck_job(0)
    complete_folds(round_)
    foreign = SealedRecord.seal(
        precommit.schema, {**precommit.payload, "created_at": "2000-01-01T00:00:00Z"}
    )
    path = round_.run_dir / "arms" / "candidate" / "fold-0" / "fold.json"
    original = SealedRecord.open(path, schema="round-contract-test/fold/1")
    body = {k: v for k, v in original.payload.items() if k != "parent_sealed_sha256"}
    foreign.derive("fold", body).write(path)
    with pytest.raises(JudgmentError, match="다른 precommit"):
        round_.compare()


# ------------------------------------------------------------ 게시


def test_publish_requires_comparison_and_refuses_existing_dir(tmp_path):
    env = Environment(tmp_path / "root")
    round_ = make_round(env, make_spec(env))
    round_.precommit()
    with pytest.raises(SealedRecordError):
        round_.publish()
    round_.selfcheck_job(0)
    complete_folds(round_)
    round_.compare()
    round_.report()
    round_.publish()
    with pytest.raises(JudgmentError, match="게시 폴더가 이미 있다"):
        round_.publish()


# ------------------------------------------------------ 스펙 검증


def test_spec_validation_rejects_bad_declarations(tmp_path):
    env = Environment(tmp_path / "root")
    good = make_spec(env)
    with pytest.raises(JudgmentError, match="두 분절"):
        make_spec(env, round_id="no-issue-segment")
    with pytest.raises(JudgmentError, match="결합기 이름"):
        make_spec(env, combiner="no-such-combiner")
    with pytest.raises(JudgmentError, match="재현 기대값"):
        SelfCheckSpec(
            grade=SELF_CHECK_HASH_IDENTITY,
            source="출처",
            expected={0: ExpectedFold(auc=0.5)},
        )
    with pytest.raises(JudgmentError, match="정확히 분할 1개"):
        SelfCheckSpec(
            grade=SELF_CHECK_SEALED_FOLD_REPLAY,
            source="출처",
            expected={0: ExpectedFold(auc=0.5), 1: ExpectedFold(auc=0.5)},
        )
    with pytest.raises(JudgmentError, match="구성원 출처가 필요"):
        make_spec(
            env,
            reference=ReferenceArm(name="reference", values=good.reference.values, source=None),
        )
    with pytest.raises(JudgmentError, match="팔 이름이 중복"):
        make_spec(
            env,
            candidates=(
                CandidateArm(name="reference", source=env.source("dup", ["m1", "m2"])),
            ),
        )
