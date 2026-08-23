"""fold 0 선별 결과의 MLflow 기록 계약 테스트. (#385)

핵심 계약은 "이 실행이 판정 대상으로 오인되지 않는다"이다.
- 판정 metric 이름(`auc_oof*`)을 쓰지 않는다.
- 판정에 필요한 산출물(`oof.parquet`, `test_pred.parquet`)을 붙이지 않는다.
- `judgment.eligible=false`와 `screening.scope=fold0_only` 태그가 붙는다.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location(
        "record_fold0_screening", REPO / "scripts" / "record_fold0_screening.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _records() -> list[dict]:
    return [
        {
            "candidate": "0.002:1",
            "lr": 0.002,
            "muon_lr_multiplier": 1.0,
            "epochs": 32,
            "seed": 42,
            "fold": 0,
            "auc": 0.9686805,
            "fit_seconds": 822.0,
            "member_best_epochs": [9, 9, 11],
            "model_training_diagnostics": {"fold_initialization_members": [{"a": 1}]},
        },
        {
            "candidate": "0.003:1",
            "lr": 0.003,
            "muon_lr_multiplier": 1.0,
            "epochs": 32,
            "seed": 42,
            "fold": 0,
            "auc": 0.9686031,
            "fit_seconds": 780.0,
            "member_best_epochs": [9, 9, 9],
        },
    ]


def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "screening.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_records_rejects_duplicate_candidates(module, tmp_path):
    duplicated = _records() + [_records()[0]]
    with pytest.raises(ValueError, match="겹친다"):
        module.load_records(_write(tmp_path, duplicated))


def test_load_records_rejects_empty_file(module, tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="비어 있다"):
        module.load_records(path)


def test_guard_rejects_judgment_metric_names(module):
    """fold 하나짜리 값을 OOF 이름으로 남기려는 시도를 막는다."""
    for name in ("auc_oof", "auc_oof_seed_42"):
        with pytest.raises(ValueError, match="판정 metric"):
            module.guard_metric_names([name])
    module.guard_metric_names(["auc_fold_0_seed_42", "diff_vs_baseline"])


def test_scalar_fields_split_numbers_and_text(module):
    metrics, params = module.scalar_fields(_records()[0])
    assert metrics["lr"] == pytest.approx(0.002)
    assert metrics["fit_seconds"] == pytest.approx(822.0)
    assert "candidate" not in metrics and "candidate" not in params
    assert "model_training_diagnostics" not in params
    # 목록은 param으로 직렬화한다.
    assert json.loads(params["member_best_epochs"]) == [9, 9, 11]


def test_summarize_reports_pairwise_range_against_the_baseline(module):
    summary = module.summarize(_records(), "0.002:1")
    assert summary["candidate_count"] == pytest.approx(2.0)
    assert summary["auc_fold_0_baseline"] == pytest.approx(0.9686805)
    assert summary["diff_vs_baseline_worst"] == pytest.approx(-0.0000774)
    assert summary["positive_candidate_count"] == pytest.approx(0.0)


def test_summarize_rejects_an_unknown_baseline(module):
    with pytest.raises(ValueError, match="기준 후보"):
        module.summarize(_records(), "9e-9:1")


def test_record_marks_the_group_as_ineligible_for_judgment(module, tmp_path):
    """기록된 실행이 판정 경로에 잘못 실릴 수 없는 상태인지 확인한다."""
    from mlflow.tracking import MlflowClient

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    records_path = _write(tmp_path, _records())
    parent_id = module.record(
        records=_records(),
        group_name="screen-test",
        note="시험",
        baseline="0.002:1",
        execution_commit="0" * 40,
        attachments=[],
        records_path=records_path,
        tracking_uri=tracking_uri,
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    parent = client.get_run(parent_id)
    assert parent.data.tags["judgment.eligible"] == "false"
    assert parent.data.tags["screening.scope"] == "fold0_only"
    assert parent.data.tags["execution.git_commit"] == "0" * 40
    assert not any(name.startswith("auc_oof") for name in parent.data.metrics)

    children = client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
    )
    assert len(children) == 2
    for child in children:
        assert child.data.tags["judgment.eligible"] == "false"
        assert not any(name.startswith("auc_oof") for name in child.data.metrics)
        assert child.data.metrics["auc_fold_0_seed_42"] > 0.9
        names = {item.path for item in client.list_artifacts(child.info.run_id)}
        assert "oof.parquet" not in names and "test_pred.parquet" not in names

    baseline = next(
        c for c in children if c.data.tags["screening.candidate"] == "0.002:1"
    )
    assert baseline.data.tags["screening.is_baseline"] == "True"
    assert baseline.data.metrics["diff_vs_baseline"] == pytest.approx(0.0)
    assert "model_training_diagnostics.json" in {
        item.path for item in client.list_artifacts(baseline.info.run_id)
    }
