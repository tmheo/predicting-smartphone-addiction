from __future__ import annotations

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

from kagglesdk.competitions.types.submission_status import SubmissionStatus
from mlflow.tracking import MlflowClient
import pytest

from pipeline.submit import record_existing_submission
from pipeline.tracking import EXPERIMENT_NAME


class FakeKaggleApi:
    def __init__(self, submission) -> None:
        self.submission = submission
        self.calls = 0

    def competition_submissions(self, competition, *, page_size):
        self.calls += 1
        assert competition == "playground-series-s6e8"
        assert page_size == 200
        return [self.submission]


def make_tracking(tmp_path: Path) -> tuple[str, str]:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment(
        EXPERIMENT_NAME, artifact_location=(tmp_path / "mlartifacts").as_uri()
    )
    source = client.create_run(experiment_id, run_name="source-ensemble")
    client.log_metric(source.info.run_id, "auc_oof", 0.96951)
    client.set_terminated(source.info.run_id)
    return tracking_uri, source.info.run_id


def test_record_existing_submission_creates_complete_derived_run(tmp_path):
    tracking_uri, source_run_id = make_tracking(tmp_path)
    submission_path = tmp_path / "submission_cv_full.csv"
    submission_path.write_text("id,addicted_label\n1,0.25\n2,0.75\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"schema_version": 1}\n')
    submitted_at = datetime.datetime(2026, 8, 18, 2, 5, tzinfo=datetime.UTC)
    api = FakeKaggleApi(
        SimpleNamespace(
            ref=55590060,
            date=submitted_at,
            description="issue66 cv-full 5:1",
            file_name="submission_cv_full.csv",
            public_score="0.97063",
            status=SubmissionStatus.COMPLETE,
        )
    )

    result = record_existing_submission(
        api=api,
        submission_ref=55590060,
        submission_path=submission_path,
        run_name="submission_issue66_cv_full",
        source_run_id=source_run_id,
        git_commit="a" * 40,
        artifacts=(manifest_path,),
        params={"ensemble.cv_model_weight": "5", "ensemble.full_model_weight": "1"},
        tags={"source.issue": "66"},
        tracking_uri=tracking_uri,
    )

    assert result.created is True
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["stage"] == "final_submission"
    assert run.data.params["model.kind"] == "derived_submission"
    assert run.data.params["source.run_id"] == source_run_id
    assert run.data.params["ensemble.cv_model_weight"] == "5"
    assert run.data.metrics["public_auc"] == 0.97063
    assert run.data.metrics["source_auc_oof"] == 0.96951
    assert run.data.tags["kaggle.submission_ref"] == "55590060"
    assert run.data.tags["kaggle.submission_message"] == "issue66 cv-full 5:1"
    assert run.data.tags["submitted_at"] == "2026-08-18T02:05:00+00:00"
    assert run.data.tags["source.issue"] == "66"
    assert run.data.tags["git_commit"] == "a" * 40

    downloaded = Path(client.download_artifacts(result.run_id, "submission.csv"))
    assert downloaded.read_bytes() == submission_path.read_bytes()
    record_path = Path(client.download_artifacts(result.run_id, "submission_record.json"))
    record = json.loads(record_path.read_text())
    assert record["kaggle"]["submission_ref"] == 55590060
    assert record["submission"]["rows"] == 2
    assert record["source_run_id"] == source_run_id
    assert Path(client.download_artifacts(result.run_id, "manifest.json")).exists()

    repeated = record_existing_submission(
        api=api,
        submission_ref=55590060,
        submission_path=submission_path,
        run_name="submission_issue66_cv_full",
        source_run_id=source_run_id,
        git_commit="a" * 40,
        tracking_uri=tracking_uri,
    )
    assert repeated.created is False
    assert repeated.run_id == result.run_id
    assert api.calls == 1

    submission_path.write_text("id,addicted_label\n1,0.5\n2,0.5\n")
    with pytest.raises(ValueError, match="CSV SHA-256이 다르다"):
        record_existing_submission(
            api=api,
            submission_ref=55590060,
            submission_path=submission_path,
            run_name="submission_issue66_cv_full",
            source_run_id=source_run_id,
            git_commit="a" * 40,
            tracking_uri=tracking_uri,
        )
    assert api.calls == 1
