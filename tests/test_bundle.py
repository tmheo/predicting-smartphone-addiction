"""실행 기록 묶음의 export·반입 테스트. (#98)

원격(다른 tracking uri)의 완료된 실행을 export한 zip을, 로컬 자료·git 검증·재채점을
거쳐 별도 tracking uri로 반입하는 왕복을 검증한다. git 검증은 helper 두 개를
monkeypatch로 대체한다(진짜 git 대조는 사용 시점의 저장소 상태에 의존하므로).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from pipeline import bundle as bundle_mod
from pipeline.bundle import BundleError, export_bundle, import_bundle
from pipeline.data import file_sha256
from pipeline.runs import MlflowRunStore

SEEDS = [42, 43]
N = 40
CONFIG_NAME = "exp_test.yaml"


def _make_data(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = np.arange(1, N + 1)
    y = np.tile([0, 1], N // 2)
    train = pd.DataFrame({"id": ids, "addicted_label": y, "age": rng.integers(18, 60, N)})
    folds = pd.DataFrame({"id": ids, "fold": np.tile([0, 0, 1, 1], N // 4)})
    return train, folds


def _seed_oof(rng: np.random.Generator, train: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    # 라벨과 약하게 상관된 예측: AUC가 0.5도 1.0도 아닌 실수가 되게 한다.
    pred = np.clip(
        train["addicted_label"] * 0.4 + rng.uniform(0, 0.6, len(train)), 1e-6, 1 - 1e-6
    )
    return pd.DataFrame({"id": train["id"], "fold": folds["fold"], "pred": pred})


@pytest.fixture
def env(tmp_path, monkeypatch):
    """로컬 자료(data/, artifacts/)와 원격 실행(별도 mlflow)을 함께 꾸민 반입 환경."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    train, folds = _make_data(rng)
    (tmp_path / "data").mkdir()
    (tmp_path / "artifacts").mkdir()
    train.to_csv(tmp_path / "data" / "train.csv", index=False)
    train.drop(columns=["addicted_label"]).to_csv(tmp_path / "data" / "test.csv", index=False)
    folds.to_parquet(tmp_path / "artifacts" / "folds.parquet", index=False)

    config = {
        "name": "exp_test",
        "data": {
            "train": "data/train.csv",
            "test": "data/test.csv",
            "folds": "artifacts/folds.parquet",
        },
        "model": {"kind": "lightgbm", "params": {}, "fit": {}},
    }
    config_path = tmp_path / CONFIG_NAME
    config_path.write_text(yaml.safe_dump(config))

    # 원격 실행: 시드별 OOF에서 정직하게 계산한 지표를 주장값으로 기록한다.
    seed_oofs = {seed: _seed_oof(rng, train, folds) for seed in SEEDS}
    y = train["addicted_label"].to_numpy()
    mean_pred = np.mean([o["pred"].to_numpy() for o in seed_oofs.values()], axis=0)
    metrics = {"auc_oof": float(roc_auc_score(y, mean_pred))}
    for fold in (0, 1):
        mask = folds["fold"].to_numpy() == fold
        metrics[f"auc_fold_{fold}"] = float(roc_auc_score(y[mask], mean_pred[mask]))
    for seed, oof in seed_oofs.items():
        metrics[f"auc_oof_seed_{seed}"] = float(roc_auc_score(y, oof["pred"].to_numpy()))

    source_uri = f"sqlite:///{tmp_path / 'source.db'}"
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=source_uri)
    experiment_id = client.create_experiment(
        "remote", artifact_location=(tmp_path / "source-artifacts").as_uri()
    )
    run = client.create_run(experiment_id, run_name="exp_test")
    run_id = run.info.run_id
    params = {
        "experiment": "exp_test",
        "seeds": ",".join(map(str, SEEDS)),
        "model.kind": "lightgbm",
        "features": "age,placebo_noise",
    }
    for key, value in params.items():
        client.log_param(run_id, key, value)
    for name, value in metrics.items():
        client.log_metric(run_id, name, value)
    tags = {
        "git_commit": "cafe" * 10,
        "git_dirty": "False",
        "sha256.train": file_sha256(tmp_path / "data" / "train.csv"),
        "sha256.test": file_sha256(tmp_path / "data" / "test.csv"),
        "sha256.folds": file_sha256(tmp_path / "artifacts" / "folds.parquet"),
    }
    for key, value in tags.items():
        client.set_tag(run_id, key, value)

    work = tmp_path / "work"
    work.mkdir()
    base_oof = seed_oofs[SEEDS[0]].copy()
    base_oof["pred"] = mean_pred
    base_oof.to_parquet(work / "oof.parquet", index=False)
    pd.DataFrame(
        {"feature": ["age", "placebo_noise"], "fold": [0, 0], "seed": [42, 42], "gain": [10.0, 1.0]}
    ).to_parquet(work / "feature_importance.parquet", index=False)
    pd.DataFrame({"id": train["id"], "pred": mean_pred}).to_parquet(
        work / "test_pred.parquet", index=False
    )
    (work / "submission.csv").write_text("id,addicted_label\n1,0.5\n")
    for seed, oof in seed_oofs.items():
        oof.to_parquet(work / f"oof_seed_{seed}.parquet", index=False)
    for name in sorted(p.name for p in work.iterdir()):
        client.log_artifact(run_id, str(work / name))
    client.log_artifact(run_id, str(config_path))
    (work / "run.log").write_text("log\n")
    client.log_artifact(run_id, str(work / "run.log"), artifact_path="logs")
    client.set_terminated(run_id, status="FINISHED")

    # git 검증: 커밋이 존재하고, 커밋 시점 config == 묶음 config라고 간주한다.
    monkeypatch.setattr(bundle_mod, "_git_commit_exists", lambda commit: True)
    monkeypatch.setattr(
        bundle_mod, "_git_file_sha256", lambda commit, path: file_sha256(config_path)
    )

    local_uri = f"sqlite:///{tmp_path / 'local.db'}"
    return {
        "source_uri": source_uri,
        "local_uri": local_uri,
        "run_id": run_id,
        "metrics": metrics,
        "tmp_path": tmp_path,
        "client": client,
    }


def _export(env) -> Path:
    return export_bundle(
        env["run_id"], env["tmp_path"] / "run.bundle.zip", tracking_uri=env["source_uri"]
    )


def test_roundtrip_reproduces_run(env):
    out = _export(env)
    new_run_id = import_bundle(out, tracking_uri=env["local_uri"])

    store = MlflowRunStore(tracking_uri=env["local_uri"])
    meta = store.facts_of(new_run_id)
    assert meta.params["experiment"] == "exp_test"
    assert meta.params["seeds"] == "42,43"
    # 기록된 지표는 재채점값이고, 정직한 주장값과 일치해야 한다.
    for name, value in env["metrics"].items():
        assert meta.metrics[name] == pytest.approx(value, abs=1e-12)
    assert meta.tags["git_dirty"] == "False"
    assert meta.tags["source.kind"] == "bundle"
    assert meta.tags["source.run_id"] == env["run_id"]
    # 산출물 전체가 재생되어 실행 저장소 소비자가 그대로 동작한다.
    assert len(store.oof_of(new_run_id)) == N
    assert not store.importance_of(new_run_id).empty
    assert store.config_of(new_run_id)["name"] == "exp_test"
    assert store.submission_path_of(new_run_id).exists()


def test_import_refuses_duplicate_bundle(env):
    out = _export(env)
    import_bundle(out, tracking_uri=env["local_uri"])
    with pytest.raises(BundleError, match="이미"):
        import_bundle(out, tracking_uri=env["local_uri"])


def test_import_refuses_input_hash_mismatch(env):
    out = _export(env)
    train_path = env["tmp_path"] / "data" / "train.csv"
    train_path.write_text(train_path.read_text() + "9999,1,30\n")
    with pytest.raises(BundleError, match="sha256"):
        import_bundle(out, tracking_uri=env["local_uri"])


def test_import_refuses_tampered_metric(env):
    out = _export(env)
    tampered = env["tmp_path"] / "tampered.bundle.zip"
    with zipfile.ZipFile(out) as zin, zipfile.ZipFile(tampered, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == bundle_mod.MANIFEST_NAME:
                manifest = json.loads(data)
                manifest["metrics"]["auc_oof"] += 0.01  # 주장값 조작
                data = json.dumps(manifest).encode()
            zout.writestr(item, data)
    with pytest.raises(BundleError, match="재채점"):
        import_bundle(tampered, tracking_uri=env["local_uri"])


def test_import_refuses_unknown_commit(env, monkeypatch):
    out = _export(env)
    monkeypatch.setattr(bundle_mod, "_git_commit_exists", lambda commit: False)
    with pytest.raises(BundleError, match="git_commit"):
        import_bundle(out, tracking_uri=env["local_uri"])


def test_import_refuses_config_drift(env, monkeypatch):
    out = _export(env)
    monkeypatch.setattr(bundle_mod, "_git_file_sha256", lambda commit, path: "0" * 64)
    with pytest.raises(BundleError, match="config"):
        import_bundle(out, tracking_uri=env["local_uri"])


def test_import_refuses_dirty_source_run(env):
    env["client"].set_tag(env["run_id"], "git_dirty", "True")
    out = _export(env)
    with pytest.raises(BundleError, match="git_dirty"):
        import_bundle(out, tracking_uri=env["local_uri"])


def test_export_refuses_incomplete_run(env):
    client = env["client"]
    run = client.create_run(
        client.get_experiment_by_name("remote").experiment_id, run_name="incomplete"
    )
    with pytest.raises(BundleError, match="미완료"):
        export_bundle(run.info.run_id, tracking_uri=env["source_uri"])


def test_export_refuses_run_without_seed_oofs(env, tmp_path):
    # 기록 규약 확장(#98) 이전 실행: oof_seed_*.parquet가 없다.
    client = env["client"]
    src = MlflowRunStore(tracking_uri=env["source_uri"])
    old = client.create_run(
        client.get_experiment_by_name("remote").experiment_id, run_name="old"
    )
    client.log_param(old.info.run_id, "experiment", "exp_old")
    client.log_param(old.info.run_id, "seeds", "42")
    client.log_param(old.info.run_id, "features", "age")
    client.log_metric(old.info.run_id, "auc_oof", 0.9)
    config_path = tmp_path / "exp_old.yaml"
    config_path.write_text("name: exp_old\n")
    client.log_artifact(old.info.run_id, str(config_path))
    assert src.facts_of(old.info.run_id).metrics["auc_oof"] == 0.9
    with pytest.raises(BundleError, match="시드별 OOF"):
        export_bundle(old.info.run_id, tracking_uri=env["source_uri"])
