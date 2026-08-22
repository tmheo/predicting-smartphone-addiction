from __future__ import annotations

import json
import multiprocessing
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.data import ID, TARGET, file_sha256
from pipeline.features import FrequencyEncoder
from pipeline.fold_fit_reuse import (
    EVIDENCE_NAME,
    FoldFitReuseError,
    FoldFitReuseRequest,
    FoldFitReuseStore,
    canonical_json_bytes,
    dataframe_value_sha256,
    keys_from_evidence,
    provider_identity_document,
)
from pipeline.plan import FeatureContractError, FeaturePlan


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            ID: [10, 11, 12, 13],
            "x": [1.0, -0.0, np.nan, 2.0],
            "cat": pd.Categorical(
                ["b", "a", None, "b"], categories=["b", "a"], ordered=True
            ),
        }
    )
    test = pd.DataFrame(
        {
            ID: [20, 21],
            "x": [3.0, np.nan],
            "cat": pd.Categorical(
                ["a", None], categories=["b", "a"], ordered=True
            ),
        }
    )
    return train, test


def _request(*, uses_target: bool = False, target_shift: int = 0) -> FoldFitReuseRequest:
    train, test = _frames()
    provider = {
        "kind": "test_provider",
        "implementation": "tests.TestProvider",
        "implementation_sha256": "a" * 64,
        "settings": {"width": 2},
        "input_columns": ["x", "cat"],
        "output_columns": ["z"],
        "uses_target": uses_target,
        "external_file_sha256": {},
        "execution": {"mode": "cpu"},
    }
    return FoldFitReuseRequest(
        provider=provider,
        runtime={
            "git_commit": "b" * 40,
            "python": {"implementation": "CPython", "version": "test"},
            "dependency_lock_sha256": "c" * 64,
            "installed_packages": [{"name": "pandas", "version": "test"}],
            "platform": {
                "operating_system": "test",
                "release": "test",
                "machine": "test",
            },
        },
        input_files={"train": "d" * 64, "test": "e" * 64, "folds": "f" * 64},
        seed=42,
        fold=0,
        train_input=train,
        test_input=test,
        training_ids=train.loc[:2, ID],
        validation_ids=train.loc[3:, ID],
        test_ids=test[ID],
        training_target=(
            pd.Series([0 + target_shift, 1, 0], index=train.index[:3], name=TARGET)
            if uses_target
            else None
        ),
    )


def _computed_frames(request: FoldFitReuseRequest) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {ID: request.train_input[ID], "z": request.train_input["x"].fillna(-1.0)}
    )
    test = pd.DataFrame(
        {ID: request.test_input[ID], "z": request.test_input["x"].fillna(-1.0)}
    )
    return train, test


def _concurrent_worker(root: str, counter: str, output) -> None:
    store = FoldFitReuseStore(Path(root))
    request = _request()

    def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
        with Path(counter).open("ab") as stream:
            stream.write(b"x")
            stream.flush()
        time.sleep(0.2)
        return _computed_frames(request)

    result = store.resolve(request, compute)
    output.put((result.status, result.key, result.manifest_sha256))


def test_dataframe_hash_preserves_category_missing_order_and_float_bits() -> None:
    train, _ = _frames()
    original = dataframe_value_sha256(train)

    category_order = train.copy()
    category_order["cat"] = category_order["cat"].cat.reorder_categories(["a", "b"])
    assert dataframe_value_sha256(category_order) != original

    row_order = train.iloc[::-1].reset_index(drop=True)
    assert dataframe_value_sha256(row_order) != original

    positive_zero = train.copy()
    positive_zero.loc[1, "x"] = 0.0
    assert dataframe_value_sha256(positive_zero) != original

    filled = train.copy()
    filled.loc[2, "x"] = 0.0
    assert dataframe_value_sha256(filled) != original


def test_identity_changes_for_inputs_rows_seed_fold_git_and_target_use() -> None:
    base = _request()
    base_identity = base.identity_document()
    base_key = FoldFitReuseStore.key_of(base_identity)

    assert set(base_identity["row_ids"]["training"]) == {
        "row_count",
        "dtype",
        "value_sha256",
    }
    assert not isinstance(base_identity["row_ids"]["training"], list)
    assert len(canonical_json_bytes(base_identity)) < 10_000

    changed_input = replace(base, train_input=base.train_input.assign(x=[9.0, -0.0, np.nan, 2.0]))
    changed_rows = replace(base, training_ids=base.training_ids.iloc[::-1])
    changed_seed = replace(base, seed=7)
    changed_fold = replace(base, fold=1)
    changed_runtime = replace(base, runtime={**base.runtime, "git_commit": "9" * 40})
    changed_provider = replace(
        base,
        provider={**base.provider, "settings": {"width": 3}},
    )
    for changed in (
        changed_input,
        changed_rows,
        changed_seed,
        changed_fold,
        changed_runtime,
        changed_provider,
    ):
        assert FoldFitReuseStore.key_of(changed.identity_document()) != base_key

    target_first = _request(uses_target=True)
    target_changed = _request(uses_target=True, target_shift=1)
    assert target_first.identity_document() != target_changed.identity_document()
    assert "training_target_value_sha256" not in base.identity_document()


def test_identity_normalizes_json_arrays_before_publish(tmp_path: Path) -> None:
    request = _request()
    request = replace(
        request,
        provider={**request.provider, "settings": {"pair": ("x", "cat")}},
    )
    store = FoldFitReuseStore(tmp_path / "cache")

    generated = store.resolve(request, lambda: _computed_frames(request))
    hit = store.resolve(request, lambda: pytest.fail("적중 항목을 다시 계산했다."))

    assert request.identity_document()["provider"]["settings"]["pair"] == ["x", "cat"]
    assert hit.status == "hit"
    assert hit.key == generated.key


def test_store_generates_once_hits_and_refuses_corrupt_expected_key(tmp_path: Path) -> None:
    store = FoldFitReuseStore(tmp_path / "cache")
    request = _request()
    calls = 0

    def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
        nonlocal calls
        calls += 1
        return _computed_frames(request)

    generated = store.resolve(request, compute)
    hit = store.resolve(request, compute)

    assert calls == 1
    assert generated.status == "generated"
    assert hit.status == "hit"
    assert hit.key == generated.key
    assert hit.manifest_sha256 == generated.manifest_sha256
    pd.testing.assert_frame_equal(hit.train, generated.train)
    pd.testing.assert_frame_equal(hit.test, generated.test)

    manifest_path = store.item_path(hit.key) / "manifest.json"
    manifest_path.chmod(0o644)
    manifest_path.write_text("{}\n")
    with pytest.raises(FoldFitReuseError):
        store.resolve(request, compute)
    assert calls == 1


def test_concurrent_processes_compute_one_immutable_item(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    counter = tmp_path / "compute-count"
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_worker,
            args=(str(root), str(counter), output),
        )
        for _ in range(3)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    results = [output.get(timeout=2) for _ in processes]
    assert counter.read_bytes() == b"x"
    assert sorted(status for status, _, _ in results) == ["generated", "hit", "hit"]
    assert len({(key, manifest_sha) for _, key, manifest_sha in results}) == 1


def test_leftover_temporary_directory_is_removed_before_generation(tmp_path: Path) -> None:
    store = FoldFitReuseStore(tmp_path / "cache")
    request = _request()
    key = store.key_of(request.identity_document())
    leftover = store.root / f".tmp-{key}-stopped-writer"
    leftover.mkdir()
    (leftover / "partial").write_text("partial")

    result = store.resolve(request, lambda: _computed_frames(request))

    assert result.status == "generated"
    assert not leftover.exists()


def test_bundle_round_trip_verifies_files_and_preserves_immutability(tmp_path: Path) -> None:
    source = FoldFitReuseStore(tmp_path / "source")
    request = _request()
    result = source.resolve(request, lambda: _computed_frames(request))
    evidence = tmp_path / EVIDENCE_NAME
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "status": result.status,
                        "key": result.key,
                    }
                ],
            }
        )
    )
    bundle = source.export_bundle(keys_from_evidence(evidence), tmp_path / "reuse.zip")
    destination = FoldFitReuseStore(tmp_path / "destination")

    imported = destination.import_bundle(bundle)

    assert imported == [result.key]
    assert destination.validate_item(result.key) == result.manifest_sha256
    assert file_sha256(bundle)
    assert destination.import_bundle(bundle) == [result.key]


def test_feature_plan_restricts_provider_to_declared_inputs() -> None:
    class UndeclaredReader:
        uses_target = False

        def columns(self) -> list[str]:
            return ["out"]

        def reuse_input_columns(self) -> list[str]:
            return ["x"]

        def reuse_settings(self) -> dict[str, object]:
            return {}

        def fit(self, frame: pd.DataFrame, seed: int) -> None:
            frame["secret"]

        def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"out": 1.0}, index=frame.index)

    train = pd.DataFrame(
        {ID: [1, 2], "x": [1.0, 2.0], "secret": [3.0, 4.0], TARGET: [0, 1]}
    )
    test = train.drop(columns=[TARGET]).copy()
    plan = FeaturePlan({"dataset-wide": [], "row-wise": [], "fold-fit": []}, [], [])

    with pytest.raises(FeatureContractError, match="선언하지 않은"):
        plan.materialize_fold_fit_provider(
            kind="undeclared",
            transformer=UndeclaredReader(),
            train_input=train,
            test_input=test,
            training_index=train.index[:1],
            validation_index=train.index[1:],
            seed=42,
            fold=0,
            recorder=None,
        )


def test_feature_plan_reuses_same_provider_across_plan_context(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {ID: [1, 2, 3, 4], "x": [1.0, 1.0, 2.0, 3.0], TARGET: [0, 1, 0, 1]}
    )
    test = pd.DataFrame({ID: [5, 6], "x": [1.0, 9.0]})
    store = FoldFitReuseStore(tmp_path / "cache")
    runtime = _request().runtime
    inputs = {"train": "b" * 64, "test": "c" * 64, "folds": "d" * 64}
    first_plan = FeaturePlan({"dataset-wide": [], "row-wise": [], "fold-fit": []}, [], [])
    second_plan = FeaturePlan({"dataset-wide": [], "row-wise": [], "fold-fit": []}, [], [])
    first_plan.configure_fold_fit_reuse(store, runtime_identity=runtime, input_files=inputs)
    second_plan.configure_fold_fit_reuse(store, runtime_identity=runtime, input_files=inputs)

    first = first_plan.materialize_fold_fit_provider(
        kind="frequency_encoding",
        transformer=FrequencyEncoder(["x"]),
        train_input=train,
        test_input=test,
        training_index=train.index[:3],
        validation_index=train.index[3:],
        seed=42,
        fold=0,
        recorder=None,
    )
    changed_target_train = train.assign(**{TARGET: [1, 0, 1, 0]})
    second = second_plan.materialize_fold_fit_provider(
        kind="frequency_encoding",
        transformer=FrequencyEncoder(["x"]),
        train_input=changed_target_train,
        test_input=test,
        training_index=train.index[:3],
        validation_index=train.index[3:],
        seed=42,
        fold=0,
        recorder=None,
    )

    assert first[2]["status"] == "generated"
    assert second[2]["status"] == "hit"
    assert first[2]["key"] == second[2]["key"]
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_provider_identity_excludes_plan_order_and_model_configuration() -> None:
    provider = FrequencyEncoder(["x"])
    identity = provider_identity_document(
        kind="frequency_encoding",
        provider=provider,
        input_columns=provider.reuse_input_columns(),
        output_columns=provider.columns(),
        uses_target=provider.uses_target,
        settings=provider.reuse_settings(),
        external_file_sha256={},
        execution={"mode": "cpu"},
    )

    assert "model" not in identity
    assert "plan" not in identity
    assert "order" not in identity


def test_unavailable_explicit_store_fails_instead_of_recomputing(tmp_path: Path) -> None:
    unavailable = tmp_path / "not-a-directory"
    unavailable.write_text("file")

    with pytest.raises(FoldFitReuseError, match="사용할 수 없다"):
        FoldFitReuseStore(unavailable)
