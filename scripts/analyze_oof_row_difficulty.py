"""동결한 계약으로 최종 242개 앙상블의 행별 OOF 난점을 분석한다.

사용법:
    python scripts/analyze_oof_row_difficulty.py
    python scripts/analyze_oof_row_difficulty.py --input-root /path/to/repository

작업 폴더에 비커밋 입력이 없으면 ``--input-root``로 원본 실행 저장소가 있는 같은
저장소 checkout을 지정한다.
출력은 임시 디렉터리에서 모두 만든 뒤 ``run-logs/issue459``를 한 번에 확정한다.
기존 출력이나 잠금이 있으면 실패하며 덮어쓰기 선택지는 제공하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import sklearn
import yaml
from sklearn.metrics import roc_auc_score

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pipeline import ensemble  # noqa: E402
from pipeline.features import DERIVED_REGISTRY  # noqa: E402

ISSUE = 459
FINAL_RUN_ID = "4f2466f84f8d462fb8231bd3a4274dd1"
CHAMPION_RUN_ID = "6911a461866b43dc9556553eba6783b7"
EXPECTED_NESTED_AUC = 0.9702876097776773
EXPECTED_INSAMPLE_AUC = 0.970415903422154
NESTED_TOLERANCE = 1e-10
BURDEN_TOLERANCE = 1e-12
MIN_ROWS = 200
MIN_EFFECTIVE_ROWS = 200.0
MIN_BURDEN_INCREASE = 0.10
MIN_AUC_OPPORTUNITY = 0.00005
DISAGREEMENT_THRESHOLD = 0.10
SENSITIVITY_SHARES = (0.001, 0.01, 0.05)
OFFICIAL_HARD_SHARE = 0.01
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)

ID = "id"
TARGET = "addicted_label"
FEATURES = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "gender",
    "stress_level",
    "academic_work_impact",
]
NUMERIC = FEATURES[:9]
CATEGORICAL = FEATURES[9:]
DERIVED = [
    "other_screen",
    "screen_slack",
    "sgw_sum",
    "sgw_frac",
    "slack_frac",
    "wk_minus_sgw",
    "wk_other",
    "gaming_minus_work",
    "screen_minus_work",
    "weekend_minus_daily",
    "social_share_screen",
    "gaming_share_screen",
    "work_share_screen",
    "screen_to_sleep",
]
EXPECTED_HASHES = {
    "train": "f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c",
    "test": "8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e",
    "folds": "5f5d09e9356f227ecb4a063270b175bb5cae20afb25636c563db185e18a155c4",
    "pool": "caa1b90769720a4accbe07074dbc7efe0335ab6657fea80c6839b60121dc39d3",
    "champion": "aa012114107c06532cf51c0fa9c741f5949146428cf266cf4bedded783d20e09",
    "external_ledger": "5dca2d01acc320299ae41d396a1cc6a2e5777614ec665c4b039eed4efd036d3c",
    "extended_evidence": "9893c49fa3e39306713ff6fa99e69af78dd0cb1c557cbf03ead16cb239c3b0b3",
    "assembly_manifest": "cb442519ea3120385f71c201b8fc2b313abcdb6994f2476f06d589b478aea480",
}
EXCLUDED_EXTERNAL = {"ext_szymon74:pub_rmlp", "ext_szymon74:pub_tabm"}
MAIN_VIEWS = ("final_242", "champion", "own_35", "external_207")
ROLE_FOLDS = {
    "discovery": (0, 1, 2),
    "refinement": (3,),
    "confirmation": (4,),
}
STAGE_CONTEXTS = {
    "discovery": ("fold_0", "fold_1", "fold_2", "fold_0_2"),
    "refinement": ("fold_3",),
    "confirmation": ("fold_4",),
}


class ContractError(RuntimeError):
    """동결 계약이나 입력 무결성이 깨졌을 때 분석 전체를 중단한다."""


@dataclass
class Inputs:
    train: pd.DataFrame
    test: pd.DataFrame
    folds: pd.Series
    target: pd.Series
    missing_weight: pd.Series
    members: pd.DataFrame
    champion: pd.Series
    pool: dict[str, Any]
    assembly: dict[str, Any]
    own_columns: list[str]
    external_columns: list[str]
    member_groups: dict[str, list[str]]
    input_manifest: dict[str, Any]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray | pd.Series) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def write_json(path: Path, value: Any, *, canonical: bool = False) -> None:
    data = (
        canonical_bytes(value)
        if canonical
        else (
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def one_run_artifact(root: Path, run_id: str, relative: str) -> Path:
    paths = sorted(root.glob(f"mlruns/*/{run_id}/artifacts/{relative}"))
    require(len(paths) == 1, f"실행 {run_id}의 {relative} 경로가 하나가 아니다: {paths}")
    return paths[0]


def _load_ledger_array(root: Path, spec: str) -> np.ndarray:
    match = re.fullmatch(r"(.+?)\[(.+)\]", spec)
    if match is None:
        values = np.load(root / spec).astype(np.float64)
        require(
            values.ndim == 1 or (values.ndim == 2 and values.shape[1] == 1),
            f"장부 배열 차원이 다르다: {spec} {values.shape}",
        )
        return values.reshape(-1)
    path_text, selector = match.groups()
    path = root / path_text
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=[selector])[selector].to_numpy(np.float64)
    require(path.suffix == ".npy", f"지원하지 않는 장부 배열 표기다: {spec}")
    column = int(selector.split(",")[1])
    return np.load(path)[:, column].astype(np.float64)


def _pool_audit_hashes(path: Path) -> dict[tuple[str, str], str]:
    hashes: dict[tuple[str, str], str] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8:
            continue
        run_id = cells[1].strip("`")
        oof_hash = cells[6].strip("`")
        if re.fullmatch(r"[0-9a-f]{32}", run_id) and re.fullmatch(
            r"[0-9a-f]{64}", oof_hash
        ):
            hashes[(cells[0], run_id)] = oof_hash
    return hashes


def _load_oof(root: Path, run_id: str, ids: pd.Series, folds: pd.Series) -> tuple[pd.Series, Path]:
    path = one_run_artifact(root, run_id, "oof.parquet")
    frame = pd.read_parquet(path)
    require(list(frame.columns) == [ID, "fold", "pred"], f"OOF 열이 다르다: {path}")
    require(frame[ID].equals(ids), f"OOF id 순서가 다르다: {run_id}")
    require(
        np.array_equal(frame["fold"].to_numpy(), folds.to_numpy()),
        f"OOF 분할이 다르다: {run_id}",
    )
    prediction = pd.Series(
        frame["pred"].to_numpy(np.float64), index=pd.Index(ids, name=ID), name=run_id
    )
    require(np.isfinite(prediction.to_numpy()).all(), f"OOF에 비유한 값이 있다: {run_id}")
    return prediction, path


def _missingness_weights(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    bits = np.left_shift(np.int64(1), np.arange(len(FEATURES), dtype=np.int64))

    def patterns(frame: pd.DataFrame) -> pd.Series:
        values = (frame[FEATURES].isna().to_numpy() * bits).sum(axis=1)
        return pd.Series(values, index=pd.Index(frame[ID], name=ID), dtype=np.int64)

    train_pattern = patterns(train)
    test_pattern = patterns(test)
    train_share = train_pattern.value_counts(normalize=True)
    test_share = test_pattern.value_counts(normalize=True)
    weight = (
        train_pattern.map(test_share).fillna(0.0) / train_pattern.map(train_share)
    ).astype(np.float64)
    pattern_rows = []
    train_count = train_pattern.value_counts()
    for key in sorted(train_share.index):
        pattern_rows.append(
            {
                "pattern": int(key),
                "missing_columns": [
                    column
                    for position, column in enumerate(FEATURES)
                    if int(key) >> position & 1
                ],
                "train_rows": int(train_count[key]),
                "train_share": float(train_share[key]),
                "test_share": float(test_share.get(key, 0.0)),
                "weight": float(test_share.get(key, 0.0) / train_share[key]),
            }
        )
    return weight.rename("missing_weight"), {
        "feature_order": FEATURES,
        "train_pattern_count": int(len(train_share)),
        "test_pattern_count": int(len(test_share)),
        "test_only_pattern_count": int(len(set(test_share.index) - set(train_share.index))),
        "zero_weight_rows": int((weight == 0.0).sum()),
        "effective_sample_size": kish(weight.to_numpy()),
        "patterns": pattern_rows,
    }


def _provider_kinds(config: dict[str, Any]) -> set[str]:
    return {
        str(provider["kind"])
        for provider in config.get("features", {}).get("providers", [])
        if isinstance(provider, dict) and "kind" in provider
    }


def _groups(root: Path, pool: dict[str, Any]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    information = {
        "info_original_proxy": [],
        "info_exact_value": [],
        "info_missing_restore": [],
        "info_raw": [],
    }
    for member in pool["members"]:
        name = member["config"]
        config = yaml.safe_load((root / "configs" / f"{name}.yaml").read_text())
        model_kind = str(config["model"]["kind"])
        families.setdefault(f"family_{model_kind}", []).append(name)
        kinds = _provider_kinds(config)
        initial_kind = config.get("initial_score", {}).get("kind")
        memberships = []
        if kinds & {"original_knn", "original_prior", "original_cdf_diff"} or initial_kind == "original_proxy_lightgbm":
            memberships.append("info_original_proxy")
        if kinds & {"target_encoding", "frequency_encoding", "lattice_pair_te", "categorical_copies"} or model_kind == "lookup_transformer":
            memberships.append("info_exact_value")
        if kinds & {"constrained_impute_aux", "xgb_impute_aux"}:
            memberships.append("info_missing_restore")
        if not memberships:
            memberships.append("info_raw")
        for group in memberships:
            information[group].append(name)
    groups = {**dict(sorted(families.items())), **information}
    require(all(groups.values()), "비어 있는 모델 계열 또는 정보 관점이 있다.")
    return groups


def load_inputs(root: Path) -> Inputs:
    paths = {
        "train": root / "data/train.csv",
        "test": root / "data/test.csv",
        "folds": root / "artifacts/folds.parquet",
        "pool": root / "artifacts/pool.yaml",
        "champion": root / "artifacts/champion.yaml",
        "extended_evidence": root / "docs/research/extended-stack-ladder-evidence.json",
        "assembly_manifest": root / "docs/research/extended-stack-submission-manifest.json",
        "pool_oof_ledger": root / "run-logs/issue337/pool-audit.md",
        "external_cache": root / "run-logs/issue443/cache/ext209.parquet",
        "uv_lock": root / "uv.lock",
    }
    paths["external_ledger"] = one_run_artifact(root, FINAL_RUN_ID, "external-member-ledger.json")
    paths["submission_record"] = one_run_artifact(root, FINAL_RUN_ID, "submission_record.json")
    for name, path in paths.items():
        require(path.is_file(), f"입력 파일이 없다: {name} {path}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in EXPECTED_HASHES.items():
        require(hashes[name] == expected, f"{name} SHA-256 불일치: {hashes[name]}")

    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    require(list(train.columns) == [ID, *FEATURES, TARGET], "train 열과 순서가 다르다.")
    require(list(test.columns) == [ID, *FEATURES], "test 열과 순서가 다르다.")
    require(train[ID].dtype == np.dtype("int64"), "train id가 int64가 아니다.")
    require(test[ID].dtype == np.dtype("int64"), "test id가 int64가 아니다.")
    require(not train[ID].duplicated().any(), "train id가 중복됐다.")
    require(not test[ID].duplicated().any(), "test id가 중복됐다.")
    require(set(train[TARGET].unique()) == {0, 1}, "목표값이 이진 0, 1이 아니다.")

    fold_frame = pd.read_parquet(paths["folds"])
    require(list(fold_frame.columns) == [ID, "fold"], "분할 파일 열이 다르다.")
    require(fold_frame[ID].equals(train[ID]), "분할 id 순서가 train과 다르다.")
    require(sorted(fold_frame["fold"].unique()) == [0, 1, 2, 3, 4], "분할이 0부터 4까지가 아니다.")
    ids = pd.Index(train[ID], name=ID)
    folds = pd.Series(fold_frame["fold"].to_numpy(np.int8), index=ids, name="fold")
    target = pd.Series(train[TARGET].to_numpy(np.int8), index=ids, name=TARGET)

    pool = yaml.safe_load(paths["pool"].read_text())
    champion_ledger = yaml.safe_load(paths["champion"].read_text())
    assembly = json.loads(paths["assembly_manifest"].read_text())
    extended = json.loads(paths["extended_evidence"].read_text())
    external_ledger = json.loads(paths["external_ledger"].read_text())
    submission_record = json.loads(paths["submission_record"].read_text())
    require(len(pool["members"]) == 35, "자체 후보 풀이 35개가 아니다.")
    require(champion_ledger["run_id"] == CHAMPION_RUN_ID, "champion 실행이 다르다.")
    require(assembly["assembled"]["member_count"] == 242, "조립 manifest가 242개가 아니다.")
    require(assembly["assembled"]["nested_auc"] == EXPECTED_NESTED_AUC, "조립 nested 기준값이 다르다.")
    require(assembly["combiner"]["in_sample_oof_auc"] == EXPECTED_INSAMPLE_AUC, "in-sample 참고치가 다르다.")
    require(extended["configs"]["ablate_te_leak"]["member_count"] == 242, "판정 근거의 절제 구성이 242개가 아니다.")
    require(extended["configs"]["ablate_te_leak"]["best_nested_auc"] == EXPECTED_NESTED_AUC, "판정 근거의 nested 기준값이 다르다.")
    require(submission_record["artifacts_sha256"]["external-member-ledger.json"] == EXPECTED_HASHES["external_ledger"], "파생 실행의 외부 장부 계보가 다르다.")
    require(submission_record["artifacts_sha256"]["extended-stack-submission-manifest.json"] == EXPECTED_HASHES["assembly_manifest"], "파생 실행의 조립 manifest 계보가 다르다.")

    audit_hashes = _pool_audit_hashes(paths["pool_oof_ledger"])
    own: dict[str, np.ndarray] = {}
    own_records = []
    for member in pool["members"]:
        config = member["config"]
        run_id = member["run_id"]
        prediction, path = _load_oof(root, run_id, train[ID], folds)
        digest = array_sha256(prediction)
        require(audit_hashes.get((config, run_id)) == digest, f"자체 OOF 장부 해시 불일치: {config} {digest}")
        auc = float(roc_auc_score(target.to_numpy(), prediction.to_numpy()))
        require(abs(auc - float(member["oof_auc"])) <= BURDEN_TOLERANCE, f"자체 OOF AUC 불일치: {config}")
        own[config] = prediction.to_numpy()
        own_records.append(
            {
                "column": config,
                "run_id": run_id,
                "oof_path": str(path.relative_to(root)),
                "oof_file_sha256": sha256_file(path),
                "prediction_sha256": digest,
                "auc": auc,
            }
        )
    own_frame = pd.DataFrame(own, index=ids, dtype=np.float64)

    champion, champion_path = _load_oof(root, CHAMPION_RUN_ID, train[ID], folds)
    champion_auc = float(roc_auc_score(target.to_numpy(), champion.to_numpy()))
    require(abs(champion_auc - float(champion_ledger["oof_auc"])) <= BURDEN_TOLERANCE, "champion OOF AUC가 장부와 다르다.")

    accepted = [row for row in external_ledger["members"] if row["status"] == "accepted"]
    require(len(accepted) == 209, "동결 외부 장부의 통과 구성원이 209개가 아니다.")
    external_cache = pd.read_parquet(paths["external_cache"])
    expected_cache_columns = [f"ext_{row['member_id']}" for row in accepted]
    require(list(external_cache.columns) == expected_cache_columns, "외부 캐시의 구성원 순서가 장부와 다르다.")
    require(external_cache.index.name == ID, "외부 캐시 인덱스가 id가 아니다.")
    require(external_cache.index.equals(ids), "외부 캐시 id 순서가 train과 다르다.")
    require(np.isfinite(external_cache.to_numpy()).all(), "외부 OOF 캐시에 비유한 값이 있다.")

    external_records = []
    final_external_columns = []
    for row in accepted:
        column = f"ext_{row['member_id']}"
        if column in EXCLUDED_EXTERNAL:
            continue
        oof = external_cache[column].to_numpy(np.float64)
        test_prediction = _load_ledger_array(root, row["test_path"])
        require(len(test_prediction) == len(test), f"외부 시험 예측 행 수 불일치: {column}")
        require(np.isfinite(test_prediction).all(), f"외부 시험 예측에 비유한 값이 있다: {column}")
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(oof, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(test_prediction, dtype=np.float64).tobytes())
        combined_hash = digest.hexdigest()
        require(combined_hash == row["sha256"], f"외부 배열 장부 해시 불일치: {column}")
        final_external_columns.append(column)
        external_records.append(
            {
                "column": column,
                "member_id": row["member_id"],
                "oof_test_prediction_sha256": combined_hash,
                "oof_path": row["oof_path"],
                "test_path": row["test_path"],
            }
        )
    require(len(final_external_columns) == 207, "최종 외부 구성원이 207개가 아니다.")
    external_frame = external_cache[final_external_columns].astype(np.float64)
    members = pd.concat([own_frame, external_frame], axis=1)
    manifest_columns = [row["column"] for row in assembly["members"]]
    require(list(members.columns) == manifest_columns, "최종 242개 구성원 순서가 조립 manifest와 다르다.")

    missing_weight, missingness_manifest = _missingness_weights(train, test)
    member_groups = _groups(root, pool)
    input_manifest = {
        "paths": {name: str(path.relative_to(root)) for name, path in paths.items()},
        "sha256": hashes,
        "rows": {"train": len(train), "test": len(test)},
        "columns": {"train": list(train.columns), "test": list(test.columns)},
        "fold_counts": {str(int(k)): int(v) for k, v in folds.value_counts().sort_index().items()},
        "target_counts": {str(int(k)): int(v) for k, v in target.value_counts().sort_index().items()},
        "own_members": own_records,
        "external_members": external_records,
        "member_order": list(members.columns),
        "champion": {
            "run_id": CHAMPION_RUN_ID,
            "oof_path": str(champion_path.relative_to(root)),
            "oof_file_sha256": sha256_file(champion_path),
            "prediction_sha256": array_sha256(champion),
            "auc": champion_auc,
        },
        "final_lineage": {
            "run_id": FINAL_RUN_ID,
            "source_run_id": submission_record["source_run_id"],
            "git_commit": submission_record["git_commit"],
            "submission_sha256": submission_record["submission"]["sha256"],
        },
        "missingness_reweighting": missingness_manifest,
    }
    return Inputs(
        train=train,
        test=test,
        folds=folds,
        target=target,
        missing_weight=missing_weight.reindex(ids),
        members=members,
        champion=champion,
        pool=pool,
        assembly=assembly,
        own_columns=list(own_frame.columns),
        external_columns=final_external_columns,
        member_groups=member_groups,
        input_manifest=input_manifest,
    )


def kish(weight: np.ndarray) -> float:
    weight = np.asarray(weight, dtype=np.float64)
    denominator = float(np.square(weight).sum())
    return 0.0 if denominator == 0.0 else float(weight.sum()) ** 2 / denominator


def rank_percentile(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average").to_numpy(np.float64)
    return (ranks - 0.5) / len(ranks)


def rank_burdens(
    prediction: np.ndarray, target: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    weight = np.asarray(weight, dtype=np.float64)
    standard = np.empty(len(prediction), dtype=np.float64)
    weighted = np.empty(len(prediction), dtype=np.float64)
    for value in (0, 1):
        own_mask = target == value
        opposite_mask = ~own_mask
        own_prediction = prediction[own_mask]
        opposite_prediction = prediction[opposite_mask]
        order = np.argsort(opposite_prediction, kind="mergesort")
        sorted_prediction = opposite_prediction[order]
        sorted_weight = weight[opposite_mask][order]
        cumulative_weight = np.concatenate(([0.0], np.cumsum(sorted_weight)))
        left = np.searchsorted(sorted_prediction, own_prediction, side="left")
        right = np.searchsorted(sorted_prediction, own_prediction, side="right")
        if value == 1:
            pair_loss = len(sorted_prediction) - right + 0.5 * (right - left)
            weighted_pair_loss = (
                cumulative_weight[-1]
                - cumulative_weight[right]
                + 0.5 * (cumulative_weight[right] - cumulative_weight[left])
            )
        else:
            pair_loss = left + 0.5 * (right - left)
            weighted_pair_loss = cumulative_weight[left] + 0.5 * (
                cumulative_weight[right] - cumulative_weight[left]
            )
        standard[own_mask] = pair_loss / len(sorted_prediction)
        opposite_weight_sum = float(cumulative_weight[-1])
        own_weight_sum = float(weight[own_mask].sum())
        require(opposite_weight_sum > 0.0 and own_weight_sum > 0.0, "역할 자료의 목표값 가중치 합이 0이다.")
        weighted[own_mask] = (
            weight[own_mask]
            * (int(own_mask.sum()) / own_weight_sum)
            * (weighted_pair_loss / opposite_weight_sum)
        )
    auc = float(roc_auc_score(target, prediction))
    weighted_auc = float(roc_auc_score(target, prediction, sample_weight=weight))
    checks: dict[str, float] = {"auc": auc, "weighted_auc": weighted_auc}
    for value in (0, 1):
        mask = target == value
        standard_delta = abs(float(standard[mask].mean()) - (1.0 - auc))
        weighted_delta = abs(float(weighted[mask].mean()) - (1.0 - weighted_auc))
        require(standard_delta <= BURDEN_TOLERANCE, f"표준 부담 평균 항등식 실패: y={value} {standard_delta}")
        require(weighted_delta <= BURDEN_TOLERANCE, f"가중 부담 평균 항등식 실패: y={value} {weighted_delta}")
        checks[f"standard_identity_delta_y{value}"] = standard_delta
        checks[f"weighted_identity_delta_y{value}"] = weighted_delta
    return standard, weighted, checks


def top_share_flags(values: np.ndarray, target: np.ndarray, share: float) -> tuple[np.ndarray, dict[str, float]]:
    flags = np.zeros(len(values), dtype=bool)
    thresholds = {}
    for value in (0, 1):
        mask = target == value
        block = values[mask]
        count = max(1, int(math.ceil(share * len(block))))
        threshold = float(np.partition(block, len(block) - count)[len(block) - count])
        flags[mask] = block >= threshold
        thresholds[str(value)] = threshold
    return flags, thresholds


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()


def role_frame(
    inputs: Inputs,
    role: str,
    nested_prediction: pd.Series,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    role_mask = inputs.folds.isin(ROLE_FOLDS[role]).to_numpy()
    positions = np.flatnonzero(role_mask)
    ids = inputs.members.index[positions]
    block = inputs.members.iloc[positions]
    member_ranks = block.rank(axis=0, method="average").to_numpy(
        dtype=np.float64, copy=True
    )
    member_ranks -= 0.5
    member_ranks /= len(block)
    column_position = {name: position for position, name in enumerate(block.columns)}

    predictions: dict[str, np.ndarray] = {
        "final_242": nested_prediction.iloc[positions].to_numpy(np.float64),
        "champion": inputs.champion.iloc[positions].to_numpy(np.float64),
        "own_35": member_ranks[:, : len(inputs.own_columns)].mean(axis=1),
        "external_207": member_ranks[:, len(inputs.own_columns) :].mean(axis=1),
    }
    for group, columns in inputs.member_groups.items():
        indices = [column_position[column] for column in columns]
        predictions[group] = member_ranks[:, indices].mean(axis=1)

    target = inputs.target.iloc[positions].to_numpy(np.int8)
    weight = inputs.missing_weight.iloc[positions].to_numpy(np.float64)
    frame_columns: dict[str, object] = {
        ID: ids.to_numpy(np.int64),
        "fold": inputs.folds.iloc[positions].to_numpy(np.int8),
        TARGET: target,
        "role": role,
        "missing_weight": weight,
    }
    integrity = []
    sensitivity = []
    hard: dict[tuple[str, str], np.ndarray] = {}
    rank_views: dict[str, np.ndarray] = {}
    for view, prediction in predictions.items():
        normalized = safe_name(view)
        rank = rank_percentile(prediction)
        standard, weighted, checks = rank_burdens(prediction, target, weight)
        rank_views[view] = rank
        frame_columns[f"prediction_{normalized}"] = prediction
        frame_columns[f"rank_{normalized}"] = rank
        frame_columns[f"burden_standard_{normalized}"] = standard
        frame_columns[f"burden_weighted_{normalized}"] = weighted
        for scale, burden in (("standard", standard), ("weighted", weighted)):
            official_flags, official_thresholds = top_share_flags(
                burden, target, OFFICIAL_HARD_SHARE
            )
            hard[(view, scale)] = official_flags
            frame_columns[f"hard_{scale}_{normalized}"] = official_flags
            for share in SENSITIVITY_SHARES:
                flags, thresholds = top_share_flags(burden, target, share)
                for value in (0, 1):
                    y_mask = target == value
                    sensitivity.append(
                        {
                            "role": role,
                            "view": view,
                            "scale": scale,
                            "share": share,
                            "target": value,
                            "threshold": thresholds[str(value)],
                            "rows": int((flags & y_mask).sum()),
                        }
                    )
            require(official_thresholds, "공식 난점 문턱을 만들지 못했다.")
        frame_columns[f"hard_both_{normalized}"] = (
            hard[(view, "standard")] & hard[(view, "weighted")]
        )
        integrity.append({"role": role, "view": view, **checks})

    for scale in ("standard", "weighted"):
        count = sum(hard[(view, scale)].astype(np.int8) for view in MAIN_VIEWS[1:])
        frame_columns[f"common_hard_{scale}"] = hard[("final_242", scale)] & (count >= 2)
    frame_columns["common_hard_both"] = (
        frame_columns["common_hard_standard"] & frame_columns["common_hard_weighted"]
    )

    comparisons = (
        ("own_external", "own_35", "external_207"),
        ("final_champion", "final_242", "champion"),
    )
    for name, left, right in comparisons:
        difference = rank_views[left] - rank_views[right]
        absolute = np.abs(difference)
        top_flags = absolute >= float(
            np.partition(absolute, len(absolute) - max(1, int(math.ceil(0.01 * len(absolute)))))[
                len(absolute) - max(1, int(math.ceil(0.01 * len(absolute))))
            ]
        )
        frame_columns[f"disagreement_{name}"] = difference
        frame_columns[f"disagreement_abs_{name}"] = absolute
        frame_columns[f"disagreement_direction_{name}"] = np.sign(difference).astype(np.int8)
        frame_columns[f"disagreement_official_{name}"] = absolute >= DISAGREEMENT_THRESHOLD
        frame_columns[f"disagreement_top1pct_{name}"] = top_flags
        frame_columns[f"target_advantage_{name}"] = (2 * target - 1) * difference

    own_count = len(inputs.own_columns)
    groups = {
        "final_members": member_ranks,
        "own_members": member_ranks[:, :own_count],
        "external_members": member_ranks[:, own_count:],
    }
    for name, values in groups.items():
        q25, q75 = np.quantile(values, (0.25, 0.75), axis=1, method="linear")
        frame_columns[f"iqr_{name}"] = q75 - q25
    return pd.DataFrame(frame_columns), integrity, sensitivity


def base_axes(train: pd.DataFrame) -> dict[str, np.ndarray]:
    axes: dict[str, np.ndarray] = {}
    for column in NUMERIC:
        axes[f"raw_{column}"] = train[column].to_numpy(np.float64)
    for column in CATEGORICAL:
        axes[f"raw_{column}"] = train[column].astype("object").to_numpy()
    for name in DERIVED:
        require(name in DERIVED_REGISTRY, f"파생 축 구현이 없다: {name}")
        axes[f"derived_{name}"] = DERIVED_REGISTRY[name](train).to_numpy(np.float64)
    missing = train[FEATURES].isna().to_numpy()
    bits = np.left_shift(np.int64(1), np.arange(len(FEATURES), dtype=np.int64))
    axes["missing_count"] = missing.sum(axis=1).astype(np.int8)
    axes["missing_mask"] = (missing * bits).sum(axis=1).astype(np.int64)
    for position, column in enumerate(FEATURES):
        axes[f"missing_{column}"] = missing[:, position]
    return axes


def role_axes(
    base: dict[str, np.ndarray], frame: pd.DataFrame, all_ids: pd.Series
) -> dict[str, np.ndarray]:
    position_of = pd.Series(
        np.arange(len(all_ids), dtype=np.int64), index=pd.Index(all_ids, name=ID)
    )
    positions = position_of.reindex(frame[ID]).to_numpy(np.int64)
    axes = {name: values[positions] for name, values in base.items()}
    for comparison in ("own_external", "final_champion"):
        axes[f"disagreement_abs_{comparison}"] = frame[
            f"disagreement_abs_{comparison}"
        ].to_numpy(np.float64)
        axes[f"disagreement_direction_{comparison}"] = frame[
            f"disagreement_direction_{comparison}"
        ].to_numpy(np.int8)
    for group in ("final_members", "own_members", "external_members"):
        axes[f"iqr_{group}"] = frame[f"iqr_{group}"].to_numpy(np.float64)
    return axes


def _number(value: float) -> str:
    return format(float(value), ".17g")


def condition_specs(
    discovery_axes: dict[str, np.ndarray], discovery_frame: pd.DataFrame
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    specs: list[dict[str, Any]] = []
    for column in FEATURES:
        specs.append(
            {
                "name": f"missing__{safe_name(column)}",
                "definition": f"{column} 값이 결측",
                "axis": f"missing_{column}",
                "axis_group": "missing_column",
                "kind": "equals",
                "value": True,
            }
        )
    for count in (0, 1, 2, 3):
        specs.append(
            {
                "name": f"missing_count__{count}",
                "definition": f"결측 개수가 {count}",
                "axis": "missing_count",
                "axis_group": "missing_count",
                "kind": "equals",
                "value": count,
            }
        )
    specs.append(
        {
            "name": "missing_count__4plus",
            "definition": "결측 개수가 4 이상",
            "axis": "missing_count",
            "axis_group": "missing_count",
            "kind": "at_least",
            "value": 4,
        }
    )

    target = discovery_frame[TARGET].to_numpy(np.int8)
    fold = discovery_frame["fold"].to_numpy(np.int8)
    for key in sorted(np.unique(discovery_axes["missing_mask"])):
        inside = discovery_axes["missing_mask"] == key
        qualifies = all(
            int((inside & (fold == split) & (target == value)).sum()) >= MIN_ROWS
            and int((~inside & (fold == split) & (target == value)).sum()) >= MIN_ROWS
            for split in (0, 1, 2)
            for value in (0, 1)
        )
        if not qualifies:
            continue
        missing_columns = [
            column
            for position, column in enumerate(FEATURES)
            if int(key) >> position & 1
        ]
        specs.append(
            {
                "name": f"missing_mask__{int(key):03x}",
                "definition": "정확한 결측 마스크: " + (", ".join(missing_columns) or "결측 없음"),
                "axis": "missing_mask",
                "axis_group": "missing_mask",
                "kind": "equals",
                "value": int(key),
            }
        )

    for column in CATEGORICAL:
        axis = f"raw_{column}"
        values = sorted({str(value) for value in discovery_axes[axis] if pd.notna(value)})
        for value in values:
            specs.append(
                {
                    "name": f"{axis}__{safe_name(value)}",
                    "definition": f"{column} 값이 {value}",
                    "axis": axis,
                    "axis_group": "raw_categorical",
                    "kind": "equals",
                    "value": value,
                }
            )

    direction_definitions = {
        -1: "오른쪽 관점의 순위가 높음",
        0: "두 관점의 순위가 같음",
        1: "왼쪽 관점의 순위가 높음",
    }
    for comparison in ("own_external", "final_champion"):
        axis = f"disagreement_direction_{comparison}"
        for value in sorted(np.unique(discovery_axes[axis])):
            specs.append(
                {
                    "name": f"{axis}__{int(value):+d}",
                    "definition": f"{comparison} 판단 갈림 방향: {direction_definitions[int(value)]}",
                    "axis": axis,
                    "axis_group": "disagreement_direction",
                    "kind": "equals",
                    "value": int(value),
                }
            )

    continuous_axes = [
        *(f"raw_{column}" for column in NUMERIC),
        *(f"derived_{name}" for name in DERIVED),
        "disagreement_abs_own_external",
        "disagreement_abs_final_champion",
        "iqr_final_members",
        "iqr_own_members",
        "iqr_external_members",
    ]
    boundaries: dict[str, list[float]] = {}
    for axis in continuous_axes:
        values = np.asarray(discovery_axes[axis], dtype=np.float64)
        finite = values[np.isfinite(values)]
        require(len(finite) > 0, f"연속축에 유한값이 없다: {axis}")
        raw_edges = np.quantile(finite, QUANTILES, method="linear").astype(float).tolist()
        boundaries[axis] = raw_edges
        edges: list[tuple[int, float]] = []
        for index, edge in enumerate(raw_edges):
            if not edges or edge > edges[-1][1]:
                edges.append((index, edge))
        for bin_index in range(len(edges) - 1):
            left_q, lower = edges[bin_index]
            right_q, upper = edges[bin_index + 1]
            right_closed = bin_index == len(edges) - 2
            specs.append(
                {
                    "name": f"{axis}__q{left_q:02d}_q{right_q:02d}",
                    "definition": f"{axis} 값이 {'['}{_number(lower)}, {_number(upper)}{']' if right_closed else ')'}",
                    "axis": axis,
                    "axis_group": (
                        "raw_numeric"
                        if axis.startswith("raw_")
                        else "derived_numeric"
                        if axis.startswith("derived_")
                        else "disagreement_value"
                        if axis.startswith("disagreement_")
                        else "member_iqr"
                    ),
                    "kind": "interval",
                    "lower": lower,
                    "upper": upper,
                    "right_closed": right_closed,
                }
            )
    specs.sort(key=lambda item: item["name"])
    require(len({item["name"] for item in specs}) == len(specs), "조건 이름이 중복됐다.")
    return specs, boundaries


def condition_mask(spec: dict[str, Any], axes: dict[str, np.ndarray]) -> np.ndarray:
    values = axes[spec["axis"]]
    if spec["kind"] == "equals":
        return values == spec["value"]
    if spec["kind"] == "at_least":
        return values >= spec["value"]
    if spec["kind"] == "interval":
        upper = values <= spec["upper"] if spec["right_closed"] else values < spec["upper"]
        return np.isfinite(values) & (values >= spec["lower"]) & upper
    raise AssertionError(spec["kind"])


def context_mask(frame: pd.DataFrame, context: str) -> np.ndarray:
    fold = frame["fold"].to_numpy(np.int8)
    if context == "fold_0_2":
        return np.isin(fold, (0, 1, 2))
    return fold == int(context.rsplit("_", 1)[1])


def target_stats(
    condition: np.ndarray,
    selected: np.ndarray,
    target: np.ndarray,
    burden: np.ndarray,
    weight: np.ndarray,
    value: int,
    weighted_scale: bool,
) -> dict[str, Any]:
    population = selected & (target == value)
    inside = population & condition
    outside = population & ~condition
    inside_rows = int(inside.sum())
    outside_rows = int(outside.sum())
    inside_ess = kish(weight[inside])
    outside_ess = kish(weight[outside])
    sample_ok = inside_rows >= MIN_ROWS and outside_rows >= MIN_ROWS
    if weighted_scale:
        sample_ok = sample_ok and inside_ess >= MIN_EFFECTIVE_ROWS and outside_ess >= MIN_EFFECTIVE_ROWS
    if not inside_rows or not outside_rows:
        increase = opportunity = float("nan")
    else:
        baseline = float(burden[population].mean())
        inside_mean = float(burden[inside].mean())
        outside_mean = float(burden[outside].mean())
        increase = inside_mean / baseline - 1.0
        opportunity = (inside_rows / int(population.sum())) * (inside_mean - outside_mean)
    return {
        "inside_rows": inside_rows,
        "outside_rows": outside_rows,
        "inside_effective_rows": inside_ess,
        "outside_effective_rows": outside_ess,
        "burden_increase": increase,
        "auc_loss_opportunity": opportunity,
        "sample_ok": sample_ok,
    }


def pass_status(sample_ok: bool, increase: float, opportunity: float, *, balanced_direction: bool = True) -> tuple[bool, str]:
    if not sample_ok:
        return False, "sample_too_small"
    if not balanced_direction:
        return False, "direction_not_positive"
    if not math.isfinite(increase) or increase < MIN_BURDEN_INCREASE:
        return False, "burden_increase_below_threshold"
    if not math.isfinite(opportunity) or opportunity < MIN_AUC_OPPORTUNITY:
        return False, "auc_loss_opportunity_below_threshold"
    return True, "passed"


def evaluate_condition(
    spec: dict[str, Any],
    condition: np.ndarray,
    frame: pd.DataFrame,
    context: str,
    scale: str,
    target_scope: str,
) -> dict[str, Any]:
    selected = context_mask(frame, context)
    target = frame[TARGET].to_numpy(np.int8)
    burden = frame[f"burden_{scale}_final_242"].to_numpy(np.float64)
    weight = frame["missing_weight"].to_numpy(np.float64)
    by_target = {
        value: target_stats(
            condition,
            selected,
            target,
            burden,
            weight,
            value,
            scale == "weighted",
        )
        for value in (0, 1)
    }
    if target_scope == "balanced":
        increase = float(np.mean([by_target[0]["burden_increase"], by_target[1]["burden_increase"]]))
        opportunity = float(np.mean([by_target[0]["auc_loss_opportunity"], by_target[1]["auc_loss_opportunity"]]))
        sample_ok = bool(by_target[0]["sample_ok"] and by_target[1]["sample_ok"])
        direction_ok = bool(by_target[0]["burden_increase"] > 0.0 and by_target[1]["burden_increase"] > 0.0)
        inside_rows = min(by_target[0]["inside_rows"], by_target[1]["inside_rows"])
        outside_rows = min(by_target[0]["outside_rows"], by_target[1]["outside_rows"])
        inside_ess = min(by_target[0]["inside_effective_rows"], by_target[1]["inside_effective_rows"])
        outside_ess = min(by_target[0]["outside_effective_rows"], by_target[1]["outside_effective_rows"])
    else:
        value = int(target_scope[-1])
        selected_stats = by_target[value]
        increase = selected_stats["burden_increase"]
        opportunity = selected_stats["auc_loss_opportunity"]
        sample_ok = selected_stats["sample_ok"]
        direction_ok = True
        inside_rows = selected_stats["inside_rows"]
        outside_rows = selected_stats["outside_rows"]
        inside_ess = selected_stats["inside_effective_rows"]
        outside_ess = selected_stats["outside_effective_rows"]
    passed, status = pass_status(sample_ok, increase, opportunity, balanced_direction=direction_ok)
    return {
        "condition_name": spec["name"],
        "condition_definition": spec["definition"],
        "axis": spec["axis"],
        "axis_group": spec["axis_group"],
        "context": context,
        "target_scope": target_scope,
        "loss_scale": scale,
        "inside_rows": inside_rows,
        "outside_rows": outside_rows,
        "inside_effective_rows": inside_ess,
        "outside_effective_rows": outside_ess,
        "inside_rows_y0": by_target[0]["inside_rows"],
        "outside_rows_y0": by_target[0]["outside_rows"],
        "inside_rows_y1": by_target[1]["inside_rows"],
        "outside_rows_y1": by_target[1]["outside_rows"],
        "inside_effective_rows_y0": by_target[0]["inside_effective_rows"],
        "outside_effective_rows_y0": by_target[0]["outside_effective_rows"],
        "inside_effective_rows_y1": by_target[1]["inside_effective_rows"],
        "outside_effective_rows_y1": by_target[1]["outside_effective_rows"],
        "burden_increase": increase,
        "auc_loss_opportunity": opportunity,
        "burden_increase_y0": by_target[0]["burden_increase"],
        "burden_increase_y1": by_target[1]["burden_increase"],
        "auc_loss_opportunity_y0": by_target[0]["auc_loss_opportunity"],
        "auc_loss_opportunity_y1": by_target[1]["auc_loss_opportunity"],
        "passed": passed,
        "status": status,
    }


def evaluate_stage(
    stage: str,
    specs: list[dict[str, Any]],
    axes: dict[str, np.ndarray],
    frame: pd.DataFrame,
    candidate_keys: set[tuple[str, str, str]] | None = None,
) -> tuple[list[dict[str, Any]], set[tuple[str, str, str]]]:
    rows = []
    passed_by_context: dict[tuple[str, str, str], list[bool]] = {}
    for spec in specs:
        condition = condition_mask(spec, axes)
        for scale in ("standard", "weighted"):
            for target_scope in ("target_0", "target_1", "balanced"):
                key = (spec["name"], target_scope, scale)
                if candidate_keys is not None and key not in candidate_keys:
                    continue
                for context in STAGE_CONTEXTS[stage]:
                    row = evaluate_condition(spec, condition, frame, context, scale, target_scope)
                    row["stage"] = stage
                    rows.append(row)
                    passed_by_context.setdefault(key, []).append(bool(row["passed"]))
    required = len(STAGE_CONTEXTS[stage])
    passed = {
        key
        for key, states in passed_by_context.items()
        if len(states) == required and all(states)
    }
    return rows, passed


def confirmed_summary(
    confirmed: set[tuple[str, str, str]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["context"] == "fold_0_2":
            continue
        key = (row["condition_name"], row["target_scope"], row["loss_scale"])
        if key in confirmed:
            by_key.setdefault(key, []).append(row)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for key, values in by_key.items():
        condition, target_scope, scale = key
        require(len(values) == 5, f"확인 조건의 분할 결과가 5개가 아니다: {key}")
        base = grouped.setdefault(
            (condition, target_scope),
            {
                "condition_name": condition,
                "condition_definition": values[0]["condition_definition"],
                "axis_group": values[0]["axis_group"],
                "target_scope": target_scope,
                "scales": [],
                "scale_results": {},
            },
        )
        opportunities = [float(value["auc_loss_opportunity"]) for value in values]
        increases = [float(value["burden_increase"]) for value in values]
        base["scales"].append(scale)
        base["scale_results"][scale] = {
            "auc_loss_opportunity_min": min(opportunities),
            "auc_loss_opportunity_median": float(np.median(opportunities)),
            "auc_loss_opportunity_max": max(opportunities),
            "burden_increase_min": min(increases),
            "burden_increase_median": float(np.median(increases)),
            "burden_increase_max": max(increases),
        }
    findings = []
    for value in grouped.values():
        value["scales"].sort()
        value["scale_status"] = (
            "both_scales" if value["scales"] == ["standard", "weighted"] else f"{value['scales'][0]}_only"
        )
        minima = [result["auc_loss_opportunity_min"] for result in value["scale_results"].values()]
        medians = [result["auc_loss_opportunity_median"] for result in value["scale_results"].values()]
        value["sort_min_opportunity"] = min(minima)
        value["sort_median_opportunity"] = float(np.median(medians))
        findings.append(value)
    scale_order = {"both_scales": 0, "standard_only": 1, "weighted_only": 2}
    findings.sort(
        key=lambda value: (
            scale_order[value["scale_status"]],
            -value["sort_min_opportunity"],
            -value["sort_median_opportunity"],
            value["condition_name"],
            value["target_scope"],
        )
    )
    return findings


def schema_of(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
    }


def markdown_report(evidence: dict[str, Any]) -> str:
    result = evidence["result"]
    lines = [
        "# 행별 OOF 난점 분석",
        "",
        "[동결한 측정 계약으로 행별 OOF 난점 분석을 실행한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/459)의 실행 결과다.",
        "[행별 OOF 난점 분석의 측정 계약을 완성한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/460)의 해답을 바꾸지 않고 구현했다.",
        "",
        "## 결론",
        "",
    ]
    if result["confirmed_count"] == 0:
        lines.extend(
            [
                "동결한 행별 OOF 난점 분석 계약에서 다섯 분할과 최소 손실 여지 문턱을 모두 통과한 오류 조건은 확인되지 않았다.",
                "현재 고정 탐색 축만으로는 개선 실험을 발주할 근거가 없다.",
            ]
        )
    else:
        lines.append(f"다섯 분할 확인 관문을 통과한 오류 조건 범위는 {result['confirmed_count']}개다.")
        lines.append("이 조건들은 관찰된 연관성이며 단일 변화 대조가 전체 OOF를 개선하기 전에는 원인으로 부르지 않는다.")
        lines.extend(["", "| 조건 | 목표값 범위 | 눈금 | 다섯 분할 최소 AUC 손실 여지 |", "| --- | --- | --- | ---: |"])
        for finding in result["confirmed_findings"]:
            lines.append(
                f"| `{finding['condition_name']}` | {finding['target_scope']} | {finding['scale_status']} | {finding['sort_min_opportunity']:.8f} |"
            )
    lines.extend(
        [
            "",
            "## 계약 무결성",
            "",
            f"최종 242개 `shrunk_rank_logit_logistic` nested OOF AUC는 `{evidence['nested']['auc']:.16f}`로 동결 기준값과 절대 차이 `{evidence['nested']['absolute_delta']:.3e}`다.",
            f"입력 해시, 행 수, 식별자 순서와 유일성, 목표값, 분할, 구성원 {evidence['inputs']['member_count']}개의 순서와 예측 유한성 검사를 모두 통과했다.",
            f"자체 35개 OOF는 기존 감사 장부의 little-endian float64 배열 해시와 일치했고, 외부 207개는 동결 장부의 OOF와 시험 예측 결합 해시와 일치했다.",
            f"표준 및 가중 순위 손실 부담 항등식의 최대 절대 오차는 각각 `{evidence['burden_integrity']['max_standard_identity_delta']:.3e}`, `{evidence['burden_integrity']['max_weighted_identity_delta']:.3e}`다.",
            f"계약 위반 수는 `{len(evidence['contract_violations'])}`이다.",
            "",
            "## 후보 소거",
            "",
            "| 단계 | 진입 범위 | 통과 범위 |",
            "| --- | ---: | ---: |",
            f"| 탐색 | {result['discovery_evaluated']} | {result['discovery_passed']} |",
            f"| 정제 | {result['refinement_evaluated']} | {result['refinement_passed']} |",
            f"| 확인 | {result['confirmation_evaluated']} | {result['confirmation_passed']} |",
            "",
            "한 범위는 조건, 목표값 범위와 손실 눈금의 조합 하나다.",
            "탐색은 분할 0, 1, 2 각각과 합친 자료를 모두 통과해야 하며 정제는 분할 3, 확인은 분할 4를 사용했다.",
            "",
            "## 산출물",
            "",
            "| 파일 | 행 수 | SHA-256 |",
            "| --- | ---: | --- |",
        ]
    )
    for artifact in evidence["artifacts"]:
        rows = artifact.get("rows")
        lines.append(
            f"| `{artifact['path']}` | {'-' if rows is None else rows} | `{artifact['sha256']}` |"
        )
    lines.extend(
        [
            "",
            f"분석 실행 자체는 `{evidence['execution']['analysis_seconds']:.1f}`초였고 전체 명령은 `{evidence['execution']['total_seconds']:.1f}`초였다.",
            "대용량 행별 결과는 커밋하지 않고 내용 해시로 계보를 남긴다.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(input_root: Path, output_dir: Path) -> None:
    started = time.monotonic()
    docs_evidence = REPOSITORY_ROOT / "docs/research/oof-row-difficulty-evidence.json"
    docs_report = REPOSITORY_ROOT / "docs/research/oof-row-difficulty.md"
    for path in (output_dir, docs_evidence, docs_report):
        require(not path.exists(), f"기존 출력이 있어 중단한다: {path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / ".issue459.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ContractError(f"분석 잠금이 이미 있다: {lock_path}") from exc
    os.close(lock_fd)
    temporary_root = Path(tempfile.mkdtemp(prefix=".issue459-", dir=output_dir.parent))
    large_dir = temporary_root / "large"
    commit_dir = temporary_root / "commit"
    large_dir.mkdir()
    commit_dir.mkdir()
    try:
        print("[1/7] 입력과 242개 구성원 해시를 검사한다.", flush=True)
        inputs = load_inputs(input_root)

        print("[2/7] 최종 242개 nested OOF를 다시 계산한다.", flush=True)
        nested_started = time.monotonic()
        combiner = ensemble.ShrunkRankLogitCombiner(fold_of=inputs.folds)
        evaluation = ensemble.evaluate_nested(combiner, inputs.members, inputs.folds, inputs.target)
        nested_seconds = time.monotonic() - nested_started
        nested_delta = abs(evaluation.nested_auc - EXPECTED_NESTED_AUC)
        require(nested_delta <= NESTED_TOLERANCE, f"nested OOF AUC 기준값 불일치: {evaluation.nested_auc}")
        expected_fold_aucs = inputs.assembly["assembled"]["fold_aucs"]
        fold_aucs = {str(item.fold): item.auc for item in evaluation.folds}
        require(
            all(abs(fold_aucs[key] - expected_fold_aucs[key]) <= NESTED_TOLERANCE for key in fold_aucs),
            "nested OOF 분할 AUC가 조립 manifest와 다르다.",
        )
        nested_prediction = evaluation.prediction.reindex(inputs.members.index)

        base = base_axes(inputs.train)
        all_frames = []
        burden_integrity = []
        sensitivity = []
        condition_rows: list[dict[str, Any]] = []

        print("[3/7] 분할 0부터 2까지의 탐색 난점과 조건을 고정한다.", flush=True)
        discovery_frame, checks, shares = role_frame(inputs, "discovery", nested_prediction)
        all_frames.append(discovery_frame)
        burden_integrity.extend(checks)
        sensitivity.extend(shares)
        discovery_axes = role_axes(base, discovery_frame, inputs.train[ID])
        specs, boundaries = condition_specs(discovery_axes, discovery_frame)
        discovery_rows, discovery_passed = evaluate_stage(
            "discovery", specs, discovery_axes, discovery_frame
        )
        condition_rows.extend(discovery_rows)
        discovery_freeze = {
            "schema_version": 1,
            "quantiles": list(QUANTILES),
            "conditions": specs,
            "continuous_boundaries": boundaries,
            "candidate_keys": [list(key) for key in sorted(discovery_passed)],
        }
        discovery_freeze_path = temporary_root / "discovery-freeze.json"
        write_json(discovery_freeze_path, discovery_freeze, canonical=True)
        discovery_freeze_hash = sha256_file(discovery_freeze_path)

        print("[4/7] 분할 3에서 탐색 통과 범위를 정제하고 다시 고정한다.", flush=True)
        refinement_frame, checks, shares = role_frame(inputs, "refinement", nested_prediction)
        all_frames.append(refinement_frame)
        burden_integrity.extend(checks)
        sensitivity.extend(shares)
        refinement_axes = role_axes(base, refinement_frame, inputs.train[ID])
        refinement_rows, refinement_passed = evaluate_stage(
            "refinement",
            specs,
            refinement_axes,
            refinement_frame,
            discovery_passed,
        )
        condition_rows.extend(refinement_rows)
        refinement_freeze = {
            "schema_version": 1,
            "discovery_freeze_sha256": discovery_freeze_hash,
            "candidate_keys": [list(key) for key in sorted(refinement_passed)],
        }
        refinement_freeze_path = temporary_root / "refinement-freeze.json"
        write_json(refinement_freeze_path, refinement_freeze, canonical=True)
        refinement_freeze_hash = sha256_file(refinement_freeze_path)

        print("[5/7] 분할 4에서 정제 통과 범위를 마지막으로 확인한다.", flush=True)
        confirmation_frame, checks, shares = role_frame(inputs, "confirmation", nested_prediction)
        all_frames.append(confirmation_frame)
        burden_integrity.extend(checks)
        sensitivity.extend(shares)
        confirmation_axes = role_axes(base, confirmation_frame, inputs.train[ID])
        confirmation_rows, confirmed = evaluate_stage(
            "confirmation",
            specs,
            confirmation_axes,
            confirmation_frame,
            refinement_passed,
        )
        condition_rows.extend(confirmation_rows)
        findings = confirmed_summary(confirmed, condition_rows)

        print("[6/7] 행별 결과와 조건 판독 결과를 쓴다.", flush=True)
        row_metrics = (
            pd.concat(all_frames, ignore_index=True)
            .sort_values(ID, kind="stable")
            .reset_index(drop=True)
        )
        row_metrics["fold"] = row_metrics["fold"].astype(np.int8)
        row_metrics[TARGET] = row_metrics[TARGET].astype(np.int8)
        require(row_metrics[ID].equals(inputs.train[ID]), "행별 결과 id 순서가 train과 다르다.")
        condition_results = pd.DataFrame(condition_rows)
        condition_results = condition_results.sort_values(
            ["condition_name", "target_scope", "loss_scale", "stage", "context"],
            kind="stable",
        ).reset_index(drop=True)
        row_path = large_dir / "row-metrics.parquet"
        condition_path = large_dir / "condition-results.parquet"
        row_metrics.to_parquet(row_path, index=False, compression="zstd")
        condition_results.to_parquet(condition_path, index=False, compression="zstd")

        analysis_seconds = time.monotonic() - nested_started
        runtime = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scikit_learn": sklearn.__version__,
        }
        manifest = {
            "schema_version": 1,
            "issue": ISSUE,
            "contract": {
                "nested_auc_expected": EXPECTED_NESTED_AUC,
                "nested_tolerance": NESTED_TOLERANCE,
                "burden_identity_tolerance": BURDEN_TOLERANCE,
                "minimum_rows": MIN_ROWS,
                "minimum_effective_rows": MIN_EFFECTIVE_ROWS,
                "minimum_burden_increase": MIN_BURDEN_INCREASE,
                "minimum_auc_loss_opportunity": MIN_AUC_OPPORTUNITY,
                "disagreement_threshold": DISAGREEMENT_THRESHOLD,
                "official_hard_share": OFFICIAL_HARD_SHARE,
            },
            "inputs": inputs.input_manifest,
            "member_groups": inputs.member_groups,
            "nested": {
                "strategy": "shrunk_rank_logit_logistic",
                "auc": evaluation.nested_auc,
                "absolute_delta": nested_delta,
                "fold_aucs": fold_aucs,
                "elapsed_seconds": nested_seconds,
                "prediction_sha256": array_sha256(nested_prediction),
            },
            "role_definitions": {key: list(value) for key, value in ROLE_FOLDS.items()},
            "burden_integrity": burden_integrity,
            "hard_row_sensitivity": sensitivity,
            "stage_freezes": {
                "discovery": {"sha256": discovery_freeze_hash, "content": discovery_freeze},
                "refinement": {"sha256": refinement_freeze_hash, "content": refinement_freeze},
            },
            "result": {
                "discovery_evaluated": len(specs) * 3 * 2,
                "discovery_passed": len(discovery_passed),
                "refinement_evaluated": len(discovery_passed),
                "refinement_passed": len(refinement_passed),
                "confirmation_evaluated": len(refinement_passed),
                "confirmation_passed": len(confirmed),
                "confirmed_findings": findings,
            },
            "schemas": {
                "row_metrics": schema_of(row_metrics),
                "condition_results": schema_of(condition_results),
            },
            "outputs": {
                "row-metrics.parquet": {
                    "sha256": sha256_file(row_path),
                    **schema_of(row_metrics),
                },
                "condition-results.parquet": {
                    "sha256": sha256_file(condition_path),
                    **schema_of(condition_results),
                },
            },
            "software": runtime,
            "uv_lock_sha256": inputs.input_manifest["sha256"]["uv_lock"],
            "analysis_seconds": analysis_seconds,
            "contract_violations": [],
        }
        manifest_path = large_dir / "manifest.json"
        write_json(manifest_path, manifest)

        artifact_rows = [
            {
                "path": "run-logs/issue459/row-metrics.parquet",
                "rows": len(row_metrics),
                "sha256": sha256_file(row_path),
            },
            {
                "path": "run-logs/issue459/condition-results.parquet",
                "rows": len(condition_results),
                "sha256": sha256_file(condition_path),
            },
            {
                "path": "run-logs/issue459/manifest.json",
                "rows": None,
                "sha256": sha256_file(manifest_path),
            },
        ]
        maximum_standard = max(
            check[f"standard_identity_delta_y{value}"]
            for check in burden_integrity
            for value in (0, 1)
        )
        maximum_weighted = max(
            check[f"weighted_identity_delta_y{value}"]
            for check in burden_integrity
            for value in (0, 1)
        )
        evidence = {
            "schema_version": 1,
            "issue": ISSUE,
            "result": {
                "confirmed_count": len(findings),
                "confirmed_findings": findings,
                "discovery_evaluated": len(specs) * 3 * 2,
                "discovery_passed": len(discovery_passed),
                "refinement_evaluated": len(discovery_passed),
                "refinement_passed": len(refinement_passed),
                "confirmation_evaluated": len(refinement_passed),
                "confirmation_passed": len(confirmed),
                "negative_conclusion": len(findings) == 0,
            },
            "inputs": {
                "member_count": len(inputs.members.columns),
                "sha256": inputs.input_manifest["sha256"],
            },
            "nested": manifest["nested"],
            "burden_integrity": {
                "max_standard_identity_delta": maximum_standard,
                "max_weighted_identity_delta": maximum_weighted,
            },
            "stage_freezes": {
                "discovery_sha256": discovery_freeze_hash,
                "refinement_sha256": refinement_freeze_hash,
            },
            "artifacts": artifact_rows,
            "execution": {
                "nested_seconds": nested_seconds,
                "analysis_seconds": analysis_seconds,
                "total_seconds": time.monotonic() - started,
            },
            "software": runtime,
            "contract_violations": [],
        }
        report_path = commit_dir / "oof-row-difficulty.md"
        evidence_path = commit_dir / "oof-row-difficulty-evidence.json"
        report_path.write_text(markdown_report(evidence))
        evidence["artifacts"].append(
            {
                "path": "docs/research/oof-row-difficulty.md",
                "rows": None,
                "sha256": sha256_file(report_path),
            }
        )
        evidence["script_sha256"] = sha256_file(Path(__file__).resolve())
        write_json(evidence_path, evidence)

        print("[7/7] 임시 산출물을 최종 경로에 확정한다.", flush=True)
        os.replace(large_dir, output_dir)
        os.replace(report_path, docs_report)
        os.replace(evidence_path, docs_evidence)
        print(
            f"완료: nested={evaluation.nested_auc:.16f}, 확인 조건 범위={len(findings)}, "
            f"전체 {time.monotonic() - started:.1f}초",
            flush=True,
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="행별 OOF 난점 분석 실행")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="data, mlruns, run-logs 비커밋 입력이 있는 저장소 checkout",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "run-logs/issue459",
        help="대용량 산출물 최종 디렉터리",
    )
    args = parser.parse_args()
    execute(args.input_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
