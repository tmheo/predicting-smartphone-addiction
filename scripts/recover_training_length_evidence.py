"""후보 풀 구성원의 원시 학습 길이 근거를 복원해 재학습 계획 장부를 다시 쓴다. (#374)

이 프로그램은 학습하지 않는다. 이미 끝난 실행이 남긴 자료에서 좌표별 **원시 선택값**만
다시 읽어, 재학습 계획 장부(`pipeline.refit_plan`의 문법 판본 3)를 통째로 생성한다.

원시 값의 출처는 세 가지이며 구성원마다 하나로 고정한다.

- 구조화 진단: 실행이 남긴 `model_training_diagnostics.json`의 fold 항목. 그 산출물을
  그대로 근거 산출물로 지목한다.
- 실행 로그: 그 시절 실행은 구조화 진단을 남기지 않았다. `logs/run.log`에서 좌표별
  원시 값을 다시 읽어 `training_length_evidence.json`으로 정리해 **그 실행에 붙이고**
  붙인 산출물을 근거로 지목한다. 로그 자체의 SHA-256도 그 안에 함께 적는다.
- 재실행 진단(#367): 로그의 표시 정밀도 때문에 후보가 둘 남은 네 셀은 원 실행 정체성으로
  다시 돌린 결과에서 확정값을 읽는다. 결과 파일과 결과 묶음 SHA-256을 같은 근거 산출물에
  적어 어느 셀이 어디서 왔는지 남긴다.

반복 수라는 개념이 없는 구성원은 관측을 만들지 않고 `not_applicable`로 적는다.

사용법:

    uv run --frozen python -m scripts.recover_training_length_evidence --check
    uv run --frozen python -m scripts.recover_training_length_evidence --attach --write

`--check`는 아무것도 바꾸지 않고 복원한 원시 값과 현재 장부를 견줘 보여 준다.
`--attach`는 실행 로그에서 복원한 구성원의 근거 산출물을 실행 저장소에 붙인다.
이미 붙어 있으면 그대로 두고 그 산출물의 해시를 쓴다.
`--write`는 `artifacts/full-refit-plan.yaml`을 문법 판본 3으로 다시 쓴다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import data  # noqa: E402
from pipeline.ledger import POOL_PATH, Pool  # noqa: E402
from pipeline.model import TRAINING_LENGTH_CONTRACTS  # noqa: E402
from pipeline.refit_plan import (  # noqa: E402
    SCHEMA_VERSION,
    STATUS_CONFIRMED,
    STATUS_NOT_APPLICABLE,
    TRAJECTORY_STATE_METHOD,
)
from pipeline.runs import MlflowRunStore, RunStore, RunStoreError, sha256_of  # noqa: E402
from pipeline.training_state_contract import (  # noqa: E402
    MANIFEST_NAME as TRAINING_STATE_MANIFEST,
    content_sha256,
)
from pipeline.training_length import (  # noqa: E402
    HALF_UP_ROUNDING,
    MEDIAN_STATISTIC,
    derive_refit_budgets,
    observe_training_length,
)

PLAN_PATH = Path("artifacts/full-refit-plan.yaml")
RUN_LOG_ARTIFACT = "logs/run.log"
STRUCTURED_ARTIFACT = "model_training_diagnostics.json"
RECOVERED_ARTIFACT = "training_length_evidence.json"
REMEASUREMENT_ARTIFACT = "training_length_remeasurement.json"
EVIDENCE_SCHEMA_VERSION = 1
MULTIPLIER = 1.25

# 이슈 #367의 재실행 결과. 결과 묶음 SHA-256은 그 이슈의 수락 기록과 같다.
ISSUE367_ROOT = Path("run-logs/issue367-training-length-diagnostic")
ISSUE367_BUNDLES = {
    "exp059": (
        "issue367-exp059-lookup-three-cells-v2-result",
        "f271d022517b01767ce8c2cfe9849ab8932d3a854b549bbe752344b7b76816a1",
    ),
    "exp133": (
        "issue367-exp133-seed44-fold0-result",
        "25b235525b5d2cce4eed619237a39f0d43237621a807f1c57f4acee2dec7283c",
    ),
}


class RecoveryError(RuntimeError):
    """원시 근거를 복원하지 못했다."""


# 아직 실행에 붙지 않은 근거 산출물. `--attach` 없이 돌려 볼 때만 채워진다.
PENDING: list[str] = []


@dataclass(frozen=True)
class Cell:
    """좌표 하나에서 읽은 원시 선택값과 그 출처."""

    seed: int
    outer_fold: int
    inner_member: int | None
    raw_value: int
    raw_path: str
    source: str


@dataclass(frozen=True)
class ConfirmedCell:
    """#367이 다시 돌려 확정한 셀. 로그가 남긴 후보 둘 가운데 하나를 고른다."""

    seed: int
    outer_fold: int
    bundle: str
    result_file: str


@dataclass(frozen=True)
class MemberRecovery:
    """구성원 하나의 복원 방법 선언."""

    model_family: str | None
    source: str  # "structured" | "declared" | "structured_remeasurement" | "run_log" | "none"
    reader: str | None = None
    confirmed: tuple[ConfirmedCell, ...] = ()
    measurement_run_id: str | None = None


def _cells(bundle_key: str, results: str, coordinates) -> tuple[ConfirmedCell, ...]:
    directory, _ = ISSUE367_BUNDLES[bundle_key]
    return tuple(
        ConfirmedCell(seed, fold, directory, f"{directory}/results/{results % name}")
        for seed, fold, name in coordinates
    )


# 구성원별 복원 선언. 순서는 후보 풀 순서를 따르므로 여기서는 이름으로만 찾는다.
RECOVERY: dict[str, MemberRecovery] = {
    "exp006_te_drop_gaming": MemberRecovery("lightgbm", "run_log", "lightgbm"),
    "exp011_resid_pair": MemberRecovery("lightgbm", "run_log", "lightgbm"),
    "exp022_orig_knn": MemberRecovery("lightgbm", "run_log", "lightgbm"),
    "exp023_orig_proxy_residual": MemberRecovery("lightgbm", "run_log", "lightgbm"),
    "exp025_constrained_impute": MemberRecovery("lightgbm", "run_log", "lightgbm"),
    "exp032_recon_orig_mean_top3": MemberRecovery("lightgbm", "run_log", "lightgbm"),
    "exp035_lattice_te": MemberRecovery("lightgbm", "run_log", "lightgbm"),
    "exp058_logreg_onehot": MemberRecovery(None, "none"),
    "exp059_lookup_transformer": MemberRecovery(
        "lookup_transformer",
        "run_log",
        "lookup_single",
        _cells(
            "exp059",
            "%s.json",
            ((42, 4, "seed42-fold4"), (43, 1, "seed43-fold1"), (44, 4, "seed44-fold4")),
        ),
    ),
    "exp070_cat_exact_cats": MemberRecovery("catboost", "run_log", "catboost"),
    "exp067_tabpfn3": MemberRecovery(None, "none"),
    "exp081_lookup_fold_initialization_avg3": MemberRecovery(
        "lookup_transformer", "run_log", "lookup_members"
    ),
    "exp110_lgb_kitopl_no_te": MemberRecovery("lightgbm", "run_log", "lightgbm"),
    "exp111_xgb_depth8_no_te": MemberRecovery("xgboost", "run_log", "xgboost"),
    "exp071_cat_exact_no_te": MemberRecovery("catboost", "run_log", "catboost"),
    "exp106_lookup_fixed24_train_test_preprocessing": MemberRecovery(
        "lookup_transformer", "structured", "lookup_members_structured"
    ),
    "exp117_ag25_gbm_r21": MemberRecovery("lightgbm", "run_log", "lightgbm"),
    "exp113_tab_cnn_m0": MemberRecovery("tab_cnn", "structured", "single"),
    "exp085_contextual_spline_m0": MemberRecovery(
        "contextualized_spline_transformer", "run_log", "contextualized_spline"
    ),
    "exp157_lookup_muon_initavg8": MemberRecovery(
        "lookup_transformer", "structured", "lookup_members_structured"
    ),
    "exp027_recon_ce": MemberRecovery("lightgbm", "run_log", "lightgbm"),
    "exp048_lgb_orig_cdf_diff": MemberRecovery("lightgbm", "run_log", "lightgbm"),
    "exp134_realmlp_muon": MemberRecovery("realmlp", "structured", "single"),
    "exp135_xgb_hpo_trial30": MemberRecovery("xgboost", "structured", "single"),
    "exp131_lookup_bivariate_plr5": MemberRecovery(
        "lookup_transformer", "structured", "lookup_members_structured"
    ),
    "exp136_realmlp_muon_recon_widths": MemberRecovery("realmlp", "structured", "single"),
    "exp137_tabm_recon_widths": MemberRecovery("tabm", "structured", "tabm_members"),
    "exp133_scalar_token_transformer_oof_te": MemberRecovery(
        "scalar_token_transformer",
        "run_log",
        "scalar_token",
        _cells("exp133", "exp133-%s.json", ((44, 0, "seed44-fold0"),)),
    ),
    "exp131_tab_cnn_oof_target_mean": MemberRecovery("tab_cnn", "structured", "single"),
    "exp132_tab_cnn_epochs100": MemberRecovery("tab_cnn", "structured", "single"),
    "exp139_realmlp_reference_qnormal_train_test": MemberRecovery(
        "realmlp", "structured", "single"
    ),
    "exp140_realmlp_orig_cdf_diff": MemberRecovery("realmlp", "structured", "single"),
    # 고정 반복 수 트리 변형(#413·#415·#419)은 실행이 진단 산출물에 관측 학습 길이를
    # 선언 형태로 직접 남기므로 그 관측을 그대로 읽는다.
    "exp168_issue413_lgb_no_te_fixed20": MemberRecovery("lightgbm", "declared"),
    "exp197_issue419_lgb_recon_ce_fixed20": MemberRecovery("lightgbm", "declared"),
    "exp183_issue419_cat_exact_fixed10": MemberRecovery("catboost", "declared"),
}

# 문법 판본 1이 쓰던 재학습 설정 경로. 원 실행 설정과 다른 구성원이 있어 그대로 옮긴다.
REFIT_CONFIG_PATHS = {"exp006_te_drop_gaming": "configs/refit/exp006_te_drop_gaming.yaml"}


# ----------------------------------------------------------------------------
# 실행 로그 읽기
# ----------------------------------------------------------------------------

def _fold_coordinates(seeds: tuple[int, ...], fold_count: int):
    """설정이 고정한 시드 순서와 그 안의 바깥쪽 분할 순서. 로그는 이 순서로 이어진다."""
    return [(seed, fold) for seed in seeds for fold in range(fold_count)]


def _lightgbm(lines: list[str]) -> list[list[list[int]]]:
    """조기 종료로 멈춘 fold와 최대 반복까지 간 fold가 서로 다른 문구를 쓴다."""
    picks = []
    for index, line in enumerate(lines):
        if line.lower().endswith("best iteration is:"):
            matched = re.match(r"\[(\d+)\]", lines[index + 1])
            if matched is None:
                raise RecoveryError(f"LightGBM 최적 반복 수 줄을 읽지 못했다: {lines[index + 1]!r}")
            picks.append([[int(matched.group(1))]])
    return picks


def _tagged(lines: list[str], tag: str) -> list[list[list[int]]]:
    pattern = re.compile(rf"\[{tag}\] early stopping: best_iteration=(\d+)")
    return [[[int(m.group(1))]] for m in map(pattern.match, lines) if m]


def _catboost(lines: list[str]) -> list[list[list[int]]]:
    return _tagged(lines, "catboost")


def _xgboost(lines: list[str]) -> list[list[list[int]]]:
    return _tagged(lines, "xgboost")


def _curve_argmax(
    lines: list[str], boundary: re.Pattern[str], point: re.Pattern[str]
) -> list[list[list[int]]]:
    """fold 경계로 잘라 각 fold에서 가장 높은 지표를 낸 위치 후보를 모은다.

    지표는 로그에 찍힌 문자열 그대로 견준다. 표시 정밀도 안에서 동점이면 후보가 둘
    이상 남고, 그 좌표는 로그만으로 확정할 수 없다.
    """
    folds: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] | None = None
    for line in lines:
        if boundary.search(line):
            current = []
            folds.append(current)
            continue
        matched = point.search(line)
        if matched is not None and current is not None:
            current.append((int(matched.group(1)), matched.group(2)))
    picks = []
    for curve in folds:
        if not curve:
            raise RecoveryError("fold 하나의 평가 곡선이 비었다.")
        best = max(score for _, score in curve)
        picks.append([[position for position, score in curve if score == best]])
    return picks


def _lookup_single(lines: list[str]) -> list[list[list[int]]]:
    return _curve_argmax(
        lines,
        re.compile(r"UserWarning: enable_nested_tensor"),
        re.compile(r"\[lookup_transformer\] ep\s+(\d+) valAUC=(\S+) best="),
    )


def _contextualized_spline(lines: list[str]) -> list[list[list[int]]]:
    return _curve_argmax(
        lines,
        re.compile(r"\[contextualized_spline_transformer\] \S+ parameters="),
        re.compile(r"epoch=(\d+) loss=\S+ add_auc=\S+ final_auc=(\S+)"),
    )


def _scalar_token(lines: list[str]) -> list[list[list[int]]]:
    return _curve_argmax(
        lines,
        re.compile(r"\[scalar_token_transformer\] \S+ features="),
        re.compile(r"epoch=(\d+) loss=\S+ ema_auc=(\S+)"),
    )


def _lookup_members(lines: list[str]) -> list[list[list[int]]]:
    """fold 초기화 구성원이 여럿인 판. 구성원은 자기 초기화 시드로 곡선을 구분한다."""
    folds: list[tuple[list[int], dict[int, list[tuple[int, str]]]]] = []
    order: list[int] | None = None
    curves: dict[int, list[tuple[int, str]]] | None = None
    member = re.compile(r"\[lookup_transformer\] member (\d+)/(\d+) seed=(\d+) ")
    point = re.compile(r"\[lookup_transformer\] seed=(\d+) ep\s+(\d+) valAUC=(\S+) best=")
    for line in lines:
        matched = member.match(line)
        if matched is not None:
            if matched.group(1) == "1":
                order, curves = [], {}
                folds.append((order, curves))
            if order is None or curves is None:
                raise RecoveryError("fold 초기화 구성원 줄이 첫 구성원 없이 나왔다.")
            order.append(int(matched.group(3)))
            curves[int(matched.group(3))] = []
            continue
        matched = point.search(line)
        if matched is not None and curves is not None:
            curves[int(matched.group(1))].append((int(matched.group(2)), matched.group(3)))
    picks = []
    for member_seeds, member_curves in folds:
        fold_picks = []
        for member_seed in member_seeds:
            curve = member_curves[member_seed]
            best = max(score for _, score in curve)
            fold_picks.append([position for position, score in curve if score == best])
        picks.append(fold_picks)
    return picks


LOG_READERS = {
    "lightgbm": _lightgbm,
    "catboost": _catboost,
    "xgboost": _xgboost,
    "lookup_single": _lookup_single,
    "lookup_members": _lookup_members,
    "contextualized_spline": _contextualized_spline,
    "scalar_token": _scalar_token,
}


# ----------------------------------------------------------------------------
# 구조화 진단 읽기
# ----------------------------------------------------------------------------

def _structured_single(details: dict, raw_field: str) -> list[int]:
    return [details[raw_field]]


def _structured_lookup_members(details: dict, raw_field: str) -> list[int]:
    return [member[raw_field] for member in details["fold_initialization_members"]]


def _structured_tabm_members(details: dict, raw_field: str) -> list[int]:
    return [member[raw_field] for member in details["members"]]


STRUCTURED_READERS = {
    "single": _structured_single,
    "lookup_members_structured": _structured_lookup_members,
    "tabm_members": _structured_tabm_members,
}


# ----------------------------------------------------------------------------
# 좌표별 원시 값 복원
# ----------------------------------------------------------------------------

@dataclass
class RecoveredEvidence:
    """구성원 하나의 복원 결과."""

    model_family: str
    raw_field: str
    raw_meaning: str
    cells: list[Cell]
    artifact_name: str
    artifact_payload: bytes | None = None
    sources: list[dict] = field(default_factory=list)


def _confirmed_value(cell: ConfirmedCell) -> tuple[int, str]:
    path = ISSUE367_ROOT / cell.result_file
    result = json.loads(path.read_text())
    if result.get("status") != "accepted":
        raise RecoveryError(f"{path}: 수락되지 않은 진단 결과다.")
    if result.get("seed") != cell.seed or result.get("outer_fold") != cell.outer_fold:
        raise RecoveryError(f"{path}: 결과 좌표가 선언과 다르다.")
    return int(result["raw_selection_value"]), str(path)


def _recover_from_log(
    store: RunStore, run_id: str, member: MemberRecovery, seeds: tuple[int, ...]
) -> RecoveredEvidence:
    contract = TRAINING_LENGTH_CONTRACTS[member.model_family]
    log_bytes = store.artifact_bytes_of(run_id, RUN_LOG_ARTIFACT)
    lines = log_bytes.decode(errors="replace").splitlines()
    candidates = LOG_READERS[member.reader](lines)
    coordinates = _fold_coordinates(seeds, 5)
    if len(candidates) != len(coordinates):
        raise RecoveryError(
            f"실행 로그에서 읽은 fold 수가 좌표 수와 다르다: "
            f"{len(candidates)} != {len(coordinates)}"
        )
    confirmed = {(cell.seed, cell.outer_fold): cell for cell in member.confirmed}
    cells: list[Cell] = []
    used_bundles: dict[str, list[str]] = {}
    for (seed, fold), fold_candidates in zip(coordinates, candidates, strict=True):
        multi = len(fold_candidates) > 1
        for index, choices in enumerate(fold_candidates):
            inner = index if multi else None
            if len(choices) == 1:
                cells.append(
                    Cell(seed, fold, inner, choices[0], RUN_LOG_ARTIFACT, "run_log")
                )
                continue
            cell = confirmed.get((seed, fold))
            if cell is None or multi:
                raise RecoveryError(
                    f"좌표 (seed={seed}, fold={fold})의 원시 값 후보가 {choices}로 남았고 "
                    "확정 근거가 없다."
                )
            value, raw_path = _confirmed_value(cell)
            if value not in choices:
                raise RecoveryError(
                    f"좌표 (seed={seed}, fold={fold})의 확정값 {value}가 로그 후보 {choices}에 없다."
                )
            used_bundles.setdefault(cell.bundle, []).append(f"seed{seed}-fold{fold}")
            cells.append(
                Cell(seed, fold, inner, value, raw_path, "issue367_diagnostic")
            )
    unused = set(confirmed) - {
        (cell.seed, cell.outer_fold)
        for cell in cells
        if cell.source == "issue367_diagnostic"
    }
    if unused:
        raise RecoveryError(f"쓰이지 않은 확정 근거 좌표가 있다: {sorted(unused)}")

    sources = [
        {
            "kind": "run_log",
            "artifact": RUN_LOG_ARTIFACT,
            "sha256": sha256_of(log_bytes),
        }
    ]
    for bundle, coordinates_used in sorted(used_bundles.items()):
        key = next(
            name for name, (directory, _) in ISSUE367_BUNDLES.items() if directory == bundle
        )
        sources.append(
            {
                "kind": "issue367_diagnostic",
                "issue": 367,
                "result_bundle": bundle,
                "result_bundle_sha256": ISSUE367_BUNDLES[key][1],
                "coordinates": sorted(coordinates_used),
            }
        )
    return RecoveredEvidence(
        model_family=member.model_family,
        raw_field=contract.raw_field,
        raw_meaning=contract.raw_meaning,
        cells=cells,
        artifact_name=RECOVERED_ARTIFACT,
        sources=sources,
    )


def _recover_from_structured(
    store: RunStore, run_id: str, member: MemberRecovery, seeds: tuple[int, ...]
) -> RecoveredEvidence:
    contract = TRAINING_LENGTH_CONTRACTS[member.model_family]
    entries = json.loads(store.artifact_bytes_of(run_id, STRUCTURED_ARTIFACT))
    reader = STRUCTURED_READERS[member.reader]
    raw_field = contract.raw_field.removesuffix("()")
    cells: list[Cell] = []
    for entry in entries:
        values = reader(entry["details"], raw_field)
        multi = len(values) > 1
        for index, value in enumerate(values):
            cells.append(
                Cell(
                    seed=int(entry["seed"]),
                    outer_fold=int(entry["fold"]),
                    inner_member=index if multi else None,
                    raw_value=int(value),
                    raw_path=f"{STRUCTURED_ARTIFACT}#details.{raw_field}",
                    source="structured_diagnostics",
                )
            )
    expected = set(_fold_coordinates(seeds, 5))
    if {(cell.seed, cell.outer_fold) for cell in cells} != expected:
        raise RecoveryError("구조화 진단의 좌표가 설정 시드·분할과 다르다.")
    return RecoveredEvidence(
        model_family=member.model_family,
        raw_field=contract.raw_field,
        raw_meaning=contract.raw_meaning,
        cells=cells,
        artifact_name=STRUCTURED_ARTIFACT,
        sources=[{"kind": "structured_diagnostics", "artifact": STRUCTURED_ARTIFACT}],
    )


def _recover_from_declared(
    store: RunStore, run_id: str, member: MemberRecovery, seeds: tuple[int, ...]
) -> RecoveredEvidence:
    """실행이 진단 산출물에 선언한 관측 학습 길이를 그대로 읽는다. (#413)

    조기 종료를 쓰지 않는 트리 변형은 `details` 대신 `training_length_evidence`에
    설정이 고정한 반복 수를 관측으로 남긴다.
    """
    entries = json.loads(store.artifact_bytes_of(run_id, STRUCTURED_ARTIFACT))
    cells: list[Cell] = []
    fields: set[tuple[str, str, str]] = set()
    for entry in entries:
        evidence = entry["training_length_evidence"]
        fields.add(
            (evidence["model_family"], evidence["raw_field"], evidence["raw_meaning"])
        )
        for observation in evidence["observations"]:
            cells.append(
                Cell(
                    seed=int(observation["seed"]),
                    outer_fold=int(observation["outer_fold"]),
                    inner_member=observation["inner_member"],
                    raw_value=int(observation["raw_value"]),
                    raw_path=f"{STRUCTURED_ARTIFACT}#training_length_evidence.observations",
                    source="structured_diagnostics",
                )
            )
    if len(fields) != 1:
        raise RecoveryError("선언된 관측의 계열·원시 필드·원시 의미가 하나로 모이지 않는다.")
    model_family, raw_field, raw_meaning = next(iter(fields))
    if model_family != member.model_family:
        raise RecoveryError(f"선언된 계열 {model_family}이 선언 {member.model_family}과 다르다.")
    expected = set(_fold_coordinates(seeds, 5))
    if {(cell.seed, cell.outer_fold) for cell in cells} != expected:
        raise RecoveryError("선언된 관측의 좌표가 설정 시드·분할과 다르다.")
    return RecoveredEvidence(
        model_family=model_family,
        raw_field=raw_field,
        raw_meaning=raw_meaning,
        cells=cells,
        artifact_name=STRUCTURED_ARTIFACT,
        sources=[{"kind": "structured_diagnostics", "artifact": STRUCTURED_ARTIFACT}],
    )


def _recover_from_structured_remeasurement(
    store: RunStore, member: MemberRecovery, seeds: tuple[int, ...]
) -> RecoveredEvidence:
    """별도 재측정 실행의 구조화 진단을 풀 구성원 실행에 붙일 근거로 바꾼다."""
    measurement_run_id = member.measurement_run_id
    if not measurement_run_id:
        raise RecoveryError("구조화 재측정 복원에는 measurement_run_id가 필요하다.")
    recovered = _recover_from_structured(store, measurement_run_id, member, seeds)
    source_sha256 = store.artifact_sha256_of(measurement_run_id, STRUCTURED_ARTIFACT)
    cells = [
        replace(
            cell,
            raw_path=(
                f"runs:/{measurement_run_id}/{STRUCTURED_ARTIFACT}"
                f"#details.{recovered.raw_field.removesuffix('()')}"
            ),
            source="structured_remeasurement",
        )
        for cell in recovered.cells
    ]
    return replace(
        recovered,
        cells=cells,
        artifact_name=REMEASUREMENT_ARTIFACT,
        sources=[
            {
                "kind": "structured_remeasurement",
                "run_id": measurement_run_id,
                "artifact": STRUCTURED_ARTIFACT,
                "sha256": source_sha256,
            }
        ],
    )


def _evidence_payload(run_id: str, evidence: RecoveredEvidence) -> bytes:
    """실행에 붙일 근거 산출물. 좌표마다 어느 자료에서 왔는지까지 남긴다."""
    from pipeline.training_length import observed_length_from_raw

    document = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "recovered_for_issue": 374,
        "source_run_id": run_id,
        "model_family": evidence.model_family,
        "raw_field": evidence.raw_field,
        "raw_meaning": evidence.raw_meaning,
        "converter": evidence.raw_meaning,
        "sources": evidence.sources,
        "observations": [
            {
                "seed": cell.seed,
                "outer_fold": cell.outer_fold,
                "inner_member": cell.inner_member,
                "raw_field": evidence.raw_field,
                "raw_path": cell.raw_path,
                "raw_value": cell.raw_value,
                "raw_meaning": evidence.raw_meaning,
                "observed_training_length": observed_length_from_raw(
                    cell.raw_value, evidence.raw_meaning
                ),
                "source": cell.source,
            }
            for cell in evidence.cells
        ],
    }
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


# ----------------------------------------------------------------------------
# 장부 조립
# ----------------------------------------------------------------------------

def _member_document(
    store: RunStore, pool_member, attach: bool
) -> dict:
    from pipeline.training_length import observed_length_from_raw

    name = pool_member.config
    run_id = pool_member.run_id
    seeds = tuple(pool_member.seeds)
    facts = store.facts_of(run_id)
    stored_config = store.config_of(run_id)
    source_config_name = f"{name}.yaml"
    lineage = {
        "source_run_id": run_id,
        "source_git_commit": facts.tags["git_commit"],
        "source_config_path": f"configs/{source_config_name}",
        "source_config_sha256": store.artifact_sha256_of(run_id, source_config_name),
    }
    if isinstance(stored_config, dict) and stored_config.get("training_state") is not None:
        return _training_state_member_document(
            store,
            pool_member,
            facts=facts,
            stored_config=stored_config,
            lineage=lineage,
        )

    recovery = RECOVERY.get(name)
    if recovery is None:
        raise RecoveryError(f"{name}: 복원 선언이 없다.")

    if recovery.source == "none":
        lineage["evidence_artifact_path"] = RUN_LOG_ARTIFACT
        lineage["evidence_artifact_sha256"] = store.artifact_sha256_of(
            run_id, RUN_LOG_ARTIFACT
        )
        return {
            "config": name,
            "config_path": REFIT_CONFIG_PATHS.get(name, f"configs/{name}.yaml"),
            "lineage": lineage,
            "training_length_evidence": {
                "status": STATUS_NOT_APPLICABLE,
                "model_family": _not_applicable_family(store, run_id),
                "converter": None,
                "observations": [],
            },
            "refit_budget_derivation": {
                "statistic": MEDIAN_STATISTIC,
                "multiplier": MULTIPLIER,
                "rounding": HALF_UP_ROUNDING,
                "seeds": [
                    {
                        "seed": seed,
                        "observed_lengths": [],
                        "median": None,
                        "scaled": None,
                        "budget": None,
                    }
                    for seed in _refit_seeds(store, run_id, seeds)
                ],
            },
        }

    if recovery.source == "structured_remeasurement":
        evidence = _recover_from_structured_remeasurement(store, recovery, seeds)
        payload = _evidence_payload(run_id, evidence)
        try:
            existing = store.artifact_bytes_of(run_id, evidence.artifact_name)
        except RunStoreError:
            existing = None
        if existing is None:
            if attach:
                evidence_sha256 = store.attach_artifact(
                    run_id, evidence.artifact_name, payload
                )
            else:
                PENDING.append(name)
                evidence_sha256 = sha256_of(payload)
        else:
            if existing != payload:
                raise RecoveryError(
                    f"{name}: 실행에 붙어 있는 {evidence.artifact_name}이 "
                    "재측정 복원 결과와 다르다."
                )
            evidence_sha256 = sha256_of(existing)
    elif recovery.source == "structured":
        evidence = _recover_from_structured(store, run_id, recovery, seeds)
        evidence_sha256 = store.artifact_sha256_of(run_id, evidence.artifact_name)
    elif recovery.source == "declared":
        evidence = _recover_from_declared(store, run_id, recovery, seeds)
        evidence_sha256 = store.artifact_sha256_of(run_id, evidence.artifact_name)
    else:
        evidence = _recover_from_log(store, run_id, recovery, seeds)
        payload = _evidence_payload(run_id, evidence)
        try:
            existing = store.artifact_bytes_of(run_id, evidence.artifact_name)
        except RunStoreError:
            existing = None
        if existing is None:
            if attach:
                evidence_sha256 = store.attach_artifact(
                    run_id, evidence.artifact_name, payload
                )
            else:
                PENDING.append(name)
                evidence_sha256 = sha256_of(payload)
        else:
            if existing != payload:
                raise RecoveryError(
                    f"{name}: 실행에 붙어 있는 {evidence.artifact_name}이 복원 결과와 다르다."
                )
            evidence_sha256 = sha256_of(existing)
    lineage["evidence_artifact_path"] = evidence.artifact_name
    lineage["evidence_artifact_sha256"] = evidence_sha256

    observations = [
        {
            "seed": cell.seed,
            "outer_fold": cell.outer_fold,
            "inner_member": cell.inner_member,
            "raw_field": evidence.raw_field,
            "raw_value": cell.raw_value,
            "raw_meaning": evidence.raw_meaning,
            "observed_training_length": observed_length_from_raw(
                cell.raw_value, evidence.raw_meaning
            ),
        }
        for cell in evidence.cells
    ]
    derivation = derive_refit_budgets(
        [
            observe_training_length(
                seed=cell.seed,
                outer_fold=cell.outer_fold,
                raw_field=evidence.raw_field,
                raw_value=cell.raw_value,
                raw_meaning=evidence.raw_meaning,
                inner_member=cell.inner_member,
            )
            for cell in evidence.cells
        ]
    )
    return {
        "config": name,
        "config_path": REFIT_CONFIG_PATHS.get(name, f"configs/{name}.yaml"),
        "lineage": lineage,
        "training_length_evidence": {
            "status": STATUS_CONFIRMED,
            "model_family": evidence.model_family,
            "converter": evidence.raw_meaning,
            "observations": observations,
        },
        "refit_budget_derivation": {
            "statistic": MEDIAN_STATISTIC,
            "multiplier": MULTIPLIER,
            "rounding": HALF_UP_ROUNDING,
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
        },
    }


def _training_state_member_document(
    store: RunStore,
    pool_member,
    *,
    facts,
    stored_config: dict,
    lineage: dict[str, str],
) -> dict:
    """완료된 후보 시점 실행에서 정확 종료 재학습 항목을 만든다."""
    from pipeline.training_length import observed_length_from_raw

    name = pool_member.config
    run_id = pool_member.run_id
    seeds = tuple(pool_member.seeds)
    state = stored_config.get("training_state")
    model_config = stored_config.get("model")
    if not isinstance(state, dict) or not isinstance(model_config, dict):
        raise RecoveryError(f"{name}: 학습 시점 설정 형식이 잘못됐다.")
    model_family = model_config.get("kind")
    if not isinstance(model_family, str):
        raise RecoveryError(f"{name}: 학습 시점 후보의 모델 계열이 없다.")
    expected_tags = {
        "run.kind": "training_state_snapshot",
        "training_state.ready": "true",
        "training_state.completed_epochs": str(state.get("selected")),
        "training_state.schedule_horizon_epochs": str(
            state.get("schedule_horizon_epochs")
        ),
        "training_state.state_kind": state.get("state_kind"),
        "training_state.selection_rule": "precommitted",
    }
    if facts.status != "FINISHED":
        raise RecoveryError(f"{name}: 학습 시점 후보 실행이 완료 상태가 아니다.")
    for key, expected in expected_tags.items():
        if facts.tags.get(key) != expected:
            raise RecoveryError(
                f"{name}: 학습 시점 후보 태그 {key}가 설정과 다르다."
            )

    try:
        manifest_payload = store.artifact_bytes_of(run_id, TRAINING_STATE_MANIFEST)
        manifest = json.loads(manifest_payload)
    except (RunStoreError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RecoveryError(
            f"{name}: 학습 시점 후보 manifest를 읽지 못했다: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise RecoveryError(f"{name}: 학습 시점 후보 manifest가 객체가 아니다.")
    claimed_content_sha = manifest.get("manifest_content_sha256")
    unhashed_manifest = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_content_sha256"
    }
    if claimed_content_sha != content_sha256(unhashed_manifest):
        raise RecoveryError(f"{name}: 학습 시점 후보 manifest 내용 해시가 다르다.")
    if facts.tags.get("sha256.training_state_manifest") != sha256_of(
        manifest_payload
    ):
        raise RecoveryError(f"{name}: 학습 시점 후보 manifest 파일 해시가 다르다.")

    trajectory_identity = facts.tags.get(
        "training_state.trajectory_identity_sha256"
    )
    candidate = manifest.get("candidate")
    expected_manifest = {
        "run_kind": "training_state_snapshot",
        "trajectory_identity_sha256": trajectory_identity,
        "completed_epochs": state.get("selected"),
        "schedule_horizon_epochs": state.get("schedule_horizon_epochs"),
        "trajectory_end_epochs": state.get("trajectory_end_epochs"),
        "state_kind": state.get("state_kind"),
        "selection_rule": "precommitted",
        "validation_target_used_for_selection": False,
        "precommitted_candidates": state.get("candidates"),
        "git_commit": lineage["source_git_commit"],
        "seeds": list(seeds),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RecoveryError(
                f"{name}: 학습 시점 후보 manifest의 {key}가 설정 또는 후보 풀과 다르다."
            )
    expected_candidate = {
        "config_name": name,
        "config_sha256": lineage["source_config_sha256"],
        "completed_epochs": state.get("selected"),
        "snapshot_identity_sha256": facts.tags.get(
            "training_state.snapshot_identity_sha256"
        ),
    }
    if not isinstance(candidate, dict) or any(
        candidate.get(key) != expected for key, expected in expected_candidate.items()
    ):
        raise RecoveryError(f"{name}: 학습 시점 후보 manifest의 후보 정체성이 다르다.")
    if manifest.get("candidate_set_sha256") != facts.tags.get(
        "training_state.candidate_set_sha256"
    ):
        raise RecoveryError(f"{name}: 학습 시점 후보 집합 정체성이 다르다.")
    if not isinstance(trajectory_identity, str) or len(trajectory_identity) != 64:
        raise RecoveryError(f"{name}: 학습 궤적 정체성 SHA-256 형식이 잘못됐다.")

    recovery = MemberRecovery(model_family, "declared")
    evidence = _recover_from_declared(store, run_id, recovery, seeds)
    evidence_sha256 = store.artifact_sha256_of(run_id, evidence.artifact_name)
    lineage["evidence_artifact_path"] = evidence.artifact_name
    lineage["evidence_artifact_sha256"] = evidence_sha256
    observations = [
        {
            "seed": cell.seed,
            "outer_fold": cell.outer_fold,
            "inner_member": cell.inner_member,
            "raw_field": evidence.raw_field,
            "raw_value": cell.raw_value,
            "raw_meaning": evidence.raw_meaning,
            "observed_training_length": observed_length_from_raw(
                cell.raw_value, evidence.raw_meaning
            ),
        }
        for cell in evidence.cells
    ]
    completed_epochs = state.get("selected")
    grouped = {seed: [] for seed in seeds}
    for observation in observations:
        grouped[observation["seed"]].append(observation["observed_training_length"])
    if any(
        not lengths or set(lengths) != {completed_epochs}
        for lengths in grouped.values()
    ):
        raise RecoveryError(
            f"{name}: 모든 시드와 분할의 관측 학습 길이가 선택 시점과 같지 않다."
        )
    return {
        "config": name,
        "config_path": REFIT_CONFIG_PATHS.get(name, f"configs/{name}.yaml"),
        "lineage": lineage,
        "training_length_evidence": {
            "status": STATUS_CONFIRMED,
            "model_family": evidence.model_family,
            "converter": evidence.raw_meaning,
            "observations": observations,
        },
        "refit_budget_derivation": {
            "method": TRAJECTORY_STATE_METHOD,
            "completed_epochs": completed_epochs,
            "schedule_horizon_epochs": state.get("schedule_horizon_epochs"),
            "state_kind": state.get("state_kind"),
            "trajectory_identity_sha256": trajectory_identity,
            "seeds": [
                {
                    "seed": seed,
                    "observed_lengths": grouped[seed],
                    "budget": completed_epochs,
                }
                for seed in seeds
            ],
        },
    }


def _not_applicable_family(store: RunStore, run_id: str) -> str:
    """반복 수가 없는 구성원의 계열 이름. 실행이 남긴 설정에서 그대로 읽는다."""
    return store.config_of(run_id)["model"]["kind"]


def _refit_seeds(store: RunStore, run_id: str, pool_seeds: tuple[int, ...]) -> list[int]:
    """전체 자료 재학습을 몇 번 하는가. 난수 시드가 결과를 바꾸지 않는 구성원만 하나다."""
    kind = _not_applicable_family(store, run_id)
    return [pool_seeds[0]] if kind == "logistic_onehot" else list(pool_seeds)


def build_plan(store: RunStore, pool: Pool, attach: bool) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_pool_sha256": data.file_sha256(POOL_PATH),
        "protocol": {
            "iteration_multiplier": MULTIPLIER,
            "budget_statistic": MEDIAN_STATISTIC,
            "budget_rounding": HALF_UP_ROUNDING,
            "cv_model_weight": 5,
            "full_model_weight": 1,
            "combiner": "shrunk_rank_logit_logistic",
        },
        "members": [
            _member_document(store, member, attach) for member in pool.members
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach", action="store_true", help="복원한 근거 산출물을 실행에 붙인다")
    parser.add_argument("--write", action="store_true", help="재학습 계획 장부를 다시 쓴다")
    parser.add_argument("--check", action="store_true", help="현재 장부와 견줘 보여 준다")
    parser.add_argument(
        "--member",
        help="지정한 구성원만 복원해 현재 장부의 같은 위치를 갱신한다",
    )
    args = parser.parse_args()

    store = MlflowRunStore()
    pool = Pool.load()
    current = yaml.safe_load(PLAN_PATH.read_text())
    current_budgets = _current_budgets(current)
    if args.member:
        try:
            pool_member = next(
                member for member in pool.members if member.config == args.member
            )
        except StopIteration as error:
            raise SystemExit(f"후보 풀에 없는 구성원이다: {args.member}") from error
        refreshed = _member_document(store, pool_member, attach=args.attach)
        plan = current
        plan["schema_version"] = SCHEMA_VERSION
        plan["source_pool_sha256"] = data.file_sha256(POOL_PATH)
        matches = [
            index
            for index, member in enumerate(plan["members"])
            if member["config"] == args.member
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"현재 장부에서 구성원 위치를 하나로 찾지 못했다: {args.member}"
            )
        plan["members"][matches[0]] = refreshed
        checked_members = [refreshed]
    else:
        plan = build_plan(store, pool, attach=args.attach)
        checked_members = plan["members"]

    if args.check or not args.write:
        for member in checked_members:
            name = member["config"]
            budgets = {
                seed["seed"]: seed["budget"]
                for seed in member["refit_budget_derivation"]["seeds"]
            }
            mark = "" if current_budgets.get(name) == budgets else "  <- 교정"
            print(f"{name:48} {budgets}{mark}")

    if PENDING:
        print(
            f"근거 산출물이 아직 실행에 붙지 않은 구성원 {len(PENDING)}개: "
            f"{', '.join(PENDING)}"
        )
    if args.write and PENDING:
        raise SystemExit("근거 산출물을 먼저 붙여야 장부를 쓸 수 있다. --attach를 함께 준다.")
    if args.write:
        PLAN_PATH.write_text(
            yaml.safe_dump(plan, allow_unicode=True, sort_keys=False, width=100)
        )
        print(f"{PLAN_PATH}를 문법 판본 {SCHEMA_VERSION}으로 다시 썼다.")


def _current_budgets(document: dict) -> dict[str, dict[int, int | None]]:
    budgets = {}
    for member in document.get("members", []):
        if "budgets" in member:
            budgets[member["config"]] = dict(member["budgets"])
        else:
            budgets[member["config"]] = {
                seed["seed"]: seed["budget"]
                for seed in member["refit_budget_derivation"]["seeds"]
            }
    return budgets


if __name__ == "__main__":
    main()
