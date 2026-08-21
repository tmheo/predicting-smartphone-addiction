"""후보 풀 소급 재심사의 읽기 전용 기준 장부를 동결한다. (#343)

사용법:
    uv run python -m scripts.freeze_pool_baseline
    uv run python scripts/freeze_pool_baseline.py --output artifacts/pool-baseline-2026-08-21.yaml

이 명령은 모델을 다시 학습하지 않고 장부와 실행 산출물만 읽는다.
champion·후보 풀 장부와 전체 자료 재학습 계획은 바꾸지 않는다.

동결 대상은 다섯 축이다.

1. 구성원 신원: config, 실행 식별자, 확정 시드, 진입 시점, 설정 파일 내용 해시.
2. 예측 무결성: `pipeline.pool_audit.verify_candidate`의 검증 결과와 OOF·시험 예측 배열 해시.
3. 모델 계열: 설정의 `model.kind` 하나.
4. 모델 계보 묶음: 문서로 확인된 이전판-개선판 간선의 연결 성분과 설정 차이.
5. 정보 관점: 컬럼 제공자 종류에서 유도한 예측 정보의 출처와 표현 원리.

여기에 구성원별 전체 자료 재학습 횟수를 전체 자료 재학습 계획에서 읽어 함께 고정한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline import pool_audit  # noqa: E402
from pipeline.ledger import Pool  # noqa: E402

POOL_PATH = Path("artifacts/pool.yaml")
REFIT_PLAN_PATH = Path("artifacts/full-refit-plan.yaml")
MANIFEST_PATH = Path("private-inputs.sha256")
DEFAULT_OUTPUT = Path("artifacts/pool-baseline-2026-08-21.yaml")
DEFAULT_REPORT = Path("docs/research/pool-reaudit-baseline.md")
SCHEMA_VERSION = 1
FROZEN_AT = "2026-08-21"
TICKET_ISSUE = 343
MAP_ISSUE = 338


class FreezeError(Exception):
    """기준 장부를 완결할 수 없는 전제 위반."""


@dataclass(frozen=True)
class LineageEdge:
    """문서로 확인된 이전판-개선판 간선 하나."""

    successor: str
    predecessor: str
    relation: str
    evidence: str


# 간선은 후보 풀 장부의 진입 근거나 그 근거가 가리키는 이슈가 이전판을 이름으로 지목한 경우에만 넣는다.
# 같은 모델 계열이라는 사실만으로는 간선을 만들지 않는다.
# 이전판이 현재 풀 밖이어도 간선을 남기며, 그 이전판을 공유하는 구성원들은 한 묶음이 된다.
LINEAGE_EDGES: tuple[LineageEdge, ...] = (
    LineageEdge(
        "exp011_resid_pair",
        "exp006_te_drop_gaming",
        "champion 개선판",
        "이슈 #46(현재 champion 위에서 산술 잔차 표현 결정)과 "
        "artifacts/pool.yaml 진입 근거: 새 champion exp011_resid_pair, 구 champion 대비 스피어만 0.99584",
    ),
    LineageEdge(
        "exp025_constrained_impute",
        "exp026_constrained_impute_nowidth",
        "품질 개선판 교체",
        "artifacts/pool.yaml 진입 근거(이슈 281): 폭 3열 포함판이 exp026보다 3시드 +0.00002705",
    ),
    LineageEdge(
        "exp032_recon_orig_mean_top3",
        "exp026_constrained_impute_nowidth",
        "특성 묶음 추가판",
        "artifacts/pool.yaml 진입 근거(이슈 #78): 원본 prior 상위 3열(m=20)의 exp026 베이스 재구성",
    ),
    LineageEdge(
        "exp033_recon_orig_mean_top3_raw",
        "exp032_recon_orig_mean_top3",
        "설정값 변형판",
        "artifacts/pool.yaml 진입 근거(이슈 #80): exp032의 평활 제거(m=0) 변형",
    ),
    LineageEdge(
        "exp027_recon_ce",
        "exp026_constrained_impute_nowidth",
        "특성 묶음 추가판",
        "artifacts/pool.yaml 진입 근거(이슈 282)와 설정 차이: exp026에 빈도 인코딩 8열 묶음 추가",
    ),
    LineageEdge(
        "exp048_lgb_orig_cdf_diff",
        "exp026_constrained_impute_nowidth",
        "특성 묶음 추가판",
        "artifacts/pool.yaml 진입 근거(이슈 283): CDF 차이 5열 묶음이 exp026보다 3시드 +0.00006208",
    ),
    LineageEdge(
        "exp070_cat_exact_cats",
        "exp057_cat_xgb_impute_comps5",
        "중복 교체판",
        "artifacts/pool.yaml 진입 근거(이슈 #107): exp057과 스피어만 0.99959 중복 교체",
    ),
    LineageEdge(
        "exp071_cat_exact_no_te",
        "exp070_cat_exact_cats",
        "특성 묶음 제거판",
        "artifacts/pool.yaml 진입 근거(이슈 #183): exp070에서 TE 제거",
    ),
    LineageEdge(
        "exp107_logreg_onehot_nn10",
        "exp058_logreg_onehot",
        "특성 묶음 추가판",
        "artifacts/pool.yaml 진입 근거(이슈 #200): exp058 계열 orig_nn10_mean 열 추가 변형",
    ),
    LineageEdge(
        "exp108_logreg_onehot_nn10_l1",
        "exp058_logreg_onehot",
        "특성 묶음 추가판",
        "artifacts/pool.yaml 진입 근거(이슈 #200): exp058 계열 orig_nn10_mean + L1 조합 변형",
    ),
    LineageEdge(
        "exp081_lookup_fold_initialization_avg3",
        "exp059_lookup_transformer",
        "학습 설정 개선판",
        "이슈 #127: 당시 Lookup-Transformer 기준 구성 위의 fold 내 초기화 3개 평균, 설정은 exp059의 진부분집합 확장",
    ),
    LineageEdge(
        "exp127_lookup_muon",
        "exp081_lookup_fold_initialization_avg3",
        "학습 설정 개선판",
        "artifacts/pool.yaml 진입 근거(이슈 196): exp081의 Muon 혼성 optimizer판",
    ),
    LineageEdge(
        "exp131_lookup_bivariate_plr5",
        "exp127_lookup_muon",
        "특성 묶음 추가판",
        "artifacts/pool.yaml 진입 근거(이슈 267): exp127에 수면·화면 이변수 조성 5열을 더한 판",
    ),
    LineageEdge(
        "exp110_lgb_kitopl_no_te",
        "exp074_lgb_kitopl_d2_bundle",
        "특성 묶음 제거판",
        "artifacts/pool.yaml 진입 근거(이슈 #183): exp074에서 TE 제거",
    ),
    LineageEdge(
        "exp117_ag25_gbm_r21",
        "exp074_lgb_kitopl_d2_bundle",
        "중복 교체판",
        "artifacts/pool.yaml 진입 근거(이슈 197): 최근접 exp074와 스피어만 0.99829 초과·성능 우위로 교체 진입",
    ),
    LineageEdge(
        "exp111_xgb_depth8_no_te",
        "exp045_xgb_depth8",
        "특성 묶음 제거판",
        "artifacts/pool.yaml 진입 근거(이슈 #183): exp045에서 TE 제거",
    ),
    LineageEdge(
        "exp135_xgb_hpo_trial30",
        "exp045_xgb_depth8",
        "중복 교체판",
        "artifacts/pool.yaml 진입 근거(이슈 288): exp045 중복 구성원을 더 높은 3시드 OOF AUC로 교체",
    ),
    LineageEdge(
        "exp124_realmlp_dtype_fix",
        "exp121_realmlp_fixed4_two_init",
        "결함 수정판 교체",
        "artifacts/pool.yaml 진입 근거(이슈 243): exp121의 dtype 정합 결함 수정판으로 exp121을 교체 진입",
    ),
    LineageEdge(
        "exp134_realmlp_muon",
        "exp124_realmlp_dtype_fix",
        "학습 설정 개선판",
        "artifacts/pool.yaml 진입 근거(이슈 280): exp124의 Muon 혼성 optimizer판",
    ),
    LineageEdge(
        "exp136_realmlp_muon_recon_widths",
        "exp134_realmlp_muon",
        "특성 묶음 추가판",
        "artifacts/pool.yaml 진입 근거(판정 issue302-exp136-realmlp-widths-confirm): exp134 직접 짝비교",
    ),
    LineageEdge(
        "exp139_realmlp_reference_qnormal_train_test",
        "exp136_realmlp_muon_recon_widths",
        "전처리 기준 범위 변형판",
        "이슈 #331 고정 실행 계약: 직접 기준 설정은 exp136_realmlp_muon_recon_widths",
    ),
    LineageEdge(
        "exp137_tabm_recon_widths",
        "exp065_tabm",
        "특성 묶음 추가판 교체",
        "artifacts/pool.yaml 진입 근거(판정 issue294-exp137-tabm-widths-confirm): exp065 직접 짝비교로 교체",
    ),
    LineageEdge(
        "exp131_tab_cnn_oof_target_mean",
        "exp113_tab_cnn_m0",
        "단일 변경 개선판",
        "이슈 #333: #303에서 exp113_tab_cnn_m0 대비 단일 변경으로 확인된 개선",
    ),
    LineageEdge(
        "exp132_tab_cnn_epochs100",
        "exp113_tab_cnn_m0",
        "단일 변경 개선판",
        "이슈 #333: #303에서 exp113_tab_cnn_m0 대비 단일 변경으로 확인된 개선",
    ),
)

# 컬럼 제공자 종류를 정보 관점 이름으로 옮긴다.
# 학습기 안에서만 쓰는 정보 사용은 모델 계열 축이 지므로 여기에 넣지 않는다.
PROVIDER_PERSPECTIVE: dict[str, str] = {
    "derived": "산술 파생 조합",
    "target_encoding": "목표값 부호화",
    "frequency_encoding": "정확값 빈도 부호화",
    "lattice_pair_te": "격자 이변수 목표값 부호화",
    "categorical_copies": "정확값 범주 복제",
    "original_knn": "원본 프록시 최근접 라벨",
    "original_prior": "원본 프록시 통계 사전",
    "original_cdf_diff": "원본 프록시 분포 좌표",
    "xgb_impute_aux": "학습 기반 결측 복원",
}
# 학습기 설정이 나르는 정보 관점은 CONTEXT.md가 이름 붙인 두 가지만 인정한다.
# 나머지 model.params 항목은 용량·최적화·실행 설정으로 보고 모델 계열 축이 진다.
MODEL_PARAM_PERSPECTIVE: dict[str, str] = {
    "lookup_cols": "정확값 어휘 조회",
    "exact_cols": "정확값 어휘 조회",
    "reference_qnormal_columns": "전처리 기준 집합 값 좌표",
}
DEFAULT_PREPROCESSING_SCOPE = "fold_train"
# `features.providers` 이전 스키마를 쓰는 구성원.
LEGACY_FEATURE_SCHEMA_MEMBERS = ("exp006_te_drop_gaming", "exp011_resid_pair")
CONSTRAINED_IMPUTE_KIND = "constrained_impute_aux"
CONSTRAINED_IMPUTE_VALUE = "제약 기반 결측 복원값"
CONSTRAINED_IMPUTE_WIDTH = "제약 기반 결측 복원 구간 폭"
RECON_WIDTH_SUFFIX = "_recon_width"
# 컬럼 제공자를 하나도 쓰지 않는 구성원은 원시 관측 열만 본다.
RAW_COLUMNS_ONLY = "원시 관측 열 전용"
# exp006·exp011은 `features.providers` 이전 스키마를 쓴다.
KNOWN_FEATURE_KEYS = frozenset(
    {"base", "categorical", "providers", "include", "placebo", "fold_fit", "derived"}
)


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{prefix}[{index}]"))
    else:
        flat[prefix] = value
    return flat


def config_delta(predecessor: dict, successor: dict) -> dict[str, object]:
    """설정 두 개의 잎 단위 차이를 이름 항목을 빼고 요약한다."""
    before = _flatten(predecessor)
    after = _flatten(successor)
    before.pop("name", None)
    after.pop("name", None)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(key for key in set(before) & set(after) if before[key] != after[key])
    if added and not removed and not changed:
        shape = "진부분집합 확장"
    elif removed and not added and not changed:
        shape = "진부분집합 축소"
    elif not added and not removed and not changed:
        shape = "차이 없음"
    else:
        shape = "혼합 변경"
    return {
        "shape": shape,
        "added_leaves": len(added),
        "removed_leaves": len(removed),
        "changed_leaves": len(changed),
        "changed_keys": changed[:20],
    }


def constrained_impute_perspectives(provider: dict) -> list[str]:
    """제약 기반 결측 복원 제공자가 실제로 내보내는 정보 관점을 가른다."""
    emit = provider.get("emit")
    widths = provider.get("widths", True)
    if emit is not None:
        emitted = list(emit)
        width_only = bool(emitted) and all(name.endswith(RECON_WIDTH_SUFFIX) for name in emitted)
        has_width = any(name.endswith(RECON_WIDTH_SUFFIX) for name in emitted)
        if width_only:
            return [CONSTRAINED_IMPUTE_WIDTH]
        return [CONSTRAINED_IMPUTE_VALUE] + ([CONSTRAINED_IMPUTE_WIDTH] if has_width else [])
    return [CONSTRAINED_IMPUTE_VALUE] + ([CONSTRAINED_IMPUTE_WIDTH] if widths else [])


def feature_providers(config: dict) -> list[dict]:
    """현재 스키마와 이전 스키마 양쪽에서 컬럼 제공자 목록을 읽는다.

    exp006과 exp011은 `features.providers` 이전의 `include`/`fold_fit`/`derived` 스키마를 쓴다.
    두 스키마 어느 쪽에도 해당하지 않으면 조용히 빈 목록으로 넘기지 않고 멈춘다.
    """
    features = config.get("features") or {}
    unknown = set(features) - KNOWN_FEATURE_KEYS
    if unknown:
        raise FreezeError(f"해석할 수 없는 features 항목: {sorted(unknown)}")
    if "providers" in features:
        return list(features.get("providers") or [])
    if "fold_fit" in features or "derived" in features:
        providers = list(features.get("fold_fit") or [])
        if features.get("derived"):
            providers.append({"kind": "derived", "names": list(features["derived"])})
        return providers
    raise FreezeError("features에 컬럼 제공자 항목이 없다.")


def model_param_perspectives(config: dict) -> list[str]:
    params = (config.get("model") or {}).get("params") or {}
    found = []
    for key, name in MODEL_PARAM_PERSPECTIVE.items():
        if params.get(key) and name not in found:
            found.append(name)
    return found


def preprocessing_reference_scope(config: dict) -> str:
    params = (config.get("model") or {}).get("params") or {}
    return str(params.get("preprocessing_scope") or DEFAULT_PREPROCESSING_SCOPE)


def member_perspectives(config: dict) -> list[str]:
    providers = feature_providers(config)
    from_model = model_param_perspectives(config)
    if not providers:
        return sorted(from_model) or [RAW_COLUMNS_ONLY]
    perspectives: list[str] = list(from_model)
    for provider in providers:
        kind = provider.get("kind")
        if kind == CONSTRAINED_IMPUTE_KIND:
            found = constrained_impute_perspectives(provider)
        elif kind in PROVIDER_PERSPECTIVE:
            found = [PROVIDER_PERSPECTIVE[kind]]
        else:
            raise FreezeError(f"정보 관점 이름이 없는 컬럼 제공자: {kind}")
        for name in found:
            if name not in perspectives:
                perspectives.append(name)
    return sorted(perspectives)


def config_subset_relations(configs: dict[str, dict], members: list[str]) -> dict[str, dict]:
    """풀 구성원 사이의 잎 단위 설정 포함 관계를 잰다.

    생략된 기본값은 잎으로 나타나지 않으므로 포함 관계는 의미 동일성을 뜻하지 않는다.
    모델 계보 묶음을 만들지 않고 참고 사실로만 남긴다.
    """
    leaves = {}
    for name in members:
        flat = _flatten(configs[name])
        flat.pop("name", None)
        leaves[name] = flat
    relations: dict[str, dict] = {}
    for name in members:
        subset_of, superset_of = [], []
        for other in members:
            if other == name:
                continue
            mine, theirs = leaves[name], leaves[other]
            if set(mine) < set(theirs) and all(theirs[key] == value for key, value in mine.items()):
                subset_of.append(other)
            if set(theirs) < set(mine) and all(mine[key] == value for key, value in theirs.items()):
                superset_of.append(other)
        relations[name] = {"config_subset_of": subset_of, "config_superset_of": superset_of}
    return relations


def lineage_groups(members: list[str]) -> tuple[dict[str, str], dict[str, dict]]:
    """문서 간선의 연결 성분으로 모델 계보 묶음을 만든다."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for name in members:
        find(name)
    for edge in LINEAGE_EDGES:
        if edge.successor not in members:
            raise FreezeError(f"간선의 개선판이 현재 풀에 없다: {edge.successor}")
        union(edge.predecessor, edge.successor)

    by_root: dict[str, list[str]] = {}
    for name in members:
        by_root.setdefault(find(name), []).append(name)

    order = {name: index for index, name in enumerate(members)}
    groups: dict[str, dict] = {}
    assignment: dict[str, str] = {}
    for root in sorted(by_root, key=lambda key: order[min(by_root[key], key=order.get)]):
        group_members = sorted(by_root[root], key=order.get)
        group_id = group_members[0]
        outside = sorted(
            {edge.predecessor for edge in LINEAGE_EDGES if edge.successor in group_members}
            - set(group_members)
        )
        groups[group_id] = {
            "members": group_members,
            "size": len(group_members),
            "predecessors_outside_pool": outside,
        }
        for name in group_members:
            assignment[name] = group_id
    return assignment, groups


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_hashes() -> dict[str, str]:
    hashes = {}
    for line in MANIFEST_PATH.read_text().splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        hashes[relative] = digest
    return hashes


def git_commit() -> str:
    shown = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return shown.stdout.strip()


def build(output: Path) -> dict:
    pool = Pool.load()
    members = [member.config for member in pool.members]
    if len(set(members)) != len(members):
        raise FreezeError("후보 풀에 같은 config가 두 번 있다.")

    plan = yaml.safe_load(REFIT_PLAN_PATH.read_text())
    plan_by_run = {entry["run_id"]: entry for entry in plan["members"]}
    if plan["source_pool_sha256"] != file_sha256(POOL_PATH):
        raise FreezeError("전체 자료 재학습 계획이 현재 후보 풀 해시를 가리키지 않는다.")

    context = pool_audit.load_context()
    candidates = pool_audit.load_mlflow_candidates(pool)
    checks = [pool_audit.verify_candidate(candidate, context) for candidate in candidates]
    retained, duplicates = pool_audit.deduplicate(checks)

    assignment, groups = lineage_groups(members)
    edge_by_successor = {edge.successor: edge for edge in LINEAGE_EDGES}

    # 구성원 설정은 실행 산출물의 설정을 쓴다.
    # verify_candidate가 실행 커밋의 설정과 같음을 이미 확인했고,
    # 저장소 현재 configs/ 파일은 실행 뒤에 바뀌었을 수 있다.
    configs = {check.config: yaml.safe_load(check.artifacts.config_bytes) for check in checks}
    config_source = {name: "실행 산출물" for name in configs}
    for edge in LINEAGE_EDGES:
        if edge.predecessor not in configs:
            configs[edge.predecessor] = yaml.safe_load(
                Path(f"configs/{edge.predecessor}.yaml").read_text()
            )
            config_source[edge.predecessor] = "저장소 현재 파일"

    relations = config_subset_relations(configs, members)
    frozen_members = []
    families: dict[str, list[str]] = {}
    perspectives_index: dict[str, list[str]] = {}
    for check in checks:
        name = check.config
        config = configs[name]
        family = (config.get("model") or {}).get("kind")
        if not family:
            raise FreezeError(f"{name} 설정에 model.kind가 없다.")
        families.setdefault(family, []).append(name)
        perspectives = member_perspectives(config)
        for perspective in perspectives:
            perspectives_index.setdefault(perspective, []).append(name)

        plan_entry = plan_by_run.get(check.run_id)
        if plan_entry is None:
            raise FreezeError(f"{name}이 전체 자료 재학습 계획에 없다.")

        edge = edge_by_successor.get(name)
        lineage: dict[str, object] = {"group": assignment[name]}
        if edge is None:
            lineage["role"] = "묶음 시작"
            lineage["predecessor"] = None
        else:
            lineage["role"] = edge.relation
            lineage["predecessor"] = edge.predecessor
            lineage["predecessor_in_pool"] = edge.predecessor in members
            lineage["evidence"] = edge.evidence
            lineage["predecessor_config_source"] = config_source[edge.predecessor]
            lineage["config_delta"] = config_delta(configs[edge.predecessor], config)

        frozen_members.append(
            {
                "config": name,
                "run_id": check.run_id,
                "oof_auc": float(check.auc) if check.auc is not None else None,
                "ledger_oof_auc": check.artifacts.member.oof_auc,
                "seeds": list(check.artifacts.member.seeds),
                "entered_at": str(check.artifacts.member.entered_at),
                "config_path": plan_entry["config_path"],
                "config_sha256": hashlib.sha256(check.artifacts.config_bytes).hexdigest(),
                "run_git_commit": check.artifacts.tags.get("git_commit"),
                "model_family": family,
                "lineage": lineage,
                "config_relations": relations[name],
                "preprocessing_reference_scope": preprocessing_reference_scope(config),
                "information_perspectives": perspectives,
                "full_refit_count": len(plan_entry["budgets"]),
                "integrity": {
                    "verdict": "통과" if check.valid else "실패",
                    "failures": list(check.failures),
                    "seed_mean_status": check.seed_mean_status,
                    "oof_missing": check.oof_missing,
                    "test_missing": check.test_missing,
                    "oof_sha256": check.oof_hash,
                    "test_sha256": check.test_hash,
                },
            }
        )

    manifest = manifest_hashes()
    baseline = {
        "schema_version": SCHEMA_VERSION,
        "frozen_at": FROZEN_AT,
        "ticket_issue": TICKET_ISSUE,
        "map_issue": MAP_ISSUE,
        "purpose": (
            "후보 풀 소급 재심사의 읽기 전용 기준 입력. "
            "이 파일은 제거 대조 결과를 보기 전에 고정하며 재심사 중에는 바꾸지 않는다."
        ),
        "source": {
            "git_commit": git_commit(),
            "pool_path": str(POOL_PATH),
            "pool_sha256": file_sha256(POOL_PATH),
            "full_refit_plan_path": str(REFIT_PLAN_PATH),
            "full_refit_plan_sha256": file_sha256(REFIT_PLAN_PATH),
        },
        "inputs": {
            "train_sha256": context.input_hashes["train"],
            "test_sha256": context.input_hashes["test"],
            "folds_sha256": context.input_hashes["folds"],
            "external_original_sha256": manifest[
                "data/external/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
            ],
        },
        "integrity_summary": {
            "members_total": len(checks),
            "members_passed": sum(1 for check in checks if check.valid),
            "members_failed": sorted(check.config for check in checks if not check.valid),
            "seed_mean_fully_verified": sum(
                1 for check in checks if check.seed_mean_status == "완전 확인"
            ),
            "duplicates_removed": [
                {
                    "dropped": decision.dropped,
                    "kept": decision.kept,
                    "reason": decision.reason,
                    "spearman": decision.spearman,
                }
                for decision in duplicates
            ],
            "retained_after_duplicate_check": len(retained),
        },
        "model_families": {
            family: {"size": len(names), "members": names}
            for family, names in sorted(families.items())
        },
        "lineage_groups": groups,
        "information_perspectives": {
            perspective: {"size": len(names), "members": names}
            for perspective, names in sorted(perspectives_index.items())
        },
        "full_refit_total": sum(entry["full_refit_count"] for entry in frozen_members),
        "members": frozen_members,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(baseline, allow_unicode=True, sort_keys=False, width=100) + ""
    )
    return baseline


def render_report(baseline: dict, output: Path, ledger_sha256: str) -> str:
    """동결한 기준 장부를 사람이 읽는 표로 옮긴다."""
    summary = baseline["integrity_summary"]
    lines = [
        "# 후보 풀 소급 재심사 기준 장부",
        "",
        "## 결론",
        "",
        f"현재 후보 풀 {summary['members_total']}개를 모두 다시 검증했고 "
        f"{summary['members_passed']}개가 무결성 검사를 통과했다.",
        f"정확 중복과 순위 중복 제거 뒤 {summary['retained_after_duplicate_check']}개가 남는다.",
        f"모델 계열은 {len(baseline['model_families'])}개, "
        f"모델 계보 묶음은 {len(baseline['lineage_groups'])}개, "
        f"정보 관점은 {len(baseline['information_perspectives'])}개다.",
        f"현재 구성으로 전체 자료 재학습은 {baseline['full_refit_total']}회다.",
        "",
        f"기준 장부 파일은 `{output}`이고 SHA-256은 `{ledger_sha256}`다.",
        "재심사가 끝날 때까지 이 파일과 아래 표를 바꾸지 않는다.",
        "",
        "## 동결한 입력",
        "",
        "| 입력 | SHA-256 |",
        "| --- | --- |",
        f"| `artifacts/pool.yaml` | `{baseline['source']['pool_sha256']}` |",
        f"| `artifacts/full-refit-plan.yaml` | `{baseline['source']['full_refit_plan_sha256']}` |",
        f"| `data/train.csv` | `{baseline['inputs']['train_sha256']}` |",
        f"| `data/test.csv` | `{baseline['inputs']['test_sha256']}` |",
        f"| `artifacts/folds.parquet` | `{baseline['inputs']['folds_sha256']}` |",
        f"| 원본 프록시 CSV | `{baseline['inputs']['external_original_sha256']}` |",
        "",
        f"기준 커밋은 `{baseline['source']['git_commit']}`다.",
        "",
        "## 구성원 분류",
        "",
        "| 구성원 | 모델 계열 | 모델 계보 묶음 | 계보 역할 | 정보 관점 | 전처리 기준 범위 | 전체 자료 재학습 | 무결성 |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for member in baseline["members"]:
        lineage = member["lineage"]
        lines.append(
            "| `{config}` | `{family}` | `{group}` | {role} | {perspectives} | {scope} | {refit} | {verdict} |".format(
                config=member["config"],
                family=member["model_family"],
                group=lineage["group"],
                role=lineage["role"],
                perspectives=", ".join(member["information_perspectives"]),
                scope=member["preprocessing_reference_scope"],
                refit=member["full_refit_count"],
                verdict=member["integrity"]["verdict"],
            )
        )

    lines += ["", "## 모델 계보 묶음", "", "| 묶음 | 구성원 수 | 구성원 | 풀 밖 이전판 |", "| --- | ---: | --- | --- |"]
    for group_id, group in baseline["lineage_groups"].items():
        lines.append(
            "| `{gid}` | {size} | {members} | {outside} |".format(
                gid=group_id,
                size=group["size"],
                members=", ".join(f"`{name}`" for name in group["members"]),
                outside=", ".join(f"`{name}`" for name in group["predecessors_outside_pool"]) or "없음",
            )
        )

    lines += ["", "## 정보 관점", "", "| 정보 관점 | 구성원 수 | 구성원 |", "| --- | ---: | --- |"]
    for name, entry in baseline["information_perspectives"].items():
        lines.append(
            "| {name} | {size} | {members} |".format(
                name=name,
                size=entry["size"],
                members=", ".join(f"`{member}`" for member in entry["members"]),
            )
        )

    partial = [
        member["config"]
        for member in baseline["members"]
        if member["integrity"]["seed_mean_status"] != "완전 확인"
    ]
    legacy = [
        member["config"]
        for member in baseline["members"]
        if member["config"] in LEGACY_FEATURE_SCHEMA_MEMBERS
    ]
    identical = [
        member["config"]
        for member in baseline["members"]
        if (member["lineage"].get("config_delta") or {}).get("shape") == "차이 없음"
    ]
    lines += [
        "",
        "## 재심사에 넘기는 제약",
        "",
        f"시드별 OOF 산출물이 없어 시드 평균을 독립 재계산하지 못한 구성원이 {len(partial)}개다: "
        + ", ".join(f"`{name}`" for name in partial)
        + ".",
        "이 구성원들은 #98 이전 실행이라 시드 단위로 짝지은 대조를 만들 수 없다.",
        "성능 동등 대역을 시드 단위로 잴 계획이면 이 제약을 먼저 반영해야 한다.",
        "",
        "설정 스키마가 하나가 아니다.",
        ", ".join(f"`{name}`" for name in legacy)
        + "는 `features.providers` 이전의 `include`/`fold_fit`/`derived` 스키마를 쓴다.",
        "저장소 현재 `configs/` 파일이 실행 당시 설정과 다른 사례도 있어, 이 장부는 실행 산출물의 설정만 읽는다.",
        "",
        "설정 차이로는 보이지 않는 계보가 있다.",
        ", ".join(f"`{name}`" for name in identical)
        + "는 이전판과 설정 잎이 같고 차이가 코드 수정에만 있다.",
        "따라서 계보 판단을 설정 차이만으로 대신할 수 없다.",
        "",
        "## 분류 규칙",
        "",
        "모델 계열은 실행 산출물 설정의 `model.kind` 하나로 정한다.",
        "",
        "모델 계보 묶음은 후보 풀 장부의 진입 근거나 그 근거가 가리키는 이슈가 이전판을 이름으로 지목한 간선만 모아 만든 연결 성분이다.",
        "같은 모델 계열이라는 사실만으로는 간선을 만들지 않는다.",
        "이전판이 현재 풀 밖이어도 간선을 남기므로, 같은 풀 밖 이전판을 공유하는 구성원은 한 묶음이 된다.",
        "설정 잎 단위 포함 관계는 각 구성원의 `config_relations`에 사실로만 남기고 묶음을 만들지 않는다.",
        "생략된 기본값은 잎으로 나타나지 않아 포함 관계가 의미 동일성을 뜻하지 않기 때문이다.",
        "",
        "정보 관점은 컬럼 제공자 종류와, CONTEXT.md가 정보 관점으로 이름 붙인 학습기 설정 두 가지에서 유도한다.",
        "학습기 설정 가운데 정보 관점으로 세는 항목은 정확값 어휘(`lookup_cols`, `exact_cols`)와 전처리 기준 집합 값 좌표(`reference_qnormal_columns`)뿐이다.",
        "나머지 `model.params` 항목은 용량·최적화·실행 설정으로 보고 모델 계열 축이 진다.",
        "컬럼 제공자를 쓰지 않고 이 두 설정도 없는 구성원은 `원시 관측 열 전용`으로 적는다.",
        "`preprocessing_scope`는 정보 관점이 아니라 전처리 기준 집합의 범위 사실로 따로 적는다.",
        "",
        "## 재현",
        "",
        "```",
        "uv run python scripts/freeze_pool_baseline.py",
        "```",
        "",
        "이 명령은 모델을 다시 학습하지 않고 장부와 실행 산출물만 읽는다.",
        "",
    ]
    report = "\n".join(lines)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="후보 풀 재심사 기준 장부 동결 (#343)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    try:
        baseline = build(args.output)
    except (FreezeError, pool_audit.PoolAuditError, KeyError, OSError) as exc:
        sys.exit(str(exc))
    digest = file_sha256(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(baseline, args.output, digest))
    print(f"기준 장부 저장: {args.output}")
    print(f"기준 장부 보고서 저장: {args.report}")
    print(f"기준 장부 SHA-256: {digest}")
    print(
        json.dumps(
            {
                "members": baseline["integrity_summary"]["members_total"],
                "passed": baseline["integrity_summary"]["members_passed"],
                "families": len(baseline["model_families"]),
                "lineage_groups": len(baseline["lineage_groups"]),
                "perspectives": len(baseline["information_perspectives"]),
                "full_refit_total": baseline["full_refit_total"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
