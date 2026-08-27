"""공개 OOF 라이브러리를 검증해 확장 스택용 외부 구성원 장부를 만든다. (#442)

지도 #441의 두 번째 최종 제출(확장 스택)에 넣을 수 있는 외부 구성원을 라이선스,
분할 안전성, 계보 기준으로 걸러 장부로 남긴다. 검증 항목은 #386과 같다.

- 행 수 691,369(OOF)와 296,302(test), 유한값
- 우리 라벨로 재채점한 AUC와 저자 선언 AUC의 일치(오차 1e-5 이내)
- OOF+test 바이트 해시 중복 제거
- 분할 벡터가 있으면 artifacts/folds.parquet와 일치

2단계 산출물, 10분할 배열, 라이선스 unknown·other 출처, 재현 불가 판정 구성원은
제외하고 사유를 장부에 남긴다. 이 스크립트는 읽기 전용이다. 외부 예측을 후보 풀
장부나 champion 판정에 넣지 않고 MLflow 실행도 만들지 않는다.

사용법:
    uv run python scripts/build_external_member_ledger.py

산출:
    docs/research/external-member-ledger.json  (기계가 읽는 장부)
    표준 출력 요약 (문서 작성용)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from pipeline.data import ID, TARGET, TRAIN_PATH, labels
from pipeline.judgment import FOLDS_PATH

EXT94 = Path("data/external/ext94")
EXT442 = Path("data/external/ext442")
TEST_PATH = Path("data/test.csv")
OUT_PATH = Path("docs/research/external-member-ledger.json")

N_TRAIN = 691369
N_TEST = 296302
AUC_TOLERANCE = 1e-5

FOLD_SPEC = (
    "StratifiedKFold(n_splits=5, shuffle=True, random_state=42), train.csv 원본 행 순서"
)

# 계보 조사(#174)와 #386이 확정한 사실을 그대로 옮긴다.
NAJI = {"naji01", "naji02", "naji03", "naji04", "naji05"}
GOLEM_PARTIAL = {"a", "d", "e", "f"}
# #386의 재현 가능 85구성원 집합. szymon 74에서 naji 5개와 pub_ravi를 뺀 68, FM 5, beicicc 12이다.
# beicicc 12 가운데 라이선스 other 4개(catboost screen-relation baseline·screen_relations,
# structural raw12·structural)는 이 장부에 반입하지 않으므로 여기서 이름으로만 기록한다.
# pub_evg(10분할)와 xgb_screen_relations_baseline103(바이트 중복)은 장부 안에서 제외 표시된다.
EXT85_BEICICC = {
    "lookup_fixed24",
    "lookup_fixed24_seed1042",
    "realmlp_seed01_fixed4",
    "identity_digit_raw12",
    "identity_digit_enhanced103",
    "xgb_identity_digit_raw12",
    "xgb_screen_relations_baseline103",
    "xgb_screen_relations_treatment110",
}
EXT85_NOT_IN_LEDGER = [
    {
        "member": "baseline",
        "dataset": "beicicc/s6e8-fixed4000-catboost-screen-relation-artifacts",
        "reason": "라이선스 other",
    },
    {
        "member": "screen_relations",
        "dataset": "beicicc/s6e8-fixed4000-catboost-screen-relation-artifacts",
        "reason": "라이선스 other",
    },
    {
        "member": "raw12",
        "dataset": "beicicc/s6e8-fixed900-structural-lgbm-artifacts",
        "reason": "라이선스 other",
    },
    {
        "member": "structural",
        "dataset": "beicicc/s6e8-fixed900-structural-lgbm-artifacts",
        "reason": "라이선스 other",
    },
]
TE_LEAK_CAVEAT = (
    "원 노트북이 전체 자료 TE를 쓴 판을 그대로 실행(#174), OOF가 낙관적일 수 있음"
)


@dataclass(frozen=True)
class Source:
    """공급원 하나. 데이터셋 단위의 라이선스와 분할 근거."""

    key: str
    dataset: str
    license: str
    root: str
    retrieved: str
    fold_evidence: str
    note: str


@dataclass
class Candidate:
    """검증 전 구성원 하나. 배열은 검증 뒤 버린다."""

    name: str
    source: str
    oof: np.ndarray
    test: np.ndarray
    oof_path: str
    test_path: str
    declared_auc: float | None = None
    fold_path: Path | None = None
    fold_evidence: str | None = None
    upstream: str | None = None
    caveats: list[str] = field(default_factory=list)
    exclude: str | None = None


@dataclass
class Member:
    """장부에 남는 구성원 한 줄."""

    member_id: str
    name: str
    source: str
    dataset: str
    license: str
    oof_path: str
    test_path: str
    oof_dtype: str
    oof_rows: int
    test_rows: int
    finite: bool
    auc: float | None
    declared_auc: float | None
    auc_delta: float | None
    sha256: str | None
    fold_check: str
    fold_evidence: str
    upstream: str | None
    caveats: list[str]
    in_ext85: bool
    status: str
    exclusion_reason: str | None


SOURCES = [
    Source(
        key="szymon74",
        dataset="szymonkapiski/s6e8-oof-library-47-models",
        license="CC0-1.0",
        root=str(EXT94 / "s6e8-oof-library-47-models"),
        retrieved="2026-08-23 (#386), 2026-08-04 판(74구성원)",
        fold_evidence="published_code",
        note="`mine` 구성원은 src/ 훈련 코드 공개, pub_*는 원 노트북 재실행, naji*는 najiama 재게시",
    ),
    Source(
        key="szymon_weak50",
        dataset="szymonkapiski/s6e8-50-weakest-oof-models",
        license="CC0-1.0",
        root=str(EXT442 / "szymonkapiski_s6e8-50-weakest-oof-models"),
        retrieved="2026-08-27",
        fold_evidence="author_statement",
        note="레시피 비공개, README가 고정 5분할과 자체 학습을 명시",
    ),
    Source(
        key="adarsh22",
        dataset="adarsh1077/s6e8-adarsh-oof-library",
        license="CC0-1.0",
        root=str(EXT442 / "adarsh1077_s6e8-adarsh-oof-library"),
        retrieved="2026-08-27",
        fold_evidence="author_statement",
        note="README가 고정 5분할 명시, 특성 레시피는 tomasa2 노트북 서술",
    ),
    Source(
        key="bolt47",
        dataset="boltuzamaki/s6e8-oof-prediction-library",
        license="CC0-1.0",
        root=str(EXT442 / "boltuzamaki_s6e8-oof-prediction-library"),
        retrieved="2026-08-27",
        fold_evidence="author_statement",
        note="스택 노트북이 '5분할 또는 10분할' 프로토콜이라고 서술, train_labels.parquet로 행 순서 대조",
    ),
    Source(
        key="mohan_cat",
        dataset="mohankrishnathalla/s6e8-cat-mlp-oof",
        license="CC0-1.0",
        root=str(EXT442 / "mohankrishnathalla_s6e8-cat-mlp-oof"),
        retrieved="2026-08-27",
        fold_evidence="published_code",
        note="s6e8-catboost-tuner-oof-saver 노트북이 고정 5분할 사용, 검증 fold 조기 종료",
    ),
    Source(
        key="mohan_lgb",
        dataset="mohankrishnathalla/s6e8-lgb-dart-oof",
        license="CC0-1.0",
        root=str(EXT442 / "mohankrishnathalla_s6e8-lgb-dart-oof"),
        retrieved="2026-08-27",
        fold_evidence="sibling_code",
        note="생성 노트북 미확인, 같은 저자 xgb/cat 노트북과 동일 골격(fold별 체크포인트 5개)",
    ),
    Source(
        key="mohan_xgb",
        dataset="mohankrishnathalla/s6e8-xgb-oof",
        license="CC0-1.0",
        root=str(EXT442 / "mohankrishnathalla_s6e8-xgb-oof"),
        retrieved="2026-08-27",
        fold_evidence="published_code",
        note="s6e8-xgb-tuner-oof-saver 노트북이 고정 5분할 사용, 검증 fold 조기 종료",
    ),
    Source(
        key="hboyang6",
        dataset="hboyang/s6e8-catstrall-member",
        license="CC0-1.0",
        root=str(EXT442 / "hboyang_s6e8-catstrall-member"),
        retrieved="2026-08-27",
        fold_evidence="author_statement",
        note="README가 cat_strall은 고정 5분할, 나머지는 '정렬된 독립 학습 다양성 구성원'이라고 서술",
    ),
    Source(
        key="golem",
        dataset="dariushafshar/s6e8-golem-oof-library",
        license="CC0-1.0",
        root=str(EXT94 / "s6e8-golem-oof-library"),
        retrieved="2026-08-23 (#386)",
        fold_evidence="author_statement",
        note="훈련 코드 비공개, manifest에 fold별 AUC, a·f는 검증 fold 조기 종료 공표",
    ),
    Source(
        key="fm5",
        dataset="raykkretzschmar/s6e8-fm-lattice-blend-members",
        license="Apache-2.0",
        root=str(EXT94 / "s6e8-fm-lattice-blend-members"),
        retrieved="2026-08-23 (#386)",
        fold_evidence="published_code",
        note="train_fm.py 등 훈련 코드 동봉",
    ),
]

BEICICC_DATASETS = {
    # slug: (license, [(member, contract json, auc json path)])
    "s6e8-fixed-schedule-lookup-transformer-artifacts": (
        "CC-BY-4.0",
        [("lookup_fixed24", "lookup_fixed24_manifest.json", ("overall_oof_auc",))],
    ),
    "s6e8-second-seed-fixed-schedule-lookup-artifacts": (
        "CC-BY-4.0",
        [
            (
                "lookup_fixed24_seed1042",
                "lookup_fixed24_seed1042_contract.json",
                ("metrics", "pooled_oof_auc"),
            )
        ],
    ),
    "s6e8-fixed-schedule-exact-value-catboost-artifacts": (
        "CC-BY-4.0",
        [
            (
                "exact_value_catboost_fixed4000",
                "exact_value_catboost_contract.json",
                ("metrics", "pooled_oof_auc"),
            )
        ],
    ),
    "s6e8-fixed4-realmlp-two-seed-artifacts": (
        "CC-BY-4.0",
        [
            (
                "realmlp_seed01_fixed4",
                "realmlp_seed01_fixed4_contract.json",
                ("metrics", "two_seed_average_oof_auc"),
            )
        ],
    ),
    "s6e8-fixed900-identity-digit-lightgbm-artifacts": (
        "CC-BY-4.0",
        [
            (
                "identity_digit_raw12",
                "identity_digit_contract.json",
                ("metrics", "raw12_oof_auc"),
            ),
            (
                "identity_digit_enhanced103",
                "identity_digit_contract.json",
                ("metrics", "enhanced103_oof_auc"),
            ),
        ],
    ),
    "s6e8-fixed1500-xgb-identity-digit-artifacts": (
        "CC0-1.0",
        [
            (
                "xgb_identity_digit_raw12",
                "xgb_identity_digit_contract.json",
                ("metrics", "raw12_oof_auc"),
            ),
            (
                "xgb_identity_digit_enhanced103",
                "xgb_identity_digit_contract.json",
                ("metrics", "enhanced103_oof_auc"),
            ),
        ],
    ),
    "s6e8-fixed1500-xgb-screen-relation-artifacts": (
        "CC0-1.0",
        [
            (
                "xgb_screen_relations_baseline103",
                "xgb_screen_relations_contract.json",
                ("metrics", "baseline103_oof_auc"),
            ),
            (
                "xgb_screen_relations_treatment110",
                "xgb_screen_relations_contract.json",
                ("metrics", "treatment110_oof_auc"),
            ),
        ],
    ),
}

# 반입하지 않는 출처. 장부에 사유만 남긴다.
SOURCES_NOT_IMPORTED = [
    {
        "dataset": "najiama/predicting-smartphone-addiction-oof-submission-csv",
        "license": "unknown",
        "reason": "라이선스 불명, 생성 코드 없음(#174)",
    },
    {
        "dataset": "hboyang/s6e8-150-fusion-local-members",
        "license": "unknown",
        "reason": "라이선스 불명, candidate_naji16_* 2단계 산출물 포함",
    },
    {
        "dataset": "beicicc/s6e8-fixed4000-catboost-screen-relation-artifacts",
        "license": "other",
        "reason": "라이선스 other, 저자 명시 허가 문구 없음",
    },
    {
        "dataset": "beicicc/s6e8-fixed900-structural-lgbm-artifacts",
        "license": "other",
        "reason": "라이선스 other, 저자 명시 허가 문구 없음",
    },
    {
        "dataset": "beicicc/s6e8-sixmember-crossfit-logitlr-artifacts",
        "license": "other",
        "reason": "2단계 산출물(sixmember_*)이며 라이선스 other",
    },
    {
        "dataset": "dariushafshar/s6e8-measured-findings-pack",
        "license": "CC0-1.0",
        "reason": "구성원 없음, folds_seed42.npy를 분할 대조에만 사용",
    },
]


def _sha(oof: np.ndarray, test: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(oof, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(test, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _json_path(data: dict, path: tuple[str, ...]) -> float:
    node = data
    for key in path:
        node = node[key]
    return float(node)


def _npy_pair(
    root: Path, name: str, oof_file: str, test_file: str
) -> tuple[np.ndarray, np.ndarray]:
    return np.load(root / oof_file), np.load(root / test_file)


def load_szymon74(source: Source) -> Iterator[Candidate]:
    root = Path(source.root)
    manifest = pd.read_csv(root / "manifest.csv").set_index("model")
    for path in sorted((root / "oof").glob("oof_*.npy")):
        name = path.name[4:-4]
        test_path = path.with_name(f"test_{name}.npy")
        row = manifest.loc[name] if name in manifest.index else None
        origin = None if row is None else str(row["source"])
        candidate = Candidate(
            name=name,
            source=source.key,
            oof=np.load(path),
            test=np.load(test_path),
            oof_path=str(path),
            test_path=str(test_path),
            declared_auc=None if row is None else float(row["oof_auc"]),
            fold_evidence="published_code" if origin == "mine" else "author_statement",
            upstream=None if origin in (None, "mine") else origin,
        )
        if name in NAJI:
            candidate.exclude = "원출처 najiama 데이터셋의 라이선스 불명(재게시분)"
        elif name == "pub_ravi":
            candidate.exclude = "2단계 산출물(ravi20076 L2 스택)"
        elif name == "pub_evg":
            candidate.exclude = "10분할 배열(evgendvorkin 단일 LGBM)"
        if name in {"pub_rmlp", "pub_tabm"}:
            candidate.caveats.append(TE_LEAK_CAVEAT)
        if name == "lookup":
            candidate.upstream = "architecture by tamerlanomralinov, retrained on 5 folds by szymonkapiski"
        yield candidate


def load_weak50(source: Source) -> Iterator[Candidate]:
    root = Path(source.root)
    members = pd.read_csv(root / "members.csv")
    oof = np.load(root / "oof.npy")
    test = np.load(root / "test.npy")
    assert oof.shape == (N_TRAIN, 50) and test.shape == (N_TEST, 50), (
        oof.shape,
        test.shape,
    )
    for column, row in enumerate(members.itertuples()):
        yield Candidate(
            name=str(row.id),
            source=source.key,
            oof=oof[:, column],
            test=test[:, column],
            oof_path=f"{root / 'oof.npy'}[:, {column}]",
            test_path=f"{root / 'test.npy'}[:, {column}]",
            declared_auc=float(row.solo_oof_auc),
            caveats=["레시피·모델 종류 비공개"],
        )


def _adarsh_declared(readme: str) -> dict[str, float]:
    pattern = re.compile(r"`oof_([a-z0-9_]+)\.npy`\s*\|\s*([0-9.]+)\s*\|")
    return {name: float(auc) for name, auc in pattern.findall(readme)}


def load_adarsh(source: Source) -> Iterator[Candidate]:
    root = Path(source.root)
    declared = _adarsh_declared((root / "README.md").read_text())
    for path in sorted(root.glob("oof_*.npy")):
        name = path.name[4:-4]
        test_path = path.with_name(f"test_{name}.npy")
        candidate = Candidate(
            name=name,
            source=source.key,
            oof=np.load(path),
            test=np.load(test_path),
            oof_path=str(path),
            test_path=str(test_path),
            declared_auc=declared.get(name),
        )
        if name in {"lgbte", "lgbs7"}:
            candidate.caveats.append(
                "README가 '10-fold nested TE'를 서술하나 이는 학습부 안 내부 인코딩이고 외부 분할은 5분할"
            )
        yield candidate


def load_bolt(source: Source) -> Iterator[Candidate]:
    root = Path(source.root)
    index = pd.read_csv(root / "stream_index.csv").set_index("stream")
    oof = pd.read_parquet(root / "oof_predictions.parquet")
    test = pd.read_parquet(root / "test_predictions.parquet")
    streams = [c for c in oof.columns if c != ID]
    assert set(streams) == set(index.index), "stream_index와 parquet 열이 다르다"
    for name in index.index:
        candidate = Candidate(
            name=name,
            source=source.key,
            oof=oof[name].to_numpy(),
            test=test[name].to_numpy(),
            oof_path=f"{root / 'oof_predictions.parquet'}[{name}]",
            test_path=f"{root / 'test_predictions.parquet'}[{name}]",
            declared_auc=float(index.loc[name, "oof_auc"]),
        )
        if name.endswith("_10f"):
            candidate.exclude = "10분할 배열(이름과 저자 서술: 5분할→10분할 이동 실험)"
        elif "4fold" in name:
            candidate.exclude = (
                "분할 수가 5가 아닌 것으로 이름이 명시(4fold), 보수적으로 제외"
            )
        yield candidate


def load_mohan(source: Source, tag: str) -> Iterator[Candidate]:
    root = Path(source.root)
    oof_path = root / f"oof_{tag}_v3.npy"
    test_path = root / f"test_{tag}_v3.npy"
    candidate = Candidate(
        name=f"{tag}_v3",
        source=source.key,
        oof=np.load(oof_path),
        test=np.load(test_path),
        oof_path=str(oof_path),
        test_path=str(test_path),
        caveats=["검증 fold 조기 종료(best_iteration)로 OOF가 약간 낙관적"],
    )
    if tag == "lgb":
        candidate.caveats.append(
            "생성 노트북 미확인, xgb 노트북 서술로는 순위 특성을 fold 밖에서 계산"
        )
    yield candidate


def load_hboyang(source: Source) -> Iterator[Candidate]:
    root = Path(source.root)
    for path in sorted(root.glob("oof_*.npy")):
        name = path.name[4:-4]
        test_path = path.with_name(f"test_{name}.npy")
        oof = np.load(path)
        candidate = Candidate(
            name=name,
            source=source.key,
            oof=oof,
            test=np.load(test_path),
            oof_path=str(path),
            test_path=str(test_path),
            fold_evidence="author_statement",
        )
        if name in {"kirill_o1", "koda_exact_te"}:
            candidate.caveats.append(
                "이름이 다른 공개 노트북 레시피를 가리키나 README는 독립 학습이라고 서술"
            )
        yield candidate


def load_golem(source: Source) -> Iterator[Candidate]:
    root = Path(source.root)
    manifest = pd.read_csv(root / "manifest.csv").set_index("member")
    for path in sorted(root.glob("oof_*.npy")):
        name = path.name[4:-4]
        test_path = path.with_name(f"test_{name}.npy")
        candidate = Candidate(
            name=name,
            source=source.key,
            oof=np.load(path),
            test=np.load(test_path),
            oof_path=str(path),
            test_path=str(test_path),
            declared_auc=float(manifest.loc[name, "oof_auc"]),
        )
        if name in GOLEM_PARTIAL:
            detail = (
                "검증 fold 조기 종료 낙관 공표"
                if name in {"a", "f"}
                else "하이퍼파라미터 부재"
            )
            candidate.exclude = f"부분 재현 판정(#174): {detail}"
        yield candidate


def _fm_declared(readme: str) -> dict[str, float]:
    pattern = re.compile(r"^\|\s*(fm[a-z]+)\s*\|\s*([0-9.]+)\s*\|", re.MULTILINE)
    return {name: float(auc) for name, auc in pattern.findall(readme)}


def load_fm(source: Source) -> Iterator[Candidate]:
    root = Path(source.root)
    declared = _fm_declared((root / "README.md").read_text())
    for path in sorted(root.glob("oof_*.npy")):
        name = path.name[4:-4]
        test_path = path.with_name(f"test_{name}.npy")
        yield Candidate(
            name=name,
            source=source.key,
            oof=np.load(path),
            test=np.load(test_path),
            oof_path=str(path),
            test_path=str(test_path),
            declared_auc=declared.get(name),
        )
    for path in sorted(root.glob("bandoof_*.npy")):
        name = path.name[8:-4]
        oof = np.load(path)
        yield Candidate(
            name=f"band_{name}",
            source=source.key,
            oof=oof,
            test=np.load(path.with_name(f"bandtest_{name}.npy")),
            oof_path=str(path),
            test_path=str(path.with_name(f"bandtest_{name}.npy")),
            exclude="구간 한정 구성원(전체 길이 아님, 저자가 혼합 구성원 아님을 명시)",
        )


def beicicc_sources() -> list[Source]:
    return [
        Source(
            key=f"beicicc:{slug}",
            dataset=f"beicicc/{slug}",
            license=license,
            root=str(EXT94 / "beicicc" / slug),
            retrieved="2026-08-23 (#386)",
            fold_evidence="fold_vector",
            note="훈련 코드 비공개, contract JSON에 하이퍼파라미터·고정 스케줄·fold 해시 기록, fold_id.npy 동봉(1-based)",
        )
        for slug, (license, _) in BEICICC_DATASETS.items()
    ]


def load_beicicc(source: Source) -> Iterator[Candidate]:
    slug = source.key.split(":", 1)[1]
    root = Path(source.root)
    _, members = BEICICC_DATASETS[slug]
    for name, contract_file, auc_path in members:
        contract = json.loads((root / contract_file).read_text())
        fold_candidates = sorted(root.glob("*fold_id.npy"))
        specific = root / f"{name}_fold_id.npy"
        yield Candidate(
            name=name,
            source=source.key,
            oof=np.load(root / f"{name}_oof.npy"),
            test=np.load(root / f"{name}_test.npy"),
            oof_path=str(root / f"{name}_oof.npy"),
            test_path=str(root / f"{name}_test.npy"),
            declared_auc=_json_path(contract, auc_path),
            fold_path=specific if specific.exists() else fold_candidates[0],
        )


def iter_candidates(sources: list[Source]) -> Iterator[tuple[Source, Candidate]]:
    loaders = {
        "szymon74": load_szymon74,
        "szymon_weak50": load_weak50,
        "adarsh22": load_adarsh,
        "bolt47": load_bolt,
        "mohan_cat": lambda s: load_mohan(s, "cat"),
        "mohan_lgb": lambda s: load_mohan(s, "lgb"),
        "mohan_xgb": lambda s: load_mohan(s, "xgb"),
        "hboyang6": load_hboyang,
        "golem": load_golem,
        "fm5": load_fm,
    }
    for source in sources:
        loader = (
            load_beicicc if source.key.startswith("beicicc:") else loaders[source.key]
        )
        for candidate in loader(source):
            yield source, candidate


def _fold_check(candidate: Candidate, fold_of: np.ndarray) -> str:
    if candidate.fold_path is None:
        return "위치 정렬(분할 벡터 없음)"
    external = np.load(candidate.fold_path)
    if len(external) != len(fold_of):
        return f"행 수 불일치 {len(external)}"
    if np.array_equal(external, fold_of):
        return "일치(0-based)"
    if np.array_equal(external - 1, fold_of):
        return "일치(1-based 보정)"
    return "불일치"


def verify(
    candidate: Candidate, source: Source, y: np.ndarray, fold_of: np.ndarray
) -> Member:
    # hboyang의 일부 배열은 (n, 1) 모양으로 저장돼 있다. 열 하나짜리 2차원은 평탄화한다.
    oof, test = candidate.oof, candidate.test
    if oof.ndim == 2 and oof.shape[1] == 1:
        oof = oof.reshape(-1)
    if test.ndim == 2 and test.shape[1] == 1:
        test = test.reshape(-1)
    rows_ok = oof.shape == (N_TRAIN,) and test.shape == (N_TEST,)
    caveats = list(candidate.caveats)
    if oof.dtype != np.float64 or test.dtype != np.float64:
        caveats.append(
            f"{oof.dtype} 저장(스택 정밀도 손실 가능, szymonkapiski 측정 공개 점수 -0.00001)"
        )
    finite = bool(rows_ok and np.isfinite(oof).all() and np.isfinite(test).all())
    auc = (
        float(roc_auc_score(y, oof.astype(np.float64))) if rows_ok and finite else None
    )
    delta = (
        None
        if auc is None or candidate.declared_auc is None
        else auc - candidate.declared_auc
    )
    fold_check = _fold_check(candidate, fold_of) if rows_ok else "행 수 불일치"
    digest = _sha(oof, test) if rows_ok else None

    reason = candidate.exclude
    if reason is None and not rows_ok:
        reason = f"행 수 불일치 {oof.shape}/{test.shape}"
    elif reason is None and not finite:
        reason = "비유한값 포함"
    elif reason is None and delta is not None and abs(delta) > AUC_TOLERANCE:
        reason = f"선언 AUC와 불일치 {delta:+.2e}"
    elif reason is None and fold_check.startswith("불일치"):
        reason = "분할 벡터가 artifacts/folds.parquet와 다름"

    in_ext85 = (
        (
            candidate.source == "szymon74"
            and candidate.name not in NAJI
            and candidate.name != "pub_ravi"
        )
        or (candidate.source == "fm5" and candidate.name.startswith("fm"))
        or (candidate.source.startswith("beicicc:") and candidate.name in EXT85_BEICICC)
    )
    return Member(
        member_id=f"{candidate.source}:{candidate.name}",
        name=candidate.name,
        source=candidate.source,
        dataset=source.dataset,
        license=source.license,
        oof_path=candidate.oof_path,
        test_path=candidate.test_path,
        oof_dtype=str(oof.dtype),
        oof_rows=int(oof.shape[0]),
        test_rows=int(test.shape[0]),
        finite=finite,
        auc=auc,
        declared_auc=candidate.declared_auc,
        auc_delta=delta,
        sha256=digest,
        fold_check=fold_check,
        fold_evidence=candidate.fold_evidence or source.fold_evidence,
        upstream=candidate.upstream,
        caveats=caveats,
        in_ext85=in_ext85,
        status="excluded" if reason else "accepted",
        exclusion_reason=reason,
    )


def row_order_checks(fold_of: pd.Series, y: np.ndarray) -> dict[str, object]:
    """외부 라이브러리가 전제하는 위치 정렬이 우리 기준 순서와 같은지 확인한다."""
    train_ids = pd.read_csv(TRAIN_PATH, usecols=[ID])[ID].to_numpy()
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID])[ID].to_numpy()
    checks: dict[str, object] = {
        "train_rows": len(train_ids),
        "test_rows": len(test_ids),
        "folds_id_order_equals_train_csv": bool(
            np.array_equal(fold_of.index.to_numpy(), train_ids)
        ),
    }
    szymon = EXT94 / "s6e8-oof-library-47-models"
    checks["szymon_train_keys_equals_train_csv"] = bool(
        np.array_equal(
            pd.read_parquet(szymon / "train_keys.parquet")[ID].to_numpy(), train_ids
        )
    )
    checks["szymon_train_keys_label_equals_ours"] = bool(
        np.array_equal(
            pd.read_parquet(szymon / "train_keys.parquet")[TARGET].to_numpy(), y
        )
    )
    checks["szymon_test_keys_equals_test_csv"] = bool(
        np.array_equal(
            pd.read_parquet(szymon / "test_keys.parquet")[ID].to_numpy(), test_ids
        )
    )
    bolt = EXT442 / "boltuzamaki_s6e8-oof-prediction-library"
    bolt_labels = pd.read_parquet(bolt / "train_labels.parquet")
    label_column = TARGET if TARGET in bolt_labels.columns else bolt_labels.columns[-1]
    checks["bolt_train_labels_equals_ours"] = bool(
        len(bolt_labels) == N_TRAIN
        and np.array_equal(bolt_labels[label_column].to_numpy(), y)
    )
    bolt_columns = pq.read_schema(bolt / "oof_predictions.parquet").names
    checks["bolt_oof_parquet_has_id"] = ID in bolt_columns
    if ID in bolt_columns:
        bolt_oof = pd.read_parquet(bolt / "oof_predictions.parquet", columns=[ID])
        checks["bolt_oof_id_equals_train_csv"] = bool(
            np.array_equal(bolt_oof[ID].to_numpy(), train_ids)
        )
        bolt_test = pd.read_parquet(bolt / "test_predictions.parquet", columns=[ID])
        checks["bolt_test_id_equals_test_csv"] = bool(
            np.array_equal(bolt_test[ID].to_numpy(), test_ids)
        )
    pack = EXT442 / "dariushafshar_s6e8-measured-findings-pack" / "folds_seed42.npy"
    checks["findings_pack_folds_equals_ours"] = bool(
        np.array_equal(np.load(pack).astype(np.int64), fold_of.to_numpy())
    )
    return checks


def main() -> None:
    fold_of = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"]
    y = labels(fold_of.index).to_numpy()
    fold_array = fold_of.to_numpy()
    checks = row_order_checks(fold_of, y)
    print("행 순서 대조:")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    assert checks["folds_id_order_equals_train_csv"], (
        "folds.parquet id 순서가 train.csv와 다르다"
    )
    assert checks["findings_pack_folds_equals_ours"], (
        "findings pack 분할이 우리 분할과 다르다"
    )

    sources = SOURCES + beicicc_sources()
    members: list[Member] = []
    seen: dict[str, str] = {}
    for source, candidate in iter_candidates(sources):
        member = verify(candidate, source, y, fold_array)
        if member.status == "accepted" and member.sha256 in seen:
            member.status = "excluded"
            member.exclusion_reason = f"바이트 중복: {seen[member.sha256]}"
        elif member.status == "accepted":
            seen[member.sha256] = member.member_id
        members.append(member)
        flag = (
            "ok" if member.status == "accepted" else f"제외({member.exclusion_reason})"
        )
        auc_text = "-" if member.auc is None else f"{member.auc:.7f}"
        print(f"  {member.member_id:<60s} {auc_text} {member.fold_check} {flag}")
        del candidate.oof, candidate.test

    accepted = [m for m in members if m.status == "accepted"]
    frame = pd.DataFrame([asdict(m) for m in members])
    rescored = frame.dropna(subset=["auc_delta"])
    summary = {
        "candidate_count": len(members),
        "accepted_count": len(accepted),
        "excluded_count": int(len(members) - len(accepted)),
        "accepted_by_source": {
            k: int(v)
            for k, v in frame[frame.status == "accepted"]
            .groupby("source")
            .size()
            .items()
        },
        "accepted_by_license": {
            k: int(v)
            for k, v in frame[frame.status == "accepted"]
            .groupby("license")
            .size()
            .items()
        },
        "max_abs_auc_delta_accepted": float(
            rescored[rescored.status == "accepted"]["auc_delta"].abs().max()
        ),
        "accepted_auc_min": float(min(m.auc for m in accepted)),
        "accepted_auc_max": float(max(m.auc for m in accepted)),
        "accepted_fold_checks": {
            str(k): int(v)
            for k, v in frame[frame.status == "accepted"]["fold_check"]
            .value_counts()
            .items()
        },
        "accepted_fold_evidence": {
            str(k): int(v)
            for k, v in frame[frame.status == "accepted"]["fold_evidence"]
            .value_counts()
            .items()
        },
        "accepted_with_caveats": int(sum(1 for m in accepted if m.caveats)),
        "accepted_float32": int(sum(1 for m in accepted if m.oof_dtype != "float64")),
        "ext85_in_ledger": int(sum(1 for m in members if m.in_ext85)),
        "ext85_accepted": int(sum(1 for m in accepted if m.in_ext85)),
        "ext85_not_in_ledger": len(EXT85_NOT_IN_LEDGER),
        "accepted_beyond_ext85": int(sum(1 for m in accepted if not m.in_ext85)),
    }
    assert summary["ext85_in_ledger"] + summary["ext85_not_in_ledger"] == 85, summary
    print("\n요약:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    ledger = {
        "issue": 442,
        "generated": "2026-08-27",
        "fold_spec": FOLD_SPEC,
        "auc_tolerance": AUC_TOLERANCE,
        "row_order_checks": checks,
        "sources": [asdict(s) for s in sources],
        "sources_not_imported": SOURCES_NOT_IMPORTED,
        "ext85_not_in_ledger": EXT85_NOT_IN_LEDGER,
        "summary": summary,
        "members": [asdict(m) for m in members],
    }
    OUT_PATH.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
    print(f"\n장부 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
