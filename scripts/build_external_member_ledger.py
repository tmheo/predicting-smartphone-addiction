"""공개 OOF 라이브러리를 검증해 확장 스택용 외부 구성원 장부를 만든다. (#442, #454)

지도 #441의 두 번째 최종 제출(확장 스택)에 넣을 수 있는 외부 구성원을 분할 안전성과
계보 기준으로 걸러 장부로 남긴다. 판본 1(#442)은 17개 데이터셋 226후보였고,
판본 2(#454, 지도 #451)는 #452가 찾은 새 데이터셋 4곳과 공개 노트북 출력물을
증분으로 더한다. 검증 항목은 #386과 같다.

- 행 수 691,369(OOF)와 296,302(test), 유한값
- 우리 라벨로 재채점한 AUC와 저자 선언 AUC의 일치(오차 1e-5 이내)
- OOF+test 바이트 해시 중복 제거(먼저 적재된 쪽을 남김)
- 분할 벡터가 있으면 artifacts/folds.parquet와 일치
- 선언 AUC가 없는 배열은 재채점 AUC가 0.8 이상이어야 정렬된 것으로 본다
- 통과 구성원끼리 OOF 스피어만 순위 상관이 0.998을 넘으면 군집으로 묶어
  대표가 아닌 #454 구성원에 `near_duplicate_cluster` 주의 사항을 단다

2단계 산출물, 10분할 배열, 재현 불가 판정 구성원은 제외하고 사유를 장부에 남긴다.
라이선스는 판본 2부터 검증 항목이 아니라 기록 항목이다(지도 #451 2026-08-27 사용자 결정).
unknown·other 데이터셋과 라이선스 표시가 없는 노트북 출력물도 결합 입력으로 반입하되
`license_unknown` 주의 사항을 달고, 재배포·저장소 커밋·자체 산출물 첨부는 하지 않는다.
분할 근거가 없는 구성원은 제외하지 않고 `fold_evidence_none` 주의 사항으로 표시해
판정 티켓이 절제로 기여를 따로 재게 한다.
판본 1의 통과 209개는 status와 caveats를 바꾸지 않는다.

이 스크립트는 읽기 전용이다. 외부 예측을 후보 풀 장부나 champion 판정에 넣지 않고
MLflow 실행도 만들지 않는다.

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
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from pipeline.data import ID, TARGET, TRAIN_PATH, labels
from pipeline.judgment import FOLDS_PATH

EXT94 = Path("data/external/ext94")
EXT442 = Path("data/external/ext442")
EXT454 = Path("data/external/ext454")
NOTEBOOKS454 = EXT454 / "notebooks"
TEST_PATH = Path("data/test.csv")
OUT_PATH = Path("docs/research/external-member-ledger.json")

LEDGER_VERSION = 2
LEDGER_ISSUE = 454
GENERATED = "2026-08-27"
HISTORY = [
    {
        "version": 1,
        "issue": 442,
        "generated": "2026-08-27",
        "candidate_count": 226,
        "accepted_count": 209,
        "note": "데이터셋 17개, 라이선스 CC0·CC BY 4.0·Apache 2.0·MIT만 반입",
    }
]
LICENSE_POLICY = (
    "라이선스 unknown·other 데이터셋과 라이선스 표시가 없는 노트북 출력물의 예측 배열도 "
    "결합 입력으로 사용할 수 있으나 재배포·저장소 커밋·자체 산출물 첨부는 하지 않는다. "
    "근거는 대회 규칙 Section 6(외부 자료는 공개·무료·동등 접근이면 됨)과 우승자 라이선스 None, "
    "지도 #451의 2026-08-27 사용자 결정이다. 라이선스는 검증 항목이 아니라 기록 항목이며 "
    "CC BY 4.0·Apache 2.0 구성원은 저작자 표기를 유지한다."
)

N_TRAIN = 691369
N_TEST = 296302
AUC_TOLERANCE = 1e-5
# 선언 AUC가 없는 배열의 정렬 판정 바닥. 정렬이 어긋난 배열은 0.5 근처가 나온다(#452 vladstud 0.49997).
AUC_FLOOR = 0.8
# 통과 구성원끼리의 OOF 스피어만 순위 상관 중복 기준(#452와 같다).
NEAR_DUPLICATE_SPEARMAN = 0.998

FOLD_SPEC = (
    "StratifiedKFold(n_splits=5, shuffle=True, random_state=42), train.csv 원본 행 순서"
)

# 판정 티켓이 절제 부류로 고르는 주의 사항. 앞의 낱말이 부류 이름이다.
LICENSE_CAVEATS = {
    "unknown": "license_unknown: 라이선스 unknown(사용 한정, 재배포·커밋·자체 산출물 첨부 금지)",
    "other": "license_unknown: 라이선스 other(사용 한정, 재배포·커밋·자체 산출물 첨부 금지)",
    "notebook_output": (
        "license_unknown: 노트북 출력물(소스는 Apache-2.0, 출력물은 라이선스 표시 없음, 사용 한정)"
    ),
}
NAJI_LICENSE_CAVEAT = (
    "license_unknown: 원출처 najiama 데이터셋 라이선스 불명(szymonkapiski CC0 재게시, 사용 한정)"
)
FOLD_NONE_CAVEAT = "fold_evidence_none: 분할 근거 없음(설명·코드·저자 서술 부재), 절제로 기여를 따로 잰다"

# 계보 조사(#174)와 #386이 확정한 사실을 그대로 옮긴다.
NAJI = {"naji01", "naji02", "naji03", "naji04", "naji05"}
GOLEM_PARTIAL = {"a", "d", "e", "f"}
# #386의 재현 가능 85구성원 집합. szymon 74에서 naji 5개와 pub_ravi를 뺀 68, FM 5, beicicc 12이다.
# beicicc 12 가운데 라이선스 other 4개(catboost screen-relation baseline·screen_relations,
# structural raw12·structural)는 판본 1에서 반입하지 않았고 판본 2(#454)가 사용 한정으로 반입했다.
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
    "baseline",
    "screen_relations",
    "raw12",
    "structural",
}
# 판본 2부터 85구성원 전부가 장부 안에 있다. 판본 1과 같은 키를 유지하려고 빈 목록을 남긴다.
EXT85_NOT_IN_LEDGER: list[dict[str, str]] = []
TE_LEAK_CAVEAT = (
    "원 노트북이 전체 자료 TE를 쓴 판을 그대로 실행(#174), OOF가 낙관적일 수 있음"
)


@dataclass(frozen=True)
class Source:
    """공급원 하나. 데이터셋 또는 노트북 단위의 라이선스와 분할 근거."""

    key: str
    dataset: str
    license: str
    root: str
    retrieved: str
    fold_evidence: str
    note: str
    kind: str = "dataset"  # dataset | notebook_output
    added_in: int = 442  # 이 공급원을 반입한 이슈 번호


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
    license: str | None = None  # 공급원 라이선스를 구성원 단위로 덮어쓸 때(naji 재게시분)
    reference_auc: float | None = None  # #452 조사가 독립적으로 잰 재채점 AUC
    added_in: int | None = None  # 공급원과 다른 이슈에서 통과시킨 구성원(naji)


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
    added_in: int = 442
    reference_auc: float | None = None
    reference_auc_delta: float | None = None
    spearman_max: float | None = None  # 다른 통과 구성원과의 OOF 스피어만 최댓값
    spearman_closest: str | None = None


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
# 판본 2(#454)가 사용 한정으로 반입한 beicicc의 라이선스 other 데이터셋. #386 기준선 재현에 쓰던 로컬 파일이다.
BEICICC_DATASETS_454 = {
    "s6e8-fixed4000-catboost-screen-relation-artifacts": (
        "other",
        [
            ("baseline", "manifest.json", ("results", "baseline", "overall_oof_auc")),
            (
                "screen_relations",
                "manifest.json",
                ("results", "screen_relations", "overall_oof_auc"),
            ),
        ],
    ),
    "s6e8-fixed900-structural-lgbm-artifacts": (
        "other",
        [
            ("raw12", "manifest.json", ("results", "raw12", "overall_oof_auc")),
            ("structural", "manifest.json", ("results", "structural", "overall_oof_auc")),
        ],
    ),
}

# 반입하지 않는 데이터셋. 장부에 사유만 남긴다. 판본 2에서 #452 조사 결과를 더했다.
SOURCES_NOT_IMPORTED = [
    {
        "dataset": "najiama/predicting-smartphone-addiction-oof-submission-csv",
        "license": "unknown",
        "reason": (
            "단일 5개는 szymon74:naji01~05로 같은 배열을 이미 보유(#454에서 사용 한정으로 통과), "
            "2단계 11개는 제외, 데이터셋 자체는 받지 않음"
        ),
    },
    {
        "dataset": "beicicc/s6e8-sixmember-crossfit-logitlr-artifacts",
        "license": "other",
        "reason": "2단계 산출물(sixmember_*)",
    },
    {
        "dataset": "dariushafshar/s6e8-measured-findings-pack",
        "license": "CC0-1.0",
        "reason": "구성원 없음, folds_seed42.npy를 분할 대조에만 사용",
    },
    {
        "dataset": "szymonkapiski/s6e8-oof-library-25-models",
        "license": "CC0-1.0",
        "reason": "25쌍 전부 장부 szymon74의 같은 이름 구성원을 float32로 바꾼 값과 바이트 동일, pub_ravi는 2단계(#452)",
    },
    {
        "dataset": "tamerlanomralinov/s6e8-full-best-blend-npy",
        "license": "CC0-1.0",
        "reason": "10분할 배열 9개(blend_config.json folds=10, 노트북 코드 11분할)",
    },
    {
        "dataset": "raykkretzschmar/s6e8-transductive-anti-student-signals",
        "license": "CC0-1.0",
        "reason": "soft_student는 najiama 스택(교사)의 소프트 라벨로 증류한 2단계 의심 산출물, 시드·행 순서 미명시",
    },
    {
        "dataset": "atakanaldemir/s6e8-v13-diversity-anchor-lb-0-97124",
        "license": "unknown",
        "reason": "244구성원 로지스틱 메타모델 출력(2단계)",
    },
    {
        "dataset": "wellkilo/s6e8-evidence-first-soft-student-assets",
        "license": "other",
        "reason": "정수 순위 벡터와 감사 계약만 있음(2단계)",
    },
    {
        "dataset": "kenchanhodgkin/pg-s6e8-exp000~exp012 child (16개)",
        "license": "CC0-1.0",
        "reason": "OOF만 있고 시험 예측 파일이 없음",
    },
    {
        "dataset": "thisray/s6e8-our-component",
        "license": "CC0-1.0",
        "reason": "시험 전용 열 하나(OOF 없음)",
    },
    {
        "dataset": (
            "anhadmahajan06, anthonytherrien, najiama/s6e8-psa, souvikdbiswas, qamrodz 등 제출 파일만 있는 데이터셋 7개"
        ),
        "license": "Apache-2.0·MIT·unknown",
        "reason": "OOF 없음(제출 CSV만)",
    },
]

# 반입하지 않는 공개 노트북 출력물. #452 조사 결과를 그대로 옮겼다.
NOTEBOOKS_NOT_IMPORTED = [
    {
        "notebook": "omidbaghchehsaraei/lookup-transformer-predicting-smartphone-addiction",
        "reason": "장부 hboyang6:kirill_o1과 OOF·시험 순위 상관 1.0(같은 배열의 재게시)",
    },
    {
        "notebook": "yadoy666/predicting-smartphone-addiction (fmdeep·앙상블 출력)",
        "reason": "fmdeep은 fm5:fmdeep과 스피어만 0.99936(같은 배열), 앙상블 출력은 2단계",
    },
    {
        "notebook": "najiama/single-lgbm-model-lb-0-96990-cv-0-96862",
        "reason": "혼합 제출에서 수확한 의사 라벨로 학습(2단계에 기댐)",
    },
    {
        "notebook": "kirill0212/s6e8-public-ensemble, yadoy666/94-verified-oof-gpu-accelerated-meta-stack",
        "reason": "기존 라이브러리·노트북 출력의 재수출(신규 학습 없음)",
    },
    {
        "notebook": (
            "dariushafshar 177-member·pool125·rank-logit-fusion, hboyang/s6e8-150-member-fusion, "
            "nikita7364777, darkmatternet oof-meta-ensemble, beicicc strict-meta·residual-audit, "
            "anthonytherrien stack, ravi20076 l2stack, lucifer19, georgymamarin, funguscakehead, "
            "wesleyhuan(oof_blend), stephentarter ensembling, rafanikitas"
        ),
        "reason": "2단계 스택 출력이거나 시험 예측이 없음",
    },
    {
        "notebook": "kodaifukuda0311/s6e8-xgb-the-power-of-exact-value-te-fe, zhenruiweng realmlp, harwindersingh766 xgb_sb",
        "reason": "시드별로 다른 분할을 평균한 배열",
    },
    {
        "notebook": "stephentarter catboost·histgradientboosting·lightgbm·xgboost",
        "reason": "설정 스크립트의 첫 시드가 10301",
    },
    {
        "notebook": (
            "ern711 multi-level spline(시드 21), darkmatternet catboost guide(시드 20260821), "
            "lavanyabacche catboost-fe(시드 2026), dranilkumardubey nova-sap, yusufmurtaza01, "
            "destroyer123787(반복 분할), magurodataanalysis(3분할), rv1922 xgb_seed777·2026"
        ),
        "reason": "커뮤니티 5분할이 아닌 분할",
    },
    {
        "notebook": "factualexplorer, tamerlanomralinov lookup-transformer-insights, evgendvorkin single-lgb, echloeprice",
        "reason": "10분할",
    },
    {
        "notebook": "vladstud716373618/baseline-5-fold-cv-catboost-deep-fe",
        "reason": "코드는 시드 42지만 내려받은 OOF의 재채점 AUC가 0.49997로 정렬 불명",
    },
    {
        "notebook": "dynamo14324 의사 라벨 판, yaminh 앙상블 열, danushkumarv 스택 출력",
        "reason": "시험 예측만 있거나 2단계",
    },
    {
        "notebook": "donmarch14/s6e8-catboost, s6e8-lgbm",
        "reason": "장부 szymon74:pub_cat·pub_donlgbm이 같은 노트북의 재실행분",
    },
    {
        "notebook": "shashwat1729/s6e8-lookup-pair-transformer",
        "reason": "코드는 5분할 seed 42이나 출력 내려받기가 권한 거부(kernels.get 403), #452와 #454에서 두 번 시도",
    },
    {
        "notebook": (
            "mhamza0810/s6e8-single-model-fe-cv-0-96947, udaken10/xgboost-improved, "
            "shamanthakreddymallu/s6e8-lightgbm"
        ),
        "reason": (
            "코드는 5분할 seed 42이고 출력 파일 목록에 OOF·시험 파일이 있으나 kernels output이 "
            "실행 기록만 내려주고 배열을 주지 않음(#452와 #454에서 패턴 있이·없이 두 번 시도)"
        ),
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
            # 판본 1은 라이선스 불명으로 제외했고, 판본 2(#454)가 사용 한정으로 통과시킨다.
            candidate.license = "unknown"
            candidate.added_in = 454
            candidate.caveats.append(NAJI_LICENSE_CAVEAT)
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


# ---------------------------------------------------------------------------
# 판본 2(#454) 증분: #452가 찾은 새 데이터셋 4곳과 공개 노트북 출력물
# ---------------------------------------------------------------------------

SOURCES_454 = [
    Source(
        key="paiky6",
        dataset="paiky1995/s6e8-oof-library-11-members",
        license="CC0-1.0",
        root=str(EXT454 / "paiky1995_s6e8-oof-library-11-members"),
        retrieved="2026-08-27 (#454)",
        fold_evidence="sibling_code",
        note=(
            "데이터셋 설명이 StratifiedKFold(shuffle=True, random_state=42)·원본 행 순서·fold 안 TE 재적합을 명시하고, "
            "같은 저자 노트북 s6e8-correlation-does-not-predict-contribution이 같은 Lookup-Transformer 골격을 "
            "고정 5분할로 학습(sibling_code). 훈련 코드 비공개, _10f 5개는 10분할이라 제외"
        ),
        added_in=454,
    ),
    Source(
        key="nhtquyn",
        dataset="nhtquyn/s6e8-addiction",
        license="CC0-1.0",
        root=str(EXT454 / "nhtquyn_s6e8-addiction"),
        retrieved="2026-08-27 (#454)",
        fold_evidence="fold_vector",
        note=(
            "설명·README·코드·저자 노트북 없음, fold_id.npy(0-based) 동봉, float32 행렬, "
            "고전 확률 모델 120개(qda·gmm·gnb·binned_nb·lda), 단독 AUC 0.853~0.930으로 약함"
        ),
        added_in=454,
    ),
    Source(
        key="hboyang150",
        dataset="hboyang/s6e8-150-fusion-local-members",
        license="unknown",
        root=str(EXT454 / "hboyang_s6e8-150-fusion-local-members" / "openx_our_members"),
        retrieved="2026-08-27 (#454)",
        fold_evidence="author_statement",
        note=(
            "README가 StratifiedKFold(5, shuffle, 42)와 원본 행 순서를 명시. README는 float64라 하나 6개는 float32. "
            "candidate_naji16_* 6개는 2단계라 제외. 같은 저자의 catstrall 데이터셋은 CC0이나 이 데이터셋은 unknown"
        ),
        added_in=454,
    ),
    Source(
        key="masaya",
        dataset="masayakawamata/s6e8-catstr-aug16",
        license="CC0-1.0",
        root=str(EXT454 / "masayakawamata_s6e8-catstr-aug16"),
        retrieved="2026-08-27 (#454)",
        fold_evidence="none",
        note="설명·README·코드·저자 노트북 없음, 공개 스택의 입력 목록에도 없음. 분할 근거가 없어 절제 부류로만 반입",
        added_in=454,
    ),
]

# #452 조사가 독립적으로 내려받아 잰 재채점 AUC. 이번 반입분이 조사 시점과 같은 배열인지 대조한다.
PAIKY_REFERENCE = {
    "v10_tabm": 0.9680063,
    "v13_lookup": 0.9682934,
    "v14_lookup_bag": 0.9687267,
    "v15_lookup_wide": 0.9681470,
    "v16_lookup_aug": 0.9681545,
    "v17_realmlp": 0.9682819,
}
HBOYANG150_REFERENCE = {
    "realmlp_fresh_s2026": 0.96349,
    "xgb_fresh_d6_s606": 0.96567,
    "xgb_fresh_d7_s314159": 0.96605,
    "lookup_fresh_d256_l8_s5150": 0.96732,
    "lookup_fresh_d384_l6_s2718": 0.96750,
    "local_lookup_d384_l4": 0.96757,
    "cat_fresh_d9_s606": 0.96766,
    "tabm_fresh_rich_s2026": 0.96827,
    "local_tabm_rich_seed909": 0.96863,
    "local_tabm_rich_seed3": 0.96867,
    "local_tabm_rich_alt": 0.96877,
}
MASAYA_REFERENCE = {"cat_str": 0.9670419}


def _paiky_declared(metadata_path: Path) -> dict[str, float]:
    """데이터셋 설명의 구성원 표(`file stem | model | OOF ROC-AUC`)에서 선언 AUC를 읽는다."""
    info = json.loads(metadata_path.read_text())
    description = info.get("info", info).get("description", "")
    pattern = re.compile(r"\|\s*`(v\d+_[a-z0-9_]+)`\s*\|[^|]*\|\s*([0-9.]+)\s*\|")
    return {name: float(auc) for name, auc in pattern.findall(description)}


def load_paiky(source: Source) -> Iterator[Candidate]:
    root = Path(source.root)
    declared = _paiky_declared(root / "dataset-metadata.json")
    for path in sorted(root.glob("oof_*.npy")):
        name = path.name[4:-4]
        test_path = path.with_name(f"testpred_{name}.npy")
        oof = np.load(path)
        candidate = Candidate(
            name=name,
            source=source.key,
            oof=oof,
            test=np.load(test_path),
            oof_path=str(path),
            test_path=str(test_path),
            declared_auc=declared.get(name),
            reference_auc=PAIKY_REFERENCE.get(name),
            upstream="architecture by tamerlanomralinov, retrained on 5 folds by paiky1995",
        )
        if name.endswith("_10f"):
            candidate.exclude = "10분할 배열(저자가 _10f 접미사와 설명으로 명시)"
        else:
            unique = len(np.unique(oof))
            if unique < 1000:
                candidate.caveats.append(
                    f"float16 양자화 흔적(고유값 {unique}개, 순위 결합기에서 대량 동점)"
                )
        yield candidate


def load_nhtquyn(source: Source) -> Iterator[Candidate]:
    root = Path(source.root)
    members = pd.read_csv(root / "members.csv")
    oof = np.load(root / "oof.npy")
    test = np.load(root / "test.npy")
    assert oof.shape == (N_TRAIN, len(members)) and test.shape == (N_TEST, len(members)), (
        oof.shape,
        test.shape,
    )
    fold_path = root / "fold_id.npy"
    for column, row in enumerate(members.itertuples()):
        yield Candidate(
            name=str(row.id),
            source=source.key,
            oof=oof[:, column],
            test=test[:, column],
            oof_path=f"{root / 'oof.npy'}[:, {column}]",
            test_path=f"{root / 'test.npy'}[:, {column}]",
            declared_auc=float(row.solo_oof_auc),
            fold_path=fold_path,
            caveats=["설명·코드 없음(저자 노트북 없음, 학습 자료 범위와 전처리 확인 불가)"],
        )


def load_hboyang150(source: Source) -> Iterator[Candidate]:
    root = Path(source.root)
    for path in sorted(root.glob("oof_*.npy")):
        stem = path.name[4:-4]
        test_path = path.with_name(f"test_{stem}.npy")
        name = stem[6:] if stem.startswith("fresh_") else stem
        candidate = Candidate(
            name=name,
            source=source.key,
            oof=np.load(path),
            test=np.load(test_path),
            oof_path=str(path),
            test_path=str(test_path),
            reference_auc=HBOYANG150_REFERENCE.get(name),
        )
        if stem.startswith("candidate_naji16"):
            candidate.exclude = "2단계 산출물(najiama 16구성원 융합 후보)"
        yield candidate


def load_masaya(source: Source) -> Iterator[Candidate]:
    root = Path(source.root)
    oof_path, test_path = root / "oof_cat_str.npy", root / "tep_cat_str.npy"
    yield Candidate(
        name="cat_str",
        source=source.key,
        oof=np.load(oof_path),
        test=np.load(test_path),
        oof_path=str(oof_path),
        test_path=str(test_path),
        reference_auc=MASAYA_REFERENCE["cat_str"],
    )


@dataclass(frozen=True)
class Array:
    """노트북 출력 파일 하나에서 배열 하나를 고르는 방법. column은 csv 열 이름 또는 npz 키."""

    file: str
    column: str | None = None

    def label(self) -> str:
        return self.file if self.column is None else f"{self.file}[{self.column}]"


@dataclass(frozen=True)
class Declared:
    """노트북이 함께 내놓은 manifest에서 선언 AUC를 읽는 방법. json은 키 경로, csv는 (행 키, 열 이름)."""

    file: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class NotebookMember:
    name: str
    oof: Array
    test: Array
    reference_auc: float | None = None
    declared: Declared | None = None
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class Notebook:
    """공개 노트북 하나. 출력 파일에는 라이선스 표시가 없으므로 전부 사용 한정이다."""

    key: str
    ref: str
    fold_evidence: str
    note: str
    members: tuple[NotebookMember, ...]

    def source(self) -> Source:
        return Source(
            key=self.key,
            dataset=self.ref,
            license="unknown",
            root=str(NOTEBOOKS454 / self.ref.replace("/", "_")),
            retrieved="2026-08-27 (#454)",
            fold_evidence=self.fold_evidence,
            note=self.note,
            kind="notebook_output",
            added_in=454,
        )


def _omid(tag: str, slug: str, name: str, reference: float, *caveats: str) -> Notebook:
    return Notebook(
        key=f"nb_omid_{tag}",
        ref=f"omidbaghchehsaraei/{slug}",
        fold_evidence="sibling_code",
        note="코드 미열람, 같은 저자의 lookup·realmlp 노트북이 5분할 seed 42(sibling_code)",
        members=(
            NotebookMember(
                name,
                Array("oof.csv", "oof_pred"),
                Array("submission.csv", TARGET),
                reference,
                caveats=caveats,
            ),
        ),
    )


def _cdeotte(tag: str, slug: str, reference: float, note: str) -> Notebook:
    return Notebook(
        key=f"nb_cdeotte_{tag}",
        ref=f"cdeotte/{slug}",
        fold_evidence="published_code",
        note=note,
        members=(
            NotebookMember(
                tag,
                Array("oof_predictions.csv", "oof_pred"),
                Array("submission.csv", TARGET),
                reference,
            ),
        ),
    )


QUANTIZED = "고유값 4,000대(양자화 흔적, 순위 결합기에서 동점 많음)"
NOTEBOOKS: tuple[Notebook, ...] = (
    Notebook(
        key="nb_kodaifukuda",
        ref="kodaifukuda0311/s6e8-how-to-achieve-0-97-with-realmlp-only",
        fold_evidence="published_code",
        note="RealMLP 단일, 5분할 seed 42, 원자료(jayjoshi37) 분포 통계를 특성으로 참조, 공개 점수 0.97016, 08-27 실행",
        members=(
            NotebookMember(
                "realmlp",
                Array("oof_realmlp.npy"),
                Array("pred_realmlp.npy"),
                0.9689929,
            ),
        ),
    ),
    _omid("ft", "ft-transformer-for-predicting-smartphone-addiction", "ft_transformer", 0.9665681),
    _omid("cnn", "cnn-for-predicting-smartphone-addiction", "cnn", 0.9677056, QUANTIZED),
    _omid("tabtr", "tabtransformer-predicting-smartphone-addiction", "tabtransformer", 0.9674689, QUANTIZED),
    _omid("fastai", "fastai-for-predicting-smartphone-addiction", "fastai", 0.9667558),
    _omid("xgb2", "xgboost-v2-for-predicting-smartphone-addiction", "xgboost_v2", 0.9687332),
    _omid("cat", "catboost-for-predicting-smartphone-addiction", "catboost", 0.9671499),
    Notebook(
        key="nb_zhukov",
        ref="zhukovoleksiy/ps6e8-eda-feature-engineering-pipeline",
        fold_evidence="published_code",
        note="oof/manifest.csv가 구성원별 분할(StratifiedKFold(5, shuffle=True, random_state=42))과 OOF AUC를 명시",
        members=tuple(
            NotebookMember(
                name,
                Array(f"oof/oof_lexD_{name}.npy"),
                Array(f"oof/test_lexD_{name}.npy"),
                reference,
                Declared("oof/manifest.csv", (f"lexD_{name}", "oof_auc")),
            )
            for name, reference in (
                ("cat_base", 0.9679922),
                ("lgb02", 0.9683554),
                ("xgb_base", 0.9678591),
            )
        ),
    ),
    Notebook(
        key="nb_reda_lgbm",
        ref="redamountassir/s6e8-lgbm-lb-0-96965",
        fold_evidence="published_code",
        note="kirill0212 공개 스택의 e-* 입력, 예측 열 이름이 addicted_label",
        members=(
            NotebookMember(
                "lgbm",
                Array("lgbm_oof_preds.csv", TARGET),
                Array("lgbm_test_preds.csv", TARGET),
                0.9682590,
            ),
        ),
    ),
    Notebook(
        key="nb_reda_hgb",
        ref="redamountassir/s6e8-histgradientboosting-lb-0-96945",
        fold_evidence="published_code",
        note="kirill0212 공개 스택의 e-* 입력, 예측 열 이름이 addicted_label",
        members=(
            NotebookMember(
                "hgb",
                Array("tehgbc_oof_preds.csv", TARGET),
                Array("tehgbc_test_preds.csv", TARGET),
                0.9680258,
            ),
        ),
    ),
    Notebook(
        key="nb_yaminh",
        ref="yaminh/smartphone-addiction-prediction-strong-eda-cv-eble",
        fold_evidence="published_code",
        note="기반 모델 3개만 반입, 앙상블 열은 2단계라 제외",
        members=tuple(
            NotebookMember(
                name,
                Array("oof_base_models.csv", name),
                Array("test_base_models.csv", name),
                reference,
            )
            for name, reference in (
                ("lgbm_te", 0.9675084),
                ("xgb_te", 0.9677343),
                ("catboost", 0.9619284),
            )
        ),
    ),
    Notebook(
        key="nb_sidhaarth",
        ref="sidhaarthshree/lightgbm-ensemble-based-on-eda",
        fold_evidence="published_code",
        note="npz 한 파일에 oof_*/test_* 키",
        members=tuple(
            NotebookMember(
                name,
                Array("oof_test_predictions.npz", f"oof_{name}"),
                Array("oof_test_predictions.npz", f"test_{name}"),
                reference,
            )
            for name, reference in (
                ("lgb_a", 0.9676538),
                ("lgb_b", 0.9675589),
                ("xgb", 0.9677400),
            )
        ),
    ),
    Notebook(
        key="nb_yekenot",
        ref="yekenot/ps-s6-e8-trompt-pytorch-frame",
        fold_evidence="published_code",
        note="Trompt(pytorch-frame), 장부에 없는 계열, 예측 열 이름이 addicted_label",
        members=(
            NotebookMember(
                "trompt",
                Array("oof_preds.csv", TARGET),
                Array("submission.csv", TARGET),
                0.9666711,
            ),
        ),
    ),
    Notebook(
        key="nb_lucy_xgb",
        ref="lucymlai32/phase-2-xgboost-and-model-blending",
        fold_evidence="published_code",
        note="XGBoost 단일",
        members=(
            NotebookMember(
                "xgboost",
                Array("xgboost_oof.csv", "xgboost_prediction"),
                Array("xgboost_test.csv", "xgboost_prediction"),
                0.9657988,
            ),
        ),
    ),
    Notebook(
        key="nb_lucy_cat",
        ref="lucymlai32/smartphone-addiction-prediction",
        fold_evidence="published_code",
        note="CatBoost v2 단일",
        members=(
            NotebookMember(
                "catboost_v2",
                Array("catboost_v2_oof.csv", "catboost_v2_prediction"),
                Array("catboost_v2_test.csv", "catboost_v2_prediction"),
                0.9643755,
            ),
        ),
    ),
    _cdeotte("xgb", "simple-xgb-starter", 0.9648116, "Chris Deotte 시작 노트북(XGB)"),
    _cdeotte("cat", "simple-cat-starter", 0.9629062, "Chris Deotte 시작 노트북(CatBoost)"),
    _cdeotte("nn", "simple-nn-starter", 0.9397375, "Chris Deotte 시작 노트북(NN)"),
    Notebook(
        key="nb_lavanya",
        ref="lavanyabacche/xgb-starter-01",
        fold_evidence="published_code",
        note="cdeotte XGB 시작 노트북의 포크, #452는 같은 배열로 보고 세지 않음(바이트 중복이면 제외된다)",
        members=(
            NotebookMember(
                "xgb_starter",
                Array("oof_predictions.csv", "oof_pred"),
                Array("submission.csv", TARGET),
                0.9648116,
            ),
        ),
    ),
    Notebook(
        key="nb_darius_ablation",
        ref="dariushafshar/0-97184-leader-xgb-feature-ablation",
        fold_evidence="published_code",
        note="cdeotte XGB 변형(특성 절제)",
        members=(
            NotebookMember(
                "xgb",
                Array("oof_predictions.csv", "oof_pred"),
                Array("submission.csv", TARGET),
                0.9648769,
            ),
        ),
    ),
    Notebook(
        key="nb_rv1922",
        ref="rv1922/smartphone-addiction",
        fold_evidence="published_code",
        note="model_outputs/manifest.json에 구성원별 OOF AUC, xgb seed777·2026은 다른 분할이라 제외",
        members=tuple(
            NotebookMember(
                name,
                Array(f"model_outputs/oof_{name}.npy"),
                Array(f"model_outputs/test_{name}.npy"),
                reference,
                Declared("model_outputs/manifest.json", ("individual_scores", name)),
            )
            for name, reference in (
                ("lgbm_v1_seed42", 0.9632600),
                ("lgbm_v2_seed42", 0.9637395),
                ("lgbm_v3_seed42", 0.9640291),
                ("xgb_seed42", 0.9648178),
            )
        ),
    ),
    Notebook(
        key="nb_yadoy",
        ref="yadoy666/predicting-smartphone-addiction",
        fold_evidence="published_code",
        note="catboost·xgboost 단일만 반입, fmdeep은 fm5:fmdeep과 같은 배열이고 앙상블 출력은 2단계라 제외",
        members=(
            NotebookMember(
                "catboost",
                Array("catboost_oof_predictions.csv.gz", "oof_prediction"),
                Array("catboost_test_predictions.npy"),
                0.9636971,
            ),
            NotebookMember(
                "xgboost",
                Array("xgboost_oof_predictions.csv.gz", "oof_prediction"),
                Array("xgboost_test_predictions.npy"),
                0.9647145,
            ),
        ),
    ),
    Notebook(
        key="nb_danush",
        ref="danushkumarv/smartphone-addiction-gbm-rank-blend-nb01",
        fold_evidence="published_code",
        note="기반 모델 3개, 스택 출력은 najiama 입력이라 제외",
        members=tuple(
            NotebookMember(name, Array(f"oof_{name}.npy"), Array(f"pred_{name}.npy"), reference)
            for name, reference in (("lgb", 0.9639537), ("xgb", 0.9644659), ("cb", 0.9627085))
        ),
    ),
    Notebook(
        key="nb_harwinder",
        ref="harwindersingh766/ps-s6e8-xgboost-te-lb-0-96548",
        fold_evidence="published_code",
        note="_sb 판은 시드 3개 분할 평균이라 제외",
        members=(NotebookMember("xgb", Array("oof_xgb.npy"), Array("test_xgb.npy"), 0.9640958),),
    ),
    Notebook(
        key="nb_dynamo",
        ref="dynamo14324/smartphone-addiction-championship-v11",
        fold_evidence="published_code",
        note="의사 라벨 판은 시험 예측만 있어 제외",
        members=(
            NotebookMember("lgb_v11", Array("oof_lgb_v11.npy"), Array("test_lgb_v11.npy"), 0.9633701),
            NotebookMember("xgb_v11", Array("oof_xgb_v11.npy"), Array("test_xgb_v11.npy"), 0.9632193),
        ),
    ),
    Notebook(
        key="nb_mohan_realmlp",
        ref="mohankrishnathalla/s6e8-realmlp-oof-saver",
        fold_evidence="published_code",
        note="RealMLP",
        members=(NotebookMember("realmlp", Array("oof_realmlp.npy"), Array("test_realmlp.npy"), 0.9581337),),
    ),
    Notebook(
        key="nb_mohan_tabm",
        ref="mohankrishnathalla/s6e8-tabm-oof-saver",
        fold_evidence="published_code",
        note="제목은 TabM이나 코드는 MLP",
        members=(NotebookMember("mlp", Array("oof_mlp.npy"), Array("test_mlp.npy"), 0.9414221),),
    ),
    Notebook(
        key="nb_kava1",
        ref="kava1/predicting-smartphone-addiction-resnet-fe",
        fold_evidence="published_code",
        note="ResNet",
        members=(
            NotebookMember(
                "resnet",
                Array("oof_preds_resnet_MYSELF_95687.npy"),
                Array("test_preds_resnet_MYSELF_95687.npy"),
                0.9568735,
            ),
        ),
    ),
    Notebook(
        key="nb_lopure",
        ref="lopure/hdviz-pca-parallel-with-linear-svm",
        fold_evidence="published_code",
        note="SVM 계열 3개, rbf는 장부 최근접 상관 0.895로 가장 다름",
        members=tuple(
            NotebookMember(
                f"{tag}_svm",
                Array(f"oof_{tag}_svm_gpu.csv", f"{column}_oof_pred"),
                Array(f"submission_{tag}_svm_gpu.csv", TARGET),
                reference,
            )
            for tag, column, reference in (
                ("linear", "Linear", 0.9113454),
                ("poly", "Poly", 0.9287951),
                ("rbf", "RBF", 0.9221669),
            )
        ),
    ),
    Notebook(
        key="nb_shaman_baseline",
        ref="shamanthakreddymallu/s6e8-baseline",
        fold_evidence="none",
        note="코드에서 분할 시드를 찾지 못함(저자 서술 있으면 풀림), 분할 근거가 없어 절제 부류로만 반입",
        members=(
            NotebookMember("lgb_fe", Array("oof_lgb_fe.npy"), Array("pred_lgb_fe.npy"), 0.9637670),
            NotebookMember("lr", Array("oof_lr.npy"), Array("pred_lr.npy"), 0.9366092),
        ),
    ),
)


def _load_array(root: Path, spec: Array, expected_ids: np.ndarray) -> tuple[np.ndarray, str | None]:
    """배열 하나를 읽는다. csv에 id 열이 있으면 기준 순서와 대조하고 순열이면 재정렬한다. 둘째 값은 주의 사항."""
    path = root / spec.file
    if path.suffix == ".npy":
        values = np.load(path)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.reshape(-1)
        return values, None
    if path.suffix == ".npz":
        return np.load(path)[spec.column], None
    frame = pd.read_csv(path)
    if spec.column not in frame.columns:
        raise ValueError(f"{path}: 열 {spec.column!r} 없음, 열 목록 {list(frame.columns)}")
    values = frame[spec.column].to_numpy()
    if ID in frame.columns and len(frame) == len(expected_ids):
        ids = frame[ID].to_numpy()
        if np.array_equal(ids, expected_ids):
            return values, None
        if np.array_equal(np.sort(ids), np.sort(expected_ids)):
            values = pd.Series(values, index=ids).loc[expected_ids].to_numpy()
            return values, "id 열이 기준 순서와 달라 id로 재정렬"
        raise ValueError(f"{path}: id 집합이 기준과 다름")
    return values, None


def _declared_auc(root: Path, declared: Declared | None) -> float | None:
    if declared is None:
        return None
    path = root / declared.file
    if path.suffix == ".json":
        return _json_path(json.loads(path.read_text()), declared.path)
    row_key, column = declared.path
    frame = pd.read_csv(path)
    return float(frame.set_index(frame.columns[0]).loc[row_key, column])


def _normalized(root: Path, kind: str, name: str, values: np.ndarray) -> Path:
    """csv·npz 출력은 하류 도구가 읽는 npy로 정규화해 둔다. 같은 값이면 다시 쓰지 않는다."""
    path = root / "normalized" / f"{kind}_{name}.npy"
    values = np.ascontiguousarray(values, dtype=np.float64)
    if path.exists() and np.array_equal(np.load(path), values):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    return path


def load_notebook(
    source: Source, notebook: Notebook, ids: dict[str, np.ndarray]
) -> Iterator[Candidate]:
    root = Path(source.root)
    for spec in notebook.members:
        caveats = list(spec.caveats)
        exclude = None
        paths = {"oof": root / spec.oof.file, "test": root / spec.test.file}
        try:
            oof, oof_note = _load_array(root, spec.oof, ids["oof"])
            test, test_note = _load_array(root, spec.test, ids["test"])
        except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
            oof, test = np.empty(0), np.empty(0)
            exclude = f"적재 실패: {exc}"
        else:
            caveats.extend(note for note in (oof_note, test_note) if note)
            if spec.oof.column is not None:
                paths["oof"] = _normalized(root, "oof", spec.name, oof)
            if spec.test.column is not None:
                paths["test"] = _normalized(root, "test", spec.name, test)
        yield Candidate(
            name=spec.name,
            source=source.key,
            oof=oof,
            test=test,
            oof_path=str(paths["oof"]),
            test_path=str(paths["test"]),
            declared_auc=None if exclude else _declared_auc(root, spec.declared),
            reference_auc=spec.reference_auc,
            upstream=f"{spec.oof.label()}, {spec.test.label()}",
            caveats=caveats,
            exclude=exclude,
        )


ALL_BEICICC = {**BEICICC_DATASETS, **BEICICC_DATASETS_454}


def beicicc_sources() -> list[Source]:
    return [
        Source(
            key=f"beicicc:{slug}",
            dataset=f"beicicc/{slug}",
            license=license,
            root=str(EXT94 / "beicicc" / slug),
            retrieved="2026-08-23 (#386)"
            if slug in BEICICC_DATASETS
            else "2026-08-23 (#386), 2026-08-27 반입(#454)",
            fold_evidence="fold_vector",
            note="훈련 코드 비공개, contract JSON에 하이퍼파라미터·고정 스케줄·fold 해시 기록, fold_id.npy 동봉(1-based)"
            if slug in BEICICC_DATASETS
            else "훈련 코드 비공개, manifest.json에 계약·fold 해시·AUC 기록, fold_id.npy 동봉(1-based), 라이선스 other라 판본 1은 미반입",
            added_in=442 if slug in BEICICC_DATASETS else 454,
        )
        for slug, (license, _) in ALL_BEICICC.items()
    ]


def load_beicicc(source: Source) -> Iterator[Candidate]:
    slug = source.key.split(":", 1)[1]
    root = Path(source.root)
    _, members = ALL_BEICICC[slug]
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


def iter_candidates(
    sources: list[Source], ids: dict[str, np.ndarray]
) -> Iterator[tuple[Source, Candidate]]:
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
        "paiky6": load_paiky,
        "nhtquyn": load_nhtquyn,
        "hboyang150": load_hboyang150,
        "masaya": load_masaya,
    }
    notebooks = {notebook.key: notebook for notebook in NOTEBOOKS}
    for source in sources:
        if source.kind == "notebook_output":
            candidates = load_notebook(source, notebooks[source.key], ids)
        elif source.key.startswith("beicicc:"):
            candidates = load_beicicc(source)
        else:
            candidates = loaders[source.key](source)
        for candidate in candidates:
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
    license = candidate.license or source.license
    fold_evidence = candidate.fold_evidence or source.fold_evidence
    caveats = list(candidate.caveats)
    if rows_ok and (oof.dtype != np.float64 or test.dtype != np.float64):
        caveats.append(
            f"{oof.dtype} 저장(스택 정밀도 손실 가능, szymonkapiski 측정 공개 점수 -0.00001)"
        )
    if fold_evidence == "none":
        caveats.append(FOLD_NONE_CAVEAT)
    if source.kind == "notebook_output":
        caveats.append(LICENSE_CAVEATS["notebook_output"])
    elif candidate.license is None and license in LICENSE_CAVEATS:
        caveats.append(LICENSE_CAVEATS[license])
    finite = bool(rows_ok and np.isfinite(oof).all() and np.isfinite(test).all())
    auc = (
        float(roc_auc_score(y, oof.astype(np.float64))) if rows_ok and finite else None
    )
    delta = (
        None
        if auc is None or candidate.declared_auc is None
        else auc - candidate.declared_auc
    )
    reference_delta = (
        None
        if auc is None or candidate.reference_auc is None
        else auc - candidate.reference_auc
    )
    if reference_delta is not None and abs(reference_delta) > AUC_TOLERANCE:
        caveats.append(
            f"reference_mismatch: #452 조사 재채점 AUC {candidate.reference_auc:.7f}와 "
            f"{reference_delta:+.2e} 차이(조사 뒤 재실행 가능성)"
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
    elif (
        reason is None
        and candidate.declared_auc is None
        and auc is not None
        and auc < AUC_FLOOR
    ):
        reason = f"선언 AUC 없이 재채점 AUC {auc:.5f}(정렬 불명)"
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
        license=license,
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
        fold_evidence=fold_evidence,
        upstream=candidate.upstream,
        caveats=caveats,
        in_ext85=in_ext85,
        status="excluded" if reason else "accepted",
        exclusion_reason=reason,
        added_in=candidate.added_in or source.added_in,
        reference_auc=candidate.reference_auc,
        reference_auc_delta=reference_delta,
    )


def _zrank(values: np.ndarray) -> np.ndarray:
    """OOF의 표준화된 순위. 두 벡터의 내적을 행 수로 나누면 스피어만 순위 상관이다."""
    ranks = rankdata(np.asarray(values, dtype=np.float64).reshape(-1))
    return ((ranks - ranks.mean()) / ranks.std()).astype(np.float64)


def mark_near_duplicates(members: list[Member], ranks: dict[str, np.ndarray]) -> dict[str, object]:
    """통과 구성원끼리 OOF 스피어만이 기준을 넘는 쌍을 군집으로 묶고 대표가 아닌 #454 구성원에 주의 사항을 단다.

    대표는 판본 1 구성원이 군집에 있으면 그중 AUC가 가장 높은 것, 없으면 군집에서 AUC가 가장 높은
    #454 구성원이다. 판본 1 구성원은 어느 경우에도 주의 사항이 바뀌지 않는다.
    """
    accepted = [m for m in members if m.status == "accepted"]
    ids = [m.member_id for m in accepted]
    matrix = np.vstack([ranks[i] for i in ids])
    correlation = matrix @ matrix.T / N_TRAIN
    del matrix
    np.fill_diagonal(correlation, -np.inf)
    closest = correlation.argmax(axis=1)
    for row, member in enumerate(accepted):
        member.spearman_max = float(correlation[row, closest[row]])
        member.spearman_closest = ids[closest[row]]
    parent = list(range(len(accepted)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    pairs = np.argwhere(np.triu(correlation > NEAR_DUPLICATE_SPEARMAN, 1))
    for left, right in pairs:
        parent[find(int(left))] = find(int(right))
    clusters: dict[int, list[int]] = {}
    for index in range(len(accepted)):
        clusters.setdefault(find(index), []).append(index)
    flagged = 0
    cluster_sizes: list[int] = []
    for indices in clusters.values():
        if len(indices) < 2:
            continue
        cluster_sizes.append(len(indices))
        existing = [i for i in indices if accepted[i].added_in == 442]
        pool = existing or indices
        representative = max(pool, key=lambda i: (accepted[i].auc or 0.0, -i))
        for index in indices:
            member = accepted[index]
            if index == representative or member.added_in != 454:
                continue
            member.caveats.append(
                f"near_duplicate_cluster: 대표 {accepted[representative].member_id}, "
                f"대표와 스피어만 {correlation[index, representative]:.5f}, "
                f"군집 크기 {len(indices)}({NEAR_DUPLICATE_SPEARMAN} 초과 연결)"
            )
            flagged += 1
    return {
        "threshold": NEAR_DUPLICATE_SPEARMAN,
        "pairs_over_threshold": int(len(pairs)),
        "clusters": len(cluster_sizes),
        "largest_cluster": max(cluster_sizes, default=0),
        "flagged_454": flagged,
    }


def row_order_checks(
    fold_of: pd.Series, y: np.ndarray, train_ids: np.ndarray, test_ids: np.ndarray
) -> dict[str, object]:
    """외부 라이브러리가 전제하는 위치 정렬이 우리 기준 순서와 같은지 확인한다."""
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
    # 판본 2(#454) 공급원. nhtquyn은 분할 벡터(0-based)로 대조한다. paiky1995의 csv `id` 열은
    # train 쪽이 train.csv id(=행 위치)와 같고 test 쪽은 0부터 세는 행 위치라서, 두 csv 모두
    # 위치 정렬을 뜻하며 npy와 같은 순서임을 확인한다.
    nhtquyn = EXT454 / "nhtquyn_s6e8-addiction" / "fold_id.npy"
    checks["nhtquyn_fold_id_equals_ours"] = bool(
        np.array_equal(np.load(nhtquyn).astype(np.int64), fold_of.to_numpy())
    )
    paiky = EXT454 / "paiky1995_s6e8-oof-library-11-members"
    checks["paiky_oof_csv_id_equals_train_csv"] = bool(
        np.array_equal(
            pd.read_csv(paiky / "oof_predictions.csv", usecols=[ID])[ID].to_numpy(),
            train_ids,
        )
    )
    checks["paiky_test_csv_id_is_row_position"] = bool(
        np.array_equal(
            pd.read_csv(paiky / "test_predictions.csv", usecols=[ID])[ID].to_numpy(),
            np.arange(len(test_ids)),
        )
    )
    return checks


def main() -> None:
    fold_of = pd.read_parquet(FOLDS_PATH).set_index(ID)["fold"]
    y = labels(fold_of.index).to_numpy()
    fold_array = fold_of.to_numpy()
    train_ids = pd.read_csv(TRAIN_PATH, usecols=[ID])[ID].to_numpy()
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID])[ID].to_numpy()
    checks = row_order_checks(fold_of, y, train_ids, test_ids)
    print("행 순서 대조:")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    assert checks["folds_id_order_equals_train_csv"], (
        "folds.parquet id 순서가 train.csv와 다르다"
    )
    assert checks["findings_pack_folds_equals_ours"], (
        "findings pack 분할이 우리 분할과 다르다"
    )
    assert checks["nhtquyn_fold_id_equals_ours"], "nhtquyn 분할 벡터가 우리 분할과 다르다"

    # 판본 1 공급원을 먼저 적재해 바이트 중복 제거와 군집 대표 선정에서 판본 1이 우선하게 한다.
    sources = (
        SOURCES
        + beicicc_sources()
        + SOURCES_454
        + [notebook.source() for notebook in NOTEBOOKS]
    )
    members: list[Member] = []
    seen: dict[str, str] = {}
    ranks: dict[str, np.ndarray] = {}
    for source, candidate in iter_candidates(sources, {"oof": train_ids, "test": test_ids}):
        member = verify(candidate, source, y, fold_array)
        if member.status == "accepted" and member.sha256 in seen:
            member.status = "excluded"
            member.exclusion_reason = f"바이트 중복: {seen[member.sha256]}"
        elif member.status == "accepted":
            seen[member.sha256] = member.member_id
            ranks[member.member_id] = _zrank(candidate.oof)
        members.append(member)
        flag = (
            "ok" if member.status == "accepted" else f"제외({member.exclusion_reason})"
        )
        auc_text = "-" if member.auc is None else f"{member.auc:.7f}"
        print(f"  {member.member_id:<60s} {auc_text} {member.fold_check} {flag}")
        del candidate.oof, candidate.test

    near_duplicates = mark_near_duplicates(members, ranks)
    del ranks
    print(f"\n중복 군집: {near_duplicates}")

    accepted = [m for m in members if m.status == "accepted"]
    frame = pd.DataFrame([asdict(m) for m in members])
    rescored = frame.dropna(subset=["auc_delta"])
    referenced = frame.dropna(subset=["reference_auc_delta"])
    categories = ("license_unknown", "fold_evidence_none", "near_duplicate_cluster", "reference_mismatch")
    source_kinds = {s.key: s.kind for s in sources}
    new_accepted = [m for m in accepted if m.added_in == 454]
    nhtquyn_representatives = sum(
        1
        for m in new_accepted
        if m.source == "nhtquyn"
        and not any(c.startswith("near_duplicate_cluster") for c in m.caveats)
    )
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
        # 판본 2(#454) 증분 요약
        "accepted_by_added_in": {
            str(k): int(v)
            for k, v in frame[frame.status == "accepted"].groupby("added_in").size().items()
        },
        "accepted_by_kind": {
            kind: int(sum(1 for m in accepted if source_kinds[m.source] == kind))
            for kind in ("dataset", "notebook_output")
        },
        "new_candidate_count": int(sum(1 for m in members if m.added_in == 454)),
        "new_accepted_count": len(new_accepted),
        "new_excluded_count": int(
            sum(1 for m in members if m.added_in == 454 and m.status == "excluded")
        ),
        "caveat_categories_accepted": {
            category: int(
                sum(1 for m in accepted if any(c.startswith(category) for c in m.caveats))
            )
            for category in categories
        },
        "caveat_categories_new_accepted": {
            category: int(
                sum(1 for m in new_accepted if any(c.startswith(category) for c in m.caveats))
            )
            for category in categories
        },
        "nhtquyn_cluster_representatives": int(nhtquyn_representatives),
        "near_duplicates": near_duplicates,
        "max_abs_reference_delta_accepted": (
            None
            if referenced[referenced.status == "accepted"].empty
            else float(referenced[referenced.status == "accepted"]["reference_auc_delta"].abs().max())
        ),
        "new_accepted_auc_min": float(min(m.auc for m in new_accepted)),
        "new_accepted_auc_max": float(max(m.auc for m in new_accepted)),
    }
    assert summary["ext85_in_ledger"] + summary["ext85_not_in_ledger"] == 85, summary
    print("\n요약:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    ledger = {
        "issue": LEDGER_ISSUE,
        "version": LEDGER_VERSION,
        "history": HISTORY,
        "generated": GENERATED,
        "fold_spec": FOLD_SPEC,
        "auc_tolerance": AUC_TOLERANCE,
        "auc_floor_without_declaration": AUC_FLOOR,
        "near_duplicate_spearman": NEAR_DUPLICATE_SPEARMAN,
        "license_policy": LICENSE_POLICY,
        "row_order_checks": checks,
        "sources": [asdict(s) for s in sources],
        "sources_not_imported": SOURCES_NOT_IMPORTED,
        "notebooks_not_imported": NOTEBOOKS_NOT_IMPORTED,
        "ext85_not_in_ledger": EXT85_NOT_IN_LEDGER,
        "summary": summary,
        "members": [asdict(m) for m in members],
    }
    OUT_PATH.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
    print(f"\n장부 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
