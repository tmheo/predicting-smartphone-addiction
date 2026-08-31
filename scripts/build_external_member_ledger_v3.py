"""엄격 외부 후보를 반입해 외부 구성원 장부 판본 3의 감사 기록을 만든다. (#484, #487)

판본 3은 판본 2(#454)의 단일 `status` 구조 대신 #482 결정 댓글의 계약을 따른다.

- `외부 구성원 감사 기록`: 후보 하나의 고정 공개 판본, 예측 쌍, 근거, 자격 판정, 보증 한계를 함께
  고정한 변경 불가 기록. `docs/research/external-member-ledger-v3/records/<감사 기록 식별자>.json`.
- `외부 구성원 자격 판정`: 기록의 근거에서 계산한 `자격 있음`, `자격 없음`, `근거 부족` 가운데 하나.
- 감사 진행 상태(`발견됨 → 공개 판본 고정됨 → 감사 완료`)와 자격 판정은 별도 필드다.
- 색인 `index.json`은 현행 감사 기록만 가리키며 예측 배열을 담지 않는다.
- 공개 판본이나 근거가 바뀌면 기존 기록을 고치지 않고 새 기록을 만들어 `supersedes`로 잇는다.

입력 후보는 이슈 #480(판본 2 공개 노트북 재감사 통과 11개)과 #479(장부 밖 전수 조사 통과 8개)의
고정 조사 결과이며, 이 스크립트는 그 근거 위치를 고정 공개 판본의 소스에서 다시 찾아 검증한다.

사용법:
    uv run python scripts/build_external_member_ledger_v3.py fetch     # Kaggle에서 고정 판본 소스·출력 확보
    uv run python scripts/build_external_member_ledger_v3.py audit     # 검증, 감사 기록·색인·요약 생성
    uv run python scripts/build_external_member_ledger_v3.py verify    # 변경 불가 규칙과 배열 미포함 검증

외부 파일은 `data/external/ext484/` 아래에만 두고 저장소에 커밋하거나 재배포하지 않는다.
이 스크립트는 읽기 전용이다. 외부 예측을 후보 풀이나 champion 판정에 넣지 않고 MLflow 실행도 만들지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib.numpy_pickle import NumpyUnpickler
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from pipeline.data import ID, TARGET, TRAIN_PATH
from pipeline.judgment import FOLDS_PATH

ISSUE = 484
INCREMENTAL_ISSUE = 487
LEDGER_VERSION = 3
CONTRACT_VERSION = "3.0"
CONTRACT_REF = "https://github.com/tmheo/predicting-smartphone-addiction/issues/482"
EXT = Path("data/external/ext484")
OUT_DIR = Path("docs/research/external-member-ledger-v3")
RECORDS_DIR = OUT_DIR / "records"
EVIDENCE_DIR = OUT_DIR / "evidence"
INDEX_PATH = OUT_DIR / "index.json"
RUN_PATH = OUT_DIR / "ingest-run.json"
SUMMARY_PATH = Path("docs/research/external-member-ledger-v3.md")
V2_LEDGER_PATH = Path("docs/research/external-member-ledger.json")
TEST_PATH = Path("data/test.csv")

N_TRAIN = 691369
N_TEST = 296302
AUC_TOLERANCE = 1e-5
NEAR_DUPLICATE_SPEARMAN = 0.998
FOLD_SPEC_ID = "community-skf5-shuffle-seed42-train-csv-order"
FOLD_SPEC = "StratifiedKFold(n_splits=5, shuffle=True, random_state=42), train.csv 원본 행 순서"

REAUDIT_REF = (
    "https://github.com/tmheo/predicting-smartphone-addiction/blob/dc0cc95/"
    "docs/research/external-member-ledger-v2-strict-notebook-reaudit.md"
)
CENSUS_REF = (
    "https://github.com/tmheo/predicting-smartphone-addiction/blob/bff57dc/"
    "docs/research/strict-external-member-census-2026-08-28.md"
)
INCREMENTAL_REF = "https://github.com/tmheo/predicting-smartphone-addiction/issues/487"
V2_LEDGER_REF = (
    "https://github.com/tmheo/predicting-smartphone-addiction/blob/96ec136/"
    "docs/research/external-member-ledger.json"
)
SOURCE_LICENSE_EVIDENCE = "https://www.kaggle.com/datasets/kaggle/meta-kaggle-code"
USE_SCOPE = "확장 스택 제출의 결합 입력으로만 사용"
LICENSE_CAVEAT = "license_unknown_use_limited"

AUDIT_STATES = ("발견됨", "공개 판본 고정됨", "감사 완료")
CONFIRMED, VIOLATION, UNKNOWN, NOT_APPLICABLE = "확인됨", "위반 확인", "알 수 없음", "해당 없음"
ELIGIBLE, INELIGIBLE, INSUFFICIENT = "자격 있음", "자격 없음", "근거 부족"

CLAIM_KEYS = (
    "direct_official_training",
    "single_model_lineage",
    "allowed_aggregation_only",
    "target_preprocessing_isolated",
    "outer_valid_scoring_only",
    "training_point_fixed",
    "public_config_selection_isolated",
    "no_external_prediction_retraining",
    "no_pseudo_label_training",
    "no_blend_distillation",
)
CLAIM_TEXT = {
    "direct_official_training": "공식 훈련 자료에서 직접 학습했다",
    "single_model_lineage": "하나의 모델 계보이며 서로 다른 모델이나 설정의 결합이 아니다",
    "allowed_aggregation_only": "같은 고정 설정의 여러 시드나 분할 예측 평균만 사용했다",
    "target_preprocessing_isolated": "목표값 기반 전처리를 각 바깥 학습 부분 안에 격리했다",
    "outer_valid_scoring_only": "바깥 검증 자료와 목표값은 채점에만 사용했다",
    "training_point_fixed": "학습 횟수나 가중치를 바깥 검증 결과 전에 고정했다",
    "public_config_selection_isolated": "공개 소스 안의 설정 선택에 바깥 검증 결과를 사용하지 않았다",
    "no_external_prediction_retraining": "외부 예측을 입력으로 다시 학습하지 않았다",
    "no_pseudo_label_training": "의사 목표값을 학습에 사용하지 않았다",
    "no_blend_distillation": "결합 예측을 교사로 삼아 증류하지 않았다",
}
ABSENCE_PATTERNS = {
    "pseudo": r"pseudo",
    "external_input": r"/kaggle/input/(datasets|notebooks)/",
    "distill": r"distill|teacher",
}

GUARANTEES = [
    "기록한 고정 공개 판본의 소스와 연결 산출물에서 확인할 수 있는 학습 계보",
    "OOF 691,369행과 시험 예측 296,302행의 유한값, 원래 행 순서와 정규화 예측 쌍 해시",
    "고정 5분할 명세와 소스 안의 분할 코드 위치",
]
NON_GUARANTEES = [
    "공개 이전의 비공개 설정 탐색 이력",
    "작성자가 공개하지 않은 코드와 숨은 외부 자료 사용",
    "미래 공개 판본",
    "출력 배열에 표시되지 않은 권리 범위(사용 한정을 넘는 사용)",
]


# ---------------------------------------------------------------------------
# 명세 자료형
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArraySpec:
    file: str
    fmt: str  # npy | csv | parquet | joblib_dataframe
    column: str | None = None
    column_index: int | None = None
    frame_key: str | None = None
    has_id: bool = False
    label_column: str | None = None
    dtype_note: str | None = None

    def label(self) -> str:
        if self.frame_key is not None and self.column is not None:
            return f"{self.file}[{self.frame_key}][{self.column}]"
        if self.column is not None:
            return f"{self.file}[{self.column}]"
        if self.column_index is not None:
            return f"{self.file}[:, {self.column_index}]"
        return self.file


@dataclass(frozen=True)
class Declared:
    file: str
    kind: str  # csv_row | json_key
    key: str
    column: str | None = None


@dataclass(frozen=True)
class Claim:
    status: str
    note: str
    snippets: tuple[str, ...] = ()
    kind: str = "published_code"
    file: str | None = None  # None이면 노트북 소스, 아니면 ext484 아래 경로(/ 포함) 또는 소스 디렉터리 안의 파일
    extra_files: tuple[tuple[str, str], ...] = ()  # (파일, 조각) 추가 근거


@dataclass(frozen=True)
class Dependency:
    kernel_ref: str
    script_version_id: int
    role: str
    files: dict[str, str]  # 파일 이름 → 고정 SHA-256


@dataclass(frozen=True)
class Member:
    name: str
    member_id: str
    display_name: str
    oof: ArraySpec
    test: ArraySpec
    semantics: str
    prior_pair_sha256: str
    independent_auc: float
    training_point: Claim
    lineage_snippets: tuple[str, ...] = ()
    declared: Declared | None = None
    caveats: tuple[tuple[str, str], ...] = ()
    claim_overrides: dict[str, Claim] = field(default_factory=dict)


@dataclass(frozen=True)
class Source:
    key: str
    kernel_ref: str
    script_version_id: int
    pinned_source_sha256: str
    author: str
    title: str
    population: str  # v2_reaudit | census
    prior_ref: str
    fold: Claim
    claims: dict[str, Claim]
    members: tuple[Member, ...]
    dependencies: tuple[Dependency, ...] = ()
    absence_explained: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    source_kind: str = "notebook"  # notebook | dataset
    fixed_source_file: str | None = None
    source_license: str = "Apache-2.0"
    output_license: str = "unknown"

    @property
    def directory(self) -> Path:
        owner, slug = self.kernel_ref.split("/")
        return EXT / f"{owner}_{slug}"

    @property
    def notebook_file(self) -> str:
        return self.fixed_source_file or self.kernel_ref.split("/")[1] + ".ipynb"

    @property
    def url(self) -> str:
        if self.source_kind == "dataset":
            return f"https://www.kaggle.com/datasets/{self.kernel_ref}/versions/{self.script_version_id}"
        return f"https://www.kaggle.com/code/{self.kernel_ref}?scriptVersionId={self.script_version_id}"


def _absent(note: str = "고정 판본 소스 전체에 해당 코드가 없다") -> Claim:
    return Claim(CONFIRMED, note, (), "published_code_absence")


NB_LICENSE_CAVEAT = (
    LICENSE_CAVEAT,
    "노트북 출력물(소스는 Apache-2.0, 출력 배열은 사용 조건 표시 없음, 결합 입력 전용)",
)

# ---------------------------------------------------------------------------
# 후보 명세: #480 재감사 통과 11개 (판본 2 장부 순서), #479 전수 조사 통과 8개
# ---------------------------------------------------------------------------

ZHUKOV_CLAIMS = {
    "direct_official_training": Claim(
        CONFIRMED,
        "공식 대회 train.csv만 읽고 원자료 추가 없음",
        ("train = pd.read_csv('/kaggle/input/competitions/playground-series-s6e8/train.csv')",),
    ),
    "single_model_lineage": Claim(
        CONFIRMED,
        "train_cv_lattice가 모형별로 별도 OOF·시험 배열을 만들고 구성원별 npy로 저장한다. "
        "quality_blend 결합은 제출 파일에만 쓰이고 저장 구성원 배열에는 섞이지 않는다.",
        ("m = deepcopy(models[name])", "np.save(f'/kaggle/working/oof/oof_lex{VARIANT}_{k}.npy'"),
    ),
    "allowed_aggregation_only": Claim(
        CONFIRMED,
        "seeds=[42] 단일 시드, 시험 예측은 5분할 모형 평균",
        ("splitter = Splitter(n_splits=5, seeds=[42])", "tst[name] += m.predict_proba(X_te)[:, 1] / n_fits"),
    ),
    "target_preprocessing_isolated": Claim(
        CONFIRMED,
        "LatticeEncoder를 바깥 학습 부분에서 fit_transform(안쪽 교차 적합)하고 검증·시험은 transform만 적용. "
        "fit_counts는 train+test 빈도만 세고 목표값을 읽지 않는다.",
        (
            "X_tr = enc.fit_transform(X.iloc[tr_idx], y.iloc[tr_idx])",
            "X_val = enc.transform(X.iloc[va_idx])[X_tr.columns]",
            "skf = StratifiedKFold(self.n_inner, shuffle=True, random_state=self.seed)",
        ),
    ),
    "outer_valid_scoring_only": Claim(
        CONFIRMED,
        "train_cv_lattice는 eval_set 없이 m.fit(X_tr, y_tr)만 호출하고 검증은 predict_proba 채점에만 쓴다",
        ("m.fit(X_tr, y_tr)", "va_pred = m.predict_proba(X_val)[:, 1]"),
    ),
    "public_config_selection_isolated": Claim(
        CONFIRMED,
        "공개 소스는 VARIANT='C'를 정적으로 고르며 바깥 검증 결과로 설정을 고르는 코드가 없다. "
        "CONFIGS A~D 사다리는 정의만 있고 실행 판본은 하나만 계산한다.",
        ("VARIANT = 'C'", "cfg = CONFIGS[VARIANT]"),
    ),
    "no_external_prediction_retraining": Claim(
        CONFIRMED,
        "구성원 저장(셀 79) 뒤의 셀 80이 작성자 데이터셋의 이전 OOF를 읽지만 순위 상관 비교 출력에만 쓰고 학습에 쓰지 않는다",
        ("print('A vs B lgb02:', np.corrcoef(rankdata(A_lgb02), rankdata(res['oof']['lgb02']))[0,1].round(5))",),
    ),
    "no_pseudo_label_training": _absent(),
    "no_blend_distillation": _absent(),
}


def _zhukov_member(name: str, prior: str, auc: float, training: Claim) -> Member:
    return Member(
        name=name,
        member_id=f"nb_zhukov:{name}",
        display_name=f"zhukovoleksiy {name}",
        oof=ArraySpec(f"oof/oof_lexD_{name}.npy", "npy"),
        test=ArraySpec(f"oof/test_lexD_{name}.npy", "npy"),
        semantics="양성 확률(predict_proba[:, 1])",
        prior_pair_sha256=prior,
        independent_auc=auc,
        training_point=training,
        declared=Declared("oof/manifest.csv", "csv_row", f"lexD_{name}", "oof_auc"),
        caveats=(
            NB_LICENSE_CAVEAT,
            (
                "naming_mismatch_lexD",
                "소스는 VARIANT='C'(min_count=50)를 계산하고 저장 셀의 VARIANT='D' 이름으로 파일을 쓴다. "
                "manifest.csv의 encoder 설명(min_count=50 no-pairs)이 C 설정과 일치하므로 계보 불일치는 아니다.",
            ),
        ),
    )


ZHUKOV = Source(
    key="nb_zhukov",
    kernel_ref="zhukovoleksiy/ps6e8-eda-feature-engineering-pipeline",
    script_version_id=344661133,
    pinned_source_sha256="a1e1f488bb4d68de5c4533df8f3816517d5ca65ceb2bb904f9a389360970d9c5",
    author="Oleksii Zhukov (zhukovoleksiy)",
    title="PS6E8 | EDA + Feature Engineering Pipeline",
    population="v2_reaudit",
    prior_ref=REAUDIT_REF,
    fold=Claim(
        CONFIRMED,
        "Splitter(n_splits=5, seeds=[42])가 StratifiedKFold(5, shuffle=True, random_state=42)를 train 원본 순서에 적용하고 manifest.csv에도 같은 분할을 기록한다",
        (
            "splitter = Splitter(n_splits=5, seeds=[42])",
            "skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=seed)",
            "'folds': 'StratifiedKFold(5, shuffle=True, random_state=42)',",
        ),
    ),
    claims=ZHUKOV_CLAIMS,
    absence_explained={
        "/kaggle/input/datasets/zhukovoleksiy/ps6e08-models/oof/oof_lex_lgb02.npy": "저장 뒤 순위 상관 비교 전용(셀 80), 학습 입력 아님"
    },
    members=(
        _zhukov_member(
            "cat_base",
            "ce0efc575d7b2ba918ed0046d29e1ff2ad22cb2c465412a57c217759d54b4b96",
            0.9679922323929437,
            Claim(CONFIRMED, "CatBoost iterations=3000 고정, 조기 종료 없음", ("clf.models['cat_base'] = CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=8,",)),
        ),
        _zhukov_member(
            "lgb02",
            "966f988ba097508f291356c049903b27e31a560b768238e12172621531419197",
            0.9683553568878653,
            Claim(CONFIRMED, "LightGBM n_estimators=3333 고정(Classifier 기본값), 조기 종료 없음", ("def __init__(self, seed=42, n_estimators=3333, device=\"cpu\"):", "lgb_02 = dict(")),
        ),
        _zhukov_member(
            "xgb_base",
            "2303df4128785ac3f417f0cc3258f04d2e3643a9c3735c53798192228a6732a3",
            0.9678590908574395,
            Claim(CONFIRMED, "XGBoost n_estimators=3333 고정(Classifier 기본값), 조기 종료 없음", ("def __init__(self, seed=42, n_estimators=3333, device=\"cpu\"):", "xgb_base = dict(")),
        ),
    ),
)


def _reda_claims(fit_line: str) -> dict[str, Claim]:
    return {
        "direct_official_training": Claim(CONFIRMED, "공식 대회 train.csv만 사용", ("train = pd.read_csv(f'{DATA}/train.csv')",)),
        "single_model_lineage": Claim(CONFIRMED, "단일 모형, MODEL_SEEDS=[0] 하나", ("MODEL_SEEDS = [0]",)),
        "allowed_aggregation_only": Claim(CONFIRMED, "시험 예측은 5분할 모형 평균", ("test_preds += model.predict_proba(A_te)[:, 1] / (FOLDS * len(MODEL_SEEDS))",)),
        "target_preprocessing_isolated": Claim(
            CONFIRMED,
            "sklearn TargetEncoder(cv=5)를 바깥 학습 부분(Xs.iloc[tr], y[tr])에서만 fit_transform하고 검증·시험은 transform",
            ("Z_tr = enc.fit_transform(Xs.iloc[tr], y[tr])", "Z_va = enc.transform(Xs.iloc[va])"),
        ),
        "outer_valid_scoring_only": Claim(CONFIRMED, "fit에 검증 자료를 넘기지 않고 검증은 AUC 출력에만 쓴다", (fit_line,)),
        "public_config_selection_isolated": Claim(CONFIRMED, "CONFIG 사전이 정적으로 고정돼 있고 검증 결과로 고르는 코드가 없다", ("CONFIG = dict(",)),
        "no_external_prediction_retraining": _absent(),
        "no_pseudo_label_training": _absent(),
        "no_blend_distillation": _absent(),
    }


REDA_FOLD = Claim(
    CONFIRMED,
    "FOLDS=5, SEED=42의 StratifiedKFold(shuffle=True)를 train 원본 순서에 적용",
    ("FOLDS = 5", "SEED = 42", "skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)"),
)

REDA_LGBM = Source(
    key="nb_reda_lgbm",
    kernel_ref="redamountassir/s6e8-lgbm-lb-0-96965",
    script_version_id=340546450,
    pinned_source_sha256="c5d72447072a783ce85950c261d61273e1b918872c8f72565471a72431960cbb",
    author="redamountassir",
    title="S6E8 LGBM LB 0.96965",
    population="v2_reaudit",
    prior_ref=REAUDIT_REF,
    fold=REDA_FOLD,
    claims=_reda_claims("model.fit(A_tr, y[tr], categorical_feature=STR_CATS)"),
    members=(
        Member(
            name="lgbm",
            member_id="nb_reda_lgbm:lgbm",
            display_name="redamountassir LGBM",
            oof=ArraySpec("lgbm_oof_preds.csv", "csv", column=TARGET, has_id=True),
            test=ArraySpec("lgbm_test_preds.csv", "csv", column=TARGET, has_id=True),
            semantics="양성 확률(predict_proba[:, 1])",
            prior_pair_sha256="ac29bdd21023f2dc401b4354d3f7900a02d2b1dfdc5aacf0809adc58b21ebe1a",
            independent_auc=0.9682590393147879,
            training_point=Claim(CONFIRMED, "n_estimators=780 고정, eval_set·조기 종료 없음", ("n_estimators=780,",)),
            caveats=(NB_LICENSE_CAVEAT,),
        ),
    ),
)

REDA_HGB = Source(
    key="nb_reda_hgb",
    kernel_ref="redamountassir/s6e8-histgradientboosting-lb-0-96945",
    script_version_id=340546105,
    pinned_source_sha256="33a5fe33bbf21ea3afe9a1e8eea5d8a6cc02054c053e06958f9dcc1c5e1d6110",
    author="redamountassir",
    title="S6E8 HistGradientBoosting LB 0.96945",
    population="v2_reaudit",
    prior_ref=REAUDIT_REF,
    fold=REDA_FOLD,
    claims=_reda_claims("model.fit(A_tr, y[tr])"),
    members=(
        Member(
            name="hgb",
            member_id="nb_reda_hgb:hgb",
            display_name="redamountassir HistGradientBoosting",
            oof=ArraySpec("tehgbc_oof_preds.csv", "csv", column=TARGET, has_id=True),
            test=ArraySpec("tehgbc_test_preds.csv", "csv", column=TARGET, has_id=True),
            semantics="양성 확률(predict_proba[:, 1])",
            prior_pair_sha256="a7a46658f0c4fe1e530881f872610ba2920fd07ac8feecade52df9a4a5fe1739",
            independent_auc=0.9680258266201603,
            training_point=Claim(CONFIRMED, "max_iter=780, early_stopping=False 고정", ("max_iter=780,", "early_stopping=False,")),
            caveats=(NB_LICENSE_CAVEAT,),
        ),
    ),
)

YEKENOT = Source(
    key="nb_yekenot",
    kernel_ref="yekenot/ps-s6-e8-trompt-pytorch-frame",
    script_version_id=344781653,
    pinned_source_sha256="a743e3ad4ac8c9916e94364ce77c8a8933319b31d7b050b928d97f0467c4bc37",
    author="yekenot",
    title="PS S6E8 Trompt (PyTorch Frame)",
    population="v2_reaudit",
    prior_ref=REAUDIT_REF,
    fold=Claim(
        CONFIRMED,
        "CFG.SEED=42, CFG.FOLDS=5의 StratifiedKFold(shuffle=True)",
        ("SEED = 42", "FOLDS = 5", "skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)"),
    ),
    claims={
        "direct_official_training": Claim(CONFIRMED, "공식 대회 train.csv만 사용", ('train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e8/train.csv")',)),
        "single_model_lineage": Claim(CONFIRMED, "Trompt 단일 모형", ("model = Trompt(",)),
        "allowed_aggregation_only": Claim(CONFIRMED, "시험 예측은 5분할 모형 평균", ("test_preds += fold_test_preds / CFG.FOLDS",)),
        "target_preprocessing_isolated": Claim(
            NOT_APPLICABLE,
            "목표값 기반 전처리 없음. 열 통계(col_stats)는 바깥 학습 부분 train_df에서만 계산",
            ("col_stats = {c: compute_col_stats(train_df[c], stype=col_to_stype[c]) for c in train_df.columns if c in col_to_stype}",),
        ),
        "outer_valid_scoring_only": Claim(
            CONFIRMED,
            "검증 AUC는 세대마다 출력만 하고 중단·가중치 선택에 쓰지 않으며 마지막 세대 예측을 OOF로 저장",
            ('print(f"    Epoch {epoch:02d} - train_loss: {avg_loss:.5f} val_auc: {val_auc:.5f}")', "oof_preds[val_idx] = val_preds"),
        ),
        "public_config_selection_isolated": Claim(CONFIRMED, "CFG 클래스가 정적으로 고정", ("class CFG:",)),
        "no_external_prediction_retraining": _absent(),
        "no_pseudo_label_training": _absent(),
        "no_blend_distillation": _absent(),
    },
    members=(
        Member(
            name="trompt",
            member_id="nb_yekenot:trompt",
            display_name="yekenot Trompt",
            oof=ArraySpec("oof_preds.csv", "csv", column=TARGET, has_id=True),
            test=ArraySpec("submission.csv", "csv", column=TARGET, has_id=True),
            semantics="양성 확률(softmax[:, 1], 층 평균)",
            prior_pair_sha256="93411f087c5c7e3a7071fa1d63ca2779455c0ab29709cd0ed8ffc092e0b7eb3c",
            independent_auc=0.9666710825586122,
            training_point=Claim(CONFIRMED, "EPOCHS=8 고정, OneCycleLR 총 단계도 고정", ("EPOCHS = 8", "for epoch in range(1, CFG.EPOCHS + 1):")),
            caveats=(NB_LICENSE_CAVEAT,),
        ),
    ),
)

MOHAN = Source(
    key="nb_mohan_realmlp",
    kernel_ref="mohankrishnathalla/s6e8-realmlp-oof-saver",
    script_version_id=342288210,
    pinned_source_sha256="17e2015025d753537023f1f2f90f8ec13adacad7e36f4d7b2bf5a326d028558a",
    author="mohankrishnathalla",
    title="S6E8 RealMLP OOF saver",
    population="v2_reaudit",
    prior_ref=REAUDIT_REF,
    fold=Claim(CONFIRMED, "StratifiedKFold(5, shuffle=True, random_state=42)", ("skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",)),
    claims={
        "direct_official_training": Claim(CONFIRMED, "공식 대회 train.csv만 사용", ("train = pd.read_csv(DATA_PATH / 'train.csv')",)),
        "single_model_lineage": Claim(CONFIRMED, "RealMLP_TD_Classifier 단일 모형", ("model = RealMLP_TD_Classifier(",)),
        "allowed_aggregation_only": Claim(CONFIRMED, "시험 예측은 5분할 모형 평균", ("test_realmlp += model.predict_proba(X_te_f)[:, 1] / 5",)),
        "target_preprocessing_isolated": Claim(
            CONFIRMED,
            "add_fold_te가 바깥 학습 부분 목표값(y_tr)으로만 대응표를 만들고 결측 대치 중앙값도 학습 부분에서 계산. "
            "LabelEncoder는 train+test 범주값만 보고 목표값을 읽지 않는다.",
            ("X_tr_raw, X_va_raw, X_te_raw = add_fold_te(X_tr_raw, y_tr, X_va_raw, X_te_raw)", "medians = X_tr_raw[FEATURES_FINAL].median()"),
        ),
        "outer_valid_scoring_only": Claim(
            CONFIRMED,
            "fit에 바깥 검증 자료를 넘기지 않는다. RealMLP 내부 검증 분할은 바깥 학습 부분 안에서만 자동 구성된다.",
            ("model.fit(X_tr_f, y_tr_s)",),
        ),
        "public_config_selection_isolated": Claim(CONFIRMED, "설정이 정적으로 고정", ("random_state=42,", "batch_size=256,")),
        "no_external_prediction_retraining": _absent(),
        "no_pseudo_label_training": _absent(),
        "no_blend_distillation": _absent(),
    },
    members=(
        Member(
            name="realmlp",
            member_id="nb_mohan_realmlp:realmlp",
            display_name="mohankrishnathalla RealMLP",
            oof=ArraySpec("oof_realmlp.npy", "npy"),
            test=ArraySpec("test_realmlp.npy", "npy"),
            semantics="양성 확률(predict_proba[:, 1])",
            prior_pair_sha256="5245ad2a78dcb5982153ccbfde5a14fd53eaf6c0ff98faac3ffcae294e7f625d",
            independent_auc=0.9581337349072739,
            training_point=Claim(CONFIRMED, "n_epochs=512 고정", ("n_epochs=512,",)),
            caveats=(NB_LICENSE_CAVEAT,),
        ),
    ),
)

LOPURE_CLAIMS = {
    "direct_official_training": Claim(CONFIRMED, "공식 대회 train.csv만 사용", ('train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e8/train.csv")',)),
    "single_model_lineage": Claim(CONFIRMED, "svm_variants의 모형 하나씩 별도 OOF·시험 파일로 저장", ("for model_name, classifier in svm_variants.items():", 'oof_filename = f"oof_{model_name.lower()}_svm_gpu.csv"')),
    "allowed_aggregation_only": Claim(CONFIRMED, "시험 예측은 5분할 모형 평균", ("test_preds += test_fold_preds / skf.n_splits",)),
    "target_preprocessing_isolated": Claim(
        NOT_APPLICABLE,
        "목표값 기반 전처리 없음. 대치·표준화·원핫은 바깥 학습 부분에서 fit_transform, 검증·시험은 transform",
        ("X_tr_processed = np.ascontiguousarray(preprocessor.fit_transform(X_tr).astype(np.float32))", "X_va_processed = np.ascontiguousarray(preprocessor.transform(X_va).astype(np.float32))"),
    ),
    "outer_valid_scoring_only": Claim(CONFIRMED, "fit은 학습 부분만 받고 검증은 decision_function 채점에만 쓴다", ("classifier.fit(X_tr_processed, y_tr_gpu)", "val_fold_preds = classifier.decision_function(X_va_processed)")),
    "public_config_selection_isolated": Claim(CONFIRMED, "svm_variants 설정이 정적으로 고정", ("svm_variants = {",)),
    "no_external_prediction_retraining": _absent(),
    "no_pseudo_label_training": _absent(),
    "no_blend_distillation": _absent(),
}


def _lopure_member(name: str, column: str, prior: str, auc: float, config_line: str) -> Member:
    return Member(
        name=f"{name}_svm",
        member_id=f"nb_lopure:{name}_svm",
        display_name=f"lopure {name} SVM",
        oof=ArraySpec(f"oof_{name}_svm_gpu.csv", "csv", column=column, has_id=True),
        test=ArraySpec(f"submission_{name}_svm_gpu.csv", "csv", column=TARGET, has_id=True),
        semantics="decision_function 점수(확률 아님, 순위 결합기 입력)",
        prior_pair_sha256=prior,
        independent_auc=auc,
        training_point=Claim(CONFIRMED, "cuML SVM 설정 고정, 검증 기반 중단 없음", (config_line,)),
        caveats=(NB_LICENSE_CAVEAT, ("decision_function_scores", "예측값이 확률이 아닌 SVM 결정 함수 점수다")),
    )


LOPURE = Source(
    key="nb_lopure",
    kernel_ref="lopure/hdviz-pca-parallel-with-linear-svm",
    script_version_id=342359513,
    pinned_source_sha256="a6cccd0afd1c8d46d88367d0d50d8386768be99af1bd9953800f7c0c79504b96",
    author="lopure",
    title="HDViz PCA Parallel with Linear SVM",
    population="v2_reaudit",
    prior_ref=REAUDIT_REF,
    fold=Claim(CONFIRMED, "StratifiedKFold(5, shuffle=True, random_state=42)", ("skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",)),
    claims=LOPURE_CLAIMS,
    members=(
        _lopure_member("linear", "Linear_oof_pred", "2900eb8774c78852b7c38d442027f0bbbce8a1e3b949a0ab394bcaa59295adcb", 0.9113453805990479, "'Linear': LinearSVC(C=1.0),"),
        _lopure_member("poly", "Poly_oof_pred", "953e2da3c591335868bf5d8d665307a0b578c7e153ae6ad448dd8d360711c350", 0.9287951221261161, "'Poly': SVC(kernel='poly', degree=3, C=1.0, cache_size=2000),"),
        _lopure_member("rbf", "RBF_oof_pred", "557a0da96cccaa7d336e1b75fbec0bb9049e6e8227924903ae421c13baaa24b2", 0.9221668586742819, "'RBF': SVC(kernel='rbf', C=1.0, cache_size=2000)"),
    ),
)

SHAMAN = Source(
    key="nb_shaman_baseline",
    kernel_ref="shamanthakreddymallu/s6e8-baseline",
    script_version_id=345105403,
    pinned_source_sha256="1647535be4fafd0c8bea4b32e7968fe91bfe3e5a6efd3e147e84e68d07a35aa7",
    author="shamanthakreddymallu",
    title="S6E8 baseline",
    population="v2_reaudit",
    prior_ref=REAUDIT_REF,
    fold=Claim(
        CONFIRMED,
        "SEED=42, N_SPLITS=5의 StratifiedKFold(shuffle=True). 판본 2의 fold_evidence_none을 공개 코드로 승격",
        ("SEED    = 42", "N_SPLITS = 5", "skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)"),
    ),
    claims={
        "direct_official_training": Claim(CONFIRMED, "공식 대회 train.csv만 사용", ('train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))',)),
        "single_model_lineage": Claim(CONFIRMED, "spline 로지스틱 회귀 파이프라인 단일 모형. 순위 평균 결합은 출력 비교에만 쓰이고 저장 배열에 섞이지 않는다", ('LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")', 'np.save(os.path.join(OUT_DIR, "oof_lr.npy"), oof_lr)')),
        "allowed_aggregation_only": Claim(CONFIRMED, "시험 예측은 5분할 모형 평균", ("pred_lr += lr_pipe.predict_proba(Xte)[:, 1] / N_SPLITS",)),
        "target_preprocessing_isolated": Claim(
            NOT_APPLICABLE,
            "목표값 기반 전처리 없음. 대치·표준화·스플라인·원핫은 파이프라인 안에서 학습 부분에만 적합. "
            "GaussianMixture 특성은 train+test 특성값만으로 적합하고 목표값을 읽지 않는다(비지도).",
            ('("num", make_pipeline(SimpleImputer(strategy="median"),', 'gm = GaussianMixture(n_components=2, random_state=SEED, covariance_type="full", max_iter=200)'),
        ),
        "outer_valid_scoring_only": Claim(CONFIRMED, "fit은 학습 부분(고정 부분 표본과 교집합)만 받고 검증은 predict_proba 채점에만 쓴다", ("lr_pipe.fit(Xtr.iloc[tr_i], y[tr_i])", "oof_lr[va_i] = lr_pipe.predict_proba(Xtr.iloc[va_i])[:, 1]")),
        "public_config_selection_isolated": Claim(CONFIRMED, "로지스틱 회귀 설정이 정적으로 고정되고 뒤따르는 특성 절제 실험은 저장된 LR 배열 뒤에서 별도로 실행된다", ('LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")',)),
        "no_external_prediction_retraining": _absent(),
        "no_pseudo_label_training": _absent(),
        "no_blend_distillation": _absent(),
    },
    members=(
        Member(
            name="lr",
            member_id="nb_shaman_baseline:lr",
            display_name="shamanthakreddymallu spline logistic regression",
            oof=ArraySpec("oof_lr.npy", "npy"),
            test=ArraySpec("pred_lr.npy", "npy"),
            semantics="양성 확률(predict_proba[:, 1])",
            prior_pair_sha256="e439bd0d7c03aad70bc027fcf66e5fb9036b0f826d13f127ea205a3fd6cb15dc",
            independent_auc=0.9366091741556544,
            training_point=Claim(CONFIRMED, "lbfgs max_iter=2000 고정, 검증 기반 중단 없음", ('LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")',)),
            caveats=(
                NB_LICENSE_CAVEAT,
                ("fixed_subsample_training", "각 바깥 학습 부분을 고정 난수 42의 250,000행 표본과 교집합해 학습한다(목표값 미사용)"),
            ),
            lineage_snippets=("tr_i = np.intersect1d(tr_i, samp)",),
        ),
    ),
)

BEICICC_TABNET = Source(
    key="beicicc_tabnet",
    kernel_ref="beicicc/s6e8-fold-safe-tabnet",
    script_version_id=339872430,
    pinned_source_sha256="3f97a7351a649a6a583edf3492fe4db190d926ef94778c74db3bba41c8abfeb7",
    author="Kun Zhang (beicicc)",
    title="S6E8 Fold-Safe TabNet",
    population="census",
    prior_ref=CENSUS_REF,
    fold=Claim(
        CONFIRMED,
        "SEED=42, N_SPLITS=5의 바깥 StratifiedKFold. 공개 실행 manifest도 outer_folds {5, shuffle, 42}를 기록",
        ("SEED = 42", "N_SPLITS = 5", "outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)"),
    ),
    claims={
        "direct_official_training": Claim(CONFIRMED, "공식 대회 train.csv만 읽고 다른 자료 입력이 없다(kernel-metadata.json의 dataset_sources 비어 있음)", ('train = pd.read_csv(COMPETITION_DIR / "train.csv")',), extra_files=(("kernel-metadata.json", '"dataset_sources": []'),)),
        "single_model_lineage": Claim(CONFIRMED, "TabNetClassifier 단일 모형", ("model = TabNetClassifier(",)),
        "allowed_aggregation_only": Claim(CONFIRMED, "시험 예측은 5분할 모형 평균", ("test_prediction = model.predict_proba(test_array)[:, 1].astype(np.float32)",)),
        "target_preprocessing_isolated": Claim(
            CONFIRMED,
            "목표 부호화는 바깥 학습 부분 안의 내부 5분할 교차 적합이고 검증·시험은 바깥 학습 전체 대응표를 적용",
            ("inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=inner_seed)", '"outer_valid_and_test": "complete_outer_train_mapping",'),
        ),
        "outer_valid_scoring_only": Claim(CONFIRMED, "fit은 eval_set=[], patience=0으로 실행되고 검증은 채점 전용", ("eval_set=[],", "patience=0,", '"outer_valid_role": "scoring_only",')),
        "public_config_selection_isolated": Claim(CONFIRMED, "TABNET_PARAMS와 FIXED_EPOCHS가 정적으로 고정", ("FIXED_EPOCHS = 35",)),
        "no_external_prediction_retraining": _absent(),
        "no_pseudo_label_training": _absent(),
        "no_blend_distillation": _absent(),
    },
    members=(
        Member(
            name="tabnet",
            member_id="beicicc/s6e8-fold-safe-tabnet:tabnet",
            display_name="beicicc Fold-Safe TabNet",
            oof=ArraySpec("tabnet_fold_safe_oof.csv", "csv", column="oof_pred", has_id=True, label_column=TARGET, dtype_note="float32 저장"),
            test=ArraySpec("tabnet_fold_safe_test.csv", "csv", column="tabnet_fold_safe", has_id=True, dtype_note="float32 저장"),
            semantics="양성 확률(predict_proba[:, 1])",
            prior_pair_sha256="b339d0b025bc3989e2e87c0c092b1e11d3ceb7df9ca792bfd9e4b9b645535722",
            independent_auc=0.965656810,
            training_point=Claim(CONFIRMED, "max_epochs=FIXED_EPOCHS(35) 고정, 학습 뒤 세대 수 단언", ("max_epochs=FIXED_EPOCHS,", 'assert len(model.history["loss"]) == FIXED_EPOCHS')),
            declared=Declared("tabnet_fold_safe_manifest.json", "json_key", "overall_oof_auc"),
            caveats=(NB_LICENSE_CAVEAT, ("float32_storage", "출력 CSV가 float32 정밀도로 저장됨")),
        ),
    ),
)

BEICICC_REALMLP = Source(
    key="beicicc_realmlp",
    kernel_ref="beicicc/s6e8-fold-safe-realmlp",
    script_version_id=339864149,
    pinned_source_sha256="60a0bd05332e8932468d9cc796855013be3c3798344fd75c15c016764eba58ef",
    author="Kun Zhang (beicicc)",
    title="S6E8 Fold-Safe RealMLP",
    population="census",
    prior_ref=CENSUS_REF,
    fold=Claim(
        CONFIRMED,
        "SEED=42, OUTER_SPLITS=5의 바깥 StratifiedKFold. 공개 실행 manifest도 같은 분할을 기록",
        ("SEED = 42", "OUTER_SPLITS = 5", "outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)"),
    ),
    claims={
        "direct_official_training": Claim(CONFIRMED, "공식 대회 train.csv만 읽고 다른 자료 입력이 없다(kernel-metadata.json의 dataset_sources 비어 있음)", ('train = pd.read_csv(COMPETITION_DIR / "train.csv")',), extra_files=(("kernel-metadata.json", '"dataset_sources": []'),)),
        "single_model_lineage": Claim(CONFIRMED, "FoldSafeRealMLPClassifier 단일 모형", ("model = FoldSafeRealMLPClassifier(**CONFIG)",)),
        "allowed_aggregation_only": Claim(CONFIRMED, "시험 예측은 5분할 모형 평균", ("test_prediction += test_fold_probability / OUTER_SPLITS",)),
        "target_preprocessing_isolated": Claim(
            CONFIRMED,
            "sklearn TargetEncoder를 바깥 학습 부분 안에서 교차 적합하고 검증·시험은 transform. 구간화 전처리도 학습 부분에서 적합",
            ("encoder = TargetEncoder(", "valid_values = encoder.transform(outer_valid[target_columns])"),
        ),
        "outer_valid_scoring_only": Claim(
            CONFIRMED,
            "fit은 학습 부분(X_train_fold, y[train_idx])만 받고 검증은 predict_proba 채점에만 쓴다. EMA 가중치는 학습 손실 경로의 지수 평균이며 검증을 보지 않는다",
            ("model = FoldSafeRealMLPClassifier(**CONFIG)", "valid_probability = model.predict_proba(X_valid_fold)[:, 1].astype(np.float32)", "self.model_.load_state_dict(ema_state, strict=True)"),
        ),
        "public_config_selection_isolated": Claim(CONFIRMED, "CONFIG와 FIXED_EPOCHS가 정적으로 고정", ("FIXED_EPOCHS = 4", "CONFIG = {")),
        "no_external_prediction_retraining": _absent(),
        "no_pseudo_label_training": _absent(),
        "no_blend_distillation": _absent(),
    },
    members=(
        Member(
            name="realmlp",
            member_id="beicicc/s6e8-fold-safe-realmlp:realmlp",
            display_name="beicicc Fold-Safe RealMLP",
            oof=ArraySpec("realmlp_fold_safe_oof.csv", "csv", column="oof_pred", has_id=True, label_column=TARGET, dtype_note="float32 저장"),
            test=ArraySpec("realmlp_fold_safe_test.csv", "csv", column="realmlp_fold_safe", has_id=True, dtype_note="float32 저장"),
            semantics="양성 확률(predict_proba[:, 1])",
            prior_pair_sha256="e21c22c3b2416598bd2bdc198cbbbbb2e8cdedd14f3434daa751282b97784665",
            independent_auc=0.968156387,
            training_point=Claim(CONFIRMED, "epochs=FIXED_EPOCHS(4) 고정, 학습 뒤 세대 수 단언", ("assert epochs == FIXED_EPOCHS", "assert self.trained_epochs_ == FIXED_EPOCHS")),
            declared=Declared("realmlp_fold_safe_manifest.json", "json_key", "overall_oof_auc"),
            caveats=(
                NB_LICENSE_CAVEAT,
                ("float32_storage", "출력 CSV가 float32 정밀도로 저장됨"),
                ("near_duplicate_cluster", "판본 2 장부 realmlp_seed01_fixed4와 스피어만 0.999097(#479 측정), 자격에는 영향 없음"),
            ),
        ),
    ),
)


def _busy_member(name: str, prior: str, auc: float, trees: str) -> Member:
    return Member(
        name=name,
        member_id=f"busyaprime/s6e8-tabular-baseline-that-autodetects-the-task:{name}",
        display_name=f"busyaprime {name}",
        oof=ArraySpec(f"oof_{name}.npy", "npy", column_index=1),
        test=ArraySpec(f"test_{name}.npy", "npy", column_index=1),
        semantics="양성 확률(predict_proba 2열 행렬의 1열)",
        prior_pair_sha256=prior,
        independent_auc=auc,
        training_point=Claim(CONFIRMED, f"FAST=False로 {trees} 고정, eval_set·조기 종료 없음", ("FAST = False", 'N_TREES = {"lgb": 200 if FAST else 600, "hgb": 150 if FAST else 400, "xgb": 150 if FAST else 450}')),
        caveats=(NB_LICENSE_CAVEAT,) + ((("near_duplicate_cluster", "판본 2 장부 raw12와 스피어만 0.998125(#479 측정), 자격에는 영향 없음"),) if name == "lgb" else ()),
    )


BUSYAPRIME = Source(
    key="busyaprime",
    kernel_ref="busyaprime/s6e8-tabular-baseline-that-autodetects-the-task",
    script_version_id=339485089,
    pinned_source_sha256="52c509d1b726d2ddeaddd0e07ada8c345a21483c09f0128e0070359552025235",
    author="busyaprime",
    title="S6E8 tabular baseline that autodetects the task",
    population="census",
    prior_ref=CENSUS_REF,
    fold=Claim(CONFIRMED, "SEED=42, N_FOLDS=5의 StratifiedKFold(shuffle=True)", ("SEED = 42", "N_FOLDS = 5", "folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xtr, y))")),
    claims={
        "direct_official_training": Claim(CONFIRMED, "/kaggle/input에서 찾은 대회 train.csv만 읽고 다른 자료 입력이 없다(kernel-metadata.json의 dataset_sources 비어 있음)", ('train = pd.read_csv(find("train.csv")); test = pd.read_csv(find("test.csv"))',), extra_files=(("kernel-metadata.json", '"dataset_sources": []'),)),
        "single_model_lineage": Claim(CONFIRMED, "oof_of가 모형 종류별로 별도 배열을 만들어 npy로 저장", ('for n in names: np.save(f"oof_{n}.npy", oof[n]); np.save(f"test_{n}.npy", tst[n])',)),
        "allowed_aggregation_only": Claim(CONFIRMED, "시험 예측은 5분할 모형 평균(분할별 random_state=f는 같은 설정의 시드 변화)", ("tst[:, cc] += m.predict_proba(Xte) / N_FOLDS",)),
        "target_preprocessing_isolated": Claim(NOT_APPLICABLE, "목표값 기반 전처리 없음. 범주 정수화는 train·test 범주 합집합만 쓰고 목표값을 읽지 않는다", ("mm = {v: i for i, v in enumerate(u.unique())}",)),
        "outer_valid_scoring_only": Claim(CONFIRMED, "fit은 학습 부분만 받고 검증은 predict_proba 채점에만 쓴다", ('if kind == "lgb": m.fit(Xtr.iloc[tri], y[tri], categorical_feature=CATCOLS)', "else: m.fit(Xtr.iloc[tri], y[tri])")),
        "public_config_selection_isolated": Claim(CONFIRMED, "모형 설정이 정적으로 고정", ("return C(n_estimators=nt, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8,",)),
        "no_external_prediction_retraining": _absent(),
        "no_pseudo_label_training": _absent(),
        "no_blend_distillation": _absent(),
    },
    members=(
        _busy_member("lgb", "ff58548f9868bdd4a5dd3fe330060b39ad21f18f232a9a776f7a7ecdf20e618f", 0.962557588, "LightGBM 600그루"),
        _busy_member("hgb", "b7b0afba77e4c3352a3c03b555c5c68fdf5fd9d6c234e4b5b00a402a5f02564a", 0.962048339, "HistGradientBoosting 400회"),
        _busy_member("xgb", "3f683ae1e737a53a2c220103b6c31375685f705030998628ecf2090c7e3d8351", 0.962314187, "XGBoost 450그루"),
    ),
)

RAVI_IMPORTS = Dependency(
    kernel_ref="ravi20076/playgrounds6e8-public-imports-v1",
    script_version_id=339439580,
    role="보조 코드(training.py의 ModelTrainer, myutils.py, mypp.py)",
    files={
        "training.py": "26504708be69444a8df97ac7b0ecc95e788340b88b5cfaef823dfe2c9d8a1405",
        # 아래 둘은 #479가 고정하지 않아 이번 감사(2026-08-28)에서 확보한 사본으로 고정한다.
        "mypp.py": "f5ab8352cc2c8b3233f07a0c35a833009466a5eb53739ec440c477f47c3fa98b",
        "myutils.py": "2a98eef32eca0a035473609fe7f37531f6c463dfed411dd62351a7f5216150fa",
    },
)
RAVI_DATA_CAVEAT = (
    "rehosted_training_data_private_notebook",
    "train.csv·test.csv·sample_submission.csv를 작성자의 비공개 자료 노트북 playgrounds6e8-data-v1에서 읽는다(mypp.py). "
    "그 노트북은 공개되지 않아 바이트를 대조할 수 없고, 공개 실행 로그의 형태 (691369, 14)·(296302, 13)(Source 열 포함)와 "
    "OOF의 id 순서·재채점 AUC 일치로만 공식 자료임을 확인했다. 원자료 추가는 nb_orig=0으로 꺼져 있다.",
)


def _ravi_member(name: str, prior: str, auc: float, trees_line: str, trees: str) -> Member:
    return Member(
        name=name,
        member_id=f"ravi20076/playgrounds6e8-public-baseline-v1:{name}",
        display_name=f"ravi20076 baseline v1 {name}",
        oof=ArraySpec("OOF_Preds_MLV1_1.parquet", "parquet", column=name, dtype_note="float32 저장"),
        test=ArraySpec("Mdl_Preds_MLV1_1.parquet", "parquet", column=name, dtype_note="float32 저장"),
        semantics="양성 확률(predict_proba[:, 1])",
        prior_pair_sha256=prior,
        independent_auc=auc,
        training_point=Claim(CONFIRMED, f"{trees} 고정, early_stopping_rounds 없음", (trees_line,)),
        caveats=(NB_LICENSE_CAVEAT, ("float32_storage", "출력 parquet가 float32 정밀도로 저장됨"), RAVI_DATA_CAVEAT),
    )


RAVI = Source(
    key="ravi_v1",
    kernel_ref="ravi20076/playgrounds6e8-public-baseline-v1",
    script_version_id=339444387,
    pinned_source_sha256="186d26a1aba7549fd182ed89322daff43f6083d8f9275175215c7c0207d31c30",
    author="Ravi Ramakrishnan (ravi20076)",
    title="PlaygroundS6E8 public baseline v1",
    population="census",
    prior_ref=CENSUS_REF,
    fold=Claim(
        CONFIRMED,
        "CFG.state=42, n_splits=5, mdlcv_mthd='SKF'의 StratifiedKFold(shuffle=True)를 ygrp로 고정하고 PredefinedSplit으로 재사용",
        ("state              = 42", "n_splits           = 5", "mdlcv_mthd         = 'SKF'", '"SKF"   : StratifiedKFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),', "ygrp[dev_idx] = fold_nb"),
    ),
    claims={
        "direct_official_training": Claim(
            CONFIRMED,
            "nb_orig=0으로 원자료 추가를 끄고, mypp.py가 작성자 자료 노트북 경로의 train.csv·test.csv를 읽으며 공개 로그의 형태가 대회 자료와 같다(주의 사항 rehosted_training_data_private_notebook)",
            ("nb_orig            = 0", 'ip_path            = f"/kaggle/input/notebooks/ravi20076/playgrounds6e8-data-v1"'),
            "published_code+published_log",
            extra_files=(
                ("ravi20076_playgrounds6e8-public-imports-v1/mypp.py", 'self.train             = pd.read_csv(os.path.join(CFG.ip_path,"train.csv"), index_col = \'id\')'),
                ("playgrounds6e8-public-baseline-v1.log", "Shapes = (691369, 14) (296302, 13)"),
            ),
        ),
        "single_model_lineage": Claim(CONFIRMED, "Mdl_Master의 파이프라인 하나씩 별도 열로 저장. L2 스택 정의는 저장 열에 섞이지 않는다", ('pd.DataFrame(OOF_Preds).to_parquet(f"OOF_Preds_{CFG.model_label}{CFG.model_id}.parquet")',)),
        "allowed_aggregation_only": Claim(CONFIRMED, "시험 예측은 분할별 예측의 평균(groupby(level=0).mean())", (").groupby(level = 0).mean().to_numpy()",), "published_code", "ravi20076_playgrounds6e8-public-imports-v1/training.py"),
        "target_preprocessing_isolated": Claim(CONFIRMED, "TargetEncoder가 sklearn Pipeline 단계라 각 바깥 학습 부분에서만 적합(교차 적합)", ('[("TE", TargetEncoder(random_state = CFG.state), cat_cols)],', '("PP", ct),')),
        "outer_valid_scoring_only": Claim(
            CONFIRMED,
            "ModelTrainer가 Pipeline.fit에 최상위 eval_set을 넘기면 sklearn이 단계 접두사 없는 인자를 거부해 예외가 나고, except 경로에서 검증 자료 없이 model.fit(Xtr, ytr)를 다시 호출한다. 검증은 채점 전용",
            ("model.fit(Xtr, ytr, eval_set = [(Xdev, ydev)], **fit_params)", "except:", "model.fit(Xtr, ytr, **fit_params)"),
            "published_code_inference",
            "ravi20076_playgrounds6e8-public-imports-v1/training.py",
        ),
        "public_config_selection_isolated": Claim(CONFIRMED, "CFG와 Mdl_Master 설정이 정적으로 고정", ("class CFG:",)),
        "no_external_prediction_retraining": _absent("고정 판본 소스에 외부 예측 입력이 없다. /kaggle/input/notebooks 경로는 보조 코드 exec 전용"),
        "no_pseudo_label_training": _absent(),
        "no_blend_distillation": _absent(),
    },
    dependencies=(RAVI_IMPORTS,),
    absence_explained={
        "public-imports-v1": "보조 코드(pip 요구사항, myutils.py, training.py, mypp.py exec), 예측 입력 아님",
        "playgrounds6e8-data-v1": "작성자 자료 노트북의 train/test/sample_submission 경로, 예측 입력 아님(주의 사항 rehosted_training_data_private_notebook)",
    },
    members=(
        _ravi_member("XGB1C", "d795573efce0daf7fa1f87e82bd0843f1e12960bc621e033e9d93c207be822ab", 0.964201482, "'n_estimators'          : 3000 if CFG.test_req == False else CFG.test_iter,", "XGBoost 3,000그루"),
        _ravi_member("LGBM1C", "15ea60831189c09204e17cdefbaa8e262cee346fab45ce9f7f97e32870446b66", 0.964173099, "'n_estimators'          : 2500 if CFG.test_req == False else CFG.test_iter,", "LightGBM 2,500그루"),
        _ravi_member("CB1C", "f3e04b96a6bb416cab11bf092570657e9bc6d74c7446ab8bf7f97815f17e80a0", 0.963944102, "'iterations'            : 3000 if CFG.test_req == False else CFG.test_iter,", "CatBoost 3,000회"),
    ),
)


def _sometime_member(name: str, display_name: str, prior: str, auc: float, training: Claim, semantics: str = "양성 확률") -> Member:
    return Member(
        name=name,
        member_id=f"sometimessubodh/stacking-9-models-smartphone-addiction-prediction:{name}",
        display_name=f"Subodh Deogade {display_name}",
        oof=ArraySpec("stacking_matrices.pkl", "joblib_dataframe", column=name, frame_key="oof_preds"),
        test=ArraySpec("stacking_matrices.pkl", "joblib_dataframe", column=name, frame_key="test_preds"),
        semantics=semantics,
        prior_pair_sha256=prior,
        independent_auc=auc,
        training_point=training,
        caveats=(
            NB_LICENSE_CAVEAT,
            (
                "full_feature_only_preprocessing",
                "범주 사전은 train+test 특성값, 중앙값과 표준화는 전체 train 특성값으로 만들지만 목표값은 읽지 않는다.",
            ),
        ),
    )


SOMETIME_STACKING = Source(
    key="sometime_stacking9",
    kernel_ref="sometimessubodh/stacking-9-models-smartphone-addiction-prediction",
    script_version_id=346039237,
    pinned_source_sha256="691100dcf6f0b365e4c1a5902e52218797cfe00c73dca19b8e6a2b19087473bb",
    author="Subodh Deogade (sometimessubodh)",
    title="Stacking 9 Models|Smartphone Addiction Prediction",
    population="incremental_issue_487",
    prior_ref=INCREMENTAL_REF,
    fold=Claim(
        CONFIRMED,
        "N_FOLDS=5, random_state=42의 StratifiedKFold(shuffle=True)를 원본 train 순서에 적용하고 모든 구성원이 같은 fold_indices를 재사용한다.",
        ("N_FOLDS = 5", "skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)", "fold_indices = list(skf.split(X, y))"),
    ),
    claims={
        "direct_official_training": Claim(
            CONFIRMED,
            "공식 대회 train.csv와 test.csv만 읽고 연결 자료가 없다.",
            ('TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e8/train.csv"',),
            extra_files=(("kernel-metadata.json", '"dataset_sources": []'),),
        ),
        "single_model_lineage": Claim(
            CONFIRMED,
            "stacking_matrices.pkl이 기초 모형별 OOF·시험 열을 따로 저장하며 최종 메타 결합 예측은 별도 제출 CSV에만 쓴다.",
            ('joblib.dump({"oof_preds": oof_preds, "test_preds": test_preds, "y": y}, "stacking_matrices.pkl")',),
        ),
        "allowed_aggregation_only": Claim(
            CONFIRMED,
            "각 시험 열은 같은 고정 설정의 5개 바깥 분할 모형 예측 평균이다.",
            ('test_preds["cuml_rf"] += model.predict_proba(X_test_num.values)[:, 1] / N_FOLDS',),
        ),
        "target_preprocessing_isolated": Claim(
            NOT_APPLICABLE,
            "목표값 기반 전처리가 없다. 범주 사전은 train+test 특성값만, 중앙값과 표준화는 train 특성값만 보고 목표값을 읽지 않는다.",
            ("le.fit(combined)", "num_medians = X_num.median()", "scaler.fit_transform(X_num)"),
        ),
        "outer_valid_scoring_only": Claim(
            CONFIRMED,
            "다섯 구성원은 fit에 바깥 학습 부분만 넘긴다. PyTorch 함수의 X_val은 학습 뒤 예측에만 쓰고 y_val은 읽지 않는다.",
            ("model.fit(X_num.iloc[tr_idx].values, y.iloc[tr_idx].values.astype(np.int32))", "for _ in range(epochs):", "val_probs = torch.sigmoid(val_logits).cpu().numpy()"),
        ),
        "public_config_selection_isolated": Claim(
            CONFIRMED,
            "다섯 구성원의 설정이 코드에 정적으로 고정되고 OOF 점수는 출력과 최종 메타 결합에만 쓰인다.",
            ("MODEL_NAMES = [", "base_scores = {name: roc_auc_score(y, oof_preds[name]) for name in MODEL_NAMES}"),
        ),
        "no_external_prediction_retraining": _absent(),
        "no_pseudo_label_training": _absent(),
        "no_blend_distillation": _absent(),
    },
    members=(
        _sometime_member(
            "cuml_rf",
            "cuML RandomForest",
            "b3e6a5c21a006b6f730170e55a3a37a3f8795b62734d7f12679b4919f9bc8ed4",
            0.940504101712,
            Claim(CONFIRMED, "n_estimators=400, max_depth=12 고정, 검증 기반 중단 없음", ("model = cuRF(n_estimators=400, max_depth=12, random_state=42 + fold)",)),
        ),
        _sometime_member(
            "cuml_logreg",
            "cuML LogisticRegression",
            "949716ab6998c5f084a3ea9318a0f103b35c48d27f22bfdfb41cae070e2153a2",
            0.927826905357,
            Claim(CONFIRMED, "max_iter=1000 고정, 검증 기반 중단 없음", ("model = cuLogReg(max_iter=1000)",)),
        ),
        _sometime_member(
            "cuml_knn",
            "cuML KNN",
            "6cc0389e39fd8ec6658322cf524d37d6a0b77bf5a86de587be074f77a841a90b",
            0.929086165555,
            Claim(CONFIRMED, "n_neighbors=25 고정, 검증 기반 중단 없음", ("model = cuKNN(n_neighbors=25)",)),
        ),
        _sometime_member(
            "cuml_mbsgd",
            "cuML MBSGD",
            "57c44070baf52a346fe78c881f04d47952ed9e1bf29f5736e1f754d01caa85c6",
            0.834173901922,
            Claim(CONFIRMED, "epochs=50, alpha=1e-4 고정, 검증 기반 중단 없음", ('model = cuMBSGD(loss="log", penalty="l2", alpha=1e-4, epochs=50)',)),
            "0·1 분류 예측(model.predict, 순위 결합기 입력)",
        ),
        _sometime_member(
            "torch_mlp",
            "PyTorch MLP",
            "8f3d5ea1e94adbeb6162ec49995803c9f8d88b9d30325dbd343c68374842ca8f",
            0.940873573614,
            Claim(CONFIRMED, "고정 구조와 epochs=30, Adam lr=1e-3으로 끝까지 학습하고 검증 기반 상태 선택 없음", ("def train_mlp_fold(X_tr, y_tr, X_val, y_val, n_features, epochs=30, batch_size=1024):", "opt = torch.optim.Adam(model.parameters(), lr=1e-3)")),
        ),
    ),
    notes=("2026-08-30T12:00:00Z 증분 조사에서 발견한 저장 출력이다.",),
)


MICHAEL_DEPTH9 = Source(
    key="michael_depth9_pair_te",
    kernel_ref="michaelqiu0606/s6e8-depth9-pair-te-inputs",
    script_version_id=1,
    pinned_source_sha256="ec1ff5cf164c212335cfa0c748ada21c4ed4f473315524e9e5bccc371e195ee9",
    author="Michael Y. Qiu (michaelqiu0606)",
    title="S6E8 Depth9 Pair TE Inputs",
    population="incremental_issue_487",
    prior_ref=INCREMENTAL_REF,
    fold=Claim(
        UNKNOWN,
        "README는 strict outer-fold라고 서술하지만 커뮤니티 고정 5분할과 원본 행 순서를 재현할 분할 벡터나 소스가 없다.",
        ("strict outer-fold", "aligned to the official competition train/test row order"),
    ),
    claims={
        "direct_official_training": Claim(UNKNOWN, "공식 자료 직접 학습이라는 설명만 있고 확인할 소스가 없다.", ("official competition train/test row order",)),
        "single_model_lineage": Claim(UNKNOWN, "이름은 depth9 champion 하나를 가리키지만 모형 계보를 확인할 소스가 없다.", ("depth9_m012_champion",)),
        "allowed_aggregation_only": Claim(UNKNOWN, "분할·시드 집계 방식을 확인할 소스가 없다."),
        "target_preprocessing_isolated": Claim(UNKNOWN, "pair TE의 목표값 격리 범위를 확인할 소스가 없다."),
        "outer_valid_scoring_only": Claim(UNKNOWN, "바깥 검증 목표값이 학습 시점이나 상태 선택에 닿지 않았는지 확인할 소스가 없다."),
        "public_config_selection_isolated": Claim(UNKNOWN, "README의 local CV selected 서술만으로 설정 선택 격리를 확인할 수 없다.", ("selected from local CV only",)),
        "no_external_prediction_retraining": Claim(UNKNOWN, "입력 계보를 확인할 소스가 없다."),
        "no_pseudo_label_training": Claim(UNKNOWN, "학습 자료 계보를 확인할 소스가 없다."),
        "no_blend_distillation": Claim(UNKNOWN, "교사 예측 사용 여부를 확인할 소스가 없다."),
    },
    members=(
        Member(
            name="depth9_pair_te",
            member_id="michaelqiu0606/s6e8-depth9-pair-te-inputs:depth9_pair_te",
            display_name="Michael Y. Qiu depth9 pair-TE",
            oof=ArraySpec("base_oof.npy", "npy"),
            test=ArraySpec("base_test.npy", "npy"),
            semantics="README가 선언한 OOF·시험 예측 배열",
            prior_pair_sha256="1d85e728c61ce6c177c90183b97b77e2bf20ff231a57dac6f9fd8b9bb93462d3",
            independent_auc=0.970516839533,
            training_point=Claim(UNKNOWN, "고정 학습 시점과 검증 기반 중단 여부를 확인할 소스가 없다."),
            caveats=(("lineage_source_missing", "README와 배열만 있고 학습 소스·분할 벡터·재현 가능한 manifest가 없다."),),
        ),
    ),
    notes=("계보를 현재 계약으로 확정할 수 없어 근거 부족으로 종결한다.", "자료 판본 1, lastUpdated=2026-08-29T17:27:56.327Z."),
    source_kind="dataset",
    fixed_source_file="README.md",
    source_license="CC0-1.0",
    output_license="CC0-1.0",
)


SOURCES: tuple[Source, ...] = (
    ZHUKOV, REDA_LGBM, REDA_HGB, YEKENOT, MOHAN, LOPURE, SHAMAN,
    BEICICC_TABNET, BEICICC_REALMLP, BUSYAPRIME, RAVI,
    SOMETIME_STACKING, MICHAEL_DEPTH9,
)


# ---------------------------------------------------------------------------
# 공통 도우미
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def pair_sha256(oof: np.ndarray, test: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(oof, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(test, dtype=np.float64).tobytes())
    return digest.hexdigest()


def canonical_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def git_dirty(paths: list[str]) -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "--", *paths], check=True, capture_output=True, text=True).stdout
    return bool(out.strip())


def kernel_dir(kernel_ref: str) -> Path:
    owner, slug = kernel_ref.split("/")
    return EXT / f"{owner}_{slug}"


def all_kernel_refs() -> list[tuple[str, int]]:
    refs: list[tuple[str, int]] = []
    for source in SOURCES:
        if source.source_kind == "notebook":
            refs.append((source.kernel_ref, source.script_version_id))
        for dep in source.dependencies:
            refs.append((dep.kernel_ref, dep.script_version_id))
    return refs


# ---------------------------------------------------------------------------
# fetch: Kaggle에서 고정 판본 소스와 출력을 확보한다
# ---------------------------------------------------------------------------


def _log(directory: Path, entry: dict) -> None:
    with (directory / ".download.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _run_logged(directory: Path, cmd: list[str], *, retries: int = 1) -> subprocess.CompletedProcess:
    for attempt in range(retries + 1):
        started = now_iso()
        clock = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        entry = {
            "at": started,
            "finished_at": now_iso(),
            "seconds": round(time.time() - clock, 1),
            "cmd": cmd,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
        _log(directory, entry)
        if result.returncode == 0:
            return result
        if "429" in result.stdout + result.stderr and attempt < retries:
            print(f"  429 응답, 45초 뒤 재시도: {' '.join(cmd)}", flush=True)
            time.sleep(45)
            continue
        raise RuntimeError(f"실패({result.returncode}): {' '.join(cmd)}\n{result.stderr[-1000:]}")
    raise AssertionError


def _page_version(kernel_ref: str) -> dict:
    """www.kaggle.com 노트북 페이지가 가리키는 현재 공개 scriptVersionId를 읽는다."""
    url = f"https://www.kaggle.com/code/{kernel_ref}"
    started = now_iso()
    result = subprocess.run(["curl", "-s", "-m", "20", url], capture_output=True, text=True)
    ids = sorted({int(m) for m in re.findall(r"scriptVersionId(?:%3D|=)(\d+)", result.stdout)})
    return {"url": url, "fetched_at": started, "http_ok": result.returncode == 0, "script_version_ids": ids}


def _dataset_version(dataset_ref: str) -> dict:
    """Kaggle CLI가 쓰는 Python 환경으로 공개 자료의 현재 판본 메타데이터를 읽는다."""
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError("kaggle 실행 파일을 찾을 수 없다")
    first_line = Path(executable).read_text(encoding="utf-8").splitlines()[0]
    if not first_line.startswith("#!"):
        raise RuntimeError(f"kaggle 실행 파일의 Python 경로를 읽을 수 없다: {executable}")
    python = first_line[2:]
    owner, slug = dataset_ref.split("/")
    code = """
import json, sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as client:
    request = ApiGetDatasetRequest()
    request.owner_slug, request.dataset_slug = sys.argv[1], sys.argv[2]
    print(json.dumps(client.datasets.dataset_api_client.get_dataset(request).to_dict()))
"""
    started = now_iso()
    result = subprocess.run([python, "-c", code, owner, slug], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"자료 판본 조회 실패: {dataset_ref}: {result.stderr[-1000:]}")
    metadata = json.loads(result.stdout)
    return {
        "url": f"https://www.kaggle.com/datasets/{dataset_ref}",
        "fetched_at": started,
        "http_ok": True,
        "dataset_id": metadata["id"],
        "dataset_version_number": metadata["currentVersionNumber"],
        "last_updated": metadata["lastUpdated"],
        "license": metadata.get("licenseName"),
        "versions": metadata.get("versions", []),
    }


def fetch(only: set[str] | None) -> None:
    EXT.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for kernel_ref, pinned in all_kernel_refs():
        if kernel_ref in seen or (only and kernel_ref not in only):
            continue
        seen.add(kernel_ref)
        directory = kernel_dir(kernel_ref)
        directory.mkdir(parents=True, exist_ok=True)
        print(f"== {kernel_ref} (고정 판본 {pinned})", flush=True)
        _log(directory, {"at": now_iso(), "event": "fetch_start", "kernel_ref": kernel_ref, "pinned_script_version_id": pinned})
        page = _page_version(kernel_ref)
        (directory / "page-version.json").write_text(json.dumps(page, ensure_ascii=False, indent=2) + "\n")
        _log(directory, {"at": now_iso(), "event": "page_version", **page})
        if page["script_version_ids"] != [pinned]:
            print(f"  경고: 페이지 판본 {page['script_version_ids']} != 고정 판본 {pinned}", flush=True)
        time.sleep(2)
        _run_logged(directory, ["kaggle", "kernels", "pull", kernel_ref, "-p", str(directory), "-m"])
        time.sleep(2)
        files = _run_logged(directory, ["kaggle", "kernels", "files", kernel_ref, "--format", "json", "--page-size", "200"])
        (directory / "files.json").write_text(files.stdout)
        time.sleep(2)
        _run_logged(directory, ["kaggle", "kernels", "output", kernel_ref, "-p", str(directory), "-o", "-q"])
        _log(directory, {"at": now_iso(), "event": "fetch_done", "kernel_ref": kernel_ref})
        time.sleep(3)
    for source in SOURCES:
        if source.source_kind != "dataset" or (only and source.kernel_ref not in only):
            continue
        directory = source.directory
        directory.mkdir(parents=True, exist_ok=True)
        print(f"== {source.kernel_ref} (고정 자료 판본 {source.script_version_id})", flush=True)
        _log(directory, {"at": now_iso(), "event": "fetch_start", "dataset_ref": source.kernel_ref, "pinned_dataset_version_number": source.script_version_id})
        page = _dataset_version(source.kernel_ref)
        (directory / "page-version.json").write_text(json.dumps(page, ensure_ascii=False, indent=2) + "\n")
        _log(directory, {"at": now_iso(), "event": "page_version", **page})
        if page["dataset_version_number"] != source.script_version_id:
            print(f"  경고: 자료 판본 {page['dataset_version_number']} != 고정 판본 {source.script_version_id}", flush=True)
        time.sleep(2)
        _run_logged(directory, ["kaggle", "datasets", "metadata", source.kernel_ref, "-p", str(directory)])
        time.sleep(2)
        files = _run_logged(directory, ["kaggle", "datasets", "files", source.kernel_ref, "--format", "json", "--page-size", "200"])
        (directory / "files.json").write_text(files.stdout)
        time.sleep(2)
        _run_logged(directory, ["kaggle", "datasets", "download", source.kernel_ref, "-p", str(directory), "--unzip", "-o", "-q"])
        _log(directory, {"at": now_iso(), "event": "fetch_done", "dataset_ref": source.kernel_ref})
        time.sleep(3)
    print("확보 완료", flush=True)


# ---------------------------------------------------------------------------
# audit: 검증하고 감사 기록·색인·요약을 만든다
# ---------------------------------------------------------------------------


@dataclass
class Notebook:
    path: Path
    sha256: str
    cells: list[tuple[int, str]]  # (셀 번호, 코드)


def load_notebook(path: Path) -> Notebook:
    if path.suffix != ".ipynb":
        return Notebook(path, file_sha256(path), [(0, path.read_text(encoding="utf-8", errors="replace"))])
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = [
        (index, "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"])
        for index, cell in enumerate(data["cells"])
        if cell["cell_type"] == "code"
    ]
    return Notebook(path, file_sha256(path), cells)


_SAFE_JOBLIB_GLOBALS = {
    ("builtins", "slice"),
    ("joblib.numpy_pickle", "NumpyArrayWrapper"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas.core.internals.managers", "SingleBlockManager"),
    ("pandas.core.series", "Series"),
}


class SafeNumpyUnpickler(NumpyUnpickler):
    """공개 joblib 출력에서 DataFrame과 ndarray 외의 전역 객체 복원을 거부한다."""

    def find_class(self, module: str, name: str):
        if (module, name) not in _SAFE_JOBLIB_GLOBALS:
            raise ValueError(f"허용하지 않은 joblib 전역 객체: {module}.{name}")
        return super().find_class(module, name)


@cache
def load_safe_joblib(path_text: str) -> dict:
    path = Path(path_text)
    with path.open("rb") as handle:
        value = SafeNumpyUnpickler(str(path), handle, ensure_native_byte_order=True).load()
    if not isinstance(value, dict):
        raise ValueError(f"{path}: joblib 최상위 값이 dict가 아니다")
    return value


def locate(notebook: Notebook, snippet: str) -> list[dict]:
    hits = []
    for cell_no, code in notebook.cells:
        for line_no, line in enumerate(code.splitlines(), 1):
            if snippet in line:
                hits.append({"file": str(notebook.path), "cell": cell_no, "line": line_no, "text": line.strip()[:200]})
    return hits


def locate_in_file(path: Path, snippet: str) -> list[dict]:
    hits = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if snippet in line:
            hits.append({"file": str(path), "line": line_no, "text": line.strip()[:200]})
    return hits


def resolve_claim(source: Source, notebook: Notebook, claim: Claim) -> dict:
    evidence: list[dict] = []
    missing: list[str] = []
    for snippet in claim.snippets:
        if claim.file is None:
            hits = locate(notebook, snippet)
        else:
            hits = locate_in_file(EXT / claim.file if "/" in claim.file else source.directory / claim.file, snippet)
        if not hits:
            missing.append(snippet)
        evidence.extend(hits[:3])
    for file, snippet in claim.extra_files:
        path = EXT / file if "/" in file else source.directory / file
        hits = locate_in_file(path, snippet) if path.exists() else []
        if not hits:
            missing.append(f"{file}: {snippet}")
        evidence.extend(hits[:2])
    status = claim.status
    note = claim.note
    if missing:
        status = UNKNOWN
        note = f"{claim.note} | 고정 판본에서 근거 조각을 찾지 못함: {missing}"
    return {"status": status, "evidence_kind": claim.kind, "evidence": evidence, "note": note}


def absence_scan(source: Source, notebook: Notebook) -> dict:
    hits: list[dict] = []
    unexplained = 0
    for label, pattern in ABSENCE_PATTERNS.items():
        regex = re.compile(pattern, re.IGNORECASE)
        for cell_no, code in notebook.cells:
            for line_no, line in enumerate(code.splitlines(), 1):
                if regex.search(line) and not line.strip().startswith("#"):
                    explanation = next((why for key, why in source.absence_explained.items() if key in line), None)
                    if explanation is None:
                        unexplained += 1
                    hits.append({"pattern": label, "cell": cell_no, "line": line_no, "text": line.strip()[:200], "explanation": explanation})
    return {"patterns": ABSENCE_PATTERNS, "hits": hits, "unexplained": unexplained}


def load_array(directory: Path, spec: ArraySpec, expected_ids: np.ndarray, labels: np.ndarray | None) -> tuple[np.ndarray, dict]:
    path = directory / spec.file
    info: dict = {"file": spec.file, "raw_sha256": file_sha256(path), "format": spec.fmt, "column": spec.column, "column_index": spec.column_index}
    if spec.fmt == "joblib_dataframe":
        payload = load_safe_joblib(str(path))
        frame = payload[spec.frame_key]
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{path}[{spec.frame_key}]: DataFrame이 아니다")
        info["frame_key"] = spec.frame_key
        info["raw_dtype"] = str(frame[spec.column].dtype)
        info["raw_shape"] = list(frame.shape)
        info["columns"] = list(frame.columns)
        values = frame[spec.column].to_numpy()
        positional = isinstance(frame.index, pd.RangeIndex) and frame.index.start == 0 and frame.index.step == 1
        info["row_alignment"] = {"method": "positional", "detail": f"RangeIndex(0..{len(frame) - 1}) 순서, 저장 코드가 공식 train/test 원본 순서로 채운다", "range_index": bool(positional)}
        if not positional:
            raise ValueError(f"{path}[{spec.frame_key}]: RangeIndex 원본 순서가 아니다")
        if spec.frame_key == "oof_preds" and labels is not None and "y" in payload:
            info["embedded_label_equals_ours"] = bool(np.array_equal(np.asarray(payload["y"]), labels))
    elif spec.fmt == "npy":
        raw = np.load(path)
        info["raw_dtype"] = str(raw.dtype)
        info["raw_shape"] = list(raw.shape)
        values = raw[:, spec.column_index] if spec.column_index is not None else raw.reshape(-1)
        info["row_alignment"] = {"method": "positional", "detail": "id 열 없음, 저장 코드가 train/test 원본 순서로 배열을 채운다"}
    elif spec.fmt == "parquet":
        frame = pd.read_parquet(path)
        info["raw_dtype"] = str(frame[spec.column].dtype)
        info["raw_shape"] = list(frame.shape)
        info["columns"] = list(frame.columns)
        values = frame[spec.column].to_numpy()
        positional = isinstance(frame.index, pd.RangeIndex) and frame.index.start == 0 and frame.index.step == 1
        info["row_alignment"] = {"method": "positional", "detail": f"RangeIndex(0..{len(frame) - 1}) 순서, id 열 없음", "range_index": bool(positional)}
    else:
        frame = pd.read_csv(path)
        info["raw_dtype"] = str(frame[spec.column].dtype)
        info["raw_shape"] = list(frame.shape)
        info["columns"] = list(frame.columns)
        values = frame[spec.column].to_numpy()
        if spec.has_id:
            ids = frame[ID].to_numpy()
            equal = bool(np.array_equal(ids, expected_ids))
            info["row_alignment"] = {"method": "id_column", "detail": "id 열이 기준 순서와 정확히 일치" if equal else "id 열이 기준 순서와 다름", "id_equal": equal}
            if not equal:
                raise ValueError(f"{path}: id 열이 기준 순서와 다르다")
        else:
            info["row_alignment"] = {"method": "positional", "detail": "id 열 없음"}
        if spec.label_column and labels is not None:
            info["embedded_label_equals_ours"] = bool(np.array_equal(frame[spec.label_column].to_numpy(), labels))
    values = np.ascontiguousarray(values, dtype=np.float64)
    info["rows"] = int(len(values))
    info["finite"] = bool(np.isfinite(values).all())
    info["dtype_note"] = spec.dtype_note
    return values, info


def declared_auc(directory: Path, declared: Declared | None) -> float | None:
    if declared is None:
        return None
    path = directory / declared.file
    if declared.kind == "json_key":
        return float(json.loads(path.read_text())[declared.key])
    frame = pd.read_csv(path)
    rows = frame[frame[frame.columns[0]] == declared.key]
    return float(rows[declared.column].iloc[-1])


def normalized_path(directory: Path, kind: str, name: str, values: np.ndarray) -> Path:
    path = directory / "normalized" / f"{kind}_{name}.npy"
    if path.exists() and np.array_equal(np.load(path), values):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    return path


def download_times(directory: Path) -> dict[str, str]:
    """`.download.log`에서 pull·output 명령의 완료 시각을 읽는다."""
    times: dict[str, str] = {}
    log = directory / ".download.log"
    if not log.exists():
        return times
    for line in log.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        cmd = entry.get("cmd") or []
        if len(cmd) > 2 and cmd[1] in {"kernels", "datasets"} and entry.get("returncode") == 0:
            times[cmd[2]] = entry["finished_at"]
        if entry.get("event") == "page_version":
            times["page"] = entry["fetched_at"]
    return times


def evidence_manifest(source: Source) -> dict:
    files = []
    for directory, role_prefix in [(source.directory, "source")] + [(kernel_dir(dep.kernel_ref), "dependency") for dep in source.dependencies]:
        times = download_times(directory)
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            rel = path.relative_to(EXT)
            name = path.name
            if path.parent.name == "normalized":
                role, retrieved = "normalized", None
            elif directory == source.directory and name == source.notebook_file:
                role = f"{role_prefix}_{'notebook' if source.source_kind == 'notebook' else 'description'}"
                retrieved = times.get("pull" if source.source_kind == "notebook" else "download")
            elif name in {"kernel-metadata.json", "files.json", "page-version.json", ".download.log"}:
                if source.source_kind == "dataset":
                    retrieved = times.get("metadata" if name == "kernel-metadata.json" else "page")
                else:
                    retrieved = times.get("pull" if name == "kernel-metadata.json" else "page")
                role = f"{role_prefix}_metadata"
            else:
                role = f"{role_prefix}_output"
                retrieved = times.get("output" if source.source_kind == "notebook" else "download")
            entry = {"path": str(rel), "role": role, "bytes": path.stat().st_size, "retrieved_at": retrieved}
            if name != ".download.log":
                entry["sha256"] = file_sha256(path)
            files.append(entry)
    return {"issue": ISSUE, "root": str(EXT), "kernel_ref": source.kernel_ref, "script_version_id": source.script_version_id, "files": files}


def spearman_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values).astype(np.float64)
    ranks -= ranks.mean()
    return ranks / (np.linalg.norm(ranks) + 1e-300)


def load_v2_array(spec: str) -> np.ndarray:
    match = re.fullmatch(r"(.+?)\[(.+)\]", spec)
    if match is None:
        values = np.load(spec).astype(np.float64)
        return values.reshape(-1)
    path, selector = match.groups()
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=[selector])[selector].to_numpy(np.float64)
    column = int(selector.split(",")[1])
    return np.load(path, mmap_mode="r")[:, column].astype(np.float64)


@dataclass
class Loaded:
    source: Source
    member: Member
    oof: np.ndarray
    test: np.ndarray
    oof_info: dict
    test_info: dict
    error: str | None = None


def near_duplicates(loaded: list[Loaded], v2_members: list[dict], target_ids: set[str]) -> dict[str, dict]:
    """새 기록 후보만 판본 2 통과 구성원과 현행 후보에 대해 근접 중복을 대조한다."""
    ranks = {item.member.member_id: spearman_ranks(item.oof) for item in loaded if item.error is None}
    best: dict[str, dict] = {mid: {"max": -1.0, "closest": None, "over_threshold": []} for mid in ranks if mid in target_ids}

    def consider(mid: str, other: str, corr: float) -> None:
        if corr > best[mid]["max"]:
            best[mid]["max"], best[mid]["closest"] = corr, other
        if corr >= NEAR_DUPLICATE_SPEARMAN:
            best[mid]["over_threshold"].append({"member_id": other, "spearman": corr})

    ids = list(ranks)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            corr = float(ranks[a] @ ranks[b])
            if a in best:
                consider(a, b, corr)
            if b in best:
                consider(b, a, corr)
    for row in v2_members:
        if row["member_id"] in ranks:
            continue
        other = spearman_ranks(load_v2_array(row["oof_path"]))
        for mid in best:
            consider(mid, f"v2:{row['member_id']}", float(ranks[mid] @ other))
    return best


def compute_eligibility(record: dict) -> tuple[str, list[str], list[str]]:
    """자격 판정을 기록의 근거에서 계산한다. (자격, 제외 사유 코드, 근거 부족 사유)"""
    reasons: list[str] = []
    gaps: list[str] = []
    pred = record["predictions"]
    if not record["fixed_source"]["pinned_version_confirmed"]:
        gaps.append("fixed_version_unconfirmed")
    for kind in ("oof", "test"):
        info = pred[kind]
        if info is None:
            reasons.append(f"{kind}_load_failed")
            continue
        expected = N_TRAIN if kind == "oof" else N_TEST
        if info["rows"] != expected:
            reasons.append(f"{kind}_row_count")
        if not info["finite"]:
            reasons.append(f"{kind}_non_finite")
        if info["row_alignment"]["method"] == "id_column" and not info["row_alignment"].get("id_equal"):
            reasons.append(f"{kind}_row_order")
        if info.get("embedded_label_equals_ours") is False:
            reasons.append(f"{kind}_embedded_label_mismatch")
    if pred.get("declared_auc") is not None and pred.get("rescored_auc") is not None:
        if abs(pred["declared_auc"] - pred["rescored_auc"]) > AUC_TOLERANCE:
            gaps.append("declared_auc_mismatch")
    if pred.get("independent_auc") is not None and pred.get("rescored_auc") is not None:
        if abs(pred["independent_auc"] - pred["rescored_auc"]) > AUC_TOLERANCE:
            gaps.append("independent_auc_mismatch")
    if record["identity"]["exact_duplicate_of"]:
        reasons.append("exact_duplicate")
    fold = record["fold_contract"]
    if fold["status"] == VIOLATION:
        reasons.append("fold_mismatch")
    elif fold["status"] != CONFIRMED:
        gaps.append("fold_evidence")
    for key, claim in record["training_contract"]["claims"].items():
        if claim["status"] == VIOLATION:
            reasons.append(f"claim_violation:{key}")
        elif claim["status"] == UNKNOWN:
            gaps.append(f"claim_unknown:{key}")
    if record["training_contract"]["absence_scan"]["unexplained"]:
        gaps.append("unexplained_external_input_pattern")
    if reasons:
        return INELIGIBLE, reasons, gaps
    if gaps:
        return INSUFFICIENT, [], gaps
    return ELIGIBLE, [], []


def fingerprint(record: dict) -> str:
    """감사 시각과 식별자를 뺀 내용 지문. 같으면 기존 기록을 유지한다."""
    volatile = {"audit_record_id", "audit_revision", "supersedes_audit_record_id", "record_sha256"}
    trimmed = {k: v for k, v in record.items() if k not in volatile}
    trimmed["audit"] = {k: v for k, v in record["audit"].items() if k not in {"audited_at", "audit_tool", "evidence_manifest_sha256", "transition_log"}}
    return text_sha256(canonical_json(trimmed))


def build_record(source: Source, member: Member, notebook: Notebook | None, item: Loaded, ids: dict, fold_vector_sha: str, page: dict, dependency_status: list[dict], near: dict | None, exact_dup: str | None, rescored: float | None, v2_row: dict | None) -> dict:
    directory = source.directory
    page_matches = (
        page.get("script_version_ids") == [source.script_version_id]
        if source.source_kind == "notebook"
        else page.get("dataset_version_number") == source.script_version_id
    )
    pinned_ok = notebook is not None and notebook.sha256 == source.pinned_source_sha256 and page_matches
    if notebook is None:
        fold_contract = {"status": UNKNOWN, "evidence_kind": "none", "evidence": [], "note": "고정 판본 소스 없음"}
        claims = {key: {"status": UNKNOWN, "evidence_kind": "none", "evidence": [], "note": "고정 판본 소스 없음"} for key in CLAIM_KEYS}
        scan = {"patterns": ABSENCE_PATTERNS, "hits": [], "unexplained": 1}
        lineage_evidence: list[dict] = []
    else:
        fold_contract = resolve_claim(source, notebook, source.fold)
        claims = {}
        for key in CLAIM_KEYS:
            claim = member.claim_overrides.get(key) or (member.training_point if key == "training_point_fixed" else source.claims[key])
            claims[key] = resolve_claim(source, notebook, claim)
        scan = absence_scan(source, notebook)
        lineage_evidence = [hit for snippet in member.lineage_snippets for hit in locate(notebook, snippet)[:2]]
    for key in CLAIM_KEYS:
        claims[key]["claim"] = CLAIM_TEXT[key]
    if not pinned_ok:
        if fold_contract["status"] == CONFIRMED:
            fold_contract["status"] = UNKNOWN
        for claim in claims.values():
            if claim["status"] in (CONFIRMED, NOT_APPLICABLE):
                claim["status"] = UNKNOWN
                claim["note"] += " | 고정 판본 소스를 확인하지 못해 근거 무효"

    oof_path = test_path = None
    normalized = None
    if item.error is None:
        oof_path = normalized_path(directory, "oof", member.name, item.oof)
        test_path = normalized_path(directory, "test", member.name, item.test)
        pair = pair_sha256(item.oof, item.test)
        normalized = {
            "oof_sha256": array_sha256(item.oof),
            "test_sha256": array_sha256(item.test),
            "pair_sha256": pair,
            "oof_path": str(oof_path),
            "test_path": str(test_path),
            "dtype": "float64",
            "layout": "C 연속 1차원, OOF 뒤에 시험 예측을 이어 붙여 쌍 해시 계산",
        }
    declared = None if item.error else declared_auc(directory, member.declared)
    predictions = {
        "oof": None if item.error else item.oof_info,
        "test": None if item.error else item.test_info,
        "load_error": item.error,
        "semantics": member.semantics,
        "normalized": normalized,
        "rescored_auc": rescored,
        "declared_auc": declared,
        "declared_auc_source": member.declared.file if member.declared else None,
        "declared_auc_delta": None if declared is None or rescored is None else rescored - declared,
        "independent_auc": member.independent_auc,
        "independent_auc_ref": source.prior_ref,
        "independent_auc_delta": None if rescored is None else rescored - member.independent_auc,
        "auc_tolerance": AUC_TOLERANCE,
        "prior_pair_sha256": member.prior_pair_sha256,
        "prior_pair_ref": source.prior_ref,
        "prior_pair_match": None if normalized is None else normalized["pair_sha256"] == member.prior_pair_sha256,
        "v2_ledger_member": None if v2_row is None else {"sha256": v2_row["sha256"], "auc": v2_row["auc"], "fold_evidence": v2_row["fold_evidence"], "caveats": v2_row["caveats"]},
    }
    caveats = [{"code": code, "detail": detail} for code, detail in member.caveats]
    if near is not None:
        for hit in near["over_threshold"]:
            if not any(c["code"] == "near_duplicate_cluster" for c in caveats):
                caveats.append({"code": "near_duplicate_cluster", "detail": f"{hit['member_id']}와 스피어만 {hit['spearman']:.6f}(이번 측정), 자격에는 영향 없음"})
    record = {
        "contract_version": CONTRACT_VERSION,
        "contract_ref": CONTRACT_REF,
        "ledger_version": LEDGER_VERSION,
        "ledger_issue": ISSUE,
        "audit_record_id": None,
        "audit_revision": None,
        "supersedes_audit_record_id": None,
        "identity": {
            "member_id": member.member_id,
            "display_name": member.display_name,
            "author": source.author,
            "source_kind": f"{source.source_kind}_output",
            "source_key": source.key,
            "model_lineage_id": f"{source.source_kind}:{source.kernel_ref}@{source.script_version_id}:{member.name}",
            "input_population": source.population,
            "exact_duplicate_of": exact_dup,
        },
        "fixed_source": {
            "source_kind": source.source_kind,
            "source_ref": source.kernel_ref,
            "kernel_ref": source.kernel_ref,
            "title": source.title,
            "url": source.url,
            "script_version_id": source.script_version_id,
            "dataset_version_number": source.script_version_id if source.source_kind == "dataset" else None,
            "source_file": str(directory / source.notebook_file),
            "source_sha256": None if notebook is None else notebook.sha256,
            "pinned_source_sha256": source.pinned_source_sha256,
            "pinned_version_confirmed": bool(pinned_ok),
            "version_confirmation": {
                "method": (
                    "kaggle kernels pull 소스 SHA-256과 www.kaggle.com의 scriptVersionId를 함께 대조"
                    if source.source_kind == "notebook"
                    else "Kaggle 자료 판본 번호와 README.md SHA-256을 함께 대조"
                ),
                "page": page,
                "page_matches_pinned": page_matches,
            },
            "retrieved_at": download_times(directory),
            "provenance_role": "direct",
            "dependency_refs": dependency_status,
            "output_listing_file": str(directory / "files.json"),
        },
        "predictions": predictions,
        "fold_contract": {
            "spec_id": FOLD_SPEC_ID,
            "spec": FOLD_SPEC,
            "fold_vector_sha256": fold_vector_sha,
            "row_order_condition": "train.csv 원본 행 순서(id 0..691368 오름차순)",
            "exact_match": fold_contract["status"] == CONFIRMED,
            **fold_contract,
        },
        "training_contract": {
            "claims": claims,
            "lineage_evidence": lineage_evidence,
            "absence_scan": scan,
            "prepublication_search_history": UNKNOWN,
        },
        "license": {
            "source_code": {
                "license": source.source_license,
                "status": "confirmed",
                "evidence": SOURCE_LICENSE_EVIDENCE if source.source_kind == "notebook" else page.get("url"),
                "attribution_required": source.source_license != "CC0-1.0",
                "attribution": f"{source.author}, {source.title}, {source.url}",
            },
            "output_arrays": {
                "license": source.output_license,
                "status": "confirmed" if source.output_license != "unknown" else "unmarked",
                "use_scope": USE_SCOPE,
                "redistribution": source.output_license == "CC0-1.0",
                "repository_commit": False,
            },
        },
        "audit": {
            "audit_state": AUDIT_STATES[2],
            "transition_log": [],
            "eligibility": None,
            "exclusion_reason_codes": [],
            "insufficiency_reasons": [],
            "caveat_codes": [c["code"] for c in caveats],
            "caveats": caveats,
            "near_duplicate": None if near is None else {"spearman_max": near["max"], "closest": near["closest"], "threshold": NEAR_DUPLICATE_SPEARMAN, "population": "판본 2 통과 구성원 + 판본 3 현행 후보"},
            "auditor": "tmheo (감사 도구 실행)",
            "audited_at": None,
            "audit_tool": None,
            "evidence_manifest_sha256": None,
            "guarantees": GUARANTEES,
            "non_guarantees": NON_GUARANTEES,
            "notes": list(source.notes),
        },
        "record_sha256": None,
    }
    eligibility, reasons, gaps = compute_eligibility(record)
    record["audit"]["eligibility"] = eligibility
    record["audit"]["exclusion_reason_codes"] = reasons
    record["audit"]["insufficiency_reasons"] = gaps
    return record


def finalize_record(record: dict, existing: dict | None, evidence: dict, audited_at: str, commit: str) -> tuple[dict, bool]:
    """기존 현행 기록과 지문이 같으면 그대로 두고, 다르면 supersedes로 잇는 새 기록을 만든다."""
    if existing is not None and fingerprint(existing) == fingerprint(record):
        return existing, False
    member_id = record["identity"]["member_id"]
    pair = (record["predictions"]["normalized"] or {}).get("pair_sha256") or "no-pair"
    revision = 1 if existing is None else existing["audit_revision"] + 1
    record["audit_record_id"] = "emar3-" + text_sha256(f"{member_id}|{record['fixed_source']['script_version_id']}|{pair}|{revision}")[:16]
    record["audit_revision"] = revision
    record["supersedes_audit_record_id"] = None if existing is None else existing["audit_record_id"]
    record["audit"]["audited_at"] = audited_at
    record["audit"]["audit_tool"] = {"script": "scripts/build_external_member_ledger_v3.py", "commit": commit}
    record["audit"]["evidence_manifest_sha256"] = text_sha256(canonical_json(evidence))
    record["audit"]["transition_log"] = [
        {"state": AUDIT_STATES[0], "at": record["fixed_source"]["retrieved_at"].get("page"), "by": "fetch"},
        {"state": AUDIT_STATES[1], "at": record["fixed_source"]["retrieved_at"].get("pull"), "by": "fetch"},
        {"state": AUDIT_STATES[2], "at": audited_at, "by": "audit"},
    ]
    record["record_sha256"] = text_sha256(canonical_json({k: v for k, v in record.items() if k != "record_sha256"}))
    return record, True


def load_index() -> dict | None:
    return json.loads(INDEX_PATH.read_text(encoding="utf-8")) if INDEX_PATH.exists() else None


def load_record(record_id: str) -> dict:
    return json.loads((RECORDS_DIR / f"{record_id}.json").read_text(encoding="utf-8"))


def audit() -> None:
    started = now_iso()
    commit = git_commit()
    train = pd.read_csv(TRAIN_PATH, usecols=[ID, TARGET])
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID])[ID].to_numpy()
    ids = {"oof": train[ID].to_numpy(), "test": test_ids}
    labels = train[TARGET].to_numpy()
    folds = pd.read_parquet(FOLDS_PATH)
    assert np.array_equal(folds[ID].to_numpy(), ids["oof"])
    fold_vector_sha = hashlib.sha256(folds["fold"].to_numpy(np.int8).tobytes()).hexdigest()
    v2 = json.loads(V2_LEDGER_PATH.read_text(encoding="utf-8"))
    v2_accepted = [row for row in v2["members"] if row["status"] == "accepted"]
    v2_by_id = {row["member_id"]: row for row in v2["members"]}
    v2_by_sha = {row["sha256"]: row["member_id"] for row in v2_accepted}

    index = load_index()
    current: dict[str, dict] = {}
    if index is not None:
        for row in index["current_records"]:
            current[row["member_id"]] = load_record(row["audit_record_id"])
    superseded_ids: list[str] = list(index["superseded_record_ids"]) if index else []

    loaded: list[Loaded] = []
    notebooks: dict[str, Notebook | None] = {}
    for source in SOURCES:
        path = source.directory / source.notebook_file
        notebooks[source.key] = load_notebook(path) if path.exists() else None
        for member in source.members:
            try:
                oof, oof_info = load_array(source.directory, member.oof, ids["oof"], labels)
                test, test_info = load_array(source.directory, member.test, ids["test"], None)
                loaded.append(Loaded(source, member, oof, test, oof_info, test_info))
            except (OSError, ValueError, KeyError) as exc:
                loaded.append(Loaded(source, member, np.empty(0), np.empty(0), {}, {}, error=f"{type(exc).__name__}: {exc}"))
            print(f"적재 {member.member_id}: {'실패 ' + loaded[-1].error if loaded[-1].error else 'OK'}", flush=True)

    pair_of = {item.member.member_id: pair_sha256(item.oof, item.test) for item in loaded if item.error is None}
    target_ids: set[str] = set()
    for item in loaded:
        member_id = item.member.member_id
        existing = current.get(member_id)
        if existing is None or item.error is not None:
            target_ids.add(member_id)
            continue
        existing_pair = (existing["predictions"].get("normalized") or {}).get("pair_sha256")
        fixed = existing["fixed_source"]
        if (
            fixed.get("script_version_id") != item.source.script_version_id
            or fixed.get("source_sha256") != item.source.pinned_source_sha256
            or existing_pair != pair_of[member_id]
        ):
            target_ids.add(member_id)

    print(f"근접 중복 대조(새 감사 기록 후보 {len(target_ids)}개, 판본 2 통과 400개)...", flush=True)
    near = near_duplicates(loaded, v2_accepted, target_ids)

    records: list[dict] = []
    created = 0
    audited_at = now_iso()
    manifests: dict[str, dict] = {}
    for item in loaded:
        source, member = item.source, item.member
        existing = current.get(member.member_id)
        if member.member_id not in target_ids and existing is not None:
            records.append(existing)
            print(f"{member.member_id}: {existing['audit']['eligibility']} ({existing['audit_record_id']}, 유지)", flush=True)
            continue
        notebook = notebooks[source.key]
        page_file = source.directory / "page-version.json"
        page = json.loads(page_file.read_text()) if page_file.exists() else {}
        deps = []
        for dep in source.dependencies:
            dep_dir = kernel_dir(dep.kernel_ref)
            dep_page_file = dep_dir / "page-version.json"
            dep_page = json.loads(dep_page_file.read_text()) if dep_page_file.exists() else {}
            files = []
            for name, pinned_sha in dep.files.items():
                dep_path = dep_dir / name
                actual = file_sha256(dep_path) if dep_path.exists() else None
                files.append({"file": str(dep_path), "sha256": actual, "pinned_sha256": pinned_sha, "match": actual == pinned_sha})
            deps.append({
                "kernel_ref": dep.kernel_ref,
                "url": f"https://www.kaggle.com/code/{dep.kernel_ref}?scriptVersionId={dep.script_version_id}",
                "script_version_id": dep.script_version_id,
                "role": dep.role,
                "page_matches_pinned": dep_page.get("script_version_ids") == [dep.script_version_id],
                "retrieved_at": download_times(dep_dir),
                "files": files,
            })
        exact = None
        if item.error is None:
            mine = pair_of[member.member_id]
            for other_id, other_pair in pair_of.items():
                if other_id != member.member_id and other_pair == mine and list(pair_of).index(other_id) < list(pair_of).index(member.member_id):
                    exact = other_id
            if exact is None and mine in v2_by_sha and v2_by_sha[mine] != member.member_id:
                exact = f"v2:{v2_by_sha[mine]}"
        rescored = None if item.error else float(roc_auc_score(labels, item.oof))
        record = build_record(source, member, notebook, item, ids, fold_vector_sha, page, deps, near.get(member.member_id), exact, rescored, v2_by_id.get(member.member_id))
        if deps and not all(f["match"] for d in deps for f in d["files"]):
            record["audit"]["insufficiency_reasons"].append("dependency_hash_mismatch")
            if record["audit"]["eligibility"] == ELIGIBLE:
                record["audit"]["eligibility"] = INSUFFICIENT
        if source.key not in manifests:
            manifests[source.key] = evidence_manifest(source)
        final, is_new = finalize_record(record, existing, manifests[source.key], audited_at, commit)
        if is_new:
            created += 1
            if existing is not None:
                superseded_ids.append(existing["audit_record_id"])
            RECORDS_DIR.mkdir(parents=True, exist_ok=True)
            EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
            (RECORDS_DIR / f"{final['audit_record_id']}.json").write_text(json.dumps(final, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            (EVIDENCE_DIR / f"{final['audit_record_id']}.json").write_text(json.dumps(manifests[source.key], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        records.append(final)
        print(f"{final['identity']['member_id']}: {final['audit']['eligibility']} ({final['audit_record_id']}, {'신규' if is_new else '유지'})", flush=True)

    write_index(records, superseded_ids, commit, fold_vector_sha, started, created)
    write_summary(records, commit)
    print(f"감사 기록 {len(records)}개(신규 {created}개), 색인 {INDEX_PATH}, 요약 {SUMMARY_PATH}", flush=True)


def write_index(records: list[dict], superseded_ids: list[str], commit: str, fold_vector_sha: str, started: str, created: int) -> None:
    rows = []
    for record in records:
        normalized = record["predictions"]["normalized"] or {}
        rows.append({
            "member_id": record["identity"]["member_id"],
            "display_name": record["identity"]["display_name"],
            "audit_record_id": record["audit_record_id"],
            "audit_revision": record["audit_revision"],
            "supersedes_audit_record_id": record["supersedes_audit_record_id"],
            "audit_state": record["audit"]["audit_state"],
            "eligibility": record["audit"]["eligibility"],
            "exclusion_reason_codes": record["audit"]["exclusion_reason_codes"],
            "insufficiency_reasons": record["audit"]["insufficiency_reasons"],
            "caveat_codes": record["audit"]["caveat_codes"],
            "source_kind": record["fixed_source"].get("source_kind", "notebook"),
            "source_ref": record["fixed_source"].get("source_ref", record["fixed_source"]["kernel_ref"]),
            "kernel_ref": record["fixed_source"]["kernel_ref"],
            "script_version_id": record["fixed_source"]["script_version_id"],
            "dataset_version_number": record["fixed_source"].get("dataset_version_number"),
            "input_population": record["identity"]["input_population"],
            "rescored_auc": record["predictions"]["rescored_auc"],
            "oof_path": normalized.get("oof_path"),
            "test_path": normalized.get("test_path"),
            "oof_sha256": normalized.get("oof_sha256"),
            "test_sha256": normalized.get("test_sha256"),
            "pair_sha256": normalized.get("pair_sha256"),
            "record_sha256": record["record_sha256"],
            "evidence_manifest_sha256": record["audit"]["evidence_manifest_sha256"],
        })
    eligible = [r for r in rows if r["eligibility"] == ELIGIBLE]
    v2 = json.loads(V2_LEDGER_PATH.read_text(encoding="utf-8"))
    index = {
        "issue": ISSUE,
        "incremental_issue": INCREMENTAL_ISSUE,
        "incremental_survey_cutoff": "2026-08-30T12:00:00Z",
        "ledger_version": LEDGER_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_ref": CONTRACT_REF,
        "history": v2["history"] + [{"version": 2, "issue": 454, "generated": v2["generated"], "candidate_count": v2["summary"]["candidate_count"], "accepted_count": v2["summary"]["accepted_count"], "note": "판본 2 장부(docs/research/external-member-ledger.json)는 과거 기록으로 보존"}],
        "generated_at": started,
        "tool": {"script": "scripts/build_external_member_ledger_v3.py", "commit": commit},
        "fold_spec": {"id": FOLD_SPEC_ID, "spec": FOLD_SPEC, "fold_vector_sha256": fold_vector_sha, "folds_path": str(FOLDS_PATH)},
        "row_contract": {"train_rows": N_TRAIN, "test_rows": N_TEST, "auc_tolerance": AUC_TOLERANCE, "near_duplicate_spearman": NEAR_DUPLICATE_SPEARMAN},
        "input_candidates": {
            "v2_reaudit_issue_480": [m.member_id for s in SOURCES if s.population == "v2_reaudit" for m in s.members],
            "census_issue_479": [m.member_id for s in SOURCES if s.population == "census" for m in s.members],
            "incremental_issue_487": [m.member_id for s in SOURCES if s.population == "incremental_issue_487" for m in s.members],
        },
        "license_policy": "각 공개 소스와 출력의 표시된 사용 조건을 기록한다. 사용 조건이 표시되지 않은 출력 배열은 license_unknown_use_limited 주의 사항과 결합 입력 전용 범위로 한정한다. 외부 배열은 커밋하거나 재배포하지 않는다.",
        "summary": {
            "record_count": len(rows),
            "created_this_run": created,
            "eligible": len(eligible),
            "ineligible": sum(r["eligibility"] == INELIGIBLE for r in rows),
            "insufficient": sum(r["eligibility"] == INSUFFICIENT for r in rows),
            "prior_pair_match": sum(bool(rec["predictions"]["prior_pair_match"]) for rec in records),
            "caveat_counts": {code: sum(code in r["caveat_codes"] for r in rows) for code in sorted({c for r in rows for c in r["caveat_codes"]})},
        },
        "eligible_current_records_in_order": [{"member_id": r["member_id"], "audit_record_id": r["audit_record_id"], "pair_sha256": r["pair_sha256"], "oof_path": r["oof_path"], "test_path": r["test_path"]} for r in eligible],
        "current_records": rows,
        "superseded_record_ids": superseded_ids,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _short(sha: str | None) -> str:
    return "-" if not sha else f"`{sha[:12]}…`"


def write_summary(records: list[dict], commit: str) -> None:
    lines: list[str] = []
    eligible = [r for r in records if r["audit"]["eligibility"] == ELIGIBLE]
    insufficient = [r for r in records if r["audit"]["eligibility"] == INSUFFICIENT]
    ineligible = [r for r in records if r["audit"]["eligibility"] == INELIGIBLE]
    incremental = [r for r in records if r["identity"]["input_population"] == "incremental_issue_487"]
    lines += [
        "# 확장 스택용 외부 구성원 장부 (판본 3, 이슈 #484·#487)",
        "",
        "## 결론",
        "",
        "판본 3은 [판본 2 공개 노트북 재감사](https://github.com/tmheo/predicting-smartphone-addiction/issues/480) 통과 11개, [장부 밖 전수 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/479) 통과 8개와 [2026-08-30 증분 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/487)에서 발견한 6개 후보를 같은 계약으로 감사했다.",
        f"현행 후보 {len(records)}개 전부 `감사 완료` 상태이며 그중 **{len(eligible)}개가 `자격 있음`**, {len(ineligible)}개가 `자격 없음`, {len(insufficient)}개가 `근거 부족`이다.",
        f"증분 조사 후보 {len(incremental)}개 가운데 {sum(r['audit']['eligibility'] == ELIGIBLE for r in incremental)}개는 `자격 있음`, {sum(r['audit']['eligibility'] == INSUFFICIENT for r in incremental)}개는 `근거 부족`이다.",
        f"정규화 예측 쌍 SHA-256은 {sum(bool(r['predictions']['prior_pair_match']) for r in records)}개가 조사 보고서의 값과 일치한다.",
        "감사 진행 상태, 자격 판정, 후보 동결은 서로 다른 축이며 이 문서와 색인은 후보를 동결하지 않는다.",
        "",
        "## 산출물",
        "",
        f"- 기계 판독 색인: `{INDEX_PATH}` (`ledger_version: 3`, 현행 감사 기록만 가리키며 예측 배열을 담지 않는다)",
        f"- 감사 기록: `{RECORDS_DIR}/<감사 기록 식별자>.json` (변경 불가, `record_sha256`는 그 필드를 뺀 정규 JSON의 SHA-256)",
        f"- 근거 묶음 manifest: `{EVIDENCE_DIR}/<감사 기록 식별자>.json` (외부 파일의 경로·바이트·SHA-256·확보 시각)",
        f"- 반입 실행 기록: `{RUN_PATH}`",
        f"- 외부 파일: `{EXT}/<owner>_<slug>/` (소스 `.ipynb`, 출력 원문, `kernel-metadata.json`, `files.json`, `page-version.json`, `.download.log`, `normalized/` float64 npy). `data/`는 커밋 제외 경로다.",
        "- 생성 도구: `scripts/build_external_member_ledger_v3.py` (`fetch` → `audit` → `verify`). 판본 1·2 장부와 `scripts/build_external_member_ledger.py`는 수정하지 않고 과거 기록으로 보존한다.",
        f"- 도구 커밋: `{commit}`",
        "",
        "## 후보별 자격 판정",
        "",
        "| 순서 | 구성원 | 고정 판본 | 자격 | 재채점 AUC | 독립 조사 AUC 차이 | 선언 AUC 차이 | 쌍 SHA-256 | 보고서 대조 | 주의 사항 |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for i, r in enumerate(records, 1):
        p = r["predictions"]
        rescored = "-" if p["rescored_auc"] is None else f"{p['rescored_auc']:.6f}"
        independent_delta = "-" if p["independent_auc_delta"] is None else f"{p['independent_auc_delta']:+.1e}"
        declared_delta = "-" if p["declared_auc_delta"] is None else f"{p['declared_auc_delta']:+.1e}"
        lines.append(
            f"| {i} | `{r['identity']['member_id']}` | [{r['fixed_source']['script_version_id']}]({r['fixed_source']['url']}) | {r['audit']['eligibility']} | "
            f"{rescored} | {independent_delta} | {declared_delta} | "
            f"{_short((p['normalized'] or {}).get('pair_sha256'))} | {'일치' if p['prior_pair_match'] else '불일치'} | {', '.join(r['audit']['caveat_codes']) or '-'} |"
        )
    lines += [
        "",
        "모든 후보의 감사 기록 식별자, 제외 사유와 근거 부족 사유는 색인의 `current_records`에 있다.",
        "",
        "## 원본·정규화 해시 대조표",
        "",
        "| 구성원 | OOF 원본 파일 | OOF 원본 SHA-256 | 시험 원본 파일 | 시험 원본 SHA-256 | 정규화 OOF SHA-256 | 정규화 시험 SHA-256 | 쌍 SHA-256 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        p = r["predictions"]
        n = p["normalized"] or {}
        oof, test = p["oof"] or {}, p["test"] or {}
        lines.append(
            f"| `{r['identity']['member_id']}` | `{oof.get('file')}` | `{oof.get('raw_sha256')}` | `{test.get('file')}` | `{test.get('raw_sha256')}` | "
            f"`{n.get('oof_sha256')}` | `{n.get('test_sha256')}` | `{n.get('pair_sha256')}` |"
        )
    lines += [
        "",
        "## 고정 공개 판본과 확보",
        "",
        "| 공개 자료 | 판본 | 소스 SHA-256 | 페이지 판본 일치 | 소스 확보 시각(UTC) | 출력 확보 시각(UTC) |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    seen: set[str] = set()
    for r in records:
        fs = r["fixed_source"]
        if fs["kernel_ref"] in seen:
            continue
        seen.add(fs["kernel_ref"])
        source_time = fs["retrieved_at"].get("pull") or fs["retrieved_at"].get("download")
        output_time = fs["retrieved_at"].get("output") or fs["retrieved_at"].get("download")
        lines.append(f"| `{fs['kernel_ref']}` | {fs['script_version_id']} | `{fs['source_sha256']}` | {'예' if fs['version_confirmation']['page_matches_pinned'] else '아니오'} | {source_time} | {output_time} |")
        for dep in fs["dependency_refs"]:
            lines.append(f"| `{dep['kernel_ref']}` (보조 코드) | {dep['script_version_id']} | {', '.join('`' + f['sha256'] + '`' for f in dep['files'])} | {'예' if dep['page_matches_pinned'] else '아니오'} | {dep['retrieved_at'].get('pull')} | {dep['retrieved_at'].get('output')} |")
    lines += [
        "",
        "노트북 고정 판본은 `kaggle kernels pull`이 받은 공개 소스 SHA-256과 공개 페이지의 `scriptVersionId`를 함께 대조했다.",
        "자료 고정 판본은 Kaggle 자료 판본 번호와 내려받은 README.md SHA-256을 함께 대조했다.",
        "Kaggle CLI 2.2.4는 특정 판본 내려받기를 거부(403)하므로 위 두 대조가 판본 고정의 근거다.",
        "",
        "## 검증 항목",
        "",
        "- 행 수 691,369(OOF)와 296,302(시험), 유한값.",
        "- 원래 행 순서: id 열이 있는 CSV는 train.csv·test.csv의 id 순서와 정확히 일치해야 하고, npy·parquet는 위치 정렬이며 저장 코드가 원본 순서로 채우는지 소스에서 확인한다.",
        "- 고정 5분할: 소스 안의 분할 코드 위치를 감사 기록에 남긴다(`fold_contract.evidence`).",
        "- 학습 격리 주장 10개: 각각 `확인됨`, `위반 확인`, `알 수 없음`, `해당 없음`과 근거 종류, 고정 판본 안의 셀·줄 위치를 기록한다. 근거 조각을 소스에서 찾지 못하면 자동으로 `알 수 없음`이 된다.",
        "- 외부 입력·의사 목표값·증류 패턴 자동 검색(`absence_scan`): 설명되지 않은 일치가 있으면 `근거 부족`.",
        "- 재채점 AUC는 독립 조사 AUC, 선언 AUC와 1e-5 안에서 맞아야 하며 산출물 동일성 확인에만 쓴다.",
        "- 정확 중복은 제외 사유, 근접 중복(스피어만 0.998 이상)은 주의 사항이며 자격을 바꾸지 않는다.",
        "- 사용 조건 미표시는 `license_unknown_use_limited` 주의 사항과 결합 입력 전용 범위로 기록하며 자격 제외 사유가 아니다.",
        "",
        "## 재현",
        "",
        "```",
        "uv run python scripts/build_external_member_ledger_v3.py fetch",
        "uv run python scripts/build_external_member_ledger_v3.py audit",
        "uv run python scripts/build_external_member_ledger_v3.py verify",
        "```",
        "",
        "`audit`는 같은 입력에서 같은 내용 지문을 얻으면 기존 감사 기록을 그대로 두고(감사 시각도 유지), 고정 판본·예측·근거·판정이 달라진 후보만 새 기록을 만들어 `supersedes_audit_record_id`로 잇는다.",
        "`verify`는 모든 기록의 `record_sha256`와 근거 manifest 해시를 다시 계산하고, 커밋된 기록에 예측 배열이 없는지 확인한다.",
        "",
        "## 범위 밖",
        "",
        "- 외부 후보 동결 명세 생성, 중첩 선별 판정, 확장 스택 조립과 제출.",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# verify: 변경 불가 규칙과 배열 미포함을 검증한다
# ---------------------------------------------------------------------------


def verify() -> None:
    index = load_index()
    assert index is not None, "색인이 없다"
    problems: list[str] = []
    seen_ids: set[str] = set()
    for row in index["current_records"]:
        record = load_record(row["audit_record_id"])
        seen_ids.add(record["audit_record_id"])
        expected = text_sha256(canonical_json({k: v for k, v in record.items() if k != "record_sha256"}))
        if record["record_sha256"] != expected or row["record_sha256"] != expected:
            problems.append(f"{row['audit_record_id']}: record_sha256 불일치(제자리 수정 의심)")
        evidence_path = EVIDENCE_DIR / f"{record['audit_record_id']}.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if text_sha256(canonical_json(evidence)) != record["audit"]["evidence_manifest_sha256"]:
            problems.append(f"{row['audit_record_id']}: 근거 manifest 해시 불일치")
        if record["audit"]["audit_state"] != AUDIT_STATES[2]:
            problems.append(f"{row['audit_record_id']}: 감사 완료 상태가 아님")
        if record["audit"]["eligibility"] not in (ELIGIBLE, INELIGIBLE, INSUFFICIENT):
            problems.append(f"{row['audit_record_id']}: 자격 판정 값 오류")
        chain = record["supersedes_audit_record_id"]
        while chain:
            if chain not in index["superseded_record_ids"]:
                problems.append(f"{row['audit_record_id']}: supersedes 대상 {chain}이 색인의 과거 기록 목록에 없음")
                break
            chain = load_record(chain)["supersedes_audit_record_id"]
    for record_id in index["superseded_record_ids"]:
        if record_id in seen_ids:
            problems.append(f"{record_id}: 과거 기록이 현행 목록에도 있음")
        path = RECORDS_DIR / f"{record_id}.json"
        if not path.exists():
            problems.append(f"{record_id}: 과거 기록 파일 없음")
    for path in list(RECORDS_DIR.glob("*.json")) + list(EVIDENCE_DIR.glob("*.json")) + [INDEX_PATH]:
        size = path.stat().st_size
        if size > 400_000:
            problems.append(f"{path}: {size} 바이트, 예측 배열 포함 의심")
        text = path.read_text(encoding="utf-8")
        if re.search(r"\[(\s*-?\d+(\.\d+)?(e-?\d+)?\s*,){100,}", text):
            problems.append(f"{path}: 긴 숫자 배열 포함")
    tracked = subprocess.run(["git", "ls-files", "--", str(RECORDS_DIR)], check=True, capture_output=True, text=True).stdout.split()
    for rel in tracked:
        head = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, text=True)
        if head.returncode == 0 and head.stdout != Path(rel).read_text(encoding="utf-8"):
            problems.append(f"{rel}: 커밋된 감사 기록이 작업 트리에서 바뀜(제자리 수정)")
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"검증 통과: 현행 기록 {len(index['current_records'])}개, 과거 기록 {len(index['superseded_record_ids'])}개, 제자리 수정 없음, 배열 미포함")


# ---------------------------------------------------------------------------


def record_run(command: str, started: str, argv: list[str]) -> None:
    runs = json.loads(RUN_PATH.read_text(encoding="utf-8")) if RUN_PATH.exists() else {"issue": ISSUE, "runs": []}
    kaggle = subprocess.run(["kaggle", "--version"], capture_output=True, text=True).stdout.strip()
    runs["runs"].append({"command": command, "argv": argv, "started_at": started, "finished_at": now_iso(), "tool_commit": git_commit(), "kaggle_cli": kaggle, "python": sys.version.split()[0]})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_PATH.write_text(json.dumps(runs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "audit", "verify"])
    parser.add_argument("--only", nargs="*", help="fetch 대상 kernel_ref 제한")
    args = parser.parse_args()
    started = now_iso()
    if args.command == "fetch":
        fetch(set(args.only) if args.only else None)
    elif args.command == "audit":
        audit()
    else:
        verify()
        return
    record_run(args.command, started, sys.argv[1:])


if __name__ == "__main__":
    main()
