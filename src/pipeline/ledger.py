"""판정 장부 module. champion.yaml·pool.yaml의 원본을 load/save 소유 타입으로 담는다. (지도 #91, #96)

장부는 git에 커밋되는 판정 결과의 기록이다: "무엇이 champion인가"(champion.yaml)와
"무엇이 풀에 있는가"(pool.yaml)라는 결정을 mlflow.db 없이 git 이력으로 남긴다.
YAML의 모양(키 순서, seed·fold의 int 키 정규화, 시드의 쉼표 문자열)을 아는 곳은
이 module의 load/save가 유일하다. 판정 규칙(judgment)은 이 타입을 읽기만 하고,
CLI(compare·pool)가 채택·등록 시점에 기록을 조립해 save를 부른다.

진입 근거(EntryEvidence)는 진입 시점의 스냅샷이다: champion_run_id와
champion_oof_auc는 진입 판정 당시의 champion을 가리키며, champion이 나중에
교체되어도 재평가하거나 갱신하지 않는다(진입 하한은 진입 시점에만 적용, ADR 0001).
당시 champion의 OOF AUC를 함께 기록하므로 그 champion run이 장부에서 사라져도
근거는 스스로 완결된다. 일괄 재심사는 P3 풀 점검(#63)과 P4 앙상블 구성의 소관이다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

CHAMPION_PATH = Path("artifacts/champion.yaml")
POOL_PATH = Path("artifacts/pool.yaml")


@dataclass(frozen=True)
class Champion:
    """champion 장부: 새 실험의 개선 판정 기준이 되는 run의 채택 기록."""

    run_id: str
    oof_auc: float
    seed_aucs: dict[int, float]  # 판정 계약(#70) 이전 champion에는 없다(빈 dict).
    fold_aucs: dict[int, float]  # 위와 같다. 시드 평균본 기준.
    config: str
    features: set[str]
    git_commit: str
    adopted_at: str
    reason: str

    @classmethod
    def load(cls, path: Path = CHAMPION_PATH) -> Champion:
        with path.open() as f:
            record = yaml.safe_load(f)
        return cls(
            run_id=record["run_id"],
            oof_auc=float(record["oof_auc"]),
            seed_aucs={int(k): float(v) for k, v in record.get("seed_aucs", {}).items()},
            fold_aucs={int(k): float(v) for k, v in record.get("fold_aucs", {}).items()},
            config=record["config"],
            features=set(record["features"].split(",")),
            git_commit=record["git_commit"],
            adopted_at=str(record["adopted_at"]),
            reason=record["reason"],
        )

    def save(self, path: Path = CHAMPION_PATH) -> None:
        record = {
            "run_id": self.run_id,
            # 작은 채택 문턱과 비교하므로 반올림 없이 전체 정밀도로 남긴다.
            "oof_auc": float(self.oof_auc),
            # 확정 재검증의 시드별 비교와 경계 구간 fold 승리 게이트의 기준값. (ADR 0001)
            "seed_aucs": {s: float(self.seed_aucs[s]) for s in sorted(self.seed_aucs)},
            "fold_aucs": {f: float(self.fold_aucs[f]) for f in sorted(self.fold_aucs)},
            "config": self.config,
            "features": ",".join(sorted(self.features)),
            "git_commit": self.git_commit,
            "adopted_at": self.adopted_at,
            "reason": self.reason,
        }
        with path.open("w") as f:
            yaml.safe_dump(record, f, allow_unicode=True, sort_keys=False)


@dataclass(frozen=True)
class EntryEvidence:
    """풀 진입 판정의 근거 기록. 진입 시점 스냅샷이다(module docstring)."""

    champion_run_id: str  # 진입 판정 당시의 champion. 교체 후에도 갱신하지 않는다.
    champion_oof_auc: float
    floor_margin: float  # 후보 OOF AUC − 진입 하한. 진입 시점의 여유 폭.
    nearest_run_id: str | None  # 진입 당시 풀이 비어 있었으면 None.
    nearest_spearman: float | None
    ensemble_auc_with: float | None  # 기여 참고값을 계산하지 않았으면 None.
    ensemble_auc_without: float | None
    contribution: float | None


@dataclass(frozen=True)
class PoolJudgmentPointer:
    """후보 풀 장부 변경을 허용한 변경 불가 판정 기록 포인터."""

    judgment_id: str
    contract_version: str
    path: str
    sha256: str


@dataclass(frozen=True)
class PoolMember:
    """후보 풀 구성원 한 명의 등록 기록."""

    run_id: str
    config: str
    oof_auc: float
    seeds: list[int]
    entered_at: str
    reason: str
    evidence: EntryEvidence
    judgment: PoolJudgmentPointer | None = None


@dataclass
class Pool:
    """후보 풀 장부. 구성원 목록이 전부이며 순서는 진입 순서다."""

    members: list[PoolMember]

    @classmethod
    def load(cls, path: Path = POOL_PATH) -> Pool:
        if not path.exists():
            return cls(members=[])
        with path.open() as f:
            record = yaml.safe_load(f)
        return cls(
            members=[
                PoolMember(
                    run_id=m["run_id"],
                    config=m["config"],
                    oof_auc=float(m["oof_auc"]),
                    seeds=[int(s) for s in str(m["seeds"]).split(",")],
                    entered_at=str(m["entered_at"]),
                    reason=m["reason"],
                    evidence=EntryEvidence(**m["evidence"]),
                    judgment=(
                        PoolJudgmentPointer(**m["judgment"])
                        if m.get("judgment") is not None
                        else None
                    ),
                )
                for m in record["members"]
            ]
        )

    def save(self, path: Path = POOL_PATH) -> None:
        record = {
            "members": [
                ({
                    "run_id": member.run_id,
                    "config": member.config,
                    # 작은 채택 문턱과 비교하므로 반올림 없이 전체 정밀도로 남긴다.
                    "oof_auc": float(member.oof_auc),
                    "seeds": ",".join(map(str, member.seeds)),
                    "entered_at": member.entered_at,
                    "reason": member.reason,
                    "evidence": asdict(member.evidence),
                } | (
                    {"judgment": asdict(member.judgment)}
                    if member.judgment is not None
                    else {}
                ))
                for member in self.members
            ]
        }
        with path.open("w") as f:
            yaml.safe_dump(record, f, allow_unicode=True, sort_keys=False, width=100)
