# PROTOTYPE (issue #17): 구조 확인용 뼈대.
"""공유 fold 분할을 한 번 생성해 커밋한다. (#15)

사용법:
    uv run python scripts/make_folds.py

StratifiedKFold(5, shuffle=True, random_state=42)로 artifacts/folds.parquet(id, fold)을 만든다.
이미 파일이 있으면 덮어쓰지 않고 실패한다. 분할은 지도 전체에서 한 번만 정한다.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

OUT = Path("artifacts/folds.parquet")


def main() -> None:
    assert not OUT.exists(), f"{OUT}이 이미 있다. 공유 분할은 다시 만들지 않는다."
    train = pd.read_csv("data/train.csv", usecols=["id", "addicted_label"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = pd.Series(-1, index=train.index, name="fold")
    for i, (_, va_idx) in enumerate(skf.split(train, train["addicted_label"])):
        folds.iloc[va_idx] = i
    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame({"id": train["id"], "fold": folds}).to_parquet(OUT, index=False)
    print(f"{OUT} 생성: {len(train)}행, fold 분포\n{folds.value_counts().sort_index()}")


if __name__ == "__main__":
    main()
