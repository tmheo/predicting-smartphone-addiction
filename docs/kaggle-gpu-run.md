# Kaggle GPU 실행 절차 (#98)

GPU가 필요한 실험(#58 Lookup-Transformer 등)을 Kaggle 노트북에서 실행하고,
그 결과를 실행 기록 묶음으로 로컬 판정 경로에 반입하는 절차다.
용어는 CONTEXT.md의 "실행 기록 묶음"과 "묶음 반입"을 따른다.

## 원칙

- Kaggle에서도 `pipeline.run <config>`를 그대로 실행한다.
  노트북 자체 학습 루프는 만들지 않는다(두 번째 CV 루프 divergence 금지).
- config는 실행 전에 main에 커밋돼 있어야 한다.
  반입이 "실행의 git_commit에 그 config가 그대로 존재하는가"를 검증한다.
- 노트북은 인터넷을 켠 상태로 실행한다(GPU 실험 노트북은 허용, 제출 노트북 아님).

## 노트북 셀 구성

1. 저장소를 커밋 고정으로 받는다.

   ```bash
   !git clone https://github.com/tmheo/predicting-smartphone-addiction.git repo
   %cd repo
   !git checkout <commit>   # 실행할 config가 포함된 커밋
   !pip install -e . -q
   ```

2. 대회 자료를 저장소가 기대하는 상대 경로에 연결한다.
   folds는 저장소에 커밋돼 있으므로 그대로 쓴다.

   ```bash
   !ln -sf /kaggle/input/playground-series-s6e8/train.csv data/train.csv
   !ln -sf /kaggle/input/playground-series-s6e8/test.csv data/test.csv
   !ln -sf /kaggle/input/playground-series-s6e8/sample_submission.csv data/sample_submission.csv
   ```

3. 실험을 실행한다. MLflow sqlite는 저장소 루트(mlflow.db)에 정상 생성된다.

   ```bash
   !uv run python -m pipeline.run configs/expNNN_*.yaml || python -m pipeline.run configs/expNNN_*.yaml
   ```

   stdout의 `run_id=`를 확보한다.

4. 마지막 셀에서 묶음을 export하고 /kaggle/working으로 옮긴다.

   ```bash
   !python -m pipeline.bundle export <run_id> --out /kaggle/working/expNNN.bundle.zip
   ```

5. 노트북 산출물에서 zip을 내려받는다.

## 로컬 반입

```bash
uv run python -m pipeline.bundle import <zip>
```

반입이 입력 해시(train·test·folds), 출처(commit 존재, config 동일, git_dirty=False),
시드별 OOF 재채점을 전부 통과하면 로컬 MLflow에 정상 run으로 재생되고 run_id를 출력한다.
이후 스크리닝·확정 재검증·풀 진입은 로컬 실행과 똑같이 그 run_id로 수행한다:

```bash
uv run python -m pipeline.compare <반입 run_id>
uv run python -m pipeline.pool <반입 run_id>
```

## 주의

- 반입 거부는 전부 이유가 출력된다. 해시 불일치는 자료·fold가 다른 것이고,
  재채점 불일치는 실행 환경이 어긋난 것이므로 노트북에서 재실행으로 다룬다.
- 기록 규약 확장(#98) 이전 실행(시드별 OOF 산출물 없음)은 export할 수 없다.
- 같은 묶음의 중복 반입은 거부된다. 다시 반입하려면 노트북에서 재실행해 새 묶음을 만든다.
