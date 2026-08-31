# 2026-08-30 외부 구성원 증분 조사

조사 기준 시각은 `2026-08-30T12:00:00Z`, 곧 2026-08-30 21:00 KST로 고정했다.
노트북 기준점 `lastRunTime=2026-08-28T00:34:34.393Z`와 자료 기준점 `lastUpdated=2026-08-28T00:21:30.937Z`보다 늦고 기준 시각 이하인 공개 판본만 이번 회차에 넣었다.

## 결론

증분 조사 범위는 완결됐다.
새 감사 기록은 6개이며, 5개가 `자격 있음`, 1개가 `근거 부족`이다.
기존 판본 3 구성원 19개의 현재 공개 판본과 예측 쌍에는 변화가 없어 기존 감사 기록을 그대로 유지했다.
따라서 판본 3 색인은 현행 기록 25개, `자격 있음` 24개, `근거 부족` 1개를 가리킨다.

`sometimessubodh/stacking-9-models-smartphone-addiction-prediction`의 저장 출력에서 cuML RandomForest, LogisticRegression, KNN, MBSGD와 PyTorch MLP의 개별 OOF·시험 예측 쌍 5개를 새로 확인했다.
다섯 쌍은 공식 훈련 자료 691,369행과 시험 자료 296,302행에 맞고 값이 모두 유한하며, 커뮤니티 고정 5분할과 원래 행 순서를 공개 소스와 저장 출력에서 확인할 수 있다.
다섯 설정은 고정 학습 횟수를 사용하고 바깥 검증 자료를 학습 시점 선택에 사용하지 않으므로 모두 `자격 있음`이다.

`michaelqiu0606/s6e8-depth9-pair-te-inputs` 자료 판본 1에는 행 수와 순서가 맞는 OOF·시험 예측 쌍이 있고 재채점 AUC는 0.970516840이다.
하지만 README의 설명 외에 학습 소스, 분할 벡터와 재현 가능한 계보가 없어 현재 계약의 분할과 학습 격리를 확정할 수 없다.
이 쌍은 `근거 부족`으로 종결했다.

## 조사 범위와 완결성

공식 Kaggle CLI 2.2.4로 대회 노트북 전체 목록과 저장 출력 자료가 있는 노트북 목록을 `dateRun` 정렬, 쪽당 100개로 읽었다.
두 목록 모두 첫째 쪽에서 기준점 이전 항목에 도달했으며, 기준 구간에 들어온 노트북은 37개이고 그중 저장 출력 자료가 있는 노트북은 30개였다.
37개 모두의 고정 공개 소스와 입력 연결 정보를 내려받았고, 저장 출력 자료가 있는 30개 모두에 파일 목록 호출을 1.5초 이상 간격으로 실행했다.
파일 목록 호출 실패나 429 응답은 없었다.

자료 검색은 다음 17개 고정 검색어를 `updated` 정렬, 쪽당 100개, 각각 3쪽까지 실행했다.

`s6e8`, `oof`, `smartphone addiction`, `playground s6e8`, `playground-series-s6e8`, `blend members`, `oof library`, `stack members`, `smartphone oof`, `addiction oof`, `s6e8 stack`, `s6e8 oof`, `s6e8 members`, `s6e8 predictions`, `s6e8 test preds`, `s6e8 artifacts`, `s6e8 submission`을 사용했다.

51번의 자료 목록 호출은 모두 완료됐고, 빈 쪽을 포함해 302개 결과와 149개 고유 자료 참조를 반환했다.
기준 구간의 새 자료와 판본이 바뀐 자료를 파일 단위로 확인했다.
새 노트북의 입력 연결 정보는 모두 열거한 뒤, 새 개별 예측 쌍으로 이어지는 연결 자료와 학습 소스를 직접 추적했다.
이미 감사한 장부 입력을 다시 가리키는 결합 노트북은 기존 감사 기록과 고정 판본을 대조했다.

기준 시각 뒤에 실행되거나 갱신된 현재 공개 판본은 목록에 보이더라도 이번 회차에서 제외했다.
따라서 이 보고서의 결론은 정한 기준 시각까지 공개된 판본에만 적용된다.

## 새 감사 기록

아래 쌍 해시는 OOF와 시험 예측을 각각 float64 연속 배열로 정규화하고 순서대로 이어 붙여 계산한 SHA-256이다.
정확 중복은 장부 판본 2 통과 구성원 400개와 판본 3 현행 후보 전체를 대조했다.
새 여섯 쌍에는 정확 중복이 없고 스피어만 0.998 이상 근접 중복도 없다.

| 출처와 구성원 | 판정 | OOF AUC | 쌍 SHA-256 | 가장 가까운 기존 구성원과 스피어만 |
| --- | --- | ---: | --- | ---: |
| `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:cuml_rf` | 자격 있음 | 0.940504102 | `b3e6a5c21a006b6f730170e55a3a37a3f8795b62734d7f12679b4919f9bc8ed4` | `v2:szymon74:et`, 0.985022 |
| `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:cuml_logreg` | 자격 있음 | 0.927826905 | `949716ab6998c5f084a3ea9318a0f103b35c48d27f22bfdfb41cae070e2153a2` | `v2:szymon_weak50:m15`, 0.976079 |
| `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:cuml_knn` | 자격 있음 | 0.929086166 | `6cc0389e39fd8ec6658322cf524d37d6a0b77bf5a86de587be074f77a841a90b` | `v2:szymon74:knn`, 0.956042 |
| `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:cuml_mbsgd` | 자격 있음 | 0.834173902 | `57c44070baf52a346fe78c881f04d47952ed9e1bf29f5736e1f754d01caa85c6` | 새 `cuml_logreg`, 0.829786 |
| `sometimessubodh/stacking-9-models-smartphone-addiction-prediction:torch_mlp` | 자격 있음 | 0.940873574 | `8f3d5ea1e94adbeb6162ec49995803c9f8d88b9d30325dbd343c68374842ca8f` | `v2:szymon74:pubmk_nn`, 0.995113 |
| `michaelqiu0606/s6e8-depth9-pair-te-inputs:depth9_pair_te` | 근거 부족 | 0.970516840 | `1d85e728c61ce6c177c90183b97b77e2bf20ff231a57dac6f9fd8b9bb93462d3` | `v2:adarsh22:gxgbd4`, 0.991578 |

Subodh 노트북의 고정 실행 판본은 `346039237`이고 소스 SHA-256은 `691100dcf6f0b365e4c1a5902e52218797cfe00c73dca19b8e6a2b19087473bb`이다.
저장 `stacking_matrices.pkl`의 원본 SHA-256은 `8c1f7d3e761c1cf66f3095cda530b05f64d8de11de757e87166421a33116c5ad`이다.
공개 소스는 범주 사전을 훈련·시험 특성값 합집합으로 만들고 중앙값과 표준화를 전체 훈련 특성값으로 만들지만 이 과정에서 목표값은 읽지 않는다.
이를 `full_feature_only_preprocessing` 주의 사항으로 기록했다.
MBSGD 열은 양성 확률이 아니라 `model.predict`의 0·1 예측이지만 장부 계약은 유한한 점수 배열을 허용하므로 자격을 바꾸지 않는다.

Michael 자료의 README SHA-256은 `ec1ff5cf164c212335cfa0c748ada21c4ed4f473315524e9e5bccc371e195ee9`이다.
원본 OOF 배열 SHA-256은 `b181851f29c67b14ee1012a723d0456725da77c51b782ec814537925f596e0f1`, 시험 배열 SHA-256은 `7859efd3e05190f89d868cac5a5c226a453d11aaddad9f9452c1ace82b24b4a3`이다.
자료 판본 1은 CC0로 표시돼 있지만 사용 조건과 모델 자격은 별개이므로 계보 부족 판정을 유지했다.

## 대표 제외와 기존 판본 대조

`aadijoshi19/s6e8-mask-augmented-oof-library`의 9개 OOF·시험 쌍은 행, 분할과 해시를 확인했으나 공개된 유일한 학습 구현이 바깥 검증 목표값으로 XGBoost와 LightGBM 조기 종료 시점을 고르므로 모두 제외했다.
이 판정은 [Asterios Rank-Gauss Stack 0.97130 판본의 외부 OOF 편입 가치를 판정한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/518)의 감사 결과와 같다.

Stephen Tarter 계열은 설정 시드가 `[10301, 42, 2026, 777, 888]`이고 첫 시드로 저장한 예측의 분할이 커뮤니티 고정 분할과 다르며 일부 모델은 바깥 검증 조기 종료도 사용하므로 제외했다.
Lam의 노트북은 개별 OOF는 있으나 대응하는 개별 시험 예측이 없고, Nawfeel과 Nikita의 출력은 외부 예측을 다시 학습하거나 결합한 2단 결과다.
나머지 기준 구간 노트북은 제출 파일만 있거나 개별 OOF·시험 예측 쌍을 함께 저장하지 않았다.

판본 3의 기존 공개 노트북 11개는 현재 공개 페이지의 `scriptVersionId`를 다시 대조했다.
11개 모두 감사 기록에 고정한 판본과 같았고, 기존 현행 구성원 19개의 정규화 예측 쌍 해시도 변하지 않았다.
따라서 기존 기록을 새로 만들거나 `supersedes_audit_record_id`로 대체한 항목은 없다.

## 장부 반영과 재현

`scripts/build_external_member_ledger_v3.py`는 공개 노트북 출력 외에 공개 자료 판본도 고정할 수 있고, 허용 목록을 적용한 읽기 방식으로 joblib DataFrame을 배열로 정규화한다.
기존 현행 기록은 공개 판본, 소스 해시와 예측 쌍이 달라진 경우에만 다시 감사한다.
이번 실행은 새 후보 6개만 새 감사 기록으로 만들고 기존 19개 기록을 유지했다.

```text
uv run python scripts/build_external_member_ledger_v3.py fetch --only sometimessubodh/stacking-9-models-smartphone-addiction-prediction michaelqiu0606/s6e8-depth9-pair-te-inputs
uv run python scripts/build_external_member_ledger_v3.py audit
uv run python scripts/build_external_member_ledger_v3.py verify
```

검증 결과는 `현행 기록 25개, 과거 기록 0개, 제자리 수정 없음, 배열 미포함`이다.
외부 원본과 정규화 배열은 `data/` 아래에만 두고 저장소에는 커밋하지 않는다.
