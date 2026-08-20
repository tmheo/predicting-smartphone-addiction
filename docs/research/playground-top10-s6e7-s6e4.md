# Playground Series S6E7-S6E4 상위권 해법과 댓글 조사

## 조사 범위와 방법

이 문서는 GitHub 이슈 [리서치: S6E7-S6E4 상위권 해법 글과 댓글의 재사용 가능한 인사이트 조사](https://github.com/tmheo/predicting-smartphone-addiction/issues/308)의 결과다.
조사 시점은 2026-08-20 JST다.

각 대회는 사용자가 지정한 `competitionWriteUps` 범주와 득표순 화면에서 시작했고, 대회 주소의 slug만 S6E7, S6E6, S6E5, S6E4로 바꿨다.
Kaggle API, 리더보드 화면, 검색 엔진, 직접 HTTP 요청은 사용하지 않았다.
`agent-browser`의 전용 세션과 Kaggle 도메인 제한 및 내용 경계를 사용해 실제 목록, 글 본문, 댓글과 답글을 읽었다.
득표순은 발견 순서로만 사용했다.
최종 순위는 글 제목이나 본문 상단의 `Solution Writeup · Nth place` 표기만 사용했다.
Private 최종 1위부터 10위라고 표기된 공식 해법 글을 모두 포함했고, 빠진 순위를 다른 글로 채우지 않았다.

목록 화면의 공식 해법 62건을 전수 확인했고, 이 중 19건이 포함 조건을 충족했다.
포함 글 목록에 표시된 댓글과 답글은 모두 합쳐 161건이었다.
실제 글 화면에서 읽을 수 있는 댓글과 답글 160건을 끝까지 확인했고, S6E4 1위 글의 삭제 댓글 자리표시자 1건도 확인했다.
글 화면의 `N Comments`는 대화 묶음 수를, 목록의 댓글 수는 답글을 포함한 게시물 수를 세는 경우가 있어 두 수가 다를 수 있었다.
별도 `더 보기`나 다음 댓글 페이지가 남은 글은 없었다.

## 전수 조사 결과

| 대회 | 과제 | 지표 | 공식 해법 글 | 포함 글 | 포함 순위 | 빠진 순위 | 포함 글 댓글·답글 |
| --- | --- | --- | ---: | ---: | --- | --- | ---: |
| S6E7 | 학생 건강 위험 3계급 분류 | Balanced Accuracy | 12 | 2 | 2, 4 | 1, 3, 5-10 | 6 |
| S6E6 | 천체 종류 3계급 분류 | Balanced Accuracy | 24 | 5 | 1, 3, 6, 8, 9 | 2, 4, 5, 7, 10 | 27 |
| S6E5 | F1 피트 스톱 이진 예측 | ROC AUC | 13 | 7 | 1, 2, 4, 5, 7, 8, 10 | 3, 6, 9 | 74 |
| S6E4 | 관개 필요도 3계급 분류 | Balanced Accuracy | 13 | 5 | 1-5 | 6-10 | 54 |
| 합계 |  |  | 62 | 19 |  |  | 161 |

Kaggle 화면에는 각 글의 작성자만 표시됐고 별도 팀명은 어느 포함 글에도 기재되지 않았다.
아래 표의 작성자는 화면의 `AUTHOR`이며 팀은 모두 미표기다.

| 대회 | 최종 순위 | 작성자 | 해법 글 | 댓글·답글 |
| --- | ---: | --- | --- | ---: |
| S6E7 | 2 | kava1 | [2nd Place Solution: Trusting CV & Mathematical Precision](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/2nd-place-solution) | 6 |
| S6E7 | 4 | Ricky | [4th Place: From #414 to #4](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/4th-place-from-414-to-4-trusting-oof-when-the) | 0 |
| S6E6 | 1 | Optimistix | [1st Place: Mission 300+ Accomplished](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/1st-place-mission-300-accomplished) | 17 |
| S6E6 | 3 | nybbler | [3rd place: same light, different space](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/same-light-different-space) | 0 |
| S6E6 | 6 | Andreas Palmgren | [6th Place Solution: Trusting The OOF Plateau](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/6th-place-solution-trusting-the-oof-plateau) | 4 |
| S6E6 | 8 | Jerry | [8th Place Solution](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/8th-place-solution) | 6 |
| S6E6 | 9 | Vaibhav Nakrani | [9th place solution](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/9th-place-solution) | 0 |
| S6E5 | 1 | Optimistix | [1st Place: By the skin of my teeth](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/1st-place-by-the-skin-of-my-teeth) | 33 |
| S6E5 | 2 | Chris Deotte | [2nd Place: Autonomous Codex Yolo](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/2nd-place-autonomous-codex-yolo) | 26 |
| S6E5 | 4 | Mahog | [4th place: 5 day rush](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/4th-place-5-day-rush) | 7 |
| S6E5 | 5 | Data User | [5th place solution: a 99-model logit stack](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/5th-place-solution-a-99-model-logit-stack) | 4 |
| S6E5 | 7 | Jerry | [7th place solution](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/7th-place-solution) | 0 |
| S6E5 | 8 | Masaya Kawamata | [8th Place: L5 Ensemble](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/l5-ensemble) | 4 |
| S6E5 | 10 | Don Mani | [Stacking stacked predictions](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/stacking-stacked-predictions) | 0 |
| S6E4 | 1 | cstdy | [1st place: One vs Rest + Multiclass Models](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/1st-place-one-vs-rest-approach) | 23 |
| S6E4 | 2 | Chris Deotte | [2nd Place: Claude Code and Codex, GPU LogReg](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/2nd-place-claude-code-and-codex-gpu-logreg) | 14 |
| S6E4 | 3 | r0tor | [Error Diversity Matters: 200-model stacking solution](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/error-diversity-matters-200-model-stacking-soluti) | 4 |
| S6E4 | 4 | Optimistix | [4th place: more ensemblers than models](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/4th-place-more-ensemblers-than-models) | 13 |
| S6E4 | 5 | Topiast | [5th Place: AI for large scale experimentation](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/5th-place-solution-writeup-ai-for-large-scale-ex) | 0 |

## 대회별 해법과 댓글 근거

### S6E7: Predicting Student Health Risk

이 대회는 약 69만 학습 행의 불균형 3계급을 Balanced Accuracy로 평가했다.

#### 2위 kava1

- **출처 사실:** LightGBM, XGBoost, CatBoost, FT-Transformer, RealMLP, HistGradientBoosting을 포함한 18개 기본 예측을 썼다.
- **출처 사실:** 계급별로 다른 54개 비중을 SLSQP로 최적화하되 목적함수는 OOF 다계급 log loss였고, 마지막에 Nelder-Mead로 계급별 확률 배수를 맞춰 Balanced Accuracy를 직접 높였다.
- **보고 결과:** 균등 비중 OOF log loss `0.105495`가 `0.086310`으로 낮아졌고, 원시 argmax Balanced Accuracy `0.889132`가 최종 `0.950737`로 높아졌다고 보고했다.
- **외부 의존성:** Masaya Kawamata의 FT-Transformer와 XGBoost OvR, Rob Schieber의 LightGBM을 비롯한 다수 공개 노트북의 예측에 크게 의존했다.
- **댓글 보충:** 작성자는 계급별 부분 표본 재학습이 확률 분포를 왜곡해 log loss를 해칠 수 있으므로 전체 자료 학습 뒤 계급별 결합 비중을 조정하는 편이 안전했다고 답했다.
- **댓글 보충:** 작성자는 GBDT 계열에는 결측을 그대로 넣었고, 신경망 계열은 공개 기준 구성의 단순 대체 또는 타깃 인코딩을 따랐다고 답했다.
- **재현성 한계:** 최종 18개 예측 파일의 고정 판본, 공통 fold, SLSQP 초기값과 수렴 조건이 없어 글만으로 동일 결과를 재현할 수 없다.
- **조사자 판단:** S6E8은 이진 ROC AUC라서 계급별 argmax 배수 조정은 순위를 바꾸지 않는 단조 변환인 한 점수에 기여하지 않으며, 현행 순위와 logit 이중 표현 nested OOF 결합을 대체할 근거가 없다.

#### 4위 Ricky

- **출처 사실:** 13개 원시 특성에 각 정확값의 3계급 타깃 비율 39개를 더하고, 7-fold마다 독립 초기화 4개를 평균한 FT-Transformer 한 계열을 썼다.
- **출처 사실:** 바깥쪽 fold 학습 행의 인코딩도 그 안의 OOF로 만들었고, 검증과 시험 행은 바깥쪽 학습 구간 전체의 평균표로 변환했다.
- **출처 사실:** 일반 argmax 대신 학습 계급 사전확률로 나눈 확률의 argmax를 써 Balanced Accuracy의 의사결정 규칙을 맞췄다.
- **보고 결과:** OOF Balanced Accuracy는 일반 argmax `0.8918688`에서 사전확률 보정 `0.9506326`으로 높아졌고, Public `0.95094` 414위에서 Private `0.95084` 4위가 됐다.
- **교차 확인:** 정확값 타깃 인코딩은 7만 행 약식 검사에서 약 `-0.0017`이었지만 전체 행 짝비교에서 약 `+0.0012`였고, XGBoost 독립 대조도 `0.9489047`에서 `0.9495564`로 같은 방향이었다.
- **외부 의존성과 재현성:** Masaya Kawamata의 공개 FT-Transformer 구현과 저장 OOF 및 시험 확률을 썼지만, 작성자는 fold별 인코딩, 지표, 최종 CSV를 다시 계산해 제출 파일과 정확히 일치시켰다고 밝혔다.
- **반증과 실패:** 전체 PLR은 약 `+0.00003`, 추가 중간 해상도 타깃 인코딩은 `-0.00004`, kNN 밀도는 `+0.00014`라 최종 구성에서 뺐다.
- **댓글:** 댓글은 없었다.
- **조사자 판단:** 정확값 타깃 인코딩과 fold 내 초기화 평균은 S6E8의 `exp070_cat_exact_cats`, Lookup 계열, RealMLP 계열에서 이미 검증 또는 반영됐으며, 작은 표본으로 고카디널리티 통계를 거르지 말라는 교훈만 추가로 강화한다.

### S6E6: Predicting Stellar Class

이 대회는 천체 종류 3계급을 Balanced Accuracy로 평가했다.

#### 1위 Optimistix

- **출처 사실:** XGBoost, LightGBM, CatBoost, RealMLP, TabM, ExtraTrees, HistGradientBoosting, RandomForest, YDF, FT-Transformer, TabNet과 여러 자동화 학습 도구를 사용했다.
- **출처 사실:** 200개가 넘는 특성 변형도 만들었고, logit 로지스틱 회귀와 단순 신경망 결합기가 가장 좋았다고 썼다.
- **외부 의존성:** 공개 노트북의 모델과 특성을 재사용했고 AutoGluon, Light AutoML, FLAML, PyTabKit을 사용했다.
- **보고 결과:** 49개 결합은 CV `0.970510`, Public `0.97179`였고, 최종 당선된 89개 신경망 결합은 CV `0.970598`, Public `0.97172`였다고 보고했다.
- **본문 불일치:** 78개 로지스틱 결합의 CV를 `0.97573`이라고 적었는데 주변 수치와 맞지 않아 오타 가능성이 있으며, 조사자가 임의로 고치지 않았다.
- **반증:** 모델 수가 100개를 넘어서자 오히려 성능이 낮아져 이전 풀로 돌아갔다고 밝혔다.
- **댓글 보충:** 새 OOF는 새 하이퍼파라미터가 아니라 새 특성 표현으로도 만들 수 있고, 큰 앙상블에서는 개별 하이퍼파라미터 최적화의 중요도가 줄어든다고 답했다.
- **댓글 보충:** 다른 3위 참가자도 하이퍼파라미터 탐색보다 표현과 오류의 폭이 중요했다고 덧붙였다.
- **재현성 한계:** 최종 89개 구성원 장부와 공통 fold, 결합기 설정이 공개되지 않아 정확 재현은 불가능하다.
- **조사자 판단:** 모델 수 확대 자체가 목적이 아니라는 직접 반증이며, S6E8의 29개 후보 풀을 계속 늘리기보다 중복 게이트와 nested OOF 선택을 유지해야 한다.

#### 3위 nybbler

- **출처 사실:** 약 180개 모델을 결합했고, 강한 모델들이 같은 오류를 내므로 약하지만 다른 GMM Bayes 분류기 같은 구성원이 유용했다고 썼다.
- **출처 사실:** 합성 생성 과정이 광도 정보와 하늘 평면 좌표는 보존했지만 redshift가 나타내는 방사 거리는 흐렸다고 진단했다.
- **출처 사실:** 적경과 적위를 HEALPix 구역 번호로 바꿔 공간 타깃 인코딩에 사용했다.
- **외부 의존성:** SDSS17의 천문 영역 지식과 공개 공간 kNN 노트북을 참고했다.
- **재현성 한계:** 여행 중 기억에 의존해 작성했고 원 파일, CV 수치, 설정, 코드를 제시하지 않았다고 직접 밝혔다.
- **댓글:** 댓글은 없었다.
- **조사자 판단:** HEALPix 자체는 S6E8에 옮길 수 없지만, 생성 과정이 보존한 결합 구조와 흐린 구조를 구분한 뒤 특성을 설계한다는 절차는 현재 합성 생성 포렌식 방향과 일치한다.

#### 6위 Andreas Palmgren

- **출처 사실:** 5-fold로 만든 92개 OOF 확률을 결합했고, RealMLP 24개, XGBoost 19개, CatBoost 12개가 큰 비중을 차지했다.
- **보고 결과:** 최종 OOF `0.970718`, Public `0.97130`, Private `0.97054`였고, 풀에서 가장 강한 단일 공개 스택 OOF `0.970350`보다 전체 결합이 약 `0.000368` 높았다.
- **외부 의존성:** 공개 OOF 결합기, Chris Deotte의 GPU 로지스틱 결합기, 여러 공개 RealMLP와 트리 예측을 같은 로컬 OOF 검사에 넣었다.
- **재현성:** 공개 노트북을 저장 예측으로 그대로 쓰기보다 로컬 저장소에 옮겨 OOF와 시험 예측을 등록하고 포함 또는 제외 사유를 기록했다고 설명했다.
- **댓글 보충:** 댓글은 `0.0004`에 가까운 작은 한계 이득이 최상단 대규모 앙상블의 현실임을 강조했으며, 작성자는 수치 정정 없이 동의했다.
- **조사자 판단:** 큰 풀의 기대 이득 규모와 포화 상태를 보여 주며, S6E8의 후보 풀은 구성원 단독 점수보다 nested OOF 한계 기여로 읽어야 한다.

#### 8위 Jerry

- **출처 사실:** Balanced Accuracy의 모집단 최적 의사결정과 같은 argmax를 갖는 엄밀 점수로 계급 가중 log loss를 제시했다.
- **출처 사실:** 표본 가중치를 지원하는 모델은 역계급빈도 가중 학습을, 지원하지 않는 RealMLP와 TabPFN은 일반 확률을 사전확률로 나눈 뒤 다시 정규화하는 경로를 썼다.
- **출처 사실:** 최종 제출은 60개 기본 모델과 27개 결합기 출력을 입력으로 한 로지스틱 회귀였다.
- **외부 의존성:** Chris Deotte의 GPU 결합 노트북과 다수 공개 기본 모델을 사용했다고 밝혔지만 구성원별 판본은 열거하지 않았다.
- **재현성:** 수학 전개는 글에 완결돼 있지만 최종 OOF 수치, 모델 장부와 코드가 없어 결과 재현성은 제한된다.
- **댓글:** 6건은 축하와 방법론 인용이었고 작성자의 추가 설명은 없었다.
- **조사자 판단:** 불연속 Balanced Accuracy 대신 엄밀 점수로 검증한다는 교훈은 강하지만, S6E8의 평가지표 자체가 이미 엄밀하고 순위 기반인 ROC AUC라 별도 가중 log loss 실험으로 옮길 이유는 없다.

#### 9위 Vaibhav Nakrani

- **출처 사실:** 63개 서로 다른 기본 모델의 확률 위에 L2 로지스틱 회귀를 쌓았다.
- **외부 의존성:** 공개 노트북과 Chris Deotte의 기준 노트북을 출발점으로 썼지만 최종 63개 중 외부 구성원 장부는 공개하지 않았다.
- **보고 결과:** 가장 강한 단일 RealMLP는 약 `0.9693`, 최종 결합은 CV `0.970355`, Public `0.97121`이었다.
- **실패 근거:** cleanlab 기반 행 제거, flux와 luptitude 및 RBF 표현, 더 큰 RealMLP, 더 많은 epoch, Optuna, 100개 bag, 의사라벨, 생성형 사전학습, 원본 좌표 라벨 복구가 모두 무효 또는 하락이었다.
- **다양성 반증:** RBF와 Nyström 표현은 오류 상관을 약 `0.48-0.54`까지 낮췄지만 단독 Balanced Accuracy가 약 `0.94`라 결합을 개선하지 못했다.
- **재현성:** 총 144개 실험, 최종 제출은 91번째 실험이며 가능한 한 실험당 한 변화만 적용하고 발견과 제안을 기록했다고 밝혔다.
- **댓글:** 댓글은 없었다.
- **조사자 판단:** 낮은 상관만으로 다양성 구성원을 채택해서는 안 된다는 반증이며, 현행 champion - 0.01 진입 하한과 nested OOF 판정을 지지한다.

### S6E5: Predicting F1 Pit Stops

이 대회는 피트 스톱 발생 여부를 ROC AUC로 평가해 S6E8과 지표가 같다.

#### 1위 Optimistix

- **출처 사실:** XGBoost, LightGBM, CatBoost, RealMLP, TabM, HistGradientBoosting, RandomForest, YDF, FT-Transformer, MLP-PLR와 여러 자동화 학습 도구를 썼다.
- **출처 사실:** 200-400개 특성을 가진 여러 표현을 만들었고, AutoGluon 결합기가 가장 좋았으며 logit 로지스틱 회귀가 근소하게 뒤따랐다.
- **외부 의존성:** 여러 공개 RealMLP와 특성 아이디어, AutoGluon, Light AutoML, FLAML, PyTabKit을 사용했다.
- **출처 사실:** 원본 자료와 대회 자료의 차이가 가장 큰 `Driver`를 뺀 모델, 원본 행을 쓰지 않은 모델, 원본 행 표본 비중 변형을 다양성 축으로 만들었다.
- **보고 결과:** L2 OOF 4개를 포함한 186개 logit 결합은 Public `0.95487`, Private `0.95503`이었고, L1 182개만 쓴 결합은 Public `0.95480`, Private `0.95495`였다.
- **보고 결과:** 마지막에 AutoGluon과 logit 결합을 50:50으로 섞은 제출이 Public `0.95488`, Private `0.95503`으로 1위가 됐다.
- **반증:** FeatureTools는 낮은 AUC였고 AutoFeat는 12시간 뒤에도 유용한 결과를 내지 못했으며, 작성자는 L2 OOF 포함이 누출 위험을 가진다고 직접 경고했다.
- **댓글 보충:** 대규모 풀에서는 완전한 하이퍼파라미터 탐색보다 기존 설정과 공개 설정을 바꾸어 여러 변형을 만드는 속도와 다양성이 중요하다고 답했다.
- **댓글 보충:** 공개 노트북은 보통 다시 학습하지만 같은 CV와 신뢰할 만한 작성자의 산출물이거나 시간이 부족하면 예외적으로 그대로 쓴다고 답했다.
- **댓글 보충:** 186개 노트북 상태를 자동화 도구가 통합 관리한 것이 아니라 작성자 자신이 중간 관리층이었다고 밝혔다.
- **조사자 판단:** ROC AUC에서 서로 다른 확률 눈금의 결합과 OOF 자산화는 S6E8에 직접 전이되지만, 현 저장소는 이를 이미 후보 풀과 rank-logit nested OOF로 더 엄격하게 구현한다.

#### 2위 Chris Deotte

- **출처 사실:** 강한 여섯 계열인 RealMLP, XGBoost, CatBoost, TabM, LightGBM, TabICL을 집중 개선하고 31개가 넘는 다른 계열은 소수씩 만들어 다양성을 확보했다.
- **출처 사실:** 최종 제출은 218개 L1 모델의 logit을 NVIDIA cuML 로지스틱 회귀로 결합했으며, 의사라벨이나 모델 예측 위의 추가 모델 학습은 쓰지 않았다.
- **보고 결과:** 최강 단일 RealMLP OOF `0.954426`, XGBoost `0.953553`, CatBoost `0.953404`, TabM `0.953371`을 보고했고 최종 선택 제출의 Private는 `0.95502`였다.
- **외부 의존성과 계산량:** 공개 노트북 재구현, 과거 대회 코드 변환, 12개 반복 개선 묶음, 다양한 새 모델 작성에 4대 A100과 약 48시간의 Codex 작업을 사용했다.
- **댓글 보충:** 기본적으로 진입 하한을 두고 모든 서로 다른 모델을 로지스틱 회귀에 넣으며, 이 대회에서는 약 `0.930`을 하한으로 생각했고 실수로 넣은 `0.893791` LNN은 제거했어야 한다고 답했다.
- **댓글 보충:** 단순 시드 변경이나 특성 한두 개 차이는 다양성으로 세지 않고, 다른 공개 계보, 대규모 특성 변경, 다른 모델 구성에서 새 예측을 만든다고 답했다.
- **댓글 보충:** 타깃 인코딩은 중첩 fold로 만들고 L2 모델을 후보로 넣을 때도 L1을 중첩 5-fold로 만들어 누출을 막는다고 답했다.
- **댓글 보충:** 공개 노트북은 자신의 5-fold로 다시 실행해 OOF와 시험 예측을 만들며, A100에서 대부분 15-30분, 모두 1시간 안에 끝났다고 답했다.
- **반증:** 로지스틱 회귀에 모든 모델을 넣는 관행은 이전 대회 사후 결과에 근거한 작성자 경험이며, 이 글은 218개 전체와 선택 부분집합의 nested OOF 대조를 제공하지 않는다.
- **조사자 판단:** 공개 구현을 공통 fold로 다시 실행하고 누출 안전성을 확인하는 절차는 그대로 채택할 만하지만, 외부 예측 제외와 3시드 확정을 둔 현 저장소가 이미 더 엄격하다.

#### 4위 Mahog

- **출처 사실:** count 인코딩, 산술 상호작용, 일부 쌍 타깃 인코딩으로 만든 XGBoost와 공개 신경망 OOF를 힐클라이밍으로 골랐다.
- **보고 결과:** 최강 XGBoost는 CV `0.95378`, Public `0.95313`, Private `0.95346`이었고, 최종 힐클라이밍은 CV `0.95529`, Public `0.95469`, Private `0.95488`이었다.
- **보고 결과:** 다른 참가자의 결합과 섞은 최종 제출은 Public `0.95471`, Private `0.95490`이었으며 그 외부 결합이 없으면 6위였다고 밝혔다.
- **댓글 보충:** 5일 동안 특성 표현, 모델 계열, 하이퍼파라미터의 곱으로 265개 OOF를 만들었고 힐클라이밍이 57개를 골랐다고 답했다.
- **재현성 한계:** 265개 장부, 힐클라이밍 가중치, 외부 결합 산출물의 고정 판본이 없다.
- **조사자 판단:** 짧은 기간의 대량 조합 탐색이 가능함을 보여 주지만, 선택과 평가를 같은 OOF에서 하면 낙관될 수 있으므로 현행 nested OOF 부분집합 선택보다 우선할 이유는 없다.

#### 5위 Data User

- **출처 사실:** 99개 기본 모델의 OOF 확률을 잘린 logit으로 바꾸고, fold별 sklearn L2 로지스틱 회귀를 맞춘 뒤 전체 자료 재학습과 시험 예측 5시드 평균을 사용했다.
- **보고 결과:** 최종 결합 OOF AUC `0.95536`, fold 평균 `0.95537`, fold 표준편차 `0.00081`이었다.
- **출처 사실:** 4,851개 모델 쌍의 평균 스피어만 상관은 `0.950`이었고, 약한 Deep FFM과 BART 및 Nyström 계열은 평균 상관 `0.85-0.91`로 결합에 새 방향을 제공했다고 주장했다.
- **반증:** 가장 다른 Deep FFM, GRU, LNN은 시험 예측 상위 꼬리 분포 이동에도 걸려, 다양성에 분포 전이 위험이 따른다고 밝혔다.
- **반증:** OOF가 더 높은 힐클라이밍보다 평평한 L2 결합을 보수적으로 골라 5위가 됐으며, 사후 Private에서는 힐클라이밍 계열이 2-3위 수준이었다.
- **출처 사실:** 서로 눈금이 다른 logit 결합과 힐클라이밍을 확률 평균하지 않고 순위 평균하면 사후 Private `0.95505`로 더 좋았다고 보고했다.
- **외부 의존성:** 공개 RealMLP, 여러 트리와 특성 아이디어를 자신의 fold와 고정 시드로 다시 구현했다.
- **댓글 보충:** 공개 노트북은 한 기법만 가져와 자신의 파이프라인에 다시 만들고, fold 불일치, fit-before-split, 타깃 인코딩 누출, 조기 종료와 미수렴을 검사한다고 답했다.
- **조사자 판단:** ROC AUC에서 눈금이 다른 예측을 순위로 맞추는 것은 직접 전이 가능하지만 S6E8의 현 결합기는 이미 구성원별 순위와 logit을 함께 사용한다.

#### 7위 Jerry

- **출처 사실:** 직접 또는 공개에서 얻은 약 25개 모델과 AutoGluon `best_v150` 50개, `best` 101개를 합쳐 176개 OOF를 만들었다.
- **출처 사실:** 7개와 50개 결합기를 다시 결합하고 일부 시험 행을 의사라벨로 넣은 다단 결합을 공개 결합과 섞었다.
- **보고 결과:** 자체 AutoGluon 결합 CV `0.95453`, 공개 결합 CV `0.95454`, 두 예측의 균등 결합 CV `0.95463`을 보고했다.
- **외부 의존성:** yekenot의 RealMLP와 공개 `0.95454` 결합을 핵심 입력으로 썼다.
- **재현성 한계:** 결합 단계별 fold 경계, 의사라벨 선택과 L2 이상의 OOF 생성 방식이 제시되지 않아 깊은 결합의 누출 여부를 검증할 수 없다.
- **댓글:** 댓글은 없었다.
- **조사자 판단:** 깊은 결합이 S6E3과 비슷한 대회에서 작동했다는 사례지만, 완전한 nested OOF가 없는 보고 점수는 S6E8 실험 발주의 채택 근거가 될 수 없다.

#### 8위 Masaya Kawamata

- **출처 사실:** 5-fold 123개, 7-fold 11개, 10-fold 13개 OOF로 각각 4단계 결합을 만들고 세 결과를 균등 평균해 5단계 결합이라고 불렀다.
- **보고 결과:** 세 결합의 CV는 `0.95510`, `0.95505`, `0.95503`이었고 최종 평균은 Public `0.95462`, Private `0.95487`이었다.
- **출처 사실:** L2에는 원 확률, logit, 순위, 요약 통계, 쌍 차이, AUC 비중 결합을 사용했고 L4에는 L2와 L3 교사 OOF를 학습하는 자기증류 학생 모델을 사용했다.
- **출처 사실:** 원본 자료를 추가 행으로 쓰는 표현과 원본 자료로 열 통계를 만드는 표현이 서로 다른 신호를 줬다고 썼다.
- **반증:** 계산 자원 때문에 nested CV를 쓰지 못했고 모든 CV가 낙관적일 수 있다고 작성자가 직접 밝혔다.
- **외부 의존성과 계산량:** 공개 RealMLP가 기반이며, 여러 계열에 25-150회의 Optuna 탐색을 수행했다.
- **댓글:** 댓글 4건은 축하와 방법론 평가였고 작성자 보충은 없었다.
- **조사자 판단:** 사용자가 주목한 깊은 결합의 독립 사례지만, 작성자의 낙관 경고 때문에 구조를 그대로 복사하지 말고 깊이 증가분 전체를 nested OOF로 측정해야 한다.

#### 10위 Don Mani

- **출처 사실:** 기본 모델, 1단 결합, 공개 결합, 2단 결합을 다시 cuML 로지스틱 회귀에 넣는 계층형 구조를 사용했다.
- **외부 의존성:** 공개 노트북 예측을 OOF 검증 뒤 추가했고, 공개 AutoGluon과 TabM 노트북을 프로젝트 링크로 제시했다.
- **보고 결과:** 최고 CV 구성은 221개로 `0.955630`, 실제 선택 제출은 129개로 `0.955294`였으며 더 작은 구성이 Private에서 더 잘 일반화했다.
- **출처 사실:** 58개가 아니라 수백 개의 예측을 재학습하지 않고 OOF 자산으로 다루며 상관, 결합기와 단계 구조를 반복 탐색했다.
- **반증:** 최고 CV 구성은 AutoGluon과 여러 세대의 결합 예측에 크게 의존했고, 최종 로지스틱 회귀가 큰 음수 계수로 중복 신호를 상쇄해야 했다.
- **재현성 한계:** 221개와 129개 구성원 목록, 각 단계 fold, 선택 규칙과 제출 점수가 공개되지 않았다.
- **댓글:** 댓글은 없었다.
- **조사자 판단:** 더 깊고 큰 결합이 자동으로 낫지 않다는 반례이며, 단계 하나를 추가할 때마다 현재 단순 결합 대비 nested OOF 증가분을 따로 요구해야 한다.

### S6E4: Predicting Irrigation Need

이 대회는 `Low`, `Medium`, `High`의 불균형 3계급을 Balanced Accuracy로 평가했다.

#### 1위 cstdy

- **출처 사실:** `Low` 대 나머지를 먼저 예측하고, 첫 모델의 예측에 따라 `Medium` 대 `High`를 구분하는 두 이진 분류기의 확률을 3계급 확률로 조합했다.
- **출처 사실:** Low와 High가 거의 직접 혼동되지 않는 오류 행렬 관측이 문제 분해의 근거였다.
- **출처 사실:** 공개 OOF와 자체 6개 모델을 포함한 61개 예측을 사용했고, 모두 5-fold로 맞춘 뒤 logit 변환과 LogisticRegressionCV로 결합했다.
- **외부 의존성:** 30개 예측을 제공한 wguesdon을 비롯해 여러 참가자의 공개 OOF와 특성 표현에 의존했다.
- **보고 결과:** 두 단계 XGBoost와 RealMLP가 각각 약 `0.9805`, 최종 결합 CV가 `0.98155`였다고 보고했다.
- **출처 사실:** 최종 계급은 OOF Balanced Accuracy로 맞춘 두 가지 탐욕 문턱 탐색으로 정했다.
- **댓글 보충:** 여러 노트북에서 OOF와 시험 예측을 저장해 로컬 폴더나 Kaggle 자료 묶음으로 모은 뒤 별도 결합 노트북을 실행했다고 답했다.
- **댓글 보충:** 비슷한 CV의 최종 후보가 Private `0.981-0.9815`에 퍼졌고, 시험에서 예측한 High 개수가 다른 두 제출을 골라 하나가 잘 맞았다고 답했다.
- **댓글 한계:** 댓글 23건 가운데 읽을 수 없는 삭제 댓글 자리표시자 1건이 있었다.
- **조사자 판단:** 오류 행렬 구조에 따른 문제 분해는 강하지만 S6E8은 이진 문제라 추가 분해 대상이 없고, OOF 전체에서 문턱을 고른 점수는 선택 편향 통제가 부족하다.

#### 2위 Chris Deotte

- **출처 사실:** 전 달의 150개 모델 스크립트를 비슷한 새 대회로 바꾸고 GPU에서 실행해 3일 만에 강한 결합을 만들었다.
- **외부 의존성:** 외부 예측보다 자신의 전 달 코드 자산과 Claude Code 및 Codex를 사용했으며, 새 대회의 최종 구성원 목록은 공개하지 않았다.
- **출처 사실:** Balanced Accuracy에 맞춘 진짜 다항 로지스틱 회귀와 계급 비중을 구현하기 위해 PyTorch의 단일 선형층, 편향 제외 L2 감쇠, 표본 비중 교차 엔트로피를 사용했다.
- **보고 결과:** 3일 차 구성은 CV `0.98130`, Public `0.98182`, Private `0.98160`이었고, 최종 구성은 CV `0.98170`, Public `0.98195`, Private `0.98151`이었다.
- **반증:** 27일을 더 써 CV를 높였지만 Private는 약 `0.0001` 낮아졌다.
- **댓글 보충:** cuML의 다계급 출력은 OvR 로지스틱 세 개이며 자신이 원한 하나의 다항 softmax와 계급 가중 손실이 아니라고 설명했다.
- **댓글 보충:** 125개 L1 전체가 아니라 전진 선택으로 약 20개를 고른 같은 결합기는 CV `0.98175`, Public `0.98182`, Private `0.98172`였지만 최종 제출로 고르지 못했다고 답했다.
- **댓글 보충:** 30분간 대회 유사성과 특성 변환 계획을 먼저 합의하고 추적 문서, 파일명 규약, OOF와 시험 예측 규약을 만든 뒤 150개 스크립트를 자동 변환했다고 답했다.
- **재현성:** GPU 로지스틱 핵심 손실과 정규화 코드는 글에 있지만 최종 125개 구성원과 전체 코드는 글 작성 시점에 공개되지 않았다.
- **조사자 판단:** 빠른 코드 이식보다 중요한 근거는 공통 OOF 규약과 단순 선형 결합이며, 현 S6E8 저장소가 이미 이 원칙을 구현한다.

#### 3위 r0tor

- **출처 사실:** 최종 후보 풀은 XGBoost 28개, LightGBM 16개, CatBoost 19개, RealMLP 7개, TabM 49개를 비롯해 총 203개였고, 일부 SVM, YDF, GraphSAGE는 최종에서 제외했다.
- **출처 사실:** 기본 예측의 3계급 확률을 입력으로 한 LightGBM 결합기가 선형 결합과 힐클라이밍보다 좋았다고 보고했다.
- **출처 사실:** 전기전도도의 역수인 비저항, 수치 구간화, 주기 변환, 여러 신경망과 선형 모델 표현을 오류 다양성 목적으로 만들었다.
- **보고 결과:** 주요 계열에 Trompt와 Keras를 더해 Public 약 `0.98121`, HistGradientBoosting과 RandomForest를 더해 약 `0.98145`, 로지스틱과 SVM 계열을 더해 약 `0.98201`이라고 서술했다.
- **본문 불일치:** 마지막 절은 SVM을 최종 단일 결합기에서 제외했다고도 하고 SVM 추가가 가장 큰 Public 증가를 만들었다고도 해 최종 포함 범위가 일관되지 않는다.
- **반증:** YDF와 GraphSAGE를 더하면 Public이 약 `0.98201`에서 `0.98130`으로 낮아져 제외했고, SVM은 CV를 높이지만 Public을 크게 낮추며 모델당 2시간 이상 걸렸다고 썼다.
- **댓글 보충:** 공개 `.npy` OOF와 시험 예측을 모아 모양, 계급 순서, CV와 결합기 중요도를 확인했고, 모델명, 경로, CV, Public, 계급 순서와 결합 개선 여부를 일반 텍스트 표로 관리했다고 답했다.
- **재현성 한계:** 구성원 선택에 Public과 LightGBM gain을 함께 사용했고 nested OOF가 없어, 보고된 한계 기여를 누출 없는 개선으로 해석할 수 없다.
- **조사자 판단:** 약한 선형 모델이 강한 트리 결합을 보완할 수 있다는 관측은 S6E8의 정확값 one-hot 로지스틱 구성원으로 이미 반영됐고, 비선형 결합기는 현 저장소의 `xgb_rank_logit` 대조에서 별도로 판단할 수 있다.

#### 4위 Optimistix

- **출처 사실:** 12-14개 모델 위에 최대 11개 L1 결합기와 4개 L2 결합기를 만들었고, 순위 평균, 신경망, LightGBM, Ridge, logit 로지스틱, Differential Evolution, 힐클라이밍, Top-K 평균을 사용했다.
- **외부 의존성:** Chris Deotte의 S6E3 logit 결합과 여러 공개 모델을 참고했지만 최종 입력 판본은 열거하지 않았다.
- **보고 결과:** Public 최고 두 결합은 `0.98191`과 `0.98190`이었지만 Private는 `0.98132`와 `0.98140`이었고, Public `0.98200` 3자 투표는 Private `0.98133`이었다.
- **반증:** 사후 Private 최고 `0.98148` 구성은 모델이 9개뿐이었고, 최종 선택보다 나은 제출이 15-20개 있었다고 밝혔다.
- **출처 사실:** High 계급이 약 `3.3%`라 적은 행의 계급 변화가 Balanced Accuracy를 크게 움직였고, CV와 Public 모두 최종 선택을 안정적으로 구분하지 못했다고 해석했다.
- **댓글 보충:** 여러 노트북에서 OOF와 시험 예측을 저장한 뒤 결합하며, Kaggle의 12시간 제한 때문에 한 노트북에는 자료 크기에 따라 대략 5-25개 모델을 넣는다고 답했다.
- **조사자 판단:** 결합기 수가 모델 수보다 많아도 최종 선택이 안정되지 않았다는 반증이며, 깊이와 방법 수 확대보다 고정 구조의 nested OOF가 우선이다.

#### 5위 Topiast

- **출처 사실:** 119개 기본 모델의 3계급 확률을 5시드 LightGBM 결합기에 넣었다.
- **출처 사실:** 후보 예측 계급 일치율이 약 `0.9964`보다 높으면 더 강한 기존 모델과 중복으로 보고 제거했다.
- **보고 결과:** 선택 구성은 CV 약 `0.98178`, Public `0.98079`, Private `0.98133`이었다.
- **출처 사실:** 정확 규칙, 타깃 인코딩, 자릿수와 반올림, 쌍 상호작용, 규칙 경계, 자동인코더 표현과 여러 트리 및 신경망 계열을 사용했다.
- **외부 의존성:** 공개 원본 자료의 정확 규칙과 공개 XGBoost, LightGBM, TabTransformer, RealMLP 노트북을 사용했다.
- **출처 사실:** 현재 결합과 후보의 오류 차이를 비교하도록 자동 실험을 반복했지만, 최종 선택과 판단의 대부분은 사람이 수행했다고 밝혔다.
- **재현성 한계:** 일치율 문턱을 여러 값으로 훑은 뒤 가장 높은 CV 실행을 골랐고, 그 선택 절차를 바깥쪽 fold에서 다시 평가하지 않았다.
- **댓글:** 댓글은 없었다.
- **조사자 판단:** 중복 제거와 오류 보완 목표는 현 후보 풀의 스피어만 `0.998` 중복 게이트와 같다.

## 제외 장부

아래 글은 공식 해법 범주에는 있었지만 제목이나 본문 순위가 11위 이하이므로 제외했다.
순위가 제목에 없던 글은 본문 상단 순위 표기를 열어 확인했다.

### S6E7 제외 10건

- [Rank11 approach](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/rank11-approach): 11위.
- [S6E7 Solution](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/s6e7-solution): 본문 상단 55위.
- [135th Place Solution](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/135th-place-solution-my-first-full-competition): 135위.
- [Why shouldn't you even consider using LB probing](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/why-you-shouldnt-even-consider-using-lb-probing-i): 본문 상단 82위.
- [PSS6E7: CV over blending](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/pss6e7-cv-blending): 본문 상단 36위.
- [29th Place: FT-Transformer and Exact-Value Target Encoding](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/29th-place-ft-transformer-exact-value-target-en): 29위.
- [First Kaggle competition experience](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/first-kaggle-competition-experience-rank-371345): 371위.
- [S6E7 HGBC Solution](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/s6e7-hgbc-solution): 본문 상단 288위.
- [376th to 92nd](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/376th-92nd-the-signal-was-non-monotonic-and-on): 92위.
- [87+ score on leaderboard](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/85-score-on-leaderboard): 본문 상단 2787위.

### S6E6 제외 19건

- [25th Place: My Public Starter Notebook](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/25th-place-my-public-starter-notebook): 25위.
- [Predicting Stellar Class using TabPFN-3](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/predicting-stellar-class-using-tabpfn-3): 본문 상단 790위.
- [24th Place Solution](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/24th-place-solution): 24위.
- [23rd Private](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/23rd-private-merci-chris-deotte): 23위.
- [Top 4%, 93rd Place](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/top-4-93rd-place-solution-stellar-classificati): 93위.
- [Bitter lesson from a 278-place shakedown](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/bitter-lesson-from-a-278-place-shakedown): 본문 상단 324위.
- [33rd Place Solution](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/33rd-place-solution): 33위.
- [45th Place: CatBoost Stacker paired with MLP Embeddings](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/45th-place-catboost-stacker-paired-with-mlp-embed): 45위.
- [26th Place WriteUp](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/26th-place-writeup-predicting-stellar-class): 26위.
- [12th place](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/12th-place): 12위.
- [22nd Place Solution](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/22nd-place-solution): 22위.
- [Top 63 Solution](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/top-63-solution-surviving-the-shake-up-with-p): 63위.
- [Top 34 Solution](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/top-34-solution-oof-stacking-for-s6e6): 34위.
- [Blending with Quasi-MC and TPE](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/blending-with-quasi-mc-and-tpe): 본문 상단 43위.
- [15th, Additive Flip-Stacking](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/15th-2816-top-8-additive-flip-stacking-bal): 15위.
- [13 Robust Features](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/playground-ser): 본문 상단 2365위.
- [31st place](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/31st-place-first-time-in-100): 31위.
- [Teaching a Model to Read the Sky](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/teaching-a-model-to-read-the-sky): 본문 상단 705위.
- [19th place: My public SKILL.md](https://www.kaggle.com/competitions/playground-series-s6e6/writeups/19th-place-my-public-skill-md-for-your-agents): 19위.

### S6E5 제외 6건

- [Rank17 approach](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/rank17-approach-diverse-models-and-blend): 17위.
- [Finished 75th](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/finished-75th): 75위.
- [11th Place](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/11th-place-in-the-midst-of-entrance-exams): 11위.
- [568th Place Solution](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/568th-place-solution-realmlp-original-dataset): 568위.
- [My First Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/f1-pit-stop-solution): 본문 상단 1919위.
- [Stacked Gradient Boosting Ensemble](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/stacked-gradient-boosting-ensemble-for-pit-stop-pr): 본문 상단 165위.

### S6E4 제외 8건

- [24th Place: A Heavy Stacking Approach](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/24th-place-a-heavy-stacking-approach-with-166-oof): 24위.
- [Rank-87 approach](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/rank-87-approach): 87위.
- [12th Place Solution](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/12th-place-solution-stacked-ensemble-with-ordered): 12위.
- [19th Place: Ensemble of 29 models](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/19nd-place-ensemble-of-29-models): 19위.
- [Top 4% Solution](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/top-4-solution-model-diversity-fe-and-systematic): 본문 상단 168위.
- [Simple XGB with Optuna](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/simple-xgb-with-hyperparameter-tuning-with-optuna): 본문 상단 359위.
- [S6E4 retrospective](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/ps-s6e4-retrospective-42-rank-jump-to-top-0-7): 30위.
- [Reproducible LightGBM Baseline](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/reproducible-lightgbm-baseline-for-s6e4): 본문 상단 197위.

## 대회 사이에서 반복된 근거

### 1. OOF 자산과 오류 다양성은 반복됐지만 모델 수 자체는 승리 조건이 아니다

19건 가운데 대규모 결합을 쓴 글 대부분은 OOF와 시험 예측을 재사용 가능한 자산으로 저장했다.
강한 공통 골격은 GBDT 여러 계열과 RealMLP, TabM, FT-Transformer 같은 신경망, 그리고 선형 또는 고전 모델의 다른 오류를 결합하는 것이었다.

반면 S6E6 1위는 100개를 넘기자 성능이 낮아졌고, S6E6 6위의 92개 결합은 최강 단일 결합보다 약 `0.000368`만 높았다.
S6E5 10위는 221개 최고 CV보다 129개 제출이 Private에서 좋았고, S6E4 4위의 사후 최고는 모델 9개뿐이었다.
따라서 구성원 수는 목표가 아니라 후보 공간이고, 선택 과정을 평가 자료와 분리해야 한다.

### 2. 단순 선형 결합은 강한 기본값이고 깊은 결합은 검증 낙관 위험이 크다

S6E6 1위와 9위, S6E5 1위, 2위, 5위, 10위, S6E4 1위와 2위가 logit 로지스틱 회귀를 최종 또는 핵심 결합기로 썼다.
S6E5 8위의 5단계 결합과 7위의 다단 결합은 깊이의 가능성을 보여 주지만, 두 글 모두 완전한 nested CV 근거를 제공하지 않았고 8위 작성자는 CV가 낙관적일 수 있다고 직접 밝혔다.
S6E5 10위와 S6E4 4위는 깊고 큰 구조의 최고 CV가 더 작은 구조보다 일반화가 나쁠 수 있다는 반례다.

### 3. 지표에 맞춘 표현과 의사결정 규칙이 모델 교체보다 큰 이득을 내기도 했다

S6E7 2위와 4위, S6E6 8위, S6E4 1위와 2위는 Balanced Accuracy에 맞춰 계급 가중 log loss, 사전확률 보정, 문턱, 계급 분해 또는 가중 다항 로지스틱 회귀를 사용했다.
그러나 이 방법들은 다계급 hard label 지표에 특화돼 S6E8의 이진 ROC AUC에는 대부분 직접 적용되지 않는다.
S6E5 5위의 순위 평균은 ROC AUC에서 서로 다른 확률 눈금을 통일한다는 점에서 직접 전이되지만 현 저장소가 이미 이를 구현한다.

### 4. 고카디널리티 통계는 전체 자료와 엄격한 fold 경계에서 평가해야 한다

S6E7 4위의 정확값 타깃 인코딩은 7만 행 검사에서는 하락했지만 69만 행 전체에서는 약 `+0.0012`였다.
학습 행의 인코딩까지 내부 OOF로 만든 점과 다른 XGBoost에서도 방향을 확인한 점이 근거의 질을 높인다.
이 결과는 S6E8의 정확값 타깃 인코딩을 작은 하위 표본에서 다시 선별하지 말고 전체 고정 fold에서 판정해야 한다는 독립 근거다.

### 5. Public 미세 차이는 최종 선택에 반복해서 실패했다

S6E7 4위는 Public 414위에서 Private 4위가 됐고, S6E6 1위는 Public 344위에서 Private 1위가 됐다.
S6E4 2위는 CV를 높인 최종판보다 3일 차 판본의 Private가 좋았고, S6E4 4위는 선택하지 않은 15-20개 제출이 사후 더 좋았다.
댓글의 여러 참가자도 Public 추종으로 큰 Private 하락을 겪었다고 확인했다.
이는 Public을 개선 판정에서 제외하고 마일스톤 건전성 점검에만 쓰는 ADR 0001과 일치한다.

## 현재 S6E8에 대조한 적용 판단

현재 저장소에는 OOF AUC `0.9692840450`의 `exp127_lookup_muon` champion과 29개 후보 풀이 있다.
후보 풀에는 LightGBM, XGBoost, CatBoost, 정확값 one-hot 로지스틱 회귀, Lookup 계열, TabM, TabPFN, RealMLP과 여러 특성 표현이 이미 들어 있다.
결합은 바깥쪽 fold마다 나머지 OOF에서 선택과 가중치를 다시 학습하는 nested OOF를 사용하며, 구성원별 순위와 잘린 logit 이중 표현 및 결측 구간 상호작용을 이미 지원한다.

| 과거 상위권 방법 | S6E8 현재 상태 | 판단 |
| --- | --- | --- |
| 정확값 타깃 인코딩과 원 수치 병행 | `exp070_cat_exact_cats`, lattice와 Lookup 계열로 검증됨 | 새 발주 없음 |
| fold 내 여러 초기화 평균 | Lookup과 RealMLP 계열에서 3초기화 또는 다중 초기화 대조 수행 | 새 발주 없음 |
| OOF와 시험 예측 자산화 | 모든 후보를 실행 저장소와 풀 장부로 관리 | 새 발주 없음 |
| 약하지만 다른 모델 계열 | champion - 0.01 하한, 스피어만 0.998 중복 게이트, 3시드 확정으로 관리 | 현 계약 유지 |
| 순위와 logit 결합 | `rank_logit` 이중 표현이 채택됨 | 새 발주 없음 |
| 부분집합 선택과 비선형 결합 | Optuna 부분집합, Ridge, XGBoost, 결측 구간 결합이 같은 nested OOF 평가기에 구현됨 | 현 평가기로 비교 |
| 원본 행 추가 | S6E8 짝비교에서 기각됐고 S6E5의 조건부 성공은 다른 자료 관계에 의존 | 재개하지 않음 |
| 계급 사전확률 보정, 문턱, OvR 분해 | 다계급 Balanced Accuracy 전용 | S6E8에 부적합 |
| HEALPix, Driver 제거, 관개 비저항 | 각 대회의 영역과 분포 이동에 종속 | S6E8에 부적합 |

## 남은 새 적용 후보와 우선순위

### 우선순위 1: 한 단계 추가 결합의 완전 중첩 깊이 대조

19건에서 현재 저장소에 없는 실질적 방법은 S6E5 8위와 10위가 보여 준 다단 결합의 추가 깊이다.
그러나 원문은 깊이의 효과를 누출 없이 분리하지 못했고, 더 큰 구조가 더 작은 구조보다 나빴다는 반증도 함께 제공한다.
따라서 4단계 또는 5단계 구조를 그대로 옮기지 않고, 현재 채택 결합 위에 사전에 고정한 한 단계만 추가하는 대조가 적절하다.

- **가설:** 현재 29개 후보의 기본 확률뿐 아니라 현재 단순 결합기 여러 개의 바깥쪽 학습 구간 예측을 다시 결합하면 ROC AUC에 남은 비선형 또는 선택 다양성을 얻을 수 있다.
- **고정 구조:** 현 `missing_interaction_rank_logit`을 기준으로 하고, 1단 결합기 후보는 이미 구현된 `rank_mean`, `rank_logit_logistic`, `optuna_subset_ridge_logit`, `xgb_rank_logit`처럼 사전에 고정한 소수만 쓴다.
- **누출 경계:** 각 outer fold에서 기본 구성원 선택, 1단 결합 학습, 2단 결합 학습을 모두 나머지 4개 fold 안에서 다시 수행해야 한다.
- **평가:** 같은 5개 outer fold의 현 채택 결합과 짝지어 비교하고, ADR 0001 계열 3의 `+0.00002` 및 경계 구간 3/5 fold 승리를 그대로 요구한다.
- **중단 조건:** 한 단계 추가 nested OOF가 기준보다 `+0.00002` 미만이거나 3/5 fold 승리를 못 하면 더 깊은 3단계와 자기증류는 열지 않는다.
- **중단 조건:** 2단 입력 수나 구조 변형을 결과를 보며 늘리고 싶어지면 현재 결과를 폐기하고 후보 집합을 새로 고정한 선택 절차 대조로 다시 시작한다.
- **외부 의존성:** 새 외부 예측이나 새 기본 모델은 필요 없으며 현재 29개 후보의 검증된 OOF만 사용한다.

### 우선순위 2: 후보 풀 포화 곡선은 새 모델 발주가 아니라 최종 결합 진단으로 수행

S6E6과 S6E5의 여러 상위권 글은 90개부터 221개까지 늘린 풀이 포화하거나 악화할 수 있음을 독립적으로 보여 준다.
현 저장소에는 부분집합 선택 결합기가 이미 있으므로 새 구현보다 29개 전체와 nested 선택 부분집합의 차이를 최종 풀 고정 시점에 기록하는 편이 낫다.

- **가설:** 현재 풀 전체가 단순 중복은 아니어도 결합 학습 분산 때문에 일부 outer fold에서 작은 부분집합보다 나쁠 수 있다.
- **평가:** 전체 `rank_logit` 결합, 기존 Optuna 부분집합 결합, 전체 순위 평균을 동일 nested OOF에서 비교한다.
- **중단 조건:** 부분집합 결합이 전체 결합을 이기지 못하면 별도 프루닝 규칙이나 하이퍼파라미터 탐색을 추가하지 않는다.
- **채택 제한:** 부분집합은 후보 풀 장부에서 구성원을 소급 삭제하는 근거가 아니라 최종 결합 전략 안의 fold별 선택으로만 사용한다.

### 우선순위 3: 새 기본 모델 계열은 이 조사만으로 추가하지 않음

과거 글에 등장한 Trompt, TabR, TabPFN, RealMLP, TabM, AMFormer, 여러 선형과 트리 계열 가운데 다수는 이미 S6E8에서 실행 또는 진입 진단을 마쳤다.
S6E6 9위는 낮은 상관의 약한 모델도 결합에 실패할 수 있음을 보여 줬고, S6E4 3위는 Public과 결합기 중요도로 후보를 고른 약한 근거를 남겼다.
따라서 이 조사만으로 새 모델 계열을 실험 발주하지 않는다.
새 계열은 별도 근거가 champion - 0.01 하한 또는 최근접 스피어만 0.98 미만의 기대를 구체적으로 뒷받침할 때만 연다.

## 최종 결론

최근 네 대회의 1-10위 공식 해법 19건과 댓글·답글 161건은 S6E8의 현재 방향을 뒤집지 않는다.
가장 강하게 반복된 방법은 공통 fold의 OOF 자산, 서로 다른 오류를 내는 모델 계열, 단순한 logit 또는 순위 결합, Public보다 정직한 CV를 우선하는 규율이었다.
이 저장소는 이 네 가지를 29개 후보 풀, 3시드 확정, 중복 게이트, rank-logit 결합과 nested OOF로 이미 더 엄격하게 구현하고 있다.

새 적용 후보는 깊은 결합을 그대로 복제하는 것이 아니라 현재 결합 위 한 단계의 추가 가치만 완전한 nested OOF로 재는 대조 하나다.
그 대조가 ADR 문턱을 넘지 못하면 4단계 또는 5단계 결합, 자기증류, 더 큰 OOF 물량으로 확장하지 않는 것이 근거에 맞다.
