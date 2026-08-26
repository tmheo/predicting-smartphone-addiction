# S6E8 마감 직전 공개 후보 재점검

이 문서는 [이슈 427](https://github.com/tmheo/predicting-smartphone-addiction/issues/427)이 묻는 "2026-08-26 공개 증분에서 현재 35개 후보 풀 밖의 자체 재현 후보가 생겼는가"에 답한다.
공개 자료 확인 시점은 2026-08-26 13:02:54 UTC이고, 자체 재현과 판정을 끝낼 수 있어야 하는 시한은 2026-08-29 23:59 UTC다.

## 결론

**새 실행 후보는 없다.**
확인 시점까지의 2026-08-26 공개분에는 현재 35개 풀과 계보·정보 관점이 충분히 다르고, 바깥쪽 검증 분할을 지키며, 시한 안에 자체 파이프라인으로 재현·판정할 근거까지 갖춘 항목이 하나도 없었다.

가장 눈에 띄는 신규 신경망인 [kind_of_tabnn_capable_0.9695](https://www.kaggle.com/code/sergeyqt2024/kind-of-tabnn-capable-0-9695/notebook?scriptVersionId=345119579)는 저장 실행 결과가 없고 현재 소스가 완주할 수 없으며, 학습행 목표 부호화가 자기 목표값을 포함한다.
구조도 범주 임베딩, 수치 특성 조절, 여러 채널의 작은 완전연결망을 합친 형태라 현재 풀의 TabM·일반 신경망 정보 관점과 겹친다.

실행 결과가 있는 최근 신경망 중 가장 강한 [FastAI 노트북](https://www.kaggle.com/code/omidbaghchehsaraei/fastai-for-predicting-smartphone-addiction/notebook?scriptVersionId=344317824)은 전체 OOF AUC `0.9667557595`를 남겼다.
그러나 바깥쪽 검증 분할의 목표값이 바깥쪽 학습행의 목표 부호화에 들어가므로 채택 판정에 쓸 수 없는 OOF다.
그 OOF를 후보에게 유리한 참고 진단으로만 현재 풀에 더해 봐도 두 핵심 결합 방식의 증분은 `+0.0000001546`과 `+0.0000064859`로 `+0.00002` 문턱에 못 미쳤다.

따라서 [지도 423](https://github.com/tmheo/predicting-smartphone-addiction/issues/423)의 우선순위를 바꾸지 않는다.
마감 전 후보 풀을 더 넓혀야 한다면 공개된 새 계열을 급히 이식하기보다, 이미 깨끗하게 검증된 자체 Lookup-Transformer·RealMLP 학습 궤적의 **학습 상태 대조축**을 보존하는 쪽이 근거가 더 강하다.
[#413](https://github.com/tmheo/predicting-smartphone-addiction/issues/413)과 [#419](https://github.com/tmheo/predicting-smartphone-addiction/issues/419)의 고정 반복 수 트리 변형도 같은 교훈을 준다.
즉 지금 풀의 폭을 실제로 늘린 것은 이름이 새로운 모형이 아니라, 단독 성능 하한을 지키면서 기존 구성원과 다른 오류를 남기는 깨끗한 학습 상태였다.

## 조사 계약

조사는 다음 네 조건을 모두 만족하는 항목만 실행 후보로 보았다.

1. 현재 [`artifacts/pool.yaml`](../../artifacts/pool.yaml)의 35개 구성원과 계보 또는 정보 관점이 실질적으로 다르다.
2. 바깥쪽 검증 분할의 목표값이 특성 생성, 학습, 학습 상태 선택과 결합 선택 어디에도 들어가지 않는다.
3. 공개 소스와 실행 근거로 단독 성능과 후보 풀 한계 기여를 예상할 수 있다.
4. 2026-08-29 23:59 UTC 전 자체 5분할 재현과 판정을 끝낼 수 있다.

현재 후보 풀의 SHA-256은 `caa1b90769720a4accbe07074dbc7efe0335ab6657fea80c6839b60121dc39d3`이다.
풀 구성과 기존 중복·기여 근거는 [OOF 후보 풀 감사](oof-pool-audit.md), 최종 결합 기준은 [35개 풀 최종 결합 문서](issue337-final-combiner.md)를 기준으로 삼았다.

Kaggle 공식 CLI에서 대회 노트북을 최근 실행 시각 순으로 조회하고, 관련 공개 노트북의 원본과 저장 실행 산출물을 공식 CLI로 내려받아 읽었다.
토론 목록은 Kaggle의 `Recently Posted` 정렬로 확인하고, 새 글의 본문과 댓글을 원문에서 읽었다.
노트북이나 토론의 주장만 있고 실행 출력이 없으면 저자 보고치로 구분했다.

외부 OOF와 시험 예측은 오직 읽기 전용 진단에만 사용했다.
그 파일은 임시 경로에서만 읽었고 저장소, 후보 풀, 전체 자료 재학습 계획과 최종 제출에는 넣지 않았다.
새 모형 학습이나 유료 계산 자원 실행은 하지 않았다.

## 판정표

| 공개 항목 | 공개 근거 | 현재 풀과의 차이 | 재현·검증 상태 | 판정 |
| --- | --- | --- | --- | --- |
| [kind_of_tabnn_capable_0.9695](https://www.kaggle.com/code/sergeyqt2024/kind-of-tabnn-capable-0-9695/notebook?scriptVersionId=345119579) | 본문은 다른 설정에서 `0.9696`을 얻었다고 주장하지만 저장 실행 OOF는 없음 | TabM식 다중 채널, 범주 임베딩, 완전연결망이라 기존 TabM·신경망 축과 중복 | 자기 목표값 포함 부호화, 정의 전 변수 참조, 마지막 OOF 저장 변수 오타 | 기각 |
| [Exploring Alternative Ensemble Blends](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/737590) | 다른 자료에서 다중 연산 결합이 가산 결합보다 약 `+0.0000377`, 실행은 중단됨 | 모형 후보가 아니라 결합 탐색 공간 확대 | S6E8 결과 없음, 저자도 OOF 과적합 위험을 명시 | 실행 후보 아님 |
| [FASTAI for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/fastai-for-predicting-smartphone-addiction/notebook?scriptVersionId=344317824) | 저장 OOF `0.9667557595` | 최근접 후보와 순위 상관 `0.98650`으로 겉보기 다양성은 있음 | 바깥쪽 검증 목표값이 학습 특성에 들어감, 참고 풀 진단도 최대 `+0.00000649` | 기각 |
| [TabFM zero-shot](https://www.kaggle.com/code/paiky1995/s6e8-tabfm-zero-shot-on-0-7-of-the-data/notebook?scriptVersionId=344896849) | 같은 100,000행에서 TabFM `0.955027`, 원시 특성 XGBoost `0.966113`, 결합 `+0.000050` | 사전학습형 표 모델이라는 구조 차이는 있음 | 강한 설정은 메모리 부족, 단독 하한 미달, 결합 문턱 미달, 가중치 사용 조건 별도 | 기각 |
| [A Small Neural Net Instead of Trees](https://www.kaggle.com/code/sarveshchhetri/a-small-neural-net-instead-of-trees) | 저장 OOF `0.938208` | 일반 다층 신경망으로 기존 축과 중복 | 완주했지만 단독 하한에서 크게 미달 | 기각 |
| [causalml](https://www.kaggle.com/code/udaken10/causalml) | 저장 로그의 20분할 OOF `0.95958` | 학업 영향 변수를 처치로 삼은 효과 추정 특성이 형식상 새로움 | 효과 추정기를 전체 목표값으로 먼저 학습한 뒤 검증하고 제출 생성에서 오류 | 기각 |
| [What the Generator Remembers](https://www.kaggle.com/code/abhirajhiwale/s6e8-what-the-generator-remembers-honest-cv) | 저자 보고 fold 0 `0.96795`, 공개 점수 `0.9680` | 정확값 목표·빈도 부호화와 구조 특성을 쓴 LightGBM | 기존 정확값·재구성·트리 축과 중복하고 절대 성능도 낮음 | 새 후보 없음 |
| [Residual Geometry Spline Transformer](https://www.kaggle.com/code/ern711/residual-geometry-spline-transformer)와 [tomasa2 갱신판](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t) | 잔차 기하 6개를 함께 넣은 추가분이 저자 스택에서 `+0.000003` | 현재 풀의 문맥화 스플라인과 같은 정보 관점 | 자체 선행 조사와 공개 추가분이 모두 문턱 미달 | 닫힌 계열 유지 |

## 2026-08-26 공개 증분

### kind_of_tabnn은 점수 표제가 아니라 미실행 소스다

공식 목록에서 [kind_of_tabnn_capable_0.9695](https://www.kaggle.com/code/sergeyqt2024/kind-of-tabnn-capable-0-9695/notebook?scriptVersionId=345119579)의 최근 실행 시각은 2026-08-26 12:42:52 UTC였다.
본문은 "설정을 학습 중 바꿔 `0.9696`을 얻었다"고 쓰고 현재 판은 더 나쁠 수 있다고 덧붙인다.
즉 표제 점수와 현재 고정 설정 사이의 대응부터 증명되지 않았다.

내려받은 노트북의 모든 코드 셀은 `execution_count=null`이고 출력이 비어 있다.
저장 실행 로그도 HTML 변환 기록만 있으며 학습 지표, OOF와 제출 산출물이 없다.

현재 소스는 다음 세 곳에서 자체 재현 관문을 통과하지 못한다.

- 각 바깥쪽 학습 분할의 범주 목표 평균표를 그 학습행 전체 목표값으로 만든 뒤 같은 학습행에 다시 매핑하므로, 학습행의 목표 부호화가 자기 목표값을 포함한다.
- 분할별 임시 제출을 만들 때 아직 정의되지 않은 `final_test_pred`를 참조하며, 바로 앞에서 계산한 `partial_test_pred`는 쓰지 않는다.
- 마지막에 실제 배열 이름 `oof_preds`가 아니라 정의되지 않은 `oof_pred`를 저장한다.

핵심 구조는 범주 임베딩과 비율 특성을 작은 보조 신경망에 넣고, 그 출력으로 수치 특성을 조절한 뒤, 512개 특성 마스크 채널을 가진 TabM식 블록의 평균을 내는 것이다.
이는 현재 풀의 TabM, RealMLP, Tab CNN, 표 토큰 변환기와 다른 이름을 갖지만 정보 원천은 같은 원시·파생 수치, 정확값 범주와 목표 평균이다.
실행 증거 없이 이 복합 구조를 다시 구현하는 것은 8월 29일 판정 시한을 쓸 만큼 새로운 정보 관점을 제공하지 않는다.

### 다중 연산 결합은 결과가 아니라 연구 가설이다

[토론 737590](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/737590)은 2026-08-26에 새로 올라온 유일한 실질적 결합 제안이었다.
가산 평균, 기하 평균과 거듭제곱 평균 가운데 하나를 구성원마다 고르게 하는 탐색을 제안한다.

글이 제시한 수치는 S6E8가 아닌 별도 자료에서 가산 결합 `0.9547683`과 다중 연산 결합 `0.95480597`을 비교한 것이고, 탐색 실행도 중간에 끊겼다.
S6E8 OOF, 고정 후보 집합, 바깥쪽 검증 분할과 재현 소스가 없으며 확인 시점 댓글도 0개였다.
저자도 탐색 공간이 커져 OOF 과적합 위험이 늘어난다고 명시한다.

현재 풀의 결합 방식은 각 바깥쪽 학습 부분에서 가중치와 변환을 다시 고르는 [중첩 OOF 절차](issue337-final-combiner.md)를 이미 사용한다.
마감 직전에 구성원별 연산 선택까지 자유도를 늘릴 근거는 없으므로 이 제안은 모형 후보도 결합 변경 후보도 아니다.

## 최근 신경망 대조 조사

엄격한 2026-08-26 증분만 보면 실행 결과가 있는 새 신경망이 없었다.
그래서 직전 며칠의 공개 실행까지 대조군으로 넓혀, 신경망 계열에서 놓친 정보 관점이 실제로 있었는지 확인했다.

### FastAI는 겉보기 다양성이 있어도 검증 계약과 기여 관문을 모두 넘지 못한다

[FastAI 노트북](https://www.kaggle.com/code/omidbaghchehsaraei/fastai-for-predicting-smartphone-addiction/notebook?scriptVersionId=344317824)은 같은 5개 분할에서 `0.96605`, `0.96675`, `0.96708`, `0.96740`, `0.96658`을 기록했다.
저장된 691,369행 OOF를 다시 채점한 전체 AUC는 `0.966755759498730`이다.

이 OOF의 가장 가까운 현재 후보는 `exp067_tabpfn3`이고 순위 상관은 `0.9865005`다.
35개와의 순위 상관 중앙값은 `0.97847`, 최솟값은 `0.94945`라 예측 모양만 보면 기존 풀에 없는 몫이 있을 수 있다.

그러나 `apply_encodings_cv`가 먼저 전체 학습 자료에 하나의 5분할 OOF 목표·빈도 부호화를 만들고, 학습 단계가 같은 분할 객체를 다시 바깥쪽 검증으로 사용한다.
바깥쪽 검증 분할이 `j`일 때, 바깥쪽 학습행 중 다른 분할 `k`에 속한 행의 부호화는 `k`만 제외한 목표값으로 만들어져 `j`의 검증 목표값을 포함한다.
따라서 검증 목표값이 바깥쪽 학습 특성에 들어가며 이 OOF는 채택 판정 자료가 아니다.

그래도 이 오염된 OOF를 후보에게 유리하게 그대로 인정하고 현재 35개 풀에 추가하는 읽기 전용 진단을 수행했다.

| 핵심 결합 방식 | 기존 35개 | FastAI 포함 | 차이 | 양수 분할 |
| --- | ---: | ---: | ---: | ---: |
| `missing_segmented_rank_logit` | `0.969805234101` | `0.969805388718` | `+0.000000154617` | 3 / 5 |
| `missing_interaction_rank_logit` | `0.969802952447` | `0.969809438375` | `+0.000006485928` | 4 / 5 |

두 결과가 모두 `+0.00002`의 약 0.8퍼센트와 32퍼센트에 불과해 세 번째 핵심 결합 방식까지 계산하지 않았다.
이 표는 불완전하고 오염된 외부 OOF를 이용한 진단이므로 공식 진입 판정이 아니다.
오염된 공개 수치를 그대로 인정한 입력에서도 남는 몫이 너무 작다는 중단 근거다.
공식 기각 근거는 이 불완전 진단이 아니라 바깥쪽 검증 분할 계약 위반과 깨끗한 자체 재현 근거의 부재다.

FastAI를 올바르게 재현하려면 각 바깥쪽 학습 부분 안에서 목표 부호화용 내부 OOF를 새로 만들고, 그 내부 학습 매핑 평균으로 바깥쪽 검증과 시험 특성을 만들어야 한다.
그러면 공개 OOF와 다른 입력으로 다시 학습하게 되며, 위 수치보다 좋아질지조차 근거가 없다.
시한 안에 새 5분할 GPU 실행을 열 우선순위가 아니다.

### TabFM은 새 구조지만 성능과 자원 관문에서 끝난다

[TabFM 노트북](https://www.kaggle.com/code/paiky1995/s6e8-tabfm-zero-shot-on-0-7-of-the-data/notebook?scriptVersionId=344896849)은 고정 분할 중 100,000행만 평가했다.
8,000개 문맥 행과 추정기 4개 설정에서 TabFM AUC는 `0.955027`, 같은 100,000행의 원시 특성 XGBoost는 `0.966113`, 두 예측의 순위 상관은 `0.9732`였다.
TabFM을 가중치 `0.0625`로 섞은 결합은 `+0.000050`이었지만, 이는 현재 35개 풀이 아니라 약한 XGBoost 하나에 더한 값이다.
저자가 미리 둔 결합 관문 `+0.000200`에도 못 미쳐 제출에서 제외됐다.

문맥 16,000개 설정과 권장 결합 설정은 16GB T4에서 메모리 부족으로 실패했다.
단독 `0.955027`은 현재 후보 진입 하한 약 `0.95937`보다 낮으므로 더 큰 장비로 전체 OOF를 만드는 작업을 정당화하지 못한다.

[Google Research TabFM 공식 저장소](https://github.com/google-research/tabfm#license-notice-for-pretrained-weights)는 소스 코드의 Apache License 2.0과 별개로 기본 사전학습 가중치에 `tabfm-non-commercial-v1.0`을 적용하고, 비상업·비운영 사용으로 제한한다고 명시한다.
따라서 성능 관문을 통과했더라도 사용 조건, 정확한 가중치 판본과 조건 수락 기록을 별도로 고정해야 한다.
이번에는 성능과 자원 관문에서 이미 기각되므로 그 절차를 열지 않는다.

### 작은 신경망과 인과 효과 특성은 절대 성능 또는 검증에서 끝난다

[작은 scikit-learn 신경망](https://www.kaggle.com/code/sarveshchhetri/a-small-neural-net-instead-of-trees)은 5개 분할 `0.937555`, `0.937885`, `0.938394`, `0.939128`, `0.938145`와 전체 OOF `0.938208`을 저장했다.
일반 다층 신경망이라는 점에서도 새 정보 관점이 아니고 단독 성능이 진입 하한보다 `0.021` 이상 낮다.

[causalml 노트북](https://www.kaggle.com/code/udaken10/causalml)은 `academic_work_impact`를 처치로 정하고 X-Learner가 만든 개별 효과 추정치를 특성으로 넣는다.
그러나 X-Learner를 전체 학습 자료의 목표값으로 먼저 맞춰 `cate_train`을 만든 뒤 그 특성을 20분할 검증에 사용한다.
각 검증행 목표값이 자기 효과 추정 특성에 들어가므로 저장 로그의 OOF `0.95958`은 유효하지 않다.
실행은 마지막에 예측 배열 전체를 정수 하나로 바꾸려다 `TypeError`로 끝나 제출도 만들지 못했다.

별도의 [FastAI 공개 실행](https://www.kaggle.com/code/casati8/s608-fast-ai)은 학습 전에 `compute_class_weight` 미정의 오류로 끝났다.
실행 결과도 고정 입력도 없으므로 후보가 아니다.

## 공개 트리와 기존 계열이 보여 주는 포화 경향

[토론 737422](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/737422)은 같은 5분할에서 LightGBM `max_bin` 255와 2047을 비교했다.
원시 12개 특성에서는 고해상도 구간이 `0.96431`에서 `0.96672`로 `+0.0024`, 18개 파생 특성에서는 `0.96541`에서 `0.96746`으로 `+0.0020` 올랐다.
정확값 빈도 부호화를 넣은 21개 특성에서는 `0.96717`에서 `0.96767`로 이득이 `+0.0005`로 줄었다.
반대로 빈도 부호화의 이득은 `max_bin=255`에서 `+0.0018`, `max_bin=2047`에서 `+0.0002`로 줄었다.

이 결과는 고해상도 분기와 정확값 빈도 부호화가 같은 합성 격자 신호를 대신 읽는다는 저자의 결론을 뒷받침한다.
현재 풀은 고해상도 XGBoost·LightGBM·CatBoost, 정확값 목표·빈도 부호화와 Lookup 계열을 모두 갖고 있다.
따라서 최근 트리 공개분은 새 정보 관점을 보여 주기보다 현재 풀의 포화를 확인한다.

[What the Generator Remembers](https://www.kaggle.com/code/abhirajhiwale/s6e8-what-the-generator-remembers-honest-cv)도 정확값 목표·빈도 부호화, 화면 시간 관계와 채운 복사본을 LightGBM에 넣는다.
저자 보고 성능은 fold 0 `0.96795`, 공개 점수 `0.9680`이고, 특성·학습기 모두 현재 풀과 [#419](https://github.com/tmheo/predicting-smartphone-addiction/issues/419)이 확장한 축의 부분집합이다.

[tomasa2 갱신판](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)은 스플라인 변환기 `0.96680`이 저자 스택에 `+0.00004`, FT-Transformer가 `+0.00001`, MLP-PLR·ResNet·FM이 0을 더했다고 정리한다.
TabM PWL의 추가분도 `+0.00002`이고, 잔차 기하 스플라인 6개를 함께 넣은 추가분은 `+0.000003`이다.
이들은 각각 현재 풀의 문맥화 스플라인, 표 토큰 변환기, TabM과 중복하며 [갱신판 선행 재검토](gaming-hours-whatmoved-notebook-recheck.md)와 [잔차 기하 선행 조사](ern711-residual-geometry-boosting-notebook.md)가 이미 같은 결론을 냈다.

값 가리기 증강 주장도 새 근거가 아니다.
[토론 증분 기록](discussion-insights.md)은 공개 실행이 0.1에서 개선되고 0.3과 0.4에서 악화됐음을 확인했고, 현재 Lookup-Transformer는 이미 0.1을 사용한다.
[실측 결측 마스크 대조](champion-value-dropout-mask-shape.md)는 더 현실적인 마스크 모양 두 개를 모두 기각했다.

공개 예측 CSV 하나를 읽어 제출만 만드는 [0826 knock the blender with a liner](https://www.kaggle.com/code/azzamradman/0826-knock-the-blender-with-a-liner) 같은 노트북은 자체 재현 후보가 아니다.
외부 OOF·시험 예측을 직접 후보나 최종 제출에 넣지 않는 이 조사의 계약에 따라 제외했다.

## 출처와 사용 조건

Kaggle 공개 노트북 소스는 [저장소의 공개 노트북 사용 조건](../agents/kaggle-public-notebook-licensing.md)에 따라 Apache License 2.0으로 참고했다.
이번 조사에서는 어떤 공개 코드도 저장소로 복사하거나 파생 구현에 넣지 않았다.
향후 공개 코드를 채택한다면 원문, 저작권·NOTICE, 변경 사실과 판본을 보존해야 한다.

노트북 소스 사용 조건은 자료, 사전학습 가중치, 패키지와 실행 산출물의 사용 조건을 대신하지 않는다.
TabFM 가중치처럼 별도 조건이 있는 자산은 소스 코드와 따로 판정해야 한다.
[FastAI 공식 저장소](https://github.com/fastai/fastai/blob/master/LICENSE)의 코드는 Apache License 2.0이지만, 이 사실은 외부 노트북 OOF의 검증 누수를 고치거나 그 예측 파일을 자체 산출물로 바꾸지 않는다.

고정한 핵심 자료의 SHA-256은 다음과 같다.

kind_of_tabnn 고정판은 `scriptVersionId=345119579`, 최근 실행 시각 2026-08-26 12:42:52 UTC다.
FastAI 고정판은 `scriptVersionId=344317824`, 최근 실행 시각 2026-08-23 07:47:37 UTC다.
TabFM 고정판은 `scriptVersionId=344896849`, 최근 실행 시각 2026-08-25 16:37:43 UTC다.

| 자료 | SHA-256 |
| --- | --- |
| kind_of_tabnn 공개 원본 | `fd6737cc839dce3a6fe74c2a67e0c2f1c851844699cad5fe28f886f21008b5a4` |
| kind_of_tabnn 저장 실행 로그 | `767a1f806a6742958287cb9a03c90c6376c39c0c0a9541d80081fe24990c8d98` |
| FastAI 공개 원본 | `09c13b3895353b28a8c16bd172c84f9f3359d97c5583704c5067b5c9c672ed32` |
| FastAI 저장 실행 로그 | `fe32d041fd5d3e04f11419b7468f6c859228a34765b1c0414d5870fcb20d9f71` |
| FastAI 외부 OOF | `0a50012a20372c978f8c7dd4af90deb08d9c27440938f0eec8c7a355ebc5823d` |
| TabFM 공개 원본 | `0b3700e412abffd3731fd3e21656671db3cdeb2e5f6afef9b0b855e5a099eca2` |
| TabFM 저장 실행 로그 | `0bd7e50c0a75eb00ec892cab4649994d57c28f179164d45dff1a58ba6b266a48` |
| 작은 신경망 공개 원본 | `7cc43296bceba4ccac7cec887eb3538c245a5e58ed9b16de1305b503a8572c5f` |
| 작은 신경망 저장 실행 로그 | `8837b37fc47e12c026300f6040f57245502549d1f924a6832a43a42c5826bd42` |
| causalml 공개 원본 | `ffdb7bc9cfb3a2834bc25a18c3b247565a448075b059de17db59161b04940496` |
| causalml 저장 실행 로그 | `4200512ada776837bae9a6262a43f3c05b30a49be8d96cfd1c0e29da7cf41b02` |

## 최종 권고와 다시 열 조건

[이슈 426](https://github.com/tmheo/predicting-smartphone-addiction/issues/426)의 공개 자료 조사 갈래는 **중단**으로 판정한다.
새 공개 후보 티켓, 새 신경망 이식, TabFM 전체 실행과 다중 연산 결합 탐색을 열지 않는다.
지도 423의 Lookup-Transformer·RealMLP 학습 상태 대조축과 이미 선정된 전체 자료 재학습을 우선한다.

시한 전 다음 중 하나가 새로 나타날 때만 이 결론을 다시 본다.

- 현재 35개 풀과 최근접 순위 상관이 낮고 단독 OOF가 최소 `0.95937`인 전체 5분할 예측이 공개된다.
- 바깥쪽 검증 분할마다 전처리와 학습을 다시 맞춘 코드와 실행 로그가 함께 공개된다.
- 현재 풀에 더했을 때 중첩 OOF `+0.00002` 이상을 내는 읽기 전용 진단이 나오고, 자체 재현을 2026-08-29 23:59 UTC 전에 끝낼 계산 경로가 있다.
- 공개 소스뿐 아니라 자료, 가중치와 패키지의 사용 조건까지 최종 제출에 맞게 고정할 수 있다.

그 조건이 없으면 새로운 이름을 하나 더 늘리는 것보다 이미 검증된 학습 궤적의 다른 상태를 보존하는 편이 단순하고 근거가 강하다.
