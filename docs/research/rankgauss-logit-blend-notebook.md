# nikita7364777 Rank-Gauss + logit-rank blending 노트북 검토

이 문서는 [리서치: 신규 최고 퍼블릭 0.97114 Rank-Gauss + logit-rank blending 노트북 검토](https://github.com/tmheo/predicting-smartphone-addiction/issues/205)의 근거다.
조사 시점은 2026-08-18이며, Kaggle이 공개한 최신 판본의 소스, 커널 메타데이터, 공개 출력(실행 기록, 감사 CSV, 계수 CSV, 제출 파일)과 이 저장소의 기존 판정 근거를 사용했다.

## 결론

[Rank-Gauss + logit-rank blending](https://www.kaggle.com/code/nikita7364777/rank-gauss-logit-rank-blending) (public 0.97114)에는 새 모델, 새 특성, 새 결합 신호가 없다.
채택하거나 자체 재현할 요소가 없고, 새 실험 티켓을 열 근거도 없다.

노트북은 두 단계다.
1단계는 adarsh1077 Diversity Beats Strength 계열의 rank-gauss 로지스틱 스택을 작성자 개인 외부 OOF 풀(186쌍) 위에서 위생 보강만 해 재실행한 것이다.
2단계는 그 제출물과 공개 제출물 두 개를 손으로 정한 가중치 50/25/25의 logit-rank 혼합으로 섞은 것이며, 점수를 만든 것은 이 2단계다.

2단계는 외부 제출물 직접 편입이므로 [지도 172](https://github.com/tmheo/predicting-smartphone-addiction/issues/172)의 범위 밖 규칙에 정확히 해당한다.
혼합 대상인 anthonytherrien NN Residual Network 0.97101은 [najiama의 감식](https://www.kaggle.com/code/najiama/where-does-the-0-97101-nn-score-really-come-from)이 공개 제출 파일 두 개의 바이트 복사 껍데기임을 이미 증명했고, amanatar 0.97092도 외부 제출물 혼합 계열이다.
따라서 0.97114는 공개 제출물 재조합의 재조합이며, 직전 최고 공개 점수 najiama 0.97113 대비 +0.00001이다.

1단계의 rank-gauss 로지스틱 결합은 이 저장소가 이미 `rank_gauss_logistic` 전략으로 fold 안전하게 구현해 [이슈 64](https://github.com/tmheo/predicting-smartphone-addiction/issues/64)에서 비교했고, 순위와 logit 이중 표현(0.969483)에 밀렸다(0.969455).
노트북 판은 rank-gauss를 outer 분할 전에 전체 OOF에서 맞추는 기존 누출을 그대로 갖고 있어 우리 구현보다 오히려 느슨하다.

## 확인한 판본과 내용 해시

커널 id는 `131140582`, 작성자는 nikita7364777(표시명 r0tor)이고, 목록 API의 최신 공개 실행 시각은 2026-08-18 10:05:57.787 UTC다.
Kaggle 공개 노트북 소스이므로 Apache License 2.0이 적용되며, 코드를 재사용하지 않고 검토만 했으므로 별도 고지 의무는 발생하지 않는다.

- 소스 `.ipynb`는 18,943바이트이며 SHA-256은 `d05af4384324d254663420c9895d8d64fd72ebe344d384b0bc1ddc2baf5b0b07`다.
- `kernel-metadata.json`은 771바이트이며 SHA-256은 `2530cc63fc96f8dfdd4f538181ac454c09f5b7b2f6b47440ccd67868e582433d`다.
- 셀은 6개(마크다운 3, 코드 3)다.
- `kernel_sources`는 anthonytherrien NN Residual Network와 amanatar Elite Rank Average 두 개이고, 입력 자료셋은 작성자 개인 자료셋 `nikita7364777/smartphone-addiction-dataset`이다.
- 개인 자료셋의 파일 목록 API는 403을 돌려주며, 사용 조건과 구성원 계보를 외부에서 검증할 수 없다.

## 1단계: 외부 OOF 풀 위의 rank-gauss 로지스틱 스택

입력은 자료셋의 `Oof_Test` 디렉터리에 있는 `.npy` 372개이고, `oof_*`/`test_*` 이름 규칙으로 186쌍을 만든다.
바이트 단위 정확 중복 1쌍(`xgb_screen_relations_baseline103` == `xgb_identity_digit_enhanced103`)을 제거해 185쌍이 된다.

격리 관문은 비유한 값 사전 검사, 단독 OOF AUC 0.93 미만 탈락, 고정 표본 40,000(시드 0)의 OOF-시험 백분위 순위 KS 0.05 초과 탈락 순서다.
adarsh1077 판의 AUC 문턱 0.90을 0.93으로 올렸고, KS 표본을 구성원마다 다시 뽑지 않고 한 번 뽑아 재사용한다.
이름에 `perp`가 들어간 구성원은 두 문턱을 면제받으며, 실제로 `sixmember_meta_perp`(단독 AUC 0.4685)가 이 우회로 유지된다.
탈락은 17개(deepfm 3, gandalf 2, svm 5, logreg 2, rf 계열 3, knn, ft_cross)이고 168개가 남는다.
`knn`(KS 0.1789), `rf`(0.0797), `rf_pairwise_v1`(0.0793)만 KS로 탈락했고 나머지는 전부 AUC 미달이다.

표현은 백분위 순위를 표준정규 분위수로 옮기는 rank-gauss(float64)인데, 전체 OOF와 전체 시험에서 각각 한 번 계산하므로 outer 분할 밖에서 맞춰진다.
결합은 StratifiedKFold 5(시드 42) nested로, outer 학습 부분에서 표준화와 L2 로지스틱 회귀(C=0.1, lbfgs, max_iter 5000, tol 1e-5)를 맞추고 수렴을 assert한다.
최종 제출은 전체 자료 재적합의 결정함수를 백분위 순위로 바꾼 값이다.

공개 실행 기록의 결과는 최고 단독 구성원 `0.96932869`, 균등 순위 평균 `0.96664913`, nested rank-gauss 스택 `0.97005974`다.
adarsh1077의 177구성원 실행 `0.970093`보다 오히려 약 0.00003 낮으므로, 관문 강화와 구성원 추가가 스택 자체를 개선하지 못했다.

로지스틱 계수 상위는 `foldsafe_te_wide` +1.178, `catnative` +0.914, `realmlp_seed01_fixed4` +0.655, `pub_rmlp` +0.627, `lookup_v2_s03` +0.518이다.
이 서열은 [상위 공개 스택 구성원의 출처 코드 추적과 재현 후보](https://github.com/tmheo/predicting-smartphone-addiction/issues/174)가 세운 재현 우선순위와 일치하고, 그 우선순위는 이미 티켓으로 소화됐다.
Lookup 2호는 [179](https://github.com/tmheo/predicting-smartphone-addiction/issues/179)로 채택, no-TE 뷰 3종은 [183](https://github.com/tmheo/predicting-smartphone-addiction/issues/183)으로 채택, catnative는 183에서 기각, 정체성·자리수는 [184](https://github.com/tmheo/predicting-smartphone-addiction/issues/184)에서 기각, 화면 관계 7특성은 [181](https://github.com/tmheo/predicting-smartphone-addiction/issues/181)에서 기각, RealMLP-TD는 [180](https://github.com/tmheo/predicting-smartphone-addiction/issues/180)이 진행 중이다.

풀에는 기존 문서에 없는 이름(`altview`, `lookup_v2_s03`, `latr1_xgb`, `gxgbnote`, `a`~`d` 등)이 있지만, 전부 계보를 검증할 수 없는 외부 OOF이므로 읽기 전용 진입 진단 이상의 용도가 없고, 그 진단 용도로는 이미 계보가 복원된 94 Verified OOFs가 있다.

마크다운이 내세우는 "key fixes"(수동 계열 제외 제거, perp 우회, 고정 KS 표본, 비유한 사전 검사, 수렴 assert, float64, 원본 행 순서 로드)는 전부 실행 위생이며 모델링 신호가 아니다.

## 2단계: 세 제출물의 logit-rank 혼합

세 제출물을 각각 백분위 순위로 바꾸고 1e-6으로 잘라 logit 공간으로 옮긴 뒤, 자체 스택 0.50, amanatar 0.25, anthonytherrien 0.25로 가중 평균하고 sigmoid 역변환 후 다시 백분위 순위를 매긴다.
가중치 선정 근거 서술이 없고 OOF 검증도 없으므로 public 점수 외에는 판정 근거가 없다.
손으로 정한 가중 혼합은 우리 nested 로지스틱 결합이 CV 안에서 가중치를 학습하는 방식보다 방법론적으로 약하다.

## 우리 쪽 반영 판정

| 요소 | 기존 근거 | 판정 |
| --- | --- | --- |
| rank-gauss 로지스틱 스택 | `rank_gauss_logistic`이 fold 안전 형태로 구현되어 이슈 64에서 rank_logit 이중 표현에 패배 | 반영할 것 없음 |
| AUC·KS 격리 관문 | 계보 미상 외부 풀 전용 방어 장치이고, 우리 풀은 고정 fold 자체 생성이라 무결성·중복 검사([이슈 63](https://github.com/tmheo/predicting-smartphone-addiction/issues/63) 계열)로 이미 대응 | 반영할 것 없음 |
| perp 잔차 보정 우회 | adarsh1077 검토에서 이미 확인된 요소이며, 잔차 보정 축은 조건부 티켓 [186](https://github.com/tmheo/predicting-smartphone-addiction/issues/186)이 담당 | 186의 조건과 내용 변경 없음 |
| 외부 OOF 풀 186쌍 | 지도 172 범위 규칙상 직접 편입 금지, 고계수 계열은 174의 재현 티켓들로 이미 소화 | 반영할 것 없음 |
| 공개 제출물 logit-rank 혼합 | 외부 제출물 직접 편입 금지(지도 172 범위 밖), 입력 두 개가 공개 제출물 재포장 | 반영할 것 없음 |

새 티켓과 안개(Not yet specified) 갱신은 없다.
public 0.97114는 지도 172의 목적지 확인 지표(0.97100) 문맥에서 외부 혼합 없이 도달할 수 있는 수준이 아니라는 점만 재확인해 주며, 이는 [public 0.97100에 필요한 자체 OOF 수준 추정](https://github.com/tmheo/predicting-smartphone-addiction/issues/175)의 결론(자체 nested 약 0.9700 필요)과 일관된다.
