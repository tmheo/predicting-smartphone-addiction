# szymonkapiski S6E8 TabM 제약 결측·격자 TE 노트북 정밀 분석

## 출처와 판본

- 원문: [S6E8 TabM with constrained imputation](https://www.kaggle.com/code/szymonkapiski/s6e8-tabm-with-constrained-imputation), 작성자 Szymon Kłapiński(`szymonkapiski`).
- Kaggle kernel id 129458801, 최근 공개판 2026-08-02T11:47:27Z(코드 노트북 목록 46번).
- 2026-08-20에 `kaggle kernels pull`로 내려받았고, `.ipynb`의 SHA-256은 `95a8fd0e2030cb34bd9fc10aaa4ea55ff9819522ac455a19abc090dd61b4d5e6`이다.
- 공개 노트북 소스는 Apache License 2.0이다(`docs/agents/kaggle-public-notebook-licensing.md`).
  이 문서는 분석만 하고 코드를 복사하지 않는다.
  코드를 옮겨 쓰게 되면 그 시점에 라이선스 절차(원문 주소·판본·수정 표시)를 구현 기록에 남긴다.
- 선언 자료는 저자의 [OOF 라이브러리](https://www.kaggle.com/datasets/szymonkapiski/s6e8-oof-library-47-models)뿐이며, 게시판은 `RUN_TRAINING = False`로 저장된 예측을 읽어 점수만 재확인한다.
  저자 스스로 "노트북을 True로 끝까지 돌린 적은 없고, 같은 코드를 스크립트로 T4에서 돌린 결과"라고 명시한다.

## 노트북 요약

단일 TabM(pytabkit `tabm-mini-normal`, 5-fold StratifiedKFold shuffle seed 42, fold 내 3시드 평균)으로 OOF 0.96867, public 0.96967을 보고한다.
공개 5-fold 규약이 우리 고정 fold와 같아 수치를 직접 비교할 수 있다.
구성 요소는 세 가지다.

1. 제약 결측 재구성: 생성 규칙 `daily >= social + gaming + work`(완전 행 421,427개에서 위반 0)를 산술 경계로 써서 iterative imputer(BayesianRidge, max_iter 12) 추정치를 실현 가능 구간으로 자른다.
   마스킹 복원 실험에서 MAE 0.679로, 무제약 iterative imputer 0.691과 중앙값 대체 1.089를 이긴다고 보고한다.
2. 신경망 전용 비율·조성 15열: 트리는 스스로 비율을 재발견하므로 주지 않고, 신경망에만 준다는 주장.
3. 전 해상도 격자 target encoding: 모든 예측 열을 문자열 범주로 만들고, r1 반올림·내림 조밀화와 쌍 셀 4종을 더하며, 평활 TE(m=10) 옆에 셀 개수(빈도) 열을 같이 내보낸다.

Honest notes에서 저자가 직접 밝히는 핵심은 다음이다.

- 구조 탐색 5종의 스프레드는 0.00039뿐이라 구조 축은 소진됐고, 실제로 점수를 움직인 것은 fold 내 3시드 평균(+0.0002 규모, 5/5 fold 승리)이다.
- 이 단일 모델은 저자의 49구성원 blend에 기여 0이다.
  기존 단일 시드 TabM들과 상관 0.9985 이상이라 "덜 노이즈한 같은 모델"은 blend에 새 정보가 없다.
- 전체 앙상블의 가치는 0.00092다(단일 public 0.96967 vs blend 0.97059).
- CV 대비 public 오프셋은 단일 모델 +0.00100, blend +0.00117로 상수가 아니다.

## 우리 저장소와의 관계

exp065_tabm(#61)이 이미 이 노트북을 우리 fold 규율로 옮긴 것이다.
TabM 하이퍼파라미터와 fold 내 3시드 평균은 원문 그대로, 결측 재구성은 fold-fit 제공자(#74, #86), 신경망 비율 역할은 #90 채택 조성 5열, TE는 내부 10-fold OOF 정확값 TE로 대체했다.
결과는 3시드 OOF 0.96833(풀 장부, 스피어만 최근접 0.99345, 앙상블 기여 +0.00008)이다.

같은 fold에서 노트북 0.96867과의 격차는 +0.00034다.
이 격차가 아래 미채택 델타들의 크기 상한 추정이 된다.

## 섹션별 정밀 분석

### Constrained imputation

노트북의 구현 세부는 다음과 같다.

- imputer는 화면 블록 6열로 train+test 결합 자료에서 한 번 fit한다(타깃 무관·행 단위라 라벨 누출은 아니라는 논거).
- daily 결측이면 관측 성분 합을 하한으로, 성분 하나 결측이면 `[0, slack]`으로 추정치를 자른다.
- 경계 자체를 특성으로 내보낸다: `daily_lower_bound`, `daily_bound_bind`(하한이 추정치를 밀어올린 크기), `*_upper_bound`(-1 채움), `*_bound_width`, `*_was_missing` 지표, `other_screen_imp`, `comp_share_imp`, `identity_violation`, `n_screen_missing`, `n_missing_all`.

우리와의 대조.

- 재구성 본체는 이미 흡수했다: `ConstrainedImputeAux`(#74)가 같은 산술 경계 클리핑을 fold-fit으로 수행하고, nowidth 변형(exp026)이 당시 champion까지 됐다.
- 경계 폭 열은 seed 42 스크리닝에서 gain importance가 플라시보 미달이라 뺐고(#74), 결측 지표·개수 열은 지도의 배제 경계이며 원저자 자신의 ablation도 무효 보고다(exp065 설정 주석).
- train+test 결합 fit은 우리 규율이 fold-fit으로 통일했고, 전이 학습의 한계 기여 분리는 `XgbImputeAux`의 조건부 통로(#86)가 이미 있다.

판정: 새로 캘 것이 없다.
남은 델타(`daily_bound_bind`, `comp_share_imp` 같은 경계 파생)는 폭 계열 기각과 원저자 ablation 무효가 겹쳐 근거가 약해 재개하지 않는다.

### Ratio and composite features

`build_continuous`의 15열을 우리 판정 이력에 대응시키면 세 무리로 나뉜다.

| 무리 | 노트북 열 | 우리 이력 |
| --- | --- | --- |
| 이미 채택 | `social_over_daily`(=social_frac), `comp_sum_imp`·`sum_all_screen`(other_screen·screen_slack·resid_frac이 같은 신호) | #90 조성 5열 채택, exp065 파생 2열 |
| GBDT 스크리닝 기각 | `weekend_over_daily`(=wk_ratio), `awake_share`(≈awake_screen_frac), `notif_per_open`, `screen_per_open`(≈min_per_open) | #90에서 12열 중 7열 기각 |
| 우리 미판정 | `over_9h`, `under_3h`(EDA S-곡선 무릎 3h·9.5h 포화 앵커), `screen_over_sleep`, `screen_minus_sleep`, `gaming_over_daily`, `weekend_minus_daily`, `screen_mean_dw` | 판정 기록 없음 |

사용자 질문("우리는 LightGBM에서 비율 특성을 안 쓰는데 신경망에서는 의미 있다고 한다")에 대한 판정.

- 전제 정정: 우리는 트리에서 비율을 전부 빼지 않았다.
  #90에서 조성 5열(social_frac, work_frac, leisure_frac, resid_frac, week_total)이 CatBoost champion을 +0.00019 개선해 채택됐고, 현 champion 계열과 LightGBM 풀 구성원 exp117도 같은 5열을 쓴다.
  따라서 노트북의 "GBDT에 조성은 -0.0003"이라는 일반화는 우리 실측과 다르다.
  비율이 트리에 유해한 게 아니라, 어느 비율이냐에 따라 갈렸다.
- 그러나 방법론 주장(트리 기각이 신경망 기각을 함의하지 않는다)은 우리 체계의 실제 빈틈이다.
  #90의 7열 기각은 전부 GBDT(LightGBM 대리 + CatBoost 본판정) 기준이고, TabM은 GBDT가 채택한 5열만 물려받았다.
  신경망에서 기각 7열과 미판정 8열을 판정한 적은 한 번도 없다.
- 전례의 무게: 트리 기각을 신경망에서 재판정한 유일한 전례 exp066(orig_nn10_mean)은 기각됐다(OOF +0.00002, 순열 중요도 플라시보 미달).
  다만 그것은 라벨 프록시 계열(정확값 TE와 중복)이고, 비율·포화 앵커는 입력 변환 계열이라 메커니즘이 다르다.
  반론도 성립한다: TabM의 pwl 수치 임베딩은 단일 변수 꺾임(over_9h, under_3h)을 이미 근사할 수 있어, 이득이 있다면 두 변수 결합(비율·차)에 몰려 있을 가능성이 크다.

### Lattice keys and target encoding

노트북의 구현 세부는 다음과 같다.

- 수치·범주 12열 전부를 문자열화한 전 해상도 키에, RICH_KEYS로 r1 반올림·내림 조밀화 18열과 쌍 셀 4종(daily×weekend, social×gaming, daily×social, stress×academic)을 더한다.
- TE는 fold의 학습부에서만 fit하는 평활 평균(m=10)이고, 모든 키에 셀 개수(빈도) 열을 병행해 "모델이 얇은 셀을 불신할 수 있게" 한다.
- 저자는 조밀화·쌍 셀이 동등 LightGBM을 0.96632에서 0.96740으로 +0.00108 올렸다고 보고한다.

우리와의 대조.

- 전 해상도 정확값 TE는 champion 구성으로 이미 있다(내부 10-fold OOF TE).
- 빈도 인코딩은 #49에서 5종 전부 스크리닝 미달, 쌍 격자 TE는 #75에서 champion 스크리닝 -0.00037(조밀화 변형 -0.00050) 기각이되 exp035가 다양성 구성원으로 풀에 있다.
- 저자의 +0.00108과 우리 -0.00037의 충돌은 기준선 차이로 읽힌다.
  저자의 LightGBM 기준선 0.96632는 낮고, 우리 champion 계열은 정확값 TE·원본 prior·복원 계열로 그 신호를 이미 다른 경로에서 흡수했다.
  GBDT 쪽 재개 근거는 없다.
- 신경망 쪽은 별개다.
  exp065의 TE 블록에는 조밀화도, 쌍 키도, 빈도 열도 없다.
  #61이 "기각 계열이라 열지 않음"으로 닫았는데, 그 기각 역시 전부 GBDT 판정이었다.
  특히 빈도 열은 "TE 평균의 신뢰도" 신호라 TE를 강하게 쓰는 신경망에서 역할이 다를 수 있다.
- TE fit 방식 차이도 있다.
  노트북은 학습부 자기 포함 평활 TE, 우리는 내부 OOF TE다.
  측정은 둘 다 외부 fold로 채점하니 정직하고, 다른 것은 학습 신호의 강도·편향 균형이다.
  규율 변경이 필요한 축이라 1차 재개 대상에서는 뺀다.

### Honest notes 판독

- fold 내 시드 평균은 #127로 이미 채택했다.
  독립 재발견이므로 수렴 증거다.
- "더 좋은 단일 모델이 자동으로 더 좋은 blend 구성원이 아니다"(상관 0.9985면 기여 0)는 우리 중복 게이트(스피어만 문턱)와 같은 설계 원리다.
  실무 함의: 아래 재개 실험이 성공해도 exp065 옆에 추가하는 게 아니라 exp065를 교체해야 한다.
- 전체 앙상블 가치 0.00092(0.96967 → 0.97059)는 우리 규모(단일 최고 0.96920, 풀 nested 0.96967, public 0.97077)와 정합한다.
  우리 파이프라인이 이 저자보다 앞서 있다.
- CV 대비 public 오프셋이 산출물 종류에 따라 다르다(+0.00100 vs +0.00117)는 관찰은 #175의 오프셋 추정과 #188 마일스톤 제출 판독에 참고 근거를 더한다.

## 재개 판정

exp065와 노트북의 격차 +0.00034 중 우리가 의도적으로 닫았던 두 축을, "GBDT 기각은 신경망 기각이 아니다"라는 새 근거로 지도 172(과거 탈락 재검토)에 재개 티켓으로 연다.

1. TabM에 신경망 전용 연속 특성 확장(미판정 8열 + GBDT 기각 4열)을 더한 짝비교.
2. TabM의 TE 블록에 조밀화·쌍 키·빈도 열을 더한 짝비교.

두 티켓 모두 LightGBM 대리 스크리닝을 쓰지 않는다.
비대칭 가설 자체가 "대리 학습기가 틀린 판정자"라는 주장이므로, seed 42 TabM 짝비교로 직접 선별한다.

재개하지 않는 것과 이유.

- 제약 결측의 경계 폭·결측 지표 파생: 폭 계열 플라시보 미달(#74)과 원저자 ablation 무효 보고가 겹친다.
- GBDT 대상 격자 조밀화·빈도 재개: 우리 champion 문맥 실측(-0.00037, #49 전부 미달)이 저자의 낮은 기준선 이득보다 우선한다.
- 자기 포함 평활 TE로의 전환: fold 규율 변경이라 위 두 티켓이 통과할 때만 2차 델타로 고려한다.
