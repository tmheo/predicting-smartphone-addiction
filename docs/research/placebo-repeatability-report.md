# 플라시보 반복성과 대표 충돌 실증 판독표

이슈 [#261](https://github.com/tmheo/predicting-smartphone-addiction/issues/261)의 결과 문서다.
이슈 [#256](https://github.com/tmheo/predicting-smartphone-addiction/issues/256)이 고정한 최소 실증 설계를 실행해, 새 피처 플라시보 게이트의 반복성과 대표 충돌을 최종 계약 결정(#262)에 쓸 원자료로 확정한다.
선행 재감사 원자료는 [충돌 장부](placebo-gate-conflict-ledger.md)와 `placebo-gate-conflict-evidence.json`(#255)이다.
이 문서의 기계 판독 자료는 `placebo-repeatability-evidence.json`이다.

## 실행 정보와 재현 명세

- 실행 커밋: `327a973898b09b9f8861adc92155d4f527cc4c6a` (깨끗한 트리, 모든 새 실행의 `git_dirty=False`).
- 환경 검증: `scripts/verify_environment_gates.sh` 통과(비공개 입력 검증, 시험 357개 전체 통과).
- 실행 도구: `scripts/diagnose_placebo_repeatability.py`.
- 새 실행은 모두 로컬 CPU LightGBM이라 외부 GPU를 쓰지 않았다.

재현 명령은 다음과 같다.

```sh
# 1) 깨끗한 같은 커밋의 짝지은 3시드·5-fold 재실행
uv run python -m pipeline.run configs/exp025_constrained_impute.yaml --stage confirm
uv run python -m pipeline.run configs/exp026_constrained_impute_nowidth.yaml --stage confirm

# 2) 플라시보 반복성 진단(모델 seed 42, fold 0, 플라시보 난수 101/202/303/404)
uv run python scripts/diagnose_placebo_repeatability.py refit \
    configs/exp027_recon_ce.yaml --placebo-seeds 101 202 303 404 --fold 0 --out <json>
uv run python scripts/diagnose_placebo_repeatability.py refit \
    configs/exp050_lgb_xgb_impute.yaml --placebo-seeds 101 202 303 404 --fold 0 --out <json>

# 3) 기존 실행 원자료의 같은 스키마 판독과 짝지은 OOF 차이
uv run python scripts/diagnose_placebo_repeatability.py readout <config 이름> \
    --importance <run>/artifacts/feature_importance.parquet --out <json>
uv run python scripts/diagnose_placebo_repeatability.py pair \
    --candidate-artifacts <run>/artifacts --baseline-artifacts <run>/artifacts --out <json>
```

## 판독 정의와 해석 제약

- 셀: fold 모델 하나(모델 seed x fold x 플라시보 난수)의 gain importance 표.
- 순위: 셀 안 전체 피처의 gain 내림차순 경쟁 순위(동률은 최고 순위).
- 백분위: 셀 안 다른 피처 중 gain이 엄격히 낮은 비율(%).
- 최솟값·중앙값·최댓값은 관측 요약이며 분위수나 오류율 추정이 아니다.
- 플라시보 난수 반복 4회를 경험적 P값, 95% 분위수, 오류율 추정으로 해석하지 않는다(#256).
- 역사 사례의 OOF 관찰을 오류율의 참값으로 간주하지 않는다(#256).

## 1. LightGBM 플라시보 반복성 진단

`exp027_recon_ce`(양성 대표)와 `exp050_lgb_xgb_impute`(음성 대표)에서 모델 seed 42와 fold 0을 고정하고 플라시보 난수 `[101, 202, 303, 404]`를 하나씩 적용해 fold 모델 8개를 다시 맞췄다.
기존 실행의 fold 0 셀(플라시보 난수 42)을 같은 판독표에 합쳐 실험당 셀 5개를 읽는다.
플라시보 열만 바뀌므로 fold 0 검증 AUC는 반복 간 0.9666~0.9667 범위에서 사실상 같았다.

### exp027_recon_ce: 빈도 인코딩 8열 (짝지은 OOF +0.00015927, fold 5/5 양수)

셀별 게이트 방향(플라시보보다 높은 새 피처 수 / 전체 새 피처 수):

| 플라시보 난수 | 42(원 실행) | 101 | 202 | 303 | 404 |
| --- | --- | --- | --- | --- | --- |
| 통과 열 수 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| 플라시보 gain | 27385.2 | 20922.5 | 31151.3 | 30890.2 | 24712.5 |

피처별 5셀 요약(비율 = 후보 gain / 플라시보 gain):

| 피처 | 통과 | 비율 최소/중앙/최대 | 순위 범위 | 백분위 범위 |
| --- | --- | --- | --- | --- |
| age_ce | 0/5 | 0.337 / 0.354 / 0.367 | 32 | 11.4 |
| app_opens_per_day_ce | 0/5 | 0.775 / 0.791 / 0.835 | 26~27 | 25.7~28.6 |
| daily_screen_time_hours_ce | 0/5 | 0.709 / 0.729 / 0.746 | 29~30 | 17.1~20.0 |
| notifications_per_day_ce | 0/5 | 0.914 / 0.919 / 0.978 | 24 | 34.3 |
| sleep_hours_ce | 0/5 | 0.783 / 0.809 / 0.820 | 26~28 | 22.9~28.6 |
| social_media_hours_ce | 0/5 | 0.702 / 0.719 / 0.811 | 27~30 | 17.1~25.7 |
| weekend_screen_time_ce | 0/5 | 0.770 / 0.771 / 0.801 | 28~29 | 20.0~22.9 |
| work_study_hours_ce | 0/5 | 0.851 / 0.883 / 0.917 | 25 | 31.4 |

40개 피처-셀 전부 미달이고, 어떤 플라시보 난수에서도 판정이 뒤집히지 않았다.
즉 exp027의 기각은 단일 플라시보 표본의 우연이 아니라 반복적이다.
짝지은 OOF가 fold 5/5 양수(+0.00015927)인 실신호 가능성 묶음을 게이트가 난수와 무관하게 기각하므로, 이 충돌은 구조적이다.

### exp050_lgb_xgb_impute: XGBoost 복원 9열 (미달 4열 제거 시 점수 유지, 올바른 기각 대조)

| 플라시보 난수 | 42(원 실행) | 101 | 202 | 303 | 404 |
| --- | --- | --- | --- | --- | --- |
| 통과 열 수 | 5/9 | 5/9 | 5/9 | 5/9 | 5/9 |
| 플라시보 gain | 19710.6 | 29664.5 | 25317.8 | 27898.8 | 28713.9 |

| 피처 | 통과 | 비율 최소/중앙/최대 | 순위 범위 | 백분위 범위 |
| --- | --- | --- | --- | --- |
| age_xgb_recon | 0/5 | 0.057 / 0.071 / 0.074 | 37 | 0.0 |
| app_opens_per_day_xgb_recon | 0/5 | 0.187 / 0.189 / 0.215 | 30~31 | 16.7~19.4 |
| daily_screen_time_hours_xgb_recon | 5/5 | 77.37 / 82.27 / 116.27 | 1 | 100.0 |
| gaming_hours_xgb_recon | 5/5 | 1.867 / 1.978 / 2.628 | 10 | 75.0 |
| notifications_per_day_xgb_recon | 0/5 | 0.153 / 0.158 / 0.162 | 35 | 5.6 |
| sleep_hours_xgb_recon | 0/5 | 0.159 / 0.173 / 0.177 | 33 | 11.1 |
| social_media_hours_xgb_recon | 5/5 | 23.90 / 25.44 / 35.91 | 2 | 97.2 |
| weekend_screen_time_xgb_recon | 5/5 | 4.007 / 4.268 / 5.822 | 7 | 83.3 |
| work_study_hours_xgb_recon | 5/5 | 1.284 / 1.338 / 1.741 | 15~18 | 52.8~61.1 |

통과·미달의 피처별 방향이 5셀 모두 같았고, 미달 4열은 원 실행의 제거 대조에서 점수가 유지된 열과 정확히 일치한다.
즉 게이트의 올바른 기각도 난수와 무관하게 반복적이다.

## 2. 깨끗한 같은 커밋의 짝지은 재실행: exp025 대 exp026

판정 단위 피처 묶음은 재구성 폭 3열(`gaming/social_media/work_study` `_recon_width`)이다.
두 실험을 커밋 `327a973`의 깨끗한 트리에서 `--stage confirm`(3시드 42/43/44, 5-fold)으로 한 번씩 재실행했다.

| 실행 | run | 3시드 평균본 OOF AUC |
| --- | --- | --- |
| exp025_constrained_impute (폭 3열 포함) | `aa32cdf9` | 0.967574689811006 |
| exp026_constrained_impute_nowidth (폭 3열 제외) | `e48c83c6` | 0.967547642359299 |

짝지은 차이(exp025 - exp026, 폭 3열 묶음의 기여):

- 3시드 평균본 OOF: **+0.00002705**.
- 시드별: 42는 -0.0000024, 43은 +0.0000650, 44는 +0.0000269로 2/3 시드 양수.
- 평균본 fold별: 4/5 양수(fold 0만 -0.0000248).

한편 같은 깨끗한 실행에서 폭 3열의 gain importance는 15셀(3시드 x 5-fold) 모두 플라시보 미달(비율 0.15~0.49)이다.

부수 확인으로, 깨끗한 exp025 재실행은 변경 사항이 남았던 옛 실행(`3f7d735f`)의 대표 OOF와 시드별 OOF를 기록 자릿수까지 그대로 재현했다.
#255가 옛 실행에 달았던 "깨끗한 재실행 필요" 단서는 해소됐고, 옛 원자료의 수치는 그대로 유효하다.

#256의 증거 단계에 대입하면 폭 3열 묶음은 3시드 평균본 양수, 시드 2/3 양수, 평균 차이가 +0.0002 미만일 때 요구되는 fold 3/5 이상 양수(4/5)를 모두 충족해 **실신호**다.
평균 차이도 채택 문턱 +0.00002 이상이다.
현행 게이트가 필수 관문이라면 이 실신호 묶음은 15/15셀 미달로 기각된다.

## 3. 기존 원자료 합류: exp070과 exp066

### exp070_cat_exact_cats (CatBoost, 정확값 범주형 9열)

원 실행(`6238d8c5`, 3시드·5-fold, 15셀)을 같은 판독 스키마로 읽었다.

- 짝지은 OOF: 3시드 평균 +0.0000364, 평균본 fold 5/5 양수, 시드별 2/3 양수.
- 후보 풀 진입 뒤 16구성원 nested OOF 바깥 분할 5개 모두 양의 계수(#255).
- 새 열 9개 중 7개가 평균 게이트 미달.
- 셀 판독: `app_opens_per_day_cat`(15/15)과 `notifications_per_day_cat`(15/15)만 전 셀 통과.
  `age_cat` 3/15, `gaming_hours_cat` 6/15로 셀 사이에서 방향이 흔들리고, 나머지 5열은 0/15다.
- 플라시보 순위는 47열 중 32~35위로, 셀마다 후보들과 순위가 교차한다.

exp070은 이미 #256이 대체 조건 충족의 근거로 고정한 사례이며, 이번 판독은 그 원자료를 같은 스키마의 기계 판독 자료로 합친 것이다.

### exp066_tabm_nn10 (TabM 순열 중요도, 단일 피처 orig_nn10_mean)

원 실행(`e5b7ba14`, seed 42, 5셀) 판독:

- `orig_nn10_mean`의 순열 중요도가 fold 4/5에서 플라시보보다 높지만, 평균은 플라시보의 0.87배로 미달.
- 플라시보 순열 중요도 자체가 음수~양수(-0.0000012~+0.0000360)를 오가며, fold 0 셀에서는 플라시보가 39열 중 38위까지 내려간다.
- 짝지은 OOF는 +0.0000221(fold 3/5)이지만 exp065 대비 중복도 0.99933이라, #256 결정대로 이 사례는 실신호 증거가 아니라 셀 사이 평균 비교의 불안정성만 뒷받침한다.

## 4. 종합 판독표

| 사례 | 묶음 | 짝지은 OOF 증거 | 게이트 판정 | 반복성 | 판독 |
| --- | --- | --- | --- | --- | --- |
| exp027_recon_ce | 빈도 인코딩 8열 | +0.00015927, fold 5/5 (단일 시드, 실신호 가능성) | 8/8열 미달 | 플라시보 난수 5종 모두 0/8 | 기각이 반복적, 충돌은 구조적 |
| exp025 - exp026 | 재구성 폭 3열 | +0.00002705, 시드 2/3, fold 4/5 (깨끗한 재실행, 실신호) | 15/15셀 미달 | 깨끗한 재실행이 옛 수치 완전 재현 | 실신호를 게이트가 기각 |
| exp070_cat_exact_cats | 정확값 범주형 9열 | +0.0000364, fold 5/5, nested 계수 5/5 | 7/9열 미달 | 15셀에서 열별 3/15~15/15로 혼재 | 실신호를 게이트가 기각(#256 대체 조건의 근거) |
| exp050_lgb_xgb_impute | XGBoost 복원 4열(미달분) | 제거해도 +0.0000039로 점수 유지 | 4/9열 미달 | 플라시보 난수 5종 모두 같은 4열 미달 | 올바른 기각도 반복적 |
| exp066_tabm_nn10 | orig_nn10_mean | +0.0000221이나 중복도 0.99933 | 평균 0.87배 미달 | fold 4/5 우위인데 평균은 미달 | 평균 비교의 불안정성 사례 |

## 5. #262 계약 결정에 넘기는 사실

- 게이트 판정은 플라시보 난수 선택에 대해 반복적이었다.
  이번 실증 범위에서 "난수에 따라 판정이 뒤집히는" 사례는 관찰되지 않았다.
- 반복적이라는 사실이 게이트를 구제하지 않는다.
  실신호로 확정된 폭 3열(깨끗한 재실행)과 대체 조건의 근거인 exp070을 게이트가 난수와 무관하게 기각하므로, 거짓 기각은 표본 우연이 아니라 게이트 정의(단일 정규 플라시보 대비 평균 gain)의 구조적 성질이다.
- #256의 대체 조건 중 "확인된 실신호 하나라도 기각"이 exp070(기존)과 폭 3열(신규 깨끗한 재실행)에서 성립한다.
- 음성 대조(exp050 미달 4열)의 올바른 기각도 반복적이므로, 게이트를 대체할 때 잃는 보호 효과의 실측 사례로 함께 넘긴다.

## 원자료 위치

- 기계 판독 자료: `docs/research/placebo-repeatability-evidence.json` (부분 판독의 병합본, 셀 단위 수치 포함).
- 새 실행 원본: 워크트리 로컬 MLflow(run `aa32cdf9`, `e48c83c6`)와 `run-placebo-repeat/` 진단 JSON.
  재현 명령과 커밋이 위에 있으므로 원본 보존 없이도 동일 수치로 재생성할 수 있다(exp025의 완전 재현으로 확인).
- 기존 실행 원본: 본 체크아웃 MLflow(run `67699e50`, `f9c37fe9`, `6238d8c5`, `e5b7ba14`, `3f7d735f`).
