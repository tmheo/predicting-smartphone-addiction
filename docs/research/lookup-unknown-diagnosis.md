# Lookup-Transformer 조회 어휘 미등록값 진단

이 문서는 [진단: Lookup-Transformer의 검증·테스트 UNK 비율 측정과 거친 키 보조 lookup 개설 여부](https://github.com/tmheo/predicting-smartphone-addiction/issues/128)의 재현 가능한 근거다.

## 질문과 판정 기준

조회 어휘 미등록값은 결측이 아니면서 해당 fold의 학습 부분에 같은 정확값이 없어 Lookup-Transformer의 미등록 식별자로 처리되는 값이다.
검증과 테스트에서 이 값이 충분히 자주 나타나고 OOF 오류에 불균형하게 크게 기여할 때만 거친 키 보조 조회 실험을 연다.

## 재현

다음 명령은 확정 champion exp067의 3시드 OOF를 기준으로 fold별 어휘 미등록 비율과 오류 기여를 함께 계산한다.

```bash
uv run python scripts/diagnose_lookup_unk.py \
  --config configs/exp067_lookup_xgb_impute_comps5.yaml \
  --run-id 2bd55026ae63430aa774bce20a359b4a
```

## 측정 결과

검증 전체 691,369행 가운데 조회 어휘 미등록값이 하나라도 있는 행은 120개, 0.0173569%였다.
한 행에 두 열 이상이 동시에 미등록인 경우는 없었다.
fold별 검증 비율은 0.0166336%에서 0.0195266%, 테스트 비율은 0.0172122%에서 0.0195746%였다.

미등록값이 나타난 열은 네 개뿐이었다.
검증에서는 `daily_screen_time_hours` 50건, `social_media_hours` 25건, `weekend_screen_time` 45건이었고 `work_study_hours`는 0건이었다.
테스트에서는 fold별로 `daily_screen_time_hours` 12건에서 18건, `social_media_hours` 9건에서 14건, `work_study_hours` 3건, `weekend_screen_time` 24건에서 27건이었다.
나머지 여덟 열은 모든 fold의 검증과 테스트에서 0건이었다.

확정 champion exp067의 OOF AUC는 0.969097840이었다.
미등록값이 없는 691,249행의 AUC는 0.969094303이고, 미등록값 포함 120행의 AUC는 0.984414643이었다.
미등록값 포함 행은 전체 행의 0.0173569%이지만 전체 로그 손실의 0.0114806%만 차지했다.

미등록값 포함 행이 적어도 하나 들어간 양성-음성 순위 쌍은 전체의 0.0335739%였다.
그 쌍에서 현재 발생한 모든 순위 오류를 완전히 없앤다는 낙관적 상한도 전체 AUC +0.000006839에 그친다.
이는 ADR 0001의 champion 확정 문턱 +0.0001의 약 6.8%다.

## 이슈 108과 exp080의 관계

[P3 보강: Lookup-Transformer의 제한적 용량·규제 재검증](https://github.com/tmheo/predicting-smartphone-addiction/issues/108)의 exp080은 seed 42 스크리닝 OOF 0.968763258로 3시드 확정 재검증 중이다.
exp080은 exp067과 같은 데이터, fold, 피처 계획과 조회 열을 쓰고 임베딩 감쇠만 바꾸므로 조회 어휘 미등록 행과 비율은 이 진단과 완전히 같다.
진단 시점에는 exp080 확정 실행이 로컬 실행 저장소에 반입되지 않아 OOF 오류 기여는 확정 champion exp067로 측정했다.
exp080의 최종 채택 여부는 이 진단의 미등록 비율이나 실험 개설 결론을 바꾸지 않는다.

## 결정

거친 키 보조 조회 실험을 열지 않는다.
조회 어휘 미등록값은 행 비율과 오류 기여가 모두 미미하고, 현재 오류를 완전히 제거해도 champion 확정 문턱에 한참 못 미친다.
반올림 조회를 모든 행에 적용해 얻을 수 있는 별도 표현 효과는 이 미등록값 손실의 근거가 아니며, 이 진단의 후속 범위에 포함하지 않는다.
