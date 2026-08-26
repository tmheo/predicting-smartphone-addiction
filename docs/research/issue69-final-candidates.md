# 이슈 #69 최종 제출 후보 두 개 확정

[P5: 최종 제출 후보 두 개 확정](https://github.com/tmheo/predicting-smartphone-addiction/issues/69)의 실행 기록이다.
판단은 [P5: 최종 제출 후보 두 개의 선정 전략 확정](https://github.com/tmheo/predicting-smartphone-addiction/issues/225)에서 끝났고, 여기서는 동결 풀 35개에 그 규칙을 기계 적용했다.
판정 기록은 `artifacts/judgments/issue69-final-candidates.yaml`, 적용 프로그램은 `scripts/select_issue69_final_candidates.py`다.

## 입력

- 후보 풀 `artifacts/pool.yaml` SHA-256 `caa1b907...`(35개), 재학습 계획 `artifacts/full-refit-plan.yaml` SHA-256 `2c56c63f...`(103회).
- 결합 전략은 [#337](https://github.com/tmheo/predicting-smartphone-addiction/issues/337)이 재확정한 `shrunk_rank_logit_logistic`이고 재학습 계획의 `protocol.combiner`와 같다.
- 전략별 nested OOF는 #337의 같은 실행(`run-logs/issue337/ensemble-evaluation.json`, SHA-256 `5faa7fc6...`)에서 읽었다.
- 실행 커밋 `54e166a`, 깨끗한 작업 폴더.

## 전체 자료 재학습

계획 파일은 #415와 #419로 구성원 3개가 더해지면서 바뀌었지만, 기존 32개 구성원의 항목은 바이트 단위로 같았다.
그런데도 조립 관문이 장부 파일 전체의 SHA-256을 요구해 [#226](https://github.com/tmheo/predicting-smartphone-addiction/issues/226)의 32개 산출물이 무효가 되고 103회를 다시 돌려야 하는 상황이었다.
사용자 결정으로 관문의 단위를 장부 전체에서 구성원 항목 해시로 바꿨다([ADR 0002](../adr/0002-full-data-refit-protocol.md)의 "관문의 단위는 장부가 아니라 구성원이다").
그 결과 32개 산출물은 그대로 관문을 통과했고, 신규 3개만 로컬 CPU에서 학습했다(exp197 1분, exp183 3분, exp168 2분, 모두 시드 3개).
Vast.ai 자원은 만들지 않았다.

`pipeline.refit --assemble`(3분 38초)이 만든 세 예측의 배열 SHA-256은 CV 전용 `f4c00fc5...`, 전체 자료 전용 `c207a4a4...`, 5:1 혼합 `ebbaeb9b...`다.
구성원별 CV 대 전체 자료 예측의 스피어만 상관은 최소 0.98333(exp113), 중앙값 0.99816, 최대 0.99985이고 신규 3개는 0.99660~0.99972다.

## 선정

| 축 | nested OOF | 첫 후보 대비 | 자격(-0.0005 이내) | 첫 후보와 스피어만 |
| --- | ---: | ---: | --- | ---: |
| 첫 후보: `shrunk_rank_logit_logistic` 5:1 혼합판 | 0.9698105828 | - | - | - |
| 같은 결합의 CV 전용판 | 0.9698105828 | 0.0 | 통과 | 0.99987 |
| 전 구성원 순위 평균의 5:1 혼합판 | 0.9691305960 | -0.00068 | 미달 | 0.99783 |

- 첫 후보는 규칙대로 `shrunk_rank_logit_logistic`의 5:1 혼합판으로 고정했다.
  파일 `artifacts/submissions/issue69-candidate-1.csv`, SHA-256 `7c57a11a2bb48624fe6ea6e3429acbad2d50f3248a140799ef4a133f9e378d4e`.
- 둘째 후보는 자격을 갖춘 축이 CV 전용판 하나뿐이라 그 축이다.
  순위 평균은 열세 상한을 넘어 자동 탈락했고, #225가 예고한 대로 별도 대비책 없이 CV 전용판이 남았다.
  파일 `artifacts/submissions/issue69-candidate-2.csv`, SHA-256 `fae78bda8c1160bfbca6e14b1813c8e2ff61fae4eb94d313e439ae2dacd59b52`.
- 둘째 후보 파일은 #337이 제출한 CV 전용판(파생 실행 `b24e5ba7`의 `submission.csv`, Kaggle ref 55791893, Public 0.97096)과 바이트 단위로 같다.
  즉 둘째 후보는 이미 Kaggle에 올라가 있고, 새로 올릴 것은 첫 후보 하나다.
- 혼합 제외 예외는 발동하지 않는다.
  #226의 짝 계측에서 5:1 혼합은 CV 전용 대비 +0.00005(0.97082 → 0.97087)로 악화가 아니었다.

## 업로드

사용자 확인 뒤 2026-08-26T12:46:51Z에 첫 후보를 올렸다(Kaggle ref 55795055).
Public 0.97099로 둘째 후보(0.97096)보다 +0.00003이고, #66·#226의 혼합 효과(+0.00006, +0.00005)와 같은 부호다.
사후 확인값이며 선택 근거로 쓰지 않는다.
파생 실행은 `pipeline.submit --record-existing`으로 `e88f706e`에 기록했고(원본 실행 `b24e5ba7`, 커밋 `69b9ca8`), 조립 manifest와 판정 기록을 산출물로 붙였다.
두 후보는 계정의 공개 점수 상위 2개라 Kaggle 기본 최종 선택에 포함된다.
