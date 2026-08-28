# 규제 강도 선택 확장 스택의 사용자 결정 조립·업로드 (이슈 #489)

[report.md](report.md)의 선택 절차 대조는 사전 고정 문턱(`+0.00002`, 바깥쪽 분할 5/5)에 미달했다(`+0.0000099`, 4/5).
2026-08-28 사용자가 결과를 본 뒤 그 문턱을 접고, nested 점추정이 높은 C 선택판으로 두 번째 최종 제출을 바꾸기로 결정했다.
판정 기록과 문턱은 바꾸지 않았고, 이 문서는 그 결정 아래의 조립과 업로드 기록이다.

## 조립

도구는 `scripts/assemble_c_selected_extended_stack.py`(커밋 `c3778b1`)이며 판정 도구의 `full`(통과 판정에서만 열림)을 쓰지 않았다.
precommit 무결성, 입력 해시, 결합기 module 해시(판정 때와 같은 코드)를 다시 확인한 뒤 전체 313 OOF에서 같은 규칙으로 `(C, λ)`를 골랐다.

- 제안 `(C, λ) = (0.03, 1.0)`. 전체 OOF 5분할 leave-one-fold-out 내부 AUC는 C=0.03이 `0.9703624`, C=0.01이 `0.9703607`, C=0.3이 `0.9703563`, C=1.0이 `0.9703532`다(선택 점수이며 판정에 쓰지 않음).
- 전체 OOF 적합 반복 208회, 계수 L2 크기 `2.3823`(C=1.0 현재 판은 `2.9`대), in-sample OOF AUC `0.9705276`(참고치).
- 시험 행렬은 현재 제출과 같다: 자체 35의 5:1 모델 수 가중 혼합판, 외부 278의 장부 시험 배열, 열·행 순서 동일(#457 manifest 해시 대조).
- 제출 CSV `artifacts/submissions/issue489-c-selected-extended-stack.csv`, SHA-256 `274553ccd79b51276ddb441167a7bc6071fec4f60655eca5845831a9476b3090`, 296,302행, 전부 유한, [0, 1] 안, 동률 없음.
- 현재 두 번째 장 `443b3a71`(`a4d9c5db…`)과 스피어만 `0.999939`, 안전판 `e88f706e`와 `0.997185`.
- 조립 manifest는 [assembly-manifest.json](assembly-manifest.json)이며 `user_override`에 결정 문구를 남겼다. git `706050e`, dirty 없음, 948초.

## 업로드와 기록

2026-08-28T13:24:52Z에 올렸다(Kaggle ref `55844886`, 당일 첫 제출).
**Public 0.97135**로 현재 두 번째 장 `443b3a71`(0.97135)과 같은 값이다.
이 값은 사후 확인값이고 판정에 쓰지 않는다.
nested 증분 `+0.0000099`는 Public 눈금(소수 5자리)의 해상도 아래다.
다만 사용자가 확인한 Public 리더보드 순위는 2계단 올랐으므로 표시 자리수 아래에서는 높아진 것으로 본다.

MLflow 기록은 `pipeline.submit --record-existing 55844886`으로 남겼다.
파생 실행 `30b6f97c30904995a79e476f02decf8f`(실행 이름 `ensemble_c_selected_shrunk_rank_logit_logistic_issue489_extended_stack_own35_ext278`, `source.run_id=443b3a71`, `git_commit=706050e`)에 제출 CSV, 조립 manifest, 판정 comparison·precommit·report를 첨부했다.
param에 구성원 수(313 = 자체 35 + 외부 278), `(C, λ)`, C 격자, 판정 수치(후보·대조군 nested, 차이, 분할 양수, 문턱, 판정)를, tag에 `user_override=true`를 적었다.
metric에는 `auc_oof = 0.9703609`(nested), `weighted_oof_auc = 0.9712273`, `auc_fold_0..4`, `current_plate_auc_oof = 0.9703509`, `delta_vs_current_plate = +0.0000099`, `delta_fold_k_vs_current_plate`, `delta_vs_pool35_source = +0.000550`, `auc_oof_insample = 0.9705276`(참고치)를 comparison.json과 assembly-manifest.json에서 올렸다.

## 최종 두 장

| 장 | 실행 | Kaggle ref | Public | nested |
| --- | --- | --- | --- | --- |
| 1 (안전판) | `e88f706e` | 55795055 | 0.97099 | 0.9698106 |
| 2 (규제 강도 선택 확장 스택) | `30b6f97c` | 55844886 | 0.97135 | 0.9703609 |

새 제출과 이전 두 번째 장 `443b3a71`(55823369)의 Public이 같으므로 Kaggle 자동 선택이 어느 쪽을 잡을지 보장되지 않는다.
www.kaggle.com에서 최종 두 장을 안전판 `e88f706e`(55795055) + 새 제출 `30b6f97c`(55844886)로 수동 고정해야 하며, 이 단계는 사용자가 직접 확인한다.
이전 두 번째 장 `443b3a71`(55823369)은 최종 선택에서 빠진다.
