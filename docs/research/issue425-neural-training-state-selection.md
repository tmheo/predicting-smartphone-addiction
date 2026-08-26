# 신경망 학습 상태 대조축의 직렬 nested OOF 판정 결과

이 문서는 [두 신경망 학습 상태 대조축을 직렬 nested OOF로 판정해 최대 2개를 등록한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/425)의 후보 자격 확인, 다중 후보 완전 중첩 선택과 공식 후보 풀 유지 근거를 기록한다.
기계 판독 결과는 `artifacts/issue425-neural-training-state-selection-results.yaml`이고 변경 불가 판정 기록은 `artifacts/judgments/issue425-bulk-selection-*.yaml`이다.

## 동결 입력과 판정 규칙

시작 후보 풀은 35개이며 `artifacts/pool.yaml`의 SHA-256은 `caa1b90769720a4accbe07074dbc7efe0335ab6657fea80c6839b60121dc39d3`이다.
결합 전략은 `missing_segmented_rank_logit`, `missing_interaction_rank_logit`, `shrunk_rank_logit_logistic` 세 개로 고정했다.
후보 순서는 Lookup-Transformer 6회, 8회, 12회와 Muon RealMLP 1회, 2회, 3회로 결과 확인 전에 고정했다.
후보를 등록하려면 현재 풀 대비 전체 nested OOF 차이가 `+0.00002` 이상이고 바깥쪽 검증 분할 5개 가운데 4개 이상에서 양수여야 한다.
통과 후보가 있으면 한 건만 등록하고 남은 후보를 갱신된 풀에서 다시 판정하되 최대 두 건에서 멈추도록 고정했다.

## 후보 자격

여섯 후보는 모두 3시드 평균본, 깨끗한 커밋, 고정 분할 내용 해시, 학습 상태 게시 계보, 카나리아와 진입 하한 검사를 통과했다.
현재 풀과의 최근접 스피어만 순위 상관계수는 모두 중복 문턱 `0.998` 미만이었다.

| 후보 | OOF AUC | 최근접 풀 구성원 | 스피어만 |
| --- | ---: | --- | ---: |
| Lookup-Transformer 6회 | `0.9651969073` | `exp168_issue413_lgb_no_te_fixed20` | `0.9902435144` |
| Lookup-Transformer 8회 | `0.9678534167` | `exp081_lookup_fold_initialization_avg3` | `0.9921785809` |
| Lookup-Transformer 12회 | `0.9689316286` | `exp081_lookup_fold_initialization_avg3` | `0.9934445805` |
| Muon RealMLP 1회 | `0.9663926684` | `exp140_realmlp_orig_cdf_diff` | `0.9845825879` |
| Muon RealMLP 2회 | `0.9678658084` | `exp134_realmlp_muon` | `0.9926121930` |
| Muon RealMLP 3회 | `0.9683137729` | `exp134_realmlp_muon` | `0.9965495142` |

## 다중 후보 판정

첫 시도 `issue425-bulk-selection-1`은 작업 공간의 `data` 기호 연결이 저장소 밖을 가리켜 입력 경로 관문에서 판정 불가로 끝났고 후보 풀은 바꾸지 않았다.
같은 원본 자료를 저장소 안 경로의 하드 링크로 제공해 내용 해시를 유지한 뒤 새 변경 불가 판정 `issue425-bulk-selection-2`를 실행했다.

유효 판정에서 현재 풀의 nested OOF AUC는 `0.9698105828357245`이고 다중 후보 선택 절차를 포함한 AUC는 `0.9698098853083991`이다.
차이는 `-0.0000006975273254550274`이고 바깥쪽 검증 분할 승수는 `3/5`다.
전체 nested OOF 차이는 `+0.00002` 문턱에 미달했고 분할 승수도 `4/5` 조건에 미달했다.

| 바깥쪽 검증 분할 | 안쪽 선택 후보 | AUC 차이 |
| ---: | --- | ---: |
| 0 | Lookup-Transformer 12회 | `+0.000000866706` |
| 1 | Lookup-Transformer 8회 | `-0.000009498247` |
| 2 | Muon RealMLP 2회 | `+0.000001002192` |
| 3 | Lookup-Transformer 8회 | `+0.000007259932` |
| 4 | Lookup-Transformer 12회 | `-0.000003118246` |

전체 OOF 참고 선택은 Muon RealMLP 2회였지만 바깥쪽 검증 분할 선택은 Lookup-Transformer 8회와 12회가 각각 두 번, Muon RealMLP 2회가 한 번으로 갈렸다.
어느 후보도 분할 과반을 얻지 못했고 선택 절차 전체도 음수였으므로 결과 확인 뒤 단독 후보를 골라 재판정하지 않았다.

## 결론

공식 후보 풀에 등록할 신규 구성원은 없다.
후보 풀은 35개와 SHA-256 `caa1b90769720a4accbe07074dbc7efe0335ab6657fea80c6839b60121dc39d3`을 그대로 유지한다.
통과 후보가 없으므로 두 번째 직렬 판정, 신규 구성원 전체 자료 재학습과 최종 제출물 재생성은 필요하지 않다.
[통과한 신규 구성원만 전체 자료로 학습하고 최종 제출물을 재생성한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/429)는 새 학습 없이 현재 35개 안전판의 해시와 제출 후보를 다시 확인하는 경로로 이어진다.
