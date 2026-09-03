# 통합판 19개와 kagglekit 문서의 겹침·충돌 대조표

이슈 628의 조사 결과다.
[회고 보고서](s6e8-top-writeups-vs-ours-retrospective.md)의 `다음 대회 운영 원칙 통합판` 절에 있는 원칙 19개를, kagglekit 형제 체크아웃의 기존 문서와 문장 단위로 대조했다.
이 대조표는 지도 #627의 초안 작성 티켓이 입력으로 쓴다.

## 읽는 법

- 대조 대상은 kagglekit `CONTEXT.md`, `docs/adr/0001`부터 `0007`, `docs/agents/verification-layers.md`, `docs/agents/vast-resource-control.md`, `docs/agents/kaggle-public-notebook-licensing.md`, `docs/agents/remote-execution.md`, `scaffold/AGENTS.md`다.
- 근거 문장의 파일 경로는 kagglekit 저장소 루트 기준이고, 줄 번호는 kagglekit 커밋 `4c563ab` 기준이다.
- 분류는 넷 중 하나다.
  겹침 없음은 대응하는 기존 문장이 없는 경우다.
  같은 뜻은 기존 문장이 원칙의 내용을 이미 담고 있는 경우다.
  더 강함은 기존 문장이 다루는 내용에 원칙이 조건, 범위 또는 주기를 더하는 경우다.
  충돌은 기존 문장이나 현행 관행과 원칙이 같은 단계에서 다른 답을 내는 경우다.
- 충돌 행의 비고에는 회고 보고서 `기존 원칙과의 충돌 해소` 절이 이미 해소 방향을 정했는지 적었다.
- 회고 보고서의 일반 해소 규칙은 단계가 다르면 병존한다는 것이다.
  기존 kagglekit 문서는 대부분 채택 관문과 기록 계약이고, 후보 생성 단계의 원칙(5, 6, 8)은 대응 문장이 없어 겹침 없음으로 두었다.

## 대조표

| 번호 | 원칙 요지 | 분류 | 근거 문장(파일:줄) | 비고 |
| --- | --- | --- | --- | --- |
| 1 | 계열별 최고 단일 목표와 결합 목표를 따로 관리하고 최고 단일 점수를 단계별로 분해 기록 | 더 강함 | `docs/adr/0001-experiment-adoption-contract.md:8` (champion 교체·다양성·앙상블을 계열별로 분리) · `CONTEXT.md:165-167` (champion 정의) | 기존은 단일 champion 하나와 앙상블 판정을 분리할 뿐, 모형 계열마다 최고 단일 목표를 두거나 원시·기본 특성·튜닝·고급 특성 단계로 분해해 기록하는 규정은 없다. |
| 2 | 각 회차를 설정, 분할, 시드, OOF, 시간, 실패 조건의 최소 공통 형식으로 기록 | 같은 뜻 | `docs/agents/verification-layers.md:10-17` (기록 계층 항목) · `scaffold/AGENTS.md:11` (첫날 계약 1번) · `docs/adr/0001-experiment-adoption-contract.md:16` (시드별 OOF AUC 기록) · `CONTEXT.md:126-128` (진행 기록의 단계별 소요 시간) | 실패 조건(중단 조건)의 사전 고정은 `docs/agents/vast-resource-control.md:23`에 Kaggle CPU 조건으로만 있어, 초안에서 기록 계층 항목으로 일반화할 여지가 있다. |
| 3 | 주 1회 대표 결합의 확인 제출을 정례화하고 결합 OOF·Public·Private를 한 표에 기록하되 판정에는 미사용 | 더 강함 | `docs/adr/0001-experiment-adoption-contract.md:27` (Public 점수는 마일스톤 건전성 점검 전용) · `docs/agents/verification-layers.md:31` (제출 직전 차단 관문: 제출물 SHA 기록) | 기존 문장의 건전성 점검 용도에 주기와 고정 기록 표를 더한다. 회고 보고서 152-155행이 이 값을 회고용 진단으로만 쓴다는 조건을 달아 공개 순위표 원칙 안에 두었다. 확인 제출도 제출 직전 차단 관문의 대상이다. |
| 4 | 특성 생성식, 자료형, 누출 경계, 적용 모형을 자체 후보 장부의 필수 항목으로 | 겹침 없음 | `CONTEXT.md:245-247` (풀 장부 항목은 성능·시드·진입 사유·진입 근거) · `CONTEXT.md:597-599` (특성 계획은 대회 저장소 소유) | 기존 장부 항목은 실행 단위이고 특성 단위 항목이 없다. 외부 구성원 장부 항목(`docs/agents/kaggle-public-notebook-licensing.md:36`)은 출처·라이선스·분할 근거로 성격이 다르다. |
| 5 | 후보는 독립 축 조합으로 생성하고 설정 하나는 한 축만 바꾸며 후보 수는 성과 지표가 아님 | 겹침 없음 | `CONTEXT.md:262-265` (일괄 생산 후보 정의) · `docs/adr/0003-candidate-pool-rebuild-boundaries.md:117` (단독 AUC나 계열로 후보를 미리 줄이지 않음) | 기존 문장은 만들어진 후보를 어떻게 판정하는지만 다루고 후보를 어떻게 만드는지는 다루지 않는다. 후보 수를 성과 지표로 삼지 않는 부분은 회고 보고서 기각 R4와 같은 방향이다. |
| 6 | 자료 생성 제약에서 파생한 열을 원시 열과 같은 지위로 두고 변환 목록 전체를 계열마다 자동 적용 | 겹침 없음 | `CONTEXT.md:587-589` (범용 코어는 자료 형태를 아는 구현을 포함하지 않음) · `CONTEXT.md:597-599` (특성 계획은 대회 저장소 소유) | 자료 형태에 묶인 원칙이라 kagglekit에 대응 문장이 없다. 원칙 문서에는 지표·자료 형태 무관 문장으로 옮기고 S6E8 제약 파생 4열은 괄호 예시로 둔다. |
| 7 | 대리 스크리닝은 탐색 순서 규칙이며 기각 규칙이 아니고, 탈락 표현도 계열당 대표 구성과 결합 구성원으로 재시험 자격 | 충돌 | `CONTEXT.md:207-210` (스크리닝은 1차 판정, 통과는 확정 재검증 자격만) · `docs/adr/0001-experiment-adoption-contract.md:31-32` (스크리닝 개선 0 이상일 때만 확정 재검증) · `docs/adr/0001-experiment-adoption-contract.md:87` (풀 진입은 새 피처 게이트를 묻지 않음) | 해소 방향 확정(회고 보고서 134-142행). kagglekit에는 대리 스크리닝 용어가 없고 스크리닝 정의는 탈락의 지위를 정하지 않아, 그 정의를 대리 스크리닝에 그대로 적용하면 탈락이 곧 기각이 된다. 충돌의 실체는 대회 저장소 `CONTEXT.md:272-274`의 대리 스크리닝 정의와 LightGBM 대리 우선 운영 규칙이다. ADR 0001 87행의 풀 진입 규칙은 결합 구성원 재시험 자격과 이미 양립한다. |
| 8 | 탐색 회차에서 여러 독립 작업자가 겹치지 않는 가설을 빠르게 시험 | 겹침 없음 | `docs/agents/verification-layers.md:3` (1인 빠른 실험 기본값) · `scaffold/AGENTS.md:31` (1인 빠른 실험이 기본값) · `CONTEXT.md:583-585` (판정 스냅숏 격리) | 기존 문장의 1인은 검증 부담의 기준이지 작업자 수의 상한이 아니라 충돌이 아니다. 판정 스냅숏 격리는 판정 중에도 본 줄기 작업을 허용해 병렬 작업자와 양립한다. |
| 9 | 이긴 변형의 변경점과 실패 경계를 다음 회차 시작 전에 공유 | 더 강함 | `CONTEXT.md:504-505` (판정 회차는 봉인·실행·비교·보고·게시 순서) · `docs/adr/0006-judgment-round-contract.md:46-47` (게시 폴더 기준 manifest) · `docs/adr/0007-toolkit-distribution-and-consumption.md:13` (커밋이 곧 대회 간 환류) | 기존은 판정 회차의 게시만 규정한다. 원칙은 탐색 회차의 이긴 변형과 실패 경계까지 다음 회차 시작 전이라는 시점 조건으로 공유 대상을 넓힌다. |
| 10 | 빠른 후보 생성과 엄격한 최종 채택을 서로 다른 단계로 | 같은 뜻 | `docs/agents/verification-layers.md:3-4` (빠른 실험 기본값과 정합성 검증 보존의 계층 구조) · `docs/agents/verification-layers.md:24-25` (차단 순간에만 관문 실행) · `CONTEXT.md:568-576` (차단 관문 묶음과 차단 순간) · `scaffold/AGENTS.md:31-33` | 검증 3계층 규약이 같은 분리를 이미 계층으로 구현했다. |
| 11 | 고정 분할, 여러 시드, 정답 사용 변환의 교차 적합, 완전한 OOF를 갖추고 채택 회차는 누출 검사와 중첩 결합 판정 통과 | 같은 뜻 | `docs/adr/0001-experiment-adoption-contract.md:13-15` (고정 5-fold, 시드 고정) · `docs/adr/0001-experiment-adoption-contract.md:17-21` (플라시보·카나리아 누출 검사) · `docs/adr/0001-experiment-adoption-contract.md:54` (채택 자격에 누출 검사 통과) · `docs/adr/0001-experiment-adoption-contract.md:58-59` (포함·제외 nested OOF 판정) · `CONTEXT.md:235-237` (채택 자격) · `docs/adr/0002-full-data-refit-protocol.md:22` (타깃 인코더 내부 OOF 표현 유지) | 지표 무관 문장으로 옮길 때 5-fold와 시드 42·43·44는 괄호 예시로 둔다(지도 #627 미확정 항목). |
| 12 | 최종 진입은 개수가 아니라 잔차 정보, 중복, 바깥 분할별 개선 부호, 사전 문턱으로 판정 | 같은 뜻 | `docs/adr/0001-experiment-adoption-contract.md:59-62` (포함 AUC - 제외 AUC > 0, 분할 승수 경고, 경계 기여) · `docs/adr/0001-experiment-adoption-contract.md:65-69` (중복 게이트와 원자 교체) · `docs/adr/0001-experiment-adoption-contract.md:99-100` (경계 구간 분할 승수 게이트) · `docs/adr/0001-experiment-adoption-contract.md:109` (OOF 상관을 하드 게이트로 쓰지 않는 이유는 잔차 구조) · `CONTEXT.md:306-310` (후보 풀 기여) | 바깥 분할별 부호는 기존 계약에서 경계 구간 게이트와 선택 불안정성 경고로 자리가 정해져 있으므로, 원칙의 부호 조건은 그 범위로 읽는다. |
| 13 | 약한 후보 묶음 판정에 같은 크기 무작위 대조군과 짝지은 부트스트랩 구간을 보조 진단으로 기록 | 더 강함 | `CONTEXT.md:364-371` (성능 동등 대역은 난수 순위 열 짝지은 영점 대조와 짝지은 행 부트스트랩으로 측정) · `CONTEXT.md:379-382` (경계 채택 후보의 부트스트랩 2.5백분위) · `CONTEXT.md:384-387` (기여 영점 대역) · `docs/adr/0003-candidate-pool-rebuild-boundaries.md:33` (동등 대역은 경고로만 기록) | 기존 영점 대조는 난수 열 하나이고 소급 제거 대조에 쓴다. 원칙은 약한 후보 묶음의 진입 판정에 묶음과 같은 크기의 무작위 대조군을 두고 부트스트랩 구간을 판정 기록의 표준 항목으로 넓힌다. 채택 관문 자체는 바꾸지 않는다. |
| 14 | 결합기는 순위·로짓 이중 표현 규제 로지스틱 회귀를 기본선으로 두고 다른 방식은 같은 중첩 절차에서만 비교 | 같은 뜻 | `docs/adr/0001-experiment-adoption-contract.md:49` (핵심 결합 방식 3개) · `docs/adr/0001-experiment-adoption-contract.md:101-102` (nested OOF 최고 방식 선택) · `docs/adr/0001-experiment-adoption-contract.md:127` (이중 표현 로지스틱 회귀 기본 전략 확정) · `CONTEXT.md:472-475` (기본 결합 전략) · `CONTEXT.md:602-604` (구체 결합 구현은 대회 저장소 소유) | 결합기 계약은 구체 구현을 대회 저장소 소유로 두므로, 원칙 문서에서 기본선은 지표 무관 문장에 S6E8 확정값을 괄호 예시로 붙이는 형태가 맞다(지도 #627 미확정 항목). |
| 15 | 외부 OOF는 출처, 분할 정합성, 배열 해시, 재채점, 라이선스, 상류 학습 계보 확인 뒤 별도 예산으로 관리하고 자체 강모형 예산과 분리 | 더 강함 | `docs/agents/kaggle-public-notebook-licensing.md:31-38` (사용 범위 사전 고정, 장부에 출처·라이선스·분할 근거, 무결성 검증 별도) · `docs/agents/verification-layers.md:35` (외부 자산 반입 차단 순간: 라이선스 확인, 정확 중복 검사) · `CONTEXT.md:116-118` (묶음 반입의 입력 해시와 재채점) | 기존은 사용 범위 고정과 무결성·라이선스 항목까지 있다. 원칙은 상류 학습 계보 확인과 자체 강모형 탐색 예산과의 분리를 더한다. |
| 16 | 계산 장비는 공급자보다 모형의 메모리·가속기 요구량과 실행 기록 계약으로 선택 | 더 강함 | `docs/agents/vast-resource-control.md:8-9` (주·예비 실행 환경은 첫날 계약 재정의 지점) · `docs/agents/vast-resource-control.md:15` (공급자 전환 조건) · `docs/agents/vast-resource-control.md:37-38` (GPU 필요만으로 Kaggle 자동 선택 금지) · `docs/agents/vast-resource-control.md:150` (GPU 메모리를 검색 조건으로 고정) · `docs/agents/remote-execution.md:11-14` (배정 원칙과 유료 발주 직전 차단 순간) · `CONTEXT.md:77-80` (원격 결과 완료는 묶음 반입 통과 상태) · `scaffold/AGENTS.md:42` | 기존은 공급자를 대회별 확정 지점으로 두고 전환 조건과 반입 관문을 갖췄다. 원칙은 그 확정 지점을 채우는 기준(모형 요구량, 실행 기록 계약)을 더한다. `vast-resource-control.md:11-12`의 Vast.ai 우선은 S6E8 확정값이라 충돌이 아니다. |
| 17 | 공개 순위표는 진단에만 쓰고 후보 생성, 결합기 선택, 최종 채택에 미사용 | 같은 뜻 | `docs/adr/0001-experiment-adoption-contract.md:27` · `docs/adr/0002-full-data-refit-protocol.md:29` · `docs/adr/0002-full-data-refit-protocol.md:231` · `docs/agents/vast-resource-control.md:30` | 네 문장이 모두 같은 규칙이다. |
| 18 | 최종 두 장은 중첩 OOF 최고 판과 정직한 OOF가 있는 구성 축이 다른 상향 판이며 OOF 없는 판은 어느 자리에도 넣지 않음 | 충돌 | `docs/adr/0002-full-data-refit-protocol.md:28` (최종 제출 후보 검토에는 CV 전용과 5:1 혼합 예측을 넘김) · `docs/agents/verification-layers.md:34` (최종 제출 두 장 고정 차단 순간) | 해소 방향 확정(회고 보고서 144-150행). kagglekit에는 안전판 용어가 없고 충돌의 실체는 자체 전용 안전판을 둘째 장에 두던 S6E8 관행이다. ADR 0002 28행의 후보 쌍은 같은 풀의 재학습 혼합 변형이라 둘째 장이 구성 축이 달라야 한다는 원칙과 자리가 겹치므로, ADR 0008에서 둘의 관계를 적어야 한다. |
| 19 | 공개되지 않은 설정과 순위 서사는 실험 가설의 출처로만 쓰고 채택 근거로 쓰지 않음 | 겹침 없음 | `docs/adr/0001-experiment-adoption-contract.md:27` (Public 점수는 판정 근거 아님) · `docs/agents/kaggle-public-notebook-licensing.md:26-27` (코드 사용 허가와 채택 가능성 구분) | 기존 문장은 점수와 코드 사용 허가만 다루고 미공개 설정이나 순위 서사는 다루지 않는다. 회고 보고서 기각 R6과 같은 방향이다. |

## 분류 집계

| 분류 | 개수 | 원칙 번호 |
| --- | --- | --- |
| 겹침 없음 | 5 | 4, 5, 6, 8, 19 |
| 같은 뜻 | 6 | 2, 10, 11, 12, 14, 17 |
| 더 강함 | 6 | 1, 3, 9, 13, 15, 16 |
| 충돌 | 2 | 7, 18 |

충돌 두 건은 모두 회고 보고서 `기존 원칙과의 충돌 해소` 절이 해소 방향을 정했다.
원칙 7은 대리 스크리닝을 탐색 순서 규칙으로 두고 탈락 표현의 재시험 자격을 남기는 방향이다.
원칙 18은 안전판을 최종 자리에서 빼고 정직한 OOF가 있는 상향 판을 두는 방향이다.
두 건 모두 kagglekit 문장 자체보다 S6E8 대회 저장소의 정의와 관행이 충돌의 실체이므로, ADR 0008은 원 출처를 대회 저장소로 명기해야 한다.

## kagglekit CONTEXT.md의 새 용어 대응

| 새 용어 | 현재 상태 | 가장 가까운 기존 용어(파일:줄) | 비고 |
| --- | --- | --- | --- |
| 상향 판 | 없음 | 없음 | 안전판 용어도 없다. 둘째 장의 성격을 정의하는 용어가 처음 들어간다. |
| 확인 제출 | 없음 | `docs/adr/0001-experiment-adoption-contract.md:27`의 마일스톤 건전성 점검 | 용어가 아니라 괄호 설명으로만 있다. |
| 탐색 순서 규칙 | 없음 | `CONTEXT.md:207-210` 스크리닝 | 스크리닝은 통과의 지위만 정하고 탈락의 지위와 순서 규칙이라는 성격은 정하지 않는다. |
| 최고 단일 목표·결합 목표 | 없음 | `CONTEXT.md:165-167` champion · `CONTEXT.md:446-449` nested OOF | champion은 계열 무관 단일 최고이고 계열별 목표가 아니다. 결합 목표는 nested OOF 평가 결과로만 존재한다. |
| 무작위 대조군 | 없음 | `CONTEXT.md:384-387` 기여 영점 대역 · `CONTEXT.md:364-371` 성능 동등 대역 | 기존 두 용어는 난수 열 하나 또는 정확 복제 하나를 더하는 영점 대조이고, 같은 크기의 무작위 묶음은 아니다. |
| 대리 스크리닝 | 없음 | `CONTEXT.md:207-210` 스크리닝 · 대회 저장소 `CONTEXT.md:272-274` | 대회 저장소 정의는 통과가 champion 판정을 대신하지 않는다고만 적고 탈락 지위는 없다. kagglekit 정의에는 탈락 표현의 지위를 함께 넣는다. |
| 짝지은 부트스트랩 구간 | 부분 있음 | `CONTEXT.md:366` 짝지은 행 부트스트랩 · `CONTEXT.md:379-382` 경계 채택 후보 | 성능 동등 대역 측정 수단과 경계 채택 표시로만 있어, 판정 기록의 보조 진단 항목이라는 용도는 새로 적어야 한다. |

기존 용어 가운데 새 용어를 이미 덮는 것은 없다.
짝지은 부트스트랩 구간만 측정 수단 수준에서 부분적으로 있다.
나머지 여섯 용어는 kagglekit `CONTEXT.md`에 새로 올려야 한다.

## 근거 범위

- 줄 번호는 kagglekit 커밋 `4c563ab`의 파일에서 `cat -n`으로 읽은 값이다.
- 회고 보고서 줄 번호는 이 저장소 커밋 `2aeb887`의 파일 기준이다.
- `docs/adr/0004`와 `docs/adr/0005`는 어느 원칙과도 겹치는 문장이 없었다.
- `docs/agents/remote-execution.md`는 원칙 16, `docs/adr/0006`과 `docs/adr/0007`은 원칙 9의 근거로만 쓰였다.
