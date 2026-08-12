# S6E8 삭제 원출처와 공개 생성 이력 후속 조사

이 문서는 GitHub 이슈 [삭제된 원출처와 공개 생성 이력 추적](https://github.com/tmheo/predicting-smartphone-addiction/issues/82)의 조사 결과다.
확인 기준일은 2026-08-12이다.
조사 질문은 삭제된 `algozee/smartphone-addiction-prediction-data`의 공개 흔적에서 원본 프록시의 생성 코드·도구·판본 계보 또는 대회 후보 생성 과정을 입증할 수 있는가이다.

## 판정

생성 코드, 사용 도구, 난수 시드, 생성 라이브러리와 권위 있는 판본 계보는 복구하지 못했다.
또한 삭제된 `algozee` 파일의 해시가 남아 있지 않아 현지 원본 프록시와의 바이트 단위 동일성을 확정할 수 없고, Kaggle이 대회 자료를 만들 때 실제로 사용한 파일이나 생성기를 입증할 수도 없다.

다만 삭제된 자료가 실재했고 2026-04-15 Kaggle 실행 환경에서 정상적으로 읽혔다는 강한 1차 증거는 복구했다.
[공개 노트북 판본](https://www.kaggle.com/code/lukhilaksh/smartphone-addiction-prediction-89-beats?scriptVersionId=311858445)의 Kaggle 메타데이터에는 자료 번호 `10083377`, 자료 판본 원천 번호 `15736495`, 묶음 판본 번호 `16678551`이 남아 있고, 실행 기록은 `algozee` 경로의 CSV를 2026-04-15T17:37:50Z에 읽었다.

현지 파일 `data/external/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`은 601,569바이트이고 SHA-256은 `2194ce1946e8559f26780049c8d972857d8378104f2c9ec25ed9ec35409f1074`이며, [Jaykumar Joshi의 Kaggle 판본 1](https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction/versions/1)과 일치한다.
따라서 후속 실험에 쓸 자료의 현지 가용성은 해결됐고, 남은 불확실성은 대회가 실제로 참고한 자료와 생성 코드의 계보다.

가장 보수적인 결론은 `algozee`가 현재까지 확인한 최초 공개자나 생성자라는 증거가 없다는 것이다.
같은 SHA-256의 파일은 `algozee` 자료가 확인된 실행보다 앞선 2026-02-19 Kaggle 판본과 2026-04-03 GitHub 커밋에 이미 공개되어 있었다.
이 연대와 Kaggle 자료 번호는 `algozee`가 후발 재게시자였을 가능성을 높이지만, 삭제본 해시와 생성자 진술이 없으므로 이는 검증된 사실이 아니라 근거가 있는 추론이다.

## 사실, 추론과 미확인 항목

| 구분 | 판정 |
| --- | --- |
| 검증된 사실 | S6E8 공식 데이터 설명은 삭제된 `algozee` 자료를 `inspired by` 대상으로 연결한다. |
| 검증된 사실 | 공개 노트북의 Kaggle 메타데이터와 셀 실행 시각은 2026-04-15에 자료 번호 `10083377`의 CSV를 `algozee` 경로에서 읽은 기록을 보존한다. |
| 검증된 사실 | 현지 원본 프록시는 Jay 판본 1과 파일 크기 및 SHA-256이 같다. |
| 검증된 사실 | 같은 SHA-256의 공개 GitHub 파일이 2026-04-03 커밋에 존재한다. |
| 검증된 사실 | 2026-04-14 Wayback 프로필은 `algozee` 게시자의 표시 이름을 Muhammad Shahzad로 보존한다. |
| 근거가 있는 추론 | 공개 시각과 Kaggle 자료 번호의 순서를 함께 보면 `algozee` 항목은 현존 프록시의 최초 공개처보다는 후발 재게시처일 가능성이 높다. |
| 근거가 있는 추론 | 삭제 전 노트북 결과와 현존 프록시의 광범위한 일치는 두 파일의 내용이 같을 가능성을 매우 높인다. |
| 확인하지 못함 | 삭제된 파일과 현지 프록시의 바이트 단위 동일성, 삭제 자료의 생성 코드·도구·난수 시드·라이선스·원게시 시각은 확인하지 못했다. |
| 확인하지 못함 | Jay, Muhammad Shahzad 또는 다른 공개 계정이 이 합성 자료를 직접 생성했다는 진술이나 코드도 찾지 못했다. |
| 확인하지 못함 | Kaggle이 S6E8 합성 자료를 만들 때 사용한 실제 입력 파일, 도구와 생성기 계보는 확인하지 못했다. |

## 확인된 공개 이력

### 2026-02-19: 현존 Kaggle 판본

[Kaggle 자료 조회 API](https://www.kaggle.com/api/v1/datasets/view/jayjoshi37/smartphone-usage-and-addiction-prediction)는 자료 번호 `9523417`, 판본 `1`, 생성 시각 `2026-02-19T05:25:36.98Z`, 전체 크기 `601569`바이트와 게시자 Jaykumar Joshi를 반환한다.
자료 설명은 7,500개가 넘는 합성 이용자 기록이라고 밝히지만 생성 코드, 도구, 난수 시드와 원출처는 제시하지 않는다.
이 판본의 CSV는 현지 파일과 SHA-256이 같다.

이것은 현재까지 확인한 바이트 일치 공개 사본 중 가장 이른 항목이지만 실제 생성 원본이라는 뜻은 아니다.
판본 설명의 `Initial release`도 해당 Kaggle 항목의 첫 판본이라는 뜻일 뿐, 자료 전체의 최초 생성이나 공개를 입증하지 않는다.

### 2026-04-03: 바이트 일치 GitHub 파일

[SmartphoneAnalisis의 최초 커밋](https://github.com/yaroslav775507/SmartphoneAnalisis/commit/d60a8580b754fe646bf8b90eee568875d791f8c3)은 2026-04-03T10:14:32Z에 `data/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`를 추가했다.
해당 커밋의 원시 파일은 601,569바이트이고 SHA-256은 현지 프록시와 같은 `2194ce1946e8559f26780049c8d972857d8378104f2c9ec25ed9ec35409f1074`다.
저장소의 코드는 이 CSV를 읽어 분석하고 모형을 학습하지만 CSV 자체를 생성하는 코드는 포함하지 않는다.

### 2026-04-14: 게시자 프로필 보존본

[2026-04-14 Wayback 프로필](https://web.archive.org/web/20260414082321id_/https://www.kaggle.com/algozee)은 페이지 제목과 공개 메타데이터에 표시 이름 `Muhammad Shahzad`와 이용자 이름 `algozee`를 보존한다.
프로필 소개는 Python, 자료 분석과 기계 학습을 공부하는 사람이라고 설명하지만 이 스마트폰 자료의 생성 방식이나 원출처는 말하지 않는다.
현재 [`algozee` Kaggle 프로필](https://www.kaggle.com/algozee)과 [삭제 자료 페이지](https://www.kaggle.com/datasets/algozee/smartphone-addiction-prediction-data)는 접근할 수 없다.

### 2026-04-15: 삭제 자료를 사용한 Kaggle 실행

[공개 노트북 판본](https://www.kaggle.com/code/lukhilaksh/smartphone-addiction-prediction-89-beats?scriptVersionId=311858445)을 [Kaggle 공식 명령줄 도구의 노트북 받기 기능](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#pull)으로 내려받아 메타데이터와 셀 실행 기록을 확인했다.
노트북의 `metadata.kaggle.dataSources`에는 다음 식별자가 남아 있다.

```json
{
  "sourceType": "datasetVersion",
  "sourceId": 15736495,
  "datasetId": 10083377,
  "databundleVersionId": 16678551
}
```

CSV 읽기 셀은 다음 경로를 사용하며 `2026-04-15T17:37:50.001053Z`에 실행을 시작해 `2026-04-15T17:37:50.037492Z`에 끝났다.

```text
/kaggle/input/datasets/algozee/smartphone-addiction-prediction-data/Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv
```

이 기록은 해당 시점에 Kaggle이 자료 판본을 실행 환경에 연결하고 파일을 읽을 수 있었음을 입증한다.
그러나 현재 내려받은 노트북의 출력 셀은 비어 있고 자료 원천 표시는 삭제된 참조를 빈 문자열로 반환하므로, 삭제 파일의 바이트와 게시 설명은 복구할 수 없다.
파일 이름의 `(1)`도 생성 계보를 뜻한다는 근거가 없으므로 해석에 사용하지 않았다.

## 삭제 파일과 현지 프록시의 관계

기존 조사 문서인 [S6E8 원본 프록시 데이터의 출처와 사용 조건](original-proxy-data.md)은 삭제 전 `algozee` 파일을 읽은 노트북 결과와 Jay 판본 1을 비교했다.
두 자료는 7,500행과 16개 열, 열 이름과 순서, 자료형, 처음과 마지막 5행, 결측 해석, 중복 수와 수치형 요약 통계 80개가 모두 일치했다.
이 결과는 같은 내용의 CSV일 가능성을 매우 높이지만 암호학적 파일 동일성은 입증하지 않는다.

[S6E8 공식 데이터 설명](https://www.kaggle.com/competitions/playground-series-s6e8/data)은 대회 train과 test가 삭제 자료에서 `inspired by`라고만 밝힌다.
따라서 삭제 자료가 대회 생성기의 직접 입력이었다거나 대회 자료가 이 CSV를 특정 도구로 변환한 결과라고 확장해서 해석할 수 없다.

Kaggle 자료 번호도 보조 단서일 뿐이다.
삭제 자료의 번호 `10083377`은 Jay 자료 번호 `9523417`보다 크고, 2026-04-13에 생성된 [jimarahman 사본](https://www.kaggle.com/api/v1/datasets/view/jimarahman/smartphone-usage-and-addiction-analysis-dataset)의 번호 `10065805`와 2026-04-16에 생성된 [danishzulfiqar5050 사본](https://www.kaggle.com/api/v1/datasets/view/danishzulfiqar5050/smartphone-addiction-prediction)의 번호 `10100822` 사이에 놓인다.
자료 번호가 생성 순서대로 부여됐다고 가정하면 `algozee` 항목은 4월 중순에 만들어진 후발 사본이라는 해석이 자연스럽다.
하지만 Kaggle은 자료 번호의 시간 순서를 공개 규약으로 보장하지 않으므로 이 해석을 확정 사실로 사용하지 않는다.

## 생성 코드와 판본 계보 탐색 결과

### Kaggle 실행 결과와 캐시

공개 노트북에서 자료 판본 식별자, 파일 경로와 실행 시각은 복구했지만 CSV 본문, 해시, 자료 설명, 생성 코드와 판본 기록은 복구하지 못했다.
삭제 자료의 [현재 API 주소](https://www.kaggle.com/api/v1/datasets/view/algozee/smartphone-addiction-prediction-data)는 권한 거부를 반환했고, 공식 명령줄 도구의 메타데이터·파일 목록·상태·받기 요청도 같은 결과를 냈다.
Kaggle 공개 자료 검색에서도 `algozee` 항목은 반환되지 않았다.

### 웹 보존본과 공개 복제본

[Wayback의 정확한 자료 주소 조회](https://web.archive.org/cdx/search/cdx?url=www.kaggle.com%2Fdatasets%2Falgozee%2Fsmartphone-addiction-prediction-data%2A&output=json&fl=timestamp%2Coriginal%2Cstatuscode%2Cmimetype%2Cdigest%2Clength&filter=statuscode%3A200&collapse=digest)는 보존본을 반환하지 않았다.
프로필은 보존됐지만 목표 자료 페이지, Kaggle API 응답과 CSV 파일은 발견되지 않았다.

[Hugging Face의 `jason1966` 공개 자료 목록](https://huggingface.co/api/datasets?author=jason1966&limit=10000&full=true)에는 2026-03-31에 생성된 `algozee_` 접두사 복제본 29개가 있다.
이 일괄 복제 목록에는 스마트폰 또는 중독 자료가 없었다.
이 음성 결과는 목표 자료가 3월 31일 뒤에 등장했을 가능성과 맞지만, 해당 계정이 `algozee`의 모든 자료를 복제했다는 보장이 없으므로 생성 시각을 입증하지 않는다.

2026년 3월, 4월, 5월과 7월 Common Crawl 색인에서도 정확한 삭제 자료 주소를 찾지 못했다.
검색 색인의 부재는 크롤링 누락일 수 있으므로 삭제 자료의 부재나 생성 시점을 입증하지 않는다.

### GitHub와 게시자 흔적

[정확한 파일 이름을 찾는 GitHub 코드 검색](https://github.com/search?q=%22Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv%22&type=code)과 [핵심 열 조합 검색](https://github.com/search?q=%22daily_screen_time_hours%22+%22addicted_label%22&type=code)으로 공개 저장소를 확인했다.
발견된 저장소는 CSV를 읽는 분석, 시각화, 모형 학습과 대회 후속 실험이었고, 7,500행 CSV를 생성하는 코드나 원생성자 진술은 찾지 못했다.
정확한 SHA-256, 정확한 파일 이름과 `generator`, `seed`, `synthetic` 조합도 별도의 생성 저장소를 찾지 못했다.

[GitHub 이용자 API의 `algozee` 조회](https://api.github.com/users/algozee)는 이용자를 반환하지 않았다.
이는 같은 이름의 GitHub 계정이 현재 없다는 뜻일 뿐, Muhammad Shahzad가 다른 이름으로 계정을 갖지 않았다는 뜻은 아니다.
게시자 표시 이름과 자료 제목을 함께 검색해 찾은 논문과 수업 문서는 모두 자료를 분석하거나 인용했을 뿐 생성 코드, 도구와 판본 계보를 제공하지 않았다.

## 음성 결과의 의미

| 조사면 | 확인 범위 | 찾지 못한 것 | 해석 경계 |
| --- | --- | --- | --- |
| Wayback | 정확한 자료 주소, 게시자 프로필, `algozee` 자료 주소 묶음 | 목표 자료 페이지, API 응답, CSV | 보존본 부재는 원래 페이지 부재의 증거가 아니다. |
| Common Crawl | 2026년 3월, 4월, 5월, 7월 색인 | 정확한 목표 주소 | 크롤링 누락 가능성이 있다. |
| Kaggle API와 명령줄 도구 | 자료 조회, 파일 목록, 상태, 받기, 공개 검색 | 삭제 자료 메타데이터와 파일 | 공개 접근 실패는 영구 삭제나 비공개 전환을 구분하지 못한다. |
| Kaggle 노트북 | 원천 식별자, 파일 경로, 셀 실행 시각, 분석 코드 | CSV 본문, 해시, 생성 코드 | 파일 사용은 확인되지만 생성 방법은 확인되지 않는다. |
| GitHub | 정확한 파일 이름, 열 조합, 해시와 생성 관련 검색 | 생성 프로그램, 난수 시드, 원생성자 진술 | 공개되지 않았거나 다른 이름으로 존재할 수 있다. |
| 자료 복제본 | Kaggle 현존 사본, GitHub 사본, Hugging Face 일괄 복제 | 권위 있는 최초 판본 계보 | 가장 이른 발견 사본은 실제 최초 생성본과 같지 않다. |

## 재현 절차

Kaggle 노트북의 현재 공개 메타데이터는 다음 절차로 다시 확인할 수 있다.

```bash
work_dir=$(mktemp -d)
uvx --from kaggle kaggle kernels pull \
  lukhilaksh/smartphone-addiction-prediction-89-beats \
  -m -p "$work_dir"

jq '.metadata.kaggle.dataSources' \
  "$work_dir/smartphone-addiction-prediction-89-beats.ipynb"

jq -r '.cells[] | select(.cell_type == "code") | .source' \
  "$work_dir/smartphone-addiction-prediction-89-beats.ipynb" \
  | rg 'algozee|read_csv'
```

현지 프록시의 고정 식별자는 다음과 같이 확인한다.

```bash
wc -c data/external/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
shasum -a 256 data/external/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
```

예상 크기는 `601569`바이트이고 예상 SHA-256은 `2194ce1946e8559f26780049c8d972857d8378104f2c9ec25ed9ec35409f1074`다.

## 모형 연구에 적용할 경계

이번 추적은 새로운 생성 공식이나 추가 설명변수를 제공하지 않았다.
따라서 현지 파일은 계속 `원본 프록시`라고만 부르고 `실제 생성 원본`, `Kaggle 학습 원본` 또는 `원생성자의 파일`이라고 부르지 않는다.
프록시의 분포, 값 눈금과 목표 규칙은 대회 생성 구조를 진단하는 관찰 근거로 사용할 수 있지만, 프록시의 생성 코드가 복구됐다고 가정한 모형이나 자료 증강은 정당화되지 않는다.

추가 계보 주장을 하려면 다음 중 하나 이상의 새 1차 증거가 필요하다.

- 삭제된 `algozee` CSV의 해시 또는 파일 본문
- 삭제 자료의 판본 생성 시각, 설명과 라이선스를 포함한 Kaggle 원응답
- 게시자가 직접 공개한 생성 코드, 도구·난수 시드 설명 또는 원출처 진술
- Kaggle이 S6E8 자료 생성에 실제로 사용한 입력과 생성 절차에 관한 공식 설명

이 증거가 나오기 전까지 가장 이른 공개 사본을 실제 원천으로 승격하거나, 프록시의 생성 규칙을 대회 생성 규칙과 동일시하지 않는다.
