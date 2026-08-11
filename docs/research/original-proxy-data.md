# S6E8 원본 프록시 데이터의 출처와 사용 조건

이 문서는 GitHub 이슈 [P0: 원본 프록시 데이터의 출처와 사용 조건 확인](https://github.com/tmheo/predicting-smartphone-addiction/issues/47)의 조사 결과다.
확인 기준일은 2026-08-11이다.

## 결론

[S6E8 공식 데이터 설명](https://www.kaggle.com/competitions/playground-series-s6e8/data)은 대회 train과 test가 [Smartphone Addiction Prediction Dataset](https://www.kaggle.com/datasets/algozee/smartphone-addiction-prediction-data)에서 영감을 받았다고 밝힌다.
이 링크가 가리키는 소유자 표시는 `algozee`지만, 현재 자료 페이지는 열리지 않고 [Kaggle Dataset API](https://www.kaggle.com/api/v1/datasets/view/algozee/smartphone-addiction-prediction-data)도 `datasets.get` 권한 거부를 반환한다.
따라서 공식 링크가 가리킨 자료의 파일 해시, 판본 이력, 게시 당시 설명과 이용 조건은 지금 직접 검증할 수 없다.

후속 실험에는 [Jaykumar Joshi가 게시한 현존 자료의 판본 1](https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction/versions/1)을 `원본 프록시`로 사용한다.
이 판본은 삭제 전 `algozee` 자료를 직접 읽은 [공개 Kaggle 노트북 판본](https://www.kaggle.com/code/lukhilaksh/smartphone-addiction-prediction-89-beats?scriptVersionId=311858445)의 행 수, 열, 자료형, 처음과 마지막 행, 결측 표시, 중복 수와 수치형 요약 통계 80개가 일치한다.
또한 대회 자료와 12개 설명변수의 이름과 범주 및 값 눈금이 강하게 겹친다.

그러나 이 근거는 `algozee` 파일과 프록시 파일의 바이트 단위 동일성이나 Kaggle이 실제 합성 생성기를 학습할 때 쓴 파일을 증명하지 못한다.
공식 표현도 `generated from`이 아니라 `inspired by`다.
따라서 이 자료는 `실제 생성 원본`이나 `원본 실데이터`가 아니라 `강한 직접 관찰 근거를 가진 원본 프록시`라고만 부른다.

## 공식 출처와 현존 판본

### 대회가 가리키는 자료

- 대회: [Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8)
- 공식 데이터 설명: [S6E8 Data](https://www.kaggle.com/competitions/playground-series-s6e8/data)
- 공식 설명이 연결한 자료: [algozee/smartphone-addiction-prediction-data](https://www.kaggle.com/datasets/algozee/smartphone-addiction-prediction-data)
- 연결된 소유자 표시: [`algozee`](https://www.kaggle.com/algozee)
- 현재 상태: 자료 페이지 접근 불가, 공개 API 조회 권한 거부

공식 데이터 설명은 원자료를 대회의 직접 생성 원천이라고 확정하지 않는다.
삭제된 자료의 소유자 이름, 판본 번호와 라이선스를 현재 공개 화면에서 복구해 다른 출처로 대신 단정해서도 안 된다.

### 사용할 프록시의 고정 식별자

[Kaggle 자료 카드](https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction), [자료 조회 API](https://www.kaggle.com/api/v1/datasets/view/jayjoshi37/smartphone-usage-and-addiction-prediction)와 [파일 목록 API](https://www.kaggle.com/api/v1/datasets/list/jayjoshi37/smartphone-usage-and-addiction-prediction)를 함께 확인했다.

| 항목 | 값 |
| --- | --- |
| Kaggle 참조 | `jayjoshi37/smartphone-usage-and-addiction-prediction` |
| 자료 번호 | `9523417` |
| 게시자 | Jaykumar Joshi (`jayjoshi37`) |
| 판본 | `1` |
| 판본 생성 시각 | `2026-02-19T05:25:36.98Z` |
| 판본 설명 | `Initial release` |
| 예상 갱신 주기 | `never` |
| 파일 | `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv` |
| 파일 크기 | `601569` bytes |
| CSV SHA-256 | `2194ce1946e8559f26780049c8d972857d8378104f2c9ec25ed9ec35409f1074` |
| 라이선스 표시 | `CC0-1.0`, 자료 카드에는 `CC0: Public Domain` |

확인한 현존 사본 중 [jimarahman 판본 1](https://www.kaggle.com/datasets/jimarahman/smartphone-usage-and-addiction-analysis-dataset/versions/1)과 [danishzulfiqar5050 판본 1](https://www.kaggle.com/datasets/danishzulfiqar5050/smartphone-addiction-prediction/versions/1)의 CSV도 위 SHA-256과 완전히 같다.
두 사본의 생성 시각은 각각 2026-04-13과 2026-04-16이고, `jayjoshi37` 판본은 2026-02-19이므로 확인한 세 현존 사본 중 가장 이르다.
같은 바이트에 서로 다른 라이선스가 붙어 있다는 사실은 이 사본들의 권리 계보가 명확하지 않다는 경고이기도 하다.

## 프록시의 스키마와 자료 품질

프록시는 헤더를 제외한 7,500행과 16개 열로 이루어져 있다.
`transaction_id`와 `user_id`는 각각 7,500개 고유값이고, 전체 행 중복은 0개다.
두 식별자를 빼고 비교해도 중복은 0개다.

| 열 | 읽힌 자료형 | 역할과 확인 가능한 의미 |
| --- | --- | --- |
| `transaction_id` | 문자열 | 프록시 전용 거래 식별자 |
| `user_id` | 문자열 | 프록시 전용 이용자 식별자 |
| `age` | 정수 | 나이 |
| `gender` | 문자열 | 성별, `Female`, `Male`, `Other` |
| `daily_screen_time_hours` | 실수 | 하루 화면 사용 시간 |
| `social_media_hours` | 실수 | 소셜 미디어 사용 시간 |
| `gaming_hours` | 실수 | 게임 사용 시간 |
| `work_study_hours` | 실수 | 업무 또는 학습 사용 시간 |
| `sleep_hours` | 실수 | 수면 시간 |
| `notifications_per_day` | 정수 | 하루 알림 수 |
| `app_opens_per_day` | 정수 | 하루 앱 실행 수 |
| `weekend_screen_time` | 실수 | 주말 화면 사용 시간 |
| `stress_level` | 문자열 | 스트레스 수준, `High`, `Low`, `Medium` |
| `academic_work_impact` | 문자열 | 학업 또는 업무 영향 여부, `No`, `Yes` |
| `addiction_level` | 문자열 | 프록시 전용 중독 단계, `Mild`, `Moderate`, `Severe`, `None` |
| `addicted_label` | 정수 | 이진 목표값, `0`은 비중독, `1`은 중독 |

열 의미의 근거는 [프록시 게시자의 자료 설명과 출처 설명](https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction)이다.
게시자는 이 자료를 실제 이용자에게서 수집하지 않고 젊은 성인의 스마트폰 사용 행태를 모사하도록 무작위이되 논리적으로 연관된 패턴으로 만든 합성 자료라고 설명한다.
따라서 사람을 직접 관찰한 실데이터라는 해석이나 의학적 중독 판정이라는 해석을 해서는 안 된다.

CSV에는 빈 필드가 없지만 `addiction_level`에 문자열 `None`이 819개 있다.
pandas의 기본 `read_csv`는 이 문자열을 결측으로 해석하므로 `df.isna().sum()`에서는 이 열에만 819개가 나온다.
대회와 공유하는 12개 설명변수와 `addicted_label`에는 빈 값이나 `None`이 없다.

목표값은 `0`이 2,192행, `1`이 5,308행이며 양성 비율은 `0.7077333333`이다.
게시자가 직접 정의한 목표 의미는 `0 = Not Addicted`, `1 = Addicted`다.

## 삭제된 자료와의 일치 근거

삭제 전 `algozee` 자료를 입력으로 사용한 [Kaggle 노트북 판본](https://www.kaggle.com/code/lukhilaksh/smartphone-addiction-prediction-89-beats?scriptVersionId=311858445)은 다음 경로를 직접 읽는다.

```text
/kaggle/input/datasets/algozee/smartphone-addiction-prediction-data/Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv
```

노트북에 보존된 실행 결과와 `jayjoshi37` 판본 1을 다시 계산해 비교한 결과는 다음과 같다.

- 크기가 모두 `7500 x 16`이다.
- 열 이름, 순서와 pandas 자료형이 모두 같다.
- 처음 5행과 마지막 5행의 모든 값이 같다.
- `addiction_level`의 `None` 819개와 나머지 열 0개라는 결측 해석 결과가 같다.
- 전체 중복 0개라는 결과가 같다.
- 수치형 10개 열의 `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max` 80개가 표시 정밀도 6자리에서 모두 같다.

예를 들어 `daily_screen_time_hours` 평균은 `7.499912`, `social_media_hours` 평균은 `3.273484`, `addicted_label` 평균은 `0.707733`으로 일치한다.
이는 같은 내용의 파일일 가능성이 매우 높다는 직접 관찰 근거다.
그러나 삭제된 파일의 해시나 원소유자의 판본 메타데이터가 없으므로 바이트 단위 동일성은 검증하지 못했다.

## 대회 자료와의 관계

직접 내려받은 공식 `train.csv`는 `691369 x 14`, `test.csv`는 `296302 x 13`이다.
`train.csv`의 SHA-256은 `f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c`이고, `test.csv`의 SHA-256은 `8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e`다.

프록시에서 `transaction_id`, `user_id`, `addiction_level`을 빼면 12개 설명변수와 `addicted_label`이 대회 train과 이름까지 정확히 같다.
대회 자료에는 이 세 열 대신 별도의 `id`가 있다.
프록시 목표 양성률 `0.707733`과 대회 train의 `0.709424` 차이는 약 0.1691%p다.

값 눈금의 겹침은 더 강한 생성 계보 단서다.
다음 표의 `행 값 포함률`은 대회 train의 결측 아닌 값 중 프록시에 정확히 같은 값이 존재하는 비율이다.

| 설명변수 | 프록시 고유값 수 | 대회 train 고유값 수 | 겹친 고유값 수 | 행 값 포함률 |
| --- | ---: | ---: | ---: | ---: |
| `age` | 18 | 18 | 18 | 100.00% |
| `gender` | 3 | 3 | 3 | 100.00% |
| `daily_screen_time_hours` | 900 | 1,389 | 899 | 92.69% |
| `social_media_hours` | 551 | 721 | 551 | 95.59% |
| `gaming_hours` | 401 | 401 | 401 | 100.00% |
| `work_study_hours` | 551 | 600 | 551 | 97.79% |
| `sleep_hours` | 451 | 451 | 451 | 100.00% |
| `notifications_per_day` | 231 | 231 | 231 | 100.00% |
| `app_opens_per_day` | 166 | 166 | 166 | 100.00% |
| `weekend_screen_time` | 1,096 | 1,437 | 1,092 | 96.53% |
| `stress_level` | 3 | 3 | 3 | 100.00% |
| `academic_work_impact` | 2 | 2 | 2 | 100.00% |

이 표는 공식 설명의 `inspired by` 관계와 양립하며 프록시를 생성 규칙 진단에 쓸 강한 근거다.
하지만 주변 분포가 모두 같은 것은 아니다.
예를 들어 프록시와 대회 train의 평균은 `social_media_hours`에서 `3.273484` 대 `2.471038`, `gaming_hours`에서 `2.014183` 대 `1.459265`, `work_study_hours`에서 `3.242420` 대 `2.366971`이다.
따라서 프록시를 대회 분포의 정확한 축소판으로 간주하거나 프록시의 목표 규칙을 대회에 그대로 적용해서는 안 된다.

## 다운로드, 대회 사용과 재배포 조건

### 프록시

`jayjoshi37` 자료 카드와 API 메타데이터는 판본 1에 `CC0-1.0`을 표시한다.
[Creative Commons의 CC0 1.0 설명](https://creativecommons.org/publicdomain/zero/1.0/)에 따르면 허락을 따로 받지 않고 복제, 수정과 재배포를 할 수 있으며 상업적 이용도 가능하다.
다만 CC0는 게시자가 보유한 저작권과 관련 권리를 포기하는 장치일 뿐, 특허와 상표, 타인의 권리까지 정리하거나 자료의 정확성과 권리 상태를 보증하지 않는다.

이번 자료는 같은 바이트의 사본마다 CC0, MIT, CC BY-NC-SA 4.0처럼 다른 라이선스를 표시하고 원소유자의 페이지가 사라져 권리 계보를 완전히 검증할 수 없다.
따라서 이 저장소에는 CSV를 재게시하지 않고 Kaggle 참조, 판본, 파일 이름, SHA-256과 내려받는 절차만 기록한다.
이 방침은 CC0가 허용하는 범위보다 보수적인 재현성 선택이며 법률 판단을 대신하지 않는다.

[S6E8 공식 규칙](https://www.kaggle.com/competitions/playground-series-s6e8/rules)은 공개적이고 모든 참가자가 같은 조건에서 무료로 접근할 수 있는 외부 자료를 제출 개발과 시험에 사용할 수 있다고 규정한다.
확인 기준일 현재 이 프록시는 공개 Kaggle Dataset이고 무료로 내려받을 수 있으므로 이 요건에 맞는다.
이 자료가 비공개되거나 무료 접근이 막히면 대회 외부 자료 요건을 다시 검토해야 한다.

### 대회 train과 test

프록시와 대회 자료의 이용 조건은 서로 다르다.
대회 자료는 공식 데이터 화면과 규칙에서 `CC BY 4.0`으로 표시되지만, 같은 규칙은 대회에 참가하지 않은 사람에게 Competition Data를 전송, 복제, 게시 또는 재배포하지 말라고 별도로 요구한다.
따라서 공식 train과 test도 이 저장소에 커밋하지 않고 각 참가자가 Kaggle에서 직접 내려받는다.

## 재현 방법

공식 [KaggleHub 사용법](https://github.com/Kaggle/kagglehub#download-dataset)은 자료 참조 뒤에 `/versions/1`을 붙여 특정 판본을 내려받는 방법을 제공한다.
다음 명령으로 판본을 고정해 파일을 내려받은 뒤 해시를 확인한다.

```bash
uv run --with kagglehub python - <<'PY'
import kagglehub

path = kagglehub.dataset_download(
    "jayjoshi37/smartphone-usage-and-addiction-prediction/versions/1",
    path="Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv",
)
print(path)
PY

shasum -a 256 <위 명령이 출력한 파일 경로>
```

예상 SHA-256은 `2194ce1946e8559f26780049c8d972857d8378104f2c9ec25ed9ec35409f1074`다.
해시가 다르면 후속 원본 기반 실험을 시작하지 않고 자료의 판본과 내용을 다시 확인한다.

## 후속 실험이 지켜야 할 경계

- 자료 이름은 `원본 프록시`로 고정하고 `실제 생성 원본`, `원본 실데이터` 또는 `실제 이용자 자료`라고 쓰지 않는다.
- 모든 산출물에 Kaggle 참조, 판본 `1`, 파일 이름과 SHA-256을 남긴다.
- `transaction_id`, `user_id`와 대회에 없는 `addiction_level`은 대회 설명변수로 직접 사용하지 않는다.
- 프록시 행을 대회 train에 이어 붙이는 실험은 지도에서 이미 범위 밖이므로 수행하지 않는다.
- 프록시의 규칙과 통계는 대회 생성 구조를 진단하고 원본 기반 특성을 계산하는 입력으로만 검토한다.
- 외부 자료 접근성이나 라이선스 표시가 바뀌면 사용 조건을 다시 확인한다.
