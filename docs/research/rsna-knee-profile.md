# RSNA Knee Abnormality Detection 대회 프로파일

작성일: 2026-09-01.
목적: 다음 대회 toolkit 설계의 입력으로 쓸 대회 사실 정리.
출처: Kaggle 대회 페이지(Overview, Data 탭), Kaggle CLI 파일 목록, RSNA 공식 발표, 공개 베이스라인 저장소.
확인된 사실과 추정을 구분해 표기한다.

## 요약

- 대회: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) (Research Code Competition, 상금 총 $77,000).
- 과제: 무릎 MRI 검사(study) 단위로 12개 이상 소견 각각의 확률을 예측하는 다중 라벨 분류.
- 지표: 12개 타깃의 macro 평균 ROC-AUC.
- 데이터: DICOM 무릎 MRI 학습 4,407 study(81만 9천여 슬라이스) + 다국어 방사선 판독문, 정식 라벨은 58 study뿐이라 판독문에서 약한 라벨을 뽑아야 한다.
- 마감: 최종 제출 2026-10-22, 참가·팀 병합 마감 2026-10-15 (UTC 23:59).
- 제출: 노트북 제출 코드 대회, CPU/GPU 각 9시간, 인터넷 차단, 공개 외부 데이터·사전학습 모델 허용.
- GPU 요구: 이미지 백본 학습이 중심이라 외부 GPU(Vast.ai) 의존이 이번 S6E8(CPU 표 데이터)보다 훨씬 커질 전망.

## 과제 정의

한 무릎 MRI 검사에서 12개 임상 소견 각각의 존재 확률을 예측한다.
평가는 12개 타깃 각각의 ROC-AUC를 평균한 macro AUC로 한다(확인됨, Overview Evaluation 절).

12개 타깃(확인됨, Data 탭):

1. `ACL` - 전방십자인대 손상.
2. `MCL` - 내측측부인대 손상.
3. `Medial Meniscus` - 내측 반월판 파열.
4. `Lateral Meniscus` - 외측 반월판 파열.
5. `Medial OA` - 내측 대퇴경골 구획 골관절염.
6. `Lateral OA` - 외측 대퇴경골 구획 골관절염.
7. `PF OA` - 슬개대퇴 골관절염.
8. `Effusion` - 관절 삼출.
9. `Synovitis` - 활막염.
10. `Baker's` - 베이커 낭종.
11. `Contusion` - 골 타박.
12. `Fracture` - 골절.

제출 파일은 `StudyInstanceUID` + 12개 확률 열의 `submission.csv` 하나다(확인됨).

## 데이터 형태와 규모

모두 Data 탭과 Kaggle CLI 파일 목록, 공개 베이스라인 저장소에서 확인한 사실이다.

- 형식: DICOM. `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm` 구조, 슬라이스 하나가 파일 하나.
- 학습 규모: 4,407 study, 24,371 series, 819,078 DICOM 파일(공개 저장소 집계, Data 탭 기술과 부합).
- series당 슬라이스: 보통 20~45장(중앙값 30), 긴 꼬리는 수백 장까지.
- 시퀀스 메타: series 단위로 `Fluid_Sensitive`(T2/PD/STIR 계열 여부), `Fat_Suppression`, `Anatomical_Plane`(Sagittal/Coronal/Axial) 세 열이 제공된다.
  시퀀스 원명(T1, T2, PD 등)은 직접 주지 않고 이 파생 열로 준다.
- 모든 학습 study가 세 평면(Sagittal, Coronal, Axial) series를 가진다(공개 저장소 집계).
- DICOM 특성: 강도·방향·해상도가 study마다 다르고, transfer syntax가 혼재한다(무압축 Explicit VR LE, JPEG Lossless, JPEG 2000, Implicit VR LE).
  메타데이터는 86개 허용 태그만 남기고 제거됐다.
- 라벨: 12개 이진 라벨이 정식으로 붙은 학습 study는 58개뿐이다.
  나머지 4,349개는 원문 방사선 판독문(`Report` 열)만 있고, 판독문은 12개 언어(터키어, 스페인어, 태국어 등 19개 기관 출신)로 섞여 있다.
  즉 판독문에서 라벨을 추출하는 약한 감독(weak supervision)이 사실상 필수 설계 요소다.
- 테스트: 약 1,300 study, 판독문은 제공되지 않는다(이미지만으로 추론).
  숨겨진 테스트로 채점되는 코드 대회라 로컬에는 예시 3 study만 보인다.
- 분포 주의: 학습·공개 리더보드·최종 평가 데이터 사이에 소견 유병률이 같다는 보장이 없다(주최 명시).

용량(추정): Kaggle CLI로 표본 추출한 학습 DICOM 파일 크기가 평균 약 0.55MB였고, 819,078 파일로 환산하면 전체 다운로드 용량은 대략 400~500GB 수준이다.
표본이 작아(한 study 인근 38개 파일) 오차가 클 수 있으니 실제 다운로드 전에 재확인이 필요하다.
어느 쪽이든 수백 GB급이라 로컬 맥 한 대에 통째로 두고 돌리기는 부담스러운 규모다.

## 일정

확인됨(Overview Timeline 절, 모두 해당일 23:59 UTC):

- 2026-07-30: 시작.
- 2026-10-15: 참가(규칙 동의) 마감.
- 2026-10-15: 팀 병합 마감.
- 2026-10-22: 최종 제출 마감.
- 2026-11-05: 수상자 의무 이행 마감(학습 코드, 영상, 방법 설명 제출).
- 2026-11 말: RSNA 연례회의(11/29~12/3, 시카고)에서 수상팀 소개.

오늘(2026-09-01) 기준 최종 제출까지 약 7주 남았다.

## 규칙과 제출 방식

확인됨(Overview Code Requirements 절):

- 노트북(코드) 제출 대회다.
- CPU 노트북 9시간, GPU 노트북 9시간 실행 제한.
- 인터넷 접근 차단.
- 자유롭게 공개된 외부 데이터와 사전학습 모델은 허용된다.
- 제출 파일 이름은 `submission.csv` 고정.

수상자 의무(확인됨, Prizes 절): 표준 Kaggle 수상 의무(오픈소스 라이선스, 솔루션 패키징·전달, 주최 측 발표)에 더해 (i) 접근법 소개 영상 제작, (ii) 공개 코드·가중치 링크를 대회 포럼에 게시, (iii) 최종 모델을 공개 배포·검증 가능하게 공유해야 한다.

추정(표준 Kaggle 관행, Rules 탭 원문은 로그인 뒤에서만 렌더링돼 이번 조사에서 원문 확인 못 함):

- 일일 제출 한도는 통상 5회, 최종 선택 제출은 2개일 가능성이 높다.
- 대회 데이터는 대회 목적과 비상업 연구 목적으로 사용이 제한되는 통상 RSNA 조항일 가능성이 높다.
- Rules 원문은 계정 로그인 상태의 브라우저로 접속해 확인해 두는 편이 안전하다.

## 상금 구조

확인됨(Overview Prizes, Efficiency Prize 절):

- 메인 리더보드: 1위 $9,000부터 10위 $5,000까지 10팀, 합계 $59,000.
- 효율 트랙: 1~3위 $7,000/$6,000/$5,000, 합계 $18,000.
- 효율 점수는 AUC와 실행 시간(RuntimeSeconds/32400, 즉 9시간 대비 비율)을 결합한 지표를 최소화하는 방식이다.
  효율 트랙 수상 대상은 메인 리더보드 선택 제출 중에서 벤치마크(`sample_submission.csv`)보다 높은 것들이다.
  효율 리더보드는 주최가 노트북으로 매일 갱신해 공개한다.

## GPU 요구 수준 추정

이 절은 추정이다.

- 학습: 81만 9천 슬라이스, study 4,407개의 다중 평면 MRI에 2.5D/3D CNN 또는 트랜스포머 백본을 학습해야 한다.
  규모로는 RSNA 2024 Lumbar(약 2,000 study)의 두 배가 넘고, S6E8처럼 CPU 트리 모델로 승부하는 구도가 아니다.
  단일 실험도 A100/4090급 GPU 수 시간~수십 시간이 걸릴 것으로 본다.
  fold·시드·백본 앙상블까지 가면 외부 GPU 시간 소요가 수백 GPU 시간 단위로 커질 수 있다.
- 판독문 트랙: 12개 언어 판독문에서 12개 라벨을 뽑는 작업은 다국어 LLM 또는 규칙 기반 추출로 가능하고, 상대적으로 가볍다(로컬 CPU 또는 소형 GPU).
  다만 테스트 시점엔 판독문이 없으므로 판독문은 학습 라벨 생성용이다.
- 추론: 9시간 안에 약 1,300 study를 처리해야 하므로 study당 약 25초 예산이다.
  앙상블 폭이 추론 예산으로 제약되고, 효율 트랙까지 노리면 더 조인다.
- 결론: Vast.ai 등 외부 GPU 의존이 이번 대회보다 훨씬 커진다.
  수백 GB 데이터를 원격 GPU 장비로 옮기는 배치·전송·체크포인트 도구가 toolkit의 핵심 요구가 된다.
  Kaggle 노트북 GPU(T4x2/P100, 주당 30시간)는 추론 검증용으로는 쓰이지만 본 학습을 감당하기 어렵다.

## 과거 유사 RSNA 대회 상위 해법 패턴

과거 대회 공개 솔루션 정리에서 확인한 패턴이다.
이번 대회 적용 방향은 추정으로 표시한다.

### RSNA 2022 Cervical Spine Fracture Detection (CT)

- 1위: 척추뼈 분할·위치 추정(1단계) 후 척추뼈 단위 2.5D CNN + LSTM(2단계).
  슬라이스 15장을 z축으로 샘플링해 이웃 슬라이스·분할 마스크와 채널로 묶고, EfficientNet-V2-S/ConvNeXt 백본 뒤 LSTM으로 시퀀스를 융합했다.
- 패턴: 위치 추정으로 관심 영역을 좁힌 뒤 2.5D CNN + 순환 계층, 백본·fold 앙상블.

### RSNA 2023 Abdominal Trauma Detection (CT)

- 1위: 3D 분할 모델로 장기 마스크를 만들어 장기 단위로 크롭하고, 96 슬라이스 볼륨을 인접 3장 묶음 32개로 재구성해 2D 백본(CoaT Lite, EfficientNet-V2-S) + GRU로 분류.
  보조 분할 헤드로 학습을 안정화하고 슬라이스 로짓 max pooling으로 집계했다.
- 패턴: 분할 기반 크롭 → 2.5D CNN + 시퀀스 헤드 → 보조 손실 → 앙상블.

### RSNA 2024 Lumbar Spine Degenerative Classification (MRI)

- 상위권: CenterNet 계열 키포인트 검출기(EfficientNet 백본 + FPN)로 디스크 레벨 좌표를 찾아 크롭하고(1단계), 레벨 단위 분류기(2단계)를 학습.
  좌표 라벨이 없는 데이터에는 학습된 모델로 의사 라벨을 만들어 전체 데이터를 활용했다.
- 패턴: 키포인트 검출 → 레벨별 크롭 → 다중 뷰(시상/축상) 분류기 → 의사 라벨로 데이터 확장.

### 공통 관행과 이번 대회 함의(추정)

- 공통 골격: (1) 위치 추정 또는 분할로 ROI 크롭, (2) 2.5D CNN(+RNN/트랜스포머 헤드) 분류, (3) 다중 백본·fold·시드 앙상블, (4) 의사 라벨로 라벨 부족 보완.
- 무릎 MRI 선행 연구로는 Stanford MRNet(평면별 2D CNN + 로지스틱 결합)이 있고, 이번 대회 구조(세 평면 series)와 잘 맞는다.
- 이번 대회 특수성: ROI가 장기·척추뼈처럼 명확히 분리되지 않아 1단계는 크롭보다 series 선택(fluid-sensitive fat-suppressed 우선)과 평면별 인코더 구성이 될 가능성이 높다.
  실제로 초기 공개 베이스라인도 평면별 2.5D EfficientNet 인코더 + 특징 연결 구조를 쓴다.
- 가장 큰 차별 요소는 판독문 약한 라벨의 품질일 것이다.
  다국어 판독문 → 12 라벨 추출 파이프라인(LLM 활용 포함)과 라벨 노이즈 대응(soft label, 확신도 가중)이 상위권을 가를 여지가 크다.
- 사전학습 활용: 공개 사전학습 모델이 허용되므로 ImageNet 계열은 물론 의료 영상 사전학습(예: DINOv2 계열, MRNet 가중치)도 검토 대상이다.
  라이선스 확인 절차는 기존 `docs/agents/kaggle-public-notebook-licensing.md` 관행을 재사용한다.

## toolkit 설계에 주는 함의

- 데이터 배치: 수백 GB DICOM을 원격 GPU 호스트로 옮기고 검증하는 도구가 최우선이다.
  Kaggle에서 원격 장비로 직접 내려받는 경로(장비에서 `kaggle competitions download`)가 로컬 경유 업로드보다 훨씬 낫고, 기존 SSH 표준 스트림 전송 절차는 산출물 회수용으로 쓰면 된다.
- 전처리 자산화: DICOM → 정규화 볼륨(또는 슬라이스 png/npz) 변환을 한 번 해서 재사용 가능한 중간 자산으로 관리하는 계층이 필요하다.
  transfer syntax 혼재 때문에 pydicom + pylibjpeg/gdcm 의존성을 실행 환경 이미지에 고정해야 한다.
- 실행 기록: 기존 pipeline.run·실행 기록 번들·판정 회차(JudgmentRound) 계약은 GPU 학습 실행에도 그대로 적용할 가치가 있으나, 실행 시간이 길어 체크포인트·재개(resume) 계약을 1급 요소로 넣어야 한다.
- 제출 파이프라인: 인터넷 차단 노트북 제출이므로 모델 가중치를 Kaggle Dataset/Model로 올리고 추론 노트북이 읽는 구조가 필요하다.
  9시간·study당 25초 예산을 지키는 추론 시간 계측 도구도 함께 둔다.
- CV 설계: 정식 라벨 58개는 검증 세트로도 부족하므로, 판독문 추출 라벨의 신뢰도 층화와 기관(사이트) 단위 그룹 분할을 지원하는 분할 도구가 필요하다.
- 효율 트랙: 메인 제출이 그대로 효율 트랙 후보가 되므로, 앙상블 폭 대비 실행 시간 트레이드오프를 기록·비교하는 눈금이 있으면 좋다.

## 참고 링크

- 대회: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- RSNA 공식 페이지: https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge
- RSNA 발표: https://www.rsna.org/news/2026/august/ai-challenge-knee-mri
- 공개 베이스라인 구조 참고: https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection
- RSNA 2023 1위 솔루션: https://github.com/Nischaydnk/RSNA-2023-1st-place-solution
- RSNA 2022 수상 알고리즘 논문: https://pubs.rsna.org/doi/10.1148/ryai.230256
