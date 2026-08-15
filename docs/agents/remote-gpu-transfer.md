# 원격 GPU 파일 전송

Runpod과 Vast.ai의 원격 계산 자원에 파일을 올리고 결과를 회수할 때 사용하는 공통 절차다.

## 확인된 제약

로컬 Mac의 회사 보안 솔루션은 `scp`가 로컬 파일을 여는 동작을 차단한다.
Runpod 실제 실행에서 `scp` 업로드는 원격 연결 전에 `Operation not permitted`로 실패했다.
보안 설정을 바꾸지 않은 상태에서도 일반 SSH 명령과 SSH 표준 입력 및 표준 출력은 정상 작동했다.

Runpod 후속 실행에서는 다음 형태로 파일 내용을 SSH 표준 입력에 보내 원격 `cat`으로 저장했고, 원격 해시가 원본과 일치했다.

```bash
ssh \
  -i "$GPU_SSH_KEY" \
  -p "$GPU_SSH_PORT" \
  "$GPU_SSH_TARGET" \
  'umask 077; cat > /workspace/job/input.tar.gz.part' \
  < input.tar.gz
```

Vast.ai 후속 실행에서도 실행 스크립트와 약 11MB의 입력 묶음을 같은 방식으로 전송하고 원격 SHA-256과 파일 크기를 검증했다.
이 제약은 공급자가 아니라 로컬 보안 솔루션에서 생기므로 Runpod과 Vast.ai에 동일하게 적용한다.

## 필수 규칙

1. `scp`, `sftp`, 브라우저 파일 업로드를 평상시 전송 경로로 사용하지 않는다.
2. 여러 파일은 로컬에서 하나의 입력 전송 묶음으로 만든다.
3. macOS 보조 메타데이터가 들어가지 않도록 묶음을 만들 때 `COPYFILE_DISABLE=1`을 사용한다.
4. 전송 전에 로컬 SHA-256을 계산한다.
5. 원격에서는 최종 경로가 아닌 `.part` 경로에 먼저 저장한다.
6. 원격 SHA-256이 로컬 값과 일치할 때만 원자적으로 최종 이름으로 바꾼다.
7. 결과 회수도 로컬 `.part` 경로에 먼저 받은 뒤 원격 해시와 일치할 때만 최종 이름으로 바꾼다.
8. 전송 실패나 연결 중단 뒤에는 부분 파일을 결과나 입력으로 사용하지 않는다.
9. SSH 호스트 키를 미리 확인하고 `StrictHostKeyChecking=yes`를 사용한다.
10. SSH 키는 원격 실행 작업별로 분리하며, 다른 작업이 같은 키를 참조하는지 확인하지 않고 삭제하지 않는다.

## 접속 변수

아래 값은 공급자 화면이나 공식 명령줄 도구로 확인한 실제 값으로 설정한다.
비밀 값과 SSH 개인 키를 저장소에 기록하지 않는다.

```bash
GPU_SSH_KEY=/absolute/path/to/task-specific-key
GPU_SSH_PORT=12345
GPU_SSH_TARGET=root@example-host
GPU_REMOTE_ROOT=/workspace/example-job
GPU_LOCAL_ARCHIVE=/absolute/path/to/input.tar.gz
```

원격 경로는 공백과 셸 제어 문자가 없는 작업 전용 절대 경로로 고정한다.

## 입력 묶음 만들기

입력 파일별 SHA-256 목록을 묶음 안에 포함한다.
아래 예시의 파일 목록은 원격 실행 명세에 고정된 실제 목록으로 바꾼다.

```bash
GPU_INPUT_ROOT=/absolute/path/to/prepared-input

(
  cd "$GPU_INPUT_ROOT"
  shasum -a 256 data/train.parquet data/test.parquet artifacts/folds.parquet \
    > input-manifest.sha256
)

COPYFILE_DISABLE=1 tar -czf "$GPU_LOCAL_ARCHIVE" \
  -C "$GPU_INPUT_ROOT" \
  data/train.parquet \
  data/test.parquet \
  artifacts/folds.parquet \
  input-manifest.sha256

GPU_LOCAL_SHA256=$(shasum -a 256 "$GPU_LOCAL_ARCHIVE" | awk '{print $1}')
```

## 입력 묶음 올리기

`scp` 대신 로컬 파일을 SSH 표준 입력에 연결한다.

```bash
GPU_REMOTE_PART="$GPU_REMOTE_ROOT/input.tar.gz.part"
GPU_REMOTE_ARCHIVE="$GPU_REMOTE_ROOT/input.tar.gz"

ssh \
  -i "$GPU_SSH_KEY" \
  -p "$GPU_SSH_PORT" \
  -o BatchMode=yes \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=12 \
  "$GPU_SSH_TARGET" \
  "umask 077; mkdir -p '$GPU_REMOTE_ROOT'; cat > '$GPU_REMOTE_PART'" \
  < "$GPU_LOCAL_ARCHIVE"

GPU_REMOTE_SHA256=$(
  ssh \
    -i "$GPU_SSH_KEY" \
    -p "$GPU_SSH_PORT" \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=yes \
    "$GPU_SSH_TARGET" \
    "sha256sum '$GPU_REMOTE_PART'" \
    | awk '{print $1}'
)

test "$GPU_LOCAL_SHA256" = "$GPU_REMOTE_SHA256"

ssh \
  -i "$GPU_SSH_KEY" \
  -p "$GPU_SSH_PORT" \
  -o BatchMode=yes \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=yes \
  "$GPU_SSH_TARGET" \
  "mv '$GPU_REMOTE_PART' '$GPU_REMOTE_ARCHIVE'"
```

해시가 다르면 `.part` 파일을 삭제하고 처음부터 다시 전송한다.
완성된 원격 파일을 이어 쓰거나 부분 전송을 재사용하지 않는다.

## 원격에서 풀기

macOS 소유권과 확장 속성 경고가 원격 환경을 바꾸지 않도록 다음 제한을 적용한다.

```bash
ssh \
  -i "$GPU_SSH_KEY" \
  -p "$GPU_SSH_PORT" \
  -o BatchMode=yes \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=yes \
  "$GPU_SSH_TARGET" \
  "mkdir -p '$GPU_REMOTE_ROOT/input' && \
   tar --no-same-owner \
       --no-same-permissions \
       --warning=no-unknown-keyword \
       --exclude='._*' \
       --exclude='*/._*' \
       -xzf '$GPU_REMOTE_ARCHIVE' \
       -C '$GPU_REMOTE_ROOT/input' && \
   cd '$GPU_REMOTE_ROOT/input' && \
   sha256sum --check input-manifest.sha256"
```

묶음 해시와 내부 파일 해시를 모두 통과해야 원격 실행을 시작할 수 있다.

## 결과 회수

원격에서 결과 묶음과 SHA-256을 먼저 만든다.
로컬에서는 표준 출력으로 결과를 받아 `.part` 파일에 저장한다.

```bash
GPU_REMOTE_RESULT="$GPU_REMOTE_ROOT/result.tar.gz"
GPU_LOCAL_RESULT=/absolute/path/to/result.tar.gz

GPU_EXPECTED_RESULT_SHA256=$(
  ssh \
    -i "$GPU_SSH_KEY" \
    -p "$GPU_SSH_PORT" \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=yes \
    "$GPU_SSH_TARGET" \
    "sha256sum '$GPU_REMOTE_RESULT'" \
    | awk '{print $1}'
)

ssh \
  -i "$GPU_SSH_KEY" \
  -p "$GPU_SSH_PORT" \
  -o BatchMode=yes \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=12 \
  "$GPU_SSH_TARGET" \
  "cat '$GPU_REMOTE_RESULT'" \
  > "$GPU_LOCAL_RESULT.part"

GPU_ACTUAL_RESULT_SHA256=$(
  shasum -a 256 "$GPU_LOCAL_RESULT.part" | awk '{print $1}'
)

test "$GPU_EXPECTED_RESULT_SHA256" = "$GPU_ACTUAL_RESULT_SHA256"
mv "$GPU_LOCAL_RESULT.part" "$GPU_LOCAL_RESULT"
```

해시 검증 뒤에도 실행 기록 묶음의 자체 반입 검증을 별도로 통과해야 원격 결과 완료로 판정한다.

## 실패 시 처리

`scp`에서 `Operation not permitted`가 발생하면 다시 시도하거나 보안 설정을 바꾸지 않는다.
즉시 SSH 표준 스트림 전송으로 전환한다.

SSH 명령 자체가 실패하면 파일 전송 방식 문제가 아니라 접속 정보, SSH 키, 호스트 키, 방화벽 또는 공급 환경 문제로 분류한다.
원격 `.part` 파일의 크기가 0보다 크더라도 해시가 일치하지 않으면 사용할 수 없다.

브라우저 업로드는 정상 운영 경로가 아니다.
SSH 표준 스트림도 사용할 수 없고 결과 손실을 막기 위한 긴급 회수만 남았을 때 사용자 승인을 받은 뒤에만 사용한다.

## 근거 기록

- `Runpod RTX A5000 실제 스크리닝 검증 수행`에서는 최초 `scp`가 `Operation not permitted`로 실패한 뒤 SSH 압축 자료 흐름으로 입력을 옮기고 세 입력 파일의 SHA-256 일치를 확인했다.
- Codex 작업 `Runpod 이슈 102 GPU 실행 확인`에서는 `scp`가 다시 차단됐고 `ssh ... 'cat > 원격파일' < 로컬파일`로 실행 스크립트와 입력 묶음을 전송했다.
- 같은 작업에서 Vast.ai 인스턴스에도 실행 스크립트와 약 11MB 입력 묶음을 SSH 표준 입력 방식으로 전송하고 원격 해시와 크기를 확인했다.
