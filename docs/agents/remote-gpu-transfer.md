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

## 대회 전용 SSH 접속 키

S6E8 동안 Vast.ai와 Runpod은 공급자별로 분리한 고정 Ed25519 키 쌍을 사용한다.
개별 원격 실행 작업마다 키를 만들거나 계정에 등록하고 삭제하지 않는다.
개인 기본 SSH 키나 두 공급자가 함께 쓰는 키도 사용하지 않는다.

- Vast.ai 개인 키: `/Users/taemyungheo/.ssh/predicting-smartphone-addiction-vast-ed25519`
- Vast.ai 공개 키 지문: `SHA256:8AgBh2VtIo6dOWddsToBBf/GRkHVWgOPiq+EoZz19Os`
- Runpod 개인 키: `/Users/taemyungheo/.ssh/predicting-smartphone-addiction-runpod-ed25519`
- Runpod 공개 키 지문: `SHA256:SLzTJ8I/aiTM3teG0Nw90iowaDfUiAyayOyMkUyGKJ0`

2026년 8월 16일 두 공급자 계정에서 각 공개 키와 로컬 공개 키가 정확히 일치하는 항목이 하나씩 등록됐음을 확인했다.
Runpod의 이전 대회용 고아 공개 키와 종료된 이슈 102·106의 작업 전용 키는 계정에서 제거했다.

개인 키 권한은 `0600`, 공개 키 권한은 `0644`로 유지한다.
공개 키는 각 공급자 계정에 한 번 등록하고 대회가 끝날 때까지 유지한다.
대회 종료, 해당 공급자 사용 종료, 로컬 장비 분실 또는 키 유출 의심 때만 계정 등록과 로컬 키 쌍을 함께 폐기한다.

유료 자원을 만들기 전에 공급자 계정의 공개 키가 로컬 공개 키와 정확히 일치하는지 읽기 전용으로 확인한다.
등록값이 없거나 다르면 유료 자원을 만들지 않고 계정 설정 경로에서 복구한다.
계정 공개 키를 새로 등록한 뒤에는 그 전에 만들어 둔 계산 자원이 자동으로 갱신된다고 가정하지 않고 새 계산 자원을 만든다.

접속에 사용하는 클라이언트 키를 고정해도 원격 서버의 SSH 호스트 키는 계산 자원마다 별도로 검증해야 한다.
각 원격 실행 작업은 전용 `known_hosts` 파일에 해당 계산 자원의 호스트 키를 고정하고 `StrictHostKeyChecking=yes`를 사용한다.

## 필수 규칙

1. `scp`, `sftp`, 브라우저 파일 업로드를 평상시 전송 경로로 사용하지 않는다.
2. 여러 파일은 로컬에서 하나의 입력 전송 묶음으로 만든다.
3. 묶음 안의 입력·결과 manifest에는 묶음 루트 기준 상대 경로만 기록한다.
4. macOS 보조 메타데이터가 들어가지 않도록 묶음을 만들 때 `COPYFILE_DISABLE=1`을 사용한다.
5. 전송 전에 로컬 SHA-256을 계산한다.
6. 원격에서는 최종 경로가 아닌 `.part` 경로에 먼저 저장한다.
7. 원격 SHA-256이 로컬 값과 일치할 때만 원자적으로 최종 이름으로 바꾼다.
8. 결과 회수도 로컬 `.part` 경로에 먼저 받은 뒤 원격 해시와 일치할 때만 최종 이름으로 바꾼다.
9. 전송 실패나 연결 중단 뒤에는 부분 파일을 결과나 입력으로 사용하지 않는다.
10. SSH 호스트 키를 미리 확인하고 `StrictHostKeyChecking=yes`를 사용한다.
11. 공급자별 대회 전용 SSH 접속 키를 사용하고 원격 실행 작업별 키를 만들거나 삭제하지 않는다.
12. SSH 호스트 키와 `known_hosts` 파일은 원격 실행 작업별로 분리한다.
13. Git 저장소를 포함한 묶음의 파일별 해시 목록에서는 `.git/` 아래를 제외하고, 저장소 무결성과 상태는 별도 Git 관문으로 확인한다.
14. 원격 저장소의 `safe.directory`는 검증된 작업 경로 하나에만 자식 프로세스 환경으로 지정하고 전역 설정이나 `*`를 사용하지 않는다.
15. 자동 시작 프로그램의 `git status`는 선택적 잠금을 끈 상태로 실행하고, 제어 세션에서 시작 직전에 별도로 실행하지 않는다.
16. 파일별 해시 검증, Git 관문과 작업 시작은 하나의 고정 시작 프로그램이 순서대로 소유한다.
17. 시작 관문을 다시 수행할 때는 사용하던 디렉터리를 고쳐 쓰지 않고 검증된 원본 묶음을 새 작업 경로에 다시 푼다.

## 접속 변수

아래 값은 공급자 화면이나 공식 명령줄 도구로 확인한 실제 값으로 설정한다.
비밀 값과 SSH 개인 키를 저장소에 기록하지 않는다.
Vast.ai에서는 `show instance`의 원시 SSH 필드를 직접 조합하지 않고 `vastai ssh-url <인스턴스 ID>` 결과에서 대상과 포트를 가져온다.
Jupyter 실행 유형에서는 명령줄 도구가 원시 `ssh_port`를 보정하므로 이 경로를 우회하면 잘못된 포트에 접속할 수 있다.

```bash
GPU_SSH_KEY=/absolute/path/to/provider-specific-competition-key
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

manifest에 절대 경로를 기록하면 원격에서 만든 결과를 로컬에 푼 뒤 기본 검증 명령이 실패한다.
결과 manifest도 결과 디렉터리 안으로 이동한 뒤 상대 경로로 만든다.

```bash
(
  cd "$GPU_REMOTE_ROOT/results"
  sha256sum * > result-manifest.sha256
)
```

zsh에서 manifest를 읽는 반복문의 변수 이름으로 `path`를 쓰지 않는다.
`path`는 zsh의 명령 검색 경로와 연결된 예약 배열이므로 덮어쓰면 반복문 안에서 `shasum`, `awk` 같은 명령을 찾지 못한다.
`file_path`처럼 역할이 분명한 이름을 사용한다.

## Git 저장소를 포함한 입력 묶음

실행 커밋을 기록하려고 `.git`을 포함한 작업 사본을 전송할 때는 Git 저장소 내부 상태와 실행 입력 내용을 같은 방식으로 검증하지 않는다.
전체 압축 파일의 SHA-256은 `.git`까지 포함한 전송 무결성을 보장하고, 파일별 해시 목록인 `input-manifest.sha256`은 제품 파일, 비공개 입력, 설정과 실행 제어 파일을 검증한다.
저장소의 객체와 실행 커밋, 작업 폴더 상태는 원격 Git 관문이 별도로 검증한다.

`.git/index`는 제품 입력이 아니라 Git이 관리하는 가변 상태다.
Git 공식 문서에 따르면 `git status`는 기본적으로 작업 폴더의 통계 정보를 인덱스에 새로 기록할 수 있다.
제품 파일이 바뀌지 않아도 전송 직후의 `.git/index` 해시가 달라질 수 있으므로 `.git/` 아래 파일을 파일별 해시 목록에 넣지 않는다.
자동화에서 상태를 읽을 때는 [`git --no-optional-locks status`](https://git-scm.com/docs/git-status#_background_refresh)를 사용한다.

입력 준비는 연결된 작업 폴더의 `.git` 연결 정보 파일이 아니라 독립된 Git 복제본을 사용한다.
로컬 준비 폴더에서 Git 상태를 먼저 확인한 뒤 `.git/`을 제외한 파일별 해시 목록을 만든다.

```bash
GPU_INPUT_ROOT=/absolute/path/to/prepared-input
GPU_STAGED_REPO_REL=repo
GPU_EXPECTED_COMMIT=0123456789abcdef0123456789abcdef01234567
GPU_STAGED_REPO="$GPU_INPUT_ROOT/$GPU_STAGED_REPO_REL"

test -d "$GPU_STAGED_REPO/.git"
test "$(git -C "$GPU_STAGED_REPO" rev-parse HEAD)" = "$GPU_EXPECTED_COMMIT"
test -z "$(git -C "$GPU_STAGED_REPO" --no-optional-locks \
  status --porcelain=v1 --untracked-files=normal)"

(
  cd "$GPU_INPUT_ROOT"
  find . -type f \
    ! -path "./$GPU_STAGED_REPO_REL/.git/*" \
    ! -name input-manifest.sha256 \
    -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 \
    > input-manifest.sha256
)
```

원격에서는 전체 묶음 해시가 일치한 뒤 새 디렉터리에 압축을 푼다.
압축을 푼 사용자와 실제 작업 사용자가 달라질 수 있으므로, 저장소의 정규화된 절대 경로 하나만 [`safe.directory`](https://git-scm.com/docs/git-config#Documentation/git-config.txt-safedirectory)로 자식 프로세스 환경에 전달한다.
`git config --global --add safe.directory ...`로 원격 계정의 지속 설정을 바꾸거나 `safe.directory=*`로 소유권 검사를 끄지 않는다.

다음 순서를 하나의 원격 시작 프로그램에 고정한다.
제어 세션은 이 프로그램을 시작하기 직전에 별도의 `git status`, `git update-index` 또는 다른 Git 명령을 실행하지 않는다.

```bash
GPU_REMOTE_INPUT="$GPU_REMOTE_ROOT/input"
GPU_REMOTE_PROJECT="$(realpath -e "$GPU_REMOTE_INPUT/repo")"
GPU_EXPECTED_COMMIT=0123456789abcdef0123456789abcdef01234567

case "$GPU_REMOTE_PROJECT" in
  "$GPU_REMOTE_INPUT"/*) ;;
  *) printf '원격 저장소가 작업 입력 경로 밖에 있다.\n' >&2; exit 1 ;;
esac

export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$GPU_REMOTE_PROJECT"
export GIT_OPTIONAL_LOCKS=0

(
  cd "$GPU_REMOTE_INPUT"
  sha256sum --check input-manifest.sha256
)
git -C "$GPU_REMOTE_PROJECT" fsck --strict --no-progress --no-dangling
test "$(git -C "$GPU_REMOTE_PROJECT" rev-parse HEAD)" = "$GPU_EXPECTED_COMMIT"
test -z "$(git -C "$GPU_REMOTE_PROJECT" \
  status --porcelain=v1 --untracked-files=normal)"
(
  cd "$GPU_REMOTE_INPUT"
  sha256sum --check input-manifest.sha256
)

exec "$GPU_REMOTE_INPUT/remote-job.sh"
```

첫 파일별 해시 검사는 전송한 실행 내용을 확인한다.
Git 관문 뒤의 두 번째 검사는 상태 확인 과정이 해시 목록 대상 파일을 바꾸지 않았음을 확인한다.
`.git/`은 파일별 해시 목록 대상이 아니므로 Git이 인덱스 통계 정보를 다뤄도 제품 파일 불일치로 잘못 판정하지 않는다.
같은 환경 변수는 `exec` 뒤의 작업과 그 자식 프로세스에도 이어져 `pipeline.entry_diagnostic` 같은 후속 Git 명령이 소유권 검사에서 멈추지 않는다.

## 실행 제어 파일 고정

실행 스크립트, 설정, 입력 manifest와 데이터는 본 실행 전에 하나의 판본으로 고정한다.
셸 문법 검사, 파일별 해시, 전체 묶음 해시와 원격 압축 해제 뒤 manifest 검증을 모두 통과한 판본만 실행한다.

원격 프로세스가 시작된 뒤에는 로컬 사본과 원격 실행 스크립트를 수정하거나 같은 경로에 덮어쓰지 않는다.
일시 정지된 프로세스도 재개 뒤 파일을 다시 읽을 수 있으므로 실행 중인 것으로 취급한다.
수정이 필요하면 기존 작업의 결과 또는 실패와 자원 정리를 먼저 확정하고, 새 작업 식별자·새 작업 루트·새 파일명으로 다시 준비한다.
새 판본은 해시와 묶음을 처음부터 다시 검증한다.

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
  "test ! -e '$GPU_REMOTE_ROOT/input' && \
   mkdir -p '$GPU_REMOTE_ROOT/input' && \
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
Git 저장소를 포함한 묶음에서 이 검사는 압축 해제 직후의 예비 확인이다.
이후 제어 세션은 Git 명령을 실행하지 않고, 위의 고정 시작 프로그램이 파일별 해시 목록을 다시 검증한 뒤 Git 관문과 작업 시작을 이어서 수행한다.

## 입력 자료를 저장소에 연결하기

저장소가 `data/`처럼 디렉터리 형태만 무시할 때는 외부 입력 디렉터리 전체를 같은 이름의 심볼릭 링크로 만들지 않는다.
Git은 디렉터리 심볼릭 링크를 디렉터리로 취급하지 않으므로 `data/` 무시 규칙이 적용되지 않고 작업 폴더에 추적되지 않은 `data` 항목이 생길 수 있다.
저장소 안에 실제 `data` 디렉터리를 만든 뒤 필요한 입력 파일을 그 안에 각각 심볼릭 링크로 연결한다.
연결 직후 입력 파일을 읽을 수 있는지와 `git status --porcelain=v1` 결과가 비어 있는지를 모두 확인하고 나서 Python 준비 관문을 시작한다.

```bash
mkdir data
ln -s "$GPU_REMOTE_INPUT/data/train.csv" data/train.csv
ln -s "$GPU_REMOTE_INPUT/data/test.csv" data/test.csv
ln -s "$GPU_REMOTE_INPUT/data/sample_submission.csv" data/sample_submission.csv

test -s data/train.csv
test -s data/test.csv
test -s data/sample_submission.csv
test -z "$(git status --porcelain=v1)"
```

## Python 준비와 실행

유료 원격 자원을 만들기 전에 새 Git 작업 폴더에서 README의 환경 관문 사전 확인을 통과해야 한다.
검증된 입력이 있는 기존 작업 폴더를 원본으로 지정해 다음 단일 진입점을 실행한다.

```bash
cd /absolute/path/to/new-worktree
scripts/verify_environment_gates.sh \
  --source-root /absolute/path/to/verified-worktree
```

이 명령은 비커밋 입력 준비와 전체 시험 수집을 확인하고, 외부 관리 Python 원격 유사 환경에서 가상환경 준비와 `pipeline.entry_diagnostic` 시작까지 검증한 뒤 전체 시험을 실행한다.
Docker를 사용할 수 없거나 어느 관문이든 실패하면 유료 원격 자원을 만들지 않는다.
이 로컬 사전 확인은 실제 공급자 자원 안의 준비 경계를 대신하지 않는다.
기본 작업 폴더에서 병합한 커밋을 확인할 때는 기존 `data/`를 검사 대상으로 삼지 않도록 `scripts/verify_environment_gates.sh --isolated`를 실행한다.
이 명령은 현재 커밋을 임시 Git 작업 폴더에서 확인하고 원래 `data/`는 변경하지 않는다.

### 대상 컨테이너 이미지의 Python 관문

저장소의 환경 관문은 공통 실행 스크립트가 외부 관리 Python을 훼손하지 않는지 검사하기 위해 시험용 컨테이너에 `python3-venv`를 먼저 설치한다.
따라서 이 검사는 실제 GPU 이미지에 가상환경 구성 요소가 들어 있는지 보장하지 않는다.
Python 실행 파일이 있어도 `venv`와 `ensurepip`는 운영체제의 별도 패키지일 수 있다.
Vast.ai나 Runpod을 이전에 성공적으로 사용했더라도 컨테이너 이미지나 꼬리표가 달라지면 이 전제를 다시 확인해야 한다.

원격 실행 명세에서 컨테이너 이미지를 고른 뒤 유료 자원을 만들기 전에 다음 검사를 통과해야 한다.
회사가 제공한 Mac처럼 호스트 구조와 원격 구조가 다르면 원격 구조를 `--platform`에 명시한다.

```bash
GPU_CONTAINER_IMAGE=registry.example/image@sha256:fixed-digest

scripts/verify_remote_image_python.sh \
  --platform linux/amd64 \
  "$GPU_CONTAINER_IMAGE"
```

이 명령은 지정한 이미지 자체에서 임시 가상환경을 만들고 그 안의 Python과 pip를 확인한 뒤 임시 경로를 삭제한다.
검사한 이미지 참조와 원격 실행 명세의 이미지 참조는 정확히 같아야 한다.
가변 꼬리표를 사용할 수밖에 없으면 작업마다 다시 검사하고 실제 이미지 식별자를 원격 실행 장부에 기록한다.

로컬 Docker가 대상 구조를 실행할 수 없거나 공급자 전용 이미지를 가져올 수 없는 경우만 생성 전 검사를 예외로 둘 수 있다.
이 경우 계산 자원이 `running`이 되고 SSH 인증이 끝난 직후, 입력 전송 전에 다음 동등 검사를 수행한다.

```bash
GPU_REMOTE_PROBE="$GPU_REMOTE_ROOT/preflight-python-env"

test ! -e "$GPU_REMOTE_PROBE"
python3 -m venv "$GPU_REMOTE_PROBE"
"$GPU_REMOTE_PROBE/bin/python" -m pip --version
rm -rf "$GPU_REMOTE_PROBE"
```

Debian 또는 Ubuntu 계열 이미지에서 `ensurepip is not available`로 실패하면 보통 `python3-venv`가 빠진 것이다.
다른 배포판에서는 `/etc/os-release`와 해당 배포판의 Python 패키지 구성을 확인한다.
호환되는 다른 이미지를 선택하거나, 필요한 운영체제 패키지 설치 명령과 설치 뒤 위 검사를 원격 실행 명세의 불변 준비 단계에 함께 고정한다.
공통 실행 스크립트 안에서 운영체제 패키지를 설치하거나 시스템 pip를 우회하지 않는다.
준비 명령을 바꿨다면 기존 실행을 재개하지 않고 새 작업 식별자, 새 작업 루트와 새 입력 묶음으로 다시 시작한다.

이 실패는 공급자 호스트 장애가 아니라 컨테이너 이미지 호환성 실패로 기록한다.
같은 이미지를 다른 호스트에서 반복 실행해도 해결되지 않으므로 공급자 전환을 위한 서로 다른 호스트 실패 횟수에 포함하지 않는다.

묶음 해시와 내부 파일 해시를 확인한 다음 모델 코드를 실행하기 전에 저장소의 공통 Python 준비 관문을 통과해야 한다.
입력 전송 묶음에는 `pyproject.toml`, `uv.lock`, `src/`, `scripts/run_remote_python.sh`와 `scripts/record_remote_python.py`를 포함한다.
`pipeline.entry_diagnostic`은 종료부에서 `git rev-parse HEAD`로 실행 커밋을 기록하므로, 진입 진단을 실행할 묶음은 `git archive` 결과물이 아니라 실행 커밋의 git 체크아웃(얕은 클론으로 `.git` 포함)이어야 한다.
이슈 199의 첫 fold 0 실행은 `.git` 없는 묶음 때문에 95분의 계산을 산출물 저장 직전에 잃었다.
작업별 `run-logs/` 실행 파일에서 사전 검사 없이 시스템 Python 패키지 설치를 즉흥적으로 구현하지 않는다.
`pip install --break-system-packages`도 사용하지 않는다.

다음 명령을 공급자와 관계없이 같은 형식으로 사용한다.
경로와 Python 인수는 원격 실행 명세에 고정한 값으로 바꾼다.

```bash
GPU_REMOTE_PROJECT="$GPU_REMOTE_ROOT/input"
GPU_REMOTE_VENV="$GPU_REMOTE_ROOT/python-env"
GPU_REMOTE_EVIDENCE="$GPU_REMOTE_ROOT/results/python-environment.json"

"$GPU_REMOTE_PROJECT/scripts/run_remote_python.sh" \
  --system-python python3 \
  --project "$GPU_REMOTE_PROJECT" \
  --venv "$GPU_REMOTE_VENV" \
  --evidence "$GPU_REMOTE_EVIDENCE" \
  -- \
  -m pipeline.entry_diagnostic configs/expNNN.yaml \
  --out-dir "$GPU_REMOTE_ROOT/results/entry-expNNN" \
  --reference \
  --expected-baseline-auc 0.968294911389327
```

위 예시는 현재 champion의 동등 단계 기준 실행을 만드는 명령이다.
challenger 원격 실행은 `README.md`의 모델 진입 진단 절에 따라 기준 진단 JSON과 검증 예측, 비교 모드와 허용 모델 차이 축을 원격 실행 명세에 함께 고정한다.

`--` 뒤는 가상환경 Python에 전달할 인수이며 별도의 `python` 또는 `pip` 실행 파일을 지정하지 않는다.
가상환경 경로는 존재하지 않는 작업 전용 경로여야 한다.
관문은 시스템 Python을 가상환경 생성에만 사용하고, 이후 설치와 사용자 명령은 가상환경 실행 파일로 수행한다.
고정한 `uv` 판본으로 선언과 `uv.lock`의 일치를 검사한 뒤 잠긴 의존성만 설치한다.
Python 실행 파일, Python 판본, 설치 도구 판본과 설치된 패키지 판본은 증거 JSON에 기록되므로 실행 기록 묶음에 포함해 회수한다.
준비 단계가 실패하면 모델 명령을 시작하지 않고 해당 원격 실행 작업을 실패로 확정한 뒤 결과 회수와 자원 정리 절차를 따른다.

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

## main MLflow 반입 관문

원격 결과 묶음과 작업 트리 안의 MLflow 기록은 최종 실행 장부가 아니다.
원격 실행 작업은 결과 묶음을 기본 `main` 작업 폴더의 `run-logs/` 아래 작업 전용 보존 경로로 회수하고, 그 작업 폴더의 MLflow에 반입해야 완료로 판정한다.
이슈 작업 트리의 `mlflow.db` 또는 `mlruns/`에만 반입한 상태에서 작업 트리나 브랜치를 삭제하지 않는다.

다음 순서를 지킨다.

1. 결과 묶음의 완성본을 기본 `main` 작업 폴더의 `run-logs/` 아래로 옮기고 원격 SHA-256과 다시 대조한다.
2. 기본 `main` 작업 폴더에서 `uv run python -m pipeline.bundle import <결과 묶음>`을 실행한다.
3. 반입 명령이 반환한 실행 식별자로 이름, 단계, 시드, OOF AUC, 출처 실행 식별자와 묶음 SHA-256을 다시 조회한다.
4. 설정, `oof.parquet`, 시드별 OOF, 중요도, 시험 예측과 제출 파일 등 실행 단계가 요구하는 산출물이 main MLflow에서 열리는지 확인한다.
5. 이슈 종료 기록에 main MLflow 실행 식별자와 반입 검증 결과를 남긴다.
6. 위 확인이 끝난 뒤에만 이슈 작업 트리와 브랜치를 삭제한다.

결과 묶음 파일은 main MLflow 반입과 산출물 조회가 끝날 때까지 지우거나 작업 트리 전용 경로에만 두지 않는다.
이슈 코멘트의 수치, 실행 식별자 또는 묶음 해시만으로 main MLflow 반입을 대신할 수 없다.
장부 누락만 복구하기 위해 같은 GPU 실험을 다시 실행하지 않는다.
원본 묶음을 복구할 수 없으면 정상 반입 실행을 꾸미지 말고, 남아 있는 원본 산출물과 수치를 불완전 복구 기록으로 명시한 뒤 사용자에게 손실 범위를 알린다.

## 실패 시 처리

`scp`에서 `Operation not permitted`가 발생하면 다시 시도하거나 보안 설정을 바꾸지 않는다.
즉시 SSH 표준 스트림 전송으로 전환한다.

SSH 명령 자체가 실패하면 파일 전송 방식 문제가 아니라 접속 정보, SSH 키, 호스트 키, 방화벽 또는 공급 환경 문제로 분류한다.
원격 `.part` 파일의 크기가 0보다 크더라도 해시가 일치하지 않으면 사용할 수 없다.

`detected dubious ownership`은 공급자 호스트 장애가 아니라 원격 시작 프로그램이 저장소의 정확한 `safe.directory` 환경을 준비하지 않은 입력 절차 실패다.
`.git/index: FAILED`가 유일한 파일별 해시 목록 불일치이고 추적 파일 상태가 깨끗하더라도 이를 제품 파일 변경으로 해석하지 않는다.
이 경우 현재 디렉터리에서 해시 목록을 다시 만들거나 실패 결과를 정상 결과로 고치지 않는다.
실패 결과를 보존하고 검증된 원본 묶음을 새 입력 경로에 다시 푼 뒤, `.git/`을 제외한 해시 목록과 단일 시작 프로그램 관문으로 다시 수행한다.
두 실패는 다른 호스트에서 반복해도 해결되지 않으므로 공급자 전환을 위한 서로 다른 호스트 실패 횟수에 포함하지 않는다.

브라우저 업로드는 정상 운영 경로가 아니다.
SSH 표준 스트림도 사용할 수 없고 결과 손실을 막기 위한 긴급 회수만 남았을 때 사용자 승인을 받은 뒤에만 사용한다.

## 근거 기록

- `Runpod RTX A5000 실제 스크리닝 검증 수행`에서는 최초 `scp`가 `Operation not permitted`로 실패한 뒤 SSH 압축 자료 흐름으로 입력을 옮기고 세 입력 파일의 SHA-256 일치를 확인했다.
- Codex 작업 `Runpod 이슈 102 GPU 실행 확인`에서는 `scp`가 다시 차단됐고 `ssh ... 'cat > 원격파일' < 로컬파일`로 실행 스크립트와 입력 묶음을 전송했다.
- 같은 작업에서 Vast.ai 인스턴스에도 실행 스크립트와 약 11MB 입력 묶음을 SSH 표준 입력 방식으로 전송하고 원격 해시와 크기를 확인했다.
- [GitHub 이슈 185](https://github.com/tmheo/predicting-smartphone-addiction/issues/185)에서는 결과 묶음을 이슈 작업 트리의 MLflow에만 반입한 뒤 작업 트리를 삭제해 main MLflow 기록이 남지 않았다.
  이 사례를 계기로 main MLflow 반입과 산출물 재조회가 작업 트리 삭제의 선행 관문이 됐다.
- [exp115 스칼라 token Transformer에 OOF 목표 평균 축을 fold 0 짝비교로 선별한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/286)와 [exp131·exp132 CNN 개선을 exp113 대조군과 3시드로 재검증한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/303)에서는 전체 Git 작업 사본을 파일별 해시 목록에 넣은 뒤 상태 확인으로 `.git/index`가 바뀌어 시작을 다시 수행했다.
- [RealMLP 전처리 기준 집합 값 좌표의 기준 범위 짝비교](https://github.com/tmheo/predicting-smartphone-addiction/issues/331)과 [구현 전 fold 실행 기준과 cv.run_cv 회귀 계약을 고정한다](https://github.com/tmheo/predicting-smartphone-addiction/issues/353)에서는 원격 저장소의 소유권 관문이 시작 프로그램에 고정되지 않아 첫 Git 명령이 중단됐다.
- `구현 전 fold 실행 기준과 cv.run_cv 회귀 계약을 고정한다`에서는 소유권 문제를 고친 뒤 제어 세션이 별도의 `git status`를 실행해 `.git/index` 파일별 해시 목록 불일치까지 연달아 일어났다.
  이 사례를 계기로 `.git/`을 파일별 해시 목록에서 제외하고, 범위가 제한된 `safe.directory`, 선택적 잠금을 끈 상태 확인과 앞뒤 해시 검증을 하나의 시작 프로그램이 소유하게 했다.
