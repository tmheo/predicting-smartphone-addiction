# 생성기 지문 진단 고정 환경

`scripts/diagnose_generator_fingerprints.py`(#89)는 프로젝트 기본 환경이 아니라 이 고정 환경에서 실행한다.
규약(docs/research/generator-fingerprint-protocol.md)이 고정한 SDV 1.38.0과 CTGAN 0.12.1을 쓴다.

```bash
uv venv genfp-env --python 3.12
uv pip install --python genfp-env/bin/python -r scripts/generator_env/requirements-lock.txt
```

TabDDPM은 공식 저장소 커밋 `b476257`을 같은 환경에서 사용한다.

```bash
git clone https://github.com/yandex-research/tab-ddpm
git -C tab-ddpm checkout b476257dd460b778ba09eb97f7a51d6490fa17f8
```

실행 예시는 진단 스크립트 상단 docstring을 따른다.
`--tabddpm-repo`에 위 클론 경로를 넘긴다.
