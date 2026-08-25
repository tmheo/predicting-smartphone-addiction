# 이슈 #410 시제품: 후보 풀 재구축 정확 검색

버리는 코드다.
공식 구현은 이슈 #412가 정하는 발주 이슈에서 `src/pipeline`에 새로 만든다.

`prototype_pool_rebuild_search.py`는 실행 당시 `scripts/` 아래에 있었고 저장소 루트를 `Path(__file__).resolve().parents[1]`로 잡는다.
다시 실행하려면 파일을 `scripts/`로 복사한 뒤 `prepare`, `search`, `finish` 순서로 실행한다.
`results/precommit.json`의 `code_sha256`는 이 파일 내용의 해시다.

결과 요약은 `results/report.md`, 원자료는 `results/report.json`과 `results/scope-*/`에 있다.
