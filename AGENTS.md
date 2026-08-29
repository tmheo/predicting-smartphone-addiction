# AGENTS.md

## Experiment iteration

Fast experiment iteration is the default for this competition.
Use lightweight, directly relevant integrity checks for routine changes, such as parsing changed configuration files, checking hashes, and running `git diff --check`.
Treat automated tests as diagnostic tools: run them only after an actual problem or failure is observed and needs reproduction or diagnosis, or when the user explicitly requests tests.
Routine issue completion, commits, merges, and pushes to the main branch proceed without running test suites or waiting for CI results.

## Agent skills

### Issue tracker

Issues are tracked as GitHub Issues on `tmheo/predicting-smartphone-addiction`, operated via the `gh` CLI.
See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage labels are used as-is: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.
See `docs/agents/triage-labels.md`.

### Discussion incremental updates

S6E8 디스커션의 새 글과 코멘트를 `docs/research/discussion-insights.md`에 반영하는 반복 절차.
See `docs/agents/discussion-update.md`.

### Kaggle public notebook licensing

Kaggle public notebook source is released under the Apache License 2.0.
When reviewing, adapting, or reusing a public notebook, follow the license and provenance procedure in `docs/agents/kaggle-public-notebook-licensing.md`.
Verify licenses for input datasets, pretrained models, packages, and external assets separately.

### Strict external candidate freeze and nested selection

외부 후보 동결 명세 생성과 ADR-0005 중첩 선별 판정의 실행 순서, 결과 확인 전에 고정할 항목, 산출물 대응.
See `docs/agents/strict-external-selection-run.md`.

### Domain docs

Single-context layout: one `CONTEXT.md` and `docs/adr/` at the repo root.
See `docs/agents/domain.md`.

### Remote GPU file transfer

The company security software on the local Mac blocks `scp` file reads even when ordinary SSH commands work.
For Runpod and Vast.ai, never use `scp`, `sftp`, or browser upload as the normal file-transfer path.
Use the SSH standard-stream procedure with SHA-256 verification described in `docs/agents/remote-gpu-transfer.md`.

### Vast.ai resource control

Use the credential, lifecycle, cleanup, and evidence rules in `docs/agents/vast-resource-control.md`.
The accepted baseline is recorded in `docs/agents/vast-control-acceptance-2026-08-15.md`.

### Remote execution provider selection

For S6E8 external GPU runs, Vast.ai is the primary execution environment and Runpod is the fallback execution environment.
If the user does not name a provider, start with Vast.ai and follow `docs/agents/vast-resource-control.md` before creating any paid resource.
Use Runpod only when the switch conditions in that document are met.

For CPU-only improvement-judgment runs, use available local CPU, Kaggle CPU and Vast.ai CPU capacity in parallel when this shortens wall-clock time.
Kaggle CPU results may enter an improvement judgment only when the configurations, folds, seeds and comparison arms were fixed before execution, the standard `pipeline.run` and execution-record bundle path was used, and the imported run passes the input-hash, source-commit, clean-state, rescoring, required-diagnostic and provider-tag checks in `docs/kaggle-gpu-run.md`.
Keep both arms of a matched comparison on the same provider and runtime class.
Different validated candidate pairs may come from local CPU, Kaggle CPU and Vast.ai CPU and may be judged together after import.
Do not use incomplete or timed-out Kaggle runs, and never use a Kaggle Public score as adoption evidence.

Kaggle GPU and Colab remain limited to human-observed compatibility checks and diagnostics unless a later explicit policy changes their scope.
An agent must not choose Kaggle merely because a model needs a GPU.
The GPU-provider policy comes from GitHub issues 123 and 126.
The Kaggle CPU policy generalizes the completed mixed Kaggle CPU and Vast.ai CPU confirmation in GitHub issue 414.
