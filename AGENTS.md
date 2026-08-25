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

### Remote GPU provider selection

For S6E8 external GPU runs, Vast.ai is the primary execution environment and Runpod is the fallback execution environment.
If the user does not name a provider, start with Vast.ai and follow `docs/agents/vast-resource-control.md` before creating any paid resource.
Use Runpod only when the switch conditions in that document are met.
Do not select Kaggle or Colab for an improvement-judgment run.
Kaggle and Colab are limited to human-observed compatibility checks and diagnostics, and an agent must not choose them merely because a model needs a GPU.
This policy comes from GitHub issues 123 and 126 and supersedes older Kaggle and Runpod execution precedents.
