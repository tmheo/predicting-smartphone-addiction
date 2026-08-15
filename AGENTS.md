# AGENTS.md

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
