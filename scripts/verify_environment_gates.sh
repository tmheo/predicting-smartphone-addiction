#!/bin/sh
set -eu

usage() {
    printf '%s\n' \
        "usage: $0 [--isolated] [--source-root VERIFIED_WORKTREE]" >&2
    exit 2
}

SOURCE_ROOT=
ISOLATED=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --isolated)
            ISOLATED=1
            shift
            ;;
        --source-root)
            [ "$#" -ge 2 ] || usage
            SOURCE_ROOT=$2
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

SCRIPT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
WORKTREE_ROOT=$(dirname -- "$SCRIPT_ROOT")

[ -f "$WORKTREE_ROOT/pyproject.toml" ] || {
    printf 'pyproject.toml not found under %s\n' "$WORKTREE_ROOT" >&2
    exit 2
}
[ -f "$WORKTREE_ROOT/private-inputs.sha256" ] || {
    printf 'private-inputs.sha256 not found under %s\n' "$WORKTREE_ROOT" >&2
    exit 2
}

if [ "$ISOLATED" -eq 1 ]; then
    command -v git >/dev/null 2>&1 || {
        printf 'git is required for isolated environment verification\n' >&2
        exit 2
    }
    git -C "$WORKTREE_ROOT" rev-parse --verify HEAD >/dev/null 2>&1 || {
        printf 'isolated verification requires a Git commit\n' >&2
        exit 2
    }
    [ -z "$(git -C "$WORKTREE_ROOT" status --porcelain --untracked-files=normal)" ] || {
        printf 'isolated verification requires a clean worktree\n' >&2
        exit 2
    }
    if [ -z "$SOURCE_ROOT" ]; then
        SOURCE_ROOT=$WORKTREE_ROOT
    fi
    case "$SOURCE_ROOT" in
        /*) ;;
        *)
            printf 'source root must be an absolute path: %s\n' "$SOURCE_ROOT" >&2
            exit 2
            ;;
    esac
    [ -d "$SOURCE_ROOT" ] || {
        printf 'source root is not a directory: %s\n' "$SOURCE_ROOT" >&2
        exit 2
    }

    ISOLATED_PARENT=$(mktemp -d "${TMPDIR:-/tmp}/environment-gate.XXXXXX")
    ISOLATED_WORKTREE=$ISOLATED_PARENT/worktree
    cleanup_isolated_worktree() {
        cleanup_status=$?
        trap - 0 1 2 15
        git -C "$WORKTREE_ROOT" worktree remove --force \
            "$ISOLATED_WORKTREE" >/dev/null 2>&1 || true
        rmdir "$ISOLATED_PARENT" >/dev/null 2>&1 || true
        exit "$cleanup_status"
    }
    trap cleanup_isolated_worktree 0 1 2 15

    git -C "$WORKTREE_ROOT" worktree add --detach "$ISOLATED_WORKTREE" HEAD
    printf 'Verifying committed HEAD in isolated worktree: %s\n' \
        "$ISOLATED_WORKTREE"
    "$ISOLATED_WORKTREE/scripts/verify_environment_gates.sh" \
        --source-root "$SOURCE_ROOT"
    exit 0
fi

command -v uv >/dev/null 2>&1 || {
    printf 'uv is required\n' >&2
    exit 2
}
command -v docker >/dev/null 2>&1 || {
    printf 'Docker is required for the remote Python environment gate\n' >&2
    exit 2
}
docker info >/dev/null 2>&1 || {
    printf 'Docker is not available for the remote Python environment gate\n' >&2
    exit 2
}

cd "$WORKTREE_ROOT"
if [ ! -e data ]; then
    [ -n "$SOURCE_ROOT" ] || {
        printf 'data/ is absent; pass --source-root with a verified worktree\n' >&2
        exit 2
    }
    case "$SOURCE_ROOT" in
        /*) ;;
        *)
            printf 'source root must be an absolute path: %s\n' "$SOURCE_ROOT" >&2
            exit 2
            ;;
    esac
    [ -d "$SOURCE_ROOT" ] || {
        printf 'source root is not a directory: %s\n' "$SOURCE_ROOT" >&2
        exit 2
    }
    uv run --frozen python -m pipeline.private_inputs prepare \
        --source-root "$SOURCE_ROOT"
fi

uv run --frozen python -m pipeline.private_inputs check
uv run --frozen pytest --collect-only
uv run --frozen pytest tests/test_remote_python_contract.py
uv run --frozen pytest
