#!/bin/sh
set -eu

UV_VERSION=0.11.7

usage() {
    printf '%s\n' \
        "usage: $0 --project DIR --venv DIR --evidence FILE [--system-python COMMAND] -- PYTHON_ARGS..." >&2
    exit 2
}

SYSTEM_PYTHON=python3
PROJECT_ROOT=
VENV_ROOT=
EVIDENCE_PATH=

while [ "$#" -gt 0 ]; do
    case "$1" in
        --system-python)
            [ "$#" -ge 2 ] || usage
            SYSTEM_PYTHON=$2
            shift 2
            ;;
        --project)
            [ "$#" -ge 2 ] || usage
            PROJECT_ROOT=$2
            shift 2
            ;;
        --venv)
            [ "$#" -ge 2 ] || usage
            VENV_ROOT=$2
            shift 2
            ;;
        --evidence)
            [ "$#" -ge 2 ] || usage
            EVIDENCE_PATH=$2
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            usage
            ;;
    esac
done

[ -n "$PROJECT_ROOT" ] || usage
[ -n "$VENV_ROOT" ] || usage
[ -n "$EVIDENCE_PATH" ] || usage
[ "$#" -gt 0 ] || usage
[ -f "$PROJECT_ROOT/pyproject.toml" ] || {
    printf 'pyproject.toml not found under %s\n' "$PROJECT_ROOT" >&2
    exit 2
}
[ -f "$PROJECT_ROOT/uv.lock" ] || {
    printf 'uv.lock not found under %s\n' "$PROJECT_ROOT" >&2
    exit 2
}
[ ! -e "$VENV_ROOT" ] || {
    printf 'virtual environment path already exists: %s\n' "$VENV_ROOT" >&2
    exit 2
}

"$SYSTEM_PYTHON" -m venv "$VENV_ROOT"
"$VENV_ROOT/bin/python" -m pip install \
    --disable-pip-version-check \
    --no-input \
    "uv==$UV_VERSION"

UV_REPORTED_VERSION=$("$VENV_ROOT/bin/python" -m uv --version)
case "$UV_REPORTED_VERSION" in
    "uv $UV_VERSION"*) ;;
    *)
        printf 'unexpected uv version: %s\n' "$UV_REPORTED_VERSION" >&2
        exit 1
        ;;
esac
if ! VIRTUAL_ENV="$VENV_ROOT" "$VENV_ROOT/bin/python" -m uv sync \
    --active \
    --locked \
    --no-dev \
    --project "$PROJECT_ROOT" \
    --python "$VENV_ROOT/bin/python"; then
    printf 'dependency lock is not current or dependency preparation failed before model entry\n' >&2
    exit 1
fi

SCRIPT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
mkdir -p "$(dirname -- "$EVIDENCE_PATH")"
"$VENV_ROOT/bin/python" "$SCRIPT_ROOT/record_remote_python.py" \
    --output "$EVIDENCE_PATH" \
    --installer uv \
    --installer-version "$UV_VERSION"

PATH="$VENV_ROOT/bin:$PATH" \
VIRTUAL_ENV="$VENV_ROOT" \
exec "$VENV_ROOT/bin/python" "$@"
