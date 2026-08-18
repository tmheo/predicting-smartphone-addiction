#!/bin/sh
set -eu

usage() {
    printf '%s\n' \
        "usage: $0 [--platform PLATFORM] IMAGE" >&2
    exit 2
}

PLATFORM=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --platform)
            [ "$#" -ge 2 ] || usage
            PLATFORM=$2
            shift 2
            ;;
        --*)
            usage
            ;;
        *)
            [ "$#" -eq 1 ] || usage
            IMAGE=$1
            shift
            ;;
    esac
done

[ -n "${IMAGE:-}" ] || usage
command -v docker >/dev/null 2>&1 || {
    printf 'Docker is required to verify the target remote image\n' >&2
    exit 2
}
docker info >/dev/null 2>&1 || {
    printf 'Docker is not available to verify the target remote image\n' >&2
    exit 2
}

if [ -n "$PLATFORM" ]; then
    set -- --platform "$PLATFORM"
else
    set --
fi

docker run --rm "$@" --entrypoint sh "$IMAGE" -c '
set -eu
command -v python3 >/dev/null 2>&1 || {
    printf "target image has no python3 executable\n" >&2
    exit 1
}
probe_root=$(mktemp -d /tmp/remote-python-venv.XXXXXX)
cleanup() {
    rm -rf "$probe_root"
}
trap cleanup EXIT HUP INT TERM
if ! python3 -m venv "$probe_root"; then
    if [ -r /etc/os-release ]; then
        sed -n "s/^\(ID\|VERSION_ID\)=/target image \1=/p" /etc/os-release >&2
    fi
    printf "%s\n" \
        "target image cannot create a virtual environment" \
        "Debian and Ubuntu images normally require python3-venv" \
        "select another image or freeze the OS package bootstrap in the remote execution specification" >&2
    exit 1
fi
"$probe_root/bin/python" -m pip --version >/dev/null
'
