from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installer", required=True)
    parser.add_argument("--installer-version", required=True)
    args = parser.parse_args()

    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    record = {
        "installer": {
            "name": args.installer,
            "version": args.installer_version,
        },
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }

    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
