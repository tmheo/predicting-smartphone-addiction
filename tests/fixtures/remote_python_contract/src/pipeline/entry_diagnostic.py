from __future__ import annotations

import argparse
from pathlib import Path
import sys

import idna


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--expected-python-prefix", required=True)
    args = parser.parse_args()

    assert idna.__version__ == "3.10"
    assert sys.prefix == args.expected_python_prefix
    args.marker.touch()


if __name__ == "__main__":
    main()
