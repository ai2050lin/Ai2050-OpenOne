#!/usr/bin/env python3
"""Run a contiguous range of local Phase tests without external PYTHONPATH setup."""

from __future__ import annotations

import argparse
import re
import sys
import unittest
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
PHASE_TEST_PATTERN = re.compile(r"test_phase(?P<phase>\d+).*\.py$")


def selected_modules(first_phase: int, last_phase: int) -> list[str]:
    modules: list[tuple[int, str]] = []
    for path in TEST_DIR.glob("test_phase*.py"):
        match = PHASE_TEST_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        phase = int(match.group("phase"))
        if first_phase <= phase <= last_phase:
            modules.append((phase, path.stem))
    return [module for _, module in sorted(modules)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_phase", type=int)
    parser.add_argument("last_phase", type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.first_phase > args.last_phase:
        parser.error("first_phase must not exceed last_phase")

    sys.path.insert(0, str(TEST_DIR))
    modules = selected_modules(args.first_phase, args.last_phase)
    if not modules:
        parser.error("no Phase tests found in the requested range")

    print(
        f"Running {len(modules)} test modules for Phase "
        f"{args.first_phase}-{args.last_phase}",
        flush=True,
    )
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
