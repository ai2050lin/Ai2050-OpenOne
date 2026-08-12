#!/usr/bin/env python3
"""Run the exact-prompt Phase1073 behavior calibration in FP16."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1071_behavior_calibration_scan as engine
import phase1073_behavior_calibration_protocol as protocol


engine.protocol = protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    engine.run(args.model)


if __name__ == "__main__":
    main()
