#!/usr/bin/env python3
"""Run the frozen Phase1086 behavior gate, one FP16 model at a time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1086_signed_shared_field_protocol as protocol

sys.modules["phase1083_same_carrier_attribute_protocol"] = protocol
import phase1083_same_carrier_attribute_behavior as engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    engine.protocol = protocol
    engine.run(args.model)


if __name__ == "__main__":
    main()
