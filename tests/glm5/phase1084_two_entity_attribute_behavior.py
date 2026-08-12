#!/usr/bin/env python3
"""Run the frozen Phase1084 behavior gate without hidden-state access."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1084_two_entity_attribute_protocol as protocol

# Reuse the audited batched FP16 behavior engine with Phase1084-owned cases,
# paths, thresholds, and digests.
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
