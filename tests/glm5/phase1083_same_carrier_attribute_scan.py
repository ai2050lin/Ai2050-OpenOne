#!/usr/bin/env python3
"""Run Phase1083 with the audited Phase1082 response-field engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1083_same_carrier_attribute_protocol as protocol

# The engine imports its protocol by the historical module name.  Substituting
# the already-frozen Phase1083 module reuses only measurement mechanics; all
# cases, roles, thresholds, output paths, and digests remain Phase1083-owned.
sys.modules["phase1082_semantic_output_operation_world_protocol"] = protocol

import phase1082_semantic_output_operation_world_scan as engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    engine.run(args.model)


if __name__ == "__main__":
    main()
