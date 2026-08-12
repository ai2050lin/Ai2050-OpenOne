#!/usr/bin/env python3
"""Run the Phase1085 preregistered middle-band scan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1085_direct_entity_attribute_protocol as protocol

sys.modules["phase1084_two_entity_attribute_protocol"] = protocol
import phase1084_two_entity_attribute_scan as targeted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    targeted.protocol = protocol
    targeted.engine.protocol = protocol
    targeted.engine.event_definitions = targeted.targeted_event_definitions
    targeted.engine.RoleCapture = targeted.MiddleBandRoleCapture
    targeted.engine.run(args.model)


if __name__ == "__main__":
    main()
