#!/usr/bin/env python3
"""Run the Phase1019 full-component scan with corrected protocol data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1018_language_pattern_scan as engine
import phase1019_corrected_pattern_protocol as protocol


def configure_engine() -> None:
    engine.CAPTURE_ROLES = protocol.CAPTURE_ROLES
    engine.FAMILIES = protocol.FAMILIES
    engine.MODELS = protocol.MODELS
    engine.OUT_ROOT = protocol.OUT_ROOT
    engine.PHASE = protocol.PHASE
    engine.PROTOCOL_REVISION = protocol.PROTOCOL_REVISION
    engine.STATES = protocol.STATES
    engine.STATE_INDEX = {
        state: index for index, state in enumerate(protocol.STATES)
    }
    engine.ROLE_INDEX = {
        role: index for index, role in enumerate(protocol.CAPTURE_ROLES)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    configure_engine()
    engine.run_model(args.model, resume=args.resume)


if __name__ == "__main__":
    main()
