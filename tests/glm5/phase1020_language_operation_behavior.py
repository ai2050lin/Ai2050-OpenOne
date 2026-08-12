#!/usr/bin/env python3
"""Run Phase1020 behavior qualification for all language-pattern families."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1018_language_pattern_behavior as engine
import phase1020_language_operation_protocol as protocol


def configure_engine() -> None:
    engine.FACTORIAL_STATES = protocol.FACTORIAL_STATES
    engine.FAMILIES = protocol.FAMILIES
    engine.MODELS = protocol.MODELS
    engine.OUT_ROOT = protocol.OUT_ROOT
    engine.PHASE = protocol.PHASE
    engine.PROMPT_MODES = protocol.PROMPT_MODES
    engine.PROTOCOL_REVISION = protocol.PROTOCOL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    configure_engine()
    engine.run_model(args.model)


if __name__ == "__main__":
    main()
