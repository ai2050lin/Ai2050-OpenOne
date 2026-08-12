#!/usr/bin/env python3
"""Finalize the Phase1083 descriptive atlas with the frozen Phase1083 protocol."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1083_same_carrier_attribute_protocol as protocol

sys.modules["phase1082_semantic_output_operation_world_protocol"] = protocol

import phase1082_semantic_output_operation_world_finalize as engine


if __name__ == "__main__":
    engine.main()
