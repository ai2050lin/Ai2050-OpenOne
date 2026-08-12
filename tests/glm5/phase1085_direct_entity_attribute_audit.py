#!/usr/bin/env python3
"""Audit Phase1085 by reusing the Phase1084 structural audit."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1085_direct_entity_attribute_protocol as protocol

sys.modules["phase1084_two_entity_attribute_protocol"] = protocol
import phase1084_two_entity_attribute_audit as engine


if __name__ == "__main__":
    engine.protocol = protocol
    engine.main()
