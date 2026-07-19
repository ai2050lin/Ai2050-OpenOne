#!/usr/bin/env python3
"""Run the frozen Phase568 explicit-relation behavior contract on one model."""

from __future__ import annotations

from pathlib import Path

import phase567_multi_relation_binding_behavior as runner


ROOT = Path(__file__).resolve().parents[2]

runner.PHASE_ID = "Phase568"
runner.SCHEMA_PREFIX = "phase568"
runner.OUT_DIR = ROOT / "tests/gpt5/result/phase568_explicit_relation_binding"
runner.CASES_PATH = runner.OUT_DIR / "phase568_open_cases.jsonl"
runner.PROTOCOL_PATH = runner.OUT_DIR / "phase568_frozen_protocol.json"
runner.AUDIT_PATH = runner.OUT_DIR / "phase568_static_audit.json"
runner.EXPECTED_MODEL_ROWS = 22464


if __name__ == "__main__":
    runner.main()
