#!/usr/bin/env python3
"""Collect Phase1094 signed fields, one unquantized FP16 model at a time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1094_semantic_topology_protocol as protocol

sys.modules["phase1086_signed_shared_field_protocol"] = protocol
import phase1086_signed_shared_field_scan as engine


def run(model_name: str) -> None:
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if not authorization["hidden_scan_authorized"]:
        raise RuntimeError("Phase1094 hidden-state scan was not authorized")
    engine.protocol = protocol
    engine.run(model_name)
    path = protocol.OUT_ROOT / "atlas" / model_name / "summary.json"
    result = protocol.read_json(path)
    result.pop("summary_digest", None)
    result["schema_version"] = "phase1094_model_signed_summary.v1"
    result["behavior_status"] = (
        "authorized" if model_name in authorization["authorized_models"] else "exploratory"
    )
    result["summary_digest"] = protocol.digest(result)
    protocol.write_json(path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
