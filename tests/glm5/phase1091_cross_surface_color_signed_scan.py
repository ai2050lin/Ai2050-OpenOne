#!/usr/bin/env python3
"""Collect Phase1091 English/Chinese cross-surface signed fields."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1091_cross_surface_color_signed_protocol as protocol

sys.modules["phase1086_signed_shared_field_protocol"] = protocol
import phase1086_signed_shared_field_scan as engine


def run(model_name: str) -> None:
    engine.protocol = protocol
    engine.run(model_name)
    path = protocol.OUT_ROOT / "atlas" / model_name / "summary.json"
    result = protocol.read_json(path)
    result.pop("summary_digest", None)
    result["schema_version"] = "phase1091_model_signed_summary.v1"
    result["summary_digest"] = protocol.digest(result)
    protocol.write_json(path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
