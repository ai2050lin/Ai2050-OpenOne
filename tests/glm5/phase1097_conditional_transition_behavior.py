#!/usr/bin/env python3
"""Run Phase1097 behavior gates, one local FP16 model at a time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1096_comparison_dynamics_behavior as shared
import phase1097_conditional_transition_protocol as protocol


def run(model_name: str) -> None:
    shared.protocol = protocol
    shared.run(model_name)
    path = protocol.OUT_ROOT / "behavior" / model_name / "summary.json"
    summary = protocol.read_json(path)
    summary["schema_version"] = "phase1097_behavior_summary.v1"
    summary.pop("summary_digest", None)
    summary["summary_digest"] = protocol.digest(summary)
    protocol.write_json(path, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
