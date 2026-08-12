#!/usr/bin/env python3
"""Run Phase1089 active and truth-matched-null behavior, one model at a time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1089_truth_matched_color_binding_protocol as protocol

sys.modules["phase1083_same_carrier_attribute_protocol"] = protocol
import phase1083_same_carrier_attribute_behavior as engine


def run(model_name: str) -> None:
    # Phase1089 must behavior-check both panels; the generic runner checks
    # active only unless this selector is replaced.
    engine.protocol = protocol
    engine.candidate_rows = lambda rows: list(rows)
    engine.run(model_name)
    path = protocol.OUT_ROOT / "pilot" / f"{model_name}.json"
    result = protocol.read_json(path)
    result.pop("result_digest", None)
    result["schema_version"] = "phase1089_behavior_gate.v1"
    result["candidate_scope"] = "active_and_truth_matched_null"
    result["result_digest"] = protocol.digest(result)
    protocol.write_json(path, result)
    print({
        "phase": protocol.PHASE,
        "model": model_name,
        "corrected_result_digest": result["result_digest"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
