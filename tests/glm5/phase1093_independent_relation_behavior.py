#!/usr/bin/env python3
"""Run Phase1093 behavior calibration, one FP16 model at a time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1092_natural_bilingual_attribute_behavior as engine
import phase1093_independent_relation_protocol as protocol


def run(model_name: str) -> None:
    engine.protocol = protocol
    engine.run(model_name)
    path = protocol.OUT_ROOT / "pilot" / f"{model_name}.json"
    result = protocol.read_json(path)
    result.pop("result_digest", None)
    result["schema_version"] = "phase1093_behavior_result.v1"
    result["result_digest"] = protocol.digest(result)
    protocol.write_json(path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
