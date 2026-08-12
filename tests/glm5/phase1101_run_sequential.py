#!/usr/bin/env python3
"""Run Phase1101 with one FP16/no-quantization model resident at a time."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1101_relation_identity_routing_protocol as protocol


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(TEST_ROOT / script), *args]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run("phase1101_relation_identity_routing_protocol.py")
    for model in protocol.MODELS:
        run("phase1101_relation_identity_routing_behavior.py", model)
    run("phase1101_relation_identity_routing_behavior_finalize.py")
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if not authorization["hidden_scan_authorized"]:
        print(json.dumps({
            "phase": protocol.PHASE,
            "decision": authorization["decision"],
            "automatic_next": False,
        }), flush=True)
        return
    for model in protocol.MODELS:
        run("phase1101_relation_identity_routing_scan.py", model)
    run("phase1101_relation_identity_routing_finalize.py")
    run("phase1101_relation_identity_routing_diagnostic.py")
    run("phase1101_relation_identity_routing_result_audit.py")
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    print(json.dumps({
        "phase": protocol.PHASE,
        "gates": final["gates"],
        "decision": final["decision"],
        "automatic_next_required": final["automatic_next_required"],
    }), flush=True)


if __name__ == "__main__":
    main()
