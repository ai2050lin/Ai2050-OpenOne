#!/usr/bin/env python3
"""Run the behavior-only Phase1102 replication sequentially."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1102_relation_identity_routing_replication_protocol as protocol


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(TEST_ROOT / script), *args]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run("phase1102_relation_identity_routing_replication_protocol.py")
    for model in protocol.MODELS:
        run("phase1102_relation_identity_routing_replication_behavior.py", model)
    run("phase1102_relation_identity_routing_replication_behavior_finalize.py")
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if authorization["hidden_scan_authorized"]:
        raise RuntimeError(
            "This frozen behavior-only replication would require a separately preregistered hidden-scan runner."
        )
    run("phase1102_relation_identity_routing_replication_finalize.py")
    run("phase1102_relation_identity_routing_replication_diagnostic.py")
    run("phase1102_relation_identity_routing_replication_result_audit.py")


if __name__ == "__main__":
    main()
