#!/usr/bin/env python3
"""Run Phase1103 sequentially to keep only one FP16 model resident."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1103_natural_relation_route_protocol as protocol


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(TEST_ROOT / script), *args]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run("phase1103_natural_relation_route_protocol.py")
    for model in protocol.MODELS:
        run("phase1103_natural_relation_route_behavior.py", model)
    run("phase1103_natural_relation_route_behavior_finalize.py")
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if authorization["causal_scan_authorized"]:
        for model in protocol.MODELS:
            run("phase1103_natural_relation_route_causal_scan.py", model)
    run("phase1103_natural_relation_route_causal_finalize.py")
    run("phase1103_natural_relation_route_finalize.py")
    run("phase1103_natural_relation_route_result_audit.py")


if __name__ == "__main__":
    main()
