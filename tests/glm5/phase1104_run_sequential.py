#!/usr/bin/env python3
"""Run Phase1104 sequentially with at most one FP16 model resident."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1104_lexical_address_execution_protocol as protocol


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(TEST_ROOT / script), *args]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run("phase1104_lexical_address_execution_protocol.py")
    for model in protocol.MODELS:
        run("phase1104_lexical_address_execution_behavior.py", model)
    run("phase1104_lexical_address_execution_behavior_finalize.py")
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    for model in protocol.MODELS:
        run("phase1104_lexical_address_execution_causal_scan.py", model)
    run("phase1104_lexical_address_execution_causal_finalize.py")
    run("phase1104_lexical_address_execution_finalize.py")
    run("phase1104_lexical_address_execution_result_audit.py")


if __name__ == "__main__":
    main()
