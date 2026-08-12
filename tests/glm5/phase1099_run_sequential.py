#!/usr/bin/env python3
"""Run Phase1099 with one FP16 model resident at a time."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1099_relation_family_atlas_protocol as protocol


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(TEST_ROOT / script), *args]
    print({"command": command}, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run("phase1099_relation_family_atlas_protocol.py")
    for model_name in protocol.MODELS:
        run("phase1099_relation_family_atlas_behavior.py", model_name)
    run("phase1099_relation_family_atlas_behavior_finalize.py")
    authorization = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    if not authorization["hidden_scan_authorized"]:
        print({"phase": protocol.PHASE, "decision": authorization["decision"]}, flush=True)
        return
    for model_name in protocol.MODELS:
        run("phase1099_relation_family_atlas_scan.py", model_name)
    run("phase1099_relation_family_atlas_finalize.py")
    run("phase1099_relation_family_atlas_result_audit.py")


if __name__ == "__main__":
    main()
