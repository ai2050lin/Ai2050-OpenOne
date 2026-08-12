#!/usr/bin/env python3
"""Run Phase1097 end to end with exactly one FP16 model resident at a time."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1097_conditional_transition_protocol as protocol


def call(script: str, *args: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, str(TEST_ROOT / script), *args]
    print({"phase": protocol.PHASE, "command": command}, flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    call("phase1097_conditional_transition_protocol.py")
    for model_name in protocol.MODELS:
        output = protocol.OUT_ROOT / "behavior" / model_name / "summary.json"
        if args.force or not output.exists():
            call("phase1097_conditional_transition_behavior.py", model_name)
    call("phase1097_conditional_transition_behavior_finalize.py")
    authorization = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    if not authorization["hidden_scan_authorized"]:
        raise SystemExit("behavior gate stopped hidden-state scan")
    for model_name in protocol.MODELS:
        output = protocol.OUT_ROOT / "atlas" / model_name / "summary.json"
        if args.force or not output.exists():
            call("phase1097_conditional_transition_scan.py", model_name)
    call("phase1097_conditional_transition_finalize.py")
    call("phase1097_conditional_transition_diagnostic.py")
    call("phase1097_result_audit.py")


if __name__ == "__main__":
    main()
