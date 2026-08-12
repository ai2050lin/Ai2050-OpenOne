#!/usr/bin/env python3
"""Run Phase1100 with one audited FP16 model resident at a time."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1100_relation_graph_inheritance_protocol as protocol


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(TEST_ROOT / script), *args]
    print({"command": command}, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run("phase1100_relation_graph_inheritance_protocol.py")
    for model_name in protocol.MODELS:
        run("phase1100_relation_graph_inheritance_source.py", model_name)
    run("phase1100_relation_graph_inheritance_finalize.py")
    run("phase1100_relation_graph_inheritance_failure_diagnostic.py")
    run("phase1100_relation_graph_inheritance_result_audit.py")


if __name__ == "__main__":
    main()
