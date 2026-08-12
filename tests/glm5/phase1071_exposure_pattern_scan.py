#!/usr/bin/env python3
"""Run the Phase1071 exposure-aware atlas with the audited Phase1070 engine.

The tensor engine is intentionally reused rather than forked: Phase1071
changes the frozen cases, capture roles, exposure controls, and final gates,
while retaining the already-audited 16-state arithmetic and FP16 loader.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1070_process_answer_scan as engine
import phase1071_exposure_pattern_protocol as protocol


engine.protocol = protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    engine.run(args.model)


if __name__ == "__main__":
    main()
