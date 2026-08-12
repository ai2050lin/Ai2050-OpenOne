#!/usr/bin/env python3
"""Freeze Phase1093 behavior authorization before hidden-state scans."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1092_natural_bilingual_attribute_behavior_finalize as engine
import phase1093_independent_relation_protocol as protocol


def main() -> None:
    engine.protocol = protocol
    engine.main()
    path = protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    result = protocol.read_json(path)
    result.pop("summary_digest", None)
    result["schema_version"] = "phase1093_behavior_authorization.v1"
    if result["hidden_scan_authorized"]:
        result["decision"] = "run_phase1093_signed_hidden_scan"
    result["summary_digest"] = protocol.digest(result)
    protocol.write_json(path, result)


if __name__ == "__main__":
    main()
