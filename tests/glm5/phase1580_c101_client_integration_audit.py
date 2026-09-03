#!/usr/bin/env python3
"""Independent audit of Phase1580 C101 client integration evidence."""
from __future__ import annotations

import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1575_c101_dual_arm"


def main() -> None:
    producer = TESTS / "phase1580_c101_client_integration.py"
    py_compile.compile(str(producer), doraise=True)
    report = json.loads((OUT / "analysis/client_integration.json").read_text(encoding="utf-8"))
    checks = {
        "producer": report["all_checks_passed"] and report["passed"] == report["total"] == 9,
        "asset_identity": len(set(report["hashes"].values())) == 1,
        "asset_size": report["built_asset_bytes"] > 10_000_000,
        "scope": report["next_authorization"]["status"] == "authorized_for_separate_preregistration",
        "sequential_models": "sequential GLM4 then DeepSeek-7B" in report["next_authorization"]["scope"],
        "coordinate_boundary": "do not compare physical coordinate numbers across model families" in report["next_authorization"]["constraints"],
    }
    result = {
        "phase": 1580,
        "campaign": "C101",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    target = OUT / "audit/independent_client_integration_audit.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
