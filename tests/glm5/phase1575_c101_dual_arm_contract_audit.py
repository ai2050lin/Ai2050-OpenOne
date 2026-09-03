#!/usr/bin/env python3
"""Independent pre-model audit for Phase1575 / C101."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1575_c101_dual_arm"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1575_c101_dual_arm_contract.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/pre_model_material_semantic_zero_audit.json")
    conf = core.rows(OUT / "material/confirmation_cases.jsonl")
    breadth = core.rows(OUT / "material/breadth_cases.jsonl")
    checks = {
        "producer": protocol["producer_sha256"] == core.sha(producer),
        "audit": audit["all_checks_passed"] and audit["passed"] == audit["total"],
        "material_digest": protocol["material_digest"] == core.digest({"confirmation": conf, "breadth": breadth}),
        "counts": len(conf) == 1152 and len(breadth) == 768,
        "partitions": Counter(row["partition"] for row in conf) == {p: 384 for p in ("response_discovery", "confirmation", "lockbox")} and Counter(row["partition"] for row in breadth) == {p: 256 for p in ("response_discovery", "confirmation", "lockbox")},
        "truth_path": all((row["path_count"] == 1) == row["truth"] for row in conf),
        "output": all(row["output_yes"] == (row["truth"] == (row["code"] == 1)) for row in [*conf, *breadth]),
        "primary_frozen": protocol["confirmation"]["primary"] == {"state": 24, "role": "boundary", "effect": "xy", "threshold": 0.5, "required": "24/24 C100-discovery to C101 confirmation/lockbox full-vector cosines"},
        "raw_scope": "all 37 states" in protocol["storage"]["raw"] and "all 2560" in protocol["storage"]["raw"],
        "authorization": protocol["authorization"] == "run_phase1576_c101_qwen_capture",
    }
    result = {"phase": 1575, "campaign": "C101", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_pre_model_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
