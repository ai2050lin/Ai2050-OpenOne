#!/usr/bin/env python3
"""Independent audit for Phase1408."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1408_c066_midstate_breadth_contract"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    pre = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    active = core.rows(OUT / "material/active_cases.jsonl")
    factors = core.rows(OUT / "material/factor_sets.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    checks = {
        "preaudit": pre["all_checks_passed"],
        "active": len(active) == 2160 and Counter(r["truth"] for r in active) == {True: 1080, False: 1080},
        "factors": len(factors) == 216 and Counter(r["partition"] for r in factors) == {p: 72 for p in protocol["material"]["partitions"]},
        "compiled": len(compiled) == 2160,
        "hashes": protocol["material"]["active_sha256"] == core.sha(OUT / "material/active_cases.jsonl") and protocol["material"]["factor_sha256"] == core.sha(OUT / "material/factor_sets.jsonl"),
        "state16_only": protocol["mechanism"]["state_index"] == 16 and len(protocol["mechanism"]["candidates"]) == 6,
        "no_discovery_search": "new candidate search" in protocol["forbidden"],
        "hidden_not_accessed": pre["checks"]["hidden_not_accessed"],
        "authorization": final["authorization"] == "run_phase1409_c066_behavior",
    }
    result = {"phase": 1408, "campaign": "C066", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
