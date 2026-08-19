#!/usr/bin/env python3
"""Independent audit for Phase1412."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1412_c067_paired_state_composition_contract"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    pre = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    active = core.rows(OUT / "material/active_cases.jsonl")
    composition = core.rows(OUT / "material/composition_sets.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    checks = {
        "preaudit": pre["all_checks_passed"],
        "active": len(active) == 2160 and Counter(r["truth"] for r in active) == {True: 1080, False: 1080},
        "composition": len(composition) == 72 and Counter(r["partition"] for r in composition) == {p: 24 for p in protocol["material"]["partitions"]},
        "compiled": len(compiled) == 2160,
        "hashes": protocol["material"]["active_sha256"] == core.sha(OUT / "material/active_cases.jsonl") and protocol["material"]["composition_sha256"] == core.sha(OUT / "material/composition_sets.jsonl"),
        "state16_catalog": protocol["mechanism"]["state_index"] == 16 and protocol["mechanism"]["surface"] == "catalog",
        "paired_arms": len(protocol["mechanism"]["arms"]) == 9 and "mismatched_dual_gh" in protocol["mechanism"]["arms"],
        "no_search": "layer search" in protocol["forbidden"] and "candidate search" in protocol["forbidden"],
        "hidden_not_accessed": pre["checks"]["hidden_not_accessed"],
        "authorization": final["authorization"] == "run_phase1413_c067_behavior",
    }
    result = {"phase": 1412, "campaign": "C067", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
