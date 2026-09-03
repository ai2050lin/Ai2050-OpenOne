#!/usr/bin/env python3
"""Independent audit for C214."""
from __future__ import annotations

import json
from pathlib import Path

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C214


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = final["headline"]
    checks = {"final": final["all_checks_passed"], "three_models": set(report["models"]) == {"qwen3", "glm4", "deepseek7b"}, "sequential": protocol["sequential_loading"], "permutation_null": all(row["role_permutations"] == 720 for row in report["pair_tests"]), "no_same_coordinates": "never compared" in protocol["topology"], "producer_hash": core.sha(Path(__file__).with_name("phase1748_c214_cross_model_functional_isomorphism.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1748, "campaign": "C214", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
