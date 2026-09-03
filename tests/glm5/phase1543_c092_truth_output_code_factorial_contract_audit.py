#!/usr/bin/env python3
"""Independent audit for Phase1543."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1543_c092_truth_output_code_factorial_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/pre_model_audit.json")
    pairs = core.rows(OUT / "material/frozen_pairs.jsonl")
    cases = core.rows(OUT / "material/active_cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    truth_counts = Counter((row["partition"], row["concreteness"], row["semantic_truth"]) for row in pairs)
    checks = {
        "pre_model_audit": audit["all_checks_passed"],
        "hashes": protocol["files"] == {
            "pairs_sha256": core.sha(OUT / "material/frozen_pairs.jsonl"),
            "cases_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl"),
        },
        "coverage": len(pairs) == 60 and len(cases) == len(compiled) == 240,
        "truth_balance": len(truth_counts) == 12 and set(truth_counts.values()) == {5},
        "answer_identity": all(row["answer_sign"] == row["truth_sign"] * row["codebook_sign"] for row in cases),
        "case_ids": [row["case_id"] for row in cases] == [row["case_id"] for row in compiled],
        "roles": all(set(row["role_positions"]) == set(protocol["roles"]) for row in compiled),
        "single_token": all(all(len(ids) == 1 for ids in row["candidate_ids"]) for row in compiled),
        "behavior_before_hidden": protocol["sequence"][0] == "behavior-only run with hidden disabled",
        "factorial_frozen": protocol["factorial_model"] == "H=S+y*T+c*C+(y*c)*A+epsilon",
        "scope": {"attention", "MLP", "PCA", "learned probes"}.issubset(protocol["forbidden"]),
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "run_phase1544_c092_behavior_only_qualification",
    }
    result = {"phase": 1543, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "checks": checks}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
