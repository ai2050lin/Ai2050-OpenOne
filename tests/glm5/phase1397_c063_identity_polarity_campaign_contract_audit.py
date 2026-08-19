#!/usr/bin/env python3
"""Independent audit for Phase1397."""
from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1397_c063_identity_polarity_campaign_contract"
SCRIPT = TESTS / "phase1397_c063_identity_polarity_campaign_contract.py"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    pre = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    active = core.rows(OUT / "material/active_cases.jsonl")
    status = core.rows(OUT / "material/status_cases.jsonl")
    factors = core.rows(OUT / "material/factor_sets.jsonl")
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    checks = {
        "preaudit_passed": pre["all_checks_passed"] and pre["passed"] == pre["total"],
        "contract_frozen": protocol["authorization"] == "run_phase1398_c063_factorized_behavior" and bool(protocol["contract_sha256"]),
        "material_hashes": protocol["material"]["active_sha256"] == core.sha(OUT / "material/active_cases.jsonl") and protocol["material"]["status_sha256"] == core.sha(OUT / "material/status_cases.jsonl") and protocol["material"]["factor_sha256"] == core.sha(OUT / "material/factor_sets.jsonl"),
        "active_answer_balance": len(active) == 1728 and Counter(r["gold_position"] for r in active) == {0: 432, 1: 432, 2: 432, 3: 432},
        "status_answer_balance": len(status) == 576 and Counter(r["gold_position"] for r in status) == {0: 144, 1: 144, 2: 144, 3: 144},
        "factor_count_partitioned": len(factors) == 288 and Counter(r["partition"] for r in factors) == {"response_discovery": 96, "confirmation": 96, "lockbox": 96},
        "factor_axes_distinct": all(len({r[k] for k in ("recipient", "family_only", "answer_only", "family_and_answer", "polarity_only", "family_and_polarity", "status_null")}) == 7 for r in factors),
        "scope_enforced": all(v in protocol["forbidden"] for v in ("attention", "MLP", "gradient", "PCA", "learned probe")),
        "no_model_import": "load_bf16" not in source and "AutoModel" not in source,
        "human_lock_disclosed": protocol["material"]["human_naturalness_lock"] is False and pre["independent_human_blind_review"] is False,
    }
    result = {"phase": 1397, "campaign": "C063", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError({k: v for k, v in checks.items() if not v})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
