#!/usr/bin/env python3
"""Independent audit for Phase1521."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1521_c089_natural_relation_observation_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1521_c089_natural_relation_observation_contract as contract
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    groups = core.rows(OUT / "material/relation_composition_sets.jsonl")
    cases = core.rows(OUT / "material/active_cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    tok = tokenizer()
    exact = sum(core.chat_ids(tok, contract.SYSTEM, case["prompt"]) == row["prompt_ids"] for case, row in zip(cases, compiled))
    words = [word for group in groups for pair in (group["pair_a"], group["pair_b"]) for word in (pair["source"], pair["target"])]
    py_compile.compile(str(TESTS / "phase1521_c089_natural_relation_observation_contract.py"), doraise=True)
    checks = {
        "status": final["status"] == "natural_relation_observation_contract_frozen",
        "counts": len(groups) == 45 and len(cases) == len(compiled) == 360,
        "balances": Counter(row["family"] for row in cases) == {family: 120 for family in contract.FAMILIES} and Counter(row["partition"] for row in cases) == {partition: 120 for partition in contract.PARTITIONS},
        "truth": Counter(row["gold_label"] for row in cases) == {"yes": 180, "no": 180},
        "lexical_disjoint": len(words) == len(set(words)) == 180,
        "exact_chat_compile": exact == 360,
        "span_and_output": all(all(1 <= len(row["role_positions"][role]) <= 4 for role in contract.ROLES) and all(len(ids) == 1 for ids in row["candidate_ids"]) for row in compiled),
        "hashes": core.sha(OUT / "material/relation_composition_sets.jsonl") == protocol["material"]["groups_sha256"] and core.sha(OUT / "compiled/qwen3_active.jsonl") == protocol["material"]["compiled_sha256"],
        "zero_models": all(value == 0.5 for value in audit["zero_models"].values()),
        "complete_input": "system plus user" in audit["complete_input_scope"],
        "scope": "universal relation vector" in protocol["claim_boundary"]["forbidden"],
        "authorization": final["authorization"] == "run_phase1522_c089_unified_forward_capture",
        "no_hidden": audit["checks"]["hidden_not_accessed"],
    }
    result = {"phase": 1521, "campaign": "C089", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
