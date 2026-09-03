#!/usr/bin/env python3
"""Independent material and contract audit for Phase1557."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
C091 = RESULT / "phase1536_c091_human_validated_chinese_relation_contract"
OUT = RESULT / "phase1557_c096_fresh_human_relation_field_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    py_compile.compile(str(TESTS / "phase1557_c096_fresh_human_relation_field_contract.py"), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    pairs = core.rows(OUT / "material/frozen_fresh_pairs.jsonl")
    cases = core.rows(OUT / "material/active_cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    old_pairs = core.rows(C091 / "material/frozen_pairs.jsonl")
    old_words = {word for row in old_pairs for word in (row["source"], row["target"])}
    new_words = [word for row in pairs for word in (row["source"], row["target"])]
    unsigned = {key: value for key, value in protocol.items() if key not in {"contract_sha256", "authorization"}}
    counts = Counter((row["partition"], row["family"], row["concreteness"]) for row in pairs)
    checks = {
        "digest": protocol["contract_sha256"] == core.digest(unsigned),
        "hashes": core.sha(OUT / "material/frozen_fresh_pairs.jsonl") == protocol["material"]["pairs_sha256"] and core.sha(OUT / "material/active_cases.jsonl") == protocol["material"]["cases_sha256"] and core.sha(OUT / "compiled/qwen3_active.jsonl") == protocol["material"]["compiled_sha256"],
        "coverage": len(pairs) == 90 and len(cases) == len(compiled) == 540,
        "balance": len(counts) == 18 and set(counts.values()) == {5},
        "lexical_independence": not (set(new_words) & old_words) and len(new_words) == len(set(new_words)) == 180,
        "roles": all(all(row["role_positions"][role] for role in protocol["roles"]) for row in compiled),
        "single_token": all(all(len(ids) == 1 for ids in row["candidate_ids"]) for row in compiled),
        "predictions": len(protocol["frozen_predictions"]) == 5,
        "layered_policy": "does not erase" in protocol["behavior_typing"]["policy"],
        "numeric_gate": protocol["numeric_integrity_gate"]["failure_action"].startswith("stop capture"),
        "forbidden": {"attention", "MLP", "PCA", "learned probe", "post-reveal threshold mutation"}.issubset(set(protocol["forbidden"])),
        "authorization": final["authorization"] == "run_phase1558_c096_unified_behavior_and_all_state_capture",
    }
    result = {"phase": 1557, "campaign": "C096", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
