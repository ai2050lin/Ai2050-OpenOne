#!/usr/bin/env python3
"""Independent audit for Phase1536."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1536_c091_human_validated_chinese_relation_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    pairs = core.rows(OUT / "material/frozen_pairs.jsonl")
    cases = core.rows(OUT / "material/active_cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    pair_counts = Counter((row["partition"], row["family"], row["concreteness"]) for row in pairs)
    checks = {
        "hash_pairs": core.sha(OUT / "material/frozen_pairs.jsonl") == protocol["material"]["pairs_sha256"],
        "hash_cases": core.sha(OUT / "material/active_cases.jsonl") == protocol["material"]["cases_sha256"],
        "hash_compiled": core.sha(OUT / "compiled/qwen3_active.jsonl") == protocol["material"]["compiled_sha256"],
        "counts": len(pairs) == 90 and len(cases) == 540 and len(compiled) == 540,
        "balance": len(pair_counts) == 18 and set(pair_counts.values()) == {5},
        "lexical_nonreuse": len({word for row in pairs for word in (row["source"], row["target"])}) == 180,
        "queries": all(sum(row["pair_id"] == pair["pair_id"] for row in cases) == 6 for pair in pairs),
        "outputs": all(all(len(ids) == 1 for ids in row["candidate_ids"]) for row in compiled),
        "fixed_length": max(map(lambda row: len(row["prompt_ids"]), compiled)) == protocol["execution"]["fixed_global_sequence_length"],
        "behavior_before_hidden": protocol["sequence"][0] == "behavior-only authoritative run",
        "route_retirement": "retire only" in protocol["behavior_gate"]["route_policy"],
        "forbidden_methods": all(value in protocol["forbidden"] for value in ("attention", "MLP", "PCA", "learned probes")),
        "authorization": protocol["authorization"] == "run_phase1537_c091_behavior_only_qualification",
    }
    result = {
        "phase": 1536,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
