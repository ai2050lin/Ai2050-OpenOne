#!/usr/bin/env python3
"""Independent audit for Phase1504."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1504_c087_cross_root_semeval_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json")
    cases = core.rows(OUT / "material/active_cases.jsonl")
    selected = core.rows(OUT / "material/selected_instances.jsonl")
    groups = core.rows(OUT / "material/composition_sets.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    pre = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    py_compile.compile(str(TESTS / "phase1504_c087_cross_root_semeval_contract.py"), doraise=True)
    candidate_positive = Counter(row["positive_candidate"] for row in selected)
    candidate_negative = Counter(row["negative_candidate"] for row in selected)
    checks = {
        "counts": len(selected) == 216 and len(cases) == 864 and len(groups) == 216 and len(compiled) == 864,
        "partitions": Counter(case["partition"] for case in cases) == {p: 288 for p in protocol["partitions"]},
        "item_disjoint": all(len({r["partition"] for r in cases if r["item"] == item}) == 1 for item in {r["item"] for r in cases}),
        "semantic": all(r["human_votes_here"] >= 2 if r["semantic_match"] else r["human_votes_here"] == 0 for r in cases),
        "cross_root": all(r["lexical_trigram_overlap"] == 0.0 for r in cases),
        "candidate_balance": candidate_positive == candidate_negative,
        "roles": all(all(len(row["role_positions"][role]) == 1 for role in protocol["roles"]) for row in compiled),
        "hashes": core.sha(OUT / "material/active_cases.jsonl") == protocol["material"]["active_sha256"] and core.sha(OUT / "compiled/qwen3_active.jsonl") == protocol["material"]["compiled_sha256"],
        "preaudit": pre["all_checks_passed"],
        "contract": protocol["contract_sha256"] == core.digest({k: v for k, v in protocol.items() if k not in ("contract_sha256", "authorization")}),
    }
    result = {"phase": 1504, "campaign": "C087", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
