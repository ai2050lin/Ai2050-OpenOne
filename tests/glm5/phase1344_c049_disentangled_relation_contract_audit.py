#!/usr/bin/env python3
"""Independent audit for Phase1344 C049 preregistration."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1344_c049_disentangled_relation_contract"
MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("discovery", "confirmation", "holdout")
FAMILIES = ("cheese", "pasta", "disease", "profession")


def main():
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    semantic = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    concepts = core.load(OUT / "material/frozen_concept_graph.json")["concepts"]
    rows = core.rows(OUT / "material/frozen_factorial_cases.jsonl")
    groups = defaultdict(list)
    for row in rows:
        groups[row["quartet_key"]].append(row)
    recompute = dict(protocol)
    authorization = recompute.pop("authorization")
    stored_digest = recompute.pop("contract_sha256")
    checks = {
        "authorization": authorization == "run_phase1345_c049_disentangled_behavior",
        "digest": stored_digest == core.digest(recompute) == final["contract_sha256"],
        "material_hashes": protocol["material"]["graph_sha256"]
        == core.sha(OUT / "material/frozen_concept_graph.json")
        and protocol["material"]["cases_sha256"] == core.sha(OUT / "material/frozen_factorial_cases.jsonl"),
        "semantic_audit": semantic["all_checks_passed"] and semantic["independent_human_blind_review"].startswith("not available"),
        "concepts": len(concepts) == 48 and len({x["word"] for x in concepts}) == 48,
        "cases": len(rows) == 1728 and Counter(x["truth"] for x in rows) == {True: 864, False: 864},
        "quartets": len(groups) == 432
        and all([x["cell"] for x in q] == ["aa", "ab", "ba", "bb"] for q in groups.values()),
        "pairing": all(
            sum(
                x["partition"] == p
                and x["family_pair"] == f"{a}__{b}"
                and x["pair_offset"] == o
                for x in rows
            )
            == 48
            for p in PARTITIONS
            for a, b in combinations(FAMILIES, 2)
            for o in (0, 1)
        ),
        "ledger_separation": protocol["behavior_ledgers"]["quartet_joint_reliability_report"]["authorization_effect"]
        == "reported independently and does not gate the interaction-field branch",
        "single_model_branch": "independently" in protocol["field_gate"]["single_model_authorization"]
        and protocol["field_gate"]["cross_model_minimum"] == 2,
        "full_dimensional": "no PCA" in protocol["field_gate"]["storage"],
        "stopping": "do not change" in protocol["stop_rule"],
        "compiled": all(
            len(core.rows(OUT / f"compiled/{model}_factorial.jsonl")) == len(rows) for model in MODELS
        ),
    }
    result = {
        "phase": 1344,
        "campaign": "C049",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
