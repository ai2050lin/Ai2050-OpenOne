#!/usr/bin/env python3
"""C321: freeze five naturalized surfaces and audit shortcuts before model use."""
from __future__ import annotations

from collections import Counter

import phase1844_c310_c335_dual_axis_common as common


def main() -> None:
    parent = common.core.load(common.OUTS["C320"] / "analysis/final.json")
    rows = common.natural_material()
    strata = Counter((row["family"], row["surface"], row["partition"], row["gold_position"]) for row in rows)
    checks = {
        "parent": parent["all_checks_passed"],
        "cases": len(rows) == 1920,
        "unique_ids": len({row["case_id"] for row in rows}) == 1920,
        "six_families": len({row["family"] for row in rows}) == 6,
        "five_surfaces": tuple(sorted({row["surface"] for row in rows})) == tuple(sorted(common.NATURAL_SURFACES)),
        "exact_candidate_balance_every_stratum": all(strata[(family, surface, partition, 0)] == strata[(family, surface, partition, 1)] == 16 for family in common.FAMILIES for surface in common.NATURAL_SURFACES for partition in ("discovery", "confirmation")),
        "semantic_graph_present": all(row["semantic_graph"]["material"] == "natural_five_surface" for row in rows),
    }
    protocol = {
        "status": "five_surface_material_frozen",
        "cases": len(rows),
        "factorization": "6 families x 5 surfaces x 8 units x 4 factorial cells x 2 answer orders",
        "partitions": "units 0-3 discovery; units 4-7 confirmation",
        "surfaces": list(common.NATURAL_SURFACES),
        "machine_naturalness_audit": ["complete grammatical sentences", "same two facts and question preserved", "candidate balance exact", "role strings present before tokenization"],
        "human_naturalness_status": "registered_no_test_external_dependency",
        "human_protocol": "At least three independent English-proficient raters, blind to factors and expected results, must score grammaticality and semantic equivalence before any future claim of human-validated naturalness.",
        "behavior_gate": {"global_min": 0.90, "family_min": 0.85, "surface_min": 0.80},
        "claim_boundary": "These are five controlled naturalized wrappers, not open-domain paraphrases. Machine checks cannot substitute for real human blind ratings.",
    }
    out = common.prepare("C321", protocol, checks)
    common.core.write_rows(out / "material/cases.jsonl", rows)
    common.core.save(out / "protocol/external_human_blind_review.json", {"status": "no_test", "reason": "No independent human raters were available in this automated run.", "required_raters": 3, "items": 30, "dimensions": ["grammaticality", "semantic_equivalence", "answer_uniqueness"]})
    headline = {"status": "five_surface_material_closed", "cases": len(rows), "strata": len(strata), "candidate_balance": "exact within family x surface x partition", "human_naturalness": "no_test", "strict_interpretation": protocol["claim_boundary"]}
    common.close("C321", headline, {"all_contract_checks": all(checks.values()), "material_rows": len(common.core.rows(out / "material/cases.jsonl")) == 1920, "human_no_test_honest": True}, "C322_qwen_behavior_qualification")


if __name__ == "__main__":
    main()
