#!/usr/bin/env python3
"""C331: freeze a four-depth graph panel before model observation."""
from __future__ import annotations

from collections import Counter

import phase1844_c310_c335_dual_axis_common as common


def main() -> None:
    parent = common.core.load(common.OUTS["C330"] / "analysis/final.json")
    rows = common.graph_material()
    strata = Counter((r["depth"], r["surface"], r["shortcut"], r["partition"], r["gold_position"]) for r in rows)
    checks = {
        "parent": parent["all_checks_passed"],
        "rows": len(rows) == 384,
        "unique_ids": len({r["case_id"] for r in rows}) == 384,
        "depths": {r["depth"] for r in rows} == {1, 2, 3, 4},
        "surfaces": {r["surface"] for r in rows} == {"registry", "briefing"},
        "shortcuts": {r["shortcut"] for r in rows} == {0, 1},
        "exact_position_balance": all(
            strata[(d, s, h, p, 0)] == strata[(d, s, h, p, 1)]
            for d in range(1, 5) for s in ("registry", "briefing")
            for h in (0, 1) for p in ("discovery", "confirmation")
        ),
        "semantic_graph": all(r["semantic_graph"]["material"] == "main" for r in rows),
    }
    protocol = {
        "status": "four_level_graph_material_frozen",
        "factorization": "12 disjoint graphs x 4 path depths x 2 surfaces x shortcut absent/present x 2 candidate orders",
        "partitions": "graphs 0-7 discovery; graphs 8-11 prospective confirmation",
        "zero_models": ["always first", "always second", "surface only", "depth only", "shortcut only"],
        "behavior_gate": {"global_min": 0.80, "depth_min": 0.65, "surface_min": 0.65},
        "observation_policy": "Capture proceeds even if a behavioral stratum misses the gate; all analyses retain behavior labels and report eligibility separately.",
        "human_naturalness_status": "no_test",
        "claim_boundary": "This is a controlled transitive type graph, not proof that natural taxonomies use the same internal computation.",
    }
    out = common.prepare("C331", protocol, checks)
    common.core.write_rows(out / "material/cases.jsonl", rows)
    zero = {"always_first": sum(r["gold_position"] == 0 for r in rows) / len(rows), "always_second": sum(r["gold_position"] == 1 for r in rows) / len(rows)}
    common.core.save(out / "analysis/zero_models.json", zero)
    common.core.save(out / "protocol/external_human_blind_review.json", {"status": "no_test", "reason": "No independent raters were available.", "required_raters": 3})
    headline = {"status": "four_level_graph_contract_closed", "cases": len(rows), "candidate_position_zero_models": zero, "human_naturalness": "no_test", "strict_interpretation": protocol["claim_boundary"]}
    common.close("C331", headline, {"checks": all(checks.values()), "saved_rows": len(common.core.rows(out / "material/cases.jsonl")) == 384, "zero_models_exact_chance": all(abs(v - 0.5) < 1e-12 for v in zero.values())}, "C332_qwen_full_field")


if __name__ == "__main__":
    main()
