#!/usr/bin/env python3
"""C326: freeze a balanced 480-case natural panel for three sequential models."""
from __future__ import annotations

from collections import Counter

import phase1844_c310_c335_dual_axis_common as common


def main() -> None:
    parent = common.core.load(common.OUTS["C325"] / "analysis/final.json")
    source = common.core.rows(common.OUTS["C321"] / "material/cases.jsonl")
    surface_index = {name: i for i, name in enumerate(common.NATURAL_SURFACES)}
    family_index = {name: i for i, name in enumerate(common.FAMILIES)}
    rows = []
    for row in source:
        if row["unit"] not in (0, 1, 4, 5):
            continue
        desired_order = 1 if (surface_index[row["surface"]] + family_index[row["family"]] + row["unit"]) % 2 == 0 else -1
        if row["order"] == desired_order:
            rows.append({**row, "partition": "discovery" if row["unit"] in (0, 1) else "confirmation"})
    counts = Counter((row["family"], row["surface"], row["partition"]) for row in rows)
    checks = {"parent": parent["all_checks_passed"], "rows": len(rows) == 480, "complete_factorials": all(value == 8 for value in counts.values()), "candidate_balance": sum(row["gold_position"] == 0 for row in rows) == 240, "models": common.MODELS == ("qwen3", "glm4", "deepseek7b")}
    protocol = {"status": "cross_model_panel_frozen", "cases": 480, "models": list(common.MODELS), "loading": "strictly sequential, one process and one model at a time", "factorization": "6 families x 5 surfaces x 4 units x 4 cells; one candidate order fixed per complete factorial group", "partitions": "units0-1 discovery, units4-5 confirmation", "comparison": "model-native full coordinate axes summarized only by role, relative depth, prediction gain, and intervention response", "claim_boundary": "The panel can support convergent response structure. It cannot prove shared physical coordinates, identical algorithms, or full functional isomorphism."}
    out = common.prepare("C326", protocol, checks)
    common.core.write_rows(out / "material/cases.jsonl", rows)
    headline = {"status": "cross_model_panel_closed", "cases": len(rows), "candidate_balance": [240, 240], "group_strata": len(counts), "strict_interpretation": protocol["claim_boundary"]}
    common.close("C326", headline, {"all_checks": all(checks.values()), "material_rows": len(common.core.rows(out / "material/cases.jsonl")) == 480}, "C327_then_C328_then_C329_sequential")


if __name__ == "__main__":
    main()
