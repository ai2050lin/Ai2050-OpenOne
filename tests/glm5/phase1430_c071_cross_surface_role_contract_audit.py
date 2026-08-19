#!/usr/bin/env python3
"""Independent audit for Phase1430."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1430_c071_cross_surface_role_contract"
ROLES = ("record_target", "record_family", "query_target", "query_family")


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    pre = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    active = core.rows(OUT / "material/active_cases.jsonl")
    composition = core.rows(OUT / "material/composition_sets.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    surfaces = protocol["material"]["surfaces"]
    signatures = {
        surface: {tuple((role, tuple(row["role_positions"][role])) for role in ROLES) for row in compiled if row["surface"] == surface}
        for surface in surfaces
    }
    checks = {
        "preaudit": pre["all_checks_passed"],
        "active": len(active) == 2880 and Counter(row["surface"] for row in active) == {surface: 1440 for surface in surfaces},
        "cells": Counter(row["cell"] for row in active) == {cell: 360 for cell in ("aa", "ab", "ac", "ad", "bb", "ba", "bc", "bd")},
        "composition": len(composition) == 72 and Counter(row["partition"] for row in composition) == {name: 24 for name in protocol["material"]["partitions"]},
        "surface_shapes": all(len({len(row["prompt_ids"]) for row in compiled if row["surface"] == surface}) == 1 for surface in surfaces),
        "different_shapes": len({len(row["prompt_ids"]) for row in compiled}) == 2,
        "role_maps": all(len(value) == 1 for value in signatures.values()) and len({next(iter(value)) for value in signatures.values()}) == 2,
        "quartet": all(len({row["role_positions"][role][0] for role in ROLES}) == 4 for row in compiled),
        "hashes": protocol["material"]["active_sha256"] == core.sha(OUT / "material/active_cases.jsonl") and protocol["material"]["composition_sha256"] == core.sha(OUT / "material/composition_sets.jsonl"),
        "zero_model_erratum": abs(pre["zero_models"]["person_only_balanced_accuracy"] - 5 / 6) < 1e-12 and abs(pre["zero_models"]["group_only_balanced_accuracy"] - 5 / 6) < 1e-12,
        "fixed_object": protocol["research_object"] == "state16 cross-surface semantic-role-mapped quartet transport" and protocol["camera"]["state_index"] == protocol["mechanism"]["state_index"] == 16,
        "five_arms": len(protocol["mechanism"]["arms"]) == 5 and "cross_surface_role_permuted" in protocol["mechanism"]["arms"],
        "forbidden": all(value in protocol["forbidden"] for value in ("attention", "MLP", "gradients", "PCA", "learned probe", "layer search", "role subset search")),
        "hidden_not_accessed": pre["checks"]["hidden_not_accessed"],
        "authorization": final["authorization"] == "run_phase1431_c071_cross_surface_behavior",
    }
    result = {
        "phase": 1430, "campaign": "C071", "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
