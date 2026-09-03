#!/usr/bin/env python3
"""Independent audit for Phase1448 C074 directional domain map."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1448, "C074"
CONTRACT = TESTS / "result/phase1445_c074_directional_domain_contract"
OUT = TESTS / "result/phase1448_c074_directional_domain_map"
SPLITS = ("confirmation", "lockbox")


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/directional_domain_summary.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "raw/directional_domain.jsonl")
    robust = core.rows(OUT / "analysis/robust_edges.jsonl")
    gate = protocol["domain"]
    expected_balance = {(split, route, direction, arm): 24 for split in SPLITS for route in gate["routes"] for direction in gate["directions"] for arm in gate["arms"]}
    recomputed_robust = sorted(f"{route}::{direction}" for route in gate["routes"] for direction in gate["directions"] if all(summary["cell_results"][route][direction][split]["pass"] for split in SPLITS))
    class_counts = {name: sum(summary["edge_results"][route][direction]["classification"] == name for route in gate["routes"] for direction in gate["directions"]) for name in gate["edge_classes"]}
    checks = {
        "count": len(rows) == gate["holdout_sets"] * len(gate["routes"]) * len(gate["directions"]) * len(gate["arms"]),
        "balance": Counter((row["partition"], row["route"], row["direction"], row["arm"]) for row in rows) == expected_balance,
        "holdout": {row["partition"] for row in rows} == set(SPLITS) and len({row["set_id"] for row in rows}) == 48,
        "finite": all(math.isfinite(row[key]) for row in rows for key in ("recipient_margin", "correct_donor_margin", "wrong_donor_margin", "swap_margin", "oriented_gain", "full_logit_max_abs_diff", "write_max_abs_diff", "complement_max_abs_diff")),
        "writes": max(max(row["write_max_abs_diff"], row["complement_max_abs_diff"]) for row in rows) <= protocol["camera"]["write_max_abs_diff"],
        "metadata": all(protocol["routes"][row["route"]][key] == row[key] for row in rows for key in ("same_surface", "same_frame", "same_order")),
        "cells": summary["cell_count"] == 64 and all(summary["cell_results"][route][direction][split]["count_per_arm"] == 24 for route in gate["routes"] for direction in gate["directions"] for split in SPLITS),
        "edges": summary["edge_count"] == 32 and sum(summary["class_counts"].values()) == 32,
        "classes": summary["class_counts"] == class_counts,
        "robust": sorted(summary["robust_edge_ids"]) == recomputed_robust == sorted(row["edge_id"] for row in robust),
        "one_shot": summary["reveal_manifest"]["one_shot"] and summary["reveal_manifest"]["holdout_count"] == 48,
        "contract": summary["contract_sha256"] == protocol["contract_sha256"],
        "execution": summary["all_execution_checks_passed"] == all(summary["checks"].values()),
        "authorization": final["authorization"] == "run_phase1449_c074_campaign_closure",
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
