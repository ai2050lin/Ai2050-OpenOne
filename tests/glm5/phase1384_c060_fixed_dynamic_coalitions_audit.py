#!/usr/bin/env python3
"""Independent audit for Phase1384."""
from __future__ import annotations

import json
import math
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1384_c060_fixed_dynamic_coalitions"
SCRIPT = TESTS / "phase1384_c060_fixed_dynamic_coalitions.py"


def main() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    summary = core.load(OUT / "analysis/qwen3_coalition_summary.json")
    final = core.load(OUT / "analysis/final.json")
    records = core.rows(OUT / "raw/qwen3_coalitions.jsonl")
    rankings = core.load(OUT / "protocol/discovery_rankings.json")
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    expected = manifest["case_count"] * len(manifest["groups"])
    groups = {g["group_id"] for g in manifest["groups"]}
    dynamic_expected = sum(g["kind"] == "dynamic" for g in manifest["groups"])
    fixed_expected = sum(g["kind"] == "fixed" for g in manifest["groups"])
    checks = {
        "record_count": len(records) == expected == summary["record_count"],
        "group_counts": fixed_expected == 16 and dynamic_expected == 25 and len(groups) == 41,
        "balanced_records": set(Counter(r["group_id"] for r in records).values()) == {144},
        "partitions": Counter(r["partition"] for r in records) == {"confirmation": expected // 2, "lockbox": expected // 2},
        "finite": summary["runtime"]["all_finite"] and all(
            math.isfinite(v)
            for r in records
            for v in list(r["suff_gain"].values()) + list(r["reverse_damage"].values())
        ),
        "self": max(r["self_max_abs_diff"] for r in records) <= manifest["fixed_gate"]["self_max_abs_diff"],
        "norm": max(r["norm_ratio_abs_error_max"] for r in records) <= 1e-5,
        "rankings_hash": core.sha(OUT / "protocol/discovery_rankings.json") == manifest["discovery_rankings_sha256"],
        "discovery_scope": rankings["selection_scope"] == "response_discovery only",
        "all_summaries": set(summary["metrics"]) == groups and set(summary["qualifications"]) == groups,
        "dynamic_summaries": len(summary["dynamic_comparison"]) == 20,
        "algebra_splits": set(summary["inherited_algebra"]) == {"pooled", "confirmation", "lockbox"},
        "final_consistent": final["inherited_S1024_reverse_replicated"] == summary["inherited_S1024_reverse_replicated"],
        "authorization": final["authorization"] == "run_phase1385_c060_early_mediation",
        "scripts_compile": True,
    }
    audit = {
        "phase": 1384,
        "campaign": "C060",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
