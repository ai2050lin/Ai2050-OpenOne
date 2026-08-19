#!/usr/bin/env python3
"""Independent audit for Phase1257."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1257_c010_factorial_head_instrument_calibration"
PROTOCOL = OUT / "protocol/preregistration.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/calibration_result.json"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/instrument_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preaudit() -> None:
    protocol = read(PROTOCOL)
    checks = {
        "contract_id": protocol.get("contract_id") == "EXP-C010-WP01-001",
        "no_model_run": any("No Qwen3" in item for item in protocol.get("hard_stops", [])),
        "no_phase1256_rescan": any("No Phase1256 component" in item for item in protocol.get("hard_stops", [])),
        "real_qwen_head_geometry": protocol.get("qwen_geometry", {}).get("query_heads") == 32 and protocol.get("qwen_geometry", {}).get("kv_heads") == 8,
        "gqa_group_registered": protocol.get("qwen_geometry", {}).get("gqa_group_size") == 4,
        "wrong_baseline_registered": "categorical_wrong_cosine_abs_error_max" in protocol.get("thresholds", {}),
        "exact_slice_gates": protocol.get("thresholds", {}).get("head_patch_reference_max_error") == 0.0,
        "trial_budget": protocol.get("tensor_trials", 0) >= 384,
        "dependencies_frozen": set(protocol.get("dependencies", {})) == {"phase1256_details", "phase1256_final"},
        "source_hashes_present": set(protocol.get("source_hashes", {})) == {"main", "auditor"},
    }
    result = {"stage": "preaudit", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    write(PREAUDIT, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def final_audit() -> None:
    protocol = read(PROTOCOL)
    raw = read(RAW)
    summary = read(SUMMARY)
    complete = read(COMPLETE)
    analysis = read(ANALYSIS)
    final = read(FINAL)
    tensor = raw["tensor_instrument"]
    null = raw["phase1256_null_identifiability"]
    checks = {
        "formal_marker": complete.get("status") == "formal_run_complete",
        "run_digest_link": complete.get("run_digest") == summary.get("run_digest"),
        "calibration_passed": raw.get("passed") is True,
        "wrong_cosine_is_half": abs(raw["categorical_geometry"]["wrong_cosine_mean"] - 0.5) <= 1.0e-12,
        "wrong_geometry_exhaustive": raw["categorical_geometry"]["case_count"] == 336,
        "null_direction_abstained": null.get("world_direction_identifiable") is False and null.get("adjudication") == "not_identifiable_from_frozen_aggregates",
        "null_norm_recomputed": abs(null["aggregate_null_norm"] - null["aggregate_null_fraction"] * null["aggregate_target_norm"]) <= 1.0e-12,
        "multiple_null_decompositions": len(null.get("same_total_distinct_decompositions", [])) >= 5,
        "slice_reference_exact": tensor["head_patch_reference_max_error"] == 0.0,
        "untouched_exact": tensor["untouched_slice_max_error"] == 0.0,
        "noop_exact": tensor["no_op_max_error"] == 0.0,
        "commutation_exact": tensor["disjoint_commutation_max_error"] == 0.0,
        "union_exact": tensor["full_union_max_error"] == 0.0,
        "gqa_exact": tensor["gqa_mapping_exact"] is True and tensor["gqa_queries_per_kv_head"] == [4] * 8,
        "authorization_scoped": analysis["authorization"]["new_natural_qwen_contract"] is True and analysis["authorization"]["phase1256_head_rescan"] is False,
        "final_verdict": final.get("verdict") == "factorial_head_instrument_calibrated",
        "artifact_set": set(final.get("artifact_hashes", {})) == {"protocol", "environment", "preaudit", "raw", "summary", "complete", "analysis"},
    }
    result = {"stage": "final_audit", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    write(FINAL_AUDIT, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "final"))
    args = parser.parse_args()
    if args.stage == "preaudit":
        preaudit()
    else:
        final_audit()


if __name__ == "__main__":
    main()
