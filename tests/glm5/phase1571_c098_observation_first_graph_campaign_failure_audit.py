#!/usr/bin/env python3
"""Independent audit of the preregistered C098 numeric-gate closure."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1571_c098_observation_first_graph_campaign"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    py_compile.compile(str(TESTS / "phase1571_c098_observation_first_graph_campaign.py"), doraise=True)
    pre = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    failure = core.load(OUT / "analysis/capture_failure.json")
    final = core.load(OUT / "analysis/final.json")
    raw_path = OUT / "raw/all_token_all_state_field.float16.npy"
    index_path = OUT / "raw/all_token_field_index.jsonl"
    raw = np.load(raw_path, mmap_mode="r")
    index_rows = sum(1 for line in index_path.open("r", encoding="utf-8") if line.strip())
    sample = np.asarray(raw[:, [0, raw.shape[1] // 2, raw.shape[1] - 1], :32], dtype=np.float32)
    affected = failure["affected_units"]
    checks = {
        "premodel": pre["all_checks_passed"] and pre["passed"] == 23,
        "formal_closure": final["status"] == "closed_at_preregistered_numeric_execution_gate",
        "hidden_not_analyzed": final["hidden_structure_analyzed"] is False and failure["hidden_structure_unblinded"] is False,
        "failed_frozen_gate": failure["failed_gate"] == "causal_prefix_effect_max_abs <= 1e-6",
        "observed_failure": failure["observed_global_max_abs"] == 3.0,
        "affected_partition": failure["affected_unit_count"] == 6 and failure["unaffected_unit_count"] == 66,
        "width_diagnosis": len(affected) == 6 and all(abs(unit["physical_widths"][1] - unit["physical_widths"][0]) == 1 for unit in affected),
        "raw_shape": tuple(raw.shape) == (37, 191600, 2560) and raw.dtype == np.float16,
        "raw_bytes": raw_path.stat().st_size == 36296704128,
        "raw_sample_finite": bool(np.isfinite(sample).all()),
        "index_coverage": index_rows == 1152,
        "authorized_correction": final["authorization"] == "run_C099_same_material_fixed_global_width",
    }
    result = {
        "phase": 1571,
        "campaign": "C098",
        "audit_type": "formal_failure_path",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_failure_path_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
