#!/usr/bin/env python3
"""Independent audit for Phase1383."""
from __future__ import annotations

import json
import math
import py_compile
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1383_c060_refined_dose_observation"
SCRIPT = TESTS / "phase1383_c060_refined_dose_observation.py"


def main() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    summary = core.load(OUT / "analysis/qwen3_refined_dose_summary.json")
    final = core.load(OUT / "analysis/final.json")
    records = core.rows(OUT / "raw/qwen3_refined_dose_response.jsonl")
    discovery_path = OUT / "raw/response_discovery_family3_differences.pt"
    discovery = torch.load(discovery_path, map_location="cpu", weights_only=False)
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    expected_records = manifest["case_count"] * len(manifest["paths"]) * sum(
        len(v) for v in manifest["mode_layouts"].values()
    )
    gate = manifest["dose_gate"]
    early_endpoint = summary["path_summary"]["family_early"]["sufficiency_endpoint"]
    mediation = all(early_endpoint[p]["passed"] for p in ("pooled", "confirmation", "lockbox"))
    checks = {
        "record_count": len(records) == expected_records == summary["record_count"],
        "partitions": {r["partition"] for r in records} == {"response_discovery", "confirmation", "lockbox"},
        "all_arms": {(r["mode"], r["direction"], r["dose"]) for r in records} == {
            (m, d, float(l))
            for m in ("sufficiency", "reverse")
            for d in gate["directions"]
            for l in gate["values"]
        },
        "finite": all(r["all_finite"] and math.isfinite(r["output_effect"]) for r in records),
        "norm_match": max(r["norm_ratio_abs_error"] for r in records) <= gate["norm_ratio_abs_error_max"],
        "lambda0_identity": max(abs(r["output_effect"]) for r in records if r["dose"] == 0.0) <= gate["self_output_max_abs_diff"],
        "split_counts": all(sum(r["partition"] == p for r in records) == expected_records // 3 for p in ("response_discovery", "confirmation", "lockbox")),
        "discovery_shape": list(discovery["vectors"].shape) == [72, 2560],
        "discovery_metadata": len(discovery["metadata"]) == 72 and all(r["partition"] == "response_discovery" for r in discovery["metadata"]),
        "discovery_hash": core.sha(discovery_path) == summary["discovery_sha256"],
        "summary_paths": set(summary["path_summary"]) == set(manifest["paths"]),
        "mediation_consistent": final["mediation_eligible"] == summary["mediation_eligible"] == mediation,
        "authorization": final["authorization"] == "run_phase1384_c060_fixed_dynamic_coalitions",
        "scripts_compile": True,
    }
    audit = {
        "phase": 1383,
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
