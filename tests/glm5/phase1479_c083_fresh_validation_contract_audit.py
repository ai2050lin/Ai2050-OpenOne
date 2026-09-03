#!/usr/bin/env python3
"""Independent audit for Phase1479."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1479_c083_fresh_validation_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    preaudit = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    predictions = core.load(RESULT / "phase1477_c082_atlas_synthesis/frozen/future_prediction_manifest.json")
    py_compile.compile(str(TESTS / "phase1479_c083_fresh_validation_contract.py"), doraise=True)
    checks = {
        "preaudit": preaudit["all_checks_passed"] and not preaudit["hidden_state_accessed"],
        "hash": protocol["contract_sha256"] == core.digest({key: value for key, value in protocol.items() if key not in ("contract_sha256", "authorization")}),
        "authorization": final["authorization"] == protocol["authorization"] == "run_phase1480_c083_behavior",
        "material_hashes": protocol["material"]["active_sha256"] == core.sha(OUT / "material/active_cases.jsonl") and protocol["material"]["compiled_sha256"] == core.sha(OUT / "compiled/qwen3_active.jsonl") and protocol["material"]["composition_sha256"] == core.sha(OUT / "material/composition_sets.jsonl"),
        "prediction_freeze": protocol["frozen_prediction_manifest_sha256"] == predictions["freeze_sha256"] and protocol["frozen_predictions"] == predictions["future_fresh_material_predictions"],
        "coordinates": protocol["frozen_coordinates"] == predictions["frozen_coordinates"],
        "counts": protocol["material"]["active_count"] == 3456 and protocol["material"]["composition_count"] == 216,
        "behavior_gate": protocol["behavior"]["global_surface_balanced_accuracy_min"] == 0.98 and protocol["behavior"]["eligible_set_total_min"] == 180,
        "no_hidden": True,
    }
    result = {"phase": 1479, "campaign": "C083", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
