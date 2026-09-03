#!/usr/bin/env python3
"""Independent audit for the C109 observation-first contract."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1603_c109_fresh_role_state_field_atlas"
SOURCE = TESTS / "result/phase1600_c108_fresh_coordinate_causality"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1603_c109_fresh_role_state_field_contract.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    source = core.load(SOURCE / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/pre_model_audit.json")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    checks = {
        "producer": core.sha(producer) == protocol["producer_sha256"],
        "source": protocol["source"]["material_digest"] == source["material_digest"],
        "source_compiled": protocol["source"]["compiled_sha256"] == core.sha(SOURCE / "compiled/qwen3.jsonl"),
        "manifest": protocol["manifest_sha256"] == core.sha(OUT / "protocol/role_occurrence_manifest.jsonl"),
        "source_checks": audit["all_checks_passed"],
        "shape": protocol["archive"]["shape"] == [37, len(manifest), 2560],
        "index": all(row["occurrence_index"] == index for index, row in enumerate(manifest)),
        "roles": len(protocol["roles"]) == 7 and sum(protocol["role_occurrence_counts"].values()) == len(manifest),
        "same_k_controls": len(protocol["supports"]["attribute_binding_k256"]) == len(protocol["supports"]["attribute_wrong_agent_k256"]) == 256 and len(protocol["supports"]["agent_patient_k128"]) == len(protocol["supports"]["agent_wrong_attribute_k128"]) == 128,
        "observation_boundary": "not a fresh confirmation" in protocol["source"]["exposure_status"] and "observation-only" in protocol["typed_missingness"]["causal_extension"],
        "authorization": protocol["authorization"] == "execute_phase1604_c109_qwen_role_state_capture",
    }
    result = {"phase": 1603, "campaign": "C109", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_pre_model_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
