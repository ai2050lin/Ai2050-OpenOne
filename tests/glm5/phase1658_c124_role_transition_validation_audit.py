#!/usr/bin/env python3
"""Independent C124 result and visualization audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1658_c124_role_transition_validation"
C123 = TESTS / "result/phase1657_c123_role_transition_atlas_discovery"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    amendment = core.load(OUT / "protocol/phase1658_execution_amendment.json")
    summary = core.load(OUT / "analysis/validation_summary.json")
    closure = core.load(OUT / "analysis/closure.json")
    results = core.rows(OUT / "analysis/validation_results.jsonl")
    manifest = core.rows(OUT / "analysis/validation_cell_manifest.jsonl")
    discovery = np.load(C123 / "analysis/discovery_selected_role_increments.float32.npy", mmap_mode="r")
    validation = np.load(OUT / "analysis/validation_selected_role_increments.float32.npy", mmap_mode="r")
    family_order = core.load(C123 / "protocol/frozen_discovery_nomination.json")["family_order"]
    recomputed = []
    for row, cell in zip(results, manifest, strict=True):
        family_index = family_order.index(cell["family"])
        transition = int(row["to_state"]) - 1
        recomputed.append(cosine(np.asarray(discovery[family_index, transition]), np.asarray(validation[int(cell["cell_index"]), transition])))
    payload = core.load(PUBLIC)
    transition_rows = payload.get("transition_rows", [])
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "execution_amendment": amendment["original_producer_sha256"] == protocol["producer_sha256"] and amendment["repaired_producer_sha256"] == core.sha(TESTS / "phase1658_c124_role_transition_validation.py") and set(amendment["unchanged"]) >= {"frozen C123 nominations", "validation cells", "gates", "wrong-state controls", "wrong-role controls", "coordinate-clock rule"},
        "validation": core.load(OUT / "audit/internal_validation_audit.json")["all_checks_passed"],
        "closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
        "source_hashes_except_mutated_public": all(name == "public_atlas" or core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
        "result_count": len(results) == 6 and len(manifest) == 6,
        "cosines": all(abs(value - row["target_increment_cosine"]) < 1e-6 for value, row in zip(recomputed, results, strict=True)),
        "shapes": list(discovery.shape) == [4, 36, 2560] and list(validation.shape) == [6, 36, 2560],
        "atlas_schema": payload["schema"] == "c109_role_state_field_atlas.v1" and payload["phase"] == 1658,
        "atlas_rows": len(transition_rows) == 584 and all(len(row["values"]) == 2560 for row in transition_rows),
        "asset_hash": core.sha(PUBLIC) == closure["heatmap"]["sha256"] == core.load(OUT / "audit/internal_closure_audit.json")["asset_sha256"],
        "boundary": "not model weights" in payload["claim_boundary"] and "not" in closure["claim_boundary"],
        "next": closure["next_authorization"].startswith("C125 fresh semantic-program family"),
    }
    report = {"phase": 1658, "campaign": "C124", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": "append_memo_and_consider_c125" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_closure_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
