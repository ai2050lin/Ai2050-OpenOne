#!/usr/bin/env python3
"""Independent C127 contract, behavior, typed checkpoint, and result audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1661_c127_typed_transition_language_family"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1661_c127_typed_transition_language_family as c127


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    behavior = core.load(OUT / "analysis/behavior_gate.json")
    confirmation = core.load(OUT / "analysis/confirmation.json")
    closure = core.load(OUT / "analysis/closure.json")
    raw = np.load(OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy", mmap_mode="r")
    fields = np.load(OUT / "analysis/unit_truth_role_typed_checkpoint.float32.npy", mmap_mode="r")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    units = core.rows(OUT / "material/units.jsonl")
    lookup = {row["unit_id"]: index for index, row in enumerate(units)}
    recomputed = np.zeros_like(fields)
    for row_index, row in enumerate(rows):
        recomputed[lookup[row["unit_id"]]] += float(row["truth_factor"]) / 8.0 * c127.decode(raw[row_index])
    payload = core.load(PUBLIC)
    batch = payload["c127_typed_transition_batch"]
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "behavior": core.load(OUT / "audit/internal_behavior_audit.json")["all_integrity_checks_passed"] and behavior["gate_passed"],
        "capture": core.load(OUT / "audit/internal_capture_audit.json")["all_checks_passed"],
        "discovery": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"],
        "confirmation": core.load(OUT / "audit/internal_confirmation_audit.json")["all_integrity_checks_passed"],
        "closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
        "parent_hashes": all(core.sha(Path(protocol["parent_paths"][name])) == digest for name, digest in protocol["parent_hashes"].items()),
        "capture_source": core.sha(Path(protocol["local_capture_semantics"]["path"])) == protocol["local_capture_semantics"]["sha256"],
        "shapes": list(raw.shape) == [256, 6, 38, 2560] and list(fields.shape) == [32, 6, 38, 2560],
        "field_recompute": np.array_equal(recomputed, np.asarray(fields)),
        "typed_checkpoints": len(protocol["checkpoints"]) == 38 and protocol["checkpoints"][0] == "embedding" and protocol["checkpoints"][-1] == "post_final_norm",
        "visualization": len(batch["effect_rows"]) == 150 and len(batch["representative_raw_rows"]) == 42 and all(len(row["values"]) == 2560 for row in [*batch["effect_rows"], *batch["representative_raw_rows"]]),
        "asset": core.sha(PUBLIC) == closure["heatmap"]["sha256"] == core.load(OUT / "audit/internal_closure_audit.json")["asset_sha256"],
        "boundary": "not weights" in closure["claim_boundary"] and "attention/MLP" in closure["claim_boundary"],
    }
    report = {"phase": 1661, "campaign": "C127", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": confirmation["all_gates_passed"], "authorization": "append_memo_and_consider_c128" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_closure_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
