#!/usr/bin/env python3
"""Independent C129 typed-transition audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1663_c129_direct_precedence_typed_transition"
C128 = TESTS / "result/phase1662_c128_direct_precedence_behavior_qualification"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1661_c127_typed_transition_language_family as c127


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); confirmation = core.load(OUT / "analysis/confirmation.json"); closure = core.load(OUT / "analysis/closure.json")
    raw = np.load(OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy", mmap_mode="r"); fields = np.load(OUT / "analysis/unit_truth_role_typed_checkpoint.float32.npy", mmap_mode="r")
    rows = core.rows(C128 / "compiled/qwen3.jsonl"); units = core.rows(C128 / "material/units.jsonl"); lookup = {row["unit_id"]: index for index, row in enumerate(units)}
    recomputed = np.zeros_like(fields)
    for row_index, row in enumerate(rows):
        recomputed[lookup[row["unit_id"]]] += float(row["truth_factor"]) / 8.0 * c127.decode(raw[row_index])
    payload = core.load(PUBLIC); batch = payload["c129_direct_precedence_typed_transition_batch"]
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "capture": core.load(OUT / "audit/internal_capture_audit.json")["all_checks_passed"], "discovery": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"], "confirmation": core.load(OUT / "audit/internal_confirmation_audit.json")["all_integrity_checks_passed"], "closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"], "source_hashes": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()), "shapes": list(raw.shape) == [256, 5, 38, 2560] and list(fields.shape) == [32, 5, 38, 2560], "field_recompute": np.array_equal(recomputed, np.asarray(fields)), "typed_checkpoints": len(protocol["checkpoints"]) == 38 and protocol["checkpoints"][0] == "embedding" and protocol["checkpoints"][-1] == "post_final_norm", "visualization": len(batch["effect_rows"]) == 150 and len(batch["representative_raw_rows"]) == 35 and all(len(row["values"]) == 2560 for row in [*batch["effect_rows"], *batch["representative_raw_rows"]]), "asset": core.sha(PUBLIC) == closure["heatmap"]["sha256"] == core.load(OUT / "audit/internal_closure_audit.json")["asset_sha256"], "boundary": "not weights" in closure["claim_boundary"] and "attention/MLP" in closure["claim_boundary"]}
    report = {"phase": 1663, "campaign": "C129", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": confirmation["all_gates_passed"], "authorization": "append_memo_and_consider_c130" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_closure_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
