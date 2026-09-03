#!/usr/bin/env python3
"""Independent C125 result and visualization audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1659_c125_final_transition_decomposition"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture = core.load(OUT / "analysis/capture_summary.json")
    adjudication = core.load(OUT / "analysis/adjudication.json")
    closure = core.load(OUT / "analysis/closure.json")
    raw = np.load(OUT / "raw/qwen3_role_checkpoint_states.float32.npy", mmap_mode="r")
    fields = np.load(OUT / "analysis/unit_truth_role_checkpoint.float32.npy", mmap_mode="r")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    units = core.rows(OUT / "material/units.jsonl")
    recomputed = np.zeros_like(fields)
    unit_index = {row["unit_id"]: index for index, row in enumerate(units)}
    for row_index, row in enumerate(rows):
        recomputed[unit_index[row["unit_id"]]] += float(row["truth_factor"]) / 16.0 * np.asarray(raw[row_index], dtype=np.float32)
    payload = core.load(PUBLIC)
    effects = payload["c125_final_transition_batch"]["effect_rows"]
    semantics = protocol["capture_semantics"]
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "capture": capture["checks"]["repeat"] and capture["checks"]["bf16"],
        "adjudication_integrity": core.load(OUT / "audit/internal_adjudication_audit.json")["all_integrity_checks_passed"],
        "closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
        "source_hashes": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
        "runtime_source_hashes": core.sha(Path(semantics["qwen_source_path"])) == semantics["qwen_source_sha256"] and core.sha(Path(semantics["capture_source_path"])) == semantics["capture_source_sha256"],
        "shapes": list(raw.shape) == [384, 2, 4, 2560] and list(fields.shape) == [24, 2, 4, 2560],
        "field_recompute": np.array_equal(recomputed, np.asarray(fields)),
        "predictions_reproduced": len(adjudication["results"]) == 3 and sum(row["frozen_prediction_passed"] for row in adjudication["results"]) == 2 and adjudication["scientific_gate_passed"] is False,
        "reconstruction": all(row["reconstruction_max_abs"] <= 1e-5 for row in adjudication["results"]),
        "effects": len(effects) == 21 and all(len(row["values"]) == 2560 for row in effects),
        "asset": core.sha(PUBLIC) == closure["heatmap"]["sha256"] == core.load(OUT / "audit/internal_closure_audit.json")["asset_sha256"],
        "boundary": "not weights" in closure["claim_boundary"] and "attention/MLP" in closure["claim_boundary"],
    }
    report = {"phase": 1659, "campaign": "C125", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": "append_memo_and_consider_c126" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_closure_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
