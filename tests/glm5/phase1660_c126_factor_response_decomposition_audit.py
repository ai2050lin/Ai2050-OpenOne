#!/usr/bin/env python3
"""Independent C126 array, decomposition, ranking, and visualization audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1660_c126_factor_response_decomposition"
C125 = TESTS / "result/phase1659_c125_final_transition_decomposition"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1660_c126_factor_response_decomposition as c126


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/adjudication.json")
    closure = core.load(OUT / "analysis/closure.json")
    raw = np.load(C125 / "raw/qwen3_role_checkpoint_states.float32.npy", mmap_mode="r")
    means = np.load(OUT / "analysis/unit_role_checkpoint_means.float32.npy", mmap_mode="r")
    effects = np.load(OUT / "analysis/unit_role_effect_checkpoint.float32.npy", mmap_mode="r")
    rows = core.rows(C125 / "compiled/qwen3.jsonl")
    units = core.rows(C125 / "material/units.jsonl")
    lookup = {row["unit_id"]: index for index, row in enumerate(units)}
    specs = c126.effect_specs()
    max_error = 0.0
    for row_index, row in enumerate(rows):
        factors = (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"])
        predicted = np.asarray(means[lookup[row["unit_id"]]], dtype=np.float32).copy()
        for effect_index, spec in enumerate(specs):
            sign = int(np.prod([factors[index] for index in range(4) if spec["mask"] & (1 << index)]))
            predicted += sign * np.asarray(effects[lookup[row["unit_id"]], :, effect_index], dtype=np.float32)
        max_error = max(max_error, float(np.max(np.abs(predicted - np.asarray(raw[row_index], dtype=np.float32)))))
    comparisons = core.rows(OUT / "analysis/factor_comparisons.jsonl")
    public = core.load(PUBLIC)
    visual = public["c126_factor_response_batch"]["effect_rows"]
    amendment = core.load(OUT / "protocol/upstream_index_correction_amendment.json")
    corrected_names = set(amendment["drift"])
    unchanged_sources = set(protocol["source_paths"]) - corrected_names
    source_hashes_valid = all(core.sha(Path(protocol["source_paths"][name])) == protocol["source_hashes"][name] for name in unchanged_sources)
    corrected_sources_valid = all(
        amendment["drift"][name]["frozen"] == protocol["source_hashes"][name]
        and amendment["drift"][name]["current"] == core.sha(Path(protocol["source_paths"][name]))
        for name in corrected_names
    )
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
        "closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
        "source_hashes_with_amendment": source_hashes_valid and corrected_sources_valid,
        "shapes": list(means.shape) == [24, 2, 4, 2560] and list(effects.shape) == [24, 2, 15, 4, 2560],
        "reconstruction": max_error == report["reconstruction_max_abs"] and max_error <= 2e-4,
        "comparisons": len(comparisons) == 45 and sum(row["signed_rank"] == 1 for row in comparisons) == 3 and sum(row["absolute_rank"] == 1 for row in comparisons) == 3,
        "effects": len(visual) == 315 and all(len(row["values"]) == 2560 for row in visual),
        "asset": core.sha(PUBLIC) == closure["heatmap"]["sha256"] == core.load(OUT / "audit/internal_closure_audit.json")["asset_sha256"],
        "boundary": "not weights" in closure["claim_boundary"] and "attention/MLP" in closure["claim_boundary"],
    }
    audit = {"phase": 1660, "campaign": "C126", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "reconstruction_max_abs": max_error, "authorization": "append_memo_and_consider_c127" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_closure_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
