#!/usr/bin/env python3
"""Independent artifact and numeric audit for Phase1558."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1557_c096_fresh_human_relation_field_contract"
OUT = RESULT / "phase1558_c096_unified_behavior_and_all_state_capture"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    py_compile.compile(str(TESTS / "phase1558_c096_unified_behavior_and_all_state_capture.py"), doraise=True)
    contract = core.load(CONTRACT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/c096_capture_and_behavior_summary.json")
    final = core.load(OUT / "analysis/final.json")
    field = np.load(OUT / "raw/c096_all_role_field.float16.npy", mmap_mode="r")
    index = core.rows(OUT / "raw/c096_all_role_field_index.jsonl")
    behavior = core.rows(OUT / "raw/c096_behavior_logits.jsonl")
    three_way = core.rows(OUT / "analysis/c096_three_way_pair_selection.jsonl")
    checks = {
        "contract_audited": core.load(CONTRACT / "audit/independent_final_audit.json")["all_checks_passed"],
        "field_hash": core.sha(OUT / "raw/c096_all_role_field.float16.npy") == report["files"]["field"]["sha256"],
        "other_hashes": all(core.sha(ROOT / report["files"][name]["path"]) == report["files"][name]["sha256"] for name in ("index", "behavior", "three_way")),
        "shape": list(field.shape) == [540, 37, 4, 2560],
        "coverage": len(index) == len(behavior) == 540 and len(three_way) == 180 and sorted(row["row_index"] for row in index) == list(range(540)),
        "finite": bool(np.isfinite(np.asarray(field[::17])).all()) and all(np.isfinite(row["candidate_logits"]).all() for row in behavior),
        "numeric_gate": all(report["checks"][key] for key in ("repeat_hidden", "repeat_logits", "postquery_causal_identity", "prequery_causal_identity", "bf16_nonquantized")),
        "behavior_scope": set(report["family_behavior"]) == set(contract["families"]) and all(row["missingness"] in {None, "M_BEHAVIOR"} for row in report["family_behavior"].values()),
        "field_behavior_identity": all(index[i]["case_id"] == behavior[i]["case_id"] for i in range(540)),
        "predictions_still_frozen": len(contract["frozen_predictions"]) == 5,
        "authorization": final["authorization"] == "run_phase1559_c096_fresh_prediction_atlas_and_adjudication",
    }
    result = {"phase": 1558, "campaign": "C096", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
