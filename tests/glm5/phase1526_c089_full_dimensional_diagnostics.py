#!/usr/bin/env python3
"""Phase1526: detect the C089 left-padding causal-prefix identity failure."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1521_c089_natural_relation_observation_contract"
CAPTURE = RESULT / "phase1522_c089_unified_forward_capture"
ATLAS = RESULT / "phase1523_c089_truth_contrast_atlas"
REVEAL = RESULT / "phase1525_c089_descriptive_holdout_reveal"
OUT = RESULT / "phase1526_c089_full_dimensional_diagnostics"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1526 exists")
    parent = core.load(REVEAL / "analysis/final.json")
    parent_audit = core.load(REVEAL / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1526_c089_full_dimensional_diagnostics" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1525 authorization missing")
    raw = np.load(CAPTURE / "raw/all_role_field.float16.npy", mmap_mode="r")
    index = core.rows(CAPTURE / "raw/all_role_field_index.jsonl")
    atlas = np.load(ATLAS / "atlas/group_truth_contrast.float16.npy", mmap_mode="r")
    atlas_index = core.rows(ATLAS / "atlas/group_truth_contrast_index.jsonl")
    source_role = protocol["roles"].index("source_word")
    source_by_state = [float(np.max(np.abs(np.asarray(atlas[:, :, state, source_role], dtype=np.float32)))) for state in range(37)]
    lookup = {(row["set_id"], row["surface"], row["cell"]): row for row in index}
    pair_rows = []
    for group in atlas_index:
        for surface in ("a_question", "b_question"):
            for left, right in (("aa", "ab"), ("bb", "ba")):
                a, b = lookup[(group["set_id"], surface, left)], lookup[(group["set_id"], surface, right)]
                state_max, state_relative = [], []
                for state in range(37):
                    x = np.asarray(raw[a["row_index"], state, source_role], dtype=np.float32)
                    y = np.asarray(raw[b["row_index"], state, source_role], dtype=np.float32)
                    state_max.append(float(np.max(np.abs(x - y))))
                    state_relative.append(float(np.linalg.norm(x - y) / (np.linalg.norm(x) + 1e-12)))
                pair_rows.append({
                    "set_id": group["set_id"], "family": group["family"], "partition": group["partition"],
                    "surface": surface, "pair": f"{left}__{right}",
                    "state0_max_abs": state_max[0], "all_state_max_abs": max(state_max),
                    "state35_max_abs": state_max[35], "state35_relative_l2": state_relative[35],
                    "first_state_over_1e_3": next((state for state, value in enumerate(state_max) if value > 1e-3), None),
                })
    violation_counts = {
        "pairs": len(pair_rows),
        "all_state_max_abs_over_1e_3": sum(row["all_state_max_abs"] > 1e-3 for row in pair_rows),
        "state35_max_abs_over_1e_2": sum(row["state35_max_abs"] > 1e-2 for row in pair_rows),
        "by_partition": {partition: sum(row["all_state_max_abs"] > 1e-3 and row["partition"] == partition for row in pair_rows) for partition in protocol["partitions"]},
    }
    selected = []
    for partition in protocol["partitions"]:
        for family in protocol["families"]:
            set_id = next(row["set_id"] for row in atlas_index if row["partition"] == partition and row["family"] == family)
            selected.extend(row for row in index if row["set_id"] == set_id)
    selected_path = OUT / "protocol/singleton_calibration_cases.jsonl"
    core.write_rows(selected_path, selected)
    calibration = {
        "case_count": len(selected), "case_sha256": core.sha(selected_path),
        "singleton_repeat_max_abs": 1e-6,
        "singleton_causal_prefix_relative_l2": 5e-3,
        "canonical_engine_rule": "authorize full recapture only if singleton repeat and causal-prefix gates pass; batch-singleton disagreement is diagnosed, not silently tolerated",
    }
    summary = {
        "phase": 1526, "campaign": "C089", "audit_type": "causal-prefix measurement identity audit",
        "mathematical_invariant": "for a causal decoder, changing only tokens after the registered source span must not materially change that source state",
        "source_truth_contrast_max_abs_by_state": source_by_state,
        "source_truth_contrast_all_state_max_abs": max(source_by_state),
        "pair_violation_counts": violation_counts,
        "calibration": calibration,
        "adjudication": {
            "phase1522": "behavior and field execution are numerically unqualified until singleton comparison",
            "phase1523": "truth-contrast atlas is superseded as a mechanism measurement because a causally impossible source response is present",
            "phase1524_1525": "descriptive discovery and holdout replication cannot be interpreted; a stable batching artifact can itself replicate",
            "semantic_result": "not tested by a qualified camera",
        },
        "model_run": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "state0_exact": source_by_state[0] == 0.0,
        "violation_detected": max(source_by_state) > 1e-2 and violation_counts["all_state_max_abs_over_1e_3"] > 0,
        "all_partitions_affected": all(value > 0 for value in violation_counts["by_partition"].values()),
        "pairs": len(pair_rows) == 180,
        "calibration_cases": len(selected) == 72,
        "supersession": "unqualified" in summary["adjudication"]["phase1522"] and "superseded" in summary["adjudication"]["phase1523"],
        "no_model": not summary["model_run"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary["checks"] = checks
    core.write_rows(OUT / "analysis/causal_prefix_pair_diagnostics.jsonl", pair_rows)
    core.save(OUT / "protocol/singleton_calibration_protocol.json", calibration)
    core.save(OUT / "analysis/causal_prefix_identity_failure.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1526, "campaign": "C089", "status": "left_padding_camera_failed_causal_prefix_identity", "authorization": "run_phase1527_c090_singleton_numeric_calibration"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
