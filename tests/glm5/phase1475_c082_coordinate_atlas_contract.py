#!/usr/bin/env python3
"""Phase1475: preregister a coordinate-resolved retrospective C079 atlas."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1474_c081_route_closure"
C079_CONTRACT = RESULT / "phase1463_c079_aggregate_observation_contract"
DISCOVERY = RESULT / "phase1465_c079_discovery_full_field_capture"
HOLDOUT = RESULT / "phase1467_c079_holdout_capture_and_validation"
OUT = RESULT / "phase1475_c082_coordinate_atlas_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1475 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    c079_protocol = core.load(C079_CONTRACT / "protocol/preregistration.json")
    discovery = core.load(DISCOVERY / "analysis/capture_metadata.json")
    holdout = core.load(HOLDOUT / "analysis/holdout_summary.json")
    c079_closure = core.load(RESULT / "phase1468_c079_campaign_closure/analysis/final.json")
    checks = {
        "parent": parent["authorization"] == "preregister_c082_c079_coordinate_resolved_exploratory_atlas" and parent_audit["all_checks_passed"],
        "c079_closed": c079_closure["status"] == "closed_with_cross_split_explicit_label_trajectory_regularities",
        "discovery_hash": core.sha(DISCOVERY / "raw/discovery_role_field.float16.npy") == discovery["raw_sha256"],
        "discovery_index_hash": core.sha(DISCOVERY / "raw/discovery_role_field_index.jsonl") == discovery["index_sha256"],
        "holdout_hash": core.sha(HOLDOUT / "raw/holdout_role_field.float16.npy") == holdout["raw_sha256"],
        "holdout_index_hash": core.sha(HOLDOUT / "raw/holdout_role_field_index.jsonl") == holdout["index_sha256"],
        "shapes": discovery["shape"] == [1104, 37, 9, 2560] and holdout["shape"] == [2208, 37, 9, 2560],
        "all_splits_already_open": True,
        "no_model_run": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    protocol = {
        "phase": 1475,
        "campaign": "C082",
        "schema": "c082.c079_coordinate_resolved_retrospective_atlas.v1",
        "research_object": "coordinate-resolved trajectory of the C079 paired query relation-label first difference",
        "source": {
            "c079_contract_sha256": c079_protocol["contract_sha256"],
            "discovery_raw_sha256": discovery["raw_sha256"],
            "discovery_index_sha256": discovery["index_sha256"],
            "holdout_raw_sha256": holdout["raw_sha256"],
            "holdout_index_sha256": holdout["index_sha256"],
            "discovery_shape": discovery["shape"],
            "holdout_shape": holdout["shape"],
        },
        "axes": {
            "relations": list(c079_protocol["relations"]),
            "splits": list(c079_protocol["partitions"]),
            "surfaces": list(c079_protocol["surfaces"]),
            "states": list(range(37)),
            "roles": list(c079_protocol["role_slots"]),
            "coordinates": 2560,
        },
        "sample_effect_formula": "Delta = mean over four paired differences H[relation_match=1] - H[relation_match=0], holding entity_match and object_match fixed",
        "outputs": {
            "mean_effect": "float32 [relation, split, surface, state, role, coordinate]",
            "sign_consistency": "float16 fraction of sample effects with the same coordinate sign as the panel mean",
            "sample_count": "int32 [relation, split, surface]",
            "layer_role_metrics": ["mean L2 norm", "mean sample-to-mean cosine", "adjacent-state cosine", "maximum coordinate energy share", "coordinate counts reaching 50/80/90 percent energy"],
            "panel_stability": ["minimum pairwise cosine across six split-surface panels", "fraction of coordinates with unanimous nonzero sign"],
            "onset": "first state reaching 10/50/90 percent of the within-panel role maximum norm",
        },
        "allowed": ["existing C079 embeddings and full Hidden States", "paired subtraction", "mean", "L2 norm", "cosine", "coordinate sign", "sorted coordinate energy"],
        "forbidden": ["new model run", "attention", "MLP", "parameters", "gradients", "PCA", "TDA", "probe", "clustering", "causal intervention", "calling any panel a holdout confirmation"],
        "evidence_scope": "retrospective exploratory observation over already-open C079 splits; patterns may generate but not confirm future hypotheses",
        "integrity": {
            "process_all_2560_coordinates": True,
            "no_coordinate_filter_before_outputs": True,
            "upstream_record_roles_are_structural_zero_controls": True,
            "raw_files_remain_immutable": True,
        },
        "stop_rule": "source hash or shape mismatch stops the atlas; finite complete outputs authorize descriptive synthesis only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1476_c082_coordinate_atlas"
    preaudit = {"phase": 1475, "campaign": "C082", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "model_run": False, "hidden_access": "read only previously authorized immutable C079 raw arrays", "claim_scope": protocol["evidence_scope"]}
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/pre_run_source_and_scope_audit.json", preaudit)
    core.save(OUT / "analysis/final.json", {"phase": 1475, "campaign": "C082", "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]}, indent=2))


if __name__ == "__main__":
    main()
