#!/usr/bin/env python3
"""Phase1484: preregister C084 batch deep mining of the immutable C079 field."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1483_existing_observation_asset_registry"
POLICY = RESULT / "phase1482_layered_observation_policy"
OUT = RESULT / "phase1484_c084_batch_deep_mining_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1484 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    registry = core.load(PARENT / "analysis/asset_registry.json")
    policy = core.load(POLICY / "protocol/layered_observation_policy.json")
    c079 = core.load(RESULT / "phase1463_c079_aggregate_observation_contract/protocol/preregistration.json")
    selected = {row["asset_id"]: row for row in registry["assets"] if row["selected_for_c084"]}
    checks = {
        "parent": parent["authorization"] == "preregister_c084_c079_batch_deep_mining" and parent_audit["all_checks_passed"],
        "registry": parent["registry_sha256"] == registry["registry_sha256"],
        "policy": registry["policy_sha256"] == policy["policy_sha256"],
        "sources": set(selected) == {"C079_discovery_full_field", "C079_confirmation_lockbox_full_field", "C082_relation_mean_atlas", "C082_sign_atlas"},
        "hashes": all(row["hash_valid"] for row in selected.values()),
        "c079_scope": c079["claim_boundary"]["allowed"].startswith("explicit labeled-carrier"),
        "no_model_run": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    contract = {
        "phase": 1484,
        "campaign": "C084",
        "schema": "c084.c079_sign_threshold_factorial_batch_atlas.v1",
        "research_object": "sign-aware, threshold-aware, layer-resolved and factorial structure of the C079 relation-match full Hidden-State field",
        "source_registry_sha256": registry["registry_sha256"],
        "source_assets": selected,
        "axes": {
            "relations": list(c079["relations"]),
            "splits": list(c079["partitions"]),
            "surfaces": list(c079["surfaces"]),
            "states": list(range(37)),
            "roles": list(c079["role_slots"]),
            "coordinates": 2560,
            "factorial_bits": ["entity_match", "object_match", "relation_match"],
        },
        "coordinate_branch": {
            "support_fractions": [0.005, 0.01, 0.02, 0.05],
            "support_counts": [13, 26, 51, 128],
            "required": [
                "sign of every relation-mean coordinate",
                "all-relation intersection and union at every state and role",
                "pairwise support Jaccard",
                "leave-one-relation common-direction cosine",
                "leave-one-panel stability",
                "state0 cyclic-contrast centroid check",
                "explicit sign audit of the formerly reported 17 coordinates",
            ],
        },
        "factorial_branch": {
            "contrasts": ["relation", "entity", "object", "relation_entity", "relation_object", "entity_object", "relation_entity_object"],
            "coding": "sum-to-zero plus/minus-one coding over the complete 2x2x2 cell cube",
            "required": ["all seven full-coordinate mean contrast fields", "coefficient-energy ledger", "surface cosine and normalized distance", "interaction-to-relation ratios by state and role"],
        },
        "examples": [
            "111 versus 110 changes only relation_match while entity_match and object_match stay true",
            "the relation_entity contrast compares the relation difference when entity_match is true against the relation difference when entity_match is false, averaged over object_match",
        ],
        "evidence_typing": {
            "material": "L0 qualified_with_limitations: machine-audited synthetic explicit-label English; no human naturalness lock",
            "behavior": "L1 qualified for the C079 eligible-set success domain",
            "observation": "L2 legal_raw M0 plus derived already-open M4",
            "pattern": "L3 exploratory; all splits are open",
            "causal": "L4 not_tested",
            "externality": "L5 single_task single_model",
        },
        "allowed": ["immutable C079 embeddings and all Hidden States", "full-coordinate subtraction", "mean", "sign", "L2 norm", "cosine", "coordinate energy", "deterministic set overlap"],
        "forbidden": ["new model run", "attention", "MLP", "parameters", "gradients", "PCA", "learned probe", "clustering", "causal language", "calling any output confirmation"],
        "route_queue": ["coordinate branch", "factorial branch", "joint synthesis and future-prediction freeze"],
        "stop_rule": "only an integrity failure stops C084; a weak or absent pattern is recorded in the complete matrix and the next preregistered branch still runs",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    contract["contract_sha256"] = core.digest(contract)
    contract["authorization"] = "run_phase1485_c084_coordinate_stability_atlas"
    core.save(OUT / "protocol/preregistration.json", contract)
    core.save(OUT / "audit/pre_run_audit.json", {"phase": 1484, "campaign": "C084", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "model_run": False})
    core.save(OUT / "analysis/final.json", {"phase": 1484, "campaign": "C084", "status": "batch_deep_mining_preregistered", "contract_sha256": contract["contract_sha256"], "authorization": contract["authorization"]})
    print(json.dumps({"checks": checks, "contract_sha256": contract["contract_sha256"], "authorization": contract["authorization"]}, indent=2))


if __name__ == "__main__":
    main()
