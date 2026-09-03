#!/usr/bin/env python3
"""Phase1482: preregister the layered-observation campaign policy."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1481_c080_c083_major_stage_closure"
OUT = RESULT / "phase1482_layered_observation_policy"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1482 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    checks = {
        "parent_closed": parent["status"] == "major_stage_closed_with_one_retrospective_structure_candidate_and_unresolved_fresh_validation",
        "parent_audited": parent_audit["all_checks_passed"],
        "restart_authorized": parent["authorization"] == "no_automatic_continuation_until_a_new_behavior_qualified_object_or_project_level_gate_policy_is_preregistered",
        "prospective_only": True,
        "no_model_run": True,
        "no_hidden_access": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    policy = {
        "phase": 1482,
        "schema": "glm5.layered_observation_predefined_missingness_batch_validation.v1",
        "name": "layered observation plus predefined missingness plus batch validation",
        "effective_scope": "prospective campaigns and retrospective reuse of already-legally-captured immutable arrays only",
        "evidence_layers": [
            {"id": "L0", "name": "material", "states": ["qualified", "qualified_with_limitations", "failed"]},
            {"id": "L1", "name": "behavior", "states": ["qualified", "mixed", "failed", "not_applicable"]},
            {"id": "L2", "name": "observation", "states": ["legal_raw", "prospective_diagnostic_raw", "missing_by_design", "unavailable"]},
            {"id": "L3", "name": "pattern", "states": ["exploratory", "frozen_candidate", "replicated", "failed"]},
            {"id": "L4", "name": "causal", "states": ["not_tested", "sufficient", "necessary", "selective", "failed"]},
            {"id": "L5", "name": "externality", "states": ["single_task", "new_material", "cross_task", "cross_model", "natural_use"]},
        ],
        "missingness_codes": {
            "M0": "observed from a legally captured immutable source",
            "M1": "not captured because the historical contract stopped before Hidden State access",
            "M2": "prospectively captured from a behavior-failed or mixed stratum for diagnostics only",
            "M3": "unavailable because of model, hardware, or source loss",
            "M4": "already-open exploratory data; cannot serve as independent confirmation",
        },
        "route_rules": [
            "A failed layer closes only claims that logically require that layer; it does not stop preauthorized sibling routes.",
            "Behavior qualification remains mandatory for successful-mechanism and natural-use claims.",
            "A future contract may authorize diagnostic Hidden State capture in mixed or failed behavior strata before reveal; those data cannot establish a successful natural mechanism.",
            "Historical M1 missingness is immutable: C080, C081, and C083 may not be retroactively recaptured under this policy.",
            "All already-open splits are exploratory even when their former names include confirmation or lockbox.",
            "Discovery, replication, diagnostic-failure, and causal evidence must remain separately typed in every result.",
            "One route failure automatically advances to the next preregistered route; the major campaign ends only when its route queue is exhausted.",
        ],
        "batch_validation": {
            "required_axes": ["material provenance", "behavior stratum", "split", "surface", "state", "role", "full coordinate"],
            "required_outputs": ["complete result matrix", "explicit missingness matrix", "pattern scope", "next route decision"],
            "forbidden": ["retroactive threshold changes", "calling open data confirmation", "silently dropping failed strata", "PCA", "learned probes", "attention", "MLP", "parameters"],
        },
        "historical_adjudication": {
            "accepted": [
                "C080 and C081 never measured the balanced interaction because behavior stopped Hidden State access.",
                "C082 is a strong retrospective structural candidate, not an independent confirmation.",
                "C083 left P082 untested rather than falsified.",
                "The C082 common boundary direction is confounded with task control and yes/no decision.",
                "The 17 residual coordinates are basis- and threshold-dependent coordinates, not identified neurons or causal anchors.",
            ],
            "corrections": [
                "The state0 mean cosine near -0.2 may be induced by the six-way cyclic contrast geometry and is not by itself a semantic law.",
                "High cosine to a mean vector is partly circular; leave-one-relation and pairwise checks are required.",
                "C081 changed both interface and materials, so its gain cannot be attributed to interface alone.",
                "No result licenses a new mathematical theory, semantic manifold, or coordinate-level mechanism breakthrough.",
                "The new policy cannot retroactively legalize absent historical captures.",
            ],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    policy["policy_sha256"] = core.digest(policy)
    policy["authorization"] = "run_phase1483_existing_observation_asset_registry"
    core.save(OUT / "protocol/layered_observation_policy.json", policy)
    core.save(OUT / "audit/preimplementation_audit.json", {"phase": 1482, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())})
    core.save(OUT / "analysis/final.json", {
        "phase": 1482,
        "status": "project_level_gate_policy_preregistered",
        "policy_sha256": policy["policy_sha256"],
        "model_run": False,
        "hidden_access": False,
        "authorization": policy["authorization"],
    })
    print(json.dumps({"checks": checks, "policy_sha256": policy["policy_sha256"], "authorization": policy["authorization"]}, indent=2))


if __name__ == "__main__":
    main()
