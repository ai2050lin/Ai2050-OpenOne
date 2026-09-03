#!/usr/bin/env python3
"""Phase1553: preregister C095 batch mining of the immutable C091 full-state field."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1552_c095_analysis_adjudication_and_layered_observation_policy"
OUT = RESULT / "phase1553_c095_existing_field_batch_mining_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1553 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    policy = core.load(PARENT / "protocol/layered_observation_policy_v2.json")
    checks = {
        "parent": parent["authorization"] == "run_phase1553_c095_existing_field_batch_mining_contract",
        "parent_audited": parent_audit["all_checks_passed"],
        "policy": policy["schema"] == "glm5.layered_observation_predefined_missingness_batch_validation.v2",
        "field_shape": policy["source_assets"]["field"]["shape"] == [540, 37, 4, 2560],
        "retrospective": "retrospective" in policy["evidence_layers"]["O3_cross_partition_repetition"],
        "no_model_run": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)

    families = ["similarity", "class_inclusion", "whole_part"]
    family_pairs = [
        ["similarity", "class_inclusion"],
        ["similarity", "whole_part"],
        ["class_inclusion", "whole_part"],
    ]
    contract = {
        "phase": 1553,
        "campaign": "C095",
        "schema": "c095.c091_triadic_full_state_coordinate_batch_mining.v1",
        "research_question": "Is K267 specific to whole-part semantics, or part of a more general late-boundary diagonal-match/output response shared by all three registered relation families?",
        "source_policy_sha256": policy["policy_sha256"],
        "source_assets": policy["source_assets"],
        "axes": {
            "partitions": ["response_discovery", "confirmation", "lockbox"],
            "surfaces": ["prequery", "postquery"],
            "concreteness": ["concrete", "abstract"],
            "families": families,
            "family_pairs": family_pairs,
            "states": list(range(37)),
            "roles": ["source_word", "target_word", "relation_anchor", "boundary"],
            "coordinates": 2560,
        },
        "triadic_interaction_branch": {
            "definition": "C_fg = 0.5 * (H_ff + H_gg - H_fg - H_gf)",
            "required_outputs": [
                "individual matched-quartet interaction vectors",
                "centroid norm and individual-to-centroid alignment",
                "all three family-pair interaction fields",
                "three pairwise family-pair cosines at every panel/state/role",
                "three pairwise cross-partition cosines for every surface/concreteness/family-pair/state/role",
            ],
            "identity_diagnostic": "If the similarity-class interaction aligns with both whole-part interactions, K267 is more consistent with a generic diagonal-match/output response than a whole-part-specific object.",
        },
        "three_by_three_decomposition": {
            "definitions": [
                "mu = mean_fq H_fq",
                "P_f = mean_q H_fq - mu",
                "Q_q = mean_f H_fq - mu",
                "I_fq = H_fq - mu - P_f - Q_q",
            ],
            "required_outputs": ["raw decomposition", "state0-subtracted dynamic decomposition", "reconstruction error", "main-effect and interaction energy ledger"],
        },
        "behavior_stratification_branch": {
            "strata": ["correct", "incorrect"],
            "group_axes": ["partition", "surface", "query_family", "state", "role"],
            "object": "mean(correct hidden state) minus mean(incorrect hidden state)",
            "predefined_missingness": "record M_CELL whenever either stratum has zero rows; do not impute or drop another route",
            "claim_scope": "diagnostic only because correctness is coupled to item difficulty and answer identity",
        },
        "raw_coordinate_branch": {
            "focus": {"states": [31, 32], "role": "boundary"},
            "support_counts": [16, 64, 256, 1024],
            "required_outputs": [
                "top-k squared-energy fraction",
                "discovery-reference top-k sign agreement in both other partitions",
                "top-k restricted cosine and complement cosine",
                "discovery-reference support overlap without dimensionality reduction",
            ],
            "restriction": "all coordinates are analyzed in the original 2560-dimensional basis",
        },
        "descriptive_bands": {
            "cosine": {"high": "x >= 0.90", "moderate": "0.50 <= x < 0.90", "weak": "0 <= x < 0.50", "opposing": "x < 0"},
            "status": "descriptive labels only; no pass/fail or post-hoc threshold change",
        },
        "examples": [
            "similarity/class interaction compares H(sim pair, sim query)+H(class pair, class query) against the two crossed cells",
            "a high late-boundary cosine among all three C_fg vectors would diagnose a shared diagonal-match or answer-boundary field, not three semantic relations",
            "a missing incorrect stratum is written as M_CELL instead of deleting the family or inventing a value",
        ],
        "evidence_typing": policy["evidence_layers"],
        "predefined_missingness": {**policy["missingness"], "M_CELL": "one requested behavior stratum is empty"},
        "allowed": ["immutable C091 full field", "subtraction", "mean", "inner product", "L2 norm", "cosine", "sign", "squared coordinate energy", "deterministic support overlap"],
        "forbidden": ["new model run", "new hidden-state capture", "attention", "MLP", "parameters", "gradients", "PCA", "learned probe", "clustering", "causal claim", "semantic-neuron claim", "new mathematics"],
        "route_queue": [
            "Phase1554 triadic interaction and 3x3 field atlas",
            "Phase1555 behavior-stratified and raw-coordinate atlas",
            "Phase1556 joint synthesis and major-stage closure",
        ],
        "stop_rule": "Only source-integrity, shape, reconstruction, or audit failure stops C095. Weak or absent patterns are retained and the next preregistered branch runs.",
        "claim_boundary": {
            "allowed": "retrospective descriptive structure of one Qwen3-4B task field",
            "forbidden": ["new independent confirmation", "pure semantic vector", "causal mechanism", "cross-model law", "complete encoding theory"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    contract["contract_sha256"] = core.digest(contract)
    contract["authorization"] = "run_phase1554_c095_triadic_interaction_and_field_atlas"
    core.save(OUT / "protocol/preregistration.json", contract)
    core.save(OUT / "audit/pre_run_audit.json", {"phase": 1553, "campaign": "C095", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "model_run": False})
    core.save(OUT / "analysis/final.json", {"phase": 1553, "campaign": "C095", "status": "batch_mining_contract_frozen", "contract_sha256": contract["contract_sha256"], "authorization": contract["authorization"]})
    print(json.dumps({"checks": checks, "contract_sha256": contract["contract_sha256"], "authorization": contract["authorization"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
