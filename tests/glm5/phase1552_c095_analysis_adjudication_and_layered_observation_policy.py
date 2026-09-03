#!/usr/bin/env python3
"""Phase1552: adjudicate the supplied analyses and restore layered observation for C095."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1551_c094_and_output_orthogonalization_final_adjudication"
POLICY_SOURCE = RESULT / "phase1482_layered_observation_policy"
FIELD_SOURCE = RESULT / "phase1539_c091_canonical_all_state_capture"
BEHAVIOR_SOURCE = RESULT / "phase1537_c091_behavior_only_qualification"
OUT = RESULT / "phase1552_c095_analysis_adjudication_and_layered_observation_policy"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1552 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    old_policy = core.load(POLICY_SOURCE / "protocol/layered_observation_policy.json")
    old_policy_audit = core.load(POLICY_SOURCE / "audit/independent_final_audit.json")
    field_final = core.load(FIELD_SOURCE / "analysis/final.json")
    field_audit = core.load(FIELD_SOURCE / "audit/independent_final_audit.json")
    behavior_audit = core.load(BEHAVIOR_SOURCE / "audit/independent_final_audit.json")
    if parent["authorization"] != "no_automatic_model_run_until_C095_object_contract" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1551 closure missing")
    field_path = FIELD_SOURCE / "raw/canonical_all_role_field.float16.npy"
    index_path = FIELD_SOURCE / "raw/canonical_all_role_field_index.jsonl"
    behavior_path = BEHAVIOR_SOURCE / "raw/behavior_logits.jsonl"
    field = np.load(field_path, mmap_mode="r")
    index = core.rows(index_path)
    behavior = core.rows(behavior_path)
    checks = {
        "parent_audited": True,
        "phase1482_policy_audited": old_policy_audit["all_checks_passed"],
        "phase1482_policy_identity": old_policy["name"] == "layered observation plus predefined missingness plus batch validation",
        "field_source_audited": field_audit["all_checks_passed"] and field_final["status"] == "canonical_all_state_capture_numeric_gate_pass",
        "field_shape": list(field.shape) == [540, 37, 4, 2560],
        "index_coverage": len(index) == 540 and len({row["case_id"] for row in index}) == 540,
        "behavior_coverage": len(behavior) == 540 and len({row["case_id"] for row in behavior}) == 540,
        "behavior_source_audited": behavior_audit["all_checks_passed"],
        "no_model_run": True,
        "no_new_hidden_capture": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    policy = {
        "phase": 1552,
        "campaign": "C095",
        "schema": "glm5.layered_observation_predefined_missingness_batch_validation.v2",
        "inherits": {"phase": 1482, "policy_sha256": old_policy["policy_sha256"]},
        "purpose": "deep-mine the already legal C091 embedding-hidden-state-logit field without converting behavior failures into project-level stops",
        "analysis_adjudication": {
            "accepted": [
                "K267 is a prospectively repeated late-boundary descriptive response in one behavior-qualified whole-part route.",
                "C092-C094 did not identify truth, codebook, and emitted-token components.",
                "The nearest useful work is observation-first reuse of immutable full-state data.",
                "Comparison-relation families are a plausible future campaign, but remain untested.",
            ],
            "corrections": [
                "C091 shows only that whole-part passed this frozen interface; it does not show that Qwen3 only understands whole-part.",
                "Post-publication release excludes direct training on the released table, not prior exposure to ordinary word pairs.",
                "K267 does not prove semantic understanding or output preparation; its identity is unresolved.",
                "C092-C094 do not show that the model cannot reverse mappings in general; only the registered prompts failed joint qualification.",
                "K267 is not safely claimed as the first behavior-qualified prospective hidden-state observation in the entire project without a full historical priority audit.",
                "Natural true-versus-false contrasts do not remove answer-token confounding when truth and answer identity remain coupled.",
                "The proposed elephant/cat and whale/mouse comparison examples are not valid negative controls for size.",
                "A shared comparison operation across size, length, type, weight, speed, and temperature is a hypothesis, not an accumulated result.",
                "No evidence currently licenses a new mathematical theory or a future K268.",
            ],
        },
        "evidence_layers": {
            "O0_execution": "all 540 rows; exact numeric identity already passed",
            "O1_semantic_qualified": "whole_part query rows, with behavior qualification inherited from discovery and split scope explicitly recorded",
            "O2_behavior_diagnostic": "similarity/class queries and every behavior-incorrect row; descriptive diagnostics only",
            "O3_cross_partition_repetition": "all three already-open partitions; retrospective repetition, not new independent confirmation",
            "O4_coordinate_structure": "raw 2560 coordinates only; no PCA, learned probes, attention, MLP, parameters, or gradients",
            "O5_causal": "missing by design",
        },
        "missingness": {
            "M_BEHAVIOR": "similarity and class-inclusion lack behavior qualification in this interface",
            "M_OUTPUT": "truth and emitted answer identity are not orthogonalized",
            "M_CAUSAL": "no intervention in C095",
            "M_EXTERNAL": "no new model, language, or task",
            "M_BLIND": "all C091 partitions have been opened previously; C095 is retrospective deep mining",
        },
        "route_rules": [
            "A missing evidence layer limits only claims requiring that layer.",
            "All behavior strata remain in the observation matrix and carry explicit scope tags.",
            "No scalar hard gate may erase a legally captured full-state route.",
            "Patterns are reported as continuous matrices plus predefined descriptive bands, not pass/fail truth claims.",
            "Batch validation spans every partition, surface, state, role, family-pair, and raw coordinate.",
            "Only an independently preregistered future dataset may provide new confirmation.",
        ],
        "source_assets": {
            "field": {"path": str(field_path.relative_to(ROOT)), "sha256": core.sha(field_path), "shape": list(field.shape)},
            "index": {"path": str(index_path.relative_to(ROOT)), "sha256": core.sha(index_path), "rows": len(index)},
            "behavior": {"path": str(behavior_path.relative_to(ROOT)), "sha256": core.sha(behavior_path), "rows": len(behavior)},
        },
        "forbidden_claims": ["pure semantic vector", "causal circuit", "identified neuron group", "cross-model law", "new mathematics"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    policy["policy_sha256"] = core.digest(policy)
    policy["authorization"] = "run_phase1553_c095_existing_field_batch_mining_contract"
    core.save(OUT / "protocol/layered_observation_policy_v2.json", policy)
    core.save(OUT / "audit/preimplementation_audit.json", {"phase": 1552, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())})
    core.save(OUT / "analysis/final.json", {"phase": 1552, "campaign": "C095", "status": "layered_observation_policy_restored", "authorization": policy["authorization"]})
    print(json.dumps({"checks": checks, "policy": policy}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
