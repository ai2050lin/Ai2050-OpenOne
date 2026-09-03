#!/usr/bin/env python3
"""Phase1556: synthesize and close C095, then authorize fresh-material C096."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P1552 = RESULT / "phase1552_c095_analysis_adjudication_and_layered_observation_policy"
P1553 = RESULT / "phase1553_c095_existing_field_batch_mining_contract"
P1554 = RESULT / "phase1554_c095_triadic_interaction_and_field_atlas"
P1555 = RESULT / "phase1555_c095_behavior_stratified_and_raw_coordinate_atlas"
MATERIAL_SOURCE = RESULT / "phase1535_c091_external_human_material_and_analysis_adjudication"
OUT = RESULT / "phase1556_c095_joint_synthesis_and_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1556 exists")
    finals = [core.load(path / "analysis/final.json") for path in (P1552, P1553, P1554, P1555)]
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (P1552, P1553, P1554, P1555)]
    triadic = core.load(P1554 / "analysis/triadic_and_field_summary.json")
    coordinate = core.load(P1555 / "analysis/behavior_and_coordinate_summary.json")
    source_manifest = core.load(MATERIAL_SOURCE / "protocol/source_manifest.json")
    checks = {
        "sequence": [row["phase"] for row in finals] == [1552, 1553, 1554, 1555],
        "all_audited": all(row["all_checks_passed"] for row in audits),
        "authorization": finals[-1]["authorization"] == "run_phase1556_c095_joint_synthesis_and_major_stage_closure",
        "no_model_run": triadic["model_run"] is False and coordinate["model_run"] is False,
        "complete_matrices": triadic["coverage"]["interaction_atlas_rows"] == 5328 and coordinate["coverage"]["coordinate_stability_rows"] == 192,
        "source_available": all(item["bytes"] > 0 for item in source_manifest.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)

    k268 = {
        "id": "K268",
        "grade": "E2-OBS-retrospective-candidate",
        "name": "late-boundary common-plus-conditional signed coordinate field",
        "statement": "In the immutable C091 Qwen3-4B field, all three pair/query diagonal interactions share a strong late-boundary component while preserving word-order- and family-pair-conditioned residuals; the field is distributed across raw coordinates with highly stable reference-support signs but non-invariant top-k identities.",
        "support": {
            "triadic_focus_min_median_max": [
                triadic["triadic_focus"]["minimum_of_minimum_pairwise_cosine"],
                triadic["triadic_focus"]["median_minimum_pairwise_cosine"],
                triadic["triadic_focus"]["maximum_of_minimum_pairwise_cosine"],
            ],
            "cross_partition_focus_min_median": [
                triadic["cross_partition_focus"]["minimum_of_minimum_partition_cosine"],
                triadic["cross_partition_focus"]["median_minimum_partition_cosine"],
            ],
            "top64_energy_min_median_max": [
                coordinate["coordinate_concentration_k64"]["minimum_energy_fraction"],
                coordinate["coordinate_concentration_k64"]["median_energy_fraction"],
                coordinate["coordinate_concentration_k64"]["maximum_energy_fraction"],
            ],
            "top64_sign_min_median": [
                coordinate["cross_partition_k64"]["minimum_sign_agreement"],
                coordinate["cross_partition_k64"]["median_sign_agreement"],
            ],
            "top64_jaccard_min_median": [
                coordinate["cross_partition_k64"]["minimum_support_jaccard"],
                coordinate["cross_partition_k64"]["median_support_jaccard"],
            ],
        },
        "missing": ["fresh blind replication", "similarity/class behavior qualification", "output-code orthogonalization", "causal necessity/sufficiency", "external model/task"],
        "forbidden_upgrade": ["semantic neuron group", "whole-part truth vector", "generic relation law", "causal mechanism", "new mathematics"],
    }
    synthesis = {
        "phase": 1556,
        "campaign": "C095",
        "status": "major_stage_complete_with_fresh_validation_authorized",
        "methodological_result": "Layered observation retained every legal stratum, exposed both common and conditional structure, and did not convert missing behavior/causal layers into either data erasure or mechanism claims.",
        "adjudication": {
            "K267": "preserved as the original whole-part-scoped prospective observation, but narrowed: it is not identified as pure whole-part truth or output preparation.",
            "K268": k268,
            "behavior_diagnostic": "Correct-minus-incorrect is mostly anti-aligned with the generic triadic field, so K267/K268 are not simple correctness directions; this branch is low precision because some error strata contain one item.",
            "coordinate_identity": "Stable signed distributed raw-coordinate field, not a fixed top-k neuron set.",
        },
        "unified_working_equations": {
            "field_decomposition": "H = mu + P_pair_family + Q_query_family + I_pair,query + epsilon",
            "late_boundary": "I_fq = G_boundary(partition,surface,concreteness) + R_fq(partition,surface,concreteness)",
            "coordinate": "v = sum_j v_j e_j; S_k(v)=TopK(|v|); stable sign/response does not imply invariant support identity",
            "evidence_chain": "execution -> observation -> retrospective repetition; behavior qualification is partial; causal and external layers are missing",
        },
        "theory": {
            "name": "conditional output field closure theory",
            "organizing_principle": "reuse-difference-conditioning (RDC)",
            "update": "The late relation task field is better represented as a reusable boundary component plus conditional family/order residuals distributed over a stable signed coordinate field.",
            "math_status": "Existing arithmetic, linear algebra, factorial contrasts, and conditional dynamical notation are sufficient. No invariant composition law or conservation object licenses new mathematics.",
        },
        "hard_limits": [
            "single Qwen3-4B and one controlled Chinese relation task",
            "all C091 partitions were previously opened before C095",
            "similarity and class-inclusion were behavior-diagnostic rather than semantically qualified",
            "truth and emitted-token identity remain entangled",
            "no coordinate intervention or rescue",
            "top-k support membership varies materially across partitions",
        ],
        "next_stage": {
            "campaign": "C096",
            "object": "fresh unused human-validated Chinese relation pairs with no lexical overlap with C091",
            "purpose": "prospectively test the common-plus-conditional late-boundary field and raw-coordinate predictions",
            "sample_target": "90 pairs: 30 per family, balanced 15 concrete/15 abstract, three frozen partitions",
            "policy": "behavior is measured and typed per family; numeric integrity gates hidden use, while behavior failure creates M_BEHAVIOR rather than deleting legally captured observation strata",
            "predictions_to_freeze_before_capture": [
                "prequery state31-32 boundary triadic minimum pairwise cosine >= 0.75 in every panel",
                "same family-pair cross-partition cosine >= 0.75 at state31-32 boundary",
                "top64 median energy fraction in [0.15, 0.25]",
                "discovery-reference top64 minimum sign agreement >= 0.90 and minimum restricted cosine >= 0.85",
                "postquery is more condition-sensitive than prequery and is reported as a full matrix rather than forced to pass a shared-direction gate",
            ],
            "routes_after_reveal": [
                "all predictions pass: retain K268 as a fresh-material candidate and authorize a separately preregistered causal coordinate perturbation",
                "common field repeats but coordinate support fails: retain field-level law and reject coordinate identity",
                "coordinate response repeats but triadic common field fails: retain family-conditional coordinate objects",
                "all fail: retire K268 while preserving K267 and the C095 retrospective atlas",
            ],
        },
        "checks": checks,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "run_phase1557_c096_fresh_human_relation_field_contract",
    }
    core.save(OUT / "analysis/c095_major_stage_synthesis.json", synthesis)
    core.save(OUT / "protocol/c096_requirements.json", synthesis["next_stage"])
    core.save(OUT / "analysis/final.json", {"phase": 1556, "campaign": "C095", "status": synthesis["status"], "authorization": synthesis["authorization"]})
    print(json.dumps(synthesis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
