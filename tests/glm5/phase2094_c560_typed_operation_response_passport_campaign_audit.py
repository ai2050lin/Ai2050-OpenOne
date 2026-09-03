#!/usr/bin/env python3
"""Independent artifact audit for Phase2076-2093 / C542-C559."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c557_typed_operation_response_passport_atlas.json"
OUT = RESULT / "phase2094_c560_typed_operation_response_passport_campaign_independent_audit"
MAIN = ROOT / "tests/glm5/phase2076_c542_c559_typed_operation_response_passport_campaign.py"
WORKER = ROOT / "tests/glm5/phase2090_c556_cross_model_worker.py"

SLUGS = {
    542: "evidence_adjudication_and_typed_operation_master_contract",
    543: "typed_linguistic_operation_ontology_and_program_graph",
    544: "large_material_compiler_semantic_balance_and_naturalness_audit",
    545: "qwen_behavior_and_all_token_all_coordinate_capture",
    546: "within_domain_operation_response_passports",
    547: "cross_domain_same_type_response_transfer",
    548: "truth_output_surface_and_equal_norm_confound_ledger",
    549: "independent_unit_evidence_and_candidate_adjudication",
    550: "minimal_sufficient_response_history_tournament",
    551: "attitude_event_atomic_composition_response",
    552: "graph_path_completion_interaction_response",
    553: "first_last_mean_token_granularity_tournament",
    554: "typed_response_causal_eligibility_adjudication",
    555: "qualified_hiddenstate_causal_branch_or_registered_na",
    556: "cross_model_functional_replication_branch_or_registered_na",
    557: "response_passport_full_coordinate_visual_atlas",
    558: "raw_field_cleanup_and_next_stage_adjudication",
    559: "campaign_synthesis_and_theory_ledger",
}


def path(campaign: int) -> Path:
    phase = 2076 + campaign - 542
    return RESULT / f"phase{phase}_c{campaign}_{SLUGS[campaign]}"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    checks: dict[str, bool] = {}
    finals = {}
    for campaign in range(542, 560):
        final_path = path(campaign) / "analysis/final.json"
        checks[f"c{campaign}_final_exists"] = final_path.exists()
        finals[campaign] = load(final_path)
        expected_phase = 2076 + campaign - 542
        checks[f"c{campaign}_phase"] = finals[campaign]["phase"] == expected_phase
        checks[f"c{campaign}_identity"] = finals[campaign]["campaign"] == f"C{campaign}"
        checks[f"c{campaign}_closed"] = finals[campaign]["status"] == "closed"

    h544 = finals[544]["headline"]
    for key, expected in {
        "rows": 2808, "compiled_rows": 2808, "pair_count": 1188,
        "duplicate_prompts": 180, "unique_physical_prompts": 2628,
        "shared_prompt_groups": 162, "cross_partition_shared_groups": 0,
        "inconsistent_shared_groups": 0,
    }.items():
        checks[f"c544_{key}"] = h544[key] == expected
    checks["c544_balanced_options"] = abs(h544["candidate_a_first_rate"] - 0.5) < 1e-12
    checks["c544_global_unique_gate_failed"] = h544["formal_global_unique_prompt_gate_passed"] is False
    checks["c544_shared_accounting"] = h544["shared_prompt_accounting_authorized"] is True

    h545 = finals[545]["headline"]
    checks["c545_rows"] = h545["rows"] == 2808
    checks["c545_behavior"] = abs(h545["accuracy"] - 0.9408831908831908) < 1e-12
    checks["c545_mean_shape"] = h545["mean_shape"] == [2808, 38, 6, 2560]
    checks["c545_last_shape"] = h545["last_shape"] == [2808, 38, 6, 2560]
    checks["c545_full_shape"] = h545["full_shape"] == [2808, 38, 123, 2560]
    checks["c545_bf16"] = h545["quantization"]["has_bf16_parameters"] is True
    checks["c545_not_quantized"] = h545["quantization"]["has_quantized_modules"] is False

    for campaign, passed, total in ((546, 534, 792), (547, 385, 792), (548, 80, 144)):
        gate = finals[campaign]["headline"]["gate_summary"]
        checks[f"c{campaign}_passed"] = gate["passed"] == passed
        checks[f"c{campaign}_total"] = gate["total"] == total
        checks[f"c{campaign}_arithmetic"] = abs(gate["pass_rate"] - passed / total) < 1e-15

    h549 = finals[549]["headline"]
    expected_candidates = ["path_depth", "discourse_permutation", "active_passive", "translation"]
    checks["c549_candidates"] = h549["qualified_types"] == expected_candidates
    checks["c549_candidate_count"] = h549["candidate_count"] == 4
    for operation in expected_candidates:
        evidence = h549["unit_evidence"][operation]
        checks[f"c549_{operation}_units"] = evidence["units"] == 48
        checks[f"c549_{operation}_positive"] = evidence["positive_fraction"] == 1.0
        checks[f"c549_{operation}_confound"] = evidence["confound_pass_rate"] == 1.0

    h550 = finals[550]["headline"]["gate_summary"]
    checks["c550_counts"] = h550["passed"] == 13 and h550["total"] == 66
    checks["c550_arithmetic"] = abs(h550["pass_rate"] - 13 / 66) < 1e-15
    h551 = finals[551]["headline"]
    checks["c551_no_composition"] = h551["gate_summary"]["passed"] == 0 and not h551["attitude_composition_candidate"]
    h552 = finals[552]["headline"]
    checks["c552_internal_candidate"] = h552["gate_summary"]["passed"] == 10 and h552["graph_composition_candidate"]
    checks["c552_behavior_not_qualified"] = finals[545]["headline"]["domain_accuracy"]["composition:graph_path_completion"] == 0.25
    h553 = finals[553]["headline"]["view_rates"]
    checks["c553_mean"] = h553["mean"]["passed"] == 83 and h553["mean"]["total"] == 132
    checks["c553_first"] = h553["first"]["passed"] == 80 and h553["first"]["total"] == 132
    checks["c553_last"] = h553["last"]["passed"] == 74 and h553["last"]["total"] == 132

    h554 = finals[554]["headline"]
    checks["c554_one_authorized"] = h554["authorized_types"] == ["active_passive"]
    checks["c554_active_behavior"] = h554["requirements"]["active_passive"]["behavior_accuracy"] == 1.0
    h555 = finals[555]["headline"]
    checks["c555_ran"] = h555["ran"] is True
    checks["c555_causal_type"] = h555["causal_types"] == ["active_passive"]
    for domain, values in h555["metrics"].items():
        checks[f"c555_{domain}_base_margin"] = values["correct_patch"]["nrmse"] <= values["base"]["nrmse"] - 0.02
        checks[f"c555_{domain}_wrong_margin"] = values["correct_patch"]["nrmse"] <= values["wrong_patch"]["nrmse"] - 0.02

    h556 = finals[556]["headline"]
    checks["c556_both_executed"] = h556["models_loaded"] == ["glm4", "deepseek7b"]
    checks["c556_glm_behavior"] = h556["metrics"]["glm4"]["behavior_accuracy"] == 1.0
    checks["c556_ds_behavior_failed"] = h556["metrics"]["deepseek7b"]["behavior_accuracy"] < 0.5
    checks["c556_glm_checkpoints"] = h556["metrics"]["glm4"]["checkpoint_count"] == 41
    checks["c556_ds_checkpoints"] = h556["metrics"]["deepseek7b"]["checkpoint_count"] == 29

    visual = load(VISUAL)
    checks["visual_exists"] = VISUAL.exists()
    checks["visual_schema"] = visual["schema"] == "ai2050.typed_operation_response_passport.v1"
    checks["visual_phase"] = visual["phase"] == 2091
    checks["visual_coordinates"] = visual["coordinate_count"] == 2560
    checks["visual_prototypes"] = len(visual["prototype_rows"]) == 66
    checks["visual_full_rows"] = len(visual["full_token_rows"]) == 4
    checks["visual_size"] = VISUAL.stat().st_size == finals[557]["headline"]["visual_bytes"]

    h558 = finals[558]["headline"]
    checks["cleanup_three"] = h558["cleanup_files"] == 3
    checks["cleanup_bytes"] = h558["cleanup_bytes"] == 73753805184
    checks["raw_absent"] = h558["raw_fields_absent"] is True
    ledger = load(path(558) / "audit/raw_field_cleanup_ledger.json")
    checks["cleanup_ledger_three"] = len(ledger["files"]) == 3
    checks["cleanup_ledger_bytes"] = ledger["total_bytes"] == h558["cleanup_bytes"]
    checks["cleanup_hashes"] = all(len(row["sha256"]) == 64 for row in ledger["files"])
    checks["raw_paths_absent"] = all(not (ROOT / row["path"]).exists() for row in ledger["files"])

    h559 = finals[559]["headline"]
    checks["math_score"] = h559["new_math_gate_score"] == 3 and h559["new_math_gate_total"] == 5
    checks["math_not_authorized"] = h559["new_foundational_math_authorized"] is False
    checks["composition_gate_false"] = h559["new_math_gates"]["behavior_qualified_composition"] is False
    checks["cross_model_gate_false"] = h559["new_math_gates"]["cross_model_functional_isomorphism"] is False
    checks["next_same_goal"] = h559["next_stage_same_goal"] is True
    checks["main_script"] = MAIN.exists()
    checks["worker_script"] = WORKER.exists()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis").mkdir(exist_ok=True)
    (OUT / "audit").mkdir(exist_ok=True)
    (OUT / "protocol").mkdir(exist_ok=True)
    failures = [name for name, passed in checks.items() if not passed]
    final = {
        "phase": 2094, "campaign": "C560", "status": "closed",
        "all_checks_passed": not failures,
        "headline": {
            "status": "independent_audit_closed", "checks": len(checks),
            "passed": len(checks) - len(failures), "failed": len(failures),
            "failures": failures, "visual_sha256": sha(VISUAL),
            "strict_conclusion": "Typed response and local causal sufficiency are supported for the registered Qwen active/passive compiler. Composition and cross-model functional isomorphism remain open.",
            "next_stage_same_goal": not failures and h559["next_stage_same_goal"],
            "next_route": h559["next_route"],
        },
    }
    save = lambda p, v: p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")
    save(OUT / "audit/independent_checks.json", checks)
    save(OUT / "analysis/final.json", final)
    save(OUT / "analysis/summary.json", final["headline"])
    save(OUT / "protocol/preregistration.json", {
        "status": "artifact_audit", "scope": "C542-C559 outputs only",
        "no_model_load": True, "no_raw_field_required": True,
    })
    print(json.dumps(final["headline"], ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
