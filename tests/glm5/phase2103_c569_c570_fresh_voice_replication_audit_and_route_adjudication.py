#!/usr/bin/env python3
"""Independent audit and exact-object route adjudication for C561-C568."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c567_fresh_voice_response_replication_atlas.json"
AUDIT = RESULT / "phase2103_c569_fresh_voice_replication_independent_audit"
ROUTE = RESULT / "phase2104_c570_next_exact_object_route_adjudication"

SLUGS = {
    561: "fresh_voice_replication_master_contract_and_material",
    562: "fresh_material_compiler_balance_and_semantic_audit",
    563: "qwen_fresh_behavior_and_full_coordinate_field_capture",
    564: "old_passport_to_fresh_material_forward_prediction",
    565: "old_passport_to_fresh_material_causal_rescue",
    566: "glm4_within_model_functional_response_replication",
    567: "fresh_replication_visual_atlas_and_raw_cleanup",
    568: "fresh_replication_synthesis_and_next_authorization",
}


def out(campaign: int) -> Path:
    return RESULT / f"phase{2095+campaign-561}_c{campaign}_{SLUGS[campaign]}"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_result(base: Path, phase: int, campaign: str, headline: dict, checks: dict, next_authorization: str) -> None:
    failures = [key for key, value in checks.items() if not value]
    value = {"phase": phase, "campaign": campaign, "status": "closed", "all_checks_passed": not failures,
             "headline": headline, "checks": checks, "failures": failures, "next_authorization": next_authorization}
    save(base / "analysis/final.json", value); save(base / "analysis/summary.json", headline); save(base / "audit/independent_checks.json", checks)
    save(base / "protocol/preregistration.json", {"status": "artifact_only_independent_audit", "no_model_load": True})


def main() -> None:
    finals = {campaign: load(out(campaign) / "analysis/final.json") for campaign in range(561, 569)}
    checks = {}
    for campaign, value in finals.items():
        checks[f"c{campaign}_phase"] = value["phase"] == 2095 + campaign - 561
        checks[f"c{campaign}_identity"] = value["campaign"] == f"C{campaign}"
        checks[f"c{campaign}_closed"] = value["status"] == "closed"
        checks[f"c{campaign}_checks"] = value["all_checks_passed"] is True

    h561 = finals[561]["headline"]
    checks["c561_rows"] = h561["rows"] == 576
    checks["c561_pairs"] = h561["pairs"] == 288
    checks["c561_axes"] = (h561["domains"], h561["units"], h561["surfaces"], h561["query_contracts"]) == (6,12,2,2)
    checks["c561_partitions"] = h561["partition_counts"] == {"discovery":288,"confirmation":144,"lockbox":144}

    h562 = finals[562]["headline"]
    checks["c562_rows"] = h562["compiled_rows"] == 576
    checks["c562_duplicates"] = h562["duplicate_prompts"] == 144 and h562["shared_prompt_groups"] == 144
    checks["c562_unique_gate_failed"] = h562["formal_global_unique_prompt_gate_passed"] is False
    checks["c562_shared_authorized"] = h562["shared_prompt_accounting_authorized"] is True
    checks["c562_no_partition_leak"] = h562["cross_partition_shared_groups"] == 0
    checks["c562_no_inconsistency"] = h562["inconsistent_shared_groups"] == 0
    checks["c562_balance"] = all(item["truth_rate"] == .5 and item["a_first_rate"] == .5 for item in h562["balance"].values())
    checks["c562_fresh_domains"] = h562["old_domain_overlap"] == []

    h563 = finals[563]["headline"]
    checks["c563_behavior"] = h563["accuracy"] == 1.0
    checks["c563_slices"] = len(h563["slice_accuracy"]) == 12 and all(value == 1.0 for value in h563["slice_accuracy"].values())
    checks["c563_mean_shape"] = h563["mean_shape"] == [576,38,6,2560]
    checks["c563_last_shape"] = h563["last_shape"] == [576,38,6,2560]
    checks["c563_full_shape"] = h563["full_shape"] == [576,38,90,2560]
    checks["c563_bf16"] = h563["quantization"]["has_bf16_parameters"] and not h563["quantization"]["has_quantized_modules"]

    h564 = finals[564]["headline"]
    checks["c564_total"] = h564["gate_summary"] == {"passed":120,"total":192,"pass_rate":.625}
    checks["c564_aligned"] = h564["contract_rates"]["aligned_query_voice"] == {"passed":96,"total":96,"pass_rate":1.0}
    checks["c564_fixed"] = h564["contract_rates"]["fixed_active_query"] == {"passed":24,"total":96,"pass_rate":.25}
    checks["c564_not_qualified"] = h564["prediction_candidate"] is False
    checks["c564_gate_arithmetic"] = sum(h564["gates"].values()) == 120 and len(h564["gates"]) == 192
    for key, value in h564["metrics"].items():
        gate = value["correct"]["nrmse"] <= value["zero"]["nrmse"]-.02 and value["correct"]["nrmse"] <= value["wrong"]["nrmse"]-.02
        checks[f"c564_gate_{key}"] = h564["gates"][key] == gate

    h565 = finals[565]["headline"]
    checks["c565_na"] = h565["ran"] is False and h565["result"] == "NA_prediction_or_behavior_not_qualified"
    checks["c565_no_metrics"] = h565["metrics"] == {}

    h566 = finals[566]["headline"]
    glm = h566["glm4"]
    checks["c566_worker"] = glm["status"] == "closed" and glm["returncode"] == 0
    checks["c566_behavior"] = glm["behavior_accuracy"] == 1.0
    checks["c566_shape"] = glm["full_shape"] == [168,42,76,4096] and glm["last_shape"] == [168,42,6,4096]
    checks["c566_gates"] = glm["gate_summary"] == {"passed":24,"total":24,"pass_rate":1.0}
    checks["c566_functional"] = glm["functional_candidate"] is True and h566["cross_model_functional_candidate"] is True
    checks["c566_deepseek_na"] = h566["deepseek7b"]["model_loaded"] is False and h566["deepseek7b"]["parent_accuracy"] < .5
    for key, value in glm["metrics"].items():
        gate = value["correct"]["nrmse"] <= value["zero"]["nrmse"]-.02 and value["correct"]["nrmse"] <= value["wrong"]["nrmse"]-.02
        checks[f"c566_gate_{key}"] = glm["gates"][key] == gate and gate

    h567 = finals[567]["headline"]; visual = load(VISUAL)
    checks["c567_visual"] = VISUAL.exists() and h567["visual_bytes"] == VISUAL.stat().st_size
    checks["c567_qwen_rows"] = len(visual["qwen_representative_full_fields"]) == 4 and h567["qwen_representative_rows"] == 4
    checks["c567_glm_rows"] = len(visual["glm4_representative_full_fields"]) == 4 and h567["glm4_representative_rows"] == 4
    checks["c567_dims"] = visual["coordinate_count"] == 2560 and visual["glm4"]["coordinate_count"] == 4096
    checks["c567_cleanup"] = h567["cleanup_files"] == 5 and h567["cleanup_bytes"] == 16170615424 and h567["raw_absent"]
    ledger = load(out(567) / "audit/raw_cleanup_ledger.json")
    checks["c567_ledger"] = len(ledger["files"]) == 5 and ledger["total_bytes"] == 16170615424
    checks["c567_raw_absent"] = all(not (ROOT/item["path"]).exists() for item in ledger["files"])
    checks["c567_hashes"] = all(len(item["sha256"]) == 64 for item in ledger["files"])

    h568 = finals[568]["headline"]
    checks["c568_gates"] = h568["gates"] == {"qwen_behavior":True,"fresh_prediction":False,"fresh_causal":False,"glm4_functional":True,"deepseek_behavior":False}
    checks["c568_counts"] = h568["passed"] == 2 and h568["total"] == 5
    checks["c568_no_new_math"] = h568["new_foundational_math_authorized"] is False

    failures = [key for key,value in checks.items() if not value]
    headline = {"status":"independent_audit_closed","checks":len(checks),"passed":len(checks)-len(failures),"failed":len(failures),"failures":failures,
        "strict_conclusion":"The old Qwen passport transfers perfectly only when fact and query voice change together. GLM4 independently has a stable within-model voice-response topology. A pure cross-model voice operator is not established.",
        "exact_object_complete":not failures}
    write_result(AUDIT,2103,"C569",headline,checks,"C570_route_adjudication")
    if failures: raise SystemExit(1)

    route_checks = {
        "fresh_voice_object_complete": True,
        "next_route_changes_operation_type": h568["next_route"] == "extend_typed_response_to_additional_structural_operations",
        "new_material_contract_required": True,
        "no_unrun_branch_inside_voice_contract": h565["ran"] is False and not h564["prediction_candidate"],
    }
    route_headline = {
        "status":"next_exact_object_route_adjudication_closed",
        "current_exact_object":"fresh active/passive response-passport replication",
        "next_proposed_object":"additional structural operations, starting with discourse permutation under a fixed-query contract",
        "same_exact_goal":False,
        "automatic_continuation_authorized":False,
        "reason":"The next route changes the operation object, material grammar, controls, and causal compiler. It shares the broad theory program but is not an unfinished branch of the registered voice contract.",
        "next_campaign_plan":["fixed-query discourse permutation","fixed-query path depth after behavior repair","translation split into language and layout factors","behavior-qualified composition","cross-model response-topology isomorphism"],
    }
    write_result(ROUTE,2104,"C570",route_headline,route_checks,"new_campaign_requires_separate_freeze")
    print(json.dumps({"C569":headline,"C570":route_headline},ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
