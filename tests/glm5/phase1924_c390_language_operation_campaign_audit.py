#!/usr/bin/env python3
"""Independent audit for Phase1903-1924 / C369-C390."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase1903_c369_c390_language_operation_graph_campaign as campaign


SUPPLEMENT_CAPTURE = TESTS / "phase1922_c388_c389_cross_tokenizer_bilingual_capture.py"
SUPPLEMENT_FINALIZE = TESTS / "phase1924_c390_language_operation_campaign_finalize.py"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    checks: dict[str, bool] = {}
    finals = {name: load(campaign.OUTS[name] / "analysis/final.json") for name in campaign.PHASES}
    checks["phase_sequence_1903_1924"] = [finals[f"C{value}"]["phase"] for value in range(369, 391)] == list(range(1903, 1925))
    checks["all_phases_closed"] = all(value["status"] == "closed" and value["all_checks_passed"] for value in finals.values())

    main_hash = sha(TESTS / "phase1903_c369_c390_language_operation_graph_campaign.py")
    supplement_hash = sha(SUPPLEMENT_CAPTURE)
    finalize_hash = sha(SUPPLEMENT_FINALIZE)
    producer_ok = True
    for value in range(369, 391):
        name = f"C{value}"
        protocol_hash = load(campaign.OUTS[name] / "protocol/preregistration.json")["producer_sha256"]
        expected = supplement_hash if name in {"C388", "C389"} else finalize_hash if name == "C390" else main_hash
        producer_ok &= protocol_hash == expected
    checks["registered_mixed_producer_hashes"] = producer_ok

    c373 = finals["C373"]["headline"]
    checks["qwen_behavior_qualified"] = c373["confirmation_accuracy"] >= 0.90 and len(c373["eligible_families"]) == 16

    atlas_path = campaign.OUTS["C375"] / "analysis/family_operation_mean_response.float16.npy"
    atlas = np.load(atlas_path, mmap_mode="r")
    checks["full_coordinate_atlas_shape"] = list(atlas.shape) == [16, 4, 38, 6, 2560]
    mapping = getattr(atlas, "_mmap", None)
    if mapping is not None:
        mapping.close()
    del atlas

    c376 = finals["C376"]["headline"]
    checks["transfer_is_local_not_universal"] = len(c376["families_with_any_qualified_cross_cell"]) == 16 and "not a universal language operator" in c376["strict_interpretation"].lower()
    c377 = finals["C377"]["headline"]
    checks["conditional_i_only_three_of_sixteen"] = c377["passed_count"] == 3 and set(c377["families_passed"]) == {"causal_direction", "negation_scope", "attribute_binding"}
    checks["conditional_i_not_causal"] = c377["causal_candidate_eligible"] is False
    c378 = finals["C378"]["headline"]
    checks["negation_scope_order_positive"] = c378["semantic_scope_result"]["gain"] > 0.20 and c378["semantic_scope_result"]["control_advantage"] > 0.40
    checks["order_not_language_algebra"] = "not be called noncommutative language algebra" in c378["strict_interpretation"].lower()

    checks["graph_behavior_failed"] = finals["C380"]["headline"]["graph_hidden_eligible"] is False
    checks["graph_field_not_run"] = finals["C381"]["headline"]["status"] == "graph_field_not_run_behavior_ineligible"
    checks["graph_recursion_not_claimed"] = finals["C382"]["headline"]["recursive_operator_established"] is False
    checks["natural_external_eligible"] = finals["C383"]["headline"]["hidden_state_eligible"] is True
    checks["known_truth_calibrated_only"] = finals["C384"]["headline"]["calibration_passed"] is True and "known-truth" in finals["C384"]["headline"]["strict_interpretation"].lower().replace(" ", "-")
    checks["causal_withheld"] = finals["C386"]["headline"]["causal_claim"] is False

    checks["qwen_bilingual_eligible"] = finals["C387"]["headline"]["abstract_response_eligible"] is True
    checks["glm_bilingual_eligible"] = finals["C388"]["headline"]["abstract_response_eligible"] is True
    checks["deepseek_bilingual_ineligible"] = finals["C389"]["headline"]["abstract_response_eligible"] is False and finals["C389"]["headline"]["lockbox_accuracy"] == 0.5
    checks["glm_span_fallback_narrow"] = finals["C388"]["headline"]["span_method_counts"].get("decoded_local_window") == 12
    checks["deepseek_spans_exact"] = finals["C389"]["headline"]["span_method_counts"].get("decoded_local_window", 0) == 0

    c390 = finals["C390"]["headline"]
    checks["new_math_gate_closed"] = c390["new_math_gate_passed"] is False and c390["gates"]["new_math"] is False
    checks["three_model_gate_closed"] = c390["gates"]["bilingual_all_models"] is False
    checks["causal_gate_closed"] = c390["gates"]["causal"] is False

    visual_path = ROOT / "frontend/public/vis_data/research_kernel/c390_language_operation_full_coordinate.json"
    visual = load(visual_path)
    all_rows = visual["family_operation_rows"] + visual["all_token_rows"]
    checks["visual_schema"] = visual["schema"] == "c390.language_operation_full_coordinate.v1"
    checks["visual_all_2560_coordinates"] = len(visual["dimensions"]) == 2560 and len(visual["family_operation_rows"]) == 320 and all(len(row["values"]) == 2560 for row in all_rows)

    cleanup = load(campaign.OUTS["C390"] / "audit/hidden_state_cleanup_manifest.json")
    checks["cleanup_six_registered_entries"] = len(cleanup) == 6
    checks["cleanup_paths_absent"] = all(not (ROOT / item["path"]).exists() and item["removed_after_analysis"] for item in cleanup)
    checks["cleanup_checksums_for_existing_files"] = sum(bool(item["sha256"]) for item in cleanup) == 5
    checks["c374_precommit_loss_disclosed"] = sum(item["status"] == "removed_by_failed_frozen_cleanup_before_manifest_commit" for item in cleanup) == 1
    checks["provisional_manifest_exists"] = (campaign.OUTS["C390"] / "audit/hidden_state_cleanup_manifest.provisional.json").exists()

    superseded = (
        "c369_c373_superseded_pre_lifecycle_fix_20260824",
        "c369_c375_superseded_pre_complete_group_audit_fix_20260824",
        "c388_superseded_pre_cross_tokenizer_span_fix_20260824",
        "c390_superseded_pre_memmap_lifecycle_fix_20260824",
    )
    checks["superseded_attempts_preserved"] = all((campaign.RESULT / name).exists() for name in superseded)

    for message, passed in checks.items():
        require(passed, message)
    result = {
        "phase": 1924,
        "campaign": "C390",
        "audit": "independent",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": "C369-C390 establishes typed full-coordinate response observations and local transfer candidates, while broad second-order, recursive graph, three-model bilingual, causal, and new-mathematics gates remain closed.",
    }
    out = campaign.OUTS["C390"] / "audit/independent_audit.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
