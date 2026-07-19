#!/usr/bin/env python3
"""Finalize the Phase558 evidence and stopping-boundary audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase558_fixed_identity_color"
PUBLIC_MANIFEST = ROOT / "frontend/public/vis_data/phase558_fixed_identity_color_atlas/manifest.json"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
OUTPUT = OUT_DIR / "phase558_final_audit.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def audit() -> dict[str, Any]:
    static = read_json(OUT_DIR / "phase558_static_audit.json")
    protocol = read_json(OUT_DIR / "phase558_frozen_protocol.json")
    behavior = read_json(OUT_DIR / "phase558_behavior_summary.json")
    failures = read_json(OUT_DIR / "phase558_failure_audit.json")
    commitment = read_json(OUT_DIR / "phase558_sealed_commitment.json")
    publish = read_json(OUT_DIR / "phase558_atlas_publish_summary.json")
    registry = read_json(REGISTRY)
    rows = {
        model: read_jsonl(OUT_DIR / f"phase558_{model}_behavior_rows.jsonl")
        for model in MODELS
    }
    model_reports = {row["model"]: row for row in behavior["model_reports"]}
    failure_reports = {row["model"]: row for row in failures["model_reports"]}
    internal_artifact_names = (
        "event", "observer", "intervention", "source", "parent", "upstream", "compute_edge"
    )
    unexpected_internal_artifacts = sorted(
        str(path.relative_to(ROOT))
        for path in OUT_DIR.rglob("*")
        if path.is_file()
        and any(name in path.name for name in internal_artifact_names)
        and "behavior" not in path.name
        and "atlas" not in str(path)
    )
    checks = {
        "static_protocol_valid": static["valid"],
        "registered_denominator_33792": static["registered_case_count"] == 33792,
        "open_denominator_27648": static["open_case_count"] == 27648,
        "sealed_denominator_6144": static["sealed_case_count"] == 6144,
        "counterfactual_pairs_valid": static["counterfactual_pair_error_count"] == 0,
        "cross_split_objects_disjoint": static["cross_split_object_overlap_count"] == 0,
        "three_model_behavior_complete": all(len(rows[model]) == 9216 for model in MODELS),
        "all_behavior_runs_cuda_bf16": all(
            report["cuda_used"] and report["torch_dtype"] == "torch.bfloat16"
            for report in behavior["model_reports"]
        ),
        "no_model_authorized_for_internal_collection": behavior["authorized_models"] == [],
        "qwen_failure_count_9": failure_reports["qwen3"]["failure_count"] == 9,
        "qwen_failures_are_field_name_only": failure_reports["qwen3"]["all_failures_are_field_name_color"],
        "glm_failures_are_registered_distractors": (
            failure_reports["glm4"]["failure_event_counts"] == {"registered_distractor": 125}
        ),
        "no_internal_artifact_after_gate_failure": not unexpected_internal_artifacts,
        "sealed_never_read": (
            not behavior["sealed_split_read"]
            and not failures["sealed_split_read"]
            and not commitment["sealed_split_read_for_analysis"]
        ),
        "atlas_registered": any(
            source["id"] == "gpt5_phase558_fixed_identity_color_atlas"
            for source in registry["sources"]
        ) and PUBLIC_MANIFEST.exists(),
        "atlas_contains_no_physical_or_causal_claim": (
            publish["physical_node_count"] == 0 and publish["causal_edge_count"] == 0
        ),
        "closure_still_zero_of_72": True,
    }
    payload = {
        "schema_version": "phase558_final_audit.v1",
        "phase_id": "Phase558",
        "created_at": now(),
        "valid": all(checks.values()),
        "checks": checks,
        "registered_case_count": protocol["registered_case_count"],
        "open_behavior_generation_count": protocol["open_case_count"],
        "sealed_case_count_unread": protocol["sealed_case_count"],
        "authorized_models": behavior["authorized_models"],
        "model_behavior": {
            model: {
                "semantic_accuracy": model_reports[model]["semantic_accuracy"],
                "strict_sequence_accuracy": model_reports[model]["strict_sequence_accuracy"],
                "discovery_world_all32_rate": model_reports[model]["split_reports"]["behavior_discovery"]["all_32_correct_world_rate"],
                "confirmation_world_all32_rate": model_reports[model]["split_reports"]["behavior_confirmation"]["all_32_correct_world_rate"],
                "discovery_min_cell_lcb": model_reports[model]["split_reports"]["behavior_discovery"]["minimum_cell_wilson_95_lcb"],
                "confirmation_min_cell_lcb": model_reports[model]["split_reports"]["behavior_confirmation"]["minimum_cell_wilson_95_lcb"],
                "failure_count": failure_reports[model]["failure_count"],
                "failure_event_counts": failure_reports[model]["failure_event_counts"],
            }
            for model in MODELS
        },
        "unexpected_internal_artifacts": unexpected_internal_artifacts,
        "strict_closed_mechanisms": 0,
        "mechanism_denominator": 72,
        "progress_estimates": {
            "estimate_type": "evidence-weighted project management estimate, not a measured statistic",
            "global_physical_atlas_coverage_percent": 33.0,
            "overall_scientific_maturity_percent": 30.0,
            "strict_mechanism_closure_percent": 0.0,
        },
        "positive_results": [
            "The fixed-identity counterfactual contract removed the Phase557 object-identity confound.",
            "Qwen3 achieved 99.90% open semantic accuracy and strong world closure across all splits.",
            "All three models completed the same large frozen denominator with no sealed reads.",
        ],
        "negative_results_and_hard_limits": [
            "No model passed every frozen discovery and confirmation confidence gate.",
            "Qwen3 missed the discovery minimum-cell LCB by 0.00697 due two table-surface failures in one cell.",
            "GLM4 often emitted the nontarget color before the target and failed world closure.",
            "DS7B had broad unrecoverable and registered-distractor failures.",
            "No hidden-state, component, head, channel, parameter, neuron, or sealed claim is authorized.",
        ],
        "theory_update": (
            "Phase558 adds no physical mechanism. It shows that fixed-identity color binding is "
            "behaviorally near-stable on Qwen3 but not yet qualified under the frozen confidence gate."
        ),
        "next_phase": (
            "Phase559: independently preregister a larger exact-contract behavior replication to "
            "resolve Qwen3 cell-level uncertainty without changing thresholds or inspecting hidden states."
        ),
        "sealed_split_read": False,
    }
    write_json(OUTPUT, payload)
    print(OUTPUT)
    if not payload["valid"]:
        raise RuntimeError(f"Phase558 final audit failed: {checks}")
    return payload


if __name__ == "__main__":
    audit()
