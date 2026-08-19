#!/usr/bin/env python3
"""Independent preregistration and result audit for Phase1256/C009."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import phase1256_c009_qwen3_typed_edge_coalition as main


def write(value: dict[str, Any], path: Path) -> None:
    main.atomic_json(path, value)
    print(main.canonical_json(value))


def preaudit() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    protocol = main.read_json(main.PROTOCOL)
    rows = main.read_jsonl(main.MATERIAL)
    expected = main.protocol_payload(rows, protocol["token_audit"])
    checks["source_hashes"] = protocol["source_hashes"] == expected["source_hashes"]
    checks["protocol_digest"] = protocol["protocol_digest"] == expected["protocol_digest"]
    checks["row_count"] = len(rows) == sum(main.WORLD_COUNTS.values()) == 128
    checks["partition_counts"] = {
        name: sum(row["partition"] == name for row in rows) for name in main.WORLD_COUNTS
    } == main.WORLD_COUNTS
    checks["row_digests"] = all(
        main.digest({key: value for key, value in row.items() if key != "row_digest"}) == row["row_digest"]
        for row in rows
    )
    checks["counterfactual_relations"] = all(
        row["values"]["target"] != row["values"]["base"]
        and row["values"]["wrong"] not in (row["values"]["base"], row["values"]["target"])
        and row["values"]["null"] == row["values"]["base"]
        for row in rows
    )
    token = protocol["token_audit"]
    value_ids = [ids[0] for ids in token["value_token_ids"].values()]
    name_ids = [ids[0] for ids in token["name_token_ids"].values()]
    checks["token_qualification"] = (
        token["all_values_single_token"]
        and token["all_names_single_token"]
        and token["all_panels_same_length"]
        and len(set(value_ids)) == len(main.VALUES)
        and len(set(name_ids)) == len(main.NAMES)
    )
    components = main.component_ids()
    checks["component_ontology"] = (
        len(components) == 108
        and len(set(components)) == 108
        and all(sum(name.endswith("." + role) for name in components) == 36 for role in ("q", "ov", "mlp"))
    )
    checks["frozen_selection"] = (
        tuple(protocol["selection"]["prefix_sizes"]) == main.PREFIX_SIZES
        and protocol["thresholds"] == main.THRESHOLDS
        and protocol["budgets"]["max_formal_runs"] == 1
        and protocol["budgets"]["max_adaptive_rounds"] == 0
    )
    checks["one_shot_clean"] = not any(
        path.exists() for path in (main.RAW, main.DETAILS, main.COMPLETE, main.ANALYSIS, main.FINAL, main.FINAL_AUDIT)
    )
    result = {
        "phase": main.PHASE,
        "audit_stage": "preaudit",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    result["audit_digest"] = main.digest(result)
    return result


def final_audit() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    protocol, _ = main.verify_protocol()
    raw = main.read_json(main.RAW)
    details = main.read_json(main.DETAILS)
    marker = main.read_json(main.COMPLETE)
    analysis = main.read_json(main.ANALYSIS)
    final = main.read_json(main.FINAL)
    checks["formal_completion"] = (
        marker["status"] == "formal_run_complete"
        and marker["raw_sha256"] == main.file_sha256(main.RAW)
        and marker["details_sha256"] == main.file_sha256(main.DETAILS)
        and raw["details_sha256"] == main.file_sha256(main.DETAILS)
        and raw["run_digest"] == marker["run_digest"] == main.digest(details)
    )
    precision = details["precision_audit"]
    checks["fp16_cuda_no_quantization"] = (
        set(precision["parameter_dtypes"]) == {"float16"}
        and precision["has_fp16_parameters"]
        and not precision["has_bf16_parameters"]
        and not precision["has_quantized_modules"]
        and details["placement"]["placement"] == "full_cuda"
    )
    behavior = details["behavior"]
    behavior_recomputed = (
        behavior["candidate_finite_fraction"] >= main.THRESHOLDS["candidate_finite_fraction_min"]
        and min(behavior["panel_accuracy"].values()) >= main.THRESHOLDS["panel_accuracy_min"]
    )
    checks["behavior_gate"] = behavior["passed"] == behavior_recomputed
    checks["trace_hard_stop"] = details["traces_captured"] == behavior_recomputed and (
        behavior_recomputed or not details.get("selected_components")
    )
    if behavior_recomputed:
        ranking = details["discovery_ranking"]
        selected = details["selected_components"]
        size = details["selected_size"]
        checks["discovery_and_prefix"] = (
            len(ranking) == len(set(ranking)) == 108
            and set(ranking) == set(main.component_ids())
            and size in main.PREFIX_SIZES
            and selected == ranking[:size]
        )
        checks["role_counts"] = details["selected_role_counts"] == {
            role: sum(name.endswith("." + role) for name in selected) for role in ("q", "ov", "mlp")
        }
        confirmation_passed = (
            details["target_effect_norm"] >= main.THRESHOLDS["target_effect_norm_min"]
            and main.passes(details["confirmation"])
        )
        checks["confirmation_gate"] = details["passed"] == confirmation_passed
    else:
        checks["discovery_and_prefix"] = "discovery_ranking" not in details
        checks["role_counts"] = "selected_role_counts" not in details
        checks["confirmation_gate"] = details["passed"] is False and "confirmation" not in details
    verdict = "qwen3_typed_edge_coalition_confirmed" if details["passed"] else "qwen3_typed_edge_coalition_not_confirmed"
    checks["verdict"] = verdict == analysis["verdict"] == final["verdict"]
    checks["authorization"] = (
        final["authorization"]["automatic_next_phase"] is False
        and final["authorization"]["glm4_or_ds7b"] is False
        and final["authorization"]["semantic_mechanism_claim"] is False
        and final["authorization"]["cross_model_claim"] is False
        and final["authorization"]["new_mathematics"] is False
    )
    checks["budget"] = raw["gpu_hours"] <= protocol["budgets"]["max_gpu_hours"]
    checks["final_hashes"] = all(
        final["artifact_hashes"][name] == main.file_sha256(path)
        for name, path in {
            "protocol": main.PROTOCOL,
            "material": main.MATERIAL,
            "environment": main.ENVIRONMENT,
            "preaudit": main.PREAUDIT,
            "raw": main.RAW,
            "details": main.DETAILS,
            "complete": main.COMPLETE,
            "analysis": main.ANALYSIS,
        }.items()
    )
    result = {
        "phase": main.PHASE,
        "audit_stage": "final",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "recomputed_behavior": behavior_recomputed,
        "recomputed_verdict": verdict,
    }
    result["audit_digest"] = main.digest(result)
    return result


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "final"))
    args = parser.parse_args()
    value = preaudit() if args.stage == "preaudit" else final_audit()
    write(value, main.PREAUDIT if args.stage == "preaudit" else main.FINAL_AUDIT)
    if not value["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
